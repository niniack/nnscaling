import warnings
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch.nn as nn
import torchinfo
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_model
from torch import Tensor

from nnscaling.models import BaseTorchModel
from nnscaling.utils import StringtoClassNonlinearity, get_device, safetensors_metadata_parser


class ScaledModel(nn.Module):
    """A model that 'scales' a given layer and manages hooks for activation capture."""

    def __init__(
        self,
        model: BaseTorchModel,
        index: Int,
        num_scaled: Int,
        nonlinearity: Literal["relu", "leakyrelu", "gelu", "sigmoid"],
        scale_style: Literal["replace", "prepend"],
        batchnorm: bool = True,
    ):
        super().__init__()

        # Validate scale style
        if scale_style not in {"replace", "prepend"}:
            raise ValueError(
                f"Invalid scale_style '{scale_style}'. Must be either 'replace' or 'prepend'."
            )

        # Validate index to ensure it's not at the start or end of the model
        if index == 0 or index == len(model.features):
            raise ValueError(
                "Scaling does not support layer insertions at the start or end of a model."
            )

        # Lookup and validate nonlinearity
        nonlinearity = (
            StringtoClassNonlinearity[nonlinearity].value
            if isinstance(nonlinearity, str)
            else nonlinearity
        )

        if not issubclass(nonlinearity, nn.Module):
            raise ValueError(f"Nonlinearity '{nonlinearity}' is not supported!")

        self._index = index  # Layer index at which the original model is split
        self._num_scaled = num_scaled  # Number of scaled layers to insert
        self._scale_style = scale_style  # Prepend or replace the layer

        self.activations_dict = OrderedDict()  # Dictionary to store activations
        self.hooks = []  # List to store hook handles

        # Create a deep copy of the model to ensure modifications do not affect the original
        object.__setattr__(self, "_model", deepcopy(model))

        # Freeze all parameters in the copied model
        for param in self._model.parameters():
            param.requires_grad = False

        # Scaling
        original_layer_to_scale = self._model.features[index]
        new_layers = []

        # NOTE: If `scale_style==prepend`, the scaled layer will not get replaced
        if scale_style == "prepend":
            for _ in range(num_scaled):
                new_layers.append(
                    model.factory.scale_layer(
                        original_layer_to_scale, batchnorm=batchnorm, nonlinearity=nonlinearity
                    )
                )

        # NOTE: If `scale_style==replace`, the scaled layer will get replaced
        elif scale_style == "replace":
            # First layer
            new_layers.append(
                model.factory.scale_layer(
                    original_layer_to_scale, batchnorm=batchnorm, nonlinearity=nonlinearity
                )
            )

            # Middle layers
            for _ in range(1, num_scaled - 1):
                new_layers.append(
                    model.factory.scale_layer(
                        original_layer_to_scale, batchnorm=batchnorm, nonlinearity=nonlinearity
                    )
                )

            # Last layer
            # TODO: `last` controls the shape of the layers, this is poor naming
            # Setting `last=True` makes the layer the same shape as the layer being scaled.
            new_layers.append(
                model.factory.scale_layer(
                    original_layer_to_scale,
                    last=True,
                    batchnorm=batchnorm,
                    nonlinearity=nonlinearity,
                )
            )

        # Split the model into pre-insert layers (up to the index)
        self.pre_inserts = nn.Sequential(*list(self._model.features.children())[:index])

        # Sequential container for scaled layers
        self.scaling_layers = nn.Sequential(*new_layers)

        # Capture the scalalbe layer at the index and split remaining layers as post-inserts
        self.scalable_layer = list(self._model.features.children())[index]
        self.post_inserts = nn.Sequential(*list(self._model.features.children())[index + 1 :])

    def __getattr__(self, name):
        # Prioritize locally defined attributes
        if name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]
        # If not found, look for the attribute in the original model
        model = self.__dict__["_model"]
        return getattr(model, name)

    @property
    def index(self) -> Int:
        return self._index

    @property
    def num_scaled(self) -> Int:
        return self._num_scaled

    @property
    def scale_style(self) -> str:
        return self._scale_style

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        # Warn that `forward_scaled` is used by default
        warnings.warn(
            "The `forward` method calls the `forward_scaled` method. For vanilla forward, call `forward_raw`."
        )
        return self.forward_scaled(*args, **kwargs)

    def forward_raw(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the original model"""
        self.activations_dict.clear()

        x = self.pre_inserts(x)
        ### NOTE: We skip the scaling layers here because its the forward raw pass! ###
        x = self.scalable_layer(x)
        return self.post_inserts(x)

    def forward_scaled(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the scaled model"""
        self.activations_dict.clear()  # Clear previous activations

        x = self.pre_inserts(x)
        x = self.scaling_layers(x)  # Gradients on
        # NOTE: We include the scalable if prepending, but it's gradients will be off.
        if self.scale_style == "prepend":
            x = self.scalable_layer(x)
        return self.post_inserts(x)

    def hook_model(self, pre: Bool = False, scaled: Bool = False, post: Bool = False) -> None:
        """Hook selected groups of layers to capture activations."""
        # Clear out all previous hooks
        self.clear_hooks()

        if not (pre or scaled or post):
            raise ValueError("Set something to True!")

        # Hook the last layer of pre-inserts if `pre` is True
        if pre:
            pre_layers = list(self.pre_inserts.children())
            self.hooks.append(self._model.factory.hook_layer(pre_layers[-1]))

        # Hook each scaled layer if `scaled` is True
        if scaled:
            for s_layer in self.scaling_layers:
                self.hooks.append(self._model.factory.hook_layer(s_layer))

        # Hook the scalable layer if `post` is True
        if post:
            self.hooks.append(self._model.factory.hook_layer(self.scalable_layer))

    def clear_hooks(self) -> None:
        """Remove all hooks and clear the hooks list."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def get_activations(self, detach: bool = True) -> OrderedDict:
        """Retrieve activations from all hooked layers."""

        def process_layer(layer, key):
            """Helper to process activations with optional detach."""
            if layer.is_hooked:
                activations[key] = (
                    layer.forward_activations.detach()
                    if detach and isinstance(layer.forward_activations, Tensor)
                    else layer.forward_activations
                )

        activations = OrderedDict()

        # Process pre-insert layer
        pre_insert_layer = list(self.pre_inserts.children())[-1]
        process_layer(pre_insert_layer, f"pre_insert_{len(list(self.pre_inserts.children())) - 1}")

        # Process scaled layers
        for i, layer in enumerate(self.scaling_layers):
            process_layer(layer, f"scaled_{i}")

        # Process scalable layer
        process_layer(self.scalable_layer, "post_insert_0")

        return activations

    @classmethod
    def load_model(cls, model: BaseTorchModel, file_path: str | Path) -> nn.Module:
        """Load a model from a file, including scaled layers."""
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        metadata = safetensors_metadata_parser(file_path=file_path)

        scaled_model = cls(
            model=model,
            index=literal_eval(metadata["layer_start"]),
            num_scaled=literal_eval(metadata["added_layers"]),
            nonlinearity=str(metadata["scale_nonlinearity"]),
            scale_style=str(metadata["scale_style"]),
        )

        # Load weights into the model using safetensors
        load_model(scaled_model, file_path, device=get_device())

        return scaled_model

    def summary(self) -> torchinfo.ModelStatistics:
        """Generate a summary of the model structure and parameters."""
        return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])
