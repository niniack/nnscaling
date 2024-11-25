import warnings
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch.nn as nn
import torchinfo
from jaxtyping import Bool, Float, Int
from safetensors.torch import load_model
from torch import Tensor

from nnscaling.models import BaseTorchModel
from nnscaling.utils import get_device, safetensors_metadata_parser


class ScaledModel(nn.Module):
    """A model that inserts scaled layers into a base model and manages hooks for activation capture."""

    def __init__(self, model: BaseTorchModel, index: Int, num_scaled: Int):
        super().__init__()

        self._index = index  # Index at which the original model is split
        self._num_scaled = num_scaled  # Number of scaled layers to insert
        self.activations_dict = OrderedDict()  # Dictionary to store activations
        self.hooks = []  # List to store hook handles

        # Validate index to ensure it's not at the start or end of the model
        if index == 0 or index == len(model.features):
            raise ValueError(
                "Scaling does not support layer insertions at the start or end of a model."
            )

        # Create a deep copy of the model to ensure modifications do not affect the original
        object.__setattr__(self, "_model", deepcopy(model))
        # Freeze all parameters in the copied model
        for param in self._model.parameters():
            param.requires_grad = False

        # Split the model into pre-insert layers (up to the index)
        self.pre_inserts = nn.Sequential(*list(self._model.features.children())[:index])

        # Create scaled layers by applying the factory method `scale_layer`
        og_layer_to_scale = self._model.features[index]
        new_layers = [model.factory.scale_layer(og_layer_to_scale) for _ in range(num_scaled)]
        self.scaled_layers = nn.Sequential(*new_layers)  # Sequential container for scaled layers

        # Capture the replaceable layer at the index and split remaining layers as post-inserts
        self.replaceable = list(self._model.features.children())[index]
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
        return self._index  # Index property

    @property
    def num_scaled(self) -> Int:
        return self._num_scaled  # Number of scaled layers property

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        # Warn that `forward_scaled` is used by default
        warnings.warn(
            "The `forward` method calls the `forward_scaled` method. For vanilla forward, call `forward_raw`."
        )
        return self.forward_scaled(*args, **kwargs)

    def forward_raw(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the original model"""
        self.activations_dict.clear()  # Clear previous activations
        return self._model(x)  # Use the original model's forward method

    def forward_scaled(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the scaled model"""
        self.activations_dict.clear()  # Clear previous activations
        x = self.pre_inserts(x)  # Pass input through pre-insert layers
        x = self.scaled_layers(x)  # Pass through scaled layers
        x = self.replaceable(x)  # Pass through the replaceable layer
        return self.post_inserts(x)  # Pass through post-insert layers

    def hook_model(self, pre: Bool = False, scaled: Bool = False, post: Bool = False) -> None:
        """Hook selected layers to capture activations."""
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
            for idx, scaled_layer in enumerate(self.scaled_layers):
                self.hooks.append(self._model.factory.hook_layer(scaled_layer))

        # Hook the replaceable layer if `post` is True
        if post:
            self.hooks.append(self._model.factory.hook_layer(self.replaceable))

    def clear_hooks(self) -> None:
        """Remove all hooks and clear the hooks list."""
        for hook in self.hooks:
            hook.remove()  # Remove each hook
        self.hooks.clear()  # Clear the list of hooks

    def get_activations(self, detach: bool = True) -> OrderedDict:
        """Retrieve activations from all hooked layers."""
        activations = OrderedDict()

        # Collect activations from pre-insert layers
        for i, layer in enumerate(self.pre_inserts):
            if layer.is_hooked:  # Check if the layer is hooked
                activations[f"pre_insert_{i}"] = (
                    layer.forward_activations.detach() if detach else layer.forward_activations
                )

        # Collect activations from scaled layers
        for i, layer in enumerate(self.scaled_layers):
            if layer.is_hooked:  # Check if the layer is hooked
                activations[f"scaled_{i}"] = (
                    layer.forward_activations.detach() if detach else layer.forward_activations
                )

        # Collect activations from the replaceable layer
        if self.replaceable.is_hooked:  # Check if the layer is hooked
            activations["post_insert_0"] = (
                self.replaceable.forward_activations.detach()
                if detach
                else self.replaceable.forward_activations
            )

        return activations  # Return the collected activations

    @classmethod
    def load_model(cls, model: BaseTorchModel, file_path: str | Path) -> nn.Module:
        """Load a model from a file, including scaled layers and weights."""
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata to determine where to insert scaled layers
        metadata = safetensors_metadata_parser(file_path=file_path)

        # Create a ScaledModel instance using the parsed metadata
        scaled_model = cls(
            model=model,
            index=literal_eval(metadata["layer_start"]),
            num_scaled=literal_eval(metadata["added_layers"]),
        )

        # Load weights into the model using safetensors
        load_model(scaled_model, file_path, device=get_device())

        return scaled_model  # Return the loaded model

    def summary(self) -> torchinfo.ModelStatistics:
        """Generate a summary of the model structure and parameters."""
        return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])
