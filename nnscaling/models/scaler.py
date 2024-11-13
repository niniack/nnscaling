import warnings
from ast import literal_eval
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch.nn as nn
import torchinfo
from jaxtyping import Bool, Float, Int
from matrepr import mprint
from safetensors.torch import load_model
from torch import Tensor

from nnscaling.models import BaseTorchModel
from nnscaling.utils import get_device, safetensors_metadata_parser


class ScaledModel(nn.Module):
    """ """

    def __init__(self, model: BaseTorchModel, index: Int, num_scaled: Int):
        super().__init__()

        self._index = index
        self._num_scaled = num_scaled
        self.activations_dict = OrderedDict()
        self.hooks = []

        # Validate index
        if index == 0 or index == len(model.features):
            raise ValueError(
                "Scaling does not support layer insertions at the start or end of a model."
            )

        # Copy and freeze model
        object.__setattr__(self, "_model", deepcopy(model))
        for param in self._model.parameters():
            param.requires_grad = False

        # Frankenstein Part 1
        self.pre_inserts = nn.Sequential(*list(self._model.features.children())[:index])

        # Layer scaling
        og_layer_to_scale = self._model.features[index]
        new_layers = [None] * num_scaled
        for i in range(num_scaled):
            new_layers[i] = model.factory.scale_layer(og_layer_to_scale)
        self.scaled_layers = nn.Sequential(*new_layers)

        # Frankenstein Part 2
        self.replaceable = list(self._model.features.children())[index]
        self.post_inserts = nn.Sequential(*list(self._model.features.children())[index + 1 :])

    def __getattr__(self, name):
        # Prioritise locally defined attributes
        if name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]
        # Look in `model` for undefined attributes
        model = self.__dict__["_model"]
        return getattr(model, name)

    @property
    def index(self) -> Int:
        return self._index

    @property
    def num_scaled(self) -> Int:
        return self._num_scaled

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        warnings.warn(
            "The `forward` method calls the `forward_scaled` method. For vanilla forward, call `forward_raw`."
        )
        return self.forward_scaled(*args, **kwargs)

    def forward_raw(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the original model"""
        self.activations_dict.clear()
        return self._model(x)

    def forward_scaled(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """Forward pass through the scaled model"""
        self.activations_dict.clear()
        x = self.pre_inserts(x)
        x = self.scaled_layers(x)
        x = self.replaceable(x)
        return self.post_inserts(x)

    def hook_model(self, pre: Bool = True, scaled: Bool = True, post: Bool = True) -> None:
        # Clear out all previous hooks
        self.clear_hooks()

        # Hook function
        def _hook_fn(key):
            def hook(module, input, output):
                self.activations_dict[key] = output

            return hook

        # Hook last layer in pre_inserts
        if pre:
            pre_layers = [child for child in self.pre_inserts.children()]
            last_pre_layer = pre_layers[-1]
            self.hooks.append(
                last_pre_layer.register_forward_hook(_hook_fn(f"pre_insert_{len(pre_layers)-1}"))
            )

        # Hook all scaled layers
        if scaled:
            for idx, scaled_layer in enumerate(self.scaled_layers):
                self.hooks.append(scaled_layer.register_forward_hook(_hook_fn(f"scaled_{idx}")))

        # Hook first layer in post_inserts
        if post:
            replaceable_layer = self.replaceable
            self.hooks.append(replaceable_layer.register_forward_hook(_hook_fn(f"post_insert_{0}")))

    def clear_hooks(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    @classmethod
    def load_model(cls, model: BaseTorchModel, file_path: str | Path) -> nn.Module:
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = safetensors_metadata_parser(file_path=file_path)

        # Load model
        scaled_model = cls(
            model=model,
            index=literal_eval(metadata["layer_start"]),
            num_scaled=literal_eval(metadata["added_layers"]),
        )

        # Load weights
        load_model(scaled_model, file_path, device=get_device())

        return scaled_model

    def summary(self) -> torchinfo.ModelStatistics:
        return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])
