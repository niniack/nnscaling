__all__ = ["ScaledModel"]

from copy import deepcopy

import torch.nn as nn
import torchinfo
from jaxtyping import Float
from torch import Tensor

from nnscaling.models.mlp import BaseTorchModel


class ScaledModel(nn.Module):
    """ """

    def __init__(self, model: BaseTorchModel, index: int, num_scaled: int):
        super().__init__()

        self.index = index
        self.num_scaled = num_scaled

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
        self.post_inserts = nn.Sequential(
            *list(self._model.features.children())[index:]
        )

    def __getattr__(self, name):
        # Prioritise locally defined attributes
        if name in self.__dict__["_modules"]:
            return self.__dict__["_modules"][name]
        # Look in `model` for undefined attributes
        model = super().__getattr__("model")
        return getattr(model, name)

    def forward_raw(self, x: Float[Tensor, "batch features"]):
        return self._model(x)

    def forward(self, x: Float[Tensor, "batch features"]):
        x = self.pre_inserts(x)
        x = self.scaled_layers(x)
        return self.post_inserts(x)

    def summary(self) -> torchinfo.ModelStatistics:
        return torchinfo.summary(self)
