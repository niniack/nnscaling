__all__ = ["Layer", "LinearLayer"]

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch.nn as nn


def is_from_module(obj, module_name):
    return obj.__module__ == module_name


class Layer(nn.Module, ABC):
    def __init__(self, nonlinearity: nn.Module, bias: bool):
        super().__init__()

        if nonlinearity is not None and not is_from_module(
            nonlinearity, "torch.nn.modules.activation"
        ):
            raise ValueError("Please use a valid activation function from torch.nn")

    @property
    @abstractmethod
    def kwargs(self) -> dict:
        pass


class LinearLayer(Layer):
    def __init__(
        self, in_features: int, out_features: int, nonlinearity: nn.Module, bias: bool
    ):
        super().__init__(nonlinearity, bias)

        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.activation = nonlinearity() if nonlinearity else None

        self._kwargs = OrderedDict(
            {
                "in_features": in_features,
                "out_features": out_features,
                "nonlinearity": nonlinearity,
                "bias": bias,
            }
        )

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x
