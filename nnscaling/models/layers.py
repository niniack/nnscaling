__all__ = ["Layer", "LinearLayer"]

from abc import ABC, abstractmethod
from collections import OrderedDict

import torch.nn as nn
from torch import Tensor


def is_from_module(obj, module_name):
    return obj.__module__ == module_name


class Layer(nn.Module, ABC):
    def __init__(self, nonlinearity: nn.Module, bias: bool, hook: bool):
        super().__init__()

        if nonlinearity is not None and not is_from_module(
            nonlinearity, "torch.nn.modules.activation"
        ):
            raise ValueError("Please use a valid activation function from torch.nn")

    @property
    @abstractmethod
    def is_hooked(self) -> bool:
        pass

    @property
    @abstractmethod
    def kwargs(self) -> dict:
        pass

    @property
    @abstractmethod
    def out_features(self) -> dict:
        pass

    @property
    @abstractmethod
    def forward_activations(self) -> Tensor:
        pass


class LinearLayer(Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        nonlinearity: nn.Module,
        bias: bool,
        hook: bool,
        batchnorm: bool = False,
    ):
        super().__init__(nonlinearity, bias, hook)

        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        self.batchnorm_layer = nn.BatchNorm1d(out_features) if batchnorm else None
        self.activation_layer = nonlinearity() if nonlinearity is not None else None

        self._kwargs = OrderedDict(
            {
                "in_features": in_features,
                "out_features": out_features,
                "nonlinearity": nonlinearity,
                "bias": bias,
            }
        )

        self._forward_activations = None

        if hook:
            self.setup_hook()

    @property
    def kwargs(self) -> dict:
        return self._kwargs

    @property
    def out_features(self) -> dict:
        return self._kwargs["out_features"]

    @property
    def is_hooked(self) -> bool:
        if self.activation_layer and len(self.activation_layer._forward_hooks) > 0:
            return True
        if len(self.linear._forward_hooks) > 0:
            return True
        return False

    @property
    def forward_activations(self) -> Tensor:
        return self._forward_activations

    def _hook(self, module, input, output):
        self._forward_activations = output

    def setup_hook(self):
        self._kwargs["hook"] = True
        if self.activation_layer:
            handle = self.activation_layer.register_forward_hook(self._hook)
        else:
            handle = self.linear.register_forward_hook(self._hook)

        return handle  # Return the handle

    def forward(self, x):
        self._forward_activations = -1
        x = self.linear(x)
        if self.batchnorm_layer:
            x = self.batchnorm_layer(x)
        if self.activation_layer:
            x = self.activation_layer(x)
        return x
