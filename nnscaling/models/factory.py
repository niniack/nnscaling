__all__ = ["LayerFactory", "LinearLayerFactory"]

from abc import ABC, abstractmethod
from copy import deepcopy

import torch.nn as nn

from nnscaling.models.layers import Layer, LinearLayer


class LayerFactory(ABC):
    @abstractmethod
    def create_layer(cls, bias: bool) -> Layer:
        pass

    @abstractmethod
    def scale_layer(cls, layer: Layer) -> Layer:
        pass

    @abstractmethod
    def init_weights(module: nn.Module):
        pass


class LinearLayerFactory(LayerFactory):
    @classmethod
    def create_layer(
        cls, bias: bool, in_features: int, out_features: int, nonlinearity: nn.Module
    ) -> LinearLayer:
        return LinearLayer(
            in_features=in_features,
            out_features=out_features,
            nonlinearity=nonlinearity,
            bias=bias,
        )

    @classmethod
    def scale_layer(cls, layer: LinearLayer) -> LinearLayer:
        kwargs = deepcopy(layer.kwargs)
        kwargs["out_features"] = kwargs["in_features"]
        return cls.create_layer(**kwargs)

    @staticmethod
    def init_weights(module: nn.Module):
        nn.init.kaiming_normal_(module.linear.weight, nonlinearity="relu")
        if module.linear.bias is not None:
            module.linear.bias.data.fill_(0.01)
