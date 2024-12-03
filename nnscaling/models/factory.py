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
    def hook_layer(cls, layer: Layer) -> Layer:
        pass

    @abstractmethod
    def init_weights(module: nn.Module):
        pass


class LinearLayerFactory(LayerFactory):
    @classmethod
    def create_layer(
        cls,
        bias: bool,
        in_features: int,
        out_features: int,
        nonlinearity: nn.Module,
        hook: bool,
        batchnorm: bool = False,
    ) -> LinearLayer:
        return LinearLayer(
            in_features=in_features,
            out_features=out_features,
            nonlinearity=nonlinearity,
            bias=bias,
            hook=hook,
            batchnorm=batchnorm,
        )

    @classmethod
    def scale_layer(
        cls, layer: LinearLayer, last=False, batchnorm=False, nonlinearity=None
    ) -> LinearLayer:
        kwargs = deepcopy(layer.kwargs)
        if nonlinearity:
            kwargs["nonlinearity"] = nonlinearity

        # TODO: Depends on scaling, the scaled layers could include different dimensionality
        kwargs["out_features"] = kwargs["in_features"]
        # if not last:
        #     kwargs["out_features"] = kwargs["in_features"]
        # If `last` set in_features to out_features
        # if last:
        #     kwargs["in_features"] = kwargs["out_features"]
        return cls.create_layer(**kwargs, hook=False, batchnorm=batchnorm)

    @classmethod
    def hook_layer(cls, layer: Layer) -> Layer:
        return layer.setup_hook()

    @staticmethod
    def init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)
