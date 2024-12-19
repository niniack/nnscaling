__all__ = ["LayerFactory", "LinearLayerFactory"]

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

import torch.nn as nn

from nnscaling.models.layers import Conv1DLayer, Conv2DLayer, Layer, LinearLayer


class LayerFactory(ABC):
    @abstractmethod
    def create_layer(cls, bias: bool) -> Layer:
        pass

    @abstractmethod
    def scale_layer(cls, layer: Layer) -> Layer:
        pass

    @abstractmethod
    def hook_layer(cls, layer: Layer) -> Any:
        pass

    @abstractmethod
    def init_weights(module: nn.Module):
        pass


class LinearLayerFactory(LayerFactory):
    @classmethod
    def create_layer(
        cls,
        in_features: int,
        out_features: int,
        nonlinearity: nn.Module,
        bias: bool,
        hook: bool,
        batchnorm: bool,
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
        if not last:
            kwargs["out_features"] = kwargs["in_features"]

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


class Conv2DLayerFactory(LayerFactory):
    @classmethod
    def create_layer(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        nonlinearity: nn.Module,
        hook: bool,
        batchnorm: bool,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> Conv2DLayer:
        return Conv2DLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            nonlinearity=nonlinearity,
            bias=bias,
            hook=hook,
            batchnorm=batchnorm,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    @classmethod
    def scale_layer(
        cls, layer: Conv2DLayer, last=False, batchnorm=False, nonlinearity=None
    ) -> None:
        raise NotImplementedError("Scaling is not yet implemented for convolutional layers!")

    @classmethod
    def hook_layer(cls, layer: Conv2DLayer) -> Any:
        return layer.setup_hook()

    @staticmethod
    def init_weights(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)


class Conv1DLayerFactory(LayerFactory):
    @classmethod
    def create_layer(
        cls,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        bias: bool,
        nonlinearity: nn.Module,
        hook: bool,
        batchnorm: bool,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
    ) -> Conv1DLayer:
        """
        Factory method for creating a Conv1DLayer.
        """
        return Conv1DLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            nonlinearity=nonlinearity,
            bias=bias,
            hook=hook,
            batchnorm=batchnorm,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    @classmethod
    def scale_layer(
        cls, layer: Conv1DLayer, last=False, batchnorm=False, nonlinearity=None
    ) -> None:
        """
        Placeholder for layer scaling logic, if needed.
        """
        raise NotImplementedError("Scaling is not yet implemented for convolutional layers!")

    @classmethod
    def hook_layer(cls, layer: Conv1DLayer) -> Any:
        """
        Sets up a forward hook on the given layer.
        """
        return layer.setup_hook()

    @staticmethod
    def init_weights(module: nn.Module):
        """
        Initializes the weights of the given module using He initialization for Conv1D layers.
        """
        if isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="leaky_relu")
            if module.bias is not None:
                module.bias.data.fill_(0.01)
