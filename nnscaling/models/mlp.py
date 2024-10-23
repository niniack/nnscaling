__all__ = ["MLP"]


import torch.nn as nn
from jaxtyping import Float
from torch import Tensor

from nnscaling.models.base import BaseTorchModel
from nnscaling.models.factory import LinearLayerFactory


class MLP(BaseTorchModel):
    """
    Multi-layer perceptron.
    """

    def __init__(
        self,
        in_features: int = 2,  # Number of input features to the model.
        config: list = [8],  # Number of neurons per hidden layer.
        out_features: int = 2,  # Number of output features.
        nonlinearity: nn.Module = nn.ReLU,
        bias: bool = True,
    ):
        super().__init__()

        self.factory = LinearLayerFactory()
        self.in_features = in_features
        self.out_features = out_features
        self.nonlinearity = nonlinearity
        self.bias = bias

        full_config = [in_features, *config, out_features]

        layers = [None] * (len(full_config) - 1)
        for i in range(len(full_config) - 1):
            layers[i] = self.factory.create_layer(
                in_features=full_config[i],
                out_features=full_config[i + 1],
                nonlinearity=nonlinearity if not i == len(full_config) - 2 else None,
                bias=bias,
            )
            self.factory.init_weights(layers[i])
        self._features = nn.Sequential(*layers)

    @property
    def features(self) -> nn.Sequential:
        return self._features

    def forward(
        self,
        x: Float[Tensor, "batch features"],
    ):
        return self.features(x)
