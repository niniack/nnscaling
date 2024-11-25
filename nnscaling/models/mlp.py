__all__ = ["MLP"]


from ast import literal_eval
from collections import OrderedDict
from enum import Enum
from pathlib import Path

import torch.nn as nn
from jaxtyping import Float, Int
from safetensors.torch import load_model
from torch import Tensor

from nnscaling.models.base import BaseTorchModel
from nnscaling.models.factory import LinearLayerFactory
from nnscaling.utils import get_device, safetensors_metadata_parser


class NonlinearityLookup(Enum):
    relu = nn.ReLU


class MLP(BaseTorchModel):
    """
    Multi-layer perceptron.
    """

    def __init__(
        self,
        in_features: int = 2,  # Number of input features to the model.
        config: list = [8],  # Number of neurons per hidden layer.
        out_features: int = 2,  # Number of output features.
        nonlinearity: nn.Module | str = nn.ReLU,
        bias: bool = True,
    ):
        super().__init__()

        self.factory = LinearLayerFactory()
        self.in_features = in_features
        self.out_features = out_features
        nonlinearity = (
            NonlinearityLookup[nonlinearity].value
            if isinstance(nonlinearity, str)
            else nonlinearity
        )
        self.nonlinearity = nonlinearity
        self.bias = bias
        self.handles = []
        full_config = [in_features, *config, out_features]

        layers = [None] * (len(full_config) - 1)
        for i in range(len(full_config) - 1):
            layers[i] = self.factory.create_layer(
                in_features=full_config[i],
                out_features=full_config[i + 1],
                nonlinearity=nonlinearity if not i == len(full_config) - 2 else None,
                bias=bias,
                hook=False,
            )
            layers[i].apply(self.factory.init_weights)
        self._features = nn.Sequential(*layers)

    @property
    def features(self) -> nn.Sequential:
        return self._features

    def hook_model(self):
        # Remove all previous hooks
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        for layer in self.features:
            # if not layer.is_hooked:
            self.handles.append(self.factory.hook_layer(layer))

    def get_activations(self, detach=True) -> OrderedDict:
        activations = OrderedDict()
        for i, layer in enumerate(self.features):
            if layer.is_hooked:
                activations[i] = (
                    layer.forward_activations if not detach else layer.forward_activations.detach()
                )

        return activations

    def forward(
        self,
        x: Float[Tensor, "batch features"],
    ):
        return self.features(x)

    @classmethod
    def load_model(cls, file_path: str | Path, in_features: Int):
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = safetensors_metadata_parser(file_path=file_path)
        print(metadata)

        # Load base model
        model = cls(
            config=literal_eval(metadata["config"]),
            in_features=in_features,
            out_features=literal_eval(metadata["out_features"]),
            nonlinearity=nn.ReLU if metadata["nonlinearity"] == "relu" else ValueError(),
        )

        # Load weights
        load_model(model, file_path, device=get_device())

        return model
