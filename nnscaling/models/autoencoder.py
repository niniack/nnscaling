__all__ = ["Autoencoder"]


from ast import literal_eval
from collections import OrderedDict, namedtuple
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from safetensors.torch import load_model
from torch import Tensor
from torch.distributions import Bernoulli, Categorical, Uniform

from nnscaling.models.base import BaseTorchModel
from nnscaling.models.factory import Conv2DLayerFactory, LinearLayerFactory
from nnscaling.utils import StringtoClassNonlinearity, get_device, safetensors_metadata_parser

AutoencoderResult = namedtuple("AutoencoderResult", "predictions reconstruction")


def eigeninit(weight: torch.Tensor, theta: float = 0.7) -> None:
    """
    Custom initialization function for Koopman matrix weights.
    Directly modifies the input tensor `weight` in-place.
    """
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eig(weight)
    polar_mags = torch.abs(eigenvalues)
    polar_phase = torch.angle(eigenvalues)

    # Sample with slab-spike distribution
    num_unique = len(torch.unique(polar_mags, sorted=False))
    bernoulli_trials = torch.distributions.Bernoulli(theta).sample([num_unique])
    uniform_trials = torch.distributions.Uniform(0, 1).sample([num_unique]) * (1 - bernoulli_trials)
    result_trials = bernoulli_trials + uniform_trials

    # Build new magnitudes, preserving pairs!
    new_polar_mags = torch.empty_like(polar_mags)
    new_polar_mags[0] = result_trials[0]
    j = 1
    for i in range(1, new_polar_mags.size(0)):
        if torch.isclose(polar_mags[i], polar_mags[i - 1]):
            new_polar_mags[i] = new_polar_mags[i - 1]
        else:
            new_polar_mags[i] = result_trials[j]
            j += 1

    # Rebuild eigenvalues
    new_eigenvalues = torch.polar(new_polar_mags, polar_phase)

    # Construct new weight matrix in-place
    with torch.no_grad():  # Ensure this works in-place without affecting autograd
        weight.copy_(
            torch.real(eigenvectors @ torch.diag(new_eigenvalues) @ torch.linalg.inv(eigenvectors))
        )


class Autoencoder(BaseTorchModel):
    """
    Autoencoder model.
    """

    def __init__(
        self,
        nonlinearity: nn.Module | str = "leakyrelu",
        in_features: int = 2,  # Number of input features to the model.
        observable_features: int = 2,  # Number of output features.
    ):
        super().__init__()

        self.linear_factory = LinearLayerFactory()
        self.in_features = in_features
        self.observable_features = observable_features
        nonlinearity = (
            StringtoClassNonlinearity[nonlinearity].value
            if isinstance(nonlinearity, str)
            else nonlinearity
        )
        self.nonlinearity = nonlinearity
        self.handles = []

        ######################
        # Encoder
        ######################
        channel_dims = [(in_features, in_features * 2), (in_features * 2, observable_features)]
        self._encoder = nn.Sequential()
        for i in range(0, len(channel_dims), 1):
            self._encoder.append(
                self.linear_factory.create_layer(
                    in_features=channel_dims[i][0],
                    out_features=channel_dims[i][1],
                    nonlinearity=nonlinearity,
                    batchnorm=False,
                    bias=True,
                    hook=False,
                )
            )
            self._encoder.apply(self.linear_factory.init_weights)

        ######################
        # Koopman matrix
        ######################
        self._koopman_matrix = self.linear_factory.create_layer(
            in_features=observable_features,
            out_features=observable_features,
            nonlinearity=None,
            batchnorm=False,
            bias=False,
            hook=False,
        )
        eigeninit(self._koopman_matrix.linear.weight, theta=0.7)

        ######################
        # Decoder
        ######################
        self._decoder = nn.Sequential()
        for i in range(len(channel_dims) - 1, -1, -1):
            self._decoder.append(
                self.linear_factory.create_layer(
                    in_features=channel_dims[i][1],
                    out_features=channel_dims[i][0],
                    nonlinearity=nonlinearity if (i != 0) else None,
                    batchnorm=False,
                    bias=True,
                    hook=False,
                )
            )
            self._decoder.apply(self.linear_factory.init_weights)

    @property
    def encoder(self) -> nn.Sequential:
        return self._encoder

    @property
    def decoder(self) -> nn.Sequential:
        return self._decoder

    @property
    def koopman_matrix(self) -> nn.Sequential:
        return self._koopman_matrix

    @property
    def features(self) -> nn.Sequential:
        # Combine encoder and decoder modules
        return nn.Sequential(self._koopman_matrix, *(list(self._encoder) + list(self._decoder)))

    def hook_model(self):
        # Remove all previous hooks
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

        for layer in self.features:
            # if not layer.is_hooked:
            self.handles.append(self.conv2d_factory.hook_layer(layer))

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
        k: Int = 0,
    ):
        # Pre-encoder bias
        x_bar = x - self.decoder[-1].linear.bias

        # Encode
        phi_x = self.encoder(x_bar)

        # Reconstruct
        x_recons = self.decoder(phi_x)

        # Advance latent variable k times
        prediction = [phi_x]
        for i in range(1, k + 1):
            prev_pred = prediction[i - 1]
            prediction.append(self.koopman_matrix(prev_pred))

        ## Batched decoding
        # Shape: [steps, batch, latent_dim]
        stacked_predictions = torch.stack(prediction, dim=0)
        steps, batch_size, latent_dim = stacked_predictions.size()
        # Shape: [steps * batch, feature_dim]
        reshaped_predictions = stacked_predictions.view(-1, latent_dim)
        decoded = self.decoder(reshaped_predictions)
        # Shape: [steps, batch, latent_dim]
        x_k = decoded.view(steps, batch_size, -1)

        return AutoencoderResult(x_k, x_recons)

    @classmethod
    def load_model(cls, file_path: str | Path, **kwargs):
        assert Path(file_path).exists(), f"Model file {file_path} does not exist."

        # Parse metadata
        metadata = safetensors_metadata_parser(file_path=file_path)
        if metadata["autoencoder_nonlinearity"] == "None":
            nonlinearity = None
        else:
            nonlinearity = metadata["autoencoder_nonlinearity"]

        # Load base model
        model = cls(
            nonlinearity=nonlinearity,
            in_features=literal_eval(metadata["in_features"]),
            observable_features=literal_eval(metadata["observable_features"]),
            **kwargs,
        )

        # Load weights
        load_model(model, file_path, device=get_device())

        return model
