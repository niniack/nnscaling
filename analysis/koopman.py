import sys
import warnings
from typing import TYPE_CHECKING, Any

import einops
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor, nn

if TYPE_CHECKING:
    from pykoopman import Koopman

T_DEFAULT = object()


class KoopmanWrapper(nn.Module):
    def __init__(self, scaled_model, dmd_model: "Koopman", n_delays: Int):
        super().__init__()
        self._scaled_model = scaled_model
        self._dmd_model = dmd_model
        self._n_delays = n_delays
        self._tde = TorchDelayEmbedder(delay=1, n_delays=n_delays)

        self.tensor_A = torch.from_numpy(dmd_model.A).float()
        self.tensor_C = torch.from_numpy(dmd_model.C).float()

    def simulate(
        self, x0: Float[Tensor, "batch state iteration"], n_steps: Int = 1
    ) -> Float[Tensor, "batch state iteration"]:
        """
        A drop-in replacement of PyKoopman's `simulate` method, but with batched inputs!
        """
        if isinstance(x0, torch.Tensor):
            x0 = x0.numpy()

        # NOTE: Time-delay embedding (TDE) on the batched input tensor
        # returns tensor `Hx` with shape [batch iteration feature],
        # where feature is a spatio-temporal dimension.
        # PyKooopman's time delay observable is NOT batched; it returns [feature iteration].
        x0 = x0[:, :, : self._n_delays + 1]
        tensor_obs = self._tde(x0).float()

        # NOTE: The original implementation computes U.T @ Hx.
        # However, because X is batched, we compute (U.T @ H.x).T = H.x.T @ U
        # taking a page from the F.linear book. This keeps the batch dimension intact
        tensor_ur = tensor_obs @ torch.from_numpy(self._dmd_model.regressor.ur).float()

        # Flatten to [batch, RSV]
        # y = tensor_ur.T.flatten(start_dim=0, end_dim=1).T
        y = tensor_ur.flatten(start_dim=1)

        y_steps = []
        # Step forward and return to original space
        # NOTE: We use the transpose matmult to preserve batch
        for _ in range(n_steps):
            y = y @ self.tensor_A.T
            y_steps.append(y)

        # Return back from observable space
        y = torch.stack(y_steps, dim=0) @ self.tensor_C.T

        # Rearrange for standard order
        y = einops.rearrange(y, "iteration batch state -> batch state iteration")

        return y

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        warnings.warn("The `forward` method calls the `forward_koopman` method.")
        return self.forward_koopman(*args, **kwargs)

    def forward_koopman(self, x: Float[Tensor, "batch features"]) -> Tensor:
        """
        Forward pass through the Koopman model.
        """
        # Hook forward pass
        self._scaled_model.hook_model(pre=True, scaled=True, post=False)

        # Forward on scaled
        self._scaled_model.eval()
        _ = self._scaled_model.forward_scaled(x)

        # Grab activations and rearrange
        scaled_model_activations = self._scaled_model.get_activations().copy()

        # NOTE: If we change how we scale, then we have to pop certain components from the activations
        # Alternatively, we can change arguments in `hook_model`
        # Remove last layer
        if self._scaled_model.scale_style == "replace":
            _, acts_to_predict = scaled_model_activations.popitem(last=True)
        # Stack activations and rearrange
        scaler_acts = torch.stack(list(scaled_model_activations.values())).detach()
        scaler_acts = einops.rearrange(
            scaler_acts, "iteration batch state -> batch state iteration"
        )

        # Run through DMD model
        n_iterations = scaler_acts.shape[-1]
        x_koopman = self.simulate(x0=scaler_acts, n_steps=n_iterations - self._n_delays)
        valid_states = (
            self._scaled_model.scalable_layer.out_features
        )  # TODO: improve this, it feels a bit brittle
        x_koopman = x_koopman[:, :valid_states, -1]

        # NOTE: This is the baseline! It must beat the original linear + batchnorm
        # x_koopman = self._scaled_model.scaling_layers[-1].linear(scaler_acts[:, :, -1])
        # x_koopman = self._scaled_model.scaling_layers[-1].batchnorm_layer(x_koopman)

        # Run through post inserts
        scaled_koopman_out = self._scaled_model.post_inserts(x_koopman)

        return scaled_koopman_out


class TorchDelayEmbedder:
    """
    A class for creating time-delay observables. These observables are formed by
    taking time-lagged measurements of state variables and interpreting them as new
    state variables.
    """

    def __init__(self, delay=1, n_delays=2):
        """
        Initialize the TimeDelay class with given parameters.

        Args:
            delay (int, optional): The length of each delay. Defaults to 1. Or
                we say this is the "stride of delay".
            n_delays (int, optional): The number of delays to compute for each
                variable. Defaults to 2.

        Raises:
            ValueError: If delay or n_delays are negative.
        """
        if delay < 0:
            raise ValueError("delay must be a nonnegative int")
        if n_delays < 0:
            raise ValueError("n_delays must be a nonnegative int")

        self.include_state = True
        self.delay = int(delay)
        self.n_delays = int(n_delays)
        self._n_consumed_samples = self.delay * self.n_delays

    def __call__(
        self, x: Float[Tensor, "batch state iteration"]
    ) -> Float[Tensor, "batch iteration feature"]:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if x.ndim == 3:
            # For readability
            pass
        elif x.ndim == 2:
            warnings.warn(
                """
                A 2D input was expanded to include a batch dimension.
                Typically, the delay embedder expects an input tensor of 
                shape [n_batch, n_states, n_iters].
                """
            )
            # Expand batch dimensison
            x = x.unsqueeze(0)

        else:
            raise ValueError(
                "The delay embedder expects an input tensor of shape [n_batch, n_states, n_iters]"
            )
        n_batch, n_states, n_iters = x.shape

        # Size of each row
        unfold_size = n_iters - self._n_consumed_samples
        if unfold_size <= 0:
            raise ValueError(
                f"Bad delay parameter selection! You need at least {(self.delay * self.n_delays) + 1} samples"
            )

        # Unfold in iterations (second) dimension
        x = x.unfold(
            dimension=2,
            size=unfold_size,
            step=self.delay,
        )

        # Rearrange to achieve spatio-temporal mix
        x = einops.rearrange(x, "batch state delay iteration -> batch delay state iteration")

        # To mimic PyKoopman, reverse the delay dimension to have later slices first
        x = torch.flip(x, dims=(1,))

        # Collapse middle two dimensions and move to end
        x = einops.rearrange(x, "batch delay state iteration -> batch iteration (delay state) ")

        return x

    def fit(self, x):
        """
        Don't really worry about this for now. We never use it. Fit the model to measurement data.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)

        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            raise ValueError("Input must be a 2D Tensor.")

        x = x.T
        n_samples, n_features = x.shape

        self.n_input_features_ = n_features
        self.n_output_features_ = n_features * (1 + self.n_delays)

        self.measurement_matrix_ = torch.zeros((self.n_input_features_, self.n_output_features_))
        self.measurement_matrix_[: self.n_input_features_, : self.n_input_features_] = torch.eye(
            self.n_input_features_
        )

        return
