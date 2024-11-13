from collections import OrderedDict
from pathlib import Path
from typing import Optional

import einops
import numpy as np
import torch
from jaxtyping import Int
from safetensors.torch import save_file
from torch import Tensor, nn

from nnscaling.data import (
    DatasetConfig,
    get_dataset_class,
)


class TrajectoryEngine:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        scaled_model: nn.Module,
    ):
        self.dataset_config = dataset_config
        DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
        self.dataset = DatasetClass(config=dataset_config)
        self.scaled_model = scaled_model

    @torch.no_grad()
    def _run_forward(self):
        # Hook and forward pass on original model
        self.scaled_model.hook_model(pre=True, scaled=False, post=True)
        _ = self.scaled_model.forward_raw(self.dataset.features)
        raw_model_activations = self.scaled_model.activations_dict.copy()

        # Hook and forward pass on scaled model
        self.scaled_model.hook_model(pre=True, scaled=True, post=True)
        _ = self.scaled_model.forward_scaled(self.dataset.features)
        scaled_model_activations = self.scaled_model.activations_dict.copy()

        return raw_model_activations, scaled_model_activations

    def _collect_timestep_tensor(
        self,
        activations_dict: OrderedDict,
        magic_number: Int,
    ):
        values = list(activations_dict.values())
        dim = values[0].shape[-1]

        for i, v in enumerate(values):
            if v.shape[-1] != dim:
                temp = torch.full((v.shape[0], dim), magic_number)
                temp[:, : v.shape[-1]] = v
                values[i] = temp

        return torch.stack(values)

    def generate_trajectory(
        self,
        num_train: Int,
        num_test: Optional[Int] = None,
        seed: Int = 0,
        magic_number: Int = -5,
    ) -> dict[str, Tensor]:
        """
        Returns a dictionary with keys: raw_train, raw_test, scaled_train, scaled_test.
        Each key holds a tensor of trajectories shaped [layers, trajectories, neurons]
        (or [iterations, trajectories, states]).
        """

        # Set seed
        np.random.seed(seed)
        dataset_size = self.dataset_config.num_samples
        assert (
            num_train <= dataset_size
        ), "The number of train and test samples cannot exceed the size of the dataset!"

        if num_test is None:
            num_test = dataset_size - num_train

        assert (
            num_train + num_test <= dataset_size
        ), "The number of train and test samples cannot exceed the size of the dataset!"

        # Randomly pick indices for train and test dataset
        train_indices = np.random.choice(
            self.dataset.features.shape[0], size=num_train, replace=False
        )
        test_indices = np.random.choice(
            np.setdiff1d(np.arange(dataset_size), train_indices),
            size=num_test,
            replace=False,
        )

        raw_model_act_dict, scaled_model_act_dict = self._run_forward()
        raw_model_act = self._collect_timestep_tensor(
            activations_dict=raw_model_act_dict, magic_number=magic_number
        )
        scaled_model_act = self._collect_timestep_tensor(
            activations_dict=scaled_model_act_dict, magic_number=magic_number
        )

        return {
            "raw_train": einops.rearrange(raw_model_act[:, train_indices, :], "i b s -> b s i"),
            "raw_test": einops.rearrange(raw_model_act[:, test_indices, :], "i b s -> b s i"),
            "scaled_train": einops.rearrange(
                scaled_model_act[:, train_indices, :], "i b s -> b s i"
            ),
            "scaled_test": einops.rearrange(scaled_model_act[:, test_indices, :], "i b s -> b s i"),
        }

    @classmethod
    def save_trajectory(cls, traj_dict: dict[str, Tensor], save_dir: Path | str):
        # TODO: complete this method
        save_file(
            traj_dict,
            Path(save_dir, "trajectories.safetensors"),
        )


def reshape_append_time(x: torch.Tensor):
    return einops.rearrange(x, "batch state iteration -> state (batch iteration)")


# def reshape_append_state(x: torch.Tensor):
#     return einops.rearrange(x, "i t s -> (t s) i")


# def reshape_interleave_time(x: torch.Tensor):
#     return einops.rearrange(x, "i t s -> s (i t)")


# def reshape_interleave_state(x: torch.Tensor):
#     return einops.rearrange(x, "i t s -> (s t) i")
