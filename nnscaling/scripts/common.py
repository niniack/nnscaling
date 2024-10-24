from pathlib import Path
from typing import TypeVar

import torch
import yaml
from pydantic import BaseModel
from torch.utils.data import DataLoader

from nnscaling.models import BaseTorchModel

T = TypeVar("T", bound=BaseModel)


def train_one_epoch(
    model: BaseTorchModel,
    train_loader: DataLoader,
    device: torch.device,
    criterion: torch.nn,
    optimizer: torch.nn,
) -> None:
    model.train()
    loss_epoch = 0

    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        label = label.squeeze()

        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, label.long())
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()  # Accumulate loss

    epoch_loss = loss_epoch / len(train_loader)  # Average loss over all batches
    return epoch_loss


def load_config(config_path_or_obj: Path | str | T | dict, config_model: type[T]) -> T:
    """Load the config of class `config_model`, either from YAML file or existing config object.
    https://github.com/ApolloResearch/e2e_sae/blob/main/e2e_sae/utils.py

    Args:
        config_path_or_obj (Union[Path, str, `config_model`]): if config object, must be instance
            of `config_model`. If str or Path, this must be the path to a .yaml.
        config_model: the class of the config that we are loading
    """
    if isinstance(config_path_or_obj, config_model):
        return config_path_or_obj

    if isinstance(config_path_or_obj, dict):
        return config_model(**config_path_or_obj)

    if isinstance(config_path_or_obj, str):
        config_path_or_obj = Path(config_path_or_obj)

    assert isinstance(
        config_path_or_obj, Path
    ), f"passed config is of invalid type {type(config_path_or_obj)}"
    assert (
        config_path_or_obj.suffix == ".yaml"
    ), f"Config file {config_path_or_obj} must be a YAML file."
    assert Path(
        config_path_or_obj
    ).exists(), f"Config file {config_path_or_obj} does not exist."
    with open(config_path_or_obj) as f:
        config_dict = yaml.safe_load(f)

    return config_model(**config_dict)
