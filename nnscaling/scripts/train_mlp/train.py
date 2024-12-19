import os
import pdb
import sys
from pathlib import Path
from typing import Optional

import fire
import torch
import wandb
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from safetensors.torch import save_model
from torch import nn, optim
from torch.utils.data import DataLoader

from nnscaling.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from nnscaling.log import logger
from nnscaling.models import MLP, BaseTorchModel
from nnscaling.scripts.common import load_config
from nnscaling.utils import get_device, set_seed


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt = 0
    train_data: DatasetConfig
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: NonNegativeFloat
    out_features: PositiveInt
    print_freq: PositiveInt
    hidden_neurons: list[PositiveInt]
    save_dir: str | None
    save_name: str | None
    num_epochs: PositiveInt | None = None


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


def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Initialize wandb
    wandb.init(entity="nishantaswani", project="scaling")

    # Validate
    if config_path_or_obj is None and not dict(wandb.config):
        sys.exit("No configuration found for the run! Provide a file")

    # Update wandb config
    wandb_config_dict = dict(wandb.config)
    if wandb_config_dict:
        wandb_config_dict.update({"save_dir": None, "print_freq": 500})

    config = load_config(
        config_path_or_obj or wandb_config_dict,
        config_model=Config,
    )
    logger.info(config)

    device = get_device()
    set_seed(config.seed)

    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    model = MLP(
        config=config.hidden_neurons,
        in_features=dataset.in_features,
        out_features=config.out_features,
        nonlinearity=nn.ReLU,
    )
    model.to(device).train()

    all_param_names = [name for name, _ in model.named_parameters()]
    assert len(all_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable parameters: {len(all_param_names)}")
    # model.summary()

    # Define loss and optimiser
    loss = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Loop over epochs
    for epoch in range(config.num_epochs):
        # Train step
        train_loss = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            criterion=loss,
            optimizer=optimizer,
        )

        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}")

    if config.save_dir:
        metadata_dict = {
            "config": str(config.hidden_neurons),
            "dataset": dataset.name(),
            "out_features": str(config.out_features),
            "nonlinearity": "relu",
        }
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        save_model(
            model,
            Path(config.save_dir, f"{config.save_name}.safetensors"),
            metadata=metadata_dict,
        )


if __name__ == "__main__":
    fire.Fire(main)
