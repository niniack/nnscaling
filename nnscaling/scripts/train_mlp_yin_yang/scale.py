import os
import sys
from ast import literal_eval
from pathlib import Path
from typing import Optional

import fire
import torch
import wandb
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)
from safetensors.torch import load_model, save_model
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nnscaling.data import (
    DatasetConfig,
    YinYangNoDotsBinaryDataset,
    create_data_loader,
)
from nnscaling.log import logger
from nnscaling.models import MLP, BaseTorchModel, ScaledModel
from nnscaling.scripts.common import load_config
from nnscaling.utils import get_device, safetensors_metadata_parser, set_seed


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    seed: NonNegativeInt = 0
    train_data: DatasetConfig
    batch_size: PositiveInt
    learning_rate: PositiveFloat
    weight_decay: PositiveFloat
    out_features: PositiveInt
    print_freq: PositiveInt
    save_dir: str | None
    model_to_scale: str
    scale_location: PositiveInt
    num_scaled_layers: PositiveInt
    betas: list[PositiveFloat] | None = None
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

        model.hook_model(post=True)
        with torch.no_grad():
            _ = model.forward_raw(input)
            target_activation = model.activations_dict["post_insert_0"]
        _ = model.forward_scaled(input)
        output_activation = model.activations_dict["post_insert_0"]

        loss = criterion(output_activation, target_activation)
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
        wandb_config_dict.update(
            {"betas": [wandb_config_dict["beta_one"], wandb_config_dict["beta_two"]]}
        )
        del wandb_config_dict["beta_one"], wandb_config_dict["beta_two"]

    # Load config
    config = load_config(
        config_path_or_obj or wandb_config_dict,
        config_model=Config,
    )
    logger.info(config)

    # Setup
    device = get_device()
    set_seed(config.seed)

    # Load data
    dataset_config = config.train_data
    dataset = YinYangNoDotsBinaryDataset(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    # Load model
    metadata = safetensors_metadata_parser(file_path=config.model_to_scale)
    unscaled_model = MLP(
        config=literal_eval(metadata["config"]), nonlinearity=metadata["nonlinearity"]
    )
    _ = load_model(unscaled_model, config.model_to_scale, device=device)
    model = ScaledModel(
        model=unscaled_model,
        index=config.scale_location,
        num_scaled=config.num_scaled_layers,
    )
    model.summary()
    model.to(device).train()

    all_param_names = [name for name, _ in model.named_parameters()]
    assert len(all_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable layers: {len(all_param_names)}")

    # Define loss and optimiser
    loss = nn.HuberLoss(delta=1)
    optimizer = optim.AdamW(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=tuple(config.betas) or (0.9, 0.999),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=500, threshold=0.0001, verbose=True
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

        scheduler.step(train_loss)
        # Log metrics
        wandb.log({"epoch": epoch, "train_loss": train_loss})

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {train_loss:.4f}")

    if config.save_dir:
        metadata_dict = {
            "config": str(metadata["config"]),
            "dataset": dataset.name(),
            "nonlinearity": str(metadata["nonlinearity"]),
            "added_layers": str(config.num_scaled_layers),
            "layer_start": str(config.scale_location),
        }
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        save_model(
            model,
            Path(
                config.save_dir,
                f"loc_{config.scale_location}_scaled_yinyang_model.safetensors",
            ),
            metadata=metadata_dict,
        )


if __name__ == "__main__":
    fire.Fire(main)
