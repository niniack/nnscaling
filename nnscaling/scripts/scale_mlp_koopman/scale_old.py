import os
import pdb
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Literal, Optional

import fire
import torch
import torch.nn.functional as F
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nnscaling.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from nnscaling.log import logger
from nnscaling.models import MLP, Autoencoder, ScaledModel
from nnscaling.scripts.common import load_config
from nnscaling.utils import (
    StringtoClassNonlinearity,
    get_device,
    safetensors_metadata_parser,
    set_seed,
)


# OptimConfig for optimizer-related parameters
class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    learning_rate: PositiveFloat
    betas: list[PositiveFloat] | None = None
    weight_decay: NonNegativeFloat
    num_epochs: PositiveInt | None = None
    batch_size: PositiveInt


# ScaleConfig for scale-related parameters
class ScaleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    model_to_scale: str
    scale_location: PositiveInt
    num_scaled_layers: PositiveInt
    scale_nonlinearity: str
    scale_style: Literal["replace", "prepend"]
    lambda_reconstruction: NonNegativeFloat
    lambda_prediction: NonNegativeFloat


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    scale: ScaleConfig
    ae_dim: PositiveInt
    print_freq: PositiveInt
    seed: NonNegativeInt = 0
    save_name: Optional[str] = None
    save_dir: Optional[str] = None


def train_one_epoch(
    model: nn.Module,
    autoencoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> float:
    model.train()
    model.to(device)
    criterion = nn.MSELoss()
    scalable_params = list(model.scalable_layer.parameters())

    def pad_act(x):
        target_size = autoencoder.encoder[0].in_features
        current_size = x.size(1)
        if current_size < target_size:
            pad_size = target_size - current_size
            x = F.pad(x, (0, pad_size), mode="constant", value=0)

        return x

    def compute_recons_loss(activations: OrderedDict):
        # shaped: [iterations, batch, act_size]
        original_acts = torch.stack([pad_act(act) for act in activations.values()])
        recons_acts = torch.stack([autoencoder(x=act, k=0).reconstruction for act in original_acts])
        recons_loss = F.mse_loss(original_acts, recons_acts, reduction="sum")
        return recons_loss

    def compute_prediction_loss(activations: OrderedDict):
        # shaped: [iterations, batch, act_size]
        original_acts = torch.stack([pad_act(act) for act in activations.values()])

        predictions = OrderedDict()
        for i, key in enumerate(activations.keys()):
            k_preds = autoencoder(x=original_acts[i], k=(len(original_acts) - (i + 1))).predictions
            predictions[key] = k_preds

        prediction_loss = torch.tensor(0.0)
        for k_preds in predictions.values():
            k = k_preds.shape[0]
            prediction_loss += F.mse_loss(original_acts[-k:], k_preds, reduction="sum")

        return prediction_loss

    loss_epoch = 0
    recons_loss_epoch = 0
    prediction_loss_epoch = 0

    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        label = label.squeeze()

        optimizer.zero_grad()
        model.hook_model(pre=True, scaled=True, post=True)

        # Forward pass for target and output activations
        with torch.no_grad():
            _ = model.forward_raw(input)
            target_activation = model.get_activations()["post_insert_0"]

        _ = model.forward_scaled(input)

        if config.scale.scale_style == "replace":
            output_activation = model.get_activations(detach=False)[
                f"scaled_{config.scale.num_scaled_layers-1}"
            ]
        elif config.scale.scale_style == "prepend":
            output_activation = model.get_activations(detach=False)["post_insert_0"]

        # Compute the primary loss
        loss = criterion(output_activation, target_activation)

        # Compute regularization terms
        scaled_activations = model.get_activations(detach=False)
        scaled_activations.popitem(last=True)
        recons_loss = compute_recons_loss(scaled_activations)
        prediction_loss = compute_prediction_loss(scaled_activations)

        # Combine total loss
        loss += (
            config.scale.lambda_reconstruction * recons_loss
            + config.scale.lambda_prediction * prediction_loss
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses
        loss_epoch += loss.item()
        recons_loss_epoch += recons_loss.item()
        prediction_loss_epoch += prediction_loss.item()

    # Average losses over all batches
    epoch_loss = loss_epoch / len(train_loader)
    avg_recons_loss = recons_loss_epoch / len(train_loader)
    avg_prediction_loss = prediction_loss_epoch / len(train_loader)

    # Return all metrics as a dictionary
    return {
        "epoch_loss": epoch_loss,
        "recons_loss": avg_recons_loss,
        "prediction_loss": avg_prediction_loss,
    }


def main(config_path_or_obj: Optional[Path | str | Config] = None):
    # Initialize wandb
    wandb.init(entity="nishantaswani", project="scaling")

    # Validate
    if config_path_or_obj is None and not dict(wandb.config):
        sys.exit("No configuration found for the run! Provide a file")

    # Update wandb config
    wandb_config_dict = dict(wandb.config)
    if wandb_config_dict:
        wandb_config_dict.update({"save_dir": None, "save_name": None, "print_freq": 500})
        wandb_config_dict.update(
            {"betas": [wandb_config_dict["beta_one"], wandb_config_dict["beta_two"]]}
        )
        del wandb_config_dict["beta_one"], wandb_config_dict["beta_two"]

    # Load config
    logger.info(wandb_config_dict)
    logger.info(config_path_or_obj)
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
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.optim.batch_size, global_seed=config.seed
    )

    # Load model
    metadata = safetensors_metadata_parser(file_path=config.scale.model_to_scale)
    unscaled_model = MLP.load_model(
        file_path=config.scale.model_to_scale,
        in_features=dataset.in_features,
    )
    unscaled_model.eval()
    model = ScaledModel(
        model=unscaled_model,
        index=config.scale.scale_location,
        nonlinearity=StringtoClassNonlinearity[config.scale.scale_nonlinearity].value,
        scale_style=config.scale.scale_style,
        num_scaled=config.scale.num_scaled_layers,
        batchnorm=True,
    )
    model.summary()
    model.to(device).train()

    # Build autoencoder
    ae_nonlinearity = "relu"
    autoencoder = Autoencoder(
        nonlinearity=ae_nonlinearity,
        in_features=model.scalable_layer.in_features,
        observable_features=config.ae_dim,
    )
    autoencoder.summary()
    autoencoder.to(device).train()

    # Define optimiser and scheduler
    optimizer = optim.AdamW(
        params=list(model.parameters()) + list(autoencoder.parameters()),
        lr=config.optim.learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=tuple(config.optim.betas) or (0.9, 0.999),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=500, threshold=0.001, verbose=True
    )

    # Loop over epochs
    for epoch in range(config.optim.num_epochs):
        # Train step
        losses = train_one_epoch(
            model=model,
            autoencoder=autoencoder,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            config=config,
        )

        scheduler.step(losses["epoch_loss"])

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": losses["epoch_loss"],
                "recons_loss": losses["recons_loss"],
                "prediction_loss": losses["prediction_loss"],
            }
        )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.optim.num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
                f"Reconstruction Loss: {losses['recons_loss']:.4f}, "
                f"Prediction Loss: {losses['prediction_loss']:.4f}"
            )

    if config.save_dir:
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)

        metadata_dict = {
            "config": str(metadata["config"]),
            "dataset": dataset.name(),
            "out_features": str(unscaled_model.out_features),
            "nonlinearity": str(metadata["nonlinearity"]),
            "scale_style": str(config.scale.scale_style),
            "scale_nonlinearity": str(config.scale.scale_nonlinearity),
            "added_layers": str(config.scale.num_scaled_layers),
            "layer_start": str(config.scale.scale_location),
        }
        save_model(
            model,
            Path(
                config.save_dir,
                f"loc_{config.scale.scale_location}_scaled_{config.save_name}.safetensors",
            ),
            metadata=metadata_dict,
        )

        metadata_dict = {
            "autoencoder_nonlinearity": str(ae_nonlinearity),
            "in_features": str(autoencoder.in_features),
            "observable_features": str(autoencoder.observable_features),
        }
        save_model(
            autoencoder,
            Path(
                config.save_dir,
                f"loc_{config.scale.scale_location}_autoencoder_{config.save_name}.safetensors",
            ),
            metadata=metadata_dict,
        )


if __name__ == "__main__":
    fire.Fire(main)
