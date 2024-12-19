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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from nnscaling.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from nnscaling.log import logger
from nnscaling.models import MLP, Autoencoder, ScaledModel
from nnscaling.scripts.common import load_config
from nnscaling.topology import SignatureLoss, VietorisRipsComplex
from nnscaling.utils import (
    get_device,
    safetensors_metadata_parser,
    set_seed,
)

################### EXP ##############
signature_loss = SignatureLoss(p=2)
vr = VietorisRipsComplex(dim=0)
################### EXP ##############


# OptimConfig for optimizer-related parameters
class OptimConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
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
    learning_rate: PositiveFloat
    scale_nonlinearity: str
    scale_style: Literal["replace", "prepend"]
    lambda_reconstruction: NonNegativeFloat
    lambda_prediction: NonNegativeFloat


class AutoencoderConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    ae_dim: PositiveInt
    learning_rate: PositiveFloat
    ae_nonlinearity: Optional[str] = None
    lambda_reconstruction: NonNegativeFloat
    lambda_prediction: NonNegativeFloat
    train_fraction: NonNegativeFloat
    reuse_file: Optional[str] = None


# Main Config class
class Config(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    train_data: DatasetConfig
    optim: OptimConfig
    scale: ScaleConfig
    autoencoder: AutoencoderConfig
    print_freq: PositiveInt
    seed: NonNegativeInt = 0
    save_name: Optional[str] = None
    save_dir: Optional[str] = None


def pad_act(x, target_size):
    current_size = x.size(1)
    if current_size < target_size:
        pad_size = target_size - current_size
        x = F.pad(x, (0, pad_size), mode="constant", value=0)

    return x


def compute_k_prediction_loss(act_dict, autoencoder, k):
    # Extract activations from the dictionary
    act_list = list(act_dict.values())
    target_acts = act_list[-1]  # shape: batch, neurons
    total_variance = (target_acts - target_acts.mean(dim=0)).pow(2).sum()

    # Compute the prediction for the first activation
    # Autoencoder outputs are shaped [layers, batch, neurons]
    all_preds = autoencoder(x=act_list[0], k=k).predictions
    pred_k = all_preds[-1, :, : target_acts.size(-1)]
    recons_error = (pred_k - target_acts).pow(2).sum()

    # Compute and return the loss
    return recons_error / total_variance


def compute_recons_loss(act_dict, autoencoder):
    padded_acts = []
    masks = []
    ae_input_size = autoencoder.encoder[0].in_features
    for act in act_dict.values():
        # Pad activations
        padded_acts.append(pad_act(act, ae_input_size))

        # Build a mask to ignore "extra" neurons in downstream activations
        # Only relevant for the second layer
        mask = torch.zeros(ae_input_size, device=act.device)
        curr_size = act.size(-1)
        mask[:curr_size] = 1
        masks.append(mask)

    # Stack lists
    padded_acts = torch.stack(padded_acts, dim=1)  # shape: batch, layers, neurons
    masks = torch.stack(masks, dim=0)  # shape: layers, neurons

    # Total variance, used as a denominator for scaling the reconstruction loss
    masked_centered_acts = padded_acts - padded_acts.mean(dim=0) * masks.unsqueeze(dim=0)
    total_variance = masked_centered_acts.pow(2).sum()

    # Reconstruction with AE
    recons_acts = [autoencoder(x=act, k=0).reconstruction for act in padded_acts.unbind(dim=1)]
    recons_acts = torch.stack(recons_acts, dim=1)

    # ################### EXP ##############
    # vr_state = vr(padded_acts[:, 0, :])
    # vr_obs = vr(autoencoder.encoder(padded_acts[:, 0, :]))

    # topo_loss = signature_loss(
    #     [padded_acts[:, 0, :], vr_state],
    #     [autoencoder.encoder(padded_acts[:, 0, :]), vr_obs],
    # )

    # vr_state = vr(padded_acts[:, 1, :])
    # vr_obs = vr(autoencoder.encoder(padded_acts[:, 1, :]))
    # topo_loss += signature_loss(
    #     [padded_acts[:, 1, :], vr_state],
    #     [autoencoder.encoder(padded_acts[:, 1, :]), vr_obs],
    # )
    # ################### EXP ##############

    # Reconstruction error
    masked_diff = (padded_acts - recons_acts) * masks.unsqueeze(dim=0)
    recons_error = (masked_diff).pow(2).sum()

    # Error is scaled with total_variance
    # Note that we didn't bother with dividing either value
    # by the numel because it would cancel out!
    return recons_error / total_variance


def compute_scaler_prediction_loss(act_dict, autoencoder):
    k = len(act_dict.keys()) - 1
    padded_acts = []
    masks = []
    ae_input_size = autoencoder.encoder[0].in_features
    for act in act_dict.values():
        # Pad activations
        padded_acts.append(pad_act(act, ae_input_size))

        # Build a mask to ignore "extra" neurons in downstream activations
        # Only relevant for the second layer
        mask = torch.zeros(ae_input_size, device=act.device)
        curr_size = act.size(-1)
        mask[:curr_size] = 1
        masks.append(mask)

    # Stack lists
    padded_acts = torch.stack(padded_acts, dim=1)  # shape: batch, layers, neurons
    masks = torch.stack(masks, dim=0)  # shape: layers, neurons

    # Split activations
    first_layer_act = padded_acts[:, 0, :]  # shape: batch, neurons
    remaining_acts = padded_acts[:, -k:, :]  # shape: batch, layers-1, neurons
    remaining_masks = masks[-k:].unsqueeze(0)  # shape: 1, layers-1, neurons

    # Predict 0,...,k steps
    # shape: layers, batch, neurons
    k_plus_one_preds = autoencoder(x=first_layer_act, k=k).predictions
    # shape: batch, layers, neurons
    remaining_pred_acts = k_plus_one_preds.permute(1, 0, 2)[:, -k:, :]

    # Compute total variance
    masked_centered_acts = (remaining_pred_acts - remaining_pred_acts.mean(dim=0)) * remaining_masks
    total_variance = masked_centered_acts.pow(2).sum()

    # Compute masked difference
    masked_diff = (remaining_pred_acts - remaining_acts) * remaining_masks
    loss = masked_diff.pow(2).sum()

    return loss / total_variance


# TODO: Is the implementation below what we want irl?
# def compute_scaler_prediction_loss(act_dict, autoencoder):
#     padded_acts = []
#     masks = []
#     ae_input_size = autoencoder.encoder[0].in_features
#     for act in act_dict.values():
#         # Pad activations
#         padded_acts.append(pad_act(act, ae_input_size))

#         # Build a mask to ignore "extra" neurons in downstream activations
#         # Only relevant for the second layer
#         mask = torch.zeros(ae_input_size, device=act.device)
#         curr_size = act.size(-1)
#         mask[:curr_size] = 1
#         masks.append(mask)

#     # Stack lists
#     padded_acts = torch.stack(padded_acts, dim=1)  # shape: batch, layers, neurons
#     masks = torch.stack(masks, dim=0)  # shape: layers, neurons

#     predictions = []
#     for i, act in enumerate(padded_acts.unbind(dim=1)):
#         k_preds = autoencoder(
#             x=act, k=(len(act_dict.keys()) - (i + 1))
#         ).predictions  # shape: layers, batch, neurons
#         predictions.append(k_preds)

#     prediction_loss = torch.tensor(0.0)
#     for k_preds in predictions:
#         # Num predictions from current layer
#         k = k_preds.shape[0]
#         # Last k masks
#         masks_last_k = masks[-k, :].unsqueeze(0)
#         # Last k padded activations
#         padded_acts_last_k_ = padded_acts[:, -k:, :]

#         # Compute and mask centered activations
#         masked_centered_acts = (
#             padded_acts_last_k_ - padded_acts_last_k_.mean(dim=0)
#         ) * masks_last_k
#         total_variance = masked_centered_acts.pow(2).sum()

#         # Compute masked difference
#         masked_diff = (padded_acts_last_k_ - k_preds.permute(1, 0, 2)) * masks_last_k
#         loss = masked_diff.pow(2).sum()

#         # Normalize loss
#         prediction_loss += loss / total_variance

#     return prediction_loss


def train_one_epoch(
    scaler: nn.Module,
    autoencoder: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer_autoencoder: torch.optim.Optimizer,
    optimizer_scaler: torch.optim.Optimizer,
    config: Config,
    train_target: Literal["autoencoder", "scaler"],
) -> dict:
    # Per epoch losses
    loss_epoch = 0
    ae_recons_loss_epoch = 0
    ae_prediction_loss_epoch = 0
    scaling_recons_epoch = 0
    scaling_pred_epoch = 0

    for input, label in train_loader:
        input, label = input.to(device), label.to(device).squeeze()

        if train_target == "autoencoder":
            autoencoder.train()
            scaler.eval()
            optimizer = optimizer_autoencoder

        elif train_target == "scaler":
            autoencoder.eval()
            scaler.train()
            optimizer = optimizer_scaler

        else:
            raise ValueError(f"Unknown train_target: {train_target}")

        # Hook scaler
        scaler.hook_model(pre=True, scaled=True, post=True)

        # Zero out gradients
        optimizer.zero_grad()

        # Raw forward pass
        with torch.no_grad():
            _ = scaler.forward_raw(input)

        # NOTE: Does not include scaled activations!
        # TODO: Rethink detach depending on train_target
        all_acts = scaler.get_activations(detach=True)

        # Get target for scaler
        target_activation = all_acts["post_insert_0"]

        # NOTE: Does not include scaled activations!
        # Get first and last elements without modifying the dict
        first_key, first_value = next(iter(all_acts.items()))
        last_key, last_value = next(iter(reversed(all_acts.items())))
        first_last_acts = {first_key: first_value, last_key: last_value}

        # Compute autoencoder losses
        ae_recons_loss = compute_recons_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
        )
        ae_pred_loss = compute_k_prediction_loss(
            act_dict=first_last_acts,
            autoencoder=autoencoder,
            k=config.scale.num_scaled_layers,
        )

        # Scaled forward pass
        _ = scaler.forward_scaled(input)
        all_acts = scaler.get_activations(detach=False)
        if config.scale.scale_style == "replace":
            output_activation = all_acts[f"scaled_{config.scale.num_scaled_layers-1}"]
            all_acts.popitem(last=True)
        elif config.scale.scale_style == "prepend":
            output_activation = all_acts["post_insert_0"]

        # Compute scaling losses
        scaler_recons_loss = ((output_activation - target_activation).pow(2).sum()) / (
            (target_activation - target_activation.mean(dim=0)).pow(2).sum()
        )
        scaler_pred_loss = compute_scaler_prediction_loss(
            act_dict=all_acts, autoencoder=autoencoder
        )

        # Combine total loss
        loss = (
            config.autoencoder.lambda_reconstruction * ae_recons_loss
            + config.autoencoder.lambda_prediction * ae_pred_loss
            + config.scale.lambda_reconstruction * scaler_recons_loss
            + (
                config.scale.lambda_prediction
                * (scaler_pred_loss if train_target == "scaler" else scaler_pred_loss.detach())
            )
        )

        loss.backward()
        optimizer.step()

        # Accumulate losses
        loss_epoch += loss.item()
        ae_recons_loss_epoch += ae_recons_loss.item()
        ae_prediction_loss_epoch += ae_pred_loss.item()
        scaling_recons_epoch += scaler_recons_loss.item()
        scaling_pred_epoch += scaler_pred_loss.item()

    # Average losses over all batches
    epoch_loss = loss_epoch / len(train_loader)
    avg_recons_loss = ae_recons_loss_epoch / len(train_loader)
    avg_prediction_loss = ae_prediction_loss_epoch / len(train_loader)
    avg_scaling_recons_loss = scaling_recons_epoch / len(train_loader)
    avg_scaling_pred_loss = scaling_pred_epoch / len(train_loader)

    # Return all metrics as a dictionary
    return {
        "epoch_loss": epoch_loss,
        "recons_loss": avg_recons_loss,
        "prediction_loss": avg_prediction_loss,
        "scaling_recons_loss": avg_scaling_recons_loss,
        "scaling_pred_loss": avg_scaling_pred_loss,
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
    ).eval()
    model = ScaledModel(
        model=unscaled_model,
        index=config.scale.scale_location,
        nonlinearity=config.scale.scale_nonlinearity,
        scale_style=config.scale.scale_style,
        num_scaled=config.scale.num_scaled_layers,
        batchnorm=True,
    )
    model.summary()
    model.to(device)

    # Build autoencoder
    if config.autoencoder.reuse_file:
        autoencoder = Autoencoder.load_model(config.autoencoder.reuse_file)
    else:
        autoencoder = Autoencoder(
            nonlinearity=config.autoencoder.ae_nonlinearity,
            in_features=model.scalable_layer.in_features,
            observable_features=config.autoencoder.ae_dim,
        )
    autoencoder.summary()
    autoencoder.to(device)

    # Set number of epochs
    num_epochs = config.optim.num_epochs
    switch_epoch = (
        config.autoencoder.train_fraction * num_epochs if not config.autoencoder.reuse_file else 0
    )

    # Define optimiser and scheduler
    optimizer_ae = optim.AdamW(
        params=list(autoencoder.parameters()),
        lr=config.autoencoder.learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=tuple(config.optim.betas) or (0.9, 0.999),
    )

    optimizer_scaler = optim.AdamW(
        params=list(model.parameters()),
        lr=config.scale.learning_rate,
        weight_decay=config.optim.weight_decay,
        betas=tuple(config.optim.betas) or (0.9, 0.999),
    )

    # Loop over epochs
    for epoch in range(num_epochs):
        # Set train target
        if epoch < switch_epoch:
            train_target = "autoencoder"
        elif epoch >= switch_epoch:
            train_target = "scaler"

        # Train step
        losses = train_one_epoch(
            scaler=model,
            autoencoder=autoencoder,
            train_loader=train_loader,
            device=device,
            optimizer_autoencoder=optimizer_ae,
            optimizer_scaler=optimizer_scaler,
            config=config,
            train_target=train_target,
        )

        # scheduler.step(losses["epoch_loss"])

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": losses["epoch_loss"],
                "recons_loss": losses["recons_loss"],
                "prediction_loss": losses["prediction_loss"],
                "scaling_recons_loss": losses["scaling_recons_loss"],
                "scaling_pred_loss": losses["scaling_pred_loss"],
            }
        )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
                f"Reconstruction Loss: {losses['recons_loss']:.4f}, "
                f"Prediction Loss: {losses['prediction_loss']:.4f} "
                f"Scaling Recons Loss: {losses['scaling_recons_loss']:.4f} "
                f"Scaling Pred Loss: {losses['scaling_pred_loss']:.4f}"
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
            "autoencoder_nonlinearity": str(config.autoencoder.ae_nonlinearity),
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
