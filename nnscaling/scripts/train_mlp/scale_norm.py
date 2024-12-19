import os
import pdb
import random
import sys
from ast import literal_eval
from pathlib import Path
from typing import Optional

import fire
import numpy as np
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
from safetensors.torch import load_model, save_model
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nnscaling.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
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
    weight_decay: NonNegativeFloat
    lambda_complexity: NonNegativeFloat
    lambda_uniformity: NonNegativeFloat
    out_features: PositiveInt
    print_freq: PositiveInt
    save_name: str | None
    save_dir: str | None
    model_to_scale: str
    scale_location: PositiveInt
    num_scaled_layers: PositiveInt
    betas: list[PositiveFloat] | None = None
    num_epochs: PositiveInt | None = None


# def train_one_epoch(
#     model: nn.Module,
#     train_loader: DataLoader,
#     device: torch.device,
#     optimizer: torch.optim.Optimizer,
#     lambda_reg: float = 100.0,
# ) -> float:
#     model.train()
#     model.to(device)
#     loss_epoch = 0
#     criterion = nn.CrossEntropyLoss()

#     replaceable_params = list(model.replaceable.parameters())

#     for batch_idx, (input, label) in enumerate(train_loader):
#         input, label = input.to(device), label.to(device)
#         label = label.squeeze()

#         optimizer.zero_grad()
#         output = model.forward_scaled(input)

#         loss = criterion(output, label.long())

#         # Compute the L2 norms
#         trainable_params = [p for p in model.parameters() if p.requires_grad]
#         total_l2 = torch.norm(torch.cat([p.flatten() for p in trainable_params]))
#         replaceable_l2 = torch.norm(torch.cat([p.flatten() for p in replaceable_params]))

#         # Add the L2 regularization loss component
#         reg_loss = (total_l2 - replaceable_l2).pow(2)
#         loss += lambda_reg * reg_loss

#         loss.backward()
#         optimizer.step()

#         loss_epoch += loss.item()  # Accumulate loss

#     epoch_loss = loss_epoch / len(train_loader)  # Average loss over all batches
#     return epoch_loss


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    lambda_complexity: float = 1e-1,
    lambda_uniformity: float = 1e-5,
) -> float:
    # Helper function to filter weight matrices (exclude biases)
    def get_weight_matrices(params):
        return [p for p in params if len(p.shape) > 1]  # Weights typically have >1 dimension

    def compute_norm_loss(trainable_params, replaceable_params):
        trainable_weights = get_weight_matrices(trainable_params)
        replaceable_weights = get_weight_matrices(replaceable_params)

        total_l2 = torch.sqrt(sum(torch.norm(p).pow(2) for p in trainable_weights))
        replaceable_l2 = torch.sqrt(sum(torch.norm(p).pow(2) for p in replaceable_weights))

        # Penalize the absolute difference in L2 norms directly
        return (total_l2 - replaceable_l2).pow(2)

    # # Updated norm loss for weight matrices
    # def compute_norm_loss(trainable_params, replaceable_params):
    #     trainable_weights = get_weight_matrices(trainable_params)

    #     # Compute identity-like regularization loss
    #     identity_like_losses = []
    #     for p in trainable_weights:
    #         m, n = p.size()
    #         # Create an identity-like matrix
    #         I = torch.zeros_like(p)
    #         min_dim = min(m, n)
    #         I[:min_dim, :min_dim] = torch.eye(min_dim, device=p.device)

    #         # Compute deviation from identity-like matrix
    #         identity_like_losses.append(torch.norm(p - I))

    #     # Regularization loss focuses on trainable weights deviation from identity-like matrices
    #     regularization_loss = torch.sum(torch.stack(identity_like_losses))

    #     return regularization_loss

    def compute_uniformity_loss(trainable_params):
        # Filter weight matrices only
        trainable_weights = get_weight_matrices(trainable_params)

        # Compute averaged singular values for trainable weights (across first dimension)
        trainable_singular_values = [torch.linalg.svdvals(p) for p in trainable_weights]
        k = len(trainable_singular_values[0])
        for tsv in trainable_singular_values:
            k = min(k, len(tsv))
        trainable_singular_values = torch.stack(
            [values[:k] for values in trainable_singular_values]
        )

        # Compute variance of singular values
        return torch.mean(torch.var(trainable_singular_values, dim=0))

    model.train()
    model.to(device)
    criterion = nn.MSELoss()
    replaceable_params = list(model.replaceable.parameters())

    loss_epoch = 0
    norm_loss_epoch = 0
    uniformity_loss_epoch = 0

    for batch_idx, (input, label) in enumerate(train_loader):
        input, label = input.to(device), label.to(device)
        label = label.squeeze()

        optimizer.zero_grad()
        model.hook_model(scaled=True, post=True)

        # Forward pass for target and output activations
        with torch.no_grad():
            _ = model.forward_raw(input)
            target_activation = model.get_activations()["post_insert_0"]

        _ = model.forward_scaled(input)
        output_activation = model.get_activations(detach=False)["scaled_7"]

        # Compute the primary loss
        loss = criterion(output_activation, target_activation)

        # Compute regularization terms for weight matrices only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        norm_loss = compute_norm_loss(trainable_params, replaceable_params)
        uniformity_loss = compute_uniformity_loss(trainable_params)

        # Combine total loss
        loss += lambda_complexity * norm_loss + lambda_uniformity * uniformity_loss

        loss.backward()
        optimizer.step()

        # Accumulate losses
        loss_epoch += loss.item()
        norm_loss_epoch += norm_loss.item()
        uniformity_loss_epoch += uniformity_loss.item()

    # Average losses over all batches
    epoch_loss = loss_epoch / len(train_loader)
    avg_norm_loss = norm_loss_epoch / len(train_loader)
    avg_uniformity_loss = uniformity_loss_epoch / len(train_loader)

    # Return all metrics as a dictionary
    return {
        "epoch_loss": epoch_loss,
        "norm_loss": avg_norm_loss,
        "uniformity_loss": avg_uniformity_loss,
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

    ##############################################################################################################################
    def print_seeds():
        """
        Print the current seeds for WandB, Python's random module, NumPy, and PyTorch.
        """
        # WandB seed (if it exists in the config)
        wandb_seed = wandb.config.get("seed", "Not set") if wandb.run else "No WandB run active"
        print(f"WandB Seed: {wandb_seed}")

        # Python's random module
        python_random_state = random.getstate()
        print(f"Python Random Module State: {python_random_state[1][:5]}")  # First few numbers

        # NumPy's random state
        numpy_random_state = np.random.get_state()
        print(f"NumPy Random State: {numpy_random_state[1][:5]}")  # First few numbers

        # PyTorch seed
        torch_random_state = torch.random.get_rng_state()
        print(f"PyTorch RNG State (First 5): {list(torch_random_state[:5])}")  # First few bytes

        if torch.cuda.is_available():
            cuda_random_state = torch.cuda.get_rng_state()
            print(f"CUDA RNG State (First 5): {list(cuda_random_state[:5])}")  # First few bytes

    # Call this function wherever you need to debug seed information
    print_seeds()
    ##############################################################################################################################

    # Load data
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    # Load model
    metadata = safetensors_metadata_parser(file_path=config.model_to_scale)
    unscaled_model = MLP.load_model(config.model_to_scale, dataset.features.shape[-1])
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

    # Define optimiser
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
        losses = train_one_epoch(
            model=model,
            train_loader=train_loader,
            device=device,
            optimizer=optimizer,
            lambda_complexity=config.lambda_complexity,
            lambda_uniformity=config.lambda_uniformity,
        )

        scheduler.step(losses["epoch_loss"])

        # Log metrics
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": losses["epoch_loss"],
                "norm_loss": losses["norm_loss"],
                "uniformity_loss": losses["uniformity_loss"],
            }
        )

        # Print loss
        if (epoch + 1) % config.print_freq == 0:
            logger.info(
                f"Epoch {epoch + 1}/{config.num_epochs}, "
                f"Train Loss: {losses['epoch_loss']:.4f}, "
                f"Norm Loss: {losses['norm_loss']:.4f}, "
                f"Uniformity Loss: {losses['uniformity_loss']:.4f}"
            )

    if config.save_dir:
        metadata_dict = {
            "config": str(metadata["config"]),
            "dataset": dataset.name(),
            "out_features": str(config.out_features),
            "nonlinearity": str(metadata["nonlinearity"]),
            "added_layers": str(config.num_scaled_layers),
            "layer_start": str(config.scale_location),
        }
        os.makedirs(os.path.dirname(config.save_dir), exist_ok=True)
        save_model(
            model,
            Path(
                config.save_dir,
                f"loc_{config.scale_location}_scaled_{config.save_name}.safetensors",
            ),
            metadata=metadata_dict,
        )


if __name__ == "__main__":
    fire.Fire(main)
