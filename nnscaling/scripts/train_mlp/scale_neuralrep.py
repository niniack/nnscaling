import os
import sys
from pathlib import Path
from typing import Literal, Optional

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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from nnscaling.data import (
    DatasetConfig,
    create_data_loader,
    get_dataset_class,
)
from nnscaling.log import logger
from nnscaling.models import MLP, ScaledModel
from nnscaling.scripts.common import load_config
from nnscaling.utils import (
    StringtoClassNonlinearity,
    get_device,
    safetensors_metadata_parser,
    set_seed,
)


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
    scale_nonlinearity: str
    scale_style: Literal["replace", "prepend"]
    betas: list[PositiveFloat] | None = None
    num_epochs: PositiveInt | None = None


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    config: Config,
) -> float:
    # Helper function to filter weight matrices (exclude biases)
    def get_weight_matrices(params):
        return [p for p in params if len(p.shape) > 1]  # Weights typically have >1 dimension

    def compute_norm_loss(activation_dict, subsample_size=80):
        def center_gram(X):
            """Center a Gram matrix."""
            n = X.size(0)
            H = torch.eye(n, device=X.device) - (1 / n) * torch.ones((n, n), device=X.device)
            return H @ X @ H

        def expected_trace(X):
            """Estimate the expected trace for finite samples."""
            return torch.trace(X) / X.size(0)

        def debiased_cka(A, B):
            """Compute the debiased CKA similarity between two activation matrices."""
            # Compute centered Gram matrices
            K_A = center_gram(
                torch.einsum("ij,kj->ik", A, A)
            )  # Using einsum for efficient computation
            K_B = center_gram(torch.einsum("ij,kj->ik", B, B))

            # Compute traces
            trace_AB = torch.trace(K_A @ K_B)
            trace_AA = torch.trace(K_A @ K_A)
            trace_BB = torch.trace(K_B @ K_B)

            # Estimate expected traces (bias terms)
            expected_AB = expected_trace(K_A @ K_B)
            expected_AA = expected_trace(K_A @ K_A)
            expected_BB = expected_trace(K_B @ K_B)

            # Debiased CKA similarity
            epsilon = 1e-10  # For numerical stability
            numerator = trace_AB - expected_AB
            denominator = torch.sqrt((trace_AA - expected_AA) * (trace_BB - expected_BB)) + epsilon

            return numerator / denominator

        def subsample_indices(matrix, size):
            n = matrix.size(0)
            if n > size:
                return torch.randperm(n, device=matrix.device)[:size]
            return torch.arange(n, device=matrix.device)

        # Preprocess activations: zero-center and subsample each activation matrix
        activations = [
            act - act.mean(dim=0, keepdim=True) for act in list(activation_dict.values())[:-1]
        ]

        # Compute subsample indices once
        subsample_idx = subsample_indices(activations[0], subsample_size)

        # Subsample all activations using the same indices
        activations = [act[subsample_idx] for act in activations]

        cka_losses = []
        for i in range(1, len(activations)):
            prev_act = activations[i - 1]
            curr_act = activations[i]

            # Compute debiased CKA similarity between consecutive activations
            # with torch.no_grad():
            similarity = debiased_cka(prev_act, curr_act)

            # Encourage similarity to be high
            cka_loss = 1 - similarity  # Higher similarity reduces the loss
            cka_losses.append(cka_loss)

        # Combine all CKA losses into a single regularization term
        regularization_loss = torch.sum(torch.stack(cka_losses))

        return regularization_loss

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
    scalable_params = list(model.scalable_layer.parameters())

    loss_epoch = 0
    norm_loss_epoch = 0
    uniformity_loss_epoch = 0

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

        if config.scale_style == "replace":
            output_activation = model.get_activations(detach=False)[
                f"scaled_{config.num_scaled_layers-1}"
            ]
        elif config.scale_style == "prepend":
            output_activation = model.get_activations(detach=False)["post_insert_0"]

        # Compute the primary loss
        loss = criterion(output_activation, target_activation)

        # Compute regularization terms for weight matrices only
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        norm_loss = compute_norm_loss(model.get_activations(detach=False))
        uniformity_loss = compute_uniformity_loss(trainable_params)

        # Combine total loss
        loss += config.lambda_complexity * norm_loss + config.lambda_uniformity * uniformity_loss

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

    # Load data
    dataset_config = config.train_data
    DatasetClass = get_dataset_class(name=dataset_config.dataset_name)
    dataset = DatasetClass(config=dataset_config)
    train_loader = create_data_loader(
        dataset, batch_size=config.batch_size, global_seed=config.seed
    )

    # Load model
    metadata = safetensors_metadata_parser(file_path=config.model_to_scale)
    unscaled_model = MLP.load_model(
        file_path=config.model_to_scale,
        in_features=dataset.in_features,
    )
    model = ScaledModel(
        model=unscaled_model,
        index=config.scale_location,
        nonlinearity=StringtoClassNonlinearity[config.scale_nonlinearity].value,
        scale_style=config.scale_style,
        num_scaled=config.num_scaled_layers,
        batchnorm=True,
    )
    model.summary()
    model.to(device).train()

    all_param_names = [name for name, _ in model.named_parameters()]
    assert len(all_param_names) > 0, "No trainable parameters found."
    logger.info(f"Trainable layers: {len(all_param_names)}")

    # Define optimiser and scheduler
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
            config=config,
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
            "scale_style": str(config.scale_style),
            "scale_nonlinearity": str(config.scale_nonlinearity),
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
