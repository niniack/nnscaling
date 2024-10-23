import torch
from pydantic import (
    BaseModel,
)
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from nnscaling.log import logger
from nnscaling.models import BaseTorchModel


@logging_redirect_tqdm()
def train(
    config: BaseModel,
    model: BaseTorchModel,
    dataloader: DataLoader,
    device: torch.device,
) -> None:
    logger.info("Starting training...")
    loss_vector = []

    criterion = CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for epoch in range(config.num_epochs):
        model.train()
        running_loss = 0.0  # Track loss within the epoch

        for batch_idx, (input, label) in enumerate(dataloader):
            input, label = input.to(device), label.to(device)
            label = label.squeeze()

            optimizer.zero_grad()
            output = model(input)

            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate loss

        epoch_loss = running_loss / len(dataloader)  # Average loss over all batches
        loss_vector.append(epoch_loss)  # Store loss for later analysis

        if (epoch + 1) % config.print_freq == 0:
            tqdm.write(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {epoch_loss:.4f}")
