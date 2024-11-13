import json
import os
import random

import numpy as np
import torch
from jaxtyping import Int
from torch import device
from torch.utils.data import DataLoader, Dataset
from torcheval.metrics import MulticlassAccuracy

from nnscaling.models import BaseTorchModel


def get_device() -> device:
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    return device


def set_seed(seed: int | None) -> None:
    """Set the random seed for random, PyTorch and NumPy"""
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)


def compute_model_accuracy(
    model: BaseTorchModel,  # Model
    dataset: Dataset,  # Dataset
    batch_size: Int = 256,
):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    metric = MulticlassAccuracy()
    for i, batch in enumerate(dataloader):
        image, target = batch
        output = model.forward(image)
        metric.update(output, target.squeeze())

    return metric.compute()


def safetensors_metadata_parser(
    file_path: str,  # thing
):
    "Parse the metadata from safetensors file"
    header_size = 8
    meta_data = {}
    if os.stat(file_path).st_size > header_size:
        with open(file_path, "rb") as f:
            b8 = f.read(header_size)
            if len(b8) == header_size:
                header_len = int.from_bytes(b8, "little", signed=False)
                headers = f.read(header_len)
                if len(headers) == header_len:
                    meta_data = sorted(
                        json.loads(headers.decode("utf-8")).get("__metadata__", meta_data).items()
                    )
    meta_data_dict = {}
    for k, v in meta_data:
        meta_data_dict[k] = v
    return meta_data_dict
