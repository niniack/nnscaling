import torch

from nnscaling.data import (
    DatasetConfig,
    YinYangBinaryDataset,
    YinYangDataset,
    YinYangNoDotsBinaryDataset,
)


def test_yin_yang_dataset():
    dataset_config = DatasetConfig(num_samples=1000, split="train", seed=42)
    dataset = YinYangDataset(config=dataset_config)
    assert len(dataset) == 1000
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1, 2]

    dataset = YinYangBinaryDataset(config=dataset_config)
    assert len(dataset) == 1000
    assert torch.unique(dataset.labels.squeeze()).tolist() == [-1, 1]

    dataset = YinYangNoDotsBinaryDataset(config=dataset_config)
    assert len(dataset) == 1000
    assert torch.unique(dataset.labels.squeeze()).tolist() == [-1, 1]
