import torch

from nnscaling.data import (
    DatasetConfig,
    YinYangBinaryDataset,
    YinYangDataset,
    YinYangNoDotsBinaryDataset,
)


def test_yin_yang_dataset():
    dataset_config = DatasetConfig(num_samples=50, split="train", seed=42)
    dataset = YinYangDataset(config=dataset_config)
    assert len(dataset) == 50
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1, 2]


def test_yin_yang_binary_dataset_label():
    dataset_config = DatasetConfig(
        num_samples=50, split="train", seed=42, negative_label=False
    )
    dataset = YinYangBinaryDataset(config=dataset_config)
    assert len(dataset) == 50
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1]

    dataset = YinYangNoDotsBinaryDataset(config=dataset_config)
    assert len(dataset) == 50
    assert torch.unique(dataset.labels.squeeze()).tolist() == [0, 1]


def test_yin_yang_binary_dataset_negative_label():
    dataset_config = DatasetConfig(
        num_samples=50, split="train", seed=42, negative_label=True
    )
    dataset = YinYangBinaryDataset(config=dataset_config)
    assert len(dataset) == 50
    assert torch.unique(dataset.labels.squeeze()).tolist() == [-1, 1]

    dataset = YinYangNoDotsBinaryDataset(config=dataset_config)
    assert len(dataset) == 50
    assert torch.unique(dataset.labels.squeeze()).tolist() == [-1, 1]
