__all__ = [
    "YinYangDataset",
    "YinYangBinaryDataset",
    "MNISTDataset",
]

import sys
from typing import Callable, Literal

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from pydantic import BaseModel, ConfigDict
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class DatasetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)
    dataset_name: Literal["YinYangDataset", "YinYangNoDotsBinaryDataset", "YinYangBinaryDataset"]
    num_samples: int
    split: str
    torch_transform: Callable | None = None
    seed: int | None = 42
    negative_label: bool = False


def get_dataset_class(name: str) -> Dataset:
    return getattr(sys.modules[__name__], name)


def create_data_loader(
    dataset: Dataset, batch_size: Int, global_seed: Int = 0, shuffle: bool = True
) -> DataLoader:
    loader = DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


class BaseYinYangDataset(Dataset):
    """
    A YinYang inspired dataset, adapted from: \\
    https://github.com/lkriener/yin_yang_data_set
    """

    def __init__(
        self,
        config: DatasetConfig,
        binary: Bool = False,
        dots: Bool = True,  # Overriden to True if binary is True
        r_small: Float = 0.1,
        r_big: Float = 0.5,
        rotate: Bool = False,
    ):
        super().__init__()
        # using a numpy RNG to allow compatibility to other deep learning frameworks
        self.rng = np.random.RandomState(config.seed)
        self.transform = config.torch_transform
        self.r_small = r_small
        self.r_big = r_big

        if binary:
            self.class_names = ["yin", "yang"]
        else:
            self.class_names = ["yin", "yang", "dot"]
        self.features = []
        self.labels = []

        self.binary = binary
        if not binary and not dots:
            self.dots = True
        else:
            self.dots = dots

        for i in range(config.num_samples):
            goal_class = self.rng.randint(2 if binary else 3)
            x, y, c = self.get_sample(goal=goal_class)

            if c == 0 and (binary and config.negative_label):
                c = -1

            val = np.array([x, y])
            self.features.append(val)
            self.labels.append(c)

        self.features = torch.FloatTensor(np.asarray(self.features))
        if rotate:
            rot_matr = torch.Tensor([[0, -1], [1, 0]])
            self.features = self.features @ rot_matr
        self.labels = torch.IntTensor(self.labels).unsqueeze(1)

    def get_sample(self, goal=None):
        # sample until goal is satisfied
        found_sample_yet = False
        while not found_sample_yet:
            # sample x,y coordinates
            x, y = self.rng.rand(2) * 2.0 * self.r_big
            # check if within yin-yang circle
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            # check if they have the same class as the goal for this sample
            c = self.which_class(x, y)
            if goal is None or c == goal:
                found_sample_yet = True
                break

        return x, y, c

    def which_class(self, x, y):
        # equations inspired by
        # https://link.springer.com/content/pdf/10.1007/11564126_19.pdf
        d_right = self.dist_to_right_dot(x, y)
        d_left = self.dist_to_left_dot(x, y)
        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3
        is_circles = d_right < self.r_small or d_left < self.r_small
        if is_circles and (not self.binary or (self.binary and not self.dots)):
            return 2
        elif is_circles and self.binary:
            return int(criterion1 or criterion2)

        return int(is_yin)

    def dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    @property
    def name(self):
        raise NotImplementedError("Subclasses must implement the 'name' property")

    def __getitem__(self, index):
        sample = (self.features[index], self.labels[index])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.labels)

    def __str__(self) -> str:
        return self.name


class YinYangDataset(BaseYinYangDataset):
    """
    3-way YinYang classification (yin, yang, dots)
    """

    def __init__(self, config: DatasetConfig, r_small=0.1, r_big=0.5):
        super().__init__(
            config=config,
            binary=False,
            dots=True,
            r_small=r_small,
            r_big=r_big,
        )

    def name(self):
        return "YinYangDataset"


class YinYangNoDotsBinaryDataset(BaseYinYangDataset):
    """
    2-way YinYang classification (yin, yang). Dots have been removed.
    """

    def __init__(self, config: DatasetConfig, r_small=0.1, r_big=0.5):
        super().__init__(
            config=config,
            binary=True,
            dots=False,
            r_small=r_small,
            r_big=r_big,
        )

    def name(self):
        return "YinYangNoDotsBinaryDataset"


class YinYangBinaryDataset(BaseYinYangDataset):
    """
    2-way (canonical) YinYang classification (yin, yang). Dots are part of Yin/Ying class.
    """

    def __init__(self, config: DatasetConfig, r_small=0.1, r_big=0.5):
        super().__init__(
            config=config,
            binary=True,
            dots=True,
            r_small=r_small,
            r_big=r_big,
        )

    def name(self):
        return "YinYangBinaryDataset"


class MNISTDataset:
    """Simple wrapper around MNIST dataset"""

    default_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # first, convert image to PyTorch tensor
            transforms.Normalize((0.1307,), (0.3081,)),  # normalize inputs
        ]
    )

    def __init__(
        self,
        seed=42,  # Randomness seed
        transform=default_transform,  # Torch transforms, uses default MNIST transform
        root="/mnt/datasets/",  # Dataset location
        train=True,  # Train or test dataset
    ):
        self.data = datasets.MNIST(root=root, train=train, download=True, transform=transform).data

    def name(self):
        return "MNISTDataset"
