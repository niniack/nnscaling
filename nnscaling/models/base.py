__all__ = ["BaseTorchModel"]

from abc import ABC, abstractmethod

import torch.nn as nn
from torchinfo import summary


class BaseTorchModel(nn.Module, ABC):
    @property
    @abstractmethod
    def features(self) -> nn.Sequential:
        pass

    @abstractmethod
    def forward(self):
        pass

    def summary(self):
        return summary(self)
