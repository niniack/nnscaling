__all__ = ["BaseTorchModel"]

from abc import ABC, abstractmethod

import torch.nn as nn
import torchinfo


class BaseTorchModel(nn.Module, ABC):
    @property
    @abstractmethod
    def features(self) -> nn.Sequential:
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    def summary(self):
        return torchinfo.summary(self, row_settings=["var_names", "ascii_only"])
