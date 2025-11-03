import torch
import torch.nn as nn
from abc import ABC, abstractmethod
class BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor):
        pass

    @abstractmethod
    def loss(self, x: torch.Tensor, y: torch.Tensor):
        pass

    @abstractmethod
    def loss_legend(self,):
        pass

    @abstractmethod
    def short_name(self,):
        pass

