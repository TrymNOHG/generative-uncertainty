import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import trange
import math

from models.base import BaseModel

class CustomCNN(BaseModel):
    def __init__(self, in_channels=1, out_channels=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 1),
            nn.LayerNorm(normalized_shape=26),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.LazyLinear(out_features=128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=out_channels),
            nn.Softmax(dim=1)
        )

        self.loss_func = nn.NLLLoss()

        # self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    def reparameterize(self, mean: torch.Tensor, log_var: torch.Tensor):
        pass

    def predict(self, x: torch.Tensor):
        return torch.argmax(self(x)).item()

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        y_hat = self(x)
        return self.loss_func(y_hat, y), self.loss_func(y_hat, y)

    def loss_legend(self,):
        return ["Total", "NLL Loss"]

    def short_name(self,):
        return "CNN"

    
    # def _init_weights(self,):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             if not isinstance(m, nn.LazyLinear):
    #                 nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')