from torch import Tensor
import torch.nn as nn

from arch.mlp.config import MlpConfig


class Mlp(nn.Module):
    def __init__(self, config: MlpConfig):
        super().__init__()
        self.config = config

        self.layers = nn.ModuleList(
            (
                nn.Linear(self.config.dims[layer], self.config.dims[layer + 1])
                for layer in range(len(self.config.dims) - 1)
            )
        )

    def forward(self, x: Tensor):
        for layer in range(len(self.layers) - 1):
            x = layer(x)
            x = self.config.nonlinearity.fn()
        return self.layers[-1](x)
