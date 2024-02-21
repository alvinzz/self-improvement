from enum import Enum
from typing import Optional

import torch


class Nonlinearity(Enum):
    NONE = 1
    RELU = 2
    SOFTMAX = 3

    def fn(self, dim: Optional[int] = None):
        if self == Nonlinearity.NONE:
            return lambda x: x
        if self == Nonlinearity.RELU:
            return torch.relu
        if self == Nonlinearity.SOFTMAX:
            return lambda x: torch.softmax(x, dim=dim)
