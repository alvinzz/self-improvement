from __future__ import annotations
from typing import Tuple

from cfg.config import Config
from arch.utils.nonlinearity import Nonlinearity


class MlpConfig(Config):
    def __init__(self, nonlinearity: Nonlinearity, dims: Tuple[int, ...]):
        self.nonlinearity = nonlinearity
        self.dims = dims
