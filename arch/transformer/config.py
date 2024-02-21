from __future__ import annotations

from cfg.config import Config
from arch.mlp.config import MlpConfig
from arch.utils.nonlinearity import Nonlinearity


class SelfAttnConfig(Config):
    def __init__(self, dim: int, nonlinearity: Nonlinearity):
        self.dim = dim
        self.nonlinearity = nonlinearity


class MultiHeadAttnConfig(Config):
    def __init__(self, n_heads: int, attn_config: SelfAttnConfig):
        self.n_heads = n_heads
        self.attn_config = attn_config


class TransformerConfig(Config):
    def __init__(
        self,
        depth: int,
        multi_head_attn_config: MultiHeadAttnConfig,
        mlp_config: MlpConfig,
        use_layernorm: bool,
    ):
        self.depth = depth
        self.multi_head_attn_config = multi_head_attn_config
        self.mlp_config = mlp_config
        self.use_layernorm = use_layernorm
