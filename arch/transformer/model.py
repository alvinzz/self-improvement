import torch.nn as nn
from torch import Tensor
from arch.mlp.config import MlpConfig

from arch.mlp.model import Mlp
from arch.transformer.config import (
    MultiHeadAttnConfig,
    SelfAttnConfig,
    TransformerConfig,
)
from arch.transformer.attention import MultiHeadAttn, SelfAttention
from arch.utils.nonlinearity import Nonlinearity


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        self.attn_layers = nn.ModuleList(
            (
                MultiHeadAttn(self.config.multi_head_attn_config)
                for _ in range(self.config.depth)
            )
        )
        self.mlps = nn.ModuleList(
            (Mlp(self.config.mlp_config) for _ in range(self.config.depth))
        )

    def forward(self, x: Tensor):
        # x: [B, T, D]
        B, T, D = x.shape
        for attn_layer, mlp in zip(self.attn_layers, self.mlps):
            x = x + attn_layer(x)
            if self.config.use_layernorm:
                x = nn.LayerNorm((D,))(x)
            x = x + mlp(x)
            if self.config.use_layernorm:
                x = nn.LayerNorm((D,))(x)
        return x
