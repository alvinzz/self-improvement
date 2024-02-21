import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from arch.transformer.config import MultiHeadAttnConfig, SelfAttnConfig
from arch.utils.nonlinearity import Nonlinearity


def self_attention(
    K: Tensor,
    Q: Tensor,
    V: Tensor,
    nonlinearity: Nonlinearity = Nonlinearity.SOFTMAX,
):
    """Compute self-attention.

    Args:
        K: Keys of shape (B, T, D_attn).
        Q: Queries of shape (B, T, D_attn).
        V: Values of shape (B, T, D_out).
        nonlinearity: `Nonlinearity` to apply.
    """
    D_attn = Q.shape[2]
    qk_similarity = torch.einsum("bqd,bkd->bqk", Q, K)  # (B, T(Q), T(K))
    qk_attn = nonlinearity.fn(dim=2)(
        qk_similarity  # / np.sqrt(D_attn)
    )  # (B, T(Q), T(K))
    return torch.einsum("bta,bav->btv", qk_attn, V)


class SelfAttention(nn.Module):
    def __init__(self, config: SelfAttnConfig):
        super().__init__()
        self.config = config

        self.register_parameter(
            "W_k",
            Parameter(torch.zeros(self.config.dim, self.config.dim)),
        )
        self.register_parameter(
            "W_q",
            Parameter(torch.zeros(self.config.dim, self.config.dim)),
        )
        self.register_parameter(
            "W_v",
            Parameter(torch.zeros(self.config.dim, self.config.dim)),
        )

        with torch.no_grad():
            torch.nn.init.xavier_normal_(self.W_k, 0.002)
            torch.nn.init.xavier_normal_(self.W_q, 0.002)
            torch.nn.init.xavier_normal_(self.W_v, 0.002)
        # self.W_k = torch.eye(self.config.dim)
        # self.W_q = torch.eye(self.config.dim)
        # self.W_v = torch.zeros(self.config.dim, self.config.dim)
        # self.W_k[-1, -1] = 0
        # self.W_q[-1, -1] = 0
        # self.W_v[-1, -1] = -1

    def forward(self, x: Tensor):
        # x: (B, T, D)
        K = torch.einsum("btd,dk->btk", x, self.W_k)
        Q = torch.einsum("btd,dq->btq", x, self.W_q)
        V = torch.einsum("btd,dv->btv", x, self.W_v)

        return self_attention(K, Q, V, self.config.nonlinearity)


class MultiHeadAttn(nn.Module):
    def __init__(self, config: MultiHeadAttnConfig):
        super().__init__()
        self.config = config

        self.heads = nn.ModuleList(
            (
                SelfAttention(self.config.attn_config)
                for _ in range(self.config.n_heads)
            )
        )

        # self.register_parameter(
        #     "projection",
        #     Parameter(
        #         torch.zeros(
        #             self.config.attn_config.dim * self.config.n_heads,
        #             self.config.attn_config.dim,
        #         )
        #     ),
        # )

        # with torch.no_grad():
        #     torch.nn.init.xavier_normal_(self.projection, 0.002)
        self.projection = torch.eye(self.config.attn_config.dim)
        # self.register_parameter("lr", Parameter(torch.ones(1)))

    def forward(self, x: Tensor):
        head_outputs = torch.cat(
            tuple(head(x) for head in self.heads), dim=2
        )  # (B, T, D * N)
        return torch.einsum("btp,pd->btd", head_outputs, self.projection)
