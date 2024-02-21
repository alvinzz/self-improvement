# reproduce results of (van Oswald 2023)

import torch

from arch.transformer.config import (
    TransformerConfig,
    SelfAttnConfig,
    MultiHeadAttnConfig,
)
from arch.mlp.config import MlpConfig
from arch.utils.nonlinearity import Nonlinearity
from arch.transformer.model import Transformer

B = 2048
D = 10
T = D + 1

cfg = TransformerConfig(
    depth=1,
    multi_head_attn_config=MultiHeadAttnConfig(
        n_heads=1,
        attn_config=SelfAttnConfig(dim=D + 1, nonlinearity=Nonlinearity.NONE),
    ),
    mlp_config=MlpConfig(
        nonlinearity=Nonlinearity.NONE,
        dims=(D + 1, D + 1),
    ),
    use_layernorm=False,
)

tf = Transformer(cfg)

optim = torch.optim.Adam(tf.parameters(), lr=0.01)

for iter in range(10000):
    W = torch.normal(
        mean=torch.zeros(B, D),
        std=torch.ones(B, D),
    )
    X = 2.0 * torch.rand(B, T, D) - 1.0
    Y = torch.einsum("bd,btd->bt", W, X)

    data = torch.cat((X, Y.unsqueeze(2)), dim=2)  # [B, T, D + 1]
    data[..., -1, -1] = 0.0

    optim.zero_grad()
    loss = (tf(data)[..., -1, -1] + Y[..., -1]).square().mean()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(tf.parameters(), 10)
    optim.step()

    print(
        loss,
        tf(data)[0, -1, -1],
        -Y[0, 0] * torch.dot(X[0, 2], X[0, 0])
        - Y[0, 1] * torch.dot(X[0, 1], X[0, 2]),
    )

print(tf.attn_layers[0].heads[0].W_v)
print(tf.attn_layers[0].heads[0].W_q @ tf.attn_layers[0].heads[0].W_k.T)

import matplotlib.pyplot as plt

plt.imshow(tf.attn_layers[0].heads[0].W_v.detach().cpu().numpy())
plt.show()
plt.imshow(
    (tf.attn_layers[0].heads[0].W_q @ tf.attn_layers[0].heads[0].W_k.T)
    .detach()
    .cpu()
    .numpy()
)
plt.show()
