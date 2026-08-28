"""阴阳编码器: 观测 -> 6 爻强度."""

import torch
import torch.nn as nn


class YinYangEncoder(nn.Module):
    """Map an observation vector to 6 signed yao strengths in (-1, 1).

    sign  -> 阴 / 阳
    |val| -> 变的势能 (老阴老阳 large, 少阴少阳 small)

    The original design used 12 dims with a forced ``y[:,i,0] = -y[:,i,1]``
    constraint; that pair is fully redundant with a single signed scalar, so
    we drop it. The state lives in R^6.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, n_yao: int = 6):
        super().__init__()
        self.n_yao = n_yao
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, n_yao),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """[batch, input_dim] -> [batch, 6]."""
        return torch.tanh(self.net(obs))
