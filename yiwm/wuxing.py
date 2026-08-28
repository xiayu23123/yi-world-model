"""五行动力学网络.

Entities are nodes in a 五行 graph. The generation/control field
(WUXING_MATRIX, +1 生 / -1 克) is combined with an external relation graph,
then drives a *multiplicative* (Lotka-Volterra style) state update:

    delta_i = rate * gate(state_i) * f(state_i, sum_j field_ij * state_j)

The ``gate(state_i)`` factor is what makes it multiplicative rather than the
purely additive message passing in the original sketch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import WUXING_MATRIX


class WuxingDynamics(nn.Module):
    def __init__(self, state_dim: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.state_dim = state_dim
        self.register_buffer("wx", WUXING_MATRIX.clone())  # [5,5], not learned
        self.msg = nn.Sequential(
            nn.Linear(state_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        self.rate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        states: torch.Tensor,      # [B, N, D]
        categories: torch.Tensor,  # [B, N] long in 0..4
        adj: torch.Tensor,         # [B, N, N] external relation graph
    ) -> torch.Tensor:
        B, N, D = states.shape
        oh = F.one_hot(categories.long(), num_classes=5).float()        # [B,N,5]
        wx = self.wx.unsqueeze(0).expand(B, 5, 5)
        effect = torch.bmm(torch.bmm(oh, wx), oh.transpose(1, 2))       # [B,N,N] in {-1,0,1}
        field = adj * effect
        neigh = torch.bmm(field, states)                               # [B,N,D]
        inter = self.msg(torch.cat([states, neigh], dim=-1))           # [B,N,D]
        gate = torch.tanh(states)                                      # multiplicative gating
        return self.rate * gate * inter
