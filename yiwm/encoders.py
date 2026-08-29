"""Grid observation encoder for the spatial kingdom envs.

Stage-1 mapped an N×N grid to R^6 by hand — too lossy, the transition lost
action-dependence (`kingdom_2d`: sensitivity 0.026 < 0.05). `GridEncoder` is a
small CNN with **two heads off a shared backbone**:

  yao : [B, 6]  tanh   — the interpretable hexagram vector (as before)
  dyn : [B, D]  tanh    — a richer space the transition actually learns in,
                          so "进 left" vs "进 right" stay distinguishable
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GridEncoder(nn.Module):
    def __init__(self, channels: int = 3, size: int = 5, player_dim: int = 3,
                 dyn_dim: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        feat = 32 * size * size + player_dim
        # yao: bounded + interpretable.  dyn: LayerNorm, NOT tanh -- a tanh on a
        # 32-d embedding saturates to a constant vector under DQN and kills it.
        self.yao_head = nn.Sequential(
            nn.Linear(feat, 64), nn.ReLU(), nn.Linear(64, 6), nn.Tanh())
        self.dyn_head = nn.Sequential(
            nn.Linear(feat, 64), nn.ReLU(), nn.Linear(64, dyn_dim), nn.LayerNorm(dyn_dim))

    def forward(self, grid: torch.Tensor, player: torch.Tensor) -> dict:
        if grid.dim() == 3:
            grid, player = grid.unsqueeze(0), player.unsqueeze(0)
        f = torch.cat([self.cnn(grid), player], dim=-1)
        return {"yao": self.yao_head(f), "dyn": self.dyn_head(f)}
