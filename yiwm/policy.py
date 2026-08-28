"""時位策略网络: 卦 feature + 时位 (当位/不当位) -> 行动分布.

Fixes vs the original sketch:
  * ``hex_feat`` is passed in (the model feeds the distribution-weighted
    hexagram embedding); no undefined ``self.hex_embed``.
  * ``yao_advice`` is a real ``nn.Linear`` head, not a fresh ``torch.randn``
    every forward.
  * fixed 爻位 parity is a registered buffer.
"""

import torch
import torch.nn as nn

from .constants import YAO_POS_PARITY


class TemporalPositionalPolicy(nn.Module):
    ACTIONS = ["jin", "tui", "shou", "bian", "dai"]      # 进 退 守 变 待
    ACTIONS_CN = ["进", "退", "守", "变", "待"]

    def __init__(self, embed_dim: int = 64, n_actions: int = 5):
        super().__init__()
        self.n_actions = n_actions
        self.pos_embed = nn.Embedding(6, embed_dim)
        self.proper_embed = nn.Embedding(2, embed_dim)   # 0 = 不当位, 1 = 当位
        self.head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, n_actions + 1),         # actions + 1 intensity
        )
        self.advice = nn.Linear(6, 6)
        self.register_buffer("parity", YAO_POS_PARITY.clone())          # [6]
        self.register_buffer("pos_ids", torch.arange(6))

    def forward(self, hex_feat: torch.Tensor, yao: torch.Tensor) -> dict:
        """hex_feat [B, E], yao [B, 6] -> dict."""
        is_yang = (yao > 0).float()                                     # [B, 6]
        proper = (is_yang == self.parity).long()                        # [B, 6]
        w = torch.softmax(yao.abs(), dim=1).unsqueeze(-1)               # [B, 6, 1] 动爻权重
        pos = (self.pos_embed(self.pos_ids).unsqueeze(0) * w).sum(1)    # [B, E]
        prop = (self.proper_embed(proper) * w).sum(1)                   # [B, E]
        feat = torch.cat([hex_feat, pos + prop], dim=-1)               # [B, 2E]
        out = self.head(feat)
        return {
            "action_logits": out[:, : self.n_actions],
            "intensity": torch.sigmoid(out[:, self.n_actions:]),
            "yao_advice": self.advice(yao),                             # [B, 6] 每爻吉凶
        }
