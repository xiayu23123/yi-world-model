"""易经世界模型: 观测 -> 卦 -> 之卦 -> 行动."""

import torch
import torch.nn as nn

from .change import ChangeEngine
from .constants import BINARY_TO_KING_WEN, KING_WEN_TO_BINARY
from .encoder import YinYangEncoder
from .hexagram import HexagramInference
from .policy import TemporalPositionalPolicy
from .wuxing import WuxingDynamics


class YiWorldModel(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_entities: int = 5,
        embed_dim: int = 64,
        entity_state_dim: int = 6,
        step: float = 0.1,
    ):
        super().__init__()
        assert entity_state_dim == 6, "entity state is aggregated into 6 yao"
        self.n_entities = n_entities
        self.step = step

        self.encoder = YinYangEncoder(obs_dim)
        self.hexinf = HexagramInference(embed_dim)
        self.wuxing = WuxingDynamics(state_dim=entity_state_dim)
        self.change = ChangeEngine()
        self.policy = TemporalPositionalPolicy(embed_dim)

    def forward(
        self,
        obs: torch.Tensor,             # [B, obs_dim]
        entity_states: torch.Tensor,   # [B, N, 6]
        entity_cats: torch.Tensor,     # [B, N] long 0..4
        entity_adj: torch.Tensor,      # [B, N, N]
        hard: bool = False,
    ) -> dict:
        yao = self.encoder(obs)                                   # [B, 6]
        hex_logits = self.hexinf(yao)                             # [B, 64]

        delta = self.wuxing(entity_states, entity_cats, entity_adj)
        entity_next = entity_states + self.step * delta           # [B, N, 6]

        agg = torch.tanh(entity_next.mean(dim=1))                 # [B, 6]
        yao_next = 0.5 * yao + 0.5 * agg

        hex_logits_next, change, change_energy = self.change(yao_next, hex_logits, hard=hard)

        hex_feat_all = self.hexinf.hex_features()                 # [64, E]
        hex_feat = torch.softmax(hex_logits, dim=-1) @ hex_feat_all   # [B, E]
        pol = self.policy(hex_feat, yao_next)

        return {
            "yao": yao,
            "yao_next": yao_next,
            "hex_logits": hex_logits,
            "hex_logits_next": hex_logits_next,
            "change": change,
            "change_energy": change_energy,
            "policy": pol,
            "entity_next": entity_next,
        }

    @staticmethod
    def to_king_wen(binary_idx: torch.Tensor) -> torch.Tensor:
        return BINARY_TO_KING_WEN.to(binary_idx.device)[binary_idx]

    @staticmethod
    def from_king_wen(kw_minus_1: torch.Tensor) -> torch.Tensor:
        return KING_WEN_TO_BINARY.to(kw_minus_1.device)[kw_minus_1]
