"""易经世界模型: 观测 -> 卦 -> 之卦 -> 行动."""

import torch
import torch.nn as nn

from .change import ChangeEngine
from .constants import BINARY_TO_KING_WEN, KING_WEN_TO_BINARY, MOVING_MASKS
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

        # joint 動爻 head: predict the whole moving SET as one of 21 classes,
        # so 之卦 is not gated by per-yao independence ((per-yao acc)^6).
        self.register_buffer("moving_masks", MOVING_MASKS.clone())        # [21, 6]
        self.moving_head = nn.Sequential(
            nn.Linear(6 + embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, MOVING_MASKS.shape[0]),
        )

    def forward(
        self,
        obs: torch.Tensor,             # [B, obs_dim]
        entity_states: torch.Tensor,   # [B, N, 6]
        entity_cats: torch.Tensor,     # [B, N] long 0..4
        entity_adj: torch.Tensor,      # [B, N, N]
        hard: bool = False,
        yao_override: torch.Tensor | None = None,   # [B, 6] -- skip the encoder (structured input)
    ) -> dict:
        yao = self.encoder(obs) if yao_override is None else yao_override  # [B, 6]
        hex_logits = self.hexinf(yao)                             # [B, 64]

        delta = self.wuxing(entity_states, entity_cats, entity_adj)
        entity_next = entity_states + self.step * delta           # [B, N, 6]

        agg = torch.tanh(entity_next.mean(dim=1))                 # [B, 6]
        yao_next = 0.5 * yao + 0.5 * agg

        hex_logits_next, change, change_energy = self.change(yao_next, hex_logits, hard=hard)

        hex_feat_all = self.hexinf.hex_features()                 # [64, E]
        hex_feat = torch.softmax(hex_logits, dim=-1) @ hex_feat_all   # [B, E]
        pol = self.policy(hex_feat, yao_next)

        # joint mask prediction + the 之卦 it implies (differentiable via
        # soft mask under the class distribution; hard argmax when hard=True)
        moving_logits = self.moving_head(torch.cat([yao_next, hex_feat], dim=-1))  # [B, 21]
        mprob = torch.softmax(moving_logits, dim=-1)
        mask_soft = mprob @ self.moving_masks                     # [B, 6] expected mask
        mask = self.moving_masks[moving_logits.argmax(-1)] if hard else mask_soft
        hex_prob = torch.softmax(hex_logits, dim=-1)              # [B, 64]
        cur_yao = hex_prob @ self.change.H                        # [B, 6] expected 爻
        new_yao_j = cur_yao * (1 - mask) + (1 - cur_yao) * mask
        H = self.change.H
        agree = new_yao_j @ H.t() + (1 - new_yao_j) @ (1 - H.t())  # [B, 64] in [0,6]
        hex_logits_next_joint = (agree - 3.0) * 2.0

        return {
            "yao": yao,
            "yao_next": yao_next,
            "hex_logits": hex_logits,
            "hex_logits_next": hex_logits_next,
            "hex_logits_next_joint": hex_logits_next_joint,
            "moving_logits": moving_logits,
            "change": change,
            "change_energy": change_energy,
            "policy": pol,
            "entity_next": entity_next,
        }

    def moving_mask_index(self, moving: torch.Tensor) -> torch.Tensor:
        """[B, 6] 0/1 -> [B] index into moving_masks; -1 if not a 1/2-line mask."""
        eq = (moving.unsqueeze(1) == self.moving_masks.unsqueeze(0)).all(-1)  # [B, 21]
        has = eq.any(-1)
        return torch.where(has, eq.float().argmax(-1), torch.full_like(has, -1, dtype=torch.long))

    @torch.no_grad()
    def rollout(self, yao: torch.Tensor, steps: int = 8, decay: float = 0.7):
        """Iterate 本卦 -> 之卦 -> 本卦 ... as a toy dynamical system.

        No new observation arrives, so the driver is the persisted force:
        the 老爻 flip their sign (they discharged) and the whole vector decays
        toward 0. It converges to a fixed point (no line 老 -> `stop='fixed'`)
        or a repeating 卦 (`stop='cycle'`), or runs out of `steps`.

        yao: [6] or [1,6].  Returns list of dicts, one per step:
          {hex_k, moving (6), hex_next_k, energy_mag}
        """
        y = yao.view(1, 6).float()
        hex_all = self.hexinf.hex_features()
        # below any learned 老爻 threshold -> no line can move -> equilibrium
        quiet = float(self.change.adaptive_threshold(y).min())
        seen: dict[int, int] = {}
        out = []
        for t in range(steps):
            hl = self.hexinf(y)
            k = int(hl.argmax(-1))
            hex_feat = torch.softmax(hl, -1) @ hex_all
            mlog = self.moving_head(torch.cat([y, hex_feat], dim=-1))
            mask = self.moving_masks[mlog.argmax(-1)]                 # [1,6]
            still = y.abs().max().item() < quiet
            flip = 0 if still else sum(int(b) << i for i, b in enumerate(mask[0].tolist()))
            k_next = k ^ flip
            out.append({
                "hex_k": k, "moving": [0] * 6 if still else mask[0].int().tolist(),
                "hex_next_k": k_next, "mag": round(y.abs().mean().item(), 3),
            })
            if still:
                out[-1]["stop"] = "fixed"
                break
            if k in seen:
                out[-1]["stop"] = f"cycle(len {t - seen[k]})"
                break
            seen[k] = t
            y = y * (1 - 2 * mask) * decay                           # flip 老爻, decay all
        return out

    @staticmethod
    def to_king_wen(binary_idx: torch.Tensor) -> torch.Tensor:
        return BINARY_TO_KING_WEN.to(binary_idx.device)[binary_idx]

    @staticmethod
    def from_king_wen(kw_minus_1: torch.Tensor) -> torch.Tensor:
        return KING_WEN_TO_BINARY.to(kw_minus_1.device)[kw_minus_1]
