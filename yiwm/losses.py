"""Multi-task loss for the I Ching world model."""

import torch
import torch.nn.functional as F

from .constants import BINARY_HEX, MOVING_MASKS

DEFAULT_WEIGHTS = {
    "hex": 1.0,
    "hex_next": 0.5,
    "hex_next_joint": 0.5,
    "change": 0.2,        # per-position BCE, demoted in favour of the ranking term
    "rank": 0.5,          # pairwise margin: 老爻 energy > 非老爻 energy + margin
    "moving_joint": 1.0,  # 21-way mask classification (no per-yao compounding)
    "action": 1.0,
    "yao": 0.3,
    "advice": 0.2,
    "balance": 0.05,
}
_RANK_MARGIN = 0.3


def _moving_rank_loss(energy: torch.Tensor, moving: torch.Tensor, margin: float) -> torch.Tensor:
    """The moving label is 'top-k by |y| within the sample'. Match that rule
    directly: every (老爻, 非老爻) pair should be separated in energy by margin.
    """
    m = moving > 0.5                                          # [B, 6]
    pos = energy.unsqueeze(2)                                 # [B, 6, 1]
    neg = energy.unsqueeze(1)                                 # [B, 1, 6]
    valid = m.unsqueeze(2) & (~m).unsqueeze(1)               # 老爻 x 非老爻
    viol = F.relu(margin - (pos - neg)) * valid
    return viol.sum() / valid.sum().clamp(min=1)


def _soft_hex_target(yao_target: torch.Tensor, temp: float) -> torch.Tensor:
    """Per-yao sign confidence -> a distribution over the 64 hexagrams.

    p_k = P(yao k is 陽) = sigmoid(y_k / temp).  A hexagram's target mass is the
    product of the matching Bernoullis; summed over all 64 it is exactly 1, so
    no normalisation is needed. Near y_k ~ 0 the mass spreads to the Hamming-1
    neighbour instead of forcing a confident bit the observation cannot resolve.
    """
    H = BINARY_HEX.to(yao_target.device)                      # [64, 6]
    p = torch.sigmoid(yao_target / temp).clamp(1e-4, 1 - 1e-4)  # [B, 6]
    logprob = torch.log(p) @ H.t() + torch.log1p(-p) @ (1 - H).t()  # [B, 64]
    return logprob.exp()


def yi_world_loss(
    out: dict,
    batch: dict,
    weights: dict | None = None,
    soft_hex_temp: float = 0.0,
    yao_norm: tuple | None = None,
) -> dict:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    L: dict[str, torch.Tensor] = {}

    # 1. 本卦.  Hard CE by default. With soft_hex_temp > 0, use a soft target
    #    built from per-yao sign confidence -- only helps when yao_target is on a
    #    calibrated common scale (synth: non-moving |force| < 0.74, moving > 0.92;
    #    eco z-scores are NOT, and softening there collapses argmax acc).
    if "yao_target" in batch and soft_hex_temp and soft_hex_temp > 0:
        soft_t = _soft_hex_target(batch["yao_target"], soft_hex_temp)
        L["hex"] = -(soft_t * F.log_softmax(out["hex_logits"], dim=1)).sum(1).mean()
    else:
        L["hex"] = F.cross_entropy(out["hex_logits"], batch["hex"])

    # 2. 之卦 classification
    if "hex_next" in batch:
        L["hex_next"] = F.cross_entropy(out["hex_logits_next"], batch["hex_next"])
        if "hex_logits_next_joint" in out:
            L["hex_next_joint"] = F.cross_entropy(out["hex_logits_next_joint"], batch["hex_next"])

    # 3. 动爻 mask: per-position BCE + pairwise ranking on the raw energy,
    #    plus a JOINT head that predicts the whole moving set as one of 21
    #    classes (breaks the per-yao (acc)^6 compounding on 之卦).
    if "moving" in batch:
        L["change"] = F.binary_cross_entropy(
            out["change"].clamp(1e-5, 1 - 1e-5), batch["moving"].float()
        )
        if "change_energy" in out:
            L["rank"] = _moving_rank_loss(
                out["change_energy"], batch["moving"], _RANK_MARGIN
            )
        if "moving_logits" in out:
            mm = MOVING_MASKS.to(batch["moving"].device)
            eq = (batch["moving"].unsqueeze(1) == mm.unsqueeze(0)).all(-1)      # [B, 21]
            valid = eq.any(-1)
            if valid.any():
                idx = eq.float().argmax(-1)
                L["moving_joint"] = F.cross_entropy(
                    out["moving_logits"][valid], idx[valid]
                )

    # 4. 行动
    if "action" in batch:
        L["action"] = F.cross_entropy(out["policy"]["action_logits"], batch["action"])

    # 5. encoder 的 爻向量直接回归到 yao_target.  Without this, hard 本卦 CE only
    #    constrains the SIGN of yao, the encoder saturates |yao| -> 1 everywhere,
    #    and ChangeEngine's rank-aware threshold (needs magnitude spread) starves.
    #    tanh(.) squashes to (-1,1) (also tames z-score tails -> Huber over MSE is
    #    a mild extra guard). yao_norm=(mean,std) optionally standardises channels
    #    first -- off by default: eco's small trend channel SHOULD stay small so
    #    the rank features line up with the quantile-based moving label.
    if "yao_target" in batch:
        yt = batch["yao_target"]
        if yao_norm is not None:
            mean, std = yao_norm
            yt = (yt - mean.to(yt)) / std.to(yt)
        L["yao"] = F.smooth_l1_loss(out["yao_next"], torch.tanh(yt), beta=0.1)

    # 6. 每爻吉凶 regression (auxiliary policy head)
    if "yao_target" in batch:
        L["advice"] = F.mse_loss(out["policy"]["yao_advice"], batch["yao_target"])

    # 6. 五行平衡: group yang-power by 五行 category, penalise the variance
    en = out["entity_next"]                                   # [B, N, D]
    cats = batch["entity_cats"].long()                        # [B, N]
    yang = torch.relu(en).sum(-1)                             # [B, N]
    power = torch.zeros(en.size(0), 5, device=en.device)
    power = power.scatter_add(1, cats, yang)                  # [B, 5]
    L["balance"] = power.var(dim=1).mean()

    L["total"] = sum(w.get(k, 1.0) * v for k, v in L.items())
    return L
