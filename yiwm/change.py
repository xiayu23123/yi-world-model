"""變易引擎: 老阴老阳 -> 爻变 (相变) -> 之卦分布.

Design notes:
  * no ``argmax`` on the current hexagram -- the expected yao pattern is taken
    under the full distribution (``hex_prob @ H``), so gradient flows back into
    HexagramInference through this path.
  * the map from a (continuous) yao pattern to 64-way logits uses real Hamming
    agreement ``new_yao @ H.T + (1 - new_yao) @ (1 - H).T`` (range [0, 6]).
  * Straight-Through Estimator: forward uses the hard 0/1 line-change mask,
    backward uses the sigmoid gradient.
  * RANK-AWARE THRESHOLD: the 老爻 cutoff is no longer a fixed per-position
    constant. A small net maps [ |y_k|, rank-of-k one-hot, sample mean, sample
    std ] -> an adaptive per-position threshold, so a relative-quantile moving
    rule (as in synth.py) is representable. ``thr_bias`` is initialised via
    inverse-softplus of the old constants, so training starts from the old
    behaviour.
  * every fixed tensor is a registered buffer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import BINARY_HEX


def _inv_softplus(y: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(y).clamp(min=1e-6))


class ChangeEngine(nn.Module):
    def __init__(
        self,
        n_yao: int = 6,
        init_thresholds=(0.30, 0.45, 0.50, 0.60, 0.70, 0.85),
        hidden: int = 32,
    ):
        super().__init__()
        self.n_yao = n_yao
        # feature dim: |y_k| (1) + rank one-hot (n_yao) + mean (1) + std (1)
        #            + diff_to_max = max|y| - |y_k|  (1, "距老阳还差多少")
        self.thr_net = nn.Sequential(
            nn.Linear(1 + n_yao + 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # start the net small so the threshold is governed by the bias at init
        with torch.no_grad():
            self.thr_net[-1].weight.mul_(0.01)
            self.thr_net[-1].bias.zero_()
        self.thr_bias = nn.Parameter(_inv_softplus(torch.tensor(init_thresholds)))

        self.log_tau = nn.Parameter(torch.zeros(()))       # temperature = exp(log_tau)
        self.mix = nn.Parameter(torch.tensor(-1.0))        # sigmoid -> 本卦 residual weight
        self.register_buffer("H", BINARY_HEX.clone())      # [64, 6]

    def adaptive_threshold(self, yao: torch.Tensor) -> torch.Tensor:
        """[B, 6] signed -> [B, 6] positive per-position 老爻 cutoff."""
        absy = yao.abs()                                          # [B, 6]
        order = absy.argsort(dim=1, descending=True)
        rank = order.argsort(dim=1)                              # rank[b,k], 0 = strongest
        rank_oh = F.one_hot(rank, num_classes=self.n_yao).float()  # [B, 6, 6]
        mean = absy.mean(1, keepdim=True).expand(-1, self.n_yao)
        std = absy.std(1, keepdim=True).expand(-1, self.n_yao)
        diff_to_max = absy.amax(1, keepdim=True) - absy           # [B, 6], 0 at the strongest line
        feat = torch.cat(
            [absy.unsqueeze(-1), rank_oh,
             mean.unsqueeze(-1), std.unsqueeze(-1), diff_to_max.unsqueeze(-1)],
            dim=-1,
        )                                                        # [B, 6, 10]
        raw = self.thr_net(feat).squeeze(-1) + self.thr_bias     # [B, 6]
        return F.softplus(raw)

    def forward(
        self,
        yao: torch.Tensor,          # [B, 6] signed strength
        hex_logits: torch.Tensor,   # [B, 64]
        hard: bool = False,
    ):
        tau = self.log_tau.exp().clamp(min=1e-2)
        thr = self.adaptive_threshold(yao)                # [B, 6]
        energy = yao.abs() - thr                          # >0 -> 老爻 -> 变
        hard_mask = (energy > 0).float()
        if self.training and not hard:
            soft = torch.sigmoid(energy / tau)
            change = soft + (hard_mask - soft).detach()   # STE
        else:
            change = hard_mask

        hex_prob = torch.softmax(hex_logits, dim=-1)      # [B, 64]
        cur_yao = hex_prob @ self.H                       # [B, 6] expected 爻, differentiable
        new_yao = cur_yao * (1 - change) + (1 - cur_yao) * change

        agree = new_yao @ self.H.t() + (1 - new_yao) @ (1 - self.H.t())  # [B,64] in [0,6]
        w = torch.sigmoid(self.mix)
        new_logits = w * hex_logits + (1 - w) * (agree - 3.0) * 2.0
        return new_logits, change, energy
