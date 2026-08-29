"""P3 (minimal 阶段二): a continuous, differentiable one-step transition
`f: (yao R^6, action) -> yao' R^6` that replaces the discrete flip+decay rule
inside `rollout`.

It does NOT invent new physics -- with no real environment, the only available
target is the model's own one-step dynamics, so `LearnedTransition` *distills*
the ChangeEngine flip + decay into a smooth map. The point of P3 is the
comparison: does a smooth learned map reproduce the same cycle structure the
discrete rule shows? If yes, the periodicity is a property of the dynamics, not
an artefact of the flip.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_MOVING_MASKS


class LearnedTransition(nn.Module):
    def __init__(self, state_dim: int = 6, action_dim: int = 5, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, state_dim),                    # predicts delta
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = F.one_hot(action.long(), num_classes=5).float()
        delta = self.net(torch.cat([state, a], dim=-1))
        return (state + delta).clamp(-1.0, 1.0)


@torch.no_grad()
def _one_step_target(model, yao: torch.Tensor, decay: float) -> torch.Tensor:
    """The model's own next-yao under the discrete rule (flip 老爻, decay)."""
    hl = model.hexinf(yao)
    hex_feat = torch.softmax(hl, -1) @ model.hexinf.hex_features()
    mlog = model.moving_head(torch.cat([yao, hex_feat], dim=-1))
    mask = model.moving_masks[mlog.argmax(-1)]
    return (yao * (1 - 2 * mask) * decay).clamp(-1.0, 1.0)


def train_transition(model, make_batch, steps: int = 2000, bs: int = 256,
                     lr: float = 2e-3, decay: float = 0.9, seed: int = 0,
                     pool: int = 8192):
    """Distil the model's one-step dynamics into a LearnedTransition.

    A fixed pool of `yao_target` vectors is drawn once (generating fresh synth
    batches every step is dominated by text/hash cost); targets are the model's
    discrete next-yao. Action is sampled uniformly -- the discrete rule ignores
    it, so the net learns action is ~inert here; that changes with a real env.
    """
    torch.manual_seed(seed)
    ys = torch.cat([make_batch(min(bs, pool), seed=10_000 + i)["yao_target"]
                    for i in range((pool + bs - 1) // bs)])[:pool].clamp(-1, 1)
    tr = LearnedTransition()
    opt = torch.optim.Adam(tr.parameters(), lr=lr)
    model.eval()
    loss = torch.tensor(0.0)
    for s in range(1, steps + 1):
        idx = torch.randint(0, ys.shape[0], (bs,))
        y = ys[idx]
        a = torch.randint(0, 5, (bs,))
        with torch.no_grad():
            y_next = _one_step_target(model, y, decay)        # batched
        loss = F.mse_loss(tr(y, a), y_next)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return tr, loss.item()


@torch.no_grad()
def rollout_learned(model, tr: LearnedTransition, yao: torch.Tensor,
                    steps: int = 40, action: int = 4):
    """Same bookkeeping as model.rollout but the state update is `tr`."""
    y = yao.view(1, 6).float()
    seen: dict[int, int] = {}
    out = []
    a = torch.tensor([action])
    for t in range(steps):
        k = int(model.hexinf(y).argmax(-1))
        out.append({"hex_k": k, "mag": round(y.abs().mean().item(), 3)})
        if k in seen:
            out[-1]["stop"] = f"cycle(len {t - seen[k]})"
            out[-1]["cycle_len"] = t - seen[k]
            out[-1]["cycle_members"] = [s["hex_k"] for s in out[seen[k]:t]]
            break
        seen[k] = t
        y = tr(y, a)
    else:
        out[-1]["stop"] = "maxsteps"
    return out
