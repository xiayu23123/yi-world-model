"""Full two-stage world-model pipeline on Kingdom2D:

  stage 1  wm_pretrain  -> encoder + transition (dynamics-first, no Q)
  stage 2  FREEZE both, train Q(dyn, action) by DQN on the frozen latent
  stage 3  compare: random / model-free greedy / model-based MPC
           (MPC rolls the FROZEN transition forward, scoring with Q)

Run: python -m yiwm.wm_pipeline
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kingdom_2d import Kingdom2D
from .wm_pretrain import TransitionDyn, pretrain, report

_A = 5


class QLatent(nn.Module):
    def __init__(self, dyn_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dyn_dim + _A, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, dyn, action):
        a = F.one_hot(action.view(-1).long(), _A).float()
        return self.net(torch.cat([dyn, a], dim=-1)).squeeze(-1)

    def all_q(self, dyn):
        d = dyn if dyn.dim() == 2 else dyn.unsqueeze(0)
        B = d.shape[0]
        return self.forward(d.repeat_interleave(_A, 0), torch.arange(_A).repeat(B)).view(B, _A)


@torch.no_grad()
def _z(enc, env):
    return enc(env.grid(), env.player())["dyn"][0]


def train_q_frozen(enc, size: int = 5, episodes: int = 3000, gamma: float = 0.95,
                   lr: float = 1e-3, eps: float = 0.2, seed: int = 0):
    for p in enc.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    q, qt = QLatent(), QLatent()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf: list = []
    for ep in range(episodes):
        env = Kingdom2D(size=size, seed=seed * 7919 + ep)
        z = _z(enc, env)
        for _ in range(env.max_steps):
            a = rng.randint(0, _A - 1) if rng.random() < eps else int(q.all_q(z).argmax())
            _, r, done, _ = env.step(a)
            z1 = _z(enc, env)
            buf.append((z, a, r, z1, done))
            z = z1
            if done:
                break
        buf = buf[-20_000:]
        if len(buf) >= 128:
            bt = rng.sample(buf, 128)
            Z = torch.stack([b[0] for b in bt]); A = torch.tensor([b[1] for b in bt])
            R = torch.tensor([b[2] for b in bt]).float()
            Z1 = torch.stack([b[3] for b in bt]); D = torch.tensor([b[4] for b in bt]).float()
            with torch.no_grad():
                tgt = R + gamma * (1 - D) * qt.all_q(Z1).max(1).values
            loss = F.mse_loss(q(Z, A), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 200 == 0:
            qt.load_state_dict(q.state_dict())
    return q


@torch.no_grad()
def _mpc_action(z, q, tr, horizon: int, gamma: float) -> int:
    best_a, best_v = 0, -1e9
    for a0 in range(_A):
        zt, tot, disc = z.unsqueeze(0), 0.0, 1.0
        at = torch.tensor([a0])
        for h in range(horizon):
            tot += disc * float(q(zt, at))
            zt = tr(zt, at)
            disc *= gamma
            at = q.all_q(zt).argmax(1)                 # greedy after the first step
        if tot > best_v:
            best_a, best_v = a0, tot
    return best_a


@torch.no_grad()
def eval_policies(enc, tr, q, size: int = 5, n: int = 150, horizon: int = 4,
                  gamma: float = 0.95, seed: int = 1):
    out = {"random": [], "greedy": [], "mpc": []}
    for ep in range(n):
        for name in out:
            env = Kingdom2D(size=size, seed=seed * 104729 + ep)
            z = _z(enc, env)
            tot = 0.0
            g = random.Random(ep)
            for _ in range(env.max_steps):
                if name == "random":
                    a = g.randint(0, _A - 1)
                elif name == "greedy":
                    a = int(q.all_q(z).argmax())
                else:
                    a = _mpc_action(z, q, tr, horizon, gamma)
                _, r, done, _ = env.step(a)
                z = _z(enc, env)
                tot += r
                if done:
                    break
            out[name].append(tot)
    return {k: sum(v) / len(v) for k, v in out.items()}


if __name__ == "__main__":
    enc, tr, ploss = pretrain(size=5)
    _, sens0 = report(enc, tr, size=5)
    print(f"stage 1  wm-pretrain: loss {ploss:.4f}, transition sensitivity {sens0:.3f}")

    q = train_q_frozen(enc, size=5)
    _, sens1 = report(enc, tr, size=5)                # encoder/transition unchanged
    res = eval_policies(enc, tr, q, size=5)
    base = res["random"]
    print(f"stage 2  Q on frozen latent (transition sensitivity still {sens1:.3f})")
    print(f"stage 3  policies (mean return over 15 steps):")
    for k in ("random", "greedy", "mpc"):
        d = (res[k] - base) / abs(base) * 100
        print(f"    {k:7s} {res[k]:+7.1f}   ({d:+.0f}% vs random)")
