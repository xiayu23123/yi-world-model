"""World-model pretraining (JEPA-style) on Kingdom2D — the fix Stage 2 pointed
to. Encoder + transition trained **jointly, no Q, no reward**:

    loss = ‖ transition(enc(g_t).dyn, a_t) − enc(g_{t+1}).dyn.detach() ‖²
         + β · variance_reg(enc(g_t).dyn)          # anti-collapse (VICReg-lite)

Then the check: does an action-conditioned transition in this dynamics-first
latent recover action-sensitivity (Stage 1 hand-R^6 = 0.026, Stage 2 DQN-latent
= 0.014)?

Run: python -m yiwm.wm_pretrain
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import GridEncoder
from .kingdom_2d import Kingdom2D

_A = 5


class TransitionDyn(nn.Module):
    def __init__(self, dyn_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dyn_dim + _A, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, dyn_dim),
        )

    def forward(self, dyn, action):
        a = F.one_hot(action.view(-1).long(), _A).float()
        return self.net(torch.cat([dyn, a], dim=-1))


def _var_reg(z: torch.Tensor) -> torch.Tensor:
    """Push every latent dim to have batch std >= 1 (stops collapse to a point)."""
    return F.relu(1.0 - z.std(0)).mean()


def pretrain(size: int = 5, n_ep: int = 1200, epochs: int = 40, lr: float = 1e-3,
             beta: float = 1.0, seed: int = 0):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    enc = GridEncoder(size=size)
    tr = TransitionDyn()
    opt = torch.optim.Adam(list(enc.parameters()) + list(tr.parameters()), lr=lr)

    G, P, A, G1, P1 = [], [], [], [], []
    for ep in range(n_ep):
        env = Kingdom2D(size=size, seed=seed * 61 + ep)
        g, p = env.grid(), env.player()
        for _ in range(env.max_steps):
            a = rng.randint(0, _A - 1)
            env.step(a)
            g1, p1 = env.grid(), env.player()
            G.append(g); P.append(p); A.append(a); G1.append(g1); P1.append(p1)
            g, p = g1, p1
    G, P = torch.stack(G), torch.stack(P)
    A = torch.tensor(A)
    G1, P1 = torch.stack(G1), torch.stack(P1)
    n = len(A)

    for e in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            z = enc(G[idx], P[idx])["dyn"]
            z1 = enc(G1[idx], P1[idx])["dyn"].detach()
            pred = tr(z, A[idx])
            loss = F.mse_loss(pred, z1) + beta * _var_reg(z)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(tr.parameters()), 5.0)
            opt.step()
    return enc, tr, loss.item()


@torch.no_grad()
def report(enc, tr, size: int = 5, n: int = 400, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    Z = []
    for _ in range(n):
        env = Kingdom2D(size=size, seed=int(torch.randint(0, 1 << 30, (1,), generator=g)))
        for _ in range(int(torch.randint(0, 10, (1,), generator=g))):
            env.step(int(torch.randint(0, _A, (1,), generator=g)))
        Z.append(enc(env.grid(), env.player())["dyn"][0])
    Z = torch.stack(Z)
    across = Z.std(0).mean().item()
    s0 = tr(Z, torch.zeros(n, dtype=torch.long))
    s2 = tr(Z, torch.full((n,), 2))
    sens = (s0 - s2).abs().mean().item()
    return across, sens


if __name__ == "__main__":
    enc, tr, loss = pretrain(size=5)
    across, sens = report(enc, tr, size=5)
    print(f"world-model pretrain (5x5, no Q/reward)")
    print(f"  final loss              {loss:.4f}")
    print(f"  latent across-state std {across:.3f}   (0 = collapsed)")
    print(f"  action sensitivity      {sens:.4f}   "
          f"({'HYPOTHESIS HOLDS (>0.05)' if sens > 0.05 else 'CNN insufficient at 5x5' if sens < 0.03 else 'marginal'})")
    print(f"  (stage 1 hand-R^6 0.026 / stage 2 DQN-latent 0.014)")
