"""Stage 2 ablation: 5×5 grid, full obs, **CNN encoder** (vs stage-1 hand R^6).
Only the encoder changes — same DQN, same transition-check, full observability.

Run: python -m yiwm.kingdom_stage2
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import GridEncoder
from .kingdom_2d import Kingdom2D

_A = 5


class QGrid(nn.Module):
    """Q(dyn, action) on the encoder's dynamics embedding."""

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


def _enc(encoder, env):
    with torch.no_grad():
        return encoder(env.grid(), env.player())


def train_dqn(size: int = 5, episodes: int = 3500, gamma: float = 0.95,
              lr: float = 3e-4, eps: float = 0.2, seed: int = 0, aux_w: float = 0.5):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    enc = GridEncoder(size=size)
    q, qt = QGrid(), QGrid()
    qt.load_state_dict(q.state_dict())
    # auxiliary world-model head: predict the FULL next grid + player from
    # (dyn, action). Weighted heavily so `dyn` is dynamics-complete, not just
    # Q-sufficient.
    tgt_dim = 3 * size * size + 3
    aux = nn.Sequential(nn.Linear(32 + _A, 128), nn.ReLU(),
                        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, tgt_dim))
    opt = torch.optim.Adam(list(enc.parameters()) + list(q.parameters())
                           + list(aux.parameters()), lr=lr)
    buf: list = []

    def summ(G, P):
        return torch.cat([G.flatten(1), P], dim=-1)

    for ep in range(episodes):
        env = Kingdom2D(size=size, seed=seed * 7919 + ep)
        g, p = env.grid(), env.player()
        for _ in range(env.max_steps):
            with torch.no_grad():
                dyn = enc(g, p)["dyn"]
            a = rng.randint(0, _A - 1) if rng.random() < eps else int(q.all_q(dyn).argmax())
            _, r, done, _ = env.step(a)
            g1, p1 = env.grid(), env.player()
            buf.append((g, p, a, r, g1, p1, done))
            g, p = g1, p1
            if done:
                break
        buf = buf[-20_000:]
        if len(buf) >= 128:
            bt = rng.sample(buf, 128)
            G = torch.stack([b[0] for b in bt]); P = torch.stack([b[1] for b in bt])
            A = torch.tensor([b[2] for b in bt]); R = torch.tensor([b[3] for b in bt]).float()
            G1 = torch.stack([b[4] for b in bt]); P1 = torch.stack([b[5] for b in bt])
            D = torch.tensor([b[6] for b in bt]).float()
            dyn = enc(G, P)["dyn"]
            with torch.no_grad():
                tgt = R + gamma * (1 - D) * qt.all_q(qt_enc(qt, enc, G1, P1)).max(1).values
            q_loss = F.mse_loss(q(dyn, A), tgt)
            aux_pred = aux(torch.cat([dyn, F.one_hot(A, _A).float()], -1))
            aux_loss = F.mse_loss(aux_pred, summ(G1, P1))
            loss = q_loss + aux_w * aux_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(q.parameters()) + list(aux.parameters()), 5.0)
            opt.step()
        if ep % 200 == 0:
            qt.load_state_dict(q.state_dict())
    return enc, q


def qt_enc(qt, enc, G, P):
    with torch.no_grad():
        return enc(G, P)["dyn"]


@torch.no_grad()
def eval_vs_random(enc, q, size: int = 5, n: int = 150, seed: int = 1):
    gg, rr = [], []
    for ep in range(n):
        env = Kingdom2D(size=size, seed=seed * 104729 + ep)
        tot = 0.0
        for _ in range(env.max_steps):
            a = int(q.all_q(_enc(enc, env)["dyn"]).argmax())
            _, r, done, _ = env.step(a)
            tot += r
            if done:
                break
        gg.append(tot)
        env = Kingdom2D(size=size, seed=seed * 104729 + ep)
        tot = 0.0
        rnd = random.Random(ep)
        for _ in range(env.max_steps):
            _, r, done, _ = env.step(rnd.randint(0, _A - 1))
            tot += r
            if done:
                break
        rr.append(tot)
    return sum(gg) / n, sum(rr) / n


def transition_check(enc, size: int = 5, n_ep: int = 800, steps: int = 600, seed: int = 2):
    """Freeze the encoder, learn f(dyn, action) -> dyn', measure action sensitivity."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    for pm in enc.parameters():
        pm.requires_grad_(False)
    buf = []
    for ep in range(n_ep):
        env = Kingdom2D(size=size, seed=seed * 31 + ep)
        d = _enc(enc, env)["dyn"][0]
        for _ in range(env.max_steps):
            a = rng.randint(0, _A - 1)
            env.step(a)
            d1 = _enc(enc, env)["dyn"][0]
            buf.append((d, a, d1))
            d = d1
    D = torch.stack([b[0] for b in buf]); A = torch.tensor([b[1] for b in buf])
    D1 = torch.stack([b[2] for b in buf])
    dyn_dim = D.shape[1]
    tr = nn.Sequential(nn.Linear(dyn_dim + _A, 128), nn.ReLU(),
                       nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, dyn_dim), nn.Tanh())
    opt = torch.optim.Adam(tr.parameters(), lr=2e-3)
    for s in range(steps):
        idx = torch.randint(0, len(buf), (256,))
        x = torch.cat([D[idx], F.one_hot(A[idx], _A).float()], -1)
        loss = F.mse_loss(tr(x), D1[idx])
        opt.zero_grad(); loss.backward(); opt.step()

    # |f(d, 进) - f(d, 守)|
    with torch.no_grad():
        d = D[torch.randint(0, len(buf), (400,))]
        oh0 = F.one_hot(torch.zeros(400, dtype=torch.long), _A).float()
        oh2 = F.one_hot(torch.full((400,), 2), _A).float()
        sens = (tr(torch.cat([d, oh0], -1)) - tr(torch.cat([d, oh2], -1))).abs().mean().item()
    return loss.item(), sens


if __name__ == "__main__":
    enc, q = train_dqn(size=5)
    g, r = eval_vs_random(enc, q, size=5)
    mse, sens = transition_check(enc, size=5)
    print(f"\nStage 2 (5x5, CNN encoder, full obs)")
    print(f"  greedy-Q vs random          {g:+.1f}  vs {r:+.1f}   (Δ {g - r:+.1f})")
    print(f"  transition MSE / sensitivity {mse:.4f} / {sens:.4f}   "
          f"({'recovered (>0.05)' if sens > 0.05 else 'still washed out'})")
    print(f"  (stage 1: Δ +73,  sensitivity 0.026)")
