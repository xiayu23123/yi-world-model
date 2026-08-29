"""Model-free Q-learning on `TinyKingdom` — the fix for the failed naïve MPC.

The earlier planner used an action-blind value head `V(s)` and picked
`argmax_a V(f(s,a))`, which wrongly assumes *reaching* a state = *earning* its
value. `Q(s, a)` scores "take this action in this state", so `argmax_a Q(s,a)`
is a correct 1-step planner. Trained by DQN on real env interaction (episodes
are microseconds — no need for a learned imagination model here).

Run: python -m yiwm.q_planner
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tiny_kingdom import TinyKingdom

_A = 5


class QNet(nn.Module):
    def __init__(self, state_dim: int = 6, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + _A, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        a = F.one_hot(action.view(-1).long(), _A).float()
        return self.net(torch.cat([state, a], dim=-1)).squeeze(-1)   # [B]

    @torch.no_grad()
    def all_q(self, state: torch.Tensor) -> torch.Tensor:
        s = state.unsqueeze(0) if state.dim() == 1 else state          # [B,6]
        B = s.shape[0]
        st = s.repeat_interleave(_A, 0)
        a = torch.arange(_A).repeat(B)
        return self.forward(st, a).view(B, _A)


def train_q(episodes: int = 4000, gamma: float = 0.95, lr: float = 1e-3,
            eps: float = 0.2, seed: int = 0, env_fn=None):
    env_fn = env_fn or (lambda s: TinyKingdom(seed=s))
    torch.manual_seed(seed)
    rng = random.Random(seed)
    q, qt = QNet(), QNet()
    qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)
    buf: list = []

    for ep in range(episodes):
        env = env_fn(seed * 7919 + ep)
        s = env.obs()
        for _ in range(env.max_steps):
            a = rng.randint(0, _A - 1) if rng.random() < eps else int(q.all_q(s).argmax())
            s1, r, done, _ = env.step(a)
            buf.append((s, a, r, s1, done))
            s = s1
            if done:
                break
        if len(buf) > 20_000:
            buf = buf[-20_000:]

        # one gradient step on a minibatch
        if len(buf) >= 256:
            batch = rng.sample(buf, 256)
            S = torch.stack([b[0] for b in batch])
            A = torch.tensor([b[1] for b in batch])
            R = torch.tensor([b[2] for b in batch]).float()
            S1 = torch.stack([b[3] for b in batch])
            D = torch.tensor([b[4] for b in batch]).float()
            with torch.no_grad():
                tgt = R + gamma * (1 - D) * qt.all_q(S1).max(1).values
            loss = F.mse_loss(q(S, A), tgt)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 200 == 0:
            qt.load_state_dict(q.state_dict())
    return q


@torch.no_grad()
def eval_greedy_vs_random(q: QNet, n: int = 200, seed: int = 1, env_fn=None):
    env_fn = env_fn or (lambda s: TinyKingdom(seed=s))
    gq, rr = [], []
    for ep in range(n):
        env = env_fn(seed * 104729 + ep)
        s, tot = env.obs(), 0.0
        for _ in range(env.max_steps):
            s, r, done, _ = env.step(int(q.all_q(s).argmax()))
            tot += r
            if done:
                break
        gq.append(tot)
        env = env_fn(seed * 104729 + ep)
        tot = 0.0
        g = random.Random(ep)
        for _ in range(env.max_steps):
            _, r, done, _ = env.step(g.randint(0, _A - 1))
            tot += r
            if done:
                break
        rr.append(tot)
    return sum(gq) / n, sum(rr) / n


if __name__ == "__main__":
    q = train_q()
    g, r = eval_greedy_vs_random(q)
    print(f"greedy-Q reward {g:+.2f}   random {r:+.2f}   "
          f"(Δ {g - r:+.2f}  -> {'Q wins' if g > r + 0.5 else 'no better'})")
