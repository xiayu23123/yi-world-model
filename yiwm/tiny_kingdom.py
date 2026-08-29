"""A minimal *causal* environment: actions genuinely change the next state.
This is what `market_transition` could not provide (a price series has no
action effect). Here `f(yao, action)` has something real to learn.

State: resource / morale / threat (0-100), mapped deterministically to R^6.
Actions 0..4 = 进 退 守 变 待, each with a fixed causal effect + action-
independent process noise.

Run: python -m yiwm.tiny_kingdom        # collect -> train -> verify
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transition import LearnedTransition

# active-yao -> the "proper" action at that 时位 (same map the eco/synth policy learns)
_PROPER = {0: 4, 1: 0, 2: 2, 3: 3, 4: 2, 5: 1}
# per-action (Δresource, Δmorale, Δthreat) at intensity 1
_EFFECT = {
    0: (-12, +8, +18),   # 进 expand
    1: (-5, -5, -15),    # 退 contract
    2: (+8, +12, +2),    # 守 develop
    3: (-8, +3, +10),    # 变 transform
    4: (+2, -3, +5),     # 待 wait
}


class TinyKingdom:
    def __init__(self, seed: int | None = None, max_steps: int = 20):
        self.rng = random.Random(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self) -> torch.Tensor:
        self.resource = self.rng.uniform(30, 70)
        self.morale = self.rng.uniform(30, 70)
        self.threat = self.rng.uniform(10, 50)
        self.t = 0
        return self.obs()

    def obs(self) -> torch.Tensor:
        """Deterministic R^6 (process noise lives in step, not here)."""
        r, m, k = self.resource / 100, (self.morale - 50) / 100, self.threat / 100
        y = torch.tensor([
            r * 0.9,                 # 初 resource base
            r * 0.8 + m * 0.2,       # 二 resource flow + morale
            0.0,                     # 三 (filled below, balance)
            k * 0.8,                 # 四 tension near the top
            r * 0.7 + m * 0.2,       # 五 resource peak + morale
            k * 0.9,                 # 上 extreme pressure
        ])
        y[2] = (y[0] + y[1] + y[3] + y[4]) / 4
        return y.clamp(0.0, 1.0)

    def step(self, action: int, intensity: float = 0.6):
        self.t += 1
        dr, dm, dk = _EFFECT[action]
        self.resource += dr * intensity + self.rng.uniform(-3, 5)
        self.morale += dm * intensity + self.rng.uniform(-5, 5)
        self.threat += dk * intensity + self.rng.uniform(-2, 8)
        self.resource = min(95.0, max(5.0, self.resource))
        self.morale = min(95.0, max(5.0, self.morale))
        self.threat = min(100.0, max(0.0, self.threat))

        active = int(self.obs().argmax())
        reward = 1.0 if action == _PROPER[active] else -0.5
        if self.resource < 20:
            reward -= 1.0
        if self.morale < 20:
            reward -= 1.0
        done = self.t >= self.max_steps or self.resource < 10 or self.morale < 10
        return self.obs(), reward, done, {"resource": self.resource,
                                          "morale": self.morale, "threat": self.threat}


# --------------------------------------------------------------------------- #
def collect(n_episodes: int = 1500, seed: int = 0, env_fn=None):
    """(yao_t, action, yao_next, reward) with uniform-random actions."""
    env_fn = env_fn or (lambda s: TinyKingdom(seed=s))
    rng = random.Random(seed)
    buf = []
    for ep in range(n_episodes):
        env = env_fn(seed * 9973 + ep)
        y = env.obs()
        for _ in range(env.max_steps):
            a = rng.randint(0, 4)
            y1, r, done, _ = env.step(a)
            buf.append((y, a, y1, r))
            y = y1
            if done:
                break
    return buf


def train(buf, epochs: int = 40, lr: float = 2e-3, seed: int = 0):
    torch.manual_seed(seed)
    tr = LearnedTransition(state_dim=6, action_dim=5)
    val = nn.Sequential(nn.Linear(6, 32), nn.ReLU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(list(tr.parameters()) + list(val.parameters()), lr=lr)
    Y = torch.stack([b[0] for b in buf])
    A = torch.tensor([b[1] for b in buf])
    Y1 = torch.stack([b[2] for b in buf])
    R = torch.tensor([b[3] for b in buf]).float().unsqueeze(1)
    n = len(buf)
    for e in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i + 256]
            lt = F.mse_loss(tr(Y[idx], A[idx]), Y1[idx])
            lv = F.mse_loss(val(Y[idx]), R[idx])
            loss = lt + 0.5 * lv
            opt.zero_grad(); loss.backward(); opt.step()
    return tr, val, lt.item()


@torch.no_grad()
def action_sensitivity(tr, n: int = 300, seed: int = 1, env_fn=None) -> float:
    """Mean |f(y, 进) - f(y, 守)| over random states."""
    env_fn = env_fn or (lambda s: TinyKingdom(seed=s))
    g = torch.Generator().manual_seed(seed)
    ds = []
    for _ in range(n):
        y = env_fn(int(torch.randint(0, 1 << 30, (1,), generator=g))).obs().unsqueeze(0)
        ds.append((tr(y, torch.tensor([0])) - tr(y, torch.tensor([2]))).abs().mean())
    return float(torch.stack(ds).mean())


@torch.no_grad()
def mpc_vs_random(tr, val, n_ep: int = 120, horizon: int = 3, seed: int = 2):
    m_r, r_r = [], []
    for ep in range(n_ep):
        env = TinyKingdom(seed=seed * 7919 + ep)
        y, tot = env.obs(), 0.0
        for _ in range(env.max_steps):
            best_a, best_v = 0, -1e9
            for a in range(5):
                yi = y.unsqueeze(0)
                for _ in range(horizon):
                    yi = tr(yi, torch.tensor([a]))
                v = val(yi).item()
                if v > best_v:
                    best_a, best_v = a, v
            y, rr, done, _ = env.step(best_a)
            tot += rr
            if done:
                break
        m_r.append(tot)
        env = TinyKingdom(seed=seed * 7919 + ep)
        tot = 0.0
        rng = random.Random(ep)
        for _ in range(env.max_steps):
            _, rr, done, _ = env.step(rng.randint(0, 4))
            tot += rr
            if done:
                break
        r_r.append(tot)
    return sum(m_r) / len(m_r), sum(r_r) / len(r_r)


if __name__ == "__main__":
    # env-level check: do 进 vs 守 diverge from the SAME state?
    e = TinyKingdom(seed=5); s = (e.resource, e.morale, e.threat)
    e.step(0); a = e.obs()
    e.resource, e.morale, e.threat = s; e.step(2); b = e.obs()
    print(f"env causal check |进 - 守| = {(a - b).abs().mean():.3f}")

    buf = collect(1500)
    tr, val, mse = train(buf)
    print(f"transition MSE {mse:.4f}  ({len(buf)} transitions)")
    sens = action_sensitivity(tr)
    print(f"action sensitivity |f(y,进) - f(y,守)| = {sens:.4f}  "
          f"({'learned action-dependent dynamics' if sens > 0.05 else 'action signal weak'})")
    mpc, rnd = mpc_vs_random(tr, val)
    print(f"MPC reward {mpc:.2f}  vs random {rnd:.2f}  "
          f"({(mpc - rnd) / abs(rnd) * 100:+.0f}%  -> {'MPC wins' if mpc > rnd else 'no better'})")
