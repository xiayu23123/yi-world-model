"""Kingdom2D — a spatial version of TinyKingdom: an N×N grid of
(resource, threat, control) cells + a capital, 5 discrete actions with spatial
causal effects. Same env interface as TinyKingdom (`obs() -> R^6`,
`step(a) -> (obs, r, done, info)`, `max_steps`), so `q_planner` and the
`transition` checks run on it unchanged.

Stage 1 here: small grid, full observability. Stages 2–3 (3×3 vision / POMDP,
7×7, conv encoder) are follow-on.

Run: python -m yiwm.kingdom_2d
"""

from __future__ import annotations

import random

import torch

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class Kingdom2D:
    def __init__(self, size: int = 3, seed: int | None = None, max_steps: int = 15):
        self.n = size
        self.max_steps = max_steps
        self.rng = random.Random(seed)
        self.reset()

    def reset(self) -> torch.Tensor:
        n = self.n
        r = self.rng
        self.res = torch.tensor([[r.uniform(30, 70) for _ in range(n)] for _ in range(n)])
        self.threat = torch.tensor([[r.uniform(5, 30) for _ in range(n)] for _ in range(n)])
        self.ctrl = torch.zeros(n, n)
        cx, cy = n // 2, n // 2
        self.cap = (cx, cy)
        self.ctrl[cx, cy] = 1.0
        self.p_res, self.p_mil, self.p_mor = 50.0, 20.0, 55.0
        self.t = 0
        return self.obs()

    # --- observation --------------------------------------------------------
    def _controlled(self) -> torch.Tensor:
        return self.ctrl > 0.5

    def obs(self) -> torch.Tensor:
        c = self._controlled()
        n_ctrl = int(c.sum())
        edge_ctrl = self.ctrl[c].min().item() if n_ctrl else 0.0
        y = torch.tensor([
            float(self.res.sum()) / (self.n * self.n * 100),   # 初 total resource
            self.p_res / 100.0,                                 # 二 stock in hand
            torch.sigmoid(torch.tensor(self.p_mor / (self.p_mil + 1e-6) - 1.0)).item(),  # 三 balance
            1.0 - edge_ctrl,                                    # 四 边界脆弱
            float((self.res * c).sum()) / (self.n * self.n * 60),  # 五 output capacity
            float(self.threat.mean()) / 100.0,                  # 上 external pressure
        ])
        return y.clamp(0.0, 1.0)

    # --- helpers ----------------------------------------------------------
    def _adj_uncontrolled(self):
        out = []
        c = self._controlled()
        for i in range(self.n):
            for j in range(self.n):
                if c[i, j]:
                    for di, dj in _DIRS:
                        a, b = i + di, j + dj
                        if 0 <= a < self.n and 0 <= b < self.n and not c[a, b]:
                            out.append((a, b))
        return out

    def _neighbors(self, i, j):
        return [(i + di, j + dj) for di, dj in _DIRS
                if 0 <= i + di < self.n and 0 <= j + dj < self.n]

    # --- transition -------------------------------------------------------
    def step(self, action: int, intensity: float = 0.6):
        self.t += 1
        if action == 0:                                   # 进 expand
            cand = self._adj_uncontrolled()
            if cand and self.p_res > 10:
                tgt = max(cand, key=lambda p: self.res[p] - self.threat[p])
                self.ctrl[tgt] += 0.35 * intensity
                self.p_res -= 15 * intensity
                self.p_mil += 4 * intensity
                for a, b in self._neighbors(*tgt):
                    self.threat[a, b] += 8 * intensity
        elif action == 1:                                # 守 fortify
            self.ctrl[self._controlled()] = (self.ctrl[self._controlled()] + 0.15 * intensity).clamp(max=1)
            self.p_mor += 9 * intensity
            self.p_res -= 3 * intensity
        elif action == 2:                                # 退 contract
            c = self._controlled()
            if int(c.sum()) > 1:
                idx = (self.ctrl + (~c) * 9).argmin()
                i, j = int(idx // self.n), int(idx % self.n)
                self.ctrl[i, j] -= 0.6
                self.p_res += 6 * intensity
                self.p_mor -= 4 * intensity
                for a, b in self._neighbors(i, j):
                    self.threat[a, b] -= 6 * intensity
        elif action == 3:                                # 变 convert
            self.p_res -= 10 * intensity
            self.p_mil += 6 * intensity
            self.p_mor += 6 * intensity
        else:                                            # 待 wait
            self.p_res += 4 * intensity
            self.p_mor -= 2 * intensity

        # action-independent env tick
        self.threat += torch.rand(self.n, self.n) * 3
        self.res += (60 - self.res) * 0.05 + torch.randn(self.n, self.n) * 2
        c = self._controlled()
        self.p_res += float((self.res * c).sum()) * 0.04
        self.ctrl = (self.ctrl - (self.threat / 100) * 0.1 * c.float()).clamp(0, 1)
        self.res.clamp_(0, 100); self.threat.clamp_(0, 100)
        self.p_res = min(200.0, max(0.0, self.p_res))
        self.p_mil = min(150.0, max(0.0, self.p_mil))
        self.p_mor = min(100.0, max(0.0, self.p_mor))

        n_ctrl = int(self._controlled().sum())
        reward = n_ctrl * 2.0 + self.p_mor * 0.2 - float(self.threat.mean()) * 0.3
        done = self.t >= self.max_steps or self.p_res < 5 or n_ctrl == 0
        return self.obs(), reward, done, {"n_ctrl": n_ctrl, "p_res": self.p_res,
                                          "threat": float(self.threat.mean())}


if __name__ == "__main__":
    from .q_planner import eval_greedy_vs_random, train_q
    from .tiny_kingdom import action_sensitivity, collect, train

    mk = lambda seed: Kingdom2D(size=3, seed=seed)

    e = Kingdom2D(size=3, seed=5)
    st = (e.res.clone(), e.threat.clone(), e.ctrl.clone(), e.p_res, e.p_mil, e.p_mor)
    a = e.step(0)[0]
    e.res, e.threat, e.ctrl, e.p_res, e.p_mil, e.p_mor = (st[0].clone(), st[1].clone(),
                                                          st[2].clone(), st[3], st[4], st[5])
    e.t = 0
    b = e.step(1)[0]
    print(f"env causal check |进 - 守| = {(a - b).abs().mean():.3f}")

    buf = collect(1200, env_fn=mk)
    tr, _, mse = train(buf)
    print(f"transition MSE {mse:.4f}  | action sensitivity "
          f"{action_sensitivity(tr, env_fn=mk):.4f}")

    q = train_q(episodes=4000, env_fn=mk)
    g, r = eval_greedy_vs_random(q, env_fn=mk)
    print(f"greedy-Q reward {g:+.2f}  vs random {r:+.2f}  "
          f"(Δ {g - r:+.2f} -> {'Q wins' if g > r + 0.5 else 'no better'})")
