"""Learn `f: yao_t -> yao_{t+1}` on real market data and check whether it found
real structure or just re-derived the discrete flip rule.

Action is intentionally dropped: a passive price series has no causal action
effect (the next state does not depend on whether *you* bought), so a transition
trained on it cannot learn action influence — that needs portfolio data. What it
*can* answer: is next-yao predictable beyond persistence, is the learned map
stable under iteration, and how far is it from `ChangeEngine`'s flip.

Run: python -m yiwm.market_transition --ticker AAPL
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .market_adapter import load_market, market_to_yao


class MarketTransition(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 6), nn.Tanh(),
        )

    def forward(self, yao_t):
        return self.net(yao_t)


def _splits(y, frac=0.7):
    n = int(len(y) * frac)
    return (torch.tensor(y[:n - 1], dtype=torch.float32),
            torch.tensor(y[1:n], dtype=torch.float32),
            torch.tensor(y[n:-1], dtype=torch.float32),
            torch.tensor(y[n + 1:], dtype=torch.float32))


@torch.no_grad()
def _changeengine_next(yao: torch.Tensor, decay: float = 0.9) -> torch.Tensor:
    """The yiwm flip rule on a raw yao vector, for the comparison baseline."""
    from .constants import MOVING_MASKS
    from .model import YiWorldModel

    m = YiWorldModel(obs_dim=6)
    m.eval()
    hl = m.hexinf(yao)
    hf = torch.softmax(hl, -1) @ m.hexinf.hex_features()
    mask = MOVING_MASKS[m.moving_head(torch.cat([yao, hf], -1)).argmax(-1)]
    return (yao * (1 - 2 * mask) * decay).clamp(-1, 1)


def run(ticker: str = "AAPL", epochs: int = 60, lr: float = 1e-3, seed: int = 0):
    torch.manual_seed(seed)
    df = load_market(ticker)
    y = market_to_yao(df)
    xtr, ttr, xte, tte = _splits(y)
    print(f"{ticker}: {len(y)} rows  ({df.index[0].date()} -> {df.index[-1].date()})  "
          f"train {len(xtr)} / test {len(xte)}")

    model = MarketTransition()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for e in range(epochs):
        model.train()
        for i in range(0, len(xtr), 64):
            xb, tb = xtr[i:i + 64], ttr[i:i + 64]
            loss = F.mse_loss(model(xb), tb)
            opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(xte)
    mse = F.mse_loss(pred, tte).item()
    persist = F.mse_loss(xte, tte).item()                      # yao_{t+1} = yao_t
    # direction hit rate on the two momentum dims
    dhit = ((pred[:, :2].sign() == tte[:, :2].sign()).float().mean().item())
    dbase = ((xte[:, :2].sign() == tte[:, :2].sign()).float().mean().item())
    # distance from the ChangeEngine flip rule on the same states
    ce = torch.stack([_changeengine_next(x.unsqueeze(0))[0] for x in xte[:400]])
    ce_gap = F.mse_loss(pred[:400], ce).item()

    # bootstrap: iterate the learned map from a real state
    s = xte[0:1].clone()
    traj = [s[0].clone()]
    for _ in range(30):
        s = model(s)
        traj.append(s[0].clone())
    traj = torch.stack(traj)
    drift = (traj[-1] - traj[-5]).abs().mean().item()

    print(f"\n  1-step MSE        {mse:.4f}   (persistence {persist:.4f}  -> "
          f"{'BEATS' if mse < persist else 'no better than'} persistence)")
    print(f"  momentum dir hit  {dhit:.3f}   (persistence {dbase:.3f})")
    print(f"  gap vs ChangeEngine flip   {ce_gap:.4f}   "
          f"({'learned its own map' if ce_gap > 0.2 else 'close to the flip rule'})")
    print(f"  bootstrap drift (last 5 steps of 30)   {drift:.4f}   "
          f"({'converges' if drift < 0.02 else 'still moving'})")
    return {"mse": mse, "persist": persist, "dir_hit": dhit, "ce_gap": ce_gap, "drift": drift}


def regime(ticker: str = "AAPL", last: int = 15):
    """Print the recent market-state -> 卦 sequence (deterministic, no forecast)."""
    from .market_adapter import load_market, market_regime

    r = market_regime(load_market(ticker))
    print(f"{ticker} — recent market regime (structural label, not a prediction):\n")
    for d, row in r.tail(last).iterrows():
        print(f"  {d.date()}  {row['king_wen']:2d} {row['hex']:<4s} {row['yang_dims']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--regime", action="store_true", help="print recent market->卦 labels, no training")
    a = ap.parse_args()
    if a.regime:
        regime(a.ticker)
    else:
        run(a.ticker, a.epochs)
