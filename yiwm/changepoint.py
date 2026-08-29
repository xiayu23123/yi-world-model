"""Change-point detection benchmark — the falsifiable test.

Does the 变卦 mechanism (本卦 pattern shift) flag regime changes better than
standard change-point methods, OR is it just multivariate feature-change
detection dressed up?

The `FeatureCUSUM` baseline is the control: it sees the SAME 6 features the Yi
detector does. If it ties/beats Yi, the 卦 framing adds nothing.

Fixed, arbitrary feature set (documented, not claimed optimal):
  trend slope / std / skew / kurtosis / lag-1 acf / coef-of-variation → yao 0..5

Run: python -m yiwm.changepoint --seeds 10
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from scipy import stats


# --------------------------------------------------------------------------- #
# window -> R^6
# --------------------------------------------------------------------------- #
def window_to_yao(w: np.ndarray) -> np.ndarray:
    n = len(w)
    tr = np.polyfit(np.arange(n), w, 1)[0]
    sd = w.std()
    m = w.mean()
    acf = np.corrcoef(w[:-1], w[1:])[0, 1] if n > 2 and w[:-1].std() > 0 else 0.0
    return np.tanh(np.array([
        tr * 10, sd * 3, stats.skew(w), stats.kurtosis(w) / 5,
        acf * 2, sd / (abs(m) + 1e-6) * 2,
    ]))


# --------------------------------------------------------------------------- #
# detectors -> list of change-point indices
# --------------------------------------------------------------------------- #
class YiDetector:
    """本卦 pattern shift = 变卦 event. Reported at window END (no centre lag)."""

    def __init__(self, model, window: int = 50, step: int = 5):
        self.m, self.w, self.s = model, window, step

    @torch.no_grad()
    def __call__(self, series: np.ndarray) -> list[int]:
        self.m.eval()
        cps, prev = [], None
        for i in range(0, len(series) - self.w + 1, self.s):
            y = torch.tensor(window_to_yao(series[i:i + self.w]), dtype=torch.float32)
            hexid = int(self.m.hexinf(y.unsqueeze(0)).argmax(1))
            if prev is not None and hexid != prev:
                cps.append(i + self.w)                     # end of the window
            prev = hexid
        return _dedup(cps, self.w)


def _feat_stream(series, w, s):
    idx, F = [], []
    for i in range(0, len(series) - w + 1, s):
        F.append(window_to_yao(series[i:i + w]))
        idx.append(i + w)
    return np.array(idx), np.array(F)


def feature_cusum(series, w=50, s=5, thresh=4.0) -> list[int]:
    """multivariate CUSUM on the SAME 6 features (the control)."""
    idx, F = _feat_stream(series, w, s)
    if len(F) < 5:
        return []
    mu, sd = F[:5].mean(0), F[:5].std(0) + 1e-6
    g = np.zeros(F.shape[1])
    cps = []
    for k in range(len(F)):
        z = (F[k] - mu) / sd
        g = np.maximum(0, g + np.abs(z) - 0.5)
        if g.max() > thresh:
            cps.append(int(idx[k]))
            g[:] = 0
            j = max(0, k - 4)
            mu, sd = F[j:k + 1].mean(0), F[j:k + 1].std(0) + 1e-6
    return _dedup(cps, w)


def cusum_raw(series, k=0.5, h=5.0) -> list[int]:
    mu, sd = series[:100].mean(), series[:100].std() + 1e-9
    gp = gn = 0.0
    cps = []
    for i in range(1, len(series)):
        z = (series[i] - mu) / sd
        gp = max(0, gp + z - k)
        gn = max(0, gn - z - k)
        if gp > h or gn > h:
            cps.append(i)
            gp = gn = 0.0
    return _dedup(cps, 20)


def _seg_cost(cs, cs2, a, b):        # L2 cost of series[a:b] from prefix sums
    n = b - a
    return (cs2[b] - cs2[a]) - (cs[b] - cs[a]) ** 2 / n


def pelt_l2(series, pen: float) -> list[int]:
    y = series.astype(float)
    cs = np.concatenate([[0], np.cumsum(y)])
    cs2 = np.concatenate([[0], np.cumsum(y * y)])
    n = len(y)
    F = np.full(n + 1, np.inf)
    F[0] = -pen
    back = [[] for _ in range(n + 1)]
    R = [0]
    for t in range(1, n + 1):
        best, arg = np.inf, 0
        for s in R:
            c = F[s] + _seg_cost(cs, cs2, s, t) + pen
            if c < best:
                best, arg = c, s
        F[t] = best
        back[t] = back[arg] + [arg]
        R = [s for s in R if F[s] + _seg_cost(cs, cs2, s, t) <= F[t]] + [t]
    return [b for b in back[n] if 0 < b < n]


def binseg_l2(series, n_bkps: int) -> list[int]:
    y = series.astype(float)
    cs = np.concatenate([[0], np.cumsum(y)])
    cs2 = np.concatenate([[0], np.cumsum(y * y)])
    segs = [(0, len(y))]
    bkps = []
    for _ in range(n_bkps):
        best_gain, best = -np.inf, None
        for (a, b) in segs:
            if b - a < 4:
                continue
            base = _seg_cost(cs, cs2, a, b)
            for m in range(a + 2, b - 1):
                g = base - _seg_cost(cs, cs2, a, m) - _seg_cost(cs, cs2, m, b)
                if g > best_gain:
                    best_gain, best = g, (a, m, b)
        if best is None:
            break
        a, m, b = best
        segs.remove((a, b)); segs += [(a, m), (m, b)]; bkps.append(m)
    return sorted(bkps)


def _dedup(cps, gap):
    out = []
    for c in sorted(cps):
        if not out or c - out[-1] >= gap:
            out.append(c)
    return out


# --------------------------------------------------------------------------- #
# data + eval
# --------------------------------------------------------------------------- #
def synth(n=1000, n_cp=5, seed=0):
    rng = np.random.default_rng(seed)
    cps = sorted(rng.choice(np.arange(120, n - 120), n_cp, replace=False).tolist())
    out, prev = [], 0
    for c in cps + [n]:
        seg = rng.normal(rng.uniform(-2, 2), rng.uniform(0.4, 2.2), c - prev)
        out.append(seg); prev = c
    return np.concatenate(out), cps


def score(pred, true, margin=15, n=1000):
    tp = sum(any(abs(p - t) <= margin for p in pred) for t in true)
    fp = sum(all(abs(p - t) > margin for t in true) for p in pred)
    fn = len(true) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    delays = []
    for t in true:
        hit = [p - t for p in pred if abs(p - t) <= margin]
        if hit:
            delays.append(min(hit, key=abs))
    return {"f1": f1, "fp": fp, "delay": float(np.mean(delays)) if delays else np.nan}


def _best_pelt(series, true):
    best = None
    for pen in (3, 5, 8, 12, 20, 40, 80):
        r = score(pelt_l2(series, pen), true, n=len(series))
        if best is None or r["f1"] > best[1]["f1"]:
            best = (pen, r)
    return best[1]


def benchmark(seeds=10, ckpt="checkpoints/yiwm_synth.pt"):
    from .data import get_dataset
    from .model import YiWorldModel

    blob = torch.load(ckpt)
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    _, od = get_dataset(blob.get("data", "synth") if isinstance(blob, dict) else "synth")
    model = YiWorldModel(obs_dim=od); model.load_state_dict(sd)

    acc = {k: [] for k in ("yi", "feat_cusum", "pelt*", "binseg(oracle)", "cusum_raw")}
    for s in range(seeds):
        series, true = synth(seed=s)
        acc["yi"].append(score(YiDetector(model)(series), true, n=len(series)))
        acc["feat_cusum"].append(score(feature_cusum(series), true, n=len(series)))
        acc["pelt*"].append(_best_pelt(series, true))
        acc["binseg(oracle)"].append(score(binseg_l2(series, len(true)), true, n=len(series)))
        acc["cusum_raw"].append(score(cusum_raw(series), true, n=len(series)))

    print(f"\n## Change-point benchmark — synthetic mean+var shifts, {seeds} seeds\n")
    print("| method | F1 | signed delay | false pos |")
    print("|---|---|---|---|")
    for k, rs in acc.items():
        f1 = np.array([r["f1"] for r in rs])
        dl = np.array([r["delay"] for r in rs]); dl = dl[~np.isnan(dl)]
        fp = np.array([r["fp"] for r in rs])
        print(f"| {k:16s} | {f1.mean():.3f} ± {f1.std():.3f} "
              f"| {dl.mean():+.0f} ± {dl.std():.0f} | {fp.mean():.1f} |")

    yi = np.array([r["f1"] for r in acc["yi"]])
    fc = np.array([r["f1"] for r in acc["feat_cusum"]])
    pl = np.array([r["f1"] for r in acc["pelt*"]])
    print(f"\nyi vs feature-CUSUM (same features):  {yi.mean() - fc.mean():+.3f} F1  "
          f"({'yi adds nothing over the features' if yi.mean() <= fc.mean() + 0.02 else 'yi beats the plain feature control'})")
    print(f"yi vs tuned PELT:                     {yi.mean() - pl.mean():+.3f} F1  "
          f"({'loses to PELT' if yi.mean() < pl.mean() - 0.02 else 'ties/beats PELT'})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--ckpt", default="checkpoints/yiwm_synth.pt")
    a = ap.parse_args()
    benchmark(a.seeds, a.ckpt)
