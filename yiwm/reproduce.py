"""Multi-seed reproduction of the 4-stage world-model arc + the β ablation.

Every stage runs on Kingdom2D(size=5) so the transition action-sensitivity
numbers are directly comparable.

  stage 1   hand R^6 obs            -> LearnedTransition, sensitivity
  stage 2   CNN latent, DQN-shaped  -> frozen, TransitionDyn, sensitivity
  JEPA β=1  dynamics-first + var reg -> sensitivity, latent std
  JEPA β=0  dynamics-first, NO reg   -> ablation (does it collapse?)
  frozen+Q  β=1 encoder frozen, DQN  -> greedy return vs random (%)

Run: python -m yiwm.reproduce --seeds 5 --out reproduce.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys

import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")           # windows gbk console
except Exception:
    pass

from .kingdom_2d import Kingdom2D
from .kingdom_stage2 import train_dqn, transition_check
from .tiny_kingdom import action_sensitivity, collect, train
from .wm_pipeline import eval_policies, train_q_frozen
from .wm_pretrain import pretrain, report

SIZE = 5


def _seed(s):
    random.seed(s)
    torch.manual_seed(s)


def one_seed(s: int) -> dict:
    mk = lambda k: Kingdom2D(size=SIZE, seed=k)

    _seed(s)
    tr1, _, _ = train(collect(700, seed=s, env_fn=mk), epochs=25, seed=s)
    sens1 = action_sensitivity(tr1, n=250, seed=s, env_fn=mk)

    _seed(s)
    enc2, _ = train_dqn(size=SIZE, episodes=2500, seed=s)
    _, sens2 = transition_check(enc2, size=SIZE, n_ep=600, steps=500, seed=s)

    _seed(s)
    encB1, trB1, _ = pretrain(size=SIZE, n_ep=1000, epochs=35, beta=1.0, seed=s)
    std_b1, sens_b1 = report(encB1, trB1, size=SIZE, n=350, seed=s)

    _seed(s)
    encB0, trB0, _ = pretrain(size=SIZE, n_ep=1000, epochs=35, beta=0.0, seed=s)
    std_b0, sens_b0 = report(encB0, trB0, size=SIZE, n=350, seed=s)

    _seed(s)
    q = train_q_frozen(encB1, size=SIZE, episodes=2500, seed=s)
    res = eval_policies(encB1, trB1, q, size=SIZE, n=120, horizon=4, seed=s)
    gain = (res["greedy"] - res["random"]) / abs(res["random"]) * 100
    mpc_gap = res["mpc"] - res["greedy"]

    return {
        "seed": s,
        "stage1_sens": sens1, "stage2_sens": sens2,
        "jepa_b1_sens": sens_b1, "jepa_b1_std": std_b1,
        "jepa_b0_sens": sens_b0, "jepa_b0_std": std_b0,
        "q_gain_pct": gain, "mpc_minus_greedy": mpc_gap,
    }


def _ms(xs):
    return (st.mean(xs), st.pstdev(xs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="reproduce.json")
    a = ap.parse_args()

    rows = []
    for s in range(a.seeds):
        print(f"=== seed {s} ===", flush=True)
        r = one_seed(s)
        rows.append(r)
        print("   ", {k: round(v, 3) for k, v in r.items() if k != "seed"}, flush=True)

    agg = {k: _ms([r[k] for r in rows]) for k in rows[0] if k != "seed"}
    json.dump({"seeds": a.seeds, "rows": rows,
               "agg": {k: {"mean": m, "std": d} for k, (m, d) in agg.items()}},
              open(a.out, "w"), indent=2)

    def line(name, key, fmt="{:.3f}"):
        m, d = agg[key]
        return f"| {name} | {fmt.format(m)} ± {fmt.format(d)} |"

    print(f"\n## Reproduction — Kingdom2D 5x5, {a.seeds} seeds\n")
    print("| step | transition action-sensitivity |")
    print("|---|---|")
    print(line("stage 1 — hand R^6", "stage1_sens"))
    print(line("stage 2 — CNN + DQN latent", "stage2_sens"))
    print(line("JEPA β=1 (var reg)", "jepa_b1_sens"))
    print(line("JEPA β=0 (ablation)", "jepa_b0_sens"))
    print("\n| latent across-state std | value |")
    print("|---|---|")
    print(line("JEPA β=1", "jepa_b1_std", "{:.2f}"))
    print(line("JEPA β=0", "jepa_b0_std", "{:.2f}"))
    gm, gd = agg["q_gain_pct"]
    mm, md = agg["mpc_minus_greedy"]
    print(f"\nfrozen+Q greedy vs random:  {gm:+.0f}% +/- {gd:.0f}")
    print(f"MPC minus greedy return:    {mm:+.1f} +/- {md:.1f}  (<0 = MPC loses to greedy)")

    # verdict
    s1 = [r["stage1_sens"] for r in rows]; s2 = [r["stage2_sens"] for r in rows]
    b1 = [r["jepa_b1_sens"] for r in rows]; b0 = [r["jepa_b0_sens"] for r in rows]
    sep = (st.mean(b1) - st.mean(s2)) / (st.pstdev(b1) + st.pstdev(s2) + 1e-9)
    print(f"\nJEPA vs stage-2 separation: {sep:.1f}σ  "
          f"({'robust' if sep > 2 and min(b1) > max(s2) else 'NOT clean — story is fragile'})")
    print(f"β ablation: β1 sens {st.mean(b1):.3f} / β0 sens {st.mean(b0):.3f}  "
          f"({'reg is necessary' if st.mean(b1) > 2 * st.mean(b0) else 'reg effect weak'})")


if __name__ == "__main__":
    main()
