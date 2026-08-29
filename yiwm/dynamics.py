"""System-dynamics portrait: run `rollout` + perturbation over all 384 (卦, 爻)
standard states and summarise.

  rollout   -> where does the system go?  (attractor / cycle / decay rate)
  perturb   -> how stable is it at the boundary?  (argmax flip rate vs sigma)

Run: python -m yiwm.dynamics --ckpt checkpoints/yiwm_synth.pt
Writes rollout_stats.json + perturbation.csv next to the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from .constants import BINARY_TO_KING_WEN, KING_WEN_NAMES
from .data import get_dataset
from .model import YiWorldModel
from .seed import _yao_target_for
from .synth import _entities


def _kw(k: int) -> str:
    return KING_WEN_NAMES[BINARY_TO_KING_WEN[k].item()]


def _load(ckpt: str):
    blob = torch.load(ckpt)
    if isinstance(blob, dict) and "state_dict" in blob:
        state, data, te = blob["state_dict"], blob.get("data", "eco"), blob.get("text_encoder", "hash")
    else:
        state, data, te = blob, "eco", "hash"
    _, obs_dim = get_dataset(data, te)
    m = YiWorldModel(obs_dim=obs_dim)
    m.load_state_dict(state)
    m.eval()
    return m, obs_dim


@torch.no_grad()
def analyse(ckpt: str, steps: int = 20, sigmas=(0.05, 0.10, 0.20), trials: int = 12):
    model, obs_dim = _load(ckpt)
    dummy = torch.zeros(1, obs_dim)

    per_hex: dict[int, list] = {k: [] for k in range(64)}
    perturb_rows = []                        # (hex_kw, yao_name, sigma, flip_ben, flip_zhi, flip_act)
    yao_names = ["初", "二", "三", "四", "五", "上"]

    for k in range(64):
        for pos in range(6):
            force, _ = _yao_target_for(k, pos)
            y = torch.tensor(force).view(1, 6)

            # --- rollout from this standard state -----------------------------
            traj = model.rollout(y[0], steps=steps)
            mags = [s["mag"] for s in traj]
            n = len(traj)
            decay = (mags[-1] / mags[0]) ** (1 / max(n - 1, 1)) if mags[0] > 0 else 1.0
            per_hex[k].append({
                "from_yao": pos,
                "n_steps": n,
                "stop": traj[-1].get("stop", "maxsteps"),
                "terminal_hex": traj[-1]["hex_next_k"],
                "decay": round(float(decay), 3),
                "cycle_members": [_kw(x) for x in traj[-1].get("cycle_members", [])],
            })

            # --- perturbation at this state ---------------------------------
            cats, st, adj = _entities(torch.tensor([k]), y)
            base = model(dummy, st, cats, adj, hard=True, yao_override=y)
            b_ben = base["hex_logits"].argmax(1)
            b_zhi = base["hex_logits_next_joint"].argmax(1)
            b_act = base["policy"]["action_logits"].argmax(1)
            for sig in sigmas:
                fb = fz = fa = 0
                for _ in range(trials):
                    o = model(dummy, st, cats, adj, hard=True,
                              yao_override=y + sig * torch.randn(1, 6))
                    fb += int(o["hex_logits"].argmax(1) != b_ben)
                    fz += int(o["hex_logits_next_joint"].argmax(1) != b_zhi)
                    fa += int(o["policy"]["action_logits"].argmax(1) != b_act)
                perturb_rows.append((_kw(k), yao_names[pos], sig,
                                     fb / trials, fz / trials, fa / trials))

    # --- aggregate rollout per hexagram --------------------------------------
    stats = {}
    for k in range(64):
        rs = per_hex[k]
        terms = Counter(_kw(r["terminal_hex"]) for r in rs)
        cyc = Counter(tuple(r["cycle_members"]) for r in rs if r["cycle_members"])
        top_cyc = cyc.most_common(1)[0][0] if cyc else []
        stats[_kw(k)] = {
            "mean_steps": round(sum(r["n_steps"] for r in rs) / 6, 2),
            "cycle_frac": round(sum(r["stop"].startswith("cycle") for r in rs) / 6, 2),
            "fixed_frac": round(sum(r["stop"] == "fixed" for r in rs) / 6, 2),
            "mean_decay": round(sum(r["decay"] for r in rs) / 6, 3),
            "terminal_top": terms.most_common(1)[0][0],
            "self_attractor": terms.most_common(1)[0][0] == _kw(k),
            "cycle_len": len(top_cyc),
            "cycle_members": list(top_cyc),
        }

    out = Path(ckpt).with_name("rollout_stats.json")
    out.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    csv = Path(ckpt).with_name("perturbation.csv")
    with open(csv, "w", encoding="utf-8") as f:
        f.write("hex,yao,sigma,flip_benGua,flip_zhiGua,flip_action\n")
        for r in perturb_rows:
            f.write(",".join(str(x) for x in r) + "\n")

    # --- console summary ---------------------------------------------------
    attractors = [h for h, s in stats.items() if s["self_attractor"]]
    fast = sorted(stats.items(), key=lambda kv: kv[1]["mean_steps"])[:5]
    slow = sorted(stats.items(), key=lambda kv: -kv[1]["mean_steps"])[:5]
    print(f"[{ckpt}]  wrote {out.name}, {csv.name}\n")
    print(f"self-attractors ({len(attractors)}/64): {' '.join(attractors)}")
    print("fastest converge:", ", ".join(f"{h}({s['mean_steps']})" for h, s in fast))
    print("slowest / most cyclic:", ", ".join(f"{h}({s['mean_steps']},cyc{s['cycle_frac']})" for h, s in slow))

    print("\nperturbation flip rate by 爻 position (mean over 64 卦):")
    print("  pos |    sigma 0.05     |    sigma 0.20")
    print("      | ben  zhi  act    | ben  zhi  act")
    for pi, nm in enumerate(yao_names):
        def avg(sig, idx):
            v = [r[3 + idx] for r in perturb_rows if r[1] == nm and r[2] == sig]
            return sum(v) / len(v)
        print(f"  {nm}爻 | {avg(0.05,0):.2f} {avg(0.05,1):.2f} {avg(0.05,2):.2f} | "
              f"{avg(0.20,0):.2f} {avg(0.20,1):.2f} {avg(0.20,2):.2f}")
    return stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/yiwm_synth.pt")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--trials", type=int, default=12)
    a = ap.parse_args()
    analyse(a.ckpt, a.steps, trials=a.trials)
