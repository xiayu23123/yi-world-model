"""Where does the 之卦 error live?  Run: python -m yiwm.analysis

Splits the ~0.77 之卦 accuracy gap into:
  * high-confidence wrong  -> model is sure and wrong  -> structural bug
  * low-confidence  wrong  -> model itself unsure      -> needs more features
and decomposes 之卦 errors into upstream (本卦 wrong) vs pure ChangeEngine
fault (本卦 right, 之卦 still wrong), and within the latter, moving-mask wrong
vs moving-mask right (agree-term / mix bug).
"""

import argparse
import math

import torch

from .data import get_dataset
from .model import YiWorldModel

LOG64 = math.log(64.0)


def _entropy(logits: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1) / LOG64          # normalised to [0, 1]


def _report_head(tag, logits, tgt, conf_thresh):
    p = torch.softmax(logits, dim=-1)
    conf, pred = p.max(-1)
    correct = pred == tgt
    wrong = ~correct
    Hn = _entropy(logits)
    n_wrong = int(wrong.sum())

    print(f"\n=== {tag}  acc={correct.float().mean():.3f}  n={len(tgt)}  n_wrong={n_wrong} ===")
    print(f"  norm-entropy : correct {Hn[correct].mean():.3f} | wrong {Hn[wrong].mean():.3f}")
    print(f"  confidence   : correct {conf[correct].mean():.3f} | wrong {conf[wrong].mean():.3f}")
    if n_wrong:
        cw = int((wrong & (conf > conf_thresh)).sum())
        fw = n_wrong - cw
        print(f"  wrong split @conf>{conf_thresh}: "
              f"confident-wrong {cw / n_wrong:.1%} ({cw})  |  fuzzy-wrong {fw / n_wrong:.1%} ({fw})")
        verdict = "STRUCTURAL BUG (sure & wrong)" if cw >= fw else "NEEDS FEATURES (model unsure)"
        print(f"  -> {verdict}")

    print("  calibration (conf bin -> acc / count):")
    edges = torch.linspace(0, 1, 6)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & ((conf < hi) | (hi == 1.0))
        if m.any():
            print(f"    [{lo:.1f},{hi:.1f})  acc {correct[m].float().mean():.3f}  n {int(m.sum())}")


def analyze(ckpt: str = "checkpoints/yiwm.pt", n: int = 8192, conf_thresh: float = 0.7,
            seed: int = 2024, data: str | None = None):
    try:
        blob = torch.load(ckpt)
    except FileNotFoundError:
        raise SystemExit(f"no checkpoint at {ckpt!r} -- run `python -m yiwm.train` first")
    text_encoder = "hash"
    if isinstance(blob, dict) and "state_dict" in blob:
        state, data = blob["state_dict"], data or blob.get("data", "eco")
        text_encoder = blob.get("text_encoder", "hash")
    else:
        state, data = blob, data or "eco"
    make, obs_dim = get_dataset(data, text_encoder)
    model = YiWorldModel(obs_dim=obs_dim)
    model.load_state_dict(state)
    model.eval()
    print(f"[data={data}  text_encoder={text_encoder}]")

    b = make(n, seed=seed)
    with torch.no_grad():
        o = model(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"], hard=True)

    _report_head("benGua", o["hex_logits"], b["hex"], conf_thresh)
    _report_head("zhiGua", o["hex_logits_next"], b["hex_next"], conf_thresh)

    # --- zhiGua error decomposition --------------------------------------------
    ben_ok = o["hex_logits"].argmax(-1) == b["hex"]
    zhi_ok = o["hex_logits_next"].argmax(-1) == b["hex_next"]
    mv_pred = (o["change"] > 0.5).long()
    mv_ok = (mv_pred == b["moving"]).all(-1)
    per_yao = (mv_pred == b["moving"]).float().mean().item()
    zhi_bad = ~zhi_ok
    nb = int(zhi_bad.sum())
    print(f"\n=== zhiGua error decomposition  (n_wrong={nb}) ===")
    print(f"  moving-mask acc: per-yao {per_yao:.3f}  ->  all-6-correct {mv_ok.float().mean():.3f} "
          f"(= {per_yao:.3f}^6 = {per_yao ** 6:.3f}; compounding is the ceiling)")
    if nb:
        up = int((zhi_bad & ~ben_ok).sum())
        pure = zhi_bad & ben_ok
        npure = int(pure.sum())
        print(f"  upstream (benGua already wrong)   : {up / nb:.1%} ({up})")
        print(f"  pure ChangeEngine (benGua right)  : {npure / nb:.1%} ({npure})")
        if npure:
            mv_bad = int((pure & ~mv_ok).sum())
            mv_good = npure - mv_bad
            print(f"    - moving-mask also wrong        : {mv_bad / npure:.1%} ({mv_bad})  "
                  f"<- ChangeEngine threshold model")
            print(f"    - moving-mask right, zhiGua wrong: {mv_good / npure:.1%} ({mv_good})  "
                  f"<- agree-term / mix / equidistant tie")

    # --- learned relation gates ------------------------------------------------
    g = model.hexinf.relation_weights()
    print("\n=== HexagramInference relation gates ===")
    for layer in range(g.size(0)):
        print(f"  layer {layer}: " +
              "  ".join(f"{r}={g[layer, i]:.3f}"
                        for i, r in enumerate(model.hexinf.RELATIONS)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/yiwm.pt")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--conf-thresh", type=float, default=0.7)
    ap.add_argument("--data", choices=["eco", "synth"], default=None)
    a = ap.parse_args()
    analyze(a.ckpt, a.n, a.conf_thresh, data=a.data)
