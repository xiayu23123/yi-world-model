"""Pretty-print one inference. Run: python -m yiwm.demo  [--data eco|synth]"""

import argparse
import math

import torch

from .constants import BINARY_TO_KING_WEN, KING_WEN_NAMES, WUXING_CN, WUXING_CONTROLLER, render_hexagram
from .data import get_dataset
from .model import YiWorldModel
from .policy import TemporalPositionalPolicy


def _load(ckpt, fallback_data):
    try:
        blob = torch.load(ckpt)
    except FileNotFoundError:
        return None, fallback_data, "hash"
    if isinstance(blob, dict) and "state_dict" in blob:
        return blob["state_dict"], blob.get("data", fallback_data), blob.get("text_encoder", "hash")
    return blob, fallback_data, "hash"


def run(ckpt: str = "checkpoints/yiwm.pt", seed: int = 7, data: str | None = None):
    state, data, text_encoder = _load(ckpt, data or "eco")
    make, obs_dim = get_dataset(data, text_encoder)
    model = YiWorldModel(obs_dim=obs_dim)
    if state is None:
        print("[no checkpoint -- random weights, run `python -m yiwm.train` first]\n")
    else:
        model.load_state_dict(state)
        print(f"[loaded {ckpt}  data={data}]\n")
    model.eval()

    b = make(1, seed=seed)
    with torch.no_grad():
        o = model(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"], hard=True)

    k = o["hex_logits"].argmax(1).item()
    kn = o["hex_logits_next"].argmax(1).item()
    kw = BINARY_TO_KING_WEN[k].item() + 1
    kwn = BINARY_TO_KING_WEN[kn].item() + 1
    acts = TemporalPositionalPolicy.ACTIONS_CN

    if "populations" in b:
        pops = b["populations"][0]
        dom = pops.argmax().item()
        print("五行 populations:", {WUXING_CN[i]: round(pops[i].item(), 2) for i in range(5)})
        print(f"当令: {WUXING_CN[dom]}  ->  当克之: {WUXING_CN[WUXING_CONTROLLER[dom].item()]}\n")
    if "text" in b:
        print("情境:", b["text"][0], "\n")

    print(f"本卦  {kw:2d} {KING_WEN_NAMES[kw - 1]}")
    print(render_hexagram(k))
    print("\n动爻 (下->上, 1=变):", o["change"][0].int().tolist())
    print(f"\n之卦  {kwn:2d} {KING_WEN_NAMES[kwn - 1]}")
    print(render_hexagram(kn))

    def norm_entropy(logits1d):
        logp = torch.log_softmax(logits1d, dim=-1)
        return (-(logp.exp() * logp).sum() / math.log(64.0)).item()

    hz = norm_entropy(o["hex_logits_next"][0])
    zconf = o["hex_logits_next"][0].softmax(0).max().item()
    print(f"\nbenGua_entropy (norm): {norm_entropy(o['hex_logits'][0]):.3f}")
    print(f"zhiGua_entropy (norm): {hz:.3f}   top-prob {zconf:.3f}   (0=确定, 1=均匀 64 卦)")

    a = o["policy"]["action_logits"][0].softmax(0)
    print("\n行动分布:", {acts[i]: round(a[i].item(), 2) for i in range(5)})
    print("强度:", round(o["policy"]["intensity"][0].item(), 2))

    tk = BINARY_TO_KING_WEN[b["hex"][0]].item() + 1
    line = f"\n[真值] 本卦 {tk} {KING_WEN_NAMES[tk - 1]} | 行动 {acts[b['action'][0].item()]}"
    if "timing" in b:
        from .synth import TIMINGS
        line += f" | 时序 {TIMINGS[b['timing'][0].item()]}"
    print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/yiwm.pt")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--data", choices=["eco", "synth"], default=None)
    a = ap.parse_args()
    run(a.ckpt, a.seed, a.data)
