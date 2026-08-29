"""Build `data/yao_seed.json` -- the 384-line (64 hexagram x 6 yao) skeleton.

Every *structural* field is derived deterministically from `constants.py` +
`synth.py` (same rules as the `synth` generator, with the moving line pinned to
this yao). The two text fields are left empty for you to fill:

  canonical_text : the classical 爻辞 (from a 周易 source -- not fabricated here)
  modern_text    : a jargon-free modern situation (hand-written, or via
                   `augment.build_from_seed` once canonical_text is in)

Run: python -m yiwm.seed [out_path]
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .constants import BINARY_HEX, BINARY_TO_KING_WEN, KING_WEN_NAMES
from .synth import ACTIONS, TIMINGS, _action, _structure

_POS_MID = ["二", "三", "四", "五"]


def _yao_name(pos: int, is_yang: bool) -> str:
    yy = "九" if is_yang else "六"
    if pos == 0:
        return "初" + yy
    if pos == 5:
        return "上" + yy
    return yy + _POS_MID[pos - 1]


def _yao_target_for(k: int, pos: int):
    """Structural base strength for hexagram k, with line `pos` pinned 老
    (magnitude ~0.95, sign kept). Mirrors synth._force + a single fixed 动爻."""
    bits, dw, dz, yy = _structure(torch.tensor([k]))
    mag = (0.40 + 0.12 * dw[0] + 0.08 * dz[0] + 0.08 * yy[0]).clamp(0.12, 0.74).clone()
    mag[pos] = 0.95
    force = mag * (bits[0] * 2 - 1)                              # [6] signed
    mask = torch.zeros(1, 6)
    mask[0, pos] = 1.0
    action = int(_action(torch.tensor([pos]), dw, yy, mask)[0])
    return force.tolist(), action


def build_yao_seed(out_path: str = "data/yao_seed.json") -> int:
    rows = []
    for k in range(64):
        bits = [int(x) for x in BINARY_HEX[k].tolist()]
        kw = int(BINARY_TO_KING_WEN[k])
        for pos in range(6):
            force, action = _yao_target_for(k, pos)
            rows.append({
                "hex_index": k,
                "king_wen": kw + 1,
                "hex_name": KING_WEN_NAMES[kw],
                "yao_index": pos,
                "yao_name": _yao_name(pos, bool(bits[pos])),
                "hex_bits": bits,
                "yao_target": [round(x, 3) for x in force],
                "moving": [1 if i == pos else 0 for i in range(6)],
                "timing": TIMINGS[pos],
                "action": ACTIONS[action],
                "canonical_text": "",
                "modern_text": "",
                "domains": [],
                "verified": False,
            })
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "data/yao_seed.json"
    n = build_yao_seed(out)
    print(f"wrote {n} rows -> {out}")
