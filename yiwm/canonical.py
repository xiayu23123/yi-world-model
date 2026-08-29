"""Merge public-domain 爻辞 into `data/yao_seed.json`'s `canonical_text` field.

The Zhouyi line statements are public domain, but transcribing all 372 from
memory is error-prone, so only 乾 / 坤 (well-known, high confidence) ship here.
Paste the rest from a 周易 source (e.g. Wikisource) into `CANONICAL`, or pass
your own dict as `extra=`.

Keys: hexagram name as in `constants.KING_WEN_NAMES` (乾, 坤, 屯, ...); yao name
as emitted by `seed._yao_name` (初九, 九二, ..., 上九 / 初六, ..., 上六).

Run: python -m yiwm.canonical data/yao_seed.json
"""

from __future__ import annotations

import json
from pathlib import Path

CANONICAL: dict[str, dict[str, str]] = {
    "乾": {
        "初九": "潜龙勿用",
        "九二": "见龙在田，利见大人",
        "九三": "君子终日乾乾，夕惕若厉，无咎",
        "九四": "或跃在渊，无咎",
        "九五": "飞龙在天，利见大人",
        "上九": "亢龙有悔",
    },
    "坤": {
        "初六": "履霜，坚冰至",
        "六二": "直方大，不习无不利",
        "六三": "含章可贞，或从王事，无成有终",
        "六四": "括囊，无咎无誉",
        "六五": "黄裳，元吉",
        "上六": "龙战于野，其血玄黄",
    },
}


def import_canonical(seed_path: str, extra: dict | None = None) -> dict:
    """Fill `canonical_text` in-place from CANONICAL (+ `extra` override)."""
    table: dict[str, dict[str, str]] = {k: dict(v) for k, v in CANONICAL.items()}
    for h, m in (extra or {}).items():
        table.setdefault(h, {}).update(m)

    seed = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    filled = 0
    for row in seed:
        t = table.get(row["hex_name"], {}).get(row["yao_name"])
        if t:
            row["canonical_text"] = t
            filled += 1
    Path(seed_path).write_text(json.dumps(seed, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"filled": filled, "total": len(seed),
            "hexagrams_covered": sum(1 for h in table if any(table[h].values()))}


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/yao_seed.json"
    print(import_canonical(path))
