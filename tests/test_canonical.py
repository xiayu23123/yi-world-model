import json

from yiwm.canonical import import_canonical
from yiwm.seed import build_yao_seed

_JARGON = ["卦", "爻", "阴", "阳", "初九", "六二", "潜龙", "履霜"]


def test_import_fills_qian_kun(tmp_path):
    seed = str(tmp_path / "yao_seed.json")
    build_yao_seed(seed)
    stats = import_canonical(seed)
    assert stats["canonical_filled"] == 12     # 乾 6 + 坤 6
    assert stats["modern_filled"] == 36        # P0 (12) + P1 (24)
    assert stats["p0_relabelled"] >= 12        # toy _action disagrees on many anchors
    assert stats["total"] == 384

    rows = json.loads(open(seed, encoding="utf-8").read())
    by = {(r["hex_name"], r["yao_name"]): r for r in rows}
    assert by[("乾", "初九")]["canonical_text"] == "潜龙勿用"
    assert by[("坤", "上六")]["canonical_text"] == "龙战于野，其血玄黄"
    assert by[("坤", "六二")]["action"] == "jin"          # P0 override applied
    # untouched rows stay empty
    assert by[("屯", "初九")]["canonical_text"] == ""
    assert by[("屯", "初九")]["modern_text"] == ""


def test_anchor_modern_text_is_jargon_free_and_sized():
    mod = __import__("yiwm.canonical", fromlist=["MODERN_P0", "MODERN_P1"])
    for table in (mod.MODERN_P0, mod.MODERN_P1):
        for hexn, ys in table.items():
            for yname, txt in ys.items():
                for bad in _JARGON:
                    assert bad not in txt, f"{hexn}{yname}: leaked {bad!r}"
                assert 60 <= len(txt) <= 200, f"{hexn}{yname}: len {len(txt)}"


def test_p1_contrast_pairs_present():
    from yiwm.canonical import MODERN_P1
    assert set(MODERN_P1) == {"既濟", "未濟", "泰", "否"}
    assert all(len(v) == 6 for v in MODERN_P1.values())


def test_import_extra_overrides(tmp_path):
    seed = str(tmp_path / "s.json")
    build_yao_seed(seed)
    base = import_canonical(seed)
    stats = import_canonical(
        seed,
        extra={"屯": {"初九": "磐桓，利居贞，利建侯"}},
        modern_extra={"屯": {"初九": "一个人接手了一个刚立项的团队，方向没定，先稳住阵脚，把手里的事排出优先级，再谈别的。"}},
    )
    assert stats["canonical_filled"] == base["canonical_filled"] + 1
    assert stats["modern_filled"] == base["modern_filled"] + 1
    rows = json.loads(open(seed, encoding="utf-8").read())
    hit = [r for r in rows if r["hex_name"] == "屯" and r["yao_name"] == "初九"][0]
    assert hit["canonical_text"] == "磐桓，利居贞，利建侯"
    assert hit["verified"] is True
