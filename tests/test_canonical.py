import json

from yiwm.canonical import import_canonical
from yiwm.seed import build_yao_seed


def test_import_fills_qian_kun(tmp_path):
    seed = str(tmp_path / "yao_seed.json")
    build_yao_seed(seed)
    stats = import_canonical(seed)
    assert stats["filled"] == 12          # 乾 6 + 坤 6
    assert stats["total"] == 384

    rows = json.loads(open(seed, encoding="utf-8").read())
    by = {(r["hex_name"], r["yao_name"]): r["canonical_text"] for r in rows}
    assert by[("乾", "初九")] == "潜龙勿用"
    assert by[("乾", "上九")] == "亢龙有悔"
    assert by[("坤", "初六")] == "履霜，坚冰至"
    # untouched rows stay empty
    assert by[("屯", "初九")] == ""


def test_import_extra_override(tmp_path):
    seed = str(tmp_path / "s.json")
    build_yao_seed(seed)
    stats = import_canonical(seed, extra={"屯": {"初九": "磐桓，利居贞，利建侯"}})
    assert stats["filled"] == 13
    rows = json.loads(open(seed, encoding="utf-8").read())
    hit = [r for r in rows if r["hex_name"] == "屯" and r["yao_name"] == "初九"][0]
    assert hit["canonical_text"] == "磐桓，利居贞，利建侯"
