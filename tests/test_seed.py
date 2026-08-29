import json

from yiwm.augment import build_from_seed, mock_llm_fn
from yiwm.seed import build_yao_seed
from yiwm.synth import ACTIONS, TIMINGS


def test_build_yao_seed_384(tmp_path):
    p = str(tmp_path / "yao_seed.json")
    assert build_yao_seed(p) == 384
    rows = json.loads(open(p, encoding="utf-8").read())
    assert len(rows) == 384
    assert len({(r["hex_index"], r["yao_index"]) for r in rows}) == 384

    q = [r["yao_name"] for r in rows if r["king_wen"] == 1]
    k = [r["yao_name"] for r in rows if r["king_wen"] == 2]
    assert q == ["初九", "九二", "九三", "九四", "九五", "上九"]
    assert k == ["初六", "六二", "六三", "六四", "六五", "上六"]

    for r in rows:
        assert len(r["yao_target"]) == 6 and len(r["hex_bits"]) == 6
        assert sum(r["moving"]) == 1 and r["moving"][r["yao_index"]] == 1
        assert r["timing"] in TIMINGS and r["action"] in ACTIONS
        # yao_target sign agrees with the hexagram bit at every position
        for v, b in zip(r["yao_target"], r["hex_bits"]):
            assert (v > 0) == (b == 1)
        # the pinned 动爻 is the strongest line
        assert abs(r["yao_target"][r["yao_index"]]) >= max(abs(x) for x in r["yao_target"]) - 1e-6
        assert r["canonical_text"] == "" and r["modern_text"] == ""


def test_build_from_seed_skips_empty(tmp_path):
    seed = str(tmp_path / "seed.json")
    build_yao_seed(seed)
    rows = json.loads(open(seed, encoding="utf-8").read())
    rows[0]["modern_text"] = "一个人手上项目刚起步，预算只够三个月，团队还没搭起来。"
    rows[7]["modern_text"] = "部门负责人处在顶点，资源充足但盯着的人很多，稍有差错就被抓。"
    open(seed, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False))

    out = str(tmp_path / "sem.jsonl")
    stats = build_from_seed(seed, mock_llm_fn, out, n_variants=3)
    assert stats["seeds_filled"] == 2
    assert stats["skipped_no_text"] == 382
    assert stats["written"] == 6
    lines = [json.loads(x) for x in open(out, encoding="utf-8")]
    assert len(lines) == 6
    assert set(lines[0]) >= {"text", "ben_k", "force", "moving", "action", "timing"}
