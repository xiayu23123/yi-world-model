import json

import torch

from yiwm.augment import (
    build_semantic_jsonl, consistency_filter, generate_situations,
    make_structure_rows, mock_llm_fn, paraphrase_fallback, structure_to_prompt,
)
from yiwm.model import YiWorldModel

_FORBIDDEN = ["乾", "坤", "初九", "上六", "阴气", "阳气", "当位", "五行", "爻"]


def test_prompt_has_no_jargon():
    rows = make_structure_rows(20, seed=0)
    for r in rows:
        p = structure_to_prompt(r)
        assert isinstance(p, str) and len(p) > 40
        for bad in _FORBIDDEN:
            assert bad not in p, bad


def test_structure_rows_shape():
    rows = make_structure_rows(8, seed=1)
    assert len(rows) == 8
    r = rows[0]
    assert 0 <= r["ben_k"] < 64 and 0 <= r["timing"] < 6
    assert len(r["force"]) == 6 and len(r["dangwei"]) == 6
    assert sum(r["moving"]) in (1, 2)


def test_paraphrase_fallback_varies_and_nonempty():
    base = ["眼下他在推进一个项目，资源慢慢到位，位置还算稳固。上面暂时没人管。"] * 6
    out = paraphrase_fallback(base, seed=3)
    assert len(out) == 6
    assert all(isinstance(s, str) and s for s in out)
    assert any(s != base[0] for s in out)


def test_generate_situations_handles_failure():
    rows = make_structure_rows(4, seed=2)

    def flaky(prompt):
        if "第二层" in prompt and len(prompt) % 2 == 0:
            raise RuntimeError("boom")
        return "一段没有术语的普通情境描述。"

    out = generate_situations(rows, flaky, on_error="skip")
    assert len(out) == 4
    assert any(x is None for x in out) or all(isinstance(x, str) for x in out)


def test_consistency_filter_mechanics():
    rows = make_structure_rows(12, seed=5)
    texts = [structure_to_prompt(r) for r in rows]
    texts[0] = None
    model = YiWorldModel(obs_dim=256)  # untrained; we test plumbing, not quality

    keep, hexo, sm = consistency_filter(texts, rows, model, "hash", min_sign_match=0.0)
    assert keep.shape == (12,) and hexo.shape == (12,) and sm.shape == (12,)
    assert not keep[0]                       # None row dropped
    assert keep[1:].all()                    # min_sign_match=0 keeps the rest
    assert (0.0 <= sm[1:]).all() and (sm[1:] <= 1.0).all()

    keep2, _, _ = consistency_filter(texts, rows, model, "hash", min_sign_match=1.0)
    assert keep2.sum() <= keep.sum()         # stricter threshold keeps no more


def test_jsonl_build_and_train_roundtrip(tmp_path):
    from yiwm.data import SemanticJsonlDataset
    from yiwm.losses import yi_world_loss

    path = str(tmp_path / "sem.jsonl")
    stats = build_semantic_jsonl(path, n=64, llm_fn=mock_llm_fn, filter_model=None, seed=1)
    assert stats["written"] == 64 and stats["dropped_gen"] == 0
    lines = [json.loads(x) for x in open(path, encoding="utf-8")]
    assert len(lines) == 64
    assert set(lines[0]) >= {"text", "ben_k", "force", "moving", "action", "timing"}

    ds = SemanticJsonlDataset(path, "hash")
    assert ds.obs_dim == 256
    b = ds(16, seed=0)
    assert b["obs"].shape == (16, 256) and b["hex"].shape == (16,)
    assert b["entity_states"].shape == (16, 5, 6)

    m = YiWorldModel(obs_dim=ds.obs_dim)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    l0 = yi_world_loss(m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"]), b)["total"]
    for _ in range(30):
        bb = ds(32, seed=None)
        out = m(bb["obs"], bb["entity_states"], bb["entity_cats"], bb["entity_adj"])
        L = yi_world_loss(out, bb)
        opt.zero_grad(); L["total"].backward(); opt.step()
    l1 = yi_world_loss(m(b["obs"], b["entity_states"], b["entity_cats"], b["entity_adj"]), b)["total"]
    assert l1 < l0                           # loss goes down on the semantic source


def test_build_semantic_jsonl_incremental_append(tmp_path):
    path = str(tmp_path / "inc.jsonl")
    build_semantic_jsonl(path, n=10, llm_fn=mock_llm_fn, seed=0)
    build_semantic_jsonl(path, n=10, llm_fn=mock_llm_fn, seed=99, append=True)
    assert sum(1 for _ in open(path, encoding="utf-8")) == 20
