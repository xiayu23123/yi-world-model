"""LLM back-inversion data augmentation (方向一).

Given a `synth` structural row (時序 + per-爻 當位/得中/有應 + domain), ask an LLM
to write a *plain modern-life* situation paragraph -- no 卦 names, no 爻位 terms,
no 阴阳/五行 vocabulary. This adds **within-hexagram diversity** (many surface
realisations of the same structure); it does NOT add cross-hexagram
distinguishability, so treat the output as augmentation, not ground truth.

Risk control (`consistency_filter`): run a *already-trained* model on each
generated text, recover the yao sign pattern, and drop any sample whose
recovered 本卦 disagrees with the structure it was generated from.

Nothing here runs an LLM by default. `llm_fn` is a plug: `str -> str`.
`anthropic_llm_fn()` builds one from the `anthropic` SDK; `paraphrase_fallback`
is a no-LLM surface-variation stand-in so the pipeline is runnable offline.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch

from .constants import BINARY_HEX
from .synth import ACTIONS, DOMAINS, TIMINGS, _structure

# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "你把一个「处境的结构描述」改写成一段真实、具体的现代生活情境（80-160 字）。"
    "严格禁止出现：任何卦名（乾、坤、屯……）、爻位词（初/二/三/四/五/上、九、六、爻）、"
    "「阴/阳/阴气/阳气/当位/失位/得中/相应/五行/时位」等术语。"
    "只描述态势结构：具体的人、具体场景、他的处境、动作倾向、资源状态、外部约束。"
    "不要讲道理，不要给建议，不要点评，不要用「他应该」。"
    "禁止鸡汤词：努力、坚持、善良、正直、成功、正能量、天道酬勤、心态、格局。"
    "直接叙述。"
)

_STAGE_HINT = {
    "潜": "事情还没启动，力量在积蓄，外部没人注意",
    "生": "刚起步，在稳步推进，资源慢慢到位",
    "守": "到了中段，位置稳固，重点是守住不出错",
    "跃": "关键跳跃点，可进可退，正在犹豫要不要发力",
    "显": "达到顶点，非常显眼，盛极",
    "亢": "过头了，绷得太满，开始有反噬迹象",
}


def structure_to_prompt(row: dict) -> str:
    """row: {"timing": int, "dangwei":[6], "youying":[6], "force":[6]} -> user prompt."""
    dom = row.get("domain") or random.choice(DOMAINS)
    tm = TIMINGS[int(row["timing"])]
    lines = [f"领域：{dom}", f"阶段感觉：{_STAGE_HINT[tm]}"]
    names = ["最底层的起点", "第二层", "第三层", "第四层", "第五层（近顶）", "最顶层"]
    for i in range(6):
        strong = "有力" if abs(float(row["force"][i])) > 0.8 else (
            "一般" if abs(float(row["force"][i])) > 0.5 else "薄弱")
        fit = "位置合适" if row["dangwei"][i] > 0 else "位置别扭"
        sup = "有外部呼应" if row["youying"][i] > 0 else "孤立无援"
        lines.append(f"- {names[i]}：{strong}，{fit}，{sup}")
    return (
        "结构描述：\n" + "\n".join(lines)
        + "\n\n把它写成一段现代生活情境。记住：不许出现任何术语或卦名。"
    )


# --------------------------------------------------------------------------- #
# back-ends
# --------------------------------------------------------------------------- #
def anthropic_llm_fn(model: str | None = None, max_tokens: int = 400):
    """Build an `str -> str` generator from the `anthropic` SDK. Needs a working
    ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL in the env."""
    import os

    import anthropic

    client = anthropic.Anthropic()
    model = model or os.environ.get("ANTHROPIC_DEFAULT_HAIKU_MODEL") or "claude-haiku-4-5"

    def gen(prompt: str) -> str:
        r = client.messages.create(
            model=model, max_tokens=max_tokens, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.content[0].text.strip()

    return gen


def mock_llm_fn(prompt: str) -> str:
    """Deterministic placeholder for pipeline tests. Not real generation."""
    h = abs(hash(prompt)) % 1000
    return (
        f"某人正在处理一件事，手上的条件时松时紧，外部关注度不高。"
        f"他在权衡要不要现在发力，还是再等等看。编号 {h}。"
    )


def get_llm_fn(name: str, model: str | None = None):
    """name -> a `str -> str` generator. 'mock' | 'anthropic' | 'ollama'."""
    if name == "mock":
        return mock_llm_fn
    if name == "anthropic":
        return anthropic_llm_fn(model=model)
    if name == "ollama":
        return ollama_llm_fn(model=model or "llama3.1:8b")
    raise ValueError(f"unknown --llm {name!r}")


def ollama_llm_fn(model: str = "llama3.1:8b", base_url: str = "http://localhost:11434",
                  timeout: float = 60.0):
    """Local offline generation via Ollama. Needs `requests` and a running daemon."""
    import requests

    def gen(prompt: str) -> str:
        r = requests.post(
            f"{base_url}/api/generate",
            json={"model": model, "prompt": SYSTEM_PROMPT + "\n\n" + prompt, "stream": False},
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()["response"].strip()

    return gen


_SWAPS = [
    ("推进", "推动"), ("资源", "人手和预算"), ("稳步", "一点点"), ("犹豫", "拿不定主意"),
    ("盛极", "风头正劲"), ("反噬", "反弹"), ("外部", "上面"), ("薄弱", "撑不住"),
    ("有力", "顶得住"), ("孤立无援", "没人搭把手"), ("呼应", "配合"),
]


def paraphrase_fallback(base_texts, seed: int | None = None):
    """No-LLM surface variation, so the augmentation pipeline is testable offline.
    Not a substitute for real generation -- it cannot add genuine diversity."""
    rng = random.Random(seed)
    out = []
    for t in base_texts:
        s = t
        for a, b in _SWAPS:
            if a in s and rng.random() < 0.6:
                s = s.replace(a, b)
        parts = s.split("。")
        if len(parts) > 3 and rng.random() < 0.5:
            i, j = 1, 2
            parts[i], parts[j] = parts[j], parts[i]
            s = "。".join(parts)
        if rng.random() < 0.4:
            s = "眼下，" + s
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
# generation + consistency filter
# --------------------------------------------------------------------------- #
def make_structure_rows(n: int, seed: int = 0) -> list[dict]:
    """Sample n synth structural rows (no text, no obs) for prompting."""
    from .synth import _action, _force, _moving, _timing

    g = torch.Generator()
    g.manual_seed(seed)
    ben_k = torch.randint(0, 64, (n,), generator=g)
    bits, dw, dz, yy = _structure(ben_k)
    force0 = _force(bits, dw, dz, yy, g)
    mask, force = _moving(bits, dw, dz, yy, force0, g)
    timing = _timing(bits, mask)
    action = _action(timing, dw, yy, mask)
    rows = []
    for i in range(n):
        rows.append({
            "ben_k": int(ben_k[i]),
            "timing": int(timing[i]),
            "action": int(action[i]),
            "dangwei": dw[i].tolist(),
            "youying": yy[i].tolist(),
            "force": force[i].tolist(),
            "moving": mask[i].tolist(),
        })
    return rows


def generate_situations(rows: list[dict], llm_fn, on_error: str = "skip") -> list[str | None]:
    """rows -> texts via llm_fn(structure_to_prompt(row)). None where it failed."""
    out: list[str | None] = []
    for row in rows:
        try:
            out.append(llm_fn(structure_to_prompt(row)))
        except Exception:  # noqa: BLE001
            if on_error == "raise":
                raise
            out.append(None)
    return out


@torch.no_grad()
def consistency_filter(texts, rows, model, text_encoder="hash", min_sign_match: float = 5 / 6):
    """Encode each text, run the trained model, recover the 本卦 sign pattern,
    keep rows where >= min_sign_match of the 6 signs agree with the structure.

    Returns (keep_mask: BoolTensor[n], recovered_hex: LongTensor[n], sign_match: FloatTensor[n]).
    Rows with text is None are dropped.
    """
    from .textenc import get_text_encoder

    enc_fn, _ = get_text_encoder(text_encoder)
    valid = [i for i, t in enumerate(texts) if t]
    if not valid:
        z = torch.zeros(len(texts))
        return z.bool(), z.long(), z

    obs = enc_fn([texts[i] for i in valid])
    tgt_bits = BINARY_HEX[torch.tensor([rows[i]["ben_k"] for i in valid])]   # [v,6]

    model.eval()
    yao = model.encoder(obs)                                                # [v,6] signed
    pred_bits = (yao > 0).float()
    sm = (pred_bits == tgt_bits).float().mean(1)                            # [v]
    pow6 = 2 ** torch.arange(6)
    recovered = (pred_bits.long() * pow6).sum(1)

    keep = torch.zeros(len(texts), dtype=torch.bool)
    hexo = torch.zeros(len(texts), dtype=torch.long)
    smo = torch.zeros(len(texts))
    for k, i in enumerate(valid):
        keep[i] = bool(sm[k] >= min_sign_match)
        hexo[i] = int(recovered[k])
        smo[i] = float(sm[k])
    return keep, hexo, smo


def build_semantic_jsonl(
    path: str,
    n: int,
    llm_fn,
    filter_model=None,
    text_encoder: str = "hash",
    min_sign_match: float = 5 / 6,
    seed: int = 0,
    append: bool = False,
) -> dict:
    """Generate n situations, optionally consistency-filter, write one JSON object
    per line: {text, ben_k, force, moving, action, timing, sign_match}.

    Writes incrementally (one line per accepted sample), so a crash after k
    samples keeps the first k. Returns {"written": int, "attempted": int,
    "dropped_gen": int, "dropped_filter": int}.
    """
    import json

    rows = make_structure_rows(n, seed=seed)
    texts = generate_situations(rows, llm_fn, on_error="skip")

    if filter_model is not None:
        keep, _, sm = consistency_filter(texts, rows, filter_model, text_encoder, min_sign_match)
    else:
        keep = torch.tensor([t is not None for t in texts])
        sm = torch.full((n,), float("nan"))

    dropped_gen = sum(t is None for t in texts)
    written = 0
    with open(path, "a" if append else "w", encoding="utf-8") as f:
        for i, (row, text) in enumerate(zip(rows, texts)):
            if not keep[i]:
                continue
            rec = {
                "text": text,
                "ben_k": row["ben_k"],
                "force": row["force"],
                "moving": row["moving"],
                "action": row["action"],
                "timing": row["timing"],
                "sign_match": None if sm[i].isnan() else round(float(sm[i]), 3),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            written += 1
    return {
        "written": written,
        "attempted": n,
        "dropped_gen": dropped_gen,
        "dropped_filter": n - dropped_gen - written,
    }


_VARIANT_PROMPT = (
    "把下面这段情境扩写成一段新的、细节不同的现代情境（80-160 字），"
    "人物、行业、具体约束都可以换，但「所处阶段 / 力量强弱 / 是否孤立」的结构保持不变。"
    "禁止出现卦名、爻位词、阴阳/五行术语和鸡汤词。只叙述，不点评。\n\n原情境：\n{src}"
)

_CANONICAL_PROMPT = (
    "下面是一句古代格言式的处境判断。请把它改写成一段具体的现代生活情境（80-160 字），"
    "保留它描述的「态势结构」——所处阶段、力量强弱、进退倾向——但换成当代的人物、行业和约束。"
    "禁止出现卦名、爻位词、阴阳/五行术语和鸡汤词。直接叙述，不解释、不点评。\n\n原文：\n{src}"
)


def build_from_seed(
    seed_path: str,
    llm_fn,
    out_path: str,
    n_variants: int = 3,
    text_field: str = "modern_text",
    filter_model=None,
    text_encoder: str = "hash",
    min_sign_match: float = 5 / 6,
) -> dict:
    """Expand a filled `yao_seed.json` into a training JSONL.

    For every seed row that has a non-empty `text_field`, ask `llm_fn` for
    `n_variants` re-phrasings that keep the structure, optionally consistency-
    filter them, and append `{text, ben_k, force, moving, action, timing}` lines.
    Rows with no text yet are skipped (reported in `skipped_no_text`).
    """
    import json

    seeds = json.loads(Path(seed_path).read_text(encoding="utf-8"))
    filled = [s for s in seeds if s.get(text_field, "").strip()]
    prompt_tpl = _CANONICAL_PROMPT if text_field == "canonical_text" else _VARIANT_PROMPT

    texts, rows = [], []
    for s in filled:
        src = s[text_field].strip()
        for _ in range(n_variants):
            try:
                texts.append(llm_fn(prompt_tpl.format(src=src)))
            except Exception:  # noqa: BLE001
                texts.append(None)
            rows.append({
                "ben_k": s["hex_index"],
                "force": s["yao_target"],
                "moving": s["moving"],
                "action": ACTIONS.index(s["action"]) if s["action"] in ACTIONS else 2,
                "timing": TIMINGS.index(s["timing"]) if s["timing"] in TIMINGS else 0,
            })

    if filter_model is not None:
        keep, _, sm = consistency_filter(texts, rows, filter_model, text_encoder, min_sign_match)
    else:
        keep = torch.tensor([t is not None for t in texts])
        sm = torch.full((len(texts),), float("nan"))

    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for i, (row, text) in enumerate(zip(rows, texts)):
            if not keep[i]:
                continue
            f.write(json.dumps({**{"text": text}, **row,
                                "sign_match": None if sm[i].isnan() else round(float(sm[i]), 3)},
                               ensure_ascii=False) + "\n")
            f.flush()
            written += 1
    return {
        "seeds_total": len(seeds),
        "seeds_filled": len(filled),
        "skipped_no_text": len(seeds) - len(filled),
        "variants_attempted": len(texts),
        "written": written,
    }


def _load_filter_model(ckpt: str):
    blob = torch.load(ckpt)
    if isinstance(blob, dict) and "state_dict" in blob:
        state, te = blob["state_dict"], blob.get("text_encoder", "hash")
    else:
        state, te = blob, "hash"
    from .model import YiWorldModel
    from .textenc import get_text_encoder

    _, obs_dim = get_text_encoder(te)
    m = YiWorldModel(obs_dim=obs_dim)
    m.load_state_dict(state)
    return m, te


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="expand a filled yao_seed.json into a training JSONL")
    ap.add_argument("--seed-file", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-variants", type=int, default=3)
    ap.add_argument("--llm", choices=["mock", "anthropic", "ollama"], default="mock")
    ap.add_argument("--model", default=None)
    ap.add_argument("--text-field", default="modern_text")
    ap.add_argument("--filter-ckpt", default=None,
                    help="checkpoint for the consistency filter (drops structure-inconsistent variants)")
    ap.add_argument("--min-sign-match", type=float, default=5 / 6)
    a = ap.parse_args()

    fmodel, fenc = (None, "hash")
    if a.filter_ckpt:
        fmodel, fenc = _load_filter_model(a.filter_ckpt)
    stats = build_from_seed(
        a.seed_file, get_llm_fn(a.llm, a.model), a.output,
        n_variants=a.n_variants, text_field=a.text_field,
        filter_model=fmodel, text_encoder=fenc, min_sign_match=a.min_sign_match,
    )
    print(stats)
