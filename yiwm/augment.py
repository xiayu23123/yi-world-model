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

import random

import torch

from .constants import BINARY_HEX
from .synth import DOMAINS, TIMINGS, _structure

# --------------------------------------------------------------------------- #
# prompt
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = (
    "你把一个「处境的结构描述」改写成一段真实、具体的现代生活情境（80-160 字）。"
    "严格禁止出现：任何卦名（乾、坤、屯……）、爻位词（初/二/三/四/五/上、九、六、爻）、"
    "「阴/阳/阴气/阳气/当位/失位/得中/相应/五行/时位」等术语。"
    "只用普通人会说的话，描述一个具体的人在具体场景里的处境、动作倾向、外部约束。"
    "不要给建议，不要点评，不要用「他应该」。直接叙述。"
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
    from .synth import _force, _moving, _timing

    g = torch.Generator()
    g.manual_seed(seed)
    ben_k = torch.randint(0, 64, (n,), generator=g)
    bits, dw, dz, yy = _structure(ben_k)
    force0 = _force(bits, dw, dz, yy, g)
    mask, force = _moving(bits, dw, dz, yy, force0, g)
    timing = _timing(bits, mask)
    rows = []
    for i in range(n):
        rows.append({
            "ben_k": int(ben_k[i]),
            "timing": int(timing[i]),
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
