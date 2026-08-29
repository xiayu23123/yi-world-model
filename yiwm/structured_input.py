"""Structured multiple-choice input -> yao-force vector, bypassing the text
encoder entirely. This is the honest fallback for "situation -> 卦" after the
free-text route failed to generalise (see README): a rule map, no learning on
the obs side.

LIMITATION: `polarity` is one global sign, so every line ends up the same sign
-> 本卦 is always 乾 (all yang) or 坤 (all yin). What varies is the *moving
line* (`phase`), hence the 之卦 and the action. To get other 本卦 you would
need per-line yin/yang input, which 4 MC questions do not provide.
"""

from __future__ import annotations

import torch

from .constants import YAO_POS_PARITY

QUESTIONS: dict[str, dict] = {
    "phase": {
        "text": "当前处于什么阶段？",
        "options": {
            "1": {"label": "潜藏 / 初创", "base": [0.9, 0.3, 0.3, 0.2, 0.1, 0.1]},
            "2": {"label": "生长 / 积累", "base": [0.3, 0.9, 0.3, 0.2, 0.1, 0.1]},
            "3": {"label": "守成 / 夯实", "base": [0.3, 0.3, 0.9, 0.3, 0.2, 0.1]},
            "4": {"label": "跃升 / 突破", "base": [0.2, 0.3, 0.4, 0.9, 0.3, 0.2]},
            "5": {"label": "显极 / 巅峰", "base": [0.1, 0.1, 0.2, 0.3, 0.9, 0.3]},
            "6": {"label": "衰极 / 转型", "base": [0.1, 0.1, 0.2, 0.2, 0.3, 0.9]},
        },
    },
    "polarity": {
        "text": "整体态势偏向？",
        "options": {
            "1": {"label": "积极 / 扩张 (阳)", "sign": 1.0},
            "2": {"label": "消极 / 收缩 (阴)", "sign": -1.0},
        },
    },
    "resource": {
        "text": "资源状态？",
        "options": {
            "1": {"label": "充足", "scale": 1.0},
            "2": {"label": "临界", "scale": 0.7},
            "3": {"label": "紧缺", "scale": 0.4},
        },
    },
    "domain": {
        "text": "领域（用于五行着色）",
        "options": {
            "1": {"label": "创业 / 创新", "wuxing": 0},   # 木
            "2": {"label": "关系 / 合作", "wuxing": 1},   # 火
            "3": {"label": "职场 / 管理", "wuxing": 2},   # 土
            "4": {"label": "投资 / 财务", "wuxing": 3},   # 金
            "5": {"label": "学习 / 研究", "wuxing": 4},   # 水
        },
    },
}


def choices_to_yao(choices: dict) -> torch.Tensor:
    """{"phase","polarity","resource","domain"} -> signed R^6 force vector."""
    ph = QUESTIONS["phase"]["options"][str(choices["phase"])]
    po = QUESTIONS["polarity"]["options"][str(choices["polarity"])]
    rs = QUESTIONS["resource"]["options"][str(choices["resource"])]

    y = torch.tensor(ph["base"], dtype=torch.float32) * po["sign"] * rs["scale"]
    # 当位 correction, same shape as synth._force
    proper = ((y > 0).float() == YAO_POS_PARITY).float()
    y = y + proper * 0.15 * po["sign"]
    return y.clamp(-1.0, 1.0)


@torch.no_grad()
def structured_infer(choices: dict, model) -> dict:
    """Structured input -> reading.

    本卦 / 动爻 / 之卦 are RULE-MAPPED (no learning): 本卦 from the input sign,
    动爻 = the `phase` line itself (the question literally asks which stage),
    之卦 = flip it. The trained model contributes only the *action* recommendation
    via its policy net (which weighs 当位 / 时位).
    """
    from .constants import BINARY_TO_KING_WEN, KING_WEN_NAMES, render_hexagram
    from .synth import ACTIONS_CN, _entities

    force = choices_to_yao(choices)                                  # [6]
    ben_k = int((force > 0).long() @ (2 ** torch.arange(6)))
    move_pos = int(choices["phase"]) - 1
    zhi_k = ben_k ^ (1 << move_pos)
    mask = [1 if i == move_pos else 0 for i in range(6)]

    cats, st, adj = _entities(torch.tensor([ben_k]), force.unsqueeze(0))
    model.eval()
    dummy_obs = torch.zeros(1, model.encoder.net[0].in_features)
    o = model(dummy_obs, st, cats, adj, hard=True, yao_override=force.unsqueeze(0))
    a = o["policy"]["action_logits"][0].softmax(0)
    base_act = int(a.argmax())

    # fragility: how unstable is the action recommendation under ±0.05 force noise?
    flips = 0
    for _ in range(24):
        op = model(dummy_obs, st, cats, adj, hard=True,
                   yao_override=(force + 0.05 * torch.randn(6)).unsqueeze(0))
        flips += int(op["policy"]["action_logits"].argmax(1).item() != base_act)
    frac = flips / 24
    if frac > 0.25:
        frag = f"⚠️ 高敏感区（action 翻转率 {frac:.0%}）：情境细节的微小差别可能改变建议，请确认关键信息是否遗漏。"
    elif frac > 0.12:
        frag = f"ℹ️ 中等敏感（action 翻转率 {frac:.0%}）：建议核对资源 / 阶段判断。"
    else:
        frag = None

    def kw(k):
        j = BINARY_TO_KING_WEN[k].item()
        return (j + 1, KING_WEN_NAMES[j])

    return {
        "force": [round(v, 2) for v in force.tolist()],
        "ben_hex": kw(ben_k),
        "ben_render": render_hexagram(ben_k),
        "moving_lines": mask,
        "zhi_hex": kw(zhi_k),
        "zhi_render": render_hexagram(zhi_k),
        "action": {ACTIONS_CN[i]: round(a[i].item(), 2) for i in range(5)},
        "fragility": frag,
    }


def ask_interactive() -> dict:
    """Prompt the 4 questions on stdin, return a choices dict."""
    choices = {}
    for key, q in QUESTIONS.items():
        print(f"\n{q['text']}")
        for oid, opt in q["options"].items():
            print(f"  {oid}. {opt['label']}")
        while True:
            ans = input("> ").strip()
            if ans in q["options"]:
                choices[key] = ans
                break
            print("  (无效选项)")
    return choices
