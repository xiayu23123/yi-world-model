"""時位決定論 synthetic generator.

Sample a 本卦, read its 時位 structure -- 當位 (line matches its position's
parity), 得中 (lines 2 & 5), 有應 (a 應 pair 初四/二五/三上 holds one yin one
yang) -- and 時序 (which line governs). Turn that into signed 爻 strengths,
a moving-line mask, an action, and a situation string.

Every label is a DETERMINISTIC function of structural features the model can
observe from the text, so the task is learnable. (The original draft made
action/timing a function of a fresh random index, and flipped the sign of
moving lines so the 本卦 label disagreed with the input -- both fixed here.)

``make_synth_batch`` returns the same dict keys as ``data.make_batch`` plus
``text`` and ``timing``, so train.py / losses.py / analysis.py work unchanged.
"""

import random

import torch

from .constants import (
    BINARY_HEX, HEX_HU, HEX_LOWER_TRIGRAM, HEX_UPPER_TRIGRAM, TRIGRAM_CN,
    TRIGRAM_NAMES, TRIGRAM_WUXING,
)
from .textenc import get_text_encoder

OBS_DIM_SYNTH = 256          # default ("hash") encoder width
N_ENTITIES = 5

_PARITY = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])   # 陽位 = 初三五
_ZHONG = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0])    # 得中 = 二五
_RESP = [(0, 3), (1, 4), (2, 5)]                         # 相應爻對
_POW6 = 2 ** torch.arange(6)
_YAO_CN = ["初", "二", "三", "四", "五", "上"]

TIMINGS = ["潜", "生", "守", "跃", "显", "亢"]
ACTIONS = ["jin", "tui", "shou", "bian", "dai"]          # 进 退 守 变 待 (== policy.py)
ACTIONS_CN = ["进", "退", "守", "变", "待"]
DOMAINS = ["职场", "创业", "投资", "人际", "健康", "政务", "学术", "家庭"]


# --------------------------------------------------------------------------- #
# structural features
# --------------------------------------------------------------------------- #
def _structure(ben_k: torch.Tensor):
    """ben_k [B] -> bits, 當位, 得中, 有應   each [B,6] float 0/1."""
    B = ben_k.shape[0]
    bits = BINARY_HEX[ben_k]                                  # [B,6]
    dangwei = (bits == _PARITY).float()
    dezhong = _ZHONG.expand(B, 6).clone()
    youying = torch.zeros(B, 6)
    for a, b in _RESP:
        diff = (bits[:, a] != bits[:, b]).float()
        youying[:, a] = diff
        youying[:, b] = diff
    return bits, dangwei, dezhong, youying


def _force(bits, dangwei, dezhong, youying, g):
    """Signed R^6 base strength. Non-moving lines stay well below 老 range."""
    mag = 0.40 + 0.12 * dangwei + 0.08 * dezhong + 0.08 * youying
    mag = mag + (torch.rand(bits.shape, generator=g) - 0.5) * 0.10
    mag = mag.clamp(0.12, 0.74)
    return mag * (bits * 2 - 1)                               # [B,6]


def _moving(bits, dangwei, dezhong, youying, force, g):
    """老陰老陽 mask + magnitude-boosted force (sign preserved)."""
    B = bits.shape[0]
    dw, dz, yy = dangwei.bool(), dezhong.bool(), youying == 0
    cand = ((dw & dz) | ((~dw) & yy)).float()                # 極盛 or 困而思變
    jitter = 0.05 * torch.rand(force.shape, generator=g)
    score = torch.where(
        cand.sum(1, keepdim=True) > 0,
        cand * (force.abs() + jitter),
        force.abs() + jitter,                                # fallback: strongest line
    )
    order = score.argsort(dim=1, descending=True)
    k = torch.randint(1, 3, (B,), generator=g)               # 1 or 2 老爻
    mask = torch.zeros(B, 6)
    for i in range(B):
        for j in range(int(k[i])):
            mask[i, order[i, j]] = 1.0
    ex = (0.92 + 0.07 * torch.rand(force.shape, generator=g)) * torch.sign(force)
    boosted = torch.where(mask.bool(), ex, force)
    return mask, boosted


def _timing(bits, mask):
    """時序 = position of the governing line (highest 老爻, else highest 陽爻)."""
    pos = torch.arange(6).float()
    has = mask.sum(1) > 0
    hm = (mask * pos).amax(1)
    hy = (bits * pos).amax(1)
    return torch.where(has, hm, hy).long().clamp(0, 5)


def _action(timing, dangwei, youying, mask):
    """f(時序, 主爻當位, 主爻有應) -> action index. Deterministic."""
    B = timing.shape[0]
    gov = (mask * torch.arange(6)).argmax(1)                 # governing line idx
    dw = dangwei[torch.arange(B), gov].bool()
    yy = youying[torch.arange(B), gov].bool()
    a = torch.full((B,), 2, dtype=torch.long)               # default 守
    for i in range(B):
        t, d, y = int(timing[i]), bool(dw[i]), bool(yy[i])
        if t == 0:                       # 潜: 藏而待时
            a[i] = 0 if (d and y) else 4          # 当位有应 -> 进, else 待
        elif t == 1:                     # 生: 渐进
            a[i] = 0 if y else 4                  # 有应 -> 进, else 待
        elif t == 2:                     # 守: 中位守成
            a[i] = 2 if d else 3                  # 当位 -> 守, else 变
        elif t == 3:                     # 跃: 或跃在渊
            a[i] = 3 if y else 2                  # 有应 -> 变, else 守
        elif t == 4:                     # 显: 飞龙在天, 盛极
            a[i] = 2 if d else 1                  # 当位 -> 守, else 退
        else:                            # 亢: 亢龙有悔
            a[i] = 3 if d else 1                  # 当位 -> 变, else 退
    return a


# --------------------------------------------------------------------------- #
# text + observation
# --------------------------------------------------------------------------- #
def _text(ben_k, timing, force, dangwei, youying, rng):
    dom = rng.choice(DOMAINS)
    tm = TIMINGS[int(timing)]
    lo = TRIGRAM_CN[TRIGRAM_NAMES[HEX_LOWER_TRIGRAM[ben_k].item()]]
    up = TRIGRAM_CN[TRIGRAM_NAMES[HEX_UPPER_TRIGRAM[ben_k].item()]]
    out = [f"{dom}场景，处于「{tm}」之时。下卦{lo}，上卦{up}。"]
    for p in range(6):
        yinyang = "阳" if force[p] > 0 else "阴"
        mag = force[p].abs().item()
        lvl = "势强" if mag > 0.8 else ("势中" if mag > 0.5 else "势弱")
        dw = "当位" if dangwei[p] > 0 else "失位"
        yo = "有应" if youying[p] > 0 else "无应"
        out.append(f"{_YAO_CN[p]}爻{yinyang}，{lvl}，{dw}{yo}。")
    return "".join(out)


def _entities(ben_k, force):
    lo = HEX_LOWER_TRIGRAM[ben_k]
    up = HEX_UPPER_TRIGRAM[ben_k]
    hu = HEX_HU[ben_k]
    hulo = HEX_LOWER_TRIGRAM[hu]
    huup = HEX_UPPER_TRIGRAM[hu]
    cats = torch.stack(
        [TRIGRAM_WUXING[lo], TRIGRAM_WUXING[up],
         TRIGRAM_WUXING[hulo], TRIGRAM_WUXING[huup], TRIGRAM_WUXING[up]],
        dim=1,
    )                                                        # [B,5]
    seg = torch.stack(
        [force[:, :3].mean(1), force[:, 3:].mean(1), force[:, 1:4].mean(1),
         force[:, 2:5].mean(1), force.mean(1)],
        dim=1,
    )                                                        # [B,5]
    st = torch.stack(
        [seg, seg ** 2 - 1.0, torch.tanh(seg), force[:, :5],
         torch.sign(seg) * seg.abs().sqrt(), torch.tanh(2 * seg)],
        dim=-1,
    )                                                        # [B,5,6]
    B = ben_k.shape[0]
    adj = torch.eye(5).expand(B, 5, 5).clone()
    for a, b in [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 4), (0, 4), (1, 4)]:
        adj[:, a, b] = adj[:, b, a] = 1.0
    return cats.long(), st, adj


# --------------------------------------------------------------------------- #
def make_synth_batch(
    batch_size: int,
    seed: int | None = None,
    device: str = "cpu",
    text_encoder: str = "hash",
) -> dict:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()
    rng = random.Random(seed)
    enc_fn, _ = get_text_encoder(text_encoder)

    ben_k = torch.randint(0, 64, (batch_size,), generator=g)
    bits, dw, dz, yy = _structure(ben_k)
    force0 = _force(bits, dw, dz, yy, g)
    mask, force = _moving(bits, dw, dz, yy, force0, g)
    timing = _timing(bits, mask)
    action = _action(timing, dw, yy, mask)

    hex_bin = ben_k.clone()
    hex_next = ben_k ^ (mask.long() * _POW6).sum(1)

    texts = [
        _text(ben_k[i], timing[i], force[i], dw[i], yy[i], rng)
        for i in range(batch_size)
    ]
    obs = enc_fn(texts)
    cats, st, adj = _entities(ben_k, force)

    return {
        "obs": obs.to(device),
        "entity_states": st.to(device),
        "entity_cats": cats.to(device),
        "entity_adj": adj.to(device),
        "hex": hex_bin.to(device),
        "hex_next": hex_next.to(device),
        "moving": mask.to(device),
        "action": action.to(device),
        "yao_target": force.to(device),
        "timing": timing.to(device),
        "text": texts,
    }


class SynthPool:
    """Pre-generate + embed a fixed pool once, then sample batches from it.

    A sentence-transformer obs encoder costs ~2 s / 256 texts on CPU, so fresh
    infinite generation is impractical for from-scratch training. Embedding a
    finite pool once (e.g. 20k rows ~ a few minutes) makes ST training tractable.
    With the ``hash`` encoder generation is already cheap and a pool is optional.
    Eval (demo / analysis) still uses fresh generation -- do not eval on the pool.
    """

    def __init__(self, pool_size: int, text_encoder: str = "hash", seed: int = 0):
        base = make_synth_batch(pool_size, seed=seed, text_encoder=text_encoder)
        self.texts = base.pop("text")
        self.data = base                              # tensors, on cpu
        self.n = pool_size

    def __call__(self, batch_size: int, seed: int | None = None, device: str = "cpu") -> dict:
        g = torch.Generator()
        g.manual_seed(seed) if seed is not None else g.seed()
        idx = torch.randint(0, self.n, (batch_size,), generator=g)
        out = {k: v[idx].to(device) for k, v in self.data.items()}
        out["text"] = [self.texts[i] for i in idx.tolist()]
        return out
