"""Structural priors for the I Ching. Nothing in this file is learned.

Canonical internal index `k` (0..63): hexagram k has yao i = (k >> i) & 1,
where i = 0 is the bottom line (初爻) and i = 5 is the top line (上爻).
1 = 陽 (yang), 0 = 陰 (yin).

King Wen (文王) sequence is a separate 1..64 ordering; we keep a permutation
map between it and the internal binary index.
"""

import torch

N_YAO = 6
N_HEX = 64

# --- all 64 hexagrams in canonical binary order ------------------------------
BINARY_HEX = torch.tensor(
    [[(k >> i) & 1 for i in range(N_YAO)] for k in range(N_HEX)],
    dtype=torch.float32,
)  # [64, 6]

# --- 8 trigrams (經卦), bits bottom-to-top ----------------------------------
TRIGRAMS = {
    "qian": (1, 1, 1), "dui": (1, 1, 0), "li": (1, 0, 1), "zhen": (1, 0, 0),
    "xun":  (0, 1, 1), "kan": (0, 1, 0), "gen": (0, 0, 1), "kun":  (0, 0, 0),
}
TRIGRAM_NAMES = ["qian", "dui", "li", "zhen", "xun", "kan", "gen", "kun"]
TRIGRAM_INDEX = {n: i for i, n in enumerate(TRIGRAM_NAMES)}
TRIGRAM_CN = {"qian": "乾", "dui": "兌", "li": "離", "zhen": "震",
              "xun": "巽", "kan": "坎", "gen": "艮", "kun": "坤"}


def _tri_value(name: str) -> int:
    b = TRIGRAMS[name]
    return b[0] | (b[1] << 1) | (b[2] << 2)


_TRI_VALUE_TO_INDEX = {_tri_value(n): TRIGRAM_INDEX[n] for n in TRIGRAM_NAMES}

# --- King Wen sequence: (lower trigram, upper trigram) ----------------------
# Index in this list = King Wen number - 1.
KING_WEN_TRIGRAMS = [
    ("qian", "qian"), ("kun", "kun"), ("zhen", "kan"), ("kan", "gen"),      # 1-4
    ("qian", "kan"), ("kan", "qian"), ("kan", "kun"), ("kun", "kan"),       # 5-8
    ("qian", "xun"), ("dui", "qian"), ("qian", "kun"), ("kun", "qian"),     # 9-12
    ("li", "qian"), ("qian", "li"), ("gen", "kun"), ("kun", "zhen"),        # 13-16
    ("zhen", "dui"), ("xun", "gen"), ("dui", "kun"), ("kun", "xun"),        # 17-20
    ("zhen", "li"), ("li", "gen"), ("kun", "gen"), ("zhen", "kun"),         # 21-24
    ("zhen", "qian"), ("qian", "gen"), ("zhen", "gen"), ("xun", "dui"),     # 25-28
    ("kan", "kan"), ("li", "li"), ("gen", "dui"), ("xun", "zhen"),          # 29-32
    ("gen", "qian"), ("qian", "zhen"), ("kun", "li"), ("li", "kun"),        # 33-36
    ("li", "xun"), ("dui", "li"), ("gen", "kan"), ("kan", "zhen"),          # 37-40
    ("dui", "gen"), ("zhen", "xun"), ("qian", "dui"), ("xun", "qian"),      # 41-44
    ("kun", "dui"), ("xun", "kun"), ("kan", "dui"), ("xun", "kan"),         # 45-48
    ("li", "dui"), ("xun", "li"), ("zhen", "zhen"), ("gen", "gen"),         # 49-52
    ("gen", "xun"), ("dui", "zhen"), ("li", "zhen"), ("gen", "li"),         # 53-56
    ("xun", "xun"), ("dui", "dui"), ("kan", "xun"), ("dui", "kan"),         # 57-60
    ("dui", "xun"), ("gen", "zhen"), ("li", "kan"), ("kan", "li"),          # 61-64
]
assert len(KING_WEN_TRIGRAMS) == 64

KING_WEN_NAMES = [
    "乾", "坤", "屯", "蒙", "需", "訟", "師", "比",
    "小畜", "履", "泰", "否", "同人", "大有", "謙", "豫",
    "隨", "蠱", "臨", "觀", "噬嗑", "賁", "剝", "復",
    "無妄", "大畜", "頤", "大過", "坎", "離", "咸", "恆",
    "遯", "大壯", "晉", "明夷", "家人", "睽", "蹇", "解",
    "損", "益", "夬", "姤", "萃", "升", "困", "井",
    "革", "鼎", "震", "艮", "漸", "歸妹", "豐", "旅",
    "巽", "兌", "渙", "節", "中孚", "小過", "既濟", "未濟",
]
assert len(KING_WEN_NAMES) == 64


def _king_wen_to_binary() -> list[int]:
    out = []
    for lower, upper in KING_WEN_TRIGRAMS:
        out.append(_tri_value(lower) | (_tri_value(upper) << 3))
    return out


# KING_WEN_TO_BINARY[kw_number - 1] -> internal binary index k
KING_WEN_TO_BINARY = torch.tensor(_king_wen_to_binary(), dtype=torch.long)
# BINARY_TO_KING_WEN[k] -> (kw_number - 1)
BINARY_TO_KING_WEN = torch.empty(64, dtype=torch.long)
BINARY_TO_KING_WEN[KING_WEN_TO_BINARY] = torch.arange(64)


def _hex_trigram_indices():
    lo = torch.empty(64, dtype=torch.long)
    up = torch.empty(64, dtype=torch.long)
    for k in range(64):
        lo[k] = _TRI_VALUE_TO_INDEX[k & 0b111]
        up[k] = _TRI_VALUE_TO_INDEX[(k >> 3) & 0b111]
    return lo, up


HEX_LOWER_TRIGRAM, HEX_UPPER_TRIGRAM = _hex_trigram_indices()  # [64], [64]

# --- 變: 一爻之變 adjacency (Hamming-1 graph over the internal index) --------
def _hamming1_adj():
    adj = torch.zeros(64, 64)
    for k in range(64):
        for b in range(6):
            adj[k, k ^ (1 << b)] = 1.0
    return adj


HEX_ADJ = _hamming1_adj()  # [64,64], every row sums to 6

# --- 錯卦: every line inverted --------------------------------------------------
HEX_CUO = torch.tensor([k ^ 0b111111 for k in range(64)], dtype=torch.long)

# --- 綜卦: hexagram turned upside down (top<->bottom); self for the 8 symmetric
def _zong():
    out = []
    for k in range(64):
        bits = [(k >> i) & 1 for i in range(6)][::-1]
        out.append(sum(v << i for i, v in enumerate(bits)))
    return torch.tensor(out, dtype=torch.long)


HEX_ZONG = _zong()

# --- 互卦 (nuclear): lines 2-3-4 -> lower, lines 3-4-5 -> upper (1-indexed) ----
def _hu():
    out = []
    for k in range(64):
        b = [(k >> i) & 1 for i in range(6)]
        nuc = [b[1], b[2], b[3], b[2], b[3], b[4]]
        out.append(sum(v << i for i, v in enumerate(nuc)))
    return torch.tensor(out, dtype=torch.long)


HEX_HU = _hu()

# --- 爻位: 1 = 陽位 (初三五 -> idx 0,2,4), 0 = 陰位 (二四六) -----------------
YAO_POS_PARITY = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.float32)

# --- 五行 generation / control ----------------------------------------------
# order: 木0 火1 土2 金3 水4
WUXING = ["mu", "huo", "tu", "jin", "shui"]
WUXING_CN = ["木", "火", "土", "金", "水"]
# M[a, b] = +1 if a 生 b, -1 if a 克 b, else 0
WUXING_MATRIX = torch.tensor([
    [0,  1, -1,  0,  0],   # 木 生火, 克土
    [0,  0,  1, -1,  0],   # 火 生土, 克金
    [0,  0,  0,  1, -1],   # 土 生金, 克水
    [-1, 0,  0,  0,  1],   # 金 生水, 克木
    [1, -1,  0,  0,  0],   # 水 生木, 克火
], dtype=torch.float32)
# element that controls (克) element x:  (x + 3) % 5
WUXING_CONTROLLER = torch.tensor([(x + 3) % 5 for x in range(5)], dtype=torch.long)

# 八卦 -> 五行 (纳甲/后天 convention).  index over TRIGRAM_NAMES order:
#   qian 金, dui 金, li 火, zhen 木, xun 木, kan 水, gen 土, kun 土
TRIGRAM_WUXING = torch.tensor([3, 3, 1, 0, 0, 4, 2, 2], dtype=torch.long)


def render_hexagram(k: int) -> str:
    """ASCII art, top line first."""
    rows = []
    for i in range(5, -1, -1):
        rows.append("█████████" if (k >> i) & 1 else "███   ███")
    return "\n".join(rows)
