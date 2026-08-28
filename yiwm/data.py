"""Task definition + a fully synthetic, self-consistent toy dataset.

------------------------------------------------------------------------------
TASK SCHEMA  (what "understanding the I Ching" is reduced to here)
------------------------------------------------------------------------------
A *situation* is:

  obs            : [obs_dim]      global observation of a scene
  entity_states  : [n_entities, 6]  signed state of each entity
  entity_cats    : [n_entities]   五行 class of each entity (0..4 = 木火土金水)
  entity_adj     : [n_entities, n_entities]  relation graph between entities

Supervision targets (deterministic functions of the underlying scene state,
so there IS a ground truth to learn):

  hex        : int in 0..63     本卦, internal binary index
  hex_next   : int in 0..63     之卦 after 老爻 flip
  moving     : [6] 0/1          which 爻 are 老 (changing)
  action     : int in 0..4      进 退 守 变 待
  yao_target : [6] float        target yao strengths (for the advice head)

------------------------------------------------------------------------------
TOY WORLD
------------------------------------------------------------------------------
A 5-species ecosystem, one species per 五行. Populations evolve one step of a
generalised Lotka-Volterra map whose interaction signs come from WUXING_MATRIX
(生 -> mutualist +, 克 -> antagonist -). The hexagram is read off the
z-scored log-populations; the action follows a fixed rule on the dominant
species' share and growth. Nothing about the mapping is random given the
populations, so a model can in principle fit it exactly. Success criterion:
when 木 is over-dominant and surging the model should output 变/退 and its
yao pattern should reflect the imbalance.
"""

import torch

from .constants import BINARY_HEX  # noqa: F401  (kept for reference / tests)
from .constants import WUXING_MATRIX

_POW6 = 2 ** torch.arange(6)          # [1,2,4,8,16,32]
OBS_DIM = 35                          # log p (5) + log p_next (5) + pairwise log-ratios (25)
N_ENTITIES = 5


def _step_populations(p: torch.Tensor) -> torch.Tensor:
    """p [B,5] > 0 -> next populations. Deterministic."""
    pn = p / p.sum(1, keepdim=True)
    A = 0.6 * (WUXING_MATRIX > 0).float() - 0.9 * (WUXING_MATRIX < 0).float()  # [5,5]
    growth = 0.15 + pn @ A.t()                                                # [B,5]
    return (p * torch.exp(0.3 * growth)).clamp(1e-3, 1e3)


def _yao_strength(p_prev: torch.Tensor, p_next: torch.Tensor) -> torch.Tensor:
    """[B,6] signed strength: 5 species z-scores + 1 overall-trend term."""
    lp = torch.log(p_next)
    z = (lp - lp.mean(1, keepdim=True)) / (lp.std(1, keepdim=True) + 1e-6)     # [B,5]
    trend = torch.log(p_next.sum(1)) - torch.log(p_prev.sum(1))               # [B]
    return torch.cat([z, trend.unsqueeze(1)], dim=1)                          # [B,6]


def _observe(p: torch.Tensor, p_next: torch.Tensor, g) -> torch.Tensor:
    lp, lpn = torch.log(p), torch.log(p_next)
    ratios = (lpn.unsqueeze(2) - lpn.unsqueeze(1)).reshape(p.size(0), -1)      # [B,25]
    base = torch.cat([lp, lpn, ratios], dim=1)                                # [B,35]
    return base + 0.05 * torch.randn(base.shape, generator=g)


def _entity_states(p: torch.Tensor, p_next: torch.Tensor) -> torch.Tensor:
    lp = torch.log(p_next)
    z = (lp - lp.mean(1, keepdim=True)) / (lp.std(1, keepdim=True) + 1e-6)     # [B,5]
    dz = torch.log(p_next) - torch.log(p)                                     # [B,5]
    feats = torch.stack(
        [z, z ** 2 - 1.0, torch.tanh(z), dz,
         torch.sign(z) * torch.sqrt(z.abs() + 1e-6), torch.tanh(dz)],
        dim=-1,
    )                                                                        # [B,5,6]
    return feats


def make_batch(batch_size: int, seed: int | None = None, device: str = "cpu") -> dict:
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    else:
        g.seed()

    p = torch.rand(batch_size, 5, generator=g) * 4.0 + 0.5
    p_next = _step_populations(p)

    ys = _yao_strength(p, p_next)                                             # [B,6]
    yang = (ys > 0).long()                                                   # [B,6]
    hex_bin = (yang * _POW6).sum(1)                                          # [B]

    thr = ys.abs().quantile(0.75, dim=1, keepdim=True).clamp(min=0.8)
    moving = (ys.abs() >= thr).long()                                        # [B,6]
    hex_next_bin = hex_bin ^ (moving * _POW6).sum(1)                         # [B]

    dom = p_next.argmax(1)                                                   # [B]
    share = p_next.max(1).values / p_next.sum(1)                             # [B]
    idx = torch.arange(batch_size)
    dom_growth = (torch.log(p_next) - torch.log(p))[idx, dom]               # [B]

    action = torch.full((batch_size,), 2, dtype=torch.long)                 # 守 shou
    action[(share > 0.35) & (dom_growth > 0.15)] = 1                        # 退 tui (过旺上冲)
    action[(share > 0.35) & (dom_growth < -0.05)] = 0                       # 进 jin (龙头衰, 补位)
    action[(share > 0.45) & (dom_growth > 0.05)] = 3                        # 变 bian (严重失衡)
    action[share < 0.24] = 4                                                # 待 dai (均衡)

    obs = _observe(p, p_next, g)
    ent_state = _entity_states(p, p_next)
    ent_cats = torch.arange(5).unsqueeze(0).repeat(batch_size, 1)
    ent_adj = torch.ones(batch_size, 5, 5)

    return {
        "obs": obs.to(device),
        "entity_states": ent_state.to(device),
        "entity_cats": ent_cats.to(device),
        "entity_adj": ent_adj.to(device),
        "hex": hex_bin.to(device),
        "hex_next": hex_next_bin.to(device),
        "moving": moving.to(device),
        "action": action.to(device),
        "yao_target": ys.to(device),
        "populations": p_next.to(device),
    }


class SemanticJsonlDataset:
    """Batch source backed by a JSONL file from `augment.build_semantic_jsonl`.

    Each line: {text, ben_k, force, moving, action, timing, ...}. Texts are
    embedded ONCE at construction with the given (frozen) text encoder; entity
    tensors are rebuilt from ben_k + force. Yields the same dict keys as
    `synth.make_synth_batch`, so train.py / losses.py consume it unchanged.
    Training on this only moves the `YinYangEncoder` head (the obs encoder is
    frozen by construction).
    """

    def __init__(self, path: str, text_encoder: str = "hash"):
        import json

        from .synth import _entities
        from .textenc import get_text_encoder

        recs = [json.loads(x) for x in open(path, encoding="utf-8") if x.strip()]
        if not recs:
            raise ValueError(f"{path} has no rows")
        enc_fn, self.obs_dim = get_text_encoder(text_encoder)

        self.texts = [r["text"] for r in recs]
        self.obs = enc_fn(self.texts)                                     # [N, obs_dim]
        ben_k = torch.tensor([r["ben_k"] for r in recs])
        force = torch.tensor([r["force"] for r in recs], dtype=torch.float32)
        moving = torch.tensor([r["moving"] for r in recs], dtype=torch.float32)
        cats, st, adj = _entities(ben_k, force)
        pow6 = 2 ** torch.arange(6)
        self.data = {
            "obs": self.obs,
            "entity_states": st,
            "entity_cats": cats,
            "entity_adj": adj,
            "hex": ben_k,
            "hex_next": ben_k ^ (moving.long() * pow6).sum(1),
            "moving": moving,
            "action": torch.tensor([r["action"] for r in recs]),
            "yao_target": force,
            "timing": torch.tensor([r["timing"] for r in recs]),
        }
        self.n = len(recs)

    def __call__(self, batch_size: int, seed: int | None = None, device: str = "cpu") -> dict:
        g = torch.Generator()
        g.manual_seed(seed) if seed is not None else g.seed()
        idx = torch.randint(0, self.n, (batch_size,), generator=g)
        out = {k: v[idx].to(device) for k, v in self.data.items()}
        out["text"] = [self.texts[i] for i in idx.tolist()]
        return out


def get_dataset(name: str, text_encoder: str = "hash", pool_size: int = 0):
    """name in {"eco", "synth"} -> (batch_fn, obs_dim).

    text_encoder only applies to "synth": "hash" (256-d, offline) or a frozen
    sentence-transformer ("minilm" / "minilm-ml", 384-d).
    pool_size > 0 pre-generates+embeds a fixed SynthPool (needed to make the
    slow ST encoders trainable); 0 = fresh generation every batch.
    """
    if name == "eco":
        return make_batch, OBS_DIM
    if name == "synth":
        from functools import partial

        from .synth import SynthPool, make_synth_batch
        from .textenc import get_text_encoder

        _, obs_dim = get_text_encoder(text_encoder)
        if pool_size and pool_size > 0:
            return SynthPool(pool_size, text_encoder), obs_dim
        return partial(make_synth_batch, text_encoder=text_encoder), obs_dim
    raise ValueError(f"unknown dataset {name!r}")
