"""Pluggable text -> obs encoders.  The base encoder is always FROZEN:
embeddings are produced under ``torch.no_grad()`` and detached, so no gradient
can ever reach it -- only the downstream ``YinYangEncoder`` head trains.

  hash       256-d signed feature-hashing bag of char uni/bi-grams. No deps,
             fully offline, deterministic. Zero synonym generalisation.
  minilm     384-d  sentence-transformers/all-MiniLM-L6-v2  (weak on Chinese).
  minilm-ml  384-d  paraphrase-multilingual-MiniLM-L12-v2   (real Chinese).

``get_text_encoder(name) -> (encode_fn, obs_dim)``.  encode_fn(list[str]) ->
FloatTensor [B, obs_dim] on CPU, detached.
"""

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch


def _find_local_hf_cache() -> str | None:
    """Local HF hub cache to load sentence-transformers from, if one exists.
    Order: $YIWM_ST_CACHE, $HF_HOME, ./cache next to the repo. A dir counts only
    if it contains a `hub/` subdir. If none is found the model resolves the
    normal way (default HF cache / download)."""
    for cand in (
        os.environ.get("YIWM_ST_CACHE"),
        os.environ.get("HF_HOME"),
        str(Path(__file__).resolve().parent.parent / "cache"),
    ):
        if cand and (Path(cand) / "hub").is_dir():
            return cand
    return None


# Must run before `sentence_transformers` / `huggingface_hub` import, so the
# cache path and offline flag are read from the environment at their init time.
_LOCAL_HF_CACHE = _find_local_hf_cache()
if _LOCAL_HF_CACHE:
    os.environ.setdefault("HF_HOME", _LOCAL_HF_CACHE)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

# --------------------------------------------------------------------------- #
# offline hashing bag
# --------------------------------------------------------------------------- #
HASH_DIM = 256


@lru_cache(maxsize=100_000)
def _tok_hash(tok: str, dim: int) -> int:
    h = int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")
    return (h % dim) | (0x10000 if (h >> 8) & 1 else 0)


def hash_bag(texts, dim: int = HASH_DIM) -> torch.Tensor:
    rows, cols, vals = [], [], []
    for i, t in enumerate(texts):
        toks = list(t)
        toks += [t[j:j + 2] for j in range(len(t) - 1)]
        for tok in toks:
            packed = _tok_hash(tok, dim)
            rows.append(i)
            cols.append(packed & 0xFFFF)
            vals.append(1.0 if packed & 0x10000 else -1.0)
    out = torch.zeros(len(texts), dim)
    out.index_put_((torch.tensor(rows), torch.tensor(cols)), torch.tensor(vals), accumulate=True)
    return out / (out.abs().amax(1, keepdim=True) + 1e-6)


# --------------------------------------------------------------------------- #
# frozen sentence-transformer
# --------------------------------------------------------------------------- #
_ST_NAMES = {
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "minilm-ml": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
_ST_CACHE: dict[str, object] = {}


def _load_st(key: str):
    if key not in _ST_CACHE:
        from sentence_transformers import SentenceTransformer  # lazy: optional dep

        model = SentenceTransformer(_ST_NAMES[key])
        model.eval()
        for p in model.parameters():           # frozen -- belt & braces
            p.requires_grad_(False)
        _ST_CACHE[key] = model
    return _ST_CACHE[key]


def _st_encode(key: str):
    model = _load_st(key)

    @torch.no_grad()
    def encode(texts):
        emb = model.encode(
            list(texts), convert_to_tensor=True, normalize_embeddings=True,
            show_progress_bar=False,
        )
        # .encode uses torch.inference_mode internally; clone() strips that flag
        # so the (frozen) embedding can feed the trainable head's autograd graph.
        return emb.float().cpu().clone()

    return encode


# --------------------------------------------------------------------------- #
def get_text_encoder(name: str):
    """name -> (encode_fn, obs_dim).  Raises a clear error if an optional
    backend is unavailable."""
    if name == "hash":
        return (lambda texts: hash_bag(texts, HASH_DIM)), HASH_DIM
    if name in _ST_NAMES:
        try:
            enc = _st_encode(name)
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"text encoder {name!r} needs `pip install sentence-transformers` "
                f"and a one-time download of {_ST_NAMES[name]!r}. Original: {e}"
            ) from e
        return enc, 384
    raise ValueError(f"unknown text encoder {name!r}; choices: hash, {', '.join(_ST_NAMES)}")
