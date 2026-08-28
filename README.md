# Yi-World-Model (`yiwm`)

A differentiable world model built on the **structural topology of the I Ching**
(易经). The discrete scaffolding — 64 hexagrams, the King Wen order, the
错/综/互 (inverse / reverse / nuclear) relations, the one-line-change graph, the
五行 (five-element) generation/control matrix — is hard-coded as prior knowledge.
Only the continuous parts are learned.

**Status:** research prototype. The core dynamics (本卦 / 动爻 / 之卦 / 行动)
converge on a synthetic and a noisy dynamical benchmark. The semantic-observation
path is wired and validated but not yet the default.

## What this is *not*

- Not a fortune-telling app.
- Not an LLM wrapper.
- Not a retrieval system.

## What it *is*

- A structured state-space model: 64 hexagrams as vertices of a signed
  6-cube, state = `R^6` *yao-force* (sign = 阴/阳, magnitude = 老/少, i.e. how
  close a line is to flipping).
- A differentiable **change engine**: line phase-transitions via a
  straight-through estimator with a **rank-aware adaptive threshold**.
- A **multi-relation graph net** over hexagrams (R-GCN with 4 learned-gated
  relations) and a **multiplicative graph dynamics** over 五行-typed entities.

## Pipeline

```
obs ──► YinYangEncoder ──► 6 yao-forces (R^6, signed)
                              │
                              ├─► HexagramInference ──► 64-way logits   (本卦)
                              │      R-GCN over {一爻之变, 错, 综, 互} + 经卦 embeds + polarity
entities ──► WuxingDynamics ──► Δstate   (生/克 field × relation graph, multiplicative)
                              │
                     aggregate ─► yao_next ─► ChangeEngine ──► 之卦 logits + 动爻 mask
                              │                  rank-aware threshold, STE, expected-yao (no argmax)
                     本卦 feat ─► TemporalPositionalPolicy ──► action {进 退 守 变 待} + intensity
```

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                            # 33 passed
python -m yiwm.train --data eco   --steps 12000      # ~3 min CPU
python -m yiwm.train --data synth --steps 8000  --ckpt checkpoints/yiwm_synth.pt
python -m yiwm.demo     --ckpt checkpoints/yiwm.pt        # one deterministic inference
python -m yiwm.analysis --ckpt checkpoints/yiwm.pt        # error decomposition + relation gates
```

## Results (converged)

| dataset | 本卦 | 动爻 / yao | 之卦 | 行动 |
|---|---|---|---|---|
| `eco`  (noisy Lotka-Volterra ecosystem, observation aliasing) | 0.96 | 0.97 | 0.84 | 0.97 |
| `synth` (时位决定论 generator, idealized) | 0.998 | 0.977 | 0.937 | 0.99 |

**The 之卦 ceiling is a compounding wall, not a bug.** `之卦` requires all 6
line-change bits right; per-yao accuracy `p ≈ 0.97` gives `p^6 ≈ 0.84`, which the
`eco` number tracks almost exactly. `analysis` confirms ~94% of the residual
之卦 errors are `本卦` right + 动爻 mask wrong; the structural map itself
contributes ~0. Breaking the wall needs a *joint* 6-bit mutation head or a
less lossy observation — not more tuning.

## Two data sources (`--data`), identical dict keys

- **`eco`** (`data.py`) — 5 species, one per 五行, evolved one generalized
  Lotka-Volterra step with interaction signs from the 五行 matrix. 卦 read off
  z-scored log-populations; action from a fixed rule on the dominant species.
- **`synth`** (`synth.py`) — sample a 本卦, read its 时位 structure
  (当位 / 得中 / 有应 / 时序), emit signed yao-forces, a moving-line mask, an
  action, and a Chinese situation string. Every label is a **deterministic
  function of structure the model can observe**, so the task is learnable.
  老爻 only boost *magnitude* (sign stays), so `sign(force) == 本卦` everywhere
  and `之卦` = flip the mask.

## Observation encoder (`textenc.py`, `--text-encoder`) — always **frozen**

Embeddings are produced under `torch.no_grad()` and `.clone()`d (strips the
inference-mode flag); no gradient can reach the encoder — only the downstream
`YinYangEncoder` head trains.

| name | dim | notes |
|---|---|---|
| `hash` (default) | 256 | signed feature-hashing bag of char uni/bi-grams. Offline, deterministic, zero deps. **No synonym generalization** — "融资" and "筹资" hash apart. |
| `minilm` | 384 | frozen `all-MiniLM-L6-v2`. English-trained: collapses the Chinese templates (`sim(融资紧缺, 筹资困难) ≈ 1.0`) → **regresses** 本卦. |
| `minilm-ml` | 384 | frozen `paraphrase-multilingual-MiniLM-L12-v2`. Real Chinese: `sim(融资/筹资)=0.87`, `sim(融资/天气)=-0.12`, zero-shot `sim(蛰伏/潜藏)=0.67`. At matched budget (9k steps, `--synth-pool 16000`): 本卦 0.996 / moving 0.966 / 之卦 ~0.91 / 行动 ~0.96 — near-parity with `hash`, 行动 trails ~3pts. |

`textenc.py` auto-detects a local HF hub cache at import time (`$YIWM_ST_CACHE`,
`$HF_HOME`, or `./cache/`) and sets `HF_HUB_OFFLINE=1` when found. Sentence
transformers cost ~2 s / 256 texts on CPU, so use `--synth-pool N` to
pre-generate + pre-embed a fixed pool once (`SynthPool`); evaluation still uses
fresh generation.

## What moved the needle (`eco`, fixed-threshold baseline → full stack)

| change | 动爻/yao | 之卦 | why |
|---|---|---|---|
| baseline (fixed per-position threshold, no encoder yao supervision) | 0.955 | 0.77 | |
| rank-aware threshold alone | collapses | — | hard 本卦 CE only pins the *sign*; encoder saturates `|y|→1`, the rank net has no magnitude spread |
| **+ `L_yao`** — Huber-regress the encoder's 6-d output to `tanh(yao_target)` | **0.966** | **0.83** | gives `|y|` a magnitude geometry. The actual unlock. |
| + pairwise ranking loss + `diff_to_max` feature | 0.97 | 0.85 | marginal on `eco`; on `synth` it drives the structural-map residual to 0 |

**Rejected with evidence:** vertical neighbor-diff features
(`corr(|Δ neighbor|, moving) = 0.06` on `eco` — the label is top-k by magnitude,
not a spatial gradient); per-channel adaptive soft-label temperature (the 本卦
residual is observation aliasing — different hexagrams, near-identical obs — not
label-boundary noise); soft-KL 之卦 target (does not lift the moving-mask
compounding ceiling). `--soft-hex-temp` is implemented but **off by default**:
soft labels help only when `yao_target` is on a common scale (synth), and hurt
`eco` (its `trend` channel is a different scale).

## Modules

| file | role |
|---|---|
| `constants.py` | hexagram binary codes, King Wen permutation, 错/综/互, one-line-change graph, 五行 生/克 & controller, 爻位 parity, 八卦→五行 |
| `encoder.py` | `YinYangEncoder` — obs → signed `R^6` |
| `hexagram.py` | `HexagramInference` — 4-relation R-GCN + 经卦 embeds + polarity term + learned relation gates |
| `wuxing.py` | `WuxingDynamics` — 生/克 field × relation graph, multiplicative update |
| `change.py` | `ChangeEngine` — rank-aware adaptive threshold, STE phase transition, differentiable expected-yao (no argmax), real Hamming agreement |
| `policy.py` | `TemporalPositionalPolicy` — 当位/不当位-weighted → {进 退 守 变 待} + intensity |
| `model.py` | `YiWorldModel` assembly + King Wen conversion |
| `losses.py` | multi-task loss: 本卦 CE (opt. soft) / 之卦 CE / 动爻 BCE + pairwise ranking / 行动 CE / `L_yao` Huber / 吉凶 / 五行 balance |
| `data.py` / `synth.py` / `textenc.py` | `eco` generator / `synth` generator + `SynthPool` / pluggable frozen text encoders |
| `augment.py` | LLM back-inversion augmentation: jargon-free prompt + projector consistency filter (`llm_fn` is a plug; offline `paraphrase_fallback`) |
| `train.py` / `demo.py` / `analysis.py` | training loop / one-shot visualization / 之卦 error decomposition |

## Data augmentation (`augment.py`) — opt-in, not run by default

LLM back-inversion: structure row → jargon-free prompt → LLM → situation text →
**consistency filter** (run an already-trained model, recover the 本卦 sign
pattern, drop samples that disagree with the structure they came from). This
buys *within-hexagram* diversity (many surface forms of one structure); it does
**not** buy cross-hexagram distinguishability, so it is augmentation, not
ground truth. `llm_fn` is a `str -> str` plug (`anthropic_llm_fn()` provided);
`paraphrase_fallback` is a no-LLM stand-in so the pipeline + filter are testable
offline. Prompt forbids 卦 names, 爻位 terms, and 阴阳/五行 vocabulary.

## Known limits / next

- 之卦 compounding wall (see above) — needs a joint mutation head.
- No real 爻辞 / 彖 / 象 semantics; no real "situation → 卦" labelled data. This is
  the main gap between a working prototype and "actually understands the I Ching".
  `augment.py` narrows the diversity gap but not the aliasing gap.
- 错/综/互 feed the graph conv but the `synth` target doesn't depend on them, so
  those relation gates stay near-uniform (on `eco` the 一爻之变 gate leads, ~0.41).
- `KING_WEN_TRIGRAMS` — 7 hexagram anchors + permutation property are tested;
  worth a full check against a reference table.

## License

MIT

> Built on **时位决定论** (Shi-Wei determinism): the same act yields different
> structural consequences at different times and positions.
