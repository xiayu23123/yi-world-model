# Yi-World-Model (`yiwm`)

`v0.3.0` · 52 tests · CPU-only · core deps: torch + pytest

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
pytest -q                                            # 59 passed
python -m yiwm.train --data eco   --steps 12000      # ~3 min CPU
python -m yiwm.train --data synth --steps 8000  --ckpt checkpoints/yiwm_synth.pt
python -m yiwm.demo     --ckpt checkpoints/yiwm.pt        # one deterministic inference
python -m yiwm.analysis --ckpt checkpoints/yiwm.pt        # error decomposition + relation gates
```

## Results (converged)

| dataset | 本卦 | 动爻 mask (all-6) | 之卦 (ChangeEngine) | **之卦 (joint head)** | 行动 |
|---|---|---|---|---|---|
| `eco`  (noisy Lotka-Volterra, observation aliasing) | 0.96 | 0.88 | 0.81 | **0.85** | 0.98 |
| `synth` (时位决定论 generator, idealized) | 0.998 | 0.99 | 0.96 | **0.984** | 1.00 |

**The `moving^6` compounding wall — and how the joint head lowers it.** `之卦`
needs the whole 6-bit line-change mask right. The `ChangeEngine` predicts the
6 lines *independently*, so exact-mask accuracy ≈ `(per-yao acc)^6` — on `eco`,
`0.969^6 ≈ 0.83`. Adding a **21-way joint head** (`moving_head`: predict the
whole moving set as one class of the 21 patterns with 1–2 老爻) sidesteps the
independence:

| exact moving-mask accuracy | `eco` | `synth` |
|---|---|---|
| per-yao product (`ChangeEngine`) | 0.83 | 0.89 |
| joint 21-way head | **0.885** | **0.986** |

On `synth` (clean obs) the joint head nearly eliminates the penalty — the wall
*was* the independence assumption. On `eco` it recovers ~+5pts; the residual
~11% is genuine observation aliasing (the noisy obs doesn't determine which
line moves). `hex_logits_next_joint` (之卦 via the joint mask) is the better
head and is what `analysis` now reports alongside the ChangeEngine path.

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
| `model.py` `moving_head` | 21-way joint 动爻-mask classifier + `hex_logits_next_joint` — the better 之卦 path (see the wall discussion above) |
| `policy.py` | `TemporalPositionalPolicy` — 当位/不当位-weighted → {进 退 守 变 待} + intensity |
| `structured_input.py` | 4-question rule map → yao-force; `demo --structured-input` (encoder bypassed via `yao_override`) |
| `dynamics.py` | 64-卦 rollout + 384-爻 perturbation batch → `rollout_stats.json` / `perturbation.csv` + console portrait; `--no-decay` veto test |
| `transition.py` | `LearnedTransition` MLP `f: (yao, action) → yao'` distilled from the model's one-step rule; `rollout_learned` for the smooth-map comparison |
| `model.py` | `YiWorldModel` assembly + King Wen conversion |
| `losses.py` | multi-task loss: 本卦 CE (opt. soft) / 之卦 CE / 动爻 BCE + pairwise ranking / 行动 CE / `L_yao` Huber / 吉凶 / 五行 balance |
| `data.py` / `synth.py` / `textenc.py` | `eco` generator / `synth` generator + `SynthPool` / pluggable frozen text encoders |
| `augment.py` | LLM back-inversion: jargon-free prompt, `str->str` backends (anthropic / deepseek / glm / ollama / mock / paraphrase — set the matching `*_API_KEY` env var), projector consistency filter, crash-safe `build_semantic_jsonl` / `build_from_seed` |
| `seed.py` | `build_yao_seed` — 384-line (64×6) skeleton with all structural fields derived (yao_target, moving, timing, action, 初九/六二… names); text fields left blank |
| `canonical.py` | `import_canonical` — fills `canonical_text` (**all 384 爻辞**, transcribed from ctext.org, traditional), `modern_text` (`MODERN_P0` 乾坤 + `MODERN_P1` 既濟/未濟/泰/否 = 36 rows), and `P0_ACTION` overrides where the structure-only `_action` misreads an anchor; `extra=` / `modern_extra=` to override |
| `train.py` / `demo.py` / `analysis.py` | training loop / one-shot visualization / 之卦 error decomposition |

## Data augmentation (`augment.py`) — opt-in, not run by default

LLM back-inversion: structure row → jargon-free prompt → LLM → situation text →
**consistency filter** (run an already-trained model, recover the 本卦 sign
pattern, drop samples that disagree with the structure they came from). This
buys *within-hexagram* diversity (many surface forms of one structure); it does
**not** buy cross-hexagram distinguishability, so it is augmentation, not
ground truth.

Two routes:

```bash
# A. structure-driven -- diverse surface forms of random synth structures
python -c "from yiwm.augment import build_semantic_jsonl, anthropic_llm_fn; \
  build_semantic_jsonl('data/sem.jsonl', 500, anthropic_llm_fn(), seed=0)"

# B. 爻辞-seeded. `data/yao_seed.json` ships complete: all 384 canonical_text
#    (爻辞, from ctext.org) + 36 modern_text anchors (乾坤 poles, 既濟/未濟/泰/否
#    contrast pairs) + 14 action_derived (toy _action overridden). Regenerate:
python -m yiwm.seed      data/yao_seed.json    # 384 rows; structural fields all derived
python -m yiwm.canonical data/yao_seed.json    # -> canonical_filled: 384, modern_filled: 36
# translate all 384 爻辞 -> modern situations:
python -m yiwm.augment --seed-file data/yao_seed.json --output data/sem.jsonl \
       --text-field canonical_text --n-variants 3 --llm anthropic \
       --filter-ckpt checkpoints/yiwm_synth.pt
# ...or --text-field modern_text to expand just the 36 hand-written anchors.

python -m yiwm.train --semantic-data data/sem.jsonl --steps 4000              # train the head on it
```

Route B is the one that adds *cross-hexagram* semantics (乾初九 潜龙 vs 坤初六
履霜 as opposite ends of the force space); route A only adds within-hexagram
diversity. Both writes are incremental / crash-safe.

- `llm_fn` is a `str -> str` plug: `anthropic_llm_fn()` / `deepseek_llm_fn()` /
  `glm_llm_fn()` / `ollama_llm_fn()` / `mock_llm_fn` / `paraphrase_fallback`
  (no-LLM, offline). Set the matching `*_API_KEY`. No LLM is called unless you
  pass one.
- Prompt forbids 卦 names, 爻位 terms, 阴阳/五行 vocab, and 鸡汤 words
  (努力/坚持/成功/正能量/…). Force is bucketed (有力/一般/薄弱), positions named
  (最底层/第二层/…) — nothing to leak.
- `build_semantic_jsonl` writes one JSON object per accepted line and `flush()`es,
  so a crash after *k* keeps the first *k*.
- `SemanticJsonlDataset` embeds the texts once, rebuilds entities from `ben_k`,
  and yields the standard batch dict — training on it only moves the
  `YinYangEncoder` head (obs encoder is frozen by construction).

### Result of the run (方向一, DeepSeek) — the route does not work as specified

`data/sem_all.jsonl` — 984 rows: 384 爻辞 translated ×2 (`--text-field
canonical_text`) + 36 anchors ×6. Frozen `minilm-ml` encoder, head trained on it.

| eval | benGua | moving/yao | 之卦 |
|---|---|---|---|
| synth template (hash / minilm-ml) | 0.998 / 0.996 | 0.977 / 0.966 | 0.94 / 0.90 |
| semantic prose, **train set** | 0.92 | 0.62 | 0.06 |
| semantic prose, **77 held-out structures** | **0.02 (chance)** | 0.47 | 0.01 |

The train/held-out gap is total: the head **memorizes** the ~770 training rows'
(embedding → 卦) pairs and generalizes **not at all** to unseen structures.
Causes: ~2 rows per structure (few-shot memorization), a frozen embedding that
places two different 爻 in the same domain (乾九二 vs 乾九三, both "创业") right
next to each other, and `ben_k`/`moving` labels that come from the seed
structure rather than from what DeepSeek actually wrote. So the earlier
"benGua 0.92" is meaningless — it is train-set only.

**Conclusion:** LLM back-inversion + frozen semantic encoder + small head does
not learn a generalizable situation→卦 map at this data scale. It would need
end-to-end fine-tuning of the encoder (drops the frozen constraint), 10–50×
more data, or a contrastive objective. The `moving^6` wall is untouched.
`data/sem_all.jsonl` is kept as a corpus; the training result is negative.

## Structured input (`structured_input.py`) — the honest fallback

After the free-text route failed to generalise, this is the working
"situation → 卦" path: **4 multiple-choice questions, rule-mapped, no learning
on the obs side.**

```bash
python -m yiwm.demo --structured-input     # answer 4 questions -> 卦象 + 之卦 + action
```

| question | drives |
|---|---|
| `phase` (潜/生/守/跃/显/亢) | the 老爻 line (1–6) → 之卦 |
| `polarity` (阳/阴) | global sign → 本卦 (乾 or 坤) |
| `resource` (充足/临界/紧缺) | magnitude scale |
| `domain` | 五行 colouring for the entities |

本卦 / 动爻 / 之卦 are **exact by construction**; the trained model contributes
only the *action* recommendation (its policy net weighs 当位 / 时位). Limitation:
one global `polarity` ⇒ 本卦 is always 乾 or 坤; per-line yin/yang would need
more than 4 questions. This is a deliberate trade — it keeps the world model's
core (变易 engine, 之卦, policy) usable without a working text→structure map.

## System-dynamics portrait (`dynamics.py`)

`python -m yiwm.dynamics --ckpt <ckpt>` runs `rollout` + perturbation over all
384 (卦, 爻) standard states → `rollout_stats.json` + `perturbation.csv`.

**`model.rollout(yao, steps)`** (`demo --rollout N`) iterates 本卦 → 之卦 → …
with no new observation: the 老爻 flip sign (discharged), the force vector
decays each step. Terminates at a fixed point (`|force| < the learned 老爻
threshold`) or a repeating 卦. e.g. `未濟 →(动爻)→ 未濟 → 解 [cycle]`, `|force|
0.90 → 0.63`. Over 64 卦 (synth ckpt): mean 2–5 steps to terminate, only
`鼎` is a strict self-attractor; **`乾` is among the *slowest / most cyclic*** —
its all-yang structural force keeps every line near-老, so the chain never
settles. That is emergent behaviour of the learned model, not a coded rule.

**Perturbation** — `yao += N(0, σ)`, argmax flip rate (mean over 64 卦, by 爻):

| 爻 | benGua flip @σ.05 | 之卦 flip @σ.05 | benGua @σ.20 | 之卦 @σ.20 |
|---|---|---|---|---|
| 初 | 0.07 | 0.17 | 0.38 | 0.50 |
| **三** | **0.14** | **0.24** | 0.40 | 0.52 |
| 五 | 0.11 | 0.18 | 0.39 | 0.52 |

benGua/action are noise-robust; **之卦 is ~2× as fragile at every position** —
the 老爻 sits on the adaptive threshold. **三爻 is the most fragile** (matches
易 lore: 三多凶, the awkward top-of-lower-trigram slot); 初/二爻 the most stable.
`eco` (aliased obs) is more fragile throughout (之卦 flip ~0.20–0.31 @σ.05).

`demo --structured-input` surfaces this per-input: it perturbs the force ±0.05
and, if the **action** recommendation flips >25% of the time, prints a
"高敏感区" warning (三爻 inputs trip it at ~42%). `rollout_stats.json` records
`cycle_len` / `cycle_members` for each 卦 whose rollout is periodic.

## Is the dynamics intrinsic? (`--no-decay` veto + `transition.py`)

**Veto test** — `python -m yiwm.dynamics --no-decay --steps 100` sets the
rollout decay to 1.0 (`|force|` conserved). Over all 384 start states: **100%
cycle, 0 divergence, 0 freeze**, cycle lengths 1–3 (period-1: 226, period-2:
155). Decay was *masking* the periodicity, not creating it — the iterated
`ChangeEngine + moving_head` is a bounded finite-state map with short,
structured attractor cycles (`乾 → 大壯 → 大畜 → 泰 ⇄ 大畜`).

**Learned transition** (`transition.py`) — `LearnedTransition` is an MLP
`f: (yao, action) → yao'` distilled from the model's own one-step rule (MSE
≈ 0.015). Rolling out with the *smooth* map instead of the discrete flip:
cycle-length distribution matches (`{1:276, 2:101, 3:6}` vs `{1:245, 2:132}`),
**period agreement 75%**, same attractors (乾九四 → 泰↔大畜 either way, shorter
transient). So the periodicity is a property of the learned dynamics,
reproducible by a continuous map — not an artefact of the flip. It is still a
*distillation*, not new physics; a real environment (state/action/reward) is
what would make the transition net learn something the flip rule doesn't
already contain.

## Real-data transition — **methodology validation only, no trading use**

> `market_transition.py` is a research probe: *can `LearnedTransition` ingest
> real time-series and learn a map distinct from the internal flip rule?*
> **No trading advice. No P&L backtest. Not a signal.**

`market_adapter` maps a price series to `R^6` (6 backward-looking indicators,
**no look-ahead** — a test asserts the obs is independent of the forward-return
label). `market_transition` learns `f: yao_t → yao_{t+1}` on it. AAPL 2015–2024,
held-out 2022–2024:

| check | result | reading |
|---|---|---|
| 1-step MSE vs persistence | 0.023 vs 0.027 | beats "stays put" ~17% |
| momentum direction hit | 0.865 vs 0.861 | **no edge** — model captures autocorrelation, not causation |
| gap vs `ChangeEngine` flip | **0.23** | learned a *distinct* map, not a distillation |
| bootstrap (30 steps) | bounded, drift ~0.05 | stable long-horizon, no fixed point |

Conclusion: the architecture ingests external signals and learns non-trivial
transitions — the P3 "just a distillation?" question resolves **no** on real
data. Markets are low-signal/high-noise; there is no predictive skill here.

**`--regime`** (`python -m yiwm.market_transition --regime`) reads `sign(yao)`
of the 6 indicators as a 6-bit 卦 — a *structural label* for "what the tape
looks like now" (late-2024 AAPL oscillates 履 ⇄ 中孚), no forecast, no training.

## Causal environment (`tiny_kingdom.py`)

`TinyKingdom` — a minimal env where actions genuinely change the next state
(resource / morale / threat, 5 actions with fixed causal effects + process
noise). This is what the market series lacks. `python -m yiwm.tiny_kingdom`
collects random-action trajectories, fits `LearnedTransition(yao, action)`, and
runs two checks:

| check | result | verdict |
|---|---|---|
| transition MSE | 0.0003 (27k transitions) | fits the causal map |
| action sensitivity `|f(y,进) − f(y,守)|` | **0.083** (> 0.05) | ✅ **learned action-dependent dynamics** |
| naïve MPC (imagine 3, pick max-value) vs random | −9.8 vs −5.6 | ❌ **worse than random** |

So given a real causal environment, the transition net **does** learn action
influence — the thing passive market data cannot teach.

**Planning** — `q_planner.py` closes it. The naïve MPC failed because it scored
`V(f(s,a))` with an action-blind value head (reaching a state ≠ earning its
value). `QNet(state, action)` + DQN on real env interaction (episodes are
microseconds, no imagination model needed):

| policy | mean return (20 steps) |
|---|---|
| random | −5.5 |
| naïve V-MPC (imagine 3, argmax value) | −9.8 |
| **greedy `argmax_a Q(s,a)`** | **+9.3** |

The problem was action-blind value, not the transition. `python -m yiwm.q_planner`.

## Known limits / next

- 之卦 `eco` residual ~15% is observation aliasing — the joint head took out the
  independence-assumption part; the rest needs a richer obs, not architecture.
- **Multi-step / model-based planning** — `q_planner` is a 1-step greedy Q
  policy on a toy env; deeper horizons, a learned-transition rollout as the
  Q-rollout, and a non-toy environment are the open pieces.
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
