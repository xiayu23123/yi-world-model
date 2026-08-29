---
title: "Every Dead End Named the Next Step: Building an I Ching World Model"
date: 2026-08-29
description: A research prototype that encodes the I Ching as a differentiable system. It never solved a real problem. Here is what each failure taught, and the one methodological result that survived.
---

I spent a few weeks building `yi-world-model` — a differentiable state-space
model whose scaffolding is the structural topology of the I Ching: 64 hexagrams
as vertices of a 6-dimensional signed cube, the King Wen permutation, the
错/综/互 relations, the 五行 generation/control matrix, all hard-coded as prior
knowledge. Only the continuous parts learn.

It solved zero real problems. It was never better than a baseline at anything a
baseline exists for. By the ordinary standard, it is a toy.

But the process was a clean example of research-as-a-loop: every dead end named
the next step, and one small result came out the other side intact. This is the
log.

## Why the I Ching at all

Not mysticism. A formal question: the I Ching is already a discrete dynamical
system in disguise. 64 states, a "one line changes" transition graph that is
literally the 6-cube, a notion of *pressure* on each line (老阳 / 老阴 — a line
"about to" flip), and a doctrine (时位) that the same act has different
consequences at different positions in the sequence. Can you write that down as
a differentiable model and make the pieces train?

The answer to *that* is yes, and it is not very interesting. Encoding a known
structure as a neural net is engineering, not discovery. The interesting
question was always: does the structure *buy* anything?

## Loop 1 — the boring foundation

Get the discrete facts exact. 64 six-bit hexagrams, the King Wen sequence as a
permutation (with tests anchoring 乾=63, 坤=0, 屯, 既濟, 未濟), 错 (invert all
lines), 综 (turn upside down), 互 (nuclear hexagram), the Hamming-1 change
graph, 五行 生/克 and "what controls what". None of this learns. All of it is
testable and was tested. This part is unglamorous and correct, and everything
after depends on it.

## Loop 2 — toy dynamics, and the first real "aha"

Two synthetic environments: a 5-species Lotka-Volterra ecosystem (`eco`), and a
时位-determinism generator (`synth`) that samples a hexagram, reads its
structural properties, and emits a signed 6-vector plus a moving-line mask.
Every label is a deterministic function of things the model can observe, so the
task is learnable by construction.

The model hit a wall on 之卦 (the "changed hexagram"): it needs the whole 6-bit
line-change mask right, and predicting the 6 lines *independently* caps accuracy
at `(per-yao accuracy)^6` — about `0.97^6 ≈ 0.84`. The `eco` number tracked that
almost exactly.

The fix was the first thing that felt like a finding rather than tuning:
predict the *whole moving set* as one class of the 21 patterns with 1–2 moving
lines. On the clean `synth` env this took exact-mask accuracy from 0.89 to
0.986 — the wall *was* the independence assumption. On noisy `eco` it recovered
about +5 points; the rest is genuine observation aliasing (different situations,
near-identical observations).

Lesson: a compounding ceiling is often a modelling artefact, not a hard limit.

## Loop 3 — free text → hexagram, falsified

The obvious next move: let the observation be natural language. Generate
situation text from the structure, embed it (frozen sentence-transformer), let
a small head map embedding → hexagram.

Train-set accuracy hit 0.92. Held out on unseen structures: **0.02 — chance**
(1/64). The head had memorised ~770 (embedding, label) pairs and generalised not
at all. ~2 rows per structure is few-shot memorisation; a frozen 384-d
embedding places two different lines of the *same* domain right next to each
other; and the label came from the seed structure, not from what the language
model actually wrote — so label and text drift apart at line resolution.

This killed the "situation → 卦 from prose" direction. It also produced a
concrete negative worth writing down: a frozen encoder + a small head + ~1k
examples does not learn a generalisable text→structure map, and freeform prose
does not reliably carry a 1-of-6 positional signal.

## Loop 4 — the honest retreat

If prose does not work, the working "situation → 卦" path is four
multiple-choice questions rule-mapped to a force vector. 本卦 comes out as 乾 or
坤 (one global polarity), the 之卦 is one of twelve, and the trained model only
contributes the action recommendation. Not impressive. Honest. It keeps the
change engine and policy usable without a text→structure map that does not
exist.

## Loop 5 — is it even a world model?

A single-step 本卦 → 之卦 map is not a world model. So: iterate it. With no new
observation, the moving lines flip sign (they discharged) and the force vector
decays. Over all 384 (卦, 爻) start states this converges to a fixed point or a
short cycle (period 1–3) — and removing the artificial decay does not make it
diverge, it makes the cycles *more* prominent. The periodicity is intrinsic to
the learned map, not an artefact of forced stabilisation.

The perturbation report was the nicest surprise. Add noise to the force vector,
measure how often each output flips. 本卦 and action are robust; 之卦 is
fragile — the moving-line decision sits on the adaptive threshold. And **三爻 is
the most fragile position**, which is exactly what the classical commentary says
(三多凶). That fell out of the numbers; nobody put it there.

None of this is *useful*. It is evidence that the thing has emergent structure
rather than being a lookup table.

## Loop 6 — the transition, three ways

The 变卦 rule is still hard-coded (flip the moving lines). A real world model
should *learn* the state transition. Three attempts:

1. **TinyKingdom** — a tiny causal environment (resource / morale / threat, 5
   actions with real effects). Here `f(state, action)` has something to learn.
   It does: given the same state, `f(·, 进)` and `f(·, 守)` differ meaningfully
   (sensitivity 0.083). Naïve model-predictive control on top *failed* (worse
   than random) because the value head was action-blind; a proper `Q(s, a)` +
   DQN fixed it (+71% over random). The bug was the value function, not the
   transition.

2. **Real market data** (AAPL, 2015–2024, held out). Methodology only — no
   trading use, no backtest. The transition learns a map *distinct* from the
   internal flip rule (so it is not just a distillation), but has no directional
   predictive skill. A passive price series has no causal action effect, so the
   action input is inert. Fair.

3. **Kingdom2D** — a spatial grid, CNN encoder. Q-learning scaled fine. The
   transition **collapsed**: the 32-d latent, trained end-to-end with DQN,
   saturated to a *constant vector* (per-state std = 0). "Q beats random" was a
   false positive — one fixed action beating random flailing. Even after fixing
   the saturation (LayerNorm, lower lr) and adding a heavy next-state auxiliary
   loss, the action-conditioned transition learned a sensitivity of ~0.014,
   *worse* than the hand-crafted R^6 map.

## Loop 7 — the one result that survived

The Kingdom2D failure had an obvious diagnosis: a latent shaped by a *control*
objective (Q) is policy-relevant, not dynamics-complete. It keeps whatever Q
needs and throws away the rest — including which action was taken.

So: train the encoder and transition **jointly, with no reward and no Q**, on a
JEPA-style objective — predict `enc(g_{t+1}).dyn` from `enc(g_t).dyn` and the
action, with a stop-gradient on the target — plus a small variance regulariser
so the latent cannot collapse to a point.

Action-conditioned transition sensitivity, Kingdom2D 5×5, **5 seeds**:

| latent objective | sensitivity |
|---|---|
| hand R^6 | 0.026 ± 0.002 |
| CNN + DQN | 0.013 ± 0.004 |
| **JEPA (dynamics-first)** | **0.238 ± 0.066** |
| JEPA, no variance reg | 0.001 ± 0.001 |

The separation is clean (stage-2 max 0.017 < JEPA min 0.153, zero overlap). The
variance regulariser is *causally* necessary — without it the latent collapses
to a constant on every seed, not just sometimes. And freezing that latent and
training `Q(z, a)` on top gives a policy that beats random by 69% ± 1.

This is the contribution, such as it is: **a representation learned to serve a
controller is not automatically good enough to plan with; you have to learn it
from the dynamics directly, and you have to stop it collapsing.** It is a small,
well-known-in-spirit point, but it is reproduced and ablated.

## Loop 8 — the falsifiable test, and the verdict

The honest way to ask "does the 变卦 mechanism buy anything" is to point it at a
standard task with ground truth and tuned baselines. Change-point detection:
slide a window, map it to a hexagram, call a 本卦 shift a regime change.
Benchmark against PELT, binary segmentation, CUSUM — and against a *control*
that sees the same six features but no hexagram.

| method | F1 | false positives (5 true) |
|---|---|---|
| 变卦 | 0.32 ± 0.13 | 7.6 |
| same-features CUSUM (control) | 0.23 ± 0.09 | 15.2 |
| tuned PELT | **0.69 ± 0.20** | 0.4 |

It loses to PELT by roughly 2×, and it over-fires — every noise-driven
feature-sign flip is a "变卦". It does beat the plain same-features control by
+0.09, so the hexagram quantisation does *something* beyond independent
per-feature thresholding. But nothing that competes with the right tool.

No edge. On the one test designed to be falsifiable, falsified.

## What I am left with

- **One methodological result**: dynamics-first representation learning +
  anti-collapse regularisation, reproduced across 5 seeds with an ablation.
- **A stack of clean negatives**: prose→hexagram is chance on held-out data;
  model-based planning loses to greedy Q on small environments; the 变卦
  mechanism is a worse change-point detector than a method from 1980; an
  English sentence encoder regresses on Chinese templates.
- **An internal-dynamics demonstration**: emergent short cycles, a perturbation
  fragility map that independently reproduces 三多凶.

Every environment it was tested on was authored by the same person who built
the model, so the generator and the model share assumptions. That is the
ceiling on all of it. The negatives are more valuable than the code — they are
the parts that save the next person the weeks I spent.

The meta-lesson is the one the blog is named for. The loop is not a failure
mode. Each dead end was specific enough to point at the next thing to try:
the compounding wall pointed at the joint head, the memorisation pointed at the
frozen-encoder problem, the DQN-latent collapse pointed at dynamics-first
pretraining, and the change-point loss pointed here — at stopping, and writing
it down.
