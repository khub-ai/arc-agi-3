# ARC Prize 2026 — ARC-AGI-3 — Solution Writeup

**Team:** KHUB.AI  ·  **Code:** <https://github.com/khub-ai/arc-agi-3> (MIT-0)  ·
**Approach:** Cognitive OS (COS) — a general, VLM-in-the-loop cognitive architecture.

> **Milestone #1 (June 30, 2026).** The result below is the scored Milestone #1
> submission; everything else reflects the shipped code at tag `v0.1-milestone1`.

## Summary

We play unseen ARC-AGI-3 games with **Cognitive OS (COS)** — a single,
game-agnostic agent that approaches each game the way a general problem-solver
would: it *perceives* the frame into structural facts, *builds a world model* of
the game's rules from what its actions actually do, and *acts* by means-ends
planning toward the win condition, exploring to reduce uncertainty when no path
is known. A local, frozen, open-weights **Qwen3-VL-32B-Instruct** serves as the strategy
vision-language model. No game-specific code and no per-game memory are used.

**Result (Milestone #1):** **0.00** (2026-06-30, notebook version 2). The scored
rerun reported Kaggle status **`Succeeded`** and produced a valid submission, but
credited **0 completed levels**. The rerun's short runtime (~30 minutes) is less
than a full 25-game, model-serving run would take, so we are first verifying whether
the vision model actually loaded in the scored environment (a GPU / serve question)
before attributing the 0 to the agent's ability. If the model served, the relevant
gap is cold-solving an unseen game within a single run (see "What worked, what
didn't"); if it did not, the 0 reflects a serve failure, not the agent. Diagnosis
in progress.

## Approach

COS runs one cognitive loop, identically, on every game:

- **Perception** turns each frame into structural facts — components, entities,
  relations. The substrate *measures* geometry; the VLM *reports* what it sees.
  The two are kept separate so the small model is never asked to do pixel math.
- **World model** accumulates *claims* about the game's mechanics from observed
  cause and effect (what changed when the agent acted), with credences.
- **Means-ends + exploration** choose the next action: pursue the win condition
  when a path is known; otherwise probe to reduce the highest-value uncertainty.
- **Strategy** — the VLM — arbitrates the hard steps and proposes hypotheses.

Two principles are load-bearing:

1. **Game-agnostic.** No branch keys on a game identifier. New capability is a
   generalization of the substrate (a new claim type, a planner heuristic, an
   instinct), never a per-game rule.
2. **Cold, competition-clean knowledge.** The agent starts every game from a
   fresh KB seeded with **general, Tier-0 priors only** (no per-game answers),
   enforced by a compliance gate. It still *learns within a run*
   (demonstration-from-preview, the claim/pursuit loop) — it just begins each
   unseen game cold, as the competition requires.

Internally, the competition's `Agent` (`MyAgent`) is a thin shim over COS: a
control-inversion bridge lets COS own its play loop while satisfying the
per-turn `choose_action` contract —
`MyAgent → CosAgent → ExploratoryDriver → cos_responder → local VLM server`.

## How it runs on Kaggle (and is scored)

This is an offline Code Competition. During the scored re-run, a Kaggle
**gateway sidecar** serves the hidden games at `gateway:8001`; our notebook runs
the framework's `main.py --agent myagent` against it, and the gateway records
every action and emits `submission.parquet`. Internet-off blocks only the public
internet — the model is served locally.

Scoring is **RHAE** (Relative Human Action Efficiency): per level,
`(human_actions / ai_actions)²` (capped at 1.15), averaged per game weighted by
the 1-indexed level number, then averaged across games. Only
**environment-affecting actions** count — internal reasoning, tool calls, and
retries are free, which suits COS's deliberate per-turn reasoning.

## Model and compute environment

- **Model:** **Qwen3-VL-32B-Instruct** (frozen, open-weights, Apache-2.0), served
  locally in **bf16** (`device_map="auto"`) behind a small `transformers`
  OpenAI-compatible shim. vLLM is not on the Kaggle image, and the model fits a
  single RTX PRO 6000 (Blackwell, 96 GB; ~64 GB used) **without quantization**.
  Swappable via a slug.
- **No training or fine-tuning is performed.** The "training code" deliverable is
  therefore N/A; the deliverable is the inference/agent code in this repository.
  All adaptation is *within-run* learning, not weight updates.
- **Hardware / limits:** Kaggle GPU accelerator (a single RTX PRO 6000, Blackwell,
  96 GB), 12-hour runtime, internet off.
- **Dependencies:** the ARC SDK (`arc-agi` / `arcengine`) and `flask` install from
  the competition's offline wheels; `torch` / `transformers` / `accelerate` are
  already on the Kaggle image; the agent code is the attached `cos-code` dataset.

## Reproduction

The Milestone #1 submission is the commit tagged **`v0.1-milestone1`**. The full
mechanism is documented in
[`KAGGLE_SUBMISSION.md`](usecases/arc-agi-3/submission/KAGGLE_SUBMISSION.md);
the build/submit scaffold is in
[`kaggle/`](usecases/arc-agi-3/submission/kaggle/).

1. Build the agent slice: `python usecases/arc-agi-3/submission/build_package.py`
   → `dist/cos-code` (passes a compliance + leak gate), and upload it as the
   private Kaggle Dataset `cos-code`.
2. Attach the **Qwen3-VL-32B-Instruct** Kaggle Model (`qwen-lm/qwen-3-vl`,
   `transformers/32b-instruct`).
3. In `kaggle/notebooks/kernel-metadata.json` set your username and the model
   slug; put your Kaggle token at `kaggle/.kaggle/access_token`.
4. `cd kaggle && make submit` — generates `submission.ipynb` (install SDK from
   the competition wheels → serve Qwen3-VL via the transformers shim → run COS
   against the gateway) and pushes it.
5. On Kaggle, select `submission.parquet` from the kernel output → **Submit to
   Competition** (5/day, 12-hour runtime).

A local smoke test (no Kaggle round-trip) runs the same agent through the real
framework loop: `python usecases/arc-agi-3/submission/run_framework_offline.py <game>`.

## What worked, what didn't

- **Worked:** separating a measuring *perception substrate* from the *reporting*
  VLM (the small model stays in its lane); encoding strategy as game-agnostic
  *instincts* and *claims* rather than per-game rules; RHAE letting COS reason
  freely without action-budget penalty.
- **Open limitation (stated honestly):** closing a game *cold* — within-run
  learning alone, from a clean KB — is the active gap; it is where the remaining
  capability work is focused.

## License

[MIT No Attribution (MIT-0)](LICENSE) — the permissive licensing the ARC Prize
rules require. All shipped code is in this repository; it is exactly, and only,
the runnable solution (produced from the engine by a gated, scrubbed sync).
