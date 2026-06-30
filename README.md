# arc-agi-3 — KHUB.AI · ARC Prize 2026 submission

KHUB.AI's entry for [ARC Prize 2026 — ARC‑AGI‑3](https://arcprize.org/competitions/2026/arc-agi-3):
a self‑contained, **offline** agent that plays the benchmark's interactive games
by running the **Cognitive OS (COS)** — a general, VLM‑in‑the‑loop cognitive
architecture — rather than any game‑specific code.

This repository is the **open‑sourced solution** (MIT‑0). It is a curated,
automatically‑synced slice of the COS engine (developed in a separate repository)
containing exactly the code the agent runs — nothing game‑specific, and no
per‑game answers.

## Idea

ARC‑AGI‑3 presents **unseen** interactive games and measures how efficiently an
agent explores, models each game's rules, and acts toward the win condition. COS
approaches every game the way a general problem‑solver would:

- **Perception** turns each frame into structural facts (components, entities,
  relations) — the vision‑language model reports *what it sees*; the substrate
  *measures* it.
- **World model** accumulates claims about the game's mechanics from what
  actually happens when the agent acts.
- **Means‑ends + exploration** pick actions: pursue the win condition when a path
  is known, otherwise explore to reduce uncertainty.
- **Strategy** (the VLM) arbitrates at the hard steps.

No code branches on a game identifier; new capability is added as a
generalization of the substrate, never a per‑game rule.

## How the submission runs

The competition harness calls an agent once per turn, but COS owns its own play
loop — so the two are bridged by control inversion:

```
ARC-AGI-3 harness ──(per-turn choose_action)──▶ CosAgent
                                                  │   control-inversion bridge
                                                  ▼
                  ExploratoryDriver  (perception · world model · means-ends · exploration · strategy)
                                                  │   file-handoff prompts
                                                  ▼
                  cos_responder ──▶ local model (Qwen3-VL-32B via transformers) — offline
```

Everything runs **offline**: the model is served locally (no internet), and the
agent starts each game with a **competition‑clean knowledge base** — general,
game‑agnostic priors only (`kb_seed/`), no per‑game memory. Within a run COS
still learns (demonstration‑from‑preview, instincts, the claim/pursuit loop); it
simply begins every unseen game cold, as the competition requires.

## Layout

| Path | What |
|---|---|
| `cognitive_os/` | COS engine core — world model, planning, knowledge index |
| `tools/governor_audit/perception_loop_v2/` | the exploratory driver, perception substrate, recall, instincts |
| `usecases/arc-agi-3/python/` | the ARC SDK bridge (`backends`, `dsl`, `dsl_executor`) |
| `usecases/arc-agi-3/submission/` | the competition wrapper — `cos_agent`, the bridge, `cos_responder`, the model backends, `kaggle_entry`, `kb_seed/` (seeded general knowledge), and `kaggle/` (the submission starter) |
| `ARC-AGI-3-Agents/` | the ARC‑AGI‑3 agent framework the submission plugs into |

## Running it

### The competition path (Kaggle, offline)

The submission is built by the starter in
`usecases/arc-agi-3/submission/kaggle/`, aligned to the official
[`ARC-AGI-3-Kaggle-Starter`](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter).
The agent is `MyAgent(CosAgent)` in `kaggle/agent/my_agent.py`;
`scripts/build_notebook.py` generates the Kaggle notebook. During Kaggle's
competition re‑run that notebook:

1. installs the SDK from the competition's offline wheels;
2. serves `Qwen3-VL-32B-Instruct` locally in bf16 via a small `transformers`
   OpenAI‑compatible shim (`serve_vlm.py`) — vLLM isn't on the Kaggle image, and
   the model fits a single RTX PRO 6000 (Blackwell, 96 GB) without quantization;
3. seeds a fresh, general‑only KB and puts the COS slice on `sys.path`;
4. runs the framework's `main.py --agent myagent` against the competition's
   `gateway:8001` sidecar, which serves the hidden games and records the play
   into `submission.parquet`.

The model is selected by a slug, so it is swappable — `vllm/<host>/<model>` routes
to the local OpenAI endpoint (`serve_vlm.py` or a real vLLM), or `ollama/<host>/<tag>`
for Ollama. See
[`KAGGLE_SUBMISSION.md`](usecases/arc-agi-3/submission/KAGGLE_SUBMISSION.md) for
the full mechanism.

### Local development

```
python usecases/arc-agi-3/submission/run_framework_offline.py <game-id>
```

plays one game through the real framework loop offline (fresh, seeded KB by
default; add `--warm-kb` to reuse accumulated knowledge for a cold‑vs‑warm
comparison).

## ARC Prize 2026 — Milestone #1 (June 30) reproduction

The Milestone #1 submission is the commit tagged **`v0.1-milestone1`** (immutable):

```
git checkout v0.1-milestone1
```

- **Model:** Qwen3‑VL‑32B‑Instruct (Kaggle Models `qwen-lm/qwen-3-vl`, variation
  `transformers/32b-instruct`), served locally and offline by `serve_vlm.py` in
  **bf16, no quantization**.
- **Accelerator:** a single **RTX PRO 6000** (Blackwell, 96 GB), pinned in the
  kernel metadata via `machine_shape: "NvidiaRtxPro6000"`.

Reproduce via the competition path above: build the notebook from
`usecases/arc-agi-3/submission/kaggle/`, attach the competition data, this code,
and the Qwen3‑VL‑32B‑Instruct model, then **Save & Run All**. Kaggle's re‑run plays
the hidden games against the `gateway:8001` sidecar, which records every action and
emits `submission.parquet` (the agent does not write predictions itself). Full
mechanism + scoring (RHAE):
[`KAGGLE_SUBMISSION.md`](usecases/arc-agi-3/submission/KAGGLE_SUBMISSION.md).

## Knowledge & compliance

The agent ships only **general, game‑agnostic** knowledge (`kb_seed/`), verified
by a compliance gate that rejects any per‑game entry an unseen game could reach.
The repository is produced from the engine by a sync tool that keeps only the
shipping code, scrubs environment‑specific paths, and blocks publication on any
residual — so what's here is exactly, and only, the runnable solution.

## License

[MIT No Attribution (MIT‑0)](LICENSE) — the permissive, public‑domain‑equivalent
licensing the ARC Prize rules require.
