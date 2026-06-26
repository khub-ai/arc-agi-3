# ARC-AGI-3 Kaggle submission — authoritative mechanism + plan

This is the single source of truth for how we submit. It supersedes the earlier
hand-rolled notebook (`kaggle_notebook_skeleton.py`, removed), which assumed we
iterate games offline ourselves — that is **not** how Kaggle scores ARC-AGI-3.

Verified against the competition data download and the official
`arcprize/ARC-AGI-3-Kaggle-Starter` + `docs.arcprize.org/methodology` (2026-06-26).

## How scoring actually works (the gateway sidecar)

ARC-AGI-3 is a Kaggle **Code Competition**. You submit a notebook; Kaggle re-runs
it for scoring. The key facts:

- During the competition re-run, the env var `KAGGLE_IS_COMPETITION_RERUN` is set
  and a **gateway sidecar** serves the *hidden* eval games at `http://gateway:8001`.
  Internet-off blocks only the *public* internet — the gateway is local.
- The notebook runs the framework's **`python main.py --agent myagent`** against
  that gateway in **online** mode. **The gateway records every action and emits
  `submission.parquet`** — we do not write predictions ourselves.
- On commit / save-and-run-all (no rerun var), the notebook writes a *dummy*
  `submission.parquet` so the commit succeeds. After committing you manually pick
  `submission.parquet` from the kernel output and "Submit to Competition".
- Limits: **5 submissions/day**, **12-hour** runtime, GPU available, internet off.

## Scoring metric — RHAE (favors COS)

- `level_score = (human_baseline_actions / ai_actions) ^ 2`, capped at 1.15.
- `game_score` = weighted average of per-level scores, weight = the 1-indexed
  level number (later levels matter more). Incomplete games are capped:
  4/5 levels → 66.7% ceiling. 100% requires finishing the last level.
- Final = average of all game scores.
- **Only environment-affecting actions count.** Internal reasoning, tool calls,
  retries are free — so COS's heavy per-turn deliberation costs nothing.

## What the competition provides (so we don't ship it)

In `/kaggle/input/competitions/arc-prize-2026-arc-agi-3/`:
- `arc_agi_3_wheels/` — the SDK wheels (`arc-agi`, `arcengine` + deps, cp312).
  → **no `cos-wheels` dataset needed.**
- `ARC-AGI-3-Agents/` — the framework. → we don't ship the framework either.
- `environment_files/` — games for *local* dev only (the rerun uses the gateway).

## Our integration — the COS starter (`kaggle/`)

We mirror the official starter, COS-fied. The agent is the only thing that differs.

| File | Role |
|---|---|
| `kaggle/agent/my_agent.py` | `MyAgent(CosAgent)` — puts the `cos-code` slice on sys.path, routes `vllm/...` slugs, seeds the general KB, exposes the contract class. The only file with our logic. |
| `kaggle/scripts/build_notebook.py` | Generates `submission.ipynb`: install SDK from comp wheels → **serve gemma-4-31b via vLLM** (rerun-only) → write agent → run `main.py --agent myagent` against the gateway → dummy submission on commit. |
| `kaggle/notebooks/kernel-metadata.json` | Kernel config: GPU on, internet off, inputs = competition + `cos-code` dataset + the gemma Model. |
| `kaggle/scripts/play_local.py`, `slim_framework.py`, `Makefile` | Reused verbatim from the official starter. |

`MyAgent` is just `CosAgent` (already an `agents.agent.Agent` with the right
constructor + `is_done`/`choose_action`). No other engine change is needed.

## The model — must be a QUANTIZED gemma-4-31b

gemma-4-31b in bf16 is ~62 GB and **fits no Kaggle GPU** (T4×2=32, P100=16,
RTX6000=24 GB). Attach a **quantized (w4a16 / AWQ / GPTQ)** gemma-4-31b as a Kaggle
Model. The vLLM cell auto-detects the quant method + GPU count (tensor-parallel).
The model slug stays `vllm/127.0.0.1:8000/gemma-4-31b-it`.

## Run / submit

Local COS smoke test (engine env, not the starter venv):
`python run_framework_offline.py <game>` — drives `CosAgent` through `Agent.main()`.

Build + submit (from a checkout of the public repo, with the Kaggle CLI token):
```
cd kaggle
# edit notebooks/kernel-metadata.json: set your username + the gemma model slug
make submit          # build_notebook + kaggle kernels push
make status          # watch the run
# then on kaggle.com: pick submission.parquet → Submit to Competition
```

## Open gates (must be green or the entry scores 0)

1. **Cold solve** — engine track (separate). RHAE needs levels actually completed.
2. **vLLM in the Kaggle image** — verify `import vllm` works offline on the GPU
   image; if not, ship a `cos-deps` wheelhouse dataset (the install cell picks it up).
3. **Quantized gemma-4-31b fits + serves** on the chosen accelerator (T4×2 / RTX6000).
4. **Python 3.12** — confirmed by the framework's `.python-version`; verify the
   Kaggle image matches (the SDK is 3.12-only).
