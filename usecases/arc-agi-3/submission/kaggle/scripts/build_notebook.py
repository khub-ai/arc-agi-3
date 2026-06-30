"""Splice `agent/my_agent.py` into `notebooks/submission.ipynb` — Cognitive OS variant.

Same skeleton as the official ARC-AGI-3 starter (so `submission.parquet` is still
produced by the competition gateway), with two COS-specific additions:

  * the install cell also brings in any extra COS deps from an optional
    `cos-deps` wheelhouse dataset (e.g. vLLM, if the Kaggle image lacks it);
  * a vLLM-serve cell (competition-rerun only) starts a local OpenAI-compatible
    server for the attached Qwen3-VL-8B model BEFORE the agent runs, so COS's
    `vllm/127.0.0.1:8000/...` slug resolves.

The agent itself (`agent/my_agent.py` -> `MyAgent(CosAgent)`) self-bootstraps the
COS slice onto sys.path and seeds the KB at import, so no other cell needs to.

You don't normally call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# KAGGLE ACCELERATOR.  *** Qwen3-VL-32B needs ~64 GB bf16 -> the 96 GB RTX PRO 6000
# (Blackwell). *** The GPU TYPE is set in kernel-metadata.json via
#   "machine_shape": "NvidiaRtxPro6000"
# (proven: a notebook that ran the 32B on it had exactly that; its .ipynb accelerator
# read "none", so the .ipynb field below is informational -- kernel-metadata's
# machine_shape is what the kaggle CLI push applies). VERIFY after pushing the kernel
# (pull it back / the COS_MAX_GAMES=1 smoke run) that the scored rerun got the 96 GB GPU.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "t4"   # .ipynb field only; the real GPU = kernel-metadata machine_shape (above)

_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},   # OLD 24 GB RTX 6000 -- NOT the Blackwell
    # "rtxpro6000": {"name": "<EXACT STRING>", "gpu": True},  # 96 GB Blackwell -- fill from a notebook's metadata
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
SERVE_SRC = ROOT / "serve_vlm.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {"cell_type": "code", "metadata": {"trusted": True},
            "outputs": [], "execution_count": None, "source": source}


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()
    serve_body = SERVE_SRC.read_text()

    install_cell = code_cell(dedent(
        """\
        # ARC SDK from the offline competition wheels (provided in the comp data).
        # This also brings in flask (used by the local VLM server). torch /
        # transformers / accelerate / pillow are already on the Kaggle GPU image.
        !pip install --no-index --find-links \\
            /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
            arc-agi python-dotenv
        # Optional: any extra deps NOT already on the image, shipped as a
        # `cos-deps` wheelhouse dataset with a requirements.txt. No-op normally.
        import glob, os
        for _d in glob.glob('/kaggle/input/cos-deps*'):
            req = os.path.join(_d, 'requirements.txt')
            if os.path.exists(req):
                print('cos-deps:', _d)
                os.system(f'pip install --no-index --find-links {_d} -r {req}')
        """
    ))

    write_serve_cell = code_cell("%%writefile /tmp/serve_vlm.py\n" + serve_body)

    serve_cell = code_cell(dedent(
        """\
        # Serve the attached VLM (Qwen3-VL-32B) on a local OpenAI-compatible endpoint
        # via transformers (fp16, device_map=auto across the GPUs) -- vLLM is not on
        # the Kaggle image. Competition-rerun only (skipped on commit).
        import os, glob, subprocess, time, urllib.request
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            model_dir = None
            for cfg in glob.glob('/kaggle/input/**/config.json', recursive=True):
                d = os.path.dirname(cfg)
                try:
                    if any(f.endswith('.safetensors') for f in os.listdir(d)):
                        model_dir = d; break
                except Exception:
                    pass
            assert model_dir, 'attach the Qwen3-VL-32B-Instruct Kaggle Model'
            print('serving model:', model_dir, flush=True)
            logf = open('/kaggle/working/serve_vlm.log', 'w')
            subprocess.Popen(['python', '/tmp/serve_vlm.py', '--model', model_dir,
                              '--port', '8000', '--name', 'qwen3-vl-32b'],
                             stdout=logf, stderr=subprocess.STDOUT)
            up = False
            for _ in range(180):          # up to ~15 min for the model to load
                try:
                    urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)
                    up = True; break
                except Exception:
                    time.sleep(5)
            print('VLM server up' if up
                  else 'VLM FAILED -- see /kaggle/working/serve_vlm.log', flush=True)
        """
    ))

    # Written to /tmp (not /kaggle/working) so it isn't offered as a submission file.
    write_agent_cell = code_cell("%%writefile /tmp/my_agent.py\n" + agent_body)

    run_cell = code_cell(dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar (serves the hidden eval games).
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agent in as a framework template.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py

            # Register MyAgent (rewrite __init__ to avoid eager template imports).
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': MyAgent,
        }

        # Session governor: swap the stock parallel-all-games Swarm for the
        # sequential, time-budgeted, crash-tolerant one (always writes a submission).
        import sys as _sys, glob as _glob
        for _d in _glob.glob('/kaggle/input/**/usecases/arc-agi-3/submission', recursive=True):
            if _d not in _sys.path:
                _sys.path.insert(0, _d)
        try:
            from session_governor import GovernedSwarm as _GS
            if _GS is not None:
                Swarm = _GS
                print('[governor] using GovernedSwarm (sequential, budgeted)')
        except Exception as _e:
            print('[governor] stock Swarm (governor unavailable):', _e)
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            # COS_SESSION_BUDGET_S: total wall-clock for ALL games (8 h, under the
            # scored-session limit); COS_SESSION_RESERVE_S: held back to flush the
            # submission. The governor splits the rest breadth-first across games
            # (per-game time deadline via COS_GAME_DEADLINE_EPOCH).
            # COS_GAME_TRIAGE=1: Stream A's driver-side quality abort -- the agreed
            # mid-game abort path; we don't add a second one. COS_TRIAGE_MAX_TURNS=400
            # lifts game_triage's blunt 80-turn cap (which would cut a FAST, PROGRESSING
            # game) so the active cuts are TIME (governor) + STALL/CLAIM (quality).
            # COS_ACTION_SWEEP_MAX=3: cap the early global-action sweep (it ate ~5 turns
            # on slow ka59) so a click-primary game isn't starved under a tight budget.
            # COS_PERC_MAX_VLM_CALLS=3 + COS_PERC_IDENTIFY_K=1: Stream A's recommended TIGHT
            # perception profile for the time-budgeted run -- caps per-turn VLM calls at 3
            # (identify x1 + escalation/review backstop), ~halving per-turn VLM time vs the
            # 4-call default; validated to HOLD ka59 grounding (90 / core_recall 1.0). The
            # over-seg noise-gate + edge-speck filter ride on their defaults (additive).
            # SMOKE / rerun-GPU check: prepend `COS_MAX_GAMES=1 COS_SESSION_BUDGET_S=1800`
            # to the env below to play ONE game -- cheaply confirms the rerun GPU fits the
            # model AND the gateway emits a parquet, before committing to the full 25-game run.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                COS_SESSION_BUDGET_S=28800 COS_SESSION_RESERVE_S=900 \\
                COS_GAME_TRIAGE=1 COS_TRIAGE_MAX_TURNS=400 COS_ACTION_SWEEP_MAX=3 \\
                COS_PERC_MAX_VLM_CALLS=3 COS_PERC_IDENTIFY_K=1 \\
                python main.py --agent myagent
        """
    ))

    dummy_submission_cell = code_cell(dedent(
        """\
        import os
        if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Commit / save-and-run-all mode: emit a dummy submission so the commit
            # succeeds. The real submission.parquet is produced by the gateway.
            import pandas as pd
            submission = pd.DataFrame(
                data=[['1_0', '1', True, 1]],
                columns=['row_id', 'game_id', 'end_of_game', 'score'])
            submission.to_parquet('/kaggle/working/submission.parquet', index=False)
            submission.head()
        """
    ))

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick: {sorted(_ACCELERATORS)}")
    accel = _ACCELERATORS[ACCELERATOR]

    return {
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python", "mimetype": "text/x-python",
                              "file_extension": ".py", "pygments_lexer": "ipython3"},
            "kaggle": {"accelerator": accel["name"], "isInternetEnabled": False,
                       "isGpuEnabled": accel["gpu"], "language": "python",
                       "sourceType": "notebook"},
        },
        "nbformat_minor": 4, "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 — Cognitive OS\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."),
            install_cell,
            write_serve_cell,
            serve_cell,
            write_agent_cell,
            run_cell,
            dummy_submission_cell,
        ],
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  (accelerator: {ACCELERATOR})")
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced enable_gpu={wanted} in {METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
