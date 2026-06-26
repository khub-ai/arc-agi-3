"""Splice `agent/my_agent.py` into `notebooks/submission.ipynb` — Cognitive OS variant.

Same skeleton as the official ARC-AGI-3 starter (so `submission.parquet` is still
produced by the competition gateway), with two COS-specific additions:

  * the install cell also brings in any extra COS deps from an optional
    `cos-deps` wheelhouse dataset (e.g. vLLM, if the Kaggle image lacks it);
  * a vLLM-serve cell (competition-rerun only) starts a local OpenAI-compatible
    server for the attached gemma-4-31b model BEFORE the agent runs, so COS's
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
# KAGGLE ACCELERATOR.  gemma-4-31b does NOT fit any Kaggle GPU in bf16 (~62 GB);
# attach a QUANTIZED (w4a16/AWQ/GPTQ) gemma-4-31b. The vLLM cell auto-detects the
# quant method + GPU count (tensor-parallel). "t4" = T4 x2 (32 GB total, TP=2) is
# the most headroom; "rtx6000" = 24 GB single (simpler, burns quota faster).
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "t4"

_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
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

    install_cell = code_cell(dedent(
        """\
        # ARC SDK from the offline competition wheels (provided in the comp data).
        !pip install --no-index --find-links \\
            /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\
            arc-agi python-dotenv
        # Optional: extra COS runtime deps not in the Kaggle image (e.g. vLLM),
        # shipped as a `cos-deps` wheelhouse dataset. No-op if you don't attach one.
        import glob, os
        for _d in glob.glob('/kaggle/input/cos-deps*'):
            req = os.path.join(_d, 'requirements.txt')
            cmd = f'pip install --no-index --find-links {_d} ' + (f'-r {req}' if os.path.exists(req) else 'vllm')
            print('cos-deps:', cmd); os.system(cmd)
        """
    ))

    serve_vllm_cell = code_cell(dedent(
        """\
        # Serve the attached gemma-4-31b (quantized) on a local OpenAI-compatible
        # endpoint, BEFORE the agent runs. Competition-rerun only (skipped on commit).
        import os, glob, json, subprocess, time, urllib.request
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            import torch
            ngpu = torch.cuda.device_count()
            model_dir = None
            for cfg in glob.glob('/kaggle/input/**/config.json', recursive=True):
                d = os.path.dirname(cfg)
                try:
                    if 'gemma' in d.lower() and any(f.endswith('.safetensors') for f in os.listdir(d)):
                        model_dir = d; break
                except Exception:
                    pass
            assert model_dir, 'attach a (quantized) gemma-4-31b Kaggle Model'
            quant = None
            try:
                quant = (json.load(open(os.path.join(model_dir, 'config.json')))
                         .get('quantization_config', {}).get('quant_method') or '').lower() or None
            except Exception:
                pass
            cmd = ['python', '-m', 'vllm.entrypoints.openai.api_server',
                   '--model', model_dir, '--served-model-name', 'gemma-4-31b-it',
                   '--host', '127.0.0.1', '--port', '8000',
                   '--tensor-parallel-size', str(max(1, ngpu)),
                   '--gpu-memory-utilization', '0.92', '--max-model-len', '8192',
                   '--trust-remote-code']
            if quant:
                cmd += ['--quantization', quant]
            print('serving', model_dir, '| gpus', ngpu, '| quant', quant, flush=True)
            logf = open('/kaggle/working/vllm.log', 'w')
            subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
            up = False
            for _ in range(240):          # up to ~20 min; 31B load is slow
                try:
                    urllib.request.urlopen('http://127.0.0.1:8000/v1/models', timeout=5)
                    up = True; break
                except Exception:
                    time.sleep(5)
            print('vLLM up' if up else 'vLLM FAILED -- see /kaggle/working/vllm.log', flush=True)
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
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
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
            serve_vllm_cell,
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
