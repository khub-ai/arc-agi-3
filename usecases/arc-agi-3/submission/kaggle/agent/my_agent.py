"""ARC-AGI-3 Kaggle agent = the Cognitive OS (COS), wired into the official starter.

`scripts/build_notebook.py` splices this file verbatim into the submission
notebook, and the competition rerun copies it into the framework as a template
registered as `myagent`. So at IMPORT time this file must:
  (a) put the bundled COS slice (the `cos-code` dataset) on sys.path,
  (b) point COS at the local model endpoint + a fresh, competition-clean KB,
  (c) route `vllm/...` model slugs to the local vLLM server and seed the KB, and
  (d) expose an `agents.agent.Agent` subclass NAMED `MyAgent`.

All the reasoning is COS's `CosAgent` (perception -> world model -> means-ends ->
exploration -> strategy). This file is only the contract shim + bootstrap. It is
the ONLY file you normally edit:  [edit my_agent.py] -> make play-local -> make submit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _find_cos_root() -> str:
    """Locate the COS slice: the dir that contains `cognitive_os/`.

    On Kaggle that is the attached `cos-code` dataset; locally, set COS_CODE_DIR
    to your checkout (or it falls back to a sibling cos-code/)."""
    cands = [os.environ.get("COS_CODE_DIR", ""), "/kaggle/input/cos-code"]
    inp = Path("/kaggle/input")
    if inp.is_dir():
        cands += [str(p) for p in inp.glob("cos-code*")]
    cands.append(str(Path(__file__).resolve().parents[2] / "dist" / "cos-code"))
    for c in cands:
        if c and (Path(c) / "cognitive_os").is_dir():
            return c
    raise RuntimeError(
        "COS slice not found. Attach the 'cos-code' dataset on Kaggle, or set "
        "COS_CODE_DIR to a directory containing cognitive_os/."
    )


_ROOT = _find_cos_root()
for _sub in (
    "",                                          # the cognitive_os package
    "tools/governor_audit/perception_loop_v2",   # exploratory driver, perception
    "usecases/arc-agi-3/python",                 # backends, dsl, dsl_executor
    "usecases/arc-agi-3/submission",             # cos_agent, vllm_backend, kaggle_entry
):
    _p = str(Path(_ROOT, _sub)) if _sub else _ROOT
    if _p not in sys.path:
        sys.path.insert(0, _p)

# --- COS runtime config: strict, offline-safe, competition-clean KB ----------
os.environ.setdefault("COS_STRICT", "1")
os.environ.setdefault("COS_KB_ROOT", "/kaggle/working/cos_kb")
os.environ.setdefault("COS_SESSION_DIR", "/kaggle/working/cos_session")
os.environ.setdefault("COS_WORKDIR", "/kaggle/working/cos_run")
# Model endpoint: a local OpenAI-compatible vLLM server (started by the notebook's
# vLLM cell). Swappable via QWEN_MODEL_SLUG -- vllm/<host>/<name> or ollama/<host>/<tag>.
os.environ.setdefault("QWEN_MODEL_SLUG", "vllm/127.0.0.1:8000/gemma-4-31b-it")

# Route vllm/... slugs to the local endpoint, then seed the general (Tier-0) KB.
import vllm_backend            # noqa: E402
vllm_backend.install()
import kaggle_entry            # noqa: E402
kaggle_entry.seed_general_kb()

# --- the contract class the framework's agent registry expects ---------------
from cos_agent import CosAgent     # noqa: E402


class MyAgent(CosAgent):
    """The Cognitive OS agent, under the name the framework registry expects.

    `CosAgent` already reads `arc_env.info.baseline_actions` + `arc_env.action_space`
    and implements `is_done` / `choose_action` by delegating to `CosPlayer`."""
    pass
