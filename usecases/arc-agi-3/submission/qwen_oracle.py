"""Offline local-Qwen oracle - reuses COS's existing self-host path.

COS already has the OFFLINE model path: `backends.call_oracle` routes an
`ollama/<host:port>/<tag>` slug to a local Ollama server - no API key, no
internet, `num_gpu=99`, native image passing, response caching. We reuse it
READ-ONLY rather than re-implement inference, so the submission's model call is
identical to how COS runs Qwen locally today (e.g.
`ollama/127.0.0.1:11434/qwen3-vl:8b`).

For FULL-COS integration you may not even need this class: point COS's model
selection at the same `ollama/...` slug and its actor calls `call_oracle`
itself. This wrapper is for a lean, direct chooser (cos_agent's seam) and for
smoke tests. `dry_run=True` returns a trivial reply so the pipeline runs with no
model.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# COS backends live here; import read-only (no edits to COS).
_COS_PY = Path(__file__).resolve().parents[1] / "python"
if _COS_PY.is_dir() and str(_COS_PY) not in sys.path:
    sys.path.insert(0, str(_COS_PY))


def default_slug() -> str:
    """`ollama/<host>/<tag>` from env, else a sane local default. No network
    beyond localhost (Ollama serves on COS_OLLAMA_HOST)."""
    slug = os.environ.get("QWEN_MODEL_SLUG")
    if slug:
        return slug
    host = os.environ.get("COS_OLLAMA_HOST", "127.0.0.1:11434")
    tag = os.environ.get("QWEN_MODEL_TAG", "qwen3-vl:8b-instruct")
    return f"ollama/{host}/{tag}"


class QwenOracle:
    def __init__(self, model_slug: Optional[str] = None, *, dry_run: bool = False,
                 max_tokens: int = 1024, temperature: float = 0.0,
                 timeout_s: int = 300):
        self.model_slug = model_slug or default_slug()
        self.dry_run = dry_run or os.environ.get("ORACLE_DRY_RUN") == "1"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_s = timeout_s
        self._call_oracle = None
        if not self.dry_run:
            try:
                from backends import call_oracle      # COS's offline path, reused
                self._call_oracle = call_oracle
            except Exception as exc:                  # pragma: no cover
                raise RuntimeError(
                    f"could not import COS backends.call_oracle ({exc!r}). Put "
                    "usecases/arc-agi-3/python on PYTHONPATH, or use dry_run.")

    def complete(self, prompt: str, image_b64: Optional[str] = None, *,
                 system: str = "") -> str:
        """Return the local model's text reply. `image_b64` is a base64 PNG of
        the frame for a vision Qwen (see `encode_frame`)."""
        if self.dry_run:
            return "ACTION1"
        res = self._call_oracle(
            model=self.model_slug, system=system, user=prompt,
            image_b64=image_b64, max_tokens=self.max_tokens,
            temperature=self.temperature, timeout_s=self.timeout_s)
        if isinstance(res, dict):
            return res.get("text", "") or res.get("reply", "")
        return str(res)


def encode_frame(frame) -> Optional[str]:
    """Encode a frame (numpy HxW / HxWx3, or PIL image) as a base64 PNG for a
    vision model. Returns None if PIL/numpy aren't available or encoding fails."""
    try:
        import base64
        import io

        from PIL import Image
        img = frame
        try:
            import numpy as np
            if isinstance(frame, np.ndarray):
                arr = frame if frame.dtype.name == "uint8" else frame.astype("uint8")
                img = Image.fromarray(arr)
        except Exception:
            pass
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return None
