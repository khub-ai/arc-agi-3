"""Thin wrapper around the harness's VLM backends.

The harness exposes ``backends.call_oracle(model, system, user,
image_b64, ...)`` which dispatches to the right vendor (Anthropic for
``claude-*``, OpenAI for ``gpt-*``, Together for text-only Gemma,
OpenRouter for everything else including vision-capable Gemma).

This module:

* loads a frame PNG and base64-encodes it for the API,
* calls ``backends.call_oracle`` with the perception prompt,
* returns the response dict + the parsed JSON when extractable.

Defaults to ``google/gemma-4-31b-it`` per operator preference (lower
cost than Claude Sonnet, and the existing trial harness has been
validated against it).
"""

from __future__ import annotations

import base64
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

# The harness's backends module lives in usecases/arc-agi-3/python.
# Add it to sys.path on import so this module is usable without
# packaging acrobatics.
_HARNESS_PYTHON_DIR = (Path(__file__).resolve().parents[2]
                       / "usecases" / "arc-agi-3" / "python")
if str(_HARNESS_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PYTHON_DIR))


DEFAULT_MODEL = "google/gemma-4-31b-it"
DEFAULT_MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.0


@dataclass
class VLMCall:
    """Result of one VLM perception call."""
    model:           str
    raw_response:    Mapping[str, Any]
    parsed_json:     Optional[dict]
    parse_error:     Optional[str]
    cost_usd:        float
    latency_ms:      int
    input_tokens:    int
    output_tokens:   int


def _frame_to_b64(frame_path: Path) -> str:
    """Read a PNG and return its base64 string (no data: prefix)."""
    from PIL import Image
    img = Image.open(frame_path)
    # Normalise to 64x64 RGB for consistent input size, then upscale
    # 4x so glyph features are visible at the VLM's image resolution.
    img = img.convert("RGB").resize((64, 64), Image.NEAREST)
    img = img.resize((256, 256), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(text: str) -> tuple:
    """Extract the first JSON object from a (possibly fenced) response.
    Returns (parsed_dict_or_None, error_message_or_None).
    """
    if not text:
        return None, "empty response"
    # Strip ```json ... ``` fences if present.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidate = fence.group(1)
    else:
        # Fall back to first balanced {...}.
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None, "no JSON object found in response"
        candidate = text[start:end + 1]
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as e:
        return None, f"JSON decode failed: {e}"


def call_vlm(
    *,
    prompt:      str,
    frame_path:  Path,
    model:       str  = DEFAULT_MODEL,
    max_tokens:  int  = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    system:      str  = "You are a careful vision-language analyst. Return only valid JSON.",
) -> VLMCall:
    """Run one VLM call on the given prompt + frame image."""
    import backends  # type: ignore
    image_b64 = _frame_to_b64(frame_path)
    raw = backends.call_oracle(
        model       = model,
        system      = system,
        user        = prompt,
        image_b64   = image_b64,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    text = (raw.get("reply")
            or raw.get("text")
            or raw.get("content")
            or "")
    parsed, err = _extract_json(text)
    return VLMCall(
        model         = model,
        raw_response  = raw,
        parsed_json   = parsed,
        parse_error   = err,
        cost_usd      = float(raw.get("cost_usd", 0.0) or 0.0),
        latency_ms    = int(raw.get("latency_ms", 0) or 0),
        input_tokens  = int(raw.get("input_tokens", 0) or 0),
        output_tokens = int(raw.get("output_tokens", 0) or 0),
    )


if __name__ == "__main__":
    # Sanity check — does the import path work?  No real call yet.
    from PIL import Image
    test_frame = (Path(__file__).resolve().parents[2]
                  / "tests" / "perception_samples"
                  / "ls20_4of7" / "frame.png")
    if test_frame.exists():
        b64 = _frame_to_b64(test_frame)
        print(f"frame_to_b64 ok: {len(b64)} chars of base64")
    else:
        print(f"test frame not found at {test_frame}")
