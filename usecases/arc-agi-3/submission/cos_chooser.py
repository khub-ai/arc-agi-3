"""Lean read-only COS chooser: one frame -> one action via the local oracle.

Strategy (b) from the README: bridge the harness's per-turn `choose_action` to a
single-shot model decision, reusing COS's **palette-correct** frame rendering
(`vlm_driven_play.frame_to_png_b64`) read-only.

This is deliberately NOT the full COS reasoning stack (CV tools / world model /
MEA). That stack is a stateful, multi-step, tool-calling loop that OWNS the env;
reusing it per-turn from the competition harness needs a coordinated
stateless-per-turn COS seam (option a) - a COS change. This lean chooser is the
no-COS-change first cut: a working offline agent driven by the local Qwen,
reusing COS conventions. Upgrade path to (a) is a drop-in replacement of
`LeanCosChooser.choose`.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Optional

# COS python tree (frame_to_png_b64) - read-only.
_COS_PY = Path(__file__).resolve().parents[1] / "python"
if _COS_PY.is_dir() and str(_COS_PY) not in sys.path:
    sys.path.insert(0, str(_COS_PY))

_ACTION_RE = re.compile(r"ACTION\s*([1-7])", re.I)
_RESET_RE = re.compile(r"\bRESET\b", re.I)
_XY_RE = re.compile(r"(-?\d+)\s*[, ]\s*(-?\d+)")

_SYS = ("You are playing an ARC-AGI-3 game on a 64x64 grid (cells are palette "
        "ids; identify entities by FUNCTION - agent/target/wall/key/hazard - "
        "not colour). No instructions are given; discover the mechanics by "
        "playing, and aim to make visible progress each turn.")


def _encode(grid) -> Optional[str]:
    """Palette-correct PNG of the frame grid (reuse COS), else a generic PNG,
    else None."""
    try:
        import numpy as np
        arr = np.array(grid)
        if arr.ndim == 3:                 # frame stack -> last frame
            arr = arr[-1]
        if arr.ndim != 2:
            return None
        try:
            from vlm_driven_play import frame_to_png_b64   # palette-aware
            return frame_to_png_b64(arr.astype("int64"))
        except Exception:
            from qwen_oracle import encode_frame
            return encode_frame(arr.astype("uint8"))
    except Exception:
        return None


def _action_names(latest_frame) -> list:
    out = []
    for a in (getattr(latest_frame, "available_actions", None) or []):
        name = a.name if hasattr(a, "name") else str(a)
        name = name.upper()
        if name.isdigit():
            name = f"ACTION{name}"
        out.append(name)
    # Drop RESET from the menu (the agent handles reset separately).
    out = [n for n in out if n != "RESET"]
    return out or ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION6"]


def _build_prompt(names) -> str:
    return (f"Available actions THIS turn: {', '.join(names)}.\n"
            "ACTION1-4 are usually directional moves; ACTION5/ACTION7 "
            "interact/undo; ACTION6 is a MOUSE CLICK and needs x,y in 0..63.\n"
            "Pick the single best next action to make progress. Reply with ONLY "
            "a JSON object: {\"action_id\": \"ACTION3\"} or, for a click, "
            "{\"action_id\": \"ACTION6\", \"x\": 31, \"y\": 20}.")


def parse_action(reply: str, names: Optional[list] = None):
    """Parse a model reply into an action: a name ('ACTION3'/'RESET') or a
    click tuple ('ACTION6', {'x':.., 'y':..}). Robust to JSON or bare text;
    falls back to the first available action."""
    names = names or []
    text = reply or ""
    # 1) JSON object
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            aid = str(obj.get("action_id") or obj.get("action") or "").upper()
            if "6" in aid and ("x" in obj or "y" in obj):
                return ("ACTION6", {"x": int(obj.get("x", 32)),
                                    "y": int(obj.get("y", 32))})
            am = _ACTION_RE.search(aid)
            if am:
                return f"ACTION{am.group(1)}"
            if _RESET_RE.search(aid):
                return "RESET"
        except Exception:
            pass
    # 2) bare token in free text
    am = _ACTION_RE.search(text)
    if am:
        name = f"ACTION{am.group(1)}"
        if name == "ACTION6":
            mm = _XY_RE.search(text)
            return ("ACTION6", {"x": int(mm.group(1)), "y": int(mm.group(2))}
                    if mm else {"x": 32, "y": 32})
        return name
    if _RESET_RE.search(text):
        return "RESET"
    # 3) fallback
    return names[0] if names else "ACTION1"


class LeanCosChooser:
    """One model call -> one action. Stateless across turns (carry game state on
    the agent if needed). Swap this class for the full COS stack (option a)."""

    def choose(self, frames, latest_frame, oracle):
        names = _action_names(latest_frame)
        image_b64 = _encode(getattr(latest_frame, "frame", None))
        reply = oracle.complete(_build_prompt(names), image_b64=image_b64,
                                system=_SYS)
        return parse_action(reply, names)
