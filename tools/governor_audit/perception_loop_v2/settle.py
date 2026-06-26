"""settle.py -- settle-then-read: classify an action's animation framestack into a
SETTLED STATE CHANGE vs a TRANSIENT flicker, and expose the settled frame as the
ground truth to read.

Why: an action that animates (a selection, a toggle, a commit) settles into a NEW
persistent state.  Reading a mid-animation frame -- or treating the animation as
transient noise to filter -- yields the wrong conclusion (e.g. "nothing changed /
reverted").  The correct discipline:

  - The SETTLED state is the LAST frame of the stack (frame_stack[-1]).
  - If the last frame DIFFERS from the first, a STATE CHANGE occurred: read the
    settled frame, record the delta -- it PERSISTS.  (transient_flashes_signal_state_change)
  - If mid-frames differed but the last ~= the first, it was a TRANSIENT flicker:
    safe to filter for state, but still a signal something happened.  (ignore_transient_effects)

Game-agnostic; pure; works on 2D label grids or 3D RGB frames.
"""
from __future__ import annotations

import numpy as np


def _arr(f):
    return np.asarray(f)


def settled_frame(frame_stack):
    """The ground-truth frame to READ: the last (settled) frame of the stack."""
    if frame_stack is None or len(frame_stack) == 0:
        return None
    return _arr(frame_stack[-1])


def _changed_mask(a, b):
    a, b = _arr(a), _arr(b)
    if a.shape != b.shape:
        return None
    return (a != b) if a.ndim == 2 else np.any(a != b, axis=-1)


def classify_animation(frame_stack) -> dict:
    """Classify the framestack.

    Returns {kind, frames, settled_change_bbox, n_settled_changed, transient_only}:
      - kind == "static":       <=1 frame, or every frame identical.
      - kind == "state_change": last frame differs from first -> a PERSISTENT change.
      - kind == "transient":    mid-frames differed but last ~= first -> reverted.
    bbox is [x0,y0,x1,y1] over the SETTLED (persistent) delta (None unless state_change).
    """
    n = 0 if frame_stack is None else len(frame_stack)
    if n <= 1:
        return {"kind": "static", "frames": n, "settled_change_bbox": None,
                "n_settled_changed": 0, "transient_only": False}
    first, last = _arr(frame_stack[0]), _arr(frame_stack[-1])
    settled = _changed_mask(first, last)
    if settled is None:                                   # shape mismatch -> can't compare
        return {"kind": "static", "frames": n, "settled_change_bbox": None,
                "n_settled_changed": 0, "transient_only": False}
    n_settled = int(settled.sum())
    # did anything change mid-animation OUTSIDE the settled delta? -> transient flicker
    transient_only = False
    for g in frame_stack[1:-1]:
        m = _changed_mask(first, g)
        if m is not None and bool((m & ~settled).any()):
            transient_only = True
            break
    if n_settled > 0:
        ys, xs = np.where(settled)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
        return {"kind": "state_change", "frames": n, "settled_change_bbox": bbox,
                "n_settled_changed": n_settled, "transient_only": transient_only}
    return {"kind": "transient" if transient_only else "static", "frames": n,
            "settled_change_bbox": None, "n_settled_changed": 0,
            "transient_only": transient_only}


def classify_transition(prev_settled, curr_settled) -> dict:
    """The PERSISTENT change ACROSS an action: the previous settled frame vs this
    action's settled frame.  This catches changes that are already complete by the
    first animation frame (e.g. a switch-panel config that flips the instant you
    select a direction) -- which classify_animation(framestack) alone misses because
    its first frame already shows the new state.

    Returns the same shape as classify_animation (kind state_change|static + bbox)."""
    if prev_settled is None or curr_settled is None:
        return {"kind": "static", "frames": 2, "settled_change_bbox": None,
                "n_settled_changed": 0, "transient_only": False}
    delta = _changed_mask(prev_settled, curr_settled)
    if delta is None:
        return {"kind": "static", "frames": 2, "settled_change_bbox": None,
                "n_settled_changed": 0, "transient_only": False}
    n = int(delta.sum())
    if n == 0:
        return {"kind": "static", "frames": 2, "settled_change_bbox": None,
                "n_settled_changed": 0, "transient_only": False}
    ys, xs = np.where(delta)
    return {"kind": "state_change", "frames": 2,
            "settled_change_bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "n_settled_changed": n, "transient_only": False}


def settle_note(cls: dict) -> str:
    """A one-line note for the VLM prompt describing the settled outcome."""
    if not cls or cls.get("kind") == "static":
        return ""
    if cls.get("kind") == "state_change":
        bb = cls.get("settled_change_bbox")
        return (f"SETTLED STATE CHANGE: this action animated and settled into a NEW "
                f"persistent state in region {bb} ({cls.get('n_settled_changed')} cells "
                f"changed vs before). READ the settled state there -- it PERSISTS; do "
                f"NOT treat it as a transient to ignore.")
    return ("TRANSIENT: the action flickered mid-animation but the settled frame matches "
            "the start (no persistent change). Something happened, but state did not stick.")
