"""Per-step controller for the structure-map / apply-legend pursuit -- the means-ends decision
that turns a computed target pattern into the NEXT step.  GAME-AGNOSTIC, pure + testable.

Given the per-slot TARGET keys (from target_pattern.compute_targets + glyph_read over the legend
& input), the per-slot CURRENT keys (glyph_read over the editable cells), and which slot the
cursor is on, decide the next move: get the cursor to the leftmost unsatisfied slot, then EDIT
that slot toward its target.  The driver maps the decision's `kind` to the live action (the edit
mechanic learned by edit_law / the action-effectiveness sweep -- cursor-move vs cycle).
"""
from __future__ import annotations

from typing import List, Optional, Dict


def next_step(targets: List[Optional[int]], current_keys: List[Optional[int]],
              cursor_idx: Optional[int]) -> Dict:
    """Decide the next step toward the target pattern.

    Returns {kind, slot, ...}:
      - kind 'solved'      : every slot with a known target already matches it.
      - kind 'move_cursor' : the cursor must move to `slot` (the leftmost unsatisfied slot).
      - kind 'edit'        : the cursor is ON `slot`; cycle its glyph from `current` to `target`.
      - kind 'blocked'     : no actionable slot (targets unknown).
    """
    unsatisfied = [i for i, t in enumerate(targets)
                   if t is not None and (i >= len(current_keys) or current_keys[i] != t)]
    if not unsatisfied:
        # solved only if at least one slot had a known target (else we know nothing)
        if any(t is not None for t in targets):
            return {"kind": "solved", "slot": None}
        return {"kind": "blocked", "slot": None, "reason": "no known targets"}
    slot = unsatisfied[0]
    if cursor_idx != slot:
        return {"kind": "move_cursor", "slot": slot, "from": cursor_idx}
    return {"kind": "edit", "slot": slot,
            "current": (current_keys[slot] if slot < len(current_keys) else None),
            "target": targets[slot]}
