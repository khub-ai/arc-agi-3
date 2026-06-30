"""Edit executor -- steps 3-4 of the TARGET-PATTERN PURSUIT: means-ends over per-slot glyph
difference + verify.  Composes step 1 (target_pattern: per-slot TARGET) and step 2 (edit_law:
the click->content law) into a concrete CLICK PLAN, and checks completion.

GAME-AGNOSTIC.  A "slot" is one editable cell: {name, cell_bbox, current, target} where
`current`/`target` are glyph identities (shape_identity signatures).  Means-ends here = reduce
each slot's (current -> target) glyph difference by the minimum clicks the induced law needs.
"""
from __future__ import annotations

from typing import List, Dict

try:
    import edit_law as _el
except ImportError:
    from perception_loop_v2 import edit_law as _el


def plan_edits(slots: List[Dict], law: Dict) -> Dict:
    """Per-slot click plan to drive each editable cell to its target under `law`.

    Returns {plan, skipped}: plan = [{slot, cell_bbox, clicks, from, to}, ...] for slots that
    need change AND are reachable; skipped = [{slot, reason, from, to}] for unreachable ones
    (law can't reach the target -- e.g. paint to a non-paint value, or noop/incomplete law).
    Satisfied slots (current == target) are silently omitted.
    """
    plan, skipped = [], []
    for s in slots:
        cur, tgt = s.get("current"), s.get("target")
        n = _el.clicks_to_target(law, cur, tgt)
        if n is None:
            skipped.append({"slot": s.get("name"), "reason": "unreachable",
                            "from": cur, "to": tgt})
        elif n > 0:
            plan.append({"slot": s.get("name"), "cell_bbox": s.get("cell_bbox"),
                         "clicks": n, "from": cur, "to": tgt})
    return {"plan": plan, "skipped": skipped}


def unsatisfied(slots: List[Dict]) -> List[str]:
    """Step-4 verify: names of slots whose current glyph still differs from its target (re-read
    after editing).  Empty list => the target pattern is achieved."""
    return [s.get("name") for s in slots if s.get("current") != s.get("target")]


def is_solved(slots: List[Dict]) -> bool:
    """True iff every slot's current glyph equals its target (the win for an apply-the-legend game)."""
    return bool(slots) and not unsatisfied(slots)
