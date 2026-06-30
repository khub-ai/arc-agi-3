"""Edit-law induction -- step 2 of the TARGET-PATTERN PURSUIT (the edit-to-target executor).

GAME-AGNOSTIC.  When an editable cell is CLICKED repeatedly, the substrate reads its glyph
each time (extract + shape_identity), giving a SEQUENCE of glyph identities.  This module
INDUCES the click->content law from that observed sequence -- it never assumes the mechanic
(per "discovery is observational"):

  - 'cycle' : the glyph advances through a fixed ordered set G0->G1->...->Gk->G0.
  - 'toggle': a cycle of length 2.
  - 'paint' : every click sets the cell to one fixed value (regardless of start).
  - 'noop'  : clicking the cell does nothing (the edit action is elsewhere / precondition unmet).
  - 'incomplete': all observed states distinct, no repeat yet -> probe more before deciding.

The induced law + `clicks_to_target` give step 3 (means-ends) the OPERATOR and its cost: how
many clicks take a slot from its current glyph to its computed target glyph.  Identities are
opaque hashable keys (shape_identity signatures), so this is colour/shape/game-agnostic.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Hashable


def induce_edit_law(glyph_seq: List[Hashable]) -> Dict:
    """Induce the click->content law from observed glyph identities.

    glyph_seq[0] = the glyph BEFORE the first click; glyph_seq[i] = after i clicks.
    Returns {kind, order, period, value} (order/period for cycle/toggle; value for paint).
    """
    seq = list(glyph_seq)
    if len(seq) < 2:
        return {"kind": "incomplete", "order": seq, "period": None, "value": None}
    if len(set(seq)) == 1:
        return {"kind": "noop", "order": [seq[0]], "period": 1, "value": None}

    # PAINT: every click lands on one fixed value, regardless of the starting glyph.
    post = seq[1:]
    if len(set(post)) == 1 and post[0] != seq[0]:
        return {"kind": "paint", "order": None, "period": None, "value": post[0]}

    # CYCLE: the sequence is periodic.  Find the first index where a state repeats; the prefix
    # up to it is the candidate order.  Confirm the whole sequence follows that order.
    first = {}
    period = None
    for i, g in enumerate(seq):
        if g in first:
            period = i - first[g]
            break
        first[g] = i
    if period:
        order = seq[:period]                    # cycle assumed to start from seq[0]
        if len(set(order)) == period and all(seq[i] == order[i % period] for i in range(len(seq))):
            kind = "toggle" if period == 2 else "cycle"
            return {"kind": kind, "order": order, "period": period, "value": None}
        return {"kind": "unknown", "order": None, "period": None, "value": None}

    # No repeat observed yet: distinct states so far -> need more probing to close the cycle.
    return {"kind": "incomplete", "order": seq, "period": len(seq), "value": None}


def clicks_to_target(law: Dict, current: Hashable, target: Hashable) -> Optional[int]:
    """Minimum clicks to take a cell from `current` to `target` under the induced law.
    None if unreachable (or law unknown/incomplete)."""
    if current == target:
        return 0
    kind = law.get("kind")
    if kind in ("cycle", "toggle"):
        order, period = law.get("order") or [], law.get("period")
        if current in order and target in order and period:
            return (order.index(target) - order.index(current)) % period
        return None
    if kind == "paint":
        return 1 if target == law.get("value") else None
    if kind == "noop":
        return None             # current != target already handled above
    return None                 # unknown / incomplete
