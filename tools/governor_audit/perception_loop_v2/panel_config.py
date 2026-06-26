"""panel_config.py -- drive a toggleable array of bars/cells to a TARGET configuration,
from ANY current state.

Generic manipulation capability COS should have for any game: when every cell can be
toggled at will, COS can CLEAN UP a contaminated panel to exactly match a target -- read
each cell's current state, toggle only the cells that differ, and verify.  No assumption
that the panel starts empty.  Closed-loop (read -> diff -> toggle -> re-read -> verify)
absorbs click errors, stale/contaminated state, and the 1-frame toggle lag.

Pure measurement + planning here (numpy + scene_state.measure_grid); the driver/responder
applies the clicks and re-reads.
"""
from __future__ import annotations

import numpy as np


def read_bar_states(frame, panel_bbox, on_color=(0, 0, 0), tol: int = 2) -> list:
    """Measure the panel's bars and read each one's on/off (on == on_color present).

    Returns [{rc, click, on, bbox}].  panel_bbox is [r0,c0,r1,c1] (row-first)."""
    import scene_state as ss
    f = np.asarray(frame)
    g = ss.measure_grid(f, panel_bbox, tol=tol)
    if not g:
        return []
    out = []
    for b in g["bars"]:
        r0, c0, r1, c1 = b["bbox"]
        on = bool(np.any(np.all(f[r0:r1 + 1, c0:c1 + 1] == on_color, axis=2)))
        out.append({"rc": tuple(b["rc"]), "click": b["click"], "on": on, "bbox": b["bbox"]})
    return out


def toggle_plan(bar_states, target_on) -> list:
    """The clicks that drive the panel to the EXACT target: every bar whose current state
    disagrees with the target gets toggled (on->off or off->on).  Robust to a contaminated
    start -- bars that should be OFF but are ON are cleared, not left.

    target_on: iterable of rc (row,col) that should be ON.  Returns
    [{rc, click, set(bool)}] for the bars to click."""
    target = {tuple(t) for t in target_on}
    return [{"rc": b["rc"], "click": b["click"], "set": b["rc"] in target}
            for b in bar_states if b["on"] != (b["rc"] in target)]


def matches_target(bar_states, target_on) -> bool:
    """True iff every bar's state already equals the target (panel == target exactly)."""
    target = {tuple(t) for t in target_on}
    return all(b["on"] == (b["rc"] in target) for b in bar_states)


def diff_summary(bar_states, target_on) -> dict:
    """How far the panel is from the target: counts of bars to turn on / off."""
    target = {tuple(t) for t in target_on}
    to_on = sum(1 for b in bar_states if not b["on"] and b["rc"] in target)
    to_off = sum(1 for b in bar_states if b["on"] and b["rc"] not in target)
    return {"to_on": to_on, "to_off": to_off, "ok": to_on == 0 and to_off == 0}


def code_from_bar_states(bar_states) -> set:
    """The code DISPLAYED by a reference panel = the bar-ROWS that are ON in a majority of the
    panel's (uniform) columns.  A reference/key panel shows the same pattern in every column for
    the active option, so the row-set IS the code -- robust to a stray per-column misread."""
    if not bar_states:
        return set()
    from collections import Counter
    ncols = len({b["rc"][1] for b in bar_states}) or 1
    on_per_row = Counter(b["rc"][0] for b in bar_states if b["on"])
    return {r for r, cnt in on_per_row.items() if cnt * 2 >= ncols}


def read_reference_code(frame, ref_bbox, on_color=(0, 0, 0), tol: int = 2) -> set:
    """MEASURE the code a reference/key panel is currently displaying for the active option.

    When a game DISPLAYS the answer (a legend/reference that, for the selected option, lights the
    bars encoding it), READ IT -- never guess.  Returns the set of ON bar-rows (the code); map it
    onto a program column c as {(r, c) for r in code} and drive that column there with toggle_plan.
    This is the measured-ground-truth half of 'follow the instructions'."""
    return code_from_bar_states(read_bar_states(frame, ref_bbox, on_color, tol))
