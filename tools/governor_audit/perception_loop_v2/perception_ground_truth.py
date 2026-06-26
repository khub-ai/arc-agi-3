"""Ground-truth perception guard -- the PICTURE arbitrates over symbolic claims.

The dangerous failure this prevents: a decision ("no-op" / "stuck" / "canned" /
"the agent didn't move") made on a SYMBOLIC flag that disagrees with the pixels.
On tn36 lc5 a fire moved the mover a clear several cells, but the symbolic
`agent_moved=false` (the tracker had lost the mover across the fire animation)
was trusted over the frame -- so a successful action was twice called a no-op.

This module MEASURES, from the raw frames, whether the agent's silhouette LEFT
its reported cell, and reconciles that against the symbolic claim.  The
reconciliation is ASYMMETRIC by design: it only ever CORRECTS the dangerous
false-negative (symbolic says "didn't move" while the pixels show it did) and
otherwise stays out of the way -- it never flips a reported move to a non-move.
Pure numpy CV (no model cost); degrades to "can't measure" (None) on any
uncertainty so it never overrides on a guess.
"""
from __future__ import annotations
from typing import Optional

try:
    import numpy as np
    _OK = True
except Exception:                                    # pragma: no cover
    _OK = False


def _dominant_color(block):
    """Most common packed-RGB colour in a HxWx3 block (or None if empty)."""
    if block.size == 0:
        return None
    flat = (block[:, :, 0].astype(np.int64) << 16
            | block[:, :, 1].astype(np.int64) << 8
            | block[:, :, 2].astype(np.int64)).reshape(-1)
    vals, counts = np.unique(flat, return_counts=True)
    return int(vals[counts.argmax()])


def _packed(frame_rgb):
    a = np.asarray(frame_rgb)[:, :, :3].astype(np.int64)
    return a[:, :, 0] << 16 | a[:, :, 1] << 8 | a[:, :, 2]


def _best_moved_cluster(mask, footprint, old_box):
    """Where did the agent GO?  Among the connected components of ``mask`` (the
    agent-colour pixels OUTSIDE its old box), pick the one whose size is
    compatible with the agent's footprint (0.4x..2.5x) and nearest the old box --
    that is the relocated agent, NOT a same-colour decoy (a legend glyph, the
    goal, another marker).  Returns its (row,col) centroid, or None when no
    compatible cluster is found (so the caller asserts NO new cell rather than a
    wrong one -- a wrong cell is itself the false information we are preventing)."""
    try:
        from scipy import ndimage as _ndi          # noqa: E402
        lab, n = _ndi.label(mask)
        if n == 0:
            return None
        r0, c0, r1, c1 = old_box
        oc = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        best, best_d = None, None
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            sz = len(ys)
            if sz < max(2, 0.4 * footprint) or sz > 2.5 * footprint:
                continue
            cy, cx = float(ys.mean()), float(xs.mean())
            d = (cy - oc[0]) ** 2 + (cx - oc[1]) ** 2
            if best_d is None or d < best_d:
                best, best_d = (cy, cx), d
        return best
    except Exception:
        return None                                  # scipy absent / any error -> no claim


def measure_agent_move(prev_rgb, curr_rgb, agent_bbox) -> Optional[dict]:
    """Did the agent (occupying ``agent_bbox`` = [r0,c0,r1,c1] in PREV) LEAVE that
    box by the CURR frame?

    Method (silhouette retention): the agent's colour is the dominant colour in
    its tight prev box.  Count that colour's pixels inside the box in prev vs
    curr; if most are GONE the agent vacated the cell -> it MOVED.  When it moved,
    locate the largest cluster of that colour OUTSIDE the old box as the new
    centroid (best-effort).  Returns a dict of measurements, or None if it cannot
    measure confidently (so callers never override on a guess)."""
    if not _OK or prev_rgb is None or curr_rgb is None or not agent_bbox:
        return None
    try:
        prev = np.asarray(prev_rgb)[:, :, :3]
        curr = np.asarray(curr_rgb)[:, :, :3]
        if prev.shape != curr.shape:
            return None
        H, W = prev.shape[:2]
        r0, c0, r1, c1 = [int(v) for v in agent_bbox]
        r0, c0 = max(0, r0), max(0, c0)
        r1, c1 = min(H, r1), min(W, c1)
        if r1 - r0 < 1 or c1 - c0 < 1:
            return None
        color = _dominant_color(prev[r0:r1, c0:c1])
        if color is None:
            return None
        pp, pc = _packed(prev), _packed(curr)
        box_prev = (pp[r0:r1, c0:c1] == color)
        n_prev = int(box_prev.sum())
        if n_prev < 2:                       # agent footprint too small to track
            return None
        n_curr_in_box = int((pc[r0:r1, c0:c1] == color).sum())
        retained = n_curr_in_box / n_prev
        # The agent VACATED its box when most of its coloured footprint is gone.
        # 0.5 is a structural "more gone than not" split, not a tuned magnet:
        # a static agent retains ~1.0, a moved agent retains ~0.0.
        moved = retained < 0.5
        new_centroid = None
        if moved:
            mask_curr = (pc == color)
            mask_curr[r0:r1, c0:c1] = False         # ignore any remnant in the old box
            new_centroid = _best_moved_cluster(mask_curr, n_prev, (r0, c0, r1, c1))
        return {"moved": bool(moved), "old_bbox": [r0, c0, r1, c1],
                "agent_color": color, "retained_fraction": float(retained),
                "footprint_prev": n_prev, "footprint_in_old_box": n_curr_in_box,
                "new_centroid_tick": new_centroid}
    except Exception:
        return None


def _cell_of_tick(tick_rc, origin_rc, cell_ticks):
    return (int((tick_rc[0] - origin_rc[0]) // cell_ticks),
            int((tick_rc[1] - origin_rc[1]) // cell_ticks))


def reconcile(symbolic_moved, symbolic_new_cell, measured,
              origin_rc=None, cell_ticks=None) -> dict:
    """Reconcile the symbolic agent-move claim with the measured pixels.  The
    PICTURE wins, but ONLY for the dangerous false-negative: when the symbolic
    claim says the agent did NOT move yet the measurement shows it vacated its
    cell.  Returns {moved, new_cell, corrected, note}: `corrected` True when the
    guard overrode the symbolic claim; `note` is a loud mismatch string to surface
    (or None).  Never flips a reported move to a non-move; never overrides when the
    measurement is absent/uncertain."""
    out = {"moved": bool(symbolic_moved), "new_cell": symbolic_new_cell,
           "corrected": False, "note": None}
    if not measured or not measured.get("moved"):
        return out                                   # nothing measured, or agent stayed
    if symbolic_moved:
        return out                                   # symbolic already agrees it moved
    # MISMATCH: pixels show the agent vacated its cell, symbolic says it didn't.
    new_cell = symbolic_new_cell
    nc = measured.get("new_centroid_tick")
    if nc is not None and origin_rc is not None and cell_ticks:
        new_cell = list(_cell_of_tick(nc, origin_rc, cell_ticks))
    out.update(moved=True, new_cell=new_cell, corrected=True)
    where = f" now at ~cell {new_cell}" if new_cell is not None else ""
    out["note"] = (
        "[GROUND-TRUTH MISMATCH -- PICTURE WINS] You reported the agent did NOT "
        f"move, but the pixels show it VACATED its cell (only "
        f"{measured['retained_fraction']:.0%} of its silhouette remains there){where}. "
        "The action HAD an effect -- do NOT report 'no-op' / 'canned' / 'stuck'. "
        "agent_moved was auto-corrected to true; re-read the frame.")
    return out
