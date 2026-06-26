"""click_feedback.py -- post-click change reporting + click auto-calibration.

The substrate must tell the VLM WHAT a click changed -- not just hand back the
settled frame.  change_report() diffs pre/post (and the animation) into a clean,
scene-aware report: which entities changed, colour/state flips, and TRANSIENT
effects (a flash that reverted, invisible in before/after).  verify_click()
auto-calibrates: it checks the observed change actually landed AT the clicked
cell, catching any coordinate/scale drift -- run it "when in doubt".

Pure measurement over the logical (tick) frame; game-agnostic; guarded.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

import numpy as np


def _dom(reg) -> tuple:
    flat = reg.reshape(-1, 3)
    if not len(flat):
        return (0, 0, 0)
    return tuple(int(x) for x in Counter(map(tuple, flat.tolist())).most_common(1)[0][0])


def change_report(prev_rgb, curr_rgb, scene=None, anim_frames=None,
                  clicked=None) -> dict:
    """Report what a click CHANGED.  Returns:
      changed_bbox  -- [r0,c0,r1,c1] of the settled pre/post change (or None),
      n_changed     -- changed pixel count (settled),
      entities      -- scene entity ids overlapping the change (colour/state flip),
      transient     -- True if the animation changed regions that the settled frame
                       does NOT (a flash/sweep that reverted -- otherwise invisible),
      transient_bbox-- the transient region (or None),
      summary       -- one-line text.
    Guarded -> a benign 'no change' report."""
    try:
        p = np.asarray(prev_rgb)[:, :, :3]
        c = np.asarray(curr_rgb)[:, :, :3]
        d = np.any(p != c, axis=2)
        ys, xs = np.where(d)
        changed_bbox = ([int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
                        if len(ys) else None)
        # entities overlapping the settled change
        ents = []
        if scene is not None and changed_bbox is not None:
            for eid, e in (getattr(scene, "entities", {}) or {}).items():
                bb = getattr(e, "bbox", None)
                if not bb:
                    continue
                if not (bb[2] < changed_bbox[0] or bb[0] > changed_bbox[2]
                        or bb[3] < changed_bbox[1] or bb[1] > changed_bbox[3]):
                    ents.append(eid)
        # transient: union of animation changes vs frame 0, minus the settled change
        transient, transient_bbox = False, None
        if anim_frames and len(anim_frames) >= 2:
            f0 = np.asarray(anim_frames[0])[:, :, :3]
            anim_any = np.zeros(f0.shape[:2], bool)
            for f in anim_frames[1:]:
                anim_any |= np.any(np.asarray(f)[:, :, :3] != f0, axis=2)
            extra = anim_any & ~d
            ays, axs = np.where(extra)
            if len(ays) > 4:
                transient = True
                transient_bbox = [int(ays.min()), int(axs.min()), int(ays.max()), int(axs.max())]
        summ = (f"no visible change" if changed_bbox is None
                else f"changed {int(d.sum())}px at {changed_bbox}"
                     + (f"; entities {ents}" if ents else "")
                     + ("; + a TRANSIENT effect that reverted" if transient else ""))
        return {"changed_bbox": changed_bbox, "n_changed": int(d.sum()),
                "entities": ents, "transient": transient,
                "transient_bbox": transient_bbox, "clicked": clicked, "summary": summ}
    except Exception as e:
        return {"changed_bbox": None, "n_changed": 0, "entities": [], "transient": False,
                "transient_bbox": None, "clicked": clicked, "summary": f"report error ({e})"}


def verify_click(clicked_colrow, report, tol: int = 2) -> dict:
    """Auto-calibration: did the change land AT the clicked cell?  ``clicked_colrow``
    = (col, row) in ticks.  Returns {calibrated, observed_offset, note}.  If the
    change centroid is within ``tol`` ticks of the click -> calibrated; a consistent
    offset flags a coordinate/scale drift; no change -> inconclusive (the cell may
    just be inert -- not a calibration failure)."""
    try:
        if not clicked_colrow:
            return {"calibrated": None, "observed_offset": None, "note": "no click given"}
        bb = (report or {}).get("changed_bbox")
        # ignore a HUD-only change (top band) when judging click landing
        if bb is None:
            return {"calibrated": None, "observed_offset": None,
                    "note": "no change -- inconclusive (cell may be inert)"}
        ccol = (bb[1] + bb[3]) / 2
        crow = (bb[0] + bb[2]) / 2
        off = (round(ccol - clicked_colrow[0], 1), round(crow - clicked_colrow[1], 1))
        inside = bb[1] - tol <= clicked_colrow[0] <= bb[3] + tol and \
                 bb[0] - tol <= clicked_colrow[1] <= bb[2] + tol
        return {"calibrated": bool(inside), "observed_offset": off,
                "note": ("change at the clicked cell" if inside
                         else f"change is OFFSET from the click by {off} -- recalibrate")}
    except Exception as e:
        return {"calibrated": None, "observed_offset": None, "note": f"verify error ({e})"}
