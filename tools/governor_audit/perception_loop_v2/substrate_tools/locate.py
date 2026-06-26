"""Reliable entity LOCATION / tracking — a game-agnostic substrate tool.

The recurring failure (su15 aiming loop): the actor needs the controllable
mover's PRECISE current position every turn to aim, but ad-hoc pixel detection is
error-prone — row/col confusion, MERGING the mover with a same-coloured HUD
marker far away, or catching aim-line dashes. This tool MEASURES it properly:
given the mover's appearance (a colour the perception layer reported) and/or a
last-known point to disambiguate, it returns the matching connected component's
exact centroid + bbox. With no colour it can instead find the component that
MOVED across the last action's animation (the controllable one) — so the mover is
trackable even when its colour is unknown.

Why this is the right fix: tracking a small mover among same-coloured distractors
needs (a) real connected-component analysis, not a global colour average, and
(b) a `near` disambiguator (the mover is the matching blob closest to where it
was last turn). Both live here, measured once, reused every turn.

Substrate MEASURES; the VLM INTERPRETS — it names what to find; this reports
where it is. Matching a FOREGROUND entity the actor named is a VLM-directed
measurement, NOT a hardcoded background-colour key (the banned figure-ground
assumption); when no colour is given it falls back to the structural,
palette-invariant figure-ground.
"""
from __future__ import annotations

import numpy as np

from .frameutils import clamp_bbox
from .registry import ToolContext, tool
from .visual import _panel_field_nonbg


def _parse_color(c):
    if isinstance(c, str):
        s = c.lstrip("#").strip()
        if len(s) == 6:
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
    if isinstance(c, (list, tuple)) and len(c) >= 3:
        return (int(c[0]), int(c[1]), int(c[2]))
    raise ValueError(f"bad color {c!r}")


def _components(mask):
    """4-connected components of a boolean mask -> list of cell lists."""
    h, w = mask.shape
    seen = np.zeros_like(mask, bool)
    out = []
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:
                cy, cx = st.pop(); cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; st.append((ny, nx))
            out.append(cells)
    return out


@tool(name="locate_entity", category="perception/visual",
      summary=("find the PRECISE current position of an entity (e.g. the mover) "
               "by colour and/or a last-known point — reliable connected-"
               "component location that disambiguates same-coloured distractors. "
               "Use it EVERY TURN to track a mover for aiming instead of "
               "eyeballing. With no colour it locates what MOVED in the last "
               "action's animation."),
      usage='{"op":"locate_entity","id":"k","color":"#b03ac3","near":[44,24]}',
      params={"color": "the entity's colour as #rrggbb or [r,g,b] (from its "
                       "perceived appearance); omit to use motion/structure",
              "near": "[row,col] the entity's ROUGH / last-known position. ALWAYS "
                      "pass this when a same-looking entity may exist elsewhere "
                      "(e.g. a HUD swatch the same colour as the mover) — it "
                      "returns the matching blob CLOSEST to `near`, so it can't "
                      "lock onto the wrong look-alike. Omitting it returns the "
                      "LARGEST match and flags ambiguity if there are several.",
              "bbox": "optional [r0,c0,r1,c1] region to search within",
              "tol": "colour-match tolerance (RGB distance, default 45)",
              "min_cells": "ignore blobs smaller than this (default 2)",
              "match": "'color' | 'moved' | 'structure' (default: auto)"})
def locate_entity(ctx: ToolContext, *, color=None, near=None, bbox=None,
                  tol=45, min_cells=2, match=None, **_) -> dict:
    frame, n = ctx.frame, ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox or [0, 0, n, n], n)
    region = frame[r0:r1, c0:c1].astype(int)
    anim = ctx.anim_frames
    mode = match or ("color" if color is not None
                     else ("moved" if (anim and len(anim) >= 2) else "structure"))

    if mode == "moved" and anim and len(anim) >= 2:
        a = np.asarray(anim[0])[r0:r1, c0:c1].astype(int)
        b = np.asarray(anim[-1])[r0:r1, c0:c1].astype(int)
        changed = np.abs(a - b).sum(2) > 40
        foreground = b.max(2) > 40                     # the NEW position is non-black
        mask = changed & foreground
        note = "located the component that MOVED across the animation"
    elif color is not None:
        tgt = np.array(_parse_color(color))
        mask = np.sqrt(((region - tgt) ** 2).sum(2)) <= float(tol)
        note = f"located by colour match (tol {tol})"
    else:
        mask, _bg = _panel_field_nonbg(region)         # structural figure-ground
        note = "located by structural foreground (no colour given)"

    cands = []
    for cells in _components(mask):
        if len(cells) < int(min_cells):
            continue
        ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
        cands.append({"centroid": [round(r0 + sum(ys) / len(ys), 1),
                                    round(c0 + sum(xs) / len(xs), 1)],
                      "bbox": [r0 + min(ys), c0 + min(xs), r0 + max(ys) + 1, c0 + max(xs) + 1],
                      "cells": len(cells)})
    if not cands:
        return {"found": False, "n_candidates": 0, "note": note + "; no matching component"}

    if near is not None:
        nr, nc = float(near[0]), float(near[1])
        best = min(cands, key=lambda d: (d["centroid"][0] - nr) ** 2 + (d["centroid"][1] - nc) ** 2)
        note += f"; chose the candidate nearest {[round(nr, 1), round(nc, 1)]}"
    else:
        best = max(cands, key=lambda d: d["cells"])
        note += "; chose the largest candidate"
        if len(cands) > 1:
            note += (f" — AMBIGUOUS: {len(cands)} matching components and no "
                     f"`near` given; pass `near` (the entity's rough/expected "
                     f"[row,col]) to disambiguate same-looking entities")
    cands.sort(key=lambda d: -d["cells"])
    return {"found": True, "centroid": best["centroid"], "bbox": best["bbox"],
            "cells": best["cells"], "n_candidates": len(cands),
            "candidates": [c["centroid"] for c in cands[:8]], "note": note}
