"""Trajectory / aim-ray detection — a game-agnostic substrate tool.

Some games draw a DASHED DIRECTIONAL LINE: an aim/launch trajectory, a laser, a
velocity arrow, a planned path.  COS's component extractor sees that line as N
separate little marks (or noise) and the acting VLM loses the one fact that
matters: a single RAY with an origin, a direction, and an endpoint.  This tool
recovers it.  Given a region, it finds the maximal set of components that are
roughly COLLINEAR and roughly EVENLY SPACED (the signature of a rendered
trajectory, as opposed to incidental scattered objects) and returns it as ONE
directional fact: the two endpoints, the direction vector + angle, the length,
the dash count, and the spacing regularity.

This is the perception half of "continuous directional control" (aiming / launch
games): once the VLM has the ray, it can reason that the mover at the ray's
origin travels along that direction — and that to hit a target it must RE-AIM the
ray toward the target (in a launch game a click typically sets the launch
direction toward the clicked point).

Pure geometry; MEASURES only, never interprets game meaning (the Prime
Directive).  Operates on already-extracted components, so it inherits the
palette-invariant, structural figure-ground of the `components` tool and never
keys on a single "background colour".
"""
from __future__ import annotations

import itertools
import math
from typing import List, Sequence, Tuple

import numpy as np

from .frameutils import clamp_bbox
from .registry import ToolContext, tool
from .visual import _panel_field_nonbg          # structural field/foreground split


Point = Tuple[float, float]                      # (row, col)


def _perp_distance(p: Point, a: Point, ux: float, uy: float) -> float:
    """Perpendicular distance from point p to the line through a with unit
    direction (ux, uy) in (row, col) space."""
    wr, wc = p[0] - a[0], p[1] - a[1]
    proj = wr * ux + wc * uy                     # scalar projection along the line
    pr, pc = wr - proj * ux, wc - proj * uy      # perpendicular component
    return math.hypot(pr, pc)


def find_trajectory(points: Sequence[Point], *, tol: float = 1.4,
                    min_dashes: int = 4, spacing_cv_max: float = 0.55) -> dict | None:
    """Find the maximal COLLINEAR + roughly-EVENLY-SPACED chain among `points`.

    Returns a ray dict, or None if no chain of >= min_dashes qualifies.  Pure
    geometry — no frame, no colour, no game knowledge.  RANSAC over point pairs
    (n is small: a handful of dashes), then an even-spacing gate so a few
    incidentally-collinear blobs are NOT mistaken for a drawn trajectory.
    """
    pts: List[Point] = [(float(r), float(c)) for r, c in points]
    n = len(pts)
    if n < min_dashes:
        return None

    best: dict | None = None
    for a, b in itertools.combinations(pts, 2):
        dr, dc = b[0] - a[0], b[1] - a[1]
        norm = math.hypot(dr, dc)
        if norm < 1e-6:
            continue
        ux, uy = dr / norm, dc / norm
        inliers = [p for p in pts if _perp_distance(p, a, ux, uy) <= tol]
        if len(inliers) < min_dashes:
            continue
        # order inliers along the line and measure spacing regularity
        proj = sorted(((p[0] - a[0]) * ux + (p[1] - a[1]) * uy, p) for p in inliers)
        ordered = [p for _, p in proj]
        coords = [t for t, _ in proj]
        gaps = [coords[i + 1] - coords[i] for i in range(len(coords) - 1)]
        mean_gap = sum(gaps) / len(gaps)
        if mean_gap <= 1e-6:
            continue
        cv = (np.std(gaps) / mean_gap) if len(gaps) > 1 else 0.0
        if cv > spacing_cv_max:                  # not evenly spaced -> not a trajectory
            continue
        span = math.hypot(ordered[-1][0] - ordered[0][0],
                          ordered[-1][1] - ordered[0][1])
        residual = max(_perp_distance(p, a, ux, uy) for p in inliers)
        # prefer more dashes, then longer span, then straighter (lower residual)
        key = (len(inliers), span, -residual)
        if best is None or key > best["_key"]:
            p0, p1 = ordered[0], ordered[-1]
            drow, dcol = p1[0] - p0[0], p1[1] - p0[1]
            dlen = math.hypot(drow, dcol) or 1.0
            best = {
                "_key": key,
                "end_a": [round(p0[0], 1), round(p0[1], 1)],
                "end_b": [round(p1[0], 1), round(p1[1], 1)],
                "direction_rowcol": [round(drow / dlen, 3), round(dcol / dlen, 3)],
                # angle in standard screen convention: 0=right(+col), 90=up(-row)
                "angle_deg": round(math.degrees(math.atan2(-drow, dcol)), 1),
                "length": round(span, 1),
                "n_dashes": len(inliers),
                "mean_spacing": round(mean_gap, 2),
                "spacing_cv": round(float(cv), 3),
                "straightness_residual": round(residual, 2),
                "members": [[round(p[0], 1), round(p[1], 1)] for p in ordered],
            }
    if best is None:
        return None
    best.pop("_key", None)
    return best


def _region_centroids(frame: np.ndarray, bbox, n_ticks: int,
                      max_dash_cells: int) -> tuple[list, list]:
    """Connected non-background components of a region (structural field split),
    returning (all_centroids, small_centroids) where 'small' are dash-sized."""
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    region = frame[r0:r1, c0:c1]
    nonbg, _bg = _panel_field_nonbg(region)
    h, w = nonbg.shape
    seen = np.zeros_like(nonbg, bool)
    all_c, small_c = [], []
    for y0 in range(h):
        for x0 in range(w):
            if not nonbg[y0, x0] or seen[y0, x0]:
                continue
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:
                cy, cx = st.pop(); cells.append((cy, cx))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and nonbg[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True; st.append((ny, nx))
            cy = r0 + sum(p[0] for p in cells) / len(cells)
            cx = c0 + sum(p[1] for p in cells) / len(cells)
            all_c.append((cy, cx))
            if len(cells) <= max_dash_cells:
                small_c.append((cy, cx))
    return all_c, small_c


@tool(name="detect_trajectory", category="perception/visual",
      summary=("find a DASHED DIRECTIONAL LINE in a region (an aim/launch "
               "trajectory, laser, velocity arrow, planned path): the maximal "
               "set of collinear, evenly-spaced marks -> ONE ray (two "
               "endpoints, direction vector + angle, length, dash count). Use "
               "when you see a dotted/dashed line connecting things instead of "
               "guessing it is N separate objects."),
      usage='{"op":"detect_trajectory","id":"k","bbox":[r0,c0,r1,c1]}',
      params={"bbox": "region to search (default whole frame)",
              "min_dashes": "min marks to count as a trajectory (default 4)",
              "tol": "max perpendicular slack in ticks (default 1.4)",
              "max_dash_cells": "marks larger than this many cells are treated as "
                                "endpoints/objects, not dashes (default 6)"})
def detect_trajectory(ctx: ToolContext, *, bbox=None, min_dashes=4, tol=1.4,
                      max_dash_cells=6, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    if bbox is None:
        bbox = [0, 0, n_ticks, n_ticks]
    all_c, small_c = _region_centroids(frame, bbox, n_ticks, int(max_dash_cells))
    # prefer the dash-sized marks (a drawn line is many small marks); fall back
    # to all components if too few smalls were found.
    pool = small_c if len(small_c) >= int(min_dashes) else all_c
    ray = find_trajectory(pool, tol=float(tol), min_dashes=int(min_dashes))
    out = {"region_ticks": list(clamp_bbox(bbox, n_ticks)),
           "n_components": len(all_c), "n_dash_marks": len(small_c)}
    if ray is None:
        out["trajectory"] = None
        out["note"] = ("no collinear, evenly-spaced chain of >= "
                       f"{int(min_dashes)} marks found")
    else:
        out["trajectory"] = ray
        out["note"] = ("a ray is directional but its SIGN is ambiguous from the "
                       "dashes alone: the origin is whichever endpoint sits on "
                       "the mover/launcher; it travels toward the other end.")
    return out
