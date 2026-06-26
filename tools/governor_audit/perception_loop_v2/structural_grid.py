"""Decompose a region into its repeating LATTICE of cells.

A general visual primitive, deliberately free of any game- or
domain-specific concept.  Given a rectangular region of a frame that
contains a regular grid of repeated marks -- a control panel, a game
board, a calendar, a spreadsheet block, a keypad -- it recovers the
grid: how many rows and columns, and the bounding box of every cell
INCLUDING the empty ones.

Why empty cells matter: to compare two configurations of the same grid
(e.g. "make this settable panel match that reference panel") you must
line up cell-for-cell, and a cell that is empty in one config but full
in the other is exactly the difference you act on.  Detecting only the
filled marks would miss those, and would give two configs a different
number of units, so they could not be paired.

The substrate MEASURES here (find the lattice, tile it into cells); it
does not interpret what any cell "means" -- that is the caller's job.

Method (no baked pixel constants; every scale is derived from the
measured marks):
  1. background = the modal colour of the region (the field the marks
     sit on);
  2. foreground = pixels unequal to that background; connected
     components are the marks (ARC frames are exact-palette, so the
     comparison is exact -- no tolerance needed);
  3. drop any component that spans essentially the whole region (a
     frame / separator / the field itself is not a cell);
  4. cluster the marks' centres on each axis into column centres and
     row centres, using the median mark extent on that axis as the
     gap scale (marks in one column share an x; a new column begins
     only when the x-gap exceeds a whole mark width);
  5. tile the region by the midpoints between adjacent centres ->
     one cell box per (row, column), empty cells included.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import label


def _modal_color(region: np.ndarray) -> np.ndarray:
    flat = region.reshape(-1, region.shape[-1])
    colors, counts = np.unique(flat, axis=0, return_counts=True)
    return colors[int(counts.argmax())]


def _components(region: np.ndarray, bg: np.ndarray) -> list[dict]:
    """Connected foreground components (8-connectivity), each a dict with
    its centre and extent.

    Two structural gates drop non-cell components (a FRAME / FIELD /
    SEPARATOR line is not a repeated cell):
      * a component spanning essentially the whole region in BOTH
        dimensions (the field / an outer frame);
      * a SIZE OUTLIER -- a repeated cell-mark is, by definition, near
        the median mark size, so a component several times larger on
        either axis is a separator / border line, not a cell.  This is
        derived from the measured marks (median extent), not a baked
        pixel constant; it removes e.g. the thin full-height line
        between two adjacent panels that would otherwise invent a
        spurious extra column."""
    h, w = region.shape[:2]
    mask = np.any(region != bg, axis=2)
    lab, n = label(mask, structure=np.ones((3, 3), dtype=int))
    out: list[dict] = []
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        ch = int(ys.max() - ys.min() + 1)
        cw = int(xs.max() - xs.min() + 1)
        if ch >= 0.9 * h and cw >= 0.9 * w:        # the whole region / field
            continue
        out.append({"cy": float(ys.mean()), "cx": float(xs.mean()),
                    "h": ch, "w": cw, "size": int(len(xs))})
    if len(out) >= 3:
        med_w = float(np.median([c["w"] for c in out]))
        med_h = float(np.median([c["h"] for c in out]))
        kept = [c for c in out
                if c["w"] <= 3.0 * med_w and c["h"] <= 3.0 * med_h]
        if len(kept) >= 2:                          # never prune below a lattice
            out = kept
    return out


def _cluster_centers(positions: list[float], scale: float) -> list[float]:
    """1-D clustering by gap: a new cluster starts when the gap to the
    previous point exceeds ``scale`` (one mark extent).  Returns the
    sorted cluster means."""
    if not positions:
        return []
    pts = sorted(positions)
    clusters: list[list[float]] = [[pts[0]]]
    s = max(float(scale), 1.0)
    for p in pts[1:]:
        if p - clusters[-1][-1] > s:
            clusters.append([p])
        else:
            clusters[-1].append(p)
    return [sum(c) / len(c) for c in clusters]


def _boundaries(centers: list[float], lo: float, hi: float) -> list[float]:
    """Edges that tile [lo, hi]: outer edges at lo/hi, inner edges at the
    midpoints between adjacent centres."""
    if not centers:
        return [lo, hi]
    edges = [lo]
    for a, b in zip(centers, centers[1:]):
        edges.append((a + b) / 2.0)
    edges.append(hi)
    return edges


def decompose(frame_rgb: np.ndarray, bbox, *, min_cells: int = 2) -> Optional[dict]:
    """Recover the lattice inside ``bbox`` of ``frame_rgb``.

    Returns ``{"n_rows", "n_cols", "cells", "col_centers", "row_centers"}``
    where ``cells`` is a list of ``[r0, c0, r1, c1]`` ABSOLUTE-frame boxes
    in ROW-MAJOR order (top-to-bottom, left-to-right), one per grid
    position (empty cells included).  Returns ``None`` when no regular
    grid of >= ``min_cells`` cells is found (so the caller can fall back).
    """
    try:
        r0, c0, r1, c1 = [int(v) for v in bbox]
    except Exception:
        return None
    H, W = frame_rgb.shape[:2]
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H, r1), min(W, c1)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    region = frame_rgb[r0:r1, c0:c1]
    if region.ndim != 3:
        return None

    bg = _modal_color(region)
    comps = _components(region, bg)
    if len(comps) < 2:
        return None

    med_w = float(np.median([c["w"] for c in comps]))
    med_h = float(np.median([c["h"] for c in comps]))
    col_centers = _cluster_centers([c["cx"] for c in comps], med_w)
    row_centers = _cluster_centers([c["cy"] for c in comps], med_h)
    n_cols, n_rows = len(col_centers), len(row_centers)
    if n_rows * n_cols < min_cells or (n_rows == 1 and n_cols == 1):
        return None

    h, w = region.shape[:2]
    col_edges = _boundaries(col_centers, 0.0, float(w))
    row_edges = _boundaries(row_centers, 0.0, float(h))

    cells: list[list[int]] = []
    for i in range(n_rows):
        for j in range(n_cols):
            cr0 = r0 + int(round(row_edges[i]))
            cr1 = r0 + int(round(row_edges[i + 1]))
            cc0 = c0 + int(round(col_edges[j]))
            cc1 = c0 + int(round(col_edges[j + 1]))
            cells.append([cr0, cc0, cr1, cc1])
    return {"n_rows": n_rows, "n_cols": n_cols, "cells": cells,
            "col_centers": [c0 + cc for cc in col_centers],
            "row_centers": [r0 + rc for rc in row_centers]}


def read_path(frame_path: str, bbox, *, min_cells: int = 2) -> Optional[dict]:
    """Convenience: decompose a region of a frame given by file path."""
    try:
        from PIL import Image
        im = Image.open(str(frame_path)).convert("RGB")
        if im.size != (64, 64):
            im = im.resize((64, 64), Image.NEAREST)
        return decompose(np.asarray(im, dtype=int), bbox, min_cells=min_cells)
    except Exception:
        return None
