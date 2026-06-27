"""Salient 1-pixel MARK detection.

WHY THIS EXISTS
---------------
Some games encode their most important state in tiny, near-unique MARKS -- a
coloured centre/dot/cursor a few pixels in size (ka59: a WHITE 1-px centre = the
agent, a BLACK 1-px centre = a static target; clicking it swaps).  Bulk
figure-ground perception and the entity tracker's speck filter (drop < 5 px)
DISCARD exactly these pixels, so the agent and its swap are invisible to COS.

A mark is defined structurally by RARITY, not by a specific colour: a colour that
occupies only a few pixels of the whole frame is a salient mark/handle, not a
field or texture.  Game-agnostic; pure numpy (degrades to [] without numpy).
"""
from __future__ import annotations

from typing import List, Tuple

try:
    import numpy as np
    _OK = True
except Exception:                                       # pragma: no cover
    _OK = False


def _packed(grid):
    """Return an HxW int grid: RGB frames are packed to a single int per pixel."""
    g = np.asarray(grid)
    if g.ndim == 3:
        g = g[:, :, :3].astype(int)
        return (g[..., 0] << 16) | (g[..., 1] << 8) | g[..., 2]
    return g.astype(int)


def find_marks(grid, max_px: int = 8, ignore_border: bool = True
               ) -> List[Tuple[int, int, int]]:
    """Embedded 1-pixel MARKS: a pixel whose colour differs from ALL FOUR of its
    neighbours -- a locally-isolated dot/centre embedded in a host (a coloured
    centre on a piece).  Returns ``[(row, col, colour_int), ...]``.

    Local isolation, not global rarity, is the right signal: a mark colour can be
    common ELSEWHERE in the frame (ka59's white centre renders to white, which
    also paints platform edges) yet still be a unique dot WHERE it sits.  A line /
    region has same-colour neighbours and is excluded; a clean ARC palette frame
    has no anti-alias singletons, so the only isolated pixels are real marks.
    ``ignore_border`` drops the outer ring (HUD/frame edges).  ``max_px`` reserved
    for a future small-blob variant.  Pure numpy; [] without it."""
    if not _OK or grid is None:
        return []
    try:
        g = _packed(grid)
    except Exception:
        return []
    if g.ndim != 2 or g.shape[0] < 3 or g.shape[1] < 3:
        return []
    c = g[1:-1, 1:-1]
    iso = ((c != g[:-2, 1:-1]) & (c != g[2:, 1:-1])
           & (c != g[1:-1, :-2]) & (c != g[1:-1, 2:]))
    ys, xs = np.where(iso)
    H, W = g.shape
    out: List[Tuple[int, int, int]] = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        r, col = y + 1, x + 1                     # offset back into full grid
        if ignore_border and (r <= 0 or col <= 0 or r >= H - 1 or col >= W - 1):
            continue
        out.append((int(r), int(col), int(g[r, col])))
    return out
