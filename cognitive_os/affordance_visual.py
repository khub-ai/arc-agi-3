"""Visual analysis primitives for affordance structural properties.

The affordance DSL's structural nodes (``AttachedSideOf``,
``OrientationOf``) reference properties cached on the entity's
``visual_id`` bundle.  This module computes those properties from
pixel evidence in the published frame and the level's passable_grid.

Both functions are domain-agnostic: they operate on numpy arrays plus
a bbox, with no game-specific tokens.  The caller (the entity
publisher in the ARC adapter, or any equivalent in another domain)
invokes them once per entity at publish time and stores the results
on the entity's properties dict.

When the visual evidence is ambiguous (no side dominates, or the
bbox is roughly square), the analyzers return ``None`` and downstream
affordance evaluation degrades to ``Undetermined`` — the same code
path that already handles missing-information cases.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def compute_orientation_axis(
    bbox_h: int,
    bbox_w: int,
    *,
    ratio_threshold: float = 2.0,
) -> Optional[str]:
    """Derive the major orientation axis of an entity from its bbox.

    Returns ``"horizontal"`` when the bbox is significantly wider than
    tall, ``"vertical"`` when significantly taller than wide,
    ``"point"`` when the ratio is near 1 (compact / square sprite).

    The threshold defaults to 2 -- a sprite whose longer side is at
    least twice the shorter side is treated as elongated.  A 5x5 cell
    occupied by a 1-pixel-thick bar at 1x5 or 5x1 cleanly qualifies;
    a 3x3 glyph stays POINT.

    Returns ``None`` when bbox is missing (one or both dimensions
    zero).  The DSL evaluator treats this as undetermined.
    """
    if bbox_h <= 0 or bbox_w <= 0:
        return None
    long_side  = max(int(bbox_h), int(bbox_w))
    short_side = min(int(bbox_h), int(bbox_w))
    if short_side == 0:
        return None
    ratio = long_side / float(short_side)
    if ratio < ratio_threshold:
        return "point"
    return "horizontal" if int(bbox_w) > int(bbox_h) else "vertical"


def compute_attached_side(
    frame:         np.ndarray,
    bbox:          Tuple[int, int, int, int],
    passable_grid: Optional[np.ndarray] = None,
    *,
    dominance_margin: float = 0.5,
) -> Optional[str]:
    """Derive which side of an entity's bbox is fused to a wall.

    Counts, for each of the four sides of the bbox, how many of the
    bbox-edge pixels have a NON-passable neighbour immediately outside
    the bbox.  The side with the highest count, by a margin of
    ``dominance_margin`` over the runner-up, is the attached side.

    Returns ``"N"`` / ``"S"`` / ``"E"`` / ``"W"`` for a clear winner,
    ``None`` when no side dominates (the sprite floats in open space,
    or two sides tie -- e.g. a bar in the middle of a corridor with
    walls on both sides).

    Args:
        frame: HxWxC frame array (only used for shape bounds).
        bbox:  ``(r0, c0, r1, c1)`` inclusive pixel-space bbox.
        passable_grid: optional HxW bool array marking passable
            pixels (True = floor).  When ``None``, no attached side
            can be inferred -- returns ``None``.
        dominance_margin: required ratio of the winning side's count
            over the runner-up (1.5 = winner has 1.5x the runner-up).
            Lower means more tolerant of close ties.
    """
    if passable_grid is None:
        return None
    try:
        H, W = passable_grid.shape[:2]
    except (AttributeError, ValueError):
        return None
    try:
        r0, c0, r1, c1 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    except (TypeError, ValueError, IndexError):
        return None
    if r0 > r1 or c0 > c1:
        return None
    # For each side, count pixels just OUTSIDE the bbox along that
    # edge that are NON-passable.  More non-passable neighbours →
    # more wall contact on that side.
    counts = {"N": 0, "S": 0, "E": 0, "W": 0}
    if r0 - 1 >= 0:
        row = passable_grid[r0 - 1, max(0, c0):min(W, c1 + 1)]
        counts["N"] = int((~row).sum())
    if r1 + 1 < H:
        row = passable_grid[r1 + 1, max(0, c0):min(W, c1 + 1)]
        counts["S"] = int((~row).sum())
    if c0 - 1 >= 0:
        col = passable_grid[max(0, r0):min(H, r1 + 1), c0 - 1]
        counts["W"] = int((~col).sum())
    if c1 + 1 < W:
        col = passable_grid[max(0, r0):min(H, r1 + 1), c1 + 1]
        counts["E"] = int((~col).sum())
    if max(counts.values()) == 0:
        return None
    # Sort by count descending; need clear winner.
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    top, top_count = ranked[0]
    runner_up_count = ranked[1][1] if len(ranked) > 1 else 0
    if runner_up_count > 0:
        ratio = top_count / float(runner_up_count)
        if ratio < (1.0 + float(dominance_margin)):
            return None
    return top
