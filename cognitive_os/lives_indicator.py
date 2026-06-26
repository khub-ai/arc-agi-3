"""Lives-indicator detector — count remaining player lives from a frame.

Domain-agnostic primitive: arcade-style games typically draw a strip
of small icons in a corner of the frame, one per remaining life.
When a life is lost, an icon flips from "alive" to "dead" — usually
by changing palette (color) or disappearing entirely.  The lives
strip is the SOURCE OF TRUTH for life-loss events that the
framework's `state` field doesn't surface (per-life respawns inside
``state=NOT_FINISHED`` are otherwise invisible).

Why this matters
----------------
Prior to this module the harness used a frame-pixel-diff heuristic
(``diff_pixels >= max(50, 3 * cell_pixels)``) to infer "level reset
happened, agent must have died".  That heuristic produces false
positives on legitimate large-diff steps (pickup-step + meter
refill + sprite migration easily exceeds the threshold) and false
negatives in games where death produces only a localised visual
change.  The operator's 2026-05-05 directive: stop using
frame-pixel-delta for life-loss classification; use the visible
lives counter directly, the way a human player reads the screen.

Detection algorithm
-------------------
1. Group every component in the frame by ``(palette, bbox_h, bbox_w)``.
2. Filter to small icons (max axis ≤ 6 pixels) with ≥ 2 members.
3. Require row-aligned OR column-aligned within ±2 pixels.
4. Require all members within an edge-band of some frame edge.
5. Among surviving candidates, prefer the one with the MOST members
   (typical arcade lives strips are 3–5 icons; smaller groups are
   more likely to be unrelated UI dots).

The detector is conservative: false positives (mis-identifying a
non-indicator) corrupt life-loss attribution silently, so the
heuristic ranks "structurally lives-shaped" candidates first and
returns ``None`` when no clear winner exists.

Counting (per frame)
--------------------
Once the indicator is fixed at session start (its bbox + alive
palette signature), each subsequent frame counts how many alive-
palette pixels appear in the bbox and divides by the icon size.
A drop = a life loss.  The post-detection count is always
clamped to [0, initial_count].
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class LivesIndicator:
    """A detected lives-indicator's geometry and palette signature.

    Fields:
      region_bbox:    (r_min, c_min, r_max, c_max) inclusive pixel rect.
      alive_palette:  palette value of an "alive" icon at session start.
      icon_size:      pixel-count of one alive icon (used as divisor).
      initial_count:  number of alive icons at first detection.
      edge:           which frame edge the indicator hugs ('top'/'bottom'/
                      'left'/'right') — diagnostic only, not behavioural.
    """
    region_bbox:    Tuple[int, int, int, int]
    alive_palette:  int
    icon_size:      int
    initial_count:  int
    edge:           str


def detect_lives_indicator(
    comps:       List[dict],
    frame_shape: Tuple[int, int],
) -> "Optional[LivesIndicator]":
    """Find a strip of identical small components aligned near a frame edge.

    Args:
      comps:       Components extracted from the frame (each dict has
                   ``palette``, ``size``, ``extent=[h, w]``,
                   ``centroid=[r, c]``).  Caller filters out the agent
                   sprite if desired (this module doesn't know which
                   component is the agent).
      frame_shape: ``(H, W)``.

    Returns:
      A populated :class:`LivesIndicator`, or ``None`` if no
      structurally-lives-shaped candidate found.
    """
    if not comps:
        return None
    H, W = int(frame_shape[0]), int(frame_shape[1])
    edge_band = max(4, min(H, W) // 6)

    groups: "dict[tuple, list[dict]]" = defaultdict(list)
    for c in comps:
        ext = c.get("extent") or [0, 0]
        try:
            eh, ew = int(ext[0]), int(ext[1])
        except (TypeError, ValueError, IndexError):
            continue
        if eh < 1 or ew < 1:
            continue
        if max(eh, ew) > 6:
            continue   # too large to be a lives icon
        sz = int(c.get("size", 0) or 0)
        if sz < 2:
            continue
        try:
            pal = int(c.get("palette", -1))
        except (TypeError, ValueError):
            continue
        groups[(pal, eh, ew)].append(c)

    candidates: "list[LivesIndicator]" = []
    for (pal, eh, ew), members in groups.items():
        if len(members) < 2:
            continue
        try:
            rows = [int(m["centroid"][0]) for m in members]
            cols = [int(m["centroid"][1]) for m in members]
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        row_aligned = (max(rows) - min(rows)) <= 2
        col_aligned = (max(cols) - min(cols)) <= 2
        if not (row_aligned or col_aligned):
            continue
        # Edge-residency: ALL members within edge_band of some edge.
        edge_dists = []
        for r, c in zip(rows, cols):
            edge_dists.append(min(r, H - 1 - r, c, W - 1 - c))
        if max(edge_dists) > edge_band:
            continue
        # Determine which edge.
        avg_r = sum(rows) / len(rows)
        avg_c = sum(cols) / len(cols)
        edge_name = min(
            (("top",    avg_r),
             ("bottom", H - 1 - avg_r),
             ("left",   avg_c),
             ("right",  W - 1 - avg_c)),
            key=lambda kv: kv[1],
        )[0]
        # Compute the bbox covering all members (with some padding for
        # aliasing).  Use centroid ± half-extent as an approximation.
        rs = []
        cs = []
        for m in members:
            cr, cc = int(m["centroid"][0]), int(m["centroid"][1])
            rs.extend([cr - eh, cr + eh])
            cs.extend([cc - ew, cc + ew])
        bbox = (
            max(0,     min(rs)),
            max(0,     min(cs)),
            min(H - 1, max(rs)),
            min(W - 1, max(cs)),
        )
        candidates.append(LivesIndicator(
            region_bbox    = bbox,
            alive_palette  = int(pal),
            icon_size      = int(eh) * int(ew),
            initial_count  = len(members),
            edge           = edge_name,
        ))

    if not candidates:
        return None
    # Prefer the candidate with the MOST members (lives indicators are
    # typically 3–5; a 2-member candidate is more likely to be
    # something else).  Tie-break by smaller icon size (lives icons
    # are typically the smallest UI elements).
    candidates.sort(key=lambda li: (-li.initial_count, li.icon_size))
    return candidates[0]


def count_alive_lives(
    indicator: LivesIndicator,
    frame,
) -> int:
    """Count visible 'alive' icons in the indicator region.

    Strategy: count alive-palette pixels in the indicator bbox and
    divide by icon_size.  Conservative — clamps to ``[0, initial_count]``.

    The frame is treated as a 2D integer-palette numpy array; this
    function does not import numpy unconditionally (so the module
    itself remains importable in tests that don't need it), but
    expects the typical numpy semantics for ``frame[bbox] == palette``.
    """
    try:
        import numpy as _np
    except ImportError:
        return indicator.initial_count
    r0, c0, r1, c1 = indicator.region_bbox
    try:
        region = frame[r0:r1 + 1, c0:c1 + 1]
        alive_pixels = int(_np.sum(region == indicator.alive_palette))
    except Exception:
        return indicator.initial_count
    if indicator.icon_size <= 0:
        return 0
    n = alive_pixels // int(indicator.icon_size)
    return max(0, min(int(indicator.initial_count), n))


def lives_decremented(
    indicator: LivesIndicator,
    pre_frame,
    post_frame,
) -> bool:
    """Return True iff ``post_frame`` shows fewer alive lives than ``pre_frame``.

    The principled "did the agent just lose a life?" signal: replaces
    the prior frame-pixel-delta heuristic.  Robust against both per-
    life respawns inside ``NOT_FINISHED`` and the final GAME_OVER
    transition.

    Returns False if ``indicator`` is ``None`` — caller should fall
    back to other signals or skip per-life classification.
    """
    if indicator is None:
        return False
    pre_n  = count_alive_lives(indicator, pre_frame)
    post_n = count_alive_lives(indicator, post_frame)
    return post_n < pre_n
