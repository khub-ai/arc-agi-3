"""First-class frame-diff capability.

Background
----------

Every learning mechanism in the engine — action-effect mining, entity
identity tracking, generic surprise detection, causal inference — needs
to answer the question *"what changed between the previous observation
and this one?"*.  Without that answer the agent is blind to the
consequences of its own actions and there is no hope of inducing a
world model from experience.

This module provides that answer, **once per step**, in a structured
form every subsystem can consume.

Architectural rule revisited
----------------------------

``types.Observation`` carries the comment *"the engine MUST NOT inspect
``raw_frame`` — only hand it to the Observer."*  That rule guards
against the engine developing domain-specific understanding of pixel
values, point-cloud coordinates, or ROS-message payloads.  It does
**not** prohibit the engine from asking a purely structural question:
*which cells differ?*  Equality comparison treats cell values as opaque
tokens; no semantics are inferred.

This module implements only that structural question.  A cell with
``colour == 9`` that became ``colour == 0`` is reported as a change
from ``9`` to ``0``; the engine does not care that ``9`` was "agent"
and ``0`` was "floor" — that interpretation belongs to the Observer
and the miners downstream.

Input shape
-----------

A "frame" is any :class:`Sequence[Sequence[T]]` — a 2-D rectangular
grid of hashable, equality-comparable values.  This shape covers:

* ARC-AGI grids (``Sequence[Sequence[int]]`` of colour IDs)
* Gridworld robotics adapters
* State vectors re-tupled as ``1 × N``
* Depth images discretised to bucket indices

For exotic frame types (raw bytes, opaque blobs, continuous state
unsuitable for cell-level equality), :func:`compute_frame_delta`
returns ``None`` and the caller falls back to whatever alternate
observability signal the adapter provides.

Capability audit
----------------
* **Problem-solving** — PRIMARY.  Without frame-diff the planner has
  no ground-truth signal that its actions produced an effect.
* **Debugging** — PRIMARY.  ``FrameDelta`` is the cleanest input to
  surprise detection and to (action, effect) miner.
* **Tool creation** — secondary.  The regions in a ``FrameDelta``
  become candidate inputs for Option synthesis (a region that
  consistently appears after action X is a candidate landmark).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


Cell = Tuple[int, int]                  # (row, col)
Grid = Sequence[Sequence[Any]]          # rectangular grid of hashable values


class FrameShapeMismatch(ValueError):
    """Raised when two frames have different shapes.

    The engine does not silently recover from this — a shape change
    mid-episode is itself an observation the adapter should surface,
    and hiding it here would mask real bugs in the adapter's
    observation pipeline.
    """


@dataclass(frozen=True)
class CellChange:
    """One cell whose value differed between two consecutive frames."""
    row:    int
    col:    int
    before: Any
    after:  Any


@dataclass(frozen=True)
class DeltaRegion:
    """A 4-connected clump of cells that all changed in the same step.

    ``bbox`` is ``(row_min, col_min, row_max, col_max)`` inclusive.

    ``dominant_before`` / ``dominant_after`` are the most frequent
    before- and after-values across the region's cells.  They let
    downstream consumers treat a region as a colour transition
    (``9 → 0``) without iterating every cell.
    """
    cells:           Tuple[Cell, ...]
    bbox:            Tuple[int, int, int, int]
    dominant_before: Any
    dominant_after:  Any


@dataclass(frozen=True)
class FrameDelta:
    """Structural description of what changed between two frames.

    Populated once per step on :attr:`WorldState.last_frame_delta`.

    Invariants
    ----------
    * ``cells_changed == len(changed_cells) == len(before_values) == len(after_values)``
    * ``bbox`` is ``None`` iff ``cells_changed == 0``.
    * ``regions`` partitions ``changed_cells`` — every changed cell
      belongs to exactly one region.

    The engine never interprets ``before_values`` / ``after_values``
    semantically; consumers (miners, observer-queries) do.
    """
    changed_cells:  Tuple[Cell, ...]
    before_values:  Tuple[Any, ...]
    after_values:   Tuple[Any, ...]
    bbox:           Optional[Tuple[int, int, int, int]]
    cells_changed:  int
    regions:        Tuple[DeltaRegion, ...]

    @classmethod
    def empty(cls) -> "FrameDelta":
        """An explicit "nothing changed" delta (distinct from ``None``,
        which means "delta unavailable")."""
        return cls((), (), (), None, 0, ())

    @property
    def is_empty(self) -> bool:
        return self.cells_changed == 0


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def compute_frame_delta(
    pre:  Optional[Grid],
    post: Optional[Grid],
) -> Optional[FrameDelta]:
    """Return a structured diff of two consecutive frames.

    Returns
    -------
    * ``None`` — delta is unavailable.  This happens when either frame
      is ``None`` (typical on the first step of an episode before any
      prior observation exists), or when the frames are not shaped
      like a 2-D grid.
    * :meth:`FrameDelta.empty` — frames are identical.  Distinct from
      ``None``; consumers can use this to assert "the action had no
      observable effect on the frame" with confidence.
    * :class:`FrameDelta` — at least one cell differed.

    Raises
    ------
    FrameShapeMismatch
        When both frames are grid-shaped but have different
        row-counts or row-widths.  The adapter is responsible for
        reporting grid-size changes as observations; masking them
        here would hide real bugs.

    Complexity
    ----------
    O(R·C) where R·C is the frame area.  No allocations for identical
    frames beyond the empty-delta singleton.
    """
    if pre is None or post is None:
        return None
    if not _is_grid(pre) or not _is_grid(post):
        return None

    pre_rows  = len(pre)
    post_rows = len(post)
    if pre_rows != post_rows:
        raise FrameShapeMismatch(
            f"row count differs: pre={pre_rows} post={post_rows}"
        )
    if pre_rows == 0:
        return FrameDelta.empty()

    pre_cols  = len(pre[0])
    post_cols = len(post[0])
    if pre_cols != post_cols:
        raise FrameShapeMismatch(
            f"col count differs: pre={pre_cols} post={post_cols}"
        )

    # Single pass: collect per-cell changes and track bbox.
    changed: List[Cell] = []
    before_vals: List[Any] = []
    after_vals:  List[Any] = []
    r_min = c_min =  10**9
    r_max = c_max = -1

    for r, (row_pre, row_post) in enumerate(zip(pre, post)):
        if len(row_pre) != pre_cols or len(row_post) != post_cols:
            raise FrameShapeMismatch(
                f"row {r} has inconsistent width"
            )
        for c, (vb, va) in enumerate(zip(row_pre, row_post)):
            if vb != va:
                changed.append((r, c))
                before_vals.append(vb)
                after_vals.append(va)
                if r < r_min: r_min = r
                if c < c_min: c_min = c
                if r > r_max: r_max = r
                if c > c_max: c_max = c

    if not changed:
        return FrameDelta.empty()

    bbox = (r_min, c_min, r_max, c_max)
    regions = _group_connected(changed, before_vals, after_vals)

    return FrameDelta(
        changed_cells = tuple(changed),
        before_values = tuple(before_vals),
        after_values  = tuple(after_vals),
        bbox          = bbox,
        cells_changed = len(changed),
        regions       = regions,
    )


# ---------------------------------------------------------------------------
# Region-level motion extraction
# ---------------------------------------------------------------------------


def extract_region_motion(
    region: DeltaRegion,
    delta:  FrameDelta,
) -> Optional[Tuple[Any, Any, int, int]]:
    """Try to interpret ``region`` as a translation of a sprite.

    A translation is the pattern where a coloured sprite moved from
    one place to another in-frame: some cells lost a sprite colour
    (``sprite-colour → background``) while a congruent set of cells
    gained one (``background → sprite-colour``).  This helper tells
    those two cell groups apart and returns the direction the sprite
    moved, as an *abstract sign-only* vector.

    The sprite may be multi-coloured.  What must hold is that a
    single colour serves as the **background** — present on at least
    one side of *every* (before, after) transition pair in the
    region.  The complement colours are the sprite's palette; the
    departing cells (``sprite → bg``) form the sprite's old silhouette,
    the arriving cells (``bg → sprite``) form its new silhouette.

    Returns
    -------
    ``None``
        If the region's cell-level before/after pattern cannot be
        parsed as a translation — e.g. no single colour appears on
        one side of every pair (a genuine recolouring), only one
        direction of transitions (pure appearance or disappearance),
        or both centroids coincide (a symmetric exchange that is
        not a motion).
    ``(colour, background, dr_sign, dc_sign)``
        * ``colour``      — the sprite's **dominant** colour (the
                            colour that appears in the most
                            sprite-cells).  Ties broken by the most
                            frequent colour in the overall region.
                            Using the dominant colour keeps the
                            canonical key stable across episodes
                            without forcing a frozenset of colours
                            into the key.
        * ``background``  — the colour that plays the "empty" role
                            on both departing and arriving cells.
        * ``dr_sign``, ``dc_sign`` — the sign of
                            ``centroid(arriving) - centroid(departing)``,
                            each one of ``-1, 0, +1``.  Sign only
                            so the claim transfers across starting
                            positions.

    Complexity
    ----------
    O(|region.cells|) plus a constant-time lookup per cell against the
    parallel arrays on ``delta``.  For ARC frames this is negligible.
    """
    cells = getattr(region, "cells", ()) or ()
    if len(cells) < 2:
        return None

    # Build an index from delta.changed_cells -> position in parallel
    # arrays.  Done lazily to stay O(changed cells) per delta per
    # miner-pass; for realistic frames the constant overhead is fine.
    index = {cell: i for i, cell in enumerate(delta.changed_cells)}

    # Partition region cells by (before, after) pair, and record
    # per-pair cell counts so we can pick a dominant sprite colour.
    groups: dict = {}
    for cell in cells:
        i = index.get(cell)
        if i is None:
            return None
        pair = (delta.before_values[i], delta.after_values[i])
        groups.setdefault(pair, []).append(cell)

    if not groups:
        return None

    # Identify the background: a colour that appears on at least one
    # side of *every* transition pair.  For a translated sprite, that
    # colour is the surrounding empty colour; pairs are either
    # (sprite_c -> bg)  — departing, or
    # (bg -> sprite_c)  — arriving.
    #
    # If no single colour satisfies this, the region is not a pure
    # translation (e.g. it's a recolouring that involves no empty
    # cells).
    pair_keys = list(groups.keys())
    # Colours that appear on one side of every pair:
    candidates = set(pair_keys[0])
    for (a, b) in pair_keys[1:]:
        candidates &= {a, b}
        if not candidates:
            return None

    # Tie-break: prefer the candidate that is the most frequent value
    # across the region (summed over both before and after arrays).
    # This keeps the labelling deterministic when multiple colours
    # qualify (rare, but possible for very symmetric recolourings).
    def _count_in_region(colour: Any) -> int:
        n = 0
        for cell in cells:
            i = index[cell]
            if delta.before_values[i] == colour: n += 1
            if delta.after_values[i]  == colour: n += 1
        return n
    bg = max(candidates, key=_count_in_region)

    # Partition cells into departing (sprite -> bg) and arriving
    # (bg -> sprite).  Any pair that doesn't touch bg at all would
    # have already disqualified bg, so we expect every pair to match
    # exactly one of these roles.
    departing: List[Cell] = []
    arriving:  List[Cell] = []
    sprite_counts: dict = {}          # colour -> cell count (total)
    for (a, b), member_cells in groups.items():
        if a != bg and b == bg:
            # sprite colour `a` departed its cells
            departing.extend(member_cells)
            sprite_counts[a] = sprite_counts.get(a, 0) + len(member_cells)
        elif a == bg and b != bg:
            arriving.extend(member_cells)
            sprite_counts[b] = sprite_counts.get(b, 0) + len(member_cells)
        elif a == bg and b == bg:
            # Shouldn't happen (no-op cells wouldn't be in the delta),
            # but guard defensively.
            continue
        else:
            # Pair (non-bg -> non-bg) — recolouring.  Disqualifies a
            # pure translation reading.
            return None

    if not departing or not arriving:
        # Pure appearance or pure disappearance — not a translation.
        return None

    # Dominant sprite colour: most cells overall.  Ties broken by
    # most frequent occurrence in the region (which is just
    # sprite_counts again, but we can also break ties by a stable
    # ordering on the hashable form of the colour to keep output
    # deterministic).
    def _dom_key(item):
        c, n = item
        try:
            return (-n, _hashable(c))
        except TypeError:
            return (-n, repr(c))
    dominant = sorted(sprite_counts.items(), key=_dom_key)[0][0]

    # Centroids (float; we only use the sign of the difference).
    def _centroid(pts):
        rs = cs = 0
        n  = len(pts)
        for (r, c) in pts:
            rs += r
            cs += c
        return (rs / n, cs / n)

    (r_new, c_new) = _centroid(arriving)
    (r_old, c_old) = _centroid(departing)
    dr = r_new - r_old
    dc = c_new - c_old

    def _sign(x: float) -> int:
        if x > 0: return 1
        if x < 0: return -1
        return 0

    dr_sign = _sign(dr)
    dc_sign = _sign(dc)
    if dr_sign == 0 and dc_sign == 0:
        # Coincident centroids — a symmetric exchange, not a motion.
        return None

    return (dominant, bg, dr_sign, dc_sign)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _is_grid(obj: Any) -> bool:
    """Duck-type check: ``obj`` is a sequence of sequences.

    Intentionally strict: strings look like sequences-of-sequences but
    treating them as frames would surprise the caller.
    """
    if isinstance(obj, (str, bytes, bytearray)):
        return False
    try:
        first = next(iter(obj))
    except (TypeError, StopIteration):
        # Empty grid is still a grid (of zero rows); callers get an
        # empty delta.  Non-iterable is not a grid.
        return hasattr(obj, "__len__")
    if isinstance(first, (str, bytes, bytearray)):
        return False
    try:
        iter(first)
    except TypeError:
        return False
    return True


def _group_connected(
    cells:        List[Cell],
    before_vals:  List[Any],
    after_vals:   List[Any],
) -> Tuple[DeltaRegion, ...]:
    """Partition changed cells into 4-connected regions.

    Uses a simple union-find over the cell set.  For the frame sizes
    we care about (ARC: 64×64, robotics gridworlds: comparable) the
    O(N·α(N)) cost is negligible compared with the cell-diff scan
    that preceded it.
    """
    if not cells:
        return ()

    idx = {cell: i for i, cell in enumerate(cells)}
    parent = list(range(len(cells)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, (r, c) in enumerate(cells):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            j = idx.get((r + dr, c + dc))
            if j is not None:
                union(i, j)

    # Bucket by root.
    buckets: dict = {}
    for i in range(len(cells)):
        buckets.setdefault(find(i), []).append(i)

    regions = []
    for members in buckets.values():
        region_cells = tuple(cells[i] for i in members)
        r_min = min(c[0] for c in region_cells)
        c_min = min(c[1] for c in region_cells)
        r_max = max(c[0] for c in region_cells)
        c_max = max(c[1] for c in region_cells)
        dom_before = _mode(before_vals[i] for i in members)
        dom_after  = _mode(after_vals[i]  for i in members)
        regions.append(DeltaRegion(
            cells           = region_cells,
            bbox            = (r_min, c_min, r_max, c_max),
            dominant_before = dom_before,
            dominant_after  = dom_after,
        ))
    # Stable order: by bbox top-left.
    regions.sort(key=lambda reg: (reg.bbox[0], reg.bbox[1]))
    return tuple(regions)


def _mode(values) -> Any:
    """Most frequent value; ties broken by first occurrence."""
    counts: dict = {}
    order: List[Any] = []
    for v in values:
        key = _hashable(v)
        if key not in counts:
            counts[key] = [0, v]
            order.append(key)
        counts[key][0] += 1
    best_key = max(order, key=lambda k: counts[k][0])
    return counts[best_key][1]


def _hashable(v: Any) -> Any:
    """Coerce to a hashable representation for counting.

    Protects against frames of lists-of-lists (unhashable) without
    forcing callers to convert up-front.
    """
    try:
        hash(v)
        return v
    except TypeError:
        return repr(v)
