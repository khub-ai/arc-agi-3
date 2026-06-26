"""Cell-grid adapter — turns a raw frame palette and an entity list
into the per-cell role map the mechanic-aware planner consumes.

This module bridges perception output (palette ints, entity bboxes,
role tags) and the planner's structural needs (a 2D grid where each
cell is classified as walkable / wall / consumable / agent /
unknown).  Both kinds of input are already produced by the
perception / curiosity layers; this adapter just composes them.

No game-specific role assumptions are baked in.  Callers supply the
``walkable_palettes`` and ``consumable_roles`` sets they've learned
from their game; the adapter applies them generically.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, FrozenSet, Iterable, Mapping, Optional, Sequence, Set, Tuple,
)

import numpy as np


# Closed role vocabulary used in the produced cell grid.  Matches the
# substrate-level vocabulary in :mod:`cognitive_os.mechanic_catalog`.
ROLE_WALKABLE   = "walkable"
ROLE_WALL       = "wall"
ROLE_CONSUMABLE = "consumable"
ROLE_AGENT      = "agent"
ROLE_UNKNOWN    = "unknown"


@dataclass
class CellGrid:
    """Per-cell role map plus an associated passability grid.

    Attributes
    ----------
    roles
        2D ``np.array`` of dtype object, with one of the
        ``ROLE_*`` strings per cell.
    passable
        2D bool ``np.array``.  True where the agent can step
        without considering consumes.  Equivalent to
        ``(roles == ROLE_WALKABLE) | (roles == ROLE_AGENT)``.
    passable_if_all_consumed
        2D bool ``np.array``.  True where the agent could step
        if every consumable were already consumed.  The planner
        uses this for "consume-permeable" path queries.
    consumable_entities
        Entity dicts (the original input) whose ``role`` matched the
        ``consumable_roles`` set.  Carried alongside the grid for
        downstream consume action planning.
    shape
        ``(H, W)`` of the cell grid.
    """

    roles:                    np.ndarray
    passable:                 np.ndarray
    passable_if_all_consumed: np.ndarray
    consumable_entities:      Sequence[Mapping[str, Any]]
    shape:                    Tuple[int, int]


def build_cell_grid(
    frame:               np.ndarray,
    entities:            Sequence[Mapping[str, Any]],
    walkable_palettes:   Iterable[int],
    consumable_roles:    Iterable[str],
    *,
    wall_roles:          Iterable[str] = (
        "wall", "void", "divider", "hud_background", "agent_avatar",
    ),
    agent_role:          str           = "agent_avatar",
) -> CellGrid:
    """Build a :class:`CellGrid` from a frame and entity list.

    Classification precedence (highest to lowest):

    1. Agent — cells overlapped by an entity whose ``role`` matches
       ``agent_role``.  The agent's footprint is marked ``ROLE_AGENT``
       but treated as walkable in the passability grid (the agent
       occupies these cells but they become walkable on step-off).
    2. Consumable — cells overlapped by entities whose ``role`` is
       in ``consumable_roles``.
    3. Wall — cells overlapped by entities whose ``role`` is in
       ``wall_roles``.
    4. Walkable — remaining cells whose palette is in
       ``walkable_palettes``.
    5. Wall — anything else.

    ``wall_roles`` defaults to the same role set the existing planner
    uses (see ``curiosity_planner.WALL_ROLES``).  Other roles seen in
    the entity list that don't match any category fall through to
    the palette-based classification.
    """
    arr = np.asarray(frame)
    if arr.ndim != 2:
        raise ValueError(
            f"build_cell_grid expects a 2D frame; got shape {arr.shape}"
        )
    H, W = arr.shape

    walkable_set: Set[int] = {int(p) for p in walkable_palettes}
    consumable_set: Set[str] = {str(r) for r in consumable_roles}
    wall_set:       Set[str] = {str(r) for r in wall_roles}
    agent_role_str          = str(agent_role)

    # Layer 1: palette-based base classification.
    if walkable_set:
        base_walkable = np.isin(arr, list(walkable_set))
    else:
        base_walkable = np.ones((H, W), dtype=bool)
    roles = np.full((H, W), ROLE_WALL, dtype=object)
    roles[base_walkable] = ROLE_WALKABLE

    # Layer 2: entity overlay.  Order matters — process walls,
    # then consumables, then agent, so agent-on-consumable shows
    # as agent (the highest precedence).
    consumable_ents: list = []
    for ent in entities:
        role = str(ent.get("role") or "").strip().lower()
        if not role:
            continue
        bb = _bbox_pixels(ent)
        if bb is None:
            continue
        r0, c0, r1, c1 = bb
        r0 = max(0, min(H - 1, r0)); r1 = max(0, min(H - 1, r1))
        c0 = max(0, min(W - 1, c0)); c1 = max(0, min(W - 1, c1))
        # Skip degenerate bboxes.
        if r1 < r0 or c1 < c0:
            continue
        if role in wall_set and role != agent_role_str:
            roles[r0:r1 + 1, c0:c1 + 1] = ROLE_WALL
        elif role in consumable_set:
            roles[r0:r1 + 1, c0:c1 + 1] = ROLE_CONSUMABLE
            consumable_ents.append(ent)

    # Layer 3: agent (highest precedence).
    for ent in entities:
        role = str(ent.get("role") or "").strip().lower()
        if role != agent_role_str:
            continue
        bb = _bbox_pixels(ent)
        if bb is None:
            continue
        r0, c0, r1, c1 = bb
        r0 = max(0, min(H - 1, r0)); r1 = max(0, min(H - 1, r1))
        c0 = max(0, min(W - 1, c0)); c1 = max(0, min(W - 1, c1))
        if r1 < r0 or c1 < c0:
            continue
        roles[r0:r1 + 1, c0:c1 + 1] = ROLE_AGENT

    # Derived grids.
    passable = (roles == ROLE_WALKABLE) | (roles == ROLE_AGENT)
    passable_if_all_consumed = passable | (roles == ROLE_CONSUMABLE)

    return CellGrid(
        roles                    = roles,
        passable                 = passable,
        passable_if_all_consumed = passable_if_all_consumed,
        consumable_entities      = tuple(consumable_ents),
        shape                    = (H, W),
    )


def _bbox_pixels(ent: Mapping[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    bb = ent.get("bbox_pixels") or []
    if len(bb) < 4:
        return None
    try:
        return (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
    except (TypeError, ValueError):
        return None


def compute_reachable_mask(
    passable:    np.ndarray,
    start_rc:    Tuple[int, int],
) -> np.ndarray:
    """4-connected BFS reachability from ``start_rc`` over a bool
    ``passable`` grid.  Returns a same-shape bool mask: True where
    the start cell can reach without crossing impassable cells.

    Used to constrain "is this consume target useful?" decisions:
    a consumable adjacent to the reachable mask can be opened by
    one click; one separated from the mask by other walls / other
    consumables cannot, regardless of which axis it sits on.
    """
    H, W = passable.shape
    out = np.zeros((H, W), dtype=bool)
    sr, sc = int(start_rc[0]), int(start_rc[1])
    if sr < 0 or sr >= H or sc < 0 or sc >= W:
        return out
    if not bool(passable[sr, sc]):
        # Start cell is not passable -- treat as if the agent
        # occupies a passable cell anyway (the agent's own
        # footprint is masked into passable=True in the cell
        # grid).  If we still hit False, return empty.
        return out
    # Stack-based flood fill (cheaper than allocating a deque for
    # 64x64 grids).
    stack = [(sr, sc)]
    out[sr, sc] = True
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W:
                if passable[nr, nc] and not out[nr, nc]:
                    out[nr, nc] = True
                    stack.append((nr, nc))
    return out


def entity_adjacent_to_mask(
    entity_bbox:  Sequence[int],
    mask:         np.ndarray,
) -> bool:
    """True when any cell in the 1-cell border around the entity's
    bbox is marked True in ``mask``.  4-connected adjacency.

    Use to test "if I consume this entity, does that open a cell
    that's adjacent to the agent's current reachable region?"  When
    False, consuming the entity leaves the reachable region
    unchanged; the consume is wasted from the current agent
    position.
    """
    if entity_bbox is None or len(entity_bbox) < 4:
        return False
    H, W = mask.shape
    r0, c0, r1, c1 = (int(entity_bbox[0]), int(entity_bbox[1]),
                       int(entity_bbox[2]), int(entity_bbox[3]))
    # Check the 4 borders of the bbox (one cell out, clamped).
    # Top/bottom rows
    for c in range(max(0, c0), min(W, c1 + 1)):
        if r0 - 1 >= 0 and mask[r0 - 1, c]:
            return True
        if r1 + 1 < H and mask[r1 + 1, c]:
            return True
    # Left/right cols
    for r in range(max(0, r0), min(H, r1 + 1)):
        if c0 - 1 >= 0 and mask[r, c0 - 1]:
            return True
        if c1 + 1 < W and mask[r, c1 + 1]:
            return True
    return False
