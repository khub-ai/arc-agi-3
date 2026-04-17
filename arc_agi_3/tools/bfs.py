"""Breadth-first search on a 2-D integer grid.

The adapter exposes this as the ``grid.bfs`` tool.  Miners use it to
confirm or refute ``TransitionClaim``s ("can the agent reach (r,c)
from its current position under the current passability rules?"); the
planner uses it as a cost oracle when composing plans over movement
transitions; the explorer uses it to score curiosity goals by
reachability.

Implementation is deliberately explicit: no NumPy dependency (the
grids are at most 60×60 for ARC-AGI-3 so a Python deque is faster
than array construction).  Passability is supplied as a callable so
the same BFS can serve both "open-terrain" queries and
"avoid-entity-at-cell" queries.
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]          # (row, col)
Grid = Sequence[Sequence[int]]
Passable = Callable[[int, int, int], bool]   # (row, col, value) → bool


# 4-connectivity only.  8-connectivity is a deliberate non-default:
# in most ARC-style domains diagonal moves are not a valid action,
# and wrong adjacency assumptions corrupt downstream reachability
# claims.  Callers that genuinely want 8-connectivity can pass
# ``neighbours`` explicitly.
_NEIGHBOURS_4: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


def shortest_path(
    grid:        Grid,
    start:       Cell,
    goal:        Cell,
    *,
    passable:    Optional[Passable] = None,
    neighbours:  Sequence[Cell]     = _NEIGHBOURS_4,
) -> Optional[List[Cell]]:
    """Return the shortest 4-connected path from ``start`` to ``goal``.

    Returns the full cell sequence including both endpoints, or
    ``None`` if ``goal`` is unreachable under the given ``passable``
    predicate.  ``passable(r, c, grid[r][c])`` is queried for every
    candidate neighbour; the default (``lambda r, c, v: True``) treats
    every in-bounds cell as traversable, which is appropriate for
    pure reachability probes on featureless grids.

    Parameters
    ----------
    grid
        2-D integer grid (list of lists or tuple of tuples).
    start, goal
        ``(row, col)`` integer tuples.  Both must be in bounds and
        themselves pass ``passable``; otherwise ``None`` is returned
        immediately.
    passable
        Predicate ``(row, col, cell_value) -> bool``.
    neighbours
        Offsets considered for each step.  Defaults to 4-connected.
    """
    if not grid or not grid[0]:
        return None
    h, w = len(grid), len(grid[0])
    if not _in_bounds(start, h, w) or not _in_bounds(goal, h, w):
        return None

    if passable is None:
        passable = _always_passable
    if not passable(start[0], start[1], grid[start[0]][start[1]]):
        return None
    if not passable(goal[0], goal[1], grid[goal[0]][goal[1]]):
        return None
    if start == goal:
        return [start]

    parent: Dict[Cell, Optional[Cell]] = {start: None}
    queue: deque[Cell] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbours:
            nr, nc = r + dr, c + dc
            nxt: Cell = (nr, nc)
            if nxt in parent:
                continue
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not passable(nr, nc, grid[nr][nc]):
                continue
            parent[nxt] = (r, c)
            if nxt == goal:
                return _reconstruct(parent, goal)
            queue.append(nxt)
    return None


def reachable_cells(
    grid:     Grid,
    start:    Cell,
    *,
    passable: Optional[Passable] = None,
) -> List[Cell]:
    """Return every cell reachable from ``start`` under ``passable``.

    Useful for bulk reachability claims ("which cells can the agent
    currently touch?") rather than single-goal queries.  Output order
    is BFS traversal order (so the first element is always ``start``).
    """
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    if not _in_bounds(start, h, w):
        return []
    if passable is None:
        passable = _always_passable
    if not passable(start[0], start[1], grid[start[0]][start[1]]):
        return []

    seen = {start}
    order = [start]
    queue: deque[Cell] = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in _NEIGHBOURS_4:
            nr, nc = r + dr, c + dc
            nxt: Cell = (nr, nc)
            if nxt in seen:
                continue
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not passable(nr, nc, grid[nr][nc]):
                continue
            seen.add(nxt)
            order.append(nxt)
            queue.append(nxt)
    return order


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _always_passable(_r: int, _c: int, _v: int) -> bool:
    return True


def _in_bounds(cell: Cell, h: int, w: int) -> bool:
    r, c = cell
    return 0 <= r < h and 0 <= c < w


def _reconstruct(parent: Dict[Cell, Optional[Cell]], goal: Cell) -> List[Cell]:
    path: List[Cell] = [goal]
    cur: Optional[Cell] = parent[goal]
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path
