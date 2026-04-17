"""Connected-component labelling on 2-D integer grids.

Two cells are in the same component when they share a colour value
AND are 4-connected.  Components are the basic "object" abstraction
the perception layer lifts to ``EntityModel``s in the engine —
without them every pixel is its own entity and the engine has
nothing useful to track.

The labeller returns a parallel grid of component ids (``0`` means
"background" by convention — the caller supplies the background
colour explicitly; there is no heuristic for picking it).  A second
helper, :func:`extract_regions`, returns a structured description of
each non-background component (cells, bounding box, colour, area)
which is directly consumable by the perception layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

Cell = Tuple[int, int]
Grid = Sequence[Sequence[int]]

_NEIGHBOURS_4: Tuple[Cell, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


@dataclass(frozen=True)
class Region:
    """A single connected component.

    ``bbox`` is ``(min_row, min_col, max_row, max_col)`` inclusive.
    ``cells`` is returned sorted (row, col) so downstream hashing /
    equality checks are deterministic.
    """

    label:  int
    colour: int
    cells:  Tuple[Cell, ...]
    bbox:   Tuple[int, int, int, int]

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def centroid(self) -> Tuple[float, float]:
        rs = sum(c[0] for c in self.cells) / len(self.cells)
        cs = sum(c[1] for c in self.cells) / len(self.cells)
        return (rs, cs)


def label(grid: Grid, *, background: int = 0) -> List[List[int]]:
    """Return a grid of component ids parallel to ``grid``.

    Cells with ``grid[r][c] == background`` are labelled ``0``.  Other
    cells receive a positive component id, with ids numbered in
    left-to-right, top-to-bottom discovery order.  This deterministic
    ordering matters because perception keys ``EntityModel``s off the
    ids and later episodes must line up with earlier ones.
    """
    if not grid or not grid[0]:
        return []
    h, w = len(grid), len(grid[0])
    out: List[List[int]] = [[0] * w for _ in range(h)]
    next_id = 0
    for r in range(h):
        for c in range(w):
            if out[r][c] != 0 or grid[r][c] == background:
                continue
            next_id += 1
            _flood_fill(grid, out, r, c, next_id, background)
    return out


def extract_regions(grid: Grid, *, background: int = 0) -> List[Region]:
    """Label ``grid`` and return a :class:`Region` per component.

    Regions are returned in the same left-to-right / top-to-bottom
    discovery order as their labels.  Background cells are excluded.
    """
    labels = label(grid, background=background)
    if not labels:
        return []
    h, w = len(labels), len(labels[0])

    by_id: Dict[int, List[Cell]] = {}
    colours: Dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            lab = labels[r][c]
            if lab == 0:
                continue
            by_id.setdefault(lab, []).append((r, c))
            colours.setdefault(lab, grid[r][c])

    regions: List[Region] = []
    for lab in sorted(by_id):
        cells = sorted(by_id[lab])
        rs = [c[0] for c in cells]
        cs = [c[1] for c in cells]
        bbox = (min(rs), min(cs), max(rs), max(cs))
        regions.append(Region(
            label  = lab,
            colour = colours[lab],
            cells  = tuple(cells),
            bbox   = bbox,
        ))
    return regions


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _flood_fill(
    grid:       Grid,
    out:        List[List[int]],
    r0:         int,
    c0:         int,
    label_val:  int,
    background: int,
) -> None:
    # Iterative DFS; avoids recursion-limit issues on large components.
    target = grid[r0][c0]
    if target == background:
        return
    stack: List[Cell] = [(r0, c0)]
    h, w = len(grid), len(grid[0])
    while stack:
        r, c = stack.pop()
        if out[r][c] != 0:
            continue
        if grid[r][c] != target:
            continue
        out[r][c] = label_val
        for dr, dc in _NEIGHBOURS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 0 and grid[nr][nc] == target:
                stack.append((nr, nc))
