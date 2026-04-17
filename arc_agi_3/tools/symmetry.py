"""Symmetry detection over a 2-D grid region.

Symmetry is a structure-mapping primitive — ARC-style tasks often
require recognising that two halves of a shape mirror each other, or
that the whole grid has a rotational axis.  Detecting these
symbolically (rather than pushing the whole image at a VLM) is
cheap, deterministic, and gives the engine a ``StructureMappingClaim``
it can reason with directly.

The detector reports which of the four canonical symmetries a region
exhibits:

* horizontal — ``grid[r][c] == grid[r][W-1-c]``
* vertical   — ``grid[r][c] == grid[H-1-r][c]``
* diagonal   — ``grid[r][c] == grid[c][r]`` (requires square)
* rotational-180 — ``grid[r][c] == grid[H-1-r][W-1-c]``

Callers can query a sub-region by passing ``bbox``; omitting it tests
the whole grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

Grid = Sequence[Sequence[int]]


@dataclass(frozen=True)
class SymmetryReport:
    """Which symmetries a region exhibits.

    All four fields are independent booleans — a region can be both
    horizontally and vertically symmetric (e.g. a cross), neither, or
    any combination.
    """

    horizontal:      bool
    vertical:        bool
    diagonal:        bool
    rotational_180:  bool

    @property
    def any(self) -> bool:
        return self.horizontal or self.vertical or self.diagonal or self.rotational_180


def detect(
    grid: Grid,
    *,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> SymmetryReport:
    """Return which of the four canonical symmetries ``grid`` exhibits.

    ``bbox`` is ``(min_row, min_col, max_row, max_col)`` inclusive; if
    ``None``, the entire grid is tested.  An empty or degenerate
    region returns a report with every flag set to ``False`` — empty
    regions are mathematically symmetric under every axis, but the
    engine benefits from the more useful convention that "no region"
    means "no structural claim to make".
    """
    if not grid or not grid[0]:
        return SymmetryReport(False, False, False, False)

    H_full, W_full = len(grid), len(grid[0])
    if bbox is None:
        r0, c0, r1, c1 = 0, 0, H_full - 1, W_full - 1
    else:
        r0, c0, r1, c1 = bbox
        if not (0 <= r0 <= r1 < H_full and 0 <= c0 <= c1 < W_full):
            return SymmetryReport(False, False, False, False)

    H = r1 - r0 + 1
    W = c1 - c0 + 1
    if H <= 0 or W <= 0:
        return SymmetryReport(False, False, False, False)

    def g(r: int, c: int) -> int:
        return grid[r0 + r][c0 + c]

    horizontal = all(
        g(r, c) == g(r, W - 1 - c)
        for r in range(H)
        for c in range(W // 2)
    )

    vertical = all(
        g(r, c) == g(H - 1 - r, c)
        for r in range(H // 2)
        for c in range(W)
    )

    diagonal = False
    if H == W:
        diagonal = all(
            g(r, c) == g(c, r)
            for r in range(H)
            for c in range(r + 1, W)
        )

    rotational_180 = all(
        g(r, c) == g(H - 1 - r, W - 1 - c)
        for r in range(H)
        for c in range(W)
        # Avoid double-work by iterating only over the first half of the
        # linearised region — but for clarity we just scan everything;
        # the region is at most 60×60 so the cost is negligible.
    )

    return SymmetryReport(
        horizontal     = horizontal,
        vertical       = vertical,
        diagonal       = diagonal,
        rotational_180 = rotational_180,
    )
