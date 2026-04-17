"""Hand-built grids and episodes for unit tests.

Everything here is small (10×10 or less) for readability.  The real
ARC-AGI-3 grid is 60×60 but the perception and tool code is
size-agnostic, and debugging a 10×10 assertion failure is
dramatically easier than staring at a 3600-cell wall of text.
"""

from __future__ import annotations

from typing import List, Tuple


Grid = List[List[int]]


def blank_grid(h: int = 10, w: int = 10, bg: int = 0) -> Grid:
    """A uniform grid of the background colour."""
    return [[bg] * w for _ in range(h)]


def static_symmetric_grid() -> Grid:
    """A 5×5 grid symmetric about both axes (a cross).

    Used by the symmetry-tool tests.  The cross is colour ``3``
    on a ``0`` background.
    """
    g = blank_grid(5, 5, 0)
    for i in range(5):
        g[2][i] = 3
        g[i][2] = 3
    return g


def two_object_collision() -> Tuple[Grid, Grid]:
    """Two non-overlapping single-cell objects whose positions swap
    between frames.

    Frame 1:              Frame 2:
        .....                 .....
        .A...                 ...A.
        .....     →           .....
        ...B.                 .B...
        .....                 .....

    The frame-diff / motion-vectors tools should see both objects
    move; ``cell_diff`` should report four cell changes.  Colours
    ``A = 2``, ``B = 4``.
    """
    before: Grid = blank_grid(5, 5)
    after:  Grid = blank_grid(5, 5)
    before[1][1] = 2
    before[3][3] = 4
    after[1][3]  = 2
    after[3][1]  = 4
    return before, after


def moving_agent_episode() -> Tuple[List[Grid], List[str], List[int], List[List[object]]]:
    """A four-step episode where a single colour-``2`` agent moves
    one cell right each step, ending in a WIN state.

    Returns (frames, states, levels_completed, available_actions) in
    the shape :meth:`ArcAdapter.from_replay` expects.
    """
    def _with_agent(col: int) -> Grid:
        g = blank_grid(5, 5)
        g[2][col] = 2
        return g

    frames:            List[Grid]         = [_with_agent(c) for c in (0, 1, 2, 3, 4)]
    states:            List[str]          = ["PLAYING"] * 4 + ["WIN"]
    levels_completed:  List[int]          = [0, 0, 0, 0, 1]
    # The replay env ignores the actual action objects; supply a
    # minimal IntEnum-like stand-in so ``available_actions`` is non-
    # empty for every step.
    action_stubs = [_FakeAction(2)]    # ACTION2 — value is symbolic only
    available:         List[List[object]] = [list(action_stubs) for _ in frames]
    return frames, states, levels_completed, available


class _FakeAction:
    """Stands in for the arc_agi SDK's Action enum in tests.

    The action_mapping layer reads ``.value``; nothing else is used.
    """

    def __init__(self, value: int) -> None:
        self.value = value

    def __repr__(self) -> str:
        return f"ACTION{self.value}"
