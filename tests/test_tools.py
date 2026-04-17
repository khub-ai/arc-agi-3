"""Tests for the grid-primitive tool suite.

Each tool is tested in isolation against a minimal synthetic grid.
The registry wiring is tested separately (:func:`test_registry`) so
a wiring bug and a tool-logic bug surface as distinct failures.
"""

from __future__ import annotations

from cognitive_os import ToolInvocation

from arc_agi_3.tools import bfs, components, diff, symmetry
from arc_agi_3.tools.registry import build_registry, dispatch

from .fixtures import (
    blank_grid,
    static_symmetric_grid,
    two_object_collision,
)


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def test_bfs_shortest_path_open_grid() -> None:
    grid = blank_grid(5, 5)
    path = bfs.shortest_path(grid, (0, 0), (4, 4))
    assert path is not None
    assert path[0] == (0, 0) and path[-1] == (4, 4)
    # Manhattan distance on an unobstructed 5×5 grid is 8 ⇒ 9 cells.
    assert len(path) == 9


def test_bfs_shortest_path_blocked() -> None:
    grid = [[0] * 3 for _ in range(3)]
    # Make row 1 impassable except column 0 using a predicate.
    def passable(r: int, c: int, _v: int) -> bool:
        return not (r == 1 and c > 0)
    path = bfs.shortest_path(grid, (0, 2), (2, 2), passable=passable)
    # Must detour through column 0 row 1.
    assert path is not None
    assert (1, 0) in path


def test_bfs_unreachable_returns_none() -> None:
    grid = [[0] * 3 for _ in range(3)]
    def passable(r: int, c: int, _v: int) -> bool:
        return not (r == 1)  # fully block row 1
    assert bfs.shortest_path(grid, (0, 0), (2, 2), passable=passable) is None


def test_bfs_reachable_cells_matches_bfs_tree_size() -> None:
    grid = blank_grid(3, 3)
    reachable = bfs.reachable_cells(grid, (1, 1))
    assert set(reachable) == {(r, c) for r in range(3) for c in range(3)}


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_components_label_three_regions() -> None:
    grid = [
        [0, 1, 0, 2, 2],
        [0, 1, 0, 2, 2],
        [0, 0, 0, 0, 0],
        [3, 3, 0, 0, 0],
    ]
    regions = components.extract_regions(grid, background=0)
    assert len(regions) == 3
    colours = sorted(r.colour for r in regions)
    assert colours == [1, 2, 3]

    by_colour = {r.colour: r for r in regions}
    assert by_colour[1].area == 2
    assert by_colour[2].area == 4
    assert by_colour[3].area == 2
    assert by_colour[2].bbox == (0, 3, 1, 4)


def test_components_label_respects_4_connectivity() -> None:
    grid = [
        [1, 0],
        [0, 1],
    ]
    # Diagonal 1s are distinct components under 4-connectivity.
    regions = components.extract_regions(grid, background=0)
    assert len(regions) == 2


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def test_symmetry_cross_is_fully_symmetric() -> None:
    report = symmetry.detect(static_symmetric_grid())
    assert report.horizontal
    assert report.vertical
    assert report.diagonal
    assert report.rotational_180
    assert report.any


def test_symmetry_asymmetric_grid() -> None:
    grid = [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
    report = symmetry.detect(grid)
    assert not report.horizontal
    assert not report.vertical
    assert not report.diagonal
    assert not report.rotational_180


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def test_diff_cell_diff_reports_all_changes() -> None:
    before, after = two_object_collision()
    changes = diff.cell_diff(before, after)
    # Two objects each moved one cell ⇒ four differing cells total
    # (old positions emptied, new positions filled).
    assert len(changes) == 4


def test_diff_is_identical() -> None:
    grid = [[1, 2], [3, 4]]
    assert diff.is_identical(grid, [row[:] for row in grid])
    assert not diff.is_identical(grid, [[1, 2], [3, 5]])


def test_diff_motion_vectors_finds_both_translations() -> None:
    before, after = two_object_collision()
    vectors = diff.motion_vectors(before, after, background=0)
    # One confident match per object.
    assert len(vectors) == 2
    by_colour = {v.colour: v for v in vectors}
    assert by_colour[2].confident and (by_colour[2].dr, by_colour[2].dc) == (0, 2)
    assert by_colour[4].confident and (by_colour[4].dr, by_colour[4].dc) == (0, -2)


# ---------------------------------------------------------------------------
# Registry + dispatch
# ---------------------------------------------------------------------------


def test_registry_exposes_all_tool_names() -> None:
    registry, handlers = build_registry()
    for name in registry.names():
        assert name in handlers


def test_registry_dispatch_success() -> None:
    _, handlers = build_registry()
    inv = ToolInvocation(
        invocation_id = "t1",
        tool_name     = "grid.symmetry.detect",
        arguments     = {"grid": static_symmetric_grid()},
        requester     = "test",
        requested_at  = 0,
    )
    result = dispatch(inv, handlers, current_step=1)
    assert result.success
    assert result.result.any


def test_registry_dispatch_unknown_tool_fails_cleanly() -> None:
    _, handlers = build_registry()
    inv = ToolInvocation(
        invocation_id = "t2",
        tool_name     = "grid.nonsense",
        arguments     = {},
        requester     = "test",
        requested_at  = 0,
    )
    result = dispatch(inv, handlers, current_step=1)
    assert not result.success
    assert result.error and "unknown tool" in result.error


def test_registry_dispatch_tool_exception_becomes_failing_result() -> None:
    _, handlers = build_registry()
    inv = ToolInvocation(
        invocation_id = "t3",
        tool_name     = "grid.bfs.shortest_path",
        arguments     = {},   # missing required keys
        requester     = "test",
        requested_at  = 0,
    )
    result = dispatch(inv, handlers, current_step=1)
    assert not result.success
    assert result.error is not None
