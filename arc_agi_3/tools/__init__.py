"""Domain primitives registered in the engine's ToolRegistry.

Every tool in this package is a pure function over 2-D integer grids.
None of them know anything about ARC-AGI-3 game semantics — they are
the generic grid primitives that miners / planner / explorer invoke
through :class:`cognitive_os.ToolInvocation` because doing the same
work through an LLM is slow, expensive, and error-prone.

Keeping these tools domain-neutral is a deliberate choice: a future
2-D grid domain (Game-of-Life analysis, Sokoban variants, tile
classification) should be able to re-use the entire tool module
unchanged.  ARC-AGI-3 specificity lives only in
:mod:`arc_agi_3.adapter`, :mod:`arc_agi_3.perception`, and
:mod:`arc_agi_3.action_mapping`.

Public surface
--------------
* :func:`bfs.shortest_path`           — BFS on a grid with a passability predicate.
* :func:`components.label`            — 4-connected connected-component labelling.
* :func:`components.extract_regions`  — list of (label, cells, bounding_box) tuples.
* :func:`symmetry.detect`             — axis / rotational symmetries of a grid region.
* :func:`diff.cell_diff`              — list of cells that changed between two frames.
* :func:`diff.motion_vectors`         — best-effort rigid-translation match per region.
* :func:`registry.build_registry`     — constructs a :class:`ToolRegistry` populated
                                         with all of the above and returns a dispatch
                                         map the adapter can hand to
                                         :meth:`Adapter.invoke_tool`.
"""

from . import bfs, components, diff, symmetry
from .registry import TOOL_NAMES, build_registry, dispatch

__all__ = [
    "bfs",
    "components",
    "diff",
    "symmetry",
    "build_registry",
    "dispatch",
    "TOOL_NAMES",
]
