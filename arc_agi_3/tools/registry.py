"""Glue between this package's tool implementations and the engine's
:class:`cognitive_os.ToolRegistry`.

The adapter's :meth:`ArcAdapter.initialize` calls
:func:`build_registry` to obtain a registry pre-populated with typed
:class:`ToolSignature`\\s, plus a dispatch map ``name -> handler``
that :meth:`ArcAdapter.invoke_tool` uses to route
:class:`ToolInvocation`\\s to the right Python function.

Tool names are namespaced (``grid.bfs``, ``grid.components``, …)
so downstream domains that provide their own tools (3-D motion
planner, trajectory optimiser) can coexist without clashing.

Signatures are hand-authored once here rather than derived via
reflection because:

* The engine needs the ``cost`` / ``typical_latency_ms`` /
  ``determinism`` metadata, which cannot be inferred from Python
  type hints.
* The ``input_schema`` uses string type hints (``"Grid"``,
  ``"tuple[int,int]"``) so the registry is trivially serialisable
  for the Mediator's ``WorldStateSummary``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from cognitive_os import (
    ToolInvocation,
    ToolRegistry,
    ToolResult,
    ToolSignature,
)

from . import bfs, components, diff, symmetry


Handler = Callable[[Dict[str, Any]], Any]


# Public tool names — stable identifiers miners and the planner refer
# to.  Changing any of these is a breaking change for learned
# knowledge artifacts that reference tool calls in their evidence.
TOOL_NAMES: Tuple[str, ...] = (
    "grid.bfs.shortest_path",
    "grid.bfs.reachable_cells",
    "grid.components.label",
    "grid.components.extract_regions",
    "grid.symmetry.detect",
    "grid.diff.cell_diff",
    "grid.diff.motion_vectors",
    "grid.diff.is_identical",
)


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------


def _signatures() -> Tuple[ToolSignature, ...]:
    return (
        ToolSignature(
            name               = "grid.bfs.shortest_path",
            description        = "Shortest 4-connected path from start to goal on a grid.",
            input_schema       = (("grid", "Grid"), ("start", "tuple[int,int]"),
                                  ("goal",  "tuple[int,int]")),
            output_schema      = "list[tuple[int,int]] | None",
            cost               = 1.0,
            typical_latency_ms = 2.0,
            determinism        = True,
            side_effects       = False,
            is_async           = False,
        ),
        ToolSignature(
            name               = "grid.bfs.reachable_cells",
            description        = "Every cell reachable from start under 4-connectivity.",
            input_schema       = (("grid", "Grid"), ("start", "tuple[int,int]")),
            output_schema      = "list[tuple[int,int]]",
            cost               = 1.0,
            typical_latency_ms = 2.0,
            determinism        = True,
            side_effects       = False,
            is_async           = False,
        ),
        ToolSignature(
            name               = "grid.components.label",
            description        = "4-connected component labelling (parallel grid of ids).",
            input_schema       = (("grid", "Grid"), ("background", "int")),
            output_schema      = "Grid",
            cost               = 1.0,
            typical_latency_ms = 2.0,
            determinism        = True,
        ),
        ToolSignature(
            name               = "grid.components.extract_regions",
            description        = "List of non-background connected regions with bbox / colour / cells.",
            input_schema       = (("grid", "Grid"), ("background", "int")),
            output_schema      = "list[Region]",
            cost               = 1.0,
            typical_latency_ms = 3.0,
            determinism        = True,
        ),
        ToolSignature(
            name               = "grid.symmetry.detect",
            description        = "Axis and rotational symmetries of a grid region.",
            input_schema       = (("grid", "Grid"),
                                  ("bbox", "tuple[int,int,int,int] | None")),
            output_schema      = "SymmetryReport",
            cost               = 0.5,
            typical_latency_ms = 1.0,
            determinism        = True,
        ),
        ToolSignature(
            name               = "grid.diff.cell_diff",
            description        = "Cells that changed colour between two frames.",
            input_schema       = (("before", "Grid"), ("after", "Grid")),
            output_schema      = "list[CellChange]",
            cost               = 1.0,
            typical_latency_ms = 2.0,
            determinism        = True,
        ),
        ToolSignature(
            name               = "grid.diff.motion_vectors",
            description        = "Per-region rigid-translation match between two frames.",
            input_schema       = (("before", "Grid"), ("after", "Grid"),
                                  ("background", "int")),
            output_schema      = "list[MotionVector]",
            cost               = 2.0,
            typical_latency_ms = 5.0,
            determinism        = True,
        ),
        ToolSignature(
            name               = "grid.diff.is_identical",
            description        = "Whole-frame equality check (cheap).",
            input_schema       = (("before", "Grid"), ("after", "Grid")),
            output_schema      = "bool",
            cost               = 0.1,
            typical_latency_ms = 0.5,
            determinism        = True,
        ),
    )


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def _handlers() -> Dict[str, Handler]:
    return {
        "grid.bfs.shortest_path": lambda a: bfs.shortest_path(
            a["grid"], a["start"], a["goal"],
            passable=a.get("passable"),
        ),
        "grid.bfs.reachable_cells": lambda a: bfs.reachable_cells(
            a["grid"], a["start"],
            passable=a.get("passable"),
        ),
        "grid.components.label": lambda a: components.label(
            a["grid"], background=a.get("background", 0),
        ),
        "grid.components.extract_regions": lambda a: components.extract_regions(
            a["grid"], background=a.get("background", 0),
        ),
        "grid.symmetry.detect": lambda a: symmetry.detect(
            a["grid"], bbox=a.get("bbox"),
        ),
        "grid.diff.cell_diff": lambda a: diff.cell_diff(
            a["before"], a["after"],
        ),
        "grid.diff.motion_vectors": lambda a: diff.motion_vectors(
            a["before"], a["after"], background=a.get("background", 0),
        ),
        "grid.diff.is_identical": lambda a: diff.is_identical(
            a["before"], a["after"],
        ),
    }


# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------


def build_registry() -> Tuple[ToolRegistry, Dict[str, Handler]]:
    """Return a populated :class:`ToolRegistry` and a parallel dispatch
    table the adapter uses in :meth:`invoke_tool`.

    The adapter MUST hold onto the returned dispatch table separately;
    the :class:`ToolRegistry` is the engine-side view (signatures
    only) and deliberately does not contain implementations.
    """
    registry = ToolRegistry()
    for sig in _signatures():
        registry.register(sig)
    return registry, _handlers()


def dispatch(
    invocation: ToolInvocation,
    handlers:   Dict[str, Handler],
    *,
    current_step: int,
) -> ToolResult:
    """Execute a :class:`ToolInvocation` against a handler table.

    Wraps adapter-side exceptions in a failing :class:`ToolResult` so
    no tool call can take down the engine loop — a failed call
    becomes a typed error the hypothesis / planner layer can reason
    about.
    """
    handler = handlers.get(invocation.tool_name)
    if handler is None:
        return ToolResult(
            invocation_id = invocation.invocation_id,
            success       = False,
            error         = f"unknown tool: {invocation.tool_name}",
            completed_at  = current_step,
        )
    try:
        result = handler(invocation.arguments)
    except Exception as exc:   # noqa: BLE001 — deliberate: surface every failure
        return ToolResult(
            invocation_id = invocation.invocation_id,
            success       = False,
            error         = f"{type(exc).__name__}: {exc}",
            completed_at  = current_step,
        )
    return ToolResult(
        invocation_id = invocation.invocation_id,
        success       = True,
        result        = result,
        completed_at  = current_step,
    )
