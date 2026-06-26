"""Path-planning tool registry — pluggable strategies for REACH_CELL
and REACH_ENTITY plan-tree nodes.

See :doc:`docs/SPEC_plan_tree.md` §"Pluggable path-planning tools".

The motivation: the default reachability check for a plan-tree leaf
is forward-BFS over the agent's passable_grid + observed portal_map.
That misses paths that exist through *predicted* portal edges (the
affordance DSL's launcher predictions) and through backward-search
strategies that find one-way paths the forward search can't.

This module provides the registry primitive — a typed contract for
adding strategies — without prescribing which strategies belong.
Concrete tools live in adapter-side modules (``arc_plan_tools.py``,
``robotics_plan_tools.py`` etc.) and register themselves at import
time.  The plan-tree primitive consumes the registered set; engine
code stays domain-clean.

Engine-clean: the ``plan`` callable takes opaque ``start`` /
``target`` / ``world`` values.  The engine never inspects them —
only the adapter that registered the tool knows their shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# A path-plan result is also opaque to the engine — adapters define
# what a "plan" is (an action sequence, a cell path, a graph route).
Plan = Any


@dataclass(frozen=True)
class PathPlanningTool:
    """One pluggable path-planning strategy.

    Attributes
    ----------
    name
        Short identifier for telemetry and selection.  Must be
        unique within a registry (last registration wins).
    plan
        ``(start, target, world) -> Optional[Plan]``.  Returns a
        plan (truthy) when the target is reachable from start; None
        when no path is found within the tool's horizon.
    cost
        ``(plan) -> float``.  Estimates the cost of executing the
        plan (typically step count or budget consumption).  Used by
        the scheduler to break ties between strategies that both
        produce a viable plan.
    horizon
        Maximum search depth this tool will explore.  None = unbounded
        (e.g. full grid BFS); a finite value caps the search for
        cheap "is the target nearby?" checks.
    confidence
        Prior credence in this tool's plans.  Plans from
        ``forward_bfs`` (observed-only) should be 1.0; plans from
        ``via_predicted_portals`` (relies on affordance hypotheses)
        should be lower so the scheduler prefers observed paths
        when both are viable.  In [0, 1].
    """
    name:       str
    plan:       Callable[..., Optional[Plan]]
    cost:       Callable[[Plan], float] = field(default=lambda _p: 1.0)
    horizon:    Optional[int]           = None
    confidence: float                   = 1.0


class PlanToolRegistry:
    """Mutable registry of registered ``PathPlanningTool``s.

    The default module-level registry is exposed as
    :data:`DEFAULT_REGISTRY`.  Most callers should ``register`` /
    ``get_strategies`` against that; tests construct private
    registries to avoid cross-test pollution.
    """

    def __init__(self) -> None:
        self._tools: Dict[str, PathPlanningTool] = {}

    def register(self, tool: PathPlanningTool) -> None:
        """Add (or replace) a tool by name."""
        if not isinstance(tool, PathPlanningTool):
            raise TypeError(
                f"register expected PathPlanningTool, got {type(tool).__name__}"
            )
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """Remove a tool by name.  Returns True if it existed."""
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Optional[PathPlanningTool]:
        """Return the tool registered under ``name`` or None."""
        return self._tools.get(name)

    def get_strategies(self) -> List[PathPlanningTool]:
        """Return every registered tool, in insertion order."""
        return list(self._tools.values())

    def names(self) -> List[str]:
        """Return the names of every registered tool, in insertion order."""
        return list(self._tools.keys())

    def clear(self) -> None:
        """Remove every registered tool.  Primarily for test isolation."""
        self._tools.clear()


# Module-level singleton registry.  Adapters register their tools
# against this at import time; the plan-tree consumer reads from it.
DEFAULT_REGISTRY = PlanToolRegistry()


# Convenience module-level functions that target DEFAULT_REGISTRY.
def register(tool: PathPlanningTool) -> None:
    DEFAULT_REGISTRY.register(tool)


def unregister(name: str) -> bool:
    return DEFAULT_REGISTRY.unregister(name)


def get(name: str) -> Optional[PathPlanningTool]:
    return DEFAULT_REGISTRY.get(name)


def get_strategies() -> List[PathPlanningTool]:
    return DEFAULT_REGISTRY.get_strategies()


# ---------------------------------------------------------------------------
# Reachability convenience — common case where the caller just wants
# "is this target reachable from this start by any registered tool?"
# Returns the *winning* tool name + its plan when reachable, None
# otherwise.  Iterates in insertion order and stops at the first hit.
# ---------------------------------------------------------------------------


def first_reachable(
    start:    Any,
    target:   Any,
    world:    Any,
    *,
    registry: Optional[PlanToolRegistry] = None,
) -> Optional[tuple]:
    """Return ``(tool_name, plan)`` for the first tool that produces a
    viable plan, or None when no tool can path.

    Strategies are consulted in registration order — callers wanting
    a different ordering should register accordingly.  Each tool's
    ``plan`` callable is invoked at most once.
    """
    reg = registry if registry is not None else DEFAULT_REGISTRY
    for tool in reg.get_strategies():
        try:
            plan = tool.plan(start, target, world)
        except Exception:
            # Tool failure should never crash the consumer.  Move on
            # to the next strategy and let the eligibility caller
            # treat the leaf as ineligible if no tool produces a plan.
            continue
        if plan:
            return (tool.name, plan)
    return None
