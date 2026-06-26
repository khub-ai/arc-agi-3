"""Bridge between vlm_strategy's ActiveSubgoal and cognitive_os's
Goal Forest.

Why this exists
---------------

ActiveSubgoal (durable container for the strategy actor's explicit
commitments) had only ``parent_id`` for structure — no sequencing
primitive.  Sibling subgoals (those sharing a ``parent_id``) were
completely unordered: the actor could pursue them in any order
even when the task required a specific sequence, and the
substrate had no way to express or enforce "do A, then B, then C."

The cognitive_os engine, which already drives ls20 / bp35, has
TWO sequential primitives:

  1. ``Goal.depends_on: Optional[DepExpr]`` — top-level
     precondition expressions (DepRef / DepAll / DepAny) checked
     by ``goal_forest.is_actionable``.  A goal whose deps are
     unmet is filtered from selection.
  2. ``GoalNode.ordering: Ordering = SEQUENTIAL`` — within an
     AND node, children execute in declared order.

This module bridges the two stacks at the goal/subgoal level:

  * Every committed ActiveSubgoal mints a parallel cognitive_os
    ``Goal`` in a ``GoalForest`` attached to WorldKnowledge.
  * The Goal's ``depends_on`` is built from the ActiveSubgoal's
    new ``depends_on: list[str]`` field via
    ``DepAll(tuple(DepRef(id) for id in depends_on))``.
  * Engine's ``is_actionable`` becomes the single source of
    truth for which subgoals can be pursued this turn.
  * When an ActiveSubgoal's status flips to ``achieved``, the
    bridge marks the corresponding Goal's root node ACHIEVED,
    automatically unblocking dependent subgoals.

Stays minimal: we don't try to lift the full GoalForest selector
machinery (priority ranking, plan search, conflict tracking) into
the strategy-actor stack.  Only ``depends_on`` evaluation is
needed.  The bridge stores its forest as a private attribute on
WorldKnowledge and creates a minimal ``_BridgeWorldState`` shim
to satisfy the engine's ``ws.goal_forest.goals.get(...)`` API.

Substrate-agnostic, no game-specific assumptions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

# cognitive_os engine types live at repo root; ensure importable.
_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from cognitive_os.types import (                                  # noqa: E402
    DepAll, DepRef, Goal, GoalForest, GoalNode, GoalStatus,
    NodeType, Ordering,
)
from cognitive_os.goal_forest import (                            # noqa: E402
    is_actionable, _validate_no_cycle, dep_refs,
)

from world_knowledge import ActiveSubgoal, WorldKnowledge          # noqa: E402


_BRIDGE_ATTR = "_subgoal_goal_forest"
_BRIDGE_WS_ATTR = "_subgoal_bridge_ws"


class _BridgeWorldState:
    """Minimal stand-in for cognitive_os.WorldState.

    The engine's ``is_actionable`` and ``_validate_no_cycle`` only
    access ``ws.goal_forest.goals`` (a dict).  We don't need the
    full WorldState dataclass — just an object with a ``.goal_forest``
    attribute pointing at a GoalForest.
    """
    __slots__ = ("goal_forest",)

    def __init__(self, goal_forest: GoalForest):
        self.goal_forest = goal_forest


def _get_or_init_bridge(world: WorldKnowledge) -> _BridgeWorldState:
    """Lazy-initialise the GoalForest sidecar attached to ``world``.

    Stored as a private attribute so the WorldKnowledge dataclass
    schema doesn't change and so JSON serialisation skips it (the
    bridge state is regenerated from active_subgoals on reload).
    """
    ws = getattr(world, _BRIDGE_WS_ATTR, None)
    if ws is not None:
        return ws

    forest = GoalForest()
    ws = _BridgeWorldState(forest)
    # Rebuild from any existing ActiveSubgoals — supports load-then-
    # query without going through commit_subgoal.
    for sg in world.active_subgoals:
        _register_goal_for_subgoal(ws, sg, validate_cycle=False)
        # Mirror status: a subgoal already achieved keeps its
        # achievement so dependents become actionable on first
        # query.
        if sg.status == "achieved":
            g = ws.goal_forest.goals.get(sg.subgoal_id)
            if g is not None:
                g.root.status = GoalStatus.ACHIEVED

    setattr(world, _BRIDGE_WS_ATTR, ws)
    setattr(world, _BRIDGE_ATTR, forest)
    return ws


def _build_dep_expr(depends_on_ids: List[str]):
    """Translate a flat list of subgoal_ids into an engine DepExpr.

    Empty list → None (no dependencies, vacuously actionable).
    One or more ids → ``DepAll(tuple(DepRef(id) for id in ...))``,
    matching the actor-facing "all must be achieved" semantics.

    OR-style ("any of these") is intentionally not exposed in the
    actor's commit_subgoal slot — it would require a structured-
    JSON depends_on shape that adds parsing surface for marginal
    real-world benefit.
    """
    ids = [i for i in (depends_on_ids or []) if i]
    if not ids:
        return None
    return DepAll(tuple(DepRef(i) for i in ids))


def _register_goal_for_subgoal(
    ws: _BridgeWorldState,
    sg: ActiveSubgoal,
    *,
    validate_cycle: bool = True,
) -> Goal:
    """Mint and register a cognitive_os Goal mirroring ``sg``.

    The Goal's id matches the subgoal_id so they're cross-
    referenceable.  The root is a single ATOM node — we don't
    decompose into condition trees because the strategy-actor
    stack doesn't have the closed-vocab Condition language the
    engine planner needs.  Only ``status`` and ``depends_on`` are
    load-bearing for our use of the forest.
    """
    if sg.subgoal_id in ws.goal_forest.goals:
        return ws.goal_forest.goals[sg.subgoal_id]

    root = GoalNode(
        id=f"{sg.subgoal_id}:root",
        node_type=NodeType.ATOM,
        condition=None,
        status=GoalStatus.OPEN,
        priority=0.5,
        ordering=Ordering.SEQUENTIAL,
        source="bridge:active_subgoal",
        created_at=sg.created_at_turn,
    )
    goal = Goal(
        id=sg.subgoal_id,
        root=root,
        priority=0.5,
        source="bridge:active_subgoal",
        created_at=sg.created_at_turn,
        depends_on=_build_dep_expr(sg.depends_on),
    )

    if validate_cycle and goal.depends_on is not None:
        # Engine cycle check + missing-ref check.  Both raise
        # ValueError; we let those propagate to the caller (the
        # strategy-reply consumer) so the actor's commit gets
        # rejected at parse time and the warning surfaces.
        _validate_no_cycle(ws, goal)

    ws.goal_forest.goals[sg.subgoal_id] = goal
    return goal


# ---------------------------------------------------------------------------
# Public API — called from active_subgoals.commit_subgoal and
# update_subgoal_status (and from vlm_strategy's prompt formatter).
# ---------------------------------------------------------------------------


def register_subgoal(world: WorldKnowledge,
                       sg: ActiveSubgoal) -> Goal:
    """Called from commit_subgoal after the ActiveSubgoal has been
    appended to ``world.active_subgoals``.  Returns the minted Goal.

    Raises ValueError if the depends_on expression references a
    non-existent subgoal or would introduce a cycle (engine's
    standard guards).
    """
    ws = _get_or_init_bridge(world)
    return _register_goal_for_subgoal(ws, sg, validate_cycle=True)


def mark_subgoal_achieved(world: WorldKnowledge,
                            subgoal_id: str) -> None:
    """Mark the goal mirroring ``subgoal_id`` as ACHIEVED, which
    unblocks any dependent subgoals.  Idempotent — safe to call
    even if the goal is already achieved or doesn't exist."""
    ws = _get_or_init_bridge(world)
    g = ws.goal_forest.goals.get(subgoal_id)
    if g is None:
        return
    g.root.status = GoalStatus.ACHIEVED


def is_subgoal_actionable(world: WorldKnowledge,
                            subgoal_id: str) -> bool:
    """True iff the subgoal's depends_on expression is currently
    satisfied (or the subgoal has no dependencies).  Returns True
    for subgoals not registered with the bridge — back-compat for
    pre-existing subgoals that didn't go through commit_subgoal."""
    ws = _get_or_init_bridge(world)
    g = ws.goal_forest.goals.get(subgoal_id)
    if g is None:
        return True
    return is_actionable(g, ws)


def blocked_by_ids(world: WorldKnowledge,
                     subgoal_id: str) -> List[str]:
    """Return the list of unmet dependency subgoal_ids for
    ``subgoal_id`` — i.e. the ids of dependencies whose Goal isn't
    ACHIEVED yet.  Empty list if actionable or unregistered."""
    ws = _get_or_init_bridge(world)
    g = ws.goal_forest.goals.get(subgoal_id)
    if g is None or g.depends_on is None:
        return []
    unmet: List[str] = []
    for ref_id in dep_refs(g.depends_on):
        dep_g = ws.goal_forest.goals.get(ref_id)
        if dep_g is None:
            # Dangling reference — surface it as unmet so the
            # actor sees the issue.
            unmet.append(ref_id)
            continue
        if dep_g.root.status != GoalStatus.ACHIEVED:
            unmet.append(ref_id)
    return unmet


def actionable_status(
    world: WorldKnowledge, subgoal_id: str,
) -> tuple[bool, List[str]]:
    """Convenience: return ``(is_actionable, unmet_dependency_ids)``
    in one call so the prompt formatter doesn't query twice."""
    actionable = is_subgoal_actionable(world, subgoal_id)
    if actionable:
        return True, []
    return False, blocked_by_ids(world, subgoal_id)
