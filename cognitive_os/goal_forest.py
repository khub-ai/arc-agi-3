"""Goal forest — operations on the GoalForest / GoalNode / Goal types.

The :class:`GoalForest` holds all active top-level goals.  Each goal is
rooted at an AND-OR-CHANCE tree (:class:`GoalNode`).  This module
provides the operations a runner needs to manage the forest during an
episode:

* Adding a new goal (adapter-seeded primary or engine-derived subgoal)
* Selecting which goal to pursue next — priority, deadlines, conflicts
* Expanding a goal with engine-derived subgoals using committed
  :class:`CausalClaim`\\s from the hypothesis store
* Walking the tree and checking achievement / failure
* Detecting conflicts across concurrent goals

Phase 3 scope
-------------
* Full operations for ATOM / AND / OR / CHANCE nodes.
* Conflict detection for MUTEX (logically incompatible conditions) and
  TEMPORAL (deadline overlap).  RESOURCE and ADVERSARIAL conflicts
  are detected structurally (pattern in place) but their resolution
  policies rely on a full rule/principal system best exercised in
  robotics adapters (Phase 5+).
* Robotics-extension node kinds (OPTION / MAINTAIN / LOOP /
  ADVERSARIAL / INFO_SET) are recognised but pass through unchanged —
  the planner/explorer will acquire handling for them when the
  corresponding adapter is built.

Capability audit (standing invariant 7)
----------------------------------------
* **Problem-solving** — PRIMARY.  Goal decomposition is how the
  engine turns a single high-level objective into a plan-generating
  structure.
* **Debugging** — secondary.  Subgoal derivation is read-only against
  committed hypotheses, so hypothesis demotions automatically
  invalidate their corresponding subgoals at the next selection pass.
* **Tool creation** — deferred.  An Option injected into the forest
  becomes a GoalNode with ``node_type=OPTION``; Phase 7 will add
  Option-producing miners and the planner handler.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from . import telemetry_schema as _tel
from .claims import CausalClaim, PropertyClaim
from .conditions import (
    ActionTried,
    AtPosition,
    Condition,
    EntitiesEquivalent,
    EntityInState,
    MotionModelCommitted,
)
from .telemetry import emit_from_ws as _emit
from .types import (
    ConflictType,
    DepAll,
    DepAny,
    DepExpr,
    DepRef,
    Goal,
    GoalConflict,
    GoalForest,
    GoalNode,
    GoalStatus,
    NodeType,
    Ordering,
    ResolutionPolicy,
    WorldState,
)
from . import hypothesis_store as _store


# Role values that, when attached to an entity, imply the agent
# should reach that entity.  Domain-agnostic by design: "target" /
# "goal" / "exit" in ARC, "pickup_target" / "dock" / "charger" in
# robotics.  Extend per-adapter via the ``role_values`` argument
# to :func:`derive_goals_from_roles`.
REACH_ROLE_VALUES: Tuple[str, ...] = (
    "target", "goal", "exit", "pickup_target", "dock", "charger",
)


# ---------------------------------------------------------------------------
# Dependency expressions (axis 1 of SPEC_goal_classification.md)
# ---------------------------------------------------------------------------


def dep_refs(expr: Optional[DepExpr]) -> Iterator[str]:
    """Walk a DepExpr tree and yield every referenced goal_id.

    Used by cycle detection and operator-error checking.  Yields
    nothing for ``None`` (no dependencies).
    """
    if expr is None:
        return
    if isinstance(expr, DepRef):
        yield expr.goal_id
        return
    if isinstance(expr, (DepAll, DepAny)):
        for child in expr.children:
            yield from dep_refs(child)
        return
    raise TypeError(f"unknown DepExpr node: {type(expr).__name__}")


def render_dep(expr: DepExpr) -> str:
    """Pretty-print a DepExpr for telemetry / log output.

    ``all(...)`` and ``any(...)`` mirror the operator-facing
    ``requires`` schema; bare goal ids are unwrapped ``DepRef``.
    """
    if isinstance(expr, DepRef):
        return expr.goal_id
    if isinstance(expr, DepAll):
        return f"all({', '.join(render_dep(c) for c in expr.children)})"
    if isinstance(expr, DepAny):
        return f"any({', '.join(render_dep(c) for c in expr.children)})"
    raise TypeError(f"unknown DepExpr node: {type(expr).__name__}")


def _eval_dep(expr: Optional[DepExpr], ws: WorldState) -> bool:
    """Recursively evaluate a dependency expression.

    Truth table:
      DepRef(gid)          → goal exists and status == ACHIEVED
      DepAll(c1, ..., cn)  → every child is True (vacuous AND for n=0)
      DepAny(c1, ..., cn)  → at least one child is True (False for n=0)
      None                 → True (no dependencies)
    """
    if expr is None:
        return True
    if isinstance(expr, DepRef):
        g = ws.goal_forest.goals.get(expr.goal_id)
        return g is not None and g.root.status == GoalStatus.ACHIEVED
    if isinstance(expr, DepAll):
        return all(_eval_dep(c, ws) for c in expr.children)
    if isinstance(expr, DepAny):
        return any(_eval_dep(c, ws) for c in expr.children)
    raise TypeError(f"unknown DepExpr node: {type(expr).__name__}")


def is_actionable(goal: Goal, ws: WorldState) -> bool:
    """Return True iff ``goal`` is currently actionable for selection.

    A goal is actionable when its dependency expression evaluates to
    True against the forest's current state.  Goals with no
    ``depends_on`` are unconditionally actionable.

    The selector applies ``is_actionable`` as a hard filter *before*
    priority ranking — non-actionable goals do not appear in
    candidates, do not contribute to top-priority maxima, and do not
    receive recovery / curiosity bumps.  This is the structural
    replacement for priority-arithmetic encodings of preconditions
    (deltas, ceilings, attenuation-defended ordering); see
    ``docs/SPEC_goal_dependencies.md``.
    """
    return _eval_dep(goal.depends_on, ws)


def _validate_no_cycle(ws: WorldState, new_goal: Goal) -> None:
    """Reject cycles when adding a new goal with a non-empty
    ``depends_on``.

    The dependency graph treats every ``DepRef`` in any DepExpr as a
    directed edge from the dependent goal to the referenced goal.
    The check ignores AND/OR structure — even one DepRef branch
    participating in a cycle is rejected, regardless of whether it
    appears under DepAll or DepAny.

    Raises ``ValueError`` with the participating ids on detection.
    Missing references (a DepRef to a goal that does not exist in
    the forest) emit ``[gf-dep] missing`` and are not registered;
    operator authoring error.
    """
    if new_goal.depends_on is None:
        return
    for ref in dep_refs(new_goal.depends_on):
        if ref not in ws.goal_forest.goals:
            print(f"[gf-dep] missing {new_goal.id} -> {ref}")
            raise ValueError(
                f"goal {new_goal.id!r} depends on undeclared goal {ref!r}"
            )

    def _adj(gid: str) -> Iterator[str]:
        if gid == new_goal.id:
            yield from dep_refs(new_goal.depends_on)
            return
        g = ws.goal_forest.goals.get(gid)
        if g is None:
            return
        yield from dep_refs(g.depends_on)

    # DFS from the new goal; if we revisit it, there's a cycle.
    visiting: set = set()
    visited:  set = set()
    path:     List[str] = []

    def _dfs(node: str) -> Optional[List[str]]:
        if node in visiting:
            cyc_start = path.index(node)
            return path[cyc_start:] + [node]
        if node in visited:
            return None
        visiting.add(node)
        path.append(node)
        for nxt in _adj(node):
            cyc = _dfs(nxt)
            if cyc is not None:
                return cyc
        path.pop()
        visiting.remove(node)
        visited.add(node)
        return None

    cycle = _dfs(new_goal.id)
    if cycle is not None:
        chain = " -> ".join(cycle)
        print(f"[gf-dep] cycle {chain}")
        raise ValueError(f"dependency cycle: {chain}")


# ---------------------------------------------------------------------------
# Registration and selection
# ---------------------------------------------------------------------------


def add_goal(ws: WorldState, goal: Goal) -> None:
    """Register a top-level goal in the forest.

    If no goal is currently active, the newly-added goal becomes
    active.  Otherwise the forest's active pointer is unchanged; a
    subsequent :func:`select_active_goal` call will arbitrate.

    Validates ``goal.depends_on`` against the existing forest:
      * Each ``DepRef`` must resolve to a goal already in the forest
        (raises ``ValueError`` otherwise).
      * The dependency graph must remain acyclic after insertion
        (raises ``ValueError`` on cycle, naming the participating
        ids).
    """
    _validate_no_cycle(ws, goal)
    is_new = goal.id not in ws.goal_forest.goals
    ws.goal_forest.goals[goal.id] = goal
    prev_active = ws.goal_forest.active_goal_id
    if ws.goal_forest.active_goal_id is None:
        ws.goal_forest.active_goal_id = goal.id

    if not is_new:
        # Same id, replacement — don't double-emit GoalAdded.  Any
        # status/condition changes of interest come through the
        # dedicated status/derivation events.
        if prev_active != ws.goal_forest.active_goal_id:
            _emit(ws, _tel.ActiveGoalChanged(
                old_goal_id = prev_active,
                new_goal_id = ws.goal_forest.active_goal_id,
            ), subject=ws.goal_forest.active_goal_id)
        return

    cond = getattr(goal.root, "condition", None)
    _emit(ws, _tel.GoalAdded(
        goal_id           = goal.id,
        priority          = float(goal.priority),
        root_node_type    = goal.root.node_type.value
                            if hasattr(goal.root.node_type, "value")
                            else str(goal.root.node_type),
        condition_summary = repr(cond.canonical_key()) if cond is not None else "",
        parent_goal_id    = None,
    ), subject=goal.id)
    if prev_active != ws.goal_forest.active_goal_id:
        _emit(ws, _tel.ActiveGoalChanged(
            old_goal_id = prev_active,
            new_goal_id = ws.goal_forest.active_goal_id,
        ), subject=ws.goal_forest.active_goal_id)


def candidates_by_priority(ws: WorldState) -> List[str]:
    """Return eligible goal IDs ordered by selection policy
    (highest priority first).

    Same filter rules as :func:`select_active_goal` but returns the
    full ordered list so callers can iterate — e.g. try to plan each
    one until a plannable goal is found.  GAP 19 introduced this split
    so the runner can fall through from an unplannable top-priority
    goal (e.g. an episode-success ``ResourceAbove`` whose truth
    cannot be reached by motion alone) to a plannable lower-priority
    one (e.g. a role-goal with a concrete ``AtPosition`` target).

    The list is empty when no eligible goals remain.  Does NOT update
    ``ws.goal_forest.active_goal_id`` — that is the runner's job once
    a plannable candidate is identified (see
    :func:`planner.select_and_plan`).

    Robotics analogue: a "deliver package" resource-predicate goal
    is unplannable by BFS (no motion path satisfies
    ``packages_delivered > N`` in the search space), but the
    concurrent "navigate to next dropoff" goal is plannable — the
    runner must be able to pick the concrete goal when the abstract
    one has no motion plan, rather than stalling.
    """
    candidates = [g for g in ws.goal_forest.goals.values()
                  if g.root.status not in
                  (GoalStatus.ACHIEVED, GoalStatus.PRUNED, GoalStatus.ABANDONED)]
    if not candidates:
        return []

    blocked_ids = {c.goal_a for c in ws.goal_forest.conflicts
                   if c.resolution_policy == ResolutionPolicy.FAIL}
    blocked_ids |= {c.goal_b for c in ws.goal_forest.conflicts
                    if c.resolution_policy == ResolutionPolicy.FAIL}
    candidates = [g for g in candidates if g.id not in blocked_ids] or candidates

    # --- Dependency gate (axis 1, SPEC_goal_dependencies.md) ------------
    # A goal whose ``depends_on`` expression is unmet is hidden from
    # selection rather than competing on priority.  Goals without
    # depends_on (the default) are unaffected.  Unlike the conflict
    # filter above, there is no degenerate-case fallback: if every
    # candidate is dependency-gated, the result is intentionally an
    # empty list — the spec's contract is that non-actionable goals
    # do not appear, which lets callers (the runner / GF→Oracle
    # escalation / robotics fallback behaviours) detect the empty
    # result and respond appropriately rather than acting on a goal
    # whose preconditions are unmet.
    candidates = [g for g in candidates if is_actionable(g, ws)]

    # --- Competence-gated learn-priority bias ---------------------------
    # The competence gate writes a continuous priority scale onto
    # ``ws.agent["_learn_priority_scale"]`` each tick — default 1.0
    # (no bias), lower values when a top task goal is BFS-plannable
    # from committed claims.  Here we apply that scale to the sort
    # priority of learn-family goals (reduce_uncertainty / explore /
    # probe) so that the ordinary goal selector naturally picks the
    # task goal when it dominates, and falls back to learning when
    # the task goal stalls.  The underlying ``g.priority`` field is
    # NOT mutated — scaling is applied only inside this sort key — so
    # the scale can change tick-to-tick without hysteresis.
    #
    # Design: continuous scale (not binary mode flag) so that when the
    # competence check is wrong or the plan fails, learning is still
    # a live fallback at reduced priority rather than being crushed
    # to near-zero.  See
    # ``memory/project_cognitive_os_execution_mode.md``.
    _scale: float = 1.0
    _learn_prefixes: Tuple[str, ...] = ()
    cfg = getattr(ws, "config", None)
    competence_cfg = getattr(cfg, "competence", None) if cfg else None
    if (competence_cfg is not None
            and getattr(competence_cfg, "enabled", False)):
        raw_scale = ws.agent.get("_learn_priority_scale", 1.0)
        try:
            _scale = float(raw_scale)
        except (TypeError, ValueError):
            _scale = 1.0
        if _scale != 1.0:
            _learn_prefixes = tuple(competence_cfg.learn_goal_prefixes)

    def _effective_priority(g: Goal) -> float:
        if (_scale != 1.0
                and _learn_prefixes
                and any(g.id.startswith(p) for p in _learn_prefixes)):
            return g.priority * _scale
        return g.priority

    def sort_key(g: Goal) -> Tuple:
        # higher priority first; earlier deadline first; stable tiebreak
        deadline = g.deadline if g.deadline is not None else float("inf")
        return (-_effective_priority(g), deadline, g.created_at)

    candidates.sort(key=sort_key)
    return [g.id for g in candidates]


def select_active_goal(ws: WorldState) -> Optional[str]:
    """Pick the goal to pursue next.

    Selection policy (Phase 3):

    1. Filter out goals whose status is ACHIEVED, PRUNED, or ABANDONED.
    2. Filter out goals blocked by unresolved conflicts whose
       resolution policy is ``FAIL``.
    3. Among survivors, prefer higher priority; tiebreak on earlier
       deadline; final tiebreak on insertion order (dict iteration).

    Returns the selected goal ID, or ``None`` if no candidate exists.
    The forest's ``active_goal_id`` is updated in place.

    Note: This returns the highest-priority eligible goal *without*
    regard for whether the planner can produce a plan for it.  A
    higher-priority goal whose condition cannot be reached via the
    committed transition model will shadow a plannable lower-priority
    goal.  Callers that want "first plannable goal by priority"
    should use :func:`planner.select_and_plan`, which is the policy
    the episode runner uses post-GAP-19.
    """
    ordered = candidates_by_priority(ws)
    prev_active = ws.goal_forest.active_goal_id
    if not ordered:
        ws.goal_forest.active_goal_id = None
        if prev_active is not None:
            _emit(ws, _tel.ActiveGoalChanged(
                old_goal_id = prev_active,
                new_goal_id = None,
            ))
        return None
    ws.goal_forest.active_goal_id = ordered[0]
    if prev_active != ordered[0]:
        _emit(ws, _tel.ActiveGoalChanged(
            old_goal_id = prev_active,
            new_goal_id = ordered[0],
        ), subject=ordered[0])
    return ordered[0]


# ---------------------------------------------------------------------------
# Subgoal derivation from committed CausalClaims
# ---------------------------------------------------------------------------


def derive_subgoals_from_causal(ws:      WorldState,
                                goal_id: str,
                                *,
                                max_depth: int = 3,
                                step:       int = 0) -> List[str]:
    """Expand ATOM leaves of a goal by consulting committed
    :class:`CausalClaim`\\s.

    For each ATOM leaf with condition ``C``:

    1. Scan committed hypotheses for a ``CausalClaim`` whose
       ``effect.canonical_key() == C.canonical_key()``.  If present,
       the trigger condition becomes a new ATOM sibling under an AND
       node that replaces the original leaf: achieve trigger → effect
       becomes true.

    2. If multiple causal claims point at the same effect, an OR node
       is inserted with one AND branch per alternative trigger.

    Only claims with credence above the engine's commit threshold are
    used.  Contradicted or undecided claims are ignored, so the derived
    structure naturally reflects current confidence.

    Parameters
    ----------
    max_depth
        Cap on recursion depth to prevent unbounded expansion if the
        hypothesis graph contains near-cycles.
    step
        Current step, used as ``created_at`` on newly created nodes.

    Returns
    -------
    list[str]
        IDs of nodes that were newly expanded.  Empty if nothing
        changed — either no applicable CausalClaims or every leaf
        already has derived children.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return []

    expanded: List[str] = []
    _expand_node(goal.root, ws, max_depth, step, expanded)
    # Refresh AtPosition tolerances on already-expanded atoms: the
    # bridge fires at expansion time and freezes whatever tolerance
    # was derivable then.  If no motion model had committed yet,
    # tolerance stuck at 0; now that a model has committed the same
    # atom should pick up the motion-step radius so the planner can
    # close on it.  Cheap (walks one goal's tree) and idempotent.
    refresh_atposition_tolerances(ws, goal_id)
    return expanded


def refresh_atposition_tolerances(ws: WorldState, goal_id: str) -> int:
    """Walk one goal's tree and raise ``tolerance`` on planner-facing
    ``AtPosition`` atoms whose tolerance was frozen at ``0.0`` when
    the bridge ran against a still-empty motion-model set.

    Returns the number of atoms whose tolerance was lifted.

    Rationale.  ``_expand_node`` fires once per ATOM leaf and writes
    the resulting ``AtPosition`` into the tree with whatever
    tolerance the bridge could compute at that instant.  If motion
    models commit later (which is the common case — the engine
    explores, learns its motor primitives, and only then starts
    closing the loop on Mediator-sourced goals), those frozen
    tolerance-0 atoms become unreachable on the motor lattice.
    The planner returns ``None`` even though, conceptually, the
    goal is now within tolerance of a lattice point.  This pass
    reconciles the two by lifting stale tolerances to the current
    motion-step radius.

    We do NOT touch atoms whose tolerance is already > 0 — those
    came from a deliberate goal-synthesis choice (e.g. entity-pos
    bridge setting the bbox radius) and overriding them here would
    be disrespecting that intent.

    Robotics analogue.  When a manipulator newly calibrates its
    end-effector step size, existing open goals should inherit the
    new reach radius rather than stay pinned to whatever radius
    was current at goal creation.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return 0
    motion_tol = _motion_step_tolerance(ws)
    if motion_tol <= 0.0:
        return 0
    lifted = 0

    def _walk(node: GoalNode) -> None:
        nonlocal lifted
        cond = getattr(node, "condition", None)
        if isinstance(cond, AtPosition) and cond.tolerance <= 0.0:
            node.condition = AtPosition(
                pos=tuple(cond.pos),
                entity_id=cond.entity_id,
                tolerance=motion_tol,
            )
            lifted += 1
        for child in getattr(node, "children", None) or []:
            _walk(child)

    _walk(goal.root)
    return lifted


def _expand_node(node:     GoalNode,
                 ws:       WorldState,
                 depth:    int,
                 step:     int,
                 expanded: List[str]) -> None:
    """Recursively expand ATOM leaves using committed CausalClaims."""
    if depth <= 0:
        return

    # Recurse into composites first so we don't modify a structure while
    # iterating it at a higher level.
    if node.node_type in (NodeType.AND, NodeType.OR):
        for child in list(node.children):
            _expand_node(child, ws, depth - 1, step, expanded)
        return

    if node.node_type != NodeType.ATOM or node.condition is None:
        return

    causal_claims = _committed_causals_for_condition(ws, node.condition)
    if not causal_claims:
        return

    # Build subgoal structure:
    #   If one causal claim:  replace ATOM leaf with AND(trigger_atom, original_atom_condition_check)
    #     — the simplest decomposition
    #   If multiple:          OR over AND branches
    new_children: List[GoalNode] = []
    for cc_hypothesis in causal_claims:
        causal: CausalClaim = cc_hypothesis.claim
        # GAP 9 bridge — if the causal trigger is an
        # ``EntityInState(X, P, V)``, the planner has no way to plan
        # *toward* that state: the state-change is the game mechanic
        # whose mechanism we are still learning.  But if entity X's
        # position is known (live position / centroid / bbox / stored
        # initial_position), "agent arriving at X's position" is a
        # plannable surrogate that the Mediator-committed CausalClaim
        # already asserts is causally sufficient.  Substitute an
        # ``AtPosition(pos=entity_pos, entity_id='agent')`` trigger
        # and tag the rationale so downstream debugging can audit the
        # bridge.  Leave all other trigger kinds untouched.
        trigger_cond, bridge_note = _bridged_trigger_condition(causal.trigger, ws)
        trig_source = ("engine:derived-from-causal-via-entity-pos"
                       if bridge_note else "engine:derived-from-causal")
        trig_atom = GoalNode(
            id         = f"{node.id}::trig::{cc_hypothesis.id}",
            node_type  = NodeType.ATOM,
            condition  = trigger_cond,
            status     = GoalStatus.OPEN,
            supporting_hypothesis_ids = [cc_hypothesis.id],
            source     = trig_source,
            created_at = step,
        )
        new_children.append(trig_atom)

    if len(new_children) == 1:
        # Simple decomposition: achieve the trigger, then the effect holds.
        # Represent by converting this ATOM into an AND whose children are
        # the trigger atom followed by a verifier atom (the original effect).
        verifier = GoalNode(
            id         = f"{node.id}::verify",
            node_type  = NodeType.ATOM,
            condition  = node.condition,
            status     = GoalStatus.OPEN,
            source     = "engine:derived-from-causal",
            created_at = step,
            # Effect side of a CausalClaim-derived AND.  The planner
            # has no transition model for this condition (that's why
            # the CausalClaim exists in the first place), so we mark
            # it deferred: the trigger sibling gets planned normally;
            # the verifier is checked only at runtime.
            deferred_plan = True,
        )
        node.node_type = NodeType.AND
        node.children  = [new_children[0], verifier]
        node.condition = None  # AND nodes have no condition
        node.ordering  = Ordering.SEQUENTIAL
        node.supporting_hypothesis_ids = list(new_children[0].supporting_hypothesis_ids)
        expanded.append(node.id)
        _emit(ws, _tel.GoalDerived(
            parent_id       = node.id,
            child_id        = new_children[0].id,
            derivation_kind = "causal:single-trigger",
        ), subject=node.id, step=step)
    else:
        # Alternatives: OR over AND branches.  Each AND has the trigger atom
        # followed by a verifier for the original effect condition.
        verifier_template = node.condition
        or_children: List[GoalNode] = []
        for trig in new_children:
            verifier = GoalNode(
                id         = f"{trig.id}::verify",
                node_type  = NodeType.ATOM,
                condition  = verifier_template,
                status     = GoalStatus.OPEN,
                source     = "engine:derived-from-causal",
                created_at = step,
                deferred_plan = True,   # effect of CausalClaim — not
                                        # directly plannable, see
                                        # single-branch case for rationale
            )
            and_node = GoalNode(
                id         = f"{trig.id}::and",
                node_type  = NodeType.AND,
                children   = [trig, verifier],
                ordering   = Ordering.SEQUENTIAL,
                supporting_hypothesis_ids = list(trig.supporting_hypothesis_ids),
                source     = "engine:derived-from-causal",
                created_at = step,
            )
            or_children.append(and_node)
        node.node_type = NodeType.OR
        node.children  = or_children
        node.condition = None
        node.supporting_hypothesis_ids = [h for c in new_children
                                          for h in c.supporting_hypothesis_ids]
        expanded.append(node.id)
        for branch in or_children:
            _emit(ws, _tel.GoalDerived(
                parent_id       = node.id,
                child_id        = branch.id,
                derivation_kind = "causal:or-branch",
            ), subject=node.id, step=step)


def _bridged_trigger_condition(
    trigger: Condition,
    ws:      WorldState,
) -> Tuple[Condition, Optional[str]]:
    """Given a CausalClaim trigger, return ``(effective_trigger, note)``.

    Most triggers are returned unchanged (``note=None``).  The
    substitution covers exactly one case: ``EntityInState(X, P, V)``
    where entity ``X``'s position is known — replaced with
    ``AtPosition(pos=<X's position>, entity_id='agent')`` so the
    planner's positional BFS can act on it.

    Why this bridge is principled, not a game-specific hack.

    1. The substitution fires *only* when a CausalClaim has committed
       ``EntityInState(X,P,V) -> <effect>``.  That commit already
       represents a Mediator-sourced assertion (filtered by
       ``commit_min_confidence``) that the entity reaching state V
       causes the effect.
    2. The Mediator knew about entity X; the engine's entity store
       has X's position (live / centroid / bbox / stored initial_pos).
       We are not inventing the geometry — we are translating a
       symbolic state predicate into a positional plan using an
       already-agreed-upon correspondence.
    3. The implicit assumption "agent-at-entity-position causes
       entity-state-change" is what turns this from a plan
       suggestion into a plan.  It is game-specific in general, but
       the *safeguard* is the deferred-plan verifier: after
       executing the bridge plan, the AND's verifier checks the
       original ``EntityInState`` predicate at runtime.  If reaching
       the position did not cause the state change, the AND fails
       to close, the selector moves on, and the engine can revise.

    Robotics analogue.  A mediator-sourced
    ``EntityInState(fridge_door_01, property=state, value=open)``
    trigger becomes ``AtPosition(pos=<door_handle>, entity_id='gripper')``
    when the door's grip point is known — same pattern: substitute
    a plannable positional surrogate, verify the state change at
    runtime.

    Returns
    -------
    (Condition, Optional[str])
        The (possibly substituted) condition and a short note
        describing the substitution ("entity-pos bridge: scan_01
        (36.0, 12.0)") or ``None`` if the trigger was returned
        unchanged.
    """
    # Case (a) — raw AtPosition with tolerance=0 from a Mediator
    # reply.  Mediator coordinates come from a pixel frame; the
    # agent moves on a coarser motor lattice.  Attach the motion-
    # derived Chebyshev radius so the planner can close within a
    # motor step.  No-op if no motion model has committed yet
    # (tolerance stays 0; the engine will retry next step once
    # motion model is learned).
    if isinstance(trigger, AtPosition) and trigger.tolerance <= 0.0:
        tol = _motion_step_tolerance(ws)
        if tol > 0.0:
            bridged = AtPosition(pos=tuple(trigger.pos),
                                 entity_id=trigger.entity_id,
                                 tolerance=tol)
            return bridged, f"motion-step tol: {trigger.pos} tol={tol}"
        return trigger, None

    # Case (b) — EntityInState substituted for an entity position.
    if not isinstance(trigger, EntityInState):
        return trigger, None
    pos = _position_for_entity(ws, trigger.entity_id)
    if pos is None:
        return trigger, None
    tol = _tolerance_for_entity(ws, trigger.entity_id)
    bridged = AtPosition(pos=tuple(pos), entity_id="agent", tolerance=tol)
    note = f"entity-pos bridge: {trigger.entity_id} {pos} tol={tol}"
    return bridged, note


def _motion_step_tolerance(ws: WorldState) -> float:
    """Half the largest committed motion-model step magnitude.

    Used as a default Chebyshev tolerance for planner-facing
    ``AtPosition`` goals whose coordinates come from an oracle
    (Mediator) living on a finer lattice than the agent's motor
    step.  Returns ``0.0`` when no motion model has committed
    yet — the engine will retry on a later step.
    """
    from .claims import MotionModelClaim
    best = 0.0
    for h in _store.committed(ws):
        claim = h.claim
        if not isinstance(claim, MotionModelClaim):
            continue
        try:
            mag = max(abs(float(claim.delta[0])), abs(float(claim.delta[1])))
        except (TypeError, ValueError, IndexError):
            continue
        if mag > best:
            best = mag
    return best / 2.0


def _committed_causals_for_condition(ws:        WorldState,
                                     condition: Condition) -> List:
    """Return committed CausalClaim hypotheses whose effect matches
    the given condition.

    Uses ``effect.matches(condition)`` (asymmetric subsumption) rather
    than direct canonical-key equality so adapter-defined Conditions
    like ``LevelProgressed`` (parameterless) can subsume parameterised
    siblings like ``LevelAdvanced(N)``.  See
    ``Condition.matches`` for the contract.

    Default behaviour is unchanged: when neither side overrides
    ``matches``, the comparison falls through to canonical-key
    equality as before.
    """
    out = []
    for h in _store.committed(ws):
        if isinstance(h.claim, CausalClaim):
            try:
                if h.claim.effect.matches(condition):
                    out.append(h)
            except Exception:
                # Defensive — a buggy matches() impl must not break
                # the chain-derivation pass.  Fall back to canonical
                # key equality for this claim.
                if h.claim.effect.canonical_key() == condition.canonical_key():
                    out.append(h)
    return out


# ---------------------------------------------------------------------------
# Trigger-discovery walker for EntitiesEquivalent goals
# ---------------------------------------------------------------------------


# Default priority bump applied to an EntitiesEquivalent goal when
# the walker successfully expands it into a cell-targeted sub-goal.
# +0.02 is small enough not to cross category boundaries (curiosity
# 0.75; pickup 0.91; rotator-interact 0.92; alignment 0.95; level-
# advance 1.00) but enough to disambiguate the now-plannable goal
# from competitors at the same priority level.  See
# ``docs/SPEC_trigger_discovery_walker.md`` §"Priority bump magnitude".
_EE_WALKER_PRIORITY_BUMP: float = 0.02


def _trig_cell_visit_count_for_goal(
    ws:      WorldState,
    goal_id: str,
    cell:    "Tuple[int, int]",
) -> int:
    """Phase 4 helper: count how many times the trajectory shows
    a confirmed visit to ``cell`` while pursuing ``goal_id``.

    Reads the closed-loop substrate's trajectory.  An outcome
    counts as a "visit" when its prediction's active_goal_id
    matches and the overall_match is CONFIRMED (i.e., the agent
    successfully reached its target cell, and that target cell
    was ``cell``).

    Used by the goal-tree refresh logic to detect "agent has
    visited the trigger but the dim is still open — try a
    different cell."  Returns zero when the trajectory or the
    indices haven't been set up yet (defensive against early
    construction).
    """
    try:
        from .types import MatchKind
    except Exception:
        return 0
    trajectory = getattr(ws, "trajectory", None)
    if trajectory is None or not getattr(trajectory, "outcomes", None):
        return 0
    cell_t = (int(cell[0]), int(cell[1]))
    visit_count = 0
    for outcome in trajectory.outcomes:
        pred = outcome.prediction
        if str(pred.active_goal_id or "") != str(goal_id):
            continue
        if outcome.overall_match != MatchKind.CONFIRMED:
            continue
        # Confirmed-positional outcome means the agent reached the
        # asserted cell.  Check that the asserted cell matches.
        for assertion in pred.predicted_assertions:
            try:
                tk = assertion.condition.canonical_key()
            except Exception:
                continue
            if (tk and len(tk) >= 3
                    and tk[0] in ("AgentAtCell", "AtPosition")
                    and (int(tk[1]), int(tk[2])) == cell_t):
                visit_count += 1
                break
    return visit_count


def _maybe_reset_ee_and_for_reexpansion(ws: WorldState, root: GoalNode) -> None:
    """If a walker-expanded EE goal (AND[trig, verifier]) has its
    trig-dim already matched but at least one verifier dim still
    open, reset root back to an ATOM with the verifier's
    ``EntitiesEquivalent`` condition.  The caller's expansion logic
    then re-runs and picks a fresh trigger for an open dim.

    No-op when:

    * root isn't AND with the expected (trig, verifier) shape, or
    * the trig's dim is still open (current expansion still useful), or
    * no dims are matched yet (no point re-targeting), or
    * all dims are matched (verifier should close on its own next tick).

    The trig atom's id encodes its emit-for-dim as
    ``f"{root.id}::trig::{primary_dim}::{hypothesis.id}"``;
    we parse the dim out and check its current matched status.
    """
    if root.node_type != NodeType.AND:
        return
    verifier: "Optional[GoalNode]" = None
    existing_trig: "Optional[GoalNode]" = None
    for child in (root.children or []):
        if isinstance(child.condition, EntitiesEquivalent):
            verifier = child
        elif "::trig::" in str(child.id):
            existing_trig = child
    if verifier is None or existing_trig is None:
        return
    ee_cond = verifier.condition
    target = ws.entities.get(ee_cond.target_id)
    reference = ws.entities.get(ee_cond.reference_id)
    if target is None or reference is None:
        return
    open_dims: "set[str]" = set()
    matched_any = False
    for dim in ee_cond.dimensions:
        t_val = target.properties.get(dim)
        r_val = reference.properties.get(dim)
        if t_val is None or r_val is None:
            open_dims.add(str(dim))
        elif t_val != r_val:
            open_dims.add(str(dim))
        else:
            matched_any = True
    if not open_dims:
        return  # everything matched; let the verifier close
    if not matched_any:
        return  # no dim has matched yet; current trig is still doing useful work
    # Parse current trig's emit-for-dim from the id.
    parts = str(existing_trig.id).split("::trig::")
    cur_dim: "Optional[str]" = None
    if len(parts) >= 2:
        cur_dim = parts[1].split("::")[0]
    if cur_dim in open_dims:
        return  # current trig still addresses an open dim; leave it
    # The trig's dim is closed and other dims remain open.  Reset root
    # to ATOM(EE) so the caller's expansion path picks a fresh trig
    # for one of the still-open dims.
    root.node_type      = NodeType.ATOM
    root.condition      = ee_cond
    root.children       = []
    root.ordering       = None
    root.supporting_hypothesis_ids = []


# Phase 4 threshold: number of confirmed visits to a trigger cell
# (with the cell's emit-for-dim still open) before the walker
# concludes its trigger pick was wrong and should be re-chosen.
# Two is a sensible default — one visit might be the agent
# arriving without the predicted change yet propagated; two
# visits without progress is strong evidence the prediction was
# wrong.
_TRIG_VISIT_RETRACT_THRESHOLD: int = 2


def _maybe_reset_ee_and_for_failed_trig(
    ws:      WorldState,
    root:    GoalNode,
    goal_id: str,
) -> "Optional[Tuple[int, int]]":
    """Phase 4: trajectory-driven reset.  If the existing trigger's
    cell has been visited at least ``_TRIG_VISIT_RETRACT_THRESHOLD``
    times for this goal but the trig's emit-for-dim is still open,
    the walker's pick was wrong.  Reset root to ATOM and return
    the failed cell so the caller can exclude it from the next
    trigger pick.

    Returns the cell tuple of the retracted trigger when reset
    occurred; returns ``None`` when the existing trig is still
    doing useful work (or no AND structure to reset).

    Complements :func:`_maybe_reset_ee_and_for_reexpansion` which
    fires when the trig's dim has CLOSED.  This function fires
    when the trig has been TRIED unsuccessfully — both produce a
    re-pick, but for different reasons.
    """
    if root.node_type != NodeType.AND:
        return None
    verifier: "Optional[GoalNode]" = None
    existing_trig: "Optional[GoalNode]" = None
    for child in (root.children or []):
        if isinstance(child.condition, EntitiesEquivalent):
            verifier = child
        elif "::trig::" in str(child.id):
            existing_trig = child
    if verifier is None or existing_trig is None:
        return None

    # Parse the trig's cell from its condition.
    try:
        tk = existing_trig.condition.canonical_key()
    except Exception:
        return None
    if not (tk and len(tk) >= 3 and tk[0] in ("AgentAtCell", "AtPosition")):
        return None
    try:
        cell = (int(tk[1]), int(tk[2]))
    except (TypeError, ValueError):
        return None

    # Has the agent visited this cell enough times for this goal
    # without the dim closing?
    visit_count = _trig_cell_visit_count_for_goal(ws, goal_id, cell)
    if visit_count < _TRIG_VISIT_RETRACT_THRESHOLD:
        return None

    # Check that the trig's dim is still open.  If matched, the
    # other reset path handles it; we only fire on the failed-trig
    # path.
    ee_cond = verifier.condition
    target = ws.entities.get(ee_cond.target_id)
    reference = ws.entities.get(ee_cond.reference_id)
    if target is None or reference is None:
        return None
    parts = str(existing_trig.id).split("::trig::")
    cur_dim: "Optional[str]" = None
    if len(parts) >= 2:
        cur_dim = parts[1].split("::")[0]
    if cur_dim is None:
        return None
    t_val = target.properties.get(cur_dim)
    r_val = reference.properties.get(cur_dim)
    if t_val is not None and r_val is not None and t_val == r_val:
        return None  # dim is matched; let the closed-dim mechanism handle it

    # Reset to ATOM, signal which cell was the failed pick.
    root.node_type      = NodeType.ATOM
    root.condition      = ee_cond
    root.children       = []
    root.ordering       = None
    root.supporting_hypothesis_ids = []
    return cell


def derive_subgoals_for_entities_equivalent(
    ws:            WorldState,
    goal_id:       str,
    *,
    priority_bump: float = _EE_WALKER_PRIORITY_BUMP,
    step:          int   = 0,
) -> List[str]:
    """Expand an :class:`EntitiesEquivalent` goal into a cell-targeted
    sub-goal derived from a committed cell-keyed
    :class:`CausalClaim`.

    Sibling to :func:`derive_subgoals_from_causal` — same
    backward-chaining pattern, different match rule.  The existing
    function matches CausalClaim effects against the leaf condition
    by canonical-key equality (or :meth:`Condition.matches`).  This
    function matches by per-dimension property containment: claims
    whose effect is ``EntityInState(_, P, _)`` where ``P`` is one of
    the goal's currently-open dimensions.

    See ``docs/SPEC_trigger_discovery_walker.md`` for full context;
    in short, the 2026-05-02 alignment work made
    ``EntitiesEquivalent({palette, orientation, ...})`` expressible
    but did not make it actionable: the goal has no spatial target
    so the cell-reachability filter in goal selection skips it
    every turn.  This walker translates "make these two entities
    equivalent on dim D" into "go to the cell of an entity whose
    interaction has been observed to change D" — the missing
    decomposition primitive.

    v1 scope (per spec).

    * Cell-keyed triggers only.  A claim whose ``trigger`` has
      canonical key ``("AgentAtCell", r, c)`` becomes a sub-goal
      with that trigger as its ATOM condition; other trigger
      kinds (``ActionJustTaken``, ``EntityInState``) are not
      handled here — the existing causal-decomposer or curiosity
      probes cover those paths.
    * Single sub-goal per expansion.  If multiple cells flip
      properties in the open-dim set, the highest-credence claim
      wins.  GF-PRIMARY's cell extractor returns only the first
      matching ATOM in the tree, so emitting a multi-cell OR
      gives no planner benefit; the surviving claim is enough
      for v1.  Multi-cell cases will be addressed in v2.
    * No re-expansion.  Once root transitions out of ATOM,
      subsequent calls are no-ops.  v2 may re-expand once the
      first trigger fires and a new dim becomes the active
      blocker.

    Returns the list of newly-expanded node ids (matches the
    signature of :func:`derive_subgoals_from_causal`).  An empty
    list means the goal was already expanded, the goal's root is
    not an ``EntitiesEquivalent`` ATOM, all dimensions are
    currently closed, or no committed cell-keyed claim addresses
    any open dimension.

    Side effect: on successful expansion, the goal's top-level
    ``priority`` is bumped by ``priority_bump`` (capped at 0.99) so
    the now-plannable cell sub-goal beats competing same-band
    goals at selection time.  The bump is monotone — re-running
    the walker on an already-expanded goal does NOT re-bump.

    Robotics analogue: a goal expressing "the gripper's pose
    matches the part's pose" decomposes into "be at the cell of
    the rotation actuator (whose past use changed gripper pose)"
    — same primitive, different domain.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return []

    root = goal.root
    # Re-expansion path (2026-05-02): if a previous expansion produced
    # an AND[trig, verifier] but the trig's targeted dim has since
    # matched (and other dims remain open), reset root back to an
    # EntitiesEquivalent ATOM so the expansion logic below picks a
    # NEW trigger for one of the still-open dims.  This addresses the
    # ls20 L2 regression where the agent kept returning to the
    # initial trigger cell after its dim closed, undoing the match.
    #
    # Phase 4 (2026-05-02 PM): also reset when the trig's cell has
    # been visited multiple times without closing the dim — the
    # walker's pick was wrong; retract it and exclude that cell
    # from the next pick.
    failed_cells: "set[Tuple[int, int]]" = set()
    if root.node_type == NodeType.AND:
        _maybe_reset_ee_and_for_reexpansion(ws, root)
    if root.node_type == NodeType.AND:
        failed_cell = _maybe_reset_ee_and_for_failed_trig(ws, root, goal_id)
        if failed_cell is not None:
            failed_cells.add(failed_cell)
    if root.node_type != NodeType.ATOM:
        return []  # already expanded (or composite from declaration)
    cond = root.condition
    # Determine the alignment shape: EntitiesEquivalent (named dims)
    # or a duck-typed alignment condition (e.g. OrientationAligned)
    # whose canonical-key prefix names it and whose implicit
    # dimension is "orientation".  Domain-neutral: we don't import
    # OrientationAligned (it lives in the adapter); we recognise it
    # by the published canonical-key string.
    open_dims: set = set()
    target_id_for_lookup: Optional[str] = None
    reference_id_for_lookup: Optional[str] = None
    if isinstance(cond, EntitiesEquivalent):
        if not cond.dimensions:
            return []  # vacuous; nothing to chain on
        target_id_for_lookup = cond.target_id
        reference_id_for_lookup = cond.reference_id
        target = ws.entities.get(cond.target_id)
        reference = ws.entities.get(cond.reference_id)
        # Refuse expansion when entity refs don't resolve to live
        # entities.  Without entities to read property values from,
        # the walker can't tell which dims are matched vs. open —
        # the inner loop's `t_val/r_val is None` branch optimistically
        # treats all dims as open and expands to triggers for every
        # one, including dims already matched (causing the agent to
        # cycle a trigger that regresses an already-aligned dim).
        # Same shape as the trusted_win_conditions stale-seed
        # deferral: defer rather than guess.  Walker re-fires every
        # tick, so expansion happens automatically once entities
        # publish (or the operator corrects the refs).
        if target is None or reference is None:
            try:
                _flag_key = (
                    f"_ee_walker_unresolved_logged:{goal_id}"
                )
                _arc = (getattr(ws, "agent", {}) or {})
                if not _arc.get(_flag_key, False):
                    _arc[_flag_key] = True
                    _missing = []
                    if target is None:
                        _missing.append(
                            f"target={cond.target_id!r}")
                    if reference is None:
                        _missing.append(
                            f"reference={cond.reference_id!r}")
                    print(f"[ee-walker] {goal_id}: deferring "
                          f"expansion ({', '.join(_missing)} not "
                          f"resolvable in ws.entities)")
            except Exception:
                pass
            return []
        for dim in cond.dimensions:
            t_val = target.properties.get(dim)
            r_val = reference.properties.get(dim)
            if t_val is None or r_val is None:
                open_dims.add(str(dim))
            elif t_val != r_val:
                open_dims.add(str(dim))
    else:
        # Duck-typed alignment condition: must publish target_id /
        # reference_id and a recognised canonical-key name.
        target_id_for_lookup = getattr(cond, "target_id", None)
        reference_id_for_lookup = getattr(cond, "reference_id", None)
        if target_id_for_lookup is None or reference_id_for_lookup is None:
            return []
        try:
            ck = cond.canonical_key()
            ck_name = str(ck[0]) if ck else ""
        except Exception:
            return []
        # Implicit dimension table.  When other alignment-shaped
        # conditions appear they extend this map without changing the
        # walker's structure.
        _implicit_dim_for_condition = {
            "OrientationAligned": "orientation",
        }
        implicit_dim = _implicit_dim_for_condition.get(ck_name)
        if implicit_dim is None:
            return []
        # The condition is "open" unless its evaluate() returns True.
        try:
            verdict = cond.evaluate(ws) if hasattr(cond, "evaluate") else None
        except Exception:
            verdict = None
        if verdict is True:
            return []  # already aligned; nothing to derive
        open_dims.add(implicit_dim)
    if not open_dims:
        return []

    # Scan CausalClaims for a cell-keyed trigger whose effect property
    # is in the open-dim set.  Group by trigger canonical key so the
    # same cell from multiple claims collapses to one entry; keep the
    # highest-credence representative.
    #
    # Credence threshold: we use ``above_credence(ws, 0.5)`` rather
    # than the stricter ``committed(ws)`` (≥0.85).  The walker is
    # making a planning decision (which cell to target as a sub-goal),
    # not a knowledge commitment.  Even partial-evidence claims are
    # useful: if the agent visits and observes no flip, the claim
    # demotes naturally and the walker re-evaluates next turn.
    # Important for cold-start cases like dialogic-driven goals where
    # no prior evidence has accumulated to reach the commit threshold.
    # Multi-trigger decomposition (v2 of SPEC_trigger_discovery_walker.md):
    # for each open dim, find the best (highest-credence, non-failed)
    # cell-keyed trigger.  Distinct dims may share a trigger cell
    # (one cell flips multiple dims) — those collapse to a single
    # sub-goal.  Different cells per dim each get their own sub-goal.
    #
    # v1 picked a single best trigger across all open dims, which left
    # multi-dim alignments stuck on whichever dim's trigger happened to
    # win the credence tiebreak: visiting that one trigger doesn't
    # advance the other dims, but the walker doesn't re-expand to pick
    # a second trigger because its re-expansion gate keys on the
    # picked dim having matched (and that dim may already be matched
    # at level entry, so visits produce no signal).  v2 emits all
    # trigger sub-goals up front so the harness routes through each.
    best_per_dim: dict[str, tuple[float, "Hypothesis"]] = {}
    for h in _store.above_credence(ws, 0.5):
        claim = h.claim
        if not isinstance(claim, CausalClaim):
            continue
        eff = claim.effect
        if not isinstance(eff, EntityInState):
            continue
        if eff.property not in open_dims:
            continue
        trig = claim.trigger
        try:
            tk = trig.canonical_key()
        except Exception:
            continue
        # Cell-keyed trigger filter — by canonical-key prefix to keep
        # this module domain-neutral (AgentAtCell lives in the ARC
        # adapter's goal_conditions, not in cognitive_os).
        if not (tk and len(tk) >= 3 and tk[0] == "AgentAtCell"):
            continue
        # Phase 4: exclude cells that have already been tried for this
        # goal without closing the dim (the trajectory-driven reset
        # above retracted them; pulling the same cell again would
        # immediately retract again).
        try:
            cell_key = (int(tk[1]), int(tk[2]))
        except (TypeError, ValueError):
            continue
        if cell_key in failed_cells:
            continue
        cred = float(getattr(h.credence, "point", 0.0))
        prop = str(eff.property)
        cur = best_per_dim.get(prop)
        if cur is None or cred > cur[0]:
            best_per_dim[prop] = (cred, h)

    if not best_per_dim:
        return []

    # Collapse duplicate trigger cells: when two open dims share the
    # same trigger (one cell flips both), emit a single sub-goal
    # carrying the canonical-key-equal trigger condition.  Use the
    # canonical key as the dedupe key so different hypothesis ids
    # pointing at the same cell collapse cleanly.
    triggers: list[tuple[str, "Hypothesis"]] = []
    seen_trigger_keys: set = set()
    for dim, (_cred, hyp) in best_per_dim.items():
        try:
            tkey = hyp.claim.trigger.canonical_key()
        except Exception:
            continue
        if tkey in seen_trigger_keys:
            continue
        seen_trigger_keys.add(tkey)
        triggers.append((dim, hyp))

    if not triggers:
        return []

    # Build trigger atoms (one per distinct trigger cell) and a
    # verifier carrying the original EntitiesEquivalent condition.
    # Mutate root from ATOM to AND in-place.  Children are
    # UNORDERED — the harness can route through any trigger first;
    # only the verifier requires all triggers to have closed their
    # dims by the time the goal evaluates.
    children: list[GoalNode] = []
    supporting_ids: list[str] = []
    for dim, hyp in triggers:
        children.append(GoalNode(
            id         = f"{root.id}::trig::{dim}::{hyp.id}",
            node_type  = NodeType.ATOM,
            condition  = hyp.claim.trigger,
            status     = GoalStatus.OPEN,
            supporting_hypothesis_ids = [hyp.id],
            source     = "engine:ee-trigger-discovery",
            created_at = step,
        ))
        supporting_ids.append(hyp.id)
    verifier = GoalNode(
        id          = f"{root.id}::verify",
        node_type   = NodeType.ATOM,
        condition   = root.condition,
        status      = GoalStatus.OPEN,
        source      = "engine:ee-trigger-discovery",
        created_at  = step,
        deferred_plan = True,
    )
    children.append(verifier)
    root.node_type = NodeType.AND
    root.children  = children
    root.condition = None
    # UNORDERED so the harness's cell-target extractor can pick any
    # trigger first (whichever is cheapest from the agent's current
    # cell), not just the leftmost child.
    root.ordering  = Ordering.UNORDERED
    root.supporting_hypothesis_ids = supporting_ids

    # Bump goal priority so the now-cell-targeted sub-goals beat
    # competing cell goals at the same band.
    bumped = min(0.99, float(goal.priority) + float(priority_bump))
    goal.priority = bumped

    # Emit one GoalDerived event per trigger child for telemetry
    # parity with the v1 single-child path.
    for child in children[:-1]:  # all but verifier
        _emit(ws, _tel.GoalDerived(
            parent_id       = root.id,
            child_id        = child.id,
            derivation_kind = "ee:trigger-discovery",
        ), subject=root.id, step=step)

    return [root.id]


def derive_destabilizing_triggers_for_matched_dims(
    ws: WorldState,
) -> "frozenset[Condition]":
    """Trigger Conditions whose firing would regress a currently-
    matched dimension of an :class:`EntitiesEquivalent` (or
    OrientationAligned) goal.

    Domain-agnostic by design: returns the trigger ``Condition``
    objects themselves.  Adapters consume the set and extract
    whatever they need — cells (ARC nav), joint configurations
    (manipulator), tool references (warehouse robot), etc. — by
    inspecting each trigger's published structure.  The engine
    layer never knows what a "cell" is.

    Closed-dim direction of the trigger-discovery walker — the
    counterpart to :func:`derive_subgoals_for_entities_equivalent`.
    Same CausalClaim query, opposite consumer:

    * v1 (open-dim): for each OPEN dim of an EE goal, derive a
      sub-goal whose target is the trigger of an entity whose
      interaction has been observed to change that dim.  Drives
      the agent TO the trigger.
    * v2 (this function): for each currently-MATCHED dim of any
      EE goal (whether the parent goal is fully ACHIEVED or only
      partially aligned), the same triggers become
      *state-destabilizing* — the adapter routes AROUND them in
      any subsequent navigation plan, since firing them would
      flip a property the goal currently has at the desired value.

    Per-dimension semantics (refined 2026-05-02 after observing
    on ls20 L2 that the agent achieved palette match, then walked
    back through the color-changer en route to the orientation
    trigger and regressed palette).  The earlier whole-goal-only
    semantics didn't engage when the goal had multiple dims and
    only some were matched.  Per-dim covers partial-match cases:

      Goal: EntitiesEquivalent(t, r, {palette, orientation})
      Currently: t.palette == r.palette  (matched)
                 t.orientation != r.orientation  (open)
      Walker v1 sees orientation as open → emits sub-goal at the
        rotator cell.
      Walker v2 sees palette as currently-matched → marks the
        color-changer cell as destabilizing.
      BFS path to the rotator routes around the color-changer.
      No regression.

    Self-correcting: when a dim un-matches (regression DID happen
    for some other reason), that dim's cells drop from the set;
    BFS lets the agent through; agent re-fires the trigger.

    A typical adapter use (ARC nav):

      .. code-block:: python

         triggers = derive_destabilizing_triggers_for_matched_dims(ws)
         destabilizing_cells = {
             (int(ck[1]), int(ck[2]))
             for trig in triggers
             for ck in [trig.canonical_key()]
             if ck and len(ck) >= 3 and ck[0] == "AgentAtCell"
         }
         walls_for_bfs = base_walls | destabilizing_cells
         plan = bfs(start, target, walls=walls_for_bfs)

    The cell-extraction filter (``ck[0] == "AgentAtCell"``) lives
    in the adapter, not here, because what the trigger represents
    is domain-specific.  A robotics adapter might extract joint
    configurations from ``JointAngleAt`` triggers, or grasps from
    ``ToolGrasped`` triggers.  The engine produces the set; the
    adapter projects it onto its navigation primitive.

    Why this lives in cognitive_os, not the ARC adapter.  The
    abstraction is "trigger conditions whose firing would regress
    a currently-matched alignment dimension" — pure read over
    goal-forest state and CausalClaims, both domain-agnostic.

    Credence threshold: ``above_credence(ws, 0.5)`` — symmetric
    with walker v1.  Earlier versions used ``committed(ws)``
    (~0.85), but cold-start cases (dialogic-driven goals) lacked
    enough corroborating observations for any claim to commit,
    so the wall never engaged.  The trade-off: a low-credence
    claim that turns out to be wrong leads the adapter's filter
    to take a slightly longer path; the claim demotes naturally
    on contradiction.

    Coverage:
      * ``EntitiesEquivalent`` goals (per-dim matched semantics).
      * ``OrientationAligned`` goals (alignment-achieved semantics) —
        recognized via ``canonical_key`` prefix string so this module
        does not need to import the adapter-side condition class.
        When such a goal currently evaluates True (alignment achieved
        per ``principal_axis_angle`` agreement), CausalClaims whose
        effect changes ANY orientation-relevant property
        (``principal_axis_angle``, ``orientation``,
        ``orientation_index``, ``bitmap_id``) of EITHER referenced
        entity become destabilizing.

    Returns a frozenset of trigger ``Condition`` objects.
    """
    out: "set[Condition]" = set()
    for gid, goal in ws.goal_forest.goals.items():
        root = goal.root
        if root is None:
            continue
        # Skip goals that are explicitly abandoned/pruned — destabilizing
        # cells from those have no current consumer.
        if root.status in (GoalStatus.ABANDONED, GoalStatus.PRUNED):
            continue

        # ---- EntitiesEquivalent path ----
        ee_cond: "Optional[EntitiesEquivalent]" = None
        if isinstance(root.condition, EntitiesEquivalent):
            ee_cond = root.condition
        else:
            for child in (root.children or []):
                if isinstance(child.condition, EntitiesEquivalent):
                    ee_cond = child.condition
                    break
        if ee_cond is not None:
            dims = set(str(d) for d in ee_cond.dimensions)
            if dims:
                target = ws.entities.get(ee_cond.target_id)
                reference = ws.entities.get(ee_cond.reference_id)
                if target is not None and reference is not None:
                    matched_dims: "set[str]" = set()
                    for dim in dims:
                        t_val = target.properties.get(dim)
                        r_val = reference.properties.get(dim)
                        if t_val is None or r_val is None:
                            continue
                        if t_val == r_val:
                            matched_dims.add(dim)
                    # Build the set of acceptable effect entity ids: the
                    # explicit target_id and reference_id PLUS each of
                    # their cross-tier identity aliases (bitmap_id /
                    # shape_id / topo_id / scaled_id / instance_id).  A
                    # CausalClaim's effect may reference the same
                    # underlying EntityModel under any of those tiers;
                    # without this expansion the filter misses claims
                    # that share an entity but use a different tier id.
                    #
                    # Operator's framing 2026-05-04: this filter was
                    # MISSING entirely from the EE path, while the OA
                    # path below already had its analogue (oa_subjects
                    # check at line ~1268).  The asymmetry caused the
                    # destabilizing-cells set to include cells whose
                    # CausalClaims targeted unrelated entities (budget
                    # meter, pickup, etc.), bloating the set so that
                    # BFS could find no path through cells that were
                    # otherwise reachable.  Run #4 (Gemma) hit this
                    # acutely: 30 BFS-no-path failures on the trivial
                    # route (0, 4) -> (-4, 4) because every transit
                    # cell along column 4 was wrongly forbidden.
                    _ID_TIERS = ("bitmap_id", "shape_id", "topo_id",
                                 "scaled_id", "instance_id")
                    ee_subjects: "set[str]" = {
                        str(ee_cond.target_id), str(ee_cond.reference_id),
                    }
                    for ent in (target, reference):
                        ep = getattr(ent, "properties", None) or {}
                        for tier in _ID_TIERS:
                            v = ep.get(tier)
                            if v is not None:
                                ee_subjects.add(str(v))
                    if matched_dims:
                        for h in _store.above_credence(ws, 0.5):
                            claim = h.claim
                            if not isinstance(claim, CausalClaim):
                                continue
                            eff = claim.effect
                            if not isinstance(eff, EntityInState):
                                continue
                            if eff.property not in matched_dims:
                                continue
                            # Entity-identity filter: the claim's effect
                            # must touch one of the EE goal's protected
                            # entities (or any of their tier aliases).
                            eff_ent = getattr(eff, "entity_id", None)
                            if eff_ent is not None and str(eff_ent) not in ee_subjects:
                                continue
                            # Add the trigger Condition itself.  The
                            # adapter (e.g. ARC nav) extracts whatever
                            # primitive it needs from the canonical key
                            # — cells, joint configs, etc.  The engine
                            # makes no assumption about what the trigger
                            # represents.
                            out.add(claim.trigger)
            continue  # done with this goal

        # ---- OrientationAligned path (duck-typed by canonical_key) ----
        oa_cond = root.condition
        try:
            ck = oa_cond.canonical_key() if oa_cond is not None else None
        except Exception:
            ck = None
        is_oa = (
            isinstance(ck, tuple)
            and len(ck) >= 3
            and ck[0] == "OrientationAligned"
        )
        if not is_oa:
            continue
        # Only treat the goal's matched cells as destabilizing when the
        # condition currently evaluates True (alignment achieved); if
        # it's open or indeterminate, the agent still needs the trigger
        # to make progress.
        try:
            evaluated = oa_cond.evaluate(ws)
        except Exception:
            evaluated = None
        if evaluated is not True:
            continue
        # OrientationAligned uses target_id and reference_id attributes
        # exposed by the adapter's class (see goal_conditions.py).
        target_id = getattr(oa_cond, "target_id", None)
        reference_id = getattr(oa_cond, "reference_id", None)
        if not (target_id and reference_id):
            continue
        oa_subjects = {str(target_id), str(reference_id)}
        # Domain-neutral set of properties whose change would alter
        # orientation in the world frame.  Includes the published
        # principal_axis_angle from the principal-axis miner plus the
        # legacy adapter-published "orientation" / "orientation_index"
        # and "bitmap_id" (rotation breaks bitmap canonicalisation).
        orientation_props = frozenset({
            "principal_axis_angle",
            "orientation",
            "orientation_index",
            "bitmap_id",
        })
        for h in _store.above_credence(ws, 0.5):
            claim = h.claim
            if not isinstance(claim, CausalClaim):
                continue
            eff = claim.effect
            if not isinstance(eff, EntityInState):
                continue
            if eff.property not in orientation_props:
                continue
            # Effect must touch one of the OA-referenced entities.
            eff_ent = getattr(eff, "entity_id", None)
            if eff_ent is not None and str(eff_ent) not in oa_subjects:
                continue
            out.add(claim.trigger)
    return frozenset(out)


# ---------------------------------------------------------------------------
# Backward-compat shim: existing call sites that ask for cell tuples still
# work, but they now go through the engine's domain-agnostic surface.  The
# cell extraction is the adapter's job; the shim does it for callers that
# haven't migrated yet.
# ---------------------------------------------------------------------------


def derive_state_destabilizing_cells_for_achieved_goals(
    ws: WorldState,
) -> "frozenset[tuple[int, int]]":
    """Backward-compat: extracts cells from
    :func:`derive_destabilizing_triggers_for_matched_dims` by filtering
    triggers whose canonical key starts with ``"AgentAtCell"``.

    New code should call the trigger-returning function directly and
    do its own projection (cells for ARC nav; joint configs for a
    manipulator; etc.).  Kept here so existing tests and adapter
    sites continue to work during migration.
    """
    triggers = derive_destabilizing_triggers_for_matched_dims(ws)
    out: "set[tuple[int, int]]" = set()
    for trig in triggers:
        try:
            tk = trig.canonical_key()
        except Exception:
            continue
        if not (tk and len(tk) >= 3 and tk[0] == "AgentAtCell"):
            continue
        try:
            out.add((int(tk[1]), int(tk[2])))
        except (TypeError, ValueError):
            continue
    return frozenset(out)


# ---------------------------------------------------------------------------
# Goal synthesis from role PropertyClaims
# ---------------------------------------------------------------------------

# Priority floor for role-derived "reach the declared target" Goals
# (GAP 24b).
#
# Role claims land at ``llm_proposer`` prior credence (0.5) and rarely
# gather enough behavioural evidence to commit — so ``credence.point``
# alone puts a ``role:target:<e>`` Goal at 0.5.  That loses to
# ``ReduceUncertainty.interact`` on any low-coverage entity, whose
# priority ceilings at ``(1.0 - 0.0) * 0.60 = 0.60``.  The observed
# ls20 L1 failure (2026-04-18): the planner ignored
# ``role:target:scan_01`` (priority 0.5) in favour of
# ``reduce_uncertainty:interact:e4`` (priority 0.60) every tick, and
# the agent patrolled around non-target entities instead of reaching
# the declared target.
#
# A declared-target Goal is a *teleological* commitment: "this is the
# level's objective".  ``ReduceUncertainty`` modalities are *epistemic*
# exploration urges: "I don't know what this thing is, let me probe".
# The two live on different quality axes, and credence-of-the-role-
# claim is the wrong signal for the relative importance of the *Goal
# it implies*.
#
# Fix: clamp role-Goal priority to a floor *above* the max
# ReduceUncertainty priority that generic exploration can produce
# (interact 0.60, observe 0.80), but *below* ``probe_in_place`` (0.90)
# — because probe_in_place only fires when the agent is already
# standing inside an unknown's bbox, so letting it briefly finish a
# local probe before walking to the declared target is a good
# exchange.  Floor 0.85 fits that window.  Robotics analogue: "drive
# to the charging dock" should outrank "characterise the new IR
# pattern you just noticed", but if you're already in the new pattern
# and one probe closes it, finish the probe first.
ROLE_GOAL_PRIORITY_FLOOR: float = 0.85


def derive_goals_from_roles(
    ws:           WorldState,
    *,
    step:         int                     = 0,
    role_values:  Sequence[str]           = REACH_ROLE_VALUES,
    min_point:    float                   = 0.4,
) -> List[str]:
    """Synthesise ``reach(entity)`` Goals from ``role`` PropertyClaims.

    Watches the hypothesis store for :class:`PropertyClaim`\\s of the
    form ``property="role", value=v`` where ``v`` is one of the
    *reach* role values (``"target"`` / ``"goal"`` / ``"exit"`` by
    default; robotics adapters extend this set via the ``role_values``
    argument).  For each such claim whose credence point is at least
    ``min_point``, construct a Goal whose root ATOM condition is
    ``AtPosition(pos=<entity-position>, entity_id="agent")`` and
    register it with the goal forest.

    Deduplication is structural: a goal with id
    ``role:<role_value>:<entity_id>`` represents one role claim.
    Calling this function on later steps will not create a duplicate
    even if the underlying claim's credence changes.

    Both committed and proposed claims are considered so that Goals
    can be derived from freshly-installed LLM role hypotheses before
    they accumulate behavioural evidence (otherwise GAP 3 —
    ``llm_proposer`` claims sit at p=0.5 and never commit — blocks
    any planning).  Goal priority is set from the claim's point so
    that the planner favours higher-confidence targets.

    Position extraction, in precedence order:

    1. ``entity.properties["position"]`` — live positional coordinate
       (robotics, or later-phase ARC with tracking).
    2. ``entity.properties["centroid"]`` — Phase 4 segmentation
       convention for engine-derived entities.
    3. Centre of ``entity.properties["bbox"]`` in engine convention
       ``[x_min, y_min, x_max, y_max]`` — the minting path sets this
       for scan-sourced entities.
    4. Any ``PropertyClaim(entity_id=<e>, property="initial_position",
       value=[x, y])`` in the store — the oracle-integration path
       writes this for every enumerated object.

    If none of the four work, the entity is skipped — no Goal is
    created.  This is deliberate: a role claim without a position is
    not yet actionable, and a stale Goal pointing at None would
    confuse the planner.

    Returns a list of newly-added goal IDs.
    """
    added: List[str] = []
    role_set = set(role_values)

    # Iterate hypotheses once, picking out role claims of interest.
    for h in list(ws.hypotheses.values()):
        claim = h.claim
        if not isinstance(claim, PropertyClaim):
            continue
        if claim.property != "role":
            continue
        if claim.value not in role_set:
            continue
        if h.credence.point < min_point:
            continue

        goal_id = f"role:{claim.value}:{claim.entity_id}"
        if goal_id in ws.goal_forest.goals:
            continue

        pos = _position_for_entity(ws, claim.entity_id)
        if pos is None:
            continue
        tol = _tolerance_for_entity(ws, claim.entity_id)

        # GAP 24b: clamp to the role-goal priority floor so the
        # declared-target Goal outranks generic ReduceUncertainty
        # modalities firing on arbitrary unknowns.  See module-level
        # ``ROLE_GOAL_PRIORITY_FLOOR`` docstring for the rationale.
        goal_priority = max(float(h.credence.point), ROLE_GOAL_PRIORITY_FLOOR)

        root = GoalNode(
            id         = f"{goal_id}::root",
            node_type  = NodeType.ATOM,
            condition  = AtPosition(pos=tuple(pos), entity_id="agent", tolerance=tol),
            status     = GoalStatus.OPEN,
            priority   = goal_priority,
            source     = "engine:derived-from-role",
            # GAP 18 note: we deliberately do NOT set
            # ``supporting_hypothesis_ids = [h.id]`` here, even though
            # the role claim ``h`` provenance-caused this goal.  The
            # planner treats ``supporting_hypothesis_ids`` as a *causal
            # underwriting* link (GAP 13): a branch is unplannable
            # unless at least one supporting hypothesis is currently
            # committed.  That semantic is correct for CausalClaim-
            # derived sub-goals — if the causal claim has been demoted,
            # the branch is tested-vacuous and pruned.  But it is wrong
            # for role-goals, whose PropertyClaim sits at credence 0.5
            # by default (llm_proposer prior) and therefore never
            # commits: treating role-goals as gated by that credence
            # makes every role-goal permanently unplannable even when
            # BFS could trivially reach the target position.
            #
            # Provenance is still captured via ``source`` above, which
            # is sufficient for debugging and future lifecycle-coupling
            # hooks (e.g. close a role-goal when its role claim is
            # abandoned).  The *plannability* of a role-goal now
            # depends on mechanics (can BFS reach the target cell via
            # the committed motion model?), not on the role claim's
            # credence.  Robotics analogue: a "reach this fixture"
            # motion plan does not require certainty that the fixture
            # is the grasp target — that's a higher-level relevance
            # question, orthogonal to whether the arm can physically
            # get there.
            created_at = step,
        )
        goal = Goal(
            id         = goal_id,
            root       = root,
            priority   = goal_priority,
            source     = "engine:derived-from-role",
            created_at = step,
        )
        add_goal(ws, goal)
        added.append(goal_id)

    return added


def _tolerance_for_entity(ws: WorldState, entity_id: str) -> float:
    """Default Chebyshev tolerance for an ``AtPosition`` goal that
    targets an entity centroid/position.

    The oracle-derived entity position lives on a 1-unit pixel
    lattice but the agent moves in coarser motor steps.  An exact
    positional match is therefore generally unreachable; the
    engine needs a "close enough" radius.  Precedence:

    1. If the entity has a bounding box, use half the smaller bbox
       side — reaching anywhere inside the entity's footprint
       qualifies as "at the entity".  (The smaller side keeps the
       radius conservative when the bbox is elongated.)
    2. Else, use half the max motion-step magnitude across
       committed :class:`MotionModelClaim`\\s — reaching the
       nearest lattice point qualifies.
    3. Else, ``0.0`` (exact match, old behaviour).

    Robotics analogue: the same helper on a manipulator returns
    max(gripper_opening, end_effector_position_accuracy) / 2 — a
    radius that accounts for both the target object's graspable
    extent and the actuator's achievable precision.
    """
    ent = ws.entities.get(entity_id)
    if ent is not None:
        bbox = ent.properties.get("bbox")
        if bbox is not None and len(bbox) == 4:
            try:
                x0, y0, x1, y1 = (float(v) for v in bbox)
                half_w = abs(x1 - x0) / 2.0
                half_h = abs(y1 - y0) / 2.0
                bbox_tol = float(min(half_w, half_h))
            except (TypeError, ValueError):
                bbox_tol = 0.0
        else:
            bbox_tol = 0.0
    else:
        bbox_tol = 0.0
    # Take the larger of {bbox half-extent, motion half-step} so the
    # tolerance is always at least reachable by a single motor step.
    # A small bbox for a large-step actor would otherwise make the
    # goal unreachable.
    return max(bbox_tol, _motion_step_tolerance(ws))


def _position_for_entity(ws: WorldState, entity_id: str) -> Optional[Tuple[float, float]]:
    """Best-effort position lookup for role-goal synthesis.

    Returns ``None`` if no candidate is usable.  See
    :func:`derive_goals_from_roles` for the precedence order.
    """
    ent = ws.entities.get(entity_id)
    if ent is not None:
        # 1. live position
        pos = ent.properties.get("position")
        if pos is not None:
            try:
                return (float(pos[0]), float(pos[1]))
            except (TypeError, ValueError, IndexError):
                pass
        # 2. centroid (engine convention after Phase 4 segmentation)
        cen = ent.properties.get("centroid")
        if cen is not None:
            try:
                return (float(cen[0]), float(cen[1]))
            except (TypeError, ValueError, IndexError):
                pass
        # 3. centre of bbox [x_min, y_min, x_max, y_max]
        bbox = ent.properties.get("bbox")
        if bbox is not None and len(bbox) == 4:
            try:
                x0, y0, x1, y1 = (float(v) for v in bbox)
                return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
            except (TypeError, ValueError):
                pass

    # 4. fall back to an initial_position PropertyClaim, if one exists
    for h in ws.hypotheses.values():
        c = h.claim
        if not isinstance(c, PropertyClaim):
            continue
        if c.entity_id != entity_id or c.property != "initial_position":
            continue
        try:
            return (float(c.value[0]), float(c.value[1]))
        except (TypeError, ValueError, IndexError):
            continue

    return None


# ---------------------------------------------------------------------------
# Achievement / walk helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Action-probe goals
# ---------------------------------------------------------------------------


_PROBE_GOAL_PREFIX = "probe::"


def derive_action_probe_goals(ws: WorldState, step: int) -> List[str]:
    """Generate / maintain one :class:`Goal` per action in the adapter's
    action space.  Returns the list of newly-created or updated goal
    IDs (priority recomputed, abandoned on cap, etc.).

    Rationale.  Without first-hand knowledge of what each action does,
    downstream planning rests on guesses.  This function walks the
    currently-known action space (read from ``ws.agent['_action_space']``,
    populated by the runner from :meth:`Adapter.action_space`) and
    maintains one probe goal per action.

    Closing condition — continuous commitment.  Each probe goal's
    atom uses :class:`MotionModelCommitted(action_id) <MotionModelCommitted>`
    rather than :class:`ActionTried`.  The probe stays open until the
    miner has accumulated enough evidence that the planner is willing
    to use the motion model (credence ≥ :attr:`PlannerConfig.min_credence`),
    not merely until the button has been pressed once.  See
    ``SPEC_continuous_commitment.md``.

    Priority — uncertainty-weighted.  Each tick the probe's priority
    is recomputed as ``base × (1 − credence(motion_model))``.  A
    just-begun probe at credence 0 sits at full priority; a probe at
    credence 0.4 sits at 60 %; a probe that has fully committed sits
    at 0 and falls out of the selector naturally.  This is the
    information-gain policy from P2 of the spec: pick the probe
    whose next observation reduces uncertainty the most.

    Abandonment — safety rail.  If an action has been executed
    ``max_probe_attempts_per_action`` times without its motion model
    clearing the floor, the probe goal is marked ABANDONED and a
    ``PropertyClaim("_action", <action_id>, "unmappable")`` is
    proposed so downstream components can filter this action out of
    planning.  Priority competition is the primary mechanism; this
    cap catches genuinely unmappable buttons and pathological
    context-conditional effects that piece 1 cannot diagnose.

    Tunables (``EngineConfig.action_probe``):

    * ``enabled``                         — master switch.
    * ``probe_goal_priority``             — base priority.
    * ``max_actions_per_episode``         — cap on total probe goals created.
    * ``max_probe_attempts_per_action``   — per-action abandon cap.

    Domain-agnostic: an ARC game's buttons and a robot's motor
    primitives are treated identically — both are string action ids
    whose kinematic effect must be characterised before planning
    trusts them.
    """
    cfg_probe = _probe_cfg(ws)
    if cfg_probe is None or not cfg_probe.enabled:
        return []

    action_ids = list(ws.agent.get("_action_space") or ())
    if not action_ids:
        return []

    cap   = int(cfg_probe.max_actions_per_episode)
    base  = float(cfg_probe.probe_goal_priority)
    attempts_cap = int(getattr(cfg_probe, "max_probe_attempts_per_action", 15))
    attempts: Dict[str, int] = ws.agent.get("_action_attempts") or {}
    floor = _probe_credence_floor(ws)

    # Small attempts-based tiebreak.  When two probes sit at equal
    # credence (typically both at 0 early in the episode) the primary
    # priority term cannot choose between them; without a tiebreak
    # the goal selector's stable sort always picks the same action
    # and the other actions never get tried.  An epsilon reduction
    # per attempt cycles the selector through distinct actions
    # before returning to any one.  Epsilon is much smaller than
    # ``base``, so it never inverts a genuine uncertainty ordering.
    tiebreak = 0.01

    touched: List[str] = []
    for aid in action_ids:
        goal_id = f"{_PROBE_GOAL_PREFIX}{aid}"
        cred = _motion_model_credence(ws, str(aid))
        n_attempts = int(attempts.get(str(aid), 0))
        # Scale priority by remaining uncertainty.  The multiplicative
        # form means the probe naturally loses ground as credence rises.
        eff_priority = base * max(0.0, 1.0 - cred) - tiebreak * n_attempts
        existing = ws.goal_forest.goals.get(goal_id)

        if existing is None:
            # Respect the total-probes-in-episode cap before creating.
            existing_count = sum(1 for g in ws.goal_forest.goals
                                 if g.startswith(_PROBE_GOAL_PREFIX))
            if existing_count >= cap:
                continue
            root = GoalNode(
                id         = f"{goal_id}::root",
                node_type  = NodeType.ATOM,
                condition  = MotionModelCommitted(action_id=str(aid)),
                priority   = eff_priority,
                source     = "engine:action_probe",
                created_at = step,
            )
            goal = Goal(
                id         = goal_id,
                root       = root,
                priority   = eff_priority,
                source     = "engine:action_probe",
                created_at = step,
            )
            add_goal(ws, goal)
            touched.append(goal_id)
            existing = goal

        # Recompute priority live for both freshly created and
        # already-existing probes.
        existing.priority = eff_priority
        existing.root.priority = eff_priority

        # Abandonment path: too many attempts and still no credible
        # motion model → give up and mark the action unmappable.
        if (existing.root.status not in
                (GoalStatus.ACHIEVED, GoalStatus.ABANDONED, GoalStatus.PRUNED)
                and cred < floor
                and n_attempts >= attempts_cap):
            mark_status(ws, goal_id, GoalStatus.ABANDONED)
            _emit_unmappable_action_claim(ws, str(aid), step)
            if goal_id not in touched:
                touched.append(goal_id)
    return touched


def _motion_model_credence(ws: WorldState, action_id: str) -> float:
    """Return the highest credence currently carried by any
    :class:`MotionModelClaim` for ``action_id``.  ``0.0`` if none."""
    from .claims import MotionModelClaim
    best = 0.0
    for h in ws.hypotheses.values():
        claim = h.claim
        if not isinstance(claim, MotionModelClaim):
            continue
        if str(claim.action_id) != action_id:
            continue
        pt = float(getattr(h.credence, "point", 0.0))
        if pt > best:
            best = pt
    return best


def _probe_credence_floor(ws: WorldState) -> float:
    """Planner credence floor read through the engine config, falling
    back to the :class:`PlannerConfig` default.  Shared by the
    probe-closing condition and the abandonment check so both sides
    agree on what "credible enough" means."""
    cfg = getattr(ws, "config", None)
    planner_cfg = getattr(cfg, "planner", None) if cfg else None
    if planner_cfg is not None:
        return float(getattr(planner_cfg, "min_credence", 0.5))
    from .config import PlannerConfig
    return float(PlannerConfig().min_credence)


def _emit_unmappable_action_claim(ws: WorldState,
                                  action_id: str,
                                  step: int) -> None:
    """Propose ``PropertyClaim("_action", action_id, "unmappable")``.

    Idempotent via the store's canonical-key dedup path.  The entity
    id ``"_action"`` mirrors the ``"_game"`` convention for engine-
    level characterisations.
    """
    from .claims import PropertyClaim
    from .types import Scope, ScopeKind
    _store.propose(
        ws,
        claim     = PropertyClaim(
            entity_id = "_action",
            property  = str(action_id),
            value     = "unmappable",
        ),
        source    = "engine:action_probe",
        scope     = Scope(kind=ScopeKind.EPISODE),
        step      = step,
        rationale = (f"action {action_id} exceeded probe attempt cap "
                     f"without a credible motion model"),
    )


def _probe_cfg(ws: WorldState):
    """Fetch ``ws.config.action_probe`` or ``None``.  Kept private so
    the probe subsystem has exactly one central tunable-lookup site."""
    cfg = getattr(ws, "config", None)
    if cfg is None:
        return None
    return getattr(cfg, "action_probe", None)


def atomic_leaves(node: GoalNode) -> Iterator[GoalNode]:
    """Yield every ATOM descendant of the given node.

    Used by the planner to determine what concrete sub-conditions the
    current goal reduces to, and by the explorer to find
    exploration-worthy leaves (ones with open status and no plan).
    """
    if node.node_type == NodeType.ATOM:
        yield node
        return
    for child in node.children:
        yield from atomic_leaves(child)


def goals_with_tag(ws: WorldState, tag: str) -> List[Goal]:
    """Return every top-level goal whose ``tags`` set contains ``tag``.

    Axis 2 of ``SPEC_goal_classification.md``.  The canonical query
    surface for tag-based lookup, replacing string-prefix matches on
    goal ids (``gid.startswith("resource_refuel_")`` becomes
    ``goals_with_tag(ws, "refuel")``).

    Linear scan over ``goal_forest.goals``.  Forest size is bounded
    by trial duration (typically tens of goals); a scan is fast
    enough that an index would be premature optimisation.  Add a
    per-tag index field on ``GoalForest`` later if profiling shows
    the scan dominating selector time.
    """
    return [
        g for g in ws.goal_forest.goals.values()
        if tag in g.tags
    ]


def is_achieved(ws: WorldState, goal_id: str) -> bool:
    """Check whether a goal's tree is satisfied by the current
    :class:`WorldState`.

    Recursively:

    * ATOM: its condition evaluates to ``True``.
    * AND : all children are achieved.
    * OR  : any child is achieved; ``active_branch`` is recorded.
    * CHANCE: treated as achieved if the selected outcome branch is
      achieved (best effort without runtime randomness).
    * OPTION / MAINTAIN / LOOP / ADVERSARIAL / INFO_SET: not yet
      implemented — treated as not-achieved (planner/runner handles
      them separately).

    Unknown truth values (``condition.evaluate`` returns ``None``)
    are treated as not-achieved — we require positive confirmation.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return False
    return _node_achieved(goal.root, ws)


def _node_achieved(node: GoalNode, ws: WorldState) -> bool:
    if node.node_type == NodeType.ATOM:
        if node.condition is None:
            return False
        return node.condition.evaluate(ws) is True
    if node.node_type == NodeType.AND:
        return all(_node_achieved(c, ws) for c in node.children)
    if node.node_type == NodeType.OR:
        # Prefer active_branch if set
        if node.active_branch is not None:
            for c in node.children:
                if c.id == node.active_branch:
                    return _node_achieved(c, ws)
        return any(_node_achieved(c, ws) for c in node.children)
    if node.node_type == NodeType.CHANCE:
        # Optimistic: if any branch achieves the goal, treat as achieved
        return any(_node_achieved(c, ws) for c in node.children)
    # Reserved types not yet handled
    return False


def reset_progress_for_cleared_hypotheses(
    ws:                       WorldState,
    cleared_hypothesis_ids:   "Iterable[str]",
    *,
    new_status:               GoalStatus = GoalStatus.OPEN,
) -> List[str]:
    """Reset goal progress when supporting hypotheses have been retracted.

    For every goal whose tree contains a node whose
    ``supporting_hypothesis_ids`` intersects ``cleared_hypothesis_ids``,
    reset that goal's root status to ``new_status`` (default ``OPEN``).
    Returns the list of reset goal IDs for telemetry.

    This is the goal-forest companion to
    :func:`hypothesis_store.clear_by_scope`.  When a scope boundary is
    crossed (e.g. respawn → ``LIFE`` claims clear), any goal whose
    progress was justified by those claims should drop back to OPEN
    so the runner re-evaluates whether the goal is achievable under
    the new world state.

    Emits a :class:`GoalStatusChanged` telemetry event per reset so
    trace consumers can show the cycle: a goal that went OPEN →
    ACHIEVED earlier in the session, then ACHIEVED → OPEN at respawn.
    Without the event, the trace would show the goal silently
    "un-achieving" itself, which is harder to diagnose.

    No-op for goals whose status is already ``new_status`` — the
    function is idempotent and safe to call when nothing relevant
    was cleared.
    """
    cleared_set = set(cleared_hypothesis_ids)
    if not cleared_set:
        return []
    reset: List[str] = []
    for goal_id, goal in list(ws.goal_forest.goals.items()):
        if _node_depends_on_any(goal.root, cleared_set):
            old_status = goal.root.status
            if old_status == new_status:
                continue
            goal.root.status = new_status
            reset.append(goal_id)
            _emit(ws, _tel.GoalStatusChanged(
                goal_id    = goal_id,
                old_status = old_status.value if hasattr(old_status, "value") else str(old_status),
                new_status = new_status.value if hasattr(new_status, "value") else str(new_status),
            ), subject=goal_id)
    return reset


def _node_depends_on_any(node: GoalNode, cleared_set: set) -> bool:
    """True if this node OR any descendant lists any of the cleared
    hypothesis IDs in its ``supporting_hypothesis_ids``."""
    if cleared_set & set(node.supporting_hypothesis_ids or ()):
        return True
    for child in node.children or ():
        if _node_depends_on_any(child, cleared_set):
            return True
    return False


def reset_status_by_atom_kind(
    ws:           WorldState,
    kinds:        "Iterable[str]",
    *,
    new_status:   GoalStatus = GoalStatus.OPEN,
) -> List[str]:
    """Reset the root status of every goal whose tree contains an
    ATOM node whose ``Condition.canonical_key()[0]`` matches one of
    the given kinds.

    Companion to :func:`reset_progress_for_cleared_hypotheses` for
    the case where a goal's achievement is justified by *world
    state* (e.g. ``TriggerVisitedAtLeast`` reading per-life visit
    counts) rather than by hypothesis-store claims.  When that
    world state resets at a scope boundary (e.g. a respawn clears
    ``visit_counts``), goals whose ATOM kinds match are no longer
    truth-preservingly ACHIEVED and should re-open.

    Returns the list of reset goal IDs.  Idempotent — goals whose
    status already equals ``new_status`` are skipped silently.
    Emits :class:`GoalStatusChanged` per reset (same as the
    hypothesis-driven path) so the trace shows the cycle.

    Domain-specific kinds (e.g. ``TriggerVisitedAtLeast``,
    ``AgentAtCell``) live in adapters, not the engine — this helper
    takes them as a string set so the engine doesn't need to
    enumerate them itself.
    """
    target_kinds = set(kinds or ())
    if not target_kinds:
        return []
    reset: List[str] = []
    for goal_id, goal in list(ws.goal_forest.goals.items()):
        if not _node_has_atom_kind(goal.root, target_kinds):
            continue
        old_status = goal.root.status
        if old_status == new_status:
            continue
        goal.root.status = new_status
        reset.append(goal_id)
        _emit(ws, _tel.GoalStatusChanged(
            goal_id    = goal_id,
            old_status = old_status.value if hasattr(old_status, "value") else str(old_status),
            new_status = new_status.value if hasattr(new_status, "value") else str(new_status),
        ), subject=goal_id)
    return reset


def _node_has_atom_kind(node: GoalNode, kinds: set) -> bool:
    """True if this node OR any descendant is an ATOM whose
    condition's canonical-key kind is in ``kinds``."""
    if node.node_type == NodeType.ATOM and node.condition is not None:
        try:
            ck = node.condition.canonical_key()
            if ck and len(ck) > 0 and str(ck[0]) in kinds:
                return True
        except Exception:
            pass
    for child in node.children or ():
        if _node_has_atom_kind(child, kinds):
            return True
    return False


def mark_status(ws: WorldState, goal_id: str, status: GoalStatus) -> None:
    """Set the status on the root and propagate achievement downward
    when the new status is ACHIEVED."""
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return
    old_status = goal.root.status
    goal.root.status = status
    if status == GoalStatus.ACHIEVED:
        _cascade_achieved(goal.root)
    if old_status != status:
        _emit(ws, _tel.GoalStatusChanged(
            goal_id    = goal_id,
            old_status = old_status.value if hasattr(old_status, "value") else str(old_status),
            new_status = status.value if hasattr(status, "value") else str(status),
        ), subject=goal_id)


def _cascade_achieved(node: GoalNode) -> None:
    node.status = GoalStatus.ACHIEVED
    for child in node.children:
        _cascade_achieved(child)


def refresh_status(ws: WorldState, goal_id: str) -> GoalStatus:
    """Re-evaluate a goal's root status against the current
    WorldState.  Returns the updated status.

    Called by the runner each step to detect achievement or to
    surface newly-open subgoals as children complete.

    Terminal statuses (ABANDONED, PRUNED) are preserved.  Without
    this check, a goal explicitly retracted by an adapter (e.g.
    one-shot ``interact_*`` goals abandoned after first achievement,
    or refuel goals abandoned after ``[refuel-consumed-on-fire]``)
    would silently revert to ACHIEVED whenever the agent re-satisfied
    the underlying condition (e.g. revisits the cell).  An explicit
    retraction must stick until the adapter re-declares the goal.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return GoalStatus.ABANDONED
    if goal.root.status in (GoalStatus.ABANDONED, GoalStatus.PRUNED):
        return goal.root.status
    if is_achieved(ws, goal_id):
        mark_status(ws, goal_id, GoalStatus.ACHIEVED)
    return goal.root.status


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------


def detect_conflicts(ws: WorldState, step: int) -> List[GoalConflict]:
    """Scan active goals for conflicts and update
    ``ws.goal_forest.conflicts`` in place.  Returns the conflict list.

    Phase 3 implements:

    * **MUTEX** — two goals whose ATOM-leaf conditions are logical
      negations (``C`` and ``Negation(C)``) appear as leaves of
      simultaneously-active goals.
    * **TEMPORAL** — two goals with deadlines tighter than the sum
      of their minimum achievement time would require (estimated
      heuristically by leaf count).
    * **RESOURCE** — structural pattern noted (both goals reference
      the same resource in ATOM conditions); resolution policy
      defaults to PRIORITY.
    * **ADVERSARIAL** — two goals from different principals where
      one principal's ``context`` explicitly excludes the other's.
      Detected structurally; resolution delegated to the principal-
      authority arbitration that arrives with the robotics rule
      system (Phase 5).
    """
    active = [g for g in ws.goal_forest.goals.values()
              if g.root.status in (GoalStatus.OPEN, GoalStatus.ACTIVE)]
    conflicts: List[GoalConflict] = []

    for i, ga in enumerate(active):
        for gb in active[i + 1:]:
            c = _pair_conflict(ga, gb, step)
            if c is not None:
                conflicts.append(c)

    # Emit only newly-detected conflicts, not repeats of existing
    # ones — the client renders the set by id.
    prior_keys = {(c.goal_a, c.goal_b, c.conflict_type)
                  for c in ws.goal_forest.conflicts}
    for c in conflicts:
        key = (c.goal_a, c.goal_b, c.conflict_type)
        if key in prior_keys:
            continue
        kind = c.conflict_type.value if hasattr(c.conflict_type, "value") \
               else str(c.conflict_type)
        _emit(ws, _tel.ConflictDetected(
            conflict_type = kind,
            goal_ids      = [c.goal_a, c.goal_b],
        ), step=step)

    ws.goal_forest.conflicts = conflicts
    return conflicts


def _pair_conflict(ga: Goal, gb: Goal, step: int) -> Optional[GoalConflict]:
    """Return a :class:`GoalConflict` describing why two goals
    conflict, or ``None`` if they don't."""
    leaves_a = list(atomic_leaves(ga.root))
    leaves_b = list(atomic_leaves(gb.root))

    # MUTEX — any pair of leaves that are logical negations
    for la in leaves_a:
        if la.condition is None:
            continue
        for lb in leaves_b:
            if lb.condition is None:
                continue
            if _is_negation_of(la.condition, lb.condition):
                return GoalConflict(
                    goal_a            = ga.id,
                    goal_b            = gb.id,
                    conflict_type     = ConflictType.MUTEX,
                    resolution_policy = ResolutionPolicy.PRIORITY,
                    detected_at       = step,
                    rationale         = (f"ATOM leaves {la.id} and {lb.id} "
                                         f"have logically incompatible conditions"),
                )

    # TEMPORAL — overlapping deadlines with cumulative leaf count too high
    if ga.deadline is not None and gb.deadline is not None:
        leaf_total = len(leaves_a) + len(leaves_b)
        earliest = min(ga.deadline, gb.deadline)
        if earliest - step < leaf_total:  # rough heuristic: 1 step per leaf
            return GoalConflict(
                goal_a            = ga.id,
                goal_b            = gb.id,
                conflict_type     = ConflictType.TEMPORAL,
                resolution_policy = ResolutionPolicy.PRIORITY,
                detected_at       = step,
                rationale         = (f"combined leaf count ({leaf_total}) exceeds "
                                     f"steps to earliest deadline ({earliest - step})"),
            )

    # RESOURCE — shared resource reference in atoms
    res_a = _resource_refs(leaves_a)
    res_b = _resource_refs(leaves_b)
    shared = res_a & res_b
    if shared:
        return GoalConflict(
            goal_a            = ga.id,
            goal_b            = gb.id,
            conflict_type     = ConflictType.RESOURCE,
            resolution_policy = ResolutionPolicy.PRIORITY,
            detected_at       = step,
            rationale         = f"shared resources referenced: {sorted(shared)}",
        )

    # ADVERSARIAL — distinct principals with excluding contexts
    if (ga.principal is not None and gb.principal is not None
            and ga.principal.id != gb.principal.id):
        if (ga.principal.context is not None
                and _is_negation_of(ga.principal.context, gb.principal.context
                                    if gb.principal.context is not None
                                    else ga.principal.context)):
            return GoalConflict(
                goal_a            = ga.id,
                goal_b            = gb.id,
                conflict_type     = ConflictType.ADVERSARIAL,
                resolution_policy = ResolutionPolicy.USER_ARBITRATE,
                detected_at       = step,
                rationale         = (f"principals {ga.principal.id} and {gb.principal.id} "
                                     f"have mutually excluding contexts"),
            )

    return None


def _is_negation_of(a: Condition, b: Condition) -> bool:
    """True if ``a == Negation(b)`` or ``b == Negation(a)`` at the
    canonical-key level."""
    from .conditions import Negation
    if isinstance(a, Negation) and a.condition.canonical_key() == b.canonical_key():
        return True
    if isinstance(b, Negation) and b.condition.canonical_key() == a.canonical_key():
        return True
    return False


def _resource_refs(leaves: List[GoalNode]) -> set:
    """Collect resource IDs referenced by ResourceAbove/ResourceBelow
    conditions within the given ATOM leaves."""
    from .conditions import ResourceAbove, ResourceBelow
    out = set()
    for leaf in leaves:
        cond = leaf.condition
        if isinstance(cond, (ResourceAbove, ResourceBelow)):
            out.add(cond.resource_id)
    return out
