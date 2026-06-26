"""Planner — AO* search over a goal's AND-OR-CHANCE tree.

Given a :class:`Goal` and an action space, produce a :class:`Plan` that
is expected to achieve the goal while respecting active :class:`Rule`\\s
and prior evidence.  The planner is the problem-solving heart of the
engine (capability-audit primary).

Design
------
The goal tree is walked recursively:

* **ATOM**  — BFS over states reachable via committed
  :class:`TransitionClaim`\\s.  Leaf conditions that use
  :class:`AtPosition` benefit most because positions form a natural
  state space; other conditions fall back to a "direct" check (try each
  action and see if the condition becomes true after execution).
* **AND**   — Plan each child in order (``SEQUENTIAL``) or any order
  with simple cost sorting (``UNORDERED``).  The concatenated actions
  become the plan for the AND node.
* **OR**    — Plan each child, choose the cheapest that succeeds.
  Heuristic branch preference from committed :class:`StrategyClaim`\\s
  breaks ties and biases toward empirically-successful branches.
* **CHANCE**— Choose the branch with highest expected value
  (``outcome_prior * success_reward - plan_cost``).  The plan must be
  valid for the chosen branch; the runner is expected to replan if a
  different branch is observed.

Rules
-----
Before search, the action space is filtered:

* ``INVIOLABLE`` rules prohibiting an action → action removed outright.
* ``DEFEASIBLE`` rules → removed unless suppressed by a higher-authority
  rule with positive weight.
* ``ADVISORY`` rules → retained but contribute cost penalties.

Rule filtering is a single pure function that any sub-planner calls; it
does not maintain state.

Options
-------
When the active goal tree contains an ``OPTION``-kind node, the planner
invokes the Option's ``internal_plan`` as a single step with
``pre_condition = option.applicability``.  This is how learned macro-
actions collapse the branching factor of search.

Budget
------
Search is capped by ``PlannerConfig.max_plan_depth`` and
``PlannerConfig.branch_budget``.  Exceeding either returns ``None``,
which the runner takes as "planner exhausted → explore".
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .claims import (
    ActorTransitionClaim,
    MotionModelClaim,
    StrategyClaim,
    TransitionClaim,
)
from .conditions import (
    ActionTried,
    AtPosition,
    Condition,
    MotionModelCommitted,
)
from . import hypothesis_store as _store
from .types import (
    Action,
    Goal,
    GoalNode,
    NodeType,
    Option,
    Ordering,
    Plan,
    PlanStatus,
    PlannedAction,
    Rule,
    RuleConstraint,
    ConstraintKind,
    Violability,
    WorldState,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_plan(ws:            WorldState,
                 goal_id:       str,
                 action_space:  List[Action],
                 *,
                 step:           int = 0,
                 start_state:    Optional[Dict] = None) -> Optional[Plan]:
    """Compute a plan for the named goal.

    Parameters
    ----------
    ws
        WorldState.  Provides hypotheses (TransitionClaims for the
        transition model, StrategyClaims for OR-branch heuristics,
        Rules for filtering), engine config (limits), and current
        agent state (used as BFS start).
    goal_id
        Which goal in ``ws.goal_forest.goals`` to plan for.
    action_space
        Actions the adapter says are currently available.  Filtered
        by Rules before use.
    step
        Current step number — goes on the returned Plan's
        ``computed_at`` field.
    start_state
        Optional override of ``ws.agent`` as the BFS start.  Used
        internally when planning subgoals whose antecedent subgoal
        would change agent state.

    Returns
    -------
    Plan | None
        A plan whose ``steps`` are the ordered actions, whose
        ``assumptions`` are the hypothesis IDs the plan depends on,
        and whose ``branch_selections`` record OR-node choices.
        ``None`` if no plan was found within budget.
    """
    goal = ws.goal_forest.goals.get(goal_id)
    if goal is None:
        return None

    cfg = _planner_cfg(ws)
    filtered_actions = apply_rules_filter(action_space, ws)
    if not filtered_actions:
        return None

    start = start_state if start_state is not None else dict(ws.agent)

    ctx = _PlanCtx(
        ws                = ws,
        action_space      = filtered_actions,
        max_depth         = cfg.max_plan_depth,
        branch_budget     = cfg.branch_budget,
        nodes_expanded    = 0,
    )

    result = _plan_node(goal.root, start, ctx)
    if result is None:
        return None

    steps, assumptions, branch_selections = result
    return Plan(
        goal_id            = goal_id,
        steps              = steps,
        computed_at        = step,
        assumptions        = sorted(set(assumptions)),
        branch_selections  = branch_selections,
        status             = PlanStatus.ACTIVE,
    )


# ---------------------------------------------------------------------------
# Selection + planning (GAP 19: fall through priority if unplannable)
# ---------------------------------------------------------------------------


def select_and_plan(ws:           WorldState,
                    action_space: List[Action],
                    *,
                    step:         int = 0) -> Tuple[Optional[str], Optional[Plan]]:
    """Iterate goal-forest candidates in priority order and return the
    first ``(goal_id, plan)`` pair whose plan is non-empty.

    Pre-GAP-19 the runner called :func:`goal_forest.select_active_goal`
    to pick the single highest-priority eligible goal, then
    :func:`compute_plan` on it.  If that plan was ``None`` the runner
    handed off to the explorer — even when a lower-priority goal
    would have been plannable in the same state.  This was
    catastrophic whenever an adapter seeded a high-priority
    resource-predicate goal (e.g. ``ResourceAbove("episode_won",
    0.5)``) that the BFS cannot reach by motion alone: that goal
    shadowed every concrete sub-goal for the entire episode.

    The fix preserves the priority policy — we still prefer the
    highest-priority plannable goal — but adds a fallback so a
    higher-priority unplannable goal does not block a lower-priority
    plannable one.  :func:`goal_forest.select_active_goal` retains
    its original semantics (priority-only, ignores plannability); the
    runner just no longer uses it for the selection+plan combined
    decision.

    Returns ``(goal_id, plan)`` for the first plannable candidate
    with a non-empty plan; ``(None, None)`` otherwise.  In the success
    case also updates ``ws.goal_forest.active_goal_id`` to the chosen
    goal; in the no-plan case sets it to ``None``.

    An empty-plan goal — condition already satisfied, or all-deferred
    subtree — is *not* selected here: we want an actionable plan so
    the runner produces motion.  The selector's caller can still
    detect already-satisfied goals via the status machinery
    (``refresh_status``).

    Robotics analogue: a mobile manipulator with a top-level "serve
    all customers" metric-goal and a concrete "navigate to kitchen"
    sub-goal cannot plan BFS on the metric, but can plan on the
    kitchen goal.  Priority says the metric wins; plannability says
    only the kitchen goal produces motion.  GAP 19 makes the runner
    prefer the latter when the former has no plan — the metric-goal
    still wins when both are plannable, because it is tried first.
    """
    from . import goal_forest as _gf

    candidates = _gf.candidates_by_priority(ws)
    for gid in candidates:
        plan = compute_plan(ws, gid, action_space, step=step)
        if plan is None or not plan.steps:
            continue
        ws.goal_forest.active_goal_id = gid
        return gid, plan

    ws.goal_forest.active_goal_id = None
    return None, None


# ---------------------------------------------------------------------------
# Rule filter
# ---------------------------------------------------------------------------


def apply_rules_filter(actions: List[Action], ws: WorldState) -> List[Action]:
    """Return ``actions`` with those violating active Rules removed.

    * ``INVIOLABLE`` + ``PROHIBIT`` on an action name → remove.
    * ``DEFEASIBLE`` + ``PROHIBIT`` → remove (Phase 3 has no
      higher-authority override mechanism; robotics Phase 5 adds
      principal arbitration).
    * ``ADVISORY`` → retained; callers query
      :func:`advisory_penalty_for_action` to inflate cost.
    """
    if not ws.rules:
        return list(actions)

    prohibited: Set[str] = set()
    for rule in ws.rules.values():
        if rule.violability in (Violability.INVIOLABLE, Violability.DEFEASIBLE):
            constraint: RuleConstraint = rule.constraint
            if constraint.kind == ConstraintKind.PROHIBIT and isinstance(constraint.target, str):
                # Rule may be conditionally applicable; if the condition
                # is evaluable and false, the rule doesn't apply right now.
                cond_truth = rule.condition.evaluate(ws)
                if cond_truth is False:
                    continue
                prohibited.add(constraint.target)

    return [a for a in actions if a.name not in prohibited]


def advisory_penalty_for_action(action: Action, ws: WorldState) -> float:
    """Sum the cost penalties from all active ``ADVISORY`` rules that
    prefer the action is avoided."""
    penalty = 0.0
    for rule in ws.rules.values():
        if rule.violability != Violability.ADVISORY:
            continue
        constraint = rule.constraint
        if constraint.kind != ConstraintKind.PROHIBIT:
            continue
        if isinstance(constraint.target, str) and constraint.target == action.name:
            cond_truth = rule.condition.evaluate(ws)
            if cond_truth is not False:
                penalty += rule.priority * constraint.weight
    return penalty


# ---------------------------------------------------------------------------
# Internal planning context
# ---------------------------------------------------------------------------


class _PlanCtx:
    """Mutable bag passed through the recursive planner so sibling
    calls share the branch-budget counter without awkward return
    value plumbing."""
    __slots__ = ("ws", "action_space", "max_depth",
                 "branch_budget", "nodes_expanded")

    def __init__(self, ws, action_space, max_depth, branch_budget, nodes_expanded):
        self.ws             = ws
        self.action_space   = action_space
        self.max_depth      = max_depth
        self.branch_budget  = branch_budget
        self.nodes_expanded = nodes_expanded

    def charge(self) -> bool:
        """Debit one node from the budget.  Returns False when
        exhausted — the caller should bail out."""
        self.nodes_expanded += 1
        return self.nodes_expanded <= self.branch_budget


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------


_PlanResult = Tuple[List[PlannedAction], List[str], Dict[str, str]]


def _plan_node(node: GoalNode,
               state: Dict,
               ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Dispatch on node type.  Returns (steps, assumptions,
    branch_selections) or None if no plan could be found."""
    if not ctx.charge():
        return None

    # Deferred-plan nodes are accepted by the planner without producing
    # any actions.  They represent conditions we expect to become true
    # as a side-effect of other planned actions (e.g., the "episode_won"
    # verifier in an AND(trigger, verifier) bundle derived from a
    # CausalClaim).  The runtime achievement check still applies, so if
    # the deferred condition fails to materialise the parent goal will
    # simply not close and the selector will move on.
    if getattr(node, "deferred_plan", False):
        return [], [], {}

    if node.node_type == NodeType.ATOM:
        return _plan_atom(node, state, ctx)
    if node.node_type == NodeType.AND:
        return _plan_and(node, state, ctx)
    if node.node_type == NodeType.OR:
        return _plan_or(node, state, ctx)
    if node.node_type == NodeType.CHANCE:
        return _plan_chance(node, state, ctx)
    if node.node_type == NodeType.OPTION:
        return _plan_option(node, state, ctx)
    # MAINTAIN / LOOP / ADVERSARIAL / INFO_SET — reserved for later
    # phases; reject for now so the caller knows planning failed.
    return None


# ---------------------------------------------------------------------------
# ATOM — BFS over reachable states
# ---------------------------------------------------------------------------


def _plan_atom(node:  GoalNode,
               state: Dict,
               ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Find a sequence of actions that achieves ``node.condition``
    starting from ``state``.

    Uses committed :class:`TransitionClaim`\\s as the transition
    model.  If the condition is already true at ``state``, returns
    an empty plan.  If the condition is a positional predicate, BFS
    over positions using action effects.  For other conditions, try
    each action once and check the effect.

    GAP 13 — honour demotion of supporting hypotheses.  When
    ``derive_subgoals_from_causal`` expands an ATOM leaf into an
    AND(trigger, verifier) or OR of such ANDs, each trigger atom
    carries the source ``CausalClaim``'s hypothesis ID in
    ``supporting_hypothesis_ids``.  If that claim has been demoted
    below commit (because standing within its trigger's tolerance
    never produced the promised effect — the disconfirmation
    pathway in ``_evidence_for_causal``), the branch is
    tested-vacuous: the planner returns ``None`` so the enclosing
    OR picks a different alternative, or so the whole goal falls
    through to ``None`` and the runner hands off to exploration.
    We do NOT mutate the goal tree; the demotion is respected
    transiently via the hypothesis store's credence.  If fresh
    supporting evidence later lifts the claim back over commit,
    the branch becomes plannable again — no rebuild needed.

    Robotics analogue.  A derived-from-CausalClaim reach-and-grasp
    sub-goal stays in the plan tree, but if the "being at the pose
    opens the door" claim has been disconfirmed, the planner
    stops trying to reach that pose as a proxy for opening the
    door and either picks another open-door theory or asks for
    fresh information.
    """
    if node.supporting_hypothesis_ids:
        # An atom with supporting_hypothesis_ids is unplannable when
        # NONE of those hypotheses carry enough credence for the
        # planner.  Three cases collapse to the same semantics:
        #   (a) hypothesis demoted below the planner's credence floor
        #       — contradictions accumulated faster than confirmations;
        #   (b) hypothesis abandoned (pruned from the store) —
        #       credence fell below the abandon threshold, gone;
        #   (c) hypothesis ID never existed in the store — defensive.
        # The gate is :attr:`PlannerConfig.min_credence` rather than
        # the global commit threshold: under continuous commitment
        # (``SPEC_continuous_commitment.md``) the planner engages
        # claims with partial evidence and lets credence keep rising
        # through ordinary dynamics.
        floor = _min_credence(ctx)
        any_credible = any(
            h_id in ctx.ws.hypotheses
            and float(getattr(ctx.ws.hypotheses[h_id].credence, "point", 0.0)) >= floor
            for h_id in node.supporting_hypothesis_ids
        )
        if not any_credible:
            return None

    cond = node.condition
    if cond is None:
        return [], [], {}

    # Short-circuit: already satisfied?
    if _condition_holds(cond, state, ctx.ws):
        return [], [], {}

    # Fast path — :class:`ActionTried` is a self-satisfying goal: the
    # condition becomes true *by executing the named action*, so we
    # don't need a TransitionClaim model.  This is what turns an
    # action-probe goal (priority 1.01, generated by
    # :func:`goal_forest.derive_action_probe_goals`) into a trivial
    # one-step plan, the sole way the engine bootstraps its action
    # model on a brand-new domain.
    if isinstance(cond, ActionTried):
        for action in ctx.action_space:
            if action.name == cond.action_id or action.id == cond.action_id:
                step = PlannedAction(
                    action                = action,
                    expected_effects      = [],
                    depends_on_hypotheses = [],
                    pre_condition         = None,
                )
                return [step], [], {}
        # The named action isn't currently in the (possibly rule-filtered)
        # action space — treat as unplannable.
        return None

    # Fast path — :class:`MotionModelCommitted` is the
    # continuous-commitment successor to ActionTried for probe goals.
    # It does not flip True after a single press; credence must
    # accumulate.  The planner's role is to keep producing one-step
    # plans that execute the named action so the miner gets more
    # observations.  When the condition eventually flips, the goal
    # closes through the normal achievement check and the probe
    # drops out of selection.
    if isinstance(cond, MotionModelCommitted):
        for action in ctx.action_space:
            if action.name == cond.action_id or action.id == cond.action_id:
                step = PlannedAction(
                    action                = action,
                    expected_effects      = [],
                    depends_on_hypotheses = [],
                    pre_condition         = None,
                )
                return [step], [], {}
        return None

    # Gather committed transition model
    transitions = _committed_transitions(ctx)

    # Gather planner-facing motion model: action_id -> (delta, claim_id).
    # Position-independent aggregate of ActorTransitionClaim evidence,
    # minted by :class:`MotionModelMiner`.  This is what lets the planner
    # BFS over a position space it has never visited — critical for the
    # first episode where the per-position ActorTransitionClaim layer has
    # at most one datapoint per (pre, action) and therefore almost nothing
    # commits at that layer.
    motion_model = _committed_motion_model(ctx)

    # Gather wall overrides: (pre_pos, action_id) pairs where an
    # ActorTransitionClaim commits with zero delta.  These locally
    # suppress the motion model — "normally ACTION1 moves up 5; but
    # from this position, it doesn't move at all".  Kept in
    # ActorTransitionClaim layer deliberately (see MotionModelClaim
    # docstring: zero-delta observations are position-specific
    # overrides and must not pollute the open-field motor model).
    #
    # Phase 6c step 2.7: gather max wall credence per (pre, action)
    # so the BFS can compare it to any competing pos_specific
    # (non-zero-delta) claim at the same key.  Stale walls must NOT
    # override fresher successful-move observations of higher
    # credence.  wall_creds is a strict superset of the prior
    # wall_overrides set (same membership; credence value adds the
    # comparison axis).
    wall_creds = _committed_wall_overrides_with_credence(ctx)

    # Gather per-position ActorTransitionClaim deltas (non-zero only;
    # zero-delta is the wall_overrides case above).  When present for
    # a given (cur_pos, action), this delta **overrides** the
    # open-field motion_model aggregate for that cell — learned
    # position-specific dynamics beat the position-blind prior.  Cf.
    # `hypothesis_store.active_position_specific_motion` docstring for
    # the full rationale (GAP 17).
    pos_specific = _store.active_position_specific_motion(ctx.ws)

    if not transitions and not motion_model and not pos_specific:
        # No learned dynamics yet — can't plan positively.
        return None

    # BFS where state is the subset of agent-state that matters for
    # transitions (typically position and key resources).  We hash on a
    # flattened tuple of (position, lives, resources) as a coarse key.
    start_key = _state_key(state)
    frontier  = deque([(state, [], [], start_key)])
    visited:  Set[Tuple] = {start_key}
    assumptions: List[str] = []

    while frontier:
        cur, path, path_assumptions, _ = frontier.popleft()

        if not ctx.charge():
            return None

        if _condition_holds(cond, cur, ctx.ws):
            return path, path_assumptions, {}

        if len(path) >= ctx.max_depth:
            continue

        for action in ctx.action_space:
            # (a) Learned TransitionClaim edges (pre/post specified)
            for tc_id, tc in transitions:
                if tc.action not in (action.name, "*"):
                    continue
                if tc.pre.evaluate_state(cur, ctx.ws) is False:
                    continue
                next_state = _apply_transition(cur, tc)
                next_key = _state_key(next_state)
                if next_key in visited:
                    continue
                visited.add(next_key)
                step = PlannedAction(
                    action                = action,
                    expected_effects      = [tc],
                    depends_on_hypotheses = [tc_id],
                    pre_condition         = tc.pre,
                )
                frontier.append((next_state,
                                 path + [step],
                                 path_assumptions + [tc_id],
                                 next_key))

            # (b) Motion-model edges (synthetic positional transitions).
            # Priority order at each (cur_pos, action):
            #   1. wall override (zero-delta ActorTransitionClaim) —
            #      blocks the edge entirely.
            #   2. position-specific ActorTransitionClaim (non-zero) —
            #      overrides the open-field motion model with the
            #      learned local delta.  GAP 17.
            #   3. open-field MotionModelClaim aggregate — fallback
            #      for cells with no learned local transition yet.
            cur_pos = cur.get("position")
            if cur_pos is None:
                continue
            cur_key = tuple(cur_pos)
            # Wall override vs successful-move credence comparison
            # (Phase 6c step 2.7).  A zero-delta ActorTransitionClaim
            # at (cur_key, action) suggests "this action does nothing
            # here".  But if a competing non-zero-delta claim exists
            # at the same key with HIGHER credence, the agent has
            # demonstrated the action does work — the wall claim is
            # stale and the move wins.  Compare credences before
            # pruning.
            ps = (pos_specific.get((cur_key, action.name))
                  or pos_specific.get((cur_key, action.id)))
            wall_cred = (wall_creds.get((cur_key, action.name))
                         or wall_creds.get((cur_key, action.id)))
            if wall_cred is not None:
                move_cred = 0.0
                if ps is not None:
                    _move_h = ctx.ws.hypotheses.get(ps[1])
                    if _move_h is not None:
                        move_cred = float(getattr(_move_h.credence, "point", 0.0))
                if wall_cred >= move_cred:
                    # Wall wins — prune this edge.
                    continue
                # else: fall through; pos_specific lookup already
                # done above, BFS uses ps's delta below.
            if ps is not None:
                mm_delta, mm_claim_id = ps
            else:
                mm = motion_model.get(action.name) or motion_model.get(action.id)
                if mm is None:
                    continue
                mm_delta, mm_claim_id = mm
            try:
                next_pos = (cur_pos[0] + mm_delta[0],
                            cur_pos[1] + mm_delta[1])
            except (TypeError, IndexError):
                continue
            next_state = dict(cur)
            next_state["position"] = next_pos
            next_key = _state_key(next_state)
            if next_key in visited:
                continue
            visited.add(next_key)
            step = PlannedAction(
                action                = action,
                expected_effects      = [],
                depends_on_hypotheses = [mm_claim_id],
                pre_condition         = None,
            )
            frontier.append((next_state,
                             path + [step],
                             path_assumptions + [mm_claim_id],
                             next_key))
    return None


def _min_credence(ctx: _PlanCtx) -> float:
    """Planner-facing credence floor.

    Reads :attr:`PlannerConfig.min_credence` (default ``0.5``) and
    falls back to ``0.5`` if the config is missing.  Used in place
    of the old ``is_committed`` gate so that hypotheses with partial
    but non-trivial evidence — typically motion-model claims built
    from 3+ confirming observations — are usable for planning well
    before they would reach the global commit threshold.
    """
    cfg = _planner_cfg(ctx.ws)
    return float(getattr(cfg, "min_credence", 0.5))


def _committed_transitions(ctx: _PlanCtx) -> List[Tuple[str, TransitionClaim]]:
    """Collected (hypothesis_id, TransitionClaim) for every
    planner-credible transition-kind hypothesis in the store.

    "Planner-credible" means credence ≥ :attr:`PlannerConfig.min_credence`.
    The legacy name is kept for diff minimality; semantics are now
    "above the planner's credence floor" rather than "strictly
    committed".
    """
    floor = _min_credence(ctx)
    out: List[Tuple[str, TransitionClaim]] = []
    for h in _store.above_credence(ctx.ws, floor):
        if isinstance(h.claim, TransitionClaim):
            out.append((h.id, h.claim))
    return out


def _committed_motion_model(ctx: _PlanCtx) -> Dict[str, Tuple[Tuple, str]]:
    """Planner-facing action -> (delta, hypothesis_id) map from
    :class:`MotionModelClaim`\\s at or above
    :attr:`PlannerConfig.min_credence`.

    When multiple MotionModelClaims clear the floor for the same
    ``action_id`` (stochastic actions with several modal deltas), we
    keep the one with the highest credence.  Ties broken by
    insertion order.  Legacy function name kept for diff minimality;
    semantics are now "above the planner's credence floor".
    """
    floor = _min_credence(ctx)
    best: Dict[str, Tuple[Tuple, str, float]] = {}
    for h in _store.above_credence(ctx.ws, floor):
        if not isinstance(h.claim, MotionModelClaim):
            continue
        cred = float(getattr(h.credence, "point", 0.0))
        aid  = str(h.claim.action_id)
        if aid in best and best[aid][2] >= cred:
            continue
        try:
            delta = (h.claim.delta[0], h.claim.delta[1])
        except (TypeError, IndexError):
            continue
        best[aid] = (delta, h.id, cred)
    return {aid: (v[0], v[1]) for aid, v in best.items()}


def _committed_wall_overrides(ctx: _PlanCtx) -> Set[Tuple[Tuple, str]]:
    """Set of ``(pre_pos, action_id)`` for which a zero-delta
    :class:`ActorTransitionClaim` ("action produced no movement from
    this position") is *active* as a local suppression of the
    open-field motion model during BFS.

    GAP 14 lowered the eligibility threshold from the general
    commit threshold to :attr:`PlannerConfig.wall_override_credence_floor`
    (default ``0.60``).  GAP 15 factored the detection into
    :func:`hypothesis_store.active_wall_overrides` so the explorer
    can share it; this wrapper remains to keep the planner's local
    call signature stable (takes a ``_PlanCtx``).

    The legacy function name is kept for diff minimality; semantics
    are now "currently active" rather than "strictly committed".
    """
    return _store.active_wall_overrides(ctx.ws)


def _committed_wall_overrides_with_credence(
    ctx: _PlanCtx,
) -> "dict[Tuple[Tuple, str], float]":
    """Same as :func:`_committed_wall_overrides` but returns the
    max-credence per ``(pre_pos, action_id)`` instead of just set
    membership.

    Used by the BFS at :func:`_plan_atom` to compare wall credence
    against any competing non-zero-delta pos_specific claim at the
    same ``(pre, action)`` — when a successful-move observation has
    higher credence than a stale wall claim, the move wins and the
    edge is NOT pruned.  This corrects the prior behavior where walls
    unconditionally won, letting stale KB-loaded walls override
    fresh in-session move observations.

    Phase 6c step 2.7, 2026-04-27.
    """
    return _store.active_wall_overrides_with_credence(ctx.ws)


# ---------------------------------------------------------------------------
# AND
# ---------------------------------------------------------------------------


def _plan_and(node:  GoalNode,
              state: Dict,
              ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Plan each child; concatenate step sequences.

    ``SEQUENTIAL`` children are planned in order.  ``UNORDERED``
    children are sorted by a heuristic cost (number of atomic leaves)
    so cheaper-looking children go first and possibly satisfy
    preconditions for others.
    """
    children = list(node.children)
    if node.ordering == Ordering.UNORDERED:
        children.sort(key=lambda c: sum(1 for _ in _atomic_count(c)))

    steps: List[PlannedAction] = []
    assumptions: List[str] = []
    branches: Dict[str, str] = {}
    cur_state = dict(state)

    for child in children:
        sub = _plan_node(child, cur_state, ctx)
        if sub is None:
            return None
        sub_steps, sub_ass, sub_branches = sub
        steps.extend(sub_steps)
        assumptions.extend(sub_ass)
        branches.update(sub_branches)
        cur_state = _simulate_steps(cur_state, sub_steps)

    # Sanity-gate empty-plan ANDs.  If the planner produced zero
    # actions, every child returned an empty sub-plan — either
    # because its condition is already satisfied, or because it is
    # a ``deferred_plan`` node that contributes no actions by
    # design.  The only correct "empty plan" is one where the AND
    # is currently satisfied outright: every leaf condition must
    # evaluate True now.  An empty plan over an AND whose overall
    # conjunction is False means we have a
    # satisfied-trigger-but-effect-did-not-fire situation (the
    # failure mode behind the GAP 12 wedge).  Reporting such an
    # AND as plannable (cost 0) makes OR latch onto the branch
    # and the runner holds an empty plan forever, blocking all
    # further progress.  Return None instead so the OR parent
    # can try another branch and, if all branches are stuck, the
    # runner falls back to exploration while the verifier's
    # condition is re-observed.
    #
    # Robotics analogue.  A manipulation AND(reach_pose,
    # grasp_succeeded) where ``grasp_succeeded`` is a deferred
    # sensor predicate that depends on force feedback: if the
    # arm is already at the pose but the gripper sensor says
    # "no contact", the AND is not done — we should replan (try
    # a slightly different pose, retry grasp) rather than
    # silently report success.
    if not steps and not _node_fully_satisfied(node, ctx.ws):
        return None

    return steps, assumptions, branches


def _node_fully_satisfied(node: "GoalNode", ws: WorldState) -> bool:
    """Return True iff every ATOM-leaf condition in this subtree
    currently evaluates to True in ``ws``.  Deferred-plan nodes
    are NOT exempt — their conditions must also be True.

    Mirrors :func:`goal_forest._node_achieved` but inlined here
    to avoid an import cycle (``goal_forest`` imports from
    ``planner`` transitively via other modules in some paths).
    """
    if node.node_type == NodeType.ATOM:
        if node.condition is None:
            return False
        return node.condition.evaluate(ws) is True
    if node.node_type == NodeType.AND:
        return all(_node_fully_satisfied(c, ws) for c in node.children)
    if node.node_type == NodeType.OR:
        return any(_node_fully_satisfied(c, ws) for c in node.children)
    # CHANCE / OPTION / other composite types: be conservative —
    # report "not fully satisfied" unless ATOM semantics apply.
    return False


# ---------------------------------------------------------------------------
# OR
# ---------------------------------------------------------------------------


def _plan_or(node:  GoalNode,
             state: Dict,
             ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Plan each child; pick cheapest by step count adjusted for
    strategy-claim preference.

    Strategy preference: for each child, consult committed
    :class:`StrategyClaim`\\s whose ``context_pattern`` evaluates true
    in the current state.  A matching StrategyClaim biases the cost
    down by ``(1 - success_rate) * 5`` (so a 0.9-success-rate strategy
    is preferred by half a step over a 0.5-rate one).
    """
    best: Optional[_PlanResult] = None
    best_cost = float("inf")
    chosen_child_id: Optional[str] = None

    for child in node.children:
        sub = _plan_node(child, state, ctx)
        if sub is None:
            continue
        sub_steps, sub_ass, sub_branches = sub
        cost = float(len(sub_steps))
        cost -= _strategy_preference(child, ctx.ws)
        if cost < best_cost:
            best = sub
            best_cost = cost
            chosen_child_id = child.id

    if best is None:
        return None

    steps, assumptions, branches = best
    if chosen_child_id is not None:
        branches[node.id] = chosen_child_id
        node.active_branch = chosen_child_id
    return steps, assumptions, branches


def _strategy_preference(child: GoalNode, ws: WorldState) -> float:
    """Bias term for OR-branch selection from StrategyClaims.
    Returns 0 if no claim applies."""
    best_bias = 0.0
    for h in _store.committed(ws):
        if not isinstance(h.claim, StrategyClaim):
            continue
        ctx_ok = h.claim.context_pattern.evaluate(ws) is True
        if not ctx_ok:
            continue
        # Child is preferred if its id contains the strategy_type string
        if h.claim.strategy_type in child.id:
            bias = h.claim.success_rate * 5.0
            if bias > best_bias:
                best_bias = bias
    return best_bias


# ---------------------------------------------------------------------------
# CHANCE
# ---------------------------------------------------------------------------


def _plan_chance(node:  GoalNode,
                 state: Dict,
                 ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Pick the outcome branch with highest expected value.

    Expected value = ``prior(outcome) / (plan_cost + 1)``.  The ``+1``
    avoids a divide-by-zero for zero-step plans.  The chosen branch
    is recorded in ``branch_selections`` under ``node.id`` so the
    runner can detect deviation and replan.
    """
    best: Optional[_PlanResult] = None
    best_ev = -1.0
    chosen_child_id: Optional[str] = None

    for child in node.children:
        sub = _plan_node(child, state, ctx)
        if sub is None:
            continue
        sub_steps, sub_ass, sub_branches = sub
        prior = node.outcome_priors.get(child.id, 1.0 / max(1, len(node.children)))
        ev = prior / (len(sub_steps) + 1.0)
        if ev > best_ev:
            best = sub
            best_ev = ev
            chosen_child_id = child.id

    if best is None:
        return None
    steps, assumptions, branches = best
    if chosen_child_id is not None:
        branches[node.id] = chosen_child_id
        node.active_branch = chosen_child_id
    return steps, assumptions, branches


# ---------------------------------------------------------------------------
# OPTION
# ---------------------------------------------------------------------------


def _plan_option(node:  GoalNode,
                 state: Dict,
                 ctx:   _PlanCtx) -> Optional[_PlanResult]:
    """Inline an Option's internal plan as a single-step "macro".

    Option nodes carry the Option's ``id`` as a supporting hypothesis
    so the runner can record usage statistics at execution time.  The
    Option's applicability condition is checked against ``state``; if
    not applicable, fall through by returning ``None``.
    """
    # In Phase 3 the OPTION node carries the Option ID in
    # ``supporting_hypothesis_ids[0]``; the ws.options dict provides
    # the Option object.
    if not node.supporting_hypothesis_ids:
        return None
    opt_id = node.supporting_hypothesis_ids[0]
    option: Optional[Option] = ctx.ws.options.get(opt_id)
    if option is None:
        return None
    if option.applicability.evaluate(ctx.ws) is False:
        return None
    # Reuse the pre-recorded plan as the output.  Mark dependence on the
    # Option ID so the runner can update n_uses / success_rate.
    assumption_ids = [opt_id]
    return list(option.internal_plan.steps), assumption_ids, {}


# ---------------------------------------------------------------------------
# Helpers: condition evaluation against an arbitrary state dict
# ---------------------------------------------------------------------------


def _condition_holds(cond:  Condition,
                     state: Dict,
                     ws:    WorldState) -> bool:
    """Evaluate a condition against a possibly-hypothetical state.

    We swap the agent dict into the WorldState temporarily because
    :meth:`Condition.evaluate` reads from ``ws.agent``.  Other entity
    states remain as-is.  This is a conservative simulation — side
    effects on non-agent entities aren't modelled here.
    """
    saved_agent = ws.agent
    ws.agent = state
    try:
        truth = cond.evaluate(ws)
    finally:
        ws.agent = saved_agent
    return truth is True


def _apply_transition(state: Dict, tc: TransitionClaim) -> Dict:
    """Apply a TransitionClaim's ``post`` condition to ``state``,
    returning a new dict.

    Phase 3 MVP handles ``AtPosition`` specifically because grid games
    are the canonical test case; other condition types leave state
    unchanged.  A generalised state-update mechanism (supporting
    arbitrary property mutations) will arrive with the richer
    TransitionClaim schemas in Phase 4.
    """
    new_state = dict(state)
    post = tc.post
    if isinstance(post, AtPosition):
        new_state["position"] = tuple(post.pos)
    return new_state


def _simulate_steps(state: Dict,
                    steps: List[PlannedAction]) -> Dict:
    """Simulate executing steps against ``state`` by chaining
    TransitionClaims' ``post`` conditions.  Used for AND planning so
    later children see realistic intermediate state."""
    cur = dict(state)
    for s in steps:
        for claim in s.expected_effects:
            if isinstance(claim, TransitionClaim):
                cur = _apply_transition(cur, claim)
    return cur


def _state_key(state: Dict) -> Tuple:
    """Hashable key for BFS visited set.  Includes position and
    (if present) lives / resources so the same position with different
    resource counts is treated as a distinct state."""
    pos = tuple(state.get("position", ()))
    lives = state.get("lives")
    resources = tuple(sorted((state.get("resources") or {}).items()))
    return (pos, lives, resources)


def _atomic_count(node: GoalNode):
    """Yield atomic leaves for a rough cost heuristic in AND
    ordering."""
    if node.node_type == NodeType.ATOM:
        yield node
        return
    for c in node.children:
        yield from _atomic_count(c)


# ---------------------------------------------------------------------------
# Monkey patch: evaluate_state on Condition base (for planner sim)
# ---------------------------------------------------------------------------
#
# The planner needs to evaluate a Condition against a hypothetical
# state dict rather than a full WorldState.  Rather than plumbing
# through every Condition subclass, we attach a helper here.  Pure
# additive — does not change Condition's existing semantics.


def _evaluate_state(self, state: Dict, ws: WorldState) -> Optional[bool]:
    """Evaluate this condition with ``state`` swapped in as
    ``ws.agent``.  Restores the original agent state on exit."""
    saved = ws.agent
    ws.agent = state
    try:
        return self.evaluate(ws)
    finally:
        ws.agent = saved


Condition.evaluate_state = _evaluate_state  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _planner_cfg(ws: WorldState):
    if ws.config is not None and hasattr(ws.config, "planner"):
        return ws.config.planner
    from .config import PlannerConfig
    return PlannerConfig()
