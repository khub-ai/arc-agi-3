"""Goal plan search — engine-level multi-step lookahead.

See ``docs/SPEC_goal_plan_search.md`` for the design doc.

This module composes with ``cognitive_os.resource_aware_selector``:
the selector ranks ONE goal at a time at horizon ≤ 2; this module
ranks complete sequences of goals over the win-condition primary
set, inserting service-goal legs (refuels, pickups) when the
simulated budget would otherwise go negative.

The output is a **plan stack** — a ranked list of feasible plans
that persists across lives within a trial.  When a life ends
without making progress on the win condition, the adapter pops
the failed plan and the next life automatically tries the
next-best alternative.  The stack itself is the agent's "don't
blindly repeat the same plan" memory; no separate learning
module is needed at the sequencing layer.

Engine/adapter separation: the engine never reads cells, joints,
or pixel coordinates.  The adapter:

  * Precomputes the primary-set IDs (via the depends_on closure
    under the terminal goal, walker-aware children traversal,
    cell_target() filtering — all adapter-side mechanics).
  * Supplies a ``LegSimulator`` that, given an opaque
    ``SimState`` and a primary/service ID, returns the leg's
    cost / restores / post-state / side-effects / success
    probability.
  * Owns the ``SimState`` shape — engine treats it as opaque.

This module is intentionally short (~300 lines).  The full
contract lives in the spec; this is the algorithmic core.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import (
    Any, FrozenSet, Iterable, List, Mapping, Optional, Protocol,
    Sequence, Set, Tuple,
)

from .resource_aware_selector import (
    GoalId, ResourceKey, ResourceProfile, ResourceState,
)


# ===========================================================================
# Types
# ===========================================================================


@dataclass(frozen=True)
class SideEffects:
    """Set-delta a leg applies to the simulated achieved-goal set.

    ``add`` are goals the leg achieves.  ``remove`` are goals the
    leg un-achieves (toggle / cycle semantics — e.g. an ls20
    trigger whose alternating visits flip an alignment dimension).
    Both default to empty; a degenerate leg with neither is a
    no-op for goal tracking.

    The engine applies ``remove`` BEFORE ``add`` when computing
    the post-leg ``achieved_goals`` set, so a leg listing the
    same goal_id in both nets to "achieved."  This rule lets
    adapters emit a redundant pair when a CausalClaim model is
    uncertain about polarity.

    See ``docs/SPEC_goal_plan_search.md`` "Cyclic side-effects via
    CausalClaim reading" for the full rationale and the adapter-
    side rule for choosing between ``add`` and ``remove``.
    """
    add:     FrozenSet[GoalId] = field(default_factory=frozenset)
    remove:  FrozenSet[GoalId] = field(default_factory=frozenset)

    @classmethod
    def achieve(cls, *goal_ids: GoalId) -> "SideEffects":
        """Convenience for the common positive-only case."""
        return cls(add=frozenset(goal_ids))

    @classmethod
    def coerce(cls, value: Any) -> "SideEffects":
        """Coerce a value to ``SideEffects``: ``SideEffects`` instances
        pass through; sets / frozensets / iterables are interpreted
        as positive-only achievements (placed in ``add``).

        Used by :class:`LegSimulation`'s ``__post_init__`` so adapter
        call sites that previously emitted ``frozenset({goal_id})``
        continue to work without modification while the migration to
        the typed surface lands.
        """
        if isinstance(value, SideEffects):
            return value
        return cls(add=frozenset(value))


@dataclass(frozen=True)
class LegSimulation:
    """Adapter-computed cost / benefit / effect of one leg.

    ``cost`` and ``restores`` follow the same convention as
    :class:`ResourceProfile`: non-negative integer counts per
    :class:`ResourceKey`; resources not relevant to the leg are
    absent from the map.

    ``post_state`` is the adapter's opaque ``SimState`` after the
    leg completes — passed back into ``simulate_leg`` for the
    next leg.  The engine never reads its fields.

    ``side_effects`` is a :class:`SideEffects` delta on the
    simulated achieved-goal set — see that class's docstring for
    semantics.  Iterables passed here are coerced to
    ``SideEffects(add=frozenset(value))`` for compatibility with
    pre-typed-surface call sites.

    ``success_prob`` is the adapter's confidence the leg works as
    modelled (typically derived from CausalClaim credences).  The
    engine multiplies these across legs to compute plan risk.
    """
    cost:           Mapping[ResourceKey, int]
    restores:       Mapping[ResourceKey, int]
    post_state:     Any                       # opaque SimState
    side_effects:   SideEffects
    success_prob:   float                     # in [0, 1]

    def __post_init__(self) -> None:
        if not isinstance(self.side_effects, SideEffects):
            object.__setattr__(self, "side_effects",
                               SideEffects.coerce(self.side_effects))


class LegSimulator(Protocol):
    """Adapter-implemented leg simulator.

    The engine calls this to estimate the cost and effect of
    pursuing one goal from a simulated state.  ``state`` is the
    adapter's own opaque ``SimState`` type — the engine treats
    it as transparent data and only forwards it.

    Returning ``None`` means the leg isn't possible from this
    state (no path, missing precondition, target unreachable,
    goal already achieved by side effects).
    """
    def simulate_leg(
        self,
        state:    Any,
        goal_id:  GoalId,
    ) -> "Optional[LegSimulation]":
        ...


@dataclass(frozen=True)
class Plan:
    """An ordered sequence of legs that, if executed, achieves
    the win-condition primary set.

    ``feasible`` is True iff every leg's cost is satisfiable from
    leg-start resources (with restores from prior legs) AND the
    union of side_effects across legs covers ``required_goal_ids``.

    ``expected_total_cost`` is the cost summed across legs (gross).
    The engine uses it for tie-breaking when feasibility and risk
    match.
    """
    legs:                 Tuple[LegSimulation, ...]
    goal_sequence:        Tuple[GoalId, ...]
    feasible:             bool
    expected_total_cost:  Mapping[ResourceKey, int]
    risk:                 float                  # 1 - product(success_prob)
    plan_id:              str                    # stable id for stack bookkeeping


@dataclass(frozen=True)
class Constraint:
    """A leg-level veto pushed by a tactical guard, persisted on the
    plan stack so re-enumeration excludes the constrained leg until
    the constraint clears.

    The engine treats ``kind`` and ``params`` opaquely: it stores
    them, lists them on demand, and removes them when the adapter
    calls one of the clear functions.  Canonical kinds are listed
    in ``docs/SPEC_goal_plan_search.md`` "Constraint clear-
    conditions" (would_strand, regression, cost_exceeds_budget,
    per_step_regression, oracle_rejected, unreachable).  Adapters
    may add domain-specific kinds without engine changes.

    ``params`` is an adapter-defined mapping the engine doesn't
    interpret but preserves — typical contents: the resource
    threshold the verdict was computed against, the dim id, the
    BFS distance estimate at constraint-add time.  Used by the
    adapter to evaluate clear conditions and by telemetry.
    """
    leg_id:    GoalId
    kind:      str
    reason:    str
    params:    Mapping[str, Any] = field(default_factory=dict)
    added_at:  int = 0


@dataclass
class PlanStack:
    """Ranked list of candidate plans, top = best, plus active
    constraints carried across re-enumeration cycles.

    Mutable: pop on plan failure, invalidate when world state
    changes enough to require re-search.  The stack itself is the
    agent's "approaches not yet tried" memory — popping the top on
    death-without-progress means the next life automatically tries
    the next-best alternative.

    ``constraints`` are veto records added by tactical guards.  They
    persist across re-enumerations within a world_version; the
    adapter calls :func:`clear_constraints_by_kind`,
    :func:`clear_constraints_for_leg`, or :func:`clear_all_constraints`
    when its conditions are met.  See spec's "Constraint clear-
    conditions" table for the canonical mapping.
    """
    plans:           List[Plan]                  = field(default_factory=list)
    generated_at:    int                         = 0
    failed_plan_ids: Set[str]                    = field(default_factory=set)
    constraints:     List[Constraint]            = field(default_factory=list)


# ===========================================================================
# Internal helpers
# ===========================================================================


def _add_resources(
    state:   Mapping[ResourceKey, int],
    delta:   Mapping[ResourceKey, int],
    sign:    int,
) -> "dict[ResourceKey, int]":
    """Apply a per-resource delta with the given sign, returning a
    new dict.  Resources absent from one map are treated as zero."""
    out = dict(state)
    for k, v in delta.items():
        out[k] = out.get(k, 0) + sign * int(v)
    return out


def _can_afford(
    cur_resources:   Mapping[ResourceKey, int],
    cost:            Mapping[ResourceKey, int],
) -> bool:
    """Check that every cost entry is satisfiable from current
    resource levels.  Resources absent from cur_resources are
    zero (cannot afford a non-zero cost on a missing resource)."""
    for k, v in cost.items():
        if cur_resources.get(k, 0) < int(v):
            return False
    return True


def _clamp_to_full(
    state:   Mapping[ResourceKey, int],
    full:    Mapping[ResourceKey, int],
) -> "dict[ResourceKey, int]":
    """Clamp resource levels at their declared full capacity.
    Restores cannot push a resource above its cap."""
    out = dict(state)
    for k in out:
        if k in full:
            out[k] = min(int(out[k]), int(full[k]))
    return out


def _initial_resource_dict(
    resources: Sequence[ResourceState],
) -> "Tuple[dict[ResourceKey, int], dict[ResourceKey, int]]":
    """Convert ResourceState sequence to (current_dict, full_dict)."""
    current = {r.key: int(r.current) for r in resources}
    full    = {r.key: int(r.full)    for r in resources}
    return current, full


def _apply_side_effects(
    achieved:  "set[GoalId]",
    se:        SideEffects,
) -> None:
    """Mutate ``achieved`` per spec: ``remove`` BEFORE ``add``.

    The order matters when the same goal_id appears in both deltas
    (degenerate but legal): the result is "achieved," matching the
    spec's "leg listing the same goal_id in both nets to achieved."
    """
    if se.remove:
        achieved -= se.remove
    if se.add:
        achieved |= se.add


def _post_leg_strands(
    *,
    post_state:        Any,
    post_resources:    Mapping[ResourceKey, int],
    services:          Sequence[GoalId],
    simulator:         LegSimulator,
) -> bool:
    """Domain-agnostic post-leg survival check.

    Returns True iff NO service is reachable + affordable from
    ``post_state`` at ``post_resources``.  This is the engine-level
    generalization of the "residual_at_target >= dist(target,
    nearest_refuel)" rule from
    :func:`cognitive_os.resource_planner.validate_target`: after a
    primary leg, the agent must still be able to reach SOME service
    (refuel-class goal) to survive a future low-resource state.  If
    not, the plan is post-strand and the caller inserts a service
    leg before the primary.

    Note on used_services: the survival check considers ALL
    services, not just unused ones.  Plan-time ``used_services`` is
    a planning convenience (each service inserted at most once per
    plan to avoid plan-time loops); survival in execution is
    independent — the agent can always revisit a refuel pad mid-
    trip if it has the budget.  One-shot semantics (a pickup that
    truly disappears) are encoded by the simulator returning None
    from a post-consumption ``simulate_leg`` call.

    Returns False (= "doesn't strand") when ``services`` is empty —
    no services means survival isn't gated on resource recovery
    (the caller treats this case as "no post-leg constraint").
    """
    if not services:
        return False
    for svc_id in services:
        svc_sim = simulator.simulate_leg(post_state, svc_id)
        if svc_sim is None:
            continue
        if _can_afford(post_resources, svc_sim.cost):
            return False
    return True


def _simulate_ordering(
    *,
    ordering:         Sequence[GoalId],
    services:         Sequence[GoalId],
    simulator:        LegSimulator,
    initial_state:    Any,
    initial_current:  Mapping[ResourceKey, int],
    full:             Mapping[ResourceKey, int],
) -> "Optional[Plan]":
    """Walk one ordering of primaries left-to-right, inserting a
    service leg before any primary whose cost exceeds available
    resources.  Returns None if no service rescues the gap; returns
    a Plan (possibly with feasible=False) if simulation completed
    but side-effects don't cover required goals.

    Service-leg insertion strategy: try each candidate service in
    order; pick the first that (a) is reachable from the current
    state, (b) restores enough to make the next primary affordable,
    (c) and is itself affordable from current resources.  Each
    service is used at most once per plan to avoid cycles.

    The resulting Plan's ``required_goal_ids`` (for feasibility) is
    the input ``ordering`` set; achievement is checked against the
    union of all legs' side_effects.
    """
    state = initial_state
    current = dict(initial_current)
    legs:           "list[LegSimulation]" = []
    goal_sequence:  "list[GoalId]"        = []
    used_services:  "set[GoalId]"         = set()
    achieved:       "set[GoalId]"         = set()
    ordering_list = list(ordering)

    for _idx, primary_id in enumerate(ordering_list):
        if primary_id in achieved:
            # Already covered by a prior leg's side-effects — skip.
            continue

        # Try the primary directly first.
        sim = simulator.simulate_leg(state, primary_id)
        if sim is None:
            return None
        # Tentatively compute post-leg resources for survival check.
        _tentative_post = _add_resources(current, sim.cost, -1)
        _tentative_post = _add_resources(_tentative_post, sim.restores, +1)
        _tentative_post = _clamp_to_full(_tentative_post, full)
        # Two reasons to insert a service leg before the primary:
        #   (a) cost-affordability: primary's cost > current resources
        #   (b) post-leg strand: after the primary, no service is
        #       reachable + affordable from post_state.  Skip (b)
        #       when this primary IS itself a service (its own
        #       restores cover survival) AND when this is the final
        #       leg in the ordering (achievement at the terminal
        #       position is the win condition; survival after isn't
        #       gated).  Per docs/SPEC_resource_aware_selector.md
        #       and the operator's 2026-05-07 framing: "if residual
        #       budget after reaching target is less than dist
        #       (target, nearest_refuel), the trip should be taken
        #       only if refuel first."
        _is_self_service = primary_id in services
        _is_last = (_idx == len(ordering_list) - 1)
        _strands_post = (
            (not _is_self_service)
            and (not _is_last)
            and _post_leg_strands(
                post_state     = sim.post_state,
                post_resources = _tentative_post,
                services       = services,
                simulator      = simulator,
            )
        )
        if (not _can_afford(current, sim.cost)) or _strands_post:
            # Try inserting a service leg.
            rescued = False
            for svc_id in services:
                if svc_id in used_services:
                    continue
                svc_sim = simulator.simulate_leg(state, svc_id)
                if svc_sim is None:
                    continue
                if not _can_afford(current, svc_sim.cost):
                    continue
                # Apply service: post_state advances, resources
                # update with cost - restore - clamp.
                tentative_current = _add_resources(current, svc_sim.cost, -1)
                tentative_current = _add_resources(tentative_current, svc_sim.restores, +1)
                tentative_current = _clamp_to_full(tentative_current, full)
                # Re-simulate the primary from the post-service state
                # to get accurate cost (state may have moved).
                resim = simulator.simulate_leg(svc_sim.post_state, primary_id)
                if resim is None:
                    continue
                if not _can_afford(tentative_current, resim.cost):
                    continue
                # Re-check post-strand from the resim's post-state
                # — service insertion only helps if it RESOLVES the
                # post-strand, otherwise we've burned a service for
                # nothing.
                _resim_post_resources = _add_resources(tentative_current, resim.cost, -1)
                _resim_post_resources = _add_resources(_resim_post_resources, resim.restores, +1)
                _resim_post_resources = _clamp_to_full(_resim_post_resources, full)
                _resim_strands = (
                    (not _is_self_service)
                    and (not _is_last)
                    and _post_leg_strands(
                        post_state     = resim.post_state,
                        post_resources = _resim_post_resources,
                        services       = services,
                        simulator      = simulator,
                    )
                )
                if _resim_strands:
                    continue
                # Service helped — commit it to the plan.
                legs.append(svc_sim)
                goal_sequence.append(svc_id)
                _apply_side_effects(achieved, svc_sim.side_effects)
                used_services.add(svc_id)
                current = tentative_current
                state   = svc_sim.post_state
                sim     = resim
                rescued = True
                break
            if not rescued:
                return None  # primary unaffordable / post-strands; no service rescues

        # Apply the primary.
        legs.append(sim)
        goal_sequence.append(primary_id)
        _apply_side_effects(achieved, sim.side_effects)
        current = _add_resources(current, sim.cost, -1)
        current = _add_resources(current, sim.restores, +1)
        current = _clamp_to_full(current, full)
        state = sim.post_state

    # Feasibility = every primary achieved (directly or via side-effects).
    feasible = all(p in achieved for p in ordering)

    # Aggregate cost / risk.
    total_cost: "dict[ResourceKey, int]" = {}
    risk_factor = 1.0
    for leg in legs:
        for k, v in leg.cost.items():
            total_cost[k] = total_cost.get(k, 0) + int(v)
        risk_factor *= max(0.0, min(1.0, float(leg.success_prob)))
    risk = 1.0 - risk_factor

    plan_id = "|".join(goal_sequence)
    return Plan(
        legs                = tuple(legs),
        goal_sequence       = tuple(goal_sequence),
        feasible            = feasible,
        expected_total_cost = total_cost,
        risk                = risk,
        plan_id             = plan_id,
    )


# ===========================================================================
# Public API
# ===========================================================================


def enumerate_plans(
    *,
    primaries:         Sequence[GoalId],
    services:          Sequence[GoalId],
    simulator:         LegSimulator,
    initial_state:     Any,
    initial_resources: Sequence[ResourceState],
    generated_at:      int = 0,
    max_plans:         int = 20,
    excluded:          Optional[Iterable[GoalId]] = None,
) -> PlanStack:
    """Enumerate orderings of ``primaries``, simulate each, return a
    ranked plan stack.

    Service goals are inserted between primaries when budget would
    otherwise go negative.  Each service is used at most once per
    plan.

    ``excluded`` is the set of leg goal_ids the planner must NOT
    propose this turn — typically the constrained-leg set derived
    via :func:`constrained_legs` from the active stack.  Both
    primaries and services are filtered.  Without ``excluded``, the
    planner ignores constraints entirely.

    Ranking:  feasible plans before infeasible; lower risk before
    higher; lower expected_total_cost as tie-breaker.

    For ``len(primaries) ≤ 6`` exhaustive permutation enumeration
    runs in <720 trials — well within budget.  Larger primary sets
    fall back to first-N orderings; if real workloads ever surface
    that case, swap in beam search here without touching the
    public API.
    """
    initial_current, full = _initial_resource_dict(initial_resources)

    excluded_set: "set[GoalId]" = (
        {GoalId(str(g)) for g in excluded} if excluded else set()
    )

    # Distinct primaries only (dedupe on equality), filter excluded.
    seen: "set[GoalId]" = set()
    unique_primaries: "list[GoalId]" = []
    for p in primaries:
        if p in excluded_set:
            continue
        if p not in seen:
            seen.add(p)
            unique_primaries.append(p)
    services = [s for s in services if s not in excluded_set]

    if len(unique_primaries) <= 6:
        candidate_orderings = list(permutations(unique_primaries))
    else:
        # Truncate: just try the input order and a few rotations.
        # Real beam-search would replace this; flag added below.
        candidate_orderings = [
            tuple(unique_primaries[i:] + unique_primaries[:i])
            for i in range(min(6, len(unique_primaries)))
        ]

    plans: "list[Plan]" = []
    seen_plan_ids: "set[str]" = set()
    for ordering in candidate_orderings:
        plan = _simulate_ordering(
            ordering        = ordering,
            services        = services,
            simulator       = simulator,
            initial_state   = initial_state,
            initial_current = initial_current,
            full            = full,
        )
        if plan is None:
            continue
        if plan.plan_id in seen_plan_ids:
            continue
        seen_plan_ids.add(plan.plan_id)
        plans.append(plan)

    plans.sort(key=lambda p: (
        not p.feasible,                         # feasible first
        p.risk,                                 # low risk first
        sum(p.expected_total_cost.values()),    # low cost as tie-breaker
    ))
    plans = plans[:max_plans]

    return PlanStack(
        plans           = plans,
        generated_at    = int(generated_at),
        failed_plan_ids = set(),
    )


def pop_failed_plan(
    stack:    PlanStack,
    plan_id:  str,
    *,
    reason:   str = "",
) -> PlanStack:
    """Remove a plan from the top of the stack after it failed.

    Caller must verify that ``plan_id`` matches the top of the
    stack — defensive against silent reordering.  Mutates ``stack``
    in place and returns it for fluent use.
    """
    if not stack.plans:
        return stack
    if stack.plans[0].plan_id != plan_id:
        # Defensive: fail loudly rather than silently pop the wrong
        # plan.  The adapter is meant to track which plan it's
        # executing and pass that exact id back here.
        raise ValueError(
            f"pop_failed_plan: requested {plan_id!r} but top is "
            f"{stack.plans[0].plan_id!r}; refusing to pop"
        )
    stack.failed_plan_ids.add(plan_id)
    stack.plans.pop(0)
    return stack


def invalidate_stack(
    stack:    PlanStack,
    *,
    reason:   str = "",
) -> PlanStack:
    """Mark the stack as stale so the adapter re-runs
    :func:`enumerate_plans` next turn.  Also clears all active
    constraints — invalidation events (world-version bump, trial /
    level boundary, scope-demotion that affected a top-plan leg)
    are the universal clearer per the spec's "Constraint clear-
    conditions" table.

    Used when the world state changed unexpectedly: a primary was
    achieved out of order, a new goal was declared mid-plan, a
    progress-tracking regression event invalidated a previously-
    achieved side-effect, validity-scope demoted a record the top
    plan depended on.  Mutates in place and returns ``stack``.
    """
    stack.plans = []
    stack.constraints.clear()
    return stack


def top_plan(stack: PlanStack) -> "Optional[Plan]":
    """Return the top-of-stack plan, or ``None`` if empty.

    Convenience for adapters that pull the next leg's goal id from
    ``stack.plans[0].goal_sequence[0]`` each turn.
    """
    return stack.plans[0] if stack.plans else None


# ===========================================================================
# Constraints — tactical-guard veto records carried across re-enumeration
# ===========================================================================
#
# Per ``docs/SPEC_goal_plan_search.md`` "Tactical guards as constraint
# feedback".  The engine treats constraint kinds opaquely; canonical
# kinds and their clear-conditions are documented in the spec.  The
# adapter is responsible for calling clear_* at the appropriate
# moments per the spec table; the engine provides the primitives.


def add_constraint(
    stack:    PlanStack,
    leg_id:   GoalId,
    kind:     str,
    reason:   str,
    *,
    params:   Optional[Mapping[str, Any]] = None,
    added_at: int = 0,
) -> Constraint:
    """Push a leg-veto onto ``stack.constraints``.

    Idempotent on ``(leg_id, kind)``: re-adding the same key
    overwrites ``reason``, ``params``, and ``added_at`` — useful
    when the adapter re-fires a guard with refined parameters.

    Returns the canonical ``Constraint`` record (the one now stored
    on the stack; may be a freshly-constructed one or the prior
    record overwritten in place semantics).
    """
    leg = GoalId(str(leg_id))
    new = Constraint(
        leg_id   = leg,
        kind     = kind,
        reason   = reason,
        params   = dict(params or {}),
        added_at = int(added_at),
    )
    # Drop any prior (leg_id, kind) before appending.
    stack.constraints[:] = [
        c for c in stack.constraints
        if not (c.leg_id == leg and c.kind == kind)
    ]
    stack.constraints.append(new)
    return new


def list_active_constraints(stack: PlanStack) -> List[Constraint]:
    """Snapshot of currently-active constraints.

    Returns a fresh list; mutating the result does not affect the
    stack.  Telemetry and operator diagnostics consume here.
    """
    return list(stack.constraints)


def constrained_legs(stack: PlanStack) -> Set[GoalId]:
    """Set of leg goal_ids currently constrained.

    Pass into :func:`enumerate_plans` as ``excluded`` to skip these
    legs during re-enumeration.
    """
    return {c.leg_id for c in stack.constraints}


def clear_constraints_for_leg(
    stack:   PlanStack,
    leg_id:  GoalId,
) -> List[Constraint]:
    """Remove every constraint on ``leg_id``.

    Returns the cleared records (for telemetry and downstream
    consumers — e.g., a refuel observation may clear constraints
    on multiple legs that depended on that resource).
    """
    leg = GoalId(str(leg_id))
    cleared = [c for c in stack.constraints if c.leg_id == leg]
    stack.constraints[:] = [c for c in stack.constraints if c.leg_id != leg]
    return cleared


def clear_constraints_by_kind(
    stack: PlanStack,
    kind:  str,
) -> List[Constraint]:
    """Remove every constraint of the given ``kind``.

    Returns the cleared records.  Adapters call this when a kind's
    clear condition fires globally — e.g., a budget refill clears
    every ``cost_exceeds_budget`` constraint.
    """
    cleared = [c for c in stack.constraints if c.kind == kind]
    stack.constraints[:] = [c for c in stack.constraints if c.kind != kind]
    return cleared


def clear_all_constraints(
    stack:   PlanStack,
    *,
    reason:  str = "",
) -> List[Constraint]:
    """Remove every constraint regardless of leg or kind.

    Universal clearer.  Per spec, called on world-version bump and
    on trial / level boundaries — both situations where prior
    tactical judgements no longer apply because the underlying
    environment has changed.
    """
    cleared = list(stack.constraints)
    stack.constraints.clear()
    return cleared
