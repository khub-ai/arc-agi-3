"""Resource-constrained route planner.

Domain-agnostic primitive for "given a current resource budget, a
target, and known refuel sites, plan a route that won't strand the
agent."  Used by ARC-AGI-3 (action-step budget + budget-pickup cells)
but the contract is intentionally generic — positions are opaque
hashables, distances come from a caller-supplied function, budgets
are integers.  Any domain with a discrete-step resource that
replenishes at known locations can wire this in: a robot's battery
with charging stations, a vehicle's fuel with gas stations, a
multi-leg logistics route with depots.

Three-state output (no continuous "pressure" score):

* :class:`RouteStatus.OK`              — direct route fits within
  current budget AND post-target lookahead confirms a refuel is
  still reachable from the target.  Caller proceeds.
* :class:`RouteStatus.MUST_REFUEL`    — direct route doesn't fit OR
  taking it would strand the agent post-target.  Returns the chosen
  refuel position; caller should route there before resuming.
* :class:`RouteStatus.NO_VIABLE`      — no refuel insertion makes
  the target reachable.  Caller's responsibility: surface as a
  stranded event for postmortem replanning, optionally explore for
  undiscovered refuels.

Refuel choice (when more than one fits): pick the refuel that
maximises the agent's RESIDUAL budget at the target.  Among refuels
feasible for this leg (``d_to ≤ current_budget`` AND
``d_on ≤ full_budget``), the residual at target equals
``full_budget − d_on``, so minimising ``d_on`` maximises the
residual.  Scoring is ``(d_on, d_to)`` lexicographic; ``d_to`` only
breaks ties (tighter agent→refuel leg is the safer choice when the
onward leg is identical).

Refuel-as-late-as-possible (``DURABLE_PRINCIPLES.md`` P9 tier 1,
``SPEC_resource_aware_selector.md`` § "Refuel-as-late-as-possible"):
the two legs do not share a budget — steps before refuel come out
of ``current_budget``, steps after come out of ``full_budget``.  The
only thing the next primary sees is the post-arrival residual.  Once
the refuel is reached, the prior leg is sunk cost; only the onward
leg shapes the next move's choice set.  The feasibility filter
collapses the rule cleanly under tight budget: when only the
nearest refuel passes ``d_to ≤ current_budget``, the candidate set
is one and the agent takes it regardless of ``d_on``.  No
special-casing needed.

This replaces an earlier (2026-05-05) rule that scored
``(d_to, d_on)`` — nearest-to-agent first.  That codification was
derived from a single tight-budget incident, generalised to all
situations, and surfaced the wrong choice in comfortable-budget
cases.  Concretely (ls20 3/7 turn 4, trial_postfix_*):
refuelling at ``(-3, 2)`` (``d_on=5``) instead of ``(-6, 5)``
(``d_on=7``) on the way to rotation trigger ``(0, 4)`` saves 2
env.steps of residual per cycle.  The two rules agree under tight
budget; the principle change is the comfortable-budget choice.

Lookahead is parameterised.  Depth 1 checks only direct feasibility;
depth 2 (default) also verifies post-target refuel reachability;
larger depths recurse into post-refuel post-target chains.  Larger
horizons are more conservative (recommend refuels earlier) but
cost more BFS distance queries.

Prime Directive review: no magic numbers, no game-specific
assumptions.  Strictly observational + structural.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Optional


Position = Any                                                 # opaque hashable
DistanceFn = Callable[[Position, Position], "Optional[int]"]


class RouteStatus(Enum):
    OK          = "ok"
    MUST_REFUEL = "must_refuel"
    NO_VIABLE   = "no_viable"


@dataclass(frozen=True)
class RoutePlan:
    """Result of :func:`plan_resource_route`.

    ``status`` is the actionable verdict; the other fields are
    diagnostic context the caller can log / surface to a postmortem
    record.  All distances are integer step counts, never None on
    a populated field (None means "not computed for this branch").
    """
    status:                RouteStatus
    refuel:                "Position | None"  = None
    direct_cost:           "int | None"       = None
    refuel_cost:           "int | None"       = None   # agent → refuel
    onward_cost:           "int | None"       = None   # refuel → target
    reason:                str                = ""
    candidates_evaluated:  int                = 0


def plan_resource_route(
    *,
    agent_pos:           Position,
    target:              Position,
    refuel_positions:    Iterable[Position],
    current_budget:      int,
    full_budget:         int,
    distance_fn:         DistanceFn,
    lookahead:           int = 1,
) -> RoutePlan:
    """Decide whether ``agent_pos`` can reach ``target`` within
    ``current_budget`` steps, with optional refuel insertion.

    Args:
      agent_pos:        Where the agent is now.
      target:           Where the agent wants to go.
      refuel_positions: Known refuel sites (any iterable; consumed once).
      current_budget:   Steps remaining on the current resource life.
      full_budget:      Steps after a refuel (max budget).
      distance_fn:      ``(a, b) -> Optional[int]``.  Should return
                        ``None`` when ``b`` is unreachable from ``a``.
      lookahead:        How many subsequent legs to verify.  ``1`` = only
                        the direct leg; ``2`` = direct leg + post-target
                        refuel reachability (default; matches ls20-class
                        games where any single leg fits in full_budget).
                        Must be ≥ 1.

    Returns:
      A :class:`RoutePlan` whose ``status`` tells the caller what to do.

    Refuel scoring:
      Among refuels whose ``agent → r`` fits in ``current_budget`` AND
      whose ``r → target`` fits in ``full_budget``, pick the smallest
      ``r → target`` (the refuel closest to the target).  Maximises
      residual budget at target = ``full_budget − r→target``.  Ties
      broken by smaller ``agent → r`` (closer refuel wins under tight
      budget; no doubling back when ``d_on`` is identical).  See
      ``SPEC_resource_aware_selector.md`` § "Refuel-as-late-as-possible".
    """
    if lookahead < 1:
        raise ValueError("lookahead must be >= 1")
    if current_budget < 0 or full_budget < 0:
        raise ValueError("budgets must be non-negative")

    refuels = list(refuel_positions or ())

    # ── Direct feasibility ───────────────────────────────────────────
    direct = distance_fn(agent_pos, target)
    direct_ok = (direct is not None and int(direct) <= current_budget)

    # Lookahead-1: don't stranding-check post-target.
    if direct_ok and lookahead < 2:
        return RoutePlan(
            status      = RouteStatus.OK,
            direct_cost = int(direct),
            reason      = "direct route fits (lookahead=1)",
        )

    # Lookahead ≥ 2: verify post-target reachability of SOME refuel.
    if direct_ok:
        residual = current_budget - int(direct)
        post_strand = _post_target_strands(
            target          = target,
            refuels         = refuels,
            residual_budget = residual,
            full_budget     = full_budget,
            distance_fn     = distance_fn,
            depth_remaining = lookahead - 2,
        )
        if not post_strand:
            return RoutePlan(
                status      = RouteStatus.OK,
                direct_cost = int(direct),
                reason      = ("direct route fits; post-target refuel "
                               "reachable within residual"),
            )
        # else: direct fits but would strand.  Insert a refuel.

    # ── Refuel insertion ─────────────────────────────────────────────
    best: "tuple[int, int, Position, int, int] | None" = None
    # tuple = (d_on, d_to, refuel, d_to, d_on) — first two axes are
    # the lex score (d_on then d_to, smaller wins); duplicated d_to / d_on
    # at indices 3/4 keep the diagnostic shape stable for the unpack below.
    evaluated = 0
    for r in refuels:
        if r == agent_pos:
            # Already at this refuel — cost 0 to "reach" it.  After
            # refuel, full_budget governs the onward leg.
            d_to = 0
        else:
            d_to_raw = distance_fn(agent_pos, r)
            if d_to_raw is None or int(d_to_raw) > current_budget:
                continue
            d_to = int(d_to_raw)
        if r == target:
            # Refuel IS the target — onward cost is 0.
            d_on = 0
        else:
            d_on_raw = distance_fn(r, target)
            if d_on_raw is None or int(d_on_raw) > full_budget:
                continue
            d_on = int(d_on_raw)
        evaluated += 1
        # Score: refuel closest to TARGET first (maximises residual
        # budget at target = full_budget − d_on); closer-to-agent as
        # tiebreak.  Module docstring + SPEC_resource_aware_selector.md
        # § "Refuel-as-late-as-possible" carry the rationale.
        score = (d_on, d_to)
        if best is None or score < (best[0], best[1]):
            best = (d_on, d_to, r, d_to, d_on)

    if best is not None:
        # best = (score_d_on, score_d_to, refuel, d_to, d_on); only the
        # last three are needed downstream.
        _, _, chosen, d_to, d_on = best
        return RoutePlan(
            status                = RouteStatus.MUST_REFUEL,
            refuel                = chosen,
            refuel_cost           = d_to,
            onward_cost           = d_on,
            direct_cost           = int(direct) if direct is not None else None,
            reason                = ("direct infeasible" if not direct_ok
                                     else "post-target stranding avoidance"),
            candidates_evaluated  = evaluated,
        )

    # ── Nothing works ────────────────────────────────────────────────
    if direct is None:
        reason = "target unreachable on known graph (no refuel can help)"
    elif not refuels:
        if direct_ok:
            # direct fits, but lookahead-2 post-target check failed
            # AND we have no refuels to insert.
            reason = (f"direct cost {int(direct)} <= budget "
                      f"{current_budget}, but post-target lookahead "
                      f"strands (no known refuels to insert)")
        else:
            reason = (f"direct cost {int(direct)} > budget "
                      f"{current_budget}; no known refuels")
    else:
        if direct_ok:
            # Direct fits, post-target strands, AND no refuel insertion
            # avoids the strand (refuels not reachable from agent OR
            # from refuel to target, OR refuel→target exceeds full
            # budget).
            reason = (f"direct cost {int(direct)} <= budget "
                      f"{current_budget}, but post-target stranding; "
                      f"no refuel insertion fits (current={current_budget}, "
                      f"full={full_budget}, refuels considered="
                      f"{len(refuels)})")
        else:
            reason = (f"direct cost {int(direct)} > budget "
                      f"{current_budget}; no refuel allows "
                      f"agent→r→target within budgets "
                      f"(current={current_budget}, full={full_budget}, "
                      f"refuels considered={len(refuels)})")
    return RoutePlan(
        status                = RouteStatus.NO_VIABLE,
        direct_cost           = int(direct) if direct is not None else None,
        reason                = reason,
        candidates_evaluated  = evaluated,
    )


class TargetVerdict(Enum):
    """Verdict returned by :func:`validate_target`.

    * ``SAFE`` — committing to ``candidate_target`` is survivable:
      either ``candidate_target`` is itself a refuel, or some
      refuel is reachable from ``candidate_target`` within the
      residual budget.
    * ``WOULD_STRAND`` — the agent can reach ``candidate_target``
      but not refuel afterwards.  ``recommended_refuel`` (on the
      :class:`TargetValidation` payload) holds the refuel the
      caller should divert to first.  Equivalent to
      :class:`RouteStatus.MUST_REFUEL` but framed as "validate this
      target before committing".
    * ``UNREACHABLE`` — the agent cannot reach ``candidate_target``
      with the current budget at all.  No refuel insertion was
      considered (use :func:`plan_resource_route` for that).
    """
    SAFE         = "safe"
    WOULD_STRAND = "would_strand"
    UNREACHABLE  = "unreachable"


@dataclass(frozen=True)
class TargetValidation:
    """Result of :func:`validate_target`."""
    verdict:             TargetVerdict
    direct_cost:         "int | None"  = None   # agent → target
    residual_at_target:  "int | None"  = None   # current_budget − direct_cost
    nearest_refuel_from_target: "Position | None" = None
    nearest_refuel_dist_from_target: "int | None" = None
    recommended_refuel:  "Position | None" = None   # set on WOULD_STRAND
    reason:              str            = ""


def validate_target(
    *,
    agent_pos:           Position,
    candidate_target:    Position,
    refuel_positions:    Iterable[Position],
    current_budget:      int,
    full_budget:         int,
    distance_fn:         DistanceFn,
) -> TargetValidation:
    """Decide whether committing to ``candidate_target`` is survivable.

    The "foresee exhaustion" primitive: the operator's framing
    (2026-05-05) "as soon as ``0,4`` is selected, it is then clear
    that going to ``0,4`` without refueling first would cause
    budget exhaustion."  The harness calls this when accepting an
    LLM-suggested target (or any candidate not derived from the
    survival-aware route planner) to catch the certain-death case
    at decision time, before the agent commits.

    Verdicts and what the caller should do:
      * :class:`TargetVerdict.SAFE` — proceed.
      * :class:`TargetVerdict.WOULD_STRAND` — divert to the recommended
        refuel first; revisit ``candidate_target`` afterwards.
      * :class:`TargetVerdict.UNREACHABLE` — caller should drop the
        candidate (or pursue exploration / a different goal).

    Note: ``WOULD_STRAND`` does NOT recurse into "from the recommended
    refuel, can the candidate target itself be reached after refilling".
    Use :func:`plan_resource_route` when a multi-leg route plan is
    needed; ``validate_target`` is the lighter "is this single
    candidate worth committing to right now" check.
    """
    if current_budget < 0 or full_budget < 0:
        raise ValueError("budgets must be non-negative")

    refuels = list(refuel_positions or ())

    # Reachability + direct cost.
    direct = distance_fn(agent_pos, candidate_target)
    if direct is None:
        return TargetValidation(
            verdict     = TargetVerdict.UNREACHABLE,
            reason      = "target unreachable on known graph",
        )
    direct_int = int(direct)
    if direct_int > current_budget:
        return TargetValidation(
            verdict     = TargetVerdict.UNREACHABLE,
            direct_cost = direct_int,
            reason      = (f"direct cost {direct_int} > current budget "
                           f"{current_budget}"),
        )

    residual = current_budget - direct_int

    # Target is itself a refuel: arrival auto-refills.
    if any(r == candidate_target for r in refuels):
        return TargetValidation(
            verdict             = TargetVerdict.SAFE,
            direct_cost         = direct_int,
            residual_at_target  = residual,
            reason              = "candidate target is a refuel — auto-refill on arrival",
        )

    # Find nearest refuel reachable from the TARGET (post-arrival).
    closest_d:  "int | None"        = None
    closest_r:  "Position | None"   = None
    for r in refuels:
        d = distance_fn(candidate_target, r)
        if d is None:
            continue
        di = int(d)
        if closest_d is None or di < closest_d:
            closest_d = di
            closest_r = r

    if closest_d is None:
        # No refuel reachable from target on known graph.  Two
        # sub-cases:
        #
        # (a) Cold-start (refuels list is empty entirely): we have no
        #     map of refuel sites yet.  The agent is at ``agent_pos``
        #     having survived to here, so retreating to ``agent_pos``
        #     costs ``direct_int`` steps.  Round-trip safety requires
        #     ``2 * direct_int <= current_budget``; otherwise committing
        #     forfeits the retreat option and strands the agent if no
        #     refuel materialises along the way.  Flag as WOULD_STRAND
        #     with ``recommended_refuel=None`` so the caller knows
        #     there's nothing to divert to — pick a shorter target.
        #     Operator framing 2026-05-07 (trial_gate_persist L2 turn
        #     3): cold-start commit to [0,4] burned 10 of 16 budget,
        #     stranding the agent before any refuel was discovered.
        #
        # (b) Refuels exist but none reachable from ``candidate_target``
        #     on the known graph: refueling now wouldn't change this
        #     (the strand at the target is structural, not budget-
        #     driven).  Analogous to the "no waste" branch in
        #     :func:`_post_target_strands`.  Treat as SAFE.
        if not refuels and 2 * direct_int > current_budget:
            return TargetValidation(
                verdict             = TargetVerdict.WOULD_STRAND,
                direct_cost         = direct_int,
                residual_at_target  = residual,
                recommended_refuel  = None,
                reason              = (
                    f"cold-start round-trip: no refuels known and "
                    f"2*direct ({2 * direct_int}) > current_budget "
                    f"({current_budget}); commitment forfeits retreat"
                ),
            )
        return TargetValidation(
            verdict             = TargetVerdict.SAFE,
            direct_cost         = direct_int,
            residual_at_target  = residual,
            reason              = ("no refuel reachable from target on known "
                                   "graph; refueling first wouldn't help"),
        )

    if closest_d <= residual:
        return TargetValidation(
            verdict                          = TargetVerdict.SAFE,
            direct_cost                      = direct_int,
            residual_at_target               = residual,
            nearest_refuel_from_target       = closest_r,
            nearest_refuel_dist_from_target  = closest_d,
            reason                           = ("refuel reachable from target "
                                                "within residual"),
        )

    # Parsimony rule: residual > 0 means the agent has at least one
    # step of mobility after arrival.  Refuels are a finite resource
    # — consuming one preemptively (because some hypothetical *next*
    # trip might need budget) trades a real resource against a
    # speculative one.  Operator framing 2026-05-06: "system got to
    # understand the concept of parsimony, and how to budget the
    # budget."  Treat post-target stranding only when residual <= 0
    # (truly stuck after arrival, can't take any next step at all).
    # Otherwise return SAFE; if budget pressure escalates next turn
    # the budget_pressure pipeline will declare a refuel goal at
    # priority 1.5 with current observations, not speculative ones.
    if residual > 0:
        return TargetValidation(
            verdict                          = TargetVerdict.SAFE,
            direct_cost                      = direct_int,
            residual_at_target               = residual,
            nearest_refuel_from_target       = closest_r,
            nearest_refuel_dist_from_target  = closest_d,
            reason                           = (f"parsimonious: residual {residual} "
                                                f"> 0; defer refuel-or-not to "
                                                f"next-turn observation rather "
                                                f"than burn a refuel speculatively"),
        )

    # Would strand: pick a refuel for the agent to divert to FIRST.
    # Use the same nearest-first scoring as plan_resource_route so
    # the diverted-to-refuel matches the route planner's choice.
    recommended:  "Position | None" = None
    rec_score:    "tuple[int, int] | None" = None
    for r in refuels:
        d_to_raw = distance_fn(agent_pos, r)
        if d_to_raw is None or int(d_to_raw) > current_budget:
            continue
        d_to = int(d_to_raw)
        if r == candidate_target:
            d_on = 0
        else:
            d_on_raw = distance_fn(r, candidate_target)
            if d_on_raw is None or int(d_on_raw) > full_budget:
                continue
            d_on = int(d_on_raw)
        sc = (d_to, d_on)
        if rec_score is None or sc < rec_score:
            rec_score = sc
            recommended = r

    return TargetValidation(
        verdict                          = TargetVerdict.WOULD_STRAND,
        direct_cost                      = direct_int,
        residual_at_target               = residual,
        nearest_refuel_from_target       = closest_r,
        nearest_refuel_dist_from_target  = closest_d,
        recommended_refuel               = recommended,
        reason                           = (f"residual at target ({residual}) "
                                            f"< nearest-refuel-from-target "
                                            f"({closest_d}); divert to refuel "
                                            f"first"),
    )


def survival_budget(
    *,
    agent_pos:           Position,
    refuel_positions:    Iterable[Position],
    current_budget:      int,
    distance_fn:         DistanceFn,
) -> int:
    """Number of "free" steps the agent can take before survival
    requires committing to a refuel.

    Returned value: ``current_budget − dist_to_nearest_reachable_refuel``,
    floored at 0.  When no refuel is reachable within the current
    budget, returns 0 — the agent is already in a state where any
    additional step risks stranding (use :func:`plan_resource_route`
    or :func:`validate_target` for a richer verdict).

    Use cases:
      * Per-step abort guard: stop committing to a path if
        survival_budget after this step would be negative.
      * EXPLORE_TRIGGER / repeated-visit guards: ``survival_budget`` ÷
        ``cost_per_visit`` bounds the number of safe iterations.
      * Telemetry / planner hints: how much "slack" the agent has to
        spend on non-refuel work this turn.
    """
    if current_budget < 0:
        raise ValueError("current_budget must be non-negative")
    refuels = list(refuel_positions or ())
    if not refuels:
        return 0
    if agent_pos in refuels:
        return current_budget
    nearest = None
    for r in refuels:
        d = distance_fn(agent_pos, r)
        if d is None:
            continue
        di = int(d)
        if di > current_budget:
            continue
        if nearest is None or di < nearest:
            nearest = di
    if nearest is None:
        return 0
    return max(0, current_budget - nearest)


def _post_target_strands(
    *,
    target:           Position,
    refuels:          list,
    residual_budget:  int,
    full_budget:      int,
    distance_fn:      DistanceFn,
    depth_remaining:  int,
) -> bool:
    """Return True iff, having arrived at ``target`` with
    ``residual_budget`` steps left, the agent will strand AND a
    refuel insertion BEFORE the target could fix it.

    The second clause — "refuel insertion could fix it" — is the
    "no waste" principle (operator 2026-05-05): we only count a
    situation as stranded for purposes of refuel-now decision when
    refueling now actually changes the survival outcome.

    Concretely, this returns False (no relevant strand) when:
    * No refuels are known at all — refuel insertion is impossible.
    * Standing on a refuel at the target — auto-refill.
    * SOME refuel reachable from target within residual — comfortable.
    * NO refuel reachable from target on the known graph — refueling
      now puts the agent at a refuel with full budget, but post-
      refuel-then-target the structural property is the same
      (target has no reachable refuel).  The agent strands EITHER
      WAY; refuel-now is wasted effort.  The user's framing: "if
      there's no future refuel anyway, just take the target now."

    Returns True only when refueling now would convert a stranded
    leg into a survivable one.

    Recursion (depth_remaining > 0): for the chosen post-target
    refuel, verify it has its own onward refuel reachable.  Bottom
    out at depth 0.
    """
    if not refuels:
        return False   # refuel insertion impossible → not a refuel-now decision
    closest_reachable: "int | None" = None
    closest_refuel: "Position | None" = None
    for r in refuels:
        if r == target:
            return False   # standing on a refuel — refill on arrival
        d = distance_fn(target, r)
        if d is None:
            continue
        di = int(d)
        if closest_reachable is None or di < closest_reachable:
            closest_reachable = di
            closest_refuel = r
    if closest_reachable is None:
        # No refuel reachable from target on the known graph.
        # Refueling now does not change this — post-refuel state
        # would have the same structural property at the target.
        # The user's "no waste" principle: don't recommend a refuel
        # that doesn't change the survival outcome.
        return False
    if closest_reachable <= residual_budget:
        # Comfortable: SOME refuel reachable within residual.
        # Optional deeper recursion: verify the post-refuel position
        # has its own onward refuel.
        if depth_remaining <= 0:
            return False
        deeper_strand = _post_target_strands(
            target          = closest_refuel,
            refuels         = [x for x in refuels if x != closest_refuel],
            residual_budget = full_budget,
            full_budget     = full_budget,
            distance_fn     = distance_fn,
            depth_remaining = depth_remaining - 1,
        )
        return deeper_strand
    # Closest refuel is reachable but residual is insufficient.
    # Refueling now extends the residual at target — genuine strand.
    return True
