"""Goal-progress tracking substrate.

See ``docs/SPEC_goal_progress_tracking.md``.

Per-goal snapshots of which dimensions are currently matched
versus open; events derived from consecutive snapshot diffs;
recovery-priority bumps that the primary goal selector consumes
when a previously-achieved dimension regresses.

Public API
==========

* :func:`compute_snapshot` — produce one snapshot for one goal
  given the current world state.
* :func:`diff_snapshots` — compute the events implied by the
  transition from one snapshot to another.
* :func:`record_progress` — the canonical entry point: compute
  the current snapshot for every goal in the forest, derive
  events from the diff to the previous snapshot, append both to
  the per-goal history, and apply recovery-priority bumps.
* :func:`decay_recovery_bumps` — call once per turn to reduce
  outstanding recovery bumps; the bump fades naturally as
  pursuit continues.
* :func:`effective_recovery_bump` — read the current
  recovery-priority bump for a goal (zero when no recent
  regression).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .claims import GoalRegressionClaim
from .conditions import EntitiesEquivalent
from .types import (
    Goal,
    GoalNode,
    GoalProgressHistory,
    GoalProgressSnapshot,
    NodeType,
    ProgressEvent,
    ProgressEventKind,
    Scope,
    ScopeKind,
    WorldState,
)
from . import hypothesis_store as _store


# Magnitude of the recovery-priority bump applied per regression
# event.  Calibrated to put a regressed alignment goal back above
# the standard pickup band (0.91): an alignment goal at raw 0.85
# plus 0.10 bump = 0.95, comfortably ahead of pickups.
_RECOVERY_BUMP_PER_EVENT: float = 0.10

# Per-turn decay rate applied to outstanding recovery bumps.  A
# bump shrinks geometrically each turn until it reaches zero,
# representing the agent's diminishing urgency about a regression
# as time passes without re-attempt.  0.5 means a single
# regression's bump halves each turn (0.10 → 0.05 → 0.025 → 0.012
# → 0.006 → ~0); roughly five turns of meaningful bias before the
# signal fades.
_RECOVERY_DECAY_PER_TURN: float = 0.5

# Cap on recovery_bump magnitude so repeated regressions cannot
# push the effective priority arbitrarily high.  A goal that
# regresses ten times in a row gets capped at 0.30 — already
# enough to dominate any other goal kind.
_RECOVERY_BUMP_MAX: float = 0.30


# ---------------------------------------------------------------------------
# Snapshot computation
# ---------------------------------------------------------------------------


def _entities_equivalent_for_goal(goal: Goal) -> "Optional[EntitiesEquivalent]":
    """Find the EntitiesEquivalent condition for a goal, whether
    it lives on the root atom or on a verifier child of a
    walker-expanded AND structure.

    Returns ``None`` when the goal isn't an EntitiesEquivalent
    goal — the substrate doesn't compute snapshots for goals it
    doesn't understand.
    """
    root = goal.root
    if root is None:
        return None
    if isinstance(root.condition, EntitiesEquivalent):
        return root.condition
    for child in (root.children or []):
        if isinstance(child.condition, EntitiesEquivalent):
            return child.condition
    return None


def compute_snapshot(goal: Goal,
                     ws:   WorldState,
                     *,
                     turn: int) -> "Optional[GoalProgressSnapshot]":
    """Compute the current progress snapshot for one goal.

    Returns ``None`` when:

    * The goal has no EntitiesEquivalent condition (the substrate
      doesn't understand its progress shape).
    * Either the target or reference entity is missing from
      ``ws.entities`` (snapshot would be uninformative).
    * The condition's dimension set is empty.

    Otherwise: reads the target and reference entities' published
    properties, partitions the goal's dimensions into matched and
    open sets based on equality, and returns a snapshot.

    A dimension counts as matched when both entities have the
    same value for it.  Unknown values (either side missing the
    property) count as open — consistent with
    :class:`EntitiesEquivalent.evaluate`'s tri-valued logic.
    """
    ee = _entities_equivalent_for_goal(goal)
    if ee is None:
        return None
    if not ee.dimensions:
        return None
    target = ws.entities.get(ee.target_id)
    reference = ws.entities.get(ee.reference_id)
    if target is None or reference is None:
        return None
    matched = set()
    open_dims = set()
    for dim in ee.dimensions:
        t_val = target.properties.get(dim)
        r_val = reference.properties.get(dim)
        if t_val is None or r_val is None:
            open_dims.add(str(dim))
        elif t_val == r_val:
            matched.add(str(dim))
        else:
            open_dims.add(str(dim))
    total = max(1, len(ee.dimensions))
    return GoalProgressSnapshot(
        goal_id            = str(goal.id),
        dimensions         = frozenset(str(d) for d in ee.dimensions),
        dimensions_matched = frozenset(matched),
        dimensions_open    = frozenset(open_dims),
        fraction_matched   = len(matched) / total,
        recorded_at_turn   = int(turn),
    )


# ---------------------------------------------------------------------------
# Diff and event derivation
# ---------------------------------------------------------------------------


def diff_snapshots(previous: "Optional[GoalProgressSnapshot]",
                   current:  GoalProgressSnapshot,
                   ) -> Tuple[ProgressEvent, ...]:
    """Compute the events implied by the transition from
    ``previous`` to ``current``.

    The first snapshot for a goal has no previous; this returns
    DIMENSION_ACHIEVED events for any dimension already matched
    in the first snapshot, plus GOAL_ACHIEVED if all dims match
    on the first observation.  Subsequent snapshots emit events
    only for dimensions that transitioned.

    GOAL_ACHIEVED fires when the current snapshot has all
    dimensions matched AND the previous did not (or there was
    no previous).  GOAL_REGRESSED fires when the previous had
    all dimensions matched and the current does not.
    """
    events: List[ProgressEvent] = []
    prev_matched = previous.dimensions_matched if previous else frozenset()
    cur_matched  = current.dimensions_matched
    for dim in cur_matched - prev_matched:
        events.append(ProgressEvent(
            goal_id   = current.goal_id,
            dimension = str(dim),
            kind      = ProgressEventKind.DIMENSION_ACHIEVED,
            turn      = current.recorded_at_turn,
        ))
    for dim in prev_matched - cur_matched:
        events.append(ProgressEvent(
            goal_id   = current.goal_id,
            dimension = str(dim),
            kind      = ProgressEventKind.DIMENSION_REGRESSED,
            turn      = current.recorded_at_turn,
        ))
    # Whole-goal transitions.
    cur_full = (len(current.dimensions_open) == 0
                and len(current.dimensions) > 0)
    prev_full = (previous is not None
                 and len(previous.dimensions_open) == 0
                 and len(previous.dimensions) > 0)
    if cur_full and not prev_full:
        events.append(ProgressEvent(
            goal_id   = current.goal_id,
            dimension = "*",  # whole-goal sentinel
            kind      = ProgressEventKind.GOAL_ACHIEVED,
            turn      = current.recorded_at_turn,
        ))
    elif prev_full and not cur_full:
        events.append(ProgressEvent(
            goal_id   = current.goal_id,
            dimension = "*",
            kind      = ProgressEventKind.GOAL_REGRESSED,
            turn      = current.recorded_at_turn,
        ))
    return tuple(events)


# ---------------------------------------------------------------------------
# History append + recovery bump update
# ---------------------------------------------------------------------------


def _append_to_history(history:  GoalProgressHistory,
                       snapshot: GoalProgressSnapshot,
                       events:   Tuple[ProgressEvent, ...]) -> None:
    """FIFO append to a goal's history with eviction at max_size.

    Both the snapshot list and the event list are bounded by the
    same cap.  Events older than ``max_size`` snapshots ago are
    evicted to keep the history compact.
    """
    history.snapshots.append(snapshot)
    while len(history.snapshots) > history.max_size:
        history.snapshots.pop(0)
    for ev in events:
        history.events.append(ev)
    while len(history.events) > history.max_size:
        history.events.pop(0)


def _bump_recovery_for_regressions(
    history: GoalProgressHistory,
    events:  Tuple[ProgressEvent, ...],
) -> None:
    """Increase the goal's recovery_bump for each regression
    event.  Capped at ``_RECOVERY_BUMP_MAX``.

    DIMENSION_REGRESSED and GOAL_REGRESSED both contribute; a
    whole-goal regression is also reflected as one or more
    dimension regressions, so we count only the dimension events
    to avoid double-counting.
    """
    n_regressions = sum(
        1 for ev in events
        if ev.kind == ProgressEventKind.DIMENSION_REGRESSED
    )
    if n_regressions == 0:
        return
    bump = float(history.recovery_bump or 0.0)
    bump += n_regressions * _RECOVERY_BUMP_PER_EVENT
    if bump > _RECOVERY_BUMP_MAX:
        bump = _RECOVERY_BUMP_MAX
    history.recovery_bump = bump


# ---------------------------------------------------------------------------
# Public per-turn entry point
# ---------------------------------------------------------------------------


def record_progress(ws: WorldState,
                    *,
                    turn: int) -> Tuple[ProgressEvent, ...]:
    """Compute snapshots for every goal in the forest, derive
    events from snapshot diffs, append both to per-goal history,
    and apply recovery-priority bumps.

    Returns the flat tuple of all events emitted across all
    goals this turn — useful for callers that want to log or
    relay the events without re-walking the forest.

    Idempotent within a turn: calling twice with the same turn
    appends two snapshots (the second one immediately after the
    first); doesn't break invariants but isn't intended.  The
    canonical caller is the goal manager's tick once per turn.
    """
    all_events: List[ProgressEvent] = []
    for gid, goal in ws.goal_forest.goals.items():
        snapshot = compute_snapshot(goal, ws, turn=turn)
        if snapshot is None:
            continue
        history = goal.progress_history
        previous = history.snapshots[-1] if history.snapshots else None
        events = diff_snapshots(previous, snapshot)
        _append_to_history(history, snapshot, events)
        _bump_recovery_for_regressions(history, events)
        all_events.extend(events)
    return tuple(all_events)


def decay_recovery_bumps(ws: WorldState) -> None:
    """Reduce all goals' recovery_bump magnitudes by the
    configured per-turn decay rate.

    Call once per turn AFTER ``record_progress`` so newly-bumped
    goals retain their full bump for the next selection pass and
    fade only on subsequent turns.

    Bumps below a small epsilon are zeroed out to avoid
    accumulating numerical noise across many turns.
    """
    for goal in ws.goal_forest.goals.values():
        history = goal.progress_history
        bump = float(history.recovery_bump or 0.0)
        if bump <= 0.0:
            continue
        bump *= _RECOVERY_DECAY_PER_TURN
        if bump < 1e-4:
            bump = 0.0
        history.recovery_bump = bump


_MINER_NAME: str = "miner:GoalRegression"
_MINER_SCOPE: Scope = Scope(kind=ScopeKind.LIFE)
_REGRESSION_FORBIDDANCE_THRESHOLD: float = 0.4


def regression_forbidden_keys(ws: WorldState) -> "set":
    """Return the set of ``(row, col, action)`` keys that planners
    should treat as forbidden because a high-credence
    :class:`GoalRegressionClaim` records that taking that action
    from that cell regressed a currently-achieved goal.

    Direction-aware: only the specific ``(pre_state, action)`` pair
    is forbidden; the same cell with a different action remains
    permissible.

    The forbiddance lifts automatically when the regressed goal's
    status flips out of ``ACHIEVED`` — the function reads goal
    status fresh each call.

    The credence threshold matches the cell-wall forbiddance bar
    (``_REGRESSION_FORBIDDANCE_THRESHOLD``): a single observation
    mints an advisory low-credence claim; repeated observations
    cross the bar and become real forbiddances.
    """
    from .types import GoalStatus
    out = set()
    if ws is None or len(getattr(ws, "hypotheses", {})) == 0:
        return out
    goal_forest = getattr(ws, "goal_forest", None)
    if goal_forest is None:
        return out
    goals = getattr(goal_forest, "goals", {}) or {}
    for h in ws.hypotheses.values():
        if not isinstance(h.claim, GoalRegressionClaim):
            continue
        if float(h.credence.point) < _REGRESSION_FORBIDDANCE_THRESHOLD:
            continue
        goal = goals.get(h.claim.goal_id)
        if goal is None or goal.status != GoalStatus.ACHIEVED:
            continue
        try:
            r = int(h.claim.pre_state[0])
            c = int(h.claim.pre_state[1])
        except (TypeError, ValueError, IndexError):
            continue
        out.add((r, c, str(h.claim.action_id)))
    return out


def mine_regression_claims(ws:     WorldState,
                           events: Tuple[ProgressEvent, ...],
                           *,
                           step:   int) -> int:
    """For every regression event in ``events``, propose one
    :class:`GoalRegressionClaim` keyed on the pre-action state and
    last action recorded in ``ws.agent``.

    Reads ``ws.agent["_turn_pre_state"]`` and
    ``ws.agent["_last_action"]``; if either is missing the claim
    cannot be keyed and the event is skipped (recoverable — the
    next regression with both slots populated will mint).

    Returns the number of claims proposed (for caller logging).

    GOAL_REGRESSED events also reflect as one or more
    DIMENSION_REGRESSED events on the same turn; this miner only
    consumes DIMENSION_REGRESSED to avoid double-counting.
    """
    pre_state = ws.agent.get("_turn_pre_state")
    action_id = ws.agent.get("_last_action")
    if pre_state is None or action_id is None:
        return 0
    proposed = 0
    for ev in events:
        if ev.kind != ProgressEventKind.DIMENSION_REGRESSED:
            continue
        claim = GoalRegressionClaim(
            pre_state = tuple(pre_state) if isinstance(pre_state, (list, tuple))
                                         else pre_state,
            action_id = str(action_id),
            goal_id   = str(ev.goal_id),
            dimension = str(ev.dimension),
        )
        _store.propose(
            ws,
            claim  = claim,
            source = _MINER_NAME,
            scope  = _MINER_SCOPE,
            step   = int(step),
        )
        proposed += 1
    return proposed


def effective_recovery_bump(goal: Goal) -> float:
    """Return the current recovery-priority bump for a goal
    (zero when no recent regression).

    Read by the primary goal selector when computing effective
    priority for ranking.
    """
    history = getattr(goal, "progress_history", None)
    if history is None:
        return 0.0
    return float(getattr(history, "recovery_bump", 0.0) or 0.0)
