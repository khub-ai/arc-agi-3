"""Closed-loop action substrate — Phases 2 and 3 (prediction
emission, outcome recording, and sharp belief updates from
outcomes).

See ``docs/SPEC_closed_loop_action_substrate.md``.

Phase 2 wires the per-action evaluation cycle.  Adapters construct a
:class:`Prediction` before each action, execute the action, then call
:func:`record_outcome` to compare the prediction to the post-action
world state.  Comparison results land in the world state's
:class:`Trajectory` and become available to later phases — sharp
belief updates (Phase 3), living goal-tree re-decomposition
(Phase 4), reflection (Phase 5), rich predictions with property
outcomes (Phase 6).

Phase 2 is intentionally minimal: prediction shapes carry only what
the simplest adapter integration needs (positional assertion plus
provenance).  Later phases extend the prediction shape and the
update rules without changing the substrate's contract.

Public API
==========

* :func:`compare_assertion` — compare one predicted assertion to
  the current world state, returning a :class:`MatchKind`.
* :func:`reduce_overall_match` — collapse a per-assertion match
  tuple into the prediction's overall match summary.
* :func:`record_outcome` — the canonical adapter entry point;
  given a prediction and the post-action world state, build the
  :class:`Outcome` and append it to the trajectory.

Trajectory eviction follows a strict FIFO policy at the
``max_size`` limit defined on the :class:`Trajectory`.  The lazy
reverse indices are kept consistent across evictions.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .credence import update_on_contradict, update_on_support
from .types import (
    MatchKind,
    Outcome,
    PredictedAssertion,
    Prediction,
    Trajectory,
    WorldState,
)


# Default sharpness multiplier for outcome-driven credence updates
# (Phase 3).  Closed-loop outcomes are higher-quality learning
# signals than ambient miner-derived observations: the agent
# committed to a prediction, acted, and observed the result, all
# within one turn.  Applying the same update three times makes the
# resulting credence delta roughly three times that of an ordinary
# supporting / contradicting event.  Configurable per-call so
# adapters or tests can tune.
_DEFAULT_OUTCOME_SHARPNESS: int = 3


# ---------------------------------------------------------------------------
# Public comparison helpers
# ---------------------------------------------------------------------------


def compare_assertion(assertion: PredictedAssertion,
                      ws:        WorldState) -> MatchKind:
    """Compare one :class:`PredictedAssertion` to the current
    world state.

    Returns:

    * :attr:`MatchKind.UNOBSERVABLE` when the condition's
      ``evaluate`` raises an exception OR returns ``None``
      (the standard "unknown / insufficient data" sentinel
      across the engine's :class:`Condition` hierarchy).
    * :attr:`MatchKind.CONFIRMED` when ``evaluate`` returns
      a value that equals the assertion's ``expected_value``.
    * :attr:`MatchKind.CONTRADICTED` when ``evaluate`` returns
      a value distinct from the expected one.

    Phase 2 does not produce :attr:`MatchKind.AMBIGUOUS` from this
    helper — ambiguous outcomes arise in Phase 3+ when the
    comparison consults credence-weighted evidence.  The substrate
    reserves the value as a first-class possibility so callers can
    handle it without API change later.
    """
    try:
        result = assertion.condition.evaluate(ws)
    except Exception:
        # An evaluator that raises is uninformative, not contradictory —
        # treat the same as a None return.  Callers reading the
        # outcome shouldn't be punished for an evaluator bug.
        return MatchKind.UNOBSERVABLE
    if result is None:
        return MatchKind.UNOBSERVABLE
    return (MatchKind.CONFIRMED
            if bool(result) == bool(assertion.expected_value)
            else MatchKind.CONTRADICTED)


def reduce_overall_match(per_assertion: Tuple[MatchKind, ...]) -> MatchKind:
    """Collapse a per-assertion match tuple into the prediction's
    overall match summary.

    Reduction rule (per spec §"Outcome"):

    1. Any CONTRADICTED → overall is CONTRADICTED (one falsified
       assertion is enough to mark the whole prediction as failed
       for downstream learning).
    2. Otherwise any CONFIRMED → overall is CONFIRMED.
    3. Otherwise any AMBIGUOUS → overall is AMBIGUOUS.
    4. Otherwise (all UNOBSERVABLE, or empty tuple) → UNOBSERVABLE.

    The empty-prediction case (no assertions) returns UNOBSERVABLE
    rather than CONFIRMED — an action with no expressed expectation
    contributed no learnable signal, regardless of what happened
    afterwards.
    """
    if not per_assertion:
        return MatchKind.UNOBSERVABLE
    if any(m == MatchKind.CONTRADICTED for m in per_assertion):
        return MatchKind.CONTRADICTED
    if any(m == MatchKind.CONFIRMED for m in per_assertion):
        return MatchKind.CONFIRMED
    if any(m == MatchKind.AMBIGUOUS for m in per_assertion):
        return MatchKind.AMBIGUOUS
    return MatchKind.UNOBSERVABLE


# ---------------------------------------------------------------------------
# Trajectory management
# ---------------------------------------------------------------------------


def _evict_oldest(trajectory: Trajectory) -> None:
    """Remove the oldest outcome and adjust the lazy reverse
    indices.  Called by :func:`record_outcome` when the trajectory
    is at its ``max_size`` limit.

    The reverse indices store positional offsets into ``outcomes``;
    eviction shifts every remaining offset down by one.  Indices
    that become empty are removed from their dict entirely so
    subsequent index lookups don't return stale empty lists.
    """
    if not trajectory.outcomes:
        return
    trajectory.outcomes.pop(0)
    for idx_dict in (trajectory.index_by_goal,
                     trajectory.index_by_action,
                     trajectory.index_by_cell):
        empty_keys = []
        for key, positions in idx_dict.items():
            shifted = [p - 1 for p in positions if p > 0]
            if shifted:
                idx_dict[key] = shifted
            else:
                empty_keys.append(key)
        for k in empty_keys:
            idx_dict.pop(k, None)


def _index_outcome(trajectory: Trajectory,
                   outcome:    Outcome,
                   position:   int) -> None:
    """Update the lazy reverse indices to include the outcome's
    position.  Called after the outcome is appended.

    The cell index is populated from any ``AgentAtCell`` /
    ``AtPosition``-shaped condition in the prediction's assertions
    — adapters that emit movement predictions get a free
    cell-keyed query path without writing the index entry
    themselves.
    """
    pred = outcome.prediction
    # action index
    trajectory.index_by_action.setdefault(
        str(pred.action_id), []).append(position)
    # goal index
    if pred.active_goal_id:
        trajectory.index_by_goal.setdefault(
            str(pred.active_goal_id), []).append(position)
    # cell index — pull from any cell-keyed assertion's canonical_key
    for a in pred.predicted_assertions:
        try:
            tk = a.condition.canonical_key()
        except Exception:
            continue
        if (tk and len(tk) >= 3
                and tk[0] in ("AgentAtCell", "AtPosition", "TriggerVisitedAtLeast")):
            try:
                cell = (int(tk[1]), int(tk[2]))
            except (TypeError, ValueError):
                continue
            trajectory.index_by_cell.setdefault(
                cell, []).append(position)


def record_outcome(ws:         WorldState,
                   prediction: Prediction) -> Outcome:
    """Compare ``prediction`` against the current world state,
    construct the :class:`Outcome`, append it to the trajectory,
    and update the lazy reverse indices.

    This is the canonical adapter entry point for closing the loop
    on an action.  Adapters call it after executing the action and
    refreshing the world state.  No engine component currently
    consumes the recorded outcomes — Phase 3 wires belief updates,
    Phase 4 wires goal-tree re-decomposition, Phase 5 wires
    reflection.

    The current ``ws.step`` is recorded as the outcome's
    ``recorded_at_turn``.  Adapters should ensure the step counter
    reflects the post-action turn before calling.

    Returns the constructed :class:`Outcome` for the convenience
    of callers that want to log or inspect the result inline.
    """
    matches = tuple(compare_assertion(a, ws)
                    for a in prediction.predicted_assertions)
    overall = reduce_overall_match(matches)
    outcome = Outcome(
        prediction          = prediction,
        per_assertion_match = matches,
        overall_match       = overall,
        recorded_at_turn    = int(ws.step),
    )
    trajectory = ws.trajectory
    while len(trajectory.outcomes) >= int(trajectory.max_size):
        _evict_oldest(trajectory)
    position = len(trajectory.outcomes)
    trajectory.outcomes.append(outcome)
    _index_outcome(trajectory, outcome, position)
    return outcome


# ---------------------------------------------------------------------------
# Trajectory query helpers
# ---------------------------------------------------------------------------


def outcomes_for_goal(ws:      WorldState,
                      goal_id: str,
                      *,
                      limit:   Optional[int] = None) -> Tuple[Outcome, ...]:
    """All recent outcomes whose prediction's ``active_goal_id``
    matches.  Order is oldest → newest (matches the trajectory's
    natural order).  Optional ``limit`` truncates from the start
    (most recent ``limit`` entries).
    """
    positions = ws.trajectory.index_by_goal.get(str(goal_id), [])
    outcomes = [ws.trajectory.outcomes[p]
                for p in positions
                if 0 <= p < len(ws.trajectory.outcomes)]
    if limit is not None and len(outcomes) > limit:
        outcomes = outcomes[-limit:]
    return tuple(outcomes)


def outcomes_for_action(ws:        WorldState,
                        action_id: str,
                        *,
                        limit:     Optional[int] = None) -> Tuple[Outcome, ...]:
    """All recent outcomes whose prediction's ``action_id`` matches."""
    positions = ws.trajectory.index_by_action.get(str(action_id), [])
    outcomes = [ws.trajectory.outcomes[p]
                for p in positions
                if 0 <= p < len(ws.trajectory.outcomes)]
    if limit is not None and len(outcomes) > limit:
        outcomes = outcomes[-limit:]
    return tuple(outcomes)


def outcomes_for_cell(ws:    WorldState,
                      cell:  Tuple[int, int],
                      *,
                      limit: Optional[int] = None) -> Tuple[Outcome, ...]:
    """All recent outcomes whose prediction included a cell-keyed
    assertion at ``cell``.  Useful for "have I visited this cell
    and did the predicted change occur?" queries — exactly the
    signal walker re-targeting needs in Phase 4."""
    positions = ws.trajectory.index_by_cell.get(
        (int(cell[0]), int(cell[1])), [])
    outcomes = [ws.trajectory.outcomes[p]
                for p in positions
                if 0 <= p < len(ws.trajectory.outcomes)]
    if limit is not None and len(outcomes) > limit:
        outcomes = outcomes[-limit:]
    return tuple(outcomes)


# ---------------------------------------------------------------------------
# Phase 3: Sharp belief updates from outcomes
# ---------------------------------------------------------------------------


def update_credence_from_outcome(
    ws:        WorldState,
    outcome:   Outcome,
    *,
    sharpness: int = _DEFAULT_OUTCOME_SHARPNESS,
) -> int:
    """Apply outcome-driven credence updates to the beliefs that
    supported the prediction.

    Phase 3 of the closed-loop substrate.  When a prediction
    confirms or contradicts unambiguously, the beliefs whose ids
    appear in ``prediction.supporting_belief_ids`` receive a
    credence update of magnitude ``sharpness`` times the standard
    per-observation step.  Three is a sensible default: a
    closed-loop outcome is roughly three times as informative as
    an ambient miner-derived observation because it represents a
    deliberate experiment whose result the agent watched for.

    No-op when:

    * ``ws.config`` is None (defensive — credence config is
      required for the underlying update functions).
    * ``outcome.overall_match`` is :attr:`MatchKind.AMBIGUOUS` or
      :attr:`MatchKind.UNOBSERVABLE` (no clear signal to apply).
    * ``prediction.supporting_belief_ids`` is empty (no supporting
      beliefs to update — common in Phase 2's minimal positional
      predictions; non-empty in Phase 6's rich predictions).
    * A referenced belief id is not in ``ws.hypotheses`` (silently
      skipped — the belief may have been retracted).

    Returns the count of belief updates applied (zero or more
    times the number of supporting beliefs).  Useful for audit
    output and tests.

    Robotics analogue: when a robot's grip-close action carries a
    prediction "object will be in the gripper" supported by the
    belief "this object's grip-pose is X", and the post-action
    observation confirms the grip, the belief's credence rises
    sharply in one step rather than crawling up over many similar
    observations.
    """
    if ws.config is None:
        return 0
    if outcome.overall_match not in (MatchKind.CONFIRMED,
                                      MatchKind.CONTRADICTED):
        return 0
    supporting_ids = outcome.prediction.supporting_belief_ids or ()
    if not supporting_ids:
        return 0
    cfg = ws.config.credence
    step = int(outcome.recorded_at_turn)
    apply_n = max(1, int(sharpness))
    update_count = 0
    update_fn = (update_on_support
                 if outcome.overall_match == MatchKind.CONFIRMED
                 else update_on_contradict)
    for hid in supporting_ids:
        h = ws.hypotheses.get(str(hid))
        if h is None:
            continue
        new_credence = h.credence
        for _ in range(apply_n):
            new_credence = update_fn(new_credence, step, cfg)
        # Hypothesis is a frozen-ish dataclass; mutate the credence
        # field in place via dataclasses.replace would require
        # re-inserting.  In practice Hypothesis.credence is a mutable
        # field; assign directly.  (Mirrors the existing
        # update_credence_from_events pattern in hypothesis_store.)
        h.credence = new_credence
        update_count += apply_n
    return update_count
