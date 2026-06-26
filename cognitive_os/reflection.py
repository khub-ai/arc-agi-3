"""Closed-loop action substrate — Phase 5: reflection pass.

See ``docs/SPEC_closed_loop_action_substrate.md``.

Reflection runs at each turn boundary.  It scans the recent
:class:`Trajectory` and emits structured observations about the
agent's own behaviour: cycles, stalls, repeated-failure patterns,
unachievable goals.  These show up as :class:`ReflectionResult`
entries; downstream consumers (the goal forest, the planner, the
human-communication surface) read them and adjust.

Phase 5 ships two foundational detectors:

* :func:`detect_cycles` — the agent has been alternating between
  the same small set of targets across recent turns without
  achieving the goal.
* :func:`detect_stalls` — a goal has been pursued across many
  turns with most outcomes contradicted, suggesting the goal's
  current decomposition is wrong or the goal itself is
  unachievable in the current world state.

Richer detectors (predicted-but-not-observed, surprise, novelty)
are deferred to subsequent specifications layered on top of this
substrate.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

from .types import (
    MatchKind,
    ReflectionResult,
    WorldState,
)


# Default lookback window for cycle detection.  Small enough that
# the detector picks up recent oscillation; large enough to require
# at least two full cycles before flagging.  Configurable per call.
_CYCLE_LOOKBACK_DEFAULT: int = 8

# Number of distinct cells in the recent trajectory below which a
# pattern counts as "cycling."  Two cells alternating across eight
# turns is the classic cycle the substrate must catch.
_CYCLE_DISTINCT_CELLS_MAX: int = 2

# Minimum number of confirmed visits within the lookback window
# before cycle detection fires.  Avoids false positives on the
# first few turns of a session.
_CYCLE_MIN_VISITS: int = 4

# Default lookback window and threshold for stall detection.
# A stall is "many recent outcomes for this goal were
# contradicted."
_STALL_LOOKBACK_DEFAULT:    int   = 10
_STALL_CONTRADICTION_RATIO: float = 0.6


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------


def detect_cycles(ws:       WorldState,
                  *,
                  lookback: int = _CYCLE_LOOKBACK_DEFAULT,
                  ) -> "tuple[tuple[tuple[int, int, ...], int], ...]":
    """Scan the recent trajectory for behavioural cycles.

    A cycle is detected when the most recent ``lookback`` outcomes
    visited at most ``_CYCLE_DISTINCT_CELLS_MAX`` distinct cells,
    with at least ``_CYCLE_MIN_VISITS`` visits among them.  The
    classic shape this catches: agent oscillates between cell A
    and cell B for many turns without making progress on its
    parent goal.

    A "visit" is a confirmed positional assertion — the agent
    actually reached the asserted cell, regardless of whether
    other property-outcome assertions in the same prediction held
    or were contradicted.  Looking at the positional assertion
    rather than the overall match catches cycles in the
    common case where an action's positional prediction confirmed
    (agent moved to the cell as planned) but a property-outcome
    prediction failed (the cell didn't flip the property the
    agent expected).

    Returns a tuple of ``(cell_tuple, count)`` pairs.  Each
    ``cell_tuple`` is a sorted tuple of cells in the cycle;
    ``count`` is the number of qualifying visits in the lookback
    window.  Empty tuple means no cycle detected.
    """
    trajectory = ws.trajectory
    if not trajectory.outcomes:
        return ()
    recent = trajectory.outcomes[-int(lookback):] if lookback > 0 else trajectory.outcomes
    cells_visited: list = []
    for outcome in recent:
        # Walk the prediction's assertions in order; count this
        # outcome as a visit when the first cell-keyed assertion
        # confirmed (per_assertion_match[i] == CONFIRMED).
        for i, assertion in enumerate(outcome.prediction.predicted_assertions):
            try:
                tk = assertion.condition.canonical_key()
            except Exception:
                continue
            if not (tk and len(tk) >= 3
                    and tk[0] in ("AgentAtCell", "AtPosition")):
                continue
            # Found the positional assertion.  Check whether it
            # confirmed independently of the overall match.
            if i >= len(outcome.per_assertion_match):
                break
            if outcome.per_assertion_match[i] != MatchKind.CONFIRMED:
                break
            try:
                cells_visited.append((int(tk[1]), int(tk[2])))
            except (TypeError, ValueError):
                pass
            break
    if len(cells_visited) < _CYCLE_MIN_VISITS:
        return ()
    distinct = set(cells_visited)
    if len(distinct) > _CYCLE_DISTINCT_CELLS_MAX:
        return ()
    sorted_cells = tuple(sorted(distinct))
    return ((sorted_cells, len(cells_visited)),)


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------


def detect_stalls(ws:        WorldState,
                  *,
                  lookback:  int   = _STALL_LOOKBACK_DEFAULT,
                  threshold: float = _STALL_CONTRADICTION_RATIO,
                  ) -> "tuple[tuple[str, str, dict], ...]":
    """Scan the recent trajectory per-goal and flag goals where
    the contradiction-to-confirmed ratio is high.

    A stall signal is a tuple ``(goal_id, "stalled", payload)``
    where ``payload`` is a dict with ``contradicted``,
    ``confirmed``, ``total``, and ``ratio`` keys.  Returns a
    tuple of zero or more stall signals.

    The threshold ``0.6`` (default) means: if 60% or more of a
    goal's recent outcomes were contradicted, flag it.  Tunable
    per call.
    """
    trajectory = ws.trajectory
    if not trajectory.outcomes:
        return ()
    recent = trajectory.outcomes[-int(lookback):] if lookback > 0 else trajectory.outcomes
    by_goal: "dict[str, list]" = {}
    for outcome in recent:
        gid = outcome.prediction.active_goal_id
        if gid is None:
            continue
        by_goal.setdefault(str(gid), []).append(outcome)
    signals: list = []
    for gid, outcomes in by_goal.items():
        n_contradicted = sum(
            1 for o in outcomes if o.overall_match == MatchKind.CONTRADICTED
        )
        n_confirmed = sum(
            1 for o in outcomes if o.overall_match == MatchKind.CONFIRMED
        )
        n_total = len(outcomes)
        if n_total < 3:
            continue  # too few outcomes to be statistically meaningful
        ratio = n_contradicted / max(1, n_total)
        if ratio >= threshold:
            signals.append((gid, "stalled", {
                "contradicted": n_contradicted,
                "confirmed":    n_confirmed,
                "total":        n_total,
                "ratio":        round(ratio, 3),
            }))
    return tuple(signals)


# ---------------------------------------------------------------------------
# Per-turn reflection pass
# ---------------------------------------------------------------------------


def run_reflection(ws: WorldState) -> ReflectionResult:
    """Run the reflection pass over the world state's trajectory
    and return a structured :class:`ReflectionResult`.

    Combines cycle detection and stall detection.  As a side
    effect, refreshes the goal forest's
    :attr:`GoalForest.stalled_goal_ids` and
    :attr:`GoalForest.cycling_cells` annotation fields so the
    primary goal selector can bias selection away from
    unproductive goals and cells.  These annotations are
    overwritten on every call — entries drop the moment a goal
    stops qualifying as stalled or a cell stops appearing in a
    detected cycle.

    Future detectors (predicted-but-not-observed, surprise,
    novelty) plug into this entry point without changing the
    consumer contract.

    The result's ``recorded_at_turn`` is set from ``ws.step``.
    Callers are responsible for ensuring step is updated before
    invoking reflection.
    """
    cycles = detect_cycles(ws)
    stalls = detect_stalls(ws)
    # Refresh goal forest annotations.  Sets are overwritten on
    # every pass — no manual clearing needed when a goal stops
    # being stalled.
    ws.goal_forest.stalled_goal_ids = {
        str(gid) for (gid, _kind, _payload) in stalls
    }
    cycling: "set[Tuple[int, int]]" = set()
    for cells, _count in cycles:
        for cell in cells:
            try:
                cycling.add((int(cell[0]), int(cell[1])))
            except (TypeError, ValueError, IndexError):
                pass
    ws.goal_forest.cycling_cells = cycling
    return ReflectionResult(
        meta_claims       = (),  # Reserved for richer detectors
        goal_signals      = stalls,
        cycle_detections  = cycles,
        recorded_at_turn  = int(ws.step),
    )
