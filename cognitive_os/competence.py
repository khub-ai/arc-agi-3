"""Competence-gated learn-priority bias — goal-switch, not mode-switch.

Why this exists
===============

Without this module the engine runs in permanent learn-dominant mode:
at every tick the goal selector re-competes every goal against every
other, with ``reduce_uncertainty:*``, ``explore:*``, and ``probe::*``
pressures always on at the same weights.  Even after an agent has
committed all the claims it needs to finish a level, the next tick
still emits a novel-entity-interact goal at priority 0.48 that can
unseat the winning plan in favour of a side-quest.

A cognitively mature agent has two competing first-class drives:

* **learning** — "model the world better".  Fires goals of the
  ``reduce_uncertainty`` / ``explore`` / ``probe`` family.
* **completing the level** — "do the task".  Fires goals of the
  ``episode`` / ``role:target:*`` family (and any other non-learn
  family).

The engine already arbitrates goals through priority comparison.  So
rather than bolt on a second control system (a global LEARN/EXECUTE
flag with an all-or-nothing 0.01× dampener), this module answers one
narrow question each tick:

    *"What priority scale should learning sit at, given what I can
    currently plan?"*

and writes the answer onto the world state.  Normal goal arbitration
in ``goal_forest.candidates_by_priority`` then picks the right goal.
No mode flag.  No cliff.  No separate control system.

The scale is continuous, not binary:

* No task goal plannable → scale = **1.0**.  Learning runs at full
  native priority.  This is the bootstrap / unknown-world regime.
* A top-N task goal plannable → scale = **cfg.plannable_task_scale**
  (default 0.3).  Learning is demoted below the task goal but stays
  in the running — if the task goal stalls for a tick (plan fails,
  target moves, an obstacle appears), learning naturally wins the
  next sort and the agent probes once, then cedes back to the task
  goal when it becomes plannable again.  Graceful degradation by
  construction, without any need for the competence check to "flip
  back" — because there is no flag to flip.

Robotics analogue
-----------------

Identical mechanism drives a household robot's drive balance.  Given
"fetch the mug", if committed claims (kitchen layout, locomotion
primitives, gripper calibration) are enough to plan the fetch, the
delivery goal sits at its native priority and learning sits at ~0.3×
— the robot does not detour to investigate the wine glass on the way,
but if the fetch plan stalls (unexpected obstacle), a learning probe
fires in the same tick rather than the robot standing still.  When
the robot arrives and the mug is not there, the delivery plan becomes
unplannable, learning's 0.3× scale still beats a zero-priority
stalled-delivery goal, and a fresh ``reduce_uncertainty:object_
location:mug`` goal drives a search.  Same mechanism; no flag flip;
no special-case recovery code.

Out of scope for this module
----------------------------

* **Task-gated exploration ranking** (which learn-goals to prioritise
  *within* learning mode based on their relevance to the top task
  goal) — future GAP 28.
* **Plan commitment across ticks** (caching and replaying plans
  rather than recomputing each tick) — cheap follow-up, not in this
  module.
* **Skill compilation / Options** — the long-term replacement for
  real-time planning in the task-dominant regime.  Phase 7.

Historical note
---------------

Earlier drafts of this spec framed the switch as a binary LEARN vs
EXECUTE **mode**, with learn-family goals dampened by 0.01× in EXECUTE
mode.  Review feedback: a global mode flag is a parallel control
system that overrides the existing goal-arbitration machinery, and the
binary 0.01× dampener means if the competence check is wrong the
agent goes blind to curiosity until the flag flips back.  This module
is the reframe — the same plannability check is now input to a
*continuous priority scale on learning*, and the LEARN vs EXECUTE
distinction emerges organically from normal priority arbitration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import OperatingMode
from .types import Action, WorldState


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class CompetenceConfig:
    """Tuning for the competence-gated learn-priority bias.

    Attributes
    ----------
    enabled
        Master switch.  When False the scale is always 1.0 (learning
        at full native priority) — matches pre-spec behaviour for
        regression-safety in existing tests.
    plannable_task_scale
        Multiplicative factor applied to learn-goal priorities when
        the gate detects that at least one top-N task goal is
        plannable from committed claims.  Default 0.3: task-dominant
        but learning stays as a live fallback if the task goal stalls
        or turns out to be wrongly marked plannable.  Setting this to
        0.01 approximates the original binary-mode dampening behaviour
        (not recommended — that loses the graceful-degradation
        property).  Setting to 1.0 disables the bias entirely.
    learn_goal_prefixes
        Goal-id prefixes considered "learn" (exploration / curiosity /
        probing).  Anything else is treated as a task goal.  The
        default covers all learn-mode goal families emitted by the
        instinct and explorer layers as of the spec date.
    top_n_task_goals
        How many of the highest-priority task goals to test for
        plannability.  Default 3 covers ``episode`` + ``role:target:*``
        + one spare.  Larger values cost proportionally more planner
        calls per tick.

    Robotics tuning note: robotics agents typically want the default
    0.3 or slightly softer (0.4–0.5) so that opportunistic
    observations during long task stretches can bubble through.  ARC
    agents can tighten toward 0.1–0.2 if step budgets are extremely
    tight, but 0.01 is a regression to the old binary-cliff behaviour
    and should be avoided.
    """

    enabled:                          bool             = True
    plannable_task_scale:             float            = 0.3
    competition_plannable_task_scale: float            = 0.05
    learn_goal_prefixes:              Tuple[str, ...]  = (
        "reduce_uncertainty:",
        "explore:",
        "probe::",
    )
    top_n_task_goals:                 int              = 3


# ---------------------------------------------------------------------------
# Goal classification
# ---------------------------------------------------------------------------


def is_learn_goal(goal_id: str, cfg: CompetenceConfig) -> bool:
    """True iff ``goal_id`` matches one of the configured learn-goal
    prefixes.  Pure string test — no ws lookup, so this is cheap to
    call from the goal-forest sort path."""
    return any(goal_id.startswith(p) for p in cfg.learn_goal_prefixes)


def is_task_goal(goal_id: str, cfg: CompetenceConfig) -> bool:
    """True iff ``goal_id`` is NOT a learn goal.  Task goals are
    everything the competence gate considers candidates for ``can-I-
    execute-now?``.  Includes adapter primary goals (``episode``),
    role-synthesis goals (``role:target:*``), and any future
    task-family that does not adopt a learn prefix."""
    return not is_learn_goal(goal_id, cfg)


# ---------------------------------------------------------------------------
# The scale computation
# ---------------------------------------------------------------------------


def compute_learn_priority_scale(
    ws:             WorldState,
    action_space:   List[Action],
    *,
    step:           int,
    cfg:            CompetenceConfig,
    operating_mode: Optional[OperatingMode] = None,
) -> float:
    """Return the multiplicative priority scale that learn-family
    goals should be viewed through for this tick.

    * ``1.0`` — no task goal is currently BFS-plannable from committed
      claims, OR the gate is disabled, OR the forest is empty.
      Learning at full native priority.
    * ``cfg.plannable_task_scale`` — at least one of the top-N task
      goals is plannable.  Learning demoted but still live.

    Contract
    --------
    * **No mutation.**  Does not touch ``ws.goal_forest.active_goal_id``,
      goal priorities, or any other persistent state.  Callers may
      safely invoke this from within read-only diagnostic paths.
    * **Planner is queried, not committed.**  Each candidate is passed
      to :func:`planner.compute_plan` which performs a read-only BFS
      over committed claims.  A non-None non-empty plan counts as
      "reachable".
    * **"Any" beats "highest".**  If the highest-priority task goal is
      abstract and unplannable (e.g. an ``episode_won`` resource
      predicate that BFS cannot satisfy by motion alone) but a
      lower-priority concrete task goal is plannable, we still
      demote learning — the agent has a way to make progress on the
      level and should not run exploration at full priority.
    * **Degenerate cases.**
        - No task goals in the forest → 1.0 (nothing to demote toward).
        - Gate disabled in config → 1.0 (pre-spec default).
    """
    if not cfg.enabled:
        return 1.0

    gf = getattr(ws, "goal_forest", None)
    if gf is None or not gf.goals:
        return 1.0

    from . import goal_forest as _gf
    from .planner import compute_plan

    # Priority-ordered candidates via the existing filter (excludes
    # ACHIEVED / PRUNED / ABANDONED / FAIL-blocked).  Reuse rather
    # than reimplement to guarantee the gate and the selector agree
    # on what counts as an eligible goal.
    ordered_ids = _gf.candidates_by_priority(ws)
    task_ids: List[str] = []
    for gid in ordered_ids:
        if is_task_goal(gid, cfg):
            task_ids.append(gid)
            if len(task_ids) >= cfg.top_n_task_goals:
                break

    if not task_ids:
        return 1.0

    if operating_mode == OperatingMode.COMPETITION:
        plannable_scale = float(cfg.competition_plannable_task_scale)
    else:
        plannable_scale = float(cfg.plannable_task_scale)

    for gid in task_ids:
        plan = compute_plan(ws, gid, action_space, step=step)
        if plan is not None and plan.steps:
            return plannable_scale

    return 1.0


# ---------------------------------------------------------------------------
# Host-side helpers: write / read the scale on ws
# ---------------------------------------------------------------------------

#: Key under which the current learn-priority scale is stored on
#: ``ws.agent``.  Readers that want to avoid importing this module
#: can access it directly (e.g. for telemetry / diagnostics).
WS_LEARN_SCALE_KEY: str = "_learn_priority_scale"


def apply_learn_priority_scale(ws: WorldState, scale: float) -> None:
    """Write ``scale`` into ``ws.agent[WS_LEARN_SCALE_KEY]``.  The
    goal-forest sort key reads this value on every
    :func:`candidates_by_priority` call."""
    ws.agent[WS_LEARN_SCALE_KEY] = float(scale)


def current_learn_priority_scale(ws: WorldState) -> float:
    """Read back the scale written by
    :func:`apply_learn_priority_scale`.  Defaults to 1.0 if unset
    (e.g. on the very first tick before the gate has run) — which
    means "no bias; learning at full priority"."""
    raw = ws.agent.get(WS_LEARN_SCALE_KEY)
    if raw is None:
        return 1.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 1.0
