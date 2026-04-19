"""Game-goal decomposer — turns LLM-sourced scenario priors into
plannable subgoals.

:class:`GameCharacterizationTrigger` emits
``PropertyClaim("_game", k, v)`` hypotheses that describe the
scenario at a conceptual level: ``genre``, ``win_pattern``,
``characters``, ``mechanics``.  None of those can be acted on
directly — the planner needs :class:`~cognitive_os.types.Goal`
objects whose conditions evaluate over observed state.

This module bridges that gap.  Each step the decomposer inspects
the current ``_game`` claims and, based on the free-form
``win_pattern`` string, synthesises intermediate subgoals under the
adapter-seeded episode goal.  The LLM characterisation only says
*what kind* of goal to create; the actual target entity and
coordinates are chosen from the engine's own perception — no LLM
pixel trust.

Strategy dispatch (piece 2 scope)
---------------------------------
* ``win_pattern`` contains any of
  ``{reach, touch, navigate, arrive, goal, tile}`` (case-insensitive)
  → **reach strategy**: pick the most visually-distinctive stationary
  non-agent, non-wall entity and emit an ``InsideBBox`` goal over it.
* Every other pattern → no-op.  Piece 3+ extends the vocabulary
  (``collect`` / ``survive`` / ``match`` / ...).

Idempotence
-----------
The decomposer runs every step.  It never re-emits an existing
goal.  A previously-emitted reach subgoal that still references a
live entity is left untouched; if the target entity has vanished
(e.g. segmentation churn across level transition, LEVEL-scoped
claim expiry), the goal is abandoned and a new one synthesised from
the current frame.

Why this is an :class:`OracleTrigger` subclass
----------------------------------------------
The abstraction is "per-step hook with ws access", which is
exactly what the decomposer needs.  It happens to not make an LLM
call, but the trigger contract does not require one — the base
class is about dispatch cadence, not about oracle traffic.  Keeping
it inside the trigger list lets adapters compose decomposers with
Observer/Mediator triggers in a single ordered sequence.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from cognitive_os import (
    Goal,
    GoalNode,
    NodeType,
    WorldState,
)
from cognitive_os.claims import PropertyClaim
from cognitive_os.conditions import InsideBBox
from cognitive_os.credence import update_on_contradict
from cognitive_os.goal_forest import add_goal
from cognitive_os.oracle import OracleTrigger
from cognitive_os.types import GoalStatus

from .game_characterization import GAME_ENTITY_ID

if TYPE_CHECKING:  # pragma: no cover
    from cognitive_os.adapters import Adapter
    from cognitive_os.config import EngineConfig


_LOG = logging.getLogger(__name__)

# Goal id prefix for every subgoal this decomposer emits.  Stable
# across ticks so idempotence is checkable by id lookup.
DECOMPOSE_GOAL_PREFIX = "decompose::_game::"

# Keywords that indicate a reach / navigate / touch-a-goal pattern.
# Free-form substring match — the LLM is open-vocabulary so we lean
# permissive.  Misses are harmless (decomposer no-ops); false
# positives would emit a reach goal against a wrong target, which
# piece 3's credence-decay on failure recovers from.
#
# The ``collect`` family is included because a "collect every item"
# pattern decomposes operationally into a *sequence* of reach-toward-
# distinct-target.  Piece 2's target-vanished resynthesis already
# handles the sequencing: once the first item is picked up and its
# entity drops out of segmentation, the decomposer picks a fresh
# target on the next tick.  So a single strategy covers both reach-
# and collect-shaped goals.
_REACH_KEYWORDS: Tuple[str, ...] = (
    "reach",
    "touch",
    "navigate",
    "arrive",
    "goal",
    "tile",
    "exit",
    "destination",
    "collect",
    "gather",
    "pickup",
    "pick up",
    "item",
)

# Default priority for decomposed subgoals.  Sits between the
# adapter-seeded episode goal (priority 1.0, atom, unplannable) and
# the role-goal floor (0.85), so when the episode atom falls through
# for being unplannable, the decomposed reach subgoal is the next
# pick rather than a learn goal.
DEFAULT_DECOMPOSE_PRIORITY = 0.9

# Step budget for a reach subgoal before it is considered failed.
# On expiry the goal is abandoned and the underlying ``win_pattern``
# claim is contradicted once — reducing its credence so a future
# decomposer tick is less dominated by a stale prior.  80 steps is
# generous: typical reach goals on ARC frames should resolve inside
# a couple-dozen steps once the agent commits, so 80 gives the
# planner plenty of runway before we conclude the pattern was wrong.
REACH_GOAL_STEP_BUDGET = 80

# Grace period after emission before a reach goal is checked for
# planner-reachability.  The planner needs motion models committed
# and the agent on-frame to find a path; on ARC episodes the probe
# phase takes ~4 steps, and motion models typically commit around
# s3-s5 with the fast-commit miner.  Five steps gives the engine
# time to settle before we conclude a target is unreachable.
REACH_GOAL_RETARGET_GRACE = 5


class GameDecomposer(OracleTrigger):
    """Synthesises plannable subgoals from ``_game`` scenario priors.

    See module docstring for the strategy table.  The decomposer is
    stateless across episodes (nothing to reset) but holds no
    per-episode state either — idempotence is enforced by checking
    ``ws.goal_forest.goals`` for the expected id prefix.
    """

    name = "game_decomposer"

    def __init__(
        self,
        *,
        priority:           float = DEFAULT_DECOMPOSE_PRIORITY,
        reach_step_budget:  int   = REACH_GOAL_STEP_BUDGET,
        retarget_grace:     int   = REACH_GOAL_RETARGET_GRACE,
    ) -> None:
        self.priority          = priority
        self.reach_step_budget = reach_step_budget
        self.retarget_grace    = retarget_grace
        # goal_id → step-when-emitted.  Used by the decay-on-failure
        # pass to detect reach goals that have outrun their budget.
        self._emit_step: Dict[str, int] = {}
        # Entity ids we've proven unreachable this episode.  Filtered
        # out of fresh target selection so we don't loop back to a
        # target that the planner has already rejected.
        self._avoid_targets: Set[str] = set()

    def reset(self) -> None:
        """Clear per-episode tracking.  Called by the runner at
        episode boundaries."""
        self._emit_step.clear()
        self._avoid_targets.clear()

    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        # First: sweep existing emitted goals for success / churn /
        # timeout / planner-unreachable.  On timeout, contradict the
        # win_pattern claim so a stale prior loses ground.
        self._sweep_tracked_goals(ws, step, adapter)

        chars = _read_game_claims(ws)
        win_pattern = chars.get("win_pattern", "")
        if not isinstance(win_pattern, str) or not win_pattern:
            return

        pattern_lc = win_pattern.lower()
        if any(kw in pattern_lc for kw in _REACH_KEYWORDS):
            self._maybe_emit_reach_goal(ws, step, chars)
        # Other strategies land in future pieces.

    # ------------------------------------------------------------------
    # Lifecycle sweep — success / churn / timeout
    # ------------------------------------------------------------------

    def _sweep_tracked_goals(
        self,
        ws:      WorldState,
        step:    int,
        adapter: "Adapter",
    ) -> None:
        """Resolve each tracked goal's fate.

        * **Success** (ACHIEVED or condition already True): drop the
          tracking entry; piece 1's level-transition path handles
          persistence.
        * **Churn** (target entity vanished): abandon the goal, drop
          tracking — do NOT decay, since this is segmentation
          instability, not a pattern-mismatch signal.
        * **Timeout** (step - emit_step > budget, condition still
          False): abandon the goal and contradict the underlying
          ``win_pattern`` hypothesis once.  Re-emission in the same
          tick picks up a fresh target if one exists.
        * **Unreachable** (after the retarget-grace window, the BFS
          planner cannot find a plan to the target): abandon the
          goal, add the target to ``_avoid_targets`` so re-emission
          picks a different candidate.  No credence decay — this is
          a target-choice failure, not a pattern-prediction failure.

        Nothing happens to unexpired, unresolved goals — they keep
        running.
        """
        for goal_id, emit_step in list(self._emit_step.items()):
            goal = ws.goal_forest.goals.get(goal_id)
            if goal is None:
                # Gone from forest (externally pruned / reset).
                self._emit_step.pop(goal_id, None)
                continue

            target_id = goal_id[len(f"{DECOMPOSE_GOAL_PREFIX}reach::"):]
            status = goal.root.status
            if status == GoalStatus.ACHIEVED:
                self._emit_step.pop(goal_id, None)
                continue

            # Success via direct condition check — the forest may lag
            # the engine's own verdict by a step.
            cond = goal.root.condition
            if cond is not None:
                try:
                    verdict = cond.evaluate(ws)
                except Exception:  # pragma: no cover — defensive
                    verdict = None
                if verdict is True:
                    self._emit_step.pop(goal_id, None)
                    continue

            # Churn: target vanished.  Abandon, don't decay.
            if target_id not in ws.entities:
                goal.root.status = GoalStatus.ABANDONED
                self._emit_step.pop(goal_id, None)
                continue

            # Timeout: budget exhausted, still unreached.
            if step - emit_step > self.reach_step_budget:
                goal.root.status = GoalStatus.ABANDONED
                self._emit_step.pop(goal_id, None)
                _contradict_win_pattern(ws, step)
                _LOG.info(
                    "GameDecomposer: reach goal %s timed out after %d steps; "
                    "contradicting win_pattern claim.",
                    goal_id, step - emit_step,
                )
                continue

            # Unreachable: planner can't find a path.  Three
            # preconditions before we retarget:
            #   1. adapter present (unit tests may pass None);
            #   2. age past the grace window (motion models take a
            #      handful of probe steps to commit);
            #   3. full planner substrate — agent position known AND
            #      every action in the action space has a committed
            #      motion model.  A partial MM set genuinely blocks
            #      BFS paths that a complete set would find, so
            #      "unplannable now" isn't yet informative.
            age = step - emit_step
            if (
                adapter is not None
                and age >= self.retarget_grace
                and _planner_has_substrate(ws, adapter)
                and _goal_unplannable(ws, goal_id, adapter, step)
            ):
                goal.root.status = GoalStatus.ABANDONED
                self._emit_step.pop(goal_id, None)
                self._avoid_targets.add(target_id)
                _LOG.info(
                    "GameDecomposer: reach goal %s unreachable from current "
                    "pos after %d steps; avoiding target %s.",
                    goal_id, age, target_id,
                )
                continue

    # ------------------------------------------------------------------
    # Reach strategy
    # ------------------------------------------------------------------

    def _maybe_emit_reach_goal(
        self,
        ws:    WorldState,
        step:  int,
        chars: Dict[str, Any],
    ) -> None:
        """Pick the most visually-distinctive candidate and emit a
        reach subgoal toward it, unless a live one already exists."""
        # First: is a decomposed reach goal already present, live, and
        # pointing at a live entity?  If so, leave it alone.
        existing_target = _existing_live_reach_target(ws)
        if existing_target is not None and existing_target in ws.entities:
            return

        agent_colour = _agent_colour(ws)
        target_id = _pick_reach_target(
            ws,
            agent_colour   = agent_colour,
            avoid_targets  = self._avoid_targets,
        )
        if target_id is None:
            return

        goal_id = f"{DECOMPOSE_GOAL_PREFIX}reach::{target_id}"
        if goal_id in ws.goal_forest.goals:
            return

        condition = InsideBBox(probe_id="agent", entity_id=target_id)
        root = GoalNode(
            id         = f"{goal_id}::root",
            node_type  = NodeType.ATOM,
            condition  = condition,
            priority   = self.priority,
            source     = "adapter:game_decomposer",
            created_at = step,
        )
        goal = Goal(
            id         = goal_id,
            root       = root,
            priority   = self.priority,
            source     = "adapter:game_decomposer",
            created_at = step,
        )
        add_goal(ws, goal)
        self._emit_step[goal_id] = step


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------


def _planner_has_substrate(ws: WorldState, adapter: "Adapter") -> bool:
    """Guard: BFS is only meaningful once the agent has a known
    position AND every action in the action space has a committed
    motion model.  With a partial MM set, BFS misses entire
    movement directions and will falsely declare reachable targets
    unreachable — retarget-churning through every candidate during
    the probe phase.
    """
    if ws.agent.get("position") is None:
        return False
    from cognitive_os.claims import MotionModelClaim
    from cognitive_os import hypothesis_store as _store
    try:
        needed = {a.id for a in adapter.action_space()}
    except Exception:  # pragma: no cover — defensive
        return False
    if not needed:
        return False
    committed_ids = {
        h.claim.action_id
        for h in _store.committed(ws)
        if isinstance(h.claim, MotionModelClaim)
    }
    return needed.issubset(committed_ids)


def _goal_unplannable(
    ws:       WorldState,
    goal_id:  str,
    adapter:  "Adapter",
    step:     int,
) -> bool:
    """True when the planner cannot find a plan to ``goal_id`` from
    the current world state.

    Lazy-imports :mod:`cognitive_os.planner` and swallows any
    planner exceptions as "not unplannable" — the decomposer is a
    best-effort signal, and we'd rather let a goal run its normal
    timeout than treat a planner bug as a mandate to retarget.
    """
    try:
        from cognitive_os.planner import compute_plan
        plan = compute_plan(ws, goal_id, adapter.action_space(), step=step)
    except Exception:  # pragma: no cover — defensive
        return False
    return plan is None


def _contradict_win_pattern(ws: WorldState, step: int) -> None:
    """Apply one contradiction to every ``_game`` ``win_pattern``
    PropertyClaim hypothesis in the store.

    We decay specifically the field we dispatched on — ``win_pattern``
    — rather than the whole characterization.  Other fields (``genre``,
    ``mechanics``) remain at their observed credence; only the part
    that predicted "reach X" loses standing on one timeout.  If
    subsequent ticks also time out, repeated contradictions compound
    via :func:`update_on_contradict` and eventually push the claim
    below the commit threshold — at which point ``_read_game_claims``
    stops seeing it and the decomposer naturally stands down.
    """
    cfg = _credence_cfg(ws)
    for h in ws.hypotheses.values():
        claim = h.claim
        if not isinstance(claim, PropertyClaim):
            continue
        if claim.entity_id != GAME_ENTITY_ID:
            continue
        if claim.property != "win_pattern":
            continue
        h.credence = update_on_contradict(h.credence, step, cfg, strength=1.0)
        h.contradicting_steps.append(step)


def _credence_cfg(ws: WorldState):
    """Read the credence-update config off the live engine config, or
    fall back to defaults if none is attached."""
    cfg = getattr(ws, "config", None)
    if cfg is not None and hasattr(cfg, "credence"):
        return cfg.credence
    from cognitive_os.config import CredenceConfig
    return CredenceConfig()


def _read_game_claims(ws: WorldState) -> Dict[str, Any]:
    """Return the latest-observed ``_game`` PropertyClaim values.

    Multiple claims on the same property (say, successive levels
    each emit a ``genre``) are resolved by recency — the one with
    the highest ``last_confirmed`` wins.  This is the cheapest
    sensible disambiguation and matches what the store does: the
    prior for the level we're *on* is the only one that can have
    been committed at the current step.
    """
    result: Dict[str, Any] = {}
    latest: Dict[str, int] = {}
    for h in ws.hypotheses.values():
        claim = h.claim
        if not isinstance(claim, PropertyClaim):
            continue
        if claim.entity_id != GAME_ENTITY_ID:
            continue
        confirmed = getattr(h.credence, "last_confirmed", 0) or 0
        prev = latest.get(claim.property, -1)
        if confirmed >= prev:
            latest[claim.property] = confirmed
            result[claim.property] = claim.value
    return result


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def _existing_live_reach_target(ws: WorldState) -> Optional[str]:
    """Return the target_id of any existing decompose reach goal
    whose status is not terminal (ACHIEVED / PRUNED / ABANDONED),
    or None if none exists.  Abandoned goals don't block new
    emissions — that's how retargeting works."""
    prefix = f"{DECOMPOSE_GOAL_PREFIX}reach::"
    for gid, goal in ws.goal_forest.goals.items():
        if not gid.startswith(prefix):
            continue
        status = goal.root.status
        if status in (GoalStatus.ACHIEVED, GoalStatus.PRUNED, GoalStatus.ABANDONED):
            continue
        return gid[len(prefix):]
    return None


def _agent_colour(ws: WorldState) -> Optional[int]:
    """Extract the controlled-actor colour signature if committed."""
    from cognitive_os.claims import ControlledActorClaim
    for h in ws.hypotheses.values():
        if not isinstance(h.claim, ControlledActorClaim):
            continue
        if getattr(h.credence, "point", 0.0) < 0.5:
            continue
        try:
            return int(h.claim.colour)
        except (TypeError, ValueError):
            return None
    return None


def _pick_reach_target(
    ws:             WorldState,
    *,
    agent_colour:   Optional[int],
    avoid_targets:  Optional[Set[str]] = None,
) -> Optional[str]:
    """Choose the most visually-distinctive non-agent, non-wall entity.

    Heuristic:
      1. Exclude the agent entity (colour matches ``agent_colour``).
      2. Exclude likely walls — entities whose bbox spans a full
         frame edge AND whose area exceeds ``WALL_AREA_THRESHOLD``.
      3. Among the remainder, prefer entities whose colour is unique
         (only one entity has that colour).  Break ties by smallest
         area — targets are typically single distinct tiles.
      4. If no unique-colour entity exists, fall back to the
         smallest-area candidate.

    Returns the entity_id, or None if no candidates survived.
    """
    # Frame dimensions — best-effort from the latest observation.
    frame_h, frame_w = _frame_dims(ws)

    avoid = avoid_targets or set()
    candidates: List[Tuple[str, int, int, Any]] = []
    # (entity_id, area, colour, bbox)
    for eid, ent in ws.entities.items():
        if eid in avoid:
            continue
        props = ent.properties
        colour = props.get("colour")
        if agent_colour is not None and colour == agent_colour:
            continue
        bbox = props.get("bbox")
        area = props.get("area")
        if bbox is None or area is None:
            continue
        if _looks_like_wall(bbox, area, frame_h, frame_w):
            continue
        try:
            area_i = int(area)
        except (TypeError, ValueError):
            continue
        candidates.append((eid, area_i, colour, bbox))

    if not candidates:
        return None

    # Colour uniqueness: count how many candidates share each colour.
    colour_counts: Dict[Any, int] = {}
    for _eid, _a, colour, _b in candidates:
        colour_counts[colour] = colour_counts.get(colour, 0) + 1

    unique = [c for c in candidates if colour_counts.get(c[2], 0) == 1]
    pool = unique if unique else candidates
    # Smallest-area wins; ties broken by entity id for determinism.
    pool.sort(key=lambda c: (c[1], c[0]))
    return pool[0][0]


WALL_AREA_THRESHOLD = 80  # pixels; heuristic, tuned for 64×64 ARC frames


def _looks_like_wall(
    bbox:     Any,
    area:     Any,
    frame_h:  Optional[int],
    frame_w:  Optional[int],
) -> bool:
    """A bbox that spans a full frame edge AND exceeds the area
    threshold is probably a wall.  Conservative: if frame dims are
    unknown we never classify anything as a wall (accept the risk of
    aiming at a wall rather than aiming at nothing)."""
    try:
        r0, c0, r1, c1 = (int(x) for x in bbox)
        area_i = int(area)
    except (TypeError, ValueError):
        return False
    if area_i < WALL_AREA_THRESHOLD:
        return False
    if frame_h is None or frame_w is None:
        return False
    spans_row = (r0 == 0 and r1 >= frame_h - 1)
    spans_col = (c0 == 0 and c1 >= frame_w - 1)
    return spans_row or spans_col


def _frame_dims(ws: WorldState) -> Tuple[Optional[int], Optional[int]]:
    """Read (height, width) from the latest observation if possible."""
    if not ws.observation_history:
        return (None, None)
    raw = getattr(ws.observation_history[-1], "raw_frame", None)
    if not raw:
        return (None, None)
    try:
        h = len(raw)
        w = len(raw[0]) if h > 0 else 0
    except (TypeError, IndexError):
        return (None, None)
    return (h, w)
