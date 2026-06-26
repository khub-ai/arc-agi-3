"""Explorer — fallback action selection when the planner has no plan.

When :func:`planner.compute_plan` returns ``None`` — because the active
goal has no known path forward given current hypotheses — the runner
hands control to the explorer.  The explorer's job is to produce a
*useful* next action: one that either reduces epistemic uncertainty
(info-gain) or probes an entity the engine knows nothing about
(curiosity).

Design
------
Two drivers, blended by the tunable
:attr:`ExplorerConfig.curiosity_level`:

1. **Information-gain** — prefer actions that would differentiate
   between currently *competing* hypotheses (those sharing a
   canonical key but diverging on full keys).  Each such group
   represents an unresolved question in the agent's model; an action
   that would cause different competitors to predict different
   outcomes has positive info-gain.

2. **Curiosity** — prefer actions that involve entities or action
   types the engine knows little about.  Each entity has a
   :func:`claim_coverage` in ``[0, 1]`` counting how many of the
   standard claim slots (property / relational / causal / transition /
   structure-mapping) are populated for it.  Low coverage → high
   curiosity draw.

The explorer also *generates goals*: it proposes
:class:`Goal`\\s of the form "probe entity X" or "try action A" that
the runner can feed into the :class:`GoalForest` at LOW priority.
This lets curiosity integrate with the normal goal / planner loop
rather than existing as a parallel system.

Capability audit (standing invariant 7)
----------------------------------------
* **Debugging** — PRIMARY.  Info-gain exploration is the deliberate
  design of an experiment that would disambiguate competing
  hypotheses.  This is the "form hypothesis → test → iterate" loop
  made explicit at the action-selection level.
* **Problem-solving** — secondary.  Curiosity-generated goals
  augment the goal forest with unknowns the primary goal doesn't
  cover, which often surfaces the preconditions the main goal
  needs.
* **Tool creation** — minor.  ``detect_generalization_candidates``
  from the hypothesis store drives info-gain priorities when many
  related hypotheses share a pattern; resolving the dispute is a
  prerequisite for eventually compressing them into a tool.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from .claims import (
    ActorTransitionClaim,
    CausalClaim,
    Claim,
    MotionModelClaim,
    PropertyClaim,
    RelationalClaim,
    StructureMappingClaim,
    TransitionClaim,
)
from .conditions import (
    Condition,
    ActionTried,
    EntityProbed,
)
from . import hypothesis_store as _store
from .types import (
    Action,
    EntityModel,
    Goal,
    GoalNode,
    GoalStatus,
    NodeType,
    WorldState,
)


# ---------------------------------------------------------------------------
# Claim-coverage metric
# ---------------------------------------------------------------------------


# Standard slot types an entity "could" have hypotheses about.  Coverage
# is the fraction of these slots with at least one hypothesis (of any
# credence) mentioning the entity.  Used for curiosity ranking.
_STANDARD_SLOTS = ("property", "relational", "causal", "transition", "structure")


def claim_coverage(entity_id: str, ws: WorldState) -> float:
    """Return fraction of :data:`_STANDARD_SLOTS` covered by at least
    one hypothesis referencing the entity.

    The metric deliberately does not require *committed* hypotheses —
    even a provisional claim means the entity has some structure the
    engine has noticed.  Only fully unknown entities score 0.0.
    """
    filled: Set[str] = set()
    for h in ws.hypotheses.values():
        claim = h.claim
        if entity_id not in claim.referenced_entities():
            continue
        if isinstance(claim, PropertyClaim):
            filled.add("property")
        elif isinstance(claim, RelationalClaim):
            filled.add("relational")
        elif isinstance(claim, CausalClaim):
            filled.add("causal")
        elif isinstance(claim, TransitionClaim):
            filled.add("transition")
        elif isinstance(claim, StructureMappingClaim):
            filled.add("structure")
    return len(filled) / len(_STANDARD_SLOTS)


# ---------------------------------------------------------------------------
# Info-gain estimate for a candidate action
# ---------------------------------------------------------------------------


def info_gain(action: Action,
              ws:     WorldState) -> float:
    """Estimate how much uncertainty the action would resolve.

    Simple model: count the number of distinct canonical_keys among
    contested hypothesis groups where the action appears in a
    :class:`TransitionClaim` or in an entity referenced by the group.
    Each such group contributes ``1.0 / n_competitors`` — more
    competitors means more information per disambiguating outcome.

    This is a coarse but tractable proxy; Phase 4 may refine it with a
    KL-divergence estimate once transition predictions from committed
    hypotheses accumulate enough.
    """
    gain = 0.0
    groups = _store.contested_groups(ws)
    for group in groups:
        canonical = group[0].claim.canonical_key()
        # Touched if the action appears in any of the group's claims
        touches = False
        for h in group:
            claim = h.claim
            if isinstance(claim, TransitionClaim) and claim.action in (action.name, "*"):
                touches = True
                break
        if touches:
            gain += 1.0 / max(1, len(group))
    return gain


# ---------------------------------------------------------------------------
# Motion-model completeness info-gain
# ---------------------------------------------------------------------------


def motion_model_info_gain(action: Action, ws: WorldState) -> float:
    """Return how much uncertainty about *action*'s motion model
    remains, in ``[0.0, 1.0]``.

    The engine cannot plan positional goals (``AtPosition``) via the
    planner's BFS unless it has a :class:`MotionModelClaim` telling it
    the per-step delta for each action.  Until every action in the
    action space has either (a) a committed motion model or (b) strong
    evidence that it produces no motion, positional planning is
    crippled.  This function scores how much an action is still
    *owed* exploration for that purpose.

    Scoring
    -------
    - A committed :class:`MotionModelClaim` exists for the action →
      ``0.0`` (fully characterised; nothing more to learn here).
    - No :class:`ActorTransitionClaim` observations at all →
      ``1.0`` (pure unknown; action has never been tried in a way
      that let the miner see a pre/post delta).
    - Any *non-zero* ActorTransitionClaim delta exists but no
      MotionModelClaim has committed yet → ``max(0, 0.5 - 0.1 * Z)``
      where ``Z`` is the zero-delta observation count.  A fresh
      nonzero observation starts at ``0.5`` (another trial may push
      the miner over its dominance threshold), but accumulating
      zero-delta evidence indicates the motion is position-dependent
      and will not yield a global :class:`MotionModelClaim` — the
      per-state :class:`ActorTransitionClaim` records capture it
      directly, and further babbling has diminishing returns.
    - Only zero-delta observations → ``max(0, 1.0 - 0.25 * Z)``.
      The action is probably a no-op or a wall-interaction in every
      context seen so far; further babbling has rapidly diminishing
      value.  After ~4 zero-delta trials the term is ``0.0`` and
      the explorer moves on.

    Robotics analogue
    -----------------
    For a robot this is **motor babbling** info-gain: the priority
    placed on exercising a motor primitive whose kinematic effect on
    the end-effector pose is still unknown.  A freshly instantiated
    manipulation stack with a new gripper or a new joint limit has
    ``1.0`` on every primitive; as the kinematic identification
    procedure (the robotics equivalent of
    :class:`MotionModelMiner`) commits per-primitive displacement
    models, the term decays to ``0.0`` and the planner takes over.
    Priming every motor primitive with at least one committed model
    is a baseline investment — without it no pose-reaching plan is
    feasible — which is why the associated weight
    (:attr:`ExplorerConfig.motion_model_ig_weight`) stays substantial
    even at low curiosity.
    """
    # Committed motion model → nothing more to learn.
    for h in _store.committed(ws):
        claim = h.claim
        if isinstance(claim, MotionModelClaim) and claim.action_id == action.name:
            return 0.0

    nonzero_obs = 0
    zero_obs    = 0
    for h in ws.hypotheses.values():
        claim = h.claim
        if not isinstance(claim, ActorTransitionClaim):
            continue
        if claim.action_id != action.name:
            continue
        # Count by supporting_steps so that repeatedly firing the
        # same action from the same pre_state (which dedups to one
        # hypothesis) accumulates evidence instead of stalling at
        # one observation.
        n_obs = max(1, len(getattr(h, "supporting_steps", []) or [1]))
        dx, dy = claim.delta
        if dx == 0 and dy == 0:
            zero_obs += n_obs
        else:
            nonzero_obs += n_obs

    if nonzero_obs == 0 and zero_obs == 0:
        return 1.0
    if nonzero_obs > 0:
        # Nonzero motion exists but global motion-model not yet
        # committed.  Start at 0.5; accumulating zero-delta evidence
        # indicates position-dependence that global MotionModelClaim
        # cannot capture — decay to 0 rather than stall.
        return max(0.0, 0.5 - 0.1 * zero_obs)
    # Only zero-delta observations: decay.
    return max(0.0, 1.0 - 0.25 * zero_obs)


# ---------------------------------------------------------------------------
# Curiosity: generate exploration goals
# ---------------------------------------------------------------------------


def propose_curiosity_goals(ws: WorldState,
                            *,
                            step: int = 0,
                            action_space: Optional[List[Action]] = None) -> List[Goal]:
    """Produce low-priority :class:`Goal`\\s targeting unknowns.

    Two kinds:

    * **Entity-probing** — for every known entity whose
      :func:`claim_coverage` is below ``curiosity_threshold``,
      generate a Goal whose ATOM condition is
      :class:`EntityProbed`\\ ``(entity_id)``.
    * **Action-trial** — for every action in ``action_space`` that
      does not appear in any committed TransitionClaim, generate a
      Goal whose ATOM condition is
      :class:`ActionTried`\\ ``(action_id)``.

    Both kinds are returned regardless of the explorer's master switch;
    the caller (runner) decides whether to add them based on
    :attr:`ExplorerConfig.generate_curiosity_goals`.

    Parameters
    ----------
    step
        Used as ``created_at`` on newly minted Goals.
    action_space
        Needed to generate action-trial goals.  If omitted, only
        entity-probing goals are produced.
    """
    cfg = _explorer_cfg(ws)
    if not cfg.generate_curiosity_goals:
        return []

    goals: List[Goal] = []
    threshold = cfg.curiosity_threshold
    base_prio = cfg.novelty_base

    # Entity probing
    for entity_id, entity in ws.entities.items():
        cov = claim_coverage(entity_id, ws)
        if cov >= threshold:
            continue
        priority = base_prio * (1.0 - cov)
        goal_id = f"explore:entity:{entity_id}"
        if goal_id in ws.goal_forest.goals:
            continue  # already proposed
        goals.append(Goal(
            id         = goal_id,
            root       = GoalNode(
                id         = f"{goal_id}::atom",
                node_type  = NodeType.ATOM,
                condition  = EntityProbed(entity_id, coverage=threshold),
                status     = GoalStatus.OPEN,
                source     = "explorer:curiosity",
                created_at = step,
            ),
            priority   = priority,
            source     = "explorer:curiosity",
            created_at = step,
        ))

    # Action trials
    if action_space:
        tried_actions = _actions_in_transitions(ws)
        for action in action_space:
            if action.name in tried_actions:
                continue
            goal_id = f"explore:action:{action.id}"
            if goal_id in ws.goal_forest.goals:
                continue
            goals.append(Goal(
                id         = goal_id,
                root       = GoalNode(
                    id         = f"{goal_id}::atom",
                    node_type  = NodeType.ATOM,
                    condition  = ActionTried(action.id),
                    status     = GoalStatus.OPEN,
                    source     = "explorer:curiosity",
                    created_at = step,
                ),
                priority   = base_prio * 0.5,
                source     = "explorer:curiosity",
                created_at = step,
            ))

    return goals


def _actions_in_transitions(ws: WorldState) -> Set[str]:
    """Action names that appear in any TransitionClaim hypothesis,
    regardless of credence."""
    names: Set[str] = set()
    for h in ws.hypotheses.values():
        if isinstance(h.claim, TransitionClaim):
            names.add(h.claim.action)
    return names


# ---------------------------------------------------------------------------
# Master: choose an exploration action given no plan
# ---------------------------------------------------------------------------


def choose_exploration_action(ws:           WorldState,
                              action_space: List[Action]) -> Optional[Action]:
    """Pick an action when the planner cannot produce a plan.

    Scoring per action:

        score(a) = info_gain_weight       * info_gain(a, ws)
                 + novelty_base           * curiosity_bonus(a, ws)
                 + motion_model_ig_weight * motion_model_info_gain(a, ws)

    where ``curiosity_bonus`` equals
    ``1 - claim_coverage(most_relevant_entity)`` — relevant entity is
    the one the action most plausibly affects (currently a proxy:
    the agent itself).  The motion-model term is dominant while any
    action lacks a committed :class:`MotionModelClaim`, which is the
    baseline investment required before positional planning becomes
    feasible at all (see
    :func:`motion_model_info_gain`).  Robotics analogue: motor
    babbling dominates action selection until every motor primitive
    has a committed kinematic model.

    When multiple actions tie, prefer actions whose name has never
    been seen in a TransitionClaim, then lexicographic name order for
    determinism.  Returns ``None`` if ``action_space`` is empty.

    GAP 15 — wall-override filter.  Before scoring, drop any action
    for which :func:`hypothesis_store.active_wall_overrides` records
    a zero-delta outcome at the agent's current position.  The
    scoring heuristics above are position-blind, so an unfiltered
    explorer will re-pick the same top-scoring action even when the
    agent just observed that action produce no effect — the exact
    shape that kept the live ls20 run wedged at ``(48,21)`` for 32
    consecutive steps.  If every action is blocked at the current
    position (rare), the filter falls back to the full action space
    rather than returning ``None`` — "something, even if futile" is
    more useful than silent lockup, and the wedge-escape scaffolding
    in higher layers can still make progress via replanning or
    Mediator intervention.  Robotics analogue: motor babbling must
    not repeatedly invoke a motion the arm has just shown produces
    no end-effector displacement from the current pose.
    """
    if not action_space:
        return None

    cfg = _explorer_cfg(ws)
    tried = _actions_in_transitions(ws)
    agent_coverage = claim_coverage("agent", ws)

    # GAP 15: skip wall-blocked actions at the current position.
    effective_space = _filter_wall_blocked(ws, action_space)

    # GAP 16: recency penalty.  Count how many times each action
    # has been executed from the agent's *current* position within
    # the tabu window.  ``tabu_counts[action_name] = k`` means
    # ``(cur_pos, action_name)`` appears ``k`` times in the
    # rolling window; we subtract ``tabu_penalty * k`` from that
    # action's score so that breaking a cycle becomes the
    # explorer's preferred move.
    tabu_counts = _tabu_counts_at_current_position(ws)
    tabu_penalty = float(getattr(cfg, "tabu_penalty", 1.0))

    # Tiebreaker chain ends in ``action.id`` (a unique, orderable string)
    # rather than the raw ``Action``.  ``Action`` is a frozen dataclass
    # with no ordering, so for action spaces where several actions share
    # a ``name`` — parameterised skills such as ``tap(red)`` / ``tap(green)``
    # whose ``name`` is just ``"tap"`` — a tie on (score, untried, name)
    # would otherwise fall through to comparing two ``Action`` objects and
    # raise ``TypeError``.  For action spaces with unique names (e.g. ARC's
    # ``MOVE_UP`` …) ``name`` alone already breaks every tie, so ``id`` is
    # never the deciding key and behaviour is unchanged.
    scored: List[Tuple[float, int, str, str]] = []
    for action in effective_space:
        ig         = info_gain(action, ws)
        curiosity  = (1.0 - agent_coverage) * 0.5
        mm_ig      = motion_model_info_gain(action, ws)
        tabu       = tabu_counts.get(action.name, 0)
        score = (cfg.info_gain_weight       * ig
                 + cfg.novelty_base         * curiosity
                 + cfg.motion_model_ig_weight * mm_ig
                 - tabu_penalty * float(tabu))
        untried_bonus = 0 if action.name in tried else 1  # lower primary key wins
        scored.append((-score, -untried_bonus, action.name, action.id))

    scored.sort()
    winning_id = scored[0][3]
    return next(a for a in effective_space if a.id == winning_id)


def _tabu_counts_at_current_position(ws: WorldState) -> Dict[str, int]:
    """Count occurrences of each action in the tabu window that
    were executed from the agent's *current* position.

    Reads ``ws.agent['_recent_pos_actions']`` (maintained by
    :mod:`episode_runner`) and ``ws.agent['position']``.  Returns
    an empty dict if either is missing — the explorer then scores
    without the recency term, which matches pre-GAP-16 behaviour.

    The window length is controlled by
    :attr:`ExplorerConfig.tabu_window`; the runner trims the list
    to that length, so we simply count matches in the whole list.
    Keeping the trimming in one place (the runner) means the
    explorer sees a consistently-sized window regardless of who
    added entries (planner execution or prior exploration calls).

    Robotics analogue: a short-term motor memory queried at the
    moment of choosing the next motion primitive.
    """
    cur_pos = ws.agent.get("position")
    if cur_pos is None:
        return {}
    try:
        cur = tuple(cur_pos)
    except (TypeError, IndexError):
        return {}
    recent = ws.agent.get("_recent_pos_actions") or []
    counts: Dict[str, int] = {}
    for entry in recent:
        try:
            pos, aname = entry
        except (TypeError, ValueError):
            continue
        if pos is None:
            continue
        try:
            pos_t = tuple(pos)
        except (TypeError, IndexError):
            continue
        if pos_t != cur:
            continue
        counts[aname] = counts.get(aname, 0) + 1
    return counts


def _filter_wall_blocked(ws: WorldState,
                         action_space: List[Action]) -> List[Action]:
    """Return the subset of ``action_space`` that is NOT wall-blocked
    at the agent's current position.  If every action is blocked, or
    the agent has no recorded position, returns the full action space
    unchanged — a degenerate filter is worse than no filter at all.

    Kept module-private; used only by :func:`choose_exploration_action`.
    The wall set itself (GAP 14 soft-floor rule) lives in
    :func:`hypothesis_store.active_wall_overrides`.
    """
    cur_pos = ws.agent.get("position")
    if cur_pos is None:
        return action_space
    try:
        cur = (cur_pos[0], cur_pos[1])
    except (TypeError, IndexError):
        return action_space
    walls = _store.active_wall_overrides(ws)
    if not walls:
        return action_space
    blocked_here: set = set()
    for (pre, aid) in walls:
        if tuple(pre) == tuple(cur):
            blocked_here.add(aid)
    if not blocked_here:
        return action_space
    filtered = [a for a in action_space
                if a.id not in blocked_here and a.name not in blocked_here]
    return filtered if filtered else action_space


# ---------------------------------------------------------------------------
# Cell-level exploration target picker (navigation domains)
# ---------------------------------------------------------------------------


def choose_exploration_target_cell(
    ws:                "WorldState",
    *,
    agent_pos:         "tuple[int, int]",
    reachable_cells:   "Iterable[tuple[int, int]]",
    dist_map:          "Dict[tuple[int, int], int]",
    current_budget:    "int",
    visited_cells:     "Optional[Set[tuple[int, int]]]" = None,
) -> "Optional[tuple[int, int]]":
    """Pick a cell to move toward when no plan / no GF directive exists.

    Cell-level analog of :func:`choose_exploration_action` for
    navigation domains. Per
    :doc:`SPEC_per_turn_dispatch <../docs/SPEC_per_turn_dispatch>`
    §4, this is the always-available fallback when the plan stack
    is empty AND `_select_gf_target` returns None — Oracle is NOT
    consulted in that situation.

    Selection rules (in order):

    1. **Round-trip safety.** Only cells `c` where
       `2 * dist_map[c] <= current_budget` are eligible. Without
       known refuels the agent's only safe position is
       `agent_pos`, so any commitment must preserve retreat budget
       (same principle as the resource-aware selector's cold-start
       branch).
    2. **Prefer unvisited.** When `visited_cells` is supplied,
       cells absent from it outrank cells present in it.
    3. **Furthest first.** Among the surviving cells, pick the one
       maximising `dist_map[c]`. Going further per turn maximises
       observational coverage per env-step spent.
    4. **Lexicographic tiebreak.** When multiple cells tie on the
       above keys, pick the lexicographically smallest cell for
       deterministic test reproducibility.

    Returns ``None`` when no cell satisfies the round-trip rule
    (e.g. `current_budget == 0` or every reachable cell exceeds
    budget/2). The caller should treat this as the
    `abandoned:no_action` signal — the agent genuinely cannot move
    safely. Oracle is not consulted on `None`; the runner ends the
    trial with a diagnostic outcome.

    Parameters
    ----------
    ws
        Reserved for future info-gain refinements (e.g. weighting
        cells by `claim_coverage` of their resident entities). The
        minimum viable implementation does not read it; future
        versions will, hence the parameter is required up front to
        avoid signature churn.
    agent_pos
        The agent's current cell. Used to skip the agent's own cell
        when it appears in `reachable_cells` (a self-target is not
        useful exploration).
    reachable_cells
        The cells reachable from `agent_pos` on the current map
        (typically the keys of `dist_map`).
    dist_map
        ``cell -> integer step count`` from `agent_pos`. Caller
        supplies this from BFS.
    current_budget
        Steps the agent has remaining on the current life. The
        round-trip check uses this; the caller is responsible for
        providing the per-life value (not full capacity).
    visited_cells
        Optional set of cells the agent has visited recently. When
        supplied, unvisited cells are preferred for exploration.
        ``None`` (default) skips the visited tiebreaker — equivalent
        to "every cell looks fresh."

    Domain-agnostic. Works for any navigation domain where actions
    are issued as `MOVE_TO target_cell`: ARC, robotics with a
    discrete cell map, gridworld puzzles, multi-room delivery.
    """
    if current_budget <= 0:
        return None
    half_budget = current_budget // 2
    visited = visited_cells or frozenset()

    candidates: List[Tuple[int, int, Tuple[int, int]]] = []
    for cell in reachable_cells:
        if cell == agent_pos:
            continue
        d = dist_map.get(cell)
        if d is None or d <= 0:
            continue
        if d > half_budget:
            continue
        # Score key: (visited_penalty, -distance, cell) — smaller
        # is better. Visited cells get penalty=1; unvisited get 0.
        # Negative distance so larger distance wins. Cell as final
        # tiebreaker for determinism.
        was_visited = 1 if cell in visited else 0
        candidates.append((was_visited, -d, cell))

    if not candidates:
        return None
    candidates.sort()
    return candidates[0][2]


# ---------------------------------------------------------------------------
# Config access
# ---------------------------------------------------------------------------


def _explorer_cfg(ws: WorldState):
    if ws.config is not None and hasattr(ws.config, "explorer"):
        return ws.config.explorer
    from .config import ExplorerConfig
    return ExplorerConfig()
