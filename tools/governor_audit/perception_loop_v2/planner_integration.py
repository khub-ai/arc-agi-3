"""Bridge: WorldKnowledge -> cognitive_os planner + explorer.

The exploratory driver had a small mechanical actor (cell_actor BFS)
plus a per-turn VLM strategy layer.  Neither used the project's
EXISTING curiosity-driven exploration + AO* planner machinery in
``cognitive_os/``, even though that machinery was specifically
designed to interleave probing and exploitation smoothly.

This module bridges the gap.  It:

  1. Translates our per-level ``WorldKnowledge`` into a minimal
     ``cognitive_os.WorldState`` (agent, entities, goal_forest,
     hypotheses, config).

  2. Seeds the WorldState with PRIMARY goals derived from the
     world's ``goal_candidate_cells`` (translated to AtPosition
     atoms on the agent), and lets ``explorer.propose_curiosity_goals``
     add EXPLORATORY ``ActionTried`` / ``EntityProbed`` goals for
     anything still unknown.

  3. Calls ``planner.select_and_plan`` to get a Plan whose first
     step is the next action.  The planner prefers the
     highest-priority PLANNABLE goal: when a primary goal is
     reachable from learned TransitionClaims, that wins; when it
     isn't, the curiosity goals (which are always plannable as
     "do action X once") become the next choice -- a smooth
     credence-based transition between exploration and play, with
     no explicit "probe phase".

  4. Exposes the resulting Plan's first action as a string the
     LiveHarnessAdapter accepts (or "NONE" when no plan exists).

PRIME DIRECTIVE: no game-specific information anywhere.  Action
ids and names come from the adapter's available_actions list.
Entity roles come from the WorldKnowledge's open vocabulary.  No
hardcoded action semantics, no game-specific goal templates.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# cognitive_os lives at repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

sys.path.insert(0, str(Path(__file__).parent))
from world_knowledge import WorldKnowledge       # noqa: E402

from cognitive_os.types import (                   # noqa: E402
    Action, EntityModel, Goal, GoalNode, GoalStatus, NodeType,
    Plan, PlannedAction, WorldState, GoalForest, Scope, ScopeKind,
)
from cognitive_os.conditions import (              # noqa: E402
    ActionTried, AtPosition, AlwaysTrue,
)
from cognitive_os.claims import (                   # noqa: E402
    MotionModelClaim, TransitionClaim,
)
from cognitive_os.goal_forest import (             # noqa: E402
    add_goal, candidates_by_priority,
)
from cognitive_os import explorer as _explorer     # noqa: E402
from cognitive_os import planner as _planner       # noqa: E402
from cognitive_os import hypothesis_store as _hs    # noqa: E402
from cognitive_os.config import EngineConfig       # noqa: E402


# ---------------------------------------------------------------------------
# Bridge result
# ---------------------------------------------------------------------------


@dataclass
class PlannedActionChoice:
    """Result of running the planner on a WorldKnowledge.  The
    driver consumes ``action_string`` and uses ``goal_id`` /
    ``plan_kind`` for tracing."""
    action_string: str
    goal_id: Optional[str]
    plan_kind: str          # one of "primary" | "curiosity:action_trial" |
                            #        "curiosity:entity_probe" | "none"
    rationale: str
    full_plan_actions: list[str]  # for trace: the full plan, not just step 1


# ---------------------------------------------------------------------------
# Translation: WorldKnowledge -> WorldState
# ---------------------------------------------------------------------------


def _make_min_world_state(wk: WorldKnowledge) -> WorldState:
    """Construct a minimal cognitive_os.WorldState seeded from the
    per-level WorldKnowledge.  Includes the agent's current
    position and an EntityModel for each entity with a known
    bbox/cell; leaves the hypothesis store empty (the cognitive_os
    miner / claims pipeline isn't wired to our MechanicHypothesis
    yet -- that's a separate bridge for a later phase)."""
    agent_pos = None
    agent_rec = wk._find_agent()
    if agent_rec is not None and agent_rec.current_cell is not None:
        agent_pos = tuple(agent_rec.current_cell)

    entities: dict[str, EntityModel] = {}
    for name, rec in wk.entities.items():
        props: dict = {}
        if rec.current_bbox is not None:
            props["bbox"] = list(rec.current_bbox)
        if rec.current_cell is not None:
            props["position"] = tuple(rec.current_cell)
        if rec.current_role:
            props["role"] = rec.current_role
        if rec.appearance:
            props["appearance"] = rec.appearance
        entities[name] = EntityModel(
            id=name,
            first_seen_step=rec.first_seen_turn,
            last_seen_step=rec.last_seen_turn,
            properties=props,
        )

    ws = WorldState(
        step=wk.turn,
        agent={"position": agent_pos} if agent_pos else {},
        entities=entities,
        goal_forest=GoalForest(),
        config=EngineConfig(),
    )
    return ws


# ---------------------------------------------------------------------------
# Action-name <-> Action object translation
# ---------------------------------------------------------------------------


def _make_action(name: str) -> Action:
    """Lift an adapter action name (e.g. "ACTION3" / "UP" /
    "CLICK") into a cognitive_os Action object.  We keep the name
    canonical so the explorer can deduplicate by name."""
    return Action(id=name, name=name)


# ---------------------------------------------------------------------------
# Goal seeding
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# MechanicHypothesis -> cognitive_os Claim bridge
# ---------------------------------------------------------------------------


# Maps the miner's direction labels to (dr, dc) unit vectors.
# Game-agnostic: the miner produces these labels in
# `EffectSignature.direction` for any cardinal-move outcome.
_DIRECTION_UNITS: dict[str, tuple[int, int]] = {
    "UP":    (-1, 0),
    "DOWN":  (1, 0),
    "LEFT":  (0, -1),
    "RIGHT": (0, 1),
}


def _parse_action_from_trigger(trigger: str) -> Optional[str]:
    """Extract the action name from a MechanicHypothesis trigger
    string like "action=ACTION3, agent_role=..., adjacent_roles=...".
    Returns the action name or None if the trigger doesn't have one."""
    parts = [p.strip() for p in trigger.split(",")]
    for part in parts:
        if part.startswith("action="):
            return part[len("action="):]
    return None


def _parse_motion_direction(effect: str) -> Optional[str]:
    """Extract the cardinal direction from an effect string like
    "effect=agent_moved, direction=UP" or None if the effect isn't
    a directional move."""
    if "agent_moved" not in effect:
        return None
    parts = [p.strip() for p in effect.split(",")]
    for part in parts:
        if part.startswith("direction="):
            d = part[len("direction="):]
            return d if d in _DIRECTION_UNITS else None
    return None


def bridge_promoted_hypotheses(ws: WorldState, wk: WorldKnowledge,
                                  step: int) -> int:
    """Translate every promoted MechanicHypothesis in WorldKnowledge
    into cognitive_os Claims and propose them to the WorldState's
    hypothesis store.  Returns the number of claims proposed.

    Two kinds of bridge translations:

      1. ANY promoted hypothesis with a parseable action string ->
         a TransitionClaim(action=X, pre=AlwaysTrue,
         post=ActionTried(X)) so the explorer stops generating
         action-trial curiosity goals for that action.

      2. Promoted "agent_moved direction=X" hypotheses additionally
         become MotionModelClaim(action_id=A, delta=(dr,dc)) where
         the delta is the cardinal unit scaled by the world's
         inferred cell_ticks (default 1 when grid_inference has
         not committed a cell pitch).

    This is what lets the planner CHAIN moves to reach a primary
    goal: once two MotionModelClaims are committed (one per
    direction), the planner can BFS multi-step plans from the
    agent's current cell to any AtPosition goal in the connected
    component.  Game-agnostic: action names and direction labels
    flow through from the miner without any per-game
    interpretation.
    """
    n_proposed = 0
    cell_ticks = (wk.grid_inference.cell_ticks
                   if wk.grid_inference
                   and wk.grid_inference.cell_ticks else 1)

    for h in wk.mechanic_hypotheses:
        if not h.promoted:
            continue
        action_name = _parse_action_from_trigger(h.trigger)
        if action_name is None:
            continue

        # (1) Mark the action as "tried" via a TransitionClaim so
        # the explorer's action-trial check sees it.
        tc = TransitionClaim(
            action=action_name,
            pre=AlwaysTrue(),
            post=ActionTried(action_name),
        )
        try:
            _hs.propose(
                ws, tc,
                source="bridge:miner_promotion",
                scope=Scope(kind=ScopeKind.EPISODE),
                step=step,
                initial_credence=h.credence,
                rationale=(f"bridged from miner-promoted "
                            f"hypothesis {h.hypothesis_id}"),
            )
            n_proposed += 1
        except Exception as e:
            print(f"  [bridge] TransitionClaim propose failed: {e}")

        # (2) Motion hypotheses additionally become
        # MotionModelClaims the planner can plan over.
        direction = _parse_motion_direction(h.effect)
        if direction is not None:
            dr_unit, dc_unit = _DIRECTION_UNITS[direction]
            delta = (dr_unit * cell_ticks, dc_unit * cell_ticks)
            mm = MotionModelClaim(
                action_id=action_name, delta=delta,
            )
            try:
                _hs.propose(
                    ws, mm,
                    source="bridge:miner_promotion",
                    scope=Scope(kind=ScopeKind.GAME),
                        # motion model transfers across episodes
                        # (per SPEC: "MotionModelClaim IS
                        # transferable across episodes")
                    step=step,
                    initial_credence=h.credence,
                    rationale=(f"motion model from miner-promoted "
                                f"{h.hypothesis_id}"),
                )
                n_proposed += 1
            except Exception as e:
                print(f"  [bridge] MotionModelClaim propose failed: {e}")

    return n_proposed


def _seed_primary_goals(ws: WorldState, wk: WorldKnowledge,
                          step: int) -> None:
    """For every entity with a goal-like role (collectable,
    trigger_target, target, goal) that has a known cell, add a
    ``Goal`` with AtPosition condition on the agent.  The actor's
    cell_actor used the same heuristic; here we hand it to the
    real planner instead.

    Game-agnostic: the role-set is the closed catalog vocabulary
    from SPEC_perception_module.md."""
    GOAL_ROLES = {"collectable", "trigger_target", "goal", "target"}
    for name, rec in wk.entities.items():
        if rec.current_role not in GOAL_ROLES:
            continue
        if rec.current_cell is None:
            continue
        gid = f"reach:{name}"
        if gid in ws.goal_forest.goals:
            continue
        # Manhattan-based priority — same heuristic as cell_actor
        agent_rec = wk._find_agent()
        if agent_rec is not None and agent_rec.current_cell is not None:
            manhattan = (abs(rec.current_cell[0] - agent_rec.current_cell[0])
                          + abs(rec.current_cell[1] - agent_rec.current_cell[1]))
        else:
            manhattan = 100
        priority = 1.0 / (1.0 + float(manhattan))
        root = GoalNode(
            id=f"{gid}::atom", node_type=NodeType.ATOM,
            condition=AtPosition(
                pos=tuple(rec.current_cell),
                entity_id="agent",
                tolerance=0.0,
            ),
            status=GoalStatus.OPEN,
            source="bridge:primary",
            created_at=step,
        )
        goal = Goal(
            id=gid, root=root, priority=priority,
            source="bridge:primary",
            created_at=step,
            tags=frozenset({"task", "pickup"}),
        )
        add_goal(ws, goal)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def choose_planned_action(wk: WorldKnowledge,
                           available_actions: list[str],
                           ) -> PlannedActionChoice:
    """Build a minimal WorldState from the WorldKnowledge, seed
    primary goals + curiosity goals, run the planner, and return
    the first action of the chosen plan.

    Game-agnostic: the available_actions list comes from the
    adapter (filtered to actions the game actually supports).
    No assumption about action semantics.
    """
    ws = _make_min_world_state(wk)
    actions = [_make_action(name) for name in available_actions
                if name != "NONE"]

    # Bridge our miner-promoted MechanicHypotheses into the
    # cognitive_os hypothesis store as TransitionClaims +
    # MotionModelClaims.  This is what lets the planner CHAIN
    # learned action effects into multi-step plans toward a
    # primary goal -- without it the planner only has the
    # action-trial curiosity goals to work with (still useful for
    # probing untried actions, but no exploitation possible until
    # this bridge fires).
    n_bridged = bridge_promoted_hypotheses(ws, wk, step=wk.turn)
    if n_bridged:
        print(f"  [planner-bridge] bridged {n_bridged} promoted "
              f"hypotheses into TransitionClaim / MotionModelClaim")

    # Seed primary goals (reach collectables / triggers) and let
    # the explorer add curiosity goals for untested actions /
    # unprobed entities.
    _seed_primary_goals(ws, wk, step=wk.turn)
    try:
        curiosity = _explorer.propose_curiosity_goals(
            ws, step=wk.turn, action_space=actions,
        )
        # Suppress re-proposing curiosity action-trials for actions ALREADY TRIED.
        # The explorer rebuilds a fresh world-state each turn, so without this it
        # re-proposes EVERY action every turn -- COS then loops on action-probing
        # (measured: ka59 re-probed the ~7 actions ~14x each over 80 turns) and never
        # hands off.  Game-agnostic: one trial per action suffices to observe its
        # effect; once the action space is covered the action-trial goals drop out and
        # the planner falls through to entity-probes / solved-mechanic pursuits / primary
        # goals -- the explore->exploit transition.  (Entity-CLICK probes -- a different
        # 'explore:entity:' goal -- are untouched, so clicking each thing still happens.)
        tried_actions = {getattr(a, "action", None)
                         for a in getattr(wk, "actions_taken", [])}
        for g in curiosity:
            gid = getattr(g, "id", "") or ""
            if (gid.startswith("explore:action:")
                    and gid[len("explore:action:"):] in tried_actions):
                continue
            add_goal(ws, g)
    except Exception as e:
        # Explorer config might be incompatible; degrade
        # gracefully.  The miner-discovered mechanic hypotheses
        # in our WorldKnowledge are not yet bridged to the
        # cognitive_os hypothesis store, so explorer may produce
        # noisy goals.  Still useful: the action-trial goals work
        # without any hypothesis-store input.
        print(f"  [planner-bridge] explorer skipped: {e}")

    # Ask the planner for a goal + plan.
    try:
        goal_id, plan = _planner.select_and_plan(
            ws, actions, step=wk.turn,
        )
    except Exception as e:
        print(f"  [planner-bridge] planner failed: {e}")
        return PlannedActionChoice(
            action_string="NONE", goal_id=None, plan_kind="none",
            rationale=f"planner raised: {e}",
            full_plan_actions=[],
        )

    if goal_id is None or plan is None or not plan.steps:
        return PlannedActionChoice(
            action_string="NONE", goal_id=None, plan_kind="none",
            rationale="planner returned no actionable plan",
            full_plan_actions=[],
        )

    # The first step's action.name is the adapter action string.
    first_step = plan.steps[0]
    action_str = first_step.action.name

    # Classify the goal kind for the trace
    if goal_id.startswith("explore:action:"):
        kind = "curiosity:action_trial"
    elif goal_id.startswith("explore:entity:"):
        kind = "curiosity:entity_probe"
    elif goal_id.startswith("reach:"):
        kind = "primary"
    else:
        kind = "other"

    full_plan = [step.action.name for step in plan.steps]
    return PlannedActionChoice(
        action_string=action_str,
        goal_id=goal_id,
        plan_kind=kind,
        rationale=(f"planner chose {action_str!r} as step 1 of "
                    f"{len(plan.steps)}-step plan for goal "
                    f"{goal_id!r}"),
        full_plan_actions=full_plan,
    )
