"""Tests for :class:`GameDecomposer` — piece 2 of the game-priors
chain.

The decomposer consumes ``_game`` PropertyClaims already committed to
the hypothesis store (piece 1 / :class:`GameCharacterizationTrigger`'s
output) and synthesises plannable subgoals under the adapter-seeded
episode goal.  These tests don't run the LLM path — they directly
seed ``ws.hypotheses`` with ``PropertyClaim`` entries and observe
which goals the decomposer emits.

Coverage:
* Strategy dispatch by ``win_pattern`` keyword (reach → emit;
  unknown → no-op).
* Target selection heuristic (exclude agent, exclude walls, prefer
  unique-colour, tie-break by smallest area).
* Idempotence across repeat dispatches.
* Re-synthesis when the previous target entity vanishes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from cognitive_os import WorldState
from cognitive_os.claims import ControlledActorClaim, PropertyClaim
from cognitive_os.conditions import InsideBBox
from cognitive_os.credence import Credence
from cognitive_os.types import (
    EntityModel,
    GoalStatus,
    Hypothesis,
    Observation,
    Scope,
    ScopeKind,
)

from arc_agi_3.game_characterization import GAME_ENTITY_ID
from arc_agi_3.game_decomposer import (
    DECOMPOSE_GOAL_PREFIX,
    REACH_GOAL_STEP_BUDGET,
    GameDecomposer,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seed_game_claim(
    ws:       WorldState,
    property: str,
    value:    Any,
    *,
    step:     int   = 0,
    point:    float = 0.9,
) -> None:
    """Directly insert a committed PropertyClaim on _game."""
    claim = PropertyClaim(entity_id=GAME_ENTITY_ID, property=property, value=value)
    hid = f"h::_game::{property}"
    ws.hypotheses[hid] = Hypothesis(
        id        = hid,
        claim     = claim,
        credence  = Credence(point=point, last_confirmed=step),
        scope     = Scope(kind=ScopeKind.LEVEL),
        source    = "test",
        created_at= step,
    )


def _seed_agent_claim(ws: WorldState, colour: int, *, point: float = 0.9) -> None:
    claim = ControlledActorClaim(colour=colour, background=0)
    hid = f"h::controlled_actor::{colour}"
    ws.hypotheses[hid] = Hypothesis(
        id        = hid,
        claim     = claim,
        credence  = Credence(point=point, last_confirmed=0),
        scope     = Scope(kind=ScopeKind.GAME),
        source    = "test",
        created_at= 0,
    )


def _entity(
    ws:     WorldState,
    eid:    str,
    *,
    bbox:   tuple,
    area:   int,
    colour: int,
) -> None:
    ws.entities[eid] = EntityModel(
        id              = eid,
        properties      = {"bbox": bbox, "area": area, "colour": colour},
        first_seen_step = 0,
        last_seen_step  = 0,
    )


def _ws_with_frame(frame_h: int = 20, frame_w: int = 20) -> WorldState:
    ws = WorldState()
    raw = [[0] * frame_w for _ in range(frame_h)]
    ws.observation_history = [Observation(
        step             = 0,
        agent_state      = {},
        events           = [],
        entity_snapshots = {},
        raw_frame        = raw,
        metadata         = {},
    )]
    return ws


def _decompose_goals(ws: WorldState) -> Dict[str, Any]:
    return {
        gid: g for gid, g in ws.goal_forest.goals.items()
        if gid.startswith(DECOMPOSE_GOAL_PREFIX)
    }


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------


def test_no_game_claim_noop() -> None:
    ws = _ws_with_frame()
    _entity(ws, "e1", bbox=(2, 2, 3, 3), area=4, colour=5)
    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    assert _decompose_goals(ws) == {}


def test_reach_keyword_emits_goal() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach distinguished tile")
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "agent_ent", bbox=(0, 0, 1, 1), area=4, colour=1)
    _entity(ws, "target",    bbox=(5, 5, 6, 6), area=4, colour=7)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]

    goals = _decompose_goals(ws)
    assert len(goals) == 1
    (gid,) = goals
    assert gid == f"{DECOMPOSE_GOAL_PREFIX}reach::target"
    goal = goals[gid]
    assert isinstance(goal.root.condition, InsideBBox)
    assert goal.root.condition.entity_id == "target"
    assert goal.root.condition.probe_id == "agent"


def test_unknown_pattern_noop() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "collect every coin")  # no reach keyword
    _entity(ws, "e1", bbox=(5, 5, 6, 6), area=4, colour=7)
    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    assert _decompose_goals(ws) == {}


def test_empty_win_pattern_noop() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "")
    _entity(ws, "e1", bbox=(5, 5, 6, 6), area=4, colour=7)
    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    assert _decompose_goals(ws) == {}


# ---------------------------------------------------------------------------
# Target selection
# ---------------------------------------------------------------------------


def test_excludes_agent_colour() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=3)
    # Only entity is the agent itself.
    _entity(ws, "agent_ent", bbox=(0, 0, 1, 1), area=4, colour=3)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    assert _decompose_goals(ws) == {}


def test_prefers_unique_colour_over_shared() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    # Two candidates share colour 5; one has unique colour 9.
    _entity(ws, "shared_a", bbox=(2, 2, 3, 3), area=4,  colour=5)
    _entity(ws, "shared_b", bbox=(4, 4, 5, 5), area=4,  colour=5)
    _entity(ws, "unique",   bbox=(6, 6, 8, 8), area=9,  colour=9)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]

    goals = _decompose_goals(ws)
    assert len(goals) == 1
    (gid,) = goals
    assert gid.endswith("reach::unique")


def test_tie_break_smallest_area() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    # Both colours unique — smallest area wins.
    _entity(ws, "big",   bbox=(2, 2, 6, 6), area=25, colour=7)
    _entity(ws, "small", bbox=(8, 8, 9, 9), area=4,  colour=8)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]

    goals = _decompose_goals(ws)
    assert len(goals) == 1
    (gid,) = goals
    assert gid.endswith("reach::small")


def test_excludes_wall_spanning_edge() -> None:
    ws = _ws_with_frame(frame_h=20, frame_w=20)
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    # A long wall along the top edge: bbox spans column range + row 0.
    _entity(ws, "wall",   bbox=(0, 0, 0, 19), area=100, colour=2)
    _entity(ws, "target", bbox=(5, 5, 6, 6),  area=4,   colour=7)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]

    goals = _decompose_goals(ws)
    assert len(goals) == 1
    (gid,) = goals
    assert gid.endswith("reach::target")


def test_small_edge_entity_not_wall() -> None:
    """A tiny sprite that happens to sit on the edge but has small
    area should still be eligible (walls are both edge-spanning AND
    large)."""
    ws = _ws_with_frame(frame_h=20, frame_w=20)
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    # Thin sprite on top edge — not a wall (area below threshold).
    _entity(ws, "tiny_edge", bbox=(0, 5, 0, 6), area=2, colour=7)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]

    goals = _decompose_goals(ws)
    assert len(goals) == 1
    (gid,) = goals
    assert gid.endswith("reach::tiny_edge")


# ---------------------------------------------------------------------------
# Idempotence and re-synthesis
# ---------------------------------------------------------------------------


def test_idempotent_across_repeat_calls() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "target", bbox=(5, 5, 6, 6), area=4, colour=7)

    dec = GameDecomposer()
    dec.maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    dec.maybe_dispatch(ws, adapter=None, step=1, cfg=None)  # type: ignore[arg-type]
    dec.maybe_dispatch(ws, adapter=None, step=2, cfg=None)  # type: ignore[arg-type]

    goals = _decompose_goals(ws)
    assert len(goals) == 1


def test_resynthesises_when_target_vanishes() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "target_v1", bbox=(5, 5, 6, 6), area=4, colour=7)

    dec = GameDecomposer()
    dec.maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    goals = _decompose_goals(ws)
    assert list(goals) == [f"{DECOMPOSE_GOAL_PREFIX}reach::target_v1"]

    # Target vanishes (e.g. level transition churn); a new candidate
    # appears.
    del ws.entities["target_v1"]
    _entity(ws, "target_v2", bbox=(9, 9, 10, 10), area=4, colour=8)

    dec.maybe_dispatch(ws, adapter=None, step=1, cfg=None)  # type: ignore[arg-type]
    goals = _decompose_goals(ws)
    assert f"{DECOMPOSE_GOAL_PREFIX}reach::target_v2" in goals


def test_no_candidate_noop() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    # Only entity is the agent.
    _entity(ws, "agent_ent", bbox=(0, 0, 1, 1), area=4, colour=1)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    assert _decompose_goals(ws) == {}


def test_priority_between_episode_and_role_floor() -> None:
    """Decomposer goals should sit strictly between the episode-atom
    priority (1.0) and the role-derived floor (0.85)."""
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _entity(ws, "target", bbox=(5, 5, 6, 6), area=4, colour=7)

    GameDecomposer().maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    goals = _decompose_goals(ws)
    (goal,) = goals.values()
    assert 0.85 < goal.priority < 1.0


# ---------------------------------------------------------------------------
# Piece 3 — decay on subgoal failure
# ---------------------------------------------------------------------------


def _win_pattern_credence(ws: WorldState) -> float:
    for h in ws.hypotheses.values():
        if (
            isinstance(h.claim, PropertyClaim)
            and h.claim.entity_id == GAME_ENTITY_ID
            and h.claim.property == "win_pattern"
        ):
            return h.credence.point
    raise AssertionError("no win_pattern claim present")


def test_timeout_abandons_goal_and_contradicts_win_pattern() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile", point=0.9)
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "target", bbox=(5, 5, 6, 6), area=4, colour=7)

    dec = GameDecomposer(reach_step_budget=5)
    dec.maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    goal_id = f"{DECOMPOSE_GOAL_PREFIX}reach::target"
    assert ws.goal_forest.goals[goal_id].root.status != GoalStatus.ABANDONED
    pre_credence = _win_pattern_credence(ws)

    # Advance past the budget — no success signal.
    dec.maybe_dispatch(ws, adapter=None, step=10, cfg=None)  # type: ignore[arg-type]

    assert ws.goal_forest.goals[goal_id].root.status == GoalStatus.ABANDONED
    assert _win_pattern_credence(ws) < pre_credence


def test_timeout_does_not_contradict_if_target_vanished() -> None:
    """Churn (target disappears) should abandon the goal but NOT
    decay — that's segmentation noise, not a pattern-mismatch."""
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile", point=0.9)
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "target", bbox=(5, 5, 6, 6), area=4, colour=7)

    dec = GameDecomposer(reach_step_budget=5)
    dec.maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    pre_credence = _win_pattern_credence(ws)

    # Target vanishes BEFORE the budget runs out.
    del ws.entities["target"]
    dec.maybe_dispatch(ws, adapter=None, step=10, cfg=None)  # type: ignore[arg-type]

    # Churn path: no decay.
    assert _win_pattern_credence(ws) == pre_credence


def test_success_clears_tracking_without_decay() -> None:
    """If the condition is satisfied, the goal is done — drop the
    tracking entry, don't decay."""
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile", point=0.9)
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "target", bbox=(5, 5, 6, 6), area=4, colour=7)
    # Agent already inside target bbox.
    ws.agent["position"] = (5, 5)

    dec = GameDecomposer(reach_step_budget=5)
    dec.maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    pre_credence = _win_pattern_credence(ws)

    # Tick well past budget — but condition is True, so no timeout.
    dec.maybe_dispatch(ws, adapter=None, step=100, cfg=None)  # type: ignore[arg-type]

    assert _win_pattern_credence(ws) == pre_credence
    # Tracking was cleared after the success detection.
    assert not dec._emit_step


def test_reset_clears_tracking() -> None:
    ws = _ws_with_frame()
    _seed_game_claim(ws, "win_pattern", "reach the tile")
    _seed_agent_claim(ws, colour=1)
    _entity(ws, "target", bbox=(5, 5, 6, 6), area=4, colour=7)

    dec = GameDecomposer()
    dec.maybe_dispatch(ws, adapter=None, step=0, cfg=None)  # type: ignore[arg-type]
    assert dec._emit_step

    dec.reset()
    assert not dec._emit_step


def test_default_step_budget_is_sane() -> None:
    """Sanity pin on the budget.  Bumping this means the decomposer
    tolerates longer periods of apparent failure before decay."""
    assert REACH_GOAL_STEP_BUDGET >= 40
    assert REACH_GOAL_STEP_BUDGET <= 500
