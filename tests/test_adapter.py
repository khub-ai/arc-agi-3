"""Tests for :class:`ArcAdapter`.

These exercise the Adapter ABC contract end to end against a replay
fixture — ``initialize`` populates the tool registry, ``reset`` and
``observe`` produce typed :class:`Observation`\\s, ``execute`` drives
the replay cursor forward, and ``is_done`` flips when the recorded
state hits ``WIN``.

The engine itself is not invoked here; :mod:`test_end_to_end` runs a
full ``run_episode`` through the adapter.
"""

from __future__ import annotations

from cognitive_os import (
    Action,
    AgentDied,
    EntityAppeared,
    EntityStateChanged,
    GoalConditionMet,
    Observation,
    ToolInvocation,
    WorldState,
)

from arc_agi_3 import ArcAdapter

from .fixtures import moving_agent_episode
from .fixtures.synthetic import _FakeAction, blank_grid


def _make_adapter() -> ArcAdapter:
    frames, states, levels, available = moving_agent_episode()
    return ArcAdapter.from_replay(
        frames             = frames,
        states             = states,
        levels_completed   = levels,
        available_actions  = available,
        env_id             = "test_replay",
    )


def test_initialize_populates_tool_registry_and_seed_goal() -> None:
    ws = WorldState()
    adapter = _make_adapter()
    adapter.initialize(ws)
    # Every tool the registry module lists should be present.
    from arc_agi_3.tools.registry import TOOL_NAMES
    for name in TOOL_NAMES:
        assert ws.tool_registry.has(name), f"missing tool: {name}"
    # Seed goal wired up.
    assert any(g.id == "episode" for g in ws.goal_forest.goals.values())


def test_reset_produces_observation_with_agent_entity() -> None:
    adapter = _make_adapter()
    obs = adapter.reset()
    assert isinstance(obs, Observation)
    assert obs.step == 1
    # First frame has one colour-2 region ⇒ exactly one EntityAppeared
    # event and one entity in the snapshot map.
    appeared = [e for e in obs.events if isinstance(e, EntityAppeared)]
    assert len(appeared) == 1
    assert len(obs.entity_snapshots) == 1
    props = next(iter(obs.entity_snapshots.values()))
    assert props["colour"] == 2
    assert props["area"] == 1


def test_execute_advances_replay_and_observes_motion() -> None:
    adapter = _make_adapter()
    adapter.reset()
    space = adapter.action_space()
    assert space, "replay should expose at least one action"
    adapter.execute(space[0])
    obs = adapter.observe()
    # The agent cell moved from col 0 to col 1 ⇒ centroid change
    # should surface as an EntityStateChanged event.
    moves = [e for e in obs.events if isinstance(e, EntityStateChanged)]
    assert moves, "expected centroid change on successful move"


def test_is_done_flips_at_end_of_replay() -> None:
    adapter = _make_adapter()
    adapter.reset()
    space = adapter.action_space()
    action = space[0]
    # Drive through the four moves.
    for _ in range(4):
        assert not adapter.is_done()
        adapter.execute(action)
        adapter.observe()
    assert adapter.is_done()


def test_execute_with_unknown_action_is_silent_noop() -> None:
    adapter = _make_adapter()
    adapter.reset()
    before = adapter._perception.step_counter   # noqa: SLF001 — white-box inspection is fine in tests
    # Construct an Action the replay env does not expose.
    bogus = Action(id="ACTION999", name="ACTION999")
    adapter.execute(bogus)
    # execute must not raise, and must not drive perception forward
    # (observe would, but we do not call it here).
    after = adapter._perception.step_counter    # noqa: SLF001
    assert before == after


def test_invoke_tool_round_trip() -> None:
    ws = WorldState()
    adapter = _make_adapter()
    adapter.initialize(ws)
    adapter.reset()
    inv = ToolInvocation(
        invocation_id = "t1",
        tool_name     = "grid.components.extract_regions",
        arguments     = {"grid": blank_grid(3, 3, 0)},
        requester     = "test",
        requested_at  = 0,
    )
    result = adapter.invoke_tool(inv)
    assert result.success
    assert result.result == []   # blank grid has no non-background regions


def test_win_state_emits_goal_condition_met() -> None:
    adapter = _make_adapter()
    adapter.reset()
    last_obs = None
    for _ in range(4):
        adapter.execute(adapter.action_space()[0])
        last_obs = adapter.observe()
    assert last_obs is not None
    assert any(isinstance(e, GoalConditionMet) and e.goal_id == "episode"
               for e in last_obs.events)
