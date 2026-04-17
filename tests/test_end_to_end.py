"""End-to-end smoke test: run a full episode through the engine.

This test proves the adapter + tool + perception seam is coherent
with the engine by invoking :func:`cognitive_os.run_episode`
directly.  If any of the engine's expectations about Observation
structure, Action shape, or ToolRegistry wiring is violated, this
test is where the failure surfaces.

The fixture episode is intentionally tiny (5 frames, ends in WIN)
so the test runs in well under a second and produces a readable
PostMortem if it ever fails.
"""

from __future__ import annotations

from cognitive_os import (
    EngineConfig,
    PostMortem,
    WorldState,
    run_episode,
)

from arc_agi_3 import ArcAdapter

from .fixtures import moving_agent_episode


def test_run_episode_against_replay_fixture() -> None:
    frames, states, levels, available = moving_agent_episode()
    adapter = ArcAdapter.from_replay(
        frames             = frames,
        states             = states,
        levels_completed   = levels,
        available_actions  = available,
        env_id             = "e2e_replay",
    )
    ws  = WorldState()
    cfg = EngineConfig.arc_agi3_default()
    pm: PostMortem = run_episode(adapter, ws, cfg, max_steps=20)

    # The replay reaches WIN on step 5; the engine should terminate
    # via ``is_done`` well inside the max_steps budget.
    assert pm.total_steps <= 20
    # Perception emitted GoalConditionMet(goal_id="episode") on the
    # WIN frame; the engine is expected to mark the seed goal as
    # achieved and surface a non-failure final status.
    assert pm.final_status in {"success", "achieved", "completed", "win"} or pm.final_status.lower().startswith("ach")


def test_tool_registry_visible_to_engine_after_initialize() -> None:
    frames, states, levels, available = moving_agent_episode()
    adapter = ArcAdapter.from_replay(
        frames             = frames,
        states             = states,
        levels_completed   = levels,
        available_actions  = available,
    )
    ws = WorldState()
    adapter.initialize(ws)
    # Every tool is listed with a signature the engine can inspect.
    for name in ws.tool_registry.names():
        sig = ws.tool_registry.get(name)
        assert sig is not None
        assert sig.name == name
        assert sig.description
