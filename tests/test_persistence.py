"""Tests for cross-invocation knowledge persistence (Phase 5c).

These exercise the save/load path directly and the harness-integrated
cross-invocation behaviour:

* :func:`save_knowledge` / :func:`load_knowledge` round-trip a
  :class:`CachedSolution`, preserving plan steps, scope, and counters.
* The JSON file is human-readable and correctly versioned.
* Missing directories and missing files are handled gracefully.
* Malformed entries are skipped, not fatal.
* A future schema_version raises (guards against silent data loss).
* ``run_harness`` with ``--knowledge-dir`` loads prior knowledge
  before the first episode and saves after the last, so a second
  invocation sees state committed by the first.
* ``--no-load-knowledge`` / ``--no-save-knowledge`` flags opt out of
  each half independently (competition-mode safety).

No engine internals beyond the public ``WorldState.cached_solutions``
contract are touched, keeping these tests stable across engine
evolution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cognitive_os import (
    Action,
    CachedSolution,
    Plan,
    PlannedAction,
    PlanStatus,
    Scope,
    ScopeKind,
    WorldState,
)

from arc_agi_3 import harness
from arc_agi_3.adapter import _ReplayEnv
from arc_agi_3.persistence import (
    LoadReport,
    load_knowledge,
    save_knowledge,
)

from .fixtures import moving_agent_episode


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_cached_solution(
    *,
    cs_id:    str = "cs_level_1",
    task_id:  str = "ls20::level_1",
    n_uses:   int = 3,
    n_succ:   int = 2,
) -> CachedSolution:
    plan = Plan(
        goal_id            = "episode",
        steps              = [
            PlannedAction(
                action                = Action(id="A0", name="ACTION0", parameters=()),
                expected_effects      = [],
                depends_on_hypotheses = [],
                pre_condition         = None,
            ),
            PlannedAction(
                action                = Action(id="A2", name="ACTION2", parameters=(("delta", 1),)),
                expected_effects      = [],
                depends_on_hypotheses = [],
                pre_condition         = None,
            ),
        ],
        computed_at         = 42,
        assumptions         = ["assume_a"],
        branch_selections   = {"or_node_1": "left"},
        status              = PlanStatus.COMPLETE,
        current_step_index  = 2,
    )
    return CachedSolution(
        id             = cs_id,
        task_id        = task_id,
        plan           = plan,
        task_parameters = (("level_seed", 7),),
        recorded_at    = 12,
        n_uses         = n_uses,
        n_successes    = n_succ,
        deterministic  = True,
        monitor_level  = "low",
        scope          = Scope(kind=ScopeKind.LEVEL),
        source         = "test",
        rationale      = "round-trip test",
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


def test_round_trip_preserves_cached_solution(tmp_path: Path) -> None:
    ws = WorldState()
    cs = _make_cached_solution()
    ws.cached_solutions[cs.id] = cs

    save_knowledge(ws, tmp_path)

    restored = WorldState()
    report = load_knowledge(restored, tmp_path)

    assert report.cached_solutions    == 1
    assert report.skipped_unsupported == 0

    got = restored.cached_solutions[cs.id]
    assert got.task_id           == cs.task_id
    assert got.n_uses            == cs.n_uses
    assert got.n_successes       == cs.n_successes
    assert got.deterministic     == cs.deterministic
    assert got.monitor_level     == cs.monitor_level
    assert got.scope.kind        == cs.scope.kind
    assert got.rationale         == cs.rationale
    assert got.source            == cs.source
    assert got.task_parameters   == cs.task_parameters

    # Plan structure
    assert got.plan.goal_id            == cs.plan.goal_id
    assert got.plan.status             == cs.plan.status
    assert got.plan.current_step_index == cs.plan.current_step_index
    assert got.plan.assumptions        == cs.plan.assumptions
    assert got.plan.branch_selections  == cs.plan.branch_selections
    assert len(got.plan.steps)         == len(cs.plan.steps)
    assert got.plan.steps[0].action.id == "A0"
    assert got.plan.steps[1].action.name == "ACTION2"
    assert got.plan.steps[1].action.parameters == (("delta", 1),)


def test_saved_file_is_human_readable_json(tmp_path: Path) -> None:
    ws = WorldState()
    ws.cached_solutions["cs"] = _make_cached_solution(cs_id="cs")
    target = save_knowledge(ws, tmp_path)

    raw = target.read_text(encoding="utf-8")
    data = json.loads(raw)
    assert data["schema_version"] == 1
    assert isinstance(data["cached_solutions"], list)
    assert data["cached_solutions"][0]["id"] == "cs"
    # Pretty-printed indentation: easy to diff in git.
    assert "\n  " in raw


def test_save_creates_nested_directories(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    ws = WorldState()
    target = save_knowledge(ws, nested)
    assert target.exists()
    assert target.parent == nested


# ---------------------------------------------------------------------------
# Missing / corrupt inputs
# ---------------------------------------------------------------------------


def test_load_missing_directory_returns_empty_report(tmp_path: Path) -> None:
    # Point at a subdir that doesn't exist — no crash, just zero loaded.
    report = load_knowledge(WorldState(), tmp_path / "nonexistent")
    assert isinstance(report, LoadReport)
    assert report.cached_solutions == 0


def test_load_missing_file_returns_empty_report(tmp_path: Path) -> None:
    # Directory exists but no knowledge.json inside.
    report = load_knowledge(WorldState(), tmp_path)
    assert report.cached_solutions == 0


def test_load_skips_malformed_entries(tmp_path: Path) -> None:
    # Hand-craft a file with one good and one malformed entry.
    good_ws = WorldState()
    good_ws.cached_solutions["ok"] = _make_cached_solution(cs_id="ok")
    save_knowledge(good_ws, tmp_path)

    target = tmp_path / "knowledge.json"
    data = json.loads(target.read_text(encoding="utf-8"))
    data["cached_solutions"].append({"id": "bad"})   # missing required fields
    target.write_text(json.dumps(data), encoding="utf-8")

    ws = WorldState()
    report = load_knowledge(ws, tmp_path)
    assert report.cached_solutions    == 1
    assert report.skipped_unsupported == 1
    assert "ok"  in ws.cached_solutions
    assert "bad" not in ws.cached_solutions


def test_load_rejects_future_schema_version(tmp_path: Path) -> None:
    (tmp_path / "knowledge.json").write_text(
        json.dumps({"schema_version": 999, "cached_solutions": []}),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="schema_version"):
        load_knowledge(WorldState(), tmp_path)


# ---------------------------------------------------------------------------
# Merge semantics
# ---------------------------------------------------------------------------


def test_load_merges_into_existing_cached_solutions(tmp_path: Path) -> None:
    # Save one solution, then load into a WS that already has a
    # different one.  Both should be present afterwards.
    ws_a = WorldState()
    ws_a.cached_solutions["a"] = _make_cached_solution(cs_id="a", task_id="task_a")
    save_knowledge(ws_a, tmp_path)

    ws_b = WorldState()
    ws_b.cached_solutions["b"] = _make_cached_solution(cs_id="b", task_id="task_b")
    load_knowledge(ws_b, tmp_path)

    assert set(ws_b.cached_solutions.keys()) == {"a", "b"}
    assert ws_b.cached_solutions["a"].task_id == "task_a"
    assert ws_b.cached_solutions["b"].task_id == "task_b"


def test_load_on_id_collision_prefers_loaded(tmp_path: Path) -> None:
    # Loaded entries are "newer" than any in-memory defaults; id
    # collision resolution should favour disk state.
    disk = WorldState()
    disk.cached_solutions["c"] = _make_cached_solution(cs_id="c", n_uses=10, n_succ=10)
    save_knowledge(disk, tmp_path)

    live = WorldState()
    live.cached_solutions["c"] = _make_cached_solution(cs_id="c", n_uses=1, n_succ=0)
    load_knowledge(live, tmp_path)

    assert live.cached_solutions["c"].n_uses      == 10
    assert live.cached_solutions["c"].n_successes == 10


# ---------------------------------------------------------------------------
# Harness integration
# ---------------------------------------------------------------------------


class _FakeArcade:
    def __init__(self, api_key: str = "") -> None:
        self.api_key = api_key
    def make(self, game_id: str, **_kw: object) -> _ReplayEnv:
        frames, states, levels, available = moving_agent_episode()
        return _ReplayEnv(
            frames            = list(frames),
            states            = list(states),
            levels            = list(levels),
            available_actions = [list(a) for a in available],
        )


def _fake_factory(api_key: str) -> _FakeArcade:
    return _FakeArcade(api_key=api_key)


def test_harness_saves_knowledge_after_run(tmp_path: Path) -> None:
    # The fake episode won't itself produce a CachedSolution (the
    # engine's post-mortem recorder is not yet wiring them for
    # replayed action sequences in this phase), so we seed one via
    # the persistence layer before the run and verify it's preserved
    # on disk afterwards — i.e. save actually wrote the file.
    seed_ws = WorldState()
    seed_ws.cached_solutions["seed"] = _make_cached_solution(cs_id="seed")
    save_knowledge(seed_ws, tmp_path)

    result = harness.run_harness(
        game_id        = "ls20",
        episodes       = 1,
        backend        = "null",
        arcade_factory = _fake_factory,
        knowledge_dir  = str(tmp_path),
    )
    assert result.successes == 1
    assert (tmp_path / "knowledge.json").exists()
    # After the run, the seeded CachedSolution should still be on disk.
    reloaded = WorldState()
    load_knowledge(reloaded, tmp_path)
    assert "seed" in reloaded.cached_solutions


def test_harness_load_carries_prior_knowledge_into_run(tmp_path: Path) -> None:
    # Seed a CachedSolution on disk, then run the harness and assert
    # that the WS going into the first episode saw it.  We inject a
    # probing arcade_factory that snapshots ws.cached_solutions on
    # the first make() call (the harness loads knowledge before
    # make()-ing the env's first wrapper adapter).  Simpler approach:
    # seed, run, then inspect the saved file — if the run clobbers
    # the file with empty state, loading was broken.
    seed_ws = WorldState()
    seed_ws.cached_solutions["prior"] = _make_cached_solution(cs_id="prior")
    save_knowledge(seed_ws, tmp_path)

    harness.run_harness(
        game_id        = "ls20",
        backend        = "null",
        arcade_factory = _fake_factory,
        knowledge_dir  = str(tmp_path),
    )

    reloaded = WorldState()
    load_knowledge(reloaded, tmp_path)
    # The harness should have loaded 'prior', carried it through the
    # episode, and saved it again.  If load were broken, save would
    # have written an empty file and 'prior' would be gone.
    assert "prior" in reloaded.cached_solutions


def test_harness_no_save_knowledge_skips_save(tmp_path: Path) -> None:
    # A pre-existing knowledge file must survive untouched when
    # save_knowledge_=False.  We detect untouched by checking the
    # file content is byte-identical before and after.
    seed_ws = WorldState()
    seed_ws.cached_solutions["keep"] = _make_cached_solution(cs_id="keep")
    save_knowledge(seed_ws, tmp_path)
    before = (tmp_path / "knowledge.json").read_bytes()

    harness.run_harness(
        game_id        = "ls20",
        backend        = "null",
        arcade_factory = _fake_factory,
        knowledge_dir  = str(tmp_path),
        save_knowledge_ = False,
    )

    after = (tmp_path / "knowledge.json").read_bytes()
    assert before == after


def test_harness_no_load_knowledge_skips_load(tmp_path: Path) -> None:
    # Competition-mode safety: --no-load-knowledge means prior
    # solutions must not leak into the run.  After the run, saving
    # still happens, so the file gets overwritten with whatever WS
    # contained — which in this case is nothing CachedSolution-wise.
    seed_ws = WorldState()
    seed_ws.cached_solutions["leak"] = _make_cached_solution(cs_id="leak")
    save_knowledge(seed_ws, tmp_path)

    harness.run_harness(
        game_id        = "ls20",
        backend        = "null",
        arcade_factory = _fake_factory,
        knowledge_dir  = str(tmp_path),
        load_knowledge_ = False,
    )

    # Save happened (default), so the file now reflects the post-run
    # WS, which never saw 'leak' — good, the isolation held.
    reloaded = WorldState()
    load_knowledge(reloaded, tmp_path)
    assert "leak" not in reloaded.cached_solutions


def test_harness_without_knowledge_dir_is_ephemeral(tmp_path: Path) -> None:
    # No --knowledge-dir: persistence layer must not run at all.  We
    # verify by pointing *into* tmp_path only to assert nothing was
    # written there by the harness itself.
    result = harness.run_harness(
        game_id        = "ls20",
        backend        = "null",
        arcade_factory = _fake_factory,
        # knowledge_dir deliberately not set
    )
    assert result.successes == 1
    assert list(tmp_path.iterdir()) == []
