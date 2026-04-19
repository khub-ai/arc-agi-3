"""Tests for the adapter-side game characterization subsystem.

Two concerns pinned here:

1. :class:`CharacterizationStore` — tag-indexed retrieval and
   persistence round-trip.  Score is overlap-count; list-valued tags
   contribute one point per shared element.

2. :class:`GameCharacterizationTrigger` — fires exactly once per
   level, parses the Observer reply into ``_game`` PropertyClaims,
   persists the hypothesis on level-complete transitions, and
   gracefully degrades on malformed replies.

The trigger is exercised against a stub adapter whose
``observer_query`` returns a fixed :class:`ObserverAnswer`.  No live
LLM, no cached oracle file, no real frames — just the algebra.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional

import pytest

from cognitive_os import (
    ObserverAnswer,
    ObserverQuery,
    QuestionType,
    WorldState,
)
from cognitive_os.claims import PropertyClaim
from cognitive_os.types import EntityModel, Observation

from arc_agi_3.game_characterization import (
    CHARACTERIZATION_FIELDS,
    CharacterizationEntry,
    CharacterizationStore,
    GAME_ENTITY_ID,
    GameCharacterizationTrigger,
)


# ---------------------------------------------------------------------------
# Stub adapter
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Minimal fake adapter: responds to observer_query with a canned
    answer chosen by query_id prefix, and records every query for
    inspection."""

    def __init__(self, answer: ObserverAnswer) -> None:
        self.answer = answer
        self.seen_queries: List[ObserverQuery] = []

    def observer_query(self, q: ObserverQuery) -> ObserverAnswer:
        self.seen_queries.append(q)
        return self.answer


def _ws_with_frame(
    *,
    env_id:           Optional[str]     = "ls20",
    levels_completed: int               = 0,
    frame:            Optional[List[List[int]]] = None,
) -> WorldState:
    """Build a WorldState whose observation_history has one frame."""
    ws = WorldState()
    if env_id is not None:
        ws.agent["_env_id"] = env_id
    ws.agent["levels_completed"] = levels_completed
    raw = frame if frame is not None else [[0, 0], [0, 0]]
    ws.observation_history = [Observation(
        step             = 0,
        agent_state      = {"levels_completed": levels_completed},
        events           = [],
        entity_snapshots = {},
        raw_frame        = raw,
        metadata         = {},
    )]
    return ws


def _good_answer(*, confidence: float = 0.8) -> ObserverAnswer:
    return ObserverAnswer(
        query_id    = "unused",
        result      = {
            "narrative":   "Avatar navigates maze to touch a distinct tile.",
            "genre":       "maze-navigation",
            "win_pattern": "reach distinguished tile",
            "characters":  ["avatar", "target", "wall"],
            "mechanics":   "Four-direction move; touching target advances.",
        },
        confidence  = confidence,
        explanation = "Grid of walls with one bright cell.",
    )


# ---------------------------------------------------------------------------
# CharacterizationStore
# ---------------------------------------------------------------------------


def test_store_empty_file_round_trip(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    assert store.query({"game_id": "ls20"}) == []
    # Saving an empty list creates a readable file.
    store.save()
    assert (tmp_path / "gc.json").exists()
    reloaded = CharacterizationStore(tmp_path / "gc.json")
    assert len(reloaded) == 0


def test_store_persist_and_query_scalar_tags(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    store.persist(
        tags       = {"game_id": "ls20", "level_id": 1, "genre": "maze-navigation"},
        hypothesis = {"genre": "maze-navigation", "win_pattern": "reach tile"},
    )
    # Reload from disk to make sure save/load round-trips.
    reloaded = CharacterizationStore(tmp_path / "gc.json")
    hits = reloaded.query({"game_id": "ls20", "genre": "maze-navigation"})
    assert len(hits) == 1
    score, entry = hits[0]
    assert score == 2  # game_id + genre
    assert entry.hypothesis["genre"] == "maze-navigation"


def test_store_query_list_valued_tags(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    store.persist(
        tags       = {"game_id": "ls20", "characters": ["avatar", "target", "wall"]},
        hypothesis = {"characters": ["avatar", "target", "wall"]},
    )
    store.persist(
        tags       = {"game_id": "zz00", "characters": ["avatar", "hazard"]},
        hypothesis = {"characters": ["avatar", "hazard"]},
    )
    # Query with shared characters — the more-overlapping entry wins.
    hits = store.query({"characters": ["avatar", "target"]})
    assert len(hits) == 2
    assert hits[0][0] == 2  # ls20 shares avatar + target
    assert hits[1][0] == 1  # zz00 shares avatar only


def test_store_case_insensitive_string_match(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    store.persist(
        tags       = {"genre": "Maze-Navigation"},
        hypothesis = {"genre": "maze-navigation"},
    )
    hits = store.query({"genre": "MAZE-NAVIGATION"})
    assert len(hits) == 1


def test_store_dedup_on_exact_replay(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    store.persist(
        tags       = {"game_id": "ls20", "level_id": 1},
        hypothesis = {"genre": "maze"},
    )
    store.persist(
        tags       = {"game_id": "ls20", "level_id": 1},
        hypothesis = {"genre": "maze"},
    )
    assert len(store) == 1


def test_store_zero_score_entries_dropped(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    store.persist(tags={"game_id": "ls20"}, hypothesis={"genre": "x"})
    store.persist(tags={"game_id": "zz00"}, hypothesis={"genre": "y"})
    hits = store.query({"game_id": "other"})
    assert hits == []


# ---------------------------------------------------------------------------
# Trigger — detector and dispatch
# ---------------------------------------------------------------------------


def test_trigger_fires_exactly_once_per_level() -> None:
    trigger = GameCharacterizationTrigger()
    adapter = _StubAdapter(_good_answer())
    ws = _ws_with_frame(levels_completed=0)

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]
    trigger.maybe_dispatch(ws, adapter, step=1, cfg=None)  # type: ignore[arg-type]
    trigger.maybe_dispatch(ws, adapter, step=2, cfg=None)  # type: ignore[arg-type]

    # Three ticks, same level, one query.
    assert len(adapter.seen_queries) == 1
    # The query uses the right QuestionType.
    assert adapter.seen_queries[0].question == QuestionType.CHARACTERIZE_GAME


def test_trigger_re_fires_on_new_level() -> None:
    trigger = GameCharacterizationTrigger()
    adapter = _StubAdapter(_good_answer())

    ws = _ws_with_frame(levels_completed=0)
    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]
    assert len(adapter.seen_queries) == 1

    # Simulate level advancing.
    ws.agent["levels_completed"] = 1
    ws.observation_history[-1] = Observation(
        step=1, agent_state={"levels_completed": 1}, events=[],
        entity_snapshots={}, raw_frame=[[0]], metadata={},
    )
    trigger.maybe_dispatch(ws, adapter, step=1, cfg=None)  # type: ignore[arg-type]
    assert len(adapter.seen_queries) == 2


def test_trigger_emits_property_claims_on_good_answer() -> None:
    trigger = GameCharacterizationTrigger()
    adapter = _StubAdapter(_good_answer())
    ws = _ws_with_frame()

    trigger.maybe_dispatch(ws, adapter, step=5, cfg=None)  # type: ignore[arg-type]

    # One PropertyClaim per characterisation field on _game.
    game_claims = [
        h for h in ws.hypotheses.values()
        if isinstance(h.claim, PropertyClaim)
        and h.claim.entity_id == GAME_ENTITY_ID
    ]
    props = {h.claim.property: h.claim.value for h in game_claims}
    for field_name in CHARACTERIZATION_FIELDS:
        assert field_name in props, f"missing {field_name}"
    # characters should be a tuple of strings (PropertyClaim requires hashable).
    assert props["characters"] == ("avatar", "target", "wall")
    assert props["genre"] == "maze-navigation"


def test_trigger_zero_confidence_emits_no_claims() -> None:
    trigger = GameCharacterizationTrigger()
    adapter = _StubAdapter(_good_answer(confidence=0.0))
    ws = _ws_with_frame()

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]

    game_claims = [
        h for h in ws.hypotheses.values()
        if isinstance(h.claim, PropertyClaim)
        and h.claim.entity_id == GAME_ENTITY_ID
    ]
    assert game_claims == []


def test_trigger_malformed_result_emits_no_claims() -> None:
    trigger = GameCharacterizationTrigger()
    bad_answer = ObserverAnswer(
        query_id="q", result="not a dict at all", confidence=0.9, explanation="",
    )
    adapter = _StubAdapter(bad_answer)
    ws = _ws_with_frame()

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]

    game_claims = [
        h for h in ws.hypotheses.values()
        if isinstance(h.claim, PropertyClaim)
        and h.claim.entity_id == GAME_ENTITY_ID
    ]
    assert game_claims == []


def test_trigger_partial_result_emits_only_present_fields() -> None:
    trigger = GameCharacterizationTrigger()
    partial_answer = ObserverAnswer(
        query_id="q",
        result={"genre": "puzzle", "mechanics": ""},  # empty mechanics
        confidence=0.7, explanation="",
    )
    adapter = _StubAdapter(partial_answer)
    ws = _ws_with_frame()

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]

    props = {
        h.claim.property: h.claim.value
        for h in ws.hypotheses.values()
        if isinstance(h.claim, PropertyClaim)
        and h.claim.entity_id == GAME_ENTITY_ID
    }
    assert props == {"genre": "puzzle"}


def test_trigger_skips_when_no_raw_frame() -> None:
    trigger = GameCharacterizationTrigger()
    adapter = _StubAdapter(_good_answer())
    ws = _ws_with_frame(frame=None)  # default frame
    # Overwrite the observation with one lacking a frame.
    ws.observation_history[-1] = Observation(
        step=0, agent_state={"levels_completed": 0}, events=[],
        entity_snapshots={}, raw_frame=None, metadata={},
    )

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]

    assert adapter.seen_queries == []


def test_trigger_reset_reopens_all_levels() -> None:
    trigger = GameCharacterizationTrigger()
    adapter = _StubAdapter(_good_answer())
    ws = _ws_with_frame()

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]
    assert len(adapter.seen_queries) == 1

    # Second call same level — no re-fire.
    trigger.maybe_dispatch(ws, adapter, step=1, cfg=None)  # type: ignore[arg-type]
    assert len(adapter.seen_queries) == 1

    # After reset, the level is fair game again.
    trigger.reset()
    trigger.maybe_dispatch(ws, adapter, step=2, cfg=None)  # type: ignore[arg-type]
    assert len(adapter.seen_queries) == 2


# ---------------------------------------------------------------------------
# Trigger — persistence on level-complete
# ---------------------------------------------------------------------------


def test_trigger_persists_on_level_transition(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    trigger = GameCharacterizationTrigger(store=store)
    adapter = _StubAdapter(_good_answer())

    # Level 0: characterise.
    ws = _ws_with_frame(env_id="ls20", levels_completed=0)
    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]

    # Advance to level 1 — transition fires persist for level 0.
    ws.agent["levels_completed"] = 1
    ws.observation_history[-1] = Observation(
        step=1, agent_state={"levels_completed": 1}, events=[],
        entity_snapshots={}, raw_frame=[[0]], metadata={},
    )
    trigger.maybe_dispatch(ws, adapter, step=1, cfg=None)  # type: ignore[arg-type]

    hits = store.query({"game_id": "ls20", "level_id": 0})
    assert len(hits) == 1
    _score, entry = hits[0]
    assert entry.hypothesis["genre"] == "maze-navigation"
    assert entry.tags["level_id"] == 0  # the completed level, not the new one


def test_trigger_no_persist_without_store() -> None:
    trigger = GameCharacterizationTrigger(store=None)
    adapter = _StubAdapter(_good_answer())
    ws = _ws_with_frame(levels_completed=0)

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]
    # Transition
    ws.agent["levels_completed"] = 1
    ws.observation_history[-1] = Observation(
        step=1, agent_state={"levels_completed": 1}, events=[],
        entity_snapshots={}, raw_frame=[[0]], metadata={},
    )
    # Should not crash.
    trigger.maybe_dispatch(ws, adapter, step=1, cfg=None)  # type: ignore[arg-type]


def test_trigger_injects_priors_into_prompt_context(tmp_path: Path) -> None:
    store = CharacterizationStore(tmp_path / "gc.json")
    store.persist(
        tags       = {"game_id": "ls20", "level_id": 0, "genre": "maze-navigation"},
        hypothesis = {
            "narrative":   "prior run narrative",
            "genre":       "maze-navigation",
            "win_pattern": "reach distinguished tile",
            "characters":  ["avatar", "target"],
            "mechanics":   "four-direction move",
        },
    )
    trigger = GameCharacterizationTrigger(store=store)
    adapter = _StubAdapter(_good_answer())
    ws = _ws_with_frame(env_id="ls20", levels_completed=0)

    trigger.maybe_dispatch(ws, adapter, step=0, cfg=None)  # type: ignore[arg-type]

    assert len(adapter.seen_queries) == 1
    context = adapter.seen_queries[0].context or ""
    assert "PREVIOUSLY-CONFIRMED" in context
    assert "maze-navigation" in context


# ---------------------------------------------------------------------------
# Observer parse round-trip
# ---------------------------------------------------------------------------


def test_observer_parse_characterize_game_valid_json() -> None:
    from arc_agi_3 import observer
    q = ObserverQuery(
        query_id="q", question=QuestionType.CHARACTERIZE_GAME,
        targets=[], frames=[],
    )
    reply = json.dumps({
        "result": {
            "narrative":   "a game",
            "genre":       "Maze ",
            "win_pattern": "Reach Tile",
            "characters":  ["Avatar", "TARGET"],
            "mechanics":   "move",
        },
        "confidence":  0.6,
        "explanation": "because",
    })
    ans = observer.parse_answer(q, reply)
    assert isinstance(ans.result, dict)
    assert ans.result["genre"] == "maze"
    assert ans.result["win_pattern"] == "reach tile"
    assert ans.result["characters"] == ["avatar", "target"]
    assert ans.confidence == pytest.approx(0.6)


def test_observer_parse_characterize_game_rejects_non_dict() -> None:
    from arc_agi_3 import observer
    q = ObserverQuery(
        query_id="q", question=QuestionType.CHARACTERIZE_GAME,
        targets=[], frames=[],
    )
    ans = observer.parse_answer(q, json.dumps({
        "result": "not a dict", "confidence": 0.9, "explanation": "",
    }))
    assert ans.result is None
    assert ans.confidence <= 0.1
