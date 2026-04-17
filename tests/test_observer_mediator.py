"""Tests for the Observer / Mediator seam and the backend protocol.

Phase 5b-i adds pluggable LLM backends behind the adapter's
``observer_query`` / ``mediator_query`` methods.  These tests exercise:

* :class:`NullBackend` — zero-confidence answers, no network.
* :class:`MockBackend` — canned responses, budget accounting.
* Prompt construction — key fields present, JSON schema declared.
* Reply parsing — well-formed JSON produces typed answers;
  malformed replies degrade to zero confidence.
* :class:`ChatBackend` end-to-end via a fake ``chat()`` implementation
  (no real LLM touched).
* Adapter delegation — ``observer_query`` and ``mediator_query``
  forward to the injected backend.

The Anthropic backend is not exercised here — it requires an API
key and a network; its SDK-level behaviour is tested indirectly
through ``ChatBackend`` with a fake ``chat()``.
"""

from __future__ import annotations

import json
from typing import List

from cognitive_os import (
    LLMBudget,
    MediatorAnswer,
    MediatorQuery,
    MediatorQuestion,
    ObserverAnswer,
    ObserverQuery,
    PropertyClaim,
    QuestionType,
    WorldState,
    WorldStateSummary,
)

from arc_agi_3 import ArcAdapter
from arc_agi_3.backends import (
    ChatBackend,
    MockBackend,
    NullBackend,
)
from arc_agi_3.backends.base import ChatMessage
from arc_agi_3 import observer, mediator

from .fixtures import moving_agent_episode


# ---------------------------------------------------------------------------
# NullBackend
# ---------------------------------------------------------------------------


def test_null_backend_observer_returns_zero_confidence() -> None:
    backend = NullBackend()
    q = ObserverQuery(
        query_id = "q1",
        question = QuestionType.DESCRIBE,
        targets  = ["e1"],
        frames   = [[[0, 1], [2, 3]]],
    )
    answer = backend.answer_observer_query(q)
    assert isinstance(answer, ObserverAnswer)
    assert answer.query_id == "q1"
    assert answer.confidence == 0.0
    assert answer.result is None


def test_null_backend_mediator_returns_zero_confidence() -> None:
    backend = NullBackend()
    summary = WorldStateSummary(step=0, agent={})
    q = MediatorQuery(
        query_id      = "m1",
        question      = MediatorQuestion.IDENTIFY_ROLES,
        world_summary = summary,
    )
    answer = backend.answer_mediator_query(q)
    assert isinstance(answer, MediatorAnswer)
    assert answer.query_id == "m1"
    assert answer.confidence == 0.0


# ---------------------------------------------------------------------------
# MockBackend
# ---------------------------------------------------------------------------


def test_mock_backend_canned_observer_response() -> None:
    backend = MockBackend(
        observer_by_question={
            QuestionType.STILL_SIMILAR: lambda q: ObserverAnswer(
                query_id    = q.query_id,
                result      = True,
                confidence  = 0.9,
                explanation = "canned",
            ),
        },
    )
    q = ObserverQuery(
        query_id = "q2",
        question = QuestionType.STILL_SIMILAR,
        targets  = ["e1", "e2"],
        frames   = [],
    )
    answer = backend.answer_observer_query(q)
    assert answer.result is True
    assert answer.confidence == 0.9
    assert backend.usage.observer_calls == 1
    assert len(backend.observer_log) == 1


def test_mock_backend_observer_budget_exhaustion() -> None:
    backend = MockBackend()
    backend.budget = LLMBudget(observer_per_episode=2)
    q = ObserverQuery(
        query_id = "q",
        question = QuestionType.DESCRIBE,
        targets  = [],
        frames   = [],
    )
    _ = backend.answer_observer_query(q)
    _ = backend.answer_observer_query(q)
    # Third call exceeds the per-episode cap.
    answer = backend.answer_observer_query(q)
    assert answer.confidence == 0.0
    assert "exhausted" in (answer.explanation or "").lower()
    assert backend.usage.observer_exhausted_returns == 1


def test_mock_backend_reset_usage() -> None:
    backend = MockBackend()
    _ = backend.answer_observer_query(ObserverQuery(
        query_id="q", question=QuestionType.DESCRIBE, targets=[], frames=[],
    ))
    assert backend.usage.observer_calls == 1
    backend.reset_usage()
    assert backend.usage.observer_calls == 0


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def test_observer_prompt_includes_targets_and_schema() -> None:
    q = ObserverQuery(
        query_id = "q",
        question = QuestionType.CLASSIFY,
        targets  = ["e1", "e2"],
        frames   = [[[0, 1], [1, 0]]],
        context  = "initial scan",
    )
    messages = observer.prompt_for(q)
    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[1].role == "user"
    body = messages[1].content
    assert "CLASSIFY" in body
    assert "e1" in body and "e2" in body
    assert "REPLY_SCHEMA" in body
    assert "initial scan" in body


def test_mediator_prompt_serialises_summary() -> None:
    summary = WorldStateSummary(
        step  = 7,
        agent = {"state_name": "PLAYING"},
    )
    q = MediatorQuery(
        query_id      = "m",
        question      = MediatorQuestion.IDENTIFY_ROLES,
        world_summary = summary,
    )
    messages = mediator.prompt_for(q)
    body = messages[1].content
    assert "IDENTIFY_ROLES" in body
    assert '"step": 7' in body
    assert "REPLY_SCHEMA" in body


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


def test_observer_parse_valid_classify_reply() -> None:
    q = ObserverQuery(
        query_id = "q",
        question = QuestionType.CLASSIFY,
        targets  = ["e1", "e2"],
        frames   = [],
    )
    reply = json.dumps({
        "result":      {"e1": "agent", "e2": "wall"},
        "confidence":  0.7,
        "explanation": "colour-based guess",
    })
    answer = observer.parse_answer(q, reply)
    assert answer.result == {"e1": "agent", "e2": "wall"}
    assert 0.7 - 1e-9 <= answer.confidence <= 0.7 + 1e-9


def test_observer_parse_malformed_reply_degrades_gracefully() -> None:
    q = ObserverQuery(
        query_id = "q", question = QuestionType.CLASSIFY, targets = [], frames = [],
    )
    answer = observer.parse_answer(q, "this is not json, sorry")
    assert answer.confidence == 0.0
    assert "parse_error" in (answer.explanation or "")


def test_observer_parse_reply_embedded_in_prose() -> None:
    q = ObserverQuery(
        query_id = "q", question = QuestionType.DESCRIBE, targets = [], frames = [],
    )
    reply = (
        "Here is the answer you requested:\n"
        '{"result": "a red square", "confidence": 0.8}\n'
        "Hope that helps!"
    )
    answer = observer.parse_answer(q, reply)
    assert answer.result == "a red square"
    assert answer.confidence > 0


def test_mediator_parse_identify_roles_filters_invalid_ids() -> None:
    summary = WorldStateSummary(
        step=0, agent={},
        # No entities present — role assignments for phantom IDs must be dropped.
        entities={},
    )
    q = MediatorQuery(
        query_id      = "m",
        question      = MediatorQuestion.IDENTIFY_ROLES,
        world_summary = summary,
    )
    reply = json.dumps({
        "entity_roles": {"ghost_entity": "agent"},
        "confidence":   0.6,
        "explanation":  "the LLM hallucinated an id",
    })
    answer = mediator.parse_answer(q, reply)
    # Ghost entity must be filtered out.
    assert answer.entity_roles == {}
    assert answer.proposed_claims == []


def test_mediator_parse_identify_roles_emits_property_claims() -> None:
    from cognitive_os import EntityModel
    summary = WorldStateSummary(
        step=0, agent={},
        entities={
            "e1": EntityModel(id="e1"),
            "e2": EntityModel(id="e2"),
        },
    )
    q = MediatorQuery(
        query_id      = "m",
        question      = MediatorQuestion.IDENTIFY_ROLES,
        world_summary = summary,
    )
    reply = json.dumps({
        "entity_roles": {"e1": "agent", "e2": "wall"},
        "confidence":   0.8,
    })
    answer = mediator.parse_answer(q, reply)
    assert answer.entity_roles == {"e1": "agent", "e2": "wall"}
    assert len(answer.proposed_claims) == 2
    assert all(isinstance(c, PropertyClaim) for c in answer.proposed_claims)
    assigned = {c.entity_id: c.value for c in answer.proposed_claims}
    assert assigned == {"e1": "agent", "e2": "wall"}


# ---------------------------------------------------------------------------
# ChatBackend end-to-end with a fake chat()
# ---------------------------------------------------------------------------


class _FakeChatBackend(ChatBackend):
    """Records the messages it receives and replies with a canned string."""

    def __init__(self, canned_reply: str) -> None:
        super().__init__()
        self._reply      = canned_reply
        self.last_messages: List[ChatMessage] = []

    def chat(
        self,
        messages:    List[ChatMessage],
        *,
        max_tokens:  int = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.last_messages = messages
        return self._reply


def test_chat_backend_full_observer_flow() -> None:
    canned = json.dumps({"result": True, "confidence": 0.8, "explanation": "ok"})
    backend = _FakeChatBackend(canned_reply=canned)
    q = ObserverQuery(
        query_id = "q",
        question = QuestionType.STILL_SIMILAR,
        targets  = ["e1", "e2"],
        frames   = [[[1]]],
    )
    answer = backend.answer_observer_query(q)
    assert answer.result is True
    assert 0.8 - 1e-9 <= answer.confidence <= 0.8 + 1e-9
    assert backend.usage.observer_calls == 1
    assert backend.last_messages[-1].role == "user"


def test_chat_backend_full_mediator_flow() -> None:
    canned = json.dumps({"explanation": "the counter likely decremented", "confidence": 0.5})
    backend = _FakeChatBackend(canned_reply=canned)
    q = MediatorQuery(
        query_id      = "m",
        question      = MediatorQuestion.EXPLAIN_SURPRISE,
        world_summary = WorldStateSummary(step=0, agent={}),
    )
    answer = backend.answer_mediator_query(q)
    assert "counter" in (answer.explanation or "")
    assert backend.usage.mediator_calls == 1


# ---------------------------------------------------------------------------
# Adapter delegation
# ---------------------------------------------------------------------------


def test_adapter_defaults_to_null_backend() -> None:
    frames, states, levels, available = moving_agent_episode()
    adapter = ArcAdapter.from_replay(
        frames=frames, states=states, levels_completed=levels,
        available_actions=available,
    )
    assert isinstance(adapter.backend, NullBackend)


def test_adapter_observer_query_delegates_to_backend() -> None:
    frames, states, levels, available = moving_agent_episode()
    backend = MockBackend(
        observer_by_question={
            QuestionType.DESCRIBE: lambda q: ObserverAnswer(
                query_id = q.query_id, result = "mock", confidence = 0.5,
            ),
        },
    )
    adapter = ArcAdapter.from_replay(
        frames=frames, states=states, levels_completed=levels,
        available_actions=available, backend=backend,
    )
    q = ObserverQuery(
        query_id="q", question=QuestionType.DESCRIBE, targets=[], frames=[],
    )
    answer = adapter.observer_query(q)
    assert answer.result == "mock"
    assert len(backend.observer_log) == 1


def test_adapter_reset_clears_backend_usage() -> None:
    frames, states, levels, available = moving_agent_episode()
    backend = MockBackend()
    adapter = ArcAdapter.from_replay(
        frames=frames, states=states, levels_completed=levels,
        available_actions=available, backend=backend,
    )
    _ = adapter.observer_query(ObserverQuery(
        query_id="q", question=QuestionType.DESCRIBE, targets=[], frames=[],
    ))
    assert backend.usage.observer_calls == 1
    adapter.reset()
    assert backend.usage.observer_calls == 0
