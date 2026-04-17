"""MockBackend — canned answers for tests.

Tests that exercise the Observer / Mediator delegation path need a
backend that returns controlled answers without hitting a network.
:class:`MockBackend` stores pre-programmed responses keyed by
``query_id`` (or, when no keyed response exists, by the
question-type enum).  A test constructs the backend, pre-registers
the answers it expects to be requested, and asserts against the
resulting engine-side state.

This is not a production path — it is explicitly a test double.
Keeping it in the same package as the real backends means the tests
can subclass it when they need richer canned logic (see
``tests/test_observer_mediator.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from cognitive_os import (
    MediatorAnswer,
    MediatorQuery,
    MediatorQuestion,
    ObserverAnswer,
    ObserverQuery,
    QuestionType,
)

from .base import LLMBackend


# A canned-response provider can be either a literal answer or a
# callable that computes the answer from the query.  Callables get
# the raw query so tests can assert on ``query.targets``,
# ``query.frames``, or a Mediator's ``world_summary``.
ObserverResponder = Callable[[ObserverQuery], ObserverAnswer]
MediatorResponder = Callable[[MediatorQuery], MediatorAnswer]


@dataclass
class MockBackend(LLMBackend):
    """Backend with pre-programmed answers.

    Lookup order for observer queries:
        1. ``observer_by_id[query.query_id]`` if set.
        2. ``observer_by_question[query.question]`` if set.
        3. :meth:`_default_observer_answer` (zero confidence).

    The mediator path is symmetric.  Budget accounting is still
    enforced so tests exercising budget exhaustion work against the
    mock just as they would against a real backend.
    """

    observer_by_id:        Dict[str, ObserverResponder]         = field(default_factory=dict)
    observer_by_question:  Dict[QuestionType, ObserverResponder] = field(default_factory=dict)
    mediator_by_id:        Dict[str, MediatorResponder]         = field(default_factory=dict)
    mediator_by_question:  Dict[MediatorQuestion, MediatorResponder] = field(default_factory=dict)
    observer_log:          List[ObserverQuery]                  = field(default_factory=list)
    mediator_log:          List[MediatorQuery]                  = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(budget=None)

    def answer_observer_query(self, query: ObserverQuery) -> ObserverAnswer:
        self.observer_log.append(query)
        if not self.observer_budget_available():
            self.usage.observer_exhausted_returns += 1
            return ObserverAnswer(
                query_id    = query.query_id,
                result      = None,
                confidence  = 0.0,
                explanation = "observer budget exhausted",
            )
        self.usage.observer_calls += 1
        responder = self.observer_by_id.get(query.query_id) \
                    or self.observer_by_question.get(query.question)
        if responder is None:
            return self._default_observer_answer(query)
        return responder(query) if callable(responder) else responder

    def answer_mediator_query(self, query: MediatorQuery) -> MediatorAnswer:
        self.mediator_log.append(query)
        if not self.mediator_budget_available():
            self.usage.mediator_exhausted_returns += 1
            return MediatorAnswer(
                query_id    = query.query_id,
                confidence  = 0.0,
                explanation = "mediator budget exhausted",
            )
        self.usage.mediator_calls += 1
        responder = self.mediator_by_id.get(query.query_id) \
                    or self.mediator_by_question.get(query.question)
        if responder is None:
            return self._default_mediator_answer(query)
        return responder(query) if callable(responder) else responder

    # ------------------------------------------------------------------
    # Defaults
    # ------------------------------------------------------------------

    @staticmethod
    def _default_observer_answer(query: ObserverQuery) -> ObserverAnswer:
        return ObserverAnswer(
            query_id    = query.query_id,
            result      = None,
            confidence  = 0.0,
            explanation = "MockBackend: no canned observer response",
        )

    @staticmethod
    def _default_mediator_answer(query: MediatorQuery) -> MediatorAnswer:
        return MediatorAnswer(
            query_id    = query.query_id,
            confidence  = 0.0,
            explanation = "MockBackend: no canned mediator response",
        )
