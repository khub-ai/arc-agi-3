"""Abstract base classes for LLM backends.

Two levels of abstraction:

* :class:`LLMBackend` — the highest level: ``answer_observer_query``
  and ``answer_mediator_query`` take typed engine queries and
  return typed engine answers.  A backend that wants total control
  over prompt formatting (e.g. a classical-vision pipeline with no
  LLM at all) implements these directly.

* :class:`ChatBackend` — a concrete :class:`LLMBackend` that
  delegates prompt formatting and response parsing to the shared
  helpers in :mod:`arc_agi_3.observer` and :mod:`arc_agi_3.mediator`,
  and only asks the subclass to implement a single ``chat()``
  method.  This is the path every LLM-based backend should take.

Both levels manage their own :class:`LLMBudget` accounting.
Exhausted calls return zero-confidence answers with a clear
explanation string — the engine treats that the same way it treats
"adapter does not support Mediator queries" in the base Adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from cognitive_os import (
    LLMBudget,
    MediatorAnswer,
    MediatorQuery,
    ObserverAnswer,
    ObserverQuery,
)


@dataclass
class BackendUsage:
    """Running counts of oracle calls made by a backend.

    The adapter can read these (via
    :meth:`LLMBackend.usage_snapshot`) and report them in
    ``PostMortem.observer_usage`` / ``mediator_usage`` so episode
    retrospectives reflect real expenditure.
    """

    observer_calls: int = 0
    mediator_calls: int = 0
    observer_exhausted_returns: int = 0
    mediator_exhausted_returns: int = 0

    # Per-call wall-time measurements in milliseconds.  Kept as a
    # plain list so statistics can be computed at episode end
    # without committing to any aggregation strategy here.
    observer_latencies_ms: List[float] = field(default_factory=list)
    mediator_latencies_ms: List[float] = field(default_factory=list)


class LLMBackend(ABC):
    """Abstract oracle backend.

    Subclasses implement :meth:`answer_observer_query` and
    :meth:`answer_mediator_query` directly.  The base class supplies
    per-call budget accounting and an :class:`BackendUsage` tracker.
    """

    def __init__(self, budget: Optional[LLMBudget] = None) -> None:
        self.budget: LLMBudget      = budget or LLMBudget()
        self.usage: BackendUsage    = BackendUsage()

    # ------------------------------------------------------------------
    # Budget gates (used by subclasses; not meant to be overridden)
    # ------------------------------------------------------------------

    def observer_budget_available(self) -> bool:
        return self.usage.observer_calls < self.budget.observer_per_episode

    def mediator_budget_available(self) -> bool:
        return self.usage.mediator_calls < self.budget.mediator_per_episode

    def reset_usage(self) -> None:
        """Called at episode boundaries to zero the per-episode counters."""
        self.usage = BackendUsage()

    def usage_snapshot(self) -> Dict[str, float]:
        """Return a flat dict suitable for
        ``PostMortem.observer_usage`` / ``mediator_usage``."""
        return {
            "observer_calls":              float(self.usage.observer_calls),
            "mediator_calls":              float(self.usage.mediator_calls),
            "observer_exhausted_returns":  float(self.usage.observer_exhausted_returns),
            "mediator_exhausted_returns":  float(self.usage.mediator_exhausted_returns),
        }

    # ------------------------------------------------------------------
    # Abstract surface
    # ------------------------------------------------------------------

    @abstractmethod
    def answer_observer_query(self, query: ObserverQuery) -> ObserverAnswer:
        """Handle a visual query.  Implementations are responsible for
        respecting the budget (call :meth:`observer_budget_available`
        before doing work) and recording usage (increment
        ``self.usage.observer_calls``).
        """

    @abstractmethod
    def answer_mediator_query(self, query: MediatorQuery) -> MediatorAnswer:
        """Handle a common-sense query.  Same budget / usage rules as
        :meth:`answer_observer_query`."""


# ---------------------------------------------------------------------------
# ChatBackend — LLM backends that speak via a single chat() method
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """One message in a chat transcript passed to an LLM.

    ``role`` follows the OpenAI / Anthropic conventions: ``"system"``,
    ``"user"``, ``"assistant"``.  The adapter uses only ``system``
    and ``user`` when constructing prompts; ``assistant`` appears in
    the reply.
    """

    role:    str
    content: str


class ChatBackend(LLMBackend):
    """A :class:`LLMBackend` that delegates prompt formatting and
    response parsing to shared helpers, and only asks the subclass to
    implement a single :meth:`chat` method.

    The prompt and parsing logic lives in :mod:`arc_agi_3.observer`
    and :mod:`arc_agi_3.mediator`; importing them here would cause a
    circular import, so the wiring is done via late imports inside
    the two ``answer_*`` methods.
    """

    # ------------------------------------------------------------------
    # Subclass must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def chat(
        self,
        messages:    List[ChatMessage],
        *,
        max_tokens:  int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """Send ``messages`` to the underlying LLM and return the
        assistant's reply as a string.

        ``temperature=0.0`` is the intended default: the engine
        treats oracle answers as typed evidence and benefits from
        deterministic replies.  Subclasses MAY ignore the parameter
        (e.g. if the underlying model doesn't support sampling
        control) but should document that.
        """

    # ------------------------------------------------------------------
    # LLMBackend implementation — delegates to observer / mediator
    # helpers (late-imported below to break the circular dep)
    # ------------------------------------------------------------------

    def answer_observer_query(self, query: ObserverQuery) -> ObserverAnswer:
        if not self.observer_budget_available():
            self.usage.observer_exhausted_returns += 1
            return ObserverAnswer(
                query_id    = query.query_id,
                result      = None,
                confidence  = 0.0,
                explanation = "observer budget exhausted",
            )
        from ..observer import prompt_for, parse_answer
        messages = prompt_for(query)
        import time
        t0 = time.perf_counter()
        # 2000-token ceiling: STILL_SIMILAR / CLASSIFY / DESCRIBE are
        # tiny; ENUMERATE_OBJECTS on a 60x60 grid can emit ~10+ objects
        # × ~100 tokens each.  At 600 the reply truncates mid-object,
        # breaking JSON parsing and silently installing zero claims
        # (diagnosed in L1 loop attempt 1, 2026-04-17).
        reply = self.chat(messages, max_tokens=2000, temperature=0.0)
        latency = (time.perf_counter() - t0) * 1000.0
        self.usage.observer_calls += 1
        self.usage.observer_latencies_ms.append(latency)
        return parse_answer(query, reply)

    def answer_mediator_query(self, query: MediatorQuery) -> MediatorAnswer:
        if not self.mediator_budget_available():
            self.usage.mediator_exhausted_returns += 1
            return MediatorAnswer(
                query_id    = query.query_id,
                confidence  = 0.0,
                explanation = "mediator budget exhausted",
            )
        from ..mediator import prompt_for, parse_answer
        messages = prompt_for(query)
        import time
        t0 = time.perf_counter()
        reply = self.chat(messages, max_tokens=1500, temperature=0.0)
        latency = (time.perf_counter() - t0) * 1000.0
        self.usage.mediator_calls += 1
        self.usage.mediator_latencies_ms.append(latency)
        return parse_answer(query, reply)
