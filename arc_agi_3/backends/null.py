"""NullBackend — the no-LLM default.

Used in CI and whenever an adapter is constructed without an
explicit backend.  Every query returns a zero-confidence answer with
a clear explanation; the engine treats that the same as "no oracle
available" and proceeds symbolically.

Having a deterministic no-op backend (rather than a ``None`` check
sprinkled through the adapter) keeps the delegation path uniform
and the test surface simple: the adapter's observer_query /
mediator_query methods always delegate, and the tests always
receive a well-typed answer.
"""

from __future__ import annotations

from cognitive_os import (
    MediatorAnswer,
    MediatorQuery,
    ObserverAnswer,
    ObserverQuery,
)

from .base import LLMBackend


class NullBackend(LLMBackend):
    """Returns zero-confidence answers without consulting any LLM."""

    def answer_observer_query(self, query: ObserverQuery) -> ObserverAnswer:
        # NullBackend does not consume budget — there's nothing to
        # account for.  Returning zero confidence is the contract.
        return ObserverAnswer(
            query_id    = query.query_id,
            result      = None,
            confidence  = 0.0,
            explanation = "NullBackend: no oracle configured",
        )

    def answer_mediator_query(self, query: MediatorQuery) -> MediatorAnswer:
        return MediatorAnswer(
            query_id    = query.query_id,
            confidence  = 0.0,
            explanation = "NullBackend: no oracle configured",
        )
