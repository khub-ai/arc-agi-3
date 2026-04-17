"""Observer — the visual oracle.

The Observer's job is to answer :class:`cognitive_os.ObserverQuery`
objects by inspecting raw frames.  The engine never sees pixels; it
asks typed visual questions (``STILL_SIMILAR``, ``CLASSIFY``,
``DESCRIBE``, ``COMPARE``, ``STRUCTURE_MAP``) and receives typed
answers.

This module is backend-agnostic: it produces the prompt (a list of
:class:`ChatMessage`) and parses the reply back into a typed
:class:`ObserverAnswer`.  The prompt / parse split lets us swap the
underlying LLM (Claude → open-source) without touching either
half.

Phase 5b-i implements three question types fully:

* ``STILL_SIMILAR``
* ``CLASSIFY``
* ``DESCRIBE``

The other two (``COMPARE``, ``STRUCTURE_MAP``) return a
zero-confidence answer with an explicit "not yet implemented" note;
the engine's budget and evidence machinery handle that uniformly.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Sequence

from cognitive_os import (
    ObserverAnswer,
    ObserverQuery,
    QuestionType,
)

from .backends.base import ChatMessage


# Observer system prompt — deliberately terse.  The LLM is acting as
# a narrow pattern-recogniser, not as a general reasoner; keeping the
# framing small reduces latency and the chance of the model
# editorialising into decision-path text.
_SYSTEM_PROMPT = """\
You are a visual pattern-matching subsystem for a symbolic reasoning engine.
You receive a small 2-D integer grid (each cell is a palette colour 0-15)
and a narrowly-scoped question.  You MUST reply with valid JSON and nothing
else — no surrounding prose, no code fences.

Your answers feed a hypothesis store; they are treated as evidence, not
commands.  A confidence you are not sure of should be reported honestly —
a low confidence is useful to the engine; a fabricated high confidence is
harmful.
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def prompt_for(query: ObserverQuery) -> List[ChatMessage]:
    """Build the chat messages for ``query``.

    All question types share the system prompt; the user message is
    specialised per question.  Unsupported questions produce a user
    message that asks for a null answer — the parser then returns
    zero confidence.
    """
    if query.question == QuestionType.STILL_SIMILAR:
        body = _body_still_similar(query)
    elif query.question == QuestionType.CLASSIFY:
        body = _body_classify(query)
    elif query.question == QuestionType.DESCRIBE:
        body = _body_describe(query)
    else:
        # COMPARE and STRUCTURE_MAP land here in Phase 5b-i.
        body = _body_unsupported(query)
    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user",   content=body),
    ]


def _body_still_similar(query: ObserverQuery) -> str:
    return _compose_body(
        question     = "STILL_SIMILAR",
        description  = (
            "Decide whether the target entities in the provided frames still "
            "look visually similar (same colour scheme, same rough shape, "
            "same relative size)."
        ),
        reply_schema = {
            "result":      "boolean — true if still similar, false otherwise",
            "confidence":  "float in [0,1]",
            "explanation": "short string; one sentence max",
        },
        query        = query,
    )


def _body_classify(query: ObserverQuery) -> str:
    return _compose_body(
        question     = "CLASSIFY",
        description  = (
            "Assign each target entity to a coarse visual category.  Use a "
            "short lowercase noun phrase; pick a category per entity even if "
            "unsure (report lower confidence instead of refusing)."
        ),
        reply_schema = {
            "result":      "object mapping entity_id -> category string",
            "confidence":  "float in [0,1]",
            "explanation": "short string; one sentence max",
        },
        query        = query,
    )


def _body_describe(query: ObserverQuery) -> str:
    return _compose_body(
        question     = "DESCRIBE",
        description  = (
            "Describe what is visually salient about the target entities in the "
            "provided frames.  This is a logging / audit channel — the engine "
            "does not parse free-form descriptions."
        ),
        reply_schema = {
            "result":      "string description (up to 120 characters)",
            "confidence":  "float in [0,1]",
            "explanation": "string; reasoning if relevant, otherwise empty",
        },
        query        = query,
    )


def _body_unsupported(query: ObserverQuery) -> str:
    return _compose_body(
        question     = query.question.value,
        description  = (
            "This question type is not yet implemented by this Observer.  "
            "Reply with a null result and confidence 0."
        ),
        reply_schema = {
            "result":      "null",
            "confidence":  "0",
            "explanation": "short acknowledgement string",
        },
        query        = query,
    )


def _compose_body(
    *,
    question:     str,
    description:  str,
    reply_schema: Dict[str, str],
    query:        ObserverQuery,
) -> str:
    parts: List[str] = [
        f"QUESTION_TYPE: {question}",
        f"DESCRIPTION: {description}",
        f"TARGET_ENTITY_IDS: {json.dumps(query.targets)}",
    ]
    if query.context:
        parts.append(f"CONTEXT: {query.context}")
    parts.append(_render_frames(query.frames))
    parts.append("REPLY_SCHEMA: " + json.dumps(reply_schema))
    parts.append("Reply with a single JSON object matching REPLY_SCHEMA.")
    return "\n\n".join(parts)


def _render_frames(frames: Sequence[Any]) -> str:
    """Render frames for the LLM.

    For ARC-AGI-3 the frames are 2-D integer grids; we serialise each
    as JSON on a single line for compactness.  A large grid could be
    rendered as ASCII with one character per cell, but Claude handles
    JSON lists well and the token count on a 60×60 palette grid is
    manageable (< 4k tokens).
    """
    lines = ["FRAMES:"]
    for i, frame in enumerate(frames):
        lines.append(f"FRAME[{i}]:")
        lines.append(json.dumps(_normalise_frame(frame)))
    return "\n".join(lines)


def _normalise_frame(frame: Any) -> List[List[int]]:
    """Best-effort coercion of a frame to a list-of-lists of ints."""
    if frame is None:
        return []
    # Most ARC-AGI-3 frames are already list-of-list-of-int.
    try:
        return [[int(c) for c in row] for row in frame]
    except TypeError:
        return []


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(query: ObserverQuery, reply: str) -> ObserverAnswer:
    """Parse an LLM reply string into a typed :class:`ObserverAnswer`.

    Robust against the model adding incidental prose around a JSON
    blob; we extract the largest ``{...}`` span and try to parse it.
    On parse failure we return a zero-confidence answer with the
    raw reply truncated into the explanation — the engine treats
    that as "no answer" and proceeds.
    """
    obj = _extract_json(reply)
    if obj is None:
        return ObserverAnswer(
            query_id    = query.query_id,
            result      = None,
            confidence  = 0.0,
            explanation = f"parse_error: {reply[:120]}",
        )

    result      = obj.get("result")
    confidence  = _coerce_float(obj.get("confidence"), default=0.0)
    explanation = str(obj.get("explanation", ""))[:240]

    # Per-question-type shape coercion.  We don't hard-fail on shape
    # mismatch because the engine is tolerant of zero-confidence
    # answers; we just drop confidence to reflect uncertainty.
    if query.question == QuestionType.STILL_SIMILAR:
        if not isinstance(result, bool):
            if isinstance(result, str):
                low = result.strip().lower()
                result = (low == "true")
            else:
                result = None
                confidence = min(confidence, 0.1)
    elif query.question == QuestionType.CLASSIFY:
        if not isinstance(result, dict):
            result = None
            confidence = min(confidence, 0.1)
    elif query.question == QuestionType.DESCRIBE:
        if not isinstance(result, str):
            result = str(result) if result is not None else ""
    else:
        # Unsupported question types always degrade to zero-confidence.
        confidence = 0.0
        if not explanation:
            explanation = "observer: question type not yet implemented"

    return ObserverAnswer(
        query_id    = query.query_id,
        result      = result,
        confidence  = max(0.0, min(1.0, confidence)),
        explanation = explanation,
    )


def _extract_json(reply: str) -> Optional[Dict[str, Any]]:
    reply = reply.strip()
    if not reply:
        return None
    # Fast path: reply is pure JSON.
    try:
        obj = json.loads(reply)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    # Fallback: extract the largest braced span and parse it.
    match = _JSON_BLOCK_RE.search(reply)
    if match is None:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _coerce_float(x: Any, *, default: float) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default
