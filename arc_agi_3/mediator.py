"""Mediator — the common-sense oracle.

The Mediator answers :class:`cognitive_os.MediatorQuery` objects
given a :class:`WorldStateSummary`.  Unlike the Observer it does not
look at pixels; it consumes the engine's own symbolic digest and
returns structured :class:`MediatorAnswer`\\s with typed claims,
goals, and rules.

Phase 5b-i implements two question types fully:

* ``IDENTIFY_ROLES``     — entity_id → role label + PropertyClaims
* ``EXPLAIN_SURPRISE``   — free-text explanation (for audit only)

The remaining question types return a zero-confidence answer with a
"not yet implemented" note.  That is the same convention the engine
treats as "no answer"; adding a new question type later is a
prompt + parse addition here and nothing more.

Keeping the Mediator narrowly scoped at this phase is deliberate:
the goal of Phase 5b-i is to exercise the *seam*, not to trust the
LLM with irreversible decisions.  Richer claim emission (Causal,
Transition, Rule) lands after the budget + attribution machinery
has live traffic to calibrate against.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from cognitive_os import (
    Claim,
    MediatorAnswer,
    MediatorQuery,
    MediatorQuestion,
    PropertyClaim,
    WorldStateSummary,
)

from .backends.base import ChatMessage


_SYSTEM_PROMPT = """\
You are a common-sense reasoning subsystem for a symbolic agent.  You receive
a structured summary of the agent's current beliefs (entities with observed
properties, committed hypotheses, active goals, recent events) and a narrow
question.  You MUST reply with valid JSON and nothing else — no surrounding
prose, no code fences.

Your replies are parsed into typed engine objects and added to a hypothesis
store.  Do not hallucinate certainty: report a lower confidence when unsure.
Do not invent entity IDs — use only the IDs present in the summary.
"""


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def prompt_for(query: MediatorQuery) -> List[ChatMessage]:
    """Build the chat messages for ``query``."""
    summary_json = json.dumps(_serialise_summary(query.world_summary), indent=None)

    if query.question == MediatorQuestion.IDENTIFY_ROLES:
        body = _body_identify_roles(query, summary_json)
    elif query.question == MediatorQuestion.EXPLAIN_SURPRISE:
        body = _body_explain_surprise(query, summary_json)
    else:
        body = _body_unsupported(query, summary_json)

    return [
        ChatMessage(role="system", content=_SYSTEM_PROMPT),
        ChatMessage(role="user",   content=body),
    ]


def _body_identify_roles(query: MediatorQuery, summary_json: str) -> str:
    schema = {
        "entity_roles": (
            "object mapping entity_id -> short lowercase noun phrase describing "
            "the likely common-sense role (e.g. \"agent\", \"wall\", \"goal\", "
            "\"counter\", \"hazard\", \"resource\")"
        ),
        "confidence":   "float in [0,1]",
        "explanation":  "short string; one sentence justification",
    }
    return _compose_body(
        question_name = "IDENTIFY_ROLES",
        instructions  = (
            "Given the entity summaries and recent events, assign each entity "
            "a coarse role label.  Prefer common-sense categories.  If you are "
            "uncertain about an entity, omit it from the mapping rather than "
            "guessing."
        ),
        summary_json  = summary_json,
        schema        = schema,
        query         = query,
    )


def _body_explain_surprise(query: MediatorQuery, summary_json: str) -> str:
    schema = {
        "explanation": (
            "short free-text explanation (up to 240 characters) of what likely "
            "caused the surprise event"
        ),
        "confidence":  "float in [0,1]",
    }
    surprise_desc = ""
    if query.surprise is not None:
        surprise_desc = (
            f"\nSURPRISE: step={query.surprise.step} "
            f"expected={query.surprise.expected!r} actual={query.surprise.actual!r}"
        )
    return _compose_body(
        question_name = "EXPLAIN_SURPRISE",
        instructions  = (
            "Given the world-state summary and the described surprise, propose "
            "a plausible common-sense explanation.  This is an audit channel — "
            "do NOT invent specific entity-level claims here."
        ) + surprise_desc,
        summary_json  = summary_json,
        schema        = schema,
        query         = query,
    )


def _body_unsupported(query: MediatorQuery, summary_json: str) -> str:
    schema = {
        "confidence":  "0",
        "explanation": "short acknowledgement string",
    }
    return _compose_body(
        question_name = query.question.value,
        instructions  = (
            "This question type is not yet implemented by this Mediator.  "
            "Reply with confidence 0."
        ),
        summary_json  = summary_json,
        schema        = schema,
        query         = query,
    )


def _compose_body(
    *,
    question_name: str,
    instructions:  str,
    summary_json:  str,
    schema:        Dict[str, str],
    query:         MediatorQuery,
) -> str:
    parts: List[str] = [
        f"QUESTION_TYPE: {question_name}",
        f"INSTRUCTIONS: {instructions}",
    ]
    if query.context:
        parts.append(f"CONTEXT: {query.context}")
    if query.focus_entities:
        parts.append(f"FOCUS_ENTITY_IDS: {json.dumps(query.focus_entities)}")
    if query.focus_goals:
        parts.append(f"FOCUS_GOAL_IDS: {json.dumps(query.focus_goals)}")
    parts.append(f"WORLD_SUMMARY: {summary_json}")
    parts.append(f"REPLY_SCHEMA: {json.dumps(schema)}")
    parts.append("Reply with a single JSON object matching REPLY_SCHEMA.")
    return "\n\n".join(parts)


def _serialise_summary(summary: WorldStateSummary) -> Dict[str, Any]:
    """Convert a WorldStateSummary into JSON-safe primitives.

    The engine's objects are typed dataclasses; the LLM only needs
    the fields relevant to common-sense reasoning.  We skip
    ``raw_frame`` entirely (that belongs on the Observer path) and
    flatten the goal / rule / hypothesis objects into short dicts.
    """
    out: Dict[str, Any] = {
        "step":  summary.step,
        "agent": dict(summary.agent),
    }

    out["entities"] = {
        eid: {
            "kind":       ent.kind,
            "properties": dict(ent.properties),
            "first_seen": ent.first_seen_step,
            "last_seen":  ent.last_seen_step,
        }
        for eid, ent in summary.entities.items()
    }

    out["committed_hypotheses"] = [
        {"id": h.id, "claim": type(h.claim).__name__, "credence": h.credence.mean}
        for h in summary.committed_hypotheses
    ]
    out["contested_hypotheses"] = [
        {"id": h.id, "claim": type(h.claim).__name__, "credence": h.credence.mean}
        for h in summary.contested_hypotheses
    ]

    out["active_goals"] = [
        {"id": g.id, "priority": g.priority, "status": g.status.value}
        for g in summary.active_goals
    ]

    out["recent_events"] = [
        {"type": type(e).__name__, "step": getattr(e, "step", None),
         **{k: _primitive(v) for k, v in e.__dict__.items() if k != "step"}}
        for e in summary.recent_events
    ][-20:]   # cap to last 20 for token budget

    if summary.impasse_context:
        out["impasse_context"] = summary.impasse_context
    if summary.available_tools is not None:
        out["available_tools"] = [sig.name for sig in summary.available_tools.list_available()]

    return out


def _primitive(v: Any) -> Any:
    """Best-effort coercion to JSON-serialisable primitives."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_primitive(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _primitive(x) for k, x in v.items()}
    return repr(v)


# ---------------------------------------------------------------------------
# Reply parsing
# ---------------------------------------------------------------------------


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_answer(query: MediatorQuery, reply: str) -> MediatorAnswer:
    """Parse an LLM reply into a typed :class:`MediatorAnswer`."""
    obj = _extract_json(reply)
    if obj is None:
        return MediatorAnswer(
            query_id    = query.query_id,
            confidence  = 0.0,
            explanation = f"parse_error: {reply[:120]}",
        )

    confidence  = _coerce_float(obj.get("confidence"), default=0.0)
    explanation = str(obj.get("explanation", ""))[:480]

    if query.question == MediatorQuestion.IDENTIFY_ROLES:
        roles = obj.get("entity_roles", {})
        if not isinstance(roles, dict):
            roles = {}
        # Accept only roles referencing entities that were actually in
        # the summary — prevents the LLM from inventing entity IDs.
        valid_ids = set(query.world_summary.entities.keys())
        entity_roles: Dict[str, str] = {
            str(k): str(v) for k, v in roles.items()
            if str(k) in valid_ids and isinstance(v, (str, int, float))
        }
        proposed_claims: List[Claim] = [
            PropertyClaim(entity_id=ent_id, property="role", value=role)
            for ent_id, role in entity_roles.items()
        ]
        return MediatorAnswer(
            query_id        = query.query_id,
            proposed_claims = proposed_claims,
            entity_roles    = entity_roles,
            confidence      = max(0.0, min(1.0, confidence)),
            explanation     = explanation,
        )

    if query.question == MediatorQuestion.EXPLAIN_SURPRISE:
        return MediatorAnswer(
            query_id    = query.query_id,
            confidence  = max(0.0, min(1.0, confidence)),
            explanation = explanation or "(no explanation provided)",
        )

    # Unsupported question types — zero-confidence, no claims.
    return MediatorAnswer(
        query_id    = query.query_id,
        confidence  = 0.0,
        explanation = explanation or "mediator: question type not yet implemented",
    )


def _extract_json(reply: str) -> Optional[Dict[str, Any]]:
    reply = reply.strip()
    if not reply:
        return None
    try:
        obj = json.loads(reply)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
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
