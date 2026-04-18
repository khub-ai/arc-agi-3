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
    AtPosition,
    CausalClaim,
    Claim,
    EntityInState,
    MediatorAnswer,
    MediatorQuery,
    MediatorQuestion,
    PropertyClaim,
    ResourceAbove,
    ResourceBelow,
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
    elif query.question == MediatorQuestion.PROPOSE_GOAL_LINKAGE:
        body = _body_propose_goal_linkage(query, summary_json)
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


def _body_propose_goal_linkage(query: MediatorQuery, summary_json: str) -> str:
    """Ask for CausalClaim links from concrete triggers to an abstract
    effect — the bridging concept the engine needs to decompose an
    adapter-seeded resource goal into actionable subgoals.

    The reply schema is deliberately narrow: each ``trigger`` must be
    one of the Conditions the engine actually knows how to evaluate
    (``AtPosition`` / ``EntityInState``), and the ``effect`` must
    exactly mirror the target goal's leaf condition (the parser
    verifies this and drops mismatches — a wrong effect would fail
    ``derive_subgoals_from_causal``'s canonical-key match anyway).
    """
    target_effects = _extract_focus_goal_effects(query)
    effect_hint = ""
    if target_effects:
        # Include the concrete expected effect(s) inline so the LLM
        # knows exactly what to place in `effect` — preventing
        # plausible-but-useless variants like ResourceAbove with the
        # wrong threshold.
        effect_hint = (
            "\nEXPECTED_EFFECT (each returned CausalClaim MUST use "
            f"exactly one of these as its `effect`): {json.dumps(target_effects)}"
        )

    schema = {
        "causal_links": (
            "array of objects.  Each object has the keys:\n"
            "  - trigger: {kind: \"AtPosition\", entity_id: string, pos: [x, y]}\n"
            "             or {kind: \"EntityInState\", entity_id: string, "
            "property: string, value: any JSON scalar}\n"
            "  - effect:  {kind: \"ResourceAbove\"|\"ResourceBelow\", "
            "resource_id: string, threshold: number}\n"
            "  - min_occurrences: integer >= 1 (default 1)\n"
            "  - delay:           integer >= 0 (default 0)\n"
            "Each link asserts `when trigger holds, effect becomes true`."
        ),
        "confidence":  "float in [0,1]",
        "explanation": "short string; one-sentence justification",
    }
    instructions = (
        "The engine has a top-level goal whose only leaf is an abstract "
        "resource predicate.  To decompose it into actionable subgoals, "
        "propose CausalClaims whose `effect` matches that leaf exactly "
        "and whose `trigger` is a concrete atomic condition the engine "
        "can evaluate.  For grid-style games the most useful trigger is "
        "usually AtPosition — pick the cell(s) where reaching them "
        "causes the win predicate to flip.  Prefer EXISTING entity IDs "
        "from WORLD_SUMMARY.entities as the trigger's `entity_id` "
        "(typically the controlled agent); the engine treats the "
        "literal string \"agent\" as the default actor when no id is "
        "known.  If several distinct triggers plausibly cause the "
        "effect, list each as a separate element of causal_links."
    ) + effect_hint
    return _compose_body(
        question_name = "PROPOSE_GOAL_LINKAGE",
        instructions  = instructions,
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
        # Agent may carry non-JSON-safe values (e.g. set-valued
        # _actions_tried), so always round-trip through _primitive.
        "agent": _primitive(dict(summary.agent)),
    }

    out["entities"] = {
        eid: {
            "kind":       ent.kind,
            "properties": _primitive(dict(ent.properties)),
            "first_seen": ent.first_seen_step,
            "last_seen":  ent.last_seen_step,
        }
        for eid, ent in summary.entities.items()
    }

    out["committed_hypotheses"] = [
        {"id": h.id, "claim": type(h.claim).__name__, "credence": h.credence.point}
        for h in summary.committed_hypotheses
    ]
    out["contested_hypotheses"] = [
        {"id": h.id, "claim": type(h.claim).__name__, "credence": h.credence.point}
        for h in summary.contested_hypotheses
    ]

    out["active_goals"] = [
        {
            "id":        g.id,
            "priority":  g.priority,
            "status":    g.status.value,
            # Surface the root-leaf condition so Mediator questions
            # (notably PROPOSE_GOAL_LINKAGE) can see what the goal is
            # actually pointing at.  Composite roots serialise as
            # a type label only.
            "root":      _serialise_goal_root(g.root),
        }
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


def _serialise_goal_root(root: Any) -> Dict[str, Any]:
    """Flatten a GoalNode's root into JSON primitives suitable for
    inclusion in the world summary.

    Only ATOM leaves have a fully-serialised condition; composites
    surface their node_type and child count so the LLM can still
    reason about the tree shape.
    """
    out: Dict[str, Any] = {"node_type": root.node_type.value}
    if root.condition is not None:
        out["condition"] = _serialise_condition(root.condition)
    if root.children:
        out["n_children"] = len(root.children)
    return out


def _serialise_condition(cond: Any) -> Dict[str, Any]:
    """Best-effort JSON dump of a Condition instance.  Typed fields
    for the conditions the Mediator is likely to encounter; generic
    repr fallback otherwise so the LLM at least sees *something*."""
    if isinstance(cond, ResourceAbove):
        return {"kind": "ResourceAbove",
                "resource_id": cond.resource_id,
                "threshold":   float(cond.threshold)}
    if isinstance(cond, ResourceBelow):
        return {"kind": "ResourceBelow",
                "resource_id": cond.resource_id,
                "threshold":   float(cond.threshold)}
    if isinstance(cond, AtPosition):
        return {"kind": "AtPosition",
                "entity_id": cond.entity_id,
                "pos":       list(cond.pos)}
    if isinstance(cond, EntityInState):
        return {"kind": "EntityInState",
                "entity_id": cond.entity_id,
                "property":  cond.property,
                "value":     _primitive(cond.value)}
    return {"kind": type(cond).__name__, "repr": repr(cond)[:80]}


def _extract_focus_goal_effects(query: MediatorQuery) -> List[Dict[str, Any]]:
    """For each goal id in ``focus_goals``, return the serialised
    condition of its root-ATOM leaf (when present in the summary).

    This is what lets :func:`_body_propose_goal_linkage` pin the
    effect the LLM should bind to — preventing confident-but-wrong
    answers that nominate a slightly different threshold or
    resource name.
    """
    if not query.focus_goals:
        return []
    ids = set(query.focus_goals)
    effects: List[Dict[str, Any]] = []
    for goal in query.world_summary.active_goals:
        if goal.id not in ids:
            continue
        root = goal.root
        if root.condition is None:
            continue
        effects.append(_serialise_condition(root.condition))
    return effects


def _primitive(v: Any) -> Any:
    """Best-effort coercion to JSON-serialisable primitives."""
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, (list, tuple)):
        return [_primitive(x) for x in v]
    if isinstance(v, (set, frozenset)):
        # JSON has no set type — serialise as a sorted list so the
        # LLM sees stable ordering.  (sorted() falls back to repr on
        # mixed-type sets; catch that and just listify.)
        try:
            return [_primitive(x) for x in sorted(v)]
        except TypeError:
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

    if query.question == MediatorQuestion.PROPOSE_GOAL_LINKAGE:
        links = obj.get("causal_links", [])
        if not isinstance(links, list):
            links = []
        proposed: List[Claim] = []
        for link in links:
            if not isinstance(link, dict):
                continue
            claim = _build_causal_claim(link)
            if claim is not None:
                proposed.append(claim)
        return MediatorAnswer(
            query_id        = query.query_id,
            proposed_claims = proposed,
            confidence      = max(0.0, min(1.0, confidence)),
            explanation     = explanation,
        )

    # Unsupported question types — zero-confidence, no claims.
    return MediatorAnswer(
        query_id    = query.query_id,
        confidence  = 0.0,
        explanation = explanation or "mediator: question type not yet implemented",
    )


def _build_causal_claim(link: Dict[str, Any]) -> Optional[CausalClaim]:
    """Build a CausalClaim from a decoded ``causal_links`` entry.

    Returns None on any malformed entry — the engine-side handler
    rejects non-matching claims regardless, but dropping them here
    keeps the hypothesis store free of junk from LLM typos.
    """
    trig = _build_condition(link.get("trigger"))
    eff  = _build_condition(link.get("effect"))
    if trig is None or eff is None:
        return None
    # Only ResourceAbove/ResourceBelow are sensible effects for the
    # goal-linkage question; guard here so a mislabelled trigger/effect
    # swap doesn't silently commit a nonsense claim.
    if not isinstance(eff, (ResourceAbove, ResourceBelow)):
        return None
    try:
        min_occ = int(link.get("min_occurrences", 1))
    except (TypeError, ValueError):
        min_occ = 1
    try:
        delay = int(link.get("delay", 0))
    except (TypeError, ValueError):
        delay = 0
    min_occ = max(1, min_occ)
    delay   = max(0, delay)
    return CausalClaim(
        trigger         = trig,
        effect          = eff,
        min_occurrences = min_occ,
        delay           = delay,
    )


def _build_condition(obj: Any) -> Optional[Any]:
    """Inverse of :func:`_serialise_condition` for the subset the
    Mediator is allowed to emit.  Returns None on anything
    unrecognised."""
    if not isinstance(obj, dict):
        return None
    kind = obj.get("kind")
    if kind == "AtPosition":
        pos = obj.get("pos")
        if not isinstance(pos, (list, tuple)) or len(pos) < 2:
            return None
        try:
            pos_tuple = tuple(float(p) if isinstance(p, float) else int(p)
                              for p in pos)
        except (TypeError, ValueError):
            return None
        entity_id = str(obj.get("entity_id", "agent")) or "agent"
        return AtPosition(pos=pos_tuple, entity_id=entity_id)
    if kind == "EntityInState":
        entity_id = obj.get("entity_id")
        prop      = obj.get("property")
        if not isinstance(entity_id, str) or not isinstance(prop, str):
            return None
        return EntityInState(
            entity_id = entity_id,
            property  = prop,
            value     = obj.get("value"),
        )
    if kind == "ResourceAbove":
        rid = obj.get("resource_id")
        thr = obj.get("threshold")
        if not isinstance(rid, str):
            return None
        try:
            return ResourceAbove(resource_id=rid, threshold=float(thr))
        except (TypeError, ValueError):
            return None
    if kind == "ResourceBelow":
        rid = obj.get("resource_id")
        thr = obj.get("threshold")
        if not isinstance(rid, str):
            return None
        try:
            return ResourceBelow(resource_id=rid, threshold=float(thr))
        except (TypeError, ValueError):
            return None
    return None


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
