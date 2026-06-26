"""Telemetry wire schema — the event taxonomy streamed by the engine.

Every observable state change in the engine is expressible as one of the
frozen dataclasses defined in this module.  A :class:`TelemetryEnvelope`
wraps each event with routing metadata (episode, step, subject id,
monotonic timestamp) so clients — live GUI, replay tool, offline
analysis — share one wire protocol.

Standing directives
-------------------

1. **Framework-level only.**  Nothing in this module may mention a
   specific domain (ARC, robotics, a particular game).  Events carry
   primitives (ids, kinds, numbers, short strings); domain semantics
   live in the adapter and are opaque to the schema.

2. **Stream carries change, not state.**  Events are deltas.  Full
   :class:`WorldState` is never serialised here.  Clients that need a
   snapshot request one separately from the sidecar.

3. **Stable identity.**  Every event carries the id of the engine
   object it concerns (``hypothesis_id``, ``goal_id``, entity id,
   ``plan_id``).  Animation in the client depends on these ids
   remaining constant across updates — a credence change must move the
   existing dot, not recreate it.

4. **Self-contained payloads.**  A single event should render without
   the client having to reconstruct earlier state.  Deltas carry the
   *old* and *new* values, not just the delta.

5. **Additive evolution.**  Minor schema versions add fields only;
   clients ignore unknown fields.  Only a major bump (2.0, 3.0, ...)
   may remove or retype fields.

6. **No engine type imports at the payload layer.**  Payloads reference
   engine objects by id and record enum values as strings so the wire
   format stays stable even when engine internals evolve.
"""

from __future__ import annotations

import base64
import os
import secrets
import time
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type


# ===========================================================================
# Schema version
# ===========================================================================

SCHEMA_VERSION = "1.1"
"""Wire schema version.  Bump the minor for additive changes (new event
types, new optional fields); bump the major for any breaking change
(removed or retyped fields, renamed types)."""


# ===========================================================================
# Envelope
# ===========================================================================


@dataclass(frozen=True)
class TelemetryEnvelope:
    """Transport wrapper for a single event.

    One NDJSON line = one ``TelemetryEnvelope`` serialised with
    :func:`envelope_to_dict`.  Clients deserialise with
    :func:`envelope_from_dict`.

    Fields
    ------
    v
        Schema version string.  Matched against :data:`SCHEMA_VERSION`
        on the client.
    id
        Globally-unique event id.  Format: ``"<session>-<seq>"`` where
        ``<session>`` is a 6-character base62 token minted once per
        sink and ``<seq>`` is a monotonic counter.  Sortable within a
        session, unique across sessions.
    ts
        Milliseconds since the sink started.  Monotonic, drift-free —
        use this for animation timing rather than ``wall``.
    wall
        ISO-8601 wall-clock timestamp.  Present for human readability
        and cross-session alignment only; never for timing logic.
    episode
        Id of the running episode, if any.  ``None`` before the first
        :class:`EpisodeBegin`.
    step
        Step index within the current episode, if any.
    type
        Event type tag — matches the ``TYPE`` class variable of the
        payload class.  Used by :func:`envelope_from_dict` to route
        deserialisation.
    subject
        Id of the primary object this event concerns (hypothesis id,
        goal id, entity id, plan id, ...).  Clients index by this to
        build per-subject history.  ``None`` for events with no single
        subject (e.g. :class:`StepBegin`, :class:`Heartbeat`).
    prev
        Id of the previous event whose ``subject`` matches this one.
        Forms a backward-linked chain per subject so clients can walk
        refinement history without a full index pass.
    payload
        Serialised event dataclass (via :func:`asdict`), excluding the
        ``TYPE`` class variable.
    """

    v:       str
    id:      str
    ts:      float
    wall:    str
    type:    str
    payload: Dict[str, Any]
    episode: Optional[str] = None
    step:    Optional[int] = None
    subject: Optional[str] = None
    prev:    Optional[str] = None


def envelope_to_dict(env: TelemetryEnvelope) -> Dict[str, Any]:
    """Serialise an envelope to a plain dict suitable for ``json.dumps``.

    ``None``-valued optional fields are omitted from the output to keep
    NDJSON lines compact.
    """
    out: Dict[str, Any] = {
        "v":       env.v,
        "id":      env.id,
        "ts":      env.ts,
        "wall":    env.wall,
        "type":    env.type,
        "payload": env.payload,
    }
    if env.episode is not None:
        out["episode"] = env.episode
    if env.step is not None:
        out["step"] = env.step
    if env.subject is not None:
        out["subject"] = env.subject
    if env.prev is not None:
        out["prev"] = env.prev
    return out


def envelope_from_dict(data: Dict[str, Any]) -> TelemetryEnvelope:
    """Construct an envelope from a decoded NDJSON line.

    Unknown top-level fields are ignored to preserve forward
    compatibility under additive schema evolution.
    """
    return TelemetryEnvelope(
        v       = data["v"],
        id      = data["id"],
        ts      = data["ts"],
        wall    = data["wall"],
        type    = data["type"],
        payload = data.get("payload", {}),
        episode = data.get("episode"),
        step    = data.get("step"),
        subject = data.get("subject"),
        prev    = data.get("prev"),
    )


# ===========================================================================
# Id generation
# ===========================================================================


_BASE62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"


def _base62(n: int, length: int) -> str:
    """Encode ``n`` in base62, left-padded to ``length`` characters."""
    if n < 0:
        raise ValueError("base62 requires a non-negative integer")
    chars: List[str] = []
    while n > 0:
        n, rem = divmod(n, 62)
        chars.append(_BASE62_ALPHABET[rem])
    s = "".join(reversed(chars)) or "0"
    return s.rjust(length, "0")


def mint_session_id() -> str:
    """Mint a fresh 6-character base62 session id.

    Chosen for log readability over uniqueness strength; 62^6 ≈ 5.7e10
    is sufficient to avoid accidental collisions between concurrent
    training runs while keeping event ids short in dashboards.
    """
    raw = secrets.token_bytes(5)
    n = int.from_bytes(raw, "big") % (62 ** 6)
    return _base62(n, 6)


# ===========================================================================
# Event registry
# ===========================================================================


_event_registry: Dict[str, Type[Any]] = {}


def register_event(cls: Type[Any]) -> Type[Any]:
    """Decorator — add a payload class to the deserialisation registry.

    The class must set a ``TYPE`` class variable equal to the string
    used on the wire.
    """
    tag = getattr(cls, "TYPE", None)
    if not isinstance(tag, str) or not tag:
        raise TypeError(
            f"{cls.__name__} lacks a non-empty string TYPE class variable"
        )
    if tag in _event_registry and _event_registry[tag] is not cls:
        raise ValueError(f"duplicate telemetry event type tag: {tag!r}")
    _event_registry[tag] = cls
    return cls


def event_class_for(tag: str) -> Optional[Type[Any]]:
    """Return the payload dataclass registered for ``tag``, or ``None``."""
    return _event_registry.get(tag)


def registered_event_types() -> Tuple[str, ...]:
    """Return all registered event type tags in registration order."""
    return tuple(_event_registry.keys())


def payload_to_dict(payload: Any) -> Dict[str, Any]:
    """Serialise a payload dataclass to a plain dict.

    ``ClassVar`` fields (notably ``TYPE``) are excluded automatically
    by :func:`dataclasses.asdict`.  Raises :class:`TypeError` if
    ``payload`` is not a dataclass instance.
    """
    if not is_dataclass(payload) or isinstance(payload, type):
        raise TypeError(
            f"expected a dataclass instance, got {type(payload).__name__}"
        )
    return asdict(payload)


def payload_from_dict(tag: str, data: Dict[str, Any]) -> Any:
    """Reconstruct a payload dataclass from ``(tag, data)``.

    Fields present in ``data`` but not declared on the dataclass are
    ignored (additive-evolution rule).  Missing required fields raise
    :class:`TypeError`.
    """
    cls = event_class_for(tag)
    if cls is None:
        raise KeyError(f"unknown telemetry event type: {tag!r}")
    field_names = {f.name for f in fields(cls)}
    kept = {k: v for k, v in data.items() if k in field_names}
    return cls(**kept)


# ===========================================================================
# Lifecycle events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class EpisodeBegin:
    TYPE: ClassVar[str] = "EpisodeBegin"
    episode_id:      str
    adapter_kind:    str
    operating_mode:  str
    seed:            Optional[int] = None


@register_event
@dataclass(frozen=True)
class EpisodeEnd:
    TYPE: ClassVar[str] = "EpisodeEnd"
    episode_id:   str
    final_status: str
    total_steps:  int
    wall_seconds: float


@register_event
@dataclass(frozen=True)
class LevelChanged:
    TYPE: ClassVar[str] = "LevelChanged"
    from_level: Optional[str]
    to_level:   str


@register_event
@dataclass(frozen=True)
class Reset:
    TYPE: ClassVar[str] = "Reset"
    reason: str


@register_event
@dataclass(frozen=True)
class StepBegin:
    TYPE: ClassVar[str] = "StepBegin"


@register_event
@dataclass(frozen=True)
class StepEnd:
    TYPE: ClassVar[str] = "StepEnd"
    planner_latency_ms: float
    total_latency_ms:   float
    action_kind:        Optional[str] = None
    plan_id:            Optional[str] = None


# ===========================================================================
# Perception events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class ObservationIngested:
    TYPE: ClassVar[str] = "ObservationIngested"
    raw_frame_ref: Optional[str]   # hash id; client fetches via sidecar
    entity_count:  int
    event_count:   int


@register_event
@dataclass(frozen=True)
class EventEmitted:
    """Generic wrapper for adapter-emitted :class:`Event` subclasses.

    ``event_type`` is the event class name (e.g. ``"AgentMoved"``);
    ``data`` is a domain-opaque payload dict.
    """
    TYPE: ClassVar[str] = "EventEmitted"
    event_type: str
    data:       Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Belief events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class HypothesisAdded:
    TYPE: ClassVar[str] = "HypothesisAdded"
    hypothesis_id:    str
    claim_type:       str
    scope_kind:       str
    source:           str
    initial_credence: float
    canonical_key:    str


@register_event
@dataclass(frozen=True)
class HypothesisCredenceUpdated:
    """Credence-update event with provenance fields (schema 1.1+).

    The original four fields (``hypothesis_id``, ``old_credence``,
    ``new_credence``, ``evidence_delta``, ``reason``) preserve the 1.0
    contract. The extension fields below are additive — older clients
    decoding 1.1 envelopes ignore them; newer clients reading 1.0
    payloads see the defaults.

    Provenance fields (1.1)
    -----------------------
    direction
        ``"support"``, ``"contradict"``, or ``"decay"``. Redundant with
        the prefix of ``reason`` but pulled out for fast indexing.
    source_kind
        Coarse category of the evidence source: ``"event"``,
        ``"propose"``, ``"decay"``, ``"observer"``, ``"user"``,
        ``"adapter"``, ``"miner"``. Used by the inspector to group
        provenance entries.
    source_detail
        Free-form id qualifying the source: an event-class name
        (``"AgentMoved"``), a miner name (``"miner:FutilePattern"``),
        or a propose-tag (``"duplicate-propose:adapter:seed"``).
    source_strength
        The ``s`` parameter passed to the credence update rule; the
        learning-rate multiplier in [0, 1].
    learning_rate
        ``cfg.learning_rate`` in effect when the update was applied.
        Recorded so replays remain interpretable across config changes.
    evidence_weight_before
        Cumulative supporting weight before this update.
    evidence_weight_after
        Cumulative supporting weight after this update. (Contradictions
        do not increment the weight by design — see ``credence.py``.)
    triggering_event_step
        Step at which the originating event was emitted, when known and
        different from the update step. ``None`` for proposals and
        decay sweeps where there is no separable triggering event.
    """

    TYPE: ClassVar[str] = "HypothesisCredenceUpdated"
    hypothesis_id:           str
    old_credence:            float
    new_credence:            float
    evidence_delta:          float
    reason:                  str
    # --- 1.1 additive provenance fields ---
    direction:               str           = ""
    source_kind:             str           = ""
    source_detail:           str           = ""
    source_strength:         float         = 0.0
    learning_rate:           float         = 0.0
    evidence_weight_before:  float         = 0.0
    evidence_weight_after:   float         = 0.0
    triggering_event_step:   Optional[int] = None


@register_event
@dataclass(frozen=True)
class HypothesisSpecialised:
    TYPE: ClassVar[str] = "HypothesisSpecialised"
    parent_id:            str
    child_id:             str
    added_condition_kind: str


@register_event
@dataclass(frozen=True)
class HypothesisRetired:
    TYPE: ClassVar[str] = "HypothesisRetired"
    hypothesis_id: str
    reason:        str


# ===========================================================================
# Intention events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class GoalAdded:
    TYPE: ClassVar[str] = "GoalAdded"
    goal_id:           str
    priority:          float
    root_node_type:    str
    condition_summary: str
    parent_goal_id:    Optional[str] = None


@register_event
@dataclass(frozen=True)
class GoalDerived:
    TYPE: ClassVar[str] = "GoalDerived"
    parent_id:       str
    child_id:        str
    derivation_kind: str


@register_event
@dataclass(frozen=True)
class GoalStatusChanged:
    TYPE: ClassVar[str] = "GoalStatusChanged"
    goal_id:    str
    old_status: str
    new_status: str


@register_event
@dataclass(frozen=True)
class ActiveGoalChanged:
    TYPE: ClassVar[str] = "ActiveGoalChanged"
    old_goal_id: Optional[str]
    new_goal_id: Optional[str]


@register_event
@dataclass(frozen=True)
class ConflictDetected:
    TYPE: ClassVar[str] = "ConflictDetected"
    conflict_type: str
    goal_ids:      List[str]


# ===========================================================================
# Action events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class PlanComputed:
    TYPE: ClassVar[str] = "PlanComputed"
    plan_id:       str
    goal_id:       str
    step_count:    int
    expected_cost: float
    head_action:   Optional[str]


@register_event
@dataclass(frozen=True)
class PlanInvalidated:
    TYPE: ClassVar[str] = "PlanInvalidated"
    plan_id: str
    reason:  str


@register_event
@dataclass(frozen=True)
class PlanExhausted:
    TYPE: ClassVar[str] = "PlanExhausted"
    plan_id: str


@register_event
@dataclass(frozen=True)
class ActionSelected:
    TYPE: ClassVar[str] = "ActionSelected"
    action_kind: str
    source:      str   # "plan" | "explore" | "probe" | "option"
    plan_id:     Optional[str] = None


@register_event
@dataclass(frozen=True)
class ActionExecuted:
    TYPE: ClassVar[str] = "ActionExecuted"
    action_kind: str
    success:     bool
    duration_ms: float


@register_event
@dataclass(frozen=True)
class ExploreFallback:
    TYPE: ClassVar[str] = "ExploreFallback"
    reason:         str
    chosen_action:  Optional[str]


# ===========================================================================
# Anomaly events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class SurpriseEventRaised:
    TYPE: ClassVar[str] = "SurpriseEventRaised"
    surprise_kind: str
    entity_id:     Optional[str] = None
    details:       Dict[str, Any] = field(default_factory=dict)


@register_event
@dataclass(frozen=True)
class FutilePatternDetected:
    TYPE: ClassVar[str] = "FutilePatternDetected"
    pattern_key: str
    count:       int


@register_event
@dataclass(frozen=True)
class MinerFinding:
    TYPE: ClassVar[str] = "MinerFinding"
    miner_name:   str
    finding_kind: str
    details:      Dict[str, Any] = field(default_factory=dict)


# ===========================================================================
# Oracle events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class ObserverQueryFired:
    TYPE: ClassVar[str] = "ObserverQueryFired"
    query_id:      str
    question_type: str
    frame_ref:     Optional[str]


@register_event
@dataclass(frozen=True)
class ObserverAnswerReceived:
    TYPE: ClassVar[str] = "ObserverAnswerReceived"
    query_id:    str
    parsed_kind: str        # "ok" | "parse_error" | "refused"
    cache_hit:   bool
    latency_ms:  float
    cache_key:   Optional[str] = None


@register_event
@dataclass(frozen=True)
class MediatorQueryFired:
    TYPE: ClassVar[str] = "MediatorQueryFired"
    query_id:      str
    question_type: str
    summary_ref:   Optional[str]


@register_event
@dataclass(frozen=True)
class MediatorAnswerReceived:
    TYPE: ClassVar[str] = "MediatorAnswerReceived"
    query_id:    str
    parsed_kind: str
    cache_hit:   bool
    latency_ms:  float
    cache_key:   Optional[str] = None


# ===========================================================================
# Meta / cross-episode events
# ===========================================================================


@register_event
@dataclass(frozen=True)
class OptionSynthesised:
    TYPE: ClassVar[str] = "OptionSynthesised"
    option_id:       str
    scope_kind:      str
    source_plan_ids: List[str]


@register_event
@dataclass(frozen=True)
class OptionUsed:
    TYPE: ClassVar[str] = "OptionUsed"
    option_id: str


@register_event
@dataclass(frozen=True)
class PostMortemProduced:
    TYPE: ClassVar[str] = "PostMortemProduced"
    episode_id:     str
    lessons_count:  int
    options_count:  int


@register_event
@dataclass(frozen=True)
class RuleLearned:
    TYPE: ClassVar[str] = "RuleLearned"
    rule_id:            str
    principal_kind:     str
    constraint_summary: str


@register_event
@dataclass(frozen=True)
class RuleRetired:
    TYPE: ClassVar[str] = "RuleRetired"
    rule_id: str
    reason:  str


@register_event
@dataclass(frozen=True)
class Heartbeat:
    TYPE: ClassVar[str] = "Heartbeat"
    step_rate_hz: float
    uptime_s:     float


@register_event
@dataclass(frozen=True)
class LogMessage:
    TYPE: ClassVar[str] = "LogMessage"
    level:   str   # "debug" | "info" | "warning" | "error"
    logger:  str
    message: str


@register_event
@dataclass(frozen=True)
class GapMarker:
    """Sidecar emits this when it had to drop events under backpressure.

    Clients seeing a gap marker must refetch a snapshot — any cached
    per-subject chains are no longer trustworthy across the gap.
    """
    TYPE: ClassVar[str] = "GapMarker"
    dropped_count: int
    reason:        str


__all__ = [
    "SCHEMA_VERSION",
    "TelemetryEnvelope",
    "envelope_to_dict",
    "envelope_from_dict",
    "mint_session_id",
    "register_event",
    "event_class_for",
    "registered_event_types",
    "payload_to_dict",
    "payload_from_dict",
    # lifecycle
    "EpisodeBegin", "EpisodeEnd", "LevelChanged", "Reset",
    "StepBegin", "StepEnd",
    # perception
    "ObservationIngested", "EventEmitted",
    # belief
    "HypothesisAdded", "HypothesisCredenceUpdated",
    "HypothesisSpecialised", "HypothesisRetired",
    # intention
    "GoalAdded", "GoalDerived", "GoalStatusChanged",
    "ActiveGoalChanged", "ConflictDetected",
    # action
    "PlanComputed", "PlanInvalidated", "PlanExhausted",
    "ActionSelected", "ActionExecuted", "ExploreFallback",
    # anomaly
    "SurpriseEventRaised", "FutilePatternDetected", "MinerFinding",
    # oracle
    "ObserverQueryFired", "ObserverAnswerReceived",
    "MediatorQueryFired", "MediatorAnswerReceived",
    # meta
    "OptionSynthesised", "OptionUsed", "PostMortemProduced",
    "RuleLearned", "RuleRetired", "Heartbeat", "LogMessage",
    "GapMarker",
]
