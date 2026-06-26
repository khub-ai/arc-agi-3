"""Provenance — reconstruct a hypothesis's credence history from telemetry.

Every credence-affecting event in the engine emits a
:class:`HypothesisCredenceUpdated` envelope (since schema 1.1) carrying
the source, strength, learning rate, and before/after weights. This
module walks an NDJSON telemetry log and assembles, for a given
hypothesis id, the ordered list of provenance entries that led to its
current credence — answering questions like *"why does h47 still
think the goal cell is impassable?"* or *"which event tipped h12 below
the abandon threshold?"*.

The walk uses two complementary keys per envelope:

* ``subject`` — set on every hypothesis-level event; equal to the
  hypothesis id.
* ``prev`` — set by the sink to the id of the preceding event with the
  same subject, forming a per-subject backward chain.

Both keys agree on the per-hypothesis stream, so we can either filter
by ``subject`` (cheap, single pass) or follow ``prev`` from the most
recent event (chain-walk, useful when only the latest event id is
known). The functions in this module use the filter approach because
NDJSON files are typically small enough to scan once; for very large
runs the chain-walk variant is a future optimisation.

The module is deliberately self-contained — it depends only on
:mod:`cognitive_os.telemetry` for the reader and on the schema
registry. No engine state required, so the inspector works on any
NDJSON file the sink produced, even one from a different machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

from .telemetry import read_ndjson
from .telemetry_schema import (
    HypothesisAdded,
    HypothesisCredenceUpdated,
    HypothesisRetired,
    HypothesisSpecialised,
    TelemetryEnvelope,
    payload_from_dict,
)


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProvenanceEntry:
    """One credence-affecting event in a hypothesis's history.

    The entry preserves enough of the originating envelope that an
    inspector can render a human-readable line without re-reading the
    NDJSON file. ``raw`` is the full payload dict for callers that need
    fields the dataclass doesn't expose (e.g. specialisation child id).
    """

    event_type:    str            # "HypothesisAdded" | "HypothesisCredenceUpdated" | ...
    event_id:      str            # envelope id, e.g. "abc123-1f"
    step:          Optional[int]
    direction:     str            # "init" | "support" | "contradict" | "decay" | "specialise" | "retire"
    source_kind:   str
    source_detail: str
    point_before:  Optional[float]
    point_after:   Optional[float]
    weight_before: Optional[float]
    weight_after:  Optional[float]
    triggering_event_step: Optional[int]
    raw:           dict


# ---------------------------------------------------------------------------
# Trail assembly
# ---------------------------------------------------------------------------


def _envelope_for_hypothesis(env: TelemetryEnvelope, hypothesis_id: str) -> bool:
    if env.subject == hypothesis_id:
        return True
    # Some early events might not carry subject — fall back on payload field.
    payload = env.payload or {}
    return payload.get("hypothesis_id") == hypothesis_id


def _entry_from_envelope(env: TelemetryEnvelope) -> Optional[ProvenanceEntry]:
    """Translate a relevant envelope into a :class:`ProvenanceEntry`.

    Returns ``None`` for envelope types that do not carry hypothesis
    provenance (defensive — the caller already filters by subject).
    """
    payload = env.payload or {}
    if env.type == "HypothesisAdded":
        return ProvenanceEntry(
            event_type    = env.type,
            event_id      = env.id,
            step          = env.step,
            direction     = "init",
            source_kind   = "propose",
            source_detail = str(payload.get("source", "")),
            point_before  = None,
            point_after   = float(payload.get("initial_credence")) if payload.get("initial_credence") is not None else None,
            weight_before = None,
            weight_after  = None,
            triggering_event_step = None,
            raw           = dict(payload),
        )
    if env.type == "HypothesisCredenceUpdated":
        return ProvenanceEntry(
            event_type    = env.type,
            event_id      = env.id,
            step          = env.step,
            direction     = str(payload.get("direction") or _direction_from_reason(payload.get("reason", ""))),
            source_kind   = str(payload.get("source_kind", "")),
            source_detail = str(payload.get("source_detail") or payload.get("reason", "")),
            point_before  = float(payload["old_credence"]) if "old_credence" in payload else None,
            point_after   = float(payload["new_credence"]) if "new_credence" in payload else None,
            weight_before = _opt_float(payload.get("evidence_weight_before")),
            weight_after  = _opt_float(payload.get("evidence_weight_after")),
            triggering_event_step = _opt_int(payload.get("triggering_event_step")),
            raw           = dict(payload),
        )
    if env.type == "HypothesisSpecialised":
        return ProvenanceEntry(
            event_type    = env.type,
            event_id      = env.id,
            step          = env.step,
            direction     = "specialise",
            source_kind   = "refinement",
            source_detail = f"parent={payload.get('parent_id', '')} added={payload.get('added_condition_kind', '')}",
            point_before  = None,
            point_after   = None,
            weight_before = None,
            weight_after  = None,
            triggering_event_step = None,
            raw           = dict(payload),
        )
    if env.type == "HypothesisRetired":
        return ProvenanceEntry(
            event_type    = env.type,
            event_id      = env.id,
            step          = env.step,
            direction     = "retire",
            source_kind   = "store",
            source_detail = str(payload.get("reason", "")),
            point_before  = None,
            point_after   = None,
            weight_before = None,
            weight_after  = None,
            triggering_event_step = None,
            raw           = dict(payload),
        )
    return None


def _direction_from_reason(reason: str) -> str:
    """Fallback for 1.0 envelopes that lack the ``direction`` field."""
    if reason.startswith("support"):
        return "support"
    if reason.startswith("contradict"):
        return "contradict"
    if reason.startswith("decay"):
        return "decay"
    return ""


def _opt_float(v) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(v) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def trail_from_envelopes(envelopes:     Iterable[TelemetryEnvelope],
                         hypothesis_id: str) -> List[ProvenanceEntry]:
    """Filter an envelope stream to a single hypothesis's provenance.

    The result is in emission order (oldest first); callers that want
    the latest-first view should reverse it. Specialisation child
    events are emitted on the *child* subject in the runtime, so they
    appear in the child's trail; an open question (§12 of the spec) is
    whether parents should also receive a back-pointer envelope.
    """
    out: List[ProvenanceEntry] = []
    for env in envelopes:
        if not _envelope_for_hypothesis(env, hypothesis_id):
            continue
        entry = _entry_from_envelope(env)
        if entry is not None:
            out.append(entry)
    return out


def trail_from_ndjson(path:          "str",
                      hypothesis_id: str) -> List[ProvenanceEntry]:
    """Read an NDJSON telemetry log and return one hypothesis's trail.

    Convenience wrapper over :func:`trail_from_envelopes`. The file is
    read once; lines that fail to decode are skipped (consistent with
    :func:`telemetry.read_ndjson`).
    """
    return trail_from_envelopes(read_ndjson(path), hypothesis_id)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def format_entry(entry: ProvenanceEntry) -> str:
    """Single-line human-readable rendering of a provenance entry.

    Layout: ``[step] direction source — credence_change weight_change``.
    Optional fields are elided when missing so the line stays compact
    on entries from older schema versions.
    """
    parts: List[str] = []
    step_str = f"step={entry.step}" if entry.step is not None else "step=?"
    parts.append(step_str)
    parts.append(entry.direction or "?")
    src = ":".join([s for s in (entry.source_kind, entry.source_detail) if s])
    if src:
        parts.append(src)
    if entry.point_before is not None and entry.point_after is not None:
        parts.append(f"{entry.point_before:.3f}→{entry.point_after:.3f}")
    elif entry.point_after is not None:
        parts.append(f"=({entry.point_after:.3f})")
    if entry.weight_before is not None and entry.weight_after is not None:
        if entry.weight_before != entry.weight_after:
            parts.append(f"w={entry.weight_before:.2f}→{entry.weight_after:.2f}")
    if entry.triggering_event_step is not None and entry.triggering_event_step != entry.step:
        parts.append(f"evt_step={entry.triggering_event_step}")
    return " ".join(parts)


def format_trail(trail: List[ProvenanceEntry]) -> str:
    """Multi-line rendering of a trail, oldest first."""
    return "\n".join(format_entry(e) for e in trail)


__all__ = [
    "ProvenanceEntry",
    "trail_from_envelopes",
    "trail_from_ndjson",
    "format_entry",
    "format_trail",
]
