"""VisualStore — the engine's perceptual memory of entities seen.

A `VisualStore` keeps a multi-key index of visual entities (bitmaps in
the ARC adapter; could be feature vectors / embeddings / 3-D meshes in
other domains).  Every entity registered by a domain adapter gets:

  * ``bitmap_id``  — exact pixel match (most specific)
  * ``shape_id``   — palette-permutation-tolerant
  * ``topo_id``    — binary occupancy only
  * ``scaled_id``  — extent-normalised exact pixels

Each id is a key into the same underlying record.  Lookups can be made
at any abstraction level; the strongest match available is returned.

Why an explicit component
=========================

Without a dedicated store, perceptual identity is a tacit convention
spread across miners, the goal forest, and the adapter.  Every reader
of a CausalClaim has to decide what `entity_id` means — is it
position-based?  Component-id-based?  Cross-game-stable or not?

By formalising VisualStore:

  * Entity IDs become first-class persistent identifiers, not
    incidental strings minted at observation time.
  * Multi-key abstraction is centralised; readers query the level
    that fits their use case (causal-chain matching, role-transfer,
    pattern-similarity probes).
  * Cross-session and cross-game persistence has a well-defined home
    (``save() / load()``).
  * The escape hatch to a VLM (``VisualRecognitionTrigger``) has an
    obvious owner — the store decides when to escalate.

Sibling components: HypothesisStore (claim memory), GoalManager (goal
memory), VisualStore (entity memory).  The three together form the
engine's persistent state.

Tiered perception with explicit escalation
==========================================

The store is the harness side of the pattern: cheap, deterministic,
multi-key matching that handles most cases without paying for an LLM
call.  When all four abstraction levels return novel ids (no shared
key with anything previously seen) AND the entity participates in an
active goal's preconditions AND the visual-query budget allows, the
store's owner can dispatch a ``MediatorQuestion.RECOGNIZE_ENTITY``
query carrying the bitmap rendered as a PNG; the VLM's reply produces
PropertyClaims and EntityEquivalenceClaims that bind the new entity
to known ones at semantic-similarity tiers the harness couldn't reach.

See ``docs/ARCHITECTURE_OVERVIEW.md`` for the broader pattern (any
fast harness tool + LLM escalation for residual cases).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class EntityRecord:
    """One entry in a VisualStore.

    ``bitmap_id`` is the primary key; the other ids index the same
    record under coarser equivalence classes.

    ``role_hypotheses`` stores Oracle/VLM-attached semantic labels at
    associated credence (e.g. ``{"rotation_trigger": 0.7}``).

    ``observations`` is a per-(domain, instance) audit trail
    answering "where have I seen this entity?" — useful both for
    debugging and for future-session priors.

    ``payload`` is an opaque slot for the adapter to attach the raw
    representation needed to ship the entity to a VLM (e.g. a PNG-
    encoded bitmap).
    """
    bitmap_id:        str
    shape_id:         str
    topo_id:          str
    scaled_id:        str
    annotation:       str = ""
    first_seen_step:  int = 0
    role_hypotheses:  Dict[str, float] = field(default_factory=dict)
    observations:     List[Dict[str, Any]] = field(default_factory=list)
    payload:          Optional[Any] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VisualStore(Protocol):
    """Engine-level interface for perceptual memory of entities seen.

    Domain adapters subclass with concrete bitmap / feature
    representations.  The engine never reads the underlying pixels
    — only the multi-key ids the adapter computes.
    """

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[EntityRecord]:
        """Lookup by any of the four abstraction-level ids
        (bitmap_id / shape_id / topo_id / scaled_id).  Returns the
        record whose `bitmap_id` matches *or* whose coarser ids match;
        when multiple records share a coarser key, returns the one
        with the most observations (tie-broken by first_seen_step)."""

    def lookup(self, key: str) -> List[EntityRecord]:
        """Return every record whose any of its four ids match `key`.
        Use when a coarser id may map to multiple distinct entities
        (e.g. a topo_id shared by N differently-coloured shapes)."""

    def all(self) -> List[EntityRecord]:
        """Iterate every registered entity.  Order: insertion."""

    def known_at(self, abstraction: str) -> Dict[str, List[EntityRecord]]:
        """Group the store by the requested abstraction level
        (`'bitmap_id' | 'shape_id' | 'topo_id' | 'scaled_id'`).
        The returned dict's values are lists because coarser keys can
        cover multiple distinct records."""

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def register(
        self,
        *,
        bitmap_id:  str,
        shape_id:   str,
        topo_id:    str,
        scaled_id:  str,
        annotation: str = "",
        step:       int = 0,
        domain:     str = "",
        instance:   str = "",
        payload:    Optional[Any] = None,
    ) -> EntityRecord:
        """Register an entity observation.  Idempotent on
        `bitmap_id` — a second call with the same id appends to that
        record's `observations` list rather than creating a duplicate.

        Returns the record (new or merged)."""

    def annotate_role(
        self,
        bitmap_id: str,
        role:      str,
        credence:  float,
    ) -> None:
        """Attach a role hypothesis to an entity (e.g.
        ``annotate_role("bm_3f2a...", "rotation_trigger", 0.85)``).
        Same role re-annotated raises credence to the maximum of old
        and new — caller is responsible for the credence policy."""

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the store to a JSON-friendly dict.  The adapter
        decides how `payload` (often a binary blob) is encoded — base64
        is the conventional choice."""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VisualStore":
        """Deserialise from `to_dict`'s output."""


# ---------------------------------------------------------------------------
# Default in-memory implementation
# ---------------------------------------------------------------------------


class DefaultVisualStore:
    """In-memory `VisualStore` implementation.

    Domain adapters can subclass to add domain-specific helpers
    (e.g. ``render_for_vlm`` returning a PNG of the bitmap, or
    ``feature_vector`` for similarity probes).  The engine itself
    only ever calls the methods on the Protocol, so subclasses stay
    decoupled.
    """

    def __init__(self) -> None:
        self._by_bitmap: Dict[str, EntityRecord] = {}
        # Reverse indices for fast coarser-key lookup.
        self._by_shape:   Dict[str, List[str]] = {}
        self._by_topo:    Dict[str, List[str]] = {}
        self._by_scaled:  Dict[str, List[str]] = {}

    # ---- read ----

    def get(self, key: str) -> Optional[EntityRecord]:
        rec = self._by_bitmap.get(key)
        if rec is not None:
            return rec
        # Coarser keys: pick the record with the most observations
        # (proxy for "the one we know best about").
        candidates = self.lookup(key)
        if not candidates:
            return None
        candidates.sort(
            key=lambda r: (-len(r.observations), r.first_seen_step)
        )
        return candidates[0]

    def lookup(self, key: str) -> List[EntityRecord]:
        if key in self._by_bitmap:
            return [self._by_bitmap[key]]
        bitmap_ids: List[str] = []
        for idx in (self._by_shape, self._by_topo, self._by_scaled):
            bitmap_ids.extend(idx.get(key, []))
        # Dedup while preserving insertion order.
        seen = set()
        out: List[EntityRecord] = []
        for bid in bitmap_ids:
            if bid in seen:
                continue
            seen.add(bid)
            rec = self._by_bitmap.get(bid)
            if rec is not None:
                out.append(rec)
        return out

    def all(self) -> List[EntityRecord]:
        return list(self._by_bitmap.values())

    def known_at(self, abstraction: str) -> Dict[str, List[EntityRecord]]:
        if abstraction == "bitmap_id":
            return {k: [v] for k, v in self._by_bitmap.items()}
        idx_attr = {
            "shape_id":  self._by_shape,
            "topo_id":   self._by_topo,
            "scaled_id": self._by_scaled,
        }.get(abstraction)
        if idx_attr is None:
            raise ValueError(f"unknown abstraction: {abstraction!r}")
        return {
            k: [self._by_bitmap[bid] for bid in v if bid in self._by_bitmap]
            for k, v in idx_attr.items()
        }

    # ---- write ----

    def register(
        self,
        *,
        bitmap_id:  str,
        shape_id:   str,
        topo_id:    str,
        scaled_id:  str,
        annotation: str = "",
        step:       int = 0,
        domain:     str = "",
        instance:   str = "",
        payload:    Optional[Any] = None,
    ) -> EntityRecord:
        rec = self._by_bitmap.get(bitmap_id)
        if rec is None:
            rec = EntityRecord(
                bitmap_id       = bitmap_id,
                shape_id        = shape_id,
                topo_id         = topo_id,
                scaled_id       = scaled_id,
                annotation      = annotation,
                first_seen_step = int(step),
                payload         = payload,
            )
            self._by_bitmap[bitmap_id] = rec
            self._by_shape.setdefault(shape_id, []).append(bitmap_id)
            self._by_topo.setdefault(topo_id, []).append(bitmap_id)
            self._by_scaled.setdefault(scaled_id, []).append(bitmap_id)
        elif payload is not None and rec.payload is None:
            # First time we have a payload to ship — attach it.
            rec.payload = payload
        rec.observations.append({
            "step":     int(step),
            "domain":   str(domain),
            "instance": str(instance),
        })
        return rec

    def annotate_role(
        self,
        bitmap_id: str,
        role:      str,
        credence:  float,
    ) -> None:
        rec = self._by_bitmap.get(bitmap_id)
        if rec is None:
            raise KeyError(f"unknown bitmap_id: {bitmap_id!r}")
        prev = rec.role_hypotheses.get(role, 0.0)
        rec.role_hypotheses[role] = max(prev, float(credence))

    # ---- persistence ----

    def to_dict(self) -> Dict[str, Any]:
        # Payload encoding is the adapter's responsibility — subclasses
        # override `_serialise_payload` if their payload type is not
        # JSON-friendly out of the box.
        return {
            "version": 1,
            "records": [
                {
                    "bitmap_id":       r.bitmap_id,
                    "shape_id":        r.shape_id,
                    "topo_id":         r.topo_id,
                    "scaled_id":       r.scaled_id,
                    "annotation":      r.annotation,
                    "first_seen_step": r.first_seen_step,
                    "role_hypotheses": dict(r.role_hypotheses),
                    "observations":    list(r.observations),
                    "payload":         self._serialise_payload(r.payload),
                }
                for r in self._by_bitmap.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DefaultVisualStore":
        store = cls()
        for rd in (data.get("records") or []):
            store.register(
                bitmap_id  = str(rd["bitmap_id"]),
                shape_id   = str(rd["shape_id"]),
                topo_id    = str(rd["topo_id"]),
                scaled_id  = str(rd["scaled_id"]),
                annotation = str(rd.get("annotation", "")),
                step       = int(rd.get("first_seen_step", 0)),
                payload    = cls._deserialise_payload(rd.get("payload")),
            )
            rec = store._by_bitmap[rd["bitmap_id"]]
            rec.role_hypotheses = dict(rd.get("role_hypotheses") or {})
            # First observation already added by register(); replace with
            # the persisted list to preserve full history.
            rec.observations = list(rd.get("observations") or [])
        return store

    # --- payload encoding hooks (override in subclasses) ---

    @staticmethod
    def _serialise_payload(payload: Any) -> Any:
        """Default: passes through.  Adapter subclasses override for
        binary payloads (e.g. base64-encode a PNG)."""
        return payload

    @staticmethod
    def _deserialise_payload(data: Any) -> Any:
        """Inverse of _serialise_payload."""
        return data


# ---------------------------------------------------------------------------
# Helper: claims at multiple abstraction levels
# ---------------------------------------------------------------------------


# Default abstraction levels at which miners should propose claims.
# The ordering is most-specific → most-general so that anyone iterating
# without changing the order gets the natural credence scaling: more
# specific keys win when the evidence backs them, broader keys still
# accumulate when narrow keys can't reach across instances.
DEFAULT_ABSTRACTION_LEVELS: tuple = ("bitmap_id", "shape_id", "topo_id")


def abstraction_keys_for(
    store:         Optional[VisualStore],
    base_entity_id: str,
    *,
    levels: tuple = DEFAULT_ABSTRACTION_LEVELS,
) -> List[tuple]:
    """Resolve `base_entity_id` (a bitmap_id) into the entity_id strings
    to use at each requested abstraction level.

    Returns a list of `(level_name, entity_id_at_that_level)` tuples.
    If the store doesn't recognise `base_entity_id` (e.g. the entity
    was just synthesised by a miner without going through fingerprinting),
    returns a single tuple `("bitmap_id", base_entity_id)` — preserves
    the legacy single-level behaviour.

    The same entity_id may appear at multiple levels when the entity
    is the only one with its shape / topology (i.e. `bitmap_id` ==
    `shape_id` == `topo_id`).  Callers should dedup by entity_id if
    they want to avoid re-proposing the same canonical claim three
    times.
    """
    if store is None or not base_entity_id:
        return [("bitmap_id", base_entity_id)]
    rec = None
    try:
        rec = store.get(base_entity_id)
    except Exception:
        rec = None
    if rec is None:
        return [("bitmap_id", base_entity_id)]
    out: List[tuple] = []
    seen: set = set()
    for level in levels:
        key = getattr(rec, level, None)
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append((level, key))
    if not out:
        return [("bitmap_id", base_entity_id)]
    return out
