"""VLM tool surface for the perception Scene.

A closed vocabulary of tool functions a VLM (or any agent loop)
calls to interrogate the perception Scene without re-deriving facts
from pixels.  Every tool:

  - takes a Scene as the first argument plus its specific args,
  - returns a JSON-serialisable dict (or list/scalar),
  - is STATELESS — no hidden mutation; the Scene is the only state,
  - is VLM-AGNOSTIC — the function signatures are LLM-friendly but
    don't depend on any particular vendor's tool-use format.

These tools compensate the VLM's known weaknesses (bucketed counts,
vague positions, no bboxes, single-frame ambiguity) by exposing the
deterministic detector + registry results as cheap callable
functions.

VOCABULARY (closed; the VLM gets this list in its system prompt):

  count_entities         - how many tracks are active right now
                           (optionally filtered by category / colour)
  describe_entity        - all known facts about one track
  list_entities          - active tracks summarised (id + position +
                           category + signature key)
  entity_history         - per-turn observations for one track over
                           a window
  nearby_entities        - tracks within K cells of a reference track
  spatial_relationships  - relationships at a given turn (optionally
                           filtered to involve a specific track)
  frame_diff             - which tracks changed state between turn A
                           and turn B
  recent_changes         - behaviour events across all tracks in the
                           last N turns
  match_entity           - similarity score between two tracks (or
                           between a track and a free-text visual
                           description)
  annotate_entity        - set description / category_labels /
                           properties on a track (write tool the VLM
                           uses to commit its judgements)

The VLM should ask FOLLOW-UP questions through these tools rather
than try to estimate from a single image.  A typical loop:

  list_entities(turn) -> short list of ids
  describe_entity(id) -> all facts the registry has
  entity_history(id, n=5) -> recent behaviour
  annotate_entity(id, description=..., category_labels=[...])

The tools do not call the VLM back; they just answer questions.
"""

from __future__ import annotations

from typing import Optional

from .entity_detector import Entity
from .template_store import signature_similarity
from .temporal_registry import (
    Scene,
    EntityTrack,
    EntityObservation,
    SpatialRelationship,
    BehaviourEvent,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _track_summary(t: EntityTrack, turn: int) -> dict:
    """One-line summary of a track at a given turn."""
    obs = t.observation_at(turn)
    sig = t.canonical_signature()
    out: dict = {
        "track_id": t.track_id,
        "present_now": obs is not None,
        "description": t.description,
        "category_labels": t.category_labels,
        "canonical_signature_first_key": (
            sig[0][0] if sig else None
        ),
        "n_turns_observed": t.n_turns_observed(),
    }
    if obs is not None:
        out["current_position"] = obs.centroid_cell
        out["current_n_pixels"] = obs.n_pixels
    return out


def _track_full(t: EntityTrack, turn: int) -> dict:
    """All registry-owned facts about a track."""
    obs = t.observation_at(turn)
    sig = t.canonical_signature()
    return {
        "track_id": t.track_id,
        "first_seen_turn": t.first_seen_turn,
        "last_seen_turn": t.last_seen_turn,
        "n_turns_observed": t.n_turns_observed(),
        "present_at_turn": obs is not None,
        "current_position": obs.centroid_cell if obs else None,
        "current_n_pixels": obs.n_pixels if obs else None,
        "current_bbox_logical": obs.bbox_logical if obs else None,
        "description": t.description,
        "category_labels": t.category_labels,
        "properties": dict(t.properties),
        "canonical_signature": list(sig),
        "pixel_count_min": t.pixel_count_min(),
        "pixel_count_max": t.pixel_count_max(),
        "pixel_count_variance_ratio": round(
            t.pixel_count_variance_ratio(), 3,
        ),
        "position_drift_cells": t.position_drift_cells(),
        "is_stationary": t.is_stationary(),
        "behaviour_events": [
            {"turn": e.turn, "kind": e.kind, "details": e.details,
             "related_track_ids": list(e.related_track_ids)}
            for e in t.behaviour_events
        ],
        "hypothesis_ids": list(t.hypothesis_ids),
    }


def _cell_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# -----------------------------------------------------------------------------
# Tool: count_entities
# -----------------------------------------------------------------------------


def count_entities(
    scene: Scene,
    turn: int,
    *,
    signature_first_key: Optional[int] = None,
    category_label: Optional[str] = None,
) -> dict:
    """How many tracks are active at `turn`, optionally filtered.

    Filters:
      signature_first_key - only tracks whose canonical_signature
                            starts with this RGB key (e.g. all green
                            sediments share a green key)
      category_label      - only tracks with this label in
                            category_labels (e.g. "collectable")

    Returns {count, matching_track_ids}.
    """
    matching: list[int] = []
    for t in scene.registry.tracks.values():
        if not t.is_present_at(turn):
            continue
        if signature_first_key is not None:
            sig = t.canonical_signature()
            if not sig or sig[0][0] != signature_first_key:
                continue
        if category_label is not None:
            if not any(
                lbl == category_label for (lbl, _conf) in t.category_labels
            ):
                continue
        matching.append(t.track_id)
    return {"count": len(matching), "matching_track_ids": matching}


# -----------------------------------------------------------------------------
# Tool: describe_entity
# -----------------------------------------------------------------------------


def describe_entity(
    scene: Scene, track_id: int, turn: int,
) -> dict:
    """All registry facts about one track, plus its current
    observation at `turn` if present.  Returns {} if the track
    doesn't exist.
    """
    t = scene.registry.get_track(track_id)
    if t is None:
        return {}
    return _track_full(t, turn)


# -----------------------------------------------------------------------------
# Tool: list_entities
# -----------------------------------------------------------------------------


def list_entities(
    scene: Scene,
    turn: int,
    *,
    include_recently_disappeared: bool = False,
    window: int = 5,
) -> list[dict]:
    """Short summary of every active track at `turn`.

    If include_recently_disappeared is True, also include tracks
    whose last_seen_turn is within `window` turns of `turn` even
    if they're not present right now.  Useful for "what just got
    consumed" queries.
    """
    out: list[dict] = []
    for t in scene.registry.tracks.values():
        obs = t.observation_at(turn)
        if obs is None:
            if not include_recently_disappeared:
                continue
            if t.last_seen_turn < turn - window:
                continue
        out.append(_track_summary(t, turn))
    return out


# -----------------------------------------------------------------------------
# Tool: entity_history
# -----------------------------------------------------------------------------


def entity_history(
    scene: Scene, track_id: int, turn: int, *, window: int = 10,
) -> dict:
    """Per-turn observations for one track over the last `window`
    turns ending at `turn` (inclusive).  Returns {} for unknown
    track_id.
    """
    t = scene.registry.get_track(track_id)
    if t is None:
        return {}
    cutoff = turn - window + 1
    obs = [
        {
            "turn": o.turn,
            "centroid_cell": o.centroid_cell,
            "n_pixels": o.n_pixels,
            "bbox_logical": o.bbox_logical,
            "signature_first_key": (
                o.visual_signature[0][0] if o.visual_signature else None
            ),
        }
        for o in t.observations
        if cutoff <= o.turn <= turn
    ]
    events = [
        {"turn": e.turn, "kind": e.kind, "details": e.details}
        for e in t.behaviour_events
        if cutoff <= e.turn <= turn
    ]
    return {
        "track_id": t.track_id,
        "window": window,
        "observations": obs,
        "behaviour_events": events,
    }


# -----------------------------------------------------------------------------
# Tool: nearby_entities
# -----------------------------------------------------------------------------


def nearby_entities(
    scene: Scene, track_id: int, turn: int, *, max_cell_distance: int = 2,
) -> list[dict]:
    """Tracks active at `turn` whose centroid_cell is within
    `max_cell_distance` (manhattan) of the reference track's
    centroid_cell.  Excludes the reference track itself.
    """
    ref = scene.registry.get_track(track_id)
    if ref is None:
        return []
    ref_obs = ref.observation_at(turn)
    if ref_obs is None:
        return []
    out: list[dict] = []
    for t in scene.registry.tracks.values():
        if t.track_id == track_id:
            continue
        o = t.observation_at(turn)
        if o is None:
            continue
        d = _cell_distance(ref_obs.centroid_cell, o.centroid_cell)
        if d > max_cell_distance:
            continue
        summary = _track_summary(t, turn)
        summary["cell_distance"] = d
        out.append(summary)
    out.sort(key=lambda x: x["cell_distance"])
    return out


# -----------------------------------------------------------------------------
# Tool: spatial_relationships
# -----------------------------------------------------------------------------


def spatial_relationships(
    scene: Scene, turn: int, *,
    involving_track: Optional[int] = None,
    relation_type: Optional[str] = None,
) -> list[dict]:
    """Geometric relationships at `turn`, optionally filtered.

    involving_track - only relations where this track_id is a or b
    relation_type   - only relations of this kind (e.g. "adjacent_cell")
    """
    rels = scene.relationships_by_turn.get(turn, [])
    out: list[dict] = []
    for r in rels:
        if involving_track is not None:
            if r.a != involving_track and r.b != involving_track:
                continue
        if relation_type is not None and r.type != relation_type:
            continue
        out.append({
            "type": r.type, "a": r.a, "b": r.b,
            "extra": dict(r.extra),
        })
    return out


# -----------------------------------------------------------------------------
# Tool: frame_diff
# -----------------------------------------------------------------------------


def frame_diff(scene: Scene, turn_a: int, turn_b: int) -> dict:
    """What changed between turn_a and turn_b.

    Returns:
      appeared      - tracks present at turn_b but not at turn_a
      disappeared   - tracks present at turn_a but not at turn_b
      moved         - tracks present in both with different centroid
                       (each entry includes from/to)
      size_changed  - tracks present in both whose n_pixels changed
                       by >= 20%
    """
    a_present = {
        t.track_id for t in scene.registry.tracks.values()
        if t.is_present_at(turn_a)
    }
    b_present = {
        t.track_id for t in scene.registry.tracks.values()
        if t.is_present_at(turn_b)
    }
    appeared = sorted(b_present - a_present)
    disappeared = sorted(a_present - b_present)
    moved: list[dict] = []
    size_changed: list[dict] = []
    for tid in a_present & b_present:
        t = scene.registry.tracks[tid]
        oa = t.observation_at(turn_a)
        ob = t.observation_at(turn_b)
        if oa is None or ob is None:
            continue
        if oa.centroid_cell != ob.centroid_cell:
            moved.append({
                "track_id": tid,
                "from": oa.centroid_cell,
                "to": ob.centroid_cell,
            })
        if oa.n_pixels > 0:
            ratio = ob.n_pixels / oa.n_pixels
            if ratio >= 1.2 or ratio <= 0.8:
                size_changed.append({
                    "track_id": tid,
                    "from_px": oa.n_pixels,
                    "to_px": ob.n_pixels,
                })
    return {
        "from_turn": turn_a, "to_turn": turn_b,
        "appeared": appeared, "disappeared": disappeared,
        "moved": moved, "size_changed": size_changed,
    }


# -----------------------------------------------------------------------------
# Tool: recent_changes
# -----------------------------------------------------------------------------


def recent_changes(
    scene: Scene, turn: int, *, window: int = 5,
) -> list[dict]:
    """All behaviour events across all tracks in the last `window`
    turns ending at `turn` (inclusive), in chronological order.
    """
    cutoff = turn - window + 1
    out: list[dict] = []
    for t in scene.registry.tracks.values():
        for e in t.behaviour_events:
            if cutoff <= e.turn <= turn:
                out.append({
                    "turn": e.turn,
                    "track_id": t.track_id,
                    "kind": e.kind,
                    "details": e.details,
                    "related_track_ids": list(e.related_track_ids),
                })
    out.sort(key=lambda x: (x["turn"], x["track_id"]))
    return out


# -----------------------------------------------------------------------------
# Tool: match_entity
# -----------------------------------------------------------------------------


def match_entity(
    scene: Scene,
    track_id_a: int, track_id_b: int, *,
    turn_a: Optional[int] = None, turn_b: Optional[int] = None,
) -> dict:
    """Signature similarity between two tracks (using the most recent
    observation in each track, or specific turns if supplied).
    Returns {similarity, size_ratio, same_canonical_signature_key}.
    """
    ta = scene.registry.get_track(track_id_a)
    tb = scene.registry.get_track(track_id_b)
    if ta is None or tb is None:
        return {"similarity": 0.0, "error": "unknown track id"}
    if turn_a is not None:
        oa = ta.observation_at(turn_a)
    else:
        oa = ta.last_observation()
    if turn_b is not None:
        ob = tb.observation_at(turn_b)
    else:
        ob = tb.last_observation()
    if oa is None or ob is None:
        return {"similarity": 0.0, "error": "no observation at given turn"}
    sim = signature_similarity(oa.visual_signature, ob.visual_signature)
    size_ratio = (
        min(oa.n_pixels, ob.n_pixels) / max(oa.n_pixels, ob.n_pixels)
        if max(oa.n_pixels, ob.n_pixels) > 0 else 0.0
    )
    a_canon = ta.canonical_signature()
    b_canon = tb.canonical_signature()
    same_key = bool(
        a_canon and b_canon and a_canon[0][0] == b_canon[0][0]
    )
    return {
        "similarity": round(sim, 3),
        "size_ratio": round(size_ratio, 3),
        "same_canonical_signature_key": same_key,
    }


# -----------------------------------------------------------------------------
# Tool: annotate_entity -- WRITE tool the VLM uses to commit its
# judgements back to the Scene.
# -----------------------------------------------------------------------------


def annotate_entity(
    scene: Scene, track_id: int, *,
    description: Optional[str] = None,
    add_category_label: Optional[tuple[str, str]] = None,
    set_properties: Optional[dict[str, str]] = None,
    add_hypothesis_id: Optional[str] = None,
) -> dict:
    """Attach VLM-supplied facts to a track.  Returns the updated
    track summary so the VLM sees its write took effect.

    Each parameter is optional and idempotent:
      description           - overwrite the track description
      add_category_label    - append (label, confidence_word) if not
                              already present
      set_properties        - merge into the properties dict
      add_hypothesis_id     - append a hypothesis_id (pointer into
                              WorldState.hypotheses) if not already
                              present

    The write tool exists so the VLM can commit its judgements as
    *labels* the registry remembers across turns.  Credence-bearing
    claims go to the hypothesis_store; this tool only stores the
    pointer back to that hypothesis.
    """
    t = scene.registry.get_track(track_id)
    if t is None:
        return {"error": "unknown track id"}
    if description is not None:
        t.description = description
    if add_category_label is not None:
        if add_category_label not in t.category_labels:
            t.category_labels.append(add_category_label)
    if set_properties:
        t.properties.update(set_properties)
    if add_hypothesis_id is not None:
        if add_hypothesis_id not in t.hypothesis_ids:
            t.hypothesis_ids.append(add_hypothesis_id)
    last_turn = t.last_seen_turn
    return _track_summary(t, last_turn)


# -----------------------------------------------------------------------------
# Closed vocabulary -- exported so caller can list it in the VLM's
# system prompt.
# -----------------------------------------------------------------------------


TOOL_NAMES = (
    "count_entities",
    "describe_entity",
    "list_entities",
    "entity_history",
    "nearby_entities",
    "spatial_relationships",
    "frame_diff",
    "recent_changes",
    "match_entity",
    "annotate_entity",
)
