"""Entity identity stitching across frame-change reminting (GAP 24a-3).

When an adapter has no persistent entity tracking — e.g. the ARC-AGI-3
grid adapter scans each frame afresh and invents new ``entity_N`` ids —
the same physical object acquires a new id every frame.  This bloats the
goal forest (one reduce-uncertainty goal per re-mint), defeats coverage
credit (observations never accrete), and prevents InsideBBox/AtPosition
goals from stabilising long enough to drive a plan.

This module exposes :func:`stitch_entity_ids`, invoked by
``_ingest_observation`` before new snapshots are inserted into
``ws.entities``.  For each newly-minted id it searches the existing
entity store for a candidate that

    * is not co-occurring in the same snapshot batch,
    * has been seen within ``staleness_window`` steps,
    * has exactly the same ``colour`` property, and
    * has bounding-box IoU >= ``iou_threshold`` with the new id's bbox.

If such a candidate exists, the new id is aliased to the stable id.  The
caller uses the returned mapping to insert snapshot data under the
stable id, so the redundant id never enters ``ws.entities``.

The matching rule (IoU + colour + staleness) is a generic object-
permanence prior and belongs to the engine, not any individual adapter.
Adapters that already emit stable ids disable stitching via
``EngineConfig.enable_entity_stitching=False``.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple


BBox = Tuple[float, float, float, float]


def _coerce_bbox(bbox: Any) -> Optional[BBox]:
    """Return a 4-tuple of floats (r0, c0, r1, c1) or ``None`` if the
    value cannot be coerced.  Bboxes are inclusive-pixel: a 1x1 box
    at (r, c) is ``(r, c, r, c)`` with area 1."""
    if bbox is None:
        return None
    try:
        r0, c0, r1, c1 = (float(x) for x in bbox)
    except (TypeError, ValueError):
        return None
    return (r0, c0, r1, c1)


def _bbox_iou(a: BBox, b: BBox) -> float:
    """Inclusive-pixel IoU between two bboxes.  A 1x1 box has area 1,
    IoU of a box with itself is 1.0.  Non-overlapping -> 0.0."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ir0 = max(ar0, br0)
    ic0 = max(ac0, bc0)
    ir1 = min(ar1, br1)
    ic1 = min(ac1, bc1)
    if ir0 > ir1 or ic0 > ic1:
        return 0.0
    inter  = (ir1 - ir0 + 1.0) * (ic1 - ic0 + 1.0)
    area_a = max(0.0, (ar1 - ar0 + 1.0) * (ac1 - ac0 + 1.0))
    area_b = max(0.0, (br1 - br0 + 1.0) * (bc1 - bc0 + 1.0))
    union  = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def stitch_entity_ids(
    ws,
    new_snapshots: Mapping[str, Mapping[str, Any]],
    *,
    obs_step: int,
    iou_threshold:    float = 0.5,
    staleness_window: int   = 3,
) -> Dict[str, str]:
    """Return an alias map {new_id: stable_id} for newly-minted ids
    that match a recently-seen entity on colour + bbox IoU.

    * ``ws`` — current :class:`WorldState`.
    * ``new_snapshots`` — ``obs.entity_snapshots`` from the incoming
      :class:`Observation`.
    * ``obs_step`` — ``obs.step`` (authoritative step number).
    * ``iou_threshold`` — minimum bbox IoU for a match.
    * ``staleness_window`` — maximum steps since a candidate was last
      seen to still be considered a match.

    Ids already present in ``ws.entities`` are left alone (they are
    already stable).  Ids with no bbox or no colour are skipped (the
    matching rule cannot fire).  Ids whose best candidate is
    co-occurring in the same batch are skipped (two distinct objects
    observed simultaneously cannot be the same physical object).
    """
    co_occurring = set(new_snapshots.keys())
    aliases: Dict[str, str] = {}
    for new_id, snapshot in new_snapshots.items():
        if new_id in ws.entities:
            continue
        new_bbox   = _coerce_bbox(snapshot.get("bbox"))
        new_colour = snapshot.get("colour")
        if new_bbox is None or new_colour is None:
            continue
        best_id:  Optional[str] = None
        best_iou: float         = 0.0
        for cand_id, cand_ent in ws.entities.items():
            if cand_id == new_id:
                continue
            if cand_id in co_occurring:
                continue
            if obs_step - cand_ent.last_seen_step > staleness_window:
                continue
            if cand_ent.properties.get("colour") != new_colour:
                continue
            cand_bbox = _coerce_bbox(cand_ent.properties.get("bbox"))
            if cand_bbox is None:
                continue
            iou = _bbox_iou(new_bbox, cand_bbox)
            if iou > best_iou:
                best_iou = iou
                best_id  = cand_id
        if best_id is not None and best_iou >= iou_threshold:
            aliases[new_id] = best_id
    return aliases


__all__ = ["stitch_entity_ids"]
