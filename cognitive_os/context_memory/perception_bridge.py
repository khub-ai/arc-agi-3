"""Read-side bridge from perception substrate to the engine claim
machinery.

Walks a parsed-perception dict (the output of
``cognitive_os.perception.interaction.apply_to_parsed`` after
``level_memory.resolve_roles`` and
``level_memory.assign_role_credence`` have run) and emits one
:class:`Claim` per asserted role.  Each claim is proposed through
:func:`cognitive_os.hypothesis_store.propose` with a credence
derived from the substrate's resolved tier.

This is step 3 of the context_memory migration described in
``docs/SPEC_context_memory_component.md``.  It does NOT change how
perception runs; it only routes perception's existing output
through the engine's existing claim stack so that:

* Perception's role assignments become first-class hypotheses with
  credence tracking, competitor linking, and evidence-driven
  updates.
* They persist via ``persistence.save_committed_knowledge`` and
  transfer across episodes / levels of the same game.
* Other engine layers (planner, dialogic, miners) can read
  perception's beliefs through the same hypothesis-store API
  they already use for engine-side claims.

Substrate-general: this module names no game-specific role,
makes no game-specific decision.  It is a typed translation layer
between two existing data shapes.

Dual-domain note: a robotics adapter that produces equivalent
"object X has role Y" / "surface region X plays role Y" assertions
will plug into this bridge with no changes.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

from ..claims import BitmapRoleClaim, RegionPaletteClaim
from ..hypothesis_store import propose
from ..types import Scope, WorldState


# Perception's tier-name -> hypothesis-store credence-point mapping.
# Tiers above 'shape' (causal/bitmap/region) become committed
# immediately (above the default 0.85 commit threshold) because
# perception has already accumulated strong evidence to give them
# those tiers; lower tiers stay below the commit threshold so they
# remain candidates that further engine-side evidence can confirm or
# refute.
_TIER_TO_POINT: Mapping[str, float] = {
    "causal":  0.92,
    "bitmap":  0.88,
    "region":  0.86,
    "shape":   0.75,
    "topo":    0.70,
    "soft":    0.60,
    "vlm":     0.55,
}
_DEFAULT_POINT = 0.60

# Role names handled by ``RegionPaletteClaim`` rather than
# ``BitmapRoleClaim``.  Mirrors the substrate's REGION_ROLES set
# (cognitive_os/perception/interaction.py); kept local here to avoid
# a perception->context_memory->perception circular import.
_REGION_ROLES = frozenset({
    "wall",
    "play_area",
    "void",
    "floor",
    "background",
    "hud_background",
})

# Role strings that mean "no role assigned" -- skip these entirely.
# ``unknown`` and ``noise`` carry no information for the engine.
_SKIP_ROLES = frozenset({"", "unknown", "noise"})


def _tier_to_point(tier: Optional[str],
                   confidence: Optional[int] = None) -> float:
    """Map a perception tier name (and optional numeric confidence)
    to a 0..1 credence-point value usable by hypothesis_store.

    Tiers known to the substrate map to fixed points so the engine
    can reason about "what tier was this from?" from the credence
    alone if needed.  When the tier is unrecognised (older substrate
    output, custom miner), we linearly interpolate the numeric
    confidence into [0.50, 0.95]; failing that we return a neutral
    default.
    """
    if tier and tier in _TIER_TO_POINT:
        return _TIER_TO_POINT[tier]
    if confidence is not None:
        try:
            c = int(confidence)
        except (TypeError, ValueError):
            return _DEFAULT_POINT
        return max(0.50, min(0.95, 0.50 + 0.0045 * c))
    return _DEFAULT_POINT


def _primary_sub_bitmap(ent: Mapping) -> Optional[Mapping]:
    """Return the entity's primary sub-bitmap record (the largest one),
    or ``None`` when the entity has no ``_sub_bitmaps`` entries.

    Multi-palette / multi-sub-CC entities accumulate several
    sub-bitmaps; we pick the largest one as the primary signature,
    matching the heuristic used in ``interaction.py``'s post-VLM
    compound merger.  Returning the whole record (not just the
    bitmap_id) lets the caller also pull ``shape_id`` / ``topo_id``
    / ``size_px`` for the BitmapRoleClaim metadata fields.
    """
    subs = ent.get("_sub_bitmaps") or []
    if not subs:
        return None
    return max(subs, key=lambda s: int(s.get("size_px") or 0))


def _row_range_from_bbox(ent: Mapping) -> Optional[tuple]:
    """Extract a ``(row_min, row_max)`` tuple from the entity's
    ``bbox_pixels`` field, or ``None`` when the field is absent /
    malformed.  Used as RegionPaletteClaim metadata so the region
    matcher can match by vertical band even when palettes shift."""
    bbox = ent.get("bbox_pixels")
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        r0, _, r1, _ = (int(x) for x in bbox)
    except (TypeError, ValueError):
        return None
    return (r0, r1)


def _spatial_zone(ent: Mapping) -> Optional[str]:
    """Pull the substrate's 9-zone bin from the entity's properties
    dict, when present.  Returns ``None`` quietly if missing -- the
    claim metadata is optional."""
    props = ent.get("properties") or {}
    zone = props.get("spatial_zone")
    return str(zone) if zone else None


def propose_from_parsed(
    parsed:        Mapping,
    ws:            WorldState,
    *,
    scope:         Scope,
    step:          int,
    source_prefix: str = "perception",
) -> List[str]:
    """Translate perception-substrate output to engine claims.

    Walks ``parsed['entities']`` plus
    ``parsed['background_palettes']`` and proposes one
    :class:`Claim` per role assignment.  Returns the list of
    hypothesis IDs created (or merged into via dedup) so callers
    can audit which claims this call generated.

    Role -> claim type mapping:
      * Region roles (wall / play_area / void / floor / background /
        hud_background) -> ``RegionPaletteClaim`` keyed on the
        entity's palette signature.
      * Any other catalog role with a recorded bitmap_id ->
        ``BitmapRoleClaim`` keyed on the primary bitmap fingerprint.
      * Entities with role ``unknown`` / ``noise`` -> skipped.
      * Entities with no bitmap_id AND a non-region role -> skipped
        (no stable signature to key the claim on).

    Background palettes listed in ``parsed['background_palettes']``
    are proposed as ``RegionPaletteClaim(palettes=(p,),
    role='background')`` regardless of whether an entity also
    declared them.

    Credence: derived from the substrate's
    ``_role_resolved_tier`` / ``_role_resolved_confidence`` fields
    via :func:`_tier_to_point`.  Tiers at or above ``region`` land
    above the commit threshold; weaker tiers land below.  No
    perception-specific cred-policy lives here; the mapping is
    documented above and easy to tune.

    Idempotency: identical proposals (same full_key) are merged by
    the hypothesis store; calling this function repeatedly on the
    same parsed view is a no-op in terms of claim count.
    """
    out_ids: List[str] = []

    # ---- entity-level claims -----------------------------------
    for ent in parsed.get("entities") or []:
        role = str(ent.get("role") or "").strip()
        if role in _SKIP_ROLES:
            continue

        tier = ent.get("_role_resolved_tier")
        conf = ent.get("_role_resolved_confidence")
        point = _tier_to_point(tier, conf)
        tier_tag = tier or "unresolved"

        if role in _REGION_ROLES:
            pals = ent.get("palettes") or []
            if not pals:
                continue
            claim = RegionPaletteClaim.make(
                palettes     = pals,
                role         = role,
                row_range    = _row_range_from_bbox(ent),
                spatial_zone = _spatial_zone(ent),
            )
            hid = propose(
                ws,
                claim=claim,
                source=f"{source_prefix}:role:{tier_tag}",
                scope=scope,
                step=step,
                initial_credence=point,
            )
            out_ids.append(hid)
            continue

        sub = _primary_sub_bitmap(ent)
        if sub is None or not sub.get("bitmap_id"):
            # Non-region role with no bitmap fingerprint -- nothing to
            # key a stable claim on.  This can happen for substrate-
            # promoted entities that haven't been matched to a track
            # yet.  Skip silently; the next observation cycle is
            # likely to surface a bitmap.
            continue
        claim = BitmapRoleClaim(
            bitmap_id    = str(sub["bitmap_id"]),
            role         = role,
            shape_id     = (str(sub["shape_id"])
                            if sub.get("shape_id") is not None else None),
            topo_id      = (str(sub["topo_id"])
                            if sub.get("topo_id")  is not None else None),
            size_px      = (int(sub["size_px"])
                            if sub.get("size_px")  is not None else None),
            spatial_zone = _spatial_zone(ent),
        )
        hid = propose(
            ws,
            claim=claim,
            source=f"{source_prefix}:role:{tier_tag}",
            scope=scope,
            step=step,
            initial_credence=point,
        )
        out_ids.append(hid)

    # ---- background-palette claims -----------------------------
    for p in parsed.get("background_palettes") or []:
        try:
            pal_int = int(p)
        except (TypeError, ValueError):
            continue
        claim = RegionPaletteClaim.make(palettes=[pal_int], role="background")
        hid = propose(
            ws,
            claim=claim,
            source=f"{source_prefix}:background_palettes",
            scope=scope,
            step=step,
            initial_credence=_TIER_TO_POINT["region"],
        )
        out_ids.append(hid)

    return out_ids
