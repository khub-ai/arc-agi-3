"""Substrate-computed visual events — pixel-diff inside the
bboxes of entities that perception flagged as worth watching.

Motivation: many games communicate state changes (deliveries,
unlocks, progress, scoring) by changing the appearance of small
fixed-position indicators — HUD slots, status bars, lights on
buttons, fill levels in containers.  When the perception layer
only tracks entity BBOXES and ROLES, these signals are invisible
because the indicator doesn't move and doesn't change role.

The substrate response: for any entity whose role suggests it is
a state indicator (default: ``role == "hud"``; extensible via an
``watch_internal_pixels`` flag on the entity record), compute a
pixel-wise diff inside its bbox between consecutive frames and
emit a generic VISUAL EVENT.  The strategy actor reads these
events as evidence — typically as ``win_condition_observation``
support / contradict signals.

GAME-AGNOSTIC: the diff is a pure function of two frames + a
bbox.  The choice of WHICH entities to watch is perception's
role inference, which is itself game-agnostic.  No game-specific
code anywhere in this module.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world_knowledge import WorldKnowledge


_DEFAULT_MIN_DIFF_FRACTION = 0.005
    # 0.5% of pixels must differ for an event to fire.  Tuned
    # to suppress noise from compression / antialiasing artifacts
    # while still catching subtle HUD highlights.  (Kept for
    # backward-compat; durability is now the primary guard.)

_DEFAULT_PERSISTENCE_FRAMES = 2
    # A non-baseline appearance must hold for this many consecutive
    # observations before it fires as an event.  This REPLACES the
    # old background-exclusion heuristic as the false-positive guard:
    # transient one-frame flicker (compression, a sprite passing
    # through) does not persist; a real state change (a lit
    # indicator, a skewered block) does.  Durability is the signal —
    # NOT where the change lands or what colour it is (durable
    # principle P11).

_DEFAULT_MARGIN_FRACTION = 0.4
    # Expand each watched bbox by this fraction of its own size on
    # every side before diffing.  Many games render a state change on
    # the cells FLANKING an entity (a skewer through a block, a
    # highlight ring) rather than inside its own pixels; a margin
    # catches those without assuming where the change appears.

_DEFAULT_WATCH_ROLES = ("hud",)
    # Any entity whose latest role is in this set is watched.
    # Extensible via the per-entity ``watch_internal_pixels``
    # attribute (if perception sets it on the EntityRecord, the
    # entity is watched regardless of role).

# Roles that are NEVER watched for internal pixel change — large
# backdrop entities whose interior churns for irrelevant reasons.
_SCENERY_ROLES = ("scenery", "background", "wall", "decoration")

# An entity is "positionally stable" (a candidate indicator) if its
# bbox has not moved across the last N observations.  Stable,
# non-scenery, not-oversized entities are watched even if perception
# never tagged them "hud" — so any in-place state indicator surfaces
# regardless of role label.  Game-agnostic: stability + size are
# pure geometry, not game knowledge.
_STABILITY_WINDOW = 3
_MAX_WATCH_AREA_FRACTION = 0.20
    # entities larger than 20% of the frame are treated as scenery
    # and not watched (their interiors change for reasons unrelated
    # to a discrete state signal).

def _is_positionally_stable(entity_record,
                            window: int = _STABILITY_WINDOW) -> bool:
    """True iff the entity's bbox is identical across the last
    ``window`` observations (it sits still — an indicator, not a
    mover)."""
    history = getattr(entity_record, "bbox_history", None) or []
    if len(history) < 2:
        return False
    recent = history[-window:]
    bboxes = []
    for h in recent:
        if isinstance(h, (list, tuple)) and len(h) >= 2:
            b = h[1]
            if isinstance(b, (list, tuple)) and len(b) == 4:
                bboxes.append(tuple(b))
    if len(bboxes) < 2:
        return False
    return all(b == bboxes[0] for b in bboxes)


def _latest_role(entity_record) -> str:
    history = getattr(entity_record, "role_history", None) or []
    if not history:
        return "unknown"
    last = history[-1]
    # role_history elements may be (turn, role, conf) tuples or lists
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        return str(last[1])
    return "unknown"


def _latest_bbox(entity_record) -> Optional[list[int]]:
    history = getattr(entity_record, "bbox_history", None) or []
    if not history:
        return None
    last = history[-1]
    if isinstance(last, (list, tuple)) and len(last) >= 2:
        bbox = last[1]
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            return list(bbox)
    return None


def compute_internal_pixel_events(
    prev_frame_path: Optional[Path],
    curr_frame_path: Path,
    world: "WorldKnowledge",
    *,
    min_diff_fraction: float = _DEFAULT_MIN_DIFF_FRACTION,
    watch_roles: tuple[str, ...] = _DEFAULT_WATCH_ROLES,
    persistence_frames: int = _DEFAULT_PERSISTENCE_FRAMES,
    margin_fraction: float = _DEFAULT_MARGIN_FRACTION,
) -> list[dict]:
    """For each watched entity, compute the pixel-wise difference
    inside its bbox between the prev and curr frames and emit a
    VISUAL EVENT if the diff exceeds ``min_diff_fraction``.

    Watched entities are: any with ``role in watch_roles`` OR any
    with a ``watch_internal_pixels`` attribute set truthy on the
    EntityRecord.

    Returns a list of small dicts:
        {entity, kind: 'internal_pixel_change',
         pixel_diff_fraction, bbox}

    Returns [] on any error (missing frames, PIL/numpy unavailable).
    The substrate-side computation must NEVER raise into the
    driver flow; the caller will append [] silently if this fails.
    """
    if prev_frame_path is None:
        return []
    if not Path(prev_frame_path).exists() or not Path(curr_frame_path).exists():
        return []

    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return []

    try:
        prev_img = Image.open(prev_frame_path).convert("RGB")
        curr_img = Image.open(curr_frame_path).convert("RGB")
        prev = np.array(prev_img)
        curr = np.array(curr_img)
    except Exception:
        return []

    if prev.shape != curr.shape:
        return []
    h, w = prev.shape[:2]

    # Perception state must NOT carry across a level boundary — the
    # indicator strip is re-laid-out between levels, so a baseline
    # learned in one level leaks into the next as a spurious
    # activate/revert (durable principle P11 / "re-derive per level").
    # Reset learned baselines whenever the level or score changes.
    vstate = getattr(world, "_vis_state", None)
    if vstate is None:
        vstate = {}
        try:
            world._vis_state = vstate
        except Exception:
            vstate = None
    if vstate is not None:
        level_key = (getattr(world, "level", None),
                     getattr(world, "score", None))
        if vstate.get("_level_key") != level_key:
            vstate.clear()
            vstate["_level_key"] = level_key

    # All tracked entity bboxes — used to clamp each watched entity's
    # margin so it can't bleed into a NEIGHBOURING entity (the indicator
    # strip is tightly packed; an unclamped margin reads a neighbour's
    # change, or the passing arm, as this entity's state change).
    all_bboxes: list[tuple[str, list[int]]] = []
    for _nm, _en in (getattr(world, "entities", None) or {}).items():
        _bb = _latest_bbox(_en)
        if _bb is not None:
            all_bboxes.append((_nm, [int(x) for x in _bb]))

    events: list[dict] = []
    for name, ent in (getattr(world, "entities", None) or {}).items():
        # Watch decision (broadened 2026-05-29): watch if ANY of
        #   (a) role in watch_roles (e.g. "hud"),
        #   (b) entity flagged watch_internal_pixels,
        #   (c) positionally stable, non-scenery, not-oversized —
        #       a candidate in-place indicator regardless of role tag.
        # (c) removes the dependence on perception tagging the
        # indicator "hud"; any still indicator that changes internally
        # now surfaces.
        role = _latest_role(ent)
        flagged = bool(getattr(ent, "watch_internal_pixels", False))
        bbox = _latest_bbox(ent)
        if bbox is None:
            continue
        watch = role in watch_roles or flagged
        if not watch and role not in _SCENERY_ROLES:
            br1, bc1, br2, bc2 = bbox
            area = max(0, (br2 - br1 + 1)) * max(0, (bc2 - bc1 + 1))
            if (area <= _MAX_WATCH_AREA_FRACTION * h * w
                    and _is_positionally_stable(ent)):
                watch = True
        if not watch:
            continue
        # Bboxes are INCLUSIVE [r1,c1,r2,c2].  Expand by a MARGIN of
        # margin_fraction of the entity's own size on each side, so a
        # state change rendered on the FLANKING cells (a skewer through
        # a block, a highlight ring) is included.  We do NOT assume the
        # change recolours the entity's own pixels (P11).
        or1, oc1, or2, oc2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        bh = max(1, or2 - or1 + 1)
        bw = max(1, oc2 - oc1 + 1)
        mr = max(0, int(round(margin_fraction * bh)))
        mc = max(0, int(round(margin_fraction * bw)))
        r1 = max(0, or1 - mr); c1 = max(0, oc1 - mc)
        r2 = min(h, or2 + 1 + mr); c2 = min(w, oc2 + 1 + mc)
        # Neighbour-aware clamp: never let the margin overlap ANOTHER
        # tracked entity's bbox.  This keeps the flank recall (the skewer
        # sits in empty space beside the entity) while killing the
        # over-reporting where a packed-neighbour swatch or the passing
        # arm fell inside the margin.
        for _onm, (nr1, nc1, nr2, nc2) in all_bboxes:
            if _onm == name:
                continue
            if not (nr2 < or1 or nr1 > or2):        # rows overlap -> horizontal neighbour
                if nc1 > oc2:
                    c2 = min(c2, nc1)               # stop before a right neighbour
                if nc2 < oc1:
                    c1 = max(c1, nc2 + 1)            # stop after a left neighbour
            if not (nc2 < oc1 or nc1 > oc2):        # cols overlap -> vertical neighbour
                if nr1 > or2:
                    r2 = min(r2, nr1)               # stop before a below neighbour
                if nr2 < or1:
                    r1 = max(r1, nr2 + 1)            # stop after an above neighbour
        if r2 <= r1 or c2 <= c1:
            continue
        prev_patch = prev[r1:r2, c1:c2]
        curr_patch = curr[r1:r2, c1:c2]
        if prev_patch.size == 0:
            continue
        # Skin-agnostic, durable classification: compare the patch to
        # this entity's learned resting BASELINE patch via PIXEL-DIFF
        # FRACTION (sensitive to a small indicator mark — a mean-colour
        # signature would wash it out) and fire only once the change has
        # PERSISTED for persistence_frames consecutive observations.
        # We do NOT exclude background->signal changes (the skewer/flank
        # lands on former-background cells, so excluding them filtered
        # out the real pierce) and we make NO colour/location assumption
        # (durable principle P11).  Durability is the false-positive
        # guard.  Returns (direction, frac).
        try:
            direction, frac = _classify_persistent(
                world, name, curr_patch, persistence_frames,
                min_diff_fraction,
            )
        except Exception:
            direction, frac = None, 0.0
        if direction is None:
            continue
        events.append({
            "entity": name,
            "kind": "internal_pixel_change",
            "direction": direction,
            "pixel_diff_fraction": round(frac, 4),
            "bbox": [int(r1), int(c1), int(r2), int(c2)],
            "role": role,
        })
    return events


def _classify_persistent(world, name, curr_patch, persistence_frames,
                         min_diff_fraction):
    """Skin-agnostic, durable state classifier.

    Compares the (margined) patch to the entity's learned resting
    BASELINE PATCH via per-pixel diff FRACTION — sensitive to small
    indicator marks (a center dot, a skewer flank), which a mean-colour
    signature would wash out.  Fires only after a departure from
    baseline has PERSISTED for ``persistence_frames`` consecutive
    observations.  Returns ``(direction, frac)`` where direction is:
      'activated' — durable departure from baseline (a state was set),
      'reverted'  — return to baseline after an activation (set marker
                    removed, e.g. a pierced block dragged off / undone),
      None        — nothing to emit this frame.

    Per-entity state (baseline patch, activation flag, streak) persists
    across turns on ``world._vis_state`` so a settled change is not
    re-fired and a later undo is caught.  No colour or location
    assumption; the false-positive guard is durability (P11)."""
    state = getattr(world, "_vis_state", None)
    if state is None:
        state = {}
        try:
            world._vis_state = state
        except Exception:
            return None, 0.0
    rec = state.get(name)
    base = rec.get("baseline") if rec is not None else None
    if (base is None or getattr(base, "shape", None) != curr_patch.shape):
        # (Re)establish the resting baseline (first sight, or the
        # watched region changed size — e.g. a level re-layout).
        state[name] = {"baseline": curr_patch.copy(),
                       "activated": False, "streak": 0}
        return None, 0.0
    try:
        frac = float((base != curr_patch).any(axis=-1).mean())
    except Exception:
        return None, 0.0
    if frac < min_diff_fraction:
        # At (or returned to) the resting baseline.
        rec["streak"] = 0
        if rec.get("activated"):
            rec["activated"] = False
            return "reverted", frac
        return None, frac
    # Departs from baseline — require it to PERSIST before emitting.
    rec["streak"] = rec.get("streak", 0) + 1
    if rec["streak"] >= persistence_frames and not rec.get("activated"):
        rec["activated"] = True
        return "activated", frac
    return None, frac


def format_visual_events_for_prompt(
    events: list[dict],
    *,
    max_events: int = 8,
) -> str:
    """Render visual events for inclusion in a prompt block."""
    if not events:
        return "  (no visual events this turn)"
    lines: list[str] = []
    lines.append(
        f"  {len(events)} VISUAL EVENT(s) — pixel-diff inside "
        "the bboxes of entities perception flagged as worth "
        "watching (typically role=hud).  These are AFFIRMATIVE "
        "SIGNALS the substrate observed, separate from the "
        "perception VLM's symbolic delta.  Treat them as direct "
        "evidence: when an indicator changes appearance "
        "immediately after an action, that action probably caused "
        "an effect related to whatever the indicator tracks.  "
        "Use these to record `win_condition_observation` (support "
        "or contradict) for hypotheses about what triggers "
        "progress."
    )
    reverted = [e for e in events if e.get("direction") == "reverted"]
    if reverted:
        names = ", ".join(e.get("entity", "?") for e in reverted)
        lines.append(
            f"  !! REGRESSION: {names} reverted to its resting state "
            f"— a previously-SET marker was REMOVED.  This turn's "
            f"action UNDID progress (e.g. a scored block got "
            f"un-scored).  Reconsider it before continuing."
        )
    for ev in events[:max_events]:
        direction = ev.get("direction", "changed")
        dtag = {"activated": "ACTIVATED/progress",
                "reverted": "REVERTED/REGRESSION",
                "changed": "changed"}.get(direction, direction)
        lines.append(
            f"  EVENT entity={ev.get('entity','?')!r} "
            f"role={ev.get('role','?')!r} "
            f"kind={ev.get('kind','?')!r} "
            f"dir={dtag} "
            f"pixel_diff={ev.get('pixel_diff_fraction', 0):.2%} "
            f"bbox={ev.get('bbox','?')}"
        )
    if len(events) > max_events:
        lines.append(f"  ... ({len(events) - max_events} more)")
    return "\n".join(lines)
