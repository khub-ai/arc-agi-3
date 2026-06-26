"""Measured RESPONSE-ASYMMETRY facts — the pixel stage's neutral feed to the
symbolic stage (logical_abstraction).

PURELY FACTUAL, no semantic judgement.  From the observed per-turn deltas it
labels each entity by HOW IT RESPONDS to actions:

  settable        : a click ON the entity changed it IN PLACE with no animation
                    (a settable input — the click-response asymmetry that separates
                    an active control from a fixed reference: a reference's members
                    never change when clicked).
  is_trigger      : a click ON the entity produced an ANIMATION (it fires /
                    activates something — a previewing legend or a GO control).
  responds_to_fire: during a trigger's animation a mover DEPARTED FROM the entity
                    (it moved) and the entity is NOT settable — a driven output /
                    SCENE element.  Excluding settable entities is what avoids the
                    conflation where a highlight sweeping over the controls would
                    look like the controls "responding": the swept controls are
                    settable inputs, and the highlight starts at the cursor, not at
                    the marker, so only a genuinely-moving non-control (e.g. a goal
                    marker) is tagged.

The symbolic layer consumes these to TYPE sections as 'program' (settable inputs
+ triggers) vs 'scene' (responds_to_fire outputs); that interpretation is NOT made
here.  Game-agnostic: it keys on the RESPONSE to an action, never on a colour or
a position, so the labelling is identical under a re-skin or a board shuffle.

Pure functions over lightweight delta views ({action|inferred_action,
entities_changed, animation_events}) + an entity bbox map; no World dependency.
"""
from __future__ import annotations


def _click_point(action: str):
    """'CLICK:c,r' -> (row, col), else None.  ARC clicks are CLICK:col,row."""
    if not action or not action.startswith("CLICK:"):
        return None
    try:
        c, r = (int(x) for x in action[len("CLICK:"):].split(",")[:2])
        return (r, c)
    except Exception:
        return None


def _contains(bbox, pt) -> bool:
    r0, c0, r1, c1 = (int(v) for v in bbox)
    r, c = pt
    return r0 <= r <= r1 and c0 <= c <= c1


def _delta_action(d) -> str:
    return (getattr(d, "action", "") or getattr(d, "inferred_action", "") or "")


def _changed_entities(d) -> set:
    """Entities that CHANGED in place this turn -- preferring the SUBSTRATE's own
    per-entity, bbox-scoped measurement (delta.visual_events, which is HUD-immune:
    it checks each entity's OWN bbox, so a move/budget counter elsewhere can't
    leak in) over the VLM reply's entities_changed.  Falls back to the reply only
    when the substrate signal is absent (older deltas)."""
    ve = getattr(d, "visual_events", None)
    if ve is not None:                       # substrate RAN (even an empty list is a
        return {e.get("entity") for e in ve  # real 'nothing changed' measurement, so
                if isinstance(e, dict) and e.get("entity")}  # do NOT fall back to it
    return set(getattr(d, "entities_changed", None) or [])    # older delta: no signal


def _clicked_entity(pt, entity_bboxes):
    """The entity whose bbox contains the click point (smallest, so a click on a
    glyph inside a panel attributes to the glyph, not the panel).  None if the
    click hit no perceived entity."""
    best, best_area = None, None
    for name, bb in entity_bboxes.items():
        try:
            if _contains(bb, pt):
                r0, c0, r1, c1 = (int(v) for v in bb)
                area = (r1 - r0 + 1) * (c1 - c0 + 1)
                if best_area is None or area < best_area:
                    best, best_area = name, area
        except Exception:
            continue
    return best


def classify_responses(deltas, entity_bboxes) -> dict:
    """Return {name: {settable, is_trigger}} measured over the delta history.

    ``deltas`` is any iterable of records exposing action/inferred_action,
    entities_changed (list of names), and animation_events (list).  ``entity_
    bboxes`` is {name: [r0,c0,r1,c1]}.  Only flips a flag TRUE on positive
    evidence; an entity never clicked stays all-False (unknown, not 'reference')."""
    facts = {n: {"settable": False, "is_trigger": False, "responds_to_fire": False}
             for n in entity_bboxes}
    # pass 1: own-click response -> settable / trigger
    for d in (deltas or []):
        action = _delta_action(d)
        pt = _click_point(action)
        if pt is None:
            continue
        clicked = _clicked_entity(pt, entity_bboxes)
        if clicked is None:
            continue
        if bool(getattr(d, "animation_events", None) or []):
            facts[clicked]["is_trigger"] = True
            continue
        if clicked in _changed_entities(d):       # SUBSTRATE-measured, HUD-immune
            facts[clicked]["settable"] = True
    # pass 2: scene response -> a mover DEPARTED from a NON-settable entity during a
    # fire (settable controls excluded so a highlight sweeping them is not mistaken
    # for them responding).
    settable = {n for n, f in facts.items() if f["settable"]}
    _MOTION = ("moved", "transient", "grew", "shrank", "appeared", "vanished")
    for d in (deltas or []):
        if _click_point(_delta_action(d)) is None:
            continue
        for ev in (getattr(d, "animation_events", None) or []):
            verbs = " ".join(ev.get("verbs", []) or []).lower()
            if not any(v in verbs for v in _MOTION):
                continue
            frm = ev.get("from")
            if not frm or len(frm) < 2:
                continue
            start = (int(round(frm[0])), int(round(frm[1])))   # (row, col)
            for name, bb in entity_bboxes.items():
                if name in settable:
                    continue
                if _contains(bb, start):
                    facts[name]["responds_to_fire"] = True
    return facts


def response_narration(facts: dict) -> str:
    """Neutral prompt text surfacing the measured facts for the VLM to interpret.
    Returns '' when nothing has been probed yet."""
    settable = sorted(n for n, f in facts.items() if f.get("settable"))
    triggers = sorted(n for n, f in facts.items() if f.get("is_trigger"))
    driven = sorted(n for n, f in facts.items() if f.get("responds_to_fire"))
    if not settable and not triggers and not driven:
        return ""
    lines = ["[SUBSTRATE] RESPONSE FACTS (measured by probing; you interpret what "
             "they MEAN — these are not labels):"]
    if settable:
        lines.append(f"  - SETTABLE (a click on it changed it in place, no "
                     f"animation): {settable}")
    if triggers:
        lines.append(f"  - TRIGGER (a click on it produced an animation): {triggers}")
    if driven:
        lines.append(f"  - DRIVEN (a mover departed from it during a trigger's "
                     f"animation — it is moved by something): {driven}")
    lines.append("  - any similar entity NOT in these lists was either clicked "
                 "with no effect (a fixed/reference element) or not yet probed.")
    return "\n".join(lines)
