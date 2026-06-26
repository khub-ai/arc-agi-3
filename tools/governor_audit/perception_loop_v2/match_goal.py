"""match_goal.py -- TRANSFORMATIONAL conform-goal: make the mover MATCH / FILL the goal.

COS's goal instincts are POSITIONAL: reach / coincide / converge / unite -- they MOVE the mover TO the
goal. But some wins require TRANSFORMING the mover to BECOME the goal's shape and size -- grow to fill a
cavity, rotate to align an asymmetric piece. When a mover and a candidate goal differ in SIZE or
ORIENTATION, this instinct proposes the transform actions that reduce the delta (GROW / SHRINK to match
extent; ROTATE when the shapes align better under rotation), so the actor pursues 'fill/match the goal',
not just 'reach it'.

This is the eyeballing -- "the mover is tiny in a big complementary cavity, so fill it" -- turned into a
substrate signal: compare the two entities by extent + rotation/scale-invariant shape signature
(shape_identity), name the transforms. The substrate MEASURES + names; the actor verifies (score judges).
Game-agnostic; no game vocabulary.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from shape_identity import oriented_similarity, similarity

# scene region: entities above this row are in the play scene (vs the panels/legend below).
# Not a match knob -- just "which entities are candidate movers/goals" (the playfield, not the UI).
_SCENE_MAX_ROW = 32
_NONGOAL_ROLES = {"hud", "scenery", "wall", "agent"}


def _area(bbox) -> int:
    try:
        r0, c0, r1, c1 = [int(v) for v in bbox]
        return max(0, r1 - r0) * max(0, c1 - c0)
    except Exception:
        return 0


def transform_plan(mover_bbox, mover_sig, goal_bbox, goal_sig, tol: float = 0.25) -> dict:
    """The transforms that would make the mover match/fill the goal.  ``tol`` is the documented
    size-match tolerance (within +/- tol of equal extent counts as matched -- a surfacing band, not a
    tuned detector knob); the DIRECTION (grow vs shrink) is structural, and ROTATE is proposed iff a
    rotation strictly improves the shape overlap."""
    ma, ga = _area(mover_bbox), _area(goal_bbox)
    ratio = (ma / ga) if ga else 0.0
    if ratio and ratio < 1.0 - tol:
        size_action = "GROW"
    elif ratio > 1.0 + tol:
        size_action = "SHRINK"
    else:
        size_action = None
    shape_sim = similarity(mover_sig or "", goal_sig or "")
    oriented = oriented_similarity(mover_sig or "", goal_sig or "")
    # ROTATE iff the shapes overlap BETTER under some rotation than in the current pose (so the mover is
    # asymmetric AND mis-oriented relative to the goal).  Strict '>' -> no magic threshold.
    rotate_action = "ROTATE" if (shape_sim > 0.0 and shape_sim > oriented) else None
    actions = [a for a in (size_action, rotate_action) if a]
    return {"size_ratio": round(ratio, 3), "size_action": size_action,
            "shape_sim": shape_sim, "oriented_sim": oriented, "rotate_action": rotate_action,
            "actions": actions, "matched": not actions}


def match_directive(mover_name: str, goal_name: str, plan: dict) -> Optional[str]:
    """A conform-goal directive for the actor, or ``None`` if the mover already matches the goal."""
    if plan.get("matched") or not plan.get("actions"):
        return None
    bits = []
    if plan["size_action"]:
        rel = ("smaller than" if plan["size_action"] == "GROW" else "larger than")
        bits.append(f"the mover's extent is {plan['size_ratio']}x the goal ({rel} it) -> "
                    f"{plan['size_action']} it to match")
    if plan["rotate_action"]:
        bits.append(f"the shapes align better under rotation (pose-match {plan['shape_sim']:.2f} vs "
                    f"current {plan['oriented_sim']:.2f}) -> ROTATE to align")
    return ("[MATCH-GOAL] the win may require TRANSFORMING the mover to match/fill the goal "
            f"'{goal_name}', not just reaching it: " + "; ".join(bits)
            + ". Pursue: reach the goal, then use the transform actions (" + ", ".join(plan["actions"])
            + ") to make the mover match/fill it.")


def _pick_mover(entities) -> Optional[dict]:
    for e in entities:
        if e.get("role_hypothesis") == "agent" and e.get("shape_sig"):
            return e
    return None


def _pick_goal(entities, mover) -> Optional[dict]:
    """The most goal-like SCENE entity that is not the mover: a non-wall/hud/scenery scene entity,
    largest first (the cavity/target the mover must fill)."""
    best, best_area = None, -1
    for e in entities:
        if e is mover or not e.get("shape_sig"):
            continue
        if e.get("role_hypothesis") in _NONGOAL_ROLES:
            continue
        bb = e.get("bbox_ticks_turn1")
        if not bb or int(bb[0]) >= _SCENE_MAX_ROW:           # must be in the play scene, not the UI
            continue
        a = _area(bb)
        if a > best_area:
            best, best_area = e, a
    return best


def plan_for_entities(entities, tol: float = 0.25) -> Optional[Tuple[str, str, dict]]:
    """Find the mover + the best scene goal candidate and return (mover_name, goal_name, plan), or
    ``None`` if either is missing.  Used by the driver to surface the conform-goal each level."""
    mover = _pick_mover(entities)
    if not mover:
        return None
    goal = _pick_goal(entities, mover)
    if not goal:
        return None
    plan = transform_plan(mover.get("bbox_ticks_turn1"), mover.get("shape_sig"),
                          goal.get("bbox_ticks_turn1"), goal.get("shape_sig"), tol=tol)
    return mover.get("name", "mover"), goal.get("name", "goal"), plan
