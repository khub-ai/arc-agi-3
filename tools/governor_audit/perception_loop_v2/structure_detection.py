"""Game-agnostic structure / obstacle detection — keyed on BEHAVIOR, not looks.

A "structure" (wall, rail, shaft, fixed obstacle) is detected by a signature
that survives regraphing — it CANNOT use color, pattern, size, or assume an
orientation, because any of those can change between competition games (the
adversarial test). The only robust, appearance-independent signature is:

  a tracked entity that stays STATIC while OTHER things move,
  and is not the agent and not a goal/target.

This consumes `world.entities[*].bbox_history` (motion over the delta stream) +
role — nothing visual. Orientation/extent are *derived* from the bbox, never
assumed. The same classifier works whether perception is a VLM (competition) or
any other provider; it only requires that the region was *tracked* at all.

Caveat by design: "static" needs MOTION evidence, so a structure is confirmed
only after a few turns in which the agent acted — not from a single frame.
Single-frame appearance guesses are exactly what fails the adversarial test, so
they are deliberately not used here.
"""
from __future__ import annotations

from typing import Optional

from world_knowledge import WorldKnowledge


# Roles that are NOT structures (they are the actor or things to act on).
_NON_STRUCTURE_ROLES = {
    "agent", "collectable", "target", "goal", "trigger_target",
    "hud", "reference", "pickup", "key",
}


def _role(rec) -> Optional[str]:
    r = getattr(rec, "current_role", None)
    if r:
        return r
    rh = getattr(rec, "role_history", None) or []
    return rh[-1][1] if rh else None


def _bbox_at(rec, turn: int):
    chosen = None
    for (t, bb) in (getattr(rec, "bbox_history", None) or []):
        if t <= turn:
            chosen = bb
    return chosen


def _obs_at(rec, turn: int):
    """Bbox observed EXACTLY at `turn` (None if not observed that turn)."""
    for (t, bb) in (getattr(rec, "bbox_history", None) or []):
        if t == turn:
            return bb
    return None


def _changed(b0, b1, tol: int) -> bool:
    return any(abs(a - b) > tol for a, b in zip(b0, b1))


def _orientation(bb) -> str:
    if not bb:
        return "unknown"
    h, w = bb[2] - bb[0], bb[3] - bb[1]
    if h > w:
        return "vertical"
    if w > h:
        return "horizontal"
    return "square"


def classify_structures(world: WorldKnowledge, min_static_turns: int = 2,
                         tol: int = 0) -> list:
    """Return [{name, orientation, static_turns}] for entities classified as
    standing structures: static across >= `min_static_turns` transitions in
    which at least one OTHER entity moved, and not agent / not a target role.

    Appearance-independent: uses only bbox_history (motion) + role. Recolor,
    retexture, resize, or reorient a structure and this verdict does not change
    — only whether it MOVES does."""
    ents = world.entities
    # Timeline = every turn at which anything was observed (from all entities'
    # bbox_history), so each recorded transition is counted.
    turns: set = set()
    for rec in ents.values():
        for (t, _bb) in (getattr(rec, "bbox_history", None) or []):
            turns.add(t)
    timeline = sorted(turns)
    agent = next((n for n, r in ents.items() if _role(r) == "agent"), None)
    candidates = [n for n, r in ents.items()
                  if n != agent and _role(r) not in _NON_STRUCTURE_ROLES]
    static_while_others_move = {n: 0 for n in candidates}

    def _scene(turn, exclude):
        # multiset of OTHER entities' bboxes observed this turn — captures
        # motion AND appearance/disappearance (so a fast object that fragments
        # into per-turn tracks still registers as "the scene changed").
        return sorted(tuple(_obs_at(r, turn)) for n, r in ents.items()
                      if n != exclude and _obs_at(r, turn) is not None)

    for i in range(1, len(timeline)):
        t0, t1 = timeline[i - 1], timeline[i]
        for n in candidates:
            # candidate must be ACTUALLY observed at both turns (a structure is
            # present every turn); carry-forward would make a one-frame fragment
            # of a fast-moving object look static.
            o0, o1 = _obs_at(ents[n], t0), _obs_at(ents[n], t1)
            cand_static = (o0 is not None and o1 is not None
                           and not _changed(o0, o1, tol))
            if cand_static and _scene(t0, n) != _scene(t1, n):
                static_while_others_move[n] += 1

    out = []
    last_turn = timeline[-1] if timeline else None
    for n in candidates:
        if static_while_others_move[n] >= min_static_turns:
            bb = (_bbox_at(ents[n], last_turn) if last_turn is not None
                  else (ents[n].bbox_history[-1][1]
                        if getattr(ents[n], "bbox_history", None) else None))
            out.append({"name": n, "orientation": _orientation(bb),
                        "static_turns": static_while_others_move[n]})
    # most-persistent first
    out.sort(key=lambda s: -s["static_turns"])
    return out


def structure_names(world: WorldKnowledge, **kw) -> frozenset:
    return frozenset(s["name"] for s in classify_structures(world, **kw))
