"""Instinct-driven goal priors — propose TEMPORARY goals when the win condition
is unknown.

A COS player gets no goal up front; in many games the score gives no gradient
until the win itself.  But human perception doesn't sit idle: seeing structure,
it forms an *urge* to act on it — near-identical things "want" to be made the
same or brought together (the grouping / sorting instinct), an object near a
matching frame "wants" to go in it (containment), a row of things "wants" to be
completed (closure).  Games are built to exploit these instincts, so they are
fair COS priors — like the physics priors — not game-specific injection.

This module turns such instincts into ranked, TEMPORARY goal hypotheses.  The
solver pursues the top one; the SCORE is the only judge — if pursuing a goal
moves the score, the instinct was right; if achieving it does nothing, refute it
and try the next.  Each Goal carries a hard predicate (is it achieved?) and a
soft distance (how far from it) so the solver can descend toward it even when
its forward model is too weak to plan exactly.

Substrate-agnostic: priors are about RELATIONS between perceived entities, never
about a specific game.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Goal:
    kind: str
    desc: str
    predicate: Callable          # state -> bool  (achieved?)
    distance: Callable           # state -> float (>=0, 0 when achieved)
    priority: float = 0.5


def _pos(state, eid):
    e = state.get(eid)
    return e.pos if e else None


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _all_coincide(state, ids):
    ps = [_pos(state, i) for i in ids]
    ps = [p for p in ps if p is not None]
    return len(ps) >= 2 and len(set(ps)) == 1


def _spread(state, ids):
    ps = [_pos(state, i) for i in ids]
    ps = [p for p in ps if p is not None]
    if len(ps) < 2:
        return 0.0
    return float(sum(_manhattan(ps[i], ps[j])
                     for i in range(len(ps)) for j in range(i + 1, len(ps))))


def propose_goals(state) -> list[Goal]:
    """Rank temporary goals for `state` from perceptual instincts.

    Currently implements the SIMILARITY -> CONVERGE/UNIFY instinct (the one the
    benchmark leans on most): any set of >=2 entities sharing a class is a
    candidate to be brought together / made to coincide.  More priors
    (containment, alignment, completion) slot in here with the same Goal
    contract; the solver tries them in priority order.
    """
    goals: list[Goal] = []
    by_class: dict = {}
    for e in state.entities:
        by_class.setdefault(e.cls, []).append(e.eid)
    for cls, ids in by_class.items():
        if len(ids) >= 2:
            ids = sorted(ids)
            # The more similar (more identical members) and the fewer of them,
            # the stronger the urge to unite — a clean pair is the prototypical
            # case.  Priority peaks for an exact pair.
            prio = 0.8 if len(ids) == 2 else 0.6
            goals.append(Goal(
                kind="unify_similar",
                desc=f"bring the {len(ids)} similar entities {ids} together "
                     f"(make them coincide)",
                predicate=(lambda s, ids=ids: _all_coincide(s, ids)),
                distance=(lambda s, ids=ids: _spread(s, ids)),
                priority=prio,
            ))
    return sorted(goals, key=lambda g: g.priority, reverse=True)
