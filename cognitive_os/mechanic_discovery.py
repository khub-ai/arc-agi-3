"""Mechanic discovery — inducing typed mechanics from observed
gameplay events.

Per :doc:`docs/SPEC_mechanic_discovery_and_planning`, mechanics are
not hardcoded.  They are discovered by:

1. **Anomaly detection.** After every action, compare the observed
   agent displacement (and other state changes) against the action's
   expected effect.  When the observation is categorically larger
   than the prediction, record a :class:`MechanicEvent` with the
   local cell-neighborhood context.

2. **Hypothesis induction.** Periodically, scan accumulated events
   and enumerate candidate predicates from the closed precondition
   vocabulary in :mod:`cognitive_os.mechanic_catalog`.  Commit the
   shortest predicate that fits every event with zero
   counter-examples.

3. **Validation.** The planner uses the committed mechanic.
   Subsequent observations either confirm or refute it; credence
   moves accordingly via the catalog's
   ``increment_support`` / ``increment_contradict`` methods.

This module owns steps 1 and 2.  Step 3 is the planner's
responsibility (it logs mechanic firings; the discovery loop
revisits accumulated events when the catalog tells it to).

Per the Prime Directive: no game-specific primitives.  The hypothesis
space is enumerated from the closed vocabulary; new game mechanics
are accommodated by extending the vocabulary (the closed enum) in
one place, never by handcrafting a per-game rule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from cognitive_os.mechanic_catalog import (
    ACTION_AUTO,
    Effect,
    EffectKind,
    Mechanic,
    Precondition,
    PreconditionKind,
    agent_slide_effect,
    cell_in_direction_walkable,
)


# ---------------------------------------------------------------------------
# Recorded event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MechanicEvent:
    """A single categorical-surprise observation.

    Captured at the moment a state transition disagreed
    significantly with the action-effect model the catalog had at
    the time.  The fields are deliberately minimal — anything richer
    would be incidental detail that biases the rule learner toward
    one mechanic family.

    Attributes
    ----------
    pos_before, pos_after
        Agent ``(col, row)`` before and after the surprising
        transition.
    action_id
        Which action was issued.  Used both to attribute the
        observation and to recognize the special case of an
        :data:`ACTION_AUTO`-eligible mechanic (no active action
        could have caused this).
    expected_displacement
        ``(dc, dr)`` the action's median-displacement model
        predicted.  The observation is "surprising" because the
        actual displacement is much larger.
    neighborhood
        Roles of the cells immediately adjacent to ``pos_before``,
        keyed by direction label
        (``"up"``, ``"down"``, ``"left"``, ``"right"``).  Each value
        is one of ``"walkable"``, ``"wall"``, ``"consumable"``,
        ``"unknown"``.  This is the substrate signal the inducer
        uses to enumerate predicate candidates.
    """

    pos_before:            Tuple[int, int]
    pos_after:             Tuple[int, int]
    action_id:             int
    expected_displacement: Tuple[int, int]
    neighborhood:          Mapping[str, str]


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------


def neighborhood_roles(
    pos:         Tuple[int, int],
    pgrid,
    frame_shape: Tuple[int, int],
) -> Mapping[str, str]:
    """Return the roles of cells immediately adjacent to ``pos``.

    A minimal classifier: True in the passability grid → walkable,
    False → wall.  Out-of-frame neighbors are ``wall``.  This is
    sufficient for inducing auto-slide-style mechanics; richer role
    discrimination (e.g. distinguishing consumable from wall) is
    handled by the calling layer enriching the grid before calling
    here.
    """
    H, W = frame_shape
    col, row = int(pos[0]), int(pos[1])
    out = {}
    for name, (dc, dr) in (("up", (0, -1)), ("down", (0, 1)),
                           ("left", (-1, 0)), ("right", (1, 0))):
        nc = col + dc
        nr = row + dr
        if not (0 <= nr < H and 0 <= nc < W):
            out[name] = "wall"
            continue
        try:
            walkable = bool(pgrid[nr, nc])
        except Exception:
            walkable = False
        out[name] = "walkable" if walkable else "wall"
    return out


def detect_mechanic_event(
    pos_before:             Tuple[int, int],
    pos_after:              Tuple[int, int],
    action_id:              int,
    expected_displacement:  Tuple[int, int],
    pgrid,
    frame_shape:            Tuple[int, int],
    *,
    surprise_threshold:     int = 2,
) -> Optional[MechanicEvent]:
    """Return a :class:`MechanicEvent` when the observed agent
    displacement is categorically larger than the action's expected
    displacement.

    The categorical-surprise rule: an observation is surprising when
    the observed Manhattan magnitude exceeds
    ``surprise_threshold * expected_magnitude + 1``.  The ``+ 1``
    handles the zero-expected case (any displacement is suspicious
    when no displacement was expected).

    The default threshold of 2 catches auto-slides cleanly
    (slide-up-many-rows after a step-east is a > 2× surprise) while
    tolerating ordinary step-magnitude noise.

    Returns ``None`` when the observation fits the action's model.
    """
    obs_dc = int(pos_after[0]) - int(pos_before[0])
    obs_dr = int(pos_after[1]) - int(pos_before[1])
    exp_dc, exp_dr = int(expected_displacement[0]), int(expected_displacement[1])
    obs_mag = abs(obs_dc) + abs(obs_dr)
    exp_mag = abs(exp_dc) + abs(exp_dr)
    if obs_mag <= surprise_threshold * exp_mag + 1:
        return None
    return MechanicEvent(
        pos_before            = (int(pos_before[0]), int(pos_before[1])),
        pos_after             = (int(pos_after[0]),  int(pos_after[1])),
        action_id             = int(action_id),
        expected_displacement = (exp_dc, exp_dr),
        neighborhood          = neighborhood_roles(
            pos_before, pgrid, frame_shape,
        ),
    )


# ---------------------------------------------------------------------------
# Rule induction
# ---------------------------------------------------------------------------


_DIRECTION_DELTAS: Sequence[Tuple[str, int, int]] = (
    ("up",    0, -1),
    ("down",  0, +1),
    ("left", -1,  0),
    ("right", +1,  0),
)


def induce_auto_slide_mechanic(
    events:  Sequence[MechanicEvent],
    *,
    min_events: int = 2,
) -> Optional[Mechanic]:
    """Try to induce an auto-slide-style mechanic that explains the
    accumulated events.

    Hypothesis space: for each cardinal direction, the mechanic
    "agent slides in this direction while the cell in this direction
    is walkable."  For each candidate, check:

    * Every event's observed displacement points in the candidate's
      direction (sign of the relevant axis matches).
    * Every event's ``neighborhood`` has the candidate direction
      labeled ``walkable``.

    The first candidate that fits every event with zero
    counter-examples wins.  When two candidates fit (e.g. both up
    and down — only possible when no events disambiguate), neither
    is committed; the inducer prefers to wait for more evidence.

    Returns ``None`` when no candidate fits or when fewer than
    ``min_events`` events are available.
    """
    if len(events) < min_events:
        return None

    matches: List[Tuple[str, int, int]] = []
    for label, dc, dr in _DIRECTION_DELTAS:
        if all(_event_matches_slide(ev, dc, dr, label) for ev in events):
            matches.append((label, dc, dr))

    if len(matches) != 1:
        # Either no candidate fits or multiple fit — defer commit.
        return None

    label, dc, dr = matches[0]
    return Mechanic(
        name         = f"auto_slide_{label}",
        precondition = cell_in_direction_walkable(dc, dr),
        action       = ACTION_AUTO,
        effect       = agent_slide_effect(dc, dr),
        falsifier    = cell_in_direction_walkable(dc, dr),
        credence     = 0.6,
        support      = len(events),
        contradict   = 0,
    )


def _event_matches_slide(
    event:        MechanicEvent,
    slide_dc:     int,
    slide_dr:     int,
    predicate_label: str,
) -> bool:
    """Check whether ``event``'s displacement is consistent with a
    slide of ``(slide_dc, slide_dr)`` AND the corresponding
    neighborhood direction is walkable."""
    obs_dc = event.pos_after[0] - event.pos_before[0]
    obs_dr = event.pos_after[1] - event.pos_before[1]
    # Sign check on the slide axis.  The agent's net displacement
    # must have the same sign as the slide direction along the
    # slide's primary axis, and must not move significantly against
    # it on the orthogonal axis.
    if slide_dc != 0:
        if obs_dc * slide_dc <= 0:
            return False
        # Off-axis tolerance: allow a small orthogonal component
        # (the agent's enabling step on the orthogonal axis).
        if abs(obs_dr) > abs(obs_dc):
            return False
    if slide_dr != 0:
        if obs_dr * slide_dr <= 0:
            return False
        if abs(obs_dc) > abs(obs_dr):
            return False
    # Neighborhood predicate must hold at pos_before.
    if event.neighborhood.get(predicate_label) != "walkable":
        return False
    return True


# ---------------------------------------------------------------------------
# Consume-mechanic induction (sibling of auto-slide).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConsumeEvent:
    """A recorded ACTION6 (or other click-style action) that
    removed an entity.  Sibling shape to :class:`MechanicEvent`."""

    action_id:    int
    entity_role:  str
    entity_id:    str
    click_xy:     Tuple[int, int]


def induce_consume_mechanic(
    events:      Sequence[ConsumeEvent],
    *,
    min_events:  int = 2,
) -> Optional[Mechanic]:
    """Induce a consume-action mechanic when ``min_events`` or more
    consume events agree on the (action_id, entity_role) pair.
    Returns the committed mechanic, or ``None`` when evidence is
    insufficient or inconsistent."""
    if len(events) < min_events:
        return None
    # Group events by (action_id, entity_role) and pick the largest
    # group; commit if it covers every event.
    counts: dict = {}
    for ev in events:
        key = (int(ev.action_id), str(ev.entity_role))
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best_key, best_count = max(counts.items(), key=lambda kv: kv[1])
    if best_count != len(events):
        return None
    action_id, entity_role = best_key
    return Mechanic(
        name         = f"consume_{entity_role}",
        precondition = Precondition(
            kind   = PreconditionKind.AGENT_AT_ENTITY,
            params = {"entity_role": entity_role},
        ),
        action       = int(action_id),
        effect       = Effect(
            kind   = EffectKind.ENTITY_REMOVE,
            params = {"entity_role": entity_role},
        ),
        falsifier    = Precondition(
            kind   = PreconditionKind.AGENT_AT_ENTITY,
            params = {"entity_role": entity_role},
        ),
        credence     = 0.6,
        support      = len(events),
        contradict   = 0,
    )
