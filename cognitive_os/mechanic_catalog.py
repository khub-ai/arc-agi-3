"""Mechanic catalog — typed, declarative records of game mechanics.

A *mechanic* expresses a rule of the form

    when PRECONDITION holds and ACTION is applied,
    the world transitions according to EFFECT,
    and FALSIFIER must continue to hold for the rule to apply.

All four members are typed records drawn from closed vocabularies.
No free-form text at runtime, no game-specific primitives — per the
Prime Directive.  See :doc:`docs/SPEC_mechanic_discovery_and_planning`.

This module owns the data shape and the catalog container only.
Discovery (induction from observation) and use (the planner's
consultation of the catalog) live in their own modules so that
mechanics are a substrate-level concept independent of any specific
game or planner.

Vocabulary closure rationale
----------------------------

Both ``PreconditionKind`` and ``EffectKind`` are intentionally
closed enums.  When a new mechanic class is needed for a new game,
the way to introduce it is to add a member to one of these enums
(extending the closed vocabulary by one), not to invent a parallel
typed structure or a free-form-text rule.  This makes mechanics
*composable* — the planner consumes any composition, and the
discovery loop only needs to consider the existing vocabulary when
proposing hypotheses.

The starting vocabulary covers the cases anticipated for the ARC
games studied so far (auto-slide, consume, transform).  Other game
families will likely need a few more members; that growth is
expected and bounded.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)


# ---------------------------------------------------------------------------
# Closed vocabularies
# ---------------------------------------------------------------------------


class PreconditionKind(str, Enum):
    """Closed vocabulary of precondition shapes.

    Each kind has a small typed parameter dict — see
    :class:`Precondition`.
    """

    #: Cell at ``offset`` relative to agent has role ``role``.  Params:
    #: ``dc: int, dr: int, role: str`` where role is from a small
    #: catalog-level vocabulary (``walkable``, ``wall``, ``consumable``,
    #: ``any``).
    CELL_REL_TO_AGENT_HAS_ROLE = "cell_rel_to_agent_has_role"

    #: Agent's footprint overlaps an entity's bbox.  Params:
    #: ``entity_role: str`` (catalog role name).
    AGENT_AT_ENTITY = "agent_at_entity"

    #: Always true — placeholder for unconditional rules
    #: (e.g. transformations that fire on any state).
    ALWAYS = "always"

    #: Conjunction of two preconditions (referenced by index in the
    #: enclosing :class:`Mechanic`'s ``aux`` list).
    AND = "and"


class EffectKind(str, Enum):
    """Closed vocabulary of effect shapes."""

    #: Agent displaces by ``(dc, dr)`` per slide step, repeating until
    #: ``terminator`` precondition no longer holds.  Params:
    #: ``dc: int, dr: int``.  The terminator is the mechanic's own
    #: precondition by default (slides while precondition true).
    AGENT_SLIDE = "agent_slide"

    #: Entity removed from world; its cells become floor.  Params:
    #: ``entity_role: str``.
    ENTITY_REMOVE = "entity_remove"

    #: Entity changes role.  Params: ``entity_role: str``,
    #: ``new_role: str``.
    ENTITY_TRANSFORM = "entity_transform"

    #: Sequence of effects applied in order.  Effects referenced by
    #: index into ``aux``.
    SEQUENCE = "sequence"


# Sentinel for "any action" — used when a mechanic auto-fires
# regardless of which active action just executed (e.g. an auto-slide
# that fires after any move that satisfies the precondition).
ACTION_AUTO = -1


# ---------------------------------------------------------------------------
# Typed records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Precondition:
    """Typed precondition shape.  ``kind`` selects the schema for
    ``params``; ``aux`` carries child preconditions for composite
    kinds (e.g. ``AND``)."""

    kind:    PreconditionKind
    params:  Mapping[str, Any]    = field(default_factory=dict)
    aux:     Tuple["Precondition", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Effect:
    """Typed effect shape.  Same composition pattern as
    :class:`Precondition`."""

    kind:    EffectKind
    params:  Mapping[str, Any]    = field(default_factory=dict)
    aux:     Tuple["Effect", ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class Mechanic:
    """A single game mechanic.

    Attributes
    ----------
    name
        Human-readable label.  Advisory only; the planner branches
        on ``kind`` enums, not on this label.
    precondition
        Typed precondition.  When this evaluates true, the mechanic
        is applicable.
    action
        Either an explicit ``action_id`` (active mechanic) or
        :data:`ACTION_AUTO` (auto-firing mechanic that fires after
        any state advance whose precondition becomes true).
    effect
        Typed effect describing the state transformation.
    falsifier
        Structural-existence check.  When this evaluates false against
        the current frame, the mechanic is removed from the workspace
        (not merely demoted) — the anti-prior-trap discipline from
        :doc:`docs/SPEC_mechanic_discovery_and_planning`.
    credence
        In ``[0.0, 1.0]``.  Earns its way up through observation;
        demoted by counter-examples.
    support
        Number of observations consistent with this mechanic.
    contradict
        Number of observations that contradicted it.
    """

    name:         str
    precondition: Precondition
    action:       int
    effect:       Effect
    falsifier:    Precondition
    credence:     float = 0.5
    support:      int   = 0
    contradict:   int   = 0


# ---------------------------------------------------------------------------
# Catalog container
# ---------------------------------------------------------------------------


class MechanicCatalog:
    """Container for :class:`Mechanic` records.

    Backed by a list and a small set of indices for the planner's
    common queries (auto-firing mechanics, mechanics keyed by action,
    mechanics whose effect kind matches a category).

    Persistence is intentionally not implemented here — the catalog
    serializes via dataclass-to-dict at the call site, alongside the
    existing CuriosityCatalog runtime JSON.
    """

    __slots__ = ("_mechanics",)

    def __init__(self, mechanics: Iterable[Mechanic] = ()) -> None:
        self._mechanics: List[Mechanic] = list(mechanics)

    # -- mutation ---------------------------------------------------

    def add(self, m: Mechanic) -> None:
        self._mechanics.append(m)

    def replace_at(self, index: int, m: Mechanic) -> None:
        self._mechanics[index] = m

    def remove_at(self, index: int) -> None:
        del self._mechanics[index]

    def increment_support(self, index: int) -> None:
        m = self._mechanics[index]
        self._mechanics[index] = replace(
            m, support=m.support + 1,
            credence=_bump_credence(m.credence, +1),
        )

    def increment_contradict(self, index: int) -> None:
        m = self._mechanics[index]
        self._mechanics[index] = replace(
            m, contradict=m.contradict + 1,
            credence=_bump_credence(m.credence, -1),
        )

    # -- queries ----------------------------------------------------

    def __len__(self) -> int:
        return len(self._mechanics)

    def __iter__(self):
        return iter(self._mechanics)

    def all(self) -> Sequence[Mechanic]:
        return tuple(self._mechanics)

    def auto_firing(self) -> Sequence[Mechanic]:
        """Mechanics with ``action == ACTION_AUTO``.  The planner
        applies these to a fixed point after each state advance."""
        return tuple(m for m in self._mechanics if m.action == ACTION_AUTO)

    def for_action(self, action_id: int) -> Sequence[Mechanic]:
        """Mechanics whose action matches ``action_id``."""
        return tuple(m for m in self._mechanics if m.action == int(action_id))

    def by_effect_kind(self, kind: EffectKind) -> Sequence[Mechanic]:
        return tuple(m for m in self._mechanics if m.effect.kind == kind)

    def index_of(self, m: Mechanic) -> Optional[int]:
        try:
            return self._mechanics.index(m)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _bump_credence(c: float, delta: int) -> float:
    """One-step credence update — small linear nudge, clamped to
    ``[0, 1]``.  Matches the lightweight update style used elsewhere
    in the credence machinery (see
    :class:`cognitive_os.credence` for the full update rules).
    """
    step = 0.05
    return max(0.0, min(1.0, c + (step if delta > 0 else -step)))


# ---------------------------------------------------------------------------
# Construction helpers — closed-vocabulary shortcuts the planner /
# discovery loop use to build common preconditions and effects.  Not
# game-specific — these compose primitives only.
# ---------------------------------------------------------------------------


def cell_above_agent_walkable() -> Precondition:
    """Composition: 'the cell at offset (0, -1) from the agent has
    role walkable.'  Used in auto-slide-up.  Substrate-level: no
    game-specific identifiers."""
    return Precondition(
        kind=PreconditionKind.CELL_REL_TO_AGENT_HAS_ROLE,
        params={"dc": 0, "dr": -1, "role": "walkable"},
    )


def cell_in_direction_walkable(dc: int, dr: int) -> Precondition:
    """Generalization of :func:`cell_above_agent_walkable` to any
    direction.  Composes the same primitive — direction-agnostic."""
    return Precondition(
        kind=PreconditionKind.CELL_REL_TO_AGENT_HAS_ROLE,
        params={"dc": int(dc), "dr": int(dr), "role": "walkable"},
    )


def agent_slide_effect(dc: int, dr: int) -> Effect:
    """Effect: agent slides by ``(dc, dr)`` per step until the
    precondition no longer holds."""
    return Effect(
        kind=EffectKind.AGENT_SLIDE,
        params={"dc": int(dc), "dr": int(dr)},
    )


def consume_effect(entity_role: str) -> Effect:
    """Effect: the entity at the agent's position (of given role) is
    removed; its cells become floor."""
    return Effect(
        kind=EffectKind.ENTITY_REMOVE,
        params={"entity_role": str(entity_role)},
    )


def entity_visible_falsifier(entity_role: str) -> Precondition:
    """Structural falsifier: 'at least one entity of role
    ``entity_role`` is visible in the frame.'  Note: with role
    ``walkable`` this falsifier is trivially true on any playable
    frame, and the mechanic's falsifier discipline is then carried
    by the behavioral check (observed events).  Use a more specific
    role for mechanics whose existence really depends on a
    particular entity class being present."""
    return Precondition(
        kind=PreconditionKind.AGENT_AT_ENTITY,
        params={"entity_role": str(entity_role)},
    )


def always_true() -> Precondition:
    return Precondition(kind=PreconditionKind.ALWAYS)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------
#
# The catalog serializes to a list of plain dicts (and back) so it
# can live inside an existing runtime-JSON file alongside other
# accumulated state.  All structures are typed enums plus ordinary
# dicts at the leaves; the JSON form stores closed-vocabulary
# identifiers as strings (each enum's ``.value``).


def precondition_to_dict(p: Precondition) -> Dict[str, Any]:
    return {
        "kind":   p.kind.value,
        "params": dict(p.params),
        "aux":    [precondition_to_dict(c) for c in p.aux],
    }


def precondition_from_dict(d: Mapping[str, Any]) -> Precondition:
    return Precondition(
        kind   = PreconditionKind(d["kind"]),
        params = dict(d.get("params") or {}),
        aux    = tuple(
            precondition_from_dict(c) for c in (d.get("aux") or [])
        ),
    )


def effect_to_dict(e: Effect) -> Dict[str, Any]:
    return {
        "kind":   e.kind.value,
        "params": dict(e.params),
        "aux":    [effect_to_dict(c) for c in e.aux],
    }


def effect_from_dict(d: Mapping[str, Any]) -> Effect:
    return Effect(
        kind   = EffectKind(d["kind"]),
        params = dict(d.get("params") or {}),
        aux    = tuple(effect_from_dict(c) for c in (d.get("aux") or [])),
    )


def mechanic_to_dict(m: Mechanic) -> Dict[str, Any]:
    return {
        "name":         m.name,
        "precondition": precondition_to_dict(m.precondition),
        "action":       int(m.action),
        "effect":       effect_to_dict(m.effect),
        "falsifier":    precondition_to_dict(m.falsifier),
        "credence":     float(m.credence),
        "support":      int(m.support),
        "contradict":   int(m.contradict),
    }


def mechanic_from_dict(d: Mapping[str, Any]) -> Mechanic:
    return Mechanic(
        name         = str(d["name"]),
        precondition = precondition_from_dict(d["precondition"]),
        action       = int(d["action"]),
        effect       = effect_from_dict(d["effect"]),
        falsifier    = precondition_from_dict(d["falsifier"]),
        credence     = float(d.get("credence", 0.5)),
        support      = int(d.get("support", 0)),
        contradict   = int(d.get("contradict", 0)),
    )


def catalog_to_list(c: MechanicCatalog) -> List[Dict[str, Any]]:
    return [mechanic_to_dict(m) for m in c.all()]


def catalog_from_list(records: Iterable[Mapping[str, Any]]) -> MechanicCatalog:
    return MechanicCatalog(mechanic_from_dict(r) for r in records)
