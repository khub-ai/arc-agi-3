"""Affordance DSL — composable AST over closed primitive vocabularies.

An affordance is a small program in a domain-specific language whose
leaves are drawn from two closed alphabets:

* **Trigger primitives** — predicates over (world_state, agent_state,
  entity_instance) that determine when an affordance fires.
* **Effect primitives** — actions (mutations of world or agent state)
  that describe what happens when the trigger holds.

Both alphabets are extended only by deliberate engine changes; their
COMPOSITIONS via the connectives (``And``, ``Or``, ``Not``, ``Seq``,
``Par``, etc.) and through structural primitives that DERIVE properties
from an entity's appearance (``AttachedSideOf``, ``OrientationOf``,
``PerpendicularTo``, ``NeighborOf``, ``CellOf``) are unbounded.  Adding
a new game mechanic does not require new engine branches; it requires
either an existing template applied to a newly-observed visual class,
or — in genuinely novel cases — a new leaf primitive added once and
shared.

This module hosts the AST node types and the evaluator / matcher
interfaces.  See :mod:`cognitive_os.affordance_claim` for the wrapping
:class:`AffordanceClaim` and the resolver that joins claims against
live instances.  See :doc:`docs/SPEC_entity_affordances.md` for the
design rationale and the inference/prediction contract.

The DSL is engine-clean: no game ids, no specific bitmap fingerprints,
no human-readable kind labels participate in any node's semantics.  A
launcher in ls20 and a docking station in a robotics workspace evaluate
through the same primitives.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Enums — closed vocabularies for cardinal directions, axes, sides.
# ---------------------------------------------------------------------------


class Direction(Enum):
    """Cardinal directions in a grid world.  Diagonal directions are
    omitted from the initial vocabulary; add only when an observed
    mechanic requires diagonal motion as a primitive operation."""
    N = "N"
    S = "S"
    E = "E"
    W = "W"


class Axis(Enum):
    """Major axis of an elongated sprite.  ``POINT`` is the degenerate
    case for compact sprites whose orientation is undefined."""
    HORIZONTAL = "horizontal"
    VERTICAL   = "vertical"
    POINT      = "point"


class Side(Enum):
    """One of the four sides of an entity's bounding box.  Returned by
    ``AttachedSideOf`` when a bar-like sprite is fused to a wall on a
    specific side."""
    N = "N"
    S = "S"
    E = "E"
    W = "W"


_PERPENDICULAR_DIRECTION_TABLE = {
    # (axis, away_from_side) -> the Direction perpendicular to axis
    # pointing away from side.
    (Axis.HORIZONTAL, Side.N): Direction.S,
    (Axis.HORIZONTAL, Side.S): Direction.N,
    (Axis.VERTICAL,   Side.E): Direction.W,
    (Axis.VERTICAL,   Side.W): Direction.E,
}


_DIRECTION_DELTA: Mapping[Direction, Tuple[int, int]] = {
    Direction.N: (-1,  0),
    Direction.S: ( 1,  0),
    Direction.E: ( 0,  1),
    Direction.W: ( 0, -1),
}


# ---------------------------------------------------------------------------
# AST base.
# ---------------------------------------------------------------------------


class Node:
    """Base class for affordance DSL AST nodes.

    Every concrete node provides:

    * ``canonical_key()`` — hashable tuple identifying the node's
      structural form.  Used for dedup and serialisation.
    * ``evaluate(env)`` — produce a concrete value (cell, direction,
      bool, etc.) given the environment dict, which carries the live
      WorldState, the bound instance ``self``, the agent, and any
      ancillary context like ``passable_grid``.  ``env`` is read-only
      to nodes; nodes never mutate state.
    """

    def canonical_key(self) -> tuple:
        raise NotImplementedError

    def evaluate(self, env: Mapping[str, Any]) -> Any:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Structural reference primitives — derived from an entity's visual
# state, not from absolute world coordinates.  Each primitive returns a
# concrete value when bound to a specific entity instance in a specific
# frame.  ``Undetermined`` propagates upward when the visual evidence
# is insufficient.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelfRef(Node):
    """The entity instance to which the affordance is bound."""

    def canonical_key(self) -> tuple:
        return ("self",)

    def evaluate(self, env):
        return env.get("self")


@dataclass(frozen=True)
class AgentRef(Node):
    """The agent entity in the current frame."""

    def canonical_key(self) -> tuple:
        return ("agent",)

    def evaluate(self, env):
        return env.get("agent")


@dataclass(frozen=True)
class CellOf(Node):
    """Returns the centroid cell of an entity reference."""
    entity: Node

    def canonical_key(self) -> tuple:
        return ("cell_of", self.entity.canonical_key())

    def evaluate(self, env):
        ent = self.entity.evaluate(env)
        if ent is None:
            return None
        return _entity_cell(ent, env)


@dataclass(frozen=True)
class AttachedSideOf(Node):
    """For a bar-like sprite fused to a wall, the side of its bbox that
    touches the wall.  Returns a :class:`Side`, or ``None`` when the
    sprite is point-like or ambiguous."""
    entity: Node

    def canonical_key(self) -> tuple:
        return ("attached_side_of", self.entity.canonical_key())

    def evaluate(self, env):
        ent = self.entity.evaluate(env)
        if ent is None:
            return None
        # Consult the visual_id bundle for a cached attached_side
        # produced by entity publication.  Down-stream pixel analysis
        # populates this; absence => undetermined.
        side = _entity_visual_prop(ent, "attached_side")
        if side is None:
            return None
        try:
            return Side(side)
        except ValueError:
            return None


@dataclass(frozen=True)
class OrientationOf(Node):
    """``HORIZONTAL`` or ``VERTICAL`` for elongated sprites; ``POINT``
    for compact ones."""
    entity: Node

    def canonical_key(self) -> tuple:
        return ("orientation_of", self.entity.canonical_key())

    def evaluate(self, env):
        ent = self.entity.evaluate(env)
        if ent is None:
            return None
        axis = _entity_visual_prop(ent, "orientation_axis")
        if axis is None:
            # Derive from bbox dimensions as a fallback when the
            # entity publisher didn't tag the axis explicitly.
            bh = _entity_visual_prop(ent, "bbox_h") or 0
            bw = _entity_visual_prop(ent, "bbox_w") or 0
            if bh == 0 or bw == 0:
                return None
            if bw >= 2 * bh:
                return Axis.HORIZONTAL
            if bh >= 2 * bw:
                return Axis.VERTICAL
            return Axis.POINT
        try:
            return Axis(axis)
        except ValueError:
            return None


@dataclass(frozen=True)
class PerpendicularTo(Node):
    """Given an axis and a side excluded from the result, return the
    cardinal direction perpendicular to the axis pointing away from
    that side.  Both inputs are AST nodes evaluated against ``env``."""
    axis:      Node
    away_from: Node

    def canonical_key(self) -> tuple:
        return ("perpendicular_to",
                self.axis.canonical_key(),
                self.away_from.canonical_key())

    def evaluate(self, env):
        ax = self.axis.evaluate(env)
        sd = self.away_from.evaluate(env)
        if not isinstance(ax, Axis) or not isinstance(sd, Side):
            return None
        return _PERPENDICULAR_DIRECTION_TABLE.get((ax, sd))


@dataclass(frozen=True)
class NeighborOf(Node):
    """The cell adjacent to an entity on its named side, at the given
    distance (default 1 cell).  Returns ``None`` if the entity or side
    is undetermined."""
    entity:   Node
    side:     Node
    distance: int = 1

    def canonical_key(self) -> tuple:
        return ("neighbor_of",
                self.entity.canonical_key(),
                self.side.canonical_key(),
                int(self.distance))

    def evaluate(self, env):
        ent = self.entity.evaluate(env)
        sd  = self.side.evaluate(env)
        if ent is None or not isinstance(sd, Side):
            return None
        c0 = _entity_cell(ent, env)
        if c0 is None:
            return None
        # The side delta maps onto the cardinal directions: side N
        # means the neighbor is on the entity's north (delta -1 row).
        d = _DIRECTION_DELTA[Direction(sd.value)]
        return (int(c0[0]) + int(d[0]) * int(self.distance),
                int(c0[1]) + int(d[1]) * int(self.distance))


@dataclass(frozen=True)
class CellInDirection(Node):
    """The cell at ``distance`` cells away from ``origin_cell`` in the
    named ``direction``.  Useful when the direction is already derived
    (e.g. from ``PerpendicularTo``) and we want the launch cell as a
    geometric offset rather than via the side-keyed ``NeighborOf``.

    Both inputs are AST nodes evaluated against ``env``."""
    origin_cell: Node
    direction:   Node
    distance:    int = 1

    def canonical_key(self) -> tuple:
        return ("cell_in_direction",
                self.origin_cell.canonical_key(),
                self.direction.canonical_key(),
                int(self.distance))

    def evaluate(self, env):
        c0 = self.origin_cell.evaluate(env)
        d  = self.direction.evaluate(env)
        if c0 is None or not isinstance(d, Direction):
            return None
        delta = _DIRECTION_DELTA[d]
        return (int(c0[0]) + int(delta[0]) * int(self.distance),
                int(c0[1]) + int(delta[1]) * int(self.distance))


# ---------------------------------------------------------------------------
# Predicate primitives — boolean-valued nodes, evaluable against the
# live world state.  These are the leaves used in trigger expressions
# and in the ``until=`` clauses of procedural effects.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentAt(Node):
    """The agent currently occupies the given cell expression."""
    cell: Node

    def canonical_key(self) -> tuple:
        return ("agent_at", self.cell.canonical_key())

    def evaluate(self, env):
        target_cell = self.cell.evaluate(env)
        if target_cell is None:
            return None
        ag = env.get("agent_cell")
        if ag is None:
            return None
        return (int(ag[0]), int(ag[1])) == (int(target_cell[0]),
                                            int(target_cell[1]))


@dataclass(frozen=True)
class AgentAction(Node):
    """The agent dispatched the named action on the most recent step."""
    action_id: Optional[str] = None  # None means any action

    def canonical_key(self) -> tuple:
        return ("agent_action", self.action_id)

    def evaluate(self, env):
        last = env.get("last_action")
        if last is None:
            return None
        if self.action_id is None:
            return True
        return str(last) == str(self.action_id)


@dataclass(frozen=True)
class IsBarrier(Node):
    """True when the cell is impassable for agent motion: wall,
    gate-blocked, or occupied by an impassable entity.  Reads
    ``env['passable_grid']`` and ``env['cell_system']`` (passable_grid
    is pixel-keyed); a cell is a barrier when its centre pixel is not
    passable."""
    cell: Node

    def canonical_key(self) -> tuple:
        return ("is_barrier", self.cell.canonical_key())

    def evaluate(self, env):
        c = self.cell.evaluate(env)
        if c is None:
            return None
        return not _cell_is_passable(env, (int(c[0]), int(c[1])))


@dataclass(frozen=True)
class IsFloor(Node):
    """True when the cell is fully passable (the inverse of
    :class:`IsBarrier` for the simple wall case; richer predicates can
    distinguish lethal/non-lethal floor in domains where it matters)."""
    cell: Node

    def canonical_key(self) -> tuple:
        return ("is_floor", self.cell.canonical_key())

    def evaluate(self, env):
        c = self.cell.evaluate(env)
        if c is None:
            return None
        return _cell_is_passable(env, (int(c[0]), int(c[1])))


# ---------------------------------------------------------------------------
# Logical connectives.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class And(Node):
    """Conjunction of two predicates.  Tri-valued: ``False`` if any
    child is ``False``; ``None`` if any child is ``None`` and none is
    ``False``; ``True`` otherwise."""
    left:  Node
    right: Node

    def canonical_key(self) -> tuple:
        return ("and",
                self.left.canonical_key(),
                self.right.canonical_key())

    def evaluate(self, env):
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        if lv is False or rv is False:
            return False
        if lv is None or rv is None:
            return None
        return True


@dataclass(frozen=True)
class Or(Node):
    left:  Node
    right: Node

    def canonical_key(self) -> tuple:
        return ("or",
                self.left.canonical_key(),
                self.right.canonical_key())

    def evaluate(self, env):
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        if lv is True or rv is True:
            return True
        if lv is None or rv is None:
            return None
        return False


@dataclass(frozen=True)
class Not(Node):
    inner: Node

    def canonical_key(self) -> tuple:
        return ("not", self.inner.canonical_key())

    def evaluate(self, env):
        v = self.inner.evaluate(env)
        if v is None:
            return None
        return not v


# ---------------------------------------------------------------------------
# Effect primitives — procedural actions describing what happens when
# the trigger holds.  ``evaluate(env)`` for an Effect node produces a
# typed RESULT structure rather than a mutating side-effect; the caller
# decides whether to commit the predicted result.  This separation is
# what allows the same AST to serve PREDICTION (run evaluator, observe
# result) and INFERENCE (compute what the effect WOULD produce, compare
# to what was actually observed).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffectResult:
    """Typed return value for an effect's ``evaluate``.  ``kind`` is a
    closed vocabulary; ``payload`` is a small dict of typed fields.
    """
    kind:    str
    payload: tuple  # tuple-of-pairs for hashability

    def get(self, key, default=None):
        for (k, v) in self.payload:
            if k == key:
                return v
        return default


def _result(kind: str, **payload) -> EffectResult:
    return EffectResult(kind=kind,
                        payload=tuple(sorted(payload.items())))


@dataclass(frozen=True)
class Traverse(Node):
    """Procedural effect: the agent walks in ``direction``, one cell
    per step, halting when ``until`` becomes true on the next cell or
    when an external limit is reached.  Returns an :class:`EffectResult`
    with ``kind='traverse'`` containing the start cell, the terminal
    cell, the number of cells traversed, and whether the halt was
    caused by the predicate.

    Maximum traversal distance is bounded by ``max_steps`` (default
    64, the side of an ARC frame) to prevent runaway evaluation."""
    direction: Node
    until:     Node
    max_steps: int = 64

    def canonical_key(self) -> tuple:
        return ("traverse",
                self.direction.canonical_key(),
                self.until.canonical_key(),
                int(self.max_steps))

    def evaluate(self, env):
        d = self.direction.evaluate(env)
        if not isinstance(d, Direction):
            return None
        ag = env.get("agent_cell")
        if ag is None:
            return None
        delta = _DIRECTION_DELTA[d]
        cur_r, cur_c = int(ag[0]), int(ag[1])
        steps = 0
        # Iterate forward; the ``until`` predicate is evaluated
        # against an env where the candidate ``next_cell`` is set.
        for _ in range(int(self.max_steps)):
            nr, nc = cur_r + int(delta[0]), cur_c + int(delta[1])
            sub_env = dict(env)
            sub_env["next_cell"] = (nr, nc)
            halt = self.until.evaluate(sub_env)
            if halt is True:
                break
            if halt is None:
                # Insufficient information to evaluate; refuse to
                # extrapolate further.
                return None
            cur_r, cur_c = nr, nc
            steps += 1
        return _result(
            "traverse",
            start_cell  = (int(ag[0]), int(ag[1])),
            end_cell    = (cur_r, cur_c),
            steps       = steps,
            halted_by_predicate = True,
        )


@dataclass(frozen=True)
class NextCellInDir(Node):
    """The cell one step away from the current evaluation context in
    the direction set on ``env['traverse_dir']`` (the direction of an
    enclosing :class:`Traverse`).  Resolves to ``env['next_cell']``
    when present (the traverse evaluator sets it explicitly)."""

    def canonical_key(self) -> tuple:
        return ("next_cell_in_dir",)

    def evaluate(self, env):
        nc = env.get("next_cell")
        if nc is None:
            return None
        return (int(nc[0]), int(nc[1]))


@dataclass(frozen=True)
class Seq(Node):
    """Sequential effect composition: evaluate ``first`` then
    ``second``.  The result is a tuple of both child results."""
    first:  Node
    second: Node

    def canonical_key(self) -> tuple:
        return ("seq",
                self.first.canonical_key(),
                self.second.canonical_key())

    def evaluate(self, env):
        r1 = self.first.evaluate(env)
        r2 = self.second.evaluate(env)
        return _result("seq",
                       first  = r1.payload if isinstance(r1, EffectResult)
                                            else r1,
                       second = r2.payload if isinstance(r2, EffectResult)
                                            else r2)


# ---------------------------------------------------------------------------
# Helpers — env property lookups.  Kept private so the AST nodes don't
# leak implementation details (e.g. which key on the entity holds the
# cached attached_side).
# ---------------------------------------------------------------------------


def _entity_visual_prop(entity, key, default=None):
    """Read a property from an entity's ``properties`` bundle.  Tolerant
    of dict / object / EntityModel shapes — the visual store yields
    dataclasses in some paths and plain dicts in others."""
    if entity is None:
        return default
    props = getattr(entity, "properties", None)
    if props is None and isinstance(entity, Mapping):
        props = entity
    if props is None:
        return default
    try:
        return props.get(key, default)
    except AttributeError:
        return default


def _entity_cell(entity, env):
    """Resolve an entity's centroid cell.  Reads the cached
    ``centroid_cell`` if present; otherwise computes from
    ``bbox_top_left`` + ``bbox_h``/``bbox_w`` and the env's cell system.
    Returns ``None`` when neither path produces a cell."""
    if entity is None:
        return None
    cc = _entity_visual_prop(entity, "centroid_cell")
    if cc is not None:
        try:
            return (int(cc[0]), int(cc[1]))
        except (TypeError, ValueError, IndexError):
            return None
    btl = _entity_visual_prop(entity, "bbox_top_left")
    if btl is None:
        return None
    cs = env.get("cell_system")
    if cs is None:
        return None
    try:
        bh = int(_entity_visual_prop(entity, "bbox_h") or 1)
        bw = int(_entity_visual_prop(entity, "bbox_w") or 1)
        pr_mid = int(btl[0]) + bh // 2
        pc_mid = int(btl[1]) + bw // 2
        return cs.pix_to_cell(pr_mid, pc_mid)
    except (TypeError, ValueError, AttributeError, IndexError):
        return None


def _cell_is_passable(env, cell) -> Optional[bool]:
    """Check whether a cell is passable per env['passable_grid'] and
    env['cell_system'].  Returns ``None`` when either is missing
    (insufficient information for the predicate).

    Semantics: the agent's centroid lands on the cell's CENTER pixel,
    so the cell is passable when its center pixel is on a non-wall
    palette.  Cells containing entity overlays (bouncer bars, refuel
    rings, glyphs) that don't intrude on the center remain passable
    -- the agent can stand on them, and the launcher's traverse can
    continue through them.

    The all-pixels-passable check used previously was too strict; it
    rejected any cell whose bbox overlapped with a wall row even
    though the agent's centroid wouldn't land on that pixel.  That
    bug caused launcher traverses to halt one cell too early.
    """
    pg = env.get("passable_grid")
    cs = env.get("cell_system")
    if pg is None or cs is None:
        return None
    try:
        H, W = pg.shape[:2]
        r_c, c_c = cs.cell_to_pix(int(cell[0]), int(cell[1]))
        if not (0 <= r_c < H and 0 <= c_c < W):
            return False  # off-grid is barrier
        return bool(pg[r_c, c_c])
    except Exception:
        return None
