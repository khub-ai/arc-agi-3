"""Conditions — logical predicates the engine evaluates against a WorldState.

A :class:`Condition` is a structural, hashable predicate.  Two conditions
with the same :meth:`canonical_key` are considered the *same* condition for
dedup / lookup purposes.  Conditions are immutable (frozen dataclasses) so
they can be safely shared across hypotheses, goals, and plans.

Every Condition must implement:

* ``canonical_key()`` — hashable tuple identifying structural form.  Used as
  the key for dedup across the hypothesis lattice.
* ``evaluate(world)`` — returns ``True``/``False`` if the predicate's truth
  value is determinable from ``world``, or ``None`` if information is
  insufficient.  Tri-valued logic is required because many conditions
  reference entities or properties the agent has never observed.
* ``variables()`` — the set of entity IDs the condition references, used by
  the planner for subgoal expansion and by miners for co-occurrence
  tracking.

No Condition class contains any domain-specific logic.  Position tuples are
of arbitrary dimensionality (2-D grids for ARC, 3-D coords for robotics).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .types import WorldState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hashable(value: Any) -> Any:
    """Coerce a value to something hashable.

    Lists/dicts/sets become tuples with sorted keys so that logically equal
    values produce equal canonical keys regardless of original ordering.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, set) or isinstance(value, frozenset):
        return tuple(sorted(_hashable(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Condition:
    """Base class for all condition predicates.

    Subclasses are frozen dataclasses with ``eq=False`` (equality is defined
    on :meth:`canonical_key`, not dataclass field-by-field).  This lets two
    subclasses with different structural forms but equivalent canonical
    keys compare equal — useful for dedup of parameterised predicates.
    """

    def canonical_key(self) -> tuple:
        raise NotImplementedError

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        raise NotImplementedError

    def variables(self) -> FrozenSet[str]:
        return frozenset()

    def cell_target(self) -> Optional[Tuple[int, int]]:
        """Axis 3 of ``SPEC_goal_classification.md``.  Returns the
        lattice-aligned spatial target this condition refers to,
        if any, as a ``(row, col)`` integer tuple.

        Default ``None`` is the inert answer for conditions that do
        not name a navigable cell (alignment predicates, resource
        thresholds, motion-model assertions, etc.).  Cell-anchored
        subclasses (the adapter's ``AgentAtCell`` and
        ``TriggerVisitedAtLeast``) override to return the target.

        Consumers query ``c.cell_target() is not None`` rather than
        switching on ``c.canonical_key()[0]`` against a literal set
        of subclass names.
        """
        return None

    def matches(self, other: "Condition") -> bool:
        """Asymmetric subsumption check used during goal-tree
        backward-chaining: ``self.matches(other)`` returns True iff
        ``self`` being true would imply ``other`` is true (modulo
        the loose semantics of "this trigger achieves that effect").

        Default implementation: canonical-key equality.  Two
        conditions with the same canonical key are interchangeable
        for chaining purposes.

        Subclasses may override to express broader subsumption — e.g.
        a parameterless ``LevelProgressed`` overrides to subsume any
        parameterised ``LevelAdvanced(N)``, so a CausalClaim whose
        effect is ``LevelProgressed`` can chain into a goal whose
        leaf is ``LevelAdvanced(N)`` for any N.  This is what enables
        cross-level / cross-game transfer of structural causal claims.
        """
        if not isinstance(other, Condition):
            return False
        return self.canonical_key() == other.canonical_key()

    # Hash / equality delegate to canonical_key so the same predicate used in
    # different places deduplicates correctly.
    def __hash__(self) -> int:
        return hash(self.canonical_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Condition):
            return NotImplemented
        return self.canonical_key() == other.canonical_key()


# ---------------------------------------------------------------------------
# Leaf conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class AlwaysTrue(Condition):
    """Tautological condition — always satisfied.

    Useful as a placeholder for Rule.condition when a rule applies
    unconditionally, or as the trigger of a CausalClaim whose effect is
    believed to hold throughout an episode.
    """

    def canonical_key(self) -> tuple:
        return ("AlwaysTrue",)

    def evaluate(self, world: "WorldState") -> bool:
        return True


@dataclass(frozen=True, eq=False)
class AtPosition(Condition):
    """Predicate: a given entity is at a given position.

    ``pos`` is a tuple of arbitrary dimensionality — 2-D for grid domains,
    3-D (or more with orientation) for robotics.  Convention: the first
    ``entity_id`` defaults to ``"agent"``, the canonical name used by the
    engine for the acting subject; adapters may choose to override it.

    ``tolerance`` is a Chebyshev radius: the predicate is satisfied when
    every dimension of the observed position is within ``tolerance`` of
    the target ``pos``.  Default ``0.0`` preserves exact-match semantics
    used by miners that record AGENT transitions at concrete lattice
    points.  Goal-synthesis sites (role-to-goal, entity-pos bridge) set
    a positive tolerance derived from the target entity's footprint or
    the agent's motor step so that oracle-derived integer-pixel targets
    are reachable on a 5-pixel motion lattice.  Robotics analogue: an
    end-effector-pose goal always carries a tolerance because reaching
    an exact pose is meaningless given motor resolution and sensor
    noise; the same field carries the reach radius.
    """

    pos: Tuple[Any, ...]
    entity_id: str = "agent"
    tolerance: float = 0.0

    def canonical_key(self) -> tuple:
        # Tolerance participates in canonicalisation so that the same
        # target position with two different tolerances remains two
        # distinct hypotheses (they make different empirical
        # predictions).
        return ("AtPosition", self.entity_id, tuple(self.pos), float(self.tolerance))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        if self.entity_id == "agent":
            observed = world.agent.get("position")
        else:
            ent = world.entities.get(self.entity_id)
            observed = ent.properties.get("position") if ent is not None else None
        if observed is None:
            return None
        if self.tolerance <= 0.0:
            return tuple(observed) == tuple(self.pos)
        try:
            return all(
                abs(float(o) - float(p)) <= self.tolerance
                for o, p in zip(observed, self.pos)
            )
        except (TypeError, ValueError):
            return None

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


@dataclass(frozen=True, eq=False)
class InsideBBox(Condition):
    """Predicate: ``probe_id`` sits inside the bounding box of ``entity_id``.

    Where :class:`AtPosition` encodes "reach the neighbourhood of a fixed
    point" with a symmetric Chebyshev tolerance, ``InsideBBox`` encodes
    the weaker, asymmetric "enter this object's footprint".  For a long
    thin entity an ``AtPosition`` tolerance that covered the whole
    footprint would also cover cells far outside it on the short axis;
    ``InsideBBox`` captures the true shape of the hit-zone.

    Semantics
    ---------
    The bounding box lives on the target entity — ``world.entities[
    entity_id].properties["bbox"]`` — and is expected in the engine's
    canonical form ``(r_min, c_min, r_max, c_max)`` with inclusive
    integer pixel indices.  The probe's position is read the same way
    :class:`AtPosition` reads it: from ``world.agent["position"]`` when
    ``probe_id == "agent"``, else from ``world.entities[probe_id].
    properties["position"]``.  The predicate is ``True`` iff the probe
    position lies within the inclusive rectangle; ``None`` when either
    the bbox or the probe position is unavailable (tri-valued logic,
    matching :class:`AtPosition`).

    Why this exists
    ---------------
    GAP 24 navigated via ``AtPosition(centroid, tol)`` with ``tol``
    derived from the motion step size, and the interact goal flipped
    to ACHIEVED when the agent was within that tolerance of the
    centroid — often *outside* the entity's actual bbox.  GAP 24a's
    live-probe diagnosis of ls20 L1 showed the agent stopping 3 pixels
    past the cross-icon's right edge and declaring victory.  Swapping
    in ``InsideBBox(uid)`` makes "arrival" mean "agent is actually on
    the thing," which is what interact needs to hand off to local
    action-probing.

    Planner compatibility
    ---------------------
    The planner's BFS tests termination via ``_condition_holds`` which
    swaps the hypothetical agent dict into ``world.agent`` before
    calling ``evaluate`` — so this condition plans natively, no
    ``_apply_transition`` branch or new operator required.  Positions
    reachable via committed motion-model deltas that fall inside the
    target bbox terminate the search.

    Robotics analogue
    -----------------
    "End-effector is inside the graspable volume of the target
    object," where the volume is a per-object AABB rather than a
    symmetric reach tolerance.  Same predicate semantics; 3-D bbox
    substituted for 2-D pixel bbox.
    """

    entity_id: str             # target whose bbox defines the zone
    probe_id:  str = "agent"   # entity whose position is tested

    def canonical_key(self) -> tuple:
        return ("InsideBBox", self.entity_id, self.probe_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        # Probe position
        if self.probe_id == "agent":
            probe_pos = world.agent.get("position")
        else:
            probe_ent = world.entities.get(self.probe_id)
            probe_pos = probe_ent.properties.get("position") if probe_ent is not None else None
        if probe_pos is None:
            return None

        # Target bbox
        target = world.entities.get(self.entity_id)
        if target is None:
            return None
        bbox = target.properties.get("bbox")
        if bbox is None:
            return None

        try:
            # Engine convention: bbox = (r_min, c_min, r_max, c_max),
            # inclusive.  We coerce to float so stored-as-int bboxes
            # compare correctly with stored-as-float positions.
            r0, c0, r1, c1 = (float(x) for x in bbox)
            r, c = float(probe_pos[0]), float(probe_pos[1])
        except (TypeError, ValueError, IndexError):
            # Malformed bbox or position — insufficient info rather
            # than False; tri-valued logic lets the goal remain open
            # until perception produces well-formed data.
            return None

        return (r0 <= r <= r1) and (c0 <= c <= c1)

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id, self.probe_id])


@dataclass(frozen=True, eq=False)
class EntityInState(Condition):
    """Predicate: ``entity.property == value``.

    ``value`` may be any hashable-coercible value (see :func:`_hashable`).
    """

    entity_id: str
    property:  str
    value:     Any

    def canonical_key(self) -> tuple:
        return ("EntityInState", self.entity_id, self.property, _hashable(self.value))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        ent = world.entities.get(self.entity_id)
        if ent is None:
            return None
        if self.property not in ent.properties:
            return None
        return ent.properties[self.property] == self.value

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


@dataclass(frozen=True, eq=False)
class ResourceAbove(Condition):
    """Predicate: a resource's current value is strictly above a threshold."""

    resource_id: str
    threshold:   float

    def canonical_key(self) -> tuple:
        return ("ResourceAbove", self.resource_id, float(self.threshold))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        val = world.agent.get("resources", {}).get(self.resource_id)
        if val is None:
            return None
        return float(val) > self.threshold


@dataclass(frozen=True, eq=False)
class ResourceBelow(Condition):
    """Predicate: a resource's current value is strictly below a threshold."""

    resource_id: str
    threshold:   float

    def canonical_key(self) -> tuple:
        return ("ResourceBelow", self.resource_id, float(self.threshold))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        val = world.agent.get("resources", {}).get(self.resource_id)
        if val is None:
            return None
        return float(val) < self.threshold


@dataclass(frozen=True, eq=False)
class EntityProbed(Condition):
    """Curiosity-goal condition: the entity has been observed / interacted
    with enough to raise its claim-coverage above ``curiosity_threshold``.

    The ``coverage`` attribute is the required coverage; actual coverage
    is computed by the explorer at evaluation time.  Canonical key does
    NOT include coverage — two EntityProbed conditions for the same
    entity deduplicate even if they were created at different thresholds.
    """

    entity_id: str
    coverage:  float = 0.5

    def canonical_key(self) -> tuple:
        return ("EntityProbed", self.entity_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        # The explorer (not this class) implements coverage computation;
        # at Condition-level we can only answer None until the explorer
        # writes a coverage field onto the EntityModel.
        ent = world.entities.get(self.entity_id)
        if ent is None:
            return None
        cov = ent.properties.get("_claim_coverage")
        if cov is None:
            return None
        return float(cov) >= self.coverage

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


@dataclass(frozen=True, eq=False)
class ActionJustTaken(Condition):
    """Predicate: the *most recent* action executed was ``action_id``.

    Distinct from :class:`ActionTried`, which is cumulative (true forever
    once an action has been tried at least once).  ``ActionJustTaken``
    is true for exactly one step — the step immediately after execution —
    making it the natural trigger for :class:`CausalClaim`\\s that
    describe what an action *does*.

    Reads from ``world.agent['_last_action']``, which the episode
    runner sets after every :meth:`Adapter.execute` call.

    Domain-agnostic: an ARC game's ACTION1 and a robot's ``MOVE_ARM_UP``
    are both string-identified actions whose effect-claims need a
    single-step-sticky trigger.
    """

    action_id: str

    def canonical_key(self) -> tuple:
        return ("ActionJustTaken", self.action_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        last = world.agent.get("_last_action")
        if last is None:
            return None
        return str(last) == self.action_id


@dataclass(frozen=True, eq=False)
class FrameChangedPattern(Condition):
    """Predicate: the most recent frame-diff matches a structural signature.

    The signature is intentionally coarse — ``cells_changed`` plus an
    optional inclusive bounding box — because the purpose of this
    condition is to *characterise* an action's effect in a way that's
    robust to cell-level noise.  Two probes of the same action that
    produce deltas with identical ``(cells_changed, bbox)`` signatures
    support the same :class:`CausalClaim`; two probes that produce
    different signatures are treated as evidence for two competing
    claims and the store will reconcile.

    Reads from ``world.last_frame_delta``; returns ``None`` when no
    delta is available (first step, non-grid frames, etc.).

    Parameters
    ----------
    cells_changed
        Number of cells expected to differ.  ``0`` is meaningful — it
        encodes the claim "this action produces no observable frame
        change", which is valuable for pruning the effective action
        space.
    bbox
        Optional inclusive bounding box ``(r_min, c_min, r_max, c_max)``.
        When ``None`` the condition ignores location and matches any
        delta with the given ``cells_changed`` count.  Bounding boxes
        are what let probes distinguish e.g. an "agent moves north"
        from "agent moves south" when both flip the same number of
        cells.
    """

    cells_changed: int
    bbox:          Optional[Tuple[int, int, int, int]] = None

    def canonical_key(self) -> tuple:
        return ("FrameChangedPattern", int(self.cells_changed), self.bbox)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        delta = getattr(world, "last_frame_delta", None)
        if delta is None:
            return None
        if int(getattr(delta, "cells_changed", -1)) != int(self.cells_changed):
            return False
        if self.bbox is None:
            return True
        return tuple(getattr(delta, "bbox", ()) or ()) == tuple(self.bbox)


@dataclass(frozen=True, eq=False)
class RegionMotion(Condition):
    """Predicate: the most recent frame-delta contains a region whose
    cell-level before/after pattern describes a *translation* of a
    coloured blob in a particular direction.

    This condition is the first rung of the generalisation ladder above
    :class:`FrameChangedPattern`.  Where ``FrameChangedPattern`` records
    an absolute bounding box — useful for exact reproduction but
    brittle across episodes and starting positions — ``RegionMotion``
    keeps only:

    * ``colour``            — the object colour that moved
                              (the value appearing in cells that were
                              previously ``background`` and are now
                              ``colour``).
    * ``background``        — the colour surrounding the object
                              (the value appearing in cells that were
                              previously ``colour`` and are now
                              ``background``).
    * ``dr_sign``, ``dc_sign`` — the **sign** of the row/column motion
                              vector (``-1``, ``0``, or ``+1``).  Sign
                              only — no magnitudes, no absolute
                              positions — is what makes the canonical
                              key transfer across episodes and across
                              starting positions of the same mechanic.

    Rationale.  An absolute-bbox claim "ACTION1 changes cells in
    bbox=(17,23,18,23)" tells us nothing when the agent starts at a
    different position next episode.  A sign-only claim "ACTION1 moves
    colour-9 objects upward through colour-0 background" transfers
    directly.  Keeping the full ``(colour, background, dr_sign,
    dc_sign)`` tuple in the canonical key lets the hypothesis store
    keep directional claims for distinct colour pairs separate (so
    "agent moves up" and "block moves up" don't collide), while still
    deduplicating claims that describe the same mechanic observed at
    different starting positions.

    ``evaluate`` returns ``True`` iff the most recent frame-delta
    contains at least one region whose extracted motion matches this
    signature exactly; ``False`` if a delta is present but no region
    matches; ``None`` if no delta is available (first step).
    """

    colour:     Any
    background: Any
    dr_sign:    int
    dc_sign:    int

    def canonical_key(self) -> tuple:
        return (
            "RegionMotion",
            _hashable(self.colour),
            _hashable(self.background),
            int(self.dr_sign),
            int(self.dc_sign),
        )

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        delta = getattr(world, "last_frame_delta", None)
        if delta is None:
            return None
        regions = getattr(delta, "regions", ()) or ()
        if not regions:
            # A delta is present but empty — no region can match.
            return False
        # Import lazily to avoid cycles (frame_diff does not depend on
        # conditions; conditions optionally read its extractor).
        from .frame_diff import extract_region_motion
        for region in regions:
            motion = extract_region_motion(region, delta)
            if motion is None:
                continue
            c, bg, dr, dc = motion
            if (c == self.colour
                    and bg == self.background
                    and int(dr) == int(self.dr_sign)
                    and int(dc) == int(self.dc_sign)):
                return True
        return False


@dataclass(frozen=True, eq=False)
class EntitiesVisuallyMatch(Condition):
    """Oracle-gated predicate: two entity regions share the same visual
    orientation / internal pattern.

    Rather than binding to fragile entity IDs (which change when a glyph
    rotates and the perception layer assigns a new ID to the new shape),
    the predicate reads from a stable ``match_slot`` written into
    ``world.agent`` by :class:`cognitive_os.oracle.VisualOrientationTrigger`
    after each :data:`~cognitive_os.QuestionType.COMPARE_VISUAL_STATES`
    Observer query.

    The slot key is ``f"_vm:{match_slot}"``; the trigger picks
    ``match_slot`` to be stable across all trajectories on the same
    level (e.g. ``"lvl0_col9"`` for a level-0 maroon-glyph comparison).

    ``evaluate`` returns:

    * ``True``  — slot is ``True``: orientations currently match.
    * ``False`` — slot is ``False``: orientations currently mismatch.
    * ``None``  — slot absent: Oracle has not yet been queried (trigger
                  has not fired, or query returned zero confidence).
                  The planner treats ``None`` as "information insufficient"
                  and cannot plan toward this condition via BFS — the
                  oracle trigger owns first contact; the explorer drives
                  action-based convergence once a mismatch is known.

    Dual-domain.  Robotics analogues:

    * Gripper orientation matches part orientation (precondition for
      peg-in-hole insertion).
    * Camera image of a bin label matches the part's colour-code
      (precondition for placement into the correct bin).
    * Screw-driver orientation matches screw-head orientation
      (precondition for a tightening motion).

    In all cases the same oracle-gated structure applies: the robot's
    vision sub-system writes the match result into a named slot; the
    condition reads the slot; the action that changes orientation
    (rotate gripper / reposition arm) advances toward match=True.
    """

    match_slot: str

    def canonical_key(self) -> tuple:
        return ("EntitiesVisuallyMatch", self.match_slot)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        key = f"_vm:{self.match_slot}"
        val = world.agent.get(key)
        if val is None:
            return None
        return bool(val)

    # No entity variables: the slot is agent-level.  Miners that track
    # entity-level coverage are unaffected; this condition contributes
    # zero to entity coverage, which is correct — the oracle trigger (not
    # this condition) drives the entity-level evidence pipeline.
    def variables(self) -> FrozenSet[str]:
        return frozenset()


@dataclass(frozen=True, eq=False)
class EntitiesEquivalent(Condition):
    """Multi-dimensional state-equivalence: two entities match on every
    named dimension.

    Generalises pair-wise alignment beyond a single fixed comparison
    (e.g. orientation-only) to an arbitrary set of property
    dimensions: orientation, color, size, shape, position, … any
    string the adapter publishes as an :class:`EntityModel`
    property.  The condition closes when every dimension's value on
    the target entity equals the same dimension's value on the
    reference entity.

    Why a set of dimensions, not a fixed pair-comparison.  Different
    instances of the "make these two things look alike" problem
    require different match dimensions:

    * An L1 alignment with both glyphs already in matching colors:
      ``EntitiesEquivalent(t, r, {"orientation"})``.
    * An L3 alignment where the glyphs start in different colors:
      ``EntitiesEquivalent(t, r, {"orientation", "color"})``.
    * A bin-placement task in robotics where the part must match
      both the bin label's color and shape:
      ``EntitiesEquivalent(part, label, {"color", "shape"})``.

    The dimension set should be derived from observation rather
    than hardcoded — see SPEC discussion: given a target/reference
    pair the operator declares as "must align", scan their observed
    properties and emit ``EntitiesEquivalent`` with the set of
    dimensions that currently differ.  As triggers fire and
    properties converge, the set narrows; when empty the condition
    holds.

    Independent of pipeline.  Critically does NOT depend on
    same-palette pair miners (which by construction skip
    different-color pairs and would never produce a PairMatch claim
    for an L3-style cross-color alignment).  Reads each entity's
    properties directly from the world model and compares
    pair-wise.  This is the structural fix for the
    ``OrientationAligned`` blind-spot at L3.

    ``evaluate`` returns:

    * ``True``  — every dimension's values are equal on both
      entities.
    * ``False`` — at least one dimension's values are present on
      both entities and not equal.
    * ``None``  — at least one dimension is missing on at least one
      entity (insufficient information; planner should surface
      observation-side subgoals to acquire it).

    Cross-domain: in robotics, the same condition expresses "the
    gripper's orientation matches the part's orientation AND its
    grasp-axis matches the part's symmetry axis" without inventing
    new condition kinds per pair-of-properties.
    """

    target_id:    str
    reference_id: str
    dimensions:   FrozenSet[str]

    def canonical_key(self) -> tuple:
        # Symmetric pair: sort the ids so that the pair (a, b) and
        # (b, a) produce the same canonical key (matches the
        # convention in OrientationAligned and pair_match_miner).
        # Dimensions are sorted for stable hashing.
        a, b = sorted([str(self.target_id), str(self.reference_id)])
        dims = tuple(sorted(str(d) for d in self.dimensions))
        return ("EntitiesEquivalent", a, b, dims)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        target = world.entities.get(self.target_id)
        reference = world.entities.get(self.reference_id)
        if target is None or reference is None:
            return None
        if not self.dimensions:
            # Vacuous match — no dimensions to compare.  Caller is
            # expressing "these two are the same entity by some other
            # measure", which is structurally always-True.
            return True
        any_unknown = False
        for dim in self.dimensions:
            t_val = target.properties.get(dim)
            r_val = reference.properties.get(dim)
            if t_val is None or r_val is None:
                any_unknown = True
                continue
            if t_val != r_val:
                return False
        # All observed dimensions match.  If any were unknown, the
        # honest answer is None (could still flip false once observed);
        # if every dimension was observable and equal, return True.
        return None if any_unknown else True

    def variables(self) -> FrozenSet[str]:
        return frozenset([self.target_id, self.reference_id])


@dataclass(frozen=True, eq=False)
class ActionTried(Condition):
    """Curiosity-goal condition: an action has been executed at least once.

    Used by the explorer when an action's transition dynamics are
    completely unknown — a single attempt yields a first TransitionClaim,
    after which ordinary evidence accumulation takes over.
    """

    action_id: str

    def canonical_key(self) -> tuple:
        return ("ActionTried", self.action_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        tried = world.agent.get("_actions_tried")
        if tried is None:
            return None
        return self.action_id in tried


@dataclass(frozen=True, eq=False)
class AgentAtEntityClass(Condition):
    """Predicate: agent's position coincides with at least one entity
    in ``world.entities`` whose stable visual fingerprint
    (``properties["entity_class"]``) equals ``entity_class``.

    This is the engine-level trigger for the "pick up an entity of
    this class" mechanic.  Unlike :class:`AtPosition` it does not
    name a cell — the cell is computed at evaluation time from
    wherever instances of the entity-class currently sit.  So a
    CausalClaim with trigger ``AgentAtEntityClass`` and effect
    ``ResourceRestored("budget")`` survives entity motion and per-
    life re-spawning of pickups: the same claim fires wherever a
    matching-class instance currently is, not where one happened to
    be when the claim was minted.

    Robotics analogue: "end-effector is grasping a battery-class
    object" — the class is identified by stable visual fingerprint;
    the world coords vary per task.

    Returns:

    * ``True``  — at least one entity of the named class is at the
                  agent's current position.
    * ``False`` — at least one entity of the class is present
                  somewhere but none is at the agent's position
                  (so the trigger is observable-but-not-firing).
    * ``None``  — agent position or entity store has not yet been
                  populated (insufficient info; tri-valued).

    Convention: agent position is read from ``world.agent["position"]``;
    each entity's position is read from
    ``world.entities[*].properties["position"]``.  Both must be in
    matching coordinate units — the adapter is responsible for
    ensuring this.  ``"entity_class"`` is published per
    :class:`~cognitive_os.types.EntityModel` by the adapter as a
    stable visual identifier (bitmap_id, shape_id, or similar).
    """

    entity_class: str

    def canonical_key(self) -> tuple:
        return ("AgentAtEntityClass", str(self.entity_class))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        agent_pos = world.agent.get("position")
        if agent_pos is None:
            return None
        agent_t = tuple(agent_pos)
        any_seen = False
        for ent in world.entities.values():
            cls_id = ent.properties.get("entity_class")
            if cls_id is None or str(cls_id) != str(self.entity_class):
                continue
            any_seen = True
            ent_pos = ent.properties.get("position")
            if ent_pos is None:
                continue
            if tuple(ent_pos) == agent_t:
                return True
        if not any_seen:
            return None
        return False

    def variables(self) -> FrozenSet[str]:
        return frozenset()


@dataclass(frozen=True, eq=False)
class ResourceRestored(Condition):
    """Predicate: the named resource just experienced a positive
    restoration event (delta > 0) on the most recent step.

    Effect-side of refill-mechanic CausalClaims.  Reads from
    ``world.agent["_last_resource_delta"][resource_id]``, which the
    adapter writes after each ``Adapter.execute`` call: a mapping
    from ``ResourceKey`` to the observed change in that resource
    this step.  ``ResourceRestored`` flips True for exactly one
    step — the step where the restoration was observed.

    Pair with :class:`AgentAtEntityClass` (or any other trigger) in
    a :class:`~cognitive_os.claims.CausalClaim` to encode learned
    restoration mechanics ("picking up an entity of class X
    restores budget").

    Robotics analogue: "battery just reported a positive charge
    delta" — same single-step-sticky semantics as ARC budget refill.
    """

    resource_id: str

    def canonical_key(self) -> tuple:
        return ("ResourceRestored", str(self.resource_id))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        deltas = world.agent.get("_last_resource_delta")
        if deltas is None:
            return None
        d = deltas.get(self.resource_id)
        if d is None:
            return False
        try:
            return float(d) > 0.0
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True, eq=False)
class AgentAtCellRelativeToEntity(Condition):
    """Predicate: the agent occupies the cell that is offset ``(dr, dc)``
    from the position of at least one entity in ``world.entities``
    whose stable visual fingerprint (``properties["entity_class"]``)
    equals ``entity_class``.

    This is the engine-level trigger for **relational** mechanics --
    "step onto the cell east of any bar," "land on a cell adjacent
    to a docking port."  Unlike :class:`AgentAtCell` (absolute) and
    :class:`AgentAtEntityClass` (coincident with a matching entity),
    this condition lets a single hypothesis fire at every cell that
    bears the same relation to a same-class entity -- the basis for
    one-observation generalization across structurally identical
    positions.

    Robotics analogue: "end-effector is at the graspable face of any
    instance of part-class X" -- the same affordance applies at
    every part instance regardless of where it currently sits.

    Returns:

    * ``True``  -- at least one entity of the named class has the
                   agent at its ``(dr, dc)`` neighbour.
    * ``False`` -- at least one entity of the class is present but
                   the agent is not at its ``(dr, dc)`` neighbour.
    * ``None``  -- agent position or entity store has not yet been
                   populated, or no instance of the class has a
                   resolved position (tri-valued).

    Convention: agent position is read from
    ``world.agent["position"]``; each entity's position is read
    from ``world.entities[*].properties["position"]``.  Both must
    be in the same coordinate units (the adapter is responsible).
    ``entity_class`` is the same stable visual identifier published
    by the adapter per :class:`~cognitive_os.types.EntityModel`
    (bitmap_id by default; coarser tier when desired).
    """

    entity_class: str
    dr:           int
    dc:           int

    def canonical_key(self) -> tuple:
        return ("AgentAtCellRelativeToEntity",
                str(self.entity_class),
                int(self.dr),
                int(self.dc))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        agent_pos = world.agent.get("position")
        if agent_pos is None:
            return None
        try:
            agent_t = (int(agent_pos[0]), int(agent_pos[1]))
        except (TypeError, ValueError, IndexError):
            return None
        any_seen = False
        any_resolved = False
        for ent in world.entities.values():
            cls_id = ent.properties.get("entity_class")
            if cls_id is None or str(cls_id) != str(self.entity_class):
                continue
            any_seen = True
            ent_pos = ent.properties.get("position")
            if ent_pos is None:
                continue
            try:
                ept = (int(ent_pos[0]), int(ent_pos[1]))
            except (TypeError, ValueError, IndexError):
                continue
            any_resolved = True
            if (ept[0] + int(self.dr), ept[1] + int(self.dc)) == agent_t:
                return True
        if not any_seen or not any_resolved:
            return None
        return False

    def variables(self) -> FrozenSet[str]:
        return frozenset()


@dataclass(frozen=True, eq=False)
class AgentTeleportedByOffset(Condition):
    """Predicate: the agent's most recent motion produced the
    cell-space delta ``(dr, dc)``.

    Effect-side of relational portal / bouncer / launcher mechanics.
    Reads from ``world.agent["_last_motion_delta"]``, which the
    adapter writes after each ``Adapter.execute`` call as a
    ``(d_row, d_col)`` tuple in **cell** units.  Returns True for
    exactly one step -- the step immediately after the motion was
    observed.

    Pair with :class:`AgentAtCellRelativeToEntity` (or any other
    trigger) in a :class:`~cognitive_os.claims.CausalClaim` to
    encode learned relational portal mechanics ("stepping onto the
    cell east of a bar propels the agent five cells east").

    Robotics analogue: "the end-effector translated by Δpose after
    the last motor command" -- single-step sticky, same convention.
    """

    dr: int
    dc: int

    def canonical_key(self) -> tuple:
        return ("AgentTeleportedByOffset", int(self.dr), int(self.dc))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        delta = world.agent.get("_last_motion_delta")
        if delta is None:
            return None
        try:
            return (int(delta[0]), int(delta[1])) == (
                int(self.dr), int(self.dc)
            )
        except (TypeError, ValueError, IndexError):
            return False


@dataclass(frozen=True, eq=False)
class MotionModelCommitted(Condition):
    """Probe-goal termination condition: a :class:`MotionModelClaim`
    exists for ``action_id`` whose credence has cleared the
    planner's :attr:`PlannerConfig.min_credence` floor.

    This replaces :class:`ActionTried` as the closing condition for
    action-probe goals under continuous commitment
    (``SPEC_continuous_commitment.md``).  ``ActionTried`` flipped to
    True on the very first press — which in practice meant the
    probe closed before any motor stride had actually been learned
    (the press could be wall-blocked, context-dependent, or a
    no-op).  ``MotionModelCommitted`` instead requires that the
    miner has accumulated enough evidence for the planner to treat
    the motion model as usable.

    Evaluation
    ----------
    Returns ``True`` iff the store contains at least one
    :class:`MotionModelClaim` for ``action_id`` whose
    ``credence.point`` is ≥ the planner's ``min_credence`` floor
    (default ``0.5``).  Returns ``False`` otherwise — never ``None``
    — because the absence of a sufficiently-credible claim is a
    determinate, actionable state (keep probing) rather than an
    information-insufficient state.

    Canonical key omits the threshold: the condition is about
    *this* action's motion model, and all probe goals share a
    single planner-wide floor via config.

    Robotics analogue: "motor primitive X has a credible kinematic
    effect on the end-effector" — the precondition for stopping
    motor babbling on that primitive.
    """

    action_id: str

    def canonical_key(self) -> tuple:
        return ("MotionModelCommitted", self.action_id)

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        # Lazy imports avoid circular dependencies — conditions is
        # low in the import stack.
        from .claims import MotionModelClaim
        from .config import PlannerConfig
        cfg_planner = (getattr(world.config, "planner", None)
                       if world.config is not None else None)
        floor = float(getattr(cfg_planner, "min_credence", None)
                      or PlannerConfig().min_credence)
        aid = self.action_id
        for h in world.hypotheses.values():
            claim = h.claim
            if not isinstance(claim, MotionModelClaim):
                continue
            if str(claim.action_id) != aid:
                continue
            if float(getattr(h.credence, "point", 0.0)) >= floor:
                return True
        return False


# ---------------------------------------------------------------------------
# Composite conditions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Conjunction(Condition):
    """Logical AND over a set of conditions.

    Child conditions are stored as a tuple so the dataclass stays hashable;
    canonicalisation sorts child keys so that the order of construction
    does not affect identity.
    """

    conditions: Tuple[Condition, ...]

    def canonical_key(self) -> tuple:
        keys = sorted(c.canonical_key() for c in self.conditions)
        return ("Conjunction", tuple(keys))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        results = [c.evaluate(world) for c in self.conditions]
        if any(r is False for r in results):
            return False
        if any(r is None for r in results):
            return None
        return True

    def variables(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for c in self.conditions:
            out = out | c.variables()
        return out


@dataclass(frozen=True, eq=False)
class Disjunction(Condition):
    """Logical OR over a set of conditions."""

    conditions: Tuple[Condition, ...]

    def canonical_key(self) -> tuple:
        keys = sorted(c.canonical_key() for c in self.conditions)
        return ("Disjunction", tuple(keys))

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        results = [c.evaluate(world) for c in self.conditions]
        if any(r is True for r in results):
            return True
        if any(r is None for r in results):
            return None
        return False

    def variables(self) -> FrozenSet[str]:
        out: FrozenSet[str] = frozenset()
        for c in self.conditions:
            out = out | c.variables()
        return out


@dataclass(frozen=True, eq=False)
class Negation(Condition):
    """Logical NOT of a condition.  Preserves tri-valued semantics."""

    condition: Condition

    def canonical_key(self) -> tuple:
        return ("Negation", self.condition.canonical_key())

    def evaluate(self, world: "WorldState") -> Optional[bool]:
        v = self.condition.evaluate(world)
        if v is None:
            return None
        return not v

    def variables(self) -> FrozenSet[str]:
        return self.condition.variables()
