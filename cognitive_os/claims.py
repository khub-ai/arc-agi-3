"""Claims — the contents of a hypothesis.

A :class:`Hypothesis` wraps a :class:`Claim` with credence, scope, evidence,
and lattice links (parent/children).  The Claim itself is the *structural*
statement being believed.

Two orthogonal keys are exposed on every Claim:

* ``canonical_key()`` — structural identity.  Two claims with equal
  canonical keys refer to the *same phenomenon*; they compete for the same
  evidence even if their specific parameters differ.  Used by the
  HypothesisStore to link competitors.

* ``full_key()`` — exact identity.  Two claims with equal full keys are
  literally the same claim; proposing one when the other exists merges
  evidence rather than creating a duplicate.

The distinction is what lets the system *learn parameters*.  Three
``CausalClaim``\\s with the same trigger/effect but ``min_occurrences`` of
2, 3, and 4 all share the same canonical key; they are competitors; evidence
will push credence onto one and drop the others below the abandon
threshold.  No separate parameter-search mechanism is needed.

Standing directive: no Claim type encodes any domain-specific mechanic.
Every Claim form shown here is one a robot, an ARC solver, or any other
symbolic agent would equally benefit from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional, Tuple

from .conditions import Condition, _hashable


# ---------------------------------------------------------------------------
# Supporting enums / value types
# ---------------------------------------------------------------------------


class RelationType(Enum):
    """Binary relations between two entities.

    ``APPEARS_*`` relations are normally resolved by an Observer (visual
    oracle).  ``CO_OCCURS_WITH`` and ``SPATIALLY_NEAR`` are resolved
    symbolically from event history and entity positions.
    ``STRUCTURALLY_LINKED`` is resolved by accumulated CausalClaims between
    the two entities.  ``SAME_CLASS`` is a taxonomic grouping that may be
    resolved by either route.
    """

    APPEARS_SIMILAR     = "appears_similar"
    APPEARS_IDENTICAL   = "appears_identical"
    APPEARS_DISTINCT    = "appears_distinct"
    CO_OCCURS_WITH      = "co_occurs_with"
    STRUCTURALLY_LINKED = "structurally_linked"
    SAME_CLASS          = "same_class"
    SPATIALLY_NEAR      = "spatially_near"


class MappingKind(Enum):
    """What basis a StructureMappingClaim rests on."""

    VISUAL     = "visual"       # appearance-driven; resolved via Observer
    STRUCTURAL = "structural"   # relation-topology driven
    FUNCTIONAL = "functional"   # role-based correspondence


@dataclass(frozen=True)
class RelationPattern:
    """A relation schema that might hold in both source and target groups
    of a :class:`StructureMappingClaim`.

    ``relation`` names the relation (e.g. "adjacent", "larger_than",
    "triggers").  ``arity`` is typically 2 but may be higher.  The pattern
    does not instantiate the roles — the mapping itself supplies the
    concrete entities.  For example, ``RelationPattern("adjacent", 2)`` in
    a source group ``{A, B, C}`` with mapping ``{A→X, B→Y, C→Z}`` is
    preserved iff ``adjacent(X,Y)`` holds whenever ``adjacent(A,B)`` does,
    and likewise for every other pair.
    """

    relation: str
    arity:    int = 2


@dataclass(frozen=True)
class Asymmetry:
    """An element in one group with no counterpart in the other, or a
    relation that holds on one side but fails on the other under the
    current mapping.

    Asymmetries are predictive: a source entity with no target counterpart
    may indicate the target group is incomplete, suggesting an exploration
    subgoal to look for the missing element.
    """

    side: str                              # "source" or "target"
    entity_id: Optional[str] = None        # unmapped entity, if element-kind
    relation:  Optional[str] = None        # relation that fails to project, if relation-kind
    note:      str = ""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Claim:
    """Base class for all claim types.

    Subclasses MUST override :meth:`canonical_key` and :meth:`full_key`.
    They SHOULD override :meth:`referenced_entities` if they name any
    entities.  Equality and hashing derive from full_key so that two
    identical claims compare equal and can be deduplicated by
    dictionary/set membership.
    """

    def canonical_key(self) -> tuple:
        raise NotImplementedError

    def full_key(self) -> tuple:
        raise NotImplementedError

    def referenced_entities(self) -> FrozenSet[str]:
        return frozenset()

    def __hash__(self) -> int:
        return hash(self.full_key())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Claim):
            return NotImplemented
        return self.full_key() == other.full_key()


# ---------------------------------------------------------------------------
# 1. PropertyClaim — an entity has a property with a value
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class PropertyClaim(Claim):
    """Claim: ``entity.property == value``.

    Canonical key identifies the (entity, property) pair; full key adds
    the value.  So claims asserting different values of the same property
    on the same entity compete.
    """

    entity_id: str
    property:  str
    value:     Any

    def canonical_key(self) -> tuple:
        return ("PropertyClaim", self.entity_id, self.property)

    def full_key(self) -> tuple:
        return ("PropertyClaim", self.entity_id, self.property, _hashable(self.value))

    def referenced_entities(self) -> FrozenSet[str]:
        return frozenset([self.entity_id])


# ---------------------------------------------------------------------------
# 2. CausalClaim — trigger condition causes effect condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CausalClaim(Claim):
    """Claim: when ``trigger`` holds (``min_occurrences`` times, optionally
    with a ``delay``), ``effect`` becomes true.

    ``min_occurrences`` and ``delay`` are learnable parameters — multiple
    CausalClaims with the same trigger/effect but different parameter
    values are competitors and the store will drive one to commitment.

    ``delay`` is measured in steps; a zero delay means the effect should
    be observable on the same step as the trigger firing.
    """

    trigger:         Condition
    effect:          Condition
    min_occurrences: int = 1
    delay:           int = 0

    def canonical_key(self) -> tuple:
        return ("CausalClaim",
                self.trigger.canonical_key(),
                self.effect.canonical_key())

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.min_occurrences, self.delay)

    def referenced_entities(self) -> FrozenSet[str]:
        return self.trigger.variables() | self.effect.variables()


# ---------------------------------------------------------------------------
# 3. TransitionClaim — action × pre-condition → post-condition
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class TransitionClaim(Claim):
    """Claim: executing ``action`` when ``pre`` holds yields ``post``.

    The backbone of the planner's forward model.  Canonical key is
    (action, pre) so multiple TransitionClaims with the same (action, pre)
    but different post conditions are competitors — the store converges
    on the best predictor of action outcomes.
    """

    action: str
    pre:    Condition
    post:   Condition

    def canonical_key(self) -> tuple:
        return ("TransitionClaim", self.action, self.pre.canonical_key())

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.post.canonical_key())

    def referenced_entities(self) -> FrozenSet[str]:
        return self.pre.variables() | self.post.variables()


# ---------------------------------------------------------------------------
# 4. RelationalClaim — a binary relation between two entities
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RelationalClaim(Claim):
    """Claim: entities ``a`` and ``b`` stand in ``relation``.

    ``a`` and ``b`` are always stored in canonical order (lexicographic)
    so that the relation is undirected by default.  Use the factory
    :meth:`make` to construct — it normalises the argument order.  For
    directed relations, include direction as part of the ``relation``
    value (e.g. distinct relation types ``"triggers"`` vs ``"triggered_by"``).
    """

    a:          str
    b:          str
    relation:   RelationType
    properties: Tuple[Tuple[str, Any], ...] = ()   # immutable kv pairs

    @classmethod
    def make(cls,
             e1: str,
             e2: str,
             relation: RelationType,
             **properties: Any) -> "RelationalClaim":
        a, b = sorted([e1, e2])
        props = tuple(sorted((k, _hashable(v)) for k, v in properties.items()))
        return cls(a=a, b=b, relation=relation, properties=props)

    def canonical_key(self) -> tuple:
        return ("RelationalClaim", self.a, self.b, self.relation.value)

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.properties)

    def referenced_entities(self) -> FrozenSet[str]:
        return frozenset([self.a, self.b])


# ---------------------------------------------------------------------------
# 5. ConstraintClaim — a structural limitation with an implication
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ConstraintClaim(Claim):
    """Claim: when ``condition`` holds, ``implication`` (described
    informally as a string) follows.

    Used for structural/meta-level observations such as "when
    ``ResourceBelow('budget', 5)`` holds, the goal ``reach(target)`` is
    unreachable".  The planner uses ConstraintClaims as pruning hints,
    not as hard filters.

    ``implication`` is intentionally a free-form string because this
    class is a catch-all for constraints that don't fit the other four;
    a subsequent refinement step may replace it with a structured
    ConstraintClaim subtype.
    """

    condition:   Condition
    implication: str

    def canonical_key(self) -> tuple:
        return ("ConstraintClaim", self.condition.canonical_key())

    def full_key(self) -> tuple:
        return (*self.canonical_key(), self.implication)

    def referenced_entities(self) -> FrozenSet[str]:
        return self.condition.variables()


# ---------------------------------------------------------------------------
# 6. StructureMappingClaim — partial correspondence between two entity groups
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StructureMappingClaim(Claim):
    """Claim: entities in ``source_entities`` correspond to entities in
    ``target_entities`` according to ``mapping``, with the preserved
    relations in ``preserved_relations`` supporting the correspondence.

    Gentner-style structure mapping.  Distinct from :class:`RelationalClaim`
    because it is a *group-to-group* projection rather than a binary
    relation, and because it explicitly tracks asymmetries that drive
    prediction and exploration.

    A committed StructureMappingClaim lets the planner *transfer* a
    previously successful plan from the source group to the target group
    by substituting mapped entities.  Asymmetries generate exploration
    subgoals: "what should be in the target position corresponding to
    this unmapped source element?"

    Canonical key is (source_entities, target_entities, mapping_kind) —
    different mappings between the same two groups compete for evidence.
    """

    source_entities:      FrozenSet[str]
    target_entities:      FrozenSet[str]
    mapping:              Tuple[Tuple[str, str], ...]          # (src_id, tgt_id) pairs, sorted
    preserved_relations:  Tuple[RelationPattern, ...] = ()
    asymmetries:          Tuple[Asymmetry, ...]       = ()
    mapping_kind:         MappingKind = MappingKind.STRUCTURAL

    @classmethod
    def make(cls,
             source: FrozenSet[str],
             target: FrozenSet[str],
             mapping: Dict[str, str],
             preserved: Tuple[RelationPattern, ...] = (),
             asymmetries: Tuple[Asymmetry, ...] = (),
             kind: MappingKind = MappingKind.STRUCTURAL) -> "StructureMappingClaim":
        sorted_mapping = tuple(sorted(mapping.items()))
        return cls(
            source_entities     = frozenset(source),
            target_entities     = frozenset(target),
            mapping             = sorted_mapping,
            preserved_relations = preserved,
            asymmetries         = asymmetries,
            mapping_kind        = kind,
        )

    def canonical_key(self) -> tuple:
        return ("StructureMappingClaim",
                tuple(sorted(self.source_entities)),
                tuple(sorted(self.target_entities)),
                self.mapping_kind.value)

    def full_key(self) -> tuple:
        return (*self.canonical_key(),
                self.mapping,
                tuple((r.relation, r.arity) for r in self.preserved_relations))

    def referenced_entities(self) -> FrozenSet[str]:
        return self.source_entities | self.target_entities


# ---------------------------------------------------------------------------
# 7. StrategyClaim — meta-claim about OR-node branch success
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class StrategyClaim(Claim):
    """Claim: in contexts matching ``context_pattern``, the strategy named
    ``strategy_type`` (with optional ``parameter`` value) succeeds at
    approximately the stored rate.

    This is the engine's mechanism for credence-weighted *policy* claims —
    beliefs about what the agent should *do* in a situation, parallel to
    world claims (PropertyClaim, CausalClaim, …) which are beliefs about
    what *is* in the world.  See
    [`SPEC_strategy_hypotheses.md`](../SPEC_strategy_hypotheses.md) for
    the broader taxonomy (tool-invocation, goal-priority, sequence,
    heuristic-fitness, refutation, resource-allocation, mode-transition);
    the original use of this class — OR-branch selection in the planner —
    is one sub-type within it.

    Fields:

    * ``context_pattern`` — the Condition describing when this strategy
      applies.  E.g. ``ResourceAbove('budget', 20)`` for an OR-branch
      heuristic, or ``WallCredenceOnPathAbove(τ)`` for a tool-invocation
      claim.
    * ``strategy_type`` — string identifier for the strategy.  E.g.
      ``"branch:direct"``, ``"tool_choice.bfs"``, ``"goal_priority.curiosity_default"``.
    * ``parameter`` — optional learnable value associated with the
      strategy.  ``None`` when the strategy is purely categorical
      (e.g. branch selection); a numeric / dict value when the strategy
      embeds a tunable (e.g. the τ threshold's *value*, the priority's
      *number*, the retry-count's *integer*).  Two claims with the same
      ``strategy_type`` but different ``parameter`` are different
      claims competing for credence — exactly what the hypothesis
      store's ``Credence.competing`` field models.
    * ``success_rate`` / ``n_trials`` — learned from invocation outcomes.
      Note these duplicate information now also tracked on the
      enclosing :class:`Hypothesis` (supporting_steps / contradicting_steps
      counts feed the same Bayesian update); the duplication is
      retained for backward compatibility with the planner's existing
      bias logic in :func:`cognitive_os.planner._strategy_claim_bias`.

    Canonical key is ``(context_pattern, strategy_type, parameter)`` —
    parameter participates in identity because differently-parameterized
    strategies make different predictions and accumulate independent
    evidence.

    Lifecycle metadata (origin, scope, parent/child for specializations,
    creation/update timestamps, status-via-credence-threshold) lives on
    the enclosing :class:`Hypothesis` per the standard hypothesis store
    contract; this class carries only the claim's own structural shape.
    """

    context_pattern: Condition
    strategy_type:   str
    parameter:       Any   = None
    success_rate:    float = 0.5
    n_trials:        int   = 0

    def canonical_key(self) -> tuple:
        # Parameter participates in identity: two strategies with the same
        # name but different parameter values are different claims.
        # ``repr`` is used to make the key hashable regardless of the
        # parameter's concrete type (number, tuple, frozen dict-like, …).
        return ("StrategyClaim",
                self.context_pattern.canonical_key(),
                self.strategy_type,
                repr(self.parameter))

    def full_key(self) -> tuple:
        return (*self.canonical_key(),
                round(self.success_rate, 3),
                self.n_trials)

    def referenced_entities(self) -> FrozenSet[str]:
        return self.context_pattern.variables()


# ---------------------------------------------------------------------------
# 8. ControlledActorClaim — which (colour, background) pair IS the agent
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ControlledActorClaim(Claim):
    """Claim: the sprite whose dominant colour is ``colour`` on
    background ``background`` is the **agent-controlled actor** of the
    current domain.

    Epistemically this is a *second-order* belief synthesised from
    action-effect evidence: when N distinct actions all commit a
    :class:`CausalClaim(ActionJustTaken(*), RegionMotion(colour, bg, *, *))`
    whose motion vectors span more than one direction, the only
    remaining explanation is that the (colour, bg) sprite is the
    thing the agent controls.  Nothing else in a well-formed domain
    would translate in four distinct directions in one-to-one
    correspondence with the action space.

    Why this deserves its own claim type (rather than a
    :class:`PropertyClaim`):

    * It is keyed by a **colour signature**, not a per-episode
      entity id.  That signature is stable across episodes, adapters,
      and flood-fill seeds — exactly the property we want for
      cross-episode knowledge transfer.
    * It is a *theory* about the structure of the domain, not a
      per-entity attribute.  Treating it as a domain-level claim
      keeps the entity layer free of synthetic ids.
    * Any downstream miner / planner that wants to use it just asks
      "is there a committed ControlledActorClaim?" — one lookup,
      no entity matching.

    Domain-agnostic.  For a robot the analogue is identical: if
    twelve motor commands each produce rigid-body motion of the same
    link, that link is "self"; the miner that mints this claim is
    the same miner in both cases.
    """

    colour:     Any
    background: Any

    def canonical_key(self) -> tuple:
        return ("ControlledActorClaim",
                _hashable(self.colour),
                _hashable(self.background))

    def full_key(self) -> tuple:
        # No extra parameters — canonical_key fully identifies the claim.
        return self.canonical_key()


# ---------------------------------------------------------------------------
# 9. ActorTransitionClaim — "from this state, this action produced this delta
#    on the controlled actor"
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ActorTransitionClaim(Claim):
    """Claim: taking action ``action_id`` while the controlled actor
    is in ``pre_state`` produced observed ``delta`` on the actor's
    position (or, more generally, on its state representation).

    This is the engine's **learned forward-transition primitive**.
    It is deliberately not about "walls" — the word "wall" is a
    game-specific interpretation of a more basic observation, namely
    that *from a particular state, a particular action produced a
    particular delta* (possibly the zero delta).  Downstream planners
    or explorers that care about walls can query the store for
    claims where ``delta`` is zero; games without walls never mint
    such claims and the mechanism simply degenerates to a uniform
    action model.

    What the miner is actually doing.  After each step where the
    actor's pre- and post-positions are both known, the miner
    records the observed ``(pre_state, action_id, delta)`` triple as
    one claim.  Repeated identical observations dedup via the
    canonical key; conflicting observations (same pre_state and
    action, different delta) coexist as separate claims — the store
    is then free to let their credences reflect the empirical
    distribution, which captures stochastic actions for free.

    Why a dedicated claim type (vs CausalClaim).  CausalClaim is
    shaped for "when trigger X fires, event Y follows" with trigger
    and effect as general predicates — two of the existing miners
    (:class:`miners.ActionEffectMiner`, :class:`miners.RegionMotionMiner`)
    already use it for frame-level and direction-only action models
    respectively.  ``ActorTransitionClaim`` is a tighter,
    actor-conditioned schema — just ``(pre_state, action_id, delta)``
    — so canonical-key dedup is trivial and downstream consumers
    (planners, curiosity) can index claims by ``(pre_state,
    action_id)`` in one dict lookup without pattern-matching effect
    shapes.

    Why keyed on absolute ``pre_state``.  The actor-conditioned
    transition model is position-dependent by nature — "ACTION1
    produces +5 row here" may or may not hold elsewhere, and the
    whole point of building the model is to learn that distinction
    empirically (the transfer-friendly abstraction lives in
    :class:`RegionMotion` claims already).  This means
    :class:`ActorTransitionClaim` is **non-transferable across
    episodes by construction** when ``pre_state`` is an absolute
    coordinate, and will not appear in the persistence whitelist.  Transferable
    transition knowledge (e.g. "this colour of sprite moves +5 per
    ACTION1 regardless of cell") is a separate, higher-level claim
    type to land once cross-episode model transfer is actually
    needed.

    Domain-agnostic.  Identical mechanism for a robot arm: record
    ``(joint_config, motor_command, Δconfig)`` after each step to
    learn the forward transition model.  A zero Δ from a command
    that usually moves the arm indicates joint-limit contact,
    obstacle contact, or actuator fault — falling out of the same
    schema, no special "boundary" claim type needed.

    Observability / growth.  Per-state × per-action × per-delta
    claims can in principle grow to
    ``O(|visited_states| × |actions| × |distinct_deltas|)``.  Dedup
    via canonical key means repeated identical observations do not
    duplicate records; they accumulate ``supporting_steps`` on the
    existing entry.  For ls20 with ~64 reachable cells × 4 actions
    × small delta set this stays well-bounded.  For richer state
    representations, ``pre_state`` will need quantisation — a
    problem to solve when it bites, not pre-emptively.
    """

    pre_state: Any    # (row, col) tuple, or any hashable state key
    action_id: str
    delta:     Any    # (d_row, d_col) tuple, or any hashable delta key

    def canonical_key(self) -> tuple:
        return ("ActorTransitionClaim",
                _hashable(self.pre_state),
                str(self.action_id),
                _hashable(self.delta))

    def full_key(self) -> tuple:
        return self.canonical_key()


# ---------------------------------------------------------------------------
# MotionModelClaim — position-independent "typical delta per action"
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class MotionModelClaim(Claim):
    """Claim: action ``action_id`` typically moves the controlled actor
    by ``delta`` regardless of position.

    This is the **planner-facing aggregation** of
    :class:`ActorTransitionClaim` evidence.  ActorTransitionClaim is
    position-conditioned ("from pre=(22,36), ACTION1 produced delta
    (-5,0)"): honest, and per-observation fine-grained, but useless
    to a planner trying to plan from a state it has never visited.
    MotionModelClaim strips the position and records just the modal
    non-zero delta per action — the open-field transition model that
    holds away from boundaries.

    **What is deliberately NOT modelled here.**  Zero-delta
    observations (walls, joint limits, contact) are NOT folded into
    the motion model.  They are position-specific overrides and
    remain in :class:`ActorTransitionClaim`.  The planner consults
    both: motion model says "normally ACTION1 moves up 5"; specific
    zero-delta ActorTransitionClaim at (18,36) says "but from here,
    ACTION1 doesn't move at all."  This two-layer design keeps the
    motion model transferable (strides survive level resets) while
    per-position obstructions stay local.

    **Stochasticity.**  If ACTION1 sometimes moves by (-5,0) and
    sometimes by (-4,0) (stride clamping near borders, say), two
    MotionModelClaims coexist with the same canonical key
    ``(MotionModelClaim, ACTION1)``; they are competitors; the store
    converges on the modal delta via credence.

    **Cross-domain fit.**  For a robot: "motor command `forward_5cm`
    typically produces Δx=+0.05".  Obstacle contact keeps its
    position-conditioned zero-delta records; the open-field motor
    model transfers across tasks.

    **Persistence.**  Unlike ActorTransitionClaim, MotionModelClaim
    IS transferable across episodes — its canonical key has no
    absolute coordinate.  Candidate for the persistence whitelist
    once cross-episode transfer lands for this claim type.
    """

    action_id: str
    delta:     Any    # (d_row, d_col) tuple

    def canonical_key(self) -> tuple:
        return ("MotionModelClaim", str(self.action_id))

    def full_key(self) -> tuple:
        return (*self.canonical_key(), _hashable(self.delta))


# ---------------------------------------------------------------------------
# GoalRegressionClaim — direction-aware learning of regression-causing actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class GoalRegressionClaim(Claim):
    """Claim: from ``pre_state``, taking ``action_id`` regressed
    ``goal_id``'s ``dimension``.

    Direction-aware in the same shape :class:`ActorTransitionClaim`
    is direction-aware: the ``(pre_state, action_id)`` ordering
    matters and the substrate never assumes reversibility.  Going
    from cell A → B with ACTION1 may regress one dimension; going
    B → A with ACTION1 is a separate observation with its own
    claim.

    Minted by :class:`miners.GoalRegressionMiner` from the per-turn
    progress-event stream.  Consumed by the planner's BFS reader,
    which forbids ``(pre_state, action_id)`` edges whose claim
    credence has crossed the threshold and whose ``goal_id`` is
    currently in ACHIEVED status.

    Non-transferable across episodes by construction (absolute
    coordinates and goal IDs do not survive resets).  Transferable
    regression knowledge — "this entity-arrangement is fragile to
    this action class" — is a separate higher-level claim type
    when needed.
    """

    pre_state: Any    # (row, col) tuple, or any hashable state key
    action_id: str
    goal_id:   str
    dimension: str    # "*" for whole-goal regression sentinel

    def canonical_key(self) -> tuple:
        return ("GoalRegressionClaim",
                _hashable(self.pre_state),
                str(self.action_id),
                str(self.goal_id),
                str(self.dimension))

    def full_key(self) -> tuple:
        return self.canonical_key()


# ---------------------------------------------------------------------------
# 10. BitmapRoleClaim — a canonical sprite bitmap plays a perception role
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BitmapRoleClaim(Claim):
    """Claim: the sprite with canonical bitmap ``bitmap_id`` plays
    perception-role ``role``.

    Bridges the perception substrate's role assignments into the
    engine's claim/hypothesis machinery.  Today the substrate writes
    role assignments into per-game ``<game>_runtime.json`` files that
    no other engine component reads; making them
    :class:`BitmapRoleClaim` instances routes them through
    :func:`hypothesis_store.propose`, gives them credence and
    competitor-tracking, and makes them eligible for the engine's
    persistence + cross-episode transfer machinery.

    Keying.  Canonical key is the bitmap_id alone, so claims asserting
    different roles for the same bitmap (e.g. one source proposes
    ``launcher`` and another proposes ``target_slot``) compete for
    evidence.  Full key adds the role so identical (bitmap, role)
    proposals from different sources merge their support rather than
    duplicating.

    Transferability across episodes / levels.  Bitmap fingerprints are
    canonical hashes computed from sprite pixels and are stable across
    a level's lifetime — re-spawning the same level produces the same
    bitmap_id for the same sprite.  Across levels within one game the
    same sprite (agent, common collectible, etc.) usually shares its
    bitmap_id too.  Across games it does not.  This claim type is
    therefore a good fit for ``Scope.GAME`` and ``Scope.LEVEL``;
    cross-game transfer should instead use a shape-keyed claim
    (future :class:`ShapeRoleClaim`, keyed by palette-invariant
    ``topo_id``).

    Dual-domain.  Robotics analogue: ``object_signature → role``
    ("this depth-hashed object cluster is a graspable_box") works
    identically; the substrate just produces different signatures.

    Substrate-general per the standing directive.  ``role`` is an
    opaque catalog primitive id (``"launcher"`` /
    ``"target_slot"`` / ...) — this class doesn't enforce a
    vocabulary; the catalog layer does.
    """

    bitmap_id: str
    role:      str
    # Optional metadata fields, populated by the perception bridge when
    # the parsed view carries them.  None of these participate in
    # canonical_key / full_key -- they're attributes used by the
    # bitmap-role matcher's shape and topo fallback tiers, not identity.
    shape_id:     Optional[str] = None
    topo_id:      Optional[str] = None
    size_px:      Optional[int] = None
    spatial_zone: Optional[str] = None

    def canonical_key(self) -> tuple:
        return ("BitmapRoleClaim", str(self.bitmap_id))

    def full_key(self) -> tuple:
        return (*self.canonical_key(), str(self.role))


# ---------------------------------------------------------------------------
# 11. RegionPaletteClaim — a palette (or palette-set) plays a region role
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class RegionPaletteClaim(Claim):
    """Claim: the palette signature ``palettes`` plays region-role
    ``role``.

    The substrate's region-role detection (background, wall_palette,
    play_area, void, ...) currently produces an opaque
    ``background_palettes`` list in the parsed view that no other
    engine component reads.  Making each palette → role assignment a
    :class:`RegionPaletteClaim` puts it on the same footing as
    :class:`PropertyClaim` and friends: credence, competitor tracking,
    persistence, provenance.

    ``palettes`` is a tuple of palette integers in sorted order.  Use
    the :meth:`make` factory to construct so the tuple normalisation
    is enforced.  A single palette is the common case (e.g.
    ``palettes=(5,)``); multi-palette region signatures (e.g. a
    dotted background that includes both the dot palette and the
    surrounding palette) use a longer tuple.

    Keying.  Canonical key is the palette signature alone, so
    different role assignments for the same palette signature compete
    (e.g. "palette 5 is background" vs "palette 5 is wall").  Full
    key adds the role.

    Transferability across episodes / levels.  Palette integers are
    stable within an env (a game's palette indices don't drift between
    levels), so region-palette assignments transfer cleanly across
    levels of the same game.  Across games palette indices are
    NOT stable; this claim type fits ``Scope.GAME`` (and
    ``Scope.LEVEL`` when needed) but should not be persisted as a
    ``Scope.GLOBAL`` prior.

    Dual-domain.  Robotics analogue: a ``surface_classification ->
    role`` claim ("this point-cloud cluster classified as 'metal' is
    a structural support beam, not a movable object") has the same
    shape; the substrate's segmentation gives a categorical signature
    rather than a numeric palette index.
    """

    palettes: Tuple[int, ...]
    role:     str
    # Optional metadata fields for the region-role matcher.  Not part
    # of canonical_key / full_key -- two RegionPaletteClaims with the
    # same (palettes, role) but different bbox row ranges from
    # different levels merge into one claim and accumulate evidence.
    row_range:    Optional[Tuple[int, int]] = None
    spatial_zone: Optional[str]             = None

    @classmethod
    def make(cls, palettes, role: str,
             *,
             row_range:    Optional[Tuple[int, int]] = None,
             spatial_zone: Optional[str]             = None,
             ) -> "RegionPaletteClaim":
        """Build a claim from any iterable of palette ints.  The
        tuple is sorted-and-deduplicated so equivalent palette sets
        produce equal claims regardless of input ordering.

        ``row_range`` and ``spatial_zone`` are optional region-matcher
        metadata; when present they help the region-role matcher
        accept later observations as the same region under bbox
        drift."""
        normalised = tuple(sorted({int(p) for p in palettes}))
        rr_norm: Optional[Tuple[int, int]] = None
        if row_range is not None:
            r0, r1 = row_range
            rr_norm = (int(r0), int(r1))
        return cls(palettes=normalised, role=str(role),
                   row_range=rr_norm, spatial_zone=spatial_zone)

    def canonical_key(self) -> tuple:
        return ("RegionPaletteClaim", tuple(self.palettes))

    def full_key(self) -> tuple:
        return (*self.canonical_key(), str(self.role))
