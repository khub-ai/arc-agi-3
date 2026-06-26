"""Core data types of the cognitive engine.

This module consolidates the top-level types the rest of the engine
operates on: Observation, Hypothesis, Rule, Goal, Plan, WorldState, and
the Observer query protocol.  Lower-level types (Condition, Claim,
Credence) live in their own modules and are re-exported via the package
``__init__``.

All types in this module are pure data structures.  Phase 1 contains no
behaviour beyond simple accessors and factory helpers; learning,
planning, and execution are implemented in later phases.

Standing directive: nothing in this file may encode the mechanics of
any specific domain.  ``Observation.raw_frame`` is typed as ``Any``
deliberately — adapters decide what a "frame" is (2-D image, video
segment, point cloud, ROS message).  Event types are generic
(AgentMoved, ResourceChanged, EntityStateChanged); per-domain semantics
live in the adapter that emits them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .claims import Claim
from .conditions import Condition
from .credence import Credence
from .tools import (
    ToolInvocation,
    ToolProposal,
    ToolRegistry,
    ToolResult,
    ToolSignature,
)


# ===========================================================================
# Scope — where/when a hypothesis, rule, or goal applies
# ===========================================================================


class ScopeKind(Enum):
    """Temporal extent of a scoped object.

    These kinds form an implicit hierarchy; clearing a broader scope
    (e.g. ``LEVEL``) cascades to any contained narrower scopes
    (``STEP``/``LIFE``/``EPISODE``).  The clear-on-transition policy is
    enforced by the WorldState at scope boundaries.
    """

    STEP     = "step"       # single observation
    LIFE     = "life"       # between death-resets within an episode
    EPISODE  = "episode"    # one run attempt on one level/task
    LEVEL    = "level"      # one game level or task variant
    GAME     = "game"       # one game/task family
    GLOBAL   = "global"     # persistent across everything


@dataclass(frozen=True)
class Scope:
    """Structural scope of a belief.

    ``kind`` gives the temporal extent.  The optional filters narrow
    applicability further: a hypothesis scoped to a specific
    ``position_region`` is evaluated only when the agent / target entity
    is in that region.
    """

    kind:            ScopeKind = ScopeKind.EPISODE
    position_region: Optional[Tuple[Any, ...]] = None   # implementation-defined bounding region
    entity_filter:   Optional[frozenset] = None         # restrict to these entity IDs
    time_range:      Optional[Tuple[int, int]] = None   # [start_step, end_step]


# ===========================================================================
# Events — per-step occurrences emitted by Adapter
# ===========================================================================


class Event:
    """Base class for all events.  Concrete events are frozen dataclasses.

    The ``step`` attribute is set by the adapter when emitting.  Events
    are the unit of symbolic information flowing from adapter to engine.
    New event types can be added for new domains; miners consume events
    they recognise and ignore the rest.
    """

    step: int


@dataclass(frozen=True)
class AgentMoved(Event):
    """Agent's position changed from ``from_pos`` to ``to_pos``."""
    step: int
    from_pos: Tuple[Any, ...]
    to_pos:   Tuple[Any, ...]


@dataclass(frozen=True)
class AgentDied(Event):
    """Agent lost a life or was terminated.  ``cause`` is an adapter-
    provided reason string (e.g. ``"lethal_cell"``, ``"budget_exhausted"``,
    ``"task_failure"``); the engine treats cause strings opaquely for
    symbolic correlation — it does not parse them for specific domains."""
    step: int
    cause: str


@dataclass(frozen=True)
class ResourceChanged(Event):
    """Tracked resource went from ``old_val`` to ``new_val``.

    Both agent-owned resources (budget, energy, time) and world-owned
    resources (e.g. a shared tank level) are emitted through this event.
    The adapter determines the resource identity.
    """
    step: int
    resource_id: str
    old_val: float
    new_val: float


@dataclass(frozen=True)
class EntityStateChanged(Event):
    """A property of ``entity_id`` changed value from ``old`` to ``new``.

    The general-purpose way to signal "something about this entity is
    different now".  Miners consume these events to form PropertyClaims,
    CausalClaims, and RelationalClaims.
    """
    step: int
    entity_id: str
    property: str
    old: Any
    new: Any


@dataclass(frozen=True)
class GoalConditionMet(Event):
    """A tracked goal's success condition became true this step."""
    step: int
    goal_id: str


@dataclass(frozen=True)
class EntityAppeared(Event):
    """A new entity became visible / trackable for the first time."""
    step: int
    entity_id: str
    initial_state: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EntityDisappeared(Event):
    """A previously-tracked entity is no longer visible / reachable."""
    step: int
    entity_id: str


@dataclass(frozen=True)
class EntityVisualPatternChanged(Event):
    """The interior pixel pattern of an entity's bounding-box region
    changed while the entity's colour and general vicinity stayed constant.

    The canonical example is a glyph inside a scan-box being rotated by
    an action: the same colour reappears in the same region but with a
    different internal arrangement.  Perception detects this by noticing
    a same-colour entity disappear and a new same-colour entity appear
    with an overlapping bounding box in the same step.

    ``entity_id_before``
        ID of the entity that disappeared (old shape).
    ``entity_id_after``
        ID of the entity that appeared (new shape, same colour).
    ``colour``
        Shared palette index of both entities.
    ``bbox``
        Union bounding box of the old and new regions — the encompassing
        area of the mutation.  Invariant across entity-ID changes so
        oracle triggers can use it as a stable location reference.

    Dual-domain.  Robotics analogue: a screw was turned (same colour /
    same rough location, different thread orientation), a dial was
    rotated (same face colour, different angular position), or a part
    flipped on a conveyor (same outline colour, mirrored interior).
    """
    step:             int
    entity_id_before: str
    entity_id_after:  str
    colour:           Any
    bbox:             Tuple[int, int, int, int]   # (r_min, c_min, r_max, c_max)


@dataclass(frozen=True)
class ContactEvent(Event):
    """The controlled actor's new position overlaps a cell previously
    occupied by an entity of ``other_colour``.

    This is the rawest possible contact signal: a geometric
    observation that the actor painted over a non-background,
    non-self colour when it moved.  The event says **nothing** about
    what the contact *means* — whether it was a pickup, a hazard, a
    goal, or pass-through is a downstream question answered by
    correlating ``ContactEvent`` occurrences with score changes,
    ``GoalConditionMet``, ``AgentDied``, or frame-level side-effects
    in subsequent steps.

    Emitting as an ``Event`` (not a ``Claim``) is deliberate: contact
    is a per-step transient observation that feeds the event stream
    miners and the Mediator already consume.  Persistent beliefs
    about "touching colour-X wins the game" belong in a
    second-order claim (future) minted by a correlation miner that
    pairs ``ContactEvent`` with reward/terminal events.

    Dual-domain.  Robotics analogue: the end-effector's new joint-
    config-derived pose overlaps a point cloud previously occupied
    by an object.  Same mechanism; downstream code decides whether
    that contact means "grasped", "pushed", "bumped into obstacle",
    or "passed a kinematic singularity".
    """
    step:          int
    actor_colour:  Any
    other_colour:  Any
    cell:          Tuple[int, int]


@dataclass(frozen=True)
class SurpriseEvent(Event):
    """A committed prediction was violated.

    Emitted either by the adapter (when it detects a gross discrepancy)
    or synthesised by the engine itself (when an observation contradicts
    a committed TransitionClaim).  Triggers the abductive proposer and
    may trigger replanning.
    """
    step: int
    expected: Any
    actual:   Any
    context:  str = ""


# ---------------------------------------------------------------------------
# Durative-skill events — lifecycle signals for actions that take time
# ---------------------------------------------------------------------------
#
# ARC-AGI-3 actions are atomic: one tick, one effect.  Embodied domains
# (robotics, any async control surface) execute *durative* skills — a
# ``grasp`` that runs for hundreds of milliseconds, can be preempted, and
# terminates with a typed outcome.  These events let the adapter surface a
# skill's lifecycle into the same typed event stream the engine already
# consumes, WITHOUT the engine ever leaving its step-synchronous loop: the
# adapter runs (or dispatches) the skill in ``execute`` and emits the
# terminal event on the next ``observe``.  The ``SkillHandle`` abstraction
# that produces these lives adapter-side (see ``usecases/robotics``); the
# engine only ever sees the events.
#
# Miners consume the events they recognise and ignore the rest (per the
# Event base contract), so adding these is additive and breaks nothing.
# A skill's MECHANICAL outcome (did the command run?) is distinct from its
# expected POST-CONDITION (did the world reach the predicted state?): a
# ``SkillSucceeded`` with ``observed_effects`` that don't match the
# prediction is exactly the closed-loop substrate's primary learning
# signal.  See ``usecases/robotics/SPEC_milestone_0_diagnostic.md`` §3.


@dataclass(frozen=True)
class SkillStarted(Event):
    """A durative skill began executing.

    ``handle_id`` identifies the in-flight skill invocation (multiple may
    be live at once — e.g. a long manipulation skill running while a
    monitor skill ticks).  ``skill`` is the skill name; ``params`` is the
    adapter-supplied parameter mapping (e.g. ``{"object_id": "red"}``).
    """
    step:      int
    handle_id: str
    skill:     str
    params:    Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SkillProgress(Event):
    """Optional intermediate milestone for an in-flight durative skill.

    ``milestone`` is an adapter-defined opaque label (e.g.
    ``"approach_complete"``).  Emitted zero or more times between
    :class:`SkillStarted` and the skill's terminal event.
    """
    step:      int
    handle_id: str
    milestone: str


@dataclass(frozen=True)
class SkillSucceeded(Event):
    """A durative skill terminated with mechanical success.

    ``observed_effects`` is the adapter's report of what actually changed
    in the world as a result (each effect an adapter-defined mapping or
    string).  Mechanical success does NOT imply the skill's expected
    post-condition holds — the engine compares ``observed_effects`` (and
    the subsequent observation) against the action's prediction; a
    mismatch is the learning signal, not an error.
    """
    step:             int
    handle_id:        str
    skill:            str
    observed_effects: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class SkillFailed(Event):
    """A durative skill terminated without mechanical success.

    ``reason`` is an adapter-provided opaque string (the engine correlates
    it symbolically; it does not parse it).  ``partial_effects`` reports
    any world changes that occurred before the failure.
    """
    step:            int
    handle_id:       str
    skill:           str
    reason:          str
    partial_effects: Tuple[Any, ...] = ()


@dataclass(frozen=True)
class SkillPreempted(Event):
    """An in-flight durative skill was preempted before terminating.

    ``by`` is an adapter-defined label for what preempted it (another
    skill, a safety stop, an operator interrupt).
    """
    step:      int
    handle_id: str
    skill:     str
    by:        str = ""


# ===========================================================================
# Observation — what the adapter emits each step
# ===========================================================================


@dataclass
class Observation:
    """One step of symbolic data from the adapter.

    ``raw_frame`` is preserved so the engine can issue an ObserverQuery
    later and the adapter can answer it visually.  The engine itself
    MUST NOT inspect ``raw_frame`` — only hand it to the Observer via an
    ObserverQuery.  Metadata is a free-form dict for adapter-specific
    extensions (timestamps, sensor IDs, etc.).
    """

    step:             int
    agent_state:      Dict[str, Any]
    events:           List[Event]
    entity_snapshots: Dict[str, Dict[str, Any]]   # entity_id → {property: value}
    raw_frame:        Any = None
    metadata:         Dict[str, Any] = field(default_factory=dict)
    #: Wall-clock timestamp (seconds) for this observation, when the domain
    #: has real elapsed time.  ``step`` remains the logical tick; ``t_wall``
    #: lets time-referenced claims and credence decay ("object unmoved for
    #: 5s", "skill took 800ms") be expressed without conflating them with
    #: action count.  ``None`` for domains with no meaningful wall-clock
    #: (e.g. ARC-AGI-3, where one step is the only notion of time).
    t_wall:           Optional[float] = None


# ===========================================================================
# Entities
# ===========================================================================


@dataclass
class EntityModel:
    """The engine's view of one tracked entity.

    Properties accumulate over time as the adapter reports them.  A
    ``kind`` label is optional and provides a coarse semantic category
    (e.g. ``"agent"``, ``"obstacle"``, ``"resource"``, ``"gate"``) when
    the adapter can supply one; without it the engine works from
    property-by-property evidence alone.
    """

    id:               str
    properties:       Dict[str, Any] = field(default_factory=dict)
    first_seen_step:  int = -1
    last_seen_step:   int = -1
    kind:             Optional[str] = None


# ===========================================================================
# Hypothesis — the learning unit
# ===========================================================================


@dataclass
class Hypothesis:
    """A candidate claim held with some credence.

    The hypothesis store maintains a lattice of Hypotheses linked by
    parent/child relations (generalisation/specialisation).  Siblings
    are reachable via parent.child_ids.  Competing hypotheses (same
    canonical key, different parameters) are linked via the
    Credence.competing field.
    """

    id:                  str
    claim:               Claim
    credence:            Credence
    scope:               Scope
    source:              str                                  # "miner:Name" | "adapter:seed" | ...
    supporting_steps:    List[int]            = field(default_factory=list)
    contradicting_steps: List[int]            = field(default_factory=list)
    expires_at:          Optional[int]        = None
    # Lattice links
    parent_id:           Optional[str]        = None
    child_ids:           List[str]            = field(default_factory=list)
    # Metadata
    created_at:          int                  = 0
    rationale:           Optional[str]        = None


# ===========================================================================
# Rule — externally imposed, authority-weighted
# ===========================================================================


class PrincipalKind(Enum):
    """Category of the entity whose will a Rule represents.

    For ARC-AGI-3 the only principal is SYSTEM; robotics requires the
    full taxonomy for conflict resolution across multiple humans.
    """
    OWNER        = "owner"
    OPERATOR     = "operator"
    GUEST        = "guest"
    STRANGER     = "stranger"
    BYSTANDER    = "bystander"
    SYSTEM       = "system"
    SAFETY_SPEC  = "safety_spec"


@dataclass(frozen=True)
class Principal:
    """An entity whose directives produce Rules.

    ``authority`` is a 0..1 base level; the effective authority at
    decision time may be modulated by the ``context`` condition
    (e.g. the owner's authority is full at home but limited in public).
    """
    id:        str
    kind:      PrincipalKind
    authority: float
    context:   Optional[Condition] = None


class Violability(Enum):
    """How strictly a Rule must be followed.

    INVIOLABLE
        Never violate under any circumstance; planner treats as hard
        filter (prunes all plans that violate the rule).
    DEFEASIBLE
        May be violated if a higher-authority or inviolable rule forces
        it; otherwise hard filter.
    ADVISORY
        Soft preference; violation adds a cost penalty proportional to
        ``priority`` but does not prune.
    """
    INVIOLABLE = "inviolable"
    DEFEASIBLE = "defeasible"
    ADVISORY   = "advisory"


class ConstraintKind(Enum):
    """Prohibit an action/condition, require it, or merely prefer it."""
    PROHIBIT = "prohibit"
    REQUIRE  = "require"
    PREFER   = "prefer"


@dataclass(frozen=True)
class RuleConstraint:
    """The content of a Rule: what action or condition it targets.

    ``target`` is either a Condition (for state-based rules: "never
    allow Condition C to hold") or an action-name string (for
    action-based rules: "never execute action A").  ``weight`` is used
    only for PREFER-kind constraints.
    """
    kind:   ConstraintKind
    target: Union[Condition, str]
    weight: float = 1.0


@dataclass
class Rule:
    """An externally imposed constraint.

    Unlike hypotheses, Rules are authoritative — they do not need
    empirical support to be binding, and their violation is not a
    disconfirmation but an action the planner avoids.  ``priority``
    is typically derived as ``principal.authority * <rule-level weight>``
    and is recomputed when the active set of principals changes.
    """
    id:           str
    condition:    Condition                # when does this rule apply?
    constraint:   RuleConstraint           # what does it require or prohibit?
    principal:    Principal
    violability:  Violability
    priority:     float
    scope:        Scope
    source:       str
    expires_at:   Optional[int] = None
    rationale:    Optional[str] = None
    created_at:   int = 0


# ===========================================================================
# Goal tree: AND-OR-CHANCE with robotics extensions
# ===========================================================================


class NodeType(Enum):
    """Types of goal-decomposition nodes.

    Active (used by the planner today):
        ATOM    — a leaf: directly achievable condition
        AND     — all children must be achieved
        OR      — any one child suffices
        CHANCE  — environment resolves with a prior distribution over outcomes

    Future (reserved for robotics / multi-agent extensions; not yet
    implemented by the planner):
        OPTION       — macro-action / sub-plan
        MAINTAIN     — ongoing condition to preserve, not achieve once
        LOOP         — cyclic sub-plan with exit condition
        ADVERSARIAL  — opposing agent's decision
        INFO_SET     — indistinguishable states, single strategy
    """
    ATOM         = "atom"
    AND          = "and"
    OR           = "or"
    CHANCE       = "chance"
    OPTION       = "option"
    MAINTAIN     = "maintain"
    LOOP         = "loop"
    ADVERSARIAL  = "adversarial"
    INFO_SET     = "info_set"


class Ordering(Enum):
    """For AND nodes: whether children must be achieved in order or any order."""
    SEQUENTIAL = "sequential"
    UNORDERED  = "unordered"


class GoalStatus(Enum):
    OPEN      = "open"         # not yet pursued
    ACTIVE    = "active"       # currently being pursued
    ACHIEVED  = "achieved"
    BLOCKED   = "blocked"      # no plan found; may become unblocked
    PRUNED    = "pruned"       # ruled out by evidence / higher-authority rule
    ABANDONED = "abandoned"    # given up permanently (e.g. deadline missed)


# ===========================================================================
# Dependency expressions (axis 1 of SPEC_goal_classification.md)
# ===========================================================================
#
# A goal carries an optional ``depends_on: DepExpr`` that names other
# top-level goals as preconditions.  The selector filters by
# ``is_actionable`` (see ``goal_forest.is_actionable``) before priority
# ranking, so a goal whose dependencies are unmet is hidden from
# selection rather than competing with actionable goals on priority.
#
# The expression is a small algebraic data type (DepRef / DepAll /
# DepAny) supporting arbitrary AND/OR nesting.  Frozen dataclasses
# make instances hashable so they can serve as dict keys in audit
# code; tuple children (rather than list) preserve immutability.


@dataclass(frozen=True)
class DepRef:
    """Reference to another top-level goal by id.  Satisfied when the
    referenced goal's status is ``ACHIEVED``."""
    goal_id: str


@dataclass(frozen=True)
class DepAll:
    """AND of dependency expressions.  Satisfied when every child is
    satisfied.  Empty children evaluates to True (vacuous AND)."""
    children: Tuple["DepExpr", ...]


@dataclass(frozen=True)
class DepAny:
    """OR of dependency expressions.  Satisfied when at least one child
    is satisfied.  Empty children is rejected at declaration as an
    operator authoring error (vacuous OR can never be satisfied)."""
    children: Tuple["DepExpr", ...]


DepExpr = Union[DepRef, DepAll, DepAny]


@dataclass
class GoalNode:
    """Node in the AND-OR-CHANCE goal tree.

    Leaf nodes are ATOM with a Condition.  Composite nodes (AND/OR/
    CHANCE) carry children.  The planner walks this tree to produce a
    :class:`Plan`; nodes carry both their current ``status`` and, for OR
    nodes, the currently-selected ``active_branch`` child.

    ``supporting_hypothesis_ids`` records which hypotheses motivated the
    insertion of this node so that if they are later falsified the
    planner knows to prune the branch.
    """
    id:         str
    node_type:  NodeType
    condition:  Optional[Condition]         = None   # required for ATOM nodes
    children:   List["GoalNode"]            = field(default_factory=list)
    ordering:   Ordering                    = Ordering.SEQUENTIAL
    status:     GoalStatus                  = GoalStatus.OPEN
    priority:   float                       = 0.5
    deadline:   Optional[int]               = None
    active_branch: Optional[str]            = None   # child_id currently selected (OR nodes only)
    # CHANCE nodes: probability prior over named outcomes
    outcome_priors: Dict[str, float]        = field(default_factory=dict)
    # Which hypotheses justified this node's existence
    supporting_hypothesis_ids: List[str]    = field(default_factory=list)
    source:     str                         = "engine:derived"
    created_at: int                         = 0
    # --- deferred-plan marker -------------------------------------------------
    # When ``True``, the planner treats this node as already planned
    # (returns an empty step sequence) even if its ``condition`` is not
    # currently true.  The runtime achievement check (``is_achieved``)
    # still fires: when the condition *does* become true during
    # execution the AND/OR parent will accept it and the overall goal
    # can close.  This is how we wire up "do trigger X, which will
    # eventually cause effect Y" without the planner demanding a
    # transition model for the unprobed effect path.  Typical use:
    # verifier children in derive_subgoals_from_causal's AND(trigger,
    # verifier) bundles, where the verifier is the CausalClaim effect
    # (e.g., episode_won rising above 0.5) that has no learned
    # transition model and should not block planning of the trigger.
    deferred_plan: bool                     = False


@dataclass
class Goal:
    """Top-level wrapper around a goal tree.

    A Goal has a unique ID, a principal (whose goal it is — important for
    robotics conflict resolution), a priority, and a ``root`` GoalNode
    that is the AND-OR-CHANCE tree to be satisfied.

    ``progress_history`` records per-turn snapshots of which dimensions
    of the goal are currently matched, plus events derived from
    snapshot diffs.  Populated by the goal-progress-tracking
    substrate; default-empty for goals not consumed by it.
    See ``docs/SPEC_goal_progress_tracking.md``.
    """
    id:                str
    root:              GoalNode
    priority:          float
    deadline:          Optional[int]      = None
    source:            str                = "adapter:primary"
    principal:         Optional[Principal] = None
    created_at:        int                = 0
    progress_history:  "GoalProgressHistory" = field(
        default_factory=lambda: GoalProgressHistory()
    )
    # --- dependency edge (SPEC_goal_dependencies.md) ----------------------
    # An optional precondition expression over other top-level goal ids.
    # ``None`` means no dependencies — the goal is always actionable.
    # When non-None, ``goal_forest.is_actionable(goal, forest)`` returns
    # True iff every ``DepRef`` in the expression resolves to a goal
    # whose status is ``ACHIEVED``, with AND/OR composition handled by
    # ``DepAll`` / ``DepAny`` nodes.  Defaults to None so existing goals
    # are unaffected.
    depends_on:        Optional[DepExpr]  = None

    # --- structural tags (SPEC_goal_classification.md, axis 2) ----
    # Domain-agnostic role labels.  Vocabulary defined in the spec:
    # ``task``, ``terminal``, ``tracker``, ``refuel``, ``pickup``,
    # ``trigger``, ``investigation``, ``recovery``.  Tags compose:
    # an unknown refuel pad may carry ``{"refuel", "investigation"}``.
    # Consumers query ``"refuel" in g.tags`` rather than parsing the
    # goal id with ``startswith``.  Empty by default so existing goals
    # are unaffected.
    tags:              frozenset[str]     = field(default_factory=frozenset)

    @property
    def condition(self) -> Optional[Condition]:
        """Success condition of an atomic top-level goal, if applicable."""
        return self.root.condition if self.root.node_type == NodeType.ATOM else None

    @property
    def status(self) -> GoalStatus:
        return self.root.status


# --- conflicts ---


class ConflictType(Enum):
    MUTEX       = "mutex"        # success conditions logically incompatible
    RESOURCE    = "resource"     # compete for the same limited resource
    ADVERSARIAL = "adversarial"  # one principal opposes another's goal
    TEMPORAL    = "temporal"     # deadlines incompatible with sequential pursuit


class ResolutionPolicy(Enum):
    PRIORITY        = "priority"
    INTERLEAVE      = "interleave"
    USER_ARBITRATE  = "user_arbitrate"
    FAIL            = "fail"


@dataclass
class GoalConflict:
    """A detected conflict between two goals in the forest.

    Populated by :class:`GoalForest` conflict-detection logic (later
    phase).  The ``rationale`` is a human-readable explanation for
    audit / user display.
    """
    goal_a:             str
    goal_b:             str
    conflict_type:      ConflictType
    resolution_policy:  ResolutionPolicy
    detected_at:        int
    rationale:          str


@dataclass
class GoalForest:
    """Collection of active goal trees with conflict tracking.

    Only ``active_goal_id`` receives planning attention at any moment;
    the goal-selection policy (later phase) determines which goal is
    active based on priorities, deadlines, and unresolved conflicts.

    The two reflection-derived annotation fields below are populated
    by the closed-loop substrate's reflection pass each turn and
    consumed by the primary goal selector to bias selection away
    from goals or cells where recent behaviour has been
    unproductive.  The selector applies a priority penalty when
    computing effective priority for ranking; the underlying
    ``goal.priority`` is unchanged so the bias is reversible the
    moment reflection stops flagging the entries.
    """
    goals:           Dict[str, Goal]      = field(default_factory=dict)
    conflicts:       List[GoalConflict]   = field(default_factory=list)
    active_goal_id:  Optional[str]        = None
    # Goal ids reflection has flagged as stalled in the most recent
    # pass.  Refreshed each turn; entries drop when the goal stops
    # qualifying.
    stalled_goal_ids: "set[str]"          = field(default_factory=set)
    # Cells reflection has flagged as part of an active behavioural
    # cycle.  Same refresh semantics as ``stalled_goal_ids``.
    cycling_cells:    "set[Tuple[int, int]]" = field(default_factory=set)


# ===========================================================================
# Actions and plans
# ===========================================================================


@dataclass(frozen=True)
class Action:
    """An action the adapter can execute.

    ``parameters`` is a canonical (sorted) tuple of kv pairs so that two
    actions constructed from the same logical arguments hash and compare
    equal regardless of construction order.  The adapter is responsible
    for translating a canonical Action back to its domain-native form
    (e.g. an API call, a motor command, a game controller input).
    """
    id:         str
    name:       str
    parameters: Tuple[Tuple[str, Any], ...] = ()

    @classmethod
    def make(cls, name: str, **params: Any) -> "Action":
        sorted_params = tuple(sorted(params.items()))
        # Default canonical id: name + repr of params — adapters may choose
        # their own ID scheme (e.g. domain-native command strings).
        aid = f"{name}({','.join(f'{k}={v}' for k, v in sorted_params)})"
        return cls(id=aid, name=name, parameters=sorted_params)


class PlanStatus(Enum):
    ACTIVE      = "active"
    EXECUTING   = "executing"
    COMPLETE    = "complete"
    INVALIDATED = "invalidated"
    FAILED      = "failed"


@dataclass
class PlannedAction:
    """One step in a Plan: the action to take, plus metadata.

    ``depends_on_hypotheses`` lists hypothesis IDs whose truth the step
    relies on; when any of them is demoted below commit, the plan is
    invalidated.  ``expected_effects`` are the TransitionClaims the
    planner used to forecast the outcome — their post-conditions are
    checked against actual observation and any mismatch feeds back into
    hypothesis evidence.
    """
    action:                Action
    expected_effects:      List[Claim] = field(default_factory=list)
    depends_on_hypotheses: List[str]   = field(default_factory=list)
    pre_condition:         Optional[Condition] = None


@dataclass
class Plan:
    """A selected path through the goal AND-OR tree expressed as an
    ordered sequence of PlannedActions.

    ``branch_selections`` records which child was chosen at each OR
    node encountered during planning — used to back-track efficiently
    when a plan is invalidated.  ``assumptions`` is the flattened set
    of hypothesis IDs every step depends on, cached here so the
    invalidation check is an O(|assumptions|) membership test each
    step, not a walk through the full tree.
    """
    goal_id:            str
    steps:              List[PlannedAction]
    computed_at:        int
    assumptions:        List[str]           = field(default_factory=list)
    branch_selections:  Dict[str, str]      = field(default_factory=dict)
    status:             PlanStatus          = PlanStatus.ACTIVE
    current_step_index: int                 = 0


# ===========================================================================
# Option — learned macro-action (tool creation via composition)
# ===========================================================================


@dataclass
class Option:
    """A learned macro-action: a parameterised sub-plan promoted to a
    single callable unit.

    Once the engine has executed a successful plan fragment several times
    across varying contexts, an OptionSynthesiser miner abstracts the
    varying parts into parameters and registers the fragment as an
    Option.  The Option joins the action space; future plans can invoke
    it as a single step instead of searching through its constituent
    actions, collapsing the branching factor in later planning.

    Attributes
    ----------
    id
        Stable identifier.
    name
        Human-readable name (may appear in logs and Mediator contexts).
    parameters
        Ordered tuple of ``(param_name, type_hint)`` pairs describing
        inputs the caller must supply.
    internal_plan
        The concrete sub-plan with parameter placeholders.  Execution
        substitutes actual arguments at invocation time.
    applicability
        Condition under which invoking the Option is sensible.  The
        Planner uses this as a guard; invocations whose preconditions
        are not met are rejected.
    expected_effects
        Claims the Option is believed to produce on success.  Feed into
        Planner forecasting and into post-hoc evidence comparison.
    success_rate
        Empirical success rate over past invocations; updated after each
        use.  Used by the OR-node branch selector (via StrategyClaim) to
        compare competing Options.
    n_uses
        Total invocations recorded.
    source
        Provenance tag: ``"synthesiser:RepeatedFragment"`` for code-side
        synthesis, ``"mediator:synthesis"`` for Mediator-proposed options,
        ``"user:teach"`` for operator-taught macros.
    scope
        Defaults to ``ScopeKind.GAME`` — re-usable in future runs of the
        same game/task.  Promoted to ``ScopeKind.GLOBAL`` once the
        Option has proved useful across multiple games (i.e. it
        crossed a cross-domain reuse threshold).
    """

    id:                  str
    name:                str
    parameters:          Tuple[Tuple[str, str], ...]
    internal_plan:       "Plan"
    applicability:       Condition
    expected_effects:    List[Claim]         = field(default_factory=list)
    success_rate:        float               = 0.5
    n_uses:              int                 = 0
    source:              str                 = "synthesiser:RepeatedFragment"
    scope:               "Scope"             = field(default_factory=lambda: Scope(kind=ScopeKind.GAME))
    created_at:          int                 = 0
    rationale:           Optional[str]       = None


# ===========================================================================
# CachedSolution — recorded executable sequence for a known task
# ===========================================================================


@dataclass
class CachedSolution:
    """A recorded sequence of actions known to achieve a specific task.

    This is the substrate for two superficially different capabilities
    that are structurally the same:

    **Game-level replay (ARC-AGI-3)** — an action sequence that solved
    a specific level.  Loaded in training mode to skip past already-
    solved levels; purged in competition mode.  Typically
    ``scope = LEVEL``.

    **Procedural / muscle-memory skills (robotics)** — a rehearsed
    motor sequence for a task like "walk to pose X" or "grasp cup",
    executed efficiently with minimal cognitive monitoring.  Typically
    ``scope = GAME`` (task-family-specific) or ``GLOBAL``
    (cross-task locomotion primitives like "walk").

    Distinct from :class:`Hypothesis` because it is a *recording* or
    *rehearsed procedure*, not a probabilistic belief.  A single
    failure does not falsify a CachedSolution — especially a
    non-deterministic one — it just updates the empirical success
    rate.  Removal happens by explicit abandonment (low success rate
    dropping below a threshold, or mode-gated purging at episode
    start).

    Distinct from :class:`Option` because:

    * An Option is a first-class entry in the action space; the
      Planner composes it with other actions during search.
    * A CachedSolution is a *complete* procedure for a task and is
      typically invoked as a whole unit with reduced monitoring.
    * Options are always parameterised abstractions; CachedSolutions
      may be concrete (exact action sequence for one level) or
      parameterised (muscle-memory skills).

    Attributes
    ----------
    id
        Stable identifier, typically ``f"{task_id}:{params_hash}"``.
    task_id
        Adapter-defined task identifier.  Examples:
        ``"arc:ls20:L2"``, ``"robotics:walk_to"``,
        ``"robotics:grasp_from_table"``.  The engine does not parse
        this string; the adapter owns the naming convention.
    task_parameters
        For parameterised skills (e.g. walk-to-pose-X), the sorted
        tuple of ``(name, value)`` pairs.  Empty for concrete
        recordings.
    plan
        The action sequence.  For muscle-memory skills the sequence
        may reference parameters by name; for concrete recordings the
        sequence is literal.
    deterministic
        If True, repeated execution from the same start state yields
        the same outcome (exact replay).  If False, the environment
        is stochastic — the solution should be treated as a strong
        prior that informs planning rather than a guaranteed outcome.
        **Even ARC-AGI-3 levels may be stochastic** (timing-dependent
        mechanics); robotics is almost always stochastic.  Callers
        must not assume determinism by default.
    monitor_level
        How tightly the executor watches for deviation during replay:

        * ``"low"`` — muscle-memory mode.  Execute the sequence with
          minimal checking; only hard failure signals (life lost,
          safety violation) interrupt.  Fastest; best for well-
          rehearsed skills in predictable contexts.
        * ``"moderate"`` — check preconditions at key waypoints
          declared by the plan.  Catches mid-sequence drift while
          still running most steps without scrutiny.
        * ``"full"`` — standard planner monitoring throughout.  Each
          step's expected effects are compared to actual observations
          and divergence triggers replanning.  Slowest but safest for
          new or stochastic contexts.
    n_uses / n_successes
        Empirical usage statistics; updated by PostMortem after each
        invocation.  See :meth:`success_rate`.
    scope
        Default ``LEVEL`` for game-specific recordings; ``GAME`` or
        ``GLOBAL`` for transferable procedural skills.  Critical for
        competition-mode purging — see
        :class:`cognitive_os.config.OperatingMode`.
    source
        Provenance tag.  Examples:
        ``"postmortem:recording"`` (captured from a successful episode),
        ``"rehearsal:training"`` (deliberately practised), or
        ``"user:teach"`` (taught by operator demonstration).
    """

    id:               str
    task_id:          str
    plan:             "Plan"
    task_parameters:  Tuple[Tuple[str, Any], ...] = ()
    recorded_at:      int                         = 0
    n_uses:           int                         = 0
    n_successes:      int                         = 0
    deterministic:    bool                        = True
    monitor_level:    str                         = "low"
    scope:            "Scope"                     = field(
        default_factory=lambda: Scope(kind=ScopeKind.LEVEL))
    source:           str                         = "postmortem:recording"
    rationale:        Optional[str]               = None

    @property
    def success_rate(self) -> float:
        """Empirical success rate; defaults to 0.5 (uninformative prior)
        until at least one use is recorded."""
        if self.n_uses == 0:
            return 0.5
        return self.n_successes / self.n_uses


# ===========================================================================
# PostMortem — retrospective analysis at episode end
# ===========================================================================


@dataclass
class PostMortem:
    """Retrospective summary produced at episode end, driving
    cross-episode learning.

    The PostMortemAnalyzer runs once per episode (including on
    successful completion).  Its output is the mechanism by which the
    system **accumulates knowledge** across episodes: lessons are
    written back into the hypothesis store at ``Scope(kind=GAME)`` or
    broader, synthesised Options are added to the action registry,
    and failure signatures feed the Mediator at the next impasse so
    the same dead ends are avoided.

    Attributes
    ----------
    episode_id
        Stable identifier for this run.
    final_status
        ``"success"``, ``"failure"``, ``"timeout"``, or ``"abandoned"``.
        Free-form to allow adapters to introduce finer categories
        (e.g. ``"failure:budget_exhausted"`` vs ``"failure:death"``).
    final_step
        Step at which the episode terminated.
    goal_outcomes
        Final status of each top-level goal.
    failed_plans
        Plans that were invalidated or failed during the episode.
        Used by the analyser to extract ConstraintClaims and
        StrategyClaim updates.
    contradicted_hypotheses
        IDs of hypotheses demoted during the episode.  Helps the
        Mediator avoid the same dead ends in future runs.
    surprises
        Surprise events recorded.  High-surprise episodes deserve
        Mediator consultation for abductive hypothesis generation
        before the next run.
    lessons
        Claims extracted by the analyser — typically new
        ``StrategyClaim``\\s and ``ConstraintClaim``\\s to be written
        back into the store at broader scope.
    options_synthesised
        IDs of Options newly created during this episode.
    mediator_usage / observer_usage
        Usage counters by question type, for budget review and
        retuning.
    total_steps
        Episode length.
    wall_time_seconds
        Real-time cost of the episode; populated by the runner.
    """

    episode_id:              str
    final_status:            str
    final_step:              int
    goal_outcomes:           Dict[str, "GoalStatus"]  = field(default_factory=dict)
    failed_plans:            List["Plan"]              = field(default_factory=list)
    contradicted_hypotheses: List[str]                 = field(default_factory=list)
    surprises:               List["SurpriseEvent"]     = field(default_factory=list)
    lessons:                 List[Claim]               = field(default_factory=list)
    options_synthesised:     List[str]                 = field(default_factory=list)
    mediator_usage:          Dict[str, int]            = field(default_factory=dict)
    observer_usage:          Dict[str, int]            = field(default_factory=dict)
    total_steps:             int = 0
    wall_time_seconds:       float = 0.0


# ===========================================================================
# Oracle protocols — Observer (visual) and Mediator (common-sense)
# ===========================================================================
#
# The engine is code-centric: hypothesis formation, credence updates,
# planning, and action selection are algorithms, not LLM calls.  But two
# capabilities are genuinely hard to code and genuinely useful to hand to
# an LLM:
#
#   1. OBSERVER — visual / perceptual oracle.  Given one or more raw
#      frames and a typed question ("are these two entities still
#      similar?", "classify this object"), return a typed answer.  Used
#      by the engine to resolve appearance-based RelationalClaims and
#      to bootstrap initial visual structure at episode start.
#
#   2. MEDIATOR — common-sense / world-knowledge oracle.  Given a
#      SYMBOLIC summary of the current WorldState and a typed question
#      ("what roles do these entities likely play?", "what strategy is
#      reasonable at this impasse?"), return structured Claims, Goals,
#      or Rules the engine should consider.  Used when miner-based
#      learning is inadequate: new entity types, impasses, surprises
#      without local explanation.
#
# Both oracles share the same discipline:
#
#   * Typed inputs and typed outputs — no free-form text in the
#     decision path (free text is allowed in `explanation` fields for
#     audit/logging only).
#   * Stateless across calls — the Oracle receives whatever context it
#     needs inside the query; the engine retains no Oracle state.
#   * Budgeted via ResourceTracker — LLMBudget splits the cap between
#     Observer and Mediator so that one cannot starve the other.
#   * Outputs are treated as evidence, not commands — a Mediator-proposed
#     Claim enters the HypothesisStore with an LLM-source prior credence
#     and must still be validated by subsequent observation.  A
#     Mediator hallucination fails to accumulate support and is pruned.
#
# Adapters implement the oracles however suits their domain: a VLM for
# Observer, a text LLM for Mediator, a hand-written rules engine, or a
# human-in-the-loop operator.  The engine does not care.
#
# ---------------------------------------------------------------------------
# Observer — visual questions about specific frames
# ---------------------------------------------------------------------------


class QuestionType(Enum):
    """What the engine is asking the Observer about.

    STILL_SIMILAR      — are two entities still visually similar?
                         (revalidating a cached RelationalClaim)
    CLASSIFY           — what category does this entity belong to?
    COMPARE            — compare two entities / regions and report salient
                         differences
    DESCRIBE           — free-form description (for logging / audit only;
                         output is NEVER in the decision path)
    STRUCTURE_MAP      — does source group map to target group?
    ENUMERATE_OBJECTS  — given a whole frame with no pre-existing entity
                         IDs, enumerate every visible object: estimated
                         position, coarse role, shape/colour.  Used at
                         level start to bootstrap the WorldState before
                         any symbolic miner has run.  ``targets`` is
                         expected to be empty; ``result`` is a list of
                         dicts (one per enumerated object).
    COMPARE_VISUAL_STATES — given two entity regions cropped from a frame,
                         determine whether they contain the same glyph /
                         pattern type and, if so, whether the patterns are
                         in the same orientation.  Used by
                         :class:`cognitive_os.oracle.VisualOrientationTrigger`
                         to detect orientation mismatches between a goal
                         entity and a reference indicator.  ``targets``
                         must be exactly two entity IDs.  ``result`` is an
                         object: ``{same_glyph: bool, orientation_match:
                         bool}``.  Dual-domain: robotics analogue is
                         checking whether a gripper orientation matches a
                         part orientation before attempting insertion.
    CHARACTERIZE_GAME  — given the first frame of a new level, ask the LLM
                         for a free-form + structured characterization of
                         the scenario: genre, likely win condition, typical
                         roles/characters, and mechanics.  ``targets`` is
                         empty; ``result`` is an object with open-string
                         fields (``narrative``, ``genre``, ``win_pattern``,
                         ``characters``, ``mechanics``).  Used by the
                         adapter-side game-characterization trigger to
                         seed decomposition priors without trusting the
                         LLM on pixel-level perception.  Domain-agnostic
                         in principle (any scenario-based domain); lives
                         adapter-side for now pending generalisation.
    """
    STILL_SIMILAR         = "still_similar"
    CLASSIFY              = "classify"
    COMPARE               = "compare"
    DESCRIBE              = "describe"
    STRUCTURE_MAP         = "structure_map"
    ENUMERATE_OBJECTS     = "enumerate_objects"
    COMPARE_VISUAL_STATES = "compare_visual_states"
    CHARACTERIZE_GAME     = "characterize_game"


@dataclass
class ObserverQuery:
    """A question the engine poses to the adapter's Observer.

    ``frames`` is a list of raw frame references; the adapter resolves
    them to domain-native form before handing them to its Observer
    implementation (VLM, classical vision pipeline, or human).
    """
    query_id:  str
    question:  QuestionType
    targets:   List[str]            # entity_ids the question is about
    frames:    List[Any]            # frames to inspect (adapter-specific type)
    claim_id:  Optional[str] = None # hypothesis this resolves, if any
    urgency:   float = 0.5
    context:   str = ""


@dataclass
class ObserverAnswer:
    """The adapter's answer to an ObserverQuery.

    ``result`` is a typed value: bool for yes/no questions, a string
    for classification, a structured dict for comparisons and mappings.
    The engine uses ``confidence`` to weight the answer as evidence; an
    Observer that is itself uncertain should pass a lower value.
    """
    query_id:    str
    result:      Any
    confidence:  float
    explanation: str = ""


# ---------------------------------------------------------------------------
# Mediator — common-sense guidance given a symbolic world summary
# ---------------------------------------------------------------------------


class MediatorQuestion(Enum):
    """What kind of common-sense guidance the engine is seeking.

    IDENTIFY_ROLES
        Given these entities and their observed properties, what role
        does each likely play?  (e.g. "this one looks like the agent,
        this like a wall, this like a life counter, this like a goal").
        Output: ``entity_roles`` dict plus PropertyClaims encoding the
        role assignments.
    SUGGEST_MECHANICS
        Given inferred roles, what mechanics are typical?  (e.g. "life
        counters usually decrement on agent damage; walls usually block
        movement").  Output: CausalClaims and TransitionClaims with
        LLM-source prior credence.
    SUGGEST_STRATEGY
        Given the current impasse (plan exhausted or repeatedly failing),
        what high-level strategy is reasonable?  Output: a small number
        of candidate GoalNodes representing alternative strategies to
        attempt, possibly with a suggested OR-node addition.
    WARN_HAZARDS
        Given observed entities / events, what hazards does common sense
        suggest?  Output: PropertyClaims (e.g. dangerous, lethal,
        time-pressure) and optional Rules (never-enter zones).  Extra
        critical in robotics; informational in game domains.
    PROPOSE_GOALS
        Given the current WorldState, what subgoals should the agent
        consider pursuing?  Used by the engine at cold start or when
        no adapter-seeded primary goal is known.  Output: GoalNodes
        and optionally top-level Goals.
    EXPLAIN_SURPRISE
        Given a SurpriseEvent that local miners can't explain, propose
        an abductive hypothesis.  Output: Claims (typically CausalClaims
        linking the surprise to some prior event).
    PROPOSE_TOOL
        Given the current impasse and the list of tools already
        available, propose a new tool (ToolProposal) the adapter could
        implement to unblock progress.  Output: one or more
        ToolProposals.  The adapter decides whether to implement the
        proposal and its adoption gate must pass before the tool
        enters the ToolRegistry.
    PROPOSE_GOAL_LINKAGE
        Given a seed goal whose leaf condition is an abstract resource
        predicate (typically an adapter-provided episode-win
        placeholder such as ``ResourceAbove("episode_won", 0.5)``),
        propose CausalClaim(s) whose ``effect`` matches that leaf and
        whose ``trigger`` is a concrete atomic condition (AtPosition,
        EntityInState, ...) that common sense says achieves the goal.
        These links are what allow ``derive_subgoals_from_causal`` to
        expand the opaque seed goal into actionable subgoals.  Used
        once per level whenever the engine observes an adapter-seeded
        resource goal with no committed causal explanation.  Output:
        one or more CausalClaims in ``proposed_claims`` — the handler
        commits them at credence derived from ``answer.confidence`` so
        they immediately seed subgoal derivation.
    RECOGNIZE_ENTITY
        Escape hatch for the harness-side VisualStore.  When a freshly-
        observed entity has no shared multi-key fingerprint with
        anything previously seen — i.e. cheap deterministic perception
        couldn't recognize it at any abstraction level (bitmap, shape,
        topology, scaled) — the harness ships the entity's bitmap
        rendering plus low-confidence candidates from the store and
        asks the VLM "is this similar to anything you've seen before,
        and what role might it play?"  Output: PropertyClaims attaching
        a role hypothesis (e.g. ``role=rotation_trigger``) at credence
        derived from ``answer.confidence``, plus optional
        EntityEquivalenceClaims linking the new entity to known ones
        at semantic-similarity tiers the harness couldn't reach.  See
        ``cognitive_os/visual_store.py`` for the harness side.
    PREDICT_OUTCOME
        Given the current symbolic world state and a proposed action (or
        short action sequence), predict the resulting state delta.  A
        forward-prediction question: "if I do X from here, what happens?"
        Output: one or more predicted Claims (typically a TransitionClaim
        or CausalClaim) describing the expected effect, at credence
        derived from ``answer.confidence``.  The engine uses the
        prediction to seed a closed-loop expectation that the subsequent
        observation either confirms or contradicts — it never trusts the
        prediction as fact.  Designed so a world-foundation model (e.g.
        NVIDIA Cosmos) or any learned forward model can plug in as the
        Mediator backend WITHOUT further engine change: the question is
        the stable contract, the backend is swappable.  Most relevant in
        robotics, where forward simulation of a manipulation is valuable
        and a learned dynamics model exists; informational in game
        domains.  See ``usecases/robotics/SPEC_milestone_0_diagnostic.md``
        §7 point 5.
    """
    IDENTIFY_ROLES       = "identify_roles"
    SUGGEST_MECHANICS    = "suggest_mechanics"
    SUGGEST_STRATEGY     = "suggest_strategy"
    WARN_HAZARDS         = "warn_hazards"
    PROPOSE_GOALS        = "propose_goals"
    EXPLAIN_SURPRISE     = "explain_surprise"
    PROPOSE_TOOL         = "propose_tool"
    PROPOSE_GOAL_LINKAGE = "propose_goal_linkage"
    RECOGNIZE_ENTITY     = "recognize_entity"
    PREDICT_OUTCOME      = "predict_outcome"


@dataclass
class WorldStateSummary:
    """A curated, symbolic digest of WorldState passed to the Mediator.

    The engine (not the adapter) is responsible for constructing this
    summary: filtering entities to those relevant to the question,
    trimming hypotheses to committed + currently-contested, truncating
    event history to the last N steps.  The adapter's job is then to
    serialise this structured summary into whatever text format the
    underlying LLM expects — keeping domain-specific formatting choices
    in the adapter.

    ``impasse_context`` is a short, structured reason for why the
    engine is consulting the Mediator (e.g. ``"plan exhausted for goal
    g1 after 3 failed attempts"``).  It seeds the LLM's framing; it is
    NOT a free-form prompt injected into the decision path.
    """

    step:                 int
    agent:                Dict[str, Any]
    entities:             Dict[str, EntityModel]      = field(default_factory=dict)
    committed_hypotheses: List["Hypothesis"]          = field(default_factory=list)
    contested_hypotheses: List["Hypothesis"]          = field(default_factory=list)
    active_goals:         List["Goal"]                = field(default_factory=list)
    active_rules:         List["Rule"]                = field(default_factory=list)
    recent_events:        List[Event]                 = field(default_factory=list)
    impasse_context:      Optional[str]               = None
    attempted_plans:      List["Plan"]                = field(default_factory=list)
    # What the adapter can already do — prevents the Mediator from
    # proposing tools that duplicate existing primitives.
    available_tools:      Optional[ToolRegistry]      = None
    # Previously-learned Options available as macro-actions.  Useful
    # context for SUGGEST_STRATEGY and PROPOSE_TOOL questions.
    available_options:    List["Option"]              = field(default_factory=list)


@dataclass
class MediatorQuery:
    """A common-sense question the engine poses to the Mediator.

    The Mediator is stateless across calls — everything it may need to
    answer is in ``world_summary`` plus the question-specific focus
    fields.  ``focus_entities`` / ``focus_goals`` / ``surprise`` narrow
    attention so the LLM is not forced to reason about the entire
    WorldState when only part of it is relevant.
    """
    query_id:       str
    question:       MediatorQuestion
    world_summary:  WorldStateSummary
    focus_entities: List[str]                = field(default_factory=list)
    focus_goals:    List[str]                = field(default_factory=list)
    surprise:       Optional[SurpriseEvent]  = None
    urgency:        float                    = 0.5
    context:        str                      = ""


@dataclass
class MediatorAnswer:
    """The Mediator's answer — structured, not free text.

    All substantive outputs are typed engine objects (Claims, GoalNodes,
    Rules, role assignments).  The engine adds proposed_claims to the
    HypothesisStore with LLM-source prior credence, inserts
    proposed_goals under the appropriate parent, and routes
    proposed_rules through the governance pipeline (which may require
    user approval for INVIOLABLE rules).

    ``entity_roles`` is a convenience channel for the common
    IDENTIFY_ROLES question — a dict mapping entity_id to a
    common-sense role name.  The engine converts these into
    PropertyClaims internally so they flow through the same
    evidence-tracking pipeline as any other claim.

    ``tool_invocations`` lets the Mediator request an immediate
    tool call as part of its answer — useful when common-sense
    guidance is "you should probe the grid structure here" and the
    Mediator wants to drive that probe directly rather than describe
    it.  The engine enqueues each invocation through the normal
    tool-dispatch pipeline.

    ``tool_proposals`` carries new ToolProposals returned in response
    to a PROPOSE_TOOL question.  Adoption is gated by the adapter.

    ``explanation`` is free text; it is logged for audit and may be
    shown to users, but it is never parsed for decision-making.
    """
    query_id:         str
    proposed_claims:  List[Claim]            = field(default_factory=list)
    proposed_goals:   List["GoalNode"]       = field(default_factory=list)
    proposed_rules:   List["Rule"]           = field(default_factory=list)
    entity_roles:     Dict[str, str]         = field(default_factory=dict)
    tool_invocations: List[ToolInvocation]   = field(default_factory=list)
    tool_proposals:   List[ToolProposal]     = field(default_factory=list)
    confidence:       float                  = 0.5
    explanation:      str                    = ""


# ===========================================================================
# WorldState — the engine's current model
# ===========================================================================


# ===========================================================================
# Goal-progress tracking substrate (SPEC_goal_progress_tracking.md)
# ===========================================================================
#
# These types record per-goal progress trajectories and the events
# that mark transitions in match state.  The substrate distinguishes
# "I have never achieved this" from "I had this and lost it" — a
# distinction the closed-loop substrate's per-action outcomes alone
# don't carry.


class ProgressEventKind(Enum):
    """Kinds of progress events emitted when a goal's snapshot
    diff shows a transition.

    Per-dimension events are the load-bearing signal; goal-level
    events are derived from them and useful for consumers that
    want to react to the whole-goal transition without reading
    the per-dimension stream.
    """
    DIMENSION_ACHIEVED  = "dimension_achieved"
    DIMENSION_REGRESSED = "dimension_regressed"
    GOAL_ACHIEVED       = "goal_achieved"
    GOAL_REGRESSED      = "goal_regressed"


@dataclass(frozen=True)
class GoalProgressSnapshot:
    """A point-in-time record of which dimensions of a goal are
    currently matched and which are open.

    Computed by the progress-tracking substrate at a fixed point
    in the per-turn cycle (after outcome recording, before the
    next selection pass).  Diffs between consecutive snapshots
    drive :class:`ProgressEvent` emission.
    """
    goal_id:             str
    dimensions:          frozenset
    dimensions_matched:  frozenset
    dimensions_open:     frozenset
    fraction_matched:    float
    recorded_at_turn:    int


@dataclass(frozen=True)
class ProgressEvent:
    """A transition in a goal's progress between consecutive
    snapshots.

    Emitted by the progress-tracking substrate's diff pass when
    a dimension's match state changes.  Consumers (the goal
    selector for recovery-priority bumps, the dialogic outbound
    surface for human-readable progress reports, future
    reflection detectors) read the events from per-goal history.
    """
    goal_id:    str
    dimension:  str
    kind:       ProgressEventKind
    turn:       int
    payload:    Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalProgressHistory:
    """Per-goal bounded record of recent progress snapshots and
    the events derived from snapshot diffs.

    Lives on each :class:`Goal`.  Default-constructed empty;
    populated by the substrate's per-turn snapshot pass.

    The default cap of eight snapshots is suitable for typical
    pursuit horizons; goals that accumulate longer histories
    can override per-instance.
    """
    snapshots:  List[GoalProgressSnapshot] = field(default_factory=list)
    events:     List[ProgressEvent]         = field(default_factory=list)
    max_size:   int                         = 8
    # Recovery-priority bump applied by the goal selector when a
    # regression event has fired recently.  Decays each turn until
    # zero unless new regressions arrive.  Tracked separately from
    # ``goal.priority`` so the bump is reversible without touching
    # the underlying priority.
    recovery_bump:  float = 0.0


# ===========================================================================
# Closed-loop action substrate (SPEC_closed_loop_action_substrate.md, Phase 1)
# ===========================================================================
#
# These types are the foundation of the closed-loop action evaluation
# substrate.  Phase 1 introduces the data shapes only; behaviour lands
# in subsequent phases.  No engine component reads or writes these
# fields yet — adapters and the engine ignore them until Phase 2 wires
# the prediction-emission path.


class MatchKind(Enum):
    """Result of comparing one predicted assertion to its observed
    counterpart.

    Used per-assertion within an :class:`Outcome` so that compound
    predictions with several assertions can record different fates
    for each one — a prediction can be partially confirmed, partially
    contradicted, partially ambiguous in the same turn.

    The four values cover the cases the substrate actually needs to
    distinguish for downstream credence and goal-tree updates:

    * ``CONFIRMED`` — observation unambiguously matches the
      prediction.  Drives upward credence updates on supporting
      beliefs and progress markers on supporting goals.
    * ``CONTRADICTED`` — observation unambiguously falsifies the
      prediction.  Drives sharp downward credence updates and
      flags decompositions whose chosen path produced this outcome.
    * ``AMBIGUOUS`` — evidence is consistent with both the prediction
      and its negation.  No sharp update; small slow-path nudge.
    * ``UNOBSERVABLE`` — the predicted property could not be
      evaluated from the post-action observation (sensor missing,
      occluded, etc.).  No update.

    See SPEC_closed_loop_action_substrate.md §"Outcome".
    """
    CONFIRMED    = "confirmed"
    CONTRADICTED = "contradicted"
    AMBIGUOUS    = "ambiguous"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class PredictedAssertion:
    """One structured assertion within a :class:`Prediction`.

    A prediction may carry several assertions, each describing a
    different aspect of the expected post-action state.  The
    substrate compares each assertion independently against the
    actual observation and records a :class:`MatchKind` per
    assertion in the resulting :class:`Outcome`.

    ``condition`` is a domain-neutral :class:`Condition` whose
    ``evaluate`` is run against the post-action world state.
    ``expected_value`` is what the prediction asserts that
    evaluation should return — typically ``True``, but ``False``
    is valid for "predicting that this condition will *not* hold"
    cases.

    ``rationale`` is a short human-readable note explaining why
    this assertion was made; it appears in audit traces.
    """
    condition:      Condition
    expected_value: bool   = True
    rationale:      str    = ""


@dataclass(frozen=True)
class Prediction:
    """The system's structured expectation of post-action state.

    A prediction is constructed before an action is executed.  The
    component proposing the action — typically the goal selector
    working with the pathfinder — populates the prediction from the
    beliefs and goals that motivated the action choice.  After the
    action runs, the substrate compares the prediction to the
    observed result and records an :class:`Outcome`.

    ``predicted_assertions`` may be empty for actions whose outcome
    is intentionally unconstrained (e.g. exploratory probes).  An
    empty prediction still records into the trajectory; reflection
    can detect "many turns of unconstrained actions" as its own
    pattern.

    ``supporting_belief_ids`` and ``active_goal_id`` give the
    substrate the provenance it needs to apply belief updates and
    goal-progress signals when the outcome arrives.

    See SPEC_closed_loop_action_substrate.md §"Prediction".
    """
    action_id:             str
    predicted_assertions:  Tuple[PredictedAssertion, ...] = ()
    supporting_belief_ids: Tuple[str, ...]                = ()
    active_goal_id:        Optional[str]                  = None
    confidence:            float                          = 0.5
    emitted_at_turn:       int                            = 0


@dataclass(frozen=True)
class Outcome:
    """The result of comparing a :class:`Prediction` to observed state.

    Constructed by the substrate after each action.  Contains:

    * The original prediction (so consumers can see what was
      expected without joining against another store).
    * The per-assertion match results — one :class:`MatchKind`
      entry per :class:`PredictedAssertion` in the prediction, in
      the same order.
    * An overall match summary derived from the per-assertion
      results.  The reduction rule is: any CONTRADICTED makes the
      overall CONTRADICTED; otherwise any CONFIRMED with no
      CONTRADICTED makes it CONFIRMED; otherwise the most common
      remaining value (defaulting to UNOBSERVABLE for the empty
      prediction).
    * The turn at which this outcome was recorded.

    Outcomes are stored in the :class:`Trajectory` and indexed for
    reflection-time queries.

    See SPEC_closed_loop_action_substrate.md §"Outcome".
    """
    prediction:           Prediction
    per_assertion_match:  Tuple[MatchKind, ...]
    overall_match:        MatchKind
    recorded_at_turn:     int


@dataclass
class Trajectory:
    """Bounded record of recent action-prediction-observation triples.

    Lives on the world-state object alongside the hypothesis store
    and goal forest.  Other components read it through dedicated
    query helpers (added in Phase 2); no component should mutate
    the ``outcomes`` list directly outside the substrate's
    own append-and-evict logic.

    The default cap of sixty-four outcomes is suitable for ARC-AGI
    scale; robotics and other higher-frequency domains can override
    via adapter configuration.

    Indices ``index_by_goal``, ``index_by_action``, and
    ``index_by_cell`` are lazily populated query caches; the canonical
    record is the ``outcomes`` deque.

    See SPEC_closed_loop_action_substrate.md §"Trajectory".
    """
    outcomes:         List[Outcome]                     = field(default_factory=list)
    max_size:         int                               = 64
    # Lazy reverse indices for query helpers landing in Phase 2.
    # Empty until populated; cleared when outcomes are evicted.
    index_by_goal:    Dict[str, List[int]]              = field(default_factory=dict)
    index_by_action:  Dict[str, List[int]]              = field(default_factory=dict)
    index_by_cell:    Dict[Tuple[int, int], List[int]]  = field(default_factory=dict)


@dataclass(frozen=True)
class ReflectionResult:
    """Output of the per-turn reflection pass.

    Reflection scans the recent :class:`Trajectory` and emits
    structured observations about the agent's own behaviour.
    Three categories:

    * ``meta_claims`` — new entries to propose into the hypothesis
      store under a ``reflection`` source label.  Examples: "the
      claim that visiting cell X flips palette has been
      contradicted by recent observations."
    * ``goal_signals`` — advisory annotations for goals.  Each
      signal is a (goal_id, signal_kind, payload) tuple.  Common
      signals: ``stalled``, ``cycling``, ``unachievable``.
    * ``cycle_detections`` — explicit records of behavioural
      cycles, listed separately from the more general
      ``goal_signals`` because cycle detection is the most common
      and load-bearing reflection output.

    See SPEC_closed_loop_action_substrate.md §"ReflectionResult".
    """
    meta_claims:        Tuple[Any, ...]                          = ()
    goal_signals:       Tuple[Tuple[str, str, Any], ...]         = ()
    cycle_detections:   Tuple[Tuple[Tuple[Any, ...], int], ...]  = ()
    recorded_at_turn:   int                                       = 0


@dataclass
class WorldState:
    """The agent's complete current model of the environment.

    Updated each step by the engine from the incoming :class:`Observation`.
    The planner, explorer, and miners all read from this structure;
    nothing else in the engine retains state (the miners are stateless
    functions over ``observation_history`` and current hypotheses).

    ``observation_history`` is retained in full for evidence-gathering
    miners (some patterns are only detectable over long windows);
    adapters may configure an eviction policy to cap memory in very long
    episodes.
    """
    step:                 int                         = 0
    agent:                Dict[str, Any]              = field(default_factory=dict)
    entities:             Dict[str, EntityModel]      = field(default_factory=dict)
    hypotheses:           Dict[str, Hypothesis]       = field(default_factory=dict)
    rules:                Dict[str, Rule]             = field(default_factory=dict)
    goal_forest:          GoalForest                  = field(default_factory=GoalForest)
    observation_history:  List[Observation]           = field(default_factory=list)
    # Tools the adapter exposes (populated at episode start) and
    # Options learned across prior episodes at scope GAME or broader.
    tool_registry:        ToolRegistry                = field(default_factory=ToolRegistry)
    options:              Dict[str, Option]           = field(default_factory=dict)
    # Pending async tool invocations (invocation_id → invocation).  The
    # runtime tracks in-flight calls here so the planner can reason
    # about latency while awaiting results.
    pending_tool_calls:   Dict[str, ToolInvocation]   = field(default_factory=dict)
    # Cached solutions / rehearsed procedures.  Loaded at episode start
    # in TRAINING mode (all scopes); in COMPETITION mode only those
    # with scope GAME or GLOBAL are loaded (LEVEL-scoped solutions are
    # purged so the agent must re-solve each level from first
    # principles).  See config.OperatingMode.
    cached_solutions:     Dict[str, "CachedSolution"] = field(default_factory=dict)
    # Monotonic counter used by the hypothesis_store to allocate unique
    # hypothesis IDs.  Starts at 0 and never rolls back, so IDs of
    # pruned hypotheses are never reused — this is important because
    # refinement lattice references (parent_id / child_ids) would
    # otherwise dangle to a different claim after pruning+reuse.
    _next_hypothesis_id:  int = 0
    # The config is held here so subsystems can read thresholds without
    # plumbing it through every call signature.  ``Any`` to avoid a
    # circular import with the config module at typing time.
    config:               Any = None
    # Structural diff of the two most recent frames, computed once per
    # step by the engine runner after observation ingest.  Consumed by
    # miners (ActionEffectMiner, surprise detectors), conditions, and
    # future entity-identity trackers.  ``None`` means "delta
    # unavailable" (first step, or non-grid raw_frame); a FrameDelta
    # with ``is_empty`` means "frames identical" — distinct signals.
    # Typed as ``Any`` to avoid import cycles; see
    # :class:`cognitive_os.frame_diff.FrameDelta`.
    last_frame_delta:     Any = None
    # Closed-loop action substrate (SPEC_closed_loop_action_substrate.md,
    # Phase 1).  Bounded record of recent action-prediction-observation
    # triples.  Phase 1 introduces the field as a default-empty
    # Trajectory; Phase 2 wires the action-emission path to populate it;
    # subsequent phases consume it for sharp belief updates, living
    # goal-tree re-decomposition, and reflection over recent behaviour.
    # No component reads or writes this field in Phase 1 — it exists
    # to establish the shape against which later phases land.
    trajectory:           Trajectory = field(default_factory=Trajectory)

    # --- convenience queries ---

    def committed_hypotheses(self) -> List[Hypothesis]:
        """Return hypotheses whose credence is at or above commit threshold.

        Requires ``self.config`` to be an :class:`EngineConfig`; returns an
        empty list if config is not set (defensive for early construction
        before config is attached).
        """
        if self.config is None:
            return []
        cfg = self.config.credence
        return [h for h in self.hypotheses.values() if h.credence.is_committed(cfg)]

    def active_goal(self) -> Optional[Goal]:
        """Return the currently-active goal, if any."""
        gid = self.goal_forest.active_goal_id
        return self.goal_forest.goals.get(gid) if gid else None

    def hypothesis_by_claim_canonical(self, canonical_key: tuple) -> List[Hypothesis]:
        """Return all hypotheses whose claim shares the given canonical key.

        Used by the HypothesisStore's dedup logic to find competitors
        when a new claim is proposed.  Linear scan; acceptable for
        current store sizes.  A future optimisation may index by
        canonical key if stores grow past a few thousand entries.
        """
        return [h for h in self.hypotheses.values()
                if h.claim.canonical_key() == canonical_key]
