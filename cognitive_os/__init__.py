"""Cognitive engine — domain-agnostic symbolic reasoning substrate.

This package implements a generic cognitive engine that operates purely on
symbolic observations, hypotheses, goals, and plans.  It has no knowledge of
any specific domain (ARC-AGI-3, robotics, etc.) — domain-specific perception
and action translation live in Adapters (see `cognitive_os.adapters`).

Design principle (standing directive):
    Game-specific or task-specific solutions MUST NOT be injected into this
    package.  Every capability needed to solve a particular problem must be
    expressed as a generalisation of the substrate — a new miner, a new
    claim type, a new planner heuristic — that would remain useful for an
    unrelated future domain.  The engine is a learning system: knowledge
    accumulated from one episode should benefit future episodes across the
    same and other domains.

LLM seams:
    Two typed oracle protocols allow an LLM (or any equivalent oracle) to
    be consulted at bounded, audited points in the loop:

    * OBSERVER  — visual / perceptual queries (frame + typed question →
                  typed answer).  For appearance-based questions the
                  engine cannot answer symbolically.
    * MEDIATOR  — common-sense / world-knowledge queries (symbolic
                  WorldStateSummary + typed question → structured
                  Claims/Goals/Rules, optionally including
                  ToolProposals or ToolInvocations).  Used at impasses,
                  cold starts, unexplained surprises, hazard assessment,
                  and tool-creation requests.

    Both oracles produce typed outputs; free-form text is confined to
    `explanation` fields that never enter the decision path.  Budgets
    are enforced via `LLMBudget` and tracked by the ResourceTracker.

Tool system:
    The adapter exposes domain primitives (grid BFS, symmetry detection,
    motion planning, etc.) through a generic ToolRegistry.  Miners,
    Planner, Explorer, and Mediator invoke tools by name via
    ToolInvocation; the adapter dispatches and returns ToolResults.
    Sync and async modes are both supported (signature carries
    `is_async` and `typical_latency_ms`).  Learned Options — macro-
    actions synthesised from recurring successful plan fragments — are
    added to the action space across episodes, so the system's toolkit
    grows over time.

Phase 1 (current): data types only — Observation, Hypothesis, Goal, Plan,
Rule, WorldState, the Claim/Condition/Credence machinery they reference,
and the Observer + Mediator query/answer protocols.  No behaviour beyond
canonicalisation and trivial accessors.
"""

from .config import (
    CredenceConfig,
    SourcePriors,
    ExplorerConfig,
    LLMBudget,
    PlannerConfig,
    ActionProbeConfig,
    EngineConfig,
    OperatingMode,
)
from .conditions import (
    Condition,
    AlwaysTrue,
    AtPosition,
    InsideBBox,
    EntityInState,
    ResourceAbove,
    ResourceBelow,
    EntityProbed,
    ActionTried,
    ActionJustTaken,
    FrameChangedPattern,
    RegionMotion,
    EntitiesVisuallyMatch,
    EntitiesEquivalent,
    AgentAtEntityClass,
    AgentAtCellRelativeToEntity,
    AgentTeleportedByOffset,
    ResourceRestored,
    Conjunction,
    Disjunction,
    Negation,
)
from .claims import (
    Claim,
    PropertyClaim,
    CausalClaim,
    TransitionClaim,
    RelationalClaim,
    ConstraintClaim,
    StructureMappingClaim,
    StrategyClaim,
    ControlledActorClaim,
    ActorTransitionClaim,
    MotionModelClaim,
    GoalRegressionClaim,
    BitmapRoleClaim,
    RegionPaletteClaim,
    RelationType,
    MappingKind,
    RelationPattern,
    Asymmetry,
)
from .credence import (
    Credence,
    update_on_support,
    update_on_contradict,
    apply_decay,
)
from . import hypothesis_store
from .hypothesis_store import (
    propose,
    update_credence_from_events,
    apply_staleness_decay_all,
    prune_abandoned,
    committed,
    contested_groups,
    by_canonical_key,
    by_full_key,
    event_evidence_for_claim,
)
from . import refinement
from .refinement import (
    specialize_on_contradiction,
    detect_generalization_candidates,
    link_parent_child,
    prune_subsumed_children,
)
from . import resource_aware_selector
from .resource_aware_selector import (
    ResourceKey,
    ResourceProfile,
    ResourceState,
    ResourceAwareSelection,
    select_resource_aware,
)
from . import decomposer
from .decomposer import (
    DecompositionTarget,
    DecomposedGoal,
    propose_decomposition_goals,
)
from . import goal_forest
from .goal_forest import (
    add_goal,
    select_active_goal,
    derive_subgoals_from_causal,
    derive_subgoals_for_entities_equivalent,
    derive_state_destabilizing_cells_for_achieved_goals,
    derive_action_probe_goals,
    refresh_atposition_tolerances,
    atomic_leaves,
    is_achieved,
    mark_status,
    refresh_status,
    detect_conflicts,
)
from . import planner
from .planner import (
    compute_plan,
    apply_rules_filter,
    advisory_penalty_for_action,
)
from . import explorer
from .explorer import (
    claim_coverage,
    info_gain,
    motion_model_info_gain,
    propose_curiosity_goals,
    choose_exploration_action,
    choose_exploration_target_cell,
)
from . import adapters
from .adapters import Adapter
from . import miners
from .miners import (
    Miner,
    PropertyObservedMiner,
    TransitionMiner,
    ActionEffectMiner,
    RegionMotionMiner,
    ControlledEntityMiner,
    SelfLocalizationMiner,
    ActorTransitionMiner,
    MotionModelMiner,
    ContactMiner,
    FutilePatternMiner,
    SurpriseMiner,
    default_miners,
)
from . import postmortem
from .postmortem import (
    run_post_mortem,
    extract_lessons,
    OptionSynthesiser,
)
from . import persistence
from .persistence import load_committed_knowledge, save_committed_knowledge
from . import validity_scope
from .validity_scope import (
    BoundaryKind,
    RecordStatus,
    ScopedRecord,
    ValidityScope,
    current_scope,
    persist,
    read_authoritative,
    read_hint,
    set_current_scope,
    validate_at_scope_boundary,
)
from . import episode_runner
from .episode_runner import run_episode
from . import oracle
from .oracle import (
    OracleTrigger,
    InitialFrameScanTrigger,
    EpisodeGoalLinkageTrigger,
    VisualOrientationTrigger,
    VisualPatternChangedTrigger,
    default_triggers,
    build_world_summary,
)
from . import telemetry_schema
from .telemetry_schema import (
    SCHEMA_VERSION,
    TelemetryEnvelope,
    envelope_to_dict,
    envelope_from_dict,
    mint_session_id,
    register_event,
    event_class_for,
    registered_event_types,
    payload_to_dict,
    payload_from_dict,
)
from . import telemetry
from .telemetry import (
    TelemetrySink,
    NullSink,
    NDJSONSink,
    WebSocketSink,
    read_ndjson,
    decode_payload,
)
from .tools import (
    ToolSignature,
    ToolRegistry,
    ToolInvocation,
    ToolResult,
    ToolProposal,
    ToolCallback,
)
from .frame_diff import (
    CellChange,
    DeltaRegion,
    FrameDelta,
    FrameShapeMismatch,
    compute_frame_delta,
)
from . import probes
from .probes import (
    PROBE_PROPOSAL_SCHEMA,
    PROBE_RESULT_SCHEMA,
    ProbeExecutor,
    ProbeProposal,
    ProbeValidationError,
    RISK_CLASS_DESTRUCTIVE,
    RISK_CLASS_IRREVERSIBLE,
    RISK_CLASS_SAFE,
    SimProbeExecutor,
    make_probe_result,
    validate_probe_proposal,
)
from . import world_model
from . import causal_attribution
from . import causal_providers
from .types import (
    # Scope / time
    Scope,
    ScopeKind,
    # Events
    Event,
    AgentMoved,
    AgentDied,
    ResourceChanged,
    EntityStateChanged,
    GoalConditionMet,
    EntityAppeared,
    EntityDisappeared,
    EntityVisualPatternChanged,
    ContactEvent,
    SurpriseEvent,
    # Durative-skill events (embodied / async domains)
    SkillStarted,
    SkillProgress,
    SkillSucceeded,
    SkillFailed,
    SkillPreempted,
    # Observation
    Observation,
    # Entities
    EntityModel,
    # Hypothesis
    Hypothesis,
    # Rule
    Rule,
    Principal,
    PrincipalKind,
    Violability,
    RuleConstraint,
    ConstraintKind,
    # Goal
    Goal,
    GoalNode,
    NodeType,
    Ordering,
    GoalStatus,
    GoalForest,
    GoalConflict,
    ConflictType,
    ResolutionPolicy,
    # Dependency expressions (SPEC_goal_dependencies.md)
    DepExpr,
    DepRef,
    DepAll,
    DepAny,
    # Action / Plan
    Action,
    PlannedAction,
    Plan,
    PlanStatus,
    # Learned macro-actions, cached procedures, and post-episode analysis
    Option,
    CachedSolution,
    PostMortem,
    # Observer (visual oracle)
    ObserverQuery,
    ObserverAnswer,
    QuestionType,
    # Mediator (common-sense oracle)
    MediatorQuery,
    MediatorAnswer,
    MediatorQuestion,
    WorldStateSummary,
    # Closed-loop action substrate (Phase 1)
    MatchKind,
    PredictedAssertion,
    Prediction,
    Outcome,
    Trajectory,
    ReflectionResult,
    # Goal-progress tracking substrate
    ProgressEventKind,
    GoalProgressSnapshot,
    ProgressEvent,
    GoalProgressHistory,
    # World
    WorldState,
)
from . import closed_loop
from .closed_loop import (
    compare_assertion,
    reduce_overall_match,
    record_outcome,
    outcomes_for_goal,
    outcomes_for_action,
    outcomes_for_cell,
    update_credence_from_outcome,
)
from . import reflection
from .reflection import (
    detect_cycles,
    detect_stalls,
    run_reflection,
)
from . import progress
from .progress import (
    compute_snapshot,
    diff_snapshots,
    record_progress,
    decay_recovery_bumps,
    effective_recovery_bump,
    mine_regression_claims,
)

__all__ = [
    # config
    "CredenceConfig", "SourcePriors", "ExplorerConfig",
    "LLMBudget", "PlannerConfig", "ActionProbeConfig", "EngineConfig",
    "OperatingMode",
    # conditions
    "Condition", "AlwaysTrue", "AtPosition", "InsideBBox", "EntityInState",
    "ResourceAbove", "ResourceBelow", "EntityProbed", "ActionTried",
    "ActionJustTaken", "FrameChangedPattern", "RegionMotion",
    "EntitiesVisuallyMatch", "EntitiesEquivalent",
    "Conjunction", "Disjunction", "Negation",
    # claims
    "Claim", "PropertyClaim", "CausalClaim", "TransitionClaim",
    "RelationalClaim", "ConstraintClaim", "StructureMappingClaim",
    "StrategyClaim", "ControlledActorClaim", "ActorTransitionClaim",
    "MotionModelClaim", "GoalRegressionClaim",
    "BitmapRoleClaim", "RegionPaletteClaim",
    "RelationType", "MappingKind",
    "RelationPattern", "Asymmetry",
    # credence
    "Credence", "update_on_support", "update_on_contradict", "apply_decay",
    # hypothesis store (Phase 2)
    "hypothesis_store",
    "propose", "update_credence_from_events", "apply_staleness_decay_all",
    "prune_abandoned", "committed", "contested_groups",
    "by_canonical_key", "by_full_key", "event_evidence_for_claim",
    # refinement (Phase 2)
    "refinement",
    "specialize_on_contradiction", "detect_generalization_candidates",
    "link_parent_child", "prune_subsumed_children",
    # goal forest (Phase 3)
    "goal_forest",
    "add_goal", "select_active_goal", "derive_subgoals_from_causal",
    "derive_subgoals_for_entities_equivalent",
    "derive_state_destabilizing_cells_for_achieved_goals",
    "derive_action_probe_goals", "refresh_atposition_tolerances",
    "atomic_leaves", "is_achieved", "mark_status", "refresh_status",
    "detect_conflicts",
    # planner (Phase 3)
    "planner",
    "compute_plan", "apply_rules_filter", "advisory_penalty_for_action",
    # explorer (Phase 3)
    "explorer",
    "claim_coverage", "info_gain", "motion_model_info_gain",
    "propose_curiosity_goals", "choose_exploration_action",
    # adapter protocol (Phase 4)
    "adapters", "Adapter",
    # miners (Phase 4)
    "miners", "Miner",
    "PropertyObservedMiner", "TransitionMiner", "ActionEffectMiner",
    "RegionMotionMiner", "ControlledEntityMiner", "SelfLocalizationMiner",
    "ActorTransitionMiner", "MotionModelMiner", "ContactMiner",
    "FutilePatternMiner", "SurpriseMiner",
    "default_miners",
    # postmortem (Phase 4)
    "postmortem", "run_post_mortem", "extract_lessons", "OptionSynthesiser",
    # persistence (Phase 5c)
    "persistence",
    "load_committed_knowledge", "save_committed_knowledge",
    # episode runner (Phase 4)
    "episode_runner", "run_episode",
    # oracle triggers (Phase 5b)
    "oracle",
    "OracleTrigger", "InitialFrameScanTrigger",
    "EpisodeGoalLinkageTrigger",
    "VisualOrientationTrigger", "VisualPatternChangedTrigger",
    "default_triggers", "build_world_summary",
    # telemetry (GUI PR 1)
    "telemetry_schema", "telemetry",
    "SCHEMA_VERSION", "TelemetryEnvelope",
    "envelope_to_dict", "envelope_from_dict", "mint_session_id",
    "register_event", "event_class_for", "registered_event_types",
    "payload_to_dict", "payload_from_dict",
    "TelemetrySink", "NullSink", "NDJSONSink", "WebSocketSink",
    "read_ndjson", "decode_payload",
    # tools
    "ToolSignature", "ToolRegistry", "ToolInvocation", "ToolResult",
    "ToolProposal", "ToolCallback",
    # frame-diff (core capability)
    "CellChange", "DeltaRegion", "FrameDelta", "FrameShapeMismatch",
    "compute_frame_delta",
    # probes
    "probes", "PROBE_PROPOSAL_SCHEMA", "PROBE_RESULT_SCHEMA",
    "ProbeExecutor", "ProbeProposal", "ProbeValidationError", "SimProbeExecutor",
    "make_probe_result", "validate_probe_proposal",
    "RISK_CLASS_SAFE", "RISK_CLASS_DESTRUCTIVE", "RISK_CLASS_IRREVERSIBLE",
    # object world model
    "world_model",
    # causal attribution & avoidance (tier-2 loop + provider/sandbox layer)
    "causal_attribution", "causal_providers",
    # scope
    "Scope", "ScopeKind",
    # events
    "Event", "AgentMoved", "AgentDied", "ResourceChanged",
    "EntityStateChanged", "GoalConditionMet", "EntityAppeared",
    "EntityDisappeared", "EntityVisualPatternChanged",
    "ContactEvent", "SurpriseEvent",
    "SkillStarted", "SkillProgress", "SkillSucceeded",
    "SkillFailed", "SkillPreempted",
    # observation
    "Observation",
    # entities / hypothesis
    "EntityModel", "Hypothesis",
    # rule
    "Rule", "Principal", "PrincipalKind", "Violability",
    "RuleConstraint", "ConstraintKind",
    # goal
    "Goal", "GoalNode", "NodeType", "Ordering", "GoalStatus",
    "GoalForest", "GoalConflict", "ConflictType", "ResolutionPolicy",
    # action / plan
    "Action", "PlannedAction", "Plan", "PlanStatus",
    # learned macro-actions, cached procedures, and post-episode analysis
    "Option", "CachedSolution", "PostMortem",
    # observer (visual oracle)
    "ObserverQuery", "ObserverAnswer", "QuestionType",
    # mediator (common-sense oracle)
    "MediatorQuery", "MediatorAnswer", "MediatorQuestion", "WorldStateSummary",
    # closed-loop action substrate (Phase 1)
    "MatchKind", "PredictedAssertion", "Prediction", "Outcome",
    "Trajectory", "ReflectionResult",
    # closed-loop action substrate (Phase 2)
    "closed_loop",
    "compare_assertion", "reduce_overall_match", "record_outcome",
    "outcomes_for_goal", "outcomes_for_action", "outcomes_for_cell",
    # closed-loop action substrate (Phase 3)
    "update_credence_from_outcome",
    # closed-loop action substrate (Phase 5: reflection)
    "reflection",
    "detect_cycles", "detect_stalls", "run_reflection",
    # goal-progress tracking substrate
    "progress",
    "ProgressEventKind", "GoalProgressSnapshot", "ProgressEvent",
    "GoalProgressHistory",
    "compute_snapshot", "diff_snapshots", "record_progress",
    "decay_recovery_bumps", "effective_recovery_bump",
    "mine_regression_claims",
    # world
    "WorldState",
]
