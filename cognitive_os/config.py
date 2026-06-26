"""Engine configuration — all tunable parameters in one place.

Nothing in the engine algorithm code should hardcode a threshold, rate, or
budget.  All such values live here, composed into an `EngineConfig` that is
passed down to subsystems.

The `curiosity_level` parameter is the single most important knob for
tuning exploration behaviour.  It modulates several derived parameters so
that a user can dial one number rather than coordinating five.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Operating mode
# ---------------------------------------------------------------------------


class OperatingMode(Enum):
    """Top-level operating mode of the engine — determines which
    persisted state is loaded at episode start.

    All modes load **knowledge**: hypotheses at scope ``GAME``/``GLOBAL``,
    learned Options, and cached solutions at scope ``GAME``/``GLOBAL``
    (e.g. muscle-memory skills in robotics, cross-game strategies).
    This is the accumulated competence of the agent and is legitimately
    usable in competition.

    Only ``TRAINING`` and ``DEBUG`` load **level-specific solutions**:
    CachedSolutions at scope ``LEVEL``.  These are concrete recordings
    of how a specific (game, level) was solved; loading them allows
    the agent to skip past already-solved levels in order to reach a
    target level.  In competition the agent must solve each level
    from first principles — replaying a stored recording of that same
    level would be cheating.

    TRAINING
        Full access.  Load all persisted hypotheses, Options, and
        CachedSolutions (all scopes).  Used for development,
        multi-level training runs, and rapid iteration toward a target
        level via solution replay.
    COMPETITION
        Knowledge only.  LEVEL-scoped CachedSolutions are purged before
        the episode begins.  GAME-scoped and GLOBAL-scoped items are
        loaded normally.  The agent brings all it has *learned* but
        none of the specific answers it has *memorised*.
    EVALUATION
        Same purging rules as COMPETITION but with verbose logging
        turned on for benchmark analysis.  Distinguished from
        COMPETITION only so that telemetry can be filtered.
    DEBUG
        Same load rules as TRAINING but with verbose logging and
        extra diagnostics.  Used for reproducing failures.
    """

    TRAINING    = "training"
    COMPETITION = "competition"
    EVALUATION  = "evaluation"
    DEBUG       = "debug"

    def loads_level_solutions(self) -> bool:
        """Whether this mode loads LEVEL-scoped CachedSolutions at
        episode start.  Used at the single load gate in the runtime."""
        return self in (OperatingMode.TRAINING, OperatingMode.DEBUG)

    def verbose_logging(self) -> bool:
        """Whether this mode enables extra diagnostic output."""
        return self in (OperatingMode.DEBUG, OperatingMode.EVALUATION)


# ---------------------------------------------------------------------------
# Credence dynamics
# ---------------------------------------------------------------------------


@dataclass
class CredenceConfig:
    """Thresholds and rates for the Credence update rule.

    A hypothesis with ``point >= commit_threshold`` is treated as committed
    by the planner.  One with ``point <= abandon_threshold`` is pruned.
    """

    commit_threshold: float = 0.85
    abandon_threshold: float = 0.15
    learning_rate: float = 0.15
    decay_per_step: float = 0.001
    staleness_window: int = 50


# ---------------------------------------------------------------------------
# Hypothesis source priors
# ---------------------------------------------------------------------------


@dataclass
class SourcePriors:
    """Initial credence assigned when a hypothesis is proposed by a source.

    Tier semantics (must agree with hypothesis_store._prior_for_source,
    which is the fallback path used when ws.config is absent):

    - ``user`` / ``adapter`` / ``observer`` — high-trust evidence
      channels.
    - ``mediator`` / ``kb`` — prior knowledge (narrow LLM inference and
      cross-session knowledge base).  Same tier (0.65), reflecting
      durable but not first-hand observation.  KB-loaded claims at
      this prior clear the planner's 0.60 credence floor for
      ``active_position_specific_motion`` / ``active_wall_overrides``,
      so persisted walls and portals become visible to the engine
      planner without requiring a single in-session re-confirmation.
    - ``miner`` — symbolic-pattern derivation from accumulated
      observations.
    - ``oracle`` — undifferentiated monolithic LLM call (the umbrella
      label that historical ``tutor:*`` sources fell through to before
      the 2026-04-27 vocabulary cleanup).
    - ``analogy`` / ``llm`` / ``abductive`` — speculative proposals
      that must earn credence through evidence.
    """

    user_correction:       float = 0.95
    adapter_seed:          float = 0.80
    observer_full_scan:    float = 0.70
    mediator_proposed:     float = 0.65
    kb_loaded:             float = 0.65
    miner_confirmed:       float = 0.60
    oracle_proposed:       float = 0.50
    analogy_transfer:      float = 0.40
    llm_proposer:          float = 0.30
    abductive_speculation: float = 0.25

    def for_source(self, source: str) -> float:
        """Lookup initial credence for a named source; defaults to 0.5.

        The ``source`` string follows ``"<kind>:<detail>"`` convention
        (e.g. ``"miner:FutilePattern"``, ``"user:correction"``).  Only the
        ``<kind>`` prefix is used for routing.
        """
        kind = source.split(":", 1)[0]
        return {
            "user":       self.user_correction,
            "adapter":    self.adapter_seed,
            "observer":   self.observer_full_scan,
            "mediator":   self.mediator_proposed,
            "kb":         self.kb_loaded,
            "miner":      self.miner_confirmed,
            "oracle":     self.oracle_proposed,
            "analogy":    self.analogy_transfer,
            "llm":        self.llm_proposer,
            "abductive":  self.abductive_speculation,
        }.get(kind, 0.5)


# ---------------------------------------------------------------------------
# Explorer / curiosity
# ---------------------------------------------------------------------------


@dataclass
class ExplorerConfig:
    """Exploration and curiosity parameters.

    ``curiosity_level`` is the primary knob.  Setting it alone via
    :meth:`from_curiosity_level` gives a coherent default for all derived
    parameters.  Individual parameters can still be overridden afterward
    for fine-grained tuning.

    Parameters
    ----------
    curiosity_level
        0.0 = never explore for its own sake; 1.0 = maximally curious.
    curiosity_threshold
        Claim-coverage below this fraction marks an entity as "unknown"
        and therefore a candidate for curiosity-driven probing.
    novelty_base
        Base priority assigned to a newly generated curiosity goal.
    info_gain_weight
        Relative importance of discriminating between competing
        hypotheses vs. probing wholly unknown entities.
    idle_boost
        Multiplier applied to curiosity goal priority when no
        higher-priority goal is currently making progress.  Encourages
        the agent to "look around" when not busy rather than sitting still.
    generate_curiosity_goals
        Master switch.  When False the explorer produces only
        info-gain goals (never raw novelty-seeking goals).
    motion_model_ig_weight
        Weight on the motion-model completeness info-gain term
        (:func:`explorer.motion_model_info_gain`).  Boosts
        exploration toward actions whose effect on the controlled
        actor has not yet been characterised by a committed
        :class:`MotionModelClaim`.  Robotics analogue: motor
        babbling weight — the priority given to exercising motor
        primitives whose kinematic effect on the end-effector
        pose is still unknown.  Default 2.0 keeps this term
        dominant while the motion model is incomplete; once every
        action has a committed motion model (or is exhausted by
        repeated zero-delta observations), the term decays to
        0 and normal info-gain / curiosity scoring takes over.
    """

    curiosity_level:          float = 0.3
    curiosity_threshold:      float = 0.2
    novelty_base:             float = 0.1
    info_gain_weight:         float = 1.0
    idle_boost:               float = 3.0
    generate_curiosity_goals: bool  = True
    motion_model_ig_weight:   float = 2.0

    # GAP 16 — tabu/recency-aware exploration.  Without memory the
    # explorer re-selects the same top-scoring action every step,
    # which is catastrophic when the env's response forms a cycle
    # (e.g. a partial boundary snap sends the agent around a
    # 3-position loop; see ``tests/test_gap16_*`` for the live
    # shape).  The runner maintains a rolling list of recent
    # ``(pre_pos, action_name)`` pairs at
    # ``ws.agent['_recent_pos_actions']``; ``choose_exploration_action``
    # subtracts ``tabu_penalty`` from each candidate's score for
    # every appearance of ``(current_pos, candidate_action)`` in
    # that window.
    #
    # Tuning notes.
    # * ``tabu_window`` sets how far back recency reaches.  Short
    #   windows (5-10) escape tight cycles quickly but can make the
    #   explorer thrash if the env genuinely requires a short
    #   repeat.  Long windows (50+) accumulate global "I've been
    #   everywhere" pressure that may drown meaningful info-gain
    #   signals.  Default ``20`` is deliberate: long enough to
    #   catch the 3-cycles observed in ls20, short enough to
    #   forget ancient visits that are no longer relevant.
    # * ``tabu_penalty`` must be large enough to break ties across
    #   the scoring heuristics (info_gain ~0-2, curiosity ~0-0.5,
    #   motion_model_ig ~0-1 late-game).  Default ``1.0`` dominates
    #   all three, which is what we want for cycle-breaking; lower
    #   it if tabu becomes too aggressive in domains where
    #   deliberate repetition is useful.
    #
    # Robotics analogue: motor babbling with a short-term memory,
    # preferring motions the arm has not just tried from this pose.
    tabu_window:              int   = 20
    tabu_penalty:             float = 1.0

    @classmethod
    def from_curiosity_level(cls, level: float) -> "ExplorerConfig":
        """Derive a coherent ExplorerConfig from a single 0..1 curiosity knob.

        level=0.0  → no curiosity goals generated at all (pure exploit).
        level=0.5  → moderate exploration when idle, balanced with exploit.
        level=1.0  → aggressive exploration; will prefer unknowns even when
                     progress on the primary goal is possible.
        """
        level = max(0.0, min(1.0, level))
        return cls(
            curiosity_level          = level,
            curiosity_threshold      = 0.1 + 0.3 * level,
            novelty_base             = 0.05 + 0.25 * level,
            info_gain_weight         = 0.5 + 1.0 * level,
            idle_boost               = 1.0 + 4.0 * (1.0 - level),   # low-curiosity agents rely more on idle boost
            generate_curiosity_goals = level > 0.0,
            # Motion-model completeness term stays substantial even at
            # low curiosity: a missing motion model blocks *all*
            # positional planning, so characterising actions is a
            # baseline investment regardless of how eagerly the agent
            # seeks other novelties.  Scales from 1.0 (pure exploit) to
            # 3.0 (high curiosity).  Robotics analogue: motor-babbling
            # priority — a robot with no kinematic model of its
            # primitives cannot reach a target pose, so babbling is
            # prioritised independently of curiosity about the world.
            motion_model_ig_weight   = 1.0 + 2.0 * level,
        )


# ---------------------------------------------------------------------------
# LLM budgets
# ---------------------------------------------------------------------------


@dataclass
class LLMBudget:
    """Hard caps on LLM invocations, tracked by the ResourceTracker.

    The engine is code-centric; LLM calls go through two typed oracle
    seams:

    * OBSERVER — visual Q&A (frame-in, typed-answer-out).  Per-call cost
      is low (VLM call with a small frame region); calls are frequent
      during initial visual scans and whenever a cached visual relation
      needs re-validating.

    * MEDIATOR — common-sense guidance given a symbolic WorldStateSummary.
      Per-call cost is higher (large-context text LLM call); calls are
      infrequent — triggered by impasses, unexplained surprises, cold
      starts, and hazard queries.

    The two budgets are tracked separately so a burst of visual
    revalidation cannot starve a later impasse consultation (or vice
    versa).  The ``per_goal`` and ``per_hypothesis`` caps apply to the
    *combined* oracle usage attributable to a specific goal or hypothesis.
    """

    # Observer (visual) budgets
    observer_per_episode:   int = 50

    # Mediator (common-sense) budgets — typically smaller because
    # per-call cost is higher and the engine should prefer learned
    # patterns over re-consulting the Mediator on the same impasse.
    mediator_per_episode:   int = 10

    # Combined caps — apply to both oracles together
    per_goal:               int = 10
    per_hypothesis:         int = 2

    # Tolerance for budget overrun before hard-stopping
    overrun_tolerance:      int = 3   # episodes in a row we will tolerate exceeding budget before hard-stopping


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class PlannerConfig:
    """Planner configuration — controls replanning cadence and search limits."""

    # Always-replan conditions (cheap) are unconditional.
    # These are optional extensions, off by default for ARC, on for robotics.
    replan_on_surprise: bool = False
    replan_periodic:    bool = False
    replan_interval:    int  = 10

    max_plan_depth: int = 200       # max AO* search depth (guards runaway search)
    branch_budget:  int = 10_000    # max nodes expanded per planning call

    # Wall-override soft threshold.  A zero-delta
    # :class:`ActorTransitionClaim` ("action X from position P produced
    # no movement") is treated as a local suppression of the open-field
    # motion model during BFS as soon as its credence reaches this
    # floor — well before the general ``commit_threshold`` (0.85) would
    # fire.  Rationale: a single first-hand observation of "I tried
    # this and didn't move" is nearly dispositive; if we are wrong,
    # one more trial will contradict and the override's credence will
    # decay.  If we are right and ignore it, the agent wedges for
    # ``commit_threshold``-many steps re-trying a futile action.  The
    # asymmetry strongly favours believing wall observations early.
    #
    # Robotics analogue: if the end-effector bumped a fixture once, do
    # not route a plan through that pose again until the override is
    # actively falsified.
    #
    # Default ``0.60`` matches the standard ``miner_confirmed``
    # :class:`SourcePriors.miner_confirmed` initial credence so the
    # first observation engages the override.  Set to a value above
    # ``CredenceConfig.commit_threshold`` to restore the pre-GAP-14
    # strict-commit behaviour.
    wall_override_credence_floor: float = 0.60

    # Minimum credence a hypothesis must carry for the planner to
    # use it (``SPEC_continuous_commitment.md``).  The planner reads
    # the hypothesis store through this floor rather than through
    # the global commit threshold, so a motion-model claim with a
    # few confirming observations becomes plannable well before it
    # would "fully commit" at 0.85.  Credence is continuous; binary
    # commitment is rejected by construction.  Raise for cautious
    # plans (closer to 0.85 reproduces the old committed-only gate),
    # lower for exploratory plans.
    min_credence: float = 0.5


# ---------------------------------------------------------------------------
# Action probing
# ---------------------------------------------------------------------------


@dataclass
class ActionProbeConfig:
    """Systematic per-action probing — how the engine learns what each
    action in the adapter's action space *does*.

    Rationale.  An agent that does not know the effect of its own
    actions has no forward model and therefore no basis for planning.
    Default miners (``TransitionMiner``, ``FutilePatternMiner``) only
    produce claims once ``AgentMoved`` events or a non-empty agent
    position exist, which is not guaranteed on a brand-new domain.
    This config controls a domain-agnostic bootstrap: for every action
    the adapter exposes, the engine generates a *probe goal* that says
    "try this action once and observe ``ws.last_frame_delta``".  The
    ``ActionEffectMiner`` then mints a ``CausalClaim`` describing the
    observed effect (including "no observable change" as a legitimate
    outcome).  Once every action has at least one committed claim,
    probing stops and ordinary goal-directed planning resumes.

    All values live here so they can be audited and tuned in one
    place.  Do **not** hardcode these numbers in miners, goal
    derivation, or the planner.

    Fields
    ------
    enabled
        Master switch.  When ``False`` the engine performs no
        systematic action probing (behaviour reverts to pre-Phase-5c:
        action effects must be learned incidentally from other
        miners).
    probe_goal_priority
        Priority assigned to an action-probe goal.  Must be greater
        than the episode's primary-goal priority (typically ``1.0``)
        so that a just-discovered, untested action is probed *before*
        the planner commits to a long reach plan founded on unknown
        dynamics.  Default ``1.01``.
    initial_claim_point
        Initial credence ``point`` for a claim minted by the
        ``ActionEffectMiner``.  Set above ``CredenceConfig.commit_threshold``
        (``0.85``) so a single clean observation commits the claim —
        repeated probing of the same action in the same context is
        wasteful once one trial has fired.  Default ``0.90``.
    initial_claim_evidence_weight
        Initial evidence weight for the minted claim.  ``3.0`` marks
        the claim as resting on solid single-trial ground without
        making it immune to contradiction.
    min_trials_to_validate
        Minimum number of successful probes of an action before its
        probe goal is considered achieved.  Default ``1`` (one good
        observation is enough to form a baseline claim; subsequent
        evidence naturally accumulates through ordinary play).
    max_actions_per_episode
        Safety valve — cap on total probe actions in one episode.  A
        malformed action_space() with thousands of entries would
        otherwise consume the entire step budget on probing.  Default
        ``64``.
    require_non_empty_delta
        If ``True``, a probe is considered informative *only* when
        ``ws.last_frame_delta`` is non-empty (the action produced an
        observable change).  If ``False``, an empty delta also counts
        — the claim will say "action X in this context has no
        observable effect", which is itself useful knowledge for
        pruning the action space.  Default ``False``.
    """

    enabled:                        bool  = True
    probe_goal_priority:             float = 1.01
    initial_claim_point:             float = 0.90
    initial_claim_evidence_weight:   float = 3.0
    min_trials_to_validate:          int   = 1
    max_actions_per_episode:         int   = 64
    require_non_empty_delta:         bool  = False

    # Per-action attempt cap under continuous commitment
    # (``SPEC_continuous_commitment.md``).  A probe goal stays open
    # while its motion model's credence is below the planner floor,
    # but if the action has been executed this many times without
    # the miner producing a credible model the probe is abandoned
    # and a ``PropertyClaim("_action", <id>, "unmappable")`` is
    # emitted so downstream components know not to plan around it.
    # Belt-and-braces: priority × (1 − credence) already pulls
    # settled probes off the selector's top; this cap is the hard
    # safety rail for genuinely unmappable buttons and for
    # pathological context-conditional effects (e.g. "UP only
    # moves when no wall is above") that piece 1 cannot diagnose.
    max_probe_attempts_per_action:   int   = 15


# ---------------------------------------------------------------------------
# Competence-gated execution mode — lazy import to avoid circular
# dependency with cognitive_os.competence (which imports types).  The
# EngineConfig dataclass default_factory calls this helper which
# imports on first access.
# ---------------------------------------------------------------------------


def _default_competence_config():
    """Build the default CompetenceConfig without creating a module-
    level import of ``cognitive_os.competence`` — that module imports
    ``types``, which in turn is re-exported by the package root, and
    we want ``config`` to stay at the bottom of the import stack."""
    from .competence import CompetenceConfig
    return CompetenceConfig()


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Top-level engine configuration.

    The two factory methods provide sane defaults for the two target
    domains; any individual sub-config can be overridden after construction.
    """

    credence:       CredenceConfig     = field(default_factory=CredenceConfig)
    source_priors:  SourcePriors       = field(default_factory=SourcePriors)
    explorer:       ExplorerConfig     = field(default_factory=ExplorerConfig)
    llm_budget:     LLMBudget          = field(default_factory=LLMBudget)
    planner:        PlannerConfig      = field(default_factory=PlannerConfig)
    action_probe:   ActionProbeConfig  = field(default_factory=ActionProbeConfig)
    # Competence-gated learn-priority bias
    # (memory/project_cognitive_os_execution_mode.md).  When the top
    # task goal becomes plannable from committed claims, learn-family
    # goal priorities (reduce_uncertainty/explore/probe) are scaled
    # down by a continuous factor (default 0.3) so the plannable task
    # goal wins normal priority arbitration — but learning stays as a
    # live fallback if the plan later stalls.  Robotics and ARC share
    # the mechanism (framed as a goal-priority bias, not a mode flag).
    competence:     "CompetenceConfig" = field(
        default_factory=lambda: _default_competence_config()
    )
    # Perception / entity stitching (GAP 24a-3).  Adapters re-mint
    # entity ids whenever the segmenter fires on a frame change; the
    # engine opt-in stitches new ids back to recently-seen ones when
    # colour matches exactly and bbox IoU clears a threshold.  Prevents
    # the goal forest from bloating one entry per frame-changed id.
    # Disable (``enable_entity_stitching=False``) for adapters that
    # already maintain stable ids themselves.
    enable_entity_stitching:         bool  = True
    entity_stitch_iou_threshold:     float = 0.5
    entity_stitch_staleness_window:  int   = 3
    # Telemetry sink (WIP — not yet committed).  Defaults to NullSink
    # so runner reads of ``cfg.telemetry`` succeed without emitting.
    telemetry: Any = field(
        default_factory=lambda: __import__(
            "cognitive_os.telemetry", fromlist=["NullSink"]
        ).NullSink()
    )
    # Default to TRAINING: safer for development.  Harness/CLI must
    # explicitly set COMPETITION for benchmarking runs.  The runtime's
    # single load gate reads this value and filters persisted
    # CachedSolutions by scope accordingly.
    operating_mode: OperatingMode  = OperatingMode.TRAINING

    @classmethod
    def arc_agi3_default(cls) -> "EngineConfig":
        """Defaults tuned for ARC-AGI-3 gameplay.

        - Low-to-moderate curiosity: exploration only when stuck.
        - No periodic/surprise replanning: plans are stable until invalidated.
        - Tight LLM budget: most knowledge must come from miners.
        """
        return cls(
            explorer   = ExplorerConfig.from_curiosity_level(0.3),
            planner    = PlannerConfig(replan_on_surprise=False, replan_periodic=False),
            llm_budget = LLMBudget(observer_per_episode=30, mediator_per_episode=5),
        )

    @classmethod
    def robotics_default(cls) -> "EngineConfig":
        """Defaults tuned for embodied robotics.

        - Higher curiosity: open-ended environments reward exploration.
        - Replan on surprise AND periodically: safety-critical drift detection.
        - Larger LLM budget: visual queries and common-sense calls are routine.
        """
        return cls(
            explorer   = ExplorerConfig.from_curiosity_level(0.5),
            planner    = PlannerConfig(replan_on_surprise=True,
                                        replan_periodic=True,
                                        replan_interval=20),
            llm_budget = LLMBudget(observer_per_episode=200,
                                    mediator_per_episode=40,
                                    per_goal=40),
        )
