"""Episode runner — the main loop that orchestrates an episode.

One call to :func:`run_episode` consumes a fully-constructed
:class:`Adapter` and drives it through a single episode.  The runner
owns the step cadence; every subsystem in the engine is invoked from
here in a fixed order that preserves the invariants each phase
established.

Main loop structure
-------------------

::

    1. Adapter.initialize() / reset()          — populate tool registry,
                                                  seed primary goal,
                                                  full_scan frame
    2. Observer.full_scan() if not yet done    — initial visual priors
    3. Loop until done or budget exhausted:
       a. obs = adapter.observe()
       b. ingest(ws, obs)                      — append to history,
                                                  update agent state,
                                                  refresh entity snapshots
       c. miners.step()                        — PropertyObserved,
                                                  Transition, FutilePattern,
                                                  Surprise
       d. update_credence_from_events()        — evidence pass
       e. apply_staleness_decay_all()
       f. prune_abandoned()
       g. derive_subgoals_from_causal()        — expand goal tree
          detect_conflicts()
       h. select_active_goal()
       i. plan = compute_plan()                — or None if exhausted
       j. if plan invalid → replan
          elif plan is None → explore
          elif plan exhausted → add curiosity goal, retry
       k. action = plan.steps[0] or exploration fallback
       l. adapter.execute(action)              — and record _last_action
       m. step += 1
    4. run_post_mortem()                       — extract lessons,
                                                  synthesise Options,
                                                  persist cross-episode
    5. Adapter.on_episode_end()

Capability audit
----------------
* **Problem-solving** — PRIMARY.  The runner is where planning,
  exploration, and execution are actually connected to the
  environment.  Without this the engine is a collection of parts.
* **Debugging** — PRIMARY.  The runner drives the hypothesis
  lifecycle every step: ingest events, run miners, update credence,
  detect surprises, trigger specialisation.  Phase 2's machinery
  fires here per step.
* **Tool creation** — secondary.  :func:`run_post_mortem` is called
  at episode end; the :class:`OptionSynthesiser` hook is exercised,
  ready for Phase 7's fuller synthesis.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Union

from .adapters import Adapter
from .config import EngineConfig, PlannerConfig
from .miners import Miner, default_miners
from .types import (
    Action,
    AgentMoved,
    EntityModel,
    Event,
    GoalStatus,
    Observation,
    Plan,
    PlanStatus,
    PostMortem,
    SurpriseEvent,
    WorldState,
)
from . import hypothesis_store as _store
from . import goal_forest as _gf
from .goal_manager import DefaultGoalManager
from . import planner as _planner
from . import explorer as _explorer
from . import telemetry_schema as _tel
from .frame_diff import compute_frame_delta
from .instincts import InstinctRegistry
from .oracle import OracleTrigger, default_triggers
from .persistence import load_committed_knowledge, save_committed_knowledge
from .postmortem import run_post_mortem


def _plan_telemetry_id(plan: Optional[Plan]) -> Optional[str]:
    """Stable id for a :class:`Plan` on the telemetry wire.

    :class:`Plan` has no intrinsic id field (plans are identified in
    the engine by the goal they serve plus their computed step).  For
    clients that need to animate plan transitions as a single object
    we synthesise a composite key; it stays constant while the plan
    executes and changes when the planner produces a new one.
    """
    if plan is None:
        return None
    return f"plan:{plan.goal_id}:{plan.computed_at}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_episode(adapter:     Adapter,
                ws:          WorldState,
                cfg:         EngineConfig,
                *,
                episode_id:  Optional[str] = None,
                max_steps:   int = 10_000,
                miners:      Optional[List[Miner]] = None,
                triggers:    Optional[List[OracleTrigger]] = None,
                instincts:   Optional[InstinctRegistry] = None,
                step_callback: Optional[Callable[[WorldState, int, Dict[str, Any]], None]] = None,
                knowledge_dir: Optional[Union[str, Path]] = None) -> PostMortem:
    """Run one episode end-to-end against ``adapter``.

    Parameters
    ----------
    adapter
        A fully-constructed :class:`Adapter` subclass instance.
    ws
        WorldState to populate.  A fresh instance is typical for
        competition runs; a preloaded instance (with persisted
        hypotheses + options) is typical for training runs.
    cfg
        :class:`EngineConfig` to drive thresholds, budgets, and
        planner cadence.  Attached to ``ws.config`` for the
        duration.
    episode_id
        Stable identifier for logging and :class:`PostMortem`.
        Defaults to ``"{adapter.env_id}::ep_auto"``.
    max_steps
        Hard cap on iterations.  Exceeding triggers ``"timeout"``.
    miners
        Override the default miner suite.  Defaults to
        :func:`miners.default_miners()`.
    triggers
        Override the default oracle-dispatch trigger suite.  Defaults
        to :func:`oracle.default_triggers()`.  Pass an empty list to
        disable all oracle consultation (symbolic-only mode — exactly
        reproduces Phase 4 behaviour).
    instincts
        Optional :class:`InstinctRegistry` carrying registered Tier-0
        instincts (e.g. :class:`ReduceUncertainty`).  Fires once per
        tick after oracle triggers and before goal synthesis.
        ``None`` (default) disables the instinct layer — pre-GAP-23
        behaviour preserved exactly.  See
        :mod:`cognitive_os.instincts` for the design rationale and
        ``project_arc_agi3_instincts.md`` in project memory for the
        multi-landing roadmap.
    step_callback
        Optional ``(ws, step, info) -> None`` invoked at the end of
        every step (after the new observation has been ingested).
        ``info`` is a dict with these keys:

        * ``events``       — the Events processed this step
        * ``action``       — the Action executed this step (None if
                             the episode terminated before an action
                             was chosen)
        * ``active_goal_id`` — the goal the planner/explorer was
                               pursuing this step
        * ``plan_steps``   — remaining steps in ``current_plan`` (0
                             if no plan is active)

        Used by external dashboards, progress printers, and per-step
        telemetry writers.  Must not mutate ``ws``; exceptions are
        propagated to the caller.
    knowledge_dir
        If given, a directory path used for cross-episode knowledge
        persistence.  Before the first step, any claims previously
        saved under this directory are re-proposed into ``ws`` at
        their persisted credence — so an episode that ran last week
        and learned ``ACTION1→UP`` still knows it today.  After the
        episode ends (and post-mortem runs), every committed
        transferable claim in ``ws`` is written back.  Callers
        typically scope one directory per environment, e.g.
        ``.knowledge/ls20``.  ``None`` disables persistence —
        behaviour is identical to Phase-5c without this argument.
        See :mod:`cognitive_os.persistence` for the whitelist of
        claim types covered.

    Returns
    -------
    PostMortem
        The episode retrospective.  Callers persist its
        ``lessons`` and ``options_synthesised`` if operating in a
        mode that allows cross-episode accumulation.

    Side effects
    ------------
    Mutates ``ws`` extensively — hypotheses, goal forest, observation
    history, agent state all accumulate.  Calls every
    :class:`Adapter` hook in order.
    """
    ws.config = cfg
    episode_id = episode_id or f"{adapter.env_id}::ep_auto"
    # GoalManager — canonical surface for per-turn goal maintenance.
    # See cognitive_os/goal_manager.py for the Protocol; the default
    # impl wraps the existing free functions in goal_forest so this
    # is a pure refactor (no behavior change).  Adapters with their
    # own JSON-spec compilers may pass a subclass instead.
    _gm = DefaultGoalManager(ws)
    # Seed the episode id into ``ws.agent`` so subsystem emits via
    # :func:`telemetry.emit_from_ws` (hypothesis_store, goal_forest)
    # carry episode context.  The underscore prefix protects it from
    # being wiped by :func:`_ingest_observation`'s agent-state refresh.
    # The ``emit_from_ws`` docstring specifies this contract; this
    # assignment makes it hold.
    ws.agent["_episode_id"] = episode_id
    miners = miners or default_miners()
    triggers = triggers if triggers is not None else default_triggers()
    for trigger in triggers:
        trigger.reset()

    sink = cfg.telemetry
    sink.emit(
        _tel.EpisodeBegin(
            episode_id     = episode_id,
            adapter_kind   = getattr(adapter, "env_id", adapter.__class__.__name__),
            operating_mode = cfg.operating_mode.value,
            seed           = None,
        ),
        episode = episode_id,
        subject = episode_id,
    )

    # Phase 1: init
    adapter.initialize(ws)
    obs = adapter.reset()
    _ingest_observation(ws, obs)

    # Initial mining + credence pass — process the starting observation
    # immediately so ``ws.agent['position']`` is populated before the
    # first action is chosen, and so claims seeded from the initial
    # frame (e.g. the adapter's role PropertyClaims, the initial
    # entity inventory) are already visible to the first planner call.
    #
    # The mining pass was previously at the TOP of each loop iteration,
    # which left a one-step lag between ``ws.last_frame_delta`` (fresh)
    # and ``ws.agent['position']`` (stale) at ``step_callback`` time,
    # at ``adapter.is_done()`` time, and for any other consumer
    # reading position between ``_ingest_observation`` and the next
    # iter's miner pass.  Moving it to fire immediately after each
    # ingest keeps delta, position, and credence in lockstep.
    _events_initial = list(obs.events) if obs else []
    for _miner in miners:
        _miner.step(ws, _events_initial, 0)
    _events_initial = (
        list(ws.observation_history[-1].events)
        if ws.observation_history else _events_initial
    )
    _store.update_credence_from_events(ws, _events_initial, 0)
    _store.apply_staleness_decay_all(ws, 0)
    _store.prune_abandoned(ws, 0)

    # Cross-episode knowledge: re-propose every previously-committed
    # transferable claim into the store at its persisted credence, so
    # the episode starts with yesterday's learning intact.  A missing
    # knowledge file is a no-op (fresh environment).  Runs before
    # on_episode_start so the adapter's hooks can already see the
    # restored claims if they care to.
    if knowledge_dir is not None:
        load_committed_knowledge(ws, knowledge_dir, step=0)

    adapter.on_episode_start(ws)

    # Seed the action-probe subsystem.  The adapter is the sole source
    # of truth for what actions exist; we snapshot that list on
    # ``ws.agent['_action_space']`` so both :func:`goal_forest.derive_action_probe_goals`
    # and downstream conditions (:class:`ActionTried`) can read it
    # without re-querying the adapter.  ``_actions_tried`` starts
    # empty.  Both live under engine-owned underscore-prefixed keys
    # that :func:`_ingest_observation` preserves across steps.
    try:
        action_ids = [a.name for a in adapter.action_space()]
    except Exception:
        action_ids = []
    ws.agent["_action_space"]  = tuple(action_ids)
    ws.agent.setdefault("_actions_tried", set())

    start_time = perf_counter()
    step = 0
    failed_plans: List[Plan] = []
    current_plan: Optional[Plan] = None
    final_status = "in_progress"

    while step < max_steps:
        if adapter.is_done():
            final_status = _done_status(ws, adapter)
            break

        _step_t0 = perf_counter()
        _planner_ms = 0.0
        sink.emit(_tel.StepBegin(), episode=episode_id, step=step)

        # NOTE: mining + credence + staleness + prune run at END of the
        # previous iteration (and once after the initial ingest), not
        # at the top of this one.  That keeps ``ws.agent['position']``
        # in sync with ``ws.last_frame_delta`` for every consumer —
        # ``adapter.is_done()`` below, the planner, the callback.
        # ``events`` for this iter is re-read from the freshest
        # observation so oracle triggers and instincts see any
        # surprise events appended by the end-of-prev-iter miners.
        events = (
            list(ws.observation_history[-1].events)
            if ws.observation_history else []
        )

        # Oracle-dispatch pass: each trigger inspects WorldState,
        # optionally builds an ObserverQuery / MediatorQuery,
        # dispatches it via the adapter, and installs the typed
        # answer as new hypotheses.  Triggers self-gate; empty list
        # disables oracle consultation entirely.
        for trigger in triggers:
            trigger.maybe_dispatch(ws, adapter, step, cfg)

        # Tier-0 instincts: fire domain-general priors that propose
        # goals unprompted.  Runs after oracle triggers (so instincts
        # can react to freshly-installed hypotheses) but before goal
        # synthesis from roles (so their proposed goals get the same
        # downstream treatment as role-derived ones).  A ``None``
        # registry is a no-op, preserving pre-GAP-23 behaviour for
        # callers that have not opted in.  See
        # :mod:`cognitive_os.instincts`.
        if instincts is not None:
            proposals = instincts.fire_all(ws)
            instincts.apply(ws, proposals, step=step)

        # Goal synthesis from role PropertyClaims.  Runs before
        # subgoal derivation so any new Goals get the same
        # causal-expansion pass their adapter-seeded siblings do.
        # Domain-agnostic: both ARC "role=target" and robotics
        # "role=pickup_target" lift into reach-goals via the same
        # mechanism.
        _gf.derive_goals_from_roles(ws, step=step)

        # Action-probe goals: generate one high-priority goal per
        # action the agent has never executed.  Satisfied by a trivial
        # one-step plan in the planner's ActionTried fast path; once
        # tried, the :class:`ActionEffectMiner` mints a
        # :class:`CausalClaim` describing the observed frame-delta.
        # This is how the engine bootstraps its action model on a
        # brand-new domain with no hand-authored transition knowledge.
        _gf.derive_action_probe_goals(ws, step=step)

        # Subgoal derivation + conflict detection — routed through the
        # GoalManager interface (cognitive_os/goal_manager.py).  See
        # SPEC documentation: this is the canonical per-turn entry
        # point for goal maintenance; replacing the manager swaps in
        # alternate selection / derivation policies without changing
        # this site.
        _gm.tick(step=step)

        # Plan validity
        current_plan = _check_plan_validity(ws, current_plan, cfg)

        # Goal selection + planning.
        #
        # GAP 19 — priority-fallback selection.  Pre-GAP-19 the runner
        # picked the single highest-priority goal via
        # :func:`goal_forest.select_active_goal` and gave up on
        # planning (falling through to exploration) if that goal's
        # plan was None — even when a lower-priority goal would have
        # been plannable in the same state.  The ls20 probe showed
        # the adapter-seeded ``episode`` goal (priority 1.0, condition
        # ``ResourceAbove("episode_won", 0.5)``) was unplannable in
        # 60/80 steps, shadowing a plannable role-goal (priority 0.5)
        # that the BFS could reach.
        #
        # :func:`planner.select_and_plan` iterates candidates in
        # priority order and returns the first plannable one with a
        # non-empty plan, preserving the priority policy while
        # refusing to discard usable plans.  Empty-plan goals (already
        # satisfied / all-deferred subtree) are skipped here so the
        # runner produces motion; the status machinery handles their
        # closure via ``refresh_status``.
        #
        # Competence-gated learn-priority bias.  Per-tick, just
        # before the selector runs, we ask: can the agent plan any
        # of its top task goals using only committed claims?  If
        # yes, scale the priority of learn-family goals
        # (reduce_uncertainty / explore / probe) by a configured
        # factor (default 0.3) so the plannable task goal wins
        # normal priority arbitration — but learning stays as a
        # live fallback at reduced priority, for graceful
        # degradation if the plan later stalls.  If no (nothing
        # plannable yet), scale stays at 1.0 and the existing
        # priority policy applies unchanged.  The scale is written
        # to ``ws.agent['_learn_priority_scale']`` so downstream
        # readers (the sort key in candidates_by_priority plus any
        # telemetry) can see the current bias without importing
        # this module.  See
        # ``memory/project_cognitive_os_execution_mode.md``.
        from . import competence as _competence
        _scale = _competence.compute_learn_priority_scale(
            ws, adapter.action_space(), step=step,
            cfg=cfg.competence,
            operating_mode=cfg.operating_mode,
        )
        _competence.apply_learn_priority_scale(ws, _scale)

        if current_plan is None or current_plan.status != PlanStatus.ACTIVE:
            _t = perf_counter()
            active, current_plan = _planner.select_and_plan(
                ws, adapter.action_space(), step=step)
            _planner_ms += (perf_counter() - _t) * 1000.0
            if active is None:
                # No plannable goal → try curiosity goals, then one
                # more selection+plan pass.  Curiosity goals are
                # typically high-priority probes with trivial
                # one-step plans, so this retry rarely fails when
                # actions are available.
                cur_goals = _explorer.propose_curiosity_goals(
                    ws, step=step, action_space=adapter.action_space())
                for g in cur_goals:
                    _gf.add_goal(ws, g)
                _t = perf_counter()
                active, current_plan = _planner.select_and_plan(
                    ws, adapter.action_space(), step=step)
                _planner_ms += (perf_counter() - _t) * 1000.0

        # Action selection
        action: Optional[Action] = None
        if current_plan is not None and current_plan.steps:
            action = current_plan.steps[0].action
        else:
            action = _explorer.choose_exploration_action(
                ws, adapter.action_space())

        if action is None:
            # Nothing to do — exhausted.  Close the telemetry bracket
            # so clients see matched StepBegin/StepEnd even on abort.
            sink.emit(
                _tel.StepEnd(
                    planner_latency_ms = _planner_ms,
                    total_latency_ms   = (perf_counter() - _step_t0) * 1000.0,
                    action_kind        = None,
                    plan_id            = None,
                ),
                episode = episode_id,
                step    = step,
            )
            final_status = "abandoned:no_action"
            break

        # Capture plan id before the advance-pointer block may null it.
        _plan_id_this_step = _plan_telemetry_id(current_plan)

        # Execute and record
        ws.agent["_last_action"] = action.name
        # Maintain the cumulative set of action ids tried at least
        # once.  Consumed by :class:`ActionTried` (goal-achievement
        # predicate) and :func:`goal_forest.derive_action_probe_goals`
        # (skip already-probed actions).
        tried = ws.agent.setdefault("_actions_tried", set())
        if not isinstance(tried, set):
            tried = set(tried)
            ws.agent["_actions_tried"] = tried
        tried.add(action.name)
        # Per-action attempt counter (SPEC_continuous_commitment.md).
        # ``_actions_tried`` collapses to "at least once"; the probe
        # subsystem needs the raw attempt count to enforce its
        # per-action cap and eventually declare an action
        # ``unmappable``.
        attempts = ws.agent.setdefault("_action_attempts", {})
        attempts[action.name] = int(attempts.get(action.name, 0)) + 1
        # GAP 16 — maintain the tabu window for recency-aware
        # exploration.  We record the ``(pre_pos, action_name)``
        # at the moment of action selection (before execute).  Even
        # when the planner picked the action (not the explorer),
        # we still record it so future exploration calls see the
        # full recent history.  Trimmed to ``tabu_window`` length.
        tabu_window = 20
        if ws.config is not None and hasattr(ws.config, "explorer"):
            tabu_window = int(getattr(ws.config.explorer,
                                      "tabu_window", 20))
        tabu = ws.agent.setdefault("_recent_pos_actions", [])
        pre_pos = ws.agent.get("position")
        pre_key = tuple(pre_pos) if pre_pos is not None else None
        tabu.append((pre_key, action.name))
        # Trim to window.  Keep the newest entries.
        if len(tabu) > tabu_window:
            del tabu[: len(tabu) - tabu_window]
        adapter.execute(action)

        # Advance plan pointer
        if current_plan is not None and current_plan.steps:
            current_plan.current_step_index += 1
            current_plan.steps = current_plan.steps[1:]
            if not current_plan.steps:
                current_plan.status = PlanStatus.COMPLETE
                failed_plans.append(current_plan)
                current_plan = None

        sink.emit(
            _tel.StepEnd(
                planner_latency_ms = _planner_ms,
                total_latency_ms   = (perf_counter() - _step_t0) * 1000.0,
                action_kind        = action.name,
                plan_id            = _plan_id_this_step,
            ),
            episode = episode_id,
            step    = step,
        )

        step += 1
        obs = adapter.observe()
        _ingest_observation(ws, obs)

        # Mining + credence update pass — runs here, immediately after
        # the new observation is ingested, so ``ws.agent['position']``
        # (set by :class:`SelfLocalizationMiner` from
        # ``ws.last_frame_delta``) is in sync with the just-computed
        # delta for every downstream consumer:
        #
        # * the ``step_callback`` below,
        # * the top of the next iteration's ``adapter.is_done()`` check
        #   (which may read position via goal-achievement predicates),
        # * the next iteration's oracle triggers, instincts, goal
        #   synthesis, and planner.
        #
        # Previously the miner pass ran at the TOP of the loop, leaving
        # a one-step lag between the fresh delta and the miner-set
        # position.  The lag corrupted motion-CSV telemetry, could
        # cause ``_done_status`` to misclassify a terminal step whose
        # last action satisfied the goal (position hadn't been updated
        # yet), and produced hard-to-diagnose "noise" in
        # :class:`SelfLocalizationMiner` continuity fallbacks.
        events = list(obs.events) if obs else []
        for miner in miners:
            miner.step(ws, events, step)
        # Miners may have appended surprise events; re-read.
        events = (
            list(ws.observation_history[-1].events)
            if ws.observation_history else events
        )
        _store.update_credence_from_events(ws, events, step)
        _store.apply_staleness_decay_all(ws, step)
        _store.prune_abandoned(ws, step)

        if step_callback is not None:
            step_callback(ws, step, {
                "events":         events,
                "action":         action,
                "active_goal_id": ws.goal_forest.active_goal_id,
                "plan_steps":     len(current_plan.steps) if current_plan else 0,
            })

    else:
        # Loop exited via max_steps guard
        final_status = "timeout"

    # Episode cleanup: pending plan becomes a failed plan for analysis
    if current_plan is not None:
        current_plan.status = PlanStatus.INVALIDATED
        failed_plans.append(current_plan)

    wall_time = perf_counter() - start_time
    pm = run_post_mortem(
        ws                = ws,
        episode_id        = episode_id,
        final_step        = step,
        final_status      = final_status,
        failed_plans      = failed_plans,
        wall_time_seconds = wall_time,
    )
    adapter.on_episode_end(ws)

    # Cross-episode knowledge: serialise every committed transferable
    # claim back to disk.  Runs after post-mortem so that any claims
    # promoted during the post-mortem pass (e.g. :class:`OptionSynthesiser`
    # introducing new macro-level claims) are included.  Missing
    # directory is created on save.
    if knowledge_dir is not None:
        save_committed_knowledge(ws, knowledge_dir)

    sink.emit(
        _tel.EpisodeEnd(
            episode_id   = episode_id,
            final_status = final_status,
            total_steps  = step,
            wall_seconds = wall_time,
        ),
        episode = episode_id,
        subject = episode_id,
    )

    return pm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ingest_observation(ws: WorldState, obs: Observation) -> None:
    """Integrate a new Observation into WorldState.

    * Appends to ``observation_history``.
    * Updates ``ws.step`` and ``ws.agent`` (preserving engine-owned
      private keys starting with ``_``).
    * Refreshes or creates :class:`EntityModel`\\s from
      ``entity_snapshots``.
    """
    if obs is None:
        return
    ws.observation_history.append(obs)
    ws.step = obs.step

    # Preserve private keys (engine-owned).  Also preserve the
    # engine-derived ``position`` across observations when the adapter
    # does not supply one — SelfLocalizationMiner populates it from
    # the committed ControlledActorClaim, and wiping it every step
    # would defeat the point.  Adapter-provided ``position`` always
    # wins (it is authoritative when available).
    private = {k: v for k, v in ws.agent.items() if k.startswith("_")}
    preserved_position = (
        ws.agent.get("position")
        if "position" not in obs.agent_state
        else None
    )
    ws.agent = dict(obs.agent_state)
    ws.agent.update(private)
    if preserved_position is not None:
        ws.agent["position"] = preserved_position

    # GAP 24a-3: entity identity stitching.  When the adapter re-mints
    # ids on every frame change (because it has no persistent tracking),
    # the same physical object acquires a new id every frame.  This
    # bloats the goal forest, defeats coverage credit, and prevents
    # InsideBBox/AtPosition goals from stabilising.  The stitcher walks
    # recently-seen entities with matching colour and high bbox IoU and
    # maps the new id back to the stable id before insertion so that
    # ``ws.entities`` never sees the redundant id.
    aliases: Dict[str, str] = {}
    if ws.config is not None and getattr(ws.config, "enable_entity_stitching", False):
        from .entity_stitch import stitch_entity_ids
        aliases = stitch_entity_ids(
            ws,
            obs.entity_snapshots,
            obs_step         = obs.step,
            iou_threshold    = ws.config.entity_stitch_iou_threshold,
            staleness_window = ws.config.entity_stitch_staleness_window,
        )
    for entity_id, snapshot in obs.entity_snapshots.items():
        stable_id = aliases.get(entity_id, entity_id)
        ent = ws.entities.get(stable_id)
        if ent is None:
            ent = EntityModel(
                id              = stable_id,
                properties      = dict(snapshot),
                first_seen_step = obs.step,
                last_seen_step  = obs.step,
                kind            = snapshot.get("_kind"),
            )
            ws.entities[stable_id] = ent
        else:
            ent.properties.update(snapshot)
            ent.last_seen_step = obs.step

    # First-class frame-diff: populate ws.last_frame_delta from the
    # two most recent observations' raw_frame fields.  This is the
    # single source of truth for "what changed last step" — every
    # downstream subsystem reads from here rather than re-computing.
    # See cognitive_os.frame_diff for type details and the
    # structural-only inspection contract.
    if len(ws.observation_history) >= 2:
        prev_obs = ws.observation_history[-2]
        ws.last_frame_delta = compute_frame_delta(
            prev_obs.raw_frame, obs.raw_frame,
        )
    else:
        ws.last_frame_delta = None


def _check_plan_validity(ws:           WorldState,
                         plan:         Optional[Plan],
                         cfg:          EngineConfig) -> Optional[Plan]:
    """Return the plan unchanged if still valid; ``None`` if
    invalidated.

    Validity checks:

    * Plan assumptions: every hypothesis ID in ``plan.assumptions``
      must still be committed.  Any demotion invalidates.
    * Plan status: ACTIVE plans continue; COMPLETE/INVALIDATED/FAILED
      require replanning.
    """
    if plan is None:
        return None
    if plan.status != PlanStatus.ACTIVE:
        return None

    cred_cfg = cfg.credence
    for h_id in plan.assumptions:
        h = ws.hypotheses.get(h_id)
        if h is None or not h.credence.is_committed(cred_cfg):
            plan.status = PlanStatus.INVALIDATED
            return None
    return plan


def _done_status(ws: WorldState, adapter: Adapter) -> str:
    """Infer a terminal status label from the adapter + goal state.

    Uses :func:`goal_forest.is_achieved` so the check reflects the
    current :class:`WorldState` (e.g. position just updated by the
    last :meth:`Adapter.execute` → :meth:`Adapter.observe`) rather
    than a stale cached status from before the final step.
    """
    for goal_id in ws.goal_forest.goals.keys():
        if _gf.is_achieved(ws, goal_id):
            # Cascade the status so post-mortem sees it correctly.
            _gf.mark_status(ws, goal_id, GoalStatus.ACHIEVED)
            return "success"
    return "failure"
