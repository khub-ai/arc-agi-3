"""Miners — pattern detectors over the event stream.

Miners are stateless callables that observe each step's events,
along with the running :class:`WorldState`, and propose hypotheses
into the :class:`hypothesis_store`.  They are the bridge between raw
symbolic events and the structured hypothesis layer the planner
operates on.

Design
------
Every miner implements :class:`Miner.step(ws, events, step)`.  The
method inspects the events and the current WorldState and calls
:func:`hypothesis_store.propose` for each hypothesis it wants to
register.  The store takes care of dedup, competitor linking, and
credence updates — the miner only decides *what to propose*.

Miners are stateless in the engine sense: all historical context they
need is in ``ws.observation_history``.  A miner that wants to remember
something across steps must encode it as a hypothesis — that's the
entire point of the hypothesis layer.

Phase 4 miners
--------------
The four miners below cover the common patterns needed to populate a
hypothesis store from a live event stream:

* :class:`PropertyObservedMiner` — converts ``EntityStateChanged``
  events directly into ``PropertyClaim``\\s for the new value.  The
  fastest source of hypothesis bootstrap: every entity change seen
  becomes an explicit belief about current state.

* :class:`TransitionMiner` — detects (pre-state, action, post-state)
  triples in the observation/action history and proposes
  ``TransitionClaim``\\s.  Builds the transition model the planner
  uses for BFS.

* :class:`FutilePatternMiner` — detects "action in context yields no
  change" patterns and proposes a ``TransitionClaim`` with the
  trivial post == pre.  Lets the planner avoid known-ineffective
  actions.

* :class:`SurpriseMiner` — detects events that contradict committed
  hypotheses and emits a :class:`SurpriseEvent` into
  ``ws.observation_history`` so that downstream miners /
  Mediator / refinement can react to it.  This is the detector
  behind the specialisation-on-contradiction loop.

Capability audit
----------------
* **Debugging** — PRIMARY.  Miners are how the engine forms
  candidate explanations from observation.  SurpriseMiner
  specifically catches "my prediction was wrong" events that drive
  specialisation and Mediator consultation.
* **Problem-solving** — secondary.  TransitionMiner populates the
  planner's transition model; FutilePatternMiner prunes the
  effective action space.
* **Tool creation** — minor.  Patterns detected here are the raw
  material for the :class:`OptionSynthesiser` in
  :mod:`postmortem`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, Tuple



from .claims import (
    ActorTransitionClaim,
    CausalClaim,
    ControlledActorClaim,
    MotionModelClaim,
    PropertyClaim,
    TransitionClaim,
)
from .conditions import (
    ActionJustTaken,
    AtPosition,
    Condition,
    EntityInState,
    FrameChangedPattern,
    RegionMotion,
)
from .frame_diff import extract_region_motion
from . import hypothesis_store as _store
from .types import (
    AgentMoved,
    ContactEvent,
    EntityStateChanged,
    Event,
    Scope,
    ScopeKind,
    SurpriseEvent,
    WorldState,
)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class Miner(ABC):
    """Abstract base class for all miners.  Subclasses implement
    :meth:`step`.

    Each miner carries a ``name`` (used as the source tag prefix on
    proposed hypotheses) and a ``default_scope`` (applied when the
    miner doesn't have a better scope hint for a given hypothesis).
    Both default to sensible values that subclasses may override.
    """

    name: str = "miner"
    default_scope: Scope = Scope(kind=ScopeKind.GAME)

    @abstractmethod
    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        """Inspect the current step's events and (optionally) the
        :class:`WorldState` history, and propose hypotheses via
        :func:`hypothesis_store.propose`.

        Must not mutate ``ws`` directly — only through ``propose()``.
        """


# ---------------------------------------------------------------------------
# PropertyObservedMiner
# ---------------------------------------------------------------------------


class PropertyObservedMiner(Miner):
    """Convert each :class:`EntityStateChanged` event into a
    :class:`PropertyClaim` asserting the new value.

    Rationale: an observed value *is* evidence that the value holds
    right now.  Competing claims about the same (entity, property)
    appear automatically as canonical competitors and the store
    reconciles them as more observations accumulate.

    This is the simplest possible miner and yields the fastest
    hypothesis-store bootstrap.  It runs on every step.
    """

    name = "miner:PropertyObserved"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        for evt in events:
            if not isinstance(evt, EntityStateChanged):
                continue
            claim = PropertyClaim(
                entity_id = evt.entity_id,
                property  = evt.property,
                value     = evt.new,
            )
            _store.propose(
                ws,
                claim  = claim,
                source = self.name,
                scope  = self.default_scope,
                step   = step,
            )


# ---------------------------------------------------------------------------
# EntityActionCausalMiner
# ---------------------------------------------------------------------------


class EntityActionCausalMiner(Miner):
    """Pair :class:`EntityStateChanged` events with the action that
    just executed and propose :class:`CausalClaim`\\s of the form
    *"action A causes entity E to enter state property=new"*.

    Generic over every (entity, property) pair the adapter can emit:
    palette (recolor), extent (reshape), position (move).  Each
    EntityStateChanged event arriving on the same step as a recorded
    ``_last_action`` produces one CausalClaim proposal.

    Why this miner exists separately from PropertyObservedMiner: the
    latter only records *that* a property has a new value; it does
    not link the change to any cause.  Causal linkage requires a
    second observation type (the just-executed action) and a
    different claim shape.

    Repetition strengthens credence (same trigger + same effect
    canonical key on subsequent firings).  A different post-action
    outcome competes via the standard hypothesis-store competing-
    claims mechanism.

    Domain-agnostic: no game-specific palette / cell knowledge.  Any
    adapter that emits EntityStateChanged events benefits.
    """

    name = "miner:EntityActionCausal"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return
        # Resolve abstraction-level entity ids via VisualStore (when
        # present in ws.agent).  Each EntityStateChanged event yields
        # one claim per abstraction level — bitmap-keyed claims stay
        # in-instance, shape- and topo-keyed claims transfer across
        # color and color+scale variants of the same shape (the
        # cross-level / cross-game transfer property).  When no
        # VisualStore is registered, falls back to single-level
        # mining keyed by the event's raw entity_id.
        from .visual_store import abstraction_keys_for
        store = ws.agent.get("_visual_store")
        trigger = ActionJustTaken(action_id=str(last_action))
        for evt in events:
            if not isinstance(evt, EntityStateChanged):
                continue
            for level, eid in abstraction_keys_for(store, evt.entity_id):
                effect = EntityInState(
                    entity_id = eid,
                    property  = evt.property,
                    value     = evt.new,
                )
                claim = CausalClaim(
                    trigger = trigger,
                    effect  = effect,
                )
                _store.propose(
                    ws,
                    claim  = claim,
                    source = f"{self.name}:{level}",
                    scope  = self.default_scope,
                    step   = step,
                )


# ---------------------------------------------------------------------------
# TransitionMiner
# ---------------------------------------------------------------------------


class TransitionMiner(Miner):
    """Detect (pre-position, action, post-position) triples from
    :class:`AgentMoved` events and propose matching
    :class:`TransitionClaim`\\s.

    This is the miner that populates the planner's transition model.
    Without it the planner has nothing to BFS over, so the engine
    reduces to pure exploration until enough transitions accumulate
    through trial and error.

    Extension path: handle resource-changing transitions (via
    :class:`ResourceChanged`) and property-changing transitions
    (via :class:`EntityStateChanged` caused by a specific action).
    Phase 4 keeps it positional because that's the minimum viable
    model for grid / locomotion domains.
    """

    name = "miner:Transition"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        # Look for an AgentMoved in this step's events.  If the last
        # action was recorded on ws.agent['_last_action'], pair them.
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return

        for evt in events:
            if not isinstance(evt, AgentMoved):
                continue
            claim = TransitionClaim(
                action = str(last_action),
                pre    = AtPosition(tuple(evt.from_pos), entity_id="agent"),
                post   = AtPosition(tuple(evt.to_pos), entity_id="agent"),
            )
            _store.propose(
                ws,
                claim  = claim,
                source = self.name,
                scope  = self.default_scope,
                step   = step,
            )


# ---------------------------------------------------------------------------
# ManipulationTransitionMiner
# ---------------------------------------------------------------------------


class ManipulationTransitionMiner(Miner):
    """Mint :class:`TransitionClaim`\\s for action-caused state changes on
    ANY entity — not just the controlled actor.

    :class:`TransitionMiner` above captures the agent's own *position*
    transitions (``AgentMoved`` -> ``AtPosition`` pre/post).  This miner
    covers the manipulation case its docstring names as the extension path:
    an :class:`EntityStateChanged` co-occurring with an action becomes
    ``TransitionClaim(action, pre=old-state, post=new-state)``.  These are
    the operators :func:`forward_model.regress` inverts (``post`` entails a
    goal -> surface ``action`` + ``pre`` as the next sub-goal), so the planner
    can reason **backward** over what actions do to manipulated objects
    ([P15](../docs/DURABLE_PRINCIPLES.md), :doc:`docs/SPEC_forward_model.md`).

    Minimal by design.  The precondition captured is the changed entity's own
    prior state.  Richer **relational** preconditions (``same_row``,
    ``clear_column``, ``co_displacement`` that held when the effect fired) are
    tightened by the specialisation-on-contradiction loop
    (:class:`SurpriseMiner`) when the bare transition's prediction fails: the
    model starts over-general and gains preconditions as evidence
    contradicts it (P1), rather than being hand-specified.

    Distinct from :class:`EntityActionCausalMiner`, which mints a
    :class:`CausalClaim` (trigger->effect) from the same observation: that is
    a forward correlational record; this is the **invertible** planner-facing
    operator.  Both are legitimate and compete only within their own type.
    """

    name = "miner:ManipulationTransition"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return
        for evt in events:
            if not isinstance(evt, EntityStateChanged):
                continue
            claim = TransitionClaim(
                action = str(last_action),
                pre    = EntityInState(evt.entity_id, evt.property, evt.old),
                post   = EntityInState(evt.entity_id, evt.property, evt.new),
            )
            _store.propose(
                ws,
                claim  = claim,
                source = self.name,
                scope  = self.default_scope,
                step   = step,
            )


# ---------------------------------------------------------------------------
# FutilePatternMiner
# ---------------------------------------------------------------------------


class FutilePatternMiner(Miner):
    """Detect "action in context yields no observable change" patterns.

    When an action is executed but no :class:`AgentMoved`,
    :class:`EntityStateChanged`, or :class:`ResourceChanged` event is
    produced, the miner proposes a :class:`TransitionClaim` with
    identical pre and post conditions — effectively "this action is
    a no-op in this state".  The planner avoids no-op actions
    automatically because their BFS doesn't advance.

    Rationale: wall-banging (repeatedly trying actions that don't work)
    is one of the biggest wastes of episode budget.  Detecting futile
    patterns early lets the planner prune them from its action space.
    """

    name = "miner:FutilePattern"

    # Event classes that constitute "something happened"
    _SIGNIFICANT_EVENT_TYPES: Tuple = (AgentMoved, EntityStateChanged)

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return
        # Significant change observed this step?
        if any(isinstance(e, self._SIGNIFICANT_EVENT_TYPES) for e in events):
            return
        # No change — propose a futile-transition claim
        position = tuple(ws.agent.get("position") or ())
        if not position:
            return
        pre = AtPosition(position, entity_id="agent")
        claim = TransitionClaim(
            action = str(last_action),
            pre    = pre,
            post   = pre,  # no-op
        )
        _store.propose(
            ws,
            claim  = claim,
            source = self.name,
            scope  = self.default_scope,
            step   = step,
        )


# ---------------------------------------------------------------------------
# ActionEffectMiner
# ---------------------------------------------------------------------------


class ActionEffectMiner(Miner):
    """Mint :class:`CausalClaim`\\s linking an action to the frame-delta
    it produces.

    Reads two engine-maintained facts every step:

    * ``ws.agent['_last_action']`` — the action id just executed
      (set by the episode runner after :meth:`Adapter.execute`).
    * ``ws.last_frame_delta`` — the structural diff of the pre/post
      frames (set by the runner after observation ingest; see
      :mod:`cognitive_os.frame_diff`).

    When both are present, proposes::

        CausalClaim(trigger = ActionJustTaken(action_id),
                    effect  = FrameChangedPattern(cells_changed, bbox))

    Credence and other tunables are read from
    ``EngineConfig.action_probe`` so that a single clean probe will
    typically commit the claim on the first trial.  Repeated probes of
    the same action that yield *different* delta signatures land as
    canonical competitors on the same ``(ActionJustTaken, *)`` key and
    the hypothesis store reconciles them via ordinary evidence updates.

    Rationale.  Without this miner the engine has no forward model of
    its own action space on any domain where ``AgentMoved`` events are
    not emitted (which includes ARC-AGI-3).  Pairing ``ActionJustTaken``
    with a delta-derived effect yields a *first-principles* action
    model that is domain-agnostic: any symbolic action that causes any
    observable frame change will be described here.

    Empty deltas (action produced no observable change) are a
    legitimate outcome and are recorded iff
    ``ActionProbeConfig.require_non_empty_delta`` is ``False``.
    """

    name = "miner:ActionEffect"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        # Fast guards.
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return
        delta = getattr(ws, "last_frame_delta", None)
        if delta is None:
            return

        # Central config — no hardcoded numbers in this miner.
        cfg_probe = _action_probe_cfg(ws)
        if cfg_probe is None or not cfg_probe.enabled:
            return
        if cfg_probe.require_non_empty_delta and getattr(delta, "is_empty", True):
            return

        cells = int(getattr(delta, "cells_changed", 0))
        bbox  = getattr(delta, "bbox", None)
        if bbox is not None:
            bbox = tuple(bbox)

        claim = CausalClaim(
            trigger = ActionJustTaken(str(last_action)),
            effect  = FrameChangedPattern(cells_changed=cells, bbox=bbox),
        )
        _store.propose(
            ws,
            claim             = claim,
            source            = self.name,
            scope             = self.default_scope,
            step              = step,
            initial_credence  = cfg_probe.initial_claim_point,
            rationale         = (f"probe: action={last_action} "
                                 f"cells_changed={cells} bbox={bbox}"),
        )


# ---------------------------------------------------------------------------
# RegionMotionMiner
# ---------------------------------------------------------------------------


class RegionMotionMiner(Miner):
    """Mint :class:`CausalClaim`\\s whose effect is a :class:`RegionMotion`.

    Runs alongside :class:`ActionEffectMiner` and consumes the same
    engine-maintained facts:

    * ``ws.agent['_last_action']`` — the action just executed.
    * ``ws.last_frame_delta``      — the structural diff.

    But where ``ActionEffectMiner`` records the *absolute* bounding
    box of the delta (exact reproduction, brittle across episodes),
    this miner lifts each DeltaRegion whose before/after pattern is a
    translation into a direction-only claim::

        CausalClaim(
            trigger = ActionJustTaken(action_id),
            effect  = RegionMotion(colour, background, dr_sign, dc_sign),
        )

    The lifted effect has no absolute coordinates — its canonical key
    is ``(RegionMotion, colour, background, dr_sign, dc_sign)`` — so
    two probes of the same action that move a colour-9 block upward
    from different starting positions both support the *same* claim.
    That is the property that makes action-learning transfer across
    episodes and across problem instances.

    The magnitude of the motion is intentionally discarded here.  A
    later refinement layer (not in this phase) can learn
    magnitude-specific sub-claims if the evidence supports them; the
    engine's generalisation ladder prefers abstract-first so that
    absolute brittleness does not leak into the canonical keys.

    Rationale.  Every translation-based mechanic in a grid domain
    (agent locomotion, block-push, gravity drops, ball bounces)
    reduces to a sign-pair ``(dr, dc)`` attached to a colour-pair
    ``(object, background)``.  Any domain whose frames are 2-D grids
    gets this miner for free; the engine stays domain-agnostic.
    """

    name = "miner:RegionMotion"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        last_action = ws.agent.get("_last_action")
        if last_action is None:
            return
        delta = getattr(ws, "last_frame_delta", None)
        if delta is None:
            return
        regions = getattr(delta, "regions", ()) or ()
        if not regions:
            return

        cfg_probe = _action_probe_cfg(ws)
        if cfg_probe is None or not cfg_probe.enabled:
            return

        # One CausalClaim per distinct (colour, background, dr, dc)
        # signature observed this step.  Multiple regions whose
        # translations share a signature (e.g. two blocks moving down
        # together) collapse to one claim per step — the store takes
        # care of evidence accumulation across steps.
        seen: Set[Tuple] = set()
        for region in regions:
            motion = extract_region_motion(region, delta)
            if motion is None:
                continue
            colour, background, dr_sign, dc_sign = motion
            sig = (colour, background, dr_sign, dc_sign)
            if sig in seen:
                continue
            seen.add(sig)

            claim = CausalClaim(
                trigger = ActionJustTaken(str(last_action)),
                effect  = RegionMotion(
                    colour     = colour,
                    background = background,
                    dr_sign    = int(dr_sign),
                    dc_sign    = int(dc_sign),
                ),
            )
            _store.propose(
                ws,
                claim             = claim,
                source            = self.name,
                scope             = self.default_scope,
                step              = step,
                initial_credence  = cfg_probe.initial_claim_point,
                rationale         = (f"motion: action={last_action} "
                                     f"colour={colour} bg={background} "
                                     f"dr={dr_sign} dc={dc_sign}"),
            )


# ---------------------------------------------------------------------------
# ControlledEntityMiner
# ---------------------------------------------------------------------------


class ControlledEntityMiner(Miner):
    """Identify *which* sprite is agent-controlled by pattern-matching
    the committed action-effect evidence.

    Signal.  If at least :attr:`min_actions` distinct action ids have
    committed a ``CausalClaim`` whose effect is a :class:`RegionMotion`
    on the **same** ``(colour, background)`` pair, **and** those
    motions span at least :attr:`min_distinct_directions` distinct
    ``(dr_sign, dc_sign)`` tuples, then propose a
    :class:`ControlledActorClaim(colour, background)`.

    Why these thresholds.  A *passive* object that happened to move
    under some actions by coincidence would typically move in a
    single repeated direction (a timed animation, a gravity drop)
    and would not respond to most actions.  A *controlled* actor is
    the unique object whose motion vector is a direct function of
    the action id — so requiring multi-action, multi-direction
    coverage filters out coincidental correlations.

    Defaults (``min_actions=2``, ``min_distinct_directions=2``) are
    deliberately permissive: two actions moving a sprite in two
    orthogonal directions is already strong evidence, and small
    action spaces (e.g. a 2-button domain) should still produce the
    claim.  Tighter thresholds can be set by the caller if a domain
    has many spurious movers.

    Runs once per step (cheap — only scans committed hypotheses), so
    the claim commits the moment the Nth directional claim does.
    """

    name = "miner:ControlledEntity"

    min_actions:             int = 2
    min_distinct_directions: int = 2

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        cfg_probe = _action_probe_cfg(ws)
        if cfg_probe is None or not cfg_probe.enabled:
            # Tie the miner's lifecycle to the action-probe subsystem:
            # if probing is off, there are no systematically-mined
            # directional claims to aggregate, so the miner has no
            # grounds to propose anything.
            return

        # Aggregate committed directional evidence per (colour, bg).
        per_signature: Dict[Tuple[Any, Any], Dict[str, Set[Tuple[int, int]]]] = {}
        for h in _store.committed(ws):
            claim = h.claim
            if not isinstance(claim, CausalClaim):
                continue
            trig = claim.trigger
            eff  = claim.effect
            if not isinstance(trig, ActionJustTaken):
                continue
            if not isinstance(eff, RegionMotion):
                continue
            key = (eff.colour, eff.background)
            by_action = per_signature.setdefault(key, {})
            by_action.setdefault(trig.action_id, set()).add(
                (int(eff.dr_sign), int(eff.dc_sign))
            )

        # For every (colour, bg) signature that clears both thresholds,
        # propose the ControlledActorClaim.  The store dedups, so firing
        # every step is harmless — this is the ordinary "propose and let
        # the store reconcile" pattern.
        for (colour, background), by_action in per_signature.items():
            n_actions      = len(by_action)
            all_directions = set().union(*by_action.values())
            if n_actions     < self.min_actions: continue
            if len(all_directions) < self.min_distinct_directions: continue

            claim = ControlledActorClaim(colour=colour, background=background)
            _store.propose(
                ws,
                claim             = claim,
                source            = self.name,
                scope             = self.default_scope,
                step              = step,
                # Commit on first detection — the pattern is structural
                # (N actions x M directions) and not itself noisy; the
                # evidence is the aggregated committed sub-claims, not
                # a single observation.
                initial_credence  = cfg_probe.initial_claim_point,
                rationale         = (f"actor: colour={colour} bg={background} "
                                     f"actions={sorted(by_action)} "
                                     f"dirs={sorted(all_directions)}"),
            )


def _action_probe_cfg(ws: WorldState):
    """Look up ``ws.config.action_probe``; returns ``None`` if unset.

    Kept separate so ``ActionEffectMiner.step`` reads like prose — and
    so any future helper that needs probe tunables can reuse this
    lookup without re-implementing the defensive ``None`` handling.
    """
    cfg = getattr(ws, "config", None)
    if cfg is None:
        return None
    return getattr(cfg, "action_probe", None)


# ---------------------------------------------------------------------------
# SelfLocalizationMiner
# ---------------------------------------------------------------------------


class SelfLocalizationMiner(Miner):
    """Populate ``ws.agent['position']`` from the committed
    :class:`ControlledActorClaim`, grounded in observed motion.

    Why this exists.  Many engine subsystems (goal achievement via
    :class:`AtPosition`, planner path computation, curiosity goals
    that want to move toward or away from a location) need to know
    where the agent *is right now*.  Some adapters supply that; many
    don't.  For any domain where the adapter provides segmented
    entities but no explicit "agent position", the engine can derive
    it from the actor claim once self-identification has fired.

    This is the first capability that **consumes** a committed claim
    rather than producing one.  It closes a small but important loop:
    self-identification → self-localization → positional goals become
    actionable.

    Dual-domain check.  Same mechanism in robotics: given a
    ``ControlledActorClaim`` on the end-effector's visual signature,
    the miner provides live end-effector position without any
    adapter-specific hook.

    Strategy, in priority order.  A frame can contain many entities
    of the actor's colour — HUD glyphs, twin decorations, inert
    palette duplicates.  Colour alone is a poor filter; the
    disambiguator that actually works is **motion**.

    1. **Motion-grounded (primary).**  If ``ws.last_frame_delta``
       contains a region whose
       :func:`frame_diff.extract_region_motion` signature matches
       the committed actor claim's ``(colour, background)``, the
       arriving-cells centroid of that region IS the agent's current
       position.  This is behaviourally exact: the sprite that just
       moved in response to the last action is, by definition, the
       thing under our control.

    2. **Continuity (fallback when no motion this step).**  If step
       N had no matching motion (e.g. we fired into a wall and the
       agent didn't move), pick the colour-matched entity whose
       centroid is closest to the previously-known
       ``ws.agent['position']``.  "Agent is where agent was" is the
       right prior for a stationary step.

    3. **No cold-start guessing.**  Without either motion evidence
       or a prior position, the miner does nothing.  Localizing from
       area heuristics would pin the agent to whichever colour-
       matched blob happens to be largest — often a HUD bar in
       game frames.  It is better to wait one action-causing step
       than to localize onto the wrong entity and then pollute
       continuity forever.

    The miner never clobbers a position it previously set if zero
    matching signals appear in the current frame: prior position
    remains available across transient occlusions.

    Runs once per step; no-op until the actor claim commits.  Cheap:
    O(committed hypotheses) + O(delta.regions) + O(entities).
    """

    name = "miner:SelfLocalization"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        actor_claims = [
            h for h in _store.committed(ws)
            if isinstance(h.claim, ControlledActorClaim)
        ]
        if not actor_claims:
            return

        # Highest-credence wins if multiple (rare but possible during
        # refinement splits).
        actor = max(actor_claims, key=lambda h: h.credence.point).claim

        # --- Strategy 1: motion-grounded ------------------------------
        motion_pos = _motion_grounded_position(ws, actor)
        if motion_pos is not None:
            ws.agent["position"]         = motion_pos
            ws.agent["_agent_entity_id"] = _nearest_entity_id(
                ws, motion_pos, actor.colour,
            )
            return

        # --- Strategy 2: continuity via prior position ----------------
        prior = ws.agent.get("position")
        if prior is None:
            # No motion this step and no prior — cognitively silent.
            # Wait for the first post-commit action that causes
            # movement before committing to a position.
            return

        try:
            pr, pc = float(prior[0]), float(prior[1])
        except (TypeError, ValueError, IndexError):
            return

        candidates = []
        for ent in ws.entities.values():
            if ent.properties.get("colour") != actor.colour:
                continue
            pos = _entity_position(ent)
            if pos is not None:
                candidates.append((ent, pos))

        if not candidates:
            # Transient absence: preserve prior position silently.
            return

        ent, (r, c) = min(
            candidates,
            key=lambda ep: (ep[1][0] - pr) ** 2 + (ep[1][1] - pc) ** 2,
        )
        ws.agent["position"]         = (float(r), float(c))
        ws.agent["_agent_entity_id"] = ent.id


def _motion_grounded_position(
    ws:    WorldState,
    actor: ControlledActorClaim,
) -> Optional[Tuple[float, float]]:
    """If the last frame-delta contains a region whose extracted
    motion matches the actor's ``(colour, background)`` signature,
    return the arriving-cells centroid (where the sprite landed).
    Returns ``None`` otherwise.

    When *multiple* regions match the actor signature in one frame
    (e.g. the player moves in the same frame as an unrelated
    same-colour sprite, or a translation artefact region coexists
    with the real motion), picking the first in ``delta.regions`` is
    order-dependent and teleports the tracked agent.  Disambiguate:

    * If a prior position is known, prefer the candidate whose
      arriving-cells centroid is closest to it (physical continuity
      dominates — the real player moved at most one action's worth).
    * Otherwise (cold start, first post-commit motion), prefer the
      candidate with the most arriving cells (the larger sprite is
      far more likely the player than a small glyph/artefact).
    """
    delta = getattr(ws, "last_frame_delta", None)
    if delta is None or not delta.regions:
        return None

    # Build the changed-cell index once; reused across every candidate
    # region below.
    index = {cell: i for i, cell in enumerate(delta.changed_cells)}

    # Collect (centroid, arriving_count) for every region matching the
    # actor signature.  Arriving-count doubles as a proxy for sprite
    # size in the cold-start fallback.
    candidates: List[Tuple[Tuple[float, float], int]] = []
    for region in delta.regions:
        motion = extract_region_motion(region, delta)
        if motion is None:
            continue
        sprite_colour, bg, _dr, _dc = motion
        if sprite_colour != actor.colour or bg != actor.background:
            continue

        # Anchor on cells where background → ACTOR.COLOUR specifically,
        # not on every ``bg → anything`` cell in the region.  A composite
        # sprite (e.g. colour-12 header + colour-9 body) would otherwise
        # include both arrivals, producing a full-sprite centroid.  But
        # the continuity fallback in SelfLocalizationMiner.step() anchors
        # on the colour-matched ENTITY's centroid (body-only, since
        # entities are segmented per colour).  If these two conventions
        # disagree by even one row, a wall-bang step — where motion is
        # None and Strategy 2 re-anchors to the body centroid — produces
        # a spurious +1 row "drift" with zero actual motion, which then
        # contaminates every downstream motion-model observation.  Fix:
        # both paths anchor on body-only arrivals, so they agree.
        rows: List[float] = []
        cols: List[float] = []
        for cell in region.cells:
            i = index.get(cell)
            if i is None:
                continue
            if (delta.before_values[i] == actor.background
                    and delta.after_values[i] == actor.colour):
                rows.append(float(cell[0]))
                cols.append(float(cell[1]))
        if not rows:
            continue
        centroid = (sum(rows) / len(rows), sum(cols) / len(cols))
        candidates.append((centroid, len(rows)))

    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0][0]

    # Multiple matching regions in one frame — disambiguate.
    prior = ws.agent.get("position")
    pr: Optional[float] = None
    pc: Optional[float] = None
    if prior is not None:
        try:
            pr = float(prior[0])
            pc = float(prior[1])
        except (TypeError, ValueError, IndexError):
            pr = pc = None

    if pr is not None and pc is not None:
        best = min(
            candidates,
            key=lambda cnd: (cnd[0][0] - pr) ** 2 + (cnd[0][1] - pc) ** 2,
        )
    else:
        # Cold start: prefer the larger sprite.  Ties broken by first
        # occurrence, which is deterministic for a given delta.
        best = max(candidates, key=lambda cnd: cnd[1])
    return best[0]


def _nearest_entity_id(
    ws:     WorldState,
    pos:    Tuple[float, float],
    colour: Any,
) -> Optional[str]:
    """Helper: return the id of the colour-matched entity whose
    centroid is closest to ``pos``.  Returns ``None`` if no entity
    of that colour is present.  Pure book-keeping so downstream code
    knows which segmented entity corresponds to the agent.
    """
    best_id: Optional[str] = None
    best_d2 = float("inf")
    for ent in ws.entities.values():
        if ent.properties.get("colour") != colour:
            continue
        ep = _entity_position(ent)
        if ep is None:
            continue
        d2 = (ep[0] - pos[0]) ** 2 + (ep[1] - pos[1]) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_id = ent.id
    return best_id


def _entity_position(ent) -> Optional[Tuple[float, float]]:
    """Best-effort positional extraction from an :class:`EntityModel`.

    Precedence matches :func:`goal_forest._position_for_entity` so
    that a Goal's achievement check and the self-localization miner
    agree on where the agent is.  Returns ``None`` if no property
    yields a usable (row, col) pair.
    """
    p = ent.properties.get("position")
    if p is not None:
        try:
            return (float(p[0]), float(p[1]))
        except (TypeError, ValueError, IndexError):
            pass
    c = ent.properties.get("centroid")
    if c is not None:
        try:
            return (float(c[0]), float(c[1]))
        except (TypeError, ValueError, IndexError):
            pass
    b = ent.properties.get("bbox")
    if b is not None and len(b) == 4:
        try:
            r0, c0, r1, c1 = (float(v) for v in b)
            return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
        except (TypeError, ValueError):
            pass
    return None


# ---------------------------------------------------------------------------
# ActorTransitionMiner
# ---------------------------------------------------------------------------


class ActorTransitionMiner(Miner):
    """Learn the controlled actor's state-conditioned transition model.

    After each step where the actor's pre- and post-positions are
    both known, mint one :class:`ActorTransitionClaim(pre_state,
    action_id, delta)` describing what happened.  The existing
    hypothesis-store dedup-by-canonical-key absorbs repeated
    identical observations (same `(pre_state, action, delta)` triple
    just accumulates ``supporting_steps``); conflicting observations
    (same pre_state and action, different delta) coexist as separate
    claims whose credences end up reflecting the empirical
    distribution, capturing stochastic actions for free.

    Why this is distinct from the existing action-model miners.

    * :class:`ActionEffectMiner` mints frame-level
      ``CausalClaim(ActionJustTaken, FrameChangedPattern)`` — the
      absolute bbox of whatever changed on the frame.  It is
      *unconditioned on the agent's state* and therefore cannot
      answer "what does ACTION1 do when I am here vs. there?".

    * :class:`RegionMotionMiner` mints direction-only
      ``CausalClaim(ActionJustTaken, RegionMotion)`` — colour and
      sign pair with no coordinate information, intentionally
      *transfer-friendly* across episodes but also blind to
      position-dependent variation within one episode.

    * ``ActorTransitionMiner`` is the *actor-conditioned* forward
      model: "from pre_state X, action A produced delta D on the
      thing under control."  It is the missing middle layer between
      the other two, and it is the primitive that lets downstream
      code (planner, explorer, curiosity) answer location-specific
      planning questions like "which actions will actually move me
      from here" — without the engine itself ever hard-coding the
      concept of a wall, a frontier, or an obstacle.  Those are
      interpretations of the learned model, not ingredients of it.

    How it is not "wall-finding".  This miner does not mint
    anomaly-flagged claims, does not search for "walls", does not
    know the word.  It records the observed ``(pre, action, delta)``
    triple — including the triple where ``delta`` is ``(0, 0)``
    because the agent did not move — and lets the downstream
    consumer decide whether zero-delta observations are interesting
    (in an ls20-style grid, yes; in an open-field shooter, ignored).
    Games without boundaries never produce zero-delta claims; games
    without position never produce any claim at all.  Both
    degeneracies are graceful.

    State plumbing.  The miner reads ``ws.agent['_last_action']``
    (set by the episode runner after :meth:`Adapter.execute`) and
    ``ws.agent['position']`` (populated by
    :class:`SelfLocalizationMiner` — which must therefore run
    *before* this miner in ``default_miners()``).  It caches the
    previous step's position in ``ws.agent['_prev_position']`` (the
    underscore prefix makes it survive ``_ingest_observation``'s
    public-state refresh) and compares across steps to compute
    ``delta``.

    Edge cases.
      - No prior position (first step after commit): record current
        position into ``_prev_position`` and return.  Cannot mint a
        claim because there is no "pre" to key on.
      - No current position (transient absence): skip this step;
        preserve ``_prev_position`` untouched so continuity-based
        mining resumes when the actor reappears.
      - No ``_last_action``: skip (can happen on engine-side
        synthetic steps).

    Dual-domain check.  Identical mechanism on a robot arm: the
    miner records ``(joint_config, motor_command, Δconfig)``
    observations as the arm operates.  A command that usually moves
    the arm producing Δ == 0 lands as a zero-delta
    ``ActorTransitionClaim`` — downstream code can interpret that as
    joint-limit contact, obstacle contact, or actuator fault
    depending on the domain.  The miner itself stays agnostic.

    Runs once per step; no-op until self-localization is live.
    O(1) work per step — one dict read, one subtraction, one claim
    proposal.
    """

    name = "miner:ActorTransition"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        current = ws.agent.get("position")
        if current is None:
            # Transient absence — preserve _prev_position untouched
            # so continuity resumes when the actor reappears.
            return

        try:
            cr, cc = float(current[0]), float(current[1])
        except (TypeError, ValueError, IndexError):
            return

        prev = ws.agent.get("_prev_position")
        last_action = ws.agent.get("_last_action")

        # First step after localization commits — no pre-state yet.
        if prev is None or last_action is None:
            ws.agent["_prev_position"] = (cr, cc)
            return

        try:
            pr, pc = float(prev[0]), float(prev[1])
        except (TypeError, ValueError, IndexError):
            ws.agent["_prev_position"] = (cr, cc)
            return

        delta = (cr - pr, cc - pc)
        pre_state = (pr, pc)

        _store.propose(
            ws,
            claim             = ActorTransitionClaim(
                pre_state = pre_state,
                action_id = str(last_action),
                delta     = delta,
            ),
            source            = self.name,
            scope             = self.default_scope,
            step              = step,
            rationale         = (f"observed transition: pre={pre_state} "
                                 f"action={last_action} delta={delta}"),
        )

        ws.agent["_prev_position"] = (cr, cc)


# ---------------------------------------------------------------------------
# MotionModelMiner
# ---------------------------------------------------------------------------


class MotionModelMiner(Miner):
    """Aggregate :class:`ActorTransitionClaim` evidence into a
    position-independent :class:`MotionModelClaim` per action.

    Why this miner exists — the planner gap.  :class:`ActorTransitionClaim`
    records absolute-position transitions; each one is keyed on its
    ``pre_state``.  A 20-step episode visiting 16 distinct cells
    produces 16 non-overlapping claims, none of which tell the
    planner how to proceed from the 17th (never-visited) cell.  The
    planner's positional BFS needs a "per-action stride" that
    generalises; without it, :func:`planner._plan_atom` returns None
    on every ``AtPosition`` goal.

    What it does.  After each step, walk every committed or
    contested ``ActorTransitionClaim`` in the store; for each
    ``action_id``, find the modal non-zero delta (the delta seen
    most often).  Propose that as a ``MotionModelClaim(action_id,
    delta)`` with credence computed from how dominant the mode is
    vs. alternatives.  The existing canonical-key dedup absorbs
    repeated proposals.

    Dominance threshold.  Propose a claim only when the modal delta
    accounts for at least ``min_dominance`` of non-zero observations
    for that action AND at least ``min_observations`` observations
    support it.  Defaults: 0.6 dominance, 3 observations — this is a
    *floor before credence begins accruing*, not a commit threshold.
    Credence grows continuously from the source prior once the floor
    is cleared, and contradicting observations demote via normal
    credence dynamics.

    Why ``min_observations=3`` default.  "One observation is never
    evidence of a rule."  A single probe can be a coincidence, a
    wall-blocked no-op, or a context-dependent effect; acting on it
    as though the motor stride is known wedges the planner the
    moment the probe was unrepresentative (see
    ``SPEC_continuous_commitment.md``).  Three gives the miner a
    shape to commit to while keeping the time-to-first-claim
    tractable on a short 4-action probe cycle.  Stochastic or noisy
    domains can raise this further via the constructor.  The engine
    consumes hypotheses by credence, not by a binary commit flag —
    the planner's ``PlannerConfig.min_credence`` threshold decides
    when a proposed motion model becomes plannable; observations
    beyond the floor continue to raise credence.

    Zero-delta handling.  Zero-delta ``ActorTransitionClaim``s
    (walls, boundary contact) are excluded from the motion-model
    aggregation.  They remain as position-specific overrides in the
    ``ActorTransitionClaim`` layer; the planner consults both (see
    :func:`planner._plan_atom`).

    Stochasticity.  For a stochastic action where two deltas occur
    roughly equally often, neither reaches the dominance threshold
    and no MotionModelClaim commits.  The planner falls back to
    per-position ``ActorTransitionClaim`` lookups.  This is
    deliberate: better to plan conservatively from positions we've
    seen than to commit to a coin-flip stride.

    Scope.  ``ScopeKind.EPISODE`` is the default — the motion model
    is re-mined each episode until cross-episode persistence for
    this claim type lands (whitelist extension).
    """

    def __init__(self,
                 *,
                 min_observations: int   = 3,
                 min_dominance:    float = 0.6) -> None:
        self.min_observations = int(min_observations)
        self.min_dominance    = float(min_dominance)
        self.default_scope    = Scope(kind=ScopeKind.EPISODE)

    @property
    def name(self) -> str:
        return "miner:MotionModel"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        # Collect non-zero ActorTransitionClaim observations grouped
        # by action_id.  We count BOTH committed and contested claims
        # because ActorTransitionClaim commits slowly (2+ identical
        # observations required), but a single observation is already
        # informative for aggregation.
        from collections import Counter

        per_action: Dict[str, Counter] = {}
        for h in ws.hypotheses.values():
            if not isinstance(h.claim, ActorTransitionClaim):
                continue
            delta = h.claim.delta
            # Reject zero-delta (wall/contact) from motion model.
            try:
                dr, dc = float(delta[0]), float(delta[1])
            except (TypeError, ValueError, IndexError):
                continue
            if dr == 0.0 and dc == 0.0:
                continue
            action_id = str(h.claim.action_id)
            bucket = per_action.setdefault(action_id, Counter())
            # Count raw observations, not distinct canonical keys.  A
            # deterministic game will visit the same (pre, action,
            # delta) triple repeatedly, which under canonical-key
            # dedup collapses to a single ``ActorTransitionClaim``
            # whose ``supporting_steps`` lengthens on each re-entry.
            # If the miner only counted hypotheses, a cycling agent
            # with ``min_observations=3`` could never clear the
            # floor for any action — each action's claim count
            # would stay at 1 forever.  Weighting by supporting
            # steps recognises the repeats as the evidence they
            # are, while still demanding *distinct observation
            # events* before promoting a motor stride.
            weight = len(getattr(h, "supporting_steps", ())) or 1
            bucket[(dr, dc)] += weight

        for action_id, counter in per_action.items():
            if not counter:
                continue
            (modal_delta, modal_count), *rest = counter.most_common()
            total = sum(counter.values())
            if total < self.min_observations:
                continue
            dominance = modal_count / total
            if dominance < self.min_dominance:
                continue
            # No force-commit.  Initial credence comes from the
            # miner source prior (0.60); each subsequent matching
            # observation grows credence through the store's
            # exact-duplicate support path.  Credence is continuous —
            # the planner consumes this claim through
            # ``PlannerConfig.min_credence``, which sits below the
            # global commit threshold so the planner engages early
            # while still demanding more than a single observation.
            _store.propose(
                ws,
                claim            = MotionModelClaim(
                    action_id = action_id,
                    delta     = modal_delta,
                ),
                source           = self.name,
                scope            = self.default_scope,
                step             = step,
                rationale        = (f"modal delta {modal_delta} "
                                    f"({modal_count}/{total} obs, "
                                    f"dom={dominance:.2f})"),
            )


# ---------------------------------------------------------------------------
# SurpriseMiner
# ---------------------------------------------------------------------------


class SurpriseMiner(Miner):
    """Detect observations that contradict committed hypotheses.

    On each step, walk committed :class:`PropertyClaim` and
    :class:`TransitionClaim` hypotheses and check them against the
    latest events.  When a committed claim is contradicted (the
    store's :func:`event_evidence_for_claim` returns ``False``),
    emit a :class:`SurpriseEvent` so downstream consumers — the
    Mediator (``EXPLAIN_SURPRISE`` question), the refinement layer
    (``specialize_on_contradiction``), and the runner's replan
    trigger — can react.

    The miner does *not* demote the contradicted hypothesis itself —
    that happens through the normal credence-update pipeline during
    the same step.  SurpriseMiner's contribution is purely
    surfacing the event so higher-level subsystems know "a
    committed prediction just failed", which is different from the
    slow cumulative credence drift the store already handles.

    Rationale: contradiction-driven learning is the engine's
    equivalent of Claude-Code-style iterative debugging.  Without
    explicit surprise detection, the system would silently slide
    through broken predictions instead of pausing to consult the
    Mediator or specialise.
    """

    name = "miner:Surprise"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        committed_hyps = _store.committed(ws)
        if not committed_hyps:
            return
        new_surprises: List[SurpriseEvent] = []

        for h in committed_hyps:
            for evt in events:
                verdict = _store.event_evidence_for_claim(evt, h.claim, ws)
                if verdict is False:
                    # A committed claim was contradicted.
                    new_surprises.append(SurpriseEvent(
                        step     = step,
                        expected = h.claim.canonical_key(),
                        actual   = _event_signature(evt),
                        context  = (f"committed hypothesis {h.id} "
                                    f"contradicted by {type(evt).__name__}"),
                    ))
                    break  # one surprise per hypothesis per step

        # Append surprise events to the latest observation in history.
        # (The runner appends observations; we piggyback on the last.)
        if new_surprises and ws.observation_history:
            ws.observation_history[-1].events.extend(new_surprises)


def _event_signature(evt: Event) -> Any:
    """Compact, hashable-ish description of an event for surprise
    logging.  Intentionally loose: we just need something the
    Mediator can look at."""
    sig = {
        "type": type(evt).__name__,
        "step": getattr(evt, "step", None),
    }
    if hasattr(evt, "__dataclass_fields__"):
        for k in evt.__dataclass_fields__:
            if k != "step":
                sig[k] = getattr(evt, k)
    return sig


# ---------------------------------------------------------------------------
# ContactMiner
# ---------------------------------------------------------------------------


class ContactMiner(Miner):
    """Emit :class:`ContactEvent`\\s when the controlled actor moves
    onto cells that were previously painted with a non-background,
    non-self colour.

    What counts as contact.  A cell is a contact cell if, within
    ``ws.last_frame_delta``:

    * ``after_value == actor.colour`` — the actor is there now, AND
    * ``before_value != actor.colour`` — the actor wasn't there, AND
    * ``before_value != actor.background`` — it wasn't empty space either.

    In other words, the actor painted over something that wasn't
    itself and wasn't the blank canvas.  Whatever ``before_value``
    was — an obstacle, a pickup, a hazard, a goal marker — is the
    ``other_colour`` reported in the event.

    Dedup.  If the actor is a multi-pixel sprite straddling several
    cells, it can land on multiple cells with the same
    ``other_colour`` in one step.  We emit ONE ``ContactEvent`` per
    distinct ``other_colour`` per step (keyed by the first-seen
    cell), not one per cell.  This keeps the event stream proportional
    to the number of *distinct* colour contacts, not pixel count.

    Why this is not a claim.  Contact is a per-step transient
    observation, not a persistent belief.  The event stream is
    already the protocol for per-step facts that miners and the
    Mediator consume.  Persistent beliefs about *what contact with
    colour X means* (pickup? goal? hazard?) belong in a future
    second-order claim minted by a correlation miner that pairs
    ``ContactEvent`` with reward / terminal / score events.

    Why this is not hard-coded game knowledge.  The miner never
    assigns meaning to any colour.  It records the geometric fact
    of overlap; domains where overlap is never meaningful (e.g. a
    game with no interactable entities) simply never have
    downstream consumers of the event and the mechanism costs
    nothing.

    Dual-domain.  Robotics: the end-effector pose overlaps a point
    cloud point formerly assigned to an object.  Same geometric
    signal, same protocol (emit an event, let downstream logic
    interpret).

    Runs once per step; no-op until the actor claim commits and a
    non-empty motion delta is present.  Cost: O(delta.changed_cells).
    """

    name = "miner:Contact"

    def step(self,
             ws:     WorldState,
             events: List[Event],
             step:   int) -> None:
        actor_claims = [
            h for h in _store.committed(ws)
            if isinstance(h.claim, ControlledActorClaim)
        ]
        if not actor_claims:
            return
        actor = max(actor_claims, key=lambda h: h.credence.point).claim

        delta = getattr(ws, "last_frame_delta", None)
        if delta is None or getattr(delta, "is_empty", True):
            return

        # Map other_colour -> first cell where that contact was seen
        contacts: Dict[Any, Tuple[int, int]] = {}
        for i, cell in enumerate(delta.changed_cells):
            after  = delta.after_values[i]
            before = delta.before_values[i]
            if after != actor.colour:
                continue
            if before == actor.colour or before == actor.background:
                continue
            if before not in contacts:
                contacts[before] = (int(cell[0]), int(cell[1]))

        if not contacts:
            return

        new_events = [
            ContactEvent(
                step         = step,
                actor_colour = actor.colour,
                other_colour = other,
                cell         = cell,
            )
            for other, cell in contacts.items()
        ]
        if ws.observation_history:
            ws.observation_history[-1].events.extend(new_events)


# ---------------------------------------------------------------------------
# Default miner suite
# ---------------------------------------------------------------------------


def default_miners() -> List[Miner]:
    """Return the canonical Phase 4 miner suite in run order.

    Order matters: :class:`PropertyObservedMiner` and
    :class:`TransitionMiner` run first so their output is visible
    to the credence-update pass in the same step;
    :class:`FutilePatternMiner` runs next, also contributing
    hypotheses; :class:`SurpriseMiner` runs last because it needs
    access to the just-updated WorldState to evaluate committed
    hypotheses against events.
    """
    return [
        PropertyObservedMiner(),
        TransitionMiner(),
        ManipulationTransitionMiner(),   # entity (non-agent) action-caused
                                         # state transitions -> invertible
                                         # operators for forward_model.regress
        ActionEffectMiner(),
        RegionMotionMiner(),
        ControlledEntityMiner(),
        SelfLocalizationMiner(),
        ActorTransitionMiner(),   # must follow SelfLocalizationMiner:
                                  # depends on ws.agent['position']
        MotionModelMiner(),       # aggregates ActorTransitionClaim
                                  # deltas into a position-independent
                                  # motion model for planner BFS;
                                  # must follow ActorTransitionMiner
        ContactMiner(),           # emits ContactEvent for actor-entity
                                  # overlap; must follow actor claim
                                  # commit but has no dep on position
        FutilePatternMiner(),
        SurpriseMiner(),
    ]
