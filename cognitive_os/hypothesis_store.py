"""Hypothesis store — lifecycle operations over WorldState.hypotheses.

This module provides the operations that constitute the learning loop at
the hypothesis level: proposing new hypotheses, deduplicating exact
matches, linking canonical competitors, applying evidence-driven
credence updates, decaying stale claims, and pruning abandoned ones.

All functions are **module-level and stateless** — they read and mutate
:class:`WorldState` in place.  No hidden indexes, no class instance
state; the store's entire state lives in ``ws.hypotheses`` plus the
monotonic counter ``ws._next_hypothesis_id``.  This keeps
:class:`WorldState` snapshottable for persistence and testing.

Evidence matching
-----------------
``update_credence_from_events`` dispatches on ``(event_type, claim_type)``
to decide whether a given event supports, contradicts, or is neutral
with respect to each active hypothesis.  Phase 2 implements matchers
for the three claim types that directly consume Events:

* :class:`PropertyClaim`   — EntityStateChanged
* :class:`TransitionClaim` — AgentMoved + action taken
* :class:`CausalClaim`     — trigger evaluated at step t-delay,
                              effect evaluated at step t

Evidence for the remaining claim types comes from sources outside
Events:

* :class:`RelationalClaim` / :class:`StructureMappingClaim`
      — Observer answers (wired in Phase 4+)
* :class:`ConstraintClaim`
      — planner experience (Phase 3+)
* :class:`StrategyClaim`
      — branch-outcome statistics from executed Plans (Phase 3+)

Those matchers return ``None`` (neutral) here and will be extended in
the phases that introduce their evidence sources.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from . import telemetry_schema as _tel
from .claims import (
    ActorTransitionClaim,
    Claim,
    CausalClaim,
    ConstraintClaim,
    PropertyClaim,
    RelationalClaim,
    StrategyClaim,
    StructureMappingClaim,
    TransitionClaim,
)
from .conditions import Condition
from .credence import (
    Credence,
    apply_decay,
    link_competitor,
    unlink_competitor,
    update_on_contradict,
    update_on_support,
)
from .telemetry import emit_from_ws as _emit
from .types import (
    AgentDied,
    AgentMoved,
    EntityStateChanged,
    Event,
    Hypothesis,
    ResourceChanged,
    Scope,
    ScopeKind,
    SurpriseEvent,
    WorldState,
)


# ---------------------------------------------------------------------------
# Internal helpers — ID generation, indexing
# ---------------------------------------------------------------------------


def _next_id(ws: WorldState, prefix: str = "h") -> str:
    """Allocate a new, never-before-used hypothesis ID.

    IDs are monotonic — pruned IDs are never re-used.  This matters
    because lattice links (``parent_id`` / ``child_ids``) would
    otherwise silently re-target if a pruned ID were reallocated to a
    different claim.
    """
    n = ws._next_hypothesis_id
    ws._next_hypothesis_id = n + 1
    return f"{prefix}{n}"


def by_canonical_key(ws: WorldState, canonical_key: tuple) -> List[Hypothesis]:
    """Return all active hypotheses whose claim shares the given
    canonical key.  These are structural peers / competitors."""
    return [h for h in ws.hypotheses.values()
            if h.claim.canonical_key() == canonical_key]


def by_full_key(ws: WorldState, full_key: tuple) -> Optional[Hypothesis]:
    """Return the active hypothesis with the given full key, if any.
    At most one should exist (invariant maintained by :func:`propose`)."""
    for h in ws.hypotheses.values():
        if h.claim.full_key() == full_key:
            return h
    return None


# ---------------------------------------------------------------------------
# Public: propose a hypothesis
# ---------------------------------------------------------------------------


def propose(ws:                WorldState,
            claim:              Claim,
            source:             str,
            scope:              Scope,
            step:               int,
            *,
            rationale:          Optional[str] = None,
            expires_at:         Optional[int] = None,
            initial_credence:   Optional[float] = None,
            parent_id:          Optional[str] = None) -> str:
    """Propose a hypothesis; returns the hypothesis ID.

    Dedup / competition pipeline:

    1. **Exact duplicate** (same ``full_key``) — merge evidence by
       treating this proposal as a supporting confirmation of the
       existing hypothesis; return the existing ID unchanged.

    2. **Canonical competitor** (same ``canonical_key``, different
       ``full_key``) — register the new hypothesis and link it
       bidirectionally with every existing competitor via
       ``Credence.competing``.  Evidence that supports one member of
       the group can inform the others (implemented by miners in
       Phase 4).

    3. **Novel** — register a fresh hypothesis with source-prior
       credence.

    Parameters
    ----------
    ws
        WorldState to mutate.
    claim
        The Claim to register.
    source
        Provenance tag, e.g. ``"miner:FutilePattern"``,
        ``"user:correction"``.  Used to look up the initial credence
        prior from :class:`SourcePriors`.
    scope
        Temporal and structural scope of the hypothesis.
    step
        Current step (used as ``created_at`` and ``last_confirmed``).
    rationale
        Optional human-readable explanation, for audit and logging.
    expires_at
        Optional step at which the hypothesis should be re-evaluated.
    initial_credence
        If given, overrides the source-prior lookup.  Rarely used;
        reserved for adapter-seeded claims where the adapter has
        specific confidence that differs from the generic source prior.
    parent_id
        If this proposal is a specialisation of an existing
        hypothesis, the parent's ID.  The function will wire the
        lattice links (``parent.child_ids``, ``new.parent_id``).
    """
    # --- (1) exact duplicate — merge evidence ---
    existing_exact = by_full_key(ws, claim.full_key())
    if existing_exact is not None:
        cfg = _credence_cfg(ws)
        old_point = existing_exact.credence.point
        old_weight = existing_exact.credence.evidence_weight
        s = _source_strength(source)
        existing_exact.credence = update_on_support(
            existing_exact.credence, step, cfg,
            source_strength=s)
        existing_exact.supporting_steps.append(step)
        new_point = existing_exact.credence.point
        new_weight = existing_exact.credence.evidence_weight
        if new_point != old_point:
            _emit(ws, _tel.HypothesisCredenceUpdated(
                hypothesis_id          = existing_exact.id,
                old_credence           = float(old_point),
                new_credence           = float(new_point),
                evidence_delta         = float(new_point - old_point),
                reason                 = f"support:duplicate-propose:{source}",
                direction              = "support",
                source_kind            = "propose",
                source_detail          = f"duplicate-propose:{source}",
                source_strength        = float(s),
                learning_rate          = float(cfg.learning_rate),
                evidence_weight_before = float(old_weight),
                evidence_weight_after  = float(new_weight),
                triggering_event_step  = None,
            ), subject=existing_exact.id, step=step)
        return existing_exact.id

    # --- initial credence from source prior (or override) ---
    if initial_credence is None:
        initial_credence = _initial_credence_from_source(ws, source)
    cred = Credence(point=initial_credence, last_confirmed=step)

    # --- (2) canonical competitors — link bidirectionally ---
    canonical_peers = by_canonical_key(ws, claim.canonical_key())
    # These are all competitors of the new hypothesis.

    # --- (3) create and install ---
    h_id = _next_id(ws)
    h = Hypothesis(
        id                  = h_id,
        claim               = claim,
        credence            = cred,
        scope               = scope,
        source              = source,
        supporting_steps    = [step],      # the proposal itself is a datum
        contradicting_steps = [],
        expires_at          = expires_at,
        parent_id           = parent_id,
        child_ids           = [],
        created_at          = step,
        rationale           = rationale,
    )

    # Link competitors
    if canonical_peers:
        peer_ids = tuple(p.id for p in canonical_peers)
        h.credence = replace(h.credence, competing=peer_ids)
        for peer in canonical_peers:
            peer.credence = link_competitor(peer.credence, h_id)

    ws.hypotheses[h_id] = h

    # Wire lattice link up to parent, if any
    if parent_id is not None and parent_id in ws.hypotheses:
        parent = ws.hypotheses[parent_id]
        if h_id not in parent.child_ids:
            parent.child_ids.append(h_id)

    # Telemetry: a fresh hypothesis has been installed.  Emit Added
    # first, then Specialised if this is a refinement-parented child —
    # order lets the client materialise the node before linking it.
    scope_kind = scope.kind.value if hasattr(scope, "kind") else str(scope)
    _emit(ws, _tel.HypothesisAdded(
        hypothesis_id    = h_id,
        claim_type       = type(claim).__name__,
        scope_kind       = scope_kind,
        source           = source,
        initial_credence = float(initial_credence),
        canonical_key    = repr(claim.canonical_key()),
    ), subject=h_id, step=step)
    if parent_id is not None and parent_id in ws.hypotheses:
        _emit(ws, _tel.HypothesisSpecialised(
            parent_id            = parent_id,
            child_id             = h_id,
            added_condition_kind = type(claim).__name__,
        ), subject=h_id, step=step)

    return h_id


# ---------------------------------------------------------------------------
# Public: evidence-driven credence updates
# ---------------------------------------------------------------------------


def update_credence_from_events(ws:     WorldState,
                                events: List[Event],
                                step:   int) -> Dict[str, List[str]]:
    """Scan all active hypotheses against a batch of events.

    For each (hypothesis, event) pair, consult the event→claim matcher
    to decide whether the event supports, contradicts, or is neutral
    toward the hypothesis.  Apply the corresponding credence update.

    Returns a summary dict:

    * ``"newly_committed"`` — hypothesis IDs that crossed the
      commit threshold during this call.
    * ``"newly_demoted"``   — hypothesis IDs that dropped below the
      commit threshold during this call.
    * ``"supported"``       — (h_id, event_step) pairs that received
      supporting evidence.
    * ``"contradicted"``    — (h_id, event_step) pairs that received
      contradicting evidence.

    The planner reads ``newly_demoted`` to decide whether a current
    plan's ``assumptions`` are still valid; the refinement layer reads
    ``contradicted`` to decide whether to propose specialisations.
    """
    cfg = _credence_cfg(ws)
    before_committed = {h_id for h_id, h in ws.hypotheses.items()
                        if h.credence.is_committed(cfg)}

    supported:     List[Tuple[str, int]] = []
    contradicted:  List[Tuple[str, int]] = []

    for h_id, h in list(ws.hypotheses.items()):
        for evt in events:
            verdict = event_evidence_for_claim(evt, h.claim, ws)
            evt_kind = type(evt).__name__
            evt_step = getattr(evt, "step", step)
            if verdict is True:
                old_point  = h.credence.point
                old_weight = h.credence.evidence_weight
                h.credence = update_on_support(
                    h.credence, step, cfg,
                    source_strength=1.0)
                h.supporting_steps.append(evt_step)
                supported.append((h_id, evt_step))
                if h.credence.point != old_point:
                    _emit(ws, _tel.HypothesisCredenceUpdated(
                        hypothesis_id          = h_id,
                        old_credence           = float(old_point),
                        new_credence           = float(h.credence.point),
                        evidence_delta         = float(h.credence.point - old_point),
                        reason                 = f"support:{evt_kind}",
                        direction              = "support",
                        source_kind            = "event",
                        source_detail          = evt_kind,
                        source_strength        = 1.0,
                        learning_rate          = float(cfg.learning_rate),
                        evidence_weight_before = float(old_weight),
                        evidence_weight_after  = float(h.credence.evidence_weight),
                        triggering_event_step  = int(evt_step) if evt_step is not None else None,
                    ), subject=h_id, step=step)
            elif verdict is False:
                old_point  = h.credence.point
                old_weight = h.credence.evidence_weight
                h.credence = update_on_contradict(
                    h.credence, step, cfg,
                    strength=1.0)
                h.contradicting_steps.append(evt_step)
                contradicted.append((h_id, evt_step))
                if h.credence.point != old_point:
                    _emit(ws, _tel.HypothesisCredenceUpdated(
                        hypothesis_id          = h_id,
                        old_credence           = float(old_point),
                        new_credence           = float(h.credence.point),
                        evidence_delta         = float(h.credence.point - old_point),
                        reason                 = f"contradict:{evt_kind}",
                        direction              = "contradict",
                        source_kind            = "event",
                        source_detail          = evt_kind,
                        source_strength        = 1.0,
                        learning_rate          = float(cfg.learning_rate),
                        evidence_weight_before = float(old_weight),
                        evidence_weight_after  = float(h.credence.evidence_weight),
                        triggering_event_step  = int(evt_step) if evt_step is not None else None,
                    ), subject=h_id, step=step)
            # verdict is None: neutral — no change

    after_committed = {h_id for h_id, h in ws.hypotheses.items()
                       if h.credence.is_committed(cfg)}

    return {
        "newly_committed": sorted(after_committed - before_committed),
        "newly_demoted":   sorted(before_committed - after_committed),
        "supported":       supported,
        "contradicted":    contradicted,
    }


# ---------------------------------------------------------------------------
# Public: decay and pruning
# ---------------------------------------------------------------------------


def apply_staleness_decay_all(ws: WorldState, step: int) -> Dict[str, List[str]]:
    """Apply staleness decay to every active hypothesis.

    Returns the same commit/demote summary as
    :func:`update_credence_from_events` so the caller can detect plans
    that lost their supporting hypotheses to decay rather than
    contradiction.
    """
    cfg = _credence_cfg(ws)
    before_committed = {h_id for h_id, h in ws.hypotheses.items()
                        if h.credence.is_committed(cfg)}
    for h_id, h in list(ws.hypotheses.items()):
        old_point  = h.credence.point
        old_weight = h.credence.evidence_weight
        h.credence = apply_decay(h.credence, step, cfg)
        if h.credence.point != old_point:
            _emit(ws, _tel.HypothesisCredenceUpdated(
                hypothesis_id          = h_id,
                old_credence           = float(old_point),
                new_credence           = float(h.credence.point),
                evidence_delta         = float(h.credence.point - old_point),
                reason                 = "decay:staleness",
                direction              = "decay",
                source_kind            = "decay",
                source_detail          = "staleness",
                source_strength        = 0.0,
                learning_rate          = float(cfg.decay_per_step),
                evidence_weight_before = float(old_weight),
                evidence_weight_after  = float(h.credence.evidence_weight),
                triggering_event_step  = None,
            ), subject=h_id, step=step)
    after_committed = {h_id for h_id, h in ws.hypotheses.items()
                       if h.credence.is_committed(cfg)}
    return {
        "newly_committed": sorted(after_committed - before_committed),
        "newly_demoted":   sorted(before_committed - after_committed),
    }


def prune_abandoned(ws: WorldState, step: int) -> List[str]:
    """Remove hypotheses whose credence has fallen at or below the
    abandon threshold.  Returns the list of pruned IDs.

    Lattice hygiene: when a pruned hypothesis has children, the
    children's ``parent_id`` is cleared to ``None`` (they become
    top-level rather than orphaned).  When it has parents, the pruned
    ID is removed from ``parent.child_ids``.  Competitor linkage is
    cleaned up in all peers' ``Credence.competing`` lists.
    """
    cfg = _credence_cfg(ws)
    pruned: List[str] = []

    for h_id, h in list(ws.hypotheses.items()):
        if h.credence.is_abandoned(cfg):
            reason = _retire_reason_with_provenance(h)
            _remove_with_cleanup(ws, h_id)
            pruned.append(h_id)
            _emit(ws, _tel.HypothesisRetired(
                hypothesis_id = h_id,
                reason        = reason,
            ), subject=h_id, step=step)

    return pruned


def clear_by_scope(
    ws:       WorldState,
    kind:     ScopeKind,
    *,
    step:     int = 0,
    reason:   Optional[str] = None,
) -> List[str]:
    """Retract every hypothesis whose ``scope.kind == kind``.

    Sibling to :func:`prune_abandoned` — same retraction semantics
    (lattice cleanup, telemetry emission), different selection rule:
    here we drop on scope match rather than on credence floor.

    Lifecycle hook for scope boundaries: when the runtime crosses a
    boundary that invalidates everything below it (e.g. a respawn
    crosses the ``LIFE`` boundary, retracting every life-scoped
    claim), the runtime calls this with the relevant ``kind``.  The
    enum's nominal hierarchy is documented on :class:`ScopeKind`;
    this function treats ``kind`` as a literal match — callers that
    want "kind X and everything narrower" must invoke for each level.
    Keeping it literal avoids hidden cascades.

    Returns the list of retracted hypothesis IDs for telemetry.
    Emits a :class:`HypothesisRetired` event per retraction with a
    headline reason that includes the scope kind so trace consumers
    can distinguish scope-clears from credence prunes.
    """
    retracted: List[str] = []
    headline = reason or f"scope_cleared:{kind.value}"

    for h_id, h in list(ws.hypotheses.items()):
        h_scope = getattr(h, "scope", None)
        if h_scope is None or getattr(h_scope, "kind", None) != kind:
            continue
        full_reason = f"{headline}; {_retire_reason_with_provenance(h)}"
        _remove_with_cleanup(ws, h_id)
        retracted.append(h_id)
        _emit(ws, _tel.HypothesisRetired(
            hypothesis_id = h_id,
            reason        = full_reason,
        ), subject=h_id, step=step)

    return retracted


def _retire_reason_with_provenance(h: "Hypothesis") -> str:
    """Build a retire reason that summarises the hypothesis's history.

    Consumes the same provenance data emitted to telemetry — counts of
    supporting and contradicting observations, plus the most recent
    contradicting step — so a downstream reader inspecting only
    HypothesisRetired events can see *why* the hypothesis fell, not
    just that it did. The trail in the per-subject envelope chain has
    the full story; this short reason is the headline.
    """
    n_sup = len(h.supporting_steps)
    n_con = len(h.contradicting_steps)
    last_con = h.contradicting_steps[-1] if h.contradicting_steps else None
    parts = ["abandoned",
             f"sup={n_sup}",
             f"con={n_con}"]
    if last_con is not None:
        parts.append(f"last_con_step={last_con}")
    if h.source:
        parts.append(f"src={h.source}")
    return "; ".join(parts)


def _remove_with_cleanup(ws: WorldState, h_id: str) -> None:
    """Delete the hypothesis and repair all structural references."""
    h = ws.hypotheses.pop(h_id, None)
    if h is None:
        return

    # Unlink from parent's child list
    if h.parent_id is not None:
        parent = ws.hypotheses.get(h.parent_id)
        if parent is not None and h_id in parent.child_ids:
            parent.child_ids.remove(h_id)

    # Orphan children (don't cascade delete — they may still be valid)
    for child_id in list(h.child_ids):
        child = ws.hypotheses.get(child_id)
        if child is not None and child.parent_id == h_id:
            child.parent_id = None

    # Unlink from competitors
    for peer_id in h.credence.competing:
        peer = ws.hypotheses.get(peer_id)
        if peer is not None:
            peer.credence = unlink_competitor(peer.credence, h_id)


# ---------------------------------------------------------------------------
# Public: queries
# ---------------------------------------------------------------------------


def committed(ws: WorldState) -> List[Hypothesis]:
    """Return hypotheses whose credence is at or above commit threshold.

    Equivalent to :meth:`WorldState.committed_hypotheses` but available
    at module level so callers can work symmetrically with other
    hypothesis_store operations.

    **Note.** Under the continuous-commitment regime
    (``SPEC_continuous_commitment.md``) this remains a soft debug
    label — useful for diagnostics and for components that genuinely
    want the "settled" subset — but the planner-facing path prefers
    :func:`above_credence`, which takes a caller-chosen floor so that
    partial evidence is available well before the full commit
    threshold is reached.  Do not treat ``committed()`` as the gate
    that decides whether a hypothesis is plannable.
    """
    cfg = _credence_cfg(ws)
    return [h for h in ws.hypotheses.values() if h.credence.is_committed(cfg)]


def above_credence(ws: WorldState, threshold: float) -> List[Hypothesis]:
    """Return hypotheses whose credence ``point`` is at or above
    ``threshold``.

    This is the planner-facing counterpart of :func:`committed`.
    Whereas ``committed`` uses the global
    :attr:`CredenceConfig.commit_threshold` (``0.85``) and encodes
    "the store regards this as settled", ``above_credence`` takes a
    caller-chosen floor and answers the softer question "is there
    enough evidence here to be *usable* for the decision I am about
    to make?".

    The planner uses this with :attr:`PlannerConfig.min_credence`
    (default ``0.5``) so that a motion-model claim with three
    confirming observations can drive BFS even though its credence
    is still below the global commit threshold.  Cautious callers
    can pass a higher threshold; exploratory callers can pass a
    lower one.  The continuity is what matters —
    ``SPEC_continuous_commitment.md`` P1: no binary commit gate.
    """
    floor = float(threshold)
    return [h for h in ws.hypotheses.values()
            if float(getattr(h.credence, "point", 0.0)) >= floor]


def active_wall_overrides(ws: WorldState) -> "set[Tuple[Tuple, str]]":
    """Set of ``(pre_pos, action_id)`` pairs for which a zero-delta
    :class:`ActorTransitionClaim` ("action produced no movement from
    this position") is currently *active* — i.e. its credence is at
    or above :attr:`PlannerConfig.wall_override_credence_floor`
    (default ``0.60``).

    Shared helper used by both the planner (to prune BFS edges) and
    the explorer (to skip actions that will produce no effect at the
    agent's current position).  Identical eligibility rule in both
    call sites; factored here because it's a property of the
    hypothesis store, not of either subsystem alone.

    GAP 14 introduced the soft-floor rule; GAP 15 lifts it out of
    :mod:`planner` so the explorer can share it.  Robotics analogue:
    "end-effector bumped this pose" — relevant to both motion
    planning (don't plan through it) and motor babbling (don't
    retry it for exploration's sake).
    """
    from typing import Set as _Set
    from .config import PlannerConfig as _PlannerConfig
    if ws.config is not None and hasattr(ws.config, "planner"):
        floor = float(getattr(ws.config.planner,
                              "wall_override_credence_floor", 0.60))
    else:
        floor = _PlannerConfig().wall_override_credence_floor
    out: _Set[Tuple[Tuple, str]] = set()
    for h in ws.hypotheses.values():
        if not isinstance(h.claim, ActorTransitionClaim):
            continue
        if float(getattr(h.credence, "point", 0.0)) < floor:
            continue
        delta = h.claim.delta
        try:
            dr, dc = float(delta[0]), float(delta[1])
        except (TypeError, IndexError, ValueError):
            continue
        if dr != 0.0 or dc != 0.0:
            continue
        try:
            pre = (h.claim.pre_state[0], h.claim.pre_state[1])
        except (TypeError, IndexError):
            continue
        out.add((pre, str(h.claim.action_id)))
    return out


def active_wall_overrides_with_credence(
    ws: WorldState,
) -> "dict[Tuple[Tuple, str], float]":
    """Same eligibility rule as :func:`active_wall_overrides`, but returns
    a mapping ``(pre_pos, action_id) -> max_credence`` instead of a set.

    When multiple zero-delta ActorTransitionClaims clear the floor for
    the same ``(pre_pos, action_id)`` (rare but possible — e.g. one
    KB-loaded and one live-observed), the highest-credence claim's
    point estimate is recorded.  Callers that need to compare wall-
    credence against a competing non-zero-delta claim use this richer
    return type; callers that only need set-membership semantics
    continue to use :func:`active_wall_overrides`.

    Added 2026-04-27 as part of Phase 6c step 2.7's fix for the
    planner's wall-vs-pos-specific reduction: the BFS at
    ``planner.py`` was unconditionally favoring walls when both a
    wall and a successful-move claim existed at the same
    ``(pre, action)`` — letting stale KB-loaded walls override fresh
    in-session move observations.  With this helper, the planner can
    compare credences and pick the higher-confidence claim.
    """
    from .config import PlannerConfig as _PlannerConfig
    if ws.config is not None and hasattr(ws.config, "planner"):
        floor = float(getattr(ws.config.planner,
                              "wall_override_credence_floor", 0.60))
    else:
        floor = _PlannerConfig().wall_override_credence_floor
    out: "dict[Tuple[Tuple, str], float]" = {}
    for h in ws.hypotheses.values():
        if not isinstance(h.claim, ActorTransitionClaim):
            continue
        cred = float(getattr(h.credence, "point", 0.0))
        if cred < floor:
            continue
        delta = h.claim.delta
        try:
            dr, dc = float(delta[0]), float(delta[1])
        except (TypeError, IndexError, ValueError):
            continue
        if dr != 0.0 or dc != 0.0:
            continue
        try:
            pre = (h.claim.pre_state[0], h.claim.pre_state[1])
        except (TypeError, IndexError):
            continue
        key = (pre, str(h.claim.action_id))
        prev = out.get(key)
        if prev is None or cred > prev:
            out[key] = cred
    return out


def active_position_specific_motion(
    ws: WorldState,
    *,
    include_zero: bool = False,
) -> "dict[Tuple[Tuple, str], Tuple[Tuple, str]]":
    """Map ``(pre_pos, action_id) -> (delta, hypothesis_id)`` for
    :class:`ActorTransitionClaim`\\s whose credence is at or above
    :attr:`PlannerConfig.wall_override_credence_floor` (default
    ``0.60``).

    When multiple ActorTransitionClaims commit for the same
    ``(pre_pos, action_id)`` (stochastic actions where the same pose
    sometimes gives different deltas), we pick the one with the
    highest credence.  Ties broken by insertion order.  Matches
    :func:`planner._committed_motion_model`'s reduction rule so the
    two layers stay mutually consistent.

    By default **zero-delta entries are excluded** — those are the
    wall-override case, which consumers already handle separately via
    :func:`active_wall_overrides`.  Set ``include_zero=True`` to get
    the union (not used by the default planner path, but useful for
    test assertions and possible future consumers).

    Why this helper exists (GAP 17).  Before this, the planner's BFS
    only used the *open-field* :class:`MotionModelClaim` aggregate as
    the motion delta from any position, with the per-position
    :class:`ActorTransitionClaim` layer consumed only for zero-delta
    wall overrides.  Environments with partial boundary "snaps"
    (ACTION4 from (48,21) gives ``(-1,+5)`` not ``(0,+5)``) caused
    BFS to mis-predict state after that step, producing plans that
    looked fine on paper but the agent's real trajectory diverged —
    no correction mechanism fired because the motion model was
    position-blind.  This helper lets the planner prefer the more
    specific (learned) local transition when available.

    Robotics analogue.  Near fixtures or joint limits the Jacobian
    deviates from the nominal motor model — the controller must
    consult a pose-specific model when one is available.  Same
    mechanism in both domains.
    """
    from typing import Dict as _Dict
    from .config import PlannerConfig as _PlannerConfig
    if ws.config is not None and hasattr(ws.config, "planner"):
        floor = float(getattr(ws.config.planner,
                              "wall_override_credence_floor", 0.60))
    else:
        floor = _PlannerConfig().wall_override_credence_floor

    # Two-pass: gather qualifying claims, then reduce per
    # (pre, action) to the single highest-credence delta.
    # Using (best_cred, delta, hid) triples so the reduction is
    # self-contained.
    best: _Dict[Tuple[Tuple, str], Tuple[float, Tuple, str]] = {}
    for h in ws.hypotheses.values():
        if not isinstance(h.claim, ActorTransitionClaim):
            continue
        cred = float(getattr(h.credence, "point", 0.0))
        if cred < floor:
            continue
        try:
            dr, dc = float(h.claim.delta[0]), float(h.claim.delta[1])
        except (TypeError, IndexError, ValueError):
            continue
        if not include_zero and dr == 0.0 and dc == 0.0:
            continue
        try:
            pre = (h.claim.pre_state[0], h.claim.pre_state[1])
        except (TypeError, IndexError):
            continue
        key = (pre, str(h.claim.action_id))
        prev = best.get(key)
        if prev is None or cred > prev[0]:
            best[key] = (cred, (dr, dc), h.id)
    return {k: (v[1], v[2]) for k, v in best.items()}


def contested_groups(ws: WorldState) -> List[List[Hypothesis]]:
    """Group competing hypotheses.  Each returned inner list is a
    set of hypotheses that share a canonical key — these compete for
    the same phenomenon.  Singleton groups (only one hypothesis with
    a given canonical key) are excluded from the result.

    Used by the explorer to find discrimination-worthy exploration
    targets, and by generalisation miners to find parameter-learning
    situations that have converged.
    """
    groups: Dict[tuple, List[Hypothesis]] = {}
    for h in ws.hypotheses.values():
        key = h.claim.canonical_key()
        groups.setdefault(key, []).append(h)
    return [g for g in groups.values() if len(g) > 1]


# ---------------------------------------------------------------------------
# Internal: source strength + initial credence
# ---------------------------------------------------------------------------


def _source_strength(source: str) -> float:
    """How strong a piece of supporting evidence is, based on its source.

    A user correction confirming a hypothesis is stronger evidence than
    a speculative LLM proposal confirming the same claim.  This uses
    the same prior-strength convention as the source priors; a source
    prior of 0.9 means supporting evidence from that source carries
    weight 0.9 in the learning-rate multiplier.
    """
    # Delegate to SourcePriors-equivalent logic; values are in [0,1].
    return max(0.0, min(1.0, _prior_for_source(source)))


def _prior_for_source(source: str) -> float:
    """Wrapper so we can look up priors without importing EngineConfig.

    The three Oracle-related rows (``observer``, ``mediator``, ``oracle``)
    reflect dispatch granularity: ``observer`` is a narrow scene-grounded
    perception query (highest prior of the three because it's most
    constrained), ``mediator`` is a narrow semantic/inferential query
    (slightly lower because it's more interpretive), ``oracle`` is the
    undifferentiated umbrella term for monolithic LLM calls that haven't
    been narrowed yet (lowest of the three; matches the legacy default
    that historical ``tutor:*`` sources fell through to, before the
    2026-04-27 vocabulary cleanup).

    The ``kb`` row (added 2026-04-27) represents durable cross-session
    knowledge loaded from a persisted knowledge base.  Same prior tier
    as ``mediator`` (0.65): both encode prior knowledge with comparable
    reliability profiles.  The planner's
    :func:`active_position_specific_motion` and
    :func:`active_wall_overrides` use a 0.60 credence floor; KB-loaded
    claims at 0.65 clear it, so KB-seeded walls and portals become
    visible to the engine planner without a single in-session
    re-confirmation.  Without this row, ``kb:*`` sources fell through
    to the 0.5 default and the planner ignored persisted priors —
    surfaced by the Phase 6c step 2 engine-planner-audit when it
    showed the engine missing portal shortcuts that the adapter's
    ``_bfs_plan_cells`` was finding via ``portal_map``.
    """
    kind = source.split(":", 1)[0]
    return {
        "user":      0.95,
        "adapter":   0.80,
        "observer":  0.70,
        "mediator":  0.65,
        "kb":        0.65,
        "miner":     0.60,
        "oracle":    0.50,
        "analogy":   0.40,
        "llm":       0.30,
        "abductive": 0.25,
    }.get(kind, 0.5)


def _initial_credence_from_source(ws: WorldState, source: str) -> float:
    """Look up initial credence for a source, preferring the engine
    config's :class:`SourcePriors` if available, falling back to
    hard-coded defaults if no config is attached yet.

    The fallback exists because some test and construction paths build
    a :class:`WorldState` without an :class:`EngineConfig`.
    """
    if ws.config is not None and hasattr(ws.config, "source_priors"):
        return ws.config.source_priors.for_source(source)
    return _prior_for_source(source)


def _credence_cfg(ws: WorldState):
    """Return the :class:`CredenceConfig` from ``ws.config``, falling
    back to a default instance when config isn't attached (test paths).
    """
    if ws.config is not None and hasattr(ws.config, "credence"):
        return ws.config.credence
    from .config import CredenceConfig
    return CredenceConfig()


# ---------------------------------------------------------------------------
# Evidence matching — (event, claim) → True / False / None
# ---------------------------------------------------------------------------


def event_evidence_for_claim(evt:   Event,
                             claim: Claim,
                             ws:    WorldState) -> Optional[bool]:
    """Decide whether an event constitutes supporting (True),
    contradicting (False), or neutral (None) evidence for a claim.

    Dispatch table keyed on claim type.  Unknown combinations fall
    through to ``None`` (neutral) — the safe default for claim types
    whose evidence arrives from non-event sources (Observer answers,
    plan outcomes).
    """
    if isinstance(claim, PropertyClaim):
        return _evidence_for_property(evt, claim)
    if isinstance(claim, TransitionClaim):
        return _evidence_for_transition(evt, claim, ws)
    if isinstance(claim, CausalClaim):
        return _evidence_for_causal(evt, claim, ws)
    # RelationalClaim, StructureMappingClaim, ConstraintClaim,
    # StrategyClaim — evidence comes from other channels; neutral here.
    return None


def _evidence_for_property(evt:   Event,
                           claim: PropertyClaim) -> Optional[bool]:
    """PropertyClaim(entity, prop, value) matches EntityStateChanged
    events targeting the same (entity, property).

    * New value matches the claim's value → supporting.
    * New value differs from claim's value → contradicting.
    * Unrelated event → neutral.
    """
    if not isinstance(evt, EntityStateChanged):
        return None
    if evt.entity_id != claim.entity_id:
        return None
    if evt.property != claim.property:
        return None
    # (new value == claim.value) supports; otherwise contradicts
    return evt.new == claim.value


def _evidence_for_transition(evt:   Event,
                             claim: TransitionClaim,
                             ws:    WorldState) -> Optional[bool]:
    """TransitionClaim(action, pre, post) matches events that record
    the action's execution and its observed post-condition.

    Phase 2 implements a narrow matcher: an :class:`AgentMoved` event
    whose action name matches ``claim.action``, evaluated against the
    claim's ``post`` condition in the current WorldState.  The
    ``pre`` side is not re-checked here — the episode runner is
    responsible for only presenting this hypothesis with events where
    the action was actually invoked in a state satisfying ``pre``.

    Richer matchers (resource-changing transitions, entity-state
    transitions) will be layered in when the corresponding action
    event types are introduced.
    """
    # Hook for future extension: transition events not yet modelled in
    # the Event vocabulary fall through as neutral.
    if not isinstance(evt, AgentMoved):
        return None
    # In Phase 2 we don't have action labels on AgentMoved; treat any
    # AgentMoved as evidence only when claim.action is the convention
    # "MOVE" or unset.  Proper action-tagging arrives with the
    # adapter protocol in Phase 4.
    if claim.action not in ("MOVE", "*", ""):
        return None
    # Evaluate the post condition after the move.
    post_truth = claim.post.evaluate(ws)
    if post_truth is None:
        return None
    return post_truth


def _evidence_for_causal(evt:   Event,
                         claim: CausalClaim,
                         ws:    WorldState) -> Optional[bool]:
    """CausalClaim(trigger, effect, min_occurrences, delay) evidence.

    Phase 2 implements the simplest form: when the trigger condition
    is evaluable-and-true in the current WorldState, check whether the
    effect condition is evaluable-and-true.  The ``min_occurrences``
    and ``delay`` parameters are informational in Phase 2 and will be
    used by specialised miners in Phase 4 that track trigger counts
    across the observation history.

    * trigger true + effect true  → supporting.
    * trigger true + effect false → contradicting (hypothesis
      predicted an effect that did not occur).
    * trigger false or unknown    → neutral.

    The ``evt`` argument is currently only used as a "heartbeat" — we
    re-check the trigger/effect on any event, relying on the
    per-step invocation cadence in the runner.

    GAP 13 — tolerance bridge applied to the trigger.  Mediator-sourced
    ``AtPosition`` triggers arrive with ``tolerance=0`` (pixel-exact);
    the agent moves on a motor lattice of coarser step, so the exact
    pose is almost never visited and the raw ``evaluate`` returns False
    forever — meaning the "trigger true" gate below never opens and the
    disconfirmation pathway never fires, even for claims whose effect
    demonstrably fails to follow over many in-tolerance steps.  The
    goal-forest already bridges this at expansion time via
    ``_bridged_trigger_condition``; we apply the same bridge here so
    the evidence layer sees "the agent is effectively at the trigger"
    under the same geometric semantics the planner uses.  The bridge
    is a no-op (returns the original condition) when no tolerance is
    applicable, so non-position claims are unaffected.

    Robotics analogue.  A grasp CausalClaim(AtPose(handle), DoorOpen)
    demands the same semantics: "the arm is at the handle within reach
    tolerance; did the door open?"  Pixel/exact-pose matching on a
    real robot is meaningless; tolerance is the contract.
    """
    from .goal_forest import _bridged_trigger_condition
    effective_trigger, _note = _bridged_trigger_condition(claim.trigger, ws)
    trig = effective_trigger.evaluate(ws)
    if trig is not True:
        return None
    eff = claim.effect.evaluate(ws)
    if eff is None:
        return None
    return bool(eff)
