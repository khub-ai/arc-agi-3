"""Refinement — generalisation and specialisation of hypotheses.

Together with the evidence-driven credence updates in
:mod:`hypothesis_store`, these two operators form the core learning
loop of the engine:

    Miners propose specific hypotheses
            ↓
    Generalisation compresses recurring patterns into broader claims
            ↓
    Broader claims tested against new situations
            ↓
    Near-miss failure → Specialisation adds a guard condition
            ↓
    Refined claim validates or fails again → iterate

In terms of the hypothesis lattice (parent/child links), generalisation
creates *parents* (more general) and specialisation creates *children*
(more specific).  The store's pruning policy takes care of removing
subsumed children once a parent commits, or collapsing a failed parent
back onto its surviving children.

Phase 2 scope
-------------
* :func:`specialize_on_contradiction` — fully implemented for
  ``CausalClaim`` and ``TransitionClaim``.  Both tighten a
  condition (``trigger`` / ``pre``) by conjoining the negation of a
  caller-supplied ``contradicting_context``.  For other claim types
  the function returns ``None`` — specialisation of those forms
  requires extensions (entity patterns, property guards) that belong
  in later phases.

* :func:`detect_generalization_candidates` — identifies *groups* of
  hypotheses with shared structure that could profit from a more
  abstract parent.  It does not yet *construct* the parent; doing so
  requires an anti-unification / pattern mechanism that will be added
  alongside pattern-bearing Claim types in a later phase.  The
  function returns metadata (group of IDs plus the structural
  commonality) that a miner can act on once the machinery is present.

* Lattice utilities — :func:`link_parent_child`,
  :func:`prune_subsumed_children`.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from .claims import (
    CausalClaim,
    Claim,
    PropertyClaim,
    TransitionClaim,
)
from .conditions import (
    Condition,
    Conjunction,
    Negation,
)
from . import hypothesis_store as _store
from .types import (
    Hypothesis,
    Scope,
    WorldState,
)


# ---------------------------------------------------------------------------
# Specialisation
# ---------------------------------------------------------------------------


def specialize_on_contradiction(ws:                    WorldState,
                                hypothesis_id:         str,
                                contradicting_context: Condition,
                                step:                  int,
                                *,
                                source_suffix:         str = "specialised",
                                rationale:             Optional[str] = None) -> Optional[str]:
    """Propose a specialised child hypothesis that excludes the
    near-miss context.

    Parameters
    ----------
    ws
        WorldState to mutate.
    hypothesis_id
        ID of the parent hypothesis being specialised.  Typically a
        hypothesis whose credence was recently contradicted but which
        the engine still believes holds in most cases.
    contradicting_context
        A :class:`Condition` describing the contextual feature that
        distinguishes the contradicting observation from the parent's
        supporting observations.  The child's trigger/pre-condition
        becomes ``Conjunction(original, Negation(contradicting_context))``.
    step
        Current step.  Used as the child's ``created_at`` and
        ``last_confirmed``.
    source_suffix
        Appended to ``"refinement:"`` to form the child's source tag;
        default ``"specialised"``.
    rationale
        Optional human-readable explanation.

    Returns
    -------
    str | None
        The new child hypothesis ID, or ``None`` if the parent's claim
        type does not admit this form of specialisation (property
        claims and relational claims require patterns; those are
        handled by future phases).

    Notes
    -----
    The child inherits the parent's :attr:`Scope` and :attr:`source`
    (with ``source_suffix`` appended).  Its initial credence starts
    from an ``"abductive"`` prior — specialisation is a speculative
    structural inference that must earn credence through subsequent
    evidence.
    """
    parent = ws.hypotheses.get(hypothesis_id)
    if parent is None:
        return None

    # Build the specialised claim
    new_claim = _specialise_claim(parent.claim, contradicting_context)
    if new_claim is None:
        return None

    # Propose as a child.  Using ``abductive`` source prior because a
    # specialisation is a speculative structural refinement, not
    # confirmed evidence.
    child_source = f"refinement:{source_suffix}"
    # Initial credence: deliberately low — specialisations must earn it.
    child_id = _store.propose(
        ws,
        claim               = new_claim,
        source              = child_source,
        scope               = parent.scope,
        step                = step,
        rationale           = rationale or _auto_rationale(parent, contradicting_context),
        initial_credence    = ws.config.source_priors.abductive_speculation
                              if ws.config is not None and hasattr(ws.config, "source_priors")
                              else 0.25,
        parent_id           = parent.id,
    )
    return child_id


def _specialise_claim(claim: Claim,
                      guard: Condition) -> Optional[Claim]:
    """Produce a specialised variant of ``claim`` that additionally
    requires the negation of ``guard`` to hold.

    Only defined for :class:`CausalClaim` and :class:`TransitionClaim`;
    other claim types return ``None`` from this helper.
    """
    negated_guard = Negation(guard)

    if isinstance(claim, CausalClaim):
        # Tighten the trigger: trigger ∧ ¬guard → effect
        new_trigger = _conjoin(claim.trigger, negated_guard)
        return CausalClaim(
            trigger         = new_trigger,
            effect          = claim.effect,
            min_occurrences = claim.min_occurrences,
            delay           = claim.delay,
        )

    if isinstance(claim, TransitionClaim):
        # Tighten the pre-condition: action × (pre ∧ ¬guard) → post
        new_pre = _conjoin(claim.pre, negated_guard)
        return TransitionClaim(
            action = claim.action,
            pre    = new_pre,
            post   = claim.post,
        )

    # PropertyClaim, RelationalClaim, StructureMappingClaim,
    # ConstraintClaim, StrategyClaim — require extensions beyond
    # what Phase 2 provides.
    return None


def _conjoin(existing: Condition, extra: Condition) -> Condition:
    """Produce a canonicalised conjunction of two conditions.

    Flattens nested Conjunctions so that
    ``conjoin(A ∧ B, C) == Conjunction((A, B, C))`` rather than
    ``Conjunction((Conjunction((A, B)), C))``.  This keeps canonical
    keys stable across repeated specialisation rounds.
    """
    members: List[Condition] = []
    for cond in (existing, extra):
        if isinstance(cond, Conjunction):
            members.extend(cond.conditions)
        else:
            members.append(cond)
    return Conjunction(tuple(members))


def _auto_rationale(parent: Hypothesis, guard: Condition) -> str:
    """Construct a human-readable rationale string when the caller
    doesn't supply one."""
    return (f"Specialised from h={parent.id} after observation "
            f"contradicting in context {guard.canonical_key()}.")


# ---------------------------------------------------------------------------
# Generalisation candidates
# ---------------------------------------------------------------------------


def detect_generalization_candidates(
        ws:               WorldState,
        min_group_size:   int = 3,
        require_committed: bool = True) -> List[Dict]:
    """Identify groups of committed hypotheses that could be generalised.

    A *group* here is a set of hypotheses whose Claims share a
    structural sub-shape — for example, all ``PropertyClaim``\\s
    asserting ``(property='lethal', value=True)`` but differing in
    ``entity_id``.  A generalisation would replace the group with a
    single parent claim ranging over the varying field via a pattern
    or wildcard.

    Phase 2 does not *construct* the parent; doing so requires pattern-
    bearing variants of the claim types that are deferred to a later
    phase.  This function is the *detector* that prepares the input
    for the future constructor.

    Returns
    -------
    list[dict]
        Each dict describes one generalisation opportunity:

        * ``"kind"``       — the claim class name, e.g.
          ``"PropertyClaim"``.
        * ``"hypothesis_ids"`` — the member hypothesis IDs.
        * ``"shared"``     — a dict of field→value pairs shared
          across all members.
        * ``"varying"``    — the names of fields whose values differ.
        * ``"support"``    — total evidence weight (sum of members'
          evidence_weights), giving the caller a confidence signal.

    Parameters
    ----------
    min_group_size
        Minimum members required to form a candidate group.  Default 3.
        Below this, the commonality is usually coincidence.
    require_committed
        If True (default), only consider hypotheses above the commit
        threshold.  Set False when running diagnostic scans.
    """
    cfg = _credence_cfg(ws)
    pool = list(ws.hypotheses.values())
    if require_committed:
        pool = [h for h in pool if h.credence.is_committed(cfg)]

    candidates: List[Dict] = []
    # Group PropertyClaim hypotheses with shared (property, value)
    candidates.extend(_gen_candidates_property(pool, min_group_size))
    # Group CausalClaim hypotheses with shared effect
    candidates.extend(_gen_candidates_causal(pool, min_group_size))
    # Group TransitionClaim hypotheses with shared (action, post)
    candidates.extend(_gen_candidates_transition(pool, min_group_size))
    return candidates


def _gen_candidates_property(pool:            List[Hypothesis],
                             min_group_size:  int) -> List[Dict]:
    """Group committed PropertyClaims by (property, value); candidate
    iff the group has ≥ min_group_size members differing only in
    ``entity_id``."""
    groups: Dict[Tuple, List[Hypothesis]] = {}
    for h in pool:
        c = h.claim
        if isinstance(c, PropertyClaim):
            key = (c.property, _hashable(c.value))
            groups.setdefault(key, []).append(h)
    out: List[Dict] = []
    for (prop, val), members in groups.items():
        if len(members) >= min_group_size:
            out.append({
                "kind":           "PropertyClaim",
                "hypothesis_ids": [h.id for h in members],
                "shared":         {"property": prop, "value": val},
                "varying":        ["entity_id"],
                "support":        sum(h.credence.evidence_weight for h in members),
            })
    return out


def _gen_candidates_causal(pool:           List[Hypothesis],
                           min_group_size: int) -> List[Dict]:
    """Group committed CausalClaims by effect; candidate iff ≥
    min_group_size members share the same effect with varying
    triggers."""
    groups: Dict[Tuple, List[Hypothesis]] = {}
    for h in pool:
        c = h.claim
        if isinstance(c, CausalClaim):
            key = c.effect.canonical_key()
            groups.setdefault(key, []).append(h)
    out: List[Dict] = []
    for eff_key, members in groups.items():
        if len(members) >= min_group_size:
            out.append({
                "kind":           "CausalClaim",
                "hypothesis_ids": [h.id for h in members],
                "shared":         {"effect_canonical": eff_key},
                "varying":        ["trigger"],
                "support":        sum(h.credence.evidence_weight for h in members),
            })
    return out


def _gen_candidates_transition(pool:           List[Hypothesis],
                               min_group_size: int) -> List[Dict]:
    """Group committed TransitionClaims by (action, post)."""
    groups: Dict[Tuple, List[Hypothesis]] = {}
    for h in pool:
        c = h.claim
        if isinstance(c, TransitionClaim):
            key = (c.action, c.post.canonical_key())
            groups.setdefault(key, []).append(h)
    out: List[Dict] = []
    for (action, post_key), members in groups.items():
        if len(members) >= min_group_size:
            out.append({
                "kind":           "TransitionClaim",
                "hypothesis_ids": [h.id for h in members],
                "shared":         {"action": action, "post_canonical": post_key},
                "varying":        ["pre"],
                "support":        sum(h.credence.evidence_weight for h in members),
            })
    return out


# ---------------------------------------------------------------------------
# Lattice maintenance
# ---------------------------------------------------------------------------


def link_parent_child(ws: WorldState,
                      parent_id: str,
                      child_id:  str) -> None:
    """Explicitly link a parent/child pair, idempotently.

    :func:`specialize_on_contradiction` and other refinement operators
    call this implicitly; exposed for callers constructing lattice
    relationships by hand (e.g. tests, adapter-seeded taxonomies).
    """
    if parent_id == child_id:
        return
    parent = ws.hypotheses.get(parent_id)
    child  = ws.hypotheses.get(child_id)
    if parent is None or child is None:
        return
    if child.id not in parent.child_ids:
        parent.child_ids.append(child.id)
    child.parent_id = parent.id


def prune_subsumed_children(ws: WorldState) -> List[str]:
    """Remove children whose parent has committed and which are
    logically subsumed.

    A child is *subsumed* when its parent is committed and the child
    contributes no additional predictive power — in the current
    simple model, this is any child with credence below its parent.
    Pruning them keeps the store from growing stale
    post-specialisation when the parent turns out to have been the
    correct hypothesis all along.

    Returns the list of pruned child IDs.
    """
    cfg = _credence_cfg(ws)
    pruned: List[str] = []
    for p in list(ws.hypotheses.values()):
        if not p.credence.is_committed(cfg):
            continue
        for c_id in list(p.child_ids):
            child = ws.hypotheses.get(c_id)
            if child is None:
                continue
            if child.credence.point < p.credence.point:
                # Child offers no predictive lift; remove it.
                _store._remove_with_cleanup(ws, c_id)
                pruned.append(c_id)
    return pruned


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _hashable(value):
    """Coerce values to something hashable for dict-key use.

    Duplicates the logic from ``conditions._hashable`` but kept local
    to avoid importing private helpers across modules.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted(_hashable(v) for v in value))
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def _credence_cfg(ws: WorldState):
    """Return the CredenceConfig from ws.config, or a default."""
    if ws.config is not None and hasattr(ws.config, "credence"):
        return ws.config.credence
    from .config import CredenceConfig
    return CredenceConfig()
