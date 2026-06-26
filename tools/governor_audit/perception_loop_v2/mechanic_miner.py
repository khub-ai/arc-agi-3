"""Game-agnostic mechanic miner.

After each turn, the ExploratoryDriver records:
  - the action the actor chose (action_record)
  - the observed delta (delta_record)
  - the new WorldKnowledge state

This module looks for repeated trigger -> effect patterns across
the action+delta history and turns them into MechanicHypothesis
records with credence.  Credence accumulates as the same pattern is
re-observed; it decays when contradicted (an action that previously
caused effect X failed to cause it the next time).

The miner is GAME-AGNOSTIC — it knows nothing about specific games,
sprites, or rules.  It deals only in abstract trigger-effect
templates over the open vocabulary of action strings and observable
deltas (entity-appeared, entity-disappeared, entity-changed,
agent-moved-by-N-cells-in-direction-X, etc.).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from world_knowledge import (   # noqa: E402
    WorldKnowledge, MechanicHypothesis, BlockingClaim,
    DeltaRecord, ActionRecord,
)


# ---------------------------------------------------------------------------
# Trigger-effect signatures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerSignature:
    """Compact, game-agnostic representation of an action+context that
    might be causally responsible for an observed effect."""
    action: str                      # the action string
    agent_role: Optional[str] = None
        # the agent's role label at the moment of action (used to
        # capture state-dependent triggers like "agent is carrying
        # red" if role updates encode carry state)
    adjacent_entity_roles: tuple[str, ...] = ()
        # roles of entities adjacent to the agent BEFORE the action

    def as_str(self) -> str:
        parts = [f"action={self.action}"]
        if self.agent_role:
            parts.append(f"agent_role={self.agent_role}")
        if self.adjacent_entity_roles:
            parts.append(
                f"adjacent_roles={'|'.join(sorted(self.adjacent_entity_roles))}"
            )
        return ", ".join(parts)


@dataclass(frozen=True)
class EffectSignature:
    """Compact representation of an observed delta."""
    kind: str
        # one of: "agent_moved", "agent_stuck",
        # "entity_disappeared", "entity_appeared",
        # "entity_changed", "win_state_changed"
    entity_role: Optional[str] = None
        # for entity_* effects, the role of the affected entity
    direction: Optional[str] = None
        # for agent_moved, the cardinal direction (UP/DOWN/LEFT/RIGHT)

    def as_str(self) -> str:
        parts = [f"effect={self.kind}"]
        if self.entity_role:
            parts.append(f"target_role={self.entity_role}")
        if self.direction:
            parts.append(f"direction={self.direction}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Mining
# ---------------------------------------------------------------------------


def _ename(item):
    """Name of a delta entity entry.  The VLM may report entities_changed/
    appeared/disappeared either as bare name STRINGS or as richer DICTS like
    {"name": ..., "change": ...}.  Coerce to the hashable string name (or None)
    so world.entities.get(...) never sees an unhashable dict."""
    if isinstance(item, dict):
        return item.get("name")
    return item


def _classify_delta_effects(
    delta: DeltaRecord, world: WorldKnowledge,
) -> list[EffectSignature]:
    """Extract atomic effect signatures from one DeltaRecord."""
    out: list[EffectSignature] = []
    if delta.agent_moved and delta.inferred_action:
        out.append(EffectSignature(
            kind="agent_moved",
            direction=delta.inferred_action,
        ))
    elif not delta.agent_moved:
        out.append(EffectSignature(kind="agent_stuck"))
    for name in delta.entities_disappeared:
        rec = world.entities.get(_ename(name))
        role = rec.current_role if rec else "unknown"
        out.append(EffectSignature(
            kind="entity_disappeared", entity_role=role or "unknown",
        ))
    for name in delta.entities_appeared:
        rec = world.entities.get(_ename(name))
        role = rec.current_role if rec else "unknown"
        out.append(EffectSignature(
            kind="entity_appeared", entity_role=role or "unknown",
        ))
    for name in delta.entities_changed:
        rec = world.entities.get(_ename(name))
        role = rec.current_role if rec else "unknown"
        out.append(EffectSignature(
            kind="entity_changed", entity_role=role or "unknown",
        ))
    return out


def _adjacent_roles_before_action(
    world: WorldKnowledge, turn_before: int,
) -> tuple[str, ...]:
    """Roles of entities adjacent to the agent at the moment BEFORE
    the action was taken (i.e., at turn `turn_before`).  Uses the
    grid's cell_ticks (if available) to determine adjacency.  Falls
    back to empty if grid info missing."""
    agent = None
    for r in world.entities.values():
        if r.current_role == "agent":
            agent = r
            break
    if agent is None or agent.current_cell is None:
        return ()
    gi = world.grid_inference
    if gi is None or not gi.is_grid_based or gi.cell_ticks is None:
        return ()
    step = gi.cell_ticks
    ar, ac = agent.current_cell
    neighbours = {
        (ar - step, ac),       # up
        (ar + step, ac),       # down
        (ar, ac - step),       # left
        (ar, ac + step),       # right
    }
    out: list[str] = []
    for r in world.entities.values():
        if r is agent or r.current_cell is None:
            continue
        if tuple(r.current_cell) in neighbours:
            out.append(r.current_role or "unknown")
    return tuple(out)


def _hypothesis_id(trigger: TriggerSignature, effect: EffectSignature) -> str:
    return f"H[{trigger.as_str()} -> {effect.as_str()}]"


def mine_step(world: WorldKnowledge,
                action_record: ActionRecord,
                delta: DeltaRecord) -> list[str]:
    """After one action+delta pair, update the mechanic-hypothesis
    inventory in ``world``.  Returns the IDs of hypotheses that were
    created or updated this step (for the driver to log).

    Each effect observed becomes a candidate (trigger, effect)
    hypothesis.  Re-observed hypotheses gain credence; missing
    re-observations cause decay only when the same trigger DEFINITELY
    fired (not by mere passage of time)."""
    trigger = TriggerSignature(
        action=action_record.action,
        adjacent_entity_roles=_adjacent_roles_before_action(
            world, action_record.turn,
        ),
    )
    effects = _classify_delta_effects(delta, world)

    # Look up existing hypotheses by ID
    existing = {h.hypothesis_id: h for h in world.mechanic_hypotheses}
    touched: list[str] = []

    delta_idx = len(world.deltas_observed) - 1

    # For each effect observed: bump matching hypothesis or create new
    for eff in effects:
        hid = _hypothesis_id(trigger, eff)
        h = existing.get(hid)
        if h is None:
            h = MechanicHypothesis(
                hypothesis_id=hid,
                trigger=trigger.as_str(),
                effect=eff.as_str(),
                credence=0.4,   # first observation gives a starting credence
                supporting_observations=[delta_idx],
                contradicting_observations=[],
            )
            world.mechanic_hypotheses.append(h)
            existing[hid] = h
        else:
            h.supporting_observations.append(delta_idx)
            h.credence = min(1.0, h.credence + 0.15)
            if h.credence >= 0.85 and not h.promoted:
                h.promoted = True
        touched.append(hid)

    # CONTRADICTION pass: for every PROMOTED-or-near-promoted
    # hypothesis whose TRIGGER fired this turn but whose EFFECT did
    # NOT appear in the observed effects, decay its credence.
    observed_effect_strs = {e.as_str() for e in effects}
    for h in world.mechanic_hypotheses:
        if h.trigger != trigger.as_str():
            continue
        if h.effect in observed_effect_strs:
            continue
        # trigger fired, effect didn't — contradiction
        h.contradicting_observations.append(delta_idx)
        h.credence = max(0.0, h.credence - 0.3)
        if h.credence < 0.5:
            h.promoted = False

    # NEGATIVE-evidence pass: mine BlockingClaim when this action was
    # SILENT (no movement and no entity changes) in a state-class
    # we've now seen blocked multiple times.
    mine_blocking_claim(world, action_record, delta)

    # Layer B — contrastive refutation.  For any hypothesis with both
    # supporting and contradicting evidence, look for a relational
    # feature that cleanly discriminates the two sets and annotate the
    # hypothesis with it as a precondition.  Replaces the decay-only
    # contradiction policy with a learning one.
    contrastive_refine(world)

    return touched


# ---------------------------------------------------------------------------
# State-class extraction
# ---------------------------------------------------------------------------


def _agent_row_band(world: WorldKnowledge) -> str:
    """Qualitative row band of the agent on the grid.  Game-agnostic:
    derived from agent's current cell row vs grid_inference.rows."""
    agent = next(
        (r for r in world.entities.values()
         if r.current_role == "agent"),
        None,
    )
    if agent is None or agent.current_cell is None:
        return "unknown"
    gi = world.grid_inference
    if gi is None or not gi.rows:
        return "unknown"
    row = agent.current_cell[0]
    third = gi.rows / 3.0
    if row < third:
        return "top"
    if row < 2 * third:
        return "mid"
    return "bottom"


def _entity_role_counts(world: WorldKnowledge) -> str:
    """Canonical 'role=count' string summarizing the role multiset of
    all currently-present entities.  Excludes background / scenery /
    decoration / hud roles (they don't affect interaction outcomes)."""
    INERT = {"scenery", "decoration", "hud", "background"}
    counts: dict[str, int] = defaultdict(int)
    for r in world.entities.values():
        # `still_present` isn't stored on EntityRecord — it's
        # snapshot-derived (last_seen_turn == world.turn).  Treat
        # any entity last seen in the current turn as present.
        if r.last_seen_turn != world.turn:
            continue
        role = (r.current_role or "unknown").lower()
        if role in INERT:
            continue
        counts[role] += 1
    parts = [f"{k}={v}" for k, v in sorted(counts.items())]
    return ",".join(parts) if parts else "empty"


def _state_class_fingerprint(world: WorldKnowledge) -> dict[str, str]:
    """Game-agnostic state-class features at the current world state.

    Returns a small dict of qualitative features used to key
    BlockingClaim mining.  Adapters can extend this without changing
    the substrate by adding game-specific roles to the perception
    output (which already feeds entities & roles)."""
    return {
        "entity_roles": _entity_role_counts(world),
        "agent_row_band": _agent_row_band(world),
        "manip_spans_obstacle": _manip_spans_obstacle(world),
    }


def _ent_cols(rec):
    for attr in ("bbox", "current_bbox"):
        b = getattr(rec, attr, None)
        if b and len(b) >= 4:
            return (b[1], b[3])
    bh = getattr(rec, "bbox_history", None) or []
    if bh:
        last = bh[-1]
        b = last[1] if (isinstance(last, (tuple, list)) and len(last) == 2) else last
        if b and len(b) >= 4:
            return (b[1], b[3])
    return None


def _manip_spans_obstacle(world: WorldKnowledge) -> str:
    """RELATIONAL state feature: does the agent/manipulator's column span cross
    an obstacle's columns? This is what made the lc-4 'raise' a no-op (the rigid
    rod's middle pinned under the wall). Game-agnostic; 'na' if not derivable."""
    agent = None
    obstacles = []
    for r in (getattr(world, "entities", {}) or {}).values():
        tag = ((getattr(r, "current_role", "") or getattr(r, "role", "") or "")
               + " " + (getattr(r, "name", "") or "")).lower()
        if agent is None and any(k in tag for k in (
                "agent", "arm", "manipulator", "rod", "gripper", "effector")):
            agent = r
        if any(k in tag for k in ("wall", "obstacle", "barrier")):
            obstacles.append(r)
    ac = _ent_cols(agent) if agent else None
    if not ac:
        return "na"
    for o in obstacles:
        oc = _ent_cols(o)
        if oc and not (ac[1] < oc[0] or ac[0] > oc[1]):
            return "yes"
    return "no" if obstacles else "na"


def _state_class_str(features: dict[str, str]) -> str:
    """Canonical, deterministic stringification of a state-class
    fingerprint — used both as the dict key for matching and as the
    human-readable claim_id suffix."""
    return ";".join(f"{k}={v}" for k, v in sorted(features.items()))


def _is_silent_delta(delta: DeltaRecord) -> bool:
    """A 'silent' action: nothing observable changed.  This is the
    negative-evidence signal that drives BlockingClaim mining."""
    return (
        not delta.agent_moved
        and not delta.entities_appeared
        and not delta.entities_disappeared
        and not delta.entities_changed
    )


def _blocking_claim_id(action: str, state_class_str: str) -> str:
    return f"B[action={action} ; {state_class_str}]"


# ---------------------------------------------------------------------------
# BlockingClaim mining
# ---------------------------------------------------------------------------


# Mining thresholds.  A claim becomes a *candidate* after 1 silent
# observation (credence 0.4), gains credence per re-observation, and
# is "promoted" — i.e. trusted enough for the planner to treat the
# state-class as a hard precondition violation — at 0.85.
_BLOCKING_STARTING_CREDENCE = 0.4
_BLOCKING_SUPPORT_BUMP = 0.2
_BLOCKING_CONTRADICTION_DECAY = 0.4
_BLOCKING_PROMOTION_THRESHOLD = 0.85


def mine_blocking_claim(world: WorldKnowledge,
                          action_record: ActionRecord,
                          delta: DeltaRecord) -> Optional[str]:
    """Update or create a BlockingClaim for this (action, state-class).

    Called once per turn from ``mine_step``.  Returns the claim_id
    that was touched (or None if no claim activity).

    Mining logic mirrors MechanicHypothesis but on NEGATIVE evidence:
      - If action was SILENT in state-class S: create-or-bump
        BlockingClaim(action, S).
      - If action was NOT silent in state-class S but a matching
        BlockingClaim exists: that's contradicting evidence, decay
        its credence.
    """
    state_features = _state_class_fingerprint(world)
    state_class_str = _state_class_str(state_features)
    claim_id = _blocking_claim_id(action_record.action, state_class_str)
    delta_idx = len(world.deltas_observed) - 1

    # Find existing claim by id
    existing: Optional[BlockingClaim] = None
    for c in world.blocking_claims:
        if c.claim_id == claim_id:
            existing = c
            break

    if _is_silent_delta(delta):
        # Positive evidence FOR the claim
        if existing is None:
            existing = BlockingClaim(
                claim_id=claim_id,
                blocked_action=action_record.action,
                blocking_state=dict(state_features),
                credence=_BLOCKING_STARTING_CREDENCE,
                supporting_observations=[delta_idx],
            )
            world.blocking_claims.append(existing)
        else:
            existing.supporting_observations.append(delta_idx)
            existing.credence = min(
                1.0,
                existing.credence + _BLOCKING_SUPPORT_BUMP,
            )
            if (existing.credence >= _BLOCKING_PROMOTION_THRESHOLD
                    and not existing.promoted):
                existing.promoted = True
        return claim_id

    # Action had effect — contradicts any matching claim.
    if existing is not None:
        existing.contradicting_observations.append(delta_idx)
        existing.credence = max(
            0.0, existing.credence - _BLOCKING_CONTRADICTION_DECAY,
        )
        if existing.credence < 0.5:
            existing.promoted = False
        return claim_id

    return None


def promoted_blocking_claims(world: WorldKnowledge) -> list[BlockingClaim]:
    """Convenience: BlockingClaims that have crossed the promotion
    threshold.  The planner / decomposer consults these as hard
    preconditions: if the current state-class matches a promoted
    BlockingClaim for action A, A is forbidden in this state and a
    removal subgoal is required."""
    return [c for c in world.blocking_claims if c.promoted]


def matching_blocking_claims(world: WorldKnowledge,
                                action: str) -> list[BlockingClaim]:
    """All BlockingClaims (any credence) whose blocking_state matches
    the CURRENT world state-class for the given action.  Used by the
    planner-bridge to decide whether to spawn a removal subgoal
    before issuing the action.

    A claim 'matches' when EVERY feature in its blocking_state has the
    same value as the current state-class fingerprint (extra current
    features are allowed — the claim describes a subset)."""
    if not world.blocking_claims:
        return []
    current = _state_class_fingerprint(world)
    out: list[BlockingClaim] = []
    for c in world.blocking_claims:
        if c.blocked_action != action:
            continue
        if all(current.get(k) == v
                for k, v in c.blocking_state.items()):
            out.append(c)
    return out


# ---------------------------------------------------------------------------
# Read-side helpers for the actor/planner
# ---------------------------------------------------------------------------


def trusted_rules(world: WorldKnowledge,
                   min_credence: float = 0.8) -> list[MechanicHypothesis]:
    return [h for h in world.mechanic_hypotheses
            if h.credence >= min_credence]


def goal_related_rules(world: WorldKnowledge) -> list[MechanicHypothesis]:
    """Rules whose effect is likely a GOAL EVENT (entity disappeared,
    win_state changed) — these are the rules a goal-directed planner
    cares about, because they tell it which actions advance progress."""
    out = []
    for h in world.mechanic_hypotheses:
        if h.credence < 0.5:
            continue
        if ("entity_disappeared" in h.effect
                or "win_state_changed" in h.effect):
            out.append(h)
    return out


# ---------------------------------------------------------------------------
# Layer B — Contrastive Refutation
# ---------------------------------------------------------------------------
#
# When a hypothesis has both supporting and contradicting observations,
# diff the relational fingerprints of those observations to find the
# discriminating condition, and annotate the hypothesis with that
# condition as a precondition.  Replaces the decay-only contradiction
# policy with a learning one: a precondition-qualified hypothesis
# survives and is trusted in matching contexts, instead of being
# averaged away.
#
# Discriminant: a (kind, direction) relation feature that is present in
# ≥B_SUPPORT_PRESENCE of supporting deltas AND ≤B_CONTRADICT_PRESENCE
# of contradicting deltas, or the reverse (negative correlation).
# Game-agnostic: it never looks at entity names, only relation kinds
# and cardinal directions, so a discriminant found on one game type
# generalises to any game producing the same relation.


# Thresholds — kept conservative.  A discriminant must be cleanly
# differential: present in most supports and largely absent from
# contradictions (or vice versa).  Below this bar, B reports nothing
# rather than overfit on noise.
_B_SUPPORT_PRESENCE = 0.7
_B_CONTRADICT_PRESENCE = 0.3
_B_MIN_OBSERVATIONS = 2          # need at least this many on each side


def _relation_fingerprint(world: WorldKnowledge,
                            delta_idx: int) -> set[tuple]:
    """Return the (kind, direction) features present in the named
    delta's relations.  Game-agnostic: never references entity names,
    so the resulting fingerprint can be compared across deltas without
    leaking game-specific vocabulary."""
    if delta_idx < 0 or delta_idx >= len(world.deltas_observed):
        return set()
    dl = world.deltas_observed[delta_idx]
    rels = getattr(dl, "relations", None) or []
    fp: set[tuple] = set()
    for r in rels:
        if isinstance(r, dict):
            kind = r.get("kind")
            direction = r.get("direction")
        else:
            kind = getattr(r, "kind", None)
            direction = getattr(r, "direction", None)
        if kind:
            fp.add((kind, direction))
    return fp


def _find_discriminant(supporting_fps: list[set[tuple]],
                          contradicting_fps: list[set[tuple]]
                          ) -> Optional[dict]:
    """Search for the relational feature that most cleanly separates
    supporting deltas from contradicting ones.  Returns a dict
    {"feature": (kind, direction), "correlation": "positive"|"negative",
     "support_presence": float, "contradict_presence": float}, or None
    if no feature crosses the threshold.

    Tie-break by largest |support_presence − contradict_presence|.
    """
    if (len(supporting_fps) < _B_MIN_OBSERVATIONS
            or len(contradicting_fps) < _B_MIN_OBSERVATIONS):
        return None
    s_n = len(supporting_fps)
    c_n = len(contradicting_fps)
    all_features: set[tuple] = set()
    for fp in supporting_fps:
        all_features.update(fp)
    for fp in contradicting_fps:
        all_features.update(fp)
    best: Optional[dict] = None
    best_diff = 0.0
    for f in all_features:
        s_with = sum(1 for fp in supporting_fps if f in fp) / s_n
        c_with = sum(1 for fp in contradicting_fps if f in fp) / c_n
        correlation: Optional[str] = None
        if (s_with >= _B_SUPPORT_PRESENCE
                and c_with <= _B_CONTRADICT_PRESENCE):
            correlation = "positive"
        elif (s_with <= _B_CONTRADICT_PRESENCE
                and c_with >= _B_SUPPORT_PRESENCE):
            correlation = "negative"
        else:
            continue
        diff = abs(s_with - c_with)
        if diff > best_diff:
            best_diff = diff
            best = {
                "feature": list(f),         # (kind, direction) -> [k, d]
                "correlation": correlation,
                "support_presence": round(s_with, 2),
                "contradict_presence": round(c_with, 2),
            }
    return best


def contrastive_refine(world: WorldKnowledge) -> list[tuple[str, dict]]:
    """For every hypothesis with both supporting and contradicting
    observations, run a relational-fingerprint contrast and annotate
    the hypothesis with the discriminating feature as a precondition.

    Idempotent — a hypothesis whose `precondition` is already set is
    skipped until new evidence accumulates that contradicts the existing
    precondition (handled by a re-mining pass when evidence diverges
    from the recorded precondition; this v1 keeps the first finding).

    Returns the list of (hypothesis_id, precondition) pairs newly
    annotated, suitable for driver logging.
    """
    annotated: list[tuple[str, dict]] = []
    for h in world.mechanic_hypotheses:
        if h.precondition is not None:
            continue
        s_obs = h.supporting_observations or []
        c_obs = h.contradicting_observations or []
        if (len(s_obs) < _B_MIN_OBSERVATIONS
                or len(c_obs) < _B_MIN_OBSERVATIONS):
            continue
        s_fps = [_relation_fingerprint(world, i) for i in s_obs]
        c_fps = [_relation_fingerprint(world, i) for i in c_obs]
        disc = _find_discriminant(s_fps, c_fps)
        if disc is None:
            continue
        disc["discovered_at_turn"] = world.turn
        h.precondition = disc
        annotated.append((h.hypothesis_id, disc))
    return annotated


def format_precondition_qualified_rules(
    world: WorldKnowledge,
) -> str:
    """Render hypotheses that have been precondition-qualified by B
    in a compact form for the strategy prompt.  These are the most
    informative rules: they encode the discriminating CONDITION under
    which an effect appears, not just the average behavior.

    Credence is NOT used as a filter: B-qualified rules are surfaced
    regardless of unconditional credence, because the precondition
    tells the actor *when* the rule applies.  A rule that was decayed
    to credence 0 by repeated contradictions and then qualified by B
    is exactly the case where B's annotation is most valuable — it
    rescues the rule from oblivion as a conditional one.  The credence
    is still rendered alongside so the actor sees the unconditional
    rate.
    """
# Relation kinds that describe a TRANSITION (what just moved), not a stable
# pre-state.  A mined "precondition" that is one of these is almost always a
# CONFOUND: it co-occurs with the effect during the same transition rather
# than being a settable pre-condition (e.g. "ACTION4 pierces WHEN
# co_displacement(left) present" — the leftward motion is part of the event,
# not a state you arrange first).  Surfacing such a rule as an authoritative
# gate misleads the actor into trying to MANUFACTURE the motion instead of
# performing the maneuver (the sk48 lc=2 derail).  These are demoted to
# test-don't-trust hypotheses.  Static relations (same_row/col, clearance,
# adjacent, aligned, support_relation, ordered_along) are legitimate
# pre-states and stay as trusted preconditions.
_TRANSIENT_RELATION_KINDS = {
    "co_displacement", "motion_blocked", "motion_arrested_at", "penetration",
}


def _precondition_feature_str(pc: dict) -> tuple[str, str, str]:
    """Return (feature_kind, feature_str, gate) for a precondition dict."""
    feat = pc.get("feature") or []
    kind = feat[0] if len(feat) >= 1 else "?"
    feat_str = str(kind)
    if len(feat) >= 2 and feat[1]:
        feat_str += f"(dir={feat[1]})"
    corr = pc.get("correlation", "?")
    gate = ("WHEN present" if corr == "positive"
            else "WHEN absent" if corr == "negative" else "?")
    return kind, feat_str, gate


def format_precondition_qualified_rules(
    world: WorldKnowledge,
) -> str:
    qualified = [
        h for h in world.mechanic_hypotheses
        if h.precondition is not None
    ]
    if not qualified:
        return ""
    trusted, suspect = [], []
    for h in qualified:
        kind, feat_str, gate = _precondition_feature_str(h.precondition or {})
        row = (f"    - {h.trigger}  ==>  {h.effect}  "
               f"[c={h.credence:.2f}, precondition: {feat_str} "
               f"{gate}, s_pres={(h.precondition or {}).get('support_presence')}, "
               f"c_pres={(h.precondition or {}).get('contradict_presence')}]")
        (suspect if kind in _TRANSIENT_RELATION_KINDS else trusted).append(row)

    out: list[str] = []
    if trusted:
        out.append(
            "  Precondition-qualified mechanic rules "
            "(B's contrastive refinement — these tell you the CONDITION "
            "an effect requires; the bare effect alone is unreliable):")
        out.extend(trusted)
    if suspect:
        if out:
            out.append("")
        out.append(
            "  UNVERIFIED conditional correlations — TREAT AS HYPOTHESES TO "
            "TEST, NOT as facts.  The 'precondition' below is a TRANSIENT "
            "MOTION relation that describes the transition itself, not a "
            "settable pre-state, so it is very likely a CONFOUND (it merely "
            "co-occurred with the effect).  Do NOT contort your plan to "
            "MANUFACTURE this motion; perform the maneuver your own physics "
            "reasoning indicates and let the outcome confirm or refute the "
            "rule:")
        out.extend(suspect)
    return "\n".join(out)
