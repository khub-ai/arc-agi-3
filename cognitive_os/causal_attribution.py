"""Causal attribution & avoidance — fault diagnosis for an acting agent.

When an action produces an UNWANTED effect (a goal/sub-goal condition regressed,
or a move that should have advanced progress was blocked), this module runs an
active investigation: it names a *cause* and finds a way *around* it, instead of
recording a bare (action, state) correlation.

The loop (see SPEC_vlm_backward_reasoning.md § "Causal attribution & avoidance"):
  anomaly -> suspects (relations the agent/coupled set are in, incl. the action's
  swept path) -> confirm by counterfactual (negate the gating relation, re-run
  the action, restore) -> CausalClaim (cause + remedy) .

It is **modality-agnostic**: it runs entirely against a `Provider` (tier 1),
never over pixels. The decisive property — the counterfactual that *confirms*
the cause is the action that *avoids* it — so diagnosis and remedy are found
together.

Provider binding (how a concrete tier-1 provider satisfies the interface):
  - `conditions()`      -> the currently-held goal/sub-goal conditions
                            (e.g. the completion done-set from the goal-gap).
  - `relations()`       -> standing typed relations (relational_kinematics:
                            co_displacement / penetration / overlapping /
                            co_confined / motion_blocked / support_relation …).
  - `path_relations(a)` -> entities/relations lying in action `a`'s swept path
                            (the structure the carried set would traverse).
  - `agents()`          -> the agent + the set it is coupled to (carried items).
  - `actions()`         -> the available action labels.
  - `snapshot()/restore()/apply()` -> the counterfactual sandbox (e.g. env
                            deepcopy, or step + UNDO/ACTION7).

The visual provider is ONE implementation; a robot's provider emits the same
record types from contact/force/proprioception and uses its own restore.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Typed records (abstract — no modality assumptions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Relation:
    """A typed relation between entities. `entities[0]` is the subject side
    (the agent / its coupled set); the rest are the related entities."""
    kind: str
    entities: tuple

    def others(self, agents: frozenset) -> list:
        return [e for e in self.entities if e not in agents]


@dataclass
class Anomaly:
    kind: str                 # 'regression' | 'block'
    action: str
    lost: frozenset = frozenset()   # conditions lost (regression); empty for block

    def describe(self) -> str:
        if self.kind == "regression":
            return (f"action {self.action!r} regressed conditions "
                    f"{sorted(self.lost)}")
        return f"action {self.action!r} was blocked (no progress)"


@dataclass(frozen=True)
class Suspect:
    culprit: str
    relation: Relation
    rank: float


@dataclass
class CausalClaim:
    effect: str               # human description of the unwanted effect
    anomaly_kind: str
    action: str               # the action that triggered the anomaly
    culprit: str              # the entity attributed as the cause
    gating_relation: str      # the relation kind that gates the effect
    remedy_action: Optional[str]   # action that negates the gating relation
    confidence: str           # 'confirmed' | 'twin_confirmed' | 'predicted'
                              #  | 'unconfirmed' — graded by WHAT confirmed it
                              #  (real rollback / sim twin / mined-dynamics);
                              #  see SPEC_vlm_backward_reasoning §7.1 tiers.
    note: str = ""
    remedy_repeats: int = 1   # repeat the remedy action this many times to clear
    fidelity: Optional[str] = None   # for twin_confirmed: the twin's fidelity
                                     #  for THIS fault class (e.g. 'kinematic'
                                     #  high, 'contact' low) — bounds validity

    def avoidance_subgoal(self) -> Optional[str]:
        """The blocked-goal sub-goal this implies: negate the gating relation
        (then retry the action)."""
        if self.remedy_action is None:
            return None
        rep = (f" x{self.remedy_repeats}" if self.remedy_repeats != 1 else "")
        return (f"achieve ¬{self.gating_relation}(agent,{self.culprit}) "
                f"via {self.remedy_action!r}{rep}, then retry {self.action!r}")


@runtime_checkable
class Provider(Protocol):
    def conditions(self) -> frozenset: ...
    def relations(self) -> list: ...
    def path_relations(self, action: str) -> list: ...
    def agents(self) -> frozenset: ...
    def actions(self) -> list: ...
    def fingerprint(self): ...            # any hashable state digest (for block)
    def snapshot(self): ...
    def restore(self, snap) -> None: ...
    def apply(self, action: str) -> None: ...


# ---------------------------------------------------------------------------
# Stage 1 — anomaly detection
# ---------------------------------------------------------------------------

def detect_anomaly(before: frozenset, after: frozenset, action: str,
                    changed: bool) -> Optional[Anomaly]:
    """A regression (a held condition went false) or a block (an action that
    should have advanced produced no change). Returns None if the action was
    benign."""
    lost = frozenset(before) - frozenset(after)
    if lost:
        return Anomaly("regression", action, lost)
    if not changed:
        return Anomaly("block", action)
    return None


# ---------------------------------------------------------------------------
# Stage 2 — suspect generation (+ contrastive narrowing)
# ---------------------------------------------------------------------------

# Relational proximity → priority. Tighter coupling is a likelier cause.
_KIND_RANK = {
    "penetration": 5.0, "co_displacement": 4.5, "motion_blocked": 4.5,
    "motion_arrested_at": 4.0, "support_relation": 3.5, "overlapping": 3.0,
    "co_confined": 2.5, "same_row": 1.5, "same_col": 1.5, "ordered_along": 1.0,
}


def _suspect_from(rel, agents: frozenset) -> Optional[Suspect]:
    others = rel.others(agents)
    if not others:
        return None                      # purely agent-internal relation
    return Suspect(culprit=others[0], relation=rel,
                   rank=_KIND_RANK.get(rel.kind, 1.0))


def generate_suspects(provider: Provider, anomaly: Anomaly) -> list:
    """Candidate (culprit, gating-relation) pairs, ranked by proximity.

    Sources, BOTH needed (turn-99 lesson): the standing relations the agent /
    its coupled set participate in, AND the relations on the *swept path* of the
    anomaly's action — because the culprit may not be in contact yet (it lies in
    the path the carried set would traverse)."""
    agents = provider.agents()
    seen: set = set()
    out: list = []
    for rel in (list(provider.relations())
                + list(provider.path_relations(anomaly.action))):
        s = _suspect_from(rel, agents)
        if s is None:
            continue
        key = (s.culprit, s.relation.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    out.sort(key=lambda s: -s.rank)
    return out


def narrow_by_contrast(suspects: list, good_relations: list,
                        agents: frozenset) -> list:
    """If a GOOD transition of the same action is available, keep only suspects
    whose gating relation is ABSENT in the good case — the discriminant (this is
    contrastive refutation, B, aimed at one anomaly). With no baseline, returns
    the suspects unchanged (fall back to perturbing each)."""
    if not good_relations:
        return suspects
    good_keys = set()
    for rel in good_relations:
        for o in rel.others(agents):
            good_keys.add((o, rel.kind))
    return [s for s in suspects
            if (s.culprit, s.relation.kind) not in good_keys] or suspects


# ---------------------------------------------------------------------------
# Stage 3 — counterfactual confirmation (the test that also finds the remedy)
# ---------------------------------------------------------------------------

def _relation_present(provider: Provider, culprit: str, kind: str,
                       agents: frozenset, anomaly_action=None) -> bool:
    rels = list(provider.relations())
    if anomaly_action is not None:                # also check swept-path source
        rels += list(provider.path_relations(anomaly_action))
    for rel in rels:
        if rel.kind == kind and culprit in rel.others(agents):
            return True
    return False


def find_negating_actions(provider: Provider, suspect: Suspect,
                           agents: frozenset, anomaly_action=None,
                           max_repeat: int = 4) -> list:
    """All `(action, repeats)` operators that NEGATE the gating relation (break
    the agent's coupling/adjacency to the culprit) WITHOUT themselves regressing
    a held condition. An operator is an action *repeated up to `max_repeat`
    times* — the magnitude the spec anticipates ("an action or short sequence";
    turn-99 lesson: one `down` does not clear the shaft, two do). For each base
    action, repeat until it negates (record `(action, k)`) or until it regresses
    (abandon that action) or the cap. Non-destructive: snapshot/restore."""
    out = []
    for action in provider.actions():
        snap = provider.snapshot()
        try:
            before = frozenset(provider.conditions())
            found = None
            for k in range(1, max_repeat + 1):
                provider.apply(action)
                if before - frozenset(provider.conditions()):
                    break                     # this action regresses — abandon
                if not _relation_present(
                        provider, suspect.culprit, suspect.relation.kind,
                        agents, anomaly_action=anomaly_action):
                    found = (action, k)
                    break
        finally:
            provider.restore(snap)
        if found:
            out.append(found)
    return out


def _effect_occurs(provider: Provider, anomaly: Anomaly) -> bool:
    """Run the anomaly's action from the CURRENT state and report whether the
    unwanted effect occurs. Snapshot/restore-bracketed (non-destructive)."""
    snap = provider.snapshot()
    try:
        if anomaly.kind == "regression":
            before = frozenset(provider.conditions())
            provider.apply(anomaly.action)
            return bool(anomaly.lost & (before - frozenset(provider.conditions())))
        # block: the unwanted effect IS "no state change"
        fp0 = provider.fingerprint()
        provider.apply(anomaly.action)
        return provider.fingerprint() == fp0
    finally:
        provider.restore(snap)


def confirm_by_counterfactual(provider: Provider, anomaly: Anomaly,
                               negator) -> bool:
    """Confirm a cause by a TWO-ARM counterfactual — treatment AND a negative
    control — so "effect gone" cannot be credited to environment drift instead
    of the intervention (the approach-#2 confound resurfacing). Returns True
    only if BOTH hold:
      - NEGATIVE CONTROL: the effect REPRODUCES when the action is re-run
        *without* negating the relation (else the env drifted / is stochastic /
        restore is lossy → inconclusive, do not credit);
      - TREATMENT: after applying the negating operator `(action, repeats)`, the
        re-run action's effect is GONE.
    In a deterministic env with lossless restore (ARC, or a digital twin) the
    control always reproduces and this reduces to the single arm; in an
    imperfect-restore / stochastic env it correctly refuses to confirm. All
    arms are snapshot/restore-bracketed."""
    action, repeats = negator
    # Arm 1 — negative control: the effect must reproduce on its own.
    if not _effect_occurs(provider, anomaly):
        return False
    # Arm 2 — treatment: negate the relation, then re-run; effect must vanish.
    snap = provider.snapshot()
    try:
        if anomaly.kind == "regression":
            before = frozenset(provider.conditions())
            for _ in range(repeats):
                provider.apply(action)
            provider.apply(anomaly.action)
            return not (anomaly.lost & (before - frozenset(provider.conditions())))
        for _ in range(repeats):
            provider.apply(action)
        fp0 = provider.fingerprint()
        provider.apply(anomaly.action)
        return provider.fingerprint() != fp0
    finally:
        provider.restore(snap)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def attribute(provider: Provider, anomaly: Anomaly,
               good_relations: Optional[list] = None) -> Optional[CausalClaim]:
    """Run the full loop: suspects -> (contrast-narrow) -> for each, find a
    negating action and confirm by counterfactual -> return the confirmed
    CausalClaim (cause + remedy). Returns an unconfirmed claim on the top
    suspect if none confirm, or None if there were no suspects."""
    agents = provider.agents()
    # Confidence is graded by WHAT the sandbox is (real rollback / sim twin /
    # mined-dynamics), supplied by the provider (default 'confirmed' for a
    # provider that confirms on the real system, e.g. the test toys).
    _tier_fn = getattr(provider, "confirmation_tier", None)
    tier = _tier_fn() if callable(_tier_fn) else "confirmed"
    _fid_fn = getattr(provider, "confirmation_fidelity", None)
    fidelity = _fid_fn() if callable(_fid_fn) else None
    suspects = narrow_by_contrast(
        generate_suspects(provider, anomaly), good_relations or [], agents)
    if not suspects:
        return None
    for s in suspects:
        negators = find_negating_actions(provider, s, agents,
                                          anomaly_action=anomaly.action)
        if not negators:
            continue                     # culprit cannot be avoided — skip
        for neg in negators:             # try every candidate direction
            if confirm_by_counterfactual(provider, anomaly, neg):
                act, rep = neg
                return CausalClaim(
                    effect=anomaly.describe(), anomaly_kind=anomaly.kind,
                    action=anomaly.action, culprit=s.culprit,
                    gating_relation=s.relation.kind, remedy_action=act,
                    remedy_repeats=rep, confidence=tier, fidelity=fidelity,
                    note=(f"negating {s.relation.kind}(agent,{s.culprit}) via "
                          f"{act!r} x{rep} prevents the effect "
                          f"[{tier}"
                          + (f", fidelity={fidelity}" if fidelity else "")
                          + "]"))
    top = suspects[0]
    return CausalClaim(
        effect=anomaly.describe(), anomaly_kind=anomaly.kind,
        action=anomaly.action, culprit=top.culprit,
        gating_relation=top.relation.kind, remedy_action=None,
        confidence="unconfirmed",
        note="no negating action confirmed the cause")
