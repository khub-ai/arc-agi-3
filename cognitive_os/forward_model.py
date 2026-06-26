"""Forward-model query layer: run TransitionClaims *backward* (goal regression).

See ``docs/SPEC_forward_model.md``. This module is the **inversion** the spec
calls the lynchpin (§3 add #2): given a goal :class:`~cognitive_os.conditions.Condition`,
find the :class:`~cognitive_os.claims.TransitionClaim` s whose ``post`` entails
it and surface their unmet ``pre`` conjuncts as sub-goals. That single
operation is what turns forward trial-and-error into backward / means-ends
planning ([P15](../docs/DURABLE_PRINCIPLES.md)).

Design notes:

* **Condition-agnostic.** Entailment is computed structurally from
  ``canonical_key`` over flattened conjuncts, so *any* predicate
  (``EntityInState``, relation predicates, resource predicates, …) composes
  without this module knowing their internals. This is deliberately
  conservative — it never *infers* beyond conjunct membership, so a returned
  transition genuinely produces the goal (no false regressions).
* **Reuse, don't reinvent.** Operates on the existing ``TransitionClaim``
  (action, pre, post) schema and the existing ``Conjunction`` combinator. The
  forward direction (claim matches events → gains credence) is unchanged in
  ``hypothesis_store``; this adds only the reverse lookup the planner needs.
* The actual multi-step planner (recurse on ``unmet`` over ``goal_forest`` /
  ``depends_on``, resolve magnitudes, verify against frames, interleave) lives
  in ``plan_search`` / ``SPEC_goal_plan_search``; this module supplies the
  one-step regression primitive it composes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, List, Optional, Set, Tuple

from .claims import MotionModelClaim, TransitionClaim
from .conditions import Condition, Conjunction, EntityInState


# ---------------------------------------------------------------------------
# Conjunct algebra (Condition-agnostic)
# ---------------------------------------------------------------------------


def conjuncts(cond: Condition) -> Tuple[Condition, ...]:
    """Flatten a Condition into its conjuncts; nested Conjunctions flatten."""
    if isinstance(cond, Conjunction):
        out: List[Condition] = []
        for c in cond.conditions:
            out.extend(conjuncts(c))
        return tuple(out)
    return (cond,)


def post_entails(post: Condition, goal: Condition) -> bool:
    """True iff every conjunct of ``goal`` appears (by ``canonical_key``)
    among the conjuncts of ``post``. Conservative structural entailment: no
    inference beyond conjunct membership, so it never claims a transition
    achieves a goal it does not literally establish."""
    post_keys = {c.canonical_key() for c in conjuncts(post)}
    return all(g.canonical_key() in post_keys for g in conjuncts(goal))


# ---------------------------------------------------------------------------
# Reverse effect-index + one-step regression
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RegressionOption:
    """One way to achieve the goal in a single action.

    ``unmet`` are the ``pre`` conjuncts that are *not yet satisfied* (the
    sub-goals the planner must recurse on). When the planner supplies a
    ``world``, ``unmet`` are the conjuncts that do not evaluate True there;
    with no ``world`` every ``pre`` conjunct is treated as a sub-goal.
    """

    action: str
    pre: Condition
    post: Condition
    unmet: Tuple[Condition, ...]
    claim_id: Optional[str] = None
    credence: Optional[float] = None


def transitions_for_goal(
    claims: Iterable[TransitionClaim], goal: Condition
) -> List[TransitionClaim]:
    """Reverse effect-index (pure): TransitionClaims whose ``post`` entails
    ``goal``. This is the inversion of the forward (action, pre)-keyed index."""
    return [c for c in claims
            if isinstance(c, TransitionClaim) and post_entails(c.post, goal)]


def _unmet(pre: Condition, world) -> Tuple[Condition, ...]:
    out: List[Condition] = []
    for pc in conjuncts(pre):
        if world is None:
            out.append(pc)
            continue
        try:
            val = pc.evaluate(world)
        except Exception:
            val = None
        if val is not True:          # False or None (unknown) → still a sub-goal
            out.append(pc)
    return tuple(out)


def regress(
    claims: Iterable[TransitionClaim],
    goal: Condition,
    world=None,
) -> List[RegressionOption]:
    """One backward step: every transition that achieves ``goal``, paired with
    its unmet preconditions (the next sub-goals). The planner recurses on
    ``opt.unmet`` until an option has no unmet preconditions (executable now)."""
    claims = list(claims)
    return [RegressionOption(action=c.action, pre=c.pre, post=c.post,
                             unmet=_unmet(c.pre, world))
            for c in transitions_for_goal(claims, goal)]


# ---------------------------------------------------------------------------
# Contrastive precondition tightening (layer 3: over-general -> precise)
# ---------------------------------------------------------------------------
#
# Co-occurrence mining (the miner) gets a transition's EFFECT right but its
# PRECONDITION too loose — "extend -> pierced(orange)" with no precondition,
# so the planner would propose `extend` prematurely. The fix is contrastive:
# compare pre-states where the action DID produce the effect against those
# where it did NOT, and keep the (boolean/relational) facts that discriminate
# (true in every positive, false in some negative). Those are the missing
# preconditions. (SPEC_forward_model.md §3 add #4 / Layer-B contrastive pass.)


def _bool_facts(pre_obs: Mapping[str, Mapping[str, Any]]) -> Set[tuple]:
    """Boolean (entity, prop, value) facts from a pre-state snapshot. Boolean
    only: positional values (row/col) are not generalizable preconditions, and
    restricting to predicates avoids coordinate noise."""
    out: Set[tuple] = set()
    for eid, props in pre_obs.items():
        for p, v in props.items():
            if isinstance(v, bool):
                out.add((eid, p, v))
    return out


def _fact_holds(fact: tuple, obs: Mapping[str, Mapping[str, Any]]) -> bool:
    eid, p, v = fact
    return obs.get(eid, {}).get(p, _NEG_SENTINEL) == v


def _effect_holds(effect: EntityInState, post_obs: Mapping[str, Mapping[str, Any]]) -> bool:
    return post_obs.get(effect.entity_id, {}).get(effect.property, _NEG_SENTINEL) == effect.value


_NEG_SENTINEL = object()


def tighten_precondition(records: Iterable[Mapping[str, Any]],
                         action: str,
                         effect: EntityInState,
                         *,
                         relevant: Optional[Set[str]] = None) -> Optional[Condition]:
    """Learn the discriminating precondition for ``action`` producing
    ``effect``, by contrast over episode ``records``.

    Each record is ``{"action": str, "pre": obs, "post": obs}`` where ``obs``
    is the ``{entity:{prop:value}}`` snapshot the bridge already produces.
    Returns the conjunction of boolean facts that are **true in every positive
    case and false in at least one negative case** — i.e. the facts that
    explain *when* the action has the effect. Returns ``None`` when there is no
    contrast (no positives, or no negatives) or no discriminator emerges.

    ``relevant`` — optional set of **causally-relevant entity ids**. When
    given, only facts on those entities are considered, suppressing facts on
    unrelated entities that *spuriously* correlate with the effect (a real
    failure mode: with limited play, e.g. unrelated blocks' clearances flip in
    lockstep with the target effect). This is the adapter's lever (P5: the
    engine does not guess locality; the domain supplies it). NOTE: ``relevant``
    bounds, but does not eliminate, spurious correlation — a fact on a relevant
    entity can still correlate spuriously. The complete control is **varied /
    decorrelating exploration** (active learning): positives where a spurious
    fact is false, or negatives where it is true, break the correlation. That
    exploration quality — not this selector — is the load-bearing requirement
    for a *correct* learned precondition (see SPEC_forward_model.md).
    """
    records = list(records)
    pos = [r for r in records
           if r["action"] == action and _effect_holds(effect, r["post"])]
    neg = [r for r in records
           if r["action"] == action and not _effect_holds(effect, r["post"])]
    if not pos or not neg:
        return None
    common: Optional[Set[tuple]] = None
    for r in pos:
        fs = _bool_facts(r["pre"])
        common = fs if common is None else (common & fs)
    if not common:
        return None
    disc = sorted(f for f in common
                  if any(not _fact_holds(f, n["pre"]) for n in neg))
    if relevant is not None:
        # Relation-target locality: keep a fact only if its OWN entity is
        # relevant AND its property does not REFERENCE a non-relevant entity
        # (e.g. drop block_orange.col_clear_of_blue when blue is irrelevant,
        # keep block_orange.col_clear_of_red when red is). Entity short-names
        # (strip "block_") are matched as whole tokens of the property name.
        universe: Set[str] = set()
        for r in records:
            universe |= set(r.get("pre", {}).keys()) | set(r.get("post", {}).keys())
        def _short(x: str) -> str:
            return x[6:] if x.startswith("block_") else x
        nonrel = {_short(e) for e in universe} - {_short(e) for e in relevant}
        disc = [f for f in disc
                if f[0] in relevant and not (set(str(f[1]).split("_")) & nonrel)]
    if not disc:
        return None
    conds = [EntityInState(e, p, v) for (e, p, v) in disc]
    return conds[0] if len(conds) == 1 else Conjunction(tuple(conds))


# ---------------------------------------------------------------------------
# Magnitude resolution (layer 2: turn a symbolic step into an exact count)
# ---------------------------------------------------------------------------
#
# The planner emits a symbolic action ("move_down"); HOW MANY times to repeat
# it is a geometric quantity = (gap to close) / (per-action stride). The gap
# comes from perception (measured positions), the stride from a
# MotionModelClaim. Computing it is what removes the count-GUESSING that left
# sk48's green unsolved (see SPEC_forward_model.md §2 layer 2, P15 §detect).


def repeats_to_close(current: int, target: int, stride: int,
                     *, mode: str = "reach") -> int:
    """How many repeats of an action with signed per-step ``stride`` move a
    scalar from ``current`` toward ``target``.

    ``mode='reach'`` (default): the smallest count that reaches **or passes**
    ``target`` (``ceil`` — e.g. retract until the tip is at/left of a column).
    ``mode='nearest'``: the count that lands **closest** to ``target``
    (``round`` — e.g. align the arm to a row).

    Returns ``0`` when already at ``target`` or when ``stride`` points **away**
    from it (the action cannot close this gap — the planner must pick another).
    """
    if stride == 0:
        raise ValueError("stride must be non-zero")
    gap = target - current
    if gap == 0:
        return 0
    if (gap > 0) != (stride > 0):          # stride moves away from target
        return 0
    steps = abs(gap) / abs(stride)
    return int(math.ceil(steps)) if mode == "reach" else int(round(steps))


def motion_stride(claims: Iterable, action_id: str) -> Optional[Any]:
    """The per-action stride for ``action_id`` from the best
    :class:`MotionModelClaim` (the planner-facing aggregated motion model).
    Returns the delta (e.g. ``(-6, 0)``) or ``None`` if unknown.

    Accepts a claim iterable; when several MotionModelClaims compete for the
    same action (stochastic stride) the highest-credence one wins if the items
    are hypotheses, else the first claim is taken."""
    best = None
    best_cred = -1.0
    for item in claims:
        claim = getattr(item, "claim", item)
        if not isinstance(claim, MotionModelClaim) or str(claim.action_id) != str(action_id):
            continue
        cred = getattr(getattr(item, "credence", None), "point", 0.0)
        if cred >= best_cred:
            best_cred, best = cred, claim.delta
    return best


def expand_action(action_id: str, current: int, target: int, stride: int,
                  *, mode: str = "reach") -> List[str]:
    """A symbolic step + geometry -> the concrete repeated-action sequence.
    ``[action_id] * repeats_to_close(...)``. Empty if the gap is already
    closed or the stride points the wrong way."""
    return [action_id] * repeats_to_close(current, target, stride, mode=mode)


# ---------------------------------------------------------------------------
# Multi-step backward planner (means-ends regression)
# ---------------------------------------------------------------------------


def holds_in(satisfied_keys: Set[tuple]) -> Callable[[Condition], bool]:
    """Build a ``holds`` predicate from a set of satisfied canonical keys."""
    return lambda c: c.canonical_key() in satisfied_keys


def holds_in_world(world) -> Callable[[Condition], bool]:
    """Build a ``holds`` predicate that evaluates Conditions against a world
    (None/unknown is treated as not-holding, so it becomes a sub-goal)."""
    def _h(c: Condition) -> bool:
        try:
            return c.evaluate(world) is True
        except Exception:
            return False
    return _h


def plan_to_goal(
    claims: Iterable[TransitionClaim],
    goal: Condition,
    holds: Callable[[Condition], bool],
    *,
    prefer: Optional[Callable[[TransitionClaim], Any]] = None,
    max_depth: int = 64,
) -> Optional[List[str]]:
    """Means-ends backward planner: recurse :func:`regress` until every leaf
    precondition ``holds`` in the current state. Returns an **ordered action
    list (prerequisites first)**, or ``None`` if no plan exists.

    This is the recursion that turns the one-step inversion into a full plan
    ([P15](../docs/DURABLE_PRINCIPLES.md)). It is the primitive
    ``plan_search`` composes (it does not own resource/goal-sequencing — see
    [SPEC_goal_plan_search.md](../docs/SPEC_goal_plan_search.md)).

    - **Ordering & reuse**: sub-goals achieved earlier (within this plan)
      satisfy later preconditions — so a goal conjunction like
      ``pierced(red) ∧ pierced(orange)`` plans red first and reuses
      ``pierced(red)`` as a precondition of the orange maneuver.
    - **Backtracking**: when a goal has several achieving transitions, each is
      tried in ``prefer`` order; the first whose preconditions are themselves
      plannable is used.
    - **Cycle guard**: a sub-goal currently being expanded is never re-entered
      (``seen``), so mutually-dependent transitions fail that branch rather
      than looping.
    - ``holds`` decides what is already true — use :func:`holds_in` (a key
      set) or :func:`holds_in_world` (live evaluation).
    """
    claims = list(claims)

    def solve(g: Condition, satisfied: Set[tuple], seen: FrozenSet,
              depth: int) -> Optional[Tuple[List[str], Set[tuple]]]:
        if depth > max_depth:
            return None
        plan: List[str] = []
        sat = set(satisfied)
        for sub in conjuncts(g):
            k = sub.canonical_key()
            if holds(sub) or k in sat:
                continue
            if k in seen:                       # cycle — this branch fails
                return None
            chosen = None
            cands = transitions_for_goal(claims, sub)
            if prefer is not None:
                cands = sorted(cands, key=prefer)
            for c in cands:
                res = solve(c.pre, sat, seen | {k}, depth + 1)
                if res is not None:
                    chosen = (c, res)
                    break
            if chosen is None:
                return None
            c, (sub_plan, sub_sat) = chosen
            plan.extend(sub_plan)
            plan.append(c.action)
            sat = set(sub_sat)
            sat.add(k)
        return plan, sat

    out = solve(goal, set(), frozenset(), 0)
    return None if out is None else out[0]


# ---------------------------------------------------------------------------
# WorldState-backed reverse index (active, credence-filtered hypotheses)
# ---------------------------------------------------------------------------


def transitions_achieving(ws, goal: Condition, *, min_credence: float = 0.0
                          ) -> List[RegressionOption]:
    """Reverse lookup over a live WorldState's hypotheses: TransitionClaims
    whose ``post`` entails ``goal``, credence ≥ ``min_credence``, with unmet
    preconditions evaluated against ``ws`` itself."""
    opts: List[RegressionOption] = []
    for h in getattr(ws, "hypotheses", {}).values():
        claim = getattr(h, "claim", None)
        if not isinstance(claim, TransitionClaim):
            continue
        cred = getattr(getattr(h, "credence", None), "point", 1.0)
        if cred < min_credence:
            continue
        if post_entails(claim.post, goal):
            opts.append(RegressionOption(
                action=claim.action, pre=claim.pre, post=claim.post,
                unmet=_unmet(claim.pre, ws), claim_id=getattr(h, "id", None),
                credence=cred))
    return opts


# ---------------------------------------------------------------------------
# Minimal manipulation-transition miner (output shape)
# ---------------------------------------------------------------------------


def make_transition_claim(action: str, pre: Condition, post: Condition
                          ) -> TransitionClaim:
    """Mint a TransitionClaim from an observed (action, pre, post). This is the
    output of the manipulation-transition miner: it captures effects on *any*
    entity (not just the controlled actor) in the relation/affordance
    vocabulary. Wiring it into the live miner pipeline (propose with Scope/step
    each turn from the perception delta) is the integration step; the claim
    shape and its consumption by ``regress`` are what this module fixes."""
    return TransitionClaim(action=action, pre=pre, post=post)
