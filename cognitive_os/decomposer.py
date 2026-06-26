"""Observation-driven goal decomposition.

Domain-agnostic engine primitive that decomposes abstract open
goals into cell-anchored sub-goals by querying ``CausalClaim``s
in the hypothesis store.  See
``docs/SPEC_observation_driven_decomposition.md`` for the
architecture.

The decomposer is read-only against the hypothesis store and
goal forest — it produces ``GoalDeclaration`` records the adapter
declares via its runtime.  Engine module never imports adapter
code; conditions opt-in via a duck-typed ``decomposition_targets``
method.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable, List, Optional


# ===========================================================================
# Types
# ===========================================================================


@dataclass(frozen=True)
class DecompositionTarget:
    """What the decomposer should look for in the hypothesis store
    when considering an abstract goal for sub-goal declaration.

    A goal's ``Condition`` returns a list of these from
    ``decomposition_targets(ws)``.  The decomposer then iterates
    committed ``CausalClaim``s, matching each claim's effect against
    the target's ``entity_ids`` (required) and ``properties``
    (optional — empty means "any property change is a candidate").
    """
    entity_ids: FrozenSet[str]
    properties: FrozenSet[str] = frozenset()
    label:      str = ""


@dataclass(frozen=True)
class DecomposedGoal:
    """One sub-goal declaration the decomposer proposes.  The adapter
    consumes this and calls its runtime's declare method with
    domain-appropriate goal shape (cell-anchored ``AgentAtCell`` for
    ARC, ``AgentAtJointConfig`` for robotics, etc.).
    """
    sub_goal_id:    str
    parent_goal_id: str
    trigger_cell:   "tuple[int, int]"
    priority:       float
    label:          str = ""
    provenance:     str = "engine:decomposer"


# ===========================================================================
# Algorithm
# ===========================================================================


_PRIORITY_DECAY: float = 0.7


def propose_decomposition_goals(
    ws,
    *,
    forbidden_cells: "Optional[set]" = None,
    priority_decay:   float = _PRIORITY_DECAY,
) -> "List[DecomposedGoal]":
    """Walk open goals, query the hypothesis store, yield cell-
    anchored sub-goal proposals.

    Args:
      ws:              WorldState — selector + goal forest live here.
      forbidden_cells: optional set of (r, c) cells to filter out
                       (typically the destabilizing-cell set for the
                       current level — cells that would regress an
                       aligned dim).  Empty when None.
      priority_decay:  multiplier applied to parent goal priority
                       for sub-goal priority (default 0.7 — well
                       below Oracle-declared cell directives at
                       0.85 so the engine's observation-driven
                       proposal doesn't overrule a fresh Oracle
                       directive, but above the always-available
                       harness-exploration fallback so it fires
                       when the directive set is exhausted).
                       Operator framing 2026-05-08 trial_decomposer_v3
                       turn 5: agent went backward to a prior-
                       observation cell because decompose_align at
                       0.9025 (decay=0.95) outranked Oracle's
                       interact_r0_c4 at 0.85.

    Returns a list of ``DecomposedGoal`` records, one per unique
    (parent_goal, trigger_cell) pair that survives filtering.  An
    empty list means no decomposition was possible — either no
    abstract goals are open, or no CausalClaims point to a non-
    forbidden cell that would advance any open goal.

    Domain-agnostic.  The trigger cell is extracted from the
    claim's ``trigger.canonical_key()`` first element being
    ``"AgentAtCell"`` — domains with a different cell-trigger
    naming should adapt the canonical-key check.

    The function reads but does not mutate ``ws``; the caller
    declares the proposed goals.
    """
    forbidden = set(forbidden_cells or ())

    proposals: "dict[tuple[str, tuple[int, int]], DecomposedGoal]" = {}

    goals = _open_goals(ws)
    if not goals:
        return []

    causal_claims = _committed_causal_claims(ws)
    if not causal_claims:
        return []

    for goal in goals:
        cond = _condition_of(goal)
        if cond is None:
            continue
        targets = _decomposition_targets(cond, ws)
        if not targets:
            continue
        parent_id = str(getattr(goal, "id", "")) or repr(goal)
        parent_pri = float(getattr(goal, "priority", 0.0) or 0.0)
        if parent_pri <= 0.0:
            continue
        sub_pri = parent_pri * float(priority_decay)
        for target in targets:
            for cell in _matching_trigger_cells(target, causal_claims):
                if cell in forbidden:
                    continue
                key = (parent_id, cell)
                if key in proposals:
                    continue   # de-dup across (target_a, target_b)
                sub_id = (
                    f"decompose_{target.label or 'via'}"
                    f"_r{cell[0]}_c{cell[1]}"
                )
                proposals[key] = DecomposedGoal(
                    sub_goal_id    = sub_id,
                    parent_goal_id = parent_id,
                    trigger_cell   = cell,
                    priority       = sub_pri,
                    label          = target.label,
                )

    return list(proposals.values())


# ===========================================================================
# Internal helpers
# ===========================================================================


def _open_goals(ws) -> "List[Any]":
    """Iterate goal-forest goals whose root status is OPEN.  Returns
    a list (not a generator) so callers can iterate freely."""
    try:
        gf = getattr(ws, "goal_forest", None)
        if gf is None:
            return []
        out: "List[Any]" = []
        for g in gf.goals.values():
            root = getattr(g, "root", None)
            status = getattr(root, "status", None) if root is not None else None
            # status is a GoalStatus enum; "OPEN" is the .name attr or
            # the value on .value.  Duck-type to stay independent.
            status_name = getattr(status, "name", str(status))
            if status_name == "OPEN":
                out.append(g)
        return out
    except Exception:
        return []


def _condition_of(goal) -> "Optional[Any]":
    """Pull the goal's root atom condition.  Goals can be compound
    (AND/OR over atom conditions); the decomposer only handles
    atom-rooted goals — compound goals decompose via their leaves
    when those leaves are themselves declared as separate goals."""
    try:
        root = getattr(goal, "root", None)
        if root is None:
            return None
        # atom node has .condition; compound has .children
        cond = getattr(root, "condition", None)
        return cond
    except Exception:
        return None


def _decomposition_targets(
    cond,
    ws,
) -> "List[DecompositionTarget]":
    """Call the condition's ``decomposition_targets(ws)`` if it
    exposes one.  Conditions without the method are treated as
    non-decomposable (return empty)."""
    fn = getattr(cond, "decomposition_targets", None)
    if not callable(fn):
        return []
    try:
        out = fn(ws)
    except Exception:
        return []
    if out is None:
        return []
    if isinstance(out, DecompositionTarget):
        return [out]
    try:
        return [t for t in out if isinstance(t, DecompositionTarget)]
    except TypeError:
        return []


_DEFAULT_CREDENCE_FLOOR: float = 0.5
_DEFAULT_PRIORITY_DECAY:  float = 0.7


def _committed_causal_claims(
    ws,
    *,
    credence_floor: float = _DEFAULT_CREDENCE_FLOOR,
) -> "List[Any]":
    """Pull CausalClaim records from the hypothesis store whose
    credence is at or above ``credence_floor``.

    Default 0.5 is the planner-facing standard (per
    ``hypothesis_store.above_credence`` docstring and
    ``SPEC_continuous_commitment.md`` P1: no binary commit gate
    drives planning).  Calling ``committed()`` here would require
    credence ≥ 0.85 and silently exclude observed-once claims that
    are still useful evidence for decomposition.

    Stays defensive: degrade to empty on any inconsistency."""
    try:
        from cognitive_os.claims import CausalClaim
        from cognitive_os import hypothesis_store as _store
    except Exception:
        return []
    try:
        candidates = list(_store.above_credence(ws, float(credence_floor)))
    except Exception:
        return []
    out: "List[Any]" = []
    for h in candidates:
        claim = getattr(h, "claim", None)
        if claim is None:
            continue
        if isinstance(claim, CausalClaim):
            out.append(claim)
    return out


def _matching_trigger_cells(
    target:  DecompositionTarget,
    claims:  "List[Any]",
) -> "Iterable[tuple[int, int]]":
    """Yield each trigger cell whose claim matches the target.

    Matching: claim.trigger is AgentAtCell, claim.effect is
    EntityInState with entity_id in target.entity_ids, and
    (when target.properties non-empty) effect.property in target.properties.

    De-duplicates within this call so a target with three matching
    claims for the same trigger cell yields the cell once.
    """
    try:
        from cognitive_os.conditions import EntityInState
    except Exception:
        return []
    seen: "set[tuple[int, int]]" = set()
    for claim in claims:
        eff = getattr(claim, "effect", None)
        if not isinstance(eff, EntityInState):
            continue
        if str(eff.entity_id) not in target.entity_ids:
            continue
        if (target.properties
                and str(eff.property) not in target.properties):
            continue
        trig = getattr(claim, "trigger", None)
        if trig is None:
            continue
        try:
            tk = trig.canonical_key()
        except Exception:
            continue
        if not (tk and len(tk) >= 3 and tk[0] == "AgentAtCell"):
            continue
        try:
            cell = (int(tk[1]), int(tk[2]))
        except (TypeError, ValueError):
            continue
        if cell in seen:
            continue
        seen.add(cell)
        yield cell
