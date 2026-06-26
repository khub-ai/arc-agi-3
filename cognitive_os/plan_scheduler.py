"""Plan-tree scheduler — picks the active EXECUTE leaf per turn.

See :doc:`docs/SPEC_plan_tree.md` §Scheduling.  This is the
selection-flows-down half of the plan-tree contract: walk the
tree, find the highest-confidence ready leaf, return it for the
existing planner to dispatch.

Pure function over a plan tree; no side effects.  Adding new
selection criteria is a matter of writing a ranking function and
chaining it; no other engine surgery needed.

The scheduler explicitly does NOT mutate the tree.  Status
updates (a leaf closes, an ABANDONED branch propagates) flow in
the opposite direction via ``plan_tree.set_leaf_status`` /
``refresh_status_up`` and are caller-driven.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .plan_tree import (
    NodeKind,
    NodeStatus,
    PlanNode,
)


# ---------------------------------------------------------------------------
# Selection criteria.  Each is a tiny function over a single leaf
# (in tree context) returning a numerical score.  The scheduler
# combines them via a weighted sum.
# ---------------------------------------------------------------------------


def _ancestor_confidence(leaf: PlanNode) -> float:
    """Aggregated confidence of the leaf + every ancestor up to the
    root.  Multiplicative — a low-confidence ancestor demotes all
    its descendant leaves' priority by that factor.
    """
    score = max(0.0, min(1.0, leaf.confidence))
    cur = leaf.parent
    while cur is not None:
        score *= max(0.0, min(1.0, cur.confidence))
        cur = cur.parent
    return score


def _status_weight(leaf: PlanNode) -> float:
    """Selection prefers ready-to-act leaves.

    KNOWN_REACHABLE  — direct execution, weight 1.0.
    IN_PROGRESS      — already being driven, slightly lower (caller
                       should keep going only if no better choice).
    HYPOTHETICAL     — only chosen when nothing better is around;
                       weight 0.4.
    CLOSED, ABANDONED — never selected.
    """
    table = {
        NodeStatus.KNOWN_REACHABLE: 1.0,
        NodeStatus.IN_PROGRESS:     0.95,
        NodeStatus.HYPOTHETICAL:    0.4,
        NodeStatus.CLOSED:          0.0,
        NodeStatus.ABANDONED:       0.0,
    }
    return table.get(leaf.status, 0.0)


# Default weights — tunable; design call is per-deployment.
DEFAULT_WEIGHTS = {
    "ancestor_confidence": 1.0,
    "status_weight":       1.0,
}


# ---------------------------------------------------------------------------
# Selection result.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchedulerChoice:
    """The scheduler's per-turn output.

    ``leaf`` — the picked EXECUTE leaf (or ``None`` when no leaf is
       selectable).
    ``score`` — the leaf's composite score, useful for telemetry.
    ``rationale`` — short, structured strings describing why it
       was picked.  Closed vocabulary so the meta-monitor and
       audit code can branch on it.
    """
    leaf:      Optional[PlanNode]
    score:     float
    rationale: tuple


def _score(leaf: PlanNode, weights: dict) -> float:
    return (
        weights.get("ancestor_confidence", 1.0) * _ancestor_confidence(leaf)
        * weights.get("status_weight", 1.0) * _status_weight(leaf)
    )


# ---------------------------------------------------------------------------
# Dependency check.  A leaf is "ready" only when none of its
# depends_on goals point at unsatisfied siblings.  Today we
# resolve this against the goal forest via the leaf's gf_goal_id;
# a future pass will track dependencies at the plan-tree level
# directly.
# ---------------------------------------------------------------------------


def _deps_satisfied(
    leaf: PlanNode,
    root: PlanNode,
    gf_goal_status: Callable[[str], Optional[str]] = lambda _gid: None,
) -> bool:
    """Return True when the leaf's depends_on edges are all closed.

    ``gf_goal_status`` is a callable that resolves a goal-forest
    goal id to a status string ("achieved" / "open" / etc.).
    When None or "achieved" / "closed", the dep is considered
    satisfied.  Missing edges (no depends_on declared) are always
    satisfied.
    """
    for dep_id in leaf.depends_on:
        status = gf_goal_status(dep_id)
        if status is None:
            continue   # unknown -> permissive (we'll re-check at execute time)
        if status.lower() not in ("achieved", "closed"):
            return False
    return True


# ---------------------------------------------------------------------------
# The scheduler.
# ---------------------------------------------------------------------------


def pick_active_leaf(
    root: PlanNode,
    *,
    gf_goal_status: Callable[[str], Optional[str]] = lambda _gid: None,
    weights: Optional[dict] = None,
    leaf_eligible: Optional[Callable[[PlanNode], bool]] = None,
) -> SchedulerChoice:
    """Walk the plan tree, pick the best EXECUTE leaf, return it.

    Selection rule (composite, ordered by tie-break):

      1. Highest composite score (ancestor-confidence × status-weight).
      2. Tie-break on lowest depth in the tree (closer leaves to
         the root are conceptually "broader" — preferred when other
         scores tie).
      3. Tie-break on node_id (deterministic).

    ``leaf_eligible``, when supplied, is called for every otherwise-
    eligible leaf and must return True for the leaf to be considered.
    Use this to plug in transient filters (BFS reachability from the
    agent's current cell, budget gates, etc.) that aren't captured by
    the leaf's persistent status.  When None, every status-eligible
    leaf is considered.

    Returns a ``SchedulerChoice`` whose ``leaf`` is ``None`` when
    no EXECUTE leaf is ready.  Caller falls back to the existing
    flat-goal-forest behaviour in that case.
    """
    w = weights or DEFAULT_WEIGHTS

    # Walk the tree, collect candidate leaves.
    candidates: List[tuple] = []
    for node in root.walk():
        if node.kind is not NodeKind.EXECUTE:
            continue
        if node.status in (NodeStatus.CLOSED, NodeStatus.ABANDONED):
            continue
        if not _deps_satisfied(node, root, gf_goal_status):
            continue
        if leaf_eligible is not None:
            try:
                if not leaf_eligible(node):
                    continue
            except Exception:
                # A failing eligibility check is permissive — better
                # to consider the leaf than silently exclude it.
                pass
        score = _score(node, w)
        if score <= 0.0:
            continue
        # Depth: number of ancestors.
        depth = 0
        cur = node.parent
        while cur is not None:
            depth += 1
            cur = cur.parent
        nid = node.node_id or ""
        candidates.append((-score, depth, nid, node))

    if not candidates:
        return SchedulerChoice(
            leaf      = None,
            score     = 0.0,
            rationale = ("no-eligible-leaf",),
        )

    candidates.sort()   # negative score first, then depth, then id
    _, depth, _, picked = candidates[0]
    rationale = _build_rationale(picked, depth, len(candidates))
    return SchedulerChoice(
        leaf      = picked,
        score     = -candidates[0][0],
        rationale = rationale,
    )


def _build_rationale(
    leaf: PlanNode,
    depth: int,
    n_candidates: int,
) -> tuple:
    """Closed-vocab string tuple describing the leaf-pick decision."""
    parts: List[str] = []
    parts.append(f"kind={leaf.kind.value}")
    parts.append(f"status={leaf.status.value}")
    parts.append(f"confidence={leaf.confidence:.2f}")
    parts.append(f"depth={depth}")
    parts.append(f"n_candidates={n_candidates}")
    # Add parent kind context (which WIN_CONDITION flavour, etc.).
    cur = leaf.parent
    while cur is not None:
        flavor = (cur.properties or {}).get("flavor")
        if flavor:
            parts.append(f"under={cur.kind.value}:{flavor}")
            break
        if cur.kind is NodeKind.TASK:
            parts.append("under=TASK")
            break
        cur = cur.parent
    return tuple(parts)
