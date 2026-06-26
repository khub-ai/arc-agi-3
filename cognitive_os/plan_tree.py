"""Plan-tree primitive — the strategic-context layer over the goal forest.

See :doc:`docs/SPEC_plan_tree.md` for the design.  Two things this
module supplies:

* A closed vocabulary of typed node kinds:
  ``TASK / WIN_CONDITION / ALIGNMENT_DIM / REACH_CELL / REACH_ENTITY /
  INVESTIGATE / OBSERVE_EFFECT / EXECUTE``.

* Status-aggregation machinery: leaves wrap the underlying goal-
  forest goal's status; internal nodes aggregate their children's
  statuses via the kind-specific rule (TASK = OR, alignment-flavor
  WIN_CONDITION = AND, sequence-flavor WIN_CONDITION = SEQUENCE,
  etc.).

The plan tree is a typed parent-child overlay on the existing goal
forest.  Leaves reference goal-forest goals by id (``gf_goal_id``);
internal nodes are pure structure.  Status flows up via aggregation;
selection flows down via the scheduler (see ``plan_scheduler``).

Engine-clean: no game ids, no specific fingerprints, no human-
readable kind labels other than the closed-vocab kind enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)


# ---------------------------------------------------------------------------
# Closed-vocab enums.
# ---------------------------------------------------------------------------


class NodeKind(Enum):
    """Closed vocabulary of plan-tree node kinds.  New kinds are added
    only by deliberate engine change — same discipline as the
    affordance DSL's closed primitive set.  New games are absorbed
    by composition over the existing kinds, not by new kinds."""
    TASK            = "TASK"
    WIN_CONDITION   = "WIN_CONDITION"
    ALIGNMENT_DIM   = "ALIGNMENT_DIM"
    REACH_CELL      = "REACH_CELL"
    REACH_ENTITY    = "REACH_ENTITY"
    INVESTIGATE     = "INVESTIGATE"
    OBSERVE_EFFECT  = "OBSERVE_EFFECT"
    EXECUTE         = "EXECUTE"


class NodeStatus(Enum):
    """Status of a plan-tree node.

    Five values arranged on a small lattice:

      HYPOTHETICAL  — the subtree exists as a candidate; no concrete
                      path to closure yet.
      KNOWN_REACHABLE — concrete path to closure exists; not in
                      progress yet.
      IN_PROGRESS   — the scheduler is currently driving this leaf
                      (or one of this node's descendant leaves).
      CLOSED        — sub-condition satisfied / leaf's gf-goal
                      ACHIEVED.
      ABANDONED     — sub-condition proved unreachable / contradicted
                      / repeatedly failing; should not be re-tried
                      without re-evaluation.
    """
    HYPOTHETICAL    = "HYPOTHETICAL"
    KNOWN_REACHABLE = "KNOWN_REACHABLE"
    IN_PROGRESS     = "IN_PROGRESS"
    CLOSED          = "CLOSED"
    ABANDONED       = "ABANDONED"


class Combinator(Enum):
    """How an internal node aggregates its children's statuses.

    AND -- all children must close.
    OR  -- any one child closing closes the parent.
    SEQUENCE -- children close in order; status reflects the next
                child to close.
    """
    AND      = "AND"
    OR       = "OR"
    SEQUENCE = "SEQUENCE"


# ---------------------------------------------------------------------------
# Default combinator for each kind.  Some kinds have a fixed
# combinator (TASK is always OR); some carry their combinator on the
# node (WIN_CONDITION can be alignment-AND, reach-OR, or sequence).
# Values here are the DEFAULT used when a node doesn't specify.
# ---------------------------------------------------------------------------


_DEFAULT_COMBINATOR: Mapping[NodeKind, Combinator] = {
    NodeKind.TASK:           Combinator.OR,
    NodeKind.WIN_CONDITION:  Combinator.AND,
    NodeKind.ALIGNMENT_DIM:  Combinator.AND,
    NodeKind.REACH_CELL:     Combinator.OR,
    NodeKind.REACH_ENTITY:   Combinator.OR,
    NodeKind.INVESTIGATE:    Combinator.AND,
    # OBSERVE_EFFECT and EXECUTE are leaves; combinator is unused.
}


_LEAF_KINDS = frozenset({NodeKind.OBSERVE_EFFECT, NodeKind.EXECUTE})


# ---------------------------------------------------------------------------
# PlanNode -- the central data structure.
# ---------------------------------------------------------------------------


@dataclass
class PlanNode:
    """One node in the plan tree.

    All fields except ``kind`` carry sensible defaults; the typical
    construction site supplies kind + children + a handful of
    typed properties.

    Status and confidence are mutable and updated by the
    status-aggregation pass.  All other fields are intended to be
    write-once at construction time; mutation outside the
    aggregation path is an error.
    """
    kind:             NodeKind
    children:         List["PlanNode"] = field(default_factory=list)
    combinator:       Optional[Combinator] = None
    status:           NodeStatus = NodeStatus.HYPOTHETICAL
    confidence:       float = 0.5
    properties:       Mapping[str, Any] = field(default_factory=dict)
    applies_to_level: Optional[str] = None
    depends_on:       Tuple[str, ...] = ()
    provenance:       str = "registry"
    gf_goal_id:       Optional[str] = None
    node_id:          Optional[str] = None  # assigned at attach time
    parent:           Optional["PlanNode"] = field(default=None,
                                                    repr=False,
                                                    compare=False)

    def __post_init__(self) -> None:
        if self.combinator is None and self.kind not in _LEAF_KINDS:
            self.combinator = _DEFAULT_COMBINATOR[self.kind]
        if self.kind in _LEAF_KINDS and self.children:
            raise ValueError(
                f"leaf kind {self.kind.value} cannot have children; got "
                f"{len(self.children)}"
            )

    @property
    def is_leaf(self) -> bool:
        return self.kind in _LEAF_KINDS

    def walk(self) -> Iterable["PlanNode"]:
        """Depth-first iteration over self + descendants."""
        yield self
        for child in self.children:
            yield from child.walk()

    def find(
        self,
        predicate: "Callable[[PlanNode], bool]",
    ) -> List["PlanNode"]:
        """Return all nodes in self+descendants matching ``predicate``."""
        return [n for n in self.walk() if predicate(n)]

    def __repr__(self) -> str:
        base = (f"PlanNode(kind={self.kind.value}, "
                f"status={self.status.value}, "
                f"confidence={self.confidence:.2f}")
        if self.gf_goal_id:
            base += f", gf_goal_id={self.gf_goal_id!r}"
        if self.properties:
            base += f", properties={dict(self.properties)!r}"
        if self.children:
            base += f", children={len(self.children)}"
        return base + ")"


# ---------------------------------------------------------------------------
# Tree construction helpers.  Builders return the assembled tree
# with parent pointers wired and node_ids assigned via a depth-first
# numbering scheme.
# ---------------------------------------------------------------------------


def assign_node_ids(root: PlanNode, prefix: str = "n") -> None:
    """Assign deterministic node_ids to root and descendants.

    IDs are of the form ``<prefix>.<i>.<j>.<k>...`` so a node's id
    encodes its path from the root.  Re-running on the same tree
    produces identical ids.
    """
    def _assign(node: PlanNode, path: str) -> None:
        node.node_id = path
        for i, child in enumerate(node.children):
            child.parent = node
            _assign(child, f"{path}.{i}")
    _assign(root, prefix)


# ---------------------------------------------------------------------------
# Status aggregation.  The core update operation: given a leaf
# whose status changed (because its gf goal closed, was abandoned,
# etc.), walk up to the root and re-aggregate each ancestor.
# ---------------------------------------------------------------------------


def _terminal_status(s: NodeStatus) -> bool:
    """A status is terminal when no further updates can change it
    via aggregation alone (CLOSED, ABANDONED)."""
    return s in (NodeStatus.CLOSED, NodeStatus.ABANDONED)


def aggregate_status(node: PlanNode) -> NodeStatus:
    """Compute ``node``'s status from its children's statuses + its
    own combinator.

    Pure function — does not mutate.  Caller decides whether to
    write the result back to ``node.status``.

    Leaves (OBSERVE_EFFECT, EXECUTE) are not aggregated; their
    status is set externally (typically by the gf-goal-status update
    path).  Calling this on a leaf returns the leaf's current status.
    """
    if node.is_leaf:
        return node.status
    if not node.children:
        # Internal node with no children: vacuous; treat as
        # HYPOTHETICAL until children are added.  (An empty TASK is
        # an under-construction tree.)
        return NodeStatus.HYPOTHETICAL

    statuses = [c.status for c in node.children]
    combinator = node.combinator or _DEFAULT_COMBINATOR[node.kind]

    if combinator is Combinator.AND:
        # All children must close.  Any ABANDONED makes the parent
        # ABANDONED (the AND can never be satisfied if a required
        # subtree died).  All CLOSED → CLOSED.  Any IN_PROGRESS →
        # IN_PROGRESS.  Otherwise: take the strongest open status.
        if any(s is NodeStatus.ABANDONED for s in statuses):
            return NodeStatus.ABANDONED
        if all(s is NodeStatus.CLOSED for s in statuses):
            return NodeStatus.CLOSED
        if any(s is NodeStatus.IN_PROGRESS for s in statuses):
            return NodeStatus.IN_PROGRESS
        if all(s is NodeStatus.KNOWN_REACHABLE or s is NodeStatus.CLOSED
               for s in statuses):
            return NodeStatus.KNOWN_REACHABLE
        return NodeStatus.HYPOTHETICAL

    if combinator is Combinator.OR:
        # Any one closing closes the parent.  All ABANDONED →
        # ABANDONED (no viable child remains).  Any IN_PROGRESS →
        # IN_PROGRESS.  Any KNOWN_REACHABLE → KNOWN_REACHABLE.
        # Default HYPOTHETICAL.
        if any(s is NodeStatus.CLOSED for s in statuses):
            return NodeStatus.CLOSED
        if all(s is NodeStatus.ABANDONED for s in statuses):
            return NodeStatus.ABANDONED
        if any(s is NodeStatus.IN_PROGRESS for s in statuses):
            return NodeStatus.IN_PROGRESS
        if any(s is NodeStatus.KNOWN_REACHABLE for s in statuses):
            return NodeStatus.KNOWN_REACHABLE
        return NodeStatus.HYPOTHETICAL

    if combinator is Combinator.SEQUENCE:
        # Children close in order.  Once all CLOSED, parent CLOSED.
        # Status reflects the next unclosed child (or ABANDONED if
        # any required step abandoned before reaching the end).
        if all(s is NodeStatus.CLOSED for s in statuses):
            return NodeStatus.CLOSED
        next_unclosed = next(
            (c for c in node.children if c.status is not NodeStatus.CLOSED),
            None,
        )
        if next_unclosed is None:
            return NodeStatus.CLOSED  # defensive
        if next_unclosed.status is NodeStatus.ABANDONED:
            return NodeStatus.ABANDONED
        return next_unclosed.status

    # Defensive: unknown combinator.
    return NodeStatus.HYPOTHETICAL


def refresh_status_up(node: PlanNode) -> List[PlanNode]:
    """Walk up from ``node`` to the root, re-aggregating each
    ancestor's status.  Returns the list of ancestors (including
    the root) whose status changed.

    Stops walking when a level's recomputed status equals the
    cached one (further ancestors won't change).
    """
    changed: List[PlanNode] = []
    cur = node.parent
    while cur is not None:
        new_status = aggregate_status(cur)
        if new_status is cur.status:
            break
        cur.status = new_status
        changed.append(cur)
        cur = cur.parent
    return changed


def set_leaf_status(leaf: PlanNode, status: NodeStatus) -> List[PlanNode]:
    """Update a leaf's status and propagate aggregation up.  Returns
    the list of ancestors (including the root) whose status changed
    as a result.
    """
    if not leaf.is_leaf:
        raise ValueError(
            f"set_leaf_status called on non-leaf {leaf.kind.value}; "
            f"use refresh_status_up after updating children directly"
        )
    if leaf.status is status:
        return []
    leaf.status = status
    return refresh_status_up(leaf)


# ---------------------------------------------------------------------------
# Convenience builders.  Templates use these to construct subtrees
# with sensible defaults.
# ---------------------------------------------------------------------------


def task(
    children: Sequence[PlanNode],
    *,
    properties: Optional[Mapping[str, Any]] = None,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build a TASK root with the given WIN_CONDITION candidates."""
    return PlanNode(
        kind=NodeKind.TASK,
        children=list(children),
        properties=properties or {},
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def win_condition(
    kind_label: str,
    children: Sequence[PlanNode],
    *,
    combinator: Optional[Combinator] = None,
    confidence: float = 0.5,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build a WIN_CONDITION node.

    ``kind_label`` is a closed-vocab descriptor of the win flavour
    (``alignment`` / ``reach`` / ``sequence``).  It's stored in the
    node's properties and used by the dimension-aware mining /
    distillation paths.
    """
    if combinator is None:
        combinator = (Combinator.SEQUENCE if kind_label == "sequence"
                      else Combinator.AND if kind_label == "alignment"
                      else Combinator.OR)
    return PlanNode(
        kind=NodeKind.WIN_CONDITION,
        children=list(children),
        combinator=combinator,
        properties={"flavor": kind_label},
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def alignment_dim(
    property_name: str,
    children: Sequence[PlanNode],
    *,
    confidence: float = 0.5,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build an ALIGNMENT_DIM node for the named property."""
    return PlanNode(
        kind=NodeKind.ALIGNMENT_DIM,
        children=list(children),
        properties={"property": property_name},
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def reach_entity(
    entity_class: str,
    strategies: Sequence[PlanNode],
    *,
    confidence: float = 0.5,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build a REACH_ENTITY node with the given strategy children.

    Strategies are themselves PlanNodes (typically EXECUTE leaves or
    other intermediate nodes).  The REACH_ENTITY's status is OR
    over strategies — whichever produces a viable plan first wins.
    """
    return PlanNode(
        kind=NodeKind.REACH_ENTITY,
        children=list(strategies),
        properties={"entity_class": entity_class},
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def reach_cell(
    cell: Tuple[int, int],
    strategies: Sequence[PlanNode],
    *,
    confidence: float = 0.5,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build a REACH_CELL node for a specific cell coordinate."""
    return PlanNode(
        kind=NodeKind.REACH_CELL,
        children=list(strategies),
        properties={"cell": tuple(int(x) for x in cell)},
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def investigate(
    target_class: str,
    children: Sequence[PlanNode],
    *,
    confidence: float = 0.5,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build an INVESTIGATE node (typically REACH_ENTITY + OBSERVE_EFFECT)."""
    return PlanNode(
        kind=NodeKind.INVESTIGATE,
        children=list(children),
        properties={"target_class": target_class},
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def observe_effect(
    expected_kind: str,
    *,
    confidence: float = 0.5,
    properties: Optional[Mapping[str, Any]] = None,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build an OBSERVE_EFFECT leaf.

    Closes when the engine observes a state change matching the
    named effect kind.  Properties carry the specific predicate
    (e.g., expected effect signature).
    """
    props = dict(properties or {})
    props["expected_kind"] = expected_kind
    return PlanNode(
        kind=NodeKind.OBSERVE_EFFECT,
        properties=props,
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )


def execute(
    gf_goal_id: str,
    *,
    confidence: float = 0.5,
    properties: Optional[Mapping[str, Any]] = None,
    applies_to_level: Optional[str] = None,
    provenance: str = "registry",
) -> PlanNode:
    """Build an EXECUTE leaf that wraps a goal-forest goal id."""
    return PlanNode(
        kind=NodeKind.EXECUTE,
        gf_goal_id=gf_goal_id,
        properties=properties or {},
        confidence=confidence,
        applies_to_level=applies_to_level,
        provenance=provenance,
    )
