"""Integration layer between the goal forest and the plan tree.

The plan-tree primitives in :mod:`plan_tree` and the registry-driven
templates in :mod:`plan_templates` know nothing about the existing
goal-forest.  This module bridges them: it builds the level's TASK
root, decides which template-emitted WIN_CONDITION categories apply,
classifies every open goal-forest goal into one of those categories,
and emits ``EXECUTE`` leaves wrapping the goal ids so the scheduler
can pick among them.

Engine-clean: no game ids, no specific fingerprints.  The only
inputs are the world state, the affordance registry, and the open
goals' typed ``Condition.canonical_key()`` tags.

See :doc:`docs/SPEC_plan_tree.md` §"Existing GF goals from elsewhere".
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, List, Mapping, Optional, Sequence, Tuple

from .plan_tree import (
    Combinator,
    NodeKind,
    NodeStatus,
    PlanNode,
    assign_node_ids,
    execute,
    reach_cell,
    refresh_status_up,
    set_leaf_status,
    task,
    win_condition,
)
from .plan_templates import build_default_plan


# ---------------------------------------------------------------------------
# Closed-vocabulary categories.  Every open GF goal is bucketed into
# exactly one of these by its Condition's canonical_key tag.  New
# categories require a deliberate engine change — same discipline
# as the plan-tree NodeKind enum.
# ---------------------------------------------------------------------------


CAT_ALIGNMENT      = "alignment"
CAT_REACH          = "reach"
CAT_INVESTIGATE    = "investigate"
CAT_UNCATEGORIZED  = "uncategorized"


# Mapping from Condition.canonical_key()[0] (the tag) to a category.
# Atomic conditions only — composite goals (AND / OR) fall through to
# CAT_UNCATEGORIZED until a proper subtree-walk classifier is wired.
_TAG_TO_CATEGORY: Mapping[str, str] = {
    # Reach-family.
    "AgentAtCell":                  CAT_REACH,
    "AtPosition":                   CAT_REACH,
    "AgentAtEntityClass":           CAT_REACH,
    "AgentAtCellRelativeToEntity":  CAT_REACH,
    # Alignment-family.
    "OrientationAligned":           CAT_ALIGNMENT,
    "EntitiesEquivalent":           CAT_ALIGNMENT,
    "EntitiesVisuallyMatch":        CAT_ALIGNMENT,
    # Investigation-family.
    "EntityProbed":                 CAT_INVESTIGATE,
    "EntityInState":                CAT_INVESTIGATE,
}


# Closed set of non-actionable canonical_key tags.  Goals with these
# tags are progression sentinels / meta-bookkeeping markers; they
# close when *something else* drives the world into the right state,
# not when the agent dispatches an action toward them.  Including
# them as EXECUTE leaves would make the scheduler latch onto a goal
# nothing can drive.  Excluded from both the build path and the
# mid-level refresh.
_NON_ACTIONABLE_TAGS: frozenset = frozenset({
    "LevelAdvanced",      # closes when levels_completed surpasses N
    "ResourceAbove",      # closes when a resource crosses threshold
    "ResourceBelow",
    "ResourceRestored",
    "BudgetBelow",
    "MotionModelCommitted",
    "ActionTried",        # closes once the action has been tried
    "ActionJustTaken",
    "FrameChangedPattern",
    "RegionMotion",
})


# Map a category to a default confidence.  Tunable.  Reach > alignment
# is a safe v1 default for ls20-shape levels where the alignment
# subtree often depends on later-discovered effects; the agent does
# better attempting reach first while alignment evidence accumulates.
# Templates can override these per-level via build_default_plan.
_DEFAULT_CATEGORY_CONFIDENCE: Mapping[str, float] = {
    CAT_ALIGNMENT:     0.55,
    CAT_REACH:         0.50,
    CAT_INVESTIGATE:   0.30,
    CAT_UNCATEGORIZED: 0.25,
}


# ---------------------------------------------------------------------------
# Goal classification.
# ---------------------------------------------------------------------------


def _canonical_tag(goal) -> Optional[str]:
    """Return ``goal.root.condition.canonical_key()[0]`` defensively.

    Returns None when the goal lacks a condition or its canonical_key
    raises (composite goals, malformed conditions, etc.).
    """
    try:
        cond = goal.root.condition
        if cond is None:
            return None
        key = cond.canonical_key()
        if isinstance(key, tuple) and key:
            return str(key[0])
    except (AttributeError, TypeError, ValueError):
        pass
    return None


def classify_goal(goal) -> str:
    """Return the closed-vocab category for a goal-forest goal.

    Reads ``goal.root.condition.canonical_key()[0]`` and maps it through
    ``_TAG_TO_CATEGORY``.  Anything unrecognised — composite goals,
    runtime-declared goals with non-standard tags — lands in
    ``CAT_UNCATEGORIZED``.
    """
    tag = _canonical_tag(goal)
    if tag is None:
        return CAT_UNCATEGORIZED
    return _TAG_TO_CATEGORY.get(tag, CAT_UNCATEGORIZED)


# Internal alias preserved for the build path's call sites.
_classify_goal = classify_goal


def _cell_target(goal) -> Optional[Tuple[int, int]]:
    """Extract a cell coordinate from the goal's condition when one
    is exposed.  Used to group reach-category goals under per-cell
    REACH_CELL parents for ancestor-confidence sharing.
    """
    try:
        cond = goal.root.condition
        if cond is None:
            return None
        # AgentAtCell exposes cell_target() returning (r, c).
        target = getattr(cond, "cell_target", None)
        if callable(target):
            v = target()
            if v is not None:
                return (int(v[0]), int(v[1]))
        # AtPosition uses .pos field.
        pos = getattr(cond, "pos", None)
        if pos is not None and len(pos) >= 2:
            return (int(pos[0]), int(pos[1]))
    except (AttributeError, TypeError, ValueError, IndexError):
        pass
    return None


def _goal_status_to_node_status(goal) -> NodeStatus:
    """Map a goal-forest status to the plan-tree leaf status."""
    try:
        st = goal.root.status
        name = getattr(st, "name", str(st))
    except AttributeError:
        return NodeStatus.HYPOTHETICAL
    if name == "ACHIEVED":
        return NodeStatus.CLOSED
    if name in ("ABANDONED", "PRUNED"):
        return NodeStatus.ABANDONED
    if name == "OPEN":
        return NodeStatus.KNOWN_REACHABLE
    return NodeStatus.HYPOTHETICAL


def _goal_priority(goal) -> float:
    try:
        return float(getattr(goal, "priority", 0.5) or 0.5)
    except (TypeError, ValueError):
        return 0.5


# ---------------------------------------------------------------------------
# Iterate the goal forest.
# ---------------------------------------------------------------------------


def _iter_open_goals(ws) -> Iterable[Tuple[str, Any]]:
    """Yield ``(goal_id, goal)`` for every goal in ``ws.goal_forest``
    whose status is not ACHIEVED or ABANDONED *and* whose canonical
    key tag isn't in the non-actionable-sentinel set.  Defensive
    against a missing goal_forest (tests can pass an empty ws stub).
    """
    try:
        forest = getattr(ws, "goal_forest", None)
        if forest is None:
            return
        goals = getattr(forest, "goals", {}) or {}
    except AttributeError:
        return
    for gid, goal in goals.items():
        try:
            name = goal.root.status.name
        except AttributeError:
            name = "OPEN"
        if name in ("ACHIEVED", "ABANDONED", "PRUNED"):
            continue
        tag = _canonical_tag(goal)
        if tag is not None and tag in _NON_ACTIONABLE_TAGS:
            continue
        yield str(gid), goal


# ---------------------------------------------------------------------------
# Build the level's plan tree from open goals + template hints.
# ---------------------------------------------------------------------------


def build_plan_tree_for_level(
    *,
    ws,
    kb,
    level_key: str,
    affordance_registry: Optional[Mapping[str, Any]] = None,
    framed_candidates: Optional[Sequence] = None,
) -> Optional[PlanNode]:
    """Construct the level's plan tree from open GF goals + templates.

    The construction is goal-forest-driven: every open goal becomes
    an ``EXECUTE`` leaf grouped under a category-keyed WIN_CONDITION
    parent.  Templates provide priors (which categories are plausible
    and at what confidence); the actual leaf set comes from the
    forest so the scheduler can dispatch through the existing planner.

    Returns the TASK root with ids assigned, or ``None`` when no
    open goals are present (caller falls back to flat selection).
    """
    open_goals: List[Tuple[str, Any]] = list(_iter_open_goals(ws))
    if not open_goals:
        return None

    # Run templates for prior signal — which WIN_CONDITION flavors
    # apply on this level and at what confidence.
    template_hints: Mapping[str, float] = _template_category_priors(
        ws            = ws,
        kb            = kb,
        level_key     = level_key,
        affordance_registry = affordance_registry or {},
        framed_candidates   = framed_candidates or [],
    )

    # Bucket open goals by category.
    buckets: dict[str, list[tuple[str, Any]]] = {}
    for gid, goal in open_goals:
        cat = _classify_goal(goal)
        buckets.setdefault(cat, []).append((gid, goal))

    # Assemble WIN_CONDITION children of the TASK root, one per
    # non-empty category bucket.  Confidence prefers the template
    # hint when present; otherwise the closed-vocab default.
    win_conds: List[PlanNode] = []
    for cat in (CAT_ALIGNMENT, CAT_REACH, CAT_INVESTIGATE, CAT_UNCATEGORIZED):
        goals_in_cat = buckets.get(cat, [])
        if not goals_in_cat:
            continue
        conf = float(template_hints.get(
            cat, _DEFAULT_CATEGORY_CONFIDENCE.get(cat, 0.4),
        ))
        win_conds.append(_build_category_subtree(
            category        = cat,
            goals_in_cat    = goals_in_cat,
            confidence      = conf,
            level_key       = level_key,
        ))

    if not win_conds:
        return None

    root = task(
        children          = win_conds,
        properties        = {"level_key": str(level_key)},
        applies_to_level  = str(level_key),
        provenance        = "registry",
    )
    assign_node_ids(root, prefix=f"plan.l{level_key}")
    # Initial status sweep: leaves' statuses were set at build time;
    # walk up and aggregate so internal nodes reflect them.
    for leaf in root.find(lambda n: n.is_leaf):
        refresh_status_up(leaf)
    return root


def _template_category_priors(
    *,
    ws,
    kb,
    level_key: str,
    affordance_registry: Mapping[str, Any],
    framed_candidates: Sequence,
) -> Mapping[str, float]:
    """Run the registered templates and extract per-category confidences.

    Each registered template emits a WIN_CONDITION whose
    ``properties["flavor"]`` names the category.  We collect the
    flavors and their template-supplied confidence values.  Categories
    no template activated for fall back to the closed-vocab default
    at build time.
    """
    hints: dict[str, float] = {}
    try:
        candidates = build_default_plan(
            ws                  = ws,
            kb                  = kb,
            level_key           = level_key,
            affordance_registry = affordance_registry,
            framed_candidates   = framed_candidates,
        )
    except Exception:
        candidates = []
    for node in candidates:
        flavor = (node.properties or {}).get("flavor")
        if flavor is None:
            continue
        # Map template flavors to attach categories.
        cat = {
            "alignment":    CAT_ALIGNMENT,
            "reach":        CAT_REACH,
            "investigate":  CAT_INVESTIGATE,
        }.get(str(flavor))
        if cat is None:
            continue
        hints[cat] = max(hints.get(cat, 0.0), float(node.confidence))
    return hints


def _build_category_subtree(
    *,
    category:     str,
    goals_in_cat: Sequence[Tuple[str, Any]],
    confidence:   float,
    level_key:    str,
) -> PlanNode:
    """Build a WIN_CONDITION subtree for one category.

    Reach goals are further grouped under per-cell REACH_CELL parents
    so two goals targeting the same cell share a parent (their
    aggregated confidence is the cell's, not the sum).  Other
    categories attach EXECUTE leaves directly under the WIN_CONDITION.

    Every WIN_CONDITION here uses OR — any one EXECUTE leaf closing
    is sufficient for the category to be considered satisfied (the
    actual level-completion test happens at the goal-forest level;
    the plan tree is structural overlay).
    """
    if category == CAT_REACH:
        children = _build_reach_children(
            goals_in_cat = goals_in_cat,
            confidence   = confidence,
            level_key    = level_key,
        )
    else:
        children = [
            execute(
                gf_goal_id        = gid,
                confidence        = _goal_priority(goal),
                properties        = {"category": category},
                applies_to_level  = str(level_key),
                provenance        = "attach",
            )
            for gid, goal in goals_in_cat
        ]
        # Initial leaf statuses mirror the GF goal's status.
        for leaf, (_, goal) in zip(children, goals_in_cat):
            leaf.status = _goal_status_to_node_status(goal)

    return win_condition(
        kind_label        = (category if category != CAT_UNCATEGORIZED
                             else "reach"),  # uncategorized uses OR-aggregator
        children          = children,
        combinator        = Combinator.OR,
        confidence        = confidence,
        applies_to_level  = str(level_key),
        provenance        = "attach",
    )


def _build_reach_children(
    *,
    goals_in_cat: Sequence[Tuple[str, Any]],
    confidence:   float,
    level_key:    str,
) -> List[PlanNode]:
    """Group reach-category goals under per-cell REACH_CELL parents.

    A goal exposing a cell target gets a REACH_CELL parent keyed on
    that cell; multiple goals sharing the cell share the parent.
    Goals without a cell target attach directly as EXECUTE leaves.
    """
    by_cell: dict[Tuple[int, int], list[Tuple[str, Any]]] = {}
    no_cell: list[Tuple[str, Any]] = []
    for gid, goal in goals_in_cat:
        cell = _cell_target(goal)
        if cell is None:
            no_cell.append((gid, goal))
        else:
            by_cell.setdefault(cell, []).append((gid, goal))

    children: List[PlanNode] = []
    for cell, goals in by_cell.items():
        leaves: List[PlanNode] = []
        for gid, goal in goals:
            leaf = execute(
                gf_goal_id        = gid,
                confidence        = _goal_priority(goal),
                properties        = {"category": CAT_REACH, "cell": cell},
                applies_to_level  = str(level_key),
                provenance        = "attach",
            )
            leaf.status = _goal_status_to_node_status(goal)
            leaves.append(leaf)
        # REACH_CELL.confidence is structural (a grouping wrapper, not
        # an independent credence signal) — passing 1.0 prevents the
        # extra ancestor depth from penalising cell-targeted leaves
        # relative to leaves placed directly under a WIN_CONDITION.
        # The category-level confidence still multiplies via the
        # WIN_CONDITION parent above.
        children.append(reach_cell(
            cell              = cell,
            strategies        = leaves,
            confidence        = 1.0,
            applies_to_level  = str(level_key),
            provenance        = "attach",
        ))
    for gid, goal in no_cell:
        leaf = execute(
            gf_goal_id        = gid,
            confidence        = _goal_priority(goal),
            properties        = {"category": CAT_REACH},
            applies_to_level  = str(level_key),
            provenance        = "attach",
        )
        leaf.status = _goal_status_to_node_status(goal)
        children.append(leaf)
    return children


# ---------------------------------------------------------------------------
# Per-turn maintenance.
# ---------------------------------------------------------------------------


def attach_new_open_goals(
    plan_root: PlanNode,
    ws,
    *,
    level_key: str,
) -> int:
    """Attach every OPEN goal not yet represented in the tree.

    The level-entry build (``build_plan_tree_for_level``) captures
    the open-goal snapshot at level entry.  Mid-level, runtime
    machinery (the resource planner, the trigger walker, the
    framed-region helper) declares new goals into the forest.
    Without a refresh, those goals appear only as "competitors"
    in the override path and never as primary scheduler picks.

    This function walks the current goal forest, finds OPEN goals
    whose ``gf_goal_id`` isn't already wrapped by an EXECUTE leaf
    in ``plan_root``, classifies each via :func:`classify_goal`,
    and attaches them under the appropriate category WIN_CONDITION
    (creating the bucket if it doesn't exist).

    Returns the number of leaves added.  Idempotent — running it
    twice in succession adds nothing the second time.
    """
    if plan_root is None:
        return 0
    # Build the set of goal ids already in the tree.
    existing_ids: set = set()
    for leaf in plan_root.find(
        lambda n: n.kind is NodeKind.EXECUTE and n.gf_goal_id is not None
    ):
        existing_ids.add(leaf.gf_goal_id)

    # Find each existing WIN_CONDITION bucket by flavor so new leaves
    # can join the right category subtree without rebuilding the tree.
    buckets_by_flavor: dict = {}
    for wc in plan_root.children:
        if wc.kind is not NodeKind.WIN_CONDITION:
            continue
        flavor = (wc.properties or {}).get("flavor")
        if flavor:
            buckets_by_flavor[flavor] = wc

    new_goals: list = []
    for gid, goal in _iter_open_goals(ws):
        if gid in existing_ids:
            continue
        new_goals.append((gid, goal))
    if not new_goals:
        return 0

    # Bucket the new goals by category and attach.
    by_cat: dict[str, list] = {}
    for gid, goal in new_goals:
        cat = classify_goal(goal)
        by_cat.setdefault(cat, []).append((gid, goal))

    n_added = 0
    for cat, goals_in_cat in by_cat.items():
        # Flavor label used in WIN_CONDITION construction.  Uncategorised
        # goals attach under the "reach"-labelled OR-aggregator bucket
        # to match the v1 build path's convention.
        flavor_label = (cat if cat != CAT_UNCATEGORIZED else "reach")
        bucket = buckets_by_flavor.get(flavor_label)
        if bucket is None:
            # Bucket doesn't yet exist; create and attach to root.
            conf = float(_DEFAULT_CATEGORY_CONFIDENCE.get(cat, 0.4))
            bucket = _build_category_subtree(
                category     = cat,
                goals_in_cat = goals_in_cat,
                confidence   = conf,
                level_key    = level_key,
            )
            plan_root.children.append(bucket)
            buckets_by_flavor[flavor_label] = bucket
            n_added += sum(
                1 for n in bucket.walk()
                if n.kind is NodeKind.EXECUTE
            )
        else:
            # Bucket exists; build leaves and merge into it.
            if cat == CAT_REACH:
                new_children = _build_reach_children(
                    goals_in_cat = goals_in_cat,
                    confidence   = bucket.confidence,
                    level_key    = level_key,
                )
                # Merge per-cell REACH_CELL parents: if a bucket already
                # has a REACH_CELL for the same cell, append the new
                # leaves into it rather than creating a duplicate.
                existing_reach_cells: dict = {}
                for ch in bucket.children:
                    if ch.kind is NodeKind.REACH_CELL:
                        cell = (ch.properties or {}).get("cell")
                        if cell is not None:
                            existing_reach_cells[tuple(cell)] = ch
                for nc in new_children:
                    if nc.kind is NodeKind.REACH_CELL:
                        cell = tuple((nc.properties or {}).get("cell") or ())
                        if cell in existing_reach_cells:
                            for leaf in nc.children:
                                existing_reach_cells[cell].children.append(leaf)
                                n_added += 1
                        else:
                            bucket.children.append(nc)
                            n_added += sum(
                                1 for n in nc.walk()
                                if n.kind is NodeKind.EXECUTE
                            )
                    else:
                        bucket.children.append(nc)
                        n_added += 1
            else:
                for gid, goal in goals_in_cat:
                    leaf = execute(
                        gf_goal_id        = gid,
                        confidence        = _goal_priority(goal),
                        properties        = {"category": cat},
                        applies_to_level  = str(level_key),
                        provenance        = "attach",
                    )
                    leaf.status = _goal_status_to_node_status(goal)
                    bucket.children.append(leaf)
                    n_added += 1

    # Re-assign node ids so the new leaves get deterministic ids and
    # parent pointers are wired through.
    assign_node_ids(plan_root, prefix=f"plan.l{level_key}")
    # Propagate any leaf-status changes up.
    for leaf in plan_root.find(
        lambda n: n.kind is NodeKind.EXECUTE
                  and n.gf_goal_id in {gid for gid, _ in new_goals}
    ):
        refresh_status_up(leaf)
    return n_added


def sync_leaf_statuses(plan_root: PlanNode, ws) -> int:
    """Walk EXECUTE leaves; align each leaf's status to its referenced
    goal-forest goal's current status.  Returns the number of leaves
    whose status changed.

    Propagation up the tree happens automatically via
    ``set_leaf_status`` / ``refresh_status_up``.
    """
    if plan_root is None:
        return 0
    forest = getattr(ws, "goal_forest", None)
    if forest is None:
        return 0
    goals = getattr(forest, "goals", {}) or {}
    n_changed = 0
    for leaf in plan_root.find(
        lambda n: n.kind is NodeKind.EXECUTE and n.gf_goal_id is not None
    ):
        goal = goals.get(leaf.gf_goal_id)
        if goal is None:
            # Goal was retracted between build and sync.
            new_status = NodeStatus.ABANDONED
        else:
            new_status = _goal_status_to_node_status(goal)
        if new_status is not leaf.status:
            set_leaf_status(leaf, new_status)
            n_changed += 1
    return n_changed


def find_leaf_by_gf_goal(
    plan_root: PlanNode,
    gf_goal_id: str,
) -> Optional[PlanNode]:
    """Return the first EXECUTE leaf wrapping ``gf_goal_id`` (or None)."""
    if plan_root is None:
        return None
    for leaf in plan_root.find(
        lambda n: n.kind is NodeKind.EXECUTE and n.gf_goal_id == gf_goal_id
    ):
        return leaf
    return None


def goal_status_resolver(ws) -> Callable[[str], Optional[str]]:
    """Return a callable ``(gf_goal_id) -> status_string`` suitable
    for passing as ``pick_active_leaf``'s ``gf_goal_status`` arg.

    Resolves against the current goal-forest; returns the lower-cased
    status name or None when the id is unknown.
    """
    forest = getattr(ws, "goal_forest", None)
    goals: Mapping[str, Any] = getattr(forest, "goals", {}) or {}

    def _resolve(gid: str) -> Optional[str]:
        g = goals.get(gid)
        if g is None:
            return None
        try:
            return str(g.root.status.name).lower()
        except AttributeError:
            return None
    return _resolve
