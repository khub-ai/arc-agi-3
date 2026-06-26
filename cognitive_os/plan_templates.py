"""Plan-tree templates — registry-driven default plan-tree
construction at level entry.

See :doc:`docs/SPEC_plan_tree.md` (Tree construction §) and the
companion :doc:`docs/SPEC_self_improvement.md` for the long-term
view (mined templates from solved-game skeletons).

Each template is a function with signature::

    template(ws, kb) -> Optional[PlanNode]

returning a candidate ``WIN_CONDITION`` subtree to attach under the
level's TASK root, or ``None`` if the template's required role
pattern isn't present.  Multiple templates can match; each
contributes a sibling candidate.

Templates ship at low confidence and rely on observation /
mining / distillation to raise it.  A template with no empirical
attestation lives at 0.4-0.5; mined templates from the corpus
arrive with confidence tied to their recurrence count.

Engine-clean: no game ids; the only inputs are typed function
tags from the affordance registry plus generic framed-region
candidate cells.
"""

from __future__ import annotations

from typing import Any, Callable, List, Mapping, Optional, Sequence

from .plan_tree import (
    Combinator,
    PlanNode,
    alignment_dim,
    execute,
    investigate,
    observe_effect,
    reach_cell,
    reach_entity,
    win_condition,
)


# Type alias for the template callable.
Template = Callable[..., Optional[PlanNode]]


# ---------------------------------------------------------------------------
# Helper queries against the affordance registry.  These are tiny
# accessors that turn the registry's JSON-y shape into typed
# answers the templates can branch on.
# ---------------------------------------------------------------------------


def _registry_classes_with_function(
    affordance_registry: Mapping[str, Any],
    function_tag: str,
) -> List[str]:
    """Return visual class fingerprints attested to play the given
    function role in any prior level / game.

    The current registry shape (see discovery_play.py's
    ``_persist_launcher_affordance_claims``) is
    ``{class_id: [ {template, action_id, ...}, ...]}``.  A class
    plays ``function_tag`` when at least one of its entries names
    a template matching the tag.
    """
    if not affordance_registry:
        return []
    out: List[str] = []
    for cls_id, entries in affordance_registry.items():
        if not entries:
            continue
        for e in entries:
            if str(e.get("template") or "") == str(function_tag):
                out.append(str(cls_id))
                break
    return out


def _visible_entity_classes(ws) -> List[str]:
    """Return the bitmap_id of every live entity (best tier for
    cross-game matching; falls back to shape_id then topo_id)."""
    out: List[str] = []
    try:
        for ent in (getattr(ws, "entities", {}) or {}).values():
            for tier in ("bitmap_id", "shape_id", "topo_id"):
                v = ent.properties.get(tier)
                if v:
                    out.append(str(v))
                    break
    except (AttributeError, KeyError):
        pass
    return out


# ---------------------------------------------------------------------------
# Templates.  Each one inspects the world and decides whether its
# pattern applies.  Templates are intentionally small and
# composable; one template emits one WIN_CONDITION subtree.  Larger
# strategic compositions (multi-phase plans) come from the
# scheduler running multiple templates' subtrees as parallel
# candidates under the same TASK root.
# ---------------------------------------------------------------------------


def alignment_template(
    *,
    ws,
    kb,
    level_key: str,
    affordance_registry: Mapping[str, Any],
) -> Optional[PlanNode]:
    """Alignment-shape WIN_CONDITION candidate.

    Activates when both a state-changer class AND at least one
    framed-region candidate are visible on this level.  The
    framed-region candidate is interpreted as a reference glyph
    the working entity should be aligned with; the state-changer
    is interpreted as the mutator the agent must visit to change
    the working entity's state.

    The default decomposition expresses a TWO-dimensional
    alignment (palette + corner_signature), matching the
    ls20-style alignment puzzle the engine has seen.  Variants
    (single-dim, three-dim) can be added as new templates.
    """
    state_changers = _registry_classes_with_function(
        affordance_registry, "state_changer",
    )
    if not state_changers:
        return None

    visible_classes = set(_visible_entity_classes(ws))
    visible_state_changers = [
        c for c in state_changers if c in visible_classes
    ]
    if not visible_state_changers:
        return None

    # Build the alignment WIN_CONDITION.  Two ALIGNMENT_DIMs by
    # default (palette + corner_signature); each requires a
    # REACH_ENTITY of the state-changer class plus an
    # OBSERVE_EFFECT closing on the matching property.
    sc_class = visible_state_changers[0]
    dims: List[PlanNode] = []
    for property_name in ("palette", "corner_signature"):
        dims.append(alignment_dim(
            property_name = property_name,
            children = [
                reach_entity(
                    entity_class = sc_class,
                    strategies = [],  # filled by scheduler
                    confidence = 0.5,
                    applies_to_level = level_key,
                    provenance = "registry",
                ),
                observe_effect(
                    expected_kind = "property_match",
                    properties = {"property": property_name},
                    confidence = 0.5,
                    applies_to_level = level_key,
                    provenance = "registry",
                ),
            ],
            confidence = 0.5,
            applies_to_level = level_key,
            provenance = "registry",
        ))

    return win_condition(
        "alignment",
        dims,
        combinator = Combinator.AND,
        confidence = 0.5,
        applies_to_level = level_key,
        provenance = "registry",
    )


def reach_traversal_template(
    *,
    ws,
    kb,
    level_key: str,
    affordance_registry: Mapping[str, Any],
    framed_candidates: Sequence,
) -> Optional[PlanNode]:
    """Traversal-shape WIN_CONDITION candidate.

    Activates when at least one framed-region candidate exists.
    The simplest level shape: agent traverses to a target cell.
    Each framed candidate becomes an OR alternative under a
    single REACH_CELL parent.
    """
    if not framed_candidates:
        return None

    children: List[PlanNode] = []
    for cell in framed_candidates:
        try:
            cr, cc = int(cell[0]), int(cell[1])
        except (TypeError, ValueError, IndexError):
            continue
        # Each framed candidate is its own REACH_CELL subtree.
        children.append(reach_cell(
            cell = (cr, cc),
            strategies = [],   # filled by scheduler
            confidence = 0.5,
            applies_to_level = level_key,
            provenance = "registry",
        ))

    if not children:
        return None

    return win_condition(
        "reach",
        children,
        combinator = Combinator.OR,
        confidence = 0.4,   # lower default than alignment when both apply
        applies_to_level = level_key,
        provenance = "registry",
    )


def investigate_unknowns_template(
    *,
    ws,
    kb,
    level_key: str,
    affordance_registry: Mapping[str, Any],
) -> Optional[PlanNode]:
    """Discovery-shape WIN_CONDITION candidate.

    Activates when the level has visible entities whose visual class
    has NO function tag attested in the registry.  The candidate
    decomposes into one INVESTIGATE child per unfamiliar class.

    This isn't a "win condition" in the strict sense -- it cannot
    close the level on its own.  But making it a sibling
    WIN_CONDITION under the TASK root lets the scheduler treat
    "investigate the unknowns" as a competing strategy: when no
    other WIN_CONDITION has high confidence, investigation
    leaves naturally surface as the next active goal.

    The scheduler is expected to demote this candidate's
    contribution to the TASK's overall progress when a
    high-confidence aligned / reach candidate exists -- but the
    INVESTIGATE leaves remain available as fallbacks and as the
    primary mechanism for raising other templates' confidence.
    """
    visible = _visible_entity_classes(ws)
    if not visible:
        return None
    known_classes = set(affordance_registry.keys()) if affordance_registry else set()
    unknown_classes = [c for c in set(visible) if c not in known_classes]
    if not unknown_classes:
        return None

    children: List[PlanNode] = []
    for cls in unknown_classes[:6]:   # cap to keep tree size sane
        children.append(investigate(
            target_class = cls,
            children = [
                reach_entity(
                    entity_class = cls,
                    strategies = [],
                    confidence = 0.5,
                    applies_to_level = level_key,
                    provenance = "registry",
                ),
                observe_effect(
                    expected_kind = "any_change",
                    confidence = 0.5,
                    applies_to_level = level_key,
                    provenance = "registry",
                ),
            ],
            confidence = 0.4,
            applies_to_level = level_key,
            provenance = "registry",
        ))

    return win_condition(
        "investigate",
        children,
        combinator = Combinator.OR,
        confidence = 0.3,    # never the preferred candidate when
                              # alignment / reach are confident
        applies_to_level = level_key,
        provenance = "registry",
    )


# ---------------------------------------------------------------------------
# Registry of starter templates.  Order is not significant; the
# scheduler / Oracle override chooses among returned candidates
# by confidence.
# ---------------------------------------------------------------------------


REGISTERED_TEMPLATES: List[Template] = [
    alignment_template,
    reach_traversal_template,
    investigate_unknowns_template,
]


def build_default_plan(
    *,
    ws,
    kb,
    level_key: str,
    affordance_registry: Optional[Mapping[str, Any]] = None,
    framed_candidates: Optional[Sequence] = None,
) -> List[PlanNode]:
    """Run every registered template and return the list of
    WIN_CONDITION candidates that activated.

    Caller assembles these into a TASK root and assigns node ids.
    Empty list means no template matched -- the caller falls back
    to Oracle (per the hybrid construction rule) or to the legacy
    flat-goal-forest behaviour.
    """
    affordance_registry = affordance_registry or {}
    framed_candidates   = framed_candidates or []
    candidates: List[PlanNode] = []

    for tmpl in REGISTERED_TEMPLATES:
        try:
            node = tmpl(
                ws = ws,
                kb = kb,
                level_key = level_key,
                affordance_registry = affordance_registry,
                framed_candidates = framed_candidates,
            ) if tmpl is reach_traversal_template else tmpl(
                ws = ws,
                kb = kb,
                level_key = level_key,
                affordance_registry = affordance_registry,
            )
        except TypeError:
            # Templates that take only the common kwargs; rerun without
            # the optional framed_candidates.
            node = tmpl(
                ws = ws,
                kb = kb,
                level_key = level_key,
                affordance_registry = affordance_registry,
            )
        except Exception:
            node = None
        if node is not None:
            candidates.append(node)

    return candidates
