"""Role resolver — applies behaviour-grounded matchers from the
knowledge base to entities detected by entity_detector.

Matchers are GENERIC primitives.  The engine knows a small closed
set of matcher kinds; the per-game knowledge file says WHICH
matchers identify WHICH roles.  Operators add new
role-to-matcher mappings as DATA; new matcher kinds are an engine
extension.

Stage A matcher kinds:

  harness_signal           — entity overlapping a named harness signal
                              (currently: "agent_position")
  largest_static_region    — the primary-background entity (by pixel
                              count)
  second_largest_static_region — secondary-background entity

Stage B adds:

  position_pinned          — entity whose bbox is pinned to a frame
                              edge across the sequence
  disappears_after_agent_contact — entity that vanishes the frame
                              after the agent enters its cell

Stage C adds:

  visual_template_matches  — entity whose visual_signature matches
                              a corroborated template
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Callable

import numpy as np

from .entity_detector import Entity
from .knowledge_base import RoleRule
from .template_store import Template, signature_similarity


@dataclass
class ResolvedEntity:
    """Entity with a role assigned (or None if no rule matched)."""

    entity: Entity
    role: Optional[str] = None
    matched_by: Optional[str] = None    # matcher kind that fired
    projection_mode: str = "centroid"   # from RoleRule
    # Governor inspector signal: if the validator's structural-check
    # rejected this match (e.g. bbox density too low), the role was
    # cleared and the rejection reason is recorded here for tracing.
    inspector_rejected: Optional[str] = None


# Pixel-count tolerance for visual_template matches.  An entity's
# pixel count must fall within (typical_min / TOL) .. (typical_max
# * TOL) of the template's observed range.  This replaces the
# previous magic density threshold — the tolerance is anchored on
# OBSERVED template geometry, not a global constant.  TOL=2.0
# means "within 2x of observed range on either side", which absorbs
# normal sprite variation while rejecting strips that are 10-30x
# larger than the template's source.
_PIXEL_COUNT_TOLERANCE = 2.0


@dataclass
class ResolveContext:
    """Inputs the matchers need.  The resolver gathers these from the
    runtime and hands them to every matcher; matchers read only what
    they need.

    Substrate-agnostic: every field is a generic primitive (cells,
    pixels, action ids), never a game-specific concept.
    """

    rows: int
    cols: int
    agent_position: Optional[tuple[int, int]] = None
    # Sequence-aware matchers read history.  Each entry is a dict like
    # {turn, entities, agent_position, prev_action}.
    history: list[dict] = field(default_factory=list)
    # Corroborated visual templates for the current (game, level).
    # The visual_template matcher reads from this list; templates
    # come from the cross-trial template store (template_store.py),
    # optionally augmented by within-sequence learning (templates
    # promoted by other matchers in this very ingest call).
    templates: list[Template] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Matcher implementations.  Each takes (entity, context, matcher_args)
# and returns True if the entity satisfies the rule.
# -----------------------------------------------------------------------------


def _match_harness_signal(
    entity: Entity, ctx: ResolveContext, args: dict,
) -> bool:
    """Match an entity whose CENTROID is at a named harness signal
    cell.  Strict centroid match (not cells-contains) — substrate-
    agnostic disambiguation: the agent is the entity whose centre
    of mass is at the signal, not just any entity overlapping the
    signal cell.  Combined with smallest-first iteration in the
    resolver, this picks the smallest entity centred on agent_pos.

    Backgrounds rejected — a harness signal points at foreground.
    """
    if entity.is_background_primary or entity.is_background_secondary:
        return False
    signal = args.get("signal")
    if signal == "agent_position":
        if ctx.agent_position is None:
            return False
        ar, ac = ctx.agent_position
        return entity.centroid_cell == (ar, ac)
    return False


def _match_largest_static_region(
    entity: Entity, ctx: ResolveContext, args: dict,
) -> bool:
    return entity.is_background_primary


def _match_second_largest_static_region(
    entity: Entity, ctx: ResolveContext, args: dict,
) -> bool:
    return entity.is_background_secondary


# -----------------------------------------------------------------------------
# Stage B matchers — behaviour-grounded, sequence-aware.
# -----------------------------------------------------------------------------


def _match_position_pinned(
    entity: Entity, ctx: ResolveContext, args: dict,
) -> bool:
    """An entity that's anchored to a specific edge of the frame
    across the sequence — typically a HUD strip.

    Definition (substrate-agnostic):
      The entity sits within `pin_tolerance` logical pixels of the
      specified edge of the frame, AND it has appeared in the last
      `min_consecutive_turns` consecutive history entries at the
      same (or visually equivalent) bbox.

    Backgrounds are rejected — backgrounds touch every edge by
    definition.
    """
    if entity.is_background_primary or entity.is_background_secondary:
        return False
    edge = args.get("edge", "bottom")
    pin_tolerance = int(args.get("pin_tolerance_logical_px", 1))
    min_consec = int(args.get("min_consecutive_turns", 1))
    y0, x0, y1, x1 = entity.bbox_logical
    rows = ctx.rows
    cols = ctx.cols
    logical_h = rows * 8  # convention: 8 logical px per cell
    logical_w = cols * 8
    if edge == "bottom":
        if logical_h - 1 - y1 > pin_tolerance:
            return False
    elif edge == "top":
        if y0 > pin_tolerance:
            return False
    elif edge == "left":
        if x0 > pin_tolerance:
            return False
    elif edge == "right":
        if logical_w - 1 - x1 > pin_tolerance:
            return False
    else:
        return False

    if min_consec <= 1:
        return True

    # Sequence corroboration: check the last min_consec - 1 turns
    # contain SOME entity with a comparable bbox in the same pinned
    # position.  Substrate-agnostic: same bbox tolerance, no visual-
    # template assumptions yet (that's Stage C).
    needed = min_consec - 1
    found = 0
    for h in reversed(ctx.history):
        for hist_e in h.get("entities", []):
            if (hist_e.is_background_primary
                    or hist_e.is_background_secondary):
                continue
            hy0, hx0, hy1, hx1 = hist_e.bbox_logical
            if edge == "bottom" and (logical_h - 1 - hy1) > pin_tolerance:
                continue
            if edge == "top" and hy0 > pin_tolerance:
                continue
            if edge == "left" and hx0 > pin_tolerance:
                continue
            if (edge == "right"
                    and (logical_w - 1 - hx1) > pin_tolerance):
                continue
            # Same edge; close enough in the orthogonal axis?
            if abs(hy0 - y0) <= 2 and abs(hx0 - x0) <= 2:
                found += 1
                break
        if found >= needed:
            return True
    return False


def _match_disappears_after_agent_contact(
    entity: Entity, ctx: ResolveContext, args: dict,
) -> bool:
    """A consumable entity: present in a cell at turn N, agent enters
    that cell at turn N (or N+1), entity gone at turn N+1.

    The matcher works in reverse: on a given turn, we check the
    history for ENTITIES that disappeared from a cell the agent
    just entered.  The disappeared entity's visual signature is
    then used to identify CURRENTLY-PRESENT entities as the same
    role.

    Substrate-agnostic.  No "what color is a sediment" rule — just
    "matches the signature of a previously-observed consumable".
    """
    if entity.is_background_primary or entity.is_background_secondary:
        return False
    if not ctx.history:
        return False

    # Collect signatures of entities that have disappeared from the
    # agent's prior cells.  This is a per-call computation; in
    # Stage C the validator's template store moves this into a
    # cross-trial cache.
    consumable_sigs: set = set()
    for i in range(len(ctx.history) - 1):
        h_now = ctx.history[i]
        h_next = ctx.history[i + 1]
        ap_now = h_now.get("agent_position")
        ap_next = h_next.get("agent_position")
        if ap_now is None or ap_next is None:
            continue
        # The cell the agent ENTERED between i and i+1 — i.e., the
        # NEW agent cell (not the one it was previously in).
        if ap_now == ap_next:
            # Stayed; no entry event.
            continue
        new_cell = ap_next
        # Find an entity present at h_now sharing the agent's new
        # cell but absent at h_next.
        ents_now = h_now.get("entities", [])
        ents_next = h_next.get("entities", [])
        next_sigs = {e.visual_signature for e in ents_next}
        for e_now in ents_now:
            if (e_now.is_background_primary
                    or e_now.is_background_secondary):
                continue
            if new_cell in e_now.cells \
                    and e_now.visual_signature not in next_sigs:
                consumable_sigs.add(e_now.visual_signature)

    if not consumable_sigs:
        return False
    return entity.visual_signature in consumable_sigs


# -----------------------------------------------------------------------------
# Stage C matcher — visual-template matching against corroborated
# templates from ctx.templates.  This is the "learn from behaviour
# once, apply visually thereafter" path that lets perception match
# entities the current sequence doesn't directly interact with.
# -----------------------------------------------------------------------------


def _match_visual_template(
    entity: Entity, ctx: ResolveContext, args: dict,
) -> bool:
    """Match an entity whose visual signature AND geometric size
    fits a template learned for the role.

    Two checks (both substrate-agnostic, both derived from observed
    template properties — no magic global threshold):

      1. SIGNATURE: entity.visual_signature has overlap >= similarity
         threshold (default 0.7) with the template's signature.

      2. SIZE: entity.n_pixels falls within
           [template.pixel_count_min / TOL, template.pixel_count_max * TOL]
         where TOL=2.0.  Catches the bp35 lc=0 case: the budget-
         counter HUD template was seeded from a ~5-pixel counter
         entity; a wide 200-pixel strip with the SAME signature is
         40x larger than the template's source — geometry rules it
         out even though the colour matches.

      3. CORROBORATION: template.corroborations >= min_corroborations
         (default 2).  Single-observation templates are noise.

    All thresholds either come from the template's observed data
    (size) or are universal weak-evidence filters (corroboration,
    similarity).  No per-game tuning.
    """
    if entity.is_background_primary or entity.is_background_secondary:
        return False
    role = args.get("role")
    if not role:
        return False
    sim_threshold = float(args.get("similarity_threshold", 0.7))
    min_corr = int(args.get("min_corroborations", 2))
    for t in ctx.templates:
        if t.role != role:
            continue
        if t.corroborations < min_corr:
            continue
        # SIZE check: only if the template recorded a non-zero range.
        # (Legacy templates without pixel-count data skip this check.)
        if t.pixel_count_max > 0:
            lo = max(1, int(t.pixel_count_min / _PIXEL_COUNT_TOLERANCE))
            hi = int(t.pixel_count_max * _PIXEL_COUNT_TOLERANCE)
            if entity.n_pixels < lo or entity.n_pixels > hi:
                continue
        # SIGNATURE check.
        sim = signature_similarity(entity.visual_signature, t.signature)
        if sim >= sim_threshold:
            return True
    return False


# Closed set of matcher kinds.  Operator-supplied knowledge files
# can ONLY reference these kinds.  An unknown kind is a knowledge-
# file authoring error and is silently no-op'd.
_MATCHERS: dict[str, Callable[[Entity, ResolveContext, dict], bool]] = {
    "harness_signal":                _match_harness_signal,
    "largest_static_region":         _match_largest_static_region,
    "second_largest_static_region":  _match_second_largest_static_region,
    "position_pinned":               _match_position_pinned,
    "disappears_after_agent_contact": _match_disappears_after_agent_contact,
    "visual_template":               _match_visual_template,
}


# -----------------------------------------------------------------------------
# Resolver — apply rules in declared order; first match wins per entity.
# -----------------------------------------------------------------------------


def _best_template_match(
    entity: Entity,
    template_rules: list[RoleRule],
    ctx: ResolveContext,
) -> Optional[tuple[float, int, RoleRule]]:
    """For one entity, scan every visual_template rule and return the
    strongest matching (similarity, corroborations, rule) — or None
    if nothing clears the rule's similarity / corroboration / size
    gates.

    "Strongest" = highest similarity, with corroborations as the
    tiebreaker.  No magic ranking weights: similarity is the direct
    measure of fit; corroborations is a count of independent
    evidence; both are intrinsic properties of the template, not
    tunable knobs.

    This is the principled answer to "which role does this entity
    belong to when its signature matches multiple templates across
    roles?"  The previous first-rule-wins discipline (which made
    sense for categorical matchers like harness_signal) was wrong
    for scored matchers — it discarded the comparison the score
    exists to inform.
    """
    if entity.is_background_primary or entity.is_background_secondary:
        return None
    best: Optional[tuple[float, int, RoleRule]] = None
    for rule in template_rules:
        args = rule.matcher
        role = args.get("role")
        if not role:
            continue
        sim_threshold = float(args.get("similarity_threshold", 0.7))
        min_corr = int(args.get("min_corroborations", 2))
        for t in ctx.templates:
            if t.role != role:
                continue
            if t.corroborations < min_corr:
                continue
            if t.pixel_count_max > 0:
                lo = max(1, int(t.pixel_count_min / _PIXEL_COUNT_TOLERANCE))
                hi = int(t.pixel_count_max * _PIXEL_COUNT_TOLERANCE)
                if entity.n_pixels < lo or entity.n_pixels > hi:
                    continue
            sim = signature_similarity(entity.visual_signature, t.signature)
            if sim < sim_threshold:
                continue
            cand = (sim, t.corroborations, rule)
            if best is None or (cand[0], cand[1]) > (best[0], best[1]):
                best = cand
    return best


def resolve_roles(
    entities: list[Entity],
    rules: list[RoleRule],
    ctx: ResolveContext,
) -> list[ResolvedEntity]:
    """Assign roles to entities using the supplied rules.

    Rules split into two kinds:

      CATEGORICAL matchers (harness_signal, largest_static_region,
      second_largest_static_region, position_pinned,
      disappears_after_agent_contact) return a yes/no answer with no
      strength score.  These run in declared order; the first rule
      whose matcher returns True claims the entity.

      SCORED matchers (visual_template) yield a similarity score
      in [0, 1] against each template.  Running these in declared
      order would arbitrarily favour whichever role's rule appears
      first in the knowledge file even when another role's template
      fits the entity better.  Instead, for each unclaimed entity
      we evaluate ALL visual_template rules and pick the role whose
      template scores highest — with corroborations as the
      tiebreaker.  No new thresholds; just the scores the templates
      already carry.

    Entities not claimed by any rule end up with role=None and are
    treated as "?" by the cell-grid projection.
    """
    resolved: list[ResolvedEntity] = [
        ResolvedEntity(entity=e) for e in entities
    ]

    template_rules = [
        r for r in rules
        if r.matcher.get("type") == "visual_template"
    ]
    categorical_rules = [
        r for r in rules
        if r.matcher.get("type") != "visual_template"
    ]

    # Pass 1: categorical matchers, declared order, first-match-wins.
    for rule in categorical_rules:
        matcher = _MATCHERS.get(rule.matcher.get("type", ""))
        if matcher is None:
            continue
        # For non-background rules, iterate smallest-entity-first so
        # the most specific entity wins.  Critical for the agent
        # rule when an agent_position cell is also covered by a
        # larger sprite-group: the smaller, more-specific agent
        # sprite should be claimed rather than the larger cluster.
        is_bg_rule = rule.matcher.get("type") in (
            "largest_static_region", "second_largest_static_region",
        )
        if is_bg_rule:
            candidates = [re for re in resolved if re.role is None]
        else:
            candidates = sorted(
                [re for re in resolved if re.role is None],
                key=lambda re: (re.entity.n_pixels, re.entity.entity_id),
            )
        for re in candidates:
            if re.role is not None:
                continue
            try:
                if matcher(re.entity, ctx, rule.matcher):
                    re.role = rule.role
                    re.matched_by = rule.matcher.get("type")
                    re.projection_mode = rule.projection_mode
                    if rule.cardinality == "single":
                        break
            except Exception:
                continue

    # Pass 2: visual_template matchers, best-match across roles.
    if template_rules:
        for re in resolved:
            if re.role is not None:
                continue
            best = _best_template_match(re.entity, template_rules, ctx)
            if best is None:
                continue
            _sim, _corr, rule = best
            re.role = rule.role
            re.matched_by = "visual_template"
            re.projection_mode = rule.projection_mode

    return resolved


def project_to_cell_grid(
    resolved: list[ResolvedEntity],
    rows: int,
    cols: int,
    truth_codes: dict[str, str],
    *,
    logical_h: Optional[int] = None,
    logical_w: Optional[int] = None,
    bg_projection_mode: str = "bbox_membership",
) -> list[list[str]]:
    """Build a rows × cols cell-code grid from resolved entities.

    Foreground sprites project onto their centroid cell.  Backgrounds
    project by **bbox membership** with a tie-breaker preferring the
    bbox with smaller area — substrate-agnostic structural rule:
    "a cell belongs to the most-specific entity whose extent
    contains it".  This matches how truth labellers treat playfield
    regions: cells inside the interior background's bbox belong to
    that region even if the perimeter background's speckle reaches
    a few pixels into them.

    `logical_h` / `logical_w` give the logical-pixel frame size; if
    omitted, default to rows*8 / cols*8 (matches the harness's 64x64
    convention).
    """
    # Initialize all cells to "?".  Foreground sprites fill first;
    # backgrounds fill remaining "?" cells by bbox membership.
    codes: list[list[str]] = [["?"] * cols for _ in range(rows)]
    H = logical_h or rows * 8
    W = logical_w or cols * 8
    cell_h = H // rows
    cell_w = W // cols

    # First pass: foreground sprites.  Iterate in CODE PRIORITY
    # order (A before H before X before P/G before U) so that when
    # two entities claim the same cell (e.g. the agent sprite and a
    # pink-sediment cluster both have centroid at the agent's cell),
    # the more-specific role wins.  First-write-wins per cell, so
    # ordering matters.
    code_priority = {c: i for i, c in enumerate(
        ("A", "H", "X", "P", "G", "U")
    )}
    fg_resolved = [
        re for re in resolved
        if re.role is not None
        and not re.entity.is_background_primary
        and not re.entity.is_background_secondary
    ]
    fg_resolved.sort(key=lambda re: code_priority.get(
        truth_codes.get(re.role, "?"), 99
    ))
    # Projection mode comes from the matched rule:
    #   centroid       — only the centroid cell gets the code
    #   all_cells      — every cell the entity touches gets the code
    #   bbox_strip_row — every cell in the entity's bbox row(s),
    #                    full frame width
    for re in fg_resolved:
        code = truth_codes.get(re.role, "?")
        if code == "?":
            continue
        mode = re.projection_mode
        target_cells: list[tuple[int, int]] = []
        if mode == "all_cells":
            target_cells = list(re.entity.cells)
        elif mode == "bbox_strip_row":
            by0, bx0, by1, bx1 = re.entity.bbox_logical
            row_y0_cell = max(0, by0 // (H // rows))
            row_y1_cell = min(rows - 1, by1 // (H // rows))
            for r_ in range(row_y0_cell, row_y1_cell + 1):
                for c_ in range(cols):
                    target_cells.append((r_, c_))
        else:  # "centroid" (default)
            target_cells = [re.entity.centroid_cell]
        for (r, c) in target_cells:
            if codes[r][c] == "?":
                codes[r][c] = code

    # Second pass: backgrounds.  Selection rule driven by the
    # `bg_projection_mode` argument (which comes from the per-level
    # knowledge entry).  Both rules are substrate-agnostic — the
    # operator's truth-labelling convention picks which fits.
    bg_entities = [
        re for re in resolved
        if re.role is not None
        and (re.entity.is_background_primary
             or re.entity.is_background_secondary)
    ]
    for r in range(rows):
        for c in range(cols):
            if codes[r][c] != "?":
                continue
            if bg_projection_mode == "pixel_majority":
                best_px = 0
                best_code = None
                for re in bg_entities:
                    pcount = re.entity.pixel_count_per_cell.get((r, c), 0)
                    if pcount > best_px:
                        best_px = pcount
                        best_code = truth_codes.get(re.role, "?")
                if best_code is not None and best_code != "?":
                    codes[r][c] = best_code
                    continue
                # Fall through to bbox if no bg has pixels here.
            # bbox_membership (default + pixel_majority fallback).
            cy0 = r * cell_h
            cy1 = (r + 1) * cell_h - 1
            cx0 = c * cell_w
            cx1 = (c + 1) * cell_w - 1
            best_area = None
            best_code = None
            for re in bg_entities:
                by0, bx0, by1, bx1 = re.entity.bbox_logical
                if (cy1 < by0 or cy0 > by1
                        or cx1 < bx0 or cx0 > bx1):
                    continue
                area = (by1 - by0 + 1) * (bx1 - bx0 + 1)
                if best_area is None or area < best_area:
                    best_area = area
                    best_code = truth_codes.get(re.role, "?")
            if best_code is not None and best_code != "?":
                codes[r][c] = best_code

    return codes
