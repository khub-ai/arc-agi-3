"""Score a perception run against the operator-authored ground truth.

The scoring mechanism aggregates six dimensions into a 0..1 composite:

1. **Win condition recognition** (35%) — did the VLM identify the
   correct match_condition primitive?  Single most important
   signal: without the right win condition the planner has no
   target.
2. **Role accuracy** (25%) — when the VLM matches a Layer A
   candidate, did it assign the right role?  Cross-game vocabulary
   adherence.
3. **Entity coverage** (15%) — what fraction of ground-truth
   entities did the VLM identify (bbox match at IoU >= 0.3)?
4. **Relationship recognition** (10%) — did the VLM identify the
   ground truth's pairings / groups / collinear triples?
5. **Grouping accuracy** (10%) — did the VLM correctly unify
   fragmented entities (e.g., life_indicator's three dot pairs)?
6. **Catalog hygiene** (5%) — did the VLM stick to catalog
   primitive_ids, or invent new role / relationship names?  This
   is a small but real factor: a catalog-respecting VLM is more
   stable for downstream consumers.

Scores are independent of the VLM tested — same scoring function
produces comparable numbers for any model's output.

Each dimension reports both a raw score and the dimension's
contribution to the composite (raw * weight).  Per-sample reports
also include diagnostic detail so the operator can see WHY a VLM
scored low on a specific dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence


# Composite weights.  Sum to 1.0.
WEIGHTS = {
    "win_condition":      0.35,
    "role_accuracy":      0.25,
    "entity_coverage":    0.15,
    "relationship":       0.10,
    "grouping":           0.10,
    "catalog_hygiene":    0.05,
}


@dataclass
class DimensionScore:
    """One dimension of the per-sample report."""
    name:        str
    raw_score:   float         # 0..1
    weight:      float         # 0..1
    contribution: float        # raw_score * weight
    notes:       str = ""


@dataclass
class SampleScore:
    """Per-sample evaluation against ground truth."""
    sample:           str
    model:            str
    composite:        float                       # weighted sum, 0..1
    dimensions:       List[DimensionScore]
    per_dim:          Dict[str, float] = field(default_factory=dict)
    # Diagnostic detail
    n_ground_truth_entities: int = 0
    n_predicted_entities:    int = 0
    bbox_matches:            int = 0              # IoU >= threshold
    role_matches:            int = 0              # of bbox_matches, role agrees
    win_condition_pred:      str = ""
    win_condition_truth:     str = ""
    relationship_pred:       int = 0
    relationship_truth:      int = 0
    relationship_matched:    int = 0
    group_pred:              int = 0
    group_truth:             int = 0
    group_matched:           int = 0
    catalog_violations:      int = 0
    notes:                   List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bbox_iou(a: Sequence[int], b: Sequence[int]) -> float:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    rr0, cc0 = max(ar0, br0), max(ac0, bc0)
    rr1, cc1 = min(ar1, br1), min(ac1, bc1)
    if rr1 < rr0 or cc1 < cc0:
        return 0.0
    inter = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
    a_area = (ar1 - ar0 + 1) * (ac1 - ac0 + 1)
    b_area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
    return inter / float(a_area + b_area - inter)


def _gt_entities(ground_truth: Mapping[str, Any]) -> List[dict]:
    return [e for e in (ground_truth.get("entities") or [])
            if isinstance(e, dict) and not e.get("_note_only")]


def _gt_groups(ground_truth: Mapping[str, Any]) -> List[set]:
    """Extract ground-truth groups by `member_of_group` field."""
    groups: dict = {}
    for e in _gt_entities(ground_truth):
        g = e.get("member_of_group")
        if g:
            groups.setdefault(str(g), set()).add(e.get("id"))
    return [members for members in groups.values() if len(members) > 1]


def _gt_relationships(ground_truth: Mapping[str, Any]) -> List[set]:
    """Extract ground-truth pairwise relationships via related_to."""
    pairs: List[set] = []
    seen: set = set()
    by_id = {e.get("id"): e for e in _gt_entities(ground_truth)}
    for eid, e in by_id.items():
        rel = e.get("related_to")
        if not rel:
            continue
        # Symmetric: only add the pair once.
        key = frozenset({eid, rel})
        if key in seen or len(key) < 2:
            continue
        if rel not in by_id:
            continue
        seen.add(key)
        pairs.append(set(key))
    return pairs


def _gt_win_kind(ground_truth: Mapping[str, Any]) -> str:
    """Extract the ground-truth win-condition kind as a closed-vocab
    match_condition primitive_id.

    Preference order:
    1. `_game_mechanic.win_condition` when it's a primitive_id directly
       (operator-friendly: just write "alignment_match" instead of prose).
    2. `_relationship_hypothesis.kind` mapped via known suffixes.
    3. Heuristic prose categorisation as last resort.
    """
    CATALOG_KINDS = {
        "alignment_match", "overlap_match", "pair_match",
        "arrangement_match", "reach_cell",
    }
    mech = ground_truth.get("_game_mechanic") or {}
    if isinstance(mech, dict):
        wc = (mech.get("win_condition") or "").strip()
        if wc in CATALOG_KINDS:
            return wc
        # Map specific operator-authored sentences we've seen.
        if wc:
            cat = _categorise_win_prose(wc)
            if cat:
                return cat
    hyp = ground_truth.get("_relationship_hypothesis") or {}
    if isinstance(hyp, dict):
        kind = (hyp.get("kind") or "").strip()
        if kind in CATALOG_KINDS:
            return kind
        if kind:
            # Truncate hyphenated extensions like
            # multi_pair_matching_with_rotation_invariance -> pair_match.
            if "pair_match" in kind or "pair_matching" in kind:
                return "pair_match"
            if "overlap" in kind:
                return "overlap_match"
            if "alignment" in kind:
                return "alignment_match"
            if "arrangement" in kind:
                return "arrangement_match"
    return ""


def _categorise_win_prose(text: str) -> str:
    """Heuristically map operator prose to a catalog match_condition.
    Uses whole-word matching (not substring) so 'center' doesn't
    trigger the 'enter' branch of reach_cell."""
    import re
    if not text:
        return ""
    t = text.lower()
    def _word(w: str) -> bool:
        return re.search(rf"\b{re.escape(w)}\b", t) is not None
    # alignment_match
    if _word("alignment_match") or (_word("align") and _word("match")) \
            or _word("aligned"):
        return "alignment_match"
    # overlap_match (covers ".center" overlap-style win in r11l)
    if _word("overlap_match") or (_word("overlap") and _word("match")) \
            or _word(".center"):
        return "overlap_match"
    # The r11l ground truth: 'linked_midpoint.position == purple_shell.center'
    if "linked_midpoint" in t and ".center" in t:
        return "overlap_match"
    # pair_match
    if _word("pair_match") or _word("matching_with_rotation") \
            or (_word("match") and _word("reference") and _word("pair")):
        return "pair_match"
    # arrangement_match
    if _word("arrangement_match") or _word("reference_arrangement") \
            or (_word("arrangement") and _word("match")):
        return "arrangement_match"
    # reach_cell
    if _word("reach_cell") or _word("win cell") \
            or (_word("enter") and _word("cell")):
        return "reach_cell"
    return ""


def _categorise_win_kind(text: str) -> str:
    """Back-compat shim; prefer _gt_win_kind(ground_truth) which
    handles structured fields and prose."""
    if not text:
        return ""
    cat = _categorise_win_prose(text)
    return cat if cat else text.strip()


# ---------------------------------------------------------------------------
# Per-sample scoring.
# ---------------------------------------------------------------------------


def score_sample(
    parsed:        Mapping[str, Any],   # parser output (dict form)
    ground_truth:  Mapping[str, Any],
    *,
    sample:        str,
    model:         str,
    iou_threshold: float = 0.3,
) -> SampleScore:
    """Compute a per-sample SampleScore.  Pure function — no I/O."""
    gt_entities = _gt_entities(ground_truth)
    pred_entities = parsed.get("entities") or []
    n_gt = len(gt_entities)
    n_pred = len(pred_entities)

    # Match predicted entities to ground-truth entities greedily by IoU.
    matched_pairs: list = []  # (pred_idx, gt_idx, iou)
    used_gt: set = set()
    for pi, pe in enumerate(pred_entities):
        best_iou = 0.0
        best_gi = -1
        for gi, ge in enumerate(gt_entities):
            if gi in used_gt:
                continue
            iou = _bbox_iou(pe.get("bbox_pixels", (0, 0, 0, 0)),
                            ge.get("bbox_pixels", (0, 0, 0, 0)))
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_gi >= 0 and best_iou >= iou_threshold:
            matched_pairs.append((pi, best_gi, best_iou))
            used_gt.add(best_gi)

    n_bbox_matched = len(matched_pairs)
    role_correct = 0
    for pi, gi, _ in matched_pairs:
        if str(pred_entities[pi].get("role", "")) == str(gt_entities[gi].get("role", "")):
            role_correct += 1

    # Dimension 1: win condition
    pred_wc = ""
    if parsed.get("win_condition"):
        pred_wc = str(parsed["win_condition"].get("kind", "")) or ""
    gt_wc = _gt_win_kind(ground_truth)
    win_score = 1.0 if (pred_wc and pred_wc == gt_wc) else 0.0

    # Dimension 2: role accuracy
    role_score = (role_correct / float(n_bbox_matched)) if n_bbox_matched else 0.0

    # Dimension 3: entity coverage
    coverage_score = (n_bbox_matched / float(n_gt)) if n_gt else 0.0

    # Dimension 4: relationship recognition
    gt_pairs = _gt_relationships(ground_truth)
    n_gt_rels = len(gt_pairs)
    pred_rels = parsed.get("relationships") or []
    # Map pred entity-id -> set of GT entity ids it represents.  A pred
    # entity represents a GT entity when:
    #   (a) IoU-matched in the greedy bbox match above (1:1), OR
    #   (b) the pred bbox CONTAINS the GT bbox (coarse-perception case
    #       — e.g., a "bar_assembly" group containing bar_ball_middle).
    # Containment is one-directional and excludes region-type predictions
    # (play_area, wall) to keep palette regions from "containing"
    # everything.
    REGION_ROLES = {"play_area", "wall", "divider", "hud_background"}
    pred_to_gt_ids: dict = {}
    for pi, gi, _ in matched_pairs:
        pred_id = pred_entities[pi].get("id")
        gt_id = gt_entities[gi].get("id")
        if pred_id and gt_id:
            pred_to_gt_ids.setdefault(pred_id, set()).add(gt_id)
    for pe in pred_entities:
        pid = pe.get("id")
        if not pid or str(pe.get("role", "")) in REGION_ROLES:
            continue
        pbox = pe.get("bbox_pixels") or (0, 0, 0, 0)
        for ge in gt_entities:
            gbox = ge.get("bbox_pixels") or (0, 0, 0, 0)
            if (pbox[0] <= gbox[0] and pbox[1] <= gbox[1]
                    and pbox[2] >= gbox[2] and pbox[3] >= gbox[3]
                    and (pbox[2] - pbox[0] + 1) * (pbox[3] - pbox[1] + 1) >
                        (gbox[2] - gbox[0] + 1) * (gbox[3] - gbox[1] + 1)):
                gid = ge.get("id")
                if gid:
                    pred_to_gt_ids.setdefault(pid, set()).add(gid)
    # A GT relationship pair is "recognised" if EITHER:
    #   (a) a predicted relationship's two distinct members cover both
    #       ends of the pair, OR
    #   (b) some predicted entity covers BOTH ends of the pair (the
    #       VLM modelled head+tail as one device — semantically equivalent).
    # Both forms get credit per GT pair (max 1.0 per pair).
    matched_gt_pairs: set = set()
    # Pass (a): explicit relationship records.
    for rel in pred_rels:
        members = rel.get("members") or []
        member_gt_sets = [pred_to_gt_ids.get(str(m), set()) for m in members]
        for gt_pair in gt_pairs:
            key = frozenset(gt_pair)
            if key in matched_gt_pairs:
                continue
            gt_a, gt_b = tuple(gt_pair)
            for i, sa in enumerate(member_gt_sets):
                if gt_a not in sa and gt_b not in sa:
                    continue
                for j, sb in enumerate(member_gt_sets):
                    if i == j:
                        continue
                    if ((gt_a in sa and gt_b in sb)
                            or (gt_b in sa and gt_a in sb)):
                        matched_gt_pairs.add(key)
                        break
                if key in matched_gt_pairs:
                    break
    # Pass (b): an entity containing both ends of a GT pair.
    for pid, gt_ids in pred_to_gt_ids.items():
        for gt_pair in gt_pairs:
            key = frozenset(gt_pair)
            if key in matched_gt_pairs:
                continue
            if gt_pair <= gt_ids:
                matched_gt_pairs.add(key)
    rel_matched = len(matched_gt_pairs)
    rel_score = (rel_matched / float(n_gt_rels)) if n_gt_rels else (
        1.0 if not pred_rels else 0.0   # no rel ground truth -> reward not making any
    )
    rel_score = min(rel_score, 1.0)

    # Dimension 5: grouping accuracy
    gt_groups = _gt_groups(ground_truth)
    n_gt_groups = len(gt_groups)
    # Predicted groups: entities whose `candidate_ids` has > 1 candidate.
    pred_groups_from_candidates: List[set] = []
    for pe in pred_entities:
        cand_ids = pe.get("candidate_ids") or []
        if len(cand_ids) > 1:
            # Resolve member candidates back to GT ids via the match.
            gt_ids: set = set()
            for pi, gi, _ in matched_pairs:
                # If this pred entity owns any of the candidates,
                # link to its matched GT entity.
                if pred_entities[pi] is pe:
                    gt_ids.add(gt_entities[gi].get("id"))
            if gt_ids:
                pred_groups_from_candidates.append(gt_ids)
    # Match each ground-truth group against any predicted group that shares all members.
    group_matched = 0
    for gtg in gt_groups:
        for pg in pred_groups_from_candidates:
            if gtg <= pg or pg <= gtg or len(gtg & pg) >= max(1, len(gtg) - 1):
                group_matched += 1
                break
    group_score = (group_matched / float(n_gt_groups)) if n_gt_groups else 1.0

    # Dimension 6: catalog hygiene
    violations = (parsed.get("validation_messages") or [])
    n_violations = sum(1 for m in violations
                       if "not in catalog" in str(m) or "unknown role" in str(m))
    # Each violation costs 0.1; floor at 0.
    hygiene_score = max(0.0, 1.0 - 0.1 * n_violations)

    dims = [
        DimensionScore("win_condition",   win_score,      WEIGHTS["win_condition"],
                       win_score * WEIGHTS["win_condition"],
                       f"predicted={pred_wc!r} truth={gt_wc!r}"),
        DimensionScore("role_accuracy",   role_score,     WEIGHTS["role_accuracy"],
                       role_score * WEIGHTS["role_accuracy"],
                       f"{role_correct}/{n_bbox_matched} matched-role"),
        DimensionScore("entity_coverage", coverage_score, WEIGHTS["entity_coverage"],
                       coverage_score * WEIGHTS["entity_coverage"],
                       f"{n_bbox_matched}/{n_gt} ground-truth recovered"),
        DimensionScore("relationship",    rel_score,      WEIGHTS["relationship"],
                       rel_score * WEIGHTS["relationship"],
                       f"{rel_matched}/{n_gt_rels} relationships"),
        DimensionScore("grouping",        group_score,    WEIGHTS["grouping"],
                       group_score * WEIGHTS["grouping"],
                       f"{group_matched}/{n_gt_groups} groups"),
        DimensionScore("catalog_hygiene", hygiene_score,  WEIGHTS["catalog_hygiene"],
                       hygiene_score * WEIGHTS["catalog_hygiene"],
                       f"{n_violations} catalog violations"),
    ]
    composite = sum(d.contribution for d in dims)

    return SampleScore(
        sample              = sample,
        model               = model,
        composite           = composite,
        dimensions          = dims,
        per_dim             = {d.name: d.raw_score for d in dims},
        n_ground_truth_entities = n_gt,
        n_predicted_entities    = n_pred,
        bbox_matches            = n_bbox_matched,
        role_matches            = role_correct,
        win_condition_pred      = pred_wc,
        win_condition_truth     = gt_wc,
        relationship_pred       = len(pred_rels),
        relationship_truth      = n_gt_rels,
        relationship_matched    = rel_matched,
        group_pred              = len(pred_groups_from_candidates),
        group_truth             = n_gt_groups,
        group_matched           = group_matched,
        catalog_violations      = n_violations,
    )


# ---------------------------------------------------------------------------
# Aggregate across samples for a single VLM.
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkReport:
    """Per-VLM benchmark across all samples."""
    model:        str
    per_sample:   List[SampleScore]
    composite:    float                            # mean across samples
    per_dim:      Dict[str, float] = field(default_factory=dict)
    total_cost:   float = 0.0
    total_latency_ms: int = 0


def aggregate(scores: Sequence[SampleScore],
              *, model: str,
              total_cost: float = 0.0,
              total_latency_ms: int = 0) -> BenchmarkReport:
    """Aggregate per-sample scores into a single VLM-level report."""
    if not scores:
        return BenchmarkReport(model=model, per_sample=[], composite=0.0)
    mean = sum(s.composite for s in scores) / float(len(scores))
    per_dim_means: dict = {}
    for name in WEIGHTS:
        per_dim_means[name] = (
            sum(s.per_dim.get(name, 0.0) for s in scores) / float(len(scores))
        )
    return BenchmarkReport(
        model            = model,
        per_sample       = list(scores),
        composite        = mean,
        per_dim          = per_dim_means,
        total_cost       = total_cost,
        total_latency_ms = total_latency_ms,
    )


# ---------------------------------------------------------------------------
# Rendering: side-by-side table for human review.
# ---------------------------------------------------------------------------


def format_benchmark(reports: Sequence[BenchmarkReport]) -> str:
    """Render a comparison table across multiple VLM benchmark reports."""
    if not reports:
        return "(no benchmark reports)"
    lines: list = []
    # Sort by composite descending.
    reps = sorted(reports, key=lambda r: -r.composite)
    # Header.
    lines.append(f"{'model':<40}  composite  win_c  role   cover  rel    grp    hyg    cost     latency")
    lines.append("-" * 110)
    for r in reps:
        d = r.per_dim
        lines.append(
            f"{r.model:<40}  {r.composite:>9.3f}  "
            f"{d.get('win_condition', 0):>5.2f}  "
            f"{d.get('role_accuracy', 0):>5.2f}  "
            f"{d.get('entity_coverage', 0):>5.2f}  "
            f"{d.get('relationship', 0):>5.2f}  "
            f"{d.get('grouping', 0):>5.2f}  "
            f"{d.get('catalog_hygiene', 0):>5.2f}  "
            f"${r.total_cost:>6.4f}  "
            f"{r.total_latency_ms/1000:>6.1f}s"
        )
    # Per-sample breakdown for the top model.
    if reps and reps[0].per_sample:
        lines.append("")
        lines.append(f"Top model per-sample: {reps[0].model}")
        lines.append("-" * 80)
        for s in reps[0].per_sample:
            lines.append(
                f"  {s.sample:<20}  composite={s.composite:.3f}  "
                f"bbox={s.bbox_matches}/{s.n_ground_truth_entities}  "
                f"role={s.role_matches}/{s.bbox_matches}  "
                f"win_cond={s.win_condition_pred or '(none)':<20} "
                f"(truth={s.win_condition_truth})"
            )
    return "\n".join(lines)


def format_sample_score(score: SampleScore) -> str:
    """One-sample detail block for the operator."""
    lines = [
        f"=== {score.sample} (model={score.model}) ===",
        f"  composite: {score.composite:.3f}",
        "",
    ]
    for d in score.dimensions:
        lines.append(
            f"  {d.name:<18}  raw={d.raw_score:.3f}  "
            f"weight={d.weight:.2f}  contrib={d.contribution:.3f}  "
            f"({d.notes})"
        )
    return "\n".join(lines)
