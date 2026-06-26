"""Structural-claim generation -- COS's ability to form HIGHER-LEVEL, structural
hypotheses about a puzzle, the way a person glances at a board and thinks "that
second panel is probably the instructions."  Game-agnostic.

A *structural claim* is a hypothesis of the shape "this STRUCTURE implies this
RULE / AFFORDANCE", one abstraction level above a single entity's function
(function: "what does this switch do"; structural: "this whole panel is the
answer key for that one").  Two principles, both required:

  1. EVERY structural claim is just a CLAIM -- it is VERIFIED before it is
     trusted.  A pattern that worked on a previous puzzle is a strong PRIOR, not
     a fact: it enters at GUESSED credence carrying a discriminating PLAN, and is
     promoted only when that plan actually works on THIS puzzle (assume-but-
     recheck).  Nothing here is hardcoded as always-true.

  2. GENERALISATION at various degrees.  The generators detect game-agnostic
     regularities (two groups with the same structure; a marker whose distance to
     its goal equals a control count) and emit the more-abstract claim each
     implies.  They are small and composable -- a new kind of generalisation is a
     new generator, and a validated structural claim becomes a reusable prior for
     future puzzles.

Pure functions over lightweight views (name -> {role, bbox}; group -> {members});
no dependency on the live World, so they are trivially testable and reusable.
"""

from __future__ import annotations

import math


def _area(bb):
    r0, c0, r1, c1 = bb
    return max(0, (r1 - r0)) * max(0, (c1 - c0))


def _center(bb):
    r0, c0, r1, c1 = bb
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


_CONTROL_ROLES = {"control_group", "trigger_target", "control", "switch"}
_REFERENCE_ROLES = {"legend", "reference_display", "reference", "key",
                    "instruction", "scenery", "unknown", ""}


def _members(groups, gname):
    return list((groups.get(gname) or {}).get("members") or [])


def _group_role_mix(entities, members):
    """How 'interactive' a group is: fraction of members with a control-like
    role.  No member info -> treat the group entity's own role."""
    roles = [(entities.get(m) or {}).get("role", "") for m in members]
    roles = [(r or "").lower() for r in roles]
    if not roles:
        return 0.0
    return sum(1 for r in roles if r in _CONTROL_ROLES) / len(roles)


def same_structure_template(entities, groups):
    """REGULARITY: two groups with the SAME structure (same member count) where
    one is an ACTIVE control panel and the other is a STATIC reference.
    GENERALISATION: the static one is a TEMPLATE / instruction for the active one
    -- so the puzzle may be solved by making the active panel MATCH the template,
    then firing the trigger.  (The lc=1 left-panel insight, generalised.)"""
    out = []
    names = [g for g in groups if _members(groups, g)]
    for i in range(len(names)):
        for j in range(len(names)):
            if i == j:
                continue
            A, B = names[i], names[j]               # A = active, B = template
            ma, mb = _members(groups, A), _members(groups, B)
            if len(ma) < 2 or len(ma) != len(mb):    # same structure (size)
                continue
            a_active = _group_role_mix(entities, ma)
            b_active = _group_role_mix(entities, mb)
            # A is the interactive panel; B is NOT more interactive than A
            if a_active <= 0.5 or b_active >= a_active:
                continue
            out.append({
                "id": f"structural_template__{B}__instructs__{A}",
                "kind": "structural", "scope": "cross_game",
                "statement": (f"group '{B}' (same structure as the active panel "
                              f"'{A}') is a TEMPLATE / instruction: solve by making "
                              f"'{A}' match '{B}', then fire the trigger"),
                "target": [A, B], "plan": f"MATCH:{A}~{B}+TRIGGER",
                "importance": 0.85, "credence": 0.4, "provenance": "guessed"})
    return out


def _size_ratio(a_area, b_area) -> float:
    """min/max area in (0,1]; 1.0 = identical size."""
    hi = max(a_area, b_area)
    return (min(a_area, b_area) / hi) if hi else 1.0


def _structure_fingerprint(entities, members):
    """A group's internal arrangement as a position-normalised, order-stable
    fingerprint: each member's centroid mapped into the group's own bounding box
    ([0,1]x[0,1]) plus its area, sorted by normalised position.  Two structures
    with the SAME fingerprint have the same members in the same relative layout --
    the visual signature of 'the same kind of panel', independent of WHERE the
    panel sits or how big it is."""
    bbs = [(m, entities[m]["bbox"]) for m in members
           if m in entities and entities[m].get("bbox")]
    if len(bbs) < 2:
        return None
    rs = [_center(b)[0] for _, b in bbs]
    cs = [_center(b)[1] for _, b in bbs]
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    dr, dc = (r1 - r0) or 1.0, (c1 - c0) or 1.0
    fp = [((_center(b)[0] - r0) / dr, (_center(b)[1] - c0) / dc, _area(b), m)
          for m, b in bbs]
    fp.sort(key=lambda t: (round(t[0], 2), round(t[1], 2)))
    return fp


def structure_correspondences(entities, groups, *, sim_floor=0.5):
    """Every pair of groups whose internal fingerprints align above the similarity
    floor, each with a member BIJECTION (the analogy) and a score in [0,1].  This
    is the MAPPING that similar_structures reports as a claim AND that Structure
    Mapping (transfer_across_mappings) transfers other claims across.  Score =
    arrangement alignment x size agreement over the order-matched members.  Pure;
    bbox-only; the >0.5 floor is the same structural cut used elsewhere."""
    out = []
    names = [g for g in groups if len(_members(groups, g)) >= 2]
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]
            fa = _structure_fingerprint(entities, _members(groups, A))
            fb = _structure_fingerprint(entities, _members(groups, B))
            if not fa or not fb or len(fa) != len(fb):
                continue
            pos = sum(1.0 - min(1.0, abs(a[0] - b[0]) + abs(a[1] - b[1]))
                      for a, b in zip(fa, fb)) / len(fa)
            size = sum(_size_ratio(a[2], b[2]) for a, b in zip(fa, fb)) / len(fa)
            score = pos * size
            if score < sim_floor:
                continue
            out.append({"a": A, "b": B, "score": score,
                        "pairs": [(a[3], b[3]) for a, b in zip(fa, fb)]})
    return out


def similar_structures(entities, groups, *, sim_floor=0.5):
    """REGULARITY (the user's principle): a LARGE DEGREE OF VISUAL SIMILARITY
    between two STRUCTURES (groups of entities) is by itself enough to hypothesise
    they are RELATED -- a mirror / copy / instruction pair -- regardless of any
    role asymmetry.  The left and right panels are the canonical example: same
    members, same arrangement, same sizes.

    Emits a binding claim for each correspondence above the MAJORITY (>0.5)
    similarity floor; credence is GRADED by the score (clamped to the guessed
    ceiling), so a stronger resemblance -> a stronger prior.  The claim carries the
    member CORRESPONDENCE (left_i ~ right_i), which is what makes 'match one to
    the other' actionable AND what Structure Mapping transfers claims across.
    Generalises same_structure_template (which additionally needed a role
    asymmetry)."""
    out = []
    for m in structure_correspondences(entities, groups, sim_floor=sim_floor):
        A, B, score, corr = m["a"], m["b"], m["score"], m["pairs"]
        corr_txt = "; ".join(f"{x}~{y}" for x, y in corr)
        out.append({
            "id": f"structural_similar__{A}__{B}",
            "kind": "structural", "scope": "cross_game",
            "statement": (f"structures '{A}' and '{B}' are visually near-identical "
                          f"(score {round(score, 2)}: same {len(corr)} members, "
                          f"matching arrangement + sizes) -> likely RELATED (mirror "
                          f"/ copy / instruction); correspondence {corr_txt}.  "
                          f"Verify by comparing corresponding members; if one is a "
                          f"reference, MATCH the other to it."),
            "target": [A, B] + [m for pair in corr for m in pair],
            "plan": f"COMPARE:{A}~{B} (corresponding members); MATCH active to reference",
            "importance": 0.8, "credence": round(0.40 * score, 2),
            "provenance": "guessed"})
    return out


def similar_entities(entities, groups, *, sim_floor=0.5, max_claims=8):
    """REGULARITY: two UNGROUPED entities with a large degree of visual similarity
    (matching size + aspect) are likely the SAME TYPE / a related pair (e.g. the
    arch in one panel and the arch in the other).  Similarity ALONE is the trigger
    (the user's principle).  Scoped to entities NOT already captured by a group
    (within-group repetition is the group itself), graded credence, capped to
    avoid an N^2 flood.  Size+aspect is the bbox-only signal; a shape descriptor
    can refine it later without changing the contract."""
    grouped = {m for g in groups for m in _members(groups, g)}
    cand = [(n, e["bbox"]) for n, e in entities.items()
            if e.get("bbox") and n not in grouped
            and (e.get("role", "") or "").lower() not in
            {"hud", "background", "bg", "wall", "floor", "border", "frame", "scenery"}]
    scored = []
    for i in range(len(cand)):
        for j in range(i + 1, len(cand)):
            (na, ba), (nb, bb) = cand[i], cand[j]
            ha, wa = ba[2] - ba[0], ba[3] - ba[1]
            hb, wb = bb[2] - bb[0], bb[3] - bb[1]
            size = _size_ratio(_area(ba), _area(bb))
            asp = _size_ratio((ha + 1) * (wb + 1), (hb + 1) * (wa + 1))  # aspect agreement
            score = size * asp
            if score >= sim_floor:
                scored.append((score, na, nb))
    scored.sort(reverse=True)
    out = []
    for score, na, nb in scored[:max_claims]:
        out.append({
            "id": f"similar_entities__{na}__{nb}",
            "kind": "structural", "scope": "cross_game",
            "statement": (f"entities '{na}' and '{nb}' are visually similar "
                          f"(score {round(score, 2)}: matching size + aspect) -> "
                          f"likely the SAME TYPE / a related pair; verify by "
                          f"comparing their state / behaviour."),
            "target": [na, nb], "plan": f"COMPARE:{na}~{nb}",
            "importance": 0.55, "credence": round(0.40 * score, 2),
            "provenance": "guessed"})
    return out


def transfer_across_mappings(entities, groups, existing_claims, *, sim_floor=0.5):
    """STRUCTURE MAPPING (analogical transfer -- the core of human analogy,
    Gentner): given two SIMILAR structures and their member correspondence, a
    claim KNOWN about one structure plausibly holds for the other under the
    mapping.  'What happens in the left panel applies to the right panel.'  For
    each existing claim that references members of one mapped structure, emit the
    ANALOGOUS claim about the other -- each member replaced by its counterpart,
    unmapped entities (e.g. a shared goal) left as-is.

    Game-agnostic and general: it transfers ANY claim (a function, a correlation,
    a per-step program) across ANY similar-structure mapping, not just panels.
    Every transferred claim is a GUESSED hypothesis to VERIFY on the target
    structure; credence = source credence x mapping score (an analogy never
    strengthens a belief).  Bidirectional.  Does not transfer the structural /
    already-mapped claims themselves (no runaway).  ``existing_claims`` is a list
    of {id, statement, target, credence?, importance?, plan?, kind?}.
    """
    out = []
    maps = structure_correspondences(entities, groups, sim_floor=sim_floor)
    for m in maps:
        fwd = {a: b for a, b in m["pairs"]}
        rev = {b: a for a, b in m["pairs"]}
        for sub, src_g, dst_g in ((fwd, m["a"], m["b"]), (rev, m["b"], m["a"])):
            for c in (existing_claims or []):
                if (c.get("kind") or "") in ("structural", "structure_mapped"):
                    continue                       # don't transfer mappings themselves
                tgt = list(c.get("target") or [])
                involved = [t for t in tgt if t in sub]
                if not involved:
                    continue                       # claim doesn't touch this structure
                new_tgt = [sub.get(t, t) for t in tgt]
                pair_txt = "; ".join(f"{t}->{sub[t]}" for t in involved)
                cid = f"mapped::{c['id']}::to::{dst_g}"
                cred = round(min(0.40, float(c.get("credence", 0.4) or 0.4) * m["score"]), 2)
                out.append({
                    "id": cid, "kind": "structure_mapped", "scope": "level",
                    "statement": (
                        f"STRUCTURE-MAPPING ({src_g} -> {dst_g}, sim "
                        f"{round(m['score'], 2)}): the claim \"{c.get('statement', '')}\" "
                        f"plausibly holds for the corresponding members of '{dst_g}' "
                        f"too ({pair_txt}).  Transferred by analogy from {c['id']} -- "
                        f"VERIFY on '{dst_g}', do not assume."),
                    "target": new_tgt,
                    "plan": (f"on '{dst_g}', reproduce the mapped claim ({pair_txt}) "
                             f"and check the effect + score"),
                    "importance": float(c.get("importance", 0.6) or 0.6),
                    "credence": cred, "provenance": "guessed"})
    # de-dup by id (a claim touching both structures could map twice)
    best = {}
    for c in out:
        if c["id"] not in best or c["credence"] > best[c["id"]]["credence"]:
            best[c["id"]] = c
    return list(best.values())


def per_control_step(entities, groups, *, cell_ticks=4, tol=1):
    """REGULARITY: a moving MARKER, its goal TARGET, and a control group whose
    column count equals the marker->target distance (in cells).
    GENERALISATION: each column of the group sets ONE step of the marker's path to
    the target -- so the group is a per-step 'program'.  (The '4 columns == 4
    steps' insight, generalised.)"""
    out = []
    markers = [(n, e) for n, e in entities.items()
               if (e.get("role", "") or "").lower() == "goal_marker" and e.get("bbox")]
    targets = [(n, e) for n, e in entities.items()
               if (e.get("role", "") or "").lower() == "goal_target" and e.get("bbox")]
    if not markers or not targets:
        return out
    for gname, g in groups.items():
        mem = _members(groups, gname)
        if _group_role_mix(entities, mem) <= 0.5:
            continue
        ncols = _column_count(entities, mem, cell_ticks)
        if ncols < 2:
            continue
        for mn, me in markers:
            tn, te = min(targets, key=lambda kv: abs(_center(kv[1]["bbox"])[1]
                                                      - _center(me["bbox"])[1]))
            dist_cells = round(abs(_center(te["bbox"])[0] - _center(me["bbox"])[0])
                               / float(cell_ticks))
            if abs(dist_cells - ncols) <= tol:
                out.append({
                    "id": f"structural_perstep__{gname}__{mn}",
                    "kind": "structural", "scope": "cross_game",
                    "statement": (f"each of '{gname}'s {ncols} columns sets ONE step "
                                  f"of '{mn}'->'{tn}' (distance {dist_cells} cells "
                                  f"== {ncols} columns)"),
                    "target": [gname, mn, tn], "plan": None,
                    "importance": 0.6, "credence": 0.45, "provenance": "guessed"})
    return out


def _column_count(entities, members, cell_ticks):
    """How many distinct COLUMNS the group's members occupy (clustered by center
    column, cell_ticks apart)."""
    cols = sorted(_center(entities[m]["bbox"])[1] for m in members
                  if m in entities and entities[m].get("bbox"))
    if not cols:
        return 0
    n, last = 1, cols[0]
    for c in cols[1:]:
        if c - last > cell_ticks * 0.6:
            n += 1
        last = c
    return n


def propose_structural_claims(entities, groups, **kw):
    """Run every generator and return the structural claims to ingest.  Each is a
    PRIOR to verify, not a fact (low/guessed credence + a discriminating plan)."""
    claims = []
    try:
        claims += same_structure_template(entities, groups)
    except Exception:
        pass
    try:
        claims += similar_structures(entities, groups)
    except Exception:
        pass
    try:
        claims += similar_entities(entities, groups)
    except Exception:
        pass
    try:
        claims += per_control_step(entities, groups, **kw)
    except Exception:
        pass
    # De-dup by id (a similar pair may be proposed by more than one generator);
    # keep the highest-credence statement of each.
    best = {}
    for c in claims:
        cur = best.get(c["id"])
        if cur is None or c.get("credence", 0) > cur.get("credence", 0):
            best[c["id"]] = c
    return list(best.values())
