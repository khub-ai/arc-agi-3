"""Appearance-AGNOSTIC segmentation + tracking for the offline harness.

Goal: segment a frame into regions and track them across frames WITHOUT
assuming any specific colors — so the offline structure pipeline survives a
regraphed game (the adversarial test). Two ideas, both appearance-independent:

  - SEGMENT by *whatever colors are present*: drop the top-k most-common colors
    as background, then connected-components each remaining distinct color. This
    groups contiguous same-color regions for ANY palette and separates touching
    differently-colored objects. Recolor the game → the colors differ but the
    grouping is identical → the same regions.
  - TRACK by bbox overlap (IoU) across frames → per-track bbox_history.

Then `structure_detection.classify_structures` (behavior: static while others
move) names the structures. Nothing here matches a specific color, shape, or
size, so a recolor/resize/reorient leaves the output unchanged.
"""
from __future__ import annotations

from typing import Optional

import re

import numpy as np
from PIL import Image

from world_knowledge import WorldKnowledge, EntityRecord
import structure_detection as SD


# RGB-tuple extractor used by BOTH the audit and the bbox-tightener.
# Accepts: (229,58,163), (RGB 0,0,0), (rgb 30, 147, 255), case-insensitive.
# The optional "RGB" prefix lets us catch entities whose appearance reads
# "a black cutout (RGB 0,0,0)" — the original (digits-only) pattern silently
# skipped them, so the void in lc=4 never made it into either pass.
_RGB_TUPLE_RE = re.compile(
    r"\(\s*(?:RGB\s+)?(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)",
    re.IGNORECASE,
)


def _neighbors(rr, cc, gap):
    """4-directional neighbors, jumping up to `gap` empty cells (so a DASHED
    line — a rod drawn as a dotted stripe — links into one component). gap=0 is
    plain 4-connectivity."""
    for d in range(1, gap + 2):
        yield rr - d, cc
        yield rr + d, cc
        yield rr, cc - d
        yield rr, cc + d


def _cca_bboxes(mask: np.ndarray, min_pixels: int, gap: int = 0) -> list:
    """Connected components of a boolean mask -> [(r1,c1,r2,c2)] (exclusive).
    `gap` bridges up to that many empty cells along a row/column so a dashed
    structure (e.g. a dotted rod) becomes ONE component, not many fragments.
    The bbox is computed from REAL mask cells only (the gap-jump links pieces
    but never inflates the box with phantom area)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            r1 = r2 = r
            c1 = c2 = c
            n = 0
            while stack:
                rr, cc = stack.pop()
                if (rr < 0 or rr >= h or cc < 0 or cc >= w
                        or not mask[rr, cc] or seen[rr, cc]):
                    continue
                seen[rr, cc] = True
                n += 1
                r1, r2 = min(r1, rr), max(r2, rr)
                c1, c2 = min(c1, cc), max(c2, cc)
                stack += list(_neighbors(rr, cc, gap))
            if n >= min_pixels:
                out.append((r1, c1, r2 + 1, c2 + 1))
    return out


def _structural_nonbg_mask(region):
    """Boolean FOREGROUND mask, PALETTE-INVARIANT (Adversarial Test): the union of
    the entity components from silhouette_track.foreground_components, where GROUND
    (large fields + repeated lattices/texture) is excluded by STRUCTURE -- never by
    'the top-k most-common colours' (which shatters on a re-skin / textured board).
    Replaces the banned modal-background figure-ground used across this module."""
    import silhouette_track as _ST
    h, w = region.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for e in _ST.foreground_components(region):
        r0, c0, r1, c1 = e["bbox"]
        mask[r0:r1 + 1, c0:c1 + 1] |= e["mask"]
    return mask


def segment(frame, play_rows: int = 52, min_pixels: int = 4,
             bg_top_k: int = 2, gap_close: int = 0) -> list:
    """Segment the playfield into entity region bboxes, PALETTE-INVARIANTLY.
    `frame` is a path or an HxWx3 array.  Figure/ground is decided by STRUCTURE
    (silhouette_track.foreground_components: a colour spanning a large field or
    forming a repeated lattice/texture is ground), NOT by dropping the `bg_top_k`
    most-common colours -- that smuggled a single/few-colour-background assumption
    that shatters on a re-skin or a textured board (the Adversarial Test).
    `bg_top_k` / `gap_close` are retained for signature compatibility but no longer
    drive figure/ground.  bbox is (r1,c1,r2,c2) exclusive."""
    arr = (np.array(Image.open(frame).convert("RGB"))
           if not isinstance(frame, np.ndarray) else frame)
    region = arr[:play_rows]
    # PALETTE-INVARIANT: entity regions come from structural figure-ground
    # (foreground_components -- ground is large fields + repeated lattices), NOT
    # from dropping the top-k modal colours.  bbox is (r1,c1,r2,c2) exclusive.
    import silhouette_track as _ST
    return [(e["bbox"][0], e["bbox"][1], e["bbox"][2] + 1, e["bbox"][3] + 1)
            for e in _ST.foreground_components(region) if e["npix"] >= min_pixels]


def coverage(frame, entity_bboxes, play_rows: int = 52, bg_top_k: int = 2,
             min_uncovered: int = 4, gap: int = 0) -> dict:
    """COMPLETENESS GATE — reconcile a named-entity list against the actual
    pixels so perception can never SILENTLY miss a visible thing.

    Given the frame and the bboxes of whatever entities a perception layer
    named, compute the non-background pixels NOT covered by any entity box.
    Any remaining cluster (>= min_uncovered px) is an entity the layer DROPPED.

    Returns {complete, covered_fraction, uncovered_regions, n_nonbg}. The driver
    surfaces uncovered_regions as 'unclassified_present' entities (role TBD) and
    flags complete=False — turning a blind spot into an explicit, actionable
    fact instead of an invisible omission. Game-agnostic: only background-vs-not
    and geometry, no colour/shape assumptions."""
    arr = (np.array(Image.open(frame).convert("RGB"))
           if not isinstance(frame, np.ndarray) else frame)
    region = arr[:play_rows]
    h, w, _ = region.shape
    # PALETTE-INVARIANT non-background: structural foreground, not 'not-in-top-k-
    # modal-colours' (the latter flags a textured board as missing content).
    nonbg = _structural_nonbg_mask(region)
    covered = np.zeros((h, w), dtype=bool)
    for (r1, c1, r2, c2) in entity_bboxes:
        covered[max(0, r1):min(h, r2), max(0, c1):min(w, c2)] = True
    uncovered = nonbg & ~covered
    regions = _cca_bboxes(uncovered, min_uncovered, gap=gap)
    n_nonbg = int(nonbg.sum())
    n_unc = int(uncovered.sum())
    return {
        "complete": not regions,
        "covered_fraction": (1.0 - n_unc / n_nonbg) if n_nonbg else 1.0,
        "uncovered_regions": regions,
        "n_nonbg": n_nonbg,
        "n_uncovered": n_unc,
    }


def audit_entity_pixel_content(
    frame, entities: list,
    play_rows: int = 52, bg_top_k: int = 2,
    min_nonbg_fraction: float = 0.05,
    min_appearance_match_fraction: float = 0.02,
    scenery_size_multiple: int = 5,
    max_owned_by_others_fraction: float = 0.95,
) -> list:
    """PER-ENTITY PIXEL-CONTENT AUDIT — complement to coverage().

    coverage() catches DROPPED entities (visible pixels that no bbox covers).
    This catches PHANTOM and APPEARANCE-MISMATCH entities (bboxes that cover
    nothing distinct, or cover pixels inconsistent with the declared appearance).
    Both blind spots exist because coverage() validates 'are there pixels under
    every bbox' but never 'are the pixels under bbox X actually consistent with
    what entity X is claimed to be'.

    Three deterministic checks per entity, all from the FRAME, none from the
    entity name (so this is game-agnostic):

      1. PHANTOM (background_only): bbox interior is essentially all background
         colour(s). The entity exists in the declaration but not in the image.
         e.g. a 'chain_connector' declared in a region where the only pixels
         are floor.

      2. APPEARANCE_RGBS_ABSENT: the entity's ``appearance`` string mentions
         specific RGB tuples (e.g. ``"(30,147,255)"``); the bbox interior
         contains essentially zero pixels of any of those colours. Catches
         symmetry hallucinations like declaring a red chain by analogy to a
         visible blue chain.

      3. NO_UNIQUE_CONTENT: every non-background pixel inside the bbox is
         ALREADY inside the bbox of another smaller entity (i.e., the
         declaration adds no pixels that aren't accounted for by some other
         entity). Catches the trial-13 chain_connector_red pattern: its bbox
         straddles the three block_red entities, so its only non-bg pixels
         are red — and those reds are entirely owned by block_red_1/2/3.
         Other entities ARE allowed to overlap with much larger "scenery"
         entities (floor, hud_strip) -- the overlap check skips entities
         whose bbox is ``scenery_size_multiple``x larger than this one, so
         a small block sitting on the floor is not mistakenly flagged.

    The harness ONLY MEASURES; it does NOT remove anything from the inventory.
    Per the spec, the VLM gets the flagged list and decides whether each
    entity should be revised or dropped.  Returns a list of dicts, one per
    suspect entity (empty list = nothing suspicious).
    """
    arr = (np.array(Image.open(frame).convert("RGB"))
           if not isinstance(frame, np.ndarray) else frame)
    H, W, _ = arr.shape   # use the WHOLE frame so HUD entities are auditable too

    rgb_pattern = _RGB_TUPLE_RE
    suspects: list = []
    # PALETTE-INVARIANT non-bg mask (whole frame): structural foreground, not the
    # banned top-k modal background.  Used by the phantom + no-unique-content checks.
    nonbg_mask = _structural_nonbg_mask(arr)
    # Pre-compute each candidate-other entity's pixel coverage mask
    ent_areas: list = []   # (area, mask) per entity for the others-overlap step
    for ee in entities or []:
        bb2 = ee.get("bbox_ticks_turn1")
        if not isinstance(bb2, (list, tuple)) or len(bb2) != 4:
            ent_areas.append(None)
            continue
        ya, xa, yb, xb = [max(0, int(v)) for v in bb2]
        yb = min(H, yb); xb = min(W, xb)
        m = np.zeros((H, W), dtype=bool)
        m[ya:yb, xa:xb] = True
        ent_areas.append(((yb - ya) * (xb - xa), m))
    for idx, e in enumerate(entities or []):
        bb = e.get("bbox_ticks_turn1")
        if not isinstance(bb, (list, tuple)) or len(bb) != 4:
            continue
        y0, x0, y1, x1 = [int(v) for v in bb]
        # clip & guard
        if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
            continue
        interior = arr[y0:y1, x0:x1]
        total = interior.shape[0] * interior.shape[1]
        if total <= 0:
            continue
        flat_int = interior.reshape(-1, 3)
        # foreground fraction (palette-invariant): how much of the bbox interior is
        # STRUCTURAL foreground, from the precomputed nonbg_mask -- not 'not a top-k
        # modal colour'.
        nonbg_count = int(nonbg_mask[y0:y1, x0:x1].sum())
        nonbg_frac = nonbg_count / total
        # appearance-declared RGB tuples
        appearance = (e.get("appearance") or "")
        mined_rgbs = {tuple(int(g) for g in m.groups())
                      for m in rgb_pattern.finditer(appearance)}
        appearance_match_count = 0
        appearance_match_frac = None
        if mined_rgbs:
            matched = np.zeros(flat_int.shape[0], dtype=bool)
            for col in mined_rgbs:
                matched |= (flat_int == np.array(col)).all(axis=1)
            appearance_match_count = int(matched.sum())
            appearance_match_frac = (appearance_match_count / total)
        # NO_UNIQUE_CONTENT: check whether THIS entity's non-bg pixels are
        # all already inside the bbox of some OTHER entity. We skip "scenery
        # entities" (those whose bbox is much larger than ours, e.g. floor,
        # hud_strip) so the check isn't trivially satisfied by container
        # overlap. The fraction owned-by-others is reported regardless.
        owned_by_others_frac = None
        if nonbg_count > 0:
            my_area = total
            my_mask = ent_areas[idx][1] if ent_areas[idx] else None
            others_mask = np.zeros((H, W), dtype=bool)
            for j, slot in enumerate(ent_areas):
                if j == idx or slot is None:
                    continue
                other_area, other_mask = slot
                if other_area >= scenery_size_multiple * my_area:
                    continue        # scenery-size; would trivially own us
                others_mask |= other_mask
            if my_mask is not None:
                my_nonbg = nonbg_mask & my_mask
                my_nonbg_count = int(my_nonbg.sum())
                if my_nonbg_count > 0:
                    owned_count = int((my_nonbg & others_mask).sum())
                    owned_by_others_frac = owned_count / my_nonbg_count
        # build flags
        flags: list = []
        if nonbg_frac < min_nonbg_fraction:
            flags.append("phantom_background_only")
        if mined_rgbs and appearance_match_frac < min_appearance_match_fraction:
            flags.append("appearance_rgbs_absent")
        if (owned_by_others_frac is not None
                and owned_by_others_frac >= max_owned_by_others_fraction):
            flags.append("no_unique_content")
        # surface (note: entities that pass all checks return NOTHING)
        if flags:
            # dominant non-bg colours in the interior for the VLM's diagnostic
            top_colors: list = []
            uniq, ct = np.unique(flat_int, axis=0, return_counts=True)
            ord2 = ct.argsort()[::-1]
            for i in ord2[:6]:
                c = tuple(int(x) for x in uniq[i])
                top_colors.append({"rgb": list(c), "pixels": int(ct[i])})
            suspects.append({
                "name": e.get("name"),
                "bbox": [y0, x0, y1, x1],
                "flags": flags,
                "interior_size_pixels": total,
                "non_background_fraction": round(nonbg_frac, 3),
                "appearance_match_fraction": (
                    round(appearance_match_frac, 3)
                    if appearance_match_frac is not None else None),
                "interior_top_colors": top_colors,
                "declared_rgbs": [list(c) for c in mined_rgbs],
                "owned_by_others_fraction": (
                    round(owned_by_others_frac, 3)
                    if owned_by_others_frac is not None else None),
            })
    return suspects


def tighten_entity_bboxes(
    frame, entities: list,
    search_pad: int = 4, min_pixels: int = 3,
) -> list:
    """DETERMINISTIC BBOX REFINEMENT — snap declared bboxes to actual pixels.

    Run AFTER the perception VLM has named the entities + the audit has run.
    For each entity whose ``appearance`` string contains one or more
    ``(R,G,B)`` triples, build a small search window around the declared
    bbox (padded by ``search_pad`` on every side, clipped to the frame),
    find every pixel inside that window matching ANY of the declared RGBs,
    connected-component-label them, and choose the component with the
    largest intersection with the ORIGINAL bbox interior. Replace the
    declared bbox with that component's tight bbox.

    Why: VLM-supplied bboxes drift by 1-3 ticks routinely (the human-VLM
    eyeballs the grid; an ML VLM rounds to its tokenizer's stride). For
    tasks like "is the void 6 ticks wide or 8?" or "does the manipulator
    head's bbox include its outline column?", that drift propagates into
    every downstream planner. This pass is the deterministic counterpart
    to ``audit_entity_pixel_content``: the audit catches phantoms; the
    tightener fixes off-by-1/2 bboxes mechanically, without another VLM
    round-trip. Entities with no RGB in their appearance (text-only
    descriptions) are passed through unchanged.

    The pass is conservative:
      * Only triggers when the declared RGBs are unambiguous (parsed from
        text).
      * Tighter bbox must contain at least ``min_pixels`` matching pixels —
        otherwise the original bbox is kept.
      * Picks the matching component with the LARGEST OVERLAP with the
        declared bbox interior, so a stray pixel elsewhere in the search
        window can't drag the bbox off-target.
      * Only the floor/void's declared RGB *as the entity's own colour* is
        used — background colours are NOT excluded from the match set, so
        a legitimate void/cutout entity whose RGB matches the border
        background still gets tightened correctly. The search-window
        clipping prevents the border from leaking in.

    Returns a NEW entity list (originals not mutated). Tightened entities
    carry a ``_bbox_tightened: True`` marker with ``_bbox_original`` for
    audit/traceability.
    """
    arr = (np.array(Image.open(frame).convert("RGB"))
           if not isinstance(frame, np.ndarray) else frame)
    H, W = arr.shape[:2]
    rgb_pattern = _RGB_TUPLE_RE
    out: list = []
    for e in entities or []:
        bbox = e.get("bbox_ticks_turn1")
        appearance = (e.get("appearance") or "")
        rgbs = {tuple(int(g) for g in m.groups())
                for m in rgb_pattern.finditer(appearance)}
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4 and rgbs):
            out.append(e)
            continue
        y0, x0, y1, x1 = [int(v) for v in bbox]
        if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
            out.append(e)
            continue
        sy0 = max(0, y0 - search_pad)
        sx0 = max(0, x0 - search_pad)
        sy1 = min(H, y1 + search_pad)
        sx1 = min(W, x1 + search_pad)
        window = arr[sy0:sy1, sx0:sx1]
        mask = np.zeros(window.shape[:2], dtype=bool)
        for (r, g, b) in rgbs:
            mask |= ((window[..., 0] == r)
                     & (window[..., 1] == g)
                     & (window[..., 2] == b))
        if mask.sum() < min_pixels:
            out.append(e)
            continue
        # Pick the connected component with the biggest intersection with
        # the original (in-window) bbox.
        components = _cca_bboxes(mask, min_pixels=min_pixels, gap=0)
        if not components:
            out.append(e)
            continue
        ory0, orx0 = y0 - sy0, x0 - sx0
        ory1, orx1 = y1 - sy0, x1 - sx0
        best = None
        best_overlap = -1
        for (cy0, cx0, cy1, cx1) in components:
            ov_y = max(0, min(cy1, ory1) - max(cy0, ory0))
            ov_x = max(0, min(cx1, orx1) - max(cx0, orx0))
            ov = ov_y * ov_x
            if ov > best_overlap:
                best_overlap = ov
                best = (cy0, cx0, cy1, cx1)
        if best is None or best_overlap <= 0:
            out.append(e)
            continue
        # Translate window-relative bbox back to frame coordinates.
        ny0 = best[0] + sy0
        nx0 = best[1] + sx0
        ny1 = best[2] + sy0
        nx1 = best[3] + sx0
        new_bbox = [ny0, nx0, ny1, nx1]
        if new_bbox == [y0, x0, y1, x1]:
            out.append(e)
            continue
        e2 = dict(e)
        e2["bbox_ticks_turn1"] = new_bbox
        e2["_bbox_tightened"] = True
        e2["_bbox_original"] = [y0, x0, y1, x1]
        out.append(e2)
    return out


def _iou(a, b) -> float:
    lo_r, lo_c = max(a[0], b[0]), max(a[1], b[1])
    hi_r, hi_c = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, hi_r - lo_r) * max(0, hi_c - lo_c)
    if inter == 0:
        return 0.0
    area = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / area if area else 0.0


def track(frames: list, turns: Optional[list] = None, iou_thresh: float = 0.3,
           **seg_kw) -> WorldKnowledge:
    """Segment each frame and link regions across frames by best-IoU into
    tracks, returned as a WorldKnowledge whose EntityRecords carry bbox_history.
    Identity is by overlap (a static structure links to itself trivially);
    role is left unset (the behavior classifier needs no role here — moving
    things fail the static test anyway)."""
    turns = turns or list(range(1, len(frames) + 1))
    w = WorldKnowledge(game_id="agnostic", level=0)
    tracks: list[dict] = []         # {bbox, history:[(turn,bbox)]}
    for turn, fr in zip(turns, frames):
        regions = segment(fr, **seg_kw)
        used = set()
        for tk in tracks:
            best, bj = iou_thresh, -1
            for j, rb in enumerate(regions):
                if j in used:
                    continue
                s = _iou(tk["bbox"], rb)
                if s >= best:
                    best, bj = s, j
            if bj >= 0:
                used.add(bj)
                tk["bbox"] = regions[bj]
                tk["history"].append((turn, list(regions[bj])))
        for j, rb in enumerate(regions):
            if j in used:
                continue
            tracks.append({"bbox": rb, "history": [(turn, list(rb))]})
    for i, tk in enumerate(tracks):
        w.entities[f"region_{i}"] = EntityRecord(
            name=f"region_{i}", first_seen_turn=tk["history"][0][0],
            last_seen_turn=tk["history"][-1][0],
            bbox_history=tk["history"], role_history=[])
    return w


def structures_from_frames(frames: list, turns: Optional[list] = None,
                            min_static_turns: int = 2, **seg_kw) -> list:
    """End-to-end appearance-agnostic structure detection from a frame
    sequence: segment + track + classify-by-behavior. Returns the structure
    tracks (each {name, orientation, static_turns, bbox})."""
    w = track(frames, turns=turns, **seg_kw)
    structs = SD.classify_structures(w, min_static_turns=min_static_turns)
    last = {n: (rec.bbox_history[-1][1] if rec.bbox_history else None)
            for n, rec in w.entities.items()}
    for s in structs:
        s["bbox"] = last.get(s["name"])
    return structs
