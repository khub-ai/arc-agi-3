"""Colour-independent SILHOUETTE tracking + SHAPE identity (game-agnostic).

A human recognises the grey arch in a legend's preview as the (abstracted) yellow
arch in the play area by its SHAPE alone -- colour and scale are incidental.  The
substrate must do the same.  The per-colour connected-component extractor used
for ordinary animations fails here: a grey silhouette tracing a path over a grey
board merges into the modal-colour background, so the mover fragments into large
grey blobs and no clean per-frame entity track survives.

This module fixes that with two ideas:

  1. ISOLATE BY NOVELTY, not colour.  In each sub-frame the mover is whatever
     DIFFERS from the STATIC first frame (the scene before the motion).  A
     silhouette is exactly what is added/changed relative to the still scene, so
     it is recovered regardless of whether its colour matches the board.

  2. IDENTIFY BY SHAPE, not colour.  Each isolated mover is matched to a known
     static entity (e.g. the yellow arch) by SCALE-NORMALISED shape overlap --
     both masks are cropped to content and resized to a common grid, then
     compared by IoU.  The grey arch-silhouette and the yellow arch coincide in
     shape, so the mover is identified as 'the arch (abstracted)'.

Frames are discrete palette images, so 'differs from static' is an EXACT pixel
inequality -- no tuned colour tolerance.  The only similarity cut is MAJORITY
shape overlap (the normalised shapes coincide over more than half their union),
a principled structural criterion, not a tuned knob.

Pure functions over numpy arrays; no I/O, no driver dependency -> unit-testable.
"""

from __future__ import annotations

import numpy as np

_NORM = 14          # common grid for scale-normalised shape compare
_MAJORITY = 0.5     # shapes "match" when >half their normalised union coincides


def connected_components(mask):
    """4-connected components of a boolean mask.  Returns, per component,
    (r0, c0, r1, c1 inclusive, centroid_r, centroid_c, npix, submask) where
    submask is the component's own boolean shape cropped to its bbox."""
    H, W = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.where(mask)
    for y0, x0 in zip(ys.tolist(), xs.tolist()):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        cells = []
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        sub = mask[r0:r1 + 1, c0:c1 + 1].copy()
        comps.append((r0, c0, r1, c1, sum(rs) / len(rs), sum(cs) / len(cs),
                      len(cells), sub))
    return comps


def norm_shape(submask, size=_NORM):
    """Scale + position invariant shape descriptor, ASPECT-RATIO PRESERVING: crop
    a boolean mask to its content, scale it by its LARGER side so the longer axis
    fills the grid, and centre it.  Preserving aspect is essential -- otherwise a
    thin 6x1 line and a compact 3x5 arch both squash to a filled grid and falsely
    match.  A vertical line stays a thin central column; the arch stays an arch."""
    m = np.asarray(submask, dtype=bool)
    if m.size == 0 or not m.any():
        return np.zeros((size, size), dtype=bool)
    ys, xs = np.where(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    scale = float(size) / float(max(h, w))
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    small = np.zeros((nh, nw), dtype=bool)
    for i in range(nh):
        for j in range(nw):
            small[i, j] = m[min(h - 1, i * h // nh), min(w - 1, j * w // nw)]
    out = np.zeros((size, size), dtype=bool)
    r0, c0 = (size - nh) // 2, (size - nw) // 2
    out[r0:r0 + nh, c0:c0 + nw] = small
    return out


def shape_iou(a, b) -> float:
    """Intersection-over-union of two same-size boolean shape grids."""
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    u = (a | b).sum()
    return float((a & b).sum()) / float(u) if u else 0.0


def _concavity(submask):
    """(vert, horz) fill-asymmetry of a shape's CENTRE bands, in [-1, 1].

    vert = (bottom-centre fill) - (top-centre fill): > 0 means the bottom is
    fuller and the OPENING faces UP (a cup ∪); < 0 means the OPENING faces DOWN
    (an arch ∩).  horz likewise: > 0 -> opening faces LEFT, < 0 -> RIGHT.  This
    is the structural feature that separates ∩ from ∪ -- two shapes that overlap
    heavily once scale-normalised (so IoU can't tell them apart) but whose
    concavity points the OPPOSITE way.  Pure geometry, palette-independent."""
    m = np.asarray(submask, dtype=bool)
    if not m.any():
        return 0.0, 0.0
    ys, xs = np.where(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    if h < 2 or w < 2:
        return 0.0, 0.0
    cc0, cc1 = w // 4, max(w // 4 + 1, w - w // 4)     # central columns
    cr0, cr1 = h // 4, max(h // 4 + 1, h - h // 4)     # central rows
    top = float(m[:h // 2, cc0:cc1].mean())
    bottom = float(m[h - h // 2:, cc0:cc1].mean())
    left = float(m[cr0:cr1, :w // 2].mean())
    right = float(m[cr0:cr1, w - w // 2:].mean())
    return bottom - top, right - left


def opening_direction(submask):
    """The single direction a shape's OPENING faces ('up'/'down'/'left'/'right'),
    or None if the shape is ~closed/symmetric.  Derived from _concavity along its
    dominant axis -- e.g. a cup ∪ -> 'up', an arch ∩ -> 'down'.  For narration /
    transparency: a human identifies the grey cup by the gap facing up, and the
    substrate can report the same structural cue alongside the shape match."""
    vert, horz = _concavity(submask)
    if abs(vert) < 1e-6 and abs(horz) < 1e-6:
        return None
    if abs(vert) >= abs(horz):
        return "up" if vert > 0 else "down"
    return "left" if horz > 0 else "right"


def foreground_components(arr, area_frac=0.12, lattice_min=6):
    """PALETTE-INVARIANT figure-ground: connected uniform-colour components of
    EVERY colour, classifying ground by STRUCTURE only -- NEVER by designating a
    'background colour'.  A component is GROUND (skipped) when it spans a large
    fraction (> area_frac) of the frame (a field/board) OR it is one of the tiles
    of a repeated TEXTURE that FILLS a region (a checkerboard field); the rest are
    entity components.

    This is the adversarial-safe replacement for 'bg = the modal colour; entity =
    pixels != bg'.  Keying on SIZE and REPETITION (structural Core-Knowledge cues)
    instead of a colour value means it survives a re-skin, a palette remap, OR a
    multi-colour / textured background (e.g. a 2-colour checkerboard) -- the exact
    cases a single modal background shatters on.

    TEXTURE IS SPATIALLY SCOPED.  The lattice cue (many similar-sized components of
    a colour) only NOMINATES a colour as a possible texture; ground is then keyed
    on the texture's STRUCTURAL FOOTPRINT.  The tiles of every nominated colour are
    unioned into one mask; a connected blob of that mask is a FIELD (ground) only
    when it fills a large fraction of the frame.  So a checkerboard's tiles -- which
    pack edge-to-edge into one frame-filling blob -- are dropped, while the SAME
    colour's tiles forming a small cluster elsewhere (a control row/column on a
    panel, separated from the field by a border/gap) survive as entities.  This is
    why a control panel of grey/black switches that happen to match the board's
    palette is no longer swallowed by the board, and a textured background is no
    longer shattered into dozens of tile 'entities'.  Returns a list of dicts
    {bbox:(r0,c0,r1,c1), centroid:(r,c), colour:int, npix:int, mask:submask}."""
    arr = np.asarray(arr, dtype=int)
    packed = (arr[:, :, 0] << 16) | (arr[:, :, 1] << 8) | arr[:, :, 2]
    H, W = packed.shape
    area = H * W

    # 1) FIELD: per colour, drop components spanning a large fraction of the frame
    #    (a board/panel).  Do this FIRST so a colour that is BOTH a big field AND a
    #    texture (e.g. black: one large playfield + many checker tiles) has its
    #    field removed before the texture step sees the tiles.
    # No absolute pixel-size floor: ARC frames are EXACT palette images (no
    # anti-aliasing / speckle), so every connected same-colour patch is a real
    # candidate -- a switch mark can be only 3 px at native 64x64.  Backgrounds
    # are removed STRUCTURALLY below (large field comps, then texture tiles), not
    # by a tuned minimum size.
    small_by_colour: dict[int, list] = {}
    for cv in (int(v) for v in np.unique(packed)):
        comps = [c for c in connected_components(packed == cv)
                 if (c[2] - c[0] + 1) * (c[3] - c[1] + 1) <= area_frac * area]
        if comps:
            small_by_colour[cv] = comps

    # 2) TEXTURE COLOURS: a colour whose surviving components are many and
    #    similar-sized is a repeated tile colour (a checkerboard half).  This only
    #    NOMINATES the colour -- it does not drop anything yet.
    texture_colours = {
        cv for cv, comps in small_by_colour.items()
        if len(comps) >= lattice_min
        and float(np.std([c[6] for c in comps])) <= 0.6 * float(np.mean([c[6] for c in comps]))
    }

    # 3) SPATIAL SCOPE: union every nominated colour's tiles into one mask; a
    #    connected blob that fills a large fraction of the frame is a FIELD ->
    #    ground.  Small/isolated blobs (a switch row on a panel) are NOT a field.
    ground_mask = np.zeros((H, W), dtype=bool)
    if texture_colours:
        tex_mask = np.zeros((H, W), dtype=bool)
        for cv in texture_colours:
            for (r0, c0, r1, c1, cr, cc, npx, sub) in small_by_colour[cv]:
                tex_mask[r0:r1 + 1, c0:c1 + 1] |= sub
        for (r0, c0, r1, c1, cr, cc, npx, sub) in connected_components(tex_mask):
            if (r1 - r0 + 1) * (c1 - c0 + 1) >= area_frac * area:
                ground_mask[r0:r1 + 1, c0:c1 + 1] |= sub

    # 4) EMIT entities: every surviving component EXCEPT texture tiles that fall
    #    inside a field blob.  Non-texture colours (a distinct sprite on the field)
    #    are never in the texture mask, so a glyph on the board is always kept.
    out = []
    for cv, comps in small_by_colour.items():
        is_tex = cv in texture_colours
        for (r0, c0, r1, c1, cr, cc, npx, sub) in comps:
            if is_tex and ground_mask[int(round(cr)), int(round(cc))]:
                continue
            out.append({"bbox": (r0, c0, r1, c1), "centroid": (float(cr), float(cc)),
                        "colour": int(cv), "npix": int(npx), "mask": sub})
    return out


def track_per_frame_entities(frames):
    """Per-frame entities, CHANGE-DRIVEN (the efficient + identity-stable view).

    Frame 0 is fully analysed (foreground_components).  For each later frame, an
    entity from the previous frame whose pixels did NOT change is CARRIED OVER
    unchanged -- same entity, same id, no re-interpretation (the insight: a region
    with zero pixel change holds the same entities).  Only the CHANGED regions are
    re-detected; a re-detected component is MATCHED to a previous entity of the
    same colour by nearest centroid, so a moving entity keeps its id and its box
    FOLLOWS it across frames (instead of a stale fixed box that would make the VLM
    misread the motion).  Returns one list per frame of entity dicts
    {id, bbox, centroid, colour, npix, moved}.  Palette-invariant throughout."""
    frames = [np.asarray(f, dtype=int) for f in (frames or [])]
    if not frames:
        return []

    def _cen(e):
        r0, c0, r1, c1 = e["bbox"]
        return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

    first = foreground_components(frames[0])
    for k, e in enumerate(first):
        e["id"], e["moved"] = k, False
    per = [first]
    nxt = len(first)
    for i in range(1, len(frames)):
        changed = (frames[i] != frames[i - 1]).any(axis=2)
        prev = per[i - 1]
        cur, used = [], set()
        for e in prev:                                  # carry fully-unchanged
            r0, c0, r1, c1 = e["bbox"]
            if not changed[r0:r1 + 1, c0:c1 + 1].any():
                cur.append({**e, "moved": False})
                used.add(e["id"])
        for d in foreground_components(frames[i]):      # re-detect changed regions
            r0, c0, r1, c1 = d["bbox"]
            if not changed[r0:r1 + 1, c0:c1 + 1].any():
                continue                                # unchanged -> already carried
            cd, best, bd = _cen(d), None, 1e9
            for e in prev:                              # match a moved entity to its id
                if e["colour"] != d["colour"] or e["id"] in used:
                    continue
                ce = _cen(e)
                dist = abs(cd[0] - ce[0]) + abs(cd[1] - ce[1])
                if dist < bd:
                    bd, best = dist, e
            if best is not None and bd <= 24:
                d["id"] = best["id"]
                used.add(best["id"])
            else:
                d["id"] = nxt
                nxt += 1
            d["moved"] = True
            cur.append(d)
        per.append(cur)
    return per


def _glyph_mask_from_bbox(packed, bbox):
    """Isolate the FOREGROUND GLYPH inside (a generous pad around) a known
    entity's bbox, ROBUST TO A LOOSE OR MISPLACED bbox.

    Method (palette-invariant, structural figure-ground): pad the bbox, then any
    colour that TOUCHES THE BORDER of the padded region is the surrounding context
    (a field/divider/board); the glyph is the interior component(s) that overlap
    the bbox.  This fixes the failure where a loose bbox made the bbox's most-
    common colour the SURROUND -- e.g. a small yellow arch in a bbox dominated by
    the black board, whose 'dominant colour' is black, so the recovered 'shape'
    was the board, not the arch.  Keying on WHERE a colour appears (border =
    context) rather than its value survives a re-skin.  Falls back to the bbox's
    own dominant colour when no clean foreground is found (a glyph that fills the
    whole padded region)."""
    H, W = packed.shape
    r0, c0, r1, c1 = [int(v) for v in bbox]
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(H, r1), min(W, c1)
    if r1 <= r0 or c1 <= c0:
        return None
    pad = max(4, (r1 - r0) // 2, (c1 - c0) // 2)
    R0, C0 = max(0, r0 - pad), max(0, c0 - pad)
    R1, C1 = min(H, r1 + pad), min(W, c1 + pad)
    region = packed[R0:R1, C0:C1]
    if region.size == 0:
        return None
    border = set()
    border.update(int(v) for v in region[0, :])
    border.update(int(v) for v in region[-1, :])
    border.update(int(v) for v in region[:, 0])
    border.update(int(v) for v in region[:, -1])
    fg = ~np.isin(region, list(border))
    if fg.any():
        keep = np.zeros_like(fg)
        ir0, ic0, ir1, ic1 = r0 - R0, c0 - C0, r1 - R0, c1 - C0
        for (a0, b0, a1, b1, _cr, _cc, _n, sm) in connected_components(fg):
            if not (a1 < ir0 or a0 > ir1 or b1 < ic0 or b0 > ic1):
                keep[a0:a1 + 1, b0:b1 + 1] |= sm
        if keep.any():
            return keep
    # FALLBACK: the bbox's own dominant colour (the original behaviour) -- used
    # only when surround-isolation found no interior foreground.
    inner = packed[r0:r1, c0:c1]
    if inner.size == 0:
        return None
    vals, counts = np.unique(inner, return_counts=True)
    return inner == int(vals[counts.argmax()])


def entity_shape_masks(frame, entities):
    """For each known static entity (name -> (r0,c0,r1,c1) bbox), its
    scale-normalised SHAPE = the FOREGROUND GLYPH isolated from the bbox by
    SURROUND (see _glyph_mask_from_bbox).

    Palette-invariant + robust to a loose/misplaced perception bbox: the glyph is
    whatever is NOT the surrounding context (the colours touching the padded
    region's border), so a small yellow arch is recovered as an arch even when its
    bbox is dominated by the black board around it.  Under a re-skin the surround
    and glyph colours both remap and the SHAPE we compare on is unchanged."""
    frame = np.asarray(frame, dtype=int)
    packed = (frame[:, :, 0] << 16) | (frame[:, :, 1] << 8) | frame[:, :, 2]
    out = {}
    for name, bbox in entities.items():
        sub = _glyph_mask_from_bbox(packed, bbox)
        if sub is not None and np.asarray(sub).any():
            out[name] = norm_shape(sub)
    return out


def identify(submask, known_masks):
    """Identify a mover's shape with the best-matching known entity by majority
    normalised-shape overlap, with OPENING-DIRECTION as a tie-breaker.  Returns
    (name, iou) or (None, best_iou); the match must exceed majority overlap -- a
    structural cut.

    The tie-breaker (threshold-free) sorts candidates by (IoU, opening agrees with
    the query): when two references overlap the query equally, the one whose
    concavity points the SAME way wins -- so an arch ∩ and a cup ∪ (which alias
    under scale-normalised IoU) are separated by the OPPOSITE direction their
    openings face, not by a tuned margin."""
    if not known_masks:
        return None, 0.0
    q = norm_shape(submask)
    qv, qh = _concavity(q)
    vertical = abs(qv) >= abs(qh)

    def agrees(m):
        mv, mh = _concavity(m)
        if vertical:
            return 1 if (abs(qv) > 1e-6 and (qv > 0) == (mv > 0)) else 0
        return 1 if (abs(qh) > 1e-6 and (qh > 0) == (mh > 0)) else 0

    scored = sorted(((shape_iou(q, m), agrees(m), name)
                     for name, m in known_masks.items()),
                    key=lambda t: (t[0], t[1]), reverse=True)
    best_iou, _agree, best_name = scored[0]
    if best_iou >= _MAJORITY:
        return best_name, best_iou
    return None, best_iou


def scene_cuts(frames):
    """Indices i (>=1) where frame i is a VIEW CHANGE from frame i-1 -- the
    MAJORITY of the frame's pixels changed at once.  A localized entity motion
    touches a small region; a view change (zoom / overlay / different screen)
    repaints most of the frame.  The majority (>half) criterion is structural,
    not a tuned knob.  Diffing a motion tracker ACROSS such a cut is meaningless
    (it invents huge phantom movers), so callers must segment on these."""
    frames = [np.asarray(f, dtype=int) for f in (frames or [])]
    n = frames[0].shape[0] * frames[0].shape[1] if frames else 1
    cuts = []
    for i in range(1, len(frames)):
        if ((frames[i] != frames[i - 1]).any(axis=2)).sum() > 0.5 * n:
            cuts.append(i)
    return cuts


def track_silhouette(frames, known_masks):
    """Track movers across an animation (list of HxWx3 int arrays) by NOVELTY vs
    the static first frame, identify each by SHAPE, and return the IDENTIFIED
    movers.  Each result: {identity, trajectory:[(frame_idx,(r,c))...], from,
    to, net:(dr,dc), reach:(dr,dc), dir, iou}.  Unidentified movers (scan cursors,
    highlights -- shapes matching no known entity) are dropped.

    SCENE-CUT AWARE: an animation may CHANGE VIEW partway (zoom / overlay).
    Diffing across a cut repaints most of the frame and fabricates huge phantom
    movers, so we track ONLY within the first contiguous segment -- the original
    scene that the known-entity masks describe.  Frames after the first cut are a
    DIFFERENT view (their content is the VLM's to read, not shape-matched to the
    main-view entities)."""
    frames = [np.asarray(f, dtype=int) for f in (frames or [])]
    if len(frames) < 2:
        return []
    cuts = scene_cuts(frames)
    end = cuts[0] if cuts else len(frames)     # first segment = original scene
    seg = frames[:end]
    if len(seg) < 2:
        return []
    static = seg[0]
    # per-frame movers = exact-diff vs static -> components (colour-independent)
    per_frame = []
    for fi in seg[1:]:
        changed = (fi != static).any(axis=2)
        comps = [c for c in connected_components(changed) if c[6] >= 4]
        per_frame.append(comps)
    if not any(per_frame):
        return []
    # thread by nearest centroid across consecutive frames
    tracks = []                                   # {cen:[(k,(r,c))], subs:[mask]}
    for k, comps in enumerate(per_frame):
        for (r0, c0, r1, c1, cr, cc, npx, sub) in comps:
            best, bd = None, 1e9
            for t in tracks:
                lk, (lr, lc) = t["cen"][-1]
                if lk == k:
                    continue
                d = abs(lr - cr) + abs(lc - cc)
                if d < bd:
                    bd, best = d, t
            if best is not None and bd <= 8 and best["cen"][-1][0] != k:
                best["cen"].append((k, (cr, cc)))
                best["subs"].append(sub)
            else:
                tracks.append({"cen": [(k, (cr, cc))], "subs": [sub]})
    out = []
    for t in tracks:
        if len(t["cen"]) < 2:
            continue
        traj = t["cen"]
        r0c0 = traj[0][1]
        far = max(traj, key=lambda p: (p[1][0] - r0c0[0]) ** 2 + (p[1][1] - r0c0[1]) ** 2)
        dr, dc = far[1][0] - r0c0[0], far[1][1] - r0c0[1]
        # a PREVIEW mover must actually MOVE -- drop static fragments (a flicker
        # that never travels is not a demonstrated motion).
        if abs(dr) + abs(dc) < 2.0:
            continue
        # representative shape = the track's MODAL-DIMENSION mask, NOT the
        # largest.  On the FIRST step a mover's diff-vs-static merges its OLD and
        # NEW positions into one oversized blob (motion doubling); 'largest' then
        # picks that smeared blob and its shape matches nothing.  The dimension
        # that RECURS across the track is the entity's true extent; among masks of
        # that modal size take the largest (most complete) instance.
        from collections import Counter as _Counter
        _dims = _Counter(np.asarray(s).shape for s in t["subs"])
        _modal = _dims.most_common(1)[0][0]
        _cands = [s for s in t["subs"] if np.asarray(s).shape == _modal]
        rep = max(_cands, key=lambda s: int(np.asarray(s).sum()))
        name, iou = identify(rep, known_masks)
        if name is None:
            continue
        net = (traj[-1][1][0] - r0c0[0], traj[-1][1][1] - r0c0[1])
        vdir = ("up" if dr < 0 else "down") if abs(dr) >= abs(dc) \
            else ("left" if dc < 0 else "right")
        out.append({"identity": name, "iou": round(iou, 2),
                    "trajectory": [(k, (round(r, 1), round(c, 1))) for k, (r, c) in traj],
                    "from": (round(r0c0[0], 1), round(r0c0[1], 1)),
                    "to": (round(far[1][0], 1), round(far[1][1], 1)),
                    "net": (round(net[0], 1), round(net[1], 1)),
                    "reach": (round(dr, 1), round(dc, 1)), "dir": vdir,
                    "opening": opening_direction(rep)})
    return out
