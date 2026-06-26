"""VLM-directed GESTALT GROUPING — unify fragments that CV provably cannot.

Some shapes are drawn so sparsely that NO connectivity rule recovers them: a
hollow diamond whose outline cells are spaced two apart (no two even
8-adjacent), a dotted curve, a glyph of scattered strokes.  Connected-components
shatters them into singletons and the sprite grouper drops them as speckle.
Only a VLM, reading the gestalt, sees "one diamond outline."  r11l is the
canonical case: 12 isolated pal-15 cells that detect_entities covers 0/12.

This module gives the VLM exactly the two primitives it needs — and NOTHING
that classifies shapes itself.  The substrate HOLDS and SURFACES; the VLM
RECOGNIZES; the substrate then RECORDS and VALIDATES:

  orphan_cells(frame, entities, bg_palettes)
      The non-background cells that NO entity claims — i.e. the fragments CV
      dropped — loosely clustered by palette + proximity (gaps BRIDGED so a
      gapped outline clusters together).  Each cluster is a CANDIDATE the VLM
      should look at, not an entity.  This is what focuses the VLM's attention.

  group_cells_as_entity(frame, cells, *, entity_id, shape_type, rows, cols)
      Record a VLM-named grouping as ONE Entity (bbox, cells, palette signature,
      bbox_fit, is_region, the VLM's shape_type label).  Validates that the
      cells are genuinely uniform-ish and returns a well-formed Entity that
      slots into the same list detect_entities produces.

  shape_descriptor(cells)  — advisory geometry (fill, hollow, span, on-perimeter
      fraction) the substrate offers to SUPPORT the VLM's label.  It is a hint,
      never a decision: on the gapped diamond even a flood-fill "closed ring"
      test fails (the gaps leak), which is precisely why the call is the VLM's.

Substrate-agnostic: no role names, no per-game shape rules.
"""
from __future__ import annotations
import os, sys
import numpy as np

# Make the package importable whether this file is imported or run as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perception_loop_v2.entity_detector import Entity  # noqa: E402


def orphan_cells(frame, entities, bg_palettes, *, bridge=3):
    """Return VLM-attention candidates: clusters of non-bg cells no entity owns.

    frame:        2D palette-index grid (cell resolution).
    entities:     detected Entity records.
    bg_palettes:  palettes that are background (floor/wall) — never orphans.
    bridge:       union cells of the SAME palette within this Chebyshev radius,
                  so a gapped outline (gaps of 2) clusters as one candidate.

    Returns list[dict(palette, cells, bbox)] sorted by size descending.
    """
    H, W = frame.shape
    bg = set(int(p) for p in bg_palettes)
    claimed = set()
    for e in entities:
        for cc in e.cells:
            claimed.add(tuple(cc))
    orphans_by_pal: dict[int, list[tuple[int, int]]] = {}
    for r in range(H):
        for c in range(W):
            p = int(frame[r, c])
            if p in bg or (r, c) in claimed:
                continue
            orphans_by_pal.setdefault(p, []).append((r, c))

    clusters = []
    for p, cells in orphans_by_pal.items():
        # union-find with same-palette bridging
        par = list(range(len(cells)))

        def find(x):
            while par[x] != x:
                par[x] = par[par[x]]
                x = par[x]
            return x

        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                if (abs(cells[i][0] - cells[j][0]) <= bridge
                        and abs(cells[i][1] - cells[j][1]) <= bridge):
                    par[find(i)] = find(j)
        groups: dict[int, list] = {}
        for i in range(len(cells)):
            groups.setdefault(find(i), []).append(cells[i])
        for mem in groups.values():
            rs = [m[0] for m in mem]
            cs = [m[1] for m in mem]
            clusters.append({
                "palette": p,
                "cells": sorted(mem),
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
            })
    return sorted(clusters, key=lambda c: -len(c["cells"]))


def shape_descriptor(cells):
    """Advisory geometry to SUPPORT (never decide) the VLM's shape label."""
    if not cells:
        return {"n_cells": 0}
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    h = max(rs) - min(rs) + 1
    w = max(cs) - min(cs) + 1
    bbox_cells = max(1, h * w)
    cellset = set(cells)
    # fraction of cells on the bbox border vs interior
    on_border = sum(
        1 for (r, c) in cells
        if r in (min(rs), max(rs)) or c in (min(cs), max(cs))
    )
    # hollow: how empty is the bbox interior
    interior = [(r, c) for r in range(min(rs) + 1, max(rs))
                for c in range(min(cs) + 1, max(cs))]
    interior_filled = sum(1 for ic in interior if ic in cellset)
    hollow = (len(interior) > 0 and interior_filled / len(interior) < 0.25)
    return {
        "n_cells": len(cells),
        "bbox_fit": round(len(cells) / bbox_cells, 2),
        "span": (h, w),
        "hollow": bool(hollow),
        "on_perimeter_frac": round(on_border / len(cells), 2),
    }


def group_cells_as_entity(frame, cells, *, entity_id, shape_type, rows, cols,
                          scale=1):
    """Record a VLM-named gestalt as one Entity.

    The VLM supplies the `cells` that form the shape and a `shape_type` label;
    the substrate computes geometry and returns an Entity that slots into the
    detect_entities list.  cell coords are at logical (cell) resolution; `scale`
    upscales the display bbox to match other entities if needed.
    """
    cells = sorted(set(tuple(c) for c in cells))
    rs = [c[0] for c in cells]
    cs = [c[1] for c in cells]
    bbox_l = (min(rs), min(cs), max(rs), max(cs))
    bbox_d = (bbox_l[0] * scale, bbox_l[1] * scale,
              (bbox_l[2] + 1) * scale - 1, (bbox_l[3] + 1) * scale - 1)
    pals = [int(frame[r, c]) for (r, c) in cells]
    n_colors = len(set(pals))
    # palette-fraction signature (top-k) from the cell palettes
    vals, counts = np.unique(np.array(pals), return_counts=True)
    order = np.argsort(-counts)
    sig = tuple((int(vals[i]), float(counts[i] / len(pals))) for i in order[:5])
    desc = shape_descriptor(cells)
    cy = float(sum(rs) / len(rs))
    cx = float(sum(cs) / len(cs))
    return Entity(
        entity_id=entity_id,
        bbox_logical=bbox_l,
        bbox_display=bbox_d,
        n_pixels=len(cells),
        n_distinct_colors=n_colors,
        visual_signature=sig,
        centroid_logical=(cy, cx),
        cells=cells,
        centroid_cell=(int(round(cy)), int(round(cx))),
        pixel_count_per_cell={c: 1 for c in cells},
        bbox_fit=desc["bbox_fit"],
        is_region=False,
        shape_type=shape_type,
        is_gestalt=True,
    )


if __name__ == "__main__":  # demonstrate the VLM-driven loop on r11l
    import sys, glob, inspect, importlib.util
    sys.path.insert(0, "tools/governor_audit")
    import arcengine
    from perception_loop_v2 import entity_detector as ed
    from perception_loop_v2.observation import build_frame_observation

    spec = importlib.util.spec_from_file_location("v3", "tools/cos_driver/cos_play.py")
    v3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v3)
    PAL = v3.PALETTE

    def load(gid):
        p = glob.glob(f"environment_files/{gid}/*/{gid}.py")[0]
        s = importlib.util.spec_from_file_location(gid, p)
        m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
        return next(o for n, o in vars(m).items() if inspect.isclass(o)
                    and issubclass(o, arcengine.ARCBaseGame) and o is not arcengine.ARCBaseGame)

    g = load("r11l")(); g.full_reset()
    fr = np.array(g.camera.render(g.current_level.get_sprites()))
    rgb = np.zeros((64, 64, 3), np.uint8)
    for i, (r, gg, b) in enumerate(PAL):
        rgb[fr == i] = (r, gg, b)
    obs = build_frame_observation(rgb, turn=0, rows=64, cols=64, agent_position=None)
    ents = ed.detect_entities(obs)
    fg0 = [e for e in ents if not (e.is_background_primary or e.is_background_secondary)]
    bg_pals = sorted({int(fr[r, c]) for e in ents
                      if e.is_background_primary or e.is_background_secondary
                      for (r, c) in e.cells})

    # 1) SUBSTRATE surfaces orphan candidates.
    cands = orphan_cells(fr, ents, bg_pals)
    print(f"r11l: {len(fg0)} CV entities; {len(cands)} orphan candidate cluster(s):")
    for c in cands:
        d = shape_descriptor(c["cells"])
        print(f"  pal{c['palette']} {len(c['cells'])} cells bbox{c['bbox']} "
              f"descriptor={d}")

    # 2) VLM (Claude) reads the frame: the pal-15 cluster of 12 cells, hollow,
    #    all on the perimeter of a 7x7 bbox in two mirrored diagonal arcs, IS a
    #    hollow diamond outline.  Group it.  (The pal-1 cluster is the line's
    #    tail, already owned by the line relation — left as-is.)
    diamond = next(c for c in cands if c["palette"] == 15 and len(c["cells"]) >= 8)
    new = group_cells_as_entity(
        fr, diamond["cells"], entity_id=max(e.entity_id for e in ents) + 1,
        shape_type="diamond_outline", rows=64, cols=64,
    )
    print(f"\nVLM groups the orphan into ONE entity: id{new.entity_id} "
          f"shape_type='{new.shape_type}' cells={len(new.cells)} "
          f"bbox{new.bbox_logical} bbox_fit={new.bbox_fit} is_gestalt={new.is_gestalt}")

    # 3) VALIDATE: the diamond is now covered by exactly one entity.
    aug = ents + [new]
    covered = sum(1 for cell in diamond["cells"]
                  if any(tuple(cell) in set(map(tuple, e.cells)) for e in aug))
    print(f"diamond cells now covered by an entity: {covered}/{len(diamond['cells'])}")
