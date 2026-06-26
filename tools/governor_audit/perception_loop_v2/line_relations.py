"""Detect LINES as RELATIONS (connector edges) between entities — not pixels.

A connector line is a thin or DASHED collinear run of same-palette cells whose
two endpoints each lie on a DISTINCT entity.  The line means "entity A is linked
to entity B" (and possibly passes THROUGH intermediate entities).  Interpreting
it as a bag of pixel components destroys that meaning: a dashed line is a
sequence of isolated single cells that connected-components shatters into N
fragments and that the region/object detector either drops or mislabels.

Two grounding cases:
  * r11l: a palette-1 line links the white star and the dark star, passing
    THROUGH the ball that sits on the track between them.
  * su15: a palette-3 DASHED line (single cells spaced every other row) links
    the top blob to the bottom node.

The gap between dashes is bridged by COLLINEARITY (a RANSAC line fit), which no
connectivity rule can do.  The endpoint-touches-a-distinct-entity gate is what
separates a genuine connector from UI chrome (full-edge status bars are straight
lines too) and from scattered same-palette cells that merely admit a collinear
subset.  Substrate-agnostic: no role names, no game-specific code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class LineRelation:
    """A connector edge discovered between two entities."""
    palette: int
    cells: list[tuple[int, int]]          # the line's cells (dashes included)
    endpoints: tuple[tuple[int, int], tuple[int, int]]
    entity_a: int                         # entity_id touched at endpoint A
    entity_b: int                         # entity_id touched at endpoint B
    through: list[int] = field(default_factory=list)  # entity_ids the line crosses


def _perp_dist(q, a, b) -> float:
    """Perpendicular distance of cell q to the infinite line through a, b."""
    (ar, ac), (br, bc), (qr, qc) = a, b, q
    dr, dc = br - ar, bc - ac
    L = (dr * dr + dc * dc) ** 0.5
    if L == 0:
        return ((qr - ar) ** 2 + (qc - ac) ** 2) ** 0.5
    return abs(dr * (ac - qc) - dc * (ar - qr)) / L


def _along(q, a, b) -> float:
    """Normalized projection of q onto segment a->b (0 at a, 1 at b)."""
    (ar, ac), (br, bc), (qr, qc) = a, b, q
    dr, dc = br - ar, bc - ac
    L2 = dr * dr + dc * dc
    if L2 == 0:
        return 0.0
    return ((qr - ar) * dr + (qc - ac) * dc) / L2


def detect_line_relations(
    frame: np.ndarray,
    entities,
    bg_palettes,
    *,
    min_len: float = 8.0,
    max_perp: float = 1.6,
    min_inliers: int = 6,
    endpoint_tol: float = 2.5,
    region_cell_cap: int = 140,
    region_fill_cap: float = 0.25,
    max_gap_cells: float = 6.0,
) -> list[LineRelation]:
    """Return the connector lines in `frame` as entity-to-entity relations.

    frame:        2D palette-index grid (cell resolution).
    entities:     Module Entity records (need .entity_id, .cells,
                  .is_background_primary/.is_background_secondary).
    bg_palettes:  palettes to ignore (floor/wall background).

    A palette is a line candidate only if its cells are SPARSE and elongated
    (a solid region or large object is never a connector); within each
    candidate palette the longest collinear chains are peeled off by RANSAC,
    and a chain is accepted only if BOTH endpoints sit within endpoint_tol of a
    DISTINCT entity.  Endpoints that touch the same entity, no entity, or only
    the frame border (UI bars) are rejected.
    """
    H, W = frame.shape
    # cell -> entity_id, for endpoint attribution and pass-through detection.
    cell_owner: dict[tuple[int, int], int] = {}
    for e in entities:
        if getattr(e, "is_background_primary", False) or getattr(
            e, "is_background_secondary", False
        ):
            continue
        for cc in e.cells:
            cell_owner[tuple(cc)] = e.entity_id

    def nearest_entity(cell):
        """entity_id within endpoint_tol of `cell` (Chebyshev search), or None."""
        best = None
        rad = int(endpoint_tol) + 1
        for dr in range(-rad, rad + 1):
            for dc in range(-rad, rad + 1):
                d = (dr * dr + dc * dc) ** 0.5
                if d > endpoint_tol:
                    continue
                owner = cell_owner.get((cell[0] + dr, cell[1] + dc))
                if owner is not None and (best is None or d < best[1]):
                    best = (owner, d)
        return best[0] if best else None

    relations: list[LineRelation] = []
    palettes = sorted(set(int(v) for v in np.unique(frame)) - set(int(p) for p in bg_palettes))
    for p in palettes:
        ys, xs = np.where(frame == p)
        cells = list(zip(ys.tolist(), xs.tolist()))
        if len(cells) < min_inliers:
            continue
        r0, r1 = min(c[0] for c in cells), max(c[0] for c in cells)
        c0, c1 = min(c[1] for c in cells), max(c[1] for c in cells)
        bba = max(1, (r1 - r0 + 1) * (c1 - c0 + 1))
        # Skip dense/solid palettes: a connector is sparse over its bbox.
        if len(cells) > region_cell_cap or len(cells) / bba > region_fill_cap:
            continue
        remaining = set(cells)
        while len(remaining) >= min_inliers:
            pts = list(remaining)
            best = None
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    a, b = pts[i], pts[j]
                    if ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 < min_len:
                        continue
                    inl = [
                        q for q in pts
                        if _perp_dist(q, a, b) <= max_perp
                        and -0.02 <= _along(q, a, b) <= 1.02
                    ]
                    if best is None or len(inl) > len(best[2]):
                        best = (a, b, inl)
            if not best or len(best[2]) < min_inliers:
                break
            a, b, inl = best
            remaining -= set(inl)
            # True endpoints = inliers extreme along the a->b axis.
            inl_sorted = sorted(inl, key=lambda q: _along(q, a, b))
            ep_a, ep_b = inl_sorted[0], inl_sorted[-1]
            # CONTINUITY GATE: a real connector is populated ALONG its whole
            # length (dashes bridge the gap); reject sets that cluster only at
            # the two ends with an empty middle — those are two separate
            # endpoint blobs that merely happen to be collinear, not a line.
            seg = (((ep_a[0] - ep_b[0]) ** 2 + (ep_a[1] - ep_b[1]) ** 2) ** 0.5)
            proj = sorted(_along(q, ep_a, ep_b) * seg for q in inl)
            max_gap = max((proj[k + 1] - proj[k] for k in range(len(proj) - 1)),
                          default=0.0)
            if max_gap > max_gap_cells:
                continue
            ea, eb = nearest_entity(ep_a), nearest_entity(ep_b)
            # VALIDITY GATE: connects two DISTINCT entities.
            if ea is None or eb is None or ea == eb:
                continue
            through = sorted(
                {cell_owner[q] for q in inl if q in cell_owner} - {ea, eb}
            )
            relations.append(LineRelation(
                palette=p, cells=inl, endpoints=(ep_a, ep_b),
                entity_a=ea, entity_b=eb, through=through,
            ))
    return relations


if __name__ == "__main__":  # ground-truth validation on r11l + su15
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

    for gid in ("r11l", "su15"):
        g = load(gid)(); g.full_reset()
        fr = np.array(g.camera.render(g.current_level.get_sprites()))
        big = np.repeat(np.repeat(fr, 8, 0), 8, 1)
        rgb = np.zeros((*big.shape, 3), np.uint8)
        for i, (r, gg, b) in enumerate(PAL):
            rgb[big == i] = (r, gg, b)
        obs = build_frame_observation(rgb, turn=0, rows=64, cols=64, agent_position=None)
        ents = ed.detect_entities(obs)
        vals, cnts = np.unique(fr, return_counts=True)
        bg = [int(vals[np.argmax(cnts)])]
        rels = detect_line_relations(fr, ents, bg)
        ebb = {e.entity_id: e.bbox_logical for e in ents}
        print(f"\n=== {gid}: {len(rels)} connector line(s) ===")
        for L in rels:
            thru = f" through {[ebb[t][:2] for t in L.through]}" if L.through else ""
            print(f"  pal{L.palette} {len(L.cells)} cells: "
                  f"entity@{ebb[L.entity_a][:2]} <--> entity@{ebb[L.entity_b][:2]}{thru}")


# --------------------------------------------------------------------------- #
# Short SOLID connectors: a thin bridge of one palette directly linking two     #
# adjacent entities, distinct from the gap's local background.  The RANSAC      #
# line detector above targets long/dashed lines (min_inliers ~6); a legend's    #
# 1-cell-thick, ~3-cell-long tile-to-tile connector is too short for it, yet is #
# the same RELATION.  This complements it for the short-bridge regime.          #
# --------------------------------------------------------------------------- #
from collections import Counter as _Counter
from typing import NamedTuple as _NamedTuple


class Connector(_NamedTuple):
    a: int          # entity_id of one endpoint entity
    b: int          # entity_id of the other
    palette: int    # the bridge palette
    axis: str       # 'h' (horizontal bridge) or 'v' (vertical bridge)
    pos: int        # row (h) or col (v) the bridge runs along


def detect_connectors(frame: np.ndarray, entities):
    """Find solid connectors linking two entities.

    A connector is a single-palette line spanning the gap between two entities,
    whose palette DIFFERS from that gap's local-background (modal) palette — so it
    is seen even when the bridge shares a palette with some OTHER background region
    (the swallow-by-key failure).  No magic gap-length threshold: the gate is
    STRUCTURAL — the bridge must be a uniform run of one off-background palette
    spanning the WHOLE gap.  A too-wide or entity-interrupted gap fails this by
    construction (its rows aren't uniform), so a long genuine bridge still counts
    and a background-filled gap is rejected regardless of length.  Only top-level
    entities (not enclosed figures, not backgrounds) are paired.  Returns Connector
    edges.  No role names."""
    nodes = [e for e in entities
             if not getattr(e, "is_background_primary", False)
             and not getattr(e, "is_background_secondary", False)
             and getattr(e, "contained_in", None) is None]

    def bridge(a0, b0, axis):
        # a0/b0 bbox_logical = (y0,x0,y1,x1); a0 is the lower-coordinate side
        ay0, ax0, ay1, ax1 = a0
        by0, bx0, by1, bx1 = b0
        if axis == "h":
            gap = bx0 - ax1 - 1
            lo, hi = max(ay0, by0), min(ay1, by1)
            if gap < 1 or hi < lo:                   # must be a real gap with row overlap
                return None
            region = frame[lo:hi + 1, ax1 + 1:bx0]
            modal = _Counter(int(v) for v in region.flatten()).most_common(1)[0][0]
            for r in range(lo, hi + 1):
                seg = [int(v) for v in frame[r, ax1 + 1:bx0]]
                if seg and all(v == seg[0] for v in seg) and seg[0] != modal:
                    return (seg[0], r)
        else:
            gap = by0 - ay1 - 1
            lo, hi = max(ax0, bx0), min(ax1, bx1)
            if gap < 1 or hi < lo:
                return None
            region = frame[ay1 + 1:by0, lo:hi + 1]
            modal = _Counter(int(v) for v in region.flatten()).most_common(1)[0][0]
            for c in range(lo, hi + 1):
                seg = [int(v) for v in frame[ay1 + 1:by0, c]]
                if seg and all(v == seg[0] for v in seg) and seg[0] != modal:
                    return (seg[0], c)
        return None

    out = []
    for A in nodes:
        for B in nodes:
            if A.entity_id >= B.entity_id:
                continue
            ab, bb = A.bbox_logical, B.bbox_logical
            if bb[1] > ab[3]:                       # B to the right of A
                br = bridge(ab, bb, "h")
                if br:
                    out.append(Connector(A.entity_id, B.entity_id, br[0], "h", br[1]))
            elif bb[0] > ab[2]:                     # B below A
                br = bridge(ab, bb, "v")
                if br:
                    out.append(Connector(A.entity_id, B.entity_id, br[0], "v", br[1]))
    return out
