"""Cell classifier — produces per-cell codes from Observations.

Substrate-agnostic.  Reads RGB pixels and structural features of the
frame; never reads palette indices, never reads truth.json.

Approach: find connected components in the quantised-RGB frame and
classify each component by structural properties — size, color
diversity, position.  Then assign each cell to a code based on which
component(s) overlap it.

Component-class rules (in substrate-agnostic vocabulary):

  BACKGROUND_PRIMARY    — the largest static connected region.
                          Cells primarily inside it -> W.
  BACKGROUND_SECONDARY  — the second-largest connected region of a
                          different colour.  Cells primarily inside
                          it -> B.
  COMPOSITE_SPRITE      — a tight cluster of pixels containing at
                          least three distinct colours within a small
                          bbox.  Hazards (ghost: eye-dots + body
                          + yellow eye) match this pattern.
                          Cells overlapping its bbox -> H.
  SOLID_TILE            — a single-colour connected region whose colour
                          is distinct from BACKGROUND_PRIMARY and
                          BACKGROUND_SECONDARY, of small-to-medium
                          size.  Pink-sediment tiles match.  Centroid
                          cell -> P.
  BOTTOM_STRIP          — a thin horizontal strip pinned to the bottom
                          edge of the frame, content differing from
                          the surrounding background.  Cells in its
                          y-range -> U.
  AGENT                 — the cell at observation.agent_position -> A.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List

import numpy as np

from .observation import FrameObservation, CellObservation


# -----------------------------------------------------------------------------


@dataclass
class Classification:
    turn: int
    rows: int
    cols: int
    cell_codes: list[list[str]]


# Quantisation step for RGB colour bucketing.  16 means each channel
# rounded to multiples of 16, giving ~16^3 buckets in principle but
# fewer in practice (the harness palette is small).
QUANT_STEP = 16

# Minimum component size (raw pixels) for general detection (sediment,
# background fill).  Hazard detection uses a lower floor so the small
# multi-colour "mouth" pixels of a composite sprite still count as
# evidence — those are individually 1-2 pixels but cluster spatially.
MIN_COMPONENT_PIXELS = 3
HAZARD_MIN_PIXELS = 1

# Composite-sprite bbox cap (raw pixels).  Ghost sprites in lc=1
# fit in a 5x5 region; anything larger is more likely a background.
COMPOSITE_BBOX_MAX = 8


# -----------------------------------------------------------------------------
# Connected-component utilities (4-connected, on a boolean mask).
# -----------------------------------------------------------------------------


def _connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros_like(mask, dtype=np.int32)
    n = 0
    rows, cols = mask.shape
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or labels[r, c]:
                continue
            n += 1
            stack = [(r, c)]
            while stack:
                rr, cc = stack.pop()
                if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if not mask[rr, cc] or labels[rr, cc]:
                    continue
                labels[rr, cc] = n
                stack.extend([(rr - 1, cc), (rr + 1, cc),
                              (rr, cc - 1), (rr, cc + 1)])
    return labels, n


def _quantise_frame(rgb: np.ndarray, step: int) -> np.ndarray:
    """Quantise (H, W, 3) RGB to (H, W) integer keys by binning each
    channel to multiples of `step`.  Two pixels with the same key are
    treated as the same colour."""
    q = (rgb.astype(np.int32) // step) * step
    return q[:, :, 0] * (256 * 256) + q[:, :, 1] * 256 + q[:, :, 2]


# -----------------------------------------------------------------------------
# Component discovery + classification.
# -----------------------------------------------------------------------------


def _find_components(
    q_frame: np.ndarray,
    *,
    min_pixels: int = MIN_COMPONENT_PIXELS,
) -> list[dict]:
    """Return a list of components found in the quantised frame.

    Each component dict has:
      key     : the quantised colour key
      ys      : ndarray of pixel y coords
      xs      : ndarray of pixel x coords
      n_px    : count
      bbox    : (y0, x0, y1, x1) inclusive
    """
    comps: list[dict] = []
    unique_keys = np.unique(q_frame)
    for key in unique_keys:
        mask = (q_frame == key)
        if not mask.any():
            continue
        labels, n = _connected_components(mask)
        for ci in range(1, n + 1):
            ys, xs = np.where(labels == ci)
            if ys.size < min_pixels:
                continue
            comps.append({
                "key": int(key),
                "ys": ys, "xs": xs,
                "n_px": int(ys.size),
                "bbox": (int(ys.min()), int(xs.min()),
                          int(ys.max()), int(xs.max())),
            })
    return comps


def _components_in_bbox(
    components: list[dict],
    y0: int, x0: int, y1: int, x1: int,
) -> list[dict]:
    """Return components whose pixels overlap the (inclusive) bbox."""
    out = []
    for c in components:
        cy0, cx0, cy1, cx1 = c["bbox"]
        if cy1 < y0 or cy0 > y1 or cx1 < x0 or cx0 > x1:
            continue
        # Actually overlap check — bbox-touching is enough for sprites.
        out.append(c)
    return out


def _cell_pixel_bbox(
    r: int, c: int, cell_h: int, cell_w: int,
) -> tuple[int, int, int, int]:
    return (r * cell_h, c * cell_w,
            (r + 1) * cell_h - 1, (c + 1) * cell_w - 1)


# -----------------------------------------------------------------------------


def _downsample_to_logical(rgb_frame: np.ndarray) -> tuple[np.ndarray, int]:
    """Detect the harness's display-upscaling factor and downsample
    the frame to its logical resolution.

    Many harnesses (including ARC-AGI-3) render a coarse logical
    frame at an integer upscale.  Component-size thresholds are
    natural in logical-pixel units (a sediment tile is "a 5x5
    sprite"), not display-pixel units (which scale with upscale).

    Heuristic: find the largest integer S such that every SxS block
    of pixels in the frame is uniform colour.  Downsample by S.
    """
    H, W = rgb_frame.shape[:2]
    # Try common integer scales in descending order.
    for scale in (16, 8, 4, 2, 1):
        if H % scale or W % scale:
            continue
        # Quick check: do the first few blocks look uniform?
        ok = True
        for by in range(0, min(H, scale * 4), scale):
            for bx in range(0, min(W, scale * 4), scale):
                block = rgb_frame[by:by + scale, bx:bx + scale, :]
                if not np.all(block == block[0, 0]):
                    ok = False
                    break
            if not ok:
                break
        if ok:
            new_h = H // scale
            new_w = W // scale
            ys = np.arange(new_h) * scale + scale // 2
            xs = np.arange(new_w) * scale + scale // 2
            return rgb_frame[np.ix_(ys, xs)], scale
    return rgb_frame, 1


def classify(
    obs: FrameObservation,
) -> Classification:
    rows, cols = obs.rows, obs.cols
    H_disp, W_disp = obs.rgb_frame.shape[:2]
    codes: list[list[str]] = [["?"] * cols for _ in range(rows)]

    # --- 1. Agent: known from harness ----------------------------------------
    if obs.agent_position is not None:
        ar, ac = obs.agent_position
        if 0 <= ar < rows and 0 <= ac < cols:
            codes[ar][ac] = "A"

    # --- 2. Downsample to logical resolution + quantise + find components ---
    # Component-size thresholds below are in LOGICAL pixels (the units
    # the harness actually renders at), so they don't depend on the
    # display-upscale factor.
    logical_frame, scale = _downsample_to_logical(obs.rgb_frame)
    H, W = logical_frame.shape[:2]
    cell_h = H // rows
    cell_w = W // cols
    q_frame = _quantise_frame(logical_frame, QUANT_STEP)
    comps = _find_components(q_frame)
    # Hazards need 1-px component sensitivity (composite-sprite
    # "mouth" pixels are individually tiny but cluster spatially).
    hazard_comps = _find_components(q_frame, min_pixels=HAZARD_MIN_PIXELS)
    if not comps:
        return Classification(turn=obs.turn, rows=rows, cols=cols,
                               cell_codes=codes)

    # Identify two largest components by pixel count — these are the
    # primary and secondary backgrounds.
    comps_by_size = sorted(comps, key=lambda c: -c["n_px"])
    primary = comps_by_size[0]
    secondary = comps_by_size[1] if len(comps_by_size) >= 2 else None
    primary_key = primary["key"]
    secondary_key = secondary["key"] if secondary else None

    # --- 3. Composite sprites -> H -------------------------------------------
    # This MUST run before the bottom-strip pass: the HUD strip is a
    # 1-raw-pixel-tall horizontal band, but cell row 7 covers 8 raw
    # rows.  When a hazard sprite occupies the upper portion of cell
    # row 7 (raw rows ~56-60) and the HUD occupies the bottom (raw
    # row 63), the cell semantically belongs to the hazard sprite,
    # not the HUD.  Claiming H first preserves that.
    # Heuristic: find spatial clusters of small components whose colours
    # are NEITHER primary NOR secondary background AND whose combined
    # bbox is small (sprite-sized).  Group neighbouring small comps into
    # sprites by bbox proximity.
    small_non_bg = [
        c for c in hazard_comps
        if c["key"] not in (primary_key, secondary_key)
        and c["n_px"] <= 30
    ]

    # Spatial-cluster small non-background components into candidate
    # sprites.  Use the WALL-DECORATION-RESISTANT rule: a new component
    # joins a group only if it is within 2 px of EXISTING members AND
    # the resulting combined bbox stays within COMPOSITE_BBOX_MAX in
    # both dimensions.  This prevents chains of wall-decoration dots
    # from bridging two unrelated sprites into one mega-group.
    sprite_groups: list[list[dict]] = []
    used = [False] * len(small_non_bg)
    # 1-px adjacency (was 2 px).  Tighter so single-pixel wall-decoration
    # dots scattered through the wall background at 3-6 px spacing don't
    # accidentally bridge into a sprite's group when they happen to land
    # 2 px away from a body pixel.
    PROXIMITY = 1
    for i, c in enumerate(small_non_bg):
        if used[i]:
            continue
        group = [c]
        used[i] = True
        gy0, gx0, gy1, gx1 = c["bbox"]
        changed = True
        while changed:
            changed = False
            for j, oc in enumerate(small_non_bg):
                if used[j]:
                    continue
                oy0, ox0, oy1, ox1 = oc["bbox"]
                if (oy1 < gy0 - PROXIMITY or oy0 > gy1 + PROXIMITY
                        or ox1 < gx0 - PROXIMITY or ox0 > gx1 + PROXIMITY):
                    continue
                new_gy0 = min(gy0, oy0)
                new_gx0 = min(gx0, ox0)
                new_gy1 = max(gy1, oy1)
                new_gx1 = max(gx1, ox1)
                # Reject if joining would overflow the sprite cap.
                if (new_gy1 - new_gy0 + 1 > COMPOSITE_BBOX_MAX
                        or new_gx1 - new_gx0 + 1 > COMPOSITE_BBOX_MAX):
                    continue
                group.append(oc)
                used[j] = True
                gy0, gx0, gy1, gx1 = (new_gy0, new_gx0,
                                       new_gy1, new_gx1)
                changed = True
        sprite_groups.append(group)

    for group in sprite_groups:
        gy0 = min(g["bbox"][0] for g in group)
        gx0 = min(g["bbox"][1] for g in group)
        gy1 = max(g["bbox"][2] for g in group)
        gx1 = max(g["bbox"][3] for g in group)
        bbox_h = gy1 - gy0 + 1
        bbox_w = gx1 - gx0 + 1
        if bbox_h > COMPOSITE_BBOX_MAX or bbox_w > COMPOSITE_BBOX_MAX:
            continue
        distinct_colors = len({g["key"] for g in group})
        r0 = max(0, gy0 // cell_h)
        r1 = min(rows - 1, gy1 // cell_h)
        c0 = max(0, gx0 // cell_w)
        c1 = min(cols - 1, gx1 // cell_w)
        # Three target-code cases:
        #   * sprite contains the harness-given agent_position -> A
        #     (regardless of color count; lc=0's diver is 2-color
        #     blue-body+yellow-head, lc=1's agent is 1-color yellow)
        #   * non-agent sprite with >=3 distinct colors -> H
        #     (composite hazard signature: ghost = body + eyes +
        #     pupils, three different colors in a small bbox)
        #   * otherwise -> leave for the solid-tile pass to classify
        sprite_contains_agent = False
        if obs.agent_position is not None:
            ar, ac = obs.agent_position
            sprite_contains_agent = (
                r0 <= ar <= r1 and c0 <= ac <= c1
            )
        if sprite_contains_agent:
            # Agent uses CENTROID-CELL ONLY marking, matching the
            # offline labeler convention.  The agent sprite may
            # protrude into adjacent cells (e.g. lc=1's small yellow
            # body straddles the row-4/row-5 boundary), but the
            # canonical "agent cell" is where the cluster's pixel-
            # weighted centroid lands — usually a single cell.
            total_n = sum(g["n_px"] for g in group)
            cy = sum(g["ys"].mean() * g["n_px"] for g in group) / max(1, total_n)
            cx = sum(g["xs"].mean() * g["n_px"] for g in group) / max(1, total_n)
            cell_r = max(0, min(rows - 1, int(cy // cell_h)))
            cell_c = max(0, min(cols - 1, int(cx // cell_w)))
            if codes[cell_r][cell_c] in ("?", "A"):
                codes[cell_r][cell_c] = "A"
            continue
        if distinct_colors >= 3:
            target_code = "H"
        else:
            continue
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if codes[r][c] in ("?", "A"):
                    codes[r][c] = target_code

    # --- 4. Bottom strip -> U (cells the HUD strip touches that aren't
    #         already a hazard).  observation.bottom_strip_rows are in
    #         DISPLAY coords; convert to logical coords by dividing by
    #         scale.
    if obs.bottom_strip_rows is not None:
        y0_strip_logical = obs.bottom_strip_rows[0] // max(1, scale)
        y1_strip_logical = obs.bottom_strip_rows[1] // max(1, scale)
        for r in range(rows):
            for c in range(cols):
                if codes[r][c] != "?":
                    continue
                y0_cell, x0_cell, y1_cell, x1_cell = \
                    _cell_pixel_bbox(r, c, cell_h, cell_w)
                if not (y1_cell < y0_strip_logical
                        or y0_cell > y1_strip_logical):
                    codes[r][c] = "U"

    # --- 5. Sediment tiles -> P/G  AND  win marker -> X --------------------
    # A small non-background single-colour component can be one of:
    #   * a SOLID sediment tile     (high fill_ratio, square/circle shape)
    #   * a SPECIAL marker          (low fill_ratio: cross, star, etc.)
    #
    # We distinguish by fill_ratio of the bbox (a STRUCTURAL shape cue):
    #   fill_ratio >= 0.70  ->  solid block  -> P or G
    #   fill_ratio  < 0.70  ->  special shape -> X (win marker)
    #
    # Within "solid block", the two tile classes (P, G) are separated by
    # COLOUR IDENTITY *relationally* -- never by a hardcoded channel.  The
    # solid sprites are clustered by their own colour; the clusters are ranked
    # by total area (then sprite count, then top-left position), and the
    # symbols are bound to the ranks (largest cluster -> P, the rest -> G).
    # A bijective re-skin permutes colours but preserves both the clustering
    # AND that ranking (area/count/position are colour-independent), so the
    # split is palette-invariant (Adversarial Test / P11).  The old code
    # branched on "green channel dominates", which recognised the skin and
    # broke on a recolour -- contradicting this module's own no-palette claim.
    #
    # NOTE (scope): this is the LEGACY offline fixture-accuracy classifier
    # (run.py / sequence.py), NOT the live exploratory_driver play path.  Its
    # W/B/G/P/X/H/U/A codes are a fixture-specific evaluation vocabulary; the
    # live perception path keys on the palette-invariant primitives in
    # silhouette_track.foreground_components instead.

    # First pass: assign X by shape; gather solid sprites + per-colour cluster
    # stats (all colour-independent, so the ranking survives a re-skin).
    _solid_sprites: list[tuple] = []                # (component, cr, cc)
    _clusters: dict[int, dict] = {}                 # colour key -> stats
    for c in comps:
        if c["key"] in (primary_key, secondary_key):
            continue
        if c["n_px"] < 4 or c["n_px"] > 50:
            continue
        y0, x0, y1, x1 = c["bbox"]
        bh = y1 - y0 + 1
        bw = x1 - x0 + 1
        if bh > 8 or bw > 8:
            continue
        fill_ratio = c["n_px"] / max(1, bh * bw)
        if fill_ratio < 0.5:
            continue   # too sparse — wall-decoration dot pattern
        cr = max(0, min(rows - 1, int(c["ys"].mean() // cell_h)))
        cc = max(0, min(cols - 1, int(c["xs"].mean() // cell_w)))
        if codes[cr][cc] != "?":
            continue
        if fill_ratio < 0.70:
            # Special-shape sprite (cross / star / asymmetric blob)
            # — most likely a win marker.  Shape-based, palette-free.
            codes[cr][cc] = "X"
            continue
        _solid_sprites.append((c, cr, cc))
        st = _clusters.setdefault(c["key"], {"area": 0, "count": 0, "anchor": (cr, cc)})
        st["area"] += int(c["n_px"])
        st["count"] += 1
        if (cr, cc) < st["anchor"]:
            st["anchor"] = (cr, cc)

    # Rank colour clusters by (area desc, count desc, top-left position) — all
    # colour-independent quantities — and bind symbols to ranks, not to hue.
    _ranked = sorted(
        _clusters,
        key=lambda k: (-_clusters[k]["area"], -_clusters[k]["count"], _clusters[k]["anchor"]),
    )
    _primary_tile = _ranked[0] if _ranked else None
    for c, cr, cc in _solid_sprites:
        codes[cr][cc] = "P" if c["key"] == _primary_tile else "G"

    # --- 6. Background fill: per-cell dominant quantised colour -------------
    # Cells still '?': assign W if their dominant quantised colour is
    # primary, B if it's secondary, else look at adjacency.
    for r in range(rows):
        for c in range(cols):
            if codes[r][c] != "?":
                continue
            y0 = r * cell_h
            y1 = (r + 1) * cell_h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w
            patch_q = q_frame[y0:y1, x0:x1]
            vals, counts = np.unique(patch_q, return_counts=True)
            dom = int(vals[counts.argmax()])
            if dom == primary_key:
                codes[r][c] = "W"
            elif dom == secondary_key:
                codes[r][c] = "B"
            else:
                # Could be a sediment whose patch happens to have a
                # background dot dominant — fall through to whichever
                # of primary/secondary is more frequent.
                n_primary = int(
                    np.sum(patch_q == primary_key)
                ) if primary_key is not None else 0
                n_secondary = int(
                    np.sum(patch_q == secondary_key)
                ) if secondary_key is not None else 0
                codes[r][c] = "W" if n_primary >= n_secondary else "B"

    return Classification(turn=obs.turn, rows=rows, cols=cols,
                          cell_codes=codes)
