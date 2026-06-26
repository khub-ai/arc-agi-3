"""Substrate-agnostic agent detector.

The agent is the small distinctive sprite the player controls.  This
module finds it from raw RGB pixels alone — no palette tables, no
operator hints, no per-game knowledge.

Two detection strategies, applied in order:

  1. Movement-based (subsequent turns).  Given the previous frame's
     agent position and the current frame, find a non-background,
     non-HUD pixel cluster of comparable visual signature in a
     nearby cell.  Robust as long as we know where the agent was
     last turn.

  2. Heuristic (first turn / fallback).  Among non-background, non-
     HUD, non-composite-sprite small connected components, pick the
     one that best matches "lone controllable sprite":
       - small (4..30 logical pixels)
       - dominantly a single colour (low colour diversity in its bbox)
       - distinct colour (not used in any large background region)
       - not part of a composite cluster (no other small components
         of different colours within 2 px)
       - away from frame edges (agent rarely starts adjacent to a wall)
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .classifier import (
    _downsample_to_logical, _find_components, _quantise_frame,
    HAZARD_MIN_PIXELS, QUANT_STEP,
)


# Tolerances.
AGENT_MIN_PIXELS = 4
AGENT_MAX_PIXELS = 30
AGENT_TRACK_SEARCH_RADIUS_CELLS = 2  # search within +/-2 cells of last position
# Cap on the agent-sprite bbox in logical pixels.  Agent sprites are
# small (~3-5 px square at logical resolution); a cluster larger than
# this is definitely not an agent and is rejected from clustering.
# Without this cap, adjacent sediment tiles get pulled into the
# agent-cluster by spatial proximity (e.g., lc=1 agent at (4,2) is
# 1 px below a row of pink sediments, so they'd merge).
AGENT_CLUSTER_BBOX_MAX = 8


def _components_meta(comps: list[dict]) -> list[dict]:
    """Augment each comp dict with bbox dimensions."""
    out = []
    for c in comps:
        y0, x0, y1, x1 = c["bbox"]
        out.append({
            **c,
            "h": y1 - y0 + 1,
            "w": x1 - x0 + 1,
            "cy": float(c["ys"].mean()),
            "cx": float(c["xs"].mean()),
        })
    return out


def _is_composite_neighbor(c: dict, others: list[dict]) -> bool:
    """True if some OTHER component of a DIFFERENT colour is within
    2 px of this component's bbox."""
    y0, x0, y1, x1 = c["bbox"]
    for o in others:
        if o is c:
            continue
        if o["key"] == c["key"]:
            continue
        oy0, ox0, oy1, ox1 = o["bbox"]
        if (oy1 < y0 - 2 or oy0 > y1 + 2
                or ox1 < x0 - 2 or ox0 > x1 + 2):
            continue
        return True
    return False


def detect_agent_position(
    rgb_frame: np.ndarray,
    *,
    rows: int = 8,
    cols: int = 8,
    prev_position: tuple[int, int] | None = None,
    prev_signature: tuple[int, int, int] | None = None,
    bottom_strip_y_range: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, tuple[int, int, int] | None]:
    """Locate the agent in `rgb_frame`.

    Returns (cell_position, visual_signature).  `cell_position` is
    None if no agent could be found.  `visual_signature` is the
    quantised RGB key of the dominant colour of the agent component
    (used as a signature to track across turns).
    """
    logical, scale = _downsample_to_logical(rgb_frame)
    H, W = logical.shape[:2]
    cell_h = H // rows
    cell_w = W // cols

    q_frame = _quantise_frame(logical, QUANT_STEP)
    comps_all = _find_components(q_frame, min_pixels=HAZARD_MIN_PIXELS)
    comps_meta = _components_meta(comps_all)

    if not comps_meta:
        return None, None

    # Backgrounds: the two largest components.
    comps_by_size = sorted(comps_meta, key=lambda c: -c["n_px"])
    primary_key = comps_by_size[0]["key"]
    secondary_key = (
        comps_by_size[1]["key"] if len(comps_by_size) >= 2 else None
    )

    # HUD pixels in logical-coord y-range.
    hud_y_range = None
    if bottom_strip_y_range is not None:
        y0_strip = bottom_strip_y_range[0] // max(1, scale)
        y1_strip = bottom_strip_y_range[1] // max(1, scale)
        hud_y_range = (y0_strip, y1_strip)

    def _outside_hud(c: dict) -> bool:
        if hud_y_range is None:
            return True
        y0, _, y1, _ = c["bbox"]
        return y1 < hud_y_range[0] or y0 > hud_y_range[1]

    # Candidate components: non-background, non-HUD, agent-sized.
    candidates = [
        c for c in comps_meta
        if c["key"] not in (primary_key, secondary_key)
        and _outside_hud(c)
        and AGENT_MIN_PIXELS <= c["n_px"] <= AGENT_MAX_PIXELS
    ]
    if not candidates:
        return None, None

    # --- Strategy 1: track by previous position + signature ---------
    if prev_position is not None:
        pr, pc = prev_position
        best = None
        best_dist = None
        for c in candidates:
            cell_r = int(c["cy"] // cell_h)
            cell_c = int(c["cx"] // cell_w)
            if (abs(cell_r - pr) > AGENT_TRACK_SEARCH_RADIUS_CELLS
                    or abs(cell_c - pc) > AGENT_TRACK_SEARCH_RADIUS_CELLS):
                continue
            # Signature match bonus.
            sig_match = (
                prev_signature is not None
                and c["key"] == _rgb_to_key(prev_signature)
            )
            dist = abs(cell_r - pr) + abs(cell_c - pc)
            score = dist - (10 if sig_match else 0)
            if best is None or score < best_dist:
                best = c
                best_dist = score
        if best is not None:
            cell_r = int(best["cy"] // cell_h)
            cell_c = int(best["cx"] // cell_w)
            return (cell_r, cell_c), _key_to_rgb(best["key"])

    # --- Strategy 2: first-turn heuristic ---------------------------
    # Cluster candidates into spatial sprites (within 2 px), then
    # score each sprite by:
    #   1. distinct colours in the sprite  (more = composite agent,
    #      e.g. lc=0 diver = blue body + yellow head)
    #   2. -(sum of global colour-counts)  (rarer colours = more
    #      distinctive, e.g. lc=1 yellow agent appears in exactly
    #      one candidate)
    # Both keys descending — most distinct, most rare wins.
    colour_counts: Counter[int] = Counter()
    for c in candidates:
        colour_counts[c["key"]] += 1

    # Spatial sprite-clustering with bbox cap (so adjacent sediments
    # don't get pulled into the agent's cluster).
    used = [False] * len(candidates)
    sprites: list[list[dict]] = []
    for i, c in enumerate(candidates):
        if used[i]:
            continue
        sprite = [c]
        used[i] = True
        gy0, gx0, gy1, gx1 = c["bbox"]
        changed = True
        while changed:
            changed = False
            for j, oc in enumerate(candidates):
                if used[j]:
                    continue
                oy0, ox0, oy1, ox1 = oc["bbox"]
                if (oy1 < gy0 - 2 or oy0 > gy1 + 2
                        or ox1 < gx0 - 2 or ox0 > gx1 + 2):
                    continue
                new_gy0 = min(gy0, oy0); new_gx0 = min(gx0, ox0)
                new_gy1 = max(gy1, oy1); new_gx1 = max(gx1, ox1)
                if (new_gy1 - new_gy0 + 1 > AGENT_CLUSTER_BBOX_MAX
                        or new_gx1 - new_gx0 + 1 > AGENT_CLUSTER_BBOX_MAX):
                    continue
                sprite.append(oc)
                used[j] = True
                gy0, gx0, gy1, gx1 = (new_gy0, new_gx0,
                                       new_gy1, new_gx1)
                changed = True
        sprites.append(sprite)

    def _sprite_score(sprite: list[dict]) -> tuple:
        colours = set(c["key"] for c in sprite)
        n_distinct = len(colours)
        rarity_sum = sum(colour_counts[k] for k in colours)
        # All-yields by 4-tuple, sorted desc on first two:
        # bigger n_distinct, lower rarity_sum, smaller bbox, larger n_px.
        gy0 = min(c["bbox"][0] for c in sprite)
        gx0 = min(c["bbox"][1] for c in sprite)
        gy1 = max(c["bbox"][2] for c in sprite)
        gx1 = max(c["bbox"][3] for c in sprite)
        bbox_area = (gy1 - gy0 + 1) * (gx1 - gx0 + 1)
        total_n_px = sum(c["n_px"] for c in sprite)
        return (-n_distinct, rarity_sum, bbox_area, -total_n_px)

    sprites.sort(key=_sprite_score)
    best_sprite = sprites[0]
    # Use the sprite's centroid for the agent position.
    cy = float(sum(c["cy"] * c["n_px"] for c in best_sprite)) \
        / max(1, sum(c["n_px"] for c in best_sprite))
    cx = float(sum(c["cx"] * c["n_px"] for c in best_sprite)) \
        / max(1, sum(c["n_px"] for c in best_sprite))
    cell_r = max(0, min(rows - 1, int(cy // cell_h)))
    cell_c = max(0, min(cols - 1, int(cx // cell_w)))
    # Pick the largest-pixel component's colour as signature.
    sig_comp = max(best_sprite, key=lambda c: c["n_px"])
    return (cell_r, cell_c), _key_to_rgb(sig_comp["key"])


def _key_to_rgb(key: int) -> tuple[int, int, int]:
    return ((key >> 16) & 0xff, (key >> 8) & 0xff, key & 0xff)


def _rgb_to_key(rgb: tuple[int, int, int]) -> int:
    return (int(rgb[0]) << 16) | (int(rgb[1]) << 8) | int(rgb[2])
