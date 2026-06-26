"""Pattern detector — emits rule candidates from per-turn classifier
runs.

The detector inspects what the classifier inferred this turn (the
identified W/B colours, the bottom-strip y-range, the composite-
sprite signatures) and proposes rule candidates in substrate-
agnostic vocabulary.

The detector NEVER reads truth.json or the operator-relabeler tool.
Its inputs are the FrameObservation and the per-turn Classification
output.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict

import numpy as np

from .classifier import (
    _downsample_to_logical, _find_components, _quantise_frame,
    QUANT_STEP, HAZARD_MIN_PIXELS,
)
from .observation import FrameObservation
from .rules import (
    Candidate, ColorBindingBody, CompositeSpriteBody, HudStripBody,
)


def detect_for_turn(
    obs: FrameObservation,
    cell_codes: list[list[str]],
) -> list[Candidate]:
    """Produce rule candidates from a single turn's observation +
    classification output.

    The returned candidates are uncommitted; the aggregator decides
    when to elevate them to sandbox / trial / established.
    """
    candidates: list[Candidate] = []
    rgb_logical, scale = _downsample_to_logical(obs.rgb_frame)
    q_frame = _quantise_frame(rgb_logical, QUANT_STEP)
    comps = _find_components(q_frame)
    hazard_comps = _find_components(q_frame, min_pixels=HAZARD_MIN_PIXELS)

    if not comps:
        return candidates

    comps_by_size = sorted(comps, key=lambda c: -c["n_px"])
    primary = comps_by_size[0]
    secondary = comps_by_size[1] if len(comps_by_size) >= 2 else None
    primary_key = primary["key"]
    secondary_key = secondary["key"] if secondary else None

    # --- 1. Color-binding candidates ----------------------------------------
    candidates.append(Candidate(
        type="color_binding",
        body=asdict(ColorBindingBody(
            role="W",
            rgb_key=_key_to_rgb(primary_key),
            structural_rank="primary",
        )),
        supporting_turns=[obs.turn],
    ))
    if secondary_key is not None:
        candidates.append(Candidate(
            type="color_binding",
            body=asdict(ColorBindingBody(
                role="B",
                rgb_key=_key_to_rgb(secondary_key),
                structural_rank="secondary",
            )),
            supporting_turns=[obs.turn],
        ))

    # --- 2. HUD strip candidate ---------------------------------------------
    if obs.bottom_strip_rows is not None:
        y0_disp, y1_disp = obs.bottom_strip_rows
        y0_log = y0_disp // max(1, scale)
        y1_log = y1_disp // max(1, scale)
        # Dominant colour in the strip (logical pixels).
        strip = q_frame[y0_log:y1_log + 1, :]
        if strip.size:
            vals, counts = np.unique(strip, return_counts=True)
            dom = int(vals[counts.argmax()])
            candidates.append(Candidate(
                type="hud_strip",
                body=asdict(HudStripBody(
                    y_range_logical=(y0_log, y1_log),
                    edge="bottom",
                    dominant_rgb_key=_key_to_rgb(dom),
                )),
                supporting_turns=[obs.turn],
            ))

    # --- 3. Composite-sprite signature for H -------------------------------
    # Aggregate every composite-sprite cluster the classifier would have
    # found this turn into a SINGLE candidate carrying the union of
    # colours observed in those clusters (and the max bbox dims).
    color_set: set[tuple[int, int, int]] = set()
    max_h = 0
    max_w = 0
    # Re-run sprite grouping to capture composite signatures
    # (same logic as the classifier, but we look at the result rather
    # than mark cells).
    small_non_bg = [
        c for c in hazard_comps
        if c["key"] not in (primary_key, secondary_key)
        and c["n_px"] <= 30
    ]
    from .classifier import COMPOSITE_BBOX_MAX
    used = [False] * len(small_non_bg)
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
                if (oy1 < gy0 - 2 or oy0 > gy1 + 2
                        or ox1 < gx0 - 2 or ox0 > gx1 + 2):
                    continue
                new_gy0 = min(gy0, oy0); new_gx0 = min(gx0, ox0)
                new_gy1 = max(gy1, oy1); new_gx1 = max(gx1, ox1)
                if (new_gy1 - new_gy0 + 1 > COMPOSITE_BBOX_MAX
                        or new_gx1 - new_gx0 + 1 > COMPOSITE_BBOX_MAX):
                    continue
                group.append(oc)
                used[j] = True
                gy0, gx0, gy1, gx1 = (new_gy0, new_gx0, new_gy1, new_gx1)
                changed = True
        colors = {_key_to_rgb(g["key"]) for g in group}
        if len(colors) >= 3:
            bbox_h = gy1 - gy0 + 1
            bbox_w = gx1 - gx0 + 1
            color_set |= colors
            max_h = max(max_h, bbox_h)
            max_w = max(max_w, bbox_w)
    if color_set:
        candidates.append(Candidate(
            type="composite_sprite",
            body=asdict(CompositeSpriteBody(
                role="H",
                color_set=sorted(color_set),
                min_distinct_colors=3,
                bbox_max_logical=(max_h, max_w),
            )),
            supporting_turns=[obs.turn],
        ))

    return candidates


def _key_to_rgb(key: int) -> tuple[int, int, int]:
    return ((key >> 16) & 0xff, (key >> 8) & 0xff, key & 0xff)
