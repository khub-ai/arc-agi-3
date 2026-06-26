"""Substrate-agnostic entity detector.

Single job: turn an RGB frame into a list of entities.  No role
names, no game-specific knowledge, no visual heuristics for
"agent looks like X" or "win marker looks like cross".  Just
geometry — connected components, sprite groups, bboxes.

This is Layer A of SPEC_perception_module.md: geometry only.
Role assignment happens downstream in role_resolver.py, which
reads from the knowledge base (per-game data, not code) and
applies behavior-grounded matchers.

The detector tags each entity with its visual signature
(quantized-RGB histogram) so downstream matchers can compare
entities across frames without re-running the pixel-level
analysis.

No new code is needed here when a new game ships.  The detector
runs identically on every game in the corpus.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .observation import FrameObservation


# Quantisation step for RGB colour bucketing — same as the legacy
# classifier so downstream code that compares signatures (e.g. the
# validator's template store) stays compatible.
QUANT_STEP = 16

# Sprite-grouping parameters.  These describe geometry, not any
# specific game's sprites, so they don't need per-game tuning.
PROXIMITY = 1                # max gap between component bboxes —
                              # composite sprites (ghost body+eye+mouth)
                              # often have 1-pixel gaps between parts.
BBOX_MAX = 8                 # max logical-pixel extent of one sprite
MIN_SPRITE_PIXELS = 4        # below this, a sprite is wall-decoration
                              # speckle, not an entity worth marking
MIN_COMPONENT_PIXELS = 1     # the component pass itself is permissive;
                              # the sprite-group filter does the
                              # noise rejection
MIN_GROUP_DENSITY = 0.40     # reject a candidate join if the resulting
                              # group's pixel-density in its bbox drops
                              # below this threshold.  A real sprite
                              # has 40-100% density; absorbing wall-
                              # speckle drops density toward zero.
                              # Substrate-agnostic: it's a property of
                              # \"compact entities are dense\".


@dataclass
class Entity:
    """A detected entity in a frame.  Substrate-agnostic — no role
    name yet.  The role_resolver assigns one later.
    """

    entity_id: int               # 0-based index, unique within frame
    bbox_logical: tuple[int, int, int, int]    # (y0, x0, y1, x1) in logical px
    bbox_display: tuple[int, int, int, int]    # same bbox upscaled to display px
    n_pixels: int                # total component-pixel count
    n_distinct_colors: int       # palette diversity inside the bbox
    visual_signature: tuple      # top-K (quantized_rgb_key, fraction) pairs
    centroid_logical: tuple[float, float]      # (y, x) in logical-px space
    is_background_primary: bool = False        # set by background pass
    is_background_secondary: bool = False
    # Cell coverage — which cells does this entity touch.
    # Always populated for both sprites and backgrounds with every
    # cell that contains at least one of the entity's pixels.
    # Matchers use this for "does this entity overlap cell X" queries.
    cells: list[tuple[int, int]] = field(default_factory=list)
    # Centroid cell — for foreground sprites, the cell containing
    # the pixel-weighted centroid.  This is the "primary" cell for
    # compact sprites (agent, sediment); large entities like HUD
    # strips have a centroid but downstream consumers usually
    # project them onto all touched cells instead.
    centroid_cell: tuple[int, int] = (0, 0)
    # Per-cell pixel count.  Used by the cell-grid projector for
    # bg tie-breaking (smallest bg bbox wins) and by sprite
    # projection modes that need it.
    pixel_count_per_cell: dict = field(default_factory=dict)
    # Region-vs-object classification, computed on the CELLS mask (not the
    # pixel bbox).  bbox_fit = touched_cells / bbox_cell_area: an irregular
    # palette REGION (floor/wall, hollow/diagonal shape) fills only a fraction
    # of its bbox; a compact OBJECT fills it.  is_region marks a LARGE,
    # low-fill, single-palette-dominated area whose bounding rectangle is NOT a
    # meaningful object box — downstream should describe it by palette + cells,
    # never as a rectangle.  Operating on cells (not pixels) is the point: a
    # diagonal diamond or hollow outline is judged by the shape it actually
    # occupies, not the box that misleadingly encloses it.
    bbox_fit: float = 1.0
    is_region: bool = False
    # VLM-assigned gestalt label, e.g. "diamond_outline", "ring", "dotted_line".
    # Empty for CV-detected entities; set when the VLM groups orphan fragments
    # that no connectivity rule could unify (see gestalt_grouping.py).  The
    # substrate records the label; it does NOT classify the shape itself.
    shape_type: str = ""
    is_gestalt: bool = False      # True when this entity was VLM-grouped
    # entity_id of the container this entity sits INSIDE (a coloured tile/frame
    # enclosing it), or None.  Lets a figure-in-a-frame (a glyph on a tile) be
    # discovered as its own entity without losing the containment relation.
    contained_in: Optional[int] = None
    # Background-removed COLOURED MASK in cell coordinates (the entity's own pixels
    # carry their colour key; every other cell in the bbox is -1).  This is the input
    # the perceptual-equivalence engine canonicalises for pose-factored identity and
    # tracking (score_association).  None for backgrounds and VLM-grouped gestalts.
    bitmap: Optional[np.ndarray] = None
    # Palette of the local background this entity sits ON — the panel/bar for a
    # top-level sprite, the enclosing tile for a contained figure.  It is the
    # CONTEXT: two same-shaped glyphs on different-coloured backgrounds (a cyan-role
    # vs a pink-role glyph) are distinct symbols, so downstream typing keeps their
    # symbol-spaces separate by context_bg.
    context_bg: Optional[int] = None


# -----------------------------------------------------------------------------
# Connected-component utilities — copied from classifier.py to keep
# entity_detector standalone.  The legacy classifier will be removed
# once the knowledge-driven path is verified.
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
    q = (rgb.astype(np.int32) // step) * step
    return q[:, :, 0] * (256 * 256) + q[:, :, 1] * 256 + q[:, :, 2]


def _downsample_to_logical(rgb_frame: np.ndarray) -> tuple[np.ndarray, int]:
    H, W = rgb_frame.shape[:2]
    for scale in (16, 8, 4, 2, 1):
        if H % scale or W % scale:
            continue
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


def _find_components(
    q_frame: np.ndarray,
    *,
    min_pixels: int = MIN_COMPONENT_PIXELS,
) -> list[dict]:
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


def _signature_from_pixels(
    logical: np.ndarray, ys: np.ndarray, xs: np.ndarray,
) -> tuple:
    """Top-K quantized-RGB signature for the pixels of an entity.
    Same QUANT_STEP as the detector so signatures are commensurable
    across frames.
    """
    if ys.size == 0:
        return tuple()
    patch = logical[ys, xs]
    q = (patch.astype(np.int32) // QUANT_STEP) * QUANT_STEP
    keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
    unique, counts = np.unique(keys, return_counts=True)
    order = np.argsort(-counts)
    total = float(keys.size)
    top: list[tuple[int, float]] = []
    for idx in order[:5]:
        top.append((int(unique[idx]), float(counts[idx] / total)))
    return tuple(top)


# -----------------------------------------------------------------------------
# Sprite grouping — neighbouring small components combine into a single
# entity.  Substrate-agnostic: composition is bounded by BBOX_MAX
# regardless of what the component represents.
# -----------------------------------------------------------------------------


def _group_components_into_sprites(
    components: list[dict],
    bg_keys: set,
) -> list[list[dict]]:
    """Group non-background components by spatial proximity.

    Returns a list of sprite groups (each a list of constituent
    components).  The group's bbox is the union of constituent bboxes
    capped at BBOX_MAX.  Groups smaller than MIN_SPRITE_PIXELS in
    total pixel count are dropped — those are wall-decoration speckle.
    """
    fg = [c for c in components if c["key"] not in bg_keys]
    used = [False] * len(fg)
    groups: list[list[dict]] = []
    for i, c in enumerate(fg):
        if used[i]:
            continue
        used[i] = True
        members = [c]
        gy0, gx0, gy1, gx1 = c["bbox"]
        total_px = c["n_px"]
        changed = True
        while changed:
            changed = False
            for j, oc in enumerate(fg):
                if used[j]:
                    continue
                oy0, ox0, oy1, ox1 = oc["bbox"]
                if (oy1 < gy0 - PROXIMITY or oy0 > gy1 + PROXIMITY
                        or ox1 < gx0 - PROXIMITY or ox0 > gx1 + PROXIMITY):
                    continue
                ny0 = min(gy0, oy0); ny1 = max(gy1, oy1)
                nx0 = min(gx0, ox0); nx1 = max(gx1, ox1)
                if (ny1 - ny0 + 1 > BBOX_MAX
                        or nx1 - nx0 + 1 > BBOX_MAX):
                    continue
                # Density gate: after joining, the group's pixel
                # density in its bbox must stay >= MIN_GROUP_DENSITY.
                # Rejects merging speckle (1-2 px) into compact
                # sprites where the resulting bbox would be mostly
                # empty space.
                new_area = (ny1 - ny0 + 1) * (nx1 - nx0 + 1)
                new_total = total_px + oc["n_px"]
                if new_total / new_area < MIN_GROUP_DENSITY:
                    continue
                gy0, gx0, gy1, gx1 = ny0, nx0, ny1, nx1
                total_px = new_total
                members.append(oc)
                used[j] = True
                changed = True
        if total_px < MIN_SPRITE_PIXELS:
            continue
        groups.append(members)
    return groups


def _touches_border(comp: dict, gbbox: tuple) -> bool:
    gy0, gx0, gy1, gx1 = gbbox
    by0, bx0, by1, bx1 = comp["bbox"]
    return by0 <= gy0 or bx0 <= gx0 or by1 >= gy1 or bx1 >= gx1


def _ring_modal(q: np.ndarray, gbbox: tuple):
    """Modal palette in the 1-cell ring just outside `gbbox` — the local background
    the group sits on (gray panel for a legend tile, the bar for a word glyph)."""
    gy0, gx0, gy1, gx1 = gbbox
    H, W = q.shape
    vals = []
    for x in range(max(0, gx0 - 1), min(W, gx1 + 2)):
        if gy0 - 1 >= 0: vals.append(int(q[gy0 - 1, x]))
        if gy1 + 1 < H: vals.append(int(q[gy1 + 1, x]))
    for y in range(max(0, gy0 - 1), min(H, gy1 + 2)):
        if gx0 - 1 >= 0: vals.append(int(q[y, gx0 - 1]))
        if gx1 + 1 < W: vals.append(int(q[y, gx1 + 1]))
    if not vals:
        return None
    from collections import Counter
    return Counter(vals).most_common(1)[0][0]


def _partition_group(members: list[dict], gbbox: tuple,
                     local_bg=None) -> list[tuple[str, list[dict]]]:
    """Split a CONTAINER+FIGURE group (a coloured tile/frame enclosing a distinct
    figure) into the container plus its contained figure(s), so an enclosed glyph is
    discovered as its own entity.  Pure geometry: a member whose bbox never reaches
    the group border, whose colour differs from the border members, and which is
    substantial (>= MIN_SPRITE_PIXELS), is an enclosed figure.  Ordinary multi-colour
    sprites whose parts reach the border (an articulated arm) are left intact.

    A figure whose colour equals `local_bg` -- the background the container sits on --
    is a SEE-THROUGH HOLE (the background showing through), NOT a contained object, so
    it is not split out.  This is the object/hole distinction: a glyph on a gray panel
    encloses a real figure (not gray); a glyph on a pink bar encloses a pink hole."""
    if len(members) < 2:
        return [("whole", members)]
    border_keys = {m["key"] for m in members if _touches_border(m, gbbox)}

    def _is_hole(m):
        # an enclosed region whose colour is the local background = the background
        # showing THROUGH the figure, not part of it.
        return (not _touches_border(m, gbbox) and m["key"] == local_bg
                and m["n_px"] >= MIN_SPRITE_PIXELS)

    # DROP see-through holes entirely, so they don't fill in the figure's bitmap (a
    # glyph with a hole stays a sparse glyph, not a solid blob).
    kept = [m for m in members if not _is_hole(m)]
    interior = [m for m in kept
                if not _touches_border(m, gbbox)
                and m["n_px"] >= MIN_SPRITE_PIXELS
                and m["key"] not in border_keys]
    if not border_keys or not interior:
        return [("whole", kept)]
    contained_keys = {m["key"] for m in interior}
    container = [m for m in kept if m["key"] not in contained_keys]
    return [("container", container)] + [("contained", [im]) for im in interior]


# -----------------------------------------------------------------------------
# Main detector entry point.
# -----------------------------------------------------------------------------


def _classify_region(
    cells: list[tuple[int, int]],
    rows: int,
    cols: int,
    visual_signature: tuple,
) -> tuple[float, bool]:
    """Region-vs-object on the CELLS mask.  Returns (bbox_fit, is_region).

    A REGION is LARGE, irregular (fills <60% of its bbox) and dominated by ONE
    palette (>=80% of pixels) — floor, walls, a hollow/diagonal outline.  Its
    rectangle is not a meaningful object box.  A compact sprite fills its bbox;
    an articulated multi-palette object (e.g. an agent with an extended arm)
    fails the single-palette gate even when sparse — so it is NOT a region.
    Substrate-agnostic: no role names, no game-specific code.
    """
    if not cells:
        return 1.0, False
    crs = [c[0] for c in cells]
    ccs = [c[1] for c in cells]
    bbox_cells = max(1, (max(crs) - min(crs) + 1) * (max(ccs) - min(ccs) + 1))
    fit = round(len(cells) / bbox_cells, 2)
    dom = visual_signature[0][1] if visual_signature else 0.0
    big = len(cells) >= 0.01 * rows * cols
    return fit, bool(big and fit < 0.6 and dom >= 0.8)


def detect_entities(
    obs: FrameObservation,
) -> list[Entity]:
    """Detect entities in a single frame.

    Returns a list of Entity records:
      - First two entries (by index) are the primary and secondary
        backgrounds, identified as the two largest single-colour
        connected regions.  They carry is_background_primary /
        is_background_secondary flags so downstream consumers know
        these aren't sprites.
      - Subsequent entries are sprite groups in arbitrary order.

    No role names are assigned here.  No game-specific code runs.
    The same detector handles every game in the corpus.
    """
    rows = obs.rows
    cols = obs.cols
    H_disp, W_disp = obs.rgb_frame.shape[:2]

    # TRUST THE DECLARED GRID.  The caller knows the true cell size, so derive
    # scale from it (rgb_height / rows) and downsample deterministically.
    # Inferring scale from frame CONTENT (_downsample_to_logical samples only
    # the top-left blocks) is unreliable: on su15 a uniform top band makes it
    # read a native 64x64 frame as 32x32, which DESTROYS the dashed connector
    # line (dashes spaced 2 cells apart collapse) and shrinks small sprites.
    # Fall back to content inference only when the declared grid doesn't divide
    # the frame evenly.
    H_full, W_full = obs.rgb_frame.shape[:2]
    if rows and cols and H_full % rows == 0 and W_full % cols == 0:
        scale = H_full // rows
        if scale > 1:
            off = scale // 2
            logical = obs.rgb_frame[off::scale, off::scale][:rows, :cols]
        else:
            logical = obs.rgb_frame
    else:
        logical, scale = _downsample_to_logical(obs.rgb_frame)
    H, W = logical.shape[:2]
    cell_h = max(1, H // rows)
    cell_w = max(1, W // cols)
    # Per-cell colour key (r<<16 | g<<8 | b) for building entity masks below.
    _key_grid = ((logical[..., 0].astype(np.int64) << 16)
                 | (logical[..., 1].astype(np.int64) << 8)
                 | logical[..., 2].astype(np.int64))
    q = _quantise_frame(logical, QUANT_STEP)
    comps = _find_components(q)

    if not comps:
        return []

    # Backgrounds: the two largest single-colour components by
    # pixel count.  This is the only "structural prior" the detector
    # carries — and it's substrate-agnostic (no game-specific code).
    comps_by_size = sorted(comps, key=lambda c: -c["n_px"])
    primary = comps_by_size[0]
    secondary = comps_by_size[1] if len(comps_by_size) >= 2 else None
    bg_keys: set[int] = {primary["key"]}
    if secondary:
        bg_keys.add(secondary["key"])

    entities: list[Entity] = []

    def _make_bg_entity(
        comp: dict, eid: int, *,
        is_primary: bool, is_secondary: bool,
    ) -> Entity:
        ys = comp["ys"]; xs = comp["xs"]
        bbox_l = comp["bbox"]
        bbox_d = (
            bbox_l[0] * scale, bbox_l[1] * scale,
            (bbox_l[2] + 1) * scale - 1,
            (bbox_l[3] + 1) * scale - 1,
        )
        sig = _signature_from_pixels(logical, ys, xs)
        # Per-cell pixel count for tie-breaking at projection time.
        # A bg's pixel-count-per-cell tells us which background owns
        # each cell when multiple touch it.
        pcount: dict[tuple[int, int], int] = {}
        for y, x in zip(ys, xs):
            cr = max(0, min(rows - 1, int(y // cell_h)))
            cc = max(0, min(cols - 1, int(x // cell_w)))
            pcount[(cr, cc)] = pcount.get((cr, cc), 0) + 1
        cy_l = float(ys.mean())
        cx_l = float(xs.mean())
        return Entity(
            entity_id=eid,
            bbox_logical=bbox_l,
            bbox_display=bbox_d,
            n_pixels=comp["n_px"],
            n_distinct_colors=1,
            visual_signature=sig,
            centroid_logical=(cy_l, cx_l),
            is_background_primary=is_primary,
            is_background_secondary=is_secondary,
            cells=sorted(pcount.keys()),
            centroid_cell=(
                max(0, min(rows - 1, int(cy_l // cell_h))),
                max(0, min(cols - 1, int(cx_l // cell_w))),
            ),
            pixel_count_per_cell=pcount,
        )

    entities.append(_make_bg_entity(
        primary, eid=0, is_primary=True, is_secondary=False,
    ))
    if secondary:
        entities.append(_make_bg_entity(
            secondary, eid=1, is_primary=False, is_secondary=True,
        ))

    # Foreground sprite groups.
    def _make_fg_entity(members: list[dict], eid: int, contained_in,
                        context_bg=None) -> Entity:
        gy0 = min(m["bbox"][0] for m in members)
        gx0 = min(m["bbox"][1] for m in members)
        gy1 = max(m["bbox"][2] for m in members)
        gx1 = max(m["bbox"][3] for m in members)
        all_ys = np.concatenate([m["ys"] for m in members])
        all_xs = np.concatenate([m["xs"] for m in members])
        sig = _signature_from_pixels(logical, all_ys, all_xs)
        cy = float(all_ys.mean())
        cx = float(all_xs.mean())
        centroid_cell = (
            max(0, min(rows - 1, int(cy // cell_h))),
            max(0, min(cols - 1, int(cx // cell_w))),
        )
        pcount: dict[tuple[int, int], int] = {}
        for y, x in zip(all_ys, all_xs):
            cr = max(0, min(rows - 1, int(y // cell_h)))
            cc = max(0, min(cols - 1, int(x // cell_w)))
            pcount[(cr, cc)] = pcount.get((cr, cc), 0) + 1
        bbox_d = (gy0 * scale, gx0 * scale,
                  (gy1 + 1) * scale - 1, (gx1 + 1) * scale - 1)
        cell_list = sorted(pcount.keys())
        fit, is_region = _classify_region(cell_list, rows, cols, sig)
        # Coloured mask in logical/cell coords: the entity's own pixels only, bg = -1.
        bmp = np.full((gy1 - gy0 + 1, gx1 - gx0 + 1), -1, dtype=np.int64)
        bmp[all_ys - gy0, all_xs - gx0] = _key_grid[all_ys, all_xs]
        return Entity(
            entity_id=eid,
            bbox_logical=(gy0, gx0, gy1, gx1),
            bbox_display=bbox_d,
            n_pixels=int(all_ys.size),
            n_distinct_colors=len({m["key"] for m in members}),
            visual_signature=sig,
            centroid_logical=(cy, cx),
            cells=cell_list,
            centroid_cell=centroid_cell,
            pixel_count_per_cell=pcount,
            bbox_fit=fit,
            is_region=is_region,
            bitmap=bmp,
            contained_in=contained_in,
            context_bg=context_bg,
        )

    groups = _group_components_into_sprites(comps, bg_keys)
    for members in groups:
        gbbox = (min(m["bbox"][0] for m in members), min(m["bbox"][1] for m in members),
                 max(m["bbox"][2] for m in members), max(m["bbox"][3] for m in members))
        lbg = _ring_modal(q, gbbox)                  # the local background the group sits on
        specs = _partition_group(members, gbbox, lbg)
        if len(specs) == 1:
            entities.append(_make_fg_entity(specs[0][1], len(entities), None, lbg))
        else:
            # container sits on the panel (lbg); each enclosed figure sits on the
            # CONTAINER, so its context is the container's colour.
            container_members = specs[0][1]
            container_color = Counter(m["key"] for m in container_members).most_common(1)[0][0]
            container_eid = len(entities)
            entities.append(_make_fg_entity(container_members, container_eid, None, lbg))
            for _role, mem in specs[1:]:
                entities.append(_make_fg_entity(mem, len(entities), container_eid, container_color))

    return entities


# -----------------------------------------------------------------------------
# Frame-level utilities downstream consumers may need.
# -----------------------------------------------------------------------------


def get_logical_shape(obs: FrameObservation) -> tuple[int, int, int]:
    """Return (rows, cols, scale) for the observation's frame.
    Downstream matchers that need pixel/cell math but don't want to
    re-run downsampling read this."""
    rows = obs.rows; cols = obs.cols
    logical, scale = _downsample_to_logical(obs.rgb_frame)
    return rows, cols, scale
