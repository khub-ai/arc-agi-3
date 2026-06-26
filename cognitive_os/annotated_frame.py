"""Annotated frame renderer for VLM consultation.

Produces a high-resolution PNG of the current game frame with
visual scaffolding the VLM can reason about precisely:

* 4-cell and 16-cell yellow gridlines with column / row labels
* Entity bounding boxes outlined and numbered (#1, #2, ...)
* Agent marker with a bold label
* Optional slide-direction arrow at the agent's position
* Optional scroll-trigger-source markers

The substrate sends this annotated PNG to the VLM and instructs it
to refer to entities by their numbered labels and to coordinates by
the gridded col/row.  This is the shared visual language between
substrate and oracle — both can refer to "consume #6" or "the cell
at (col 27, row 35)" and mean the same thing.

No game-specific assumptions: the renderer takes any entity list
with bboxes and roles, any agent position, and any slide direction.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np


# Role -> highlight color.  These are visual hints for the VLM, not
# semantic labels.  Default fallback is light orange for unknown
# roles.  Adjust the mapping when adding new role categories.
_DEFAULT_ROLE_COLORS = {
    "agent_avatar":           (255, 230, 0,   230),  # yellow
    "movable_block":          (80,  255, 80,  230),  # bright green
    "reference_glyph":        (255, 100, 200, 230),  # pink
    "reference_pair_member":  (200, 100, 255, 230),  # purple
    "reference_arrangement":  (255, 180, 80,  230),  # amber
    "play_area":              None,                   # no outline
    "floor":                  None,
    "background":             None,
    "wall":                   None,
    "void":                   None,
}


def render_annotated_frame(
    frame:           np.ndarray,
    entities:        Sequence[Mapping[str, Any]],
    *,
    agent_centroid:  Optional[Tuple[int, int]] = None,
    slide_dirs:      Sequence[Tuple[int, int]] = (),
    scroll_sources:  Sequence[Tuple[int, int]] = (),
    out_path:        Optional[Path]            = None,
    cells:           int                        = 64,
    scale:           int                        = 8,
    role_colors:     Optional[Mapping[str, Optional[Tuple[int, int, int, int]]]] = None,
    reachable_mask:  Optional[np.ndarray]       = None,
    unreachable_entity_ids: Optional[Sequence[str]] = None,
    adjacent_entity_ids:    Optional[Sequence[str]] = None,
    tried_no_progress_ids:  Optional[Sequence[str]] = None,
    scrolled_entity_ids:    Optional[Sequence[str]] = None,
    actual_slide_endpoint:  Optional[Tuple[int, int]] = None,
    legend_summary:         Optional[str]             = None,
) -> Path:
    """Render the annotated PNG and return the file path.

    Parameters
    ----------
    frame
        Either a 2D ``(H, W)`` palette-id array or a 3D ``(H, W, 3)``
        RGB pixel array.  The renderer upsamples to ``cells * scale``
        on a side and overlays the annotations.
    entities
        Entity dicts with ``bbox_pixels`` (``[r0, c0, r1, c1]``) and
        ``role`` (string).  Each entity is outlined and numbered.
    agent_centroid
        Optional ``(col, row)`` of the agent.  Marks the agent with
        an "AGENT" label when supplied.
    slide_dirs
        Optional list of ``(dc, dr)`` slide directions from the
        mechanic catalog.  Each gets an arrow at the agent's position
        in the slide direction.
    scroll_sources
        Optional list of ``(col, row)`` scroll-trigger source cells
        from the catalog.  Each gets a small ring.
    out_path
        Where to save the PNG.  When None, the renderer chooses a
        location under ``.tmp/vlm_frames/``.
    cells / scale
        Default 64x64 logical cells, rendered at 8x → 512x512 PNG.
    role_colors
        Override the role → color mapping.  Unmapped roles get a
        default fallback outline.
    reachable_mask
        Optional 2D bool array (same H/W as the logical frame) that
        marks the cells the agent can currently reach without
        consuming anything.  When supplied, a translucent cyan
        overlay tints those cells, giving the VLM a direct visual
        cue for "what counts as 'on the agent's path'" — eliminates
        the failure mode where the VLM picks a tile in the agent's
        column but separated from it by intermediate walls.
    unreachable_entity_ids
        Optional iterable of entity ``id`` strings the caller has
        determined are NOT adjacent to ``reachable_mask`` (so a
        single consume on them opens no new ground).  These get a
        thick RED outline and an "isolated" label, signaling
        the VLM that they are spatially isolated from the agent
        and not productive single-click targets.
    adjacent_entity_ids
        Optional iterable of entity ``id`` strings the caller has
        determined ARE adjacent to ``reachable_mask`` (so a
        single consume opens new ground).  These get a thick
        GREEN outline and an "OPEN" label, signaling the VLM
        that they are the productive single-click candidates
        from the agent's current position.
    actual_slide_endpoint
        Optional ``(col, row)`` of the cell the agent would
        reach if it slid along the confirmed slide axis right
        now, accounting for intermediate obstacles.  When
        supplied, the slide arrow stops exactly at this
        endpoint (instead of drawing a long generic axis line
        through obstacles), giving the VLM a precise visual
        statement of "this is how far the slide actually
        travels."
    legend_summary
        Optional multi-line text drawn in a corner box of the
        image.  Used to enumerate "Adjacent (productive)
        consumables: #5, #8; Isolated (wasted) consumables:
        #2, #3, #4" so the VLM has the spatial conclusion
        already pre-computed in plain language.
    """
    from PIL import Image, ImageDraw, ImageFont

    if out_path is None:
        out_path = (Path(".tmp") / "vlm_frames"
                    / "annotated_frame.png")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    role_color_map = dict(_DEFAULT_ROLE_COLORS)
    if role_colors is not None:
        role_color_map.update(role_colors)

    # 1. Convert frame to a base RGB image.
    arr = np.asarray(frame)
    if arr.ndim == 2:
        # Palette-id grid.  Without a known palette mapping, render
        # each unique id with a distinct shade.  Black for id 0.
        H, W = arr.shape
        unique_ids = np.unique(arr)
        # Build a per-id color: black for 0, light-cyan for the
        # most-common non-zero id (palette inference), distinct
        # hues for others.
        rgb = np.zeros((H, W, 3), dtype=np.uint8)
        # Heuristic: id 0 black; largest non-zero id cyan; rest
        # rotate through a small palette.
        non_zero_counts = [(int((arr == int(pid)).sum()), int(pid))
                           for pid in unique_ids if int(pid) != 0]
        non_zero_counts.sort(reverse=True)
        id_to_color: dict = {0: (0, 0, 0)}
        if non_zero_counts:
            id_to_color[non_zero_counts[0][1]] = (136, 216, 241)
            fallback = [
                (80, 255, 80), (255, 100, 200), (200, 100, 255),
                (255, 180, 80), (255, 230, 0), (200, 200, 200),
            ]
            for i, (_, pid) in enumerate(non_zero_counts[1:]):
                id_to_color[pid] = fallback[i % len(fallback)]
        for pid, col in id_to_color.items():
            rgb[arr == pid] = col
        base = Image.fromarray(rgb, mode="RGB").resize(
            (cells * scale, cells * scale), Image.NEAREST,
        )
    elif arr.ndim == 3 and arr.shape[2] == 3:
        # RGB pixel array — assume input is already at frame
        # resolution (e.g., the env's raw 64x64 pixel-grid, or
        # an already-upscaled version).
        if arr.shape[0] == cells and arr.shape[1] == cells:
            base = Image.fromarray(arr.astype(np.uint8), mode="RGB").resize(
                (cells * scale, cells * scale), Image.NEAREST,
            )
        else:
            base = Image.fromarray(arr.astype(np.uint8), mode="RGB").resize(
                (cells * scale, cells * scale), Image.NEAREST,
            )
    else:
        raise ValueError(
            f"render_annotated_frame expects 2D palette or 3D RGB, "
            f"got shape {arr.shape}"
        )

    img = base.convert("RGBA")
    W_px, H_px = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    try:
        font_small = ImageFont.truetype("arial.ttf", 11)
        font_med   = ImageFont.truetype("arial.ttf", 14)
        font_big   = ImageFont.truetype("arial.ttf", 18)
    except Exception:
        font_small = ImageFont.load_default()
        font_med   = font_small
        font_big   = font_small

    def xy(col: int, row: int) -> Tuple[int, int]:
        return (col * scale + scale // 2,
                row * scale + scale // 2)

    # 2. Pixel-index rulers.  These are NOT game-cell boundaries
    # -- the underlying game's cell size varies (entities can be
    # 5 px wide on 6 px spacing, etc.).  The ruler ticks every
    # 4 pixels, with labels at every tick, purely as a coordinate
    # reference for the VLM to cite "col=21, row=38" precisely.
    # Operator feedback 2026-05-17: "your grid lines are off in
    # many places, because this layout is not based on strictly
    # square cells in all places" -- the rendering used to imply
    # the lines were cell boundaries.  We now draw them faint
    # and label the figure as "pixel coordinates" in the legend.
    for c in range(0, cells + 1, 4):
        x = c * scale
        # Thin grey lines, faint -- a coordinate reference, not
        # a topology marker.
        line_color = (180, 180, 180, 80)
        line_width = 1
        if c % 16 == 0:
            line_color = (200, 200, 200, 140); line_width = 1
        draw.line([(x, 0), (x, H_px)],
                  fill=line_color, width=line_width)
        if c % 4 == 0 and c < cells:
            label = str(c)
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            draw.rectangle(
                [x + 1, 1, x + 1 + tw + 4, 1 + th + 4],
                fill=(0, 0, 0, 180),
            )
            draw.text((x + 3, 2), label,
                      fill=(220, 220, 220, 220), font=font_small)
    for r in range(0, cells + 1, 4):
        y = r * scale
        line_color = (180, 180, 180, 80)
        line_width = 1
        if r % 16 == 0:
            line_color = (200, 200, 200, 140); line_width = 1
        draw.line([(0, y), (W_px, y)],
                  fill=line_color, width=line_width)
        if r % 4 == 0 and r < cells:
            label = str(r)
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            draw.rectangle(
                [1, y + 1, 1 + tw + 4, y + 1 + th + 4],
                fill=(0, 0, 0, 180),
            )
            draw.text((3, y + 2), label,
                      fill=(220, 220, 220, 220), font=font_small)

    # 2.5. Reachability overlay (between gridlines and entity
    # outlines).  Two-phase rendering for HIGH CONTRAST between
    # reachable vs unreachable cells:
    #
    #   Phase A: darken EVERYTHING with a translucent black layer
    #            so the unreachable region visibly dims.
    #
    #   Phase B: punch BRIGHT cells back through the dim layer
    #            where the reachable mask is True -- using a
    #            saturated golden-yellow stripe pattern that
    #            cannot be confused with the base play_area
    #            palette (which is typically cyan or green).
    #
    # Without this, a cyan tint on top of a cyan-palette frame
    # is visually indistinguishable from the base.  Operator
    # feedback 2026-05-17: "agent can only reach those cyan
    # cells contiguous to the current position; there are cyan
    # cells that could be unreachable" -- the prior cyan-tint
    # rendering made this distinction impossible to read.
    if reachable_mask is not None:
        rm = np.asarray(reachable_mask)
        if rm.ndim == 2 and rm.shape == (cells, cells):
            rm_img = Image.fromarray(
                (rm.astype(np.uint8) * 255), mode="L",
            ).resize((cells * scale, cells * scale), Image.NEAREST)
            mask_alpha = rm_img.point(lambda v: 255 if v > 128 else 0)
            # Phase A: dim everything (the unreachable region
            # becomes visibly darker than the reachable region).
            dim_layer = Image.new(
                "RGBA", img.size, (0, 0, 0, 140),
            )
            img = Image.alpha_composite(img, dim_layer)
            # Phase B: punch the reachable region back to full
            # brightness with a neutral white wash that preserves
            # the underlying palette colors (no green/cyan
            # blending).  Higher alpha than before so the
            # contrast is unambiguous.
            highlight = Image.new(
                "RGBA", img.size, (255, 255, 255, 110),
            )
            highlight_with_mask = Image.new(
                "RGBA", img.size, (0, 0, 0, 0),
            )
            highlight_with_mask.paste(
                highlight, (0, 0), mask_alpha,
            )
            img = Image.alpha_composite(img, highlight_with_mask)
            # Phase C: draw a thick magenta border around the
            # reachable region's perimeter so the boundary is
            # visible at a glance, no matter what palette colors
            # sit on either side.  Magenta is uncommon in ARC-AGI
            # game palettes; the boundary doesn't blend with
            # green / cyan / red / orange.
            try:
                # Edge pixels = mask cells whose 4-neighbours
                # include at least one unreachable cell.
                H_m, W_m = rm.shape
                edges = np.zeros((H_m, W_m), dtype=bool)
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    shifted = np.roll(rm, (dr, dc), axis=(0, 1))
                    # Wrap-around shouldn't be treated as
                    # neighbours; zero out the wrapped edge.
                    if dr == -1: shifted[-1, :] = False
                    if dr ==  1: shifted[0,  :] = False
                    if dc == -1: shifted[:, -1] = False
                    if dc ==  1: shifted[:,  0] = False
                    edges |= (rm & ~shifted)
                edge_img = Image.fromarray(
                    (edges.astype(np.uint8) * 255), mode="L",
                ).resize(
                    (cells * scale, cells * scale), Image.NEAREST,
                )
                edge_alpha = edge_img.point(
                    lambda v: 255 if v > 128 else 0,
                )
                border = Image.new(
                    "RGBA", img.size, (255, 0, 200, 230),
                )
                border_layer = Image.new(
                    "RGBA", img.size, (0, 0, 0, 0),
                )
                border_layer.paste(border, (0, 0), edge_alpha)
                img = Image.alpha_composite(img, border_layer)
            except Exception:
                pass

    # Pre-compute id sets for fast lookup in the outline loop.
    _unreachable_ids: Set = set()
    if unreachable_entity_ids:
        _unreachable_ids = set(str(x) for x in unreachable_entity_ids)
    _adjacent_ids: Set = set()
    if adjacent_entity_ids:
        _adjacent_ids = set(str(x) for x in adjacent_entity_ids)
    _tried_ids: Set = set()
    if tried_no_progress_ids:
        _tried_ids = set(str(x) for x in tried_no_progress_ids)
    _scrolled_ids: Set = set()
    if scrolled_entity_ids:
        _scrolled_ids = set(str(x) for x in scrolled_entity_ids)

    # 3. Entity bboxes with numbered labels.  Number entities in
    # order of (role-priority, top-row, left-col) so the labels
    # are stable across renders of the same frame.
    def _bbox_of(ent: Mapping[str, Any]) -> Optional[Tuple[int, int, int, int]]:
        bb = ent.get("bbox_pixels") or []
        if len(bb) < 4:
            return None
        try:
            return (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        except (TypeError, ValueError):
            return None

    drawable: list = []
    for ent in entities:
        bb = _bbox_of(ent)
        if bb is None:
            continue
        role = str(ent.get("role") or "").strip().lower()
        color = role_color_map.get(role, (255, 150, 0, 220))
        if color is None:
            # Skip non-outlined roles (play_area / floor / etc).
            continue
        drawable.append((role, bb, color, ent))

    # Sort: agent first, then by top-row, then left-col.
    drawable.sort(key=lambda t: (
        0 if t[0] in ("agent_avatar", "agent") else 1,
        t[1][0],  # r0
        t[1][1],  # c0
    ))
    for i, (role, bb, color, ent) in enumerate(drawable, start=1):
        r0, c0, r1, c1 = bb
        x0 = c0 * scale; y0 = r0 * scale
        x1 = (c1 + 1) * scale - 1; y1 = (r1 + 1) * scale - 1
        # Reachability flags: an entity is either ADJACENT (so a
        # single consume opens new ground for the agent), ISOLATED
        # (consuming opens no ground reachable from current
        # position), or neither (non-consumable / no flag info).
        # The visual encoding is high-contrast so the VLM can read
        # the spatial conclusion at a glance:
        #   ADJACENT  -> bright GREEN outline + "OPEN" tag
        #   ISOLATED  -> bright RED outline + "isolated" tag
        #   neither   -> default role color outline
        eid_str = str(ent.get("id") or "")
        is_unreachable = (eid_str in _unreachable_ids) if _unreachable_ids else False
        is_adjacent    = (eid_str in _adjacent_ids)    if _adjacent_ids    else False
        is_tried       = (eid_str in _tried_ids)       if _tried_ids       else False
        is_scrolled    = (eid_str in _scrolled_ids)    if _scrolled_ids    else False
        # Precedence: scrolled > tried > unreachable > adjacent.
        # SCROLLED entities have been EMPIRICALLY OBSERVED to
        # trigger a room transition when consumed -- the strongest
        # positive signal we can give the VLM.  TRIED-FUTILE
        # entities have been observed to consume WITHOUT
        # transitioning.  Untried adjacent entities are the
        # remaining candidates.
        if is_scrolled:
            out_color = (0, 255, 0, 255)      # bright pure green
            outline_w = 5
        elif is_tried:
            out_color = (255, 165, 0, 240)    # orange -- tried, futile
            outline_w = 4
        elif is_unreachable:
            out_color = (255, 70, 70, 240)
            outline_w = 4
        elif is_adjacent:
            out_color = (60, 230, 80, 240)
            outline_w = 4
        else:
            out_color = color
            outline_w = 2
        for w in range(outline_w):
            draw.rectangle([x0 - w, y0 - w, x1 + w, y1 + w],
                           outline=out_color, width=1)
        # Number label inside the bbox, top-left corner
        label = f"#{i}"
        bbox = draw.textbbox((0, 0), label, font=font_med)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        lx = x0 + 2; ly = y0 + 2
        draw.rectangle(
            [lx, ly, lx + tw + 6, ly + th + 4],
            fill=(0, 0, 0, 220),
        )
        draw.text((lx + 3, ly + 1), label,
                  fill=out_color, font=font_med)
        # Tag below the #N label so the spatial conclusion is
        # also textually visible inside the rendered image.
        tag: Optional[str] = None
        tag_bg = (80, 0, 0, 230)
        tag_fg = (255, 200, 200, 255)
        if is_scrolled:
            tag = "SCROLLED!"
            tag_bg = (0, 80, 0, 230)
            tag_fg = (180, 255, 180, 255)
        elif is_tried:
            tag = "TRIED-FUTILE"
            tag_bg = (90, 50, 0, 230)
            tag_fg = (255, 200, 130, 255)
        elif is_unreachable:
            tag = "isolated"
            tag_bg = (80, 0, 0, 230)
            tag_fg = (255, 200, 200, 255)
        elif is_adjacent:
            tag = "OPEN (untried)"
            tag_bg = (0, 50, 50, 230)
            tag_fg = (180, 230, 230, 255)
        if tag is not None:
            tbox = draw.textbbox((0, 0), tag, font=font_small)
            ttw = tbox[2] - tbox[0]; tth = tbox[3] - tbox[1]
            tlx = x0 + 2
            tly = ly + th + 6
            draw.rectangle(
                [tlx, tly, tlx + ttw + 4, tly + tth + 4],
                fill=tag_bg,
            )
            draw.text((tlx + 2, tly + 1), tag,
                      fill=tag_fg, font=font_small)
        # Index the entity dict in-place with the displayed number
        # so the caller can map back from "#N" to entity id.
        try:
            ent["_vlm_label_number"] = i  # type: ignore[index]
        except Exception:
            pass

    # 4. Agent marker
    if agent_centroid is not None:
        ac, ar = int(agent_centroid[0]), int(agent_centroid[1])
        ax, ay = xy(ac, ar)
        r_marker = scale + 2
        draw.ellipse(
            [ax - r_marker, ay - r_marker,
             ax + r_marker, ay + r_marker],
            outline=(255, 0, 255, 255), width=3,
        )
        label = "AGENT"
        bbox = draw.textbbox((0, 0), label, font=font_med)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        # Position label below the agent if room, above otherwise.
        ly = ay + r_marker + 4 if ay + r_marker + th + 8 < H_px \
             else ay - r_marker - th - 8
        lx = ax - tw // 2
        draw.rectangle(
            [lx - 4, ly - 2, lx + tw + 4, ly + th + 4],
            fill=(0, 0, 0, 230),
        )
        draw.text((lx, ly), label,
                  fill=(255, 0, 255, 255), font=font_med)

    # 5. Slide-axis arrow.  When the caller supplies the actual
    # slide endpoint (computed from the live passability grid),
    # the arrow stops exactly at that cell -- visually
    # communicating "this is how far the slide travels right
    # now" rather than the generic axis direction.  Without an
    # endpoint, fall back to a 12-cell projection line as a
    # rough direction hint.
    if agent_centroid is not None and slide_dirs:
        ac, ar = int(agent_centroid[0]), int(agent_centroid[1])
        ax, ay = xy(ac, ar)
        for dc, dr in slide_dirs:
            if dc == 0 and dr == 0:
                continue
            if actual_slide_endpoint is not None:
                # Use the actual stopping cell (in cell coords).
                ec, er = (int(actual_slide_endpoint[0]),
                          int(actual_slide_endpoint[1]))
                tip_x, tip_y = xy(ec, er)
                arrow_label = "slide reach (actual)"
                arrow_color = (0, 230, 100, 240)
            else:
                length = 12 * scale
                mag = math.hypot(dc, dr)
                dx = int(round(dc * length / mag))
                dy = int(round(dr * length / mag))
                tip_x = ax + dx; tip_y = ay + dy
                arrow_label = "slide axis (direction only)"
                arrow_color = (0, 200, 255, 240)
            # Start just outside the agent's circle
            mag = max(1.0, math.hypot(tip_x - ax, tip_y - ay))
            sx = ax + int(round((tip_x - ax) * (scale + 4) / mag))
            sy = ay + int(round((tip_y - ay) * (scale + 4) / mag))
            draw.line([(sx, sy), (tip_x, tip_y)],
                      fill=arrow_color, width=4)
            angle = math.atan2(tip_y - sy, tip_x - sx)
            hl = 14
            ax1 = tip_x - hl * math.cos(angle - math.pi / 6)
            ay1 = tip_y - hl * math.sin(angle - math.pi / 6)
            ax2 = tip_x - hl * math.cos(angle + math.pi / 6)
            ay2 = tip_y - hl * math.sin(angle + math.pi / 6)
            draw.polygon([(tip_x, tip_y), (ax1, ay1), (ax2, ay2)],
                         fill=arrow_color)
            # Label
            mid_x = (sx + tip_x) // 2
            mid_y = (sy + tip_y) // 2
            label = arrow_label
            bbox = draw.textbbox((0, 0), label, font=font_small)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            draw.rectangle(
                [mid_x + 4, mid_y - th // 2 - 2,
                 mid_x + 4 + tw + 6, mid_y + th // 2 + 2],
                fill=(0, 60, 80, 220),
            )
            draw.text((mid_x + 7, mid_y - th // 2),
                      label, fill=arrow_color[:3] + (255,),
                      font=font_small)

    # 6. Scroll-source markers
    for sc, sr in scroll_sources:
        sx, sy = xy(int(sc), int(sr))
        r_s = scale - 1
        draw.ellipse(
            [sx - r_s, sy - r_s, sx + r_s, sy + r_s],
            outline=(0, 255, 100, 220), width=2,
        )

    # 7. Legend / summary box.  Drawn in the bottom-right corner
    # of the image with the pre-computed spatial conclusion in
    # plain language ("ADJACENT consumables: #5, #8; ISOLATED:
    # #2, #3, #4; slide reach: (col 50, row 38)").  Putting the
    # conclusion directly on the image gives the VLM a textual
    # anchor that's hard to miss; the overlay alone is sometimes
    # under-read by smaller VLMs.
    if legend_summary:
        lines = [ln for ln in legend_summary.splitlines() if ln.strip()]
        if lines:
            # Measure block.
            line_h = 0
            max_w  = 0
            for ln in lines:
                bb = draw.textbbox((0, 0), ln, font=font_small)
                lw = bb[2] - bb[0]; lh = bb[3] - bb[1]
                max_w = max(max_w, lw)
                line_h = max(line_h, lh)
            pad_x = 8; pad_y = 6
            box_w = max_w + pad_x * 2
            box_h = (line_h + 2) * len(lines) + pad_y * 2 - 2
            x_l = W_px - box_w - 6
            y_l = H_px - box_h - 6
            draw.rectangle(
                [x_l, y_l, x_l + box_w, y_l + box_h],
                fill=(0, 0, 0, 220),
                outline=(255, 255, 255, 220),
                width=2,
            )
            for i_ln, ln in enumerate(lines):
                draw.text(
                    (x_l + pad_x,
                     y_l + pad_y + i_ln * (line_h + 2)),
                    ln,
                    fill=(255, 255, 255, 255),
                    font=font_small,
                )

    result = Image.alpha_composite(img, overlay).convert("RGB")
    result.save(out_path)
    return out_path


def entity_label_mapping(
    entities: Sequence[Mapping[str, Any]],
) -> dict:
    """Return a ``{label_number: entity_dict}`` mapping that matches
    the numbering :func:`render_annotated_frame` applies.  Useful for
    parsing VLM responses that refer to entities by ``#N``."""
    drawable: list = []
    for ent in entities:
        bb = ent.get("bbox_pixels") or []
        if len(bb) < 4:
            continue
        role = str(ent.get("role") or "").strip().lower()
        color = _DEFAULT_ROLE_COLORS.get(role, (255, 150, 0, 220))
        if color is None:
            continue
        drawable.append((role, bb, ent))
    drawable.sort(key=lambda t: (
        0 if t[0] in ("agent_avatar", "agent") else 1,
        int(t[1][0]),
        int(t[1][1]),
    ))
    return {i: ent for i, (_, _, ent) in enumerate(drawable, start=1)}
