"""Render an annotated game frame.

Takes a raw 64x64 palette frame plus a candidate list and produces
an upscaled PNG with each candidate's bounding box drawn over the
frame and a visible numeric label at each box's corner.  The result
is what the VLM sees -- a self-describing image in which every
entity the substrate believes exists is visually called out by id.

Public entry point: ``render_annotated_frame(frame, candidates, out_path)``.

Each ``candidate`` is a dict with at least:

    id        -- integer id displayed on the frame (1, 2, 3, ...)
    bbox      -- (r0, c0, r1, c1) in pixel coordinates
    role      -- short role label rendered next to the id
                 (optional; can be "unknown" or the empty string)

Optional fields used for the legend:

    palettes  -- list of palette ints
    summary   -- short behaviour description ("moved 18px", "static",
                 "animated every step", ...)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

# Distinct high-contrast outline colours, cycled by id.  Chosen so
# adjacent ids look obviously different on a typical ARC palette.
_OUTLINE_COLOURS: Tuple[Tuple[int, int, int], ...] = (
    (255, 64, 64),     # red
    (64, 200, 64),     # green
    (64, 128, 255),    # blue
    (255, 200, 32),    # yellow
    (255, 96, 255),    # magenta
    (32, 224, 224),    # cyan
    (255, 144, 32),    # orange
    (224, 224, 224),   # near-white
)

# Scaling factor when rendering: each game pixel becomes a 12x12
# block.  Big enough that 1-2 character ID labels read at a glance.
_SCALE = 12


def _hx(h: str) -> Tuple[int, int, int]:
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


def _palette_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Upscale a (H, W) palette frame to (H*scale, W*scale, 3) RGB."""
    from arc_agi.rendering import COLOR_MAP  # type: ignore
    palmap = {int(i): _hx(h) for i, h in COLOR_MAP.items()}
    H, W = frame.shape
    img = np.zeros((H * _SCALE, W * _SCALE, 3), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            img[r*_SCALE:(r+1)*_SCALE, c*_SCALE:(c+1)*_SCALE] = palmap.get(
                int(frame[r, c]), (0, 0, 0))
    return img


def _draw_rect(img: np.ndarray, r0: int, c0: int, r1: int, c1: int,
               colour: Tuple[int, int, int], thickness: int = 2) -> None:
    """Draw a hollow rectangle in image-pixel coordinates."""
    H, W, _ = img.shape
    # Top + bottom edges
    img[max(0, r0):min(H, r0 + thickness), max(0, c0):min(W, c1)] = colour
    img[max(0, r1 - thickness):min(H, r1), max(0, c0):min(W, c1)] = colour
    # Left + right edges
    img[max(0, r0):min(H, r1), max(0, c0):min(W, c0 + thickness)] = colour
    img[max(0, r0):min(H, r1), max(0, c1 - thickness):min(W, c1)] = colour


def _measure_text(draw, font, text: str) -> Tuple[int, int]:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def _draw_label(img: np.ndarray, r: int, c: int, text: str,
                fg: Tuple[int, int, int],
                bg: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """Draw ``text`` near (r, c) in image-pixel coords.

    Uses PIL for text rendering.  A black background rectangle sits
    behind the text so it stays legible regardless of frame content.
    """
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    tw, th = _measure_text(draw, font, text)
    pad = 2
    draw.rectangle(
        [(c - pad, r - pad), (c + tw + pad, r + th + pad)],
        fill=bg,
    )
    draw.text((c, r), text, fill=fg, font=font)
    img[:] = np.asarray(pil)


def _label_size(text: str) -> Tuple[int, int]:
    """Estimate label size in pixels for collision detection."""
    from PIL import ImageDraw, ImageFont, Image
    pil = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    tw, th = _measure_text(draw, font, text)
    return tw + 4, th + 4   # include padding


def _rects_overlap(r1, r2) -> bool:
    """r1 / r2 are (left, top, right, bottom).  Inclusive overlap."""
    return not (r1[2] <= r2[0] or r2[2] <= r1[0]
                or r1[3] <= r2[1] or r2[3] <= r1[1])


def render_annotated_frame(
    frame:      np.ndarray,
    candidates: Sequence[Mapping],
    out_path:   Path,
    *,
    title:      Optional[str] = None,
) -> Path:
    """Render the annotated frame to ``out_path`` and return the path.

    Each candidate gets a coloured bbox outline (colours cycled by id)
    and a label "<id>:<role>" pinned at the top-left corner of the
    bbox.  Region-role candidates (wall, play_area) covering most of
    the frame are drawn with thin corner brackets rather than a full
    outline so the frame stays visible.
    """
    img = _palette_to_rgb(frame)
    H, W, _ = img.shape
    frame_area = frame.shape[0] * frame.shape[1]

    # Sort by id so colour assignment is stable across calls.
    cands = sorted(candidates, key=lambda c: int(c.get("id", 0)))
    # First pass: draw all outlines / corner brackets.
    cand_info = []   # (cid, bbox_img, colour, label)
    for cand in cands:
        cid = int(cand.get("id", 0))
        bbox = cand.get("bbox") or [0, 0, 0, 0]
        role = str(cand.get("role") or "?")
        colour = _OUTLINE_COLOURS[(cid - 1) % len(_OUTLINE_COLOURS)]
        r0, c0, r1, c1 = bbox
        ir0 = r0 * _SCALE
        ic0 = c0 * _SCALE
        ir1 = (r1 + 1) * _SCALE
        ic1 = (c1 + 1) * _SCALE
        bbox_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        too_large = bbox_area / frame_area > 0.25
        if too_large:
            arm = 2 * _SCALE
            t = 2
            img[ir0:ir0 + t, ic0:ic0 + arm] = colour
            img[ir0:ir0 + arm, ic0:ic0 + t] = colour
            img[ir0:ir0 + t, ic1 - arm:ic1] = colour
            img[ir0:ir0 + arm, ic1 - t:ic1] = colour
            img[ir1 - t:ir1, ic0:ic0 + arm] = colour
            img[ir1 - arm:ir1, ic0:ic0 + t] = colour
            img[ir1 - t:ir1, ic1 - arm:ic1] = colour
            img[ir1 - arm:ir1, ic1 - t:ic1] = colour
        else:
            _draw_rect(img, ir0, ic0, ir1, ic1, colour, thickness=2)
        label = f"{cid}:{role}"
        cand_info.append((cid, (ir0, ic0, ir1, ic1), colour, label))

    # Second pass: place labels.  For each candidate, try a list of
    # candidate placements (above the bbox, then below, then left,
    # then right, then inside) and pick the first one that doesn't
    # overlap any already-placed label.  Labels are clamped to the
    # image bounds.
    placed_rects: List[Tuple[int, int, int, int]] = []
    LABEL_H_GAP = 4
    for cid, (ir0, ic0, ir1, ic1), colour, label in cand_info:
        tw, th = _label_size(label)
        # Candidate anchors as (label_top_left_x, label_top_left_y).
        candidates_xy = [
            (ic0, ir0 - th - LABEL_H_GAP),            # above
            (ic0, ir1 + LABEL_H_GAP),                  # below
            (ic1 + LABEL_H_GAP, ir0),                  # right
            (ic0 - tw - LABEL_H_GAP, ir0),             # left
            (ic0 + 2, ir0 + 2),                        # inside top-left
            (ic1 - tw - 2, ir1 - th - 2),              # inside bottom-right
            (ic0, ir1 - th - 2),                       # bottom-inside-left
        ]
        chosen = None
        for x, y in candidates_xy:
            # Clamp.
            x = max(0, min(W - tw, x))
            y = max(0, min(H - th, y))
            rect = (x, y, x + tw, y + th)
            if not any(_rects_overlap(rect, pr) for pr in placed_rects):
                chosen = (x, y, rect)
                break
        if chosen is None:
            # All candidate positions collided; stagger downward from
            # the first anchor until a free slot opens or we hit the
            # frame edge.
            x, y = candidates_xy[0]
            x = max(0, min(W - tw, x))
            y = max(0, min(H - th, y))
            step = th + 2
            while y + th < H and any(
                    _rects_overlap((x, y, x + tw, y + th), pr)
                    for pr in placed_rects):
                y += step
            chosen = (x, y, (x, y, x + tw, y + th))
        x, y, rect = chosen
        placed_rects.append(rect)
        _draw_label(img, y, x, label, fg=colour)

    # Optional title strip across the top.
    if title:
        from PIL import Image, ImageDraw, ImageFont
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except OSError:
            font = ImageFont.load_default()
        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except AttributeError:
            tw, th = draw.textsize(title, font=font)
        draw.rectangle([(0, 0), (W, th + 8)], fill=(20, 20, 20))
        draw.text((6, 4), title, fill=(220, 220, 220), font=font)
        img = np.asarray(pil)

    from PIL import Image
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out_path)
    return out_path


def candidates_from_parsed(parsed: Mapping) -> list:
    """Build a candidate list suitable for ``render_annotated_frame``
    from a parsed.json-style dict produced by ``apply_to_parsed``.

    Each entity becomes one candidate with its 1-based index as the
    id, its bbox_pixels, role, palettes, and a one-line behaviour
    summary derived from properties (motion / change flags).
    """
    out = []
    for i, ent in enumerate(parsed.get("entities") or [], start=1):
        bbox = ent.get("bbox_pixels") or ent.get("bbox") or [0, 0, 0, 0]
        role = ent.get("role") or "unknown"
        pr = ent.get("properties") or {}
        disp = pr.get("total_displacement_px") or 0
        npal = pr.get("n_palette_changes", 0)
        npat = pr.get("n_pattern_changes", 0)
        bits = []
        if disp:
            bits.append(f"moved {disp:.0f}px")
        if npal or npat:
            bits.append(f"changed {npal + npat}x")
        if not bits:
            bits.append("static")
        summary = ", ".join(bits)
        out.append({
            "id":       i,
            "bbox":     list(bbox),
            "role":     role,
            "palettes": list(ent.get("palettes") or []),
            "summary":  summary,
        })
    return out
