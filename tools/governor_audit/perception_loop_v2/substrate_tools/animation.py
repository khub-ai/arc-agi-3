"""Animation-inspection substrate tools (the "perception/animation" family).

The engine plays a multi-frame ANIMATION per action; the motion across those
sub-frames is usually the mechanic.  The driver already shows the VLM a coarse
filmstrip of the whole frames; these tools let the VLM inspect the animation in
DETAIL on demand — magnify a region across ALL sub-frames so fine motion (a small
ball launching, a cursor stepping, a tile flashing) is readable.  Game-agnostic:
measures/renders only; the VLM interprets the motion (object constancy across
frames = entity movement).

These operate on ctx.anim_frames (the last action's sub-frames, oldest->settled),
which the driver passes for delta turns; on a turn with no animation they return
an informative error instead of failing.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from .frameutils import clamp_bbox, font
from .registry import ToolContext, tool


@tool(name="animation_zoom", category="perception/animation", renders_image=True,
      summary=("magnify a bbox region across ALL sub-frames of the last action's "
               "animation (a per-frame filmstrip of that region) — read fine "
               "motion in detail; object constancy across frames = entity movement"),
      usage='{"op":"animation_zoom","id":"k","bbox":[r0,c0,r1,c1]}',
      params={"bbox": "[row_top,col_left,row_bottom,col_right] region to track",
              "target_px": "optional per-frame tile size (default 280)",
              "pad": "optional ticks of context around the region (default 1)"})
def animation_zoom(ctx: ToolContext, *, bbox, target_px=280, pad=1, **_) -> dict:
    frames = ctx.anim_frames
    if not frames or len(frames) < 2:
        return {"error": ("no animation this turn — the last action produced a "
                          "single settled frame, so there is no motion to zoom.")}
    n_ticks = ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    pad = int(pad)
    pr0, pc0 = max(0, r0 - pad), max(0, c0 - pad)
    pr1, pc1 = min(n_ticks, r1 + pad), min(n_ticks, c1 + pad)
    h_ticks, w_ticks = pr1 - pr0, pc1 - pc0
    factor = min(48, max(4, int(round(int(target_px) / max(h_ticks, w_ticks)))))
    margin, lab, gap = 22, 18, 8
    fnt = font(13)

    tiles = []
    for fr in frames:
        crop = fr[pr0:pr1, pc0:pc1]
        up = Image.fromarray(crop, "RGB").resize(
            (w_ticks * factor, h_ticks * factor), Image.NEAREST)
        canvas = Image.new("RGB", (up.width + margin, up.height + margin),
                           (18, 18, 18))
        canvas.paste(up, (margin, margin))
        dr = ImageDraw.Draw(canvas)
        for j in range(w_ticks + 1):
            x = margin + j * factor
            dr.line([(x, margin), (x, canvas.height)], fill=(90, 90, 90), width=1)
        for i in range(h_ticks + 1):
            y = margin + i * factor
            dr.line([(margin, y), (canvas.width, y)], fill=(90, 90, 90), width=1)
        for j in range(w_ticks):
            dr.text((margin + j * factor + 1, 5), str(pc0 + j),
                    fill=(200, 200, 200), font=font(10))
        for i in range(h_ticks):
            dr.text((2, margin + i * factor + factor // 2 - 5), str(pr0 + i),
                    fill=(200, 200, 200), font=font(10))
        tiles.append(canvas)

    n = len(tiles)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    tw, th = tiles[0].size
    W = cols * tw + (cols + 1) * gap
    H = rows * (th + lab) + (rows + 1) * gap
    out = Image.new("RGB", (W, H), (12, 12, 18))
    od = ImageDraw.Draw(out)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        x = gap + c * (tw + gap)
        y = gap + r * (th + lab + gap)
        tag = " (settled)" if i == n - 1 else ""
        od.text((x + 2, y + 2), f"frame {i}{tag}", fill=(240, 240, 140), font=fnt)
        out.paste(tile, (x, y + lab))
    out_path = ctx.out_dir / f"animzoom_{ctx.query_id}.png"
    out.save(out_path)
    return {"image": out_path.name, "n_frames": n,
            "region_ticks": [r0, c0, r1, c1], "shown_ticks": [pr0, pc0, pr1, pc1],
            "note": ("Each tile is the SAME region at successive animation frames "
                     "(frame 0 -> settled); tick labels are absolute coordinates. "
                     "Track each object's position across the tiles to read its "
                     "motion.")}
