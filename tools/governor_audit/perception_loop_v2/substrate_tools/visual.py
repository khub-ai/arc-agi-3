"""The bundled "perception/visual" substrate tools.

Each is a game-agnostic measurement/render over the logical frame, registered
via @tool so it is dispatchable by name and auto-advertised in the perception
prompt.  Handlers take a ToolContext + the VLM's query args (**kwargs) and return
a JSON-serialisable dict; rendering tools write a PNG to ctx.out_dir.  This is
the reference implementation contributors copy from — see
docs/CONTRIBUTING_substrate_tools.md.
"""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

from .frameutils import (
    background_rgb, clamp_bbox, connected_components, dominant_rgb, font, hexof,
)
from .registry import ToolContext, tool


# -----------------------------------------------------------------------------
# Rendering tools (write a PNG, return its filename)
# -----------------------------------------------------------------------------


@tool(name="zoom", category="perception/visual", renders_image=True,
      summary=("magnified, tick-labelled image of a bbox — use WHENEVER an "
               "element may contain smaller structure you can't resolve (small "
               "cursor/label/centre cells, keys, legends, glyphs, text)"),
      usage='{"op":"zoom","id":"k","bbox":[r0,c0,r1,c1]}',
      params={"bbox": "[row_top,col_left,row_bottom,col_right] region to magnify",
              "target_px": "optional target image size (default 820)",
              "pad": "optional ticks of context on each side (default 1)"})
def zoom(ctx: ToolContext, *, bbox, target_px=820, pad=1, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    pad = int(pad)
    pr0, pc0 = max(0, r0 - pad), max(0, c0 - pad)
    pr1, pc1 = min(n_ticks, r1 + pad), min(n_ticks, c1 + pad)
    crop = frame[pr0:pr1, pc0:pc1]
    h_ticks, w_ticks = crop.shape[0], crop.shape[1]
    factor = min(64, max(4, int(round(int(target_px) / max(h_ticks, w_ticks)))))
    up = Image.fromarray(crop, "RGB").resize(
        (w_ticks * factor, h_ticks * factor), Image.NEAREST)
    margin = 26
    canvas = Image.new("RGB", (up.width + margin, up.height + margin),
                       (18, 18, 18))
    canvas.paste(up, (margin, margin))
    draw = ImageDraw.Draw(canvas)
    fnt = font(11)
    for j in range(w_ticks + 1):
        x = margin + j * factor
        draw.line([(x, margin), (x, canvas.height)], fill=(90, 90, 90), width=1)
    for i in range(h_ticks + 1):
        y = margin + i * factor
        draw.line([(margin, y), (canvas.width, y)], fill=(90, 90, 90), width=1)
    for j in range(w_ticks):
        draw.text((margin + j * factor + 2, 6), str(pc0 + j),
                  fill=(210, 210, 210), font=fnt)
    for i in range(h_ticks):
        draw.text((3, margin + i * factor + factor // 2 - 6), str(pr0 + i),
                  fill=(210, 210, 210), font=fnt)
    out_path = ctx.out_dir / f"zoom_{ctx.query_id}.png"
    canvas.save(out_path)
    return {"image": out_path.name, "region_ticks": [r0, c0, r1, c1],
            "shown_ticks": [pr0, pc0, pr1, pc1], "upscale_factor": factor,
            "note": ("Magnified view; margin labels are absolute tick "
                     "coordinates. Read any internal/nested structure here.")}


@tool(name="highlight", category="perception/visual", renders_image=True,
      summary=("whole-frame image with the given regions outlined and "
               "everything else dimmed — focus attention / confirm extents"),
      usage='{"op":"highlight","id":"k","regions":[[r0,c0,r1,c1],...]}',
      params={"regions": "list of [row_top,col_left,row_bottom,col_right] bboxes",
              "bbox": "a single region (alternative to regions)",
              "dim": "optional dim factor for the rest (default 0.35)"})
def highlight(ctx: ToolContext, *, regions=None, bbox=None, dim=0.35, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    regions = regions if regions else ([bbox] if bbox is not None else [])
    factor = max(4, int(round(900 / n_ticks)))
    full = Image.fromarray(frame, "RGB").resize(
        (n_ticks * factor, n_ticks * factor), Image.NEAREST)
    dimmed = Image.fromarray(
        (np.asarray(full, dtype=np.float32) * float(dim)).astype(np.uint8), "RGB")
    boxes = []
    for reg in regions:
        r0, c0, r1, c1 = clamp_bbox(reg, n_ticks)
        boxes.append((r0, c0, r1, c1))
        crop = full.crop((c0 * factor, r0 * factor, c1 * factor, r1 * factor))
        dimmed.paste(crop, (c0 * factor, r0 * factor))
    draw = ImageDraw.Draw(dimmed)
    for (r0, c0, r1, c1) in boxes:
        draw.rectangle([c0 * factor, r0 * factor, c1 * factor - 1,
                        r1 * factor - 1], outline=(0, 255, 200), width=3)
    out_path = ctx.out_dir / f"highlight_{ctx.query_id}.png"
    dimmed.save(out_path)
    return {"image": out_path.name, "regions_ticks": [list(b) for b in boxes]}


# -----------------------------------------------------------------------------
# Numeric / geometric tools (return JSON facts)
# -----------------------------------------------------------------------------


def _panel_field_nonbg(region):
    """PALETTE-INVARIANT figure-ground for a region (Adversarial Test): the FIELD
    is the single LARGEST connected uniform-colour component (a board/panel),
    decided by STRUCTURE -- never 'the dominant/most-common colour' (a single-
    colour-background assumption that breaks on a re-skin or a two-tone panel).
    Returns (nonbg_mask, field_rgb_tuple)."""
    import silhouette_track as _ST
    reg = np.ascontiguousarray(region).astype(int)
    if reg.size == 0:
        return np.zeros(reg.shape[:2], dtype=bool), (0, 0, 0)
    packed = (reg[:, :, 0] << 16) | (reg[:, :, 1] << 8) | reg[:, :, 2]
    field, fcol = None, 0
    for cv in (int(v) for v in np.unique(packed)):
        for comp in _ST.connected_components(packed == cv):
            if field is None or comp[6] > field[6]:
                field, fcol = comp, cv
    nonbg = np.ones(packed.shape, dtype=bool)
    if field is not None:
        nonbg[field[0]:field[2] + 1, field[1]:field[3] + 1] &= ~np.asarray(field[7], dtype=bool)
    return nonbg, (fcol >> 16 & 255, fcol >> 8 & 255, fcol & 255)


def _parse_color(c):
    """Coerce a VLM-supplied colour ('#rrggbb' or [r,g,b]) to an (r,g,b) tuple, or
    None.  Lets the caller ASSERT the set/'bar' tone; the substrate just measures
    where that exact tone appears."""
    if c is None:
        return None
    try:
        if isinstance(c, str):
            s = c.strip().lstrip("#")
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        return (int(c[0]), int(c[1]), int(c[2]))
    except Exception:
        return None


def _luma(pixels):
    """ITU-R BT.601 luma for one RGB or an (N,3) array of RGB -- the brightness
    used to RANK tones (field brightest .. set-bar darkest).  Threshold-free."""
    p = np.asarray(pixels, dtype=float).reshape(-1, 3)
    out = 0.299 * p[:, 0] + 0.587 * p[:, 1] + 0.114 * p[:, 2]
    return out


def _max_bar_run(mask2d) -> int:
    """Longest contiguous run of True along any single row OR column of a 2D bool
    mask -- the length of the longest straight BAR.  A SET switch draws a bar (a
    long run, horizontal or vertical); a faint lattice tick or a stray antialias
    pixel does not.  Used so a thin set-bar diluted by a panel BORDER (which makes
    it a MINORITY of the cell's figure) still reads as set."""
    if mask2d is None or np.asarray(mask2d).size == 0:
        return 0
    m = np.asarray(mask2d, dtype=bool)
    best = 0
    for line in list(m) + list(m.T):
        run = 0
        for v in line:
            run = run + 1 if v else 0
            if run > best:
                best = run
    return best


def _panel_mark_tones(region, nonbg):
    """The panel's distinct NON-FIELD tones + the luminance that splits the darker
    'bar' (set) tones from the lighter 'lattice' (unset) tones.

    A switch panel layers up to three structural tones: the FIELD (the bright
    backdrop, already removed in ``nonbg``), a faint LATTICE tick that marks every
    cell, and a bolder, DARKER BAR that marks the *set* cells.  The lattice and the
    bar both 'fill' a cell, so fill cannot tell them apart -- but they are distinct
    TONES.  Split them at the LARGEST luminance GAP among the non-field tones
    (derived from the measured pixels, never a baked colour or magic cut); the
    darker side is the 'bar/set' tone.

    Returns ``(tones, set_lum_max)`` where ``tones`` is a darkest-first list of
    ``{hex, luma, pixels, dark}`` and ``set_lum_max`` is the split luminance (a
    mark <= it is the dark/bar tone).  ``set_lum_max`` is ``None`` when fewer than
    two non-field tones exist -- a single uniform mark cannot be read as a bar vs a
    tick without a reference, so ``set`` is then undecidable."""
    ys, xs = np.where(nonbg)
    if len(ys) == 0:
        return [], None
    px = np.asarray(region)[ys, xs].reshape(-1, 3)
    colors, counts = np.unique(px, axis=0, return_counts=True)
    lums = _luma(colors)
    order = np.argsort(lums)                       # darkest tone first
    colors, counts, lums = colors[order], counts[order], lums[order]
    set_lum_max = None
    if len(colors) >= 2:
        k = int(np.argmax(np.diff(lums)))          # the dark|light boundary gap
        set_lum_max = float((lums[k] + lums[k + 1]) / 2.0)
    tones = [{"hex": hexof(colors[i]), "luma": round(float(lums[i]), 1),
              "pixels": int(counts[i]),
              "dark": (None if set_lum_max is None else bool(lums[i] <= set_lum_max))}
             for i in range(len(colors))]
    return tones, set_lum_max


@tool(name="count", category="perception/visual",
      summary="exact count in a bbox: connected components | distinct colours | non-background cells",
      usage='{"op":"count","id":"k","bbox":[r0,c0,r1,c1],"what":"components"}',
      params={"bbox": "the region to count in",
              "what": "'components' (default) | 'colors' | 'nonbg_cells'"})
def count(ctx: ToolContext, *, bbox, what="components", **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    region = frame[r0:r1, c0:c1]
    if what == "colors":
        cols = np.unique(region.reshape(-1, 3), axis=0)
        return {"what": "colors", "region_ticks": [r0, c0, r1, c1],
                "count": int(cols.shape[0])}
    nonbg, bg = _panel_field_nonbg(region)   # palette-invariant: largest comp = field
    if what == "nonbg_cells":
        return {"what": "nonbg_cells", "region_ticks": [r0, c0, r1, c1],
                "count": int(nonbg.sum()), "background": hexof(bg)}
    return {"what": "components", "region_ticks": [r0, c0, r1, c1],
            "count": connected_components(nonbg), "background": hexof(bg)}


@tool(name="components", category="perception/visual",
      summary="split a region into its connected non-background UNITS -- the individual repeated marks of a panel/grid (each a candidate entity / control); returns each unit's bbox + centroid + cell-count + dominant colour, largest-first. Use to enumerate the individual switches/cells of a control panel you perceived as one region.",
      usage='{"op":"components","id":"k","bbox":[r0,c0,r1,c1]}',
      params={"bbox": "the region (e.g. a control-group panel) to split into units",
              "min_cells": "ignore units smaller than this many cells (default 1)",
              "max_return": "cap on units returned, largest-first (default 64)"})
def components(ctx: ToolContext, *, bbox, min_cells=1, max_return=64, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    region = frame[r0:r1, c0:c1]
    # PALETTE-INVARIANT (Adversarial Test): the panel FIELD is the single LARGEST
    # connected uniform-colour component (by STRUCTURE), so the units returned are
    # the marks ON the panel -- not the panel itself; never 'the region's dominant
    # colour'.  N.B. unlike entity extraction we do NOT lattice-suppress here:
    # enumerating the repeated marks is this tool's job.
    nonbg, bg = _panel_field_nonbg(region)
    h, w = nonbg.shape
    seen = np.zeros_like(nonbg, bool)
    comps = []
    for y0 in range(h):
        for x0 in range(w):
            if not nonbg[y0, x0] or seen[y0, x0]:
                continue
            seed_rgb = tuple(int(v) for v in region[y0, x0])
            st = [(y0, x0)]; seen[y0, x0] = True; cells = []
            while st:                                   # 4-connected flood fill, COLOUR-AWARE:
                cy, cx = st.pop(); cells.append((cy, cx))   # merge only SAME-colour neighbours, so a
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):   # distinct-colour piece sitting ON a
                    ny, nx = cy + dy, cx + dx                       # field (room/panel) separates from it
                    if (0 <= ny < h and 0 <= nx < w and nonbg[ny, nx] and not seen[ny, nx]
                            and tuple(int(v) for v in region[ny, nx]) == seed_rgb):
                        seen[ny, nx] = True; st.append((ny, nx))
            if len(cells) < int(min_cells):
                continue
            ys = [p[0] for p in cells]; xs = [p[1] for p in cells]
            pix = region[ys, xs]                        # the unit's own pixels
            vals, counts = np.unique(pix, axis=0, return_counts=True)
            rgb = tuple(int(v) for v in vals[counts.argmax()])
            comps.append({
                "bbox": [r0 + min(ys), c0 + min(xs), r0 + max(ys) + 1, c0 + max(xs) + 1],
                "centroid": [round(r0 + sum(ys) / len(ys), 1),
                             round(c0 + sum(xs) / len(xs), 1)],
                "cells": len(cells), "color": hexof(rgb)})
    comps.sort(key=lambda d: d["cells"], reverse=True)
    return {"region_ticks": [r0, c0, r1, c1], "background": hexof(bg),
            "n_components": len(comps), "components": comps[:int(max_return)]}


@tool(name="measure", category="perception/visual",
      summary="distance between two tick points: dr, dc, manhattan, chebyshev, euclidean + same-row/col",
      usage='{"op":"measure","id":"k","a":[r,c],"b":[r,c]}',
      params={"a": "[row,col] first point", "b": "[row,col] second point"})
def measure(ctx: ToolContext, *, a, b, **_) -> dict:
    ar, ac = int(a[0]), int(a[1])
    br, bc = int(b[0]), int(b[1])
    dr, dc = br - ar, bc - ac
    return {"a": [ar, ac], "b": [br, bc], "dr": dr, "dc": dc,
            "manhattan": abs(dr) + abs(dc), "chebyshev": max(abs(dr), abs(dc)),
            "euclidean": round(float(np.hypot(dr, dc)), 2),
            "same_row": dr == 0, "same_col": dc == 0}


@tool(name="align", category="perception/visual",
      summary="alignment verdict for a set of points: same row/col, collinear, evenly-spaced + pitch, dominant axis",
      usage='{"op":"align","id":"k","points":[[r,c],...]}',
      params={"points": "list of [row,col] points",
              "tol": "optional sub-tick slack (default 1.0)"})
def align(ctx: ToolContext, *, points, tol=1.0, **_) -> dict:
    tol = float(tol)
    pts = [(int(p[0]), int(p[1])) for p in points]
    out: dict = {"points": [list(p) for p in pts], "n": len(pts)}
    if len(pts) < 2:
        out["note"] = "need >= 2 points"
        return out
    rs = [p[0] for p in pts]
    cs = [p[1] for p in pts]
    out["same_row"] = (max(rs) - min(rs)) <= tol
    out["same_col"] = (max(cs) - min(cs)) <= tol
    out["centroid"] = [round(sum(rs) / len(rs), 1), round(sum(cs) / len(cs), 1)]
    p_a, p_b = max(itertools.combinations(pts, 2),
                   key=lambda pr: (pr[0][0] - pr[1][0]) ** 2
                   + (pr[0][1] - pr[1][1]) ** 2)
    (ar, ac), (br, bc) = p_a, p_b
    vlen = float(np.hypot(br - ar, bc - ac)) or 1.0
    max_dev = max(abs((bc - ac) * (ar - r) - (br - ar) * (ac - c)) / vlen
                  for (r, c) in pts)
    out["max_perp_deviation"] = round(max_dev, 2)
    out["collinear"] = max_dev <= tol
    out["axis"] = ("row" if out["same_row"] else "col" if out["same_col"]
                   else "diagonal" if out["collinear"] else "none")
    if out["collinear"]:
        key = (lambda p: p[1]) if abs(bc - ac) >= abs(br - ar) else (lambda p: p[0])
        ordered = sorted(pts, key=key)
        steps = [round(float(np.hypot(ordered[i + 1][0] - ordered[i][0],
                                      ordered[i + 1][1] - ordered[i][1])), 2)
                 for i in range(len(ordered) - 1)]
        out["consecutive_pitches"] = steps
        out["evenly_spaced"] = bool(steps) and (max(steps) - min(steps)) <= tol
        if out["evenly_spaced"]:
            out["pitch"] = round(sum(steps) / len(steps), 2)
    return out


@tool(name="palette", category="perception/visual",
      summary="exact distinct colours (hex) + pixel counts within a bbox, most-frequent first",
      usage='{"op":"palette","id":"k","bbox":[r0,c0,r1,c1]}',
      params={"bbox": "the region to read colours from",
              "top": "optional max colours returned (default 12)"})
def palette(ctx: ToolContext, *, bbox, top=12, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    region = frame[r0:r1, c0:c1].reshape(-1, 3)
    colours, counts = np.unique(region, axis=0, return_counts=True)
    order = np.argsort(-counts)
    items = [{"hex": hexof(colours[i]), "rgb": [int(x) for x in colours[i]],
              "pixels": int(counts[i])} for i in order[:int(top)]]
    return {"region_ticks": [r0, c0, r1, c1],
            "n_distinct": int(colours.shape[0]), "colors": items}


@tool(name="grid_readout", category="perception/visual",
      summary=("YOU assert a region holds an R x C sub-grid; returns each "
               "sub-cell's exact dominant colour (a directed sampler, NOT a "
               "detector). Ideal for reading a thumbnail/key you've zoomed"),
      usage='{"op":"grid_readout","id":"k","bbox":[r0,c0,r1,c1],"rows":R,"cols":C}',
      params={"bbox": "the region the sub-grid spans",
              "rows": "number of sub-rows you assert", "cols": "number of sub-cols"})
def grid_readout(ctx: ToolContext, *, bbox, rows, cols, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    rows, cols = max(1, int(rows)), max(1, int(cols))
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    rr = np.linspace(r0, r1, rows + 1)
    cc = np.linspace(c0, c1, cols + 1)
    hexgrid: list = []
    for i in range(rows):
        row_hex = []
        for j in range(cols):
            sub = frame[int(round(rr[i])):max(int(round(rr[i])) + 1,
                                              int(round(rr[i + 1]))),
                        int(round(cc[j])):max(int(round(cc[j])) + 1,
                                              int(round(cc[j + 1])))]
            row_hex.append(hexof(dominant_rgb(sub)))
        hexgrid.append(row_hex)
    return {"region_ticks": [r0, c0, r1, c1], "rows": rows, "cols": cols,
            "region_most_common": hexof(background_rgb(frame[r0:r1, c0:c1])),
            "dominant_hex": hexgrid,
            "note": ("Per-sub-cell dominant colour for the sub-grid YOU "
                     "specified; the substrate measured colour only — you "
                     "decide which tone is 'on' and what the pattern means.")}


@tool(name="decode_panel", category="perception/visual",
      summary=("DETECT a switch/control panel's cell lattice and return, per cell, "
               "its dominant colour, fill, a per-cell SET flag (the dark 'bar' tone "
               "told apart from the faint 'lattice' tick and the field) AND the "
               "[col,row] CENTRE tick to CLICK to toggle it — for reading AND "
               "programming panels. Substrate measures geometry+tone; YOU decide "
               "what 'set' means and which to set. Use on BOTH a reference and a "
               "program panel, then map one to the other yourself."),
      usage='{"op":"decode_panel","id":"k","bbox":[r0,c0,r1,c1]}',
      params={"bbox": "the panel region to decode",
              "rows": "optional: assert the row count (else the lattice is detected)",
              "cols": "optional: assert the col count (else detected)",
              "set_color": ("optional hex/[r,g,b]: assert the SET/'bar' tone (e.g. "
                            "the colour a reference shows for an active cell); a "
                            "cell reads set iff it carries that exact tone. Omit to "
                            "auto-split the panel's tones into bar(dark)/tick(light).")})
def decode_panel(ctx: ToolContext, *, bbox, rows=None, cols=None, set_color=None, **_) -> dict:
    frame, n_ticks = ctx.frame, ctx.n_ticks
    r0, c0, r1, c1 = clamp_bbox(bbox, n_ticks)
    # DETECT the lattice (structural_grid) unless the VLM asserts rows x cols.
    # Detection clusters the MARKS, so a sparsely-set panel may under-detect --
    # then `detected` is False and the VLM should re-call asserting rows x cols.
    cellboxes, nr, nc, detected = None, None, None, False
    if not (rows and cols):
        try:
            import structural_grid as _sg
            g = _sg.decompose(frame, [r0, c0, r1, c1])
            if g:
                nr, nc, cellboxes = g["n_rows"], g["n_cols"], g["cells"]
                detected = True
        except Exception:
            cellboxes = None
    if cellboxes is None:                              # asserted, or detection failed
        nr = max(1, int(rows or 1)); nc = max(1, int(cols or 1))
        rr = np.linspace(r0, r1, nr + 1); cc = np.linspace(c0, c1, nc + 1)
        cellboxes = [[int(round(rr[i])), int(round(cc[j])),
                      int(round(rr[i + 1])), int(round(cc[j + 1]))]
                     for i in range(nr) for j in range(nc)]
    # structural (non-modal-bg) figure-ground for the fill measurement
    region = frame[r0:r1, c0:c1]
    nonbg, field = _panel_field_nonbg(region)
    # Tell the dark 'bar' (set) tone apart from the faint 'lattice' (unset) tone.
    # Either the VLM asserts the set tone (set_color) or we auto-split the panel's
    # tones at the largest luminance gap -- both data-grounded, no baked colour.
    set_rgb = _parse_color(set_color)
    mark_tones, set_lum_max = _panel_mark_tones(region, nonbg)
    cells = []
    for idx, (cr0, cc0, cr1, cc1) in enumerate(cellboxes):
        row, col = divmod(idx, nc)
        cr1e, cc1e = max(cr0 + 1, cr1), max(cc0 + 1, cc1)
        sub = frame[cr0:cr1e, cc0:cc1e]
        sl = nonbg[max(0, cr0 - r0):max(1, cr1e - r0), max(0, cc0 - c0):max(1, cc1e - c0)]
        # the MARK = the foreground pixels in this cell.  Report the mark's OWN
        # colour (the switch STATE -- a gray level a field-dominated 'color' hides),
        # and a per-cell SET flag distinguishing the dark bar from the faint tick.
        cell = {"row": int(row), "col": int(col),
                "color": hexof(dominant_rgb(sub)),
                "fill": round(float(sl.mean()) if sl.size else 0.0, 3),
                "bbox": [int(cr0), int(cc0), int(cr1e), int(cc1e)]}
        ys, xs = np.where(sl)
        if len(ys):
            msub = sub[ys, xs]
            cell["mark_color"] = hexof(Counter(map(tuple, msub.tolist())).most_common(1)[0][0])
            cell["mark_bbox"] = [int(cr0 + ys.min()), int(cc0 + xs.min()),
                                 int(cr0 + ys.max() + 1), int(cc0 + xs.max() + 1)]
            # click the MARK CENTROID -- always the cell INTERIOR, never a cell/box
            # boundary (the box geometric centre can round onto a top/edge tick and
            # miss the cell, the row-boundary toggle-miss bug).
            mcr = int(cr0 + round(float(ys.mean())))
            mcc = int(cc0 + round(float(xs.mean())))
            # SET := the cell carries the dark 'bar' tone AS A BAR (a contiguous
            # run), NOT merely as a MAJORITY of the cell's figure.  A thin set-bar
            # on a panel's top/edge row is diluted by the panel BORDER pixels (also
            # non-field) to a minority -- the old majority test then read a real
            # SET bar as unset (the tn36 reference top-row miss that produced an
            # incomplete code + a canned fire).  Contiguity is immune to that:
            # a SET cell draws a bar (run >= 2), a faint lattice tick / stray pixel
            # does not.  set_color when asserted, else the dark side of the split.
            if set_rgb is not None:
                dm = np.all(sub == np.asarray(set_rgb), axis=2)
            elif set_lum_max is not None:
                lum2d = (0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1]
                         + 0.114 * sub[:, :, 2])
                dm = lum2d <= set_lum_max
            else:
                dm = None                              # single tone: undecidable
            if dm is None:
                cell["dark_fill"] = None
                cell["set"] = None
            else:
                h = min(dm.shape[0], sl.shape[0]); w = min(dm.shape[1], sl.shape[1])
                dm = dm[:h, :w] & sl[:h, :w]
                figure = int(sl[:h, :w].sum())
                cell["dark_fill"] = round(int(dm.sum()) / max(1, figure), 3)
                cell["set"] = bool(_max_bar_run(dm) >= 2)
        else:
            mcr = int((cr0 + cr1e - 1) // 2)
            mcc = int((cc0 + cc1e - 1) // 2)
            cell["mark_color"] = None
            cell["dark_fill"] = 0.0
            cell["set"] = False                        # bare field -- definitely off
        # clamp the click into the STRICT cell interior (never an outer edge row/col)
        cell["click_row"] = int(min(max(mcr, cr0 + 1), max(cr0 + 1, cr1e - 2)))
        cell["click_col"] = int(min(max(mcc, cc0 + 1), max(cc0 + 1, cc1e - 2)))
        cell["mark_click_row"], cell["mark_click_col"] = cell["click_row"], cell["click_col"]
        cells.append(cell)
    n_set = sum(1 for c in cells if c["set"])
    undecided = any(c["set"] is None for c in cells)
    return {"region_ticks": [r0, c0, r1, c1], "rows": int(nr), "cols": int(nc),
            "detected": bool(detected), "field_hex": hexof(field),
            "mark_tones": mark_tones, "set_lum_max": set_lum_max, "n_set": int(n_set),
            "cells": cells,
            "note": ("Detected switch lattice: per cell the dominant colour, the "
                     "fill, a SET flag (True=the dark 'bar' tone, False=field/faint "
                     "'lattice' tick, None=undecidable from one tone — pass "
                     "set_color or compare a reference), and the [click_col,"
                     "click_row] CELL-CENTRE tick to CLICK to toggle it. Substrate "
                     "measured geometry+tone only — YOU decide what 'set' MEANS and "
                     "which to set to program the required steps."
                     + (" NOTE: some cells share a single mark tone (set "
                        "undecidable); assert set_color or compare a reference."
                        if undecided else ""))}
