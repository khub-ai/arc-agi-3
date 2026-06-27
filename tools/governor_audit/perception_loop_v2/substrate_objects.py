"""Substrate object-set extraction: the clean, MEASURED list of distinct objects in
a frame -- figure-ground components, regrouped per colour with 8-connectivity + a 1px
close (so an over-split chain / gapped outline is ONE object), with frame-spanning
micro-textures (dotted backgrounds) suppressed.

This is the substrate's authoritative answer to "which objects exist and where" --
pure geometry, no meaning. The VLM assigns roles to these; it does not have to find or
localize them (a small VLM can name kinds but cannot localize on a 64px grid).

Shared by the benchmark generator (make_case) and COS perception (cos_responder).
Pure numpy/scipy; returns [] without them.
"""
from __future__ import annotations

from typing import Any, Dict, List

try:
    import numpy as np
    from scipy import ndimage
    import silhouette_track as _st
    _OK = True
    _8CONN = ndimage.generate_binary_structure(2, 2)
except Exception:                                       # pragma: no cover
    _OK = False


def _regroup(rgb, comps, gap: int = 1) -> List[Dict[str, Any]]:
    """Re-extract components PER COLOUR with 8-connectivity and a 1px close, so a
    diagonally-stepping chain or a 1px-gapped outline becomes ONE object. Only the
    original-colour pixels are kept; the close only bridges the grouping."""
    n = rgb.shape[0]
    fg = np.zeros((n, n), bool)
    for c in comps:
        r0, c0, r1, c1 = c["bbox"]
        fg[r0:r1 + 1, c0:c1 + 1] |= np.asarray(c["mask"], bool)
    cid = (rgb[..., 0].astype(int) << 16) | (rgb[..., 1].astype(int) << 8) | rgb[..., 2].astype(int)
    out: List[Dict[str, Any]] = []
    for col in np.unique(cid[fg]).tolist():
        mask = fg & (cid == col)
        bridged = mask
        if gap > 0:
            bridged = ndimage.binary_dilation(mask, _8CONN, iterations=gap)
            bridged = ndimage.binary_erosion(bridged, _8CONN, iterations=gap) | mask
        lab, nl = ndimage.label(bridged, structure=_8CONN)
        for i in range(1, nl + 1):
            cm = (lab == i) & mask
            if not cm.any():
                continue
            ys, xs = np.where(cm)
            r0, c0, r1, c1 = int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())
            out.append({"bbox": [r0, c0, r1, c1], "mask": cm[r0:r1 + 1, c0:c1 + 1],
                        "npix": int(cm.sum()),
                        "color": f"#{(col >> 16) & 255:02x}{(col >> 8) & 255:02x}{col & 255:02x}"})
    return out


def _texture_colors(grouped, n) -> Dict[str, int]:
    """Colours that are a frame-spanning field of many tiny components -- a background
    micro-texture, not game objects. Conservative: >=12 components, each <=2px, union
    spanning >= half the frame on both axes. A few localised small pieces are kept."""
    from collections import defaultdict
    by = defaultdict(list)
    for g in grouped:
        by[g["color"]].append(g)
    tex: Dict[str, int] = {}
    for col, gs in by.items():
        if len(gs) >= 12 and all(g["npix"] <= 2 for g in gs):
            r0 = min(g["bbox"][0] for g in gs); c0 = min(g["bbox"][1] for g in gs)
            r1 = max(g["bbox"][2] for g in gs); c1 = max(g["bbox"][3] for g in gs)
            if (r1 - r0) >= n * 0.5 and (c1 - c0) >= n * 0.5:
                tex[col] = len(gs)
    return tex


def extract_objects(rgb) -> List[Dict[str, Any]]:
    """The clean measured object set for a frame (RGB HxWx3 uint8).
    Returns [{id, bbox:[r0,c0,r1,c1] inclusive, center:[r,c], npix, color}], with
    over-split fragments grouped and background micro-textures dropped. [] on error."""
    if not _OK or rgb is None:
        return []
    try:
        rgb = np.asarray(rgb)
        if rgb.ndim != 3:
            return []
        n = rgb.shape[0]
        comps = _st.foreground_components(rgb)
        grouped = _regroup(rgb, comps)
        tex = _texture_colors(grouped, n)
        kept = [g for g in grouped if g["color"] not in tex]
        out = []
        for i, g in enumerate(kept):
            r0, c0, r1, c1 = g["bbox"]
            out.append({"id": f"obj{i}", "bbox": [r0, c0, r1, c1],
                        "center": [(r0 + r1) // 2, (c0 + c1) // 2],
                        "npix": g["npix"], "color": g["color"], "mask": g["mask"]})
        return out
    except Exception:
        return []
