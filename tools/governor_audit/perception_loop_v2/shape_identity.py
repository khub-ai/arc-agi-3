"""Rotation- and scale-invariant entity SHAPE identity.

An entity's IDENTITY ("is this the same kind of glyph as before?") is its SHAPE
up to ROTATION and SCALE -- an arch (⊓) is a cup (U) rotated 180°, and the same
mover recurs at different sizes across levels.  This module turns an entity's
crop into a compact, scale-normalised binary SHAPE SIGNATURE and compares two
signatures rotation-invariantly (8 orientations: 4 rotations x mirror).

IDENTITY (shape, here) is deliberately kept SEPARATE from ROLE (behaviour /
position, decided elsewhere): because mover and goal can be rotations of each
other, shape CANNOT tell them apart -- only behaviour can.  See
feedback_entity_crop_and_rotation_invariant_id.

Game/colour-agnostic: the foreground is figure-ground by border background, not
a colour key (palette-invariant per feedback_palette_invariant_figure_ground).
No tuned thresholds: similarity() returns a 0..1 IoU; callers decide.
"""
from __future__ import annotations

from typing import Optional

try:
    import numpy as np
    _OK = True
except Exception:                                    # pragma: no cover
    _OK = False

GRID = 16          # canonical shape resolution (this is scale-normalisation, not a tuned knob)
_BG_TOL = 24       # colour distance for "differs from background" (figure-ground, not a match knob)


def _foreground_mask(frame_rgb, bbox) -> "Optional[np.ndarray]":
    """Binary foreground of the entity inside its (tight) bbox.  Background =
    the most common colour around the crop BORDER (figure-ground, no colour key);
    foreground = pixels that differ from it.  Bottom/right EXCLUSIVE."""
    arr = np.asarray(frame_rgb)
    if arr.ndim != 3:
        return None
    H, W = arr.shape[:2]
    r0, c0, r1, c1 = [int(v) for v in bbox]
    orig = arr[max(0, r0):min(H, r1), max(0, c0):min(W, c1), :3]
    if orig.size == 0 or orig.shape[0] < 1 or orig.shape[1] < 1:
        return None
    # BACKGROUND PALETTE = every colour in the surrounding RING (the padded area
    # MINUS the bbox itself), foreground computed over the ORIGINAL tight bbox.
    # Sampling the whole ring (not just a 1-px border) captures a MULTI-COLOUR
    # background (e.g. a checkerboard's two shades) -- so a checker square showing
    # through a hollow shape's opening reads as background, not foreground.  And
    # computing fg on the tight bbox avoids both figure-ground inversion (when the
    # entity fills its own border) and aspect distortion.
    pad = 3
    pr0, pc0 = max(0, r0 - pad), max(0, c0 - pad)
    pr1, pc1 = min(H, r1 + pad), min(W, c1 + pad)
    padded = arr[pr0:pr1, pc0:pc1, :3]
    ring_mask = np.ones(padded.shape[:2], bool)
    ring_mask[r0 - pr0:r1 - pr0, c0 - pc0:c1 - pc0] = False     # exclude the entity bbox
    ring_px = padded[ring_mask]
    if ring_px.size == 0:
        ring_px = padded.reshape(-1, 3)
    bg_cols = np.unique(ring_px, axis=0).astype(int)           # (K, 3) full bg palette
    d = np.abs(orig.astype(int)[:, :, None, :] - bg_cols[None, None, :, :]).sum(3)
    return d.min(axis=2) > _BG_TOL                              # fg = unlike EVERY bg colour


def _canonical(mask: "np.ndarray") -> "np.ndarray":
    """Tight-crop the foreground and nearest-neighbour resize to GRIDxGRID
    (scale-normalised)."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((GRID, GRID), bool)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    ri = (np.arange(GRID) * h // GRID).clip(0, h - 1)
    ci = (np.arange(GRID) * w // GRID).clip(0, w - 1)
    return m[ri][:, ci]


def _serialize(cs: "np.ndarray") -> str:
    return "".join("1" if v else "0" for v in cs.flatten())


def _deserialize(s: str) -> "Optional[np.ndarray]":
    if not _OK or not s or len(s) != GRID * GRID:
        return None
    return np.array([c == "1" for c in s], dtype=bool).reshape(GRID, GRID)


def _component_mask(frame_rgb, bbox):
    """The entity's foreground via the substrate's GLOBAL figure-ground
    (silhouette_track.foreground_components), which drops fields/texture (e.g. a
    checkerboard) frame-wide -- so a checker square showing through a hollow
    shape's opening is correctly background.  Returns the mask of the component
    with the most overlap with ``bbox``, or None.  Falls back to None on any
    issue (caller then uses the local border-ring method)."""
    try:
        import silhouette_track as _st
        r0, c0, r1, c1 = [int(v) for v in bbox]
        comps = _st.foreground_components(frame_rgb)
        best, best_ov = None, 0
        for c in comps:
            br0, bc0, br1, bc1 = c["bbox"]                     # inclusive
            ov = (max(0, min(r1 - 1, br1) - max(r0, br0) + 1) *
                  max(0, min(c1 - 1, bc1) - max(c0, bc0) + 1))
            if ov > best_ov:
                best, best_ov = c, ov
        return best["mask"] if best is not None and best_ov > 0 else None
    except Exception:
        return None


def shape_signature(frame_rgb, bbox) -> str:
    """Scale-normalised binary shape signature (GRID*GRID chars) for an entity
    crop, or '' if it can't be computed.  Prefers the substrate's global
    figure-ground (handles textured backgrounds); falls back to a local
    border-ring figure-ground.  Guarded -- never raises."""
    if not _OK or frame_rgb is None or bbox is None:
        return ""
    try:
        m = _component_mask(frame_rgb, bbox)
        if m is None or not np.asarray(m).any():
            m = _foreground_mask(frame_rgb, bbox)
        if m is None or not np.asarray(m).any():
            return ""
        return _serialize(_canonical(np.asarray(m, dtype=bool)))
    except Exception:
        return ""


def _orientations(cs: "np.ndarray"):
    """The 8 dihedral orientations: 4 rotations of the shape and of its mirror."""
    for base in (cs, np.fliplr(cs)):
        m = base
        for _ in range(4):
            yield m
            m = np.rot90(m)


def similarity(sig_a: str, sig_b: str) -> float:
    """Rotation/mirror- and scale-invariant shape similarity (0..1).

    Combines the best-orientation IoU of the two scale-normalised silhouettes
    with a FILL-RATIO agreement factor (foreground fraction) -- so a HOLLOW
    bracket and a SOLID blob, which can have a deceptively high silhouette IoU
    when one contains the other, are correctly separated.  Both terms come from
    the signatures themselves; no tuned threshold."""
    a = _deserialize(sig_a)
    b = _deserialize(sig_b)
    if a is None or b is None:
        return 0.0
    best = 0.0
    for v in _orientations(b):
        inter = int(np.logical_and(a, v).sum())
        union = int(np.logical_or(a, v).sum())
        if union:
            best = max(best, inter / union)
    fill_a = a.mean()
    fill_b = b.mean()
    fill_agree = 1.0 - abs(fill_a - fill_b)        # 1 when equally dense
    return float(round(best * fill_agree, 4))


def color_crop_b64(frame_rgb, bbox, max_side: int = 48) -> str:
    """A base64 PNG of the entity's COLOUR crop (the actual cropped image), for
    carrying in knowledge so a human or VLM can VISUALLY compare identity -- text
    is lossy (this is the 'keep a cropped image of the entity' principle).
    Stores the raw bbox pixels (entities are small); guarded -- '' on any issue."""
    if not _OK or frame_rgb is None or bbox is None:
        return ""
    try:
        import base64
        import io
        from PIL import Image
        r0, c0, r1, c1 = [int(v) for v in bbox]
        arr = np.asarray(frame_rgb)
        H, W = arr.shape[:2]
        crop = arr[max(0, r0):min(H, r1), max(0, c0):min(W, c1), :3]
        if crop.size == 0:
            return ""
        im = Image.fromarray(crop.astype("uint8"), "RGB")
        if max(im.size) > max_side:                       # cap pathological sizes
            im.thumbnail((max_side, max_side), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:
        return ""


def decode_crop_b64(s: str):
    """Decode a color_crop_b64 string back to an HxWx3 uint8 array, or None."""
    if not _OK or not s:
        return None
    try:
        import base64
        import io
        from PIL import Image
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB"))
    except Exception:
        return None


def best_match(sig: str, candidates) -> tuple:
    """candidates: iterable of (key, sig).  Returns (key, score) of the best
    rotation-invariant match, or (None, 0.0)."""
    best_k, best_s = None, 0.0
    for k, s in candidates:
        sc = similarity(sig, s)
        if sc > best_s:
            best_k, best_s = k, sc
    return best_k, best_s


def oriented_similarity(sig_a: str, sig_b: str) -> float:
    """EXACT-orientation shape similarity (0..1) -- NO rotation/mirror search.

    similarity() is rotation-invariant, so a mover ⊔ and a goal ⊓ (a cup rotated 180°) score
    ~1.0 -- it tells them apart not at all.  But their ROLE is encoded by ORIENTATION, and only
    an oriented comparison preserves that.  Use the two together: similarity() says 'same KIND of
    glyph' (identity), oriented_similarity() says 'same ORIENTATION' -> so the recurring glyph
    sitting the same way up as the prior level's MOVER is the mover, the one flipped is the goal.
    Same IoU x fill-agreement form as similarity(), minus the orientation search."""
    a = _deserialize(sig_a)
    b = _deserialize(sig_b)
    if a is None or b is None:
        return 0.0
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    iou = inter / union if union else 0.0
    fill_agree = 1.0 - abs(a.mean() - b.mean())
    return float(round(iou * fill_agree, 4))


def identify_role(sig: str, role_templates) -> tuple:
    """role_templates: iterable of (role, sig) carried from prior levels.  Returns (role, score)
    of the best ORIENTED match -- so a recurring glyph is assigned the role of the prior-level
    entity it matches IN THE SAME ORIENTATION.  This is how mover vs goal is decided across
    levels by reference to carried bitmaps (not by position-guessing).  (None, 0.0) if no match."""
    best_r, best_s = None, 0.0
    for role, s in role_templates:
        sc = oriented_similarity(sig, s)
        if sc > best_s:
            best_r, best_s = role, sc
    return best_r, best_s
