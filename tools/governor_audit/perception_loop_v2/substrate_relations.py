"""Substrate-measured GEOMETRIC RELATIONS among components -- the narrow,
deterministic facts the substrate may assert authoritatively (the VLM owns what
they MEAN).  Pure measurement, no game knowledge, no tuned thresholds: relations
are decided by EXACT equality, not similarity cut-offs.

What it reports (all measurable from pixels alone):
  - identical groups: components with the SAME native size AND the SAME mask
    (translation-only) -- e.g. ka59's two green pieces (8 px each), two box rings
    (16 px each).  Same shape AND same scale.
  - same-shape groups: components whose SCALE-NORMALISED silhouette is equal under
    some rotation/mirror -- same shape, possibly different scale (looser; a square
    and a bigger square).  Uses shape_identity canonical signatures (exact match,
    not a threshold).
  - contains: component A's bbox strictly encloses component B (A larger) -- e.g.
    a ring around its interior, or a piece around the 1-px mark sitting in it.

The substrate states these as neutral facts ("two components are identical";
"A contains B").  It does NOT say "pieces", "containers", "goal" -- that is the
VLM's interpretation.  See feedback_label_observed_vs_assumed + the
measurement-vs-meaning line.
"""
from __future__ import annotations

from typing import List, Dict, Any

try:
    import shape_identity as _si        # REUSE the invariant matcher (rot/scale/mirror/colour)
except Exception:                        # pragma: no cover
    _si = None

try:
    import numpy as np
    import shape_identity as _si
    _OK = True
except Exception:                                       # pragma: no cover
    _OK = False


def _dims(bbox):
    r0, c0, r1, c1 = bbox                                # inclusive
    return (r1 - r0 + 1, c1 - c0 + 1)


def _mask_key(mask) -> str:
    """Translation-invariant exact key for a binary mask (its bit pattern)."""
    m = np.asarray(mask, dtype=bool)
    return f"{m.shape[0]}x{m.shape[1]}:" + "".join("1" if v else "0" for v in m.flatten())


def _canonical_sig(mask) -> str:
    """Scale-normalised canonical signature, orientation-folded to the
    lexicographically smallest of its 8 dihedral forms -> equal iff same shape
    modulo rotation/mirror/scale.  Exact match, no threshold."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return ""
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    G = _si.GRID
    h, w = m.shape
    ri = (np.arange(G) * h // G).clip(0, h - 1)
    ci = (np.arange(G) * w // G).clip(0, w - 1)
    cs = m[ri][:, ci]
    forms = []
    for base in (cs, np.fliplr(cs)):
        r = base
        for _ in range(4):
            forms.append("".join("1" if v else "0" for v in r.flatten()))
            r = np.rot90(r)
    return min(forms)


def _sig16(mask) -> str:
    """Scale-normalised 16x16 silhouette serialized as shape_identity expects, so we
    can REUSE shape_identity.similarity (rotation/mirror/scale/colour-invariant) rather
    than reinvent it."""
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return ""
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    G = 16
    h, w = m.shape
    ri = (np.arange(G) * h // G).clip(0, h - 1)
    ci = (np.arange(G) * w // G).clip(0, w - 1)
    return "".join("1" if v else "0" for v in m[ri][:, ci].flatten())


def _interior_dims(comp):
    """Return the interior dims (h-2, w-2) ONLY if comp encloses a real HOLE; else None.
    Gate: 1px-CLOSE the shape (8-connected dilation) so a DOTTED ring counts as a closed
    loop, then flood-fill the background from the bbox border -- background the flood
    can't reach is the enclosed hole.  This keeps a dotted ring/shell (the ball's
    container), but rejects a thin LINE (a wire encloses nothing) and a 1px centre MARK
    (the close fills it).  The dims returned are bbox-based (the hole test is a gate),
    so the fit size-match is unchanged.  Needs the mask -> None (conservative) when
    absent."""
    r0, c0, r1, c1 = comp["bbox"]
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 3 or w < 3 or comp.get("npix", 0) >= h * w:     # too small / solid -> no hole
        return None
    try:
        import numpy as np
        from scipy import ndimage
        m = np.asarray(comp.get("mask"))
        if m.ndim != 2:
            return None
        closed = ndimage.binary_dilation(m.astype(bool), structure=np.ones((3, 3), bool))
        bg = ~closed
        lab, n = ndimage.label(bg)                 # 4-connected background regions
        if n == 0:
            return None
        border = (set(lab[0, :]) | set(lab[-1, :])
                  | set(lab[:, 0]) | set(lab[:, -1]))
        border.discard(0)
        if not any(i not in border for i in range(1, n + 1)):
            return None                            # no enclosed hole -> not a container
        return (h - 2, w - 2)
    except Exception:
        return None


def _fits(piece, container) -> bool:
    """piece's FOOTPRINT fits container's enclosed interior (colour- and solidity-
    agnostic: it is about the bounding box fitting the hole).  container must be a ring
    with an interior; piece's bbox dims ~= that interior (within 1)."""
    iv = _interior_dims(container)
    if iv is None or piece is container:
        return False
    pr0, pc0, pr1, pc1 = piece["bbox"]
    ph, pw = pr1 - pr0 + 1, pc1 - pc0 + 1
    return abs(ph - iv[0]) <= 1 and abs(pw - iv[1]) <= 1 and ph <= iv[0] + 1 and pw <= iv[1] + 1


def _contains(outer, inner) -> bool:
    """outer bbox STRICTLY encloses inner bbox, and outer is larger (npix)."""
    Or0, Oc0, Or1, Oc1 = outer["bbox"]
    Ir0, Ic0, Ir1, Ic1 = inner["bbox"]
    inside = (Or0 <= Ir0 and Oc0 <= Ic0 and Or1 >= Ir1 and Oc1 >= Ic1)
    bigger = outer.get("npix", 0) > inner.get("npix", 0)
    return inside and bigger and (outer is not inner)


def find_relations(components: List[Dict[str, Any]]) -> Dict[str, Any]:
    """components: list of dicts with 'bbox' (inclusive r0,c0,r1,c1), 'mask'
    (2D bool), 'npix', and optional 'id'/'color'.  Returns measured relations:
      {identical: [[ids...], ...], same_shape: [[ids...], ...],
       contains: [{outer, inner, inner_tiny}], counts: {...}}.
    Groups have >=2 members.  Guarded -- {} on any error or without numpy."""
    if not _OK or not components:
        return {}
    try:
        comps = []
        for i, c in enumerate(components):
            cid = c.get("id", i)
            mask = c.get("mask")
            if mask is None:
                continue
            comps.append({"id": cid, "bbox": list(c["bbox"]),
                          "npix": int(c.get("npix", np.asarray(mask).sum())),
                          "mask": mask})
        # identical = same exact mask (translation-only)
        ident: Dict[str, list] = {}
        shape: Dict[str, list] = {}
        for c in comps:
            ident.setdefault(_mask_key(c["mask"]), []).append(c["id"])
            sg = _canonical_sig(c["mask"])
            if sg:
                shape.setdefault(sg, []).append(c["id"])
        identical = [sorted(v) for v in ident.values() if len(v) >= 2]
        # same-shape groups beyond what's already an identical group
        ident_pairs = {frozenset(g) for g in identical}
        same_shape = [sorted(v) for v in shape.values()
                      if len(v) >= 2 and frozenset(v) not in ident_pairs]
        # SHAPE-SIMILAR (graded, INVARIANT to rotation/mirror/scale/colour) -- reuse
        # shape_identity. Reports pairs that are NOT exactly identical/same_shape but
        # are strongly similar (a "minor difference" / same kind under a transform),
        # carrying the score so the VLM judges. 0.80 is a display floor (show strong
        # matches), not a decision threshold -- the score is surfaced.
        # Only DISTINCTIVE shapes carry scale-invariant signal: a 1px dot or a solid
        # blob canonicalises to a filled square and would "match" everything. Require
        # enough pixels AND a non-solid silhouette (the shape has internal structure).
        def _distinctive(c):
            r0, c0, r1, c1 = c["bbox"]
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            return c["npix"] >= 4 and area > 0 and (c["npix"] / area) <= 0.9
        shape_similar = []
        if _si is not None:
            dist = [c for c in comps if _distinctive(c)]
            sigs = {c["id"]: _sig16(c["mask"]) for c in dist}
            done = {frozenset(g) for g in identical} | {frozenset(g) for g in same_shape}
            for i in range(len(dist)):
                for j in range(i + 1, len(dist)):
                    a, b = dist[i]["id"], dist[j]["id"]
                    if frozenset((a, b)) in done or not sigs[a] or not sigs[b]:
                        continue
                    sc = _si.similarity(sigs[a], sigs[b])
                    if sc >= 0.85:
                        shape_similar.append({"a": a, "b": b, "score": sc})
        # FITS -- a piece's footprint fits a ring's enclosed interior (colour- and
        # solidity-agnostic). The delivery clue ka59 needs (a green fits a box hole).
        fits = []
        for p in comps:
            for r in comps:
                if _fits(p, r):
                    fits.append({"piece": p["id"], "container": r["id"]})
        # containment
        contains = []
        for a in comps:
            for b in comps:
                if _contains(a, b):
                    contains.append({"outer": a["id"], "inner": b["id"],
                                     "inner_tiny": b["npix"] <= 2})
        return {"identical": identical, "same_shape": same_shape,
                "shape_similar": shape_similar, "fits": fits,
                "contains": contains,
                "counts": {"components": len(comps),
                           "identical_groups": len(identical),
                           "same_shape_groups": len(same_shape),
                           "shape_similar_pairs": len(shape_similar),
                           "fits_pairs": len(fits)}}
    except Exception:
        return {}


def render_text(rel: Dict[str, Any], names: Dict[Any, str] | None = None) -> str:
    """Neutral one-line-per-fact rendering for the perception prompt.  ``names``
    maps component id -> a label (else the id).  No meaning words."""
    if not rel:
        return ""
    nm = (lambda i: str((names or {}).get(i, i)))
    lines = []
    for g in rel.get("identical", []):
        lines.append(f"- identical (same shape AND size): {', '.join(nm(i) for i in g)}")
    for g in rel.get("same_shape", []):
        lines.append(f"- same shape (different scale/orientation): {', '.join(nm(i) for i in g)}")
    for s in rel.get("shape_similar", []):
        lines.append(f"- similar shape (score {s['score']}, minor difference, ignoring "
                     f"orientation/scale/colour): {nm(s['a'])}, {nm(s['b'])}")
    for f in rel.get("fits", []):
        lines.append(f"- {nm(f['piece'])} fits inside the empty interior of {nm(f['container'])}")
    for c in rel.get("contains", []):
        tag = " (a tiny mark inside it)" if c.get("inner_tiny") else ""
        lines.append(f"- {nm(c['outer'])} encloses {nm(c['inner'])}{tag}")
    return "\n".join(lines)
