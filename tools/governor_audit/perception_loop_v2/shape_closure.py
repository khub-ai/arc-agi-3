"""shape_closure.py -- game-agnostic CLOSURE / completion instinct (Gestalt closure).

A shape is INCOMPLETE if it has a concavity -- a structural MOUTH (a notch / opening a complement could
fill).  The closure prior: incomplete shapes "want" to be COMPLETED -- united with the piece that fills
the mouth.  The win-condition goal it yields: CLOSE EVERY MOUTH -- pair each incomplete shape with its
complement (mouth facing opposite AND matching the incomplete shape's own attributes, e.g. colour/type)
so each union becomes a solid, gap-free figure; the win is when no concavity remains.

This derives "complete the figures" with ZERO game knowledge: "purple needs purple" falls out of
attribute-matching, not a colour rule; multiple incomplete shapes -> complete ALL of them.  It is the
operational form of the regularity/conform instinct (a concavity is a near-miss to a solid) and feeds
the means-ends goal.  Pure geometry + attributes; the substrate MEASURES, the actor VERIFIES.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from path_planning import shape_opening, openings_couple
    _OK = True
except Exception:  # pragma: no cover
    _OK = False

    def shape_opening(mask):
        return None

    def openings_couple(a, b):
        return False


def concavity(mask) -> float:
    """Fraction of the shape's bounding box that is UNFILLED -- the size of its mouth/notch/hole.
    ~0 for a solid block; large for a deep cup/arch.  A measure, not a gate."""
    if mask is None:
        return 0.0
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return 0.0
    h = ys.max() - ys.min() + 1
    w = xs.max() - xs.min() + 1
    return 1.0 - float(m.sum()) / float(h * w)


def is_incomplete(mask) -> bool:
    """A shape is incomplete iff it has a structural MOUTH (an opening a complement can mate into).
    Threshold-light (via shape_opening: walls at both ends of an edge, gap in the middle)."""
    return shape_opening(mask) is not None


def attributes_match(a: Optional[str], b: Optional[str]) -> bool:
    """Same on the matchable attribute (colour/type).  This is what makes 'purple needs purple'
    emerge generically -- match the incomplete shape's OWN attribute, no hardcoded colour."""
    return a is not None and b is not None and str(a).lower() == str(b).lower()


def complements(mask_a, color_a, mask_b, color_b) -> bool:
    """True iff a and b complete each other: mouths face OPPOSITE (so the union is solid) AND their
    attributes match (so the union is one uniform figure)."""
    return (openings_couple(shape_opening(mask_a), shape_opening(mask_b))
            and attributes_match(color_a, color_b))


# ---- entity-level (entity dict needs "mask" (bool grid) + "color" + "name") ----------------------
def incomplete_entities(entities: List[Dict]) -> List[Dict]:
    return [e for e in entities if is_incomplete(e.get("mask"))]


def find_completions(entities: List[Dict]) -> Tuple[List[Tuple[Dict, Dict]], List[Dict]]:
    """Pair each incomplete entity with a complement among ALL entities (incl. references/legend
    pieces).  Returns (pairs, unmatched): unmatched incomplete shapes have NO complement present, so
    their complement must be PRODUCED (e.g. selected from the legend) before the mouth can close."""
    inc = incomplete_entities(entities)
    pairs: List[Tuple[Dict, Dict]] = []
    used = set()
    for a in inc:
        for b in entities:
            if b is a or id(b) in used:
                continue
            if complements(a.get("mask"), a.get("color"), b.get("mask"), b.get("color")):
                pairs.append((a, b))
                used.add(id(a))
                used.add(id(b))
                break
    unmatched = [a for a in inc if id(a) not in used]
    return pairs, unmatched


def closure_directive(entities: List[Dict]) -> Optional[str]:
    inc = incomplete_entities(entities)
    if not inc:
        return None
    pairs, unmatched = find_completions(entities)
    lines = [f"[CLOSURE] {len(inc)} incomplete shape(s) (have a mouth); win = close EVERY mouth by "
             f"uniting each with its complement (opposite mouth + matching attribute):"]
    for a, b in pairs:
        lines.append(f"  - {a.get('name')} (mouth {shape_opening(a.get('mask'))}, {a.get('color')}) "
                     f"<- complement {b.get('name')} (mouth {shape_opening(b.get('mask'))})")
    for a in unmatched:
        lines.append(f"  - {a.get('name')} (mouth {shape_opening(a.get('mask'))}, {a.get('color')}) "
                     f"<- NO complement present -> must be PRODUCED (e.g. a matching piece from the legend)")
    return "\n".join(lines)


# ---- frame helpers (extract a shape's own-pixel mask + dominant colour) ---------------------------
def mask_from_bbox(frame_rgb, bbox, bg=None) -> np.ndarray:
    """Boolean mask of the non-background pixels of `frame_rgb` inside `bbox` ([r0,c0,r1,c1] in the
    frame's own pixel coords).  `bg` defaults to the bbox's most common colour (the local field)."""
    arr = np.asarray(frame_rgb)[:, :, :3]
    r0, c0, r1, c1 = [int(v) for v in bbox]
    crop = arr[r0:r1 + 1, c0:c1 + 1]
    if crop.size == 0:
        return np.zeros((0, 0), bool)
    if bg is None:
        cols, counts = np.unique(crop.reshape(-1, 3), axis=0, return_counts=True)
        bg = cols[counts.argmax()]
    return ~np.all(crop == np.asarray(bg), axis=-1)


def dominant_color(frame_rgb, bbox) -> str:
    """Hex of the dominant NON-background colour of the shape in `bbox`."""
    arr = np.asarray(frame_rgb)[:, :, :3]
    r0, c0, r1, c1 = [int(v) for v in bbox]
    crop = arr[r0:r1 + 1, c0:c1 + 1].reshape(-1, 3)
    if crop.size == 0:
        return ""
    cols, counts = np.unique(crop, axis=0, return_counts=True)
    order = counts.argsort()[::-1]
    bg = cols[order[0]]
    for i in order:
        if not np.array_equal(cols[i], bg):
            r, g, b = cols[i]
            return f"#{r:02x}{g:02x}{b:02x}"
    return ""
