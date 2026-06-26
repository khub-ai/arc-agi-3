"""Structural-similarity detection over Layer A geometry candidates.

The cardinal mechanic in ARC-AGI-3 puzzles is "make entity A match
entity B" — alignment_match, pair_match, arrangement_match all rest on
the agent noticing that two entities are visually similar.  A human
takes one glance at an L-shaped reference glyph and an L-shaped working
glyph and immediately infers the relationship; without explicit
similarity computation the agent has to hope the VLM noticed.

This module computes a four-tier fingerprint per candidate and reports
pairs that match at any tier, along with what differs and how to make
them MORE similar.  Tiers, strictest to loosest:

* ``bitmap_id``  — identical bitmap (exact match including palette).
* ``shape_id``   — palette-normalized (same shape and palette structure
                   but different absolute palette values).
* ``topo_id``    — binary occupancy (same silhouette, ignores palette).
* ``scaled_id``  — size-normalized (same shape at different scales).

When two entities share a tier but differ at stricter tiers, we
characterise the differences and produce a "convergence hint": the
transformation that would close the gap.  For the alignment_match win
condition, that hint is literally the planner target: cycle entity A
through transformations until it matches entity B at the bitmap tier.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import GeometryCandidate, GeometryResult

# Pull the harness's existing four-tier fingerprinting.  Lives under
# usecases/arc-agi-3/python; add to path so cognitive_os can import it
# without depending on harness install state.
_HARNESS_PY = (Path(__file__).resolve().parents[2]
               / "usecases" / "arc-agi-3" / "python")
if str(_HARNESS_PY) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PY))

from entity_fingerprint import (  # type: ignore
    fingerprint_from_bitmap,
    masked_bitmap_from_object,
)


@dataclass
class SimilarityPair:
    """One similar-pair finding between two candidates."""
    a_id:           int
    b_id:           int
    a_bbox:         Tuple[int, int, int, int]
    b_bbox:         Tuple[int, int, int, int]
    bitmap_match:   bool
    shape_match:    bool
    topo_match:     bool
    scaled_match:   bool
    a_palettes:     Tuple[int, ...]
    b_palettes:     Tuple[int, ...]
    a_size_pixels:  int
    b_size_pixels:  int
    differences:    List[str] = field(default_factory=list)
    convergence:    str = ""

    @property
    def strongest_tier(self) -> str:
        if self.bitmap_match: return "bitmap_id"
        if self.shape_match:  return "shape_id"
        if self.topo_match:   return "topo_id"
        if self.scaled_match: return "scaled_id"
        return "none"


def _fingerprint_for(cand: GeometryCandidate, frame: np.ndarray):
    bm = masked_bitmap_from_object(frame, cand.bbox_pixels, cells=None)
    return fingerprint_from_bitmap(
        bm,
        palettes = list(cand.palettes),
    )


def _diff_and_convergence(a: GeometryCandidate, b: GeometryCandidate,
                          *, bitmap_match: bool, shape_match: bool,
                          topo_match: bool, scaled_match: bool
                          ) -> Tuple[List[str], str]:
    diffs: List[str] = []
    a_pals = set(a.palettes)
    b_pals = set(b.palettes)
    pal_diff = a_pals.symmetric_difference(b_pals)
    if pal_diff:
        diffs.append(
            f"different palette set ({sorted(a_pals)} vs {sorted(b_pals)})"
        )
    size_ratio = a.size_pixels / max(1, b.size_pixels)
    if size_ratio < 0.85 or size_ratio > 1.18:
        diffs.append(
            f"different pixel size ({a.size_pixels} vs {b.size_pixels})"
        )
    # bbox extents
    ah = a.bbox_pixels[2] - a.bbox_pixels[0] + 1
    aw = a.bbox_pixels[3] - a.bbox_pixels[1] + 1
    bh = b.bbox_pixels[2] - b.bbox_pixels[0] + 1
    bw = b.bbox_pixels[3] - b.bbox_pixels[1] + 1
    if (ah, aw) != (bh, bw):
        diffs.append(f"different bbox extent ({ah}x{aw} vs {bh}x{bw})")
    # Convergence hint — the strictest tier they FAIL determines the
    # transformation needed.
    if bitmap_match:
        conv = "already identical at the bitmap level"
    elif shape_match:
        # Same shape and palette structure but different absolute
        # palette values — palette swap closes the gap.
        conv = (f"cycle entity {a.candidate_id}'s palette to match entity "
                f"{b.candidate_id} (or vice versa) to reach exact match")
    elif topo_match:
        # Same silhouette but different palette layout — recolor closes
        # the gap.
        conv = (f"entities {a.candidate_id} and {b.candidate_id} share the "
                "same silhouette; recolor / re-pattern one to match the other")
    elif scaled_match:
        conv = (f"entities {a.candidate_id} and {b.candidate_id} share the "
                "same shape at different sizes; scaling or growth/shrink "
                "transformations would converge them")
    else:
        conv = ""
    return diffs, conv


def find_similar_pairs(
    geometry: GeometryResult,
    frame:    np.ndarray,
    *,
    min_size_pixels: int = 3,
) -> List[SimilarityPair]:
    """Compute pairwise structural similarity over Layer A candidates.

    Tiny candidates (1-2 pixels) are skipped — their fingerprints are
    degenerate and would match every other tiny candidate, drowning the
    real pairs in noise.

    Returns a list of pairs that match at AT LEAST the loosest tier
    (scaled_id).  Sorted by strictest-tier-first: bitmap matches come
    before shape matches, etc.
    """
    cands = [c for c in geometry.candidates if c.size_pixels >= min_size_pixels]
    if len(cands) < 2:
        return []
    fps = [_fingerprint_for(c, frame) for c in cands]
    pairs: List[SimilarityPair] = []
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            a, b = cands[i], cands[j]
            fa, fb = fps[i], fps[j]
            bitmap_match = (fa.bitmap_id == fb.bitmap_id)
            shape_match  = (fa.shape_id  == fb.shape_id)
            topo_match   = (fa.topo_id   == fb.topo_id)
            scaled_match = (fa.scaled_id == fb.scaled_id)
            if not (bitmap_match or shape_match or topo_match or scaled_match):
                continue
            diffs, conv = _diff_and_convergence(
                a, b,
                bitmap_match=bitmap_match, shape_match=shape_match,
                topo_match=topo_match,   scaled_match=scaled_match,
            )
            pairs.append(SimilarityPair(
                a_id          = a.candidate_id,
                b_id          = b.candidate_id,
                a_bbox        = tuple(a.bbox_pixels),
                b_bbox        = tuple(b.bbox_pixels),
                bitmap_match  = bitmap_match,
                shape_match   = shape_match,
                topo_match    = topo_match,
                scaled_match  = scaled_match,
                a_palettes    = tuple(a.palettes),
                b_palettes    = tuple(b.palettes),
                a_size_pixels = a.size_pixels,
                b_size_pixels = b.size_pixels,
                differences   = diffs,
                convergence   = conv,
            ))
    # Sort by tier strictness so the prompt highlights the most
    # compelling pairs first.
    tier_rank = {
        "bitmap_id": 0, "shape_id": 1, "topo_id": 2, "scaled_id": 3, "none": 4,
    }
    pairs.sort(key=lambda p: tier_rank[p.strongest_tier])
    return pairs


def format_pairs_for_prompt(pairs: Sequence[SimilarityPair]) -> str:
    """Render similarity pairs as a Layer-B prompt section.

    Bitmap-identical groups (many launchers, many cards, etc.) collapse
    to ONE line per equivalence class to avoid N-squared bloat.  Shape /
    topo / scaled matches stay as per-pair entries because they carry
    the more interesting "different in palette/size" relationship the
    VLM needs to read.
    """
    if not pairs:
        return ""

    # Build bitmap-id equivalence classes from the bitmap-tier pairs.
    bm_pairs = [p for p in pairs if p.bitmap_match]
    parent: dict = {}
    def _find(x):
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x
    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra
    for p in bm_pairs:
        parent.setdefault(p.a_id, p.a_id)
        parent.setdefault(p.b_id, p.b_id)
        _union(p.a_id, p.b_id)
    classes: dict = {}
    member_meta: dict = {}
    for p in bm_pairs:
        for cid, bbox, size, pals in (
            (p.a_id, p.a_bbox, p.a_size_pixels, p.a_palettes),
            (p.b_id, p.b_bbox, p.b_size_pixels, p.b_palettes),
        ):
            classes.setdefault(_find(cid), set()).add(cid)
            member_meta[cid] = (bbox, size, pals)

    lines: List[str] = [
        "# Structural similarity (Layer A pairwise comparison)",
        "",
        "Candidates were compared via four-tier fingerprint matching.  "
        "Pairs that share shape or silhouette are STRONG candidates for "
        "game-mechanical pairing (working_reference_pair, reference_pair, "
        "complementary_shape_pair, etc.).  Use this to inform your role "
        "assignments and the relationships you emit.",
        "",
    ]

    # Bitmap-identical equivalence classes.
    if classes:
        lines.append("## Bitmap-identical equivalence classes")
        lines.append("")
        for root, members in sorted(classes.items(), key=lambda kv: -len(kv[1])):
            ms = sorted(members)
            example = member_meta[ms[0]]
            lines.append(
                f"- {len(ms)} candidates with identical bitmaps: {ms} "
                f"(each is size={example[1]}px, pals={list(example[2])})"
            )
        lines.append("")

    # Non-bitmap (interesting) pairs: shape / topo / scaled.
    interesting = [p for p in pairs if not p.bitmap_match]
    if interesting:
        lines.append("## Same-shape-or-silhouette pairs (different in palette and/or scale)")
        lines.append("")
        for p in interesting:
            kind = p.strongest_tier
            if kind == "shape_id":
                label = "same shape and palette structure (palette swap closes gap)"
            elif kind == "topo_id":
                label = "same silhouette (recolor closes gap)"
            elif kind == "scaled_id":
                label = "same shape at different scale"
            else:
                continue
            lines.append(
                f"- candidates {p.a_id} and {p.b_id}: {label}."
            )
            lines.append(
                f"  - cand {p.a_id}: bbox={list(p.a_bbox)} size={p.a_size_pixels}px "
                f"pals={list(p.a_palettes)}"
            )
            lines.append(
                f"  - cand {p.b_id}: bbox={list(p.b_bbox)} size={p.b_size_pixels}px "
                f"pals={list(p.b_palettes)}"
            )
            if p.differences:
                lines.append(f"  - differs in: {'; '.join(p.differences)}")
            if p.convergence:
                lines.append(f"  - to make more similar: {p.convergence}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"
