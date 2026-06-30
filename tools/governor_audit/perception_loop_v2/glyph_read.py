"""Substrate glyph-read -- step 5 of the TARGET-PATTERN PURSUIT.  Turns a cell's pixels into a
DISCRETE identity key so the executor can compare/track glyphs, instead of the VLM eyeballing.

GAME-AGNOSTIC.  Given the frame, a list of cell bboxes, and an ALPHABET (reference glyph
bboxes -- e.g. the legend's distinct output glyphs), each cell is labelled with the index of
the alphabet glyph it best matches (rotation/scale/colour-invariant via shape_identity), or
None if it matches nothing well.  These integer keys are exactly what edit_law (the click
sequence) and edit_executor (current vs target) consume.

This is the "read the board with the substrate, not the VLM's eye" guarantee for per-slot state.
"""
from __future__ import annotations

from typing import List, Optional

try:
    import shape_identity as _si
except ImportError:
    from perception_loop_v2 import shape_identity as _si


def identify_glyphs(frame_rgb, cell_bboxes: List[list], alphabet_bboxes: List[list],
                    min_score: float = 0.5) -> List[Optional[int]]:
    """Label each cell with the index of the best-matching alphabet glyph (invariant), or None.

    Returns one key per cell: an int alphabet index, or None when no alphabet glyph matches
    above min_score (cell empty / unknown glyph).  Deterministic; guarded.
    """
    try:
        ab_sigs = [_si.shape_signature(frame_rgb, b) for b in alphabet_bboxes]
    except Exception:
        ab_sigs = []
    keys: List[Optional[int]] = []
    for cb in cell_bboxes:
        best_i, best_s = None, min_score
        try:
            csig = _si.shape_signature(frame_rgb, cb)
            for i, asig in enumerate(ab_sigs):
                if not asig:
                    continue
                s = _si.similarity(csig, asig)
                if s >= best_s:
                    best_i, best_s = i, s
        except Exception:
            pass
        keys.append(best_i)
    return keys


def read_slot_sequence(frames: List, cell_bbox: list, alphabet_bboxes: List[list],
                       min_score: float = 0.5) -> List[Optional[int]]:
    """One cell's glyph identity across a probe sequence of frames (before click, after click 1,
    ...).  The result feeds edit_law.induce_edit_law to learn the click->content law."""
    return [identify_glyphs(f, [cell_bbox], alphabet_bboxes, min_score)[0] for f in frames]
