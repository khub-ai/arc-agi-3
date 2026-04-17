"""Frame-diff and motion extraction.

Given two successive frames the adapter can derive a lot of
information without invoking any LLM:

* Which cells changed colour (``cell_diff``).
* Whether a region moved rigidly between frames (``motion_vectors``).
* Whether anything at all changed (``is_identical``).

The perception layer turns ``cell_diff`` into per-entity property
updates and ``AgentMoved`` / ``EntityDisappeared`` / ``EntityAppeared``
events; the planner turns ``motion_vectors`` into empirical
``TransitionClaim``s.  Both are central to the engine's
hypothesis-formation loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .components import Region, extract_regions

Cell = Tuple[int, int]
Grid = Sequence[Sequence[int]]


@dataclass(frozen=True)
class CellChange:
    """A single cell whose colour differed between two frames."""
    row:     int
    col:     int
    before:  int
    after:   int


@dataclass(frozen=True)
class MotionVector:
    """A best-effort rigid-translation match from a ``before`` region
    to an ``after`` region.

    ``dr`` and ``dc`` are the integer row/column offsets; ``confident``
    is ``True`` when the match is unambiguous (same colour, same
    cell-shape up to translation).  Ambiguous matches — multiple
    plausible pairings — are returned with ``confident=False`` and
    should be treated as hypotheses, not facts.
    """
    before_label:  int
    after_label:   int
    colour:        int
    dr:            int
    dc:            int
    confident:     bool


def is_identical(before: Grid, after: Grid) -> bool:
    """Cheap whole-frame equality check."""
    if len(before) != len(after):
        return False
    for rb, ra in zip(before, after):
        if len(rb) != len(ra):
            return False
        for vb, va in zip(rb, ra):
            if vb != va:
                return False
    return True


def cell_diff(before: Grid, after: Grid) -> List[CellChange]:
    """Return every cell whose colour differs between the two frames.

    The two grids must have identical shape; shape mismatches return
    an empty list rather than raising, because mid-episode grid-size
    changes are a legitimate (surprising) observation and the
    SurpriseMiner handles them upstream.
    """
    if len(before) != len(after):
        return []
    out: List[CellChange] = []
    for r, (rb, ra) in enumerate(zip(before, after)):
        if len(rb) != len(ra):
            return []
        for c, (vb, va) in enumerate(zip(rb, ra)):
            if vb != va:
                out.append(CellChange(row=r, col=c, before=vb, after=va))
    return out


def motion_vectors(
    before: Grid,
    after:  Grid,
    *,
    background: int = 0,
) -> List[MotionVector]:
    """Match components across two frames and return their translations.

    Algorithm: label both frames; for each ``before`` region, look for
    an ``after`` region with the same colour and the same cell-shape
    under translation.  Unambiguous matches are emitted with
    ``confident=True``; multi-candidate matches emit every candidate
    with ``confident=False`` so the caller can defer the decision to
    a miner or the Mediator.

    Regions without a match are not emitted here — the
    ``EntityAppeared`` / ``EntityDisappeared`` events those imply are
    the perception layer's responsibility.
    """
    before_regions = extract_regions(before, background=background)
    after_regions  = extract_regions(after,  background=background)

    # Precompute normalised-shape signatures per region so we can
    # match by "same shape + same colour" in a single pass.
    after_by_key: Dict[Tuple[int, Tuple[Cell, ...]], List[Region]] = {}
    for reg in after_regions:
        key = (reg.colour, _normalised_shape(reg))
        after_by_key.setdefault(key, []).append(reg)

    matches: List[MotionVector] = []
    for b in before_regions:
        key = (b.colour, _normalised_shape(b))
        candidates = after_by_key.get(key, [])
        if not candidates:
            continue
        if len(candidates) == 1:
            a = candidates[0]
            dr = a.bbox[0] - b.bbox[0]
            dc = a.bbox[1] - b.bbox[1]
            matches.append(MotionVector(
                before_label = b.label,
                after_label  = a.label,
                colour       = b.colour,
                dr           = dr,
                dc           = dc,
                confident    = True,
            ))
        else:
            # Ambiguous — emit one low-confidence vector per candidate.
            for a in candidates:
                dr = a.bbox[0] - b.bbox[0]
                dc = a.bbox[1] - b.bbox[1]
                matches.append(MotionVector(
                    before_label = b.label,
                    after_label  = a.label,
                    colour       = b.colour,
                    dr           = dr,
                    dc           = dc,
                    confident    = False,
                ))
    return matches


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _normalised_shape(region: Region) -> Tuple[Cell, ...]:
    """Return ``region.cells`` translated to place the bbox origin at
    (0, 0).  Two regions are the same shape up to translation iff
    their normalised-shape tuples are equal."""
    r0, c0 = region.bbox[0], region.bbox[1]
    return tuple((r - r0, c - c0) for (r, c) in region.cells)
