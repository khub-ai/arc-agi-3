"""Interaction-grounded perception refinement.

Single-frame perception (Layer A+B) is bounded by what's visible in one
image -- the VLM has to guess which entity is the agent, which is a
trigger, which is a passive prop.  Many of those guesses are wrong.
Gameplay observation collapses the ambiguity:

- The entity whose bbox MOVES with actions is the agent_avatar.
- Cells the agent stands on AT THE MOMENT another entity changes are
  triggers for that entity (shape_changer / color_changer / etc.).
- Entities that NEVER change across the playthrough are static --
  references (in alignment puzzles), HUD, decorations, etc.
- Entities that change but don't move are operated-upon -- working
  glyphs, mutable blocks, etc.
- Pairs whose final-state bitmaps match a reference's bitmap when the
  level wins are the working/reference pair of the win condition.

This module accepts an initial ParsedPerception plus a sequence of
(frame, action, post_frame) observations and emits an updated
ParsedPerception with corrected role assignments and a per-entity
behavior log.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# Use the harness's four-tier fingerprinting.
_HARNESS_PY = (Path(__file__).resolve().parents[2]
               / "usecases" / "arc-agi-3" / "python")
if str(_HARNESS_PY) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PY))

from entity_fingerprint import (  # type: ignore
    fingerprint_from_bitmap,
    masked_bitmap_from_object,
    EntityFingerprint,
)
from cognitive_os.perception.geometry import GeometryCandidate, GeometryResult


# ---------------------------------------------------------------------------
# Per-entity behavior trace.
# ---------------------------------------------------------------------------


@dataclass
class _PerStepCand:
    """Internal: a Layer-A candidate snapshot used by the tracker."""
    bbox: Tuple[int, int, int, int]
    palettes: Tuple[int, ...]
    size: int
    bitmap_id: str
    topo_id: str


@dataclass
class EntityTrack:
    """Persistent track of one entity across frames.

    Identified by its initial bitmap_id at level start.  Across frames
    we follow it by:
    1. Same bitmap_id at a different bbox -> moved.
    2. Same topo_id (silhouette) at the same bbox -> palette / pattern change.
    3. Disappeared from the candidate list -> consumed or occluded.
    """
    initial_bitmap_id:   str
    initial_bbox:        Tuple[int, int, int, int]
    initial_size:        int
    initial_palettes:    Tuple[int, ...]
    # Per-step observations.
    bbox_history:        List[Tuple[int, int, int, int]] = field(default_factory=list)
    bitmap_id_history:   List[str] = field(default_factory=list)
    topo_id_history:     List[str] = field(default_factory=list)
    shape_id_history:    List[str] = field(default_factory=list)
    scaled_id_history:   List[str] = field(default_factory=list)
    palette_history:     List[Tuple[int, ...]] = field(default_factory=list)
    size_history:        List[int] = field(default_factory=list)
    cell_set_history:    List[frozenset] = field(default_factory=list)
    # Derived metrics across the trace.
    total_displacement:  float = 0.0
    # Displacement contributed only by within-radius matches (the P1-P5
    # tiers).  P_slide teleports do NOT count: they are the matcher's
    # weakest tier (a globally-unique-bitmap fallback that fires when
    # an entity vanished from its prior location).  If the only motion
    # the tracker recorded was a P_slide jump to a same-bitmap
    # candidate elsewhere, that "motion" is at least as plausibly a
    # mis-association as a real slide.  Continuous displacement is the
    # honest signal that an entity actually moved frame-to-frame.
    continuous_displacement: float = 0.0
    n_frames_seen:       int   = 0
    n_palette_changes:   int   = 0
    n_pattern_changes:   int   = 0     # bitmap_id changed at same bbox
    disappeared_at_step: Optional[int] = None

    @property
    def stayed_static(self) -> bool:
        """No motion AND no appearance change."""
        return (self.total_displacement == 0
                and self.n_palette_changes == 0
                and self.n_pattern_changes == 0)

    @property
    def moves_with_actions(self) -> bool:
        """The track shows real motion across the trace.

        Two pieces of evidence are required:

        * ``total_displacement > 4.0`` -- the cumulative bbox shift is
          big enough that the bbox-center moved at least one cell-ish
          across the trace.
        * ``continuous_displacement > 0.0`` -- at least one motion
          event was a within-radius (P1-P5) match.  Pure-P_slide
          tracks have ``continuous_displacement == 0`` because every
          recorded jump was a teleport-tier fallback; such tracks
          are at least as plausibly matcher mis-associations of a
          vanished entity to a same-bitmap candidate elsewhere as
          they are real long-range slides, so they don't satisfy the
          "this entity actually moves" claim on their own.  A
          legitimately moving entity in a click-positioned game
          still gets at least one within-radius match somewhere in
          the trace (when the click distance is small enough), so
          this gate is satisfied alongside the teleport events.
        """
        return (self.total_displacement > 4.0
                and self.continuous_displacement > 0.0)

    @property
    def appearance_changed(self) -> bool:
        return self.n_palette_changes > 0 or self.n_pattern_changes > 0


@dataclass
class InteractionLog:
    """Accumulated observations across a play sequence."""
    n_steps:        int = 0
    tracks:         Dict[str, EntityTrack] = field(default_factory=dict)
    # Step-by-step events:
    motion_events:  List[dict] = field(default_factory=list)
    change_events:  List[dict] = field(default_factory=list)
    frame_history:  List[np.ndarray] = field(default_factory=list)
    # Inferences after the run:
    agent_track_id: Optional[str] = None
    static_tracks:  List[str] = field(default_factory=list)
    dynamic_tracks: List[str] = field(default_factory=list)
    trigger_candidates: List[dict] = field(default_factory=list)
    relatedness_pairs:  List[Tuple[str, str, int]] = field(default_factory=list)
    permanently_changed_tracks: set = field(default_factory=set)
    # Slide events: a same-identity entity that moved more than one
    # cell in a single step (push / slide / bounce-plate motion).
    # Each entry has step, action, tid, from_bbox, to_bbox, dist.
    # Keeps the old `teleport_events` name as an alias for any
    # external consumers; the physically-accurate name is slide.
    slide_events:       List[dict] = field(default_factory=list)
    # Bounce plates derived from slide_events in finalise_inferences:
    # one entry per (entrance_centre, exit_centre) pair with count,
    # actions, tids.  ``portal_endpoints`` kept as a no-op alias.
    bounce_plates:      List[dict] = field(default_factory=list)

    @property
    def teleport_events(self):  # pragma: no cover - backwards-compat alias
        return self.slide_events

    @teleport_events.setter
    def teleport_events(self, value):  # pragma: no cover
        self.slide_events = value

    @property
    def portal_endpoints(self):  # pragma: no cover - backwards-compat alias
        return self.bounce_plates

    @portal_endpoints.setter
    def portal_endpoints(self, value):  # pragma: no cover
        self.bounce_plates = value


# ---------------------------------------------------------------------------
# Entity extraction per frame -- per-palette components.
# ---------------------------------------------------------------------------


def _extract_candidates_per_palette(
    frame: np.ndarray,
    bg_palettes: Sequence[int] = (),
    *,
    max_region_size: int = 90,
) -> List[Tuple[Tuple[int,int,int,int], Tuple[int,...], int, str, str, str, str, frozenset]]:
    """For each non-background palette region, emit
    (bbox, palettes, size, bitmap_id, topo_id) tuples.

    Uses 8-connectivity within a palette so multi-color sprites stay
    split into per-palette components (cheaper and more deterministic
    than the cross-palette object_id merging Layer A uses).

    Components LARGER than ``max_region_size`` pixels (default ~5% of a
    64x64 frame) are treated as palette regions, not entities.  Region
    bitmaps appear to "change" whenever the agent moves through them
    (their connected-component pixels reshape around the agent), which
    would generate false trigger events.  Tracking discrete sprites
    only avoids that pitfall.
    """
    from collections import deque
    H, W = frame.shape
    bg = set(int(p) for p in bg_palettes)
    visited = np.zeros((H, W), dtype=bool)
    out: List = []
    for r0 in range(H):
        for c0 in range(W):
            if visited[r0, c0]:
                continue
            pal = int(frame[r0, c0])
            if pal in bg:
                visited[r0, c0] = True
                continue
            # 8-conn BFS within same palette.
            queue = deque([(r0, c0)])
            cells: List[Tuple[int,int]] = []
            visited[r0, c0] = True
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not visited[nr, nc]:
                            if int(frame[nr, nc]) == pal:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
            if len(cells) > max_region_size:
                # Skip palette regions; they're not discrete entities
                # and their bitmaps shift as the agent moves through.
                continue
            rs = [c[0] for c in cells]
            cs = [c[1] for c in cells]
            bbox = (min(rs), min(cs), max(rs), max(cs))
            bm = masked_bitmap_from_object(frame, bbox, cells=cells)
            fp = fingerprint_from_bitmap(bm, palettes=[pal])
            # Cell set is the NON-canonical signature: it captures the
            # actual pixel positions, so rotations of the same shape
            # are distinguishable (whereas fingerprint canonicalises
            # rotations to the same id).  Used downstream to detect
            # permanent vs transient changes -- a rotation of the
            # working glyph that PERSISTS across steps is a real
            # property change even if its canonical shape is unchanged.
            cell_set = frozenset(cells)
            out.append((bbox, (pal,), len(cells),
                        fp.bitmap_id, fp.topo_id, fp.shape_id, fp.scaled_id,
                        cell_set))
    return out


# ---------------------------------------------------------------------------
# Frame-to-frame tracking.
# ---------------------------------------------------------------------------


def _bbox_center(b: Tuple[int, int, int, int]) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _bbox_distance(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ar, ac = _bbox_center(a)
    br, bc = _bbox_center(b)
    return ((ar - br) ** 2 + (ac - bc) ** 2) ** 0.5


def initialise_log(initial_frame: np.ndarray,
                   bg_palettes: Sequence[int] = ()) -> InteractionLog:
    """Seed the log from the level's initial frame."""
    cands = _extract_candidates_per_palette(initial_frame, bg_palettes=bg_palettes)
    log = InteractionLog()
    # Keep full frames so per-entity bbox histograms / arrangements
    # can be computed downstream for transient-vs-permanent
    # classification at the perception-entity level.
    log.frame_history.append(initial_frame.copy())
    for bbox, pals, size, bm_id, topo_id, shape_id, scaled_id, cell_set in cands:
        # Track identity is keyed by initial bitmap+location.  This key
        # NEVER changes for the lifetime of the track even when the
        # entity's bitmap, palette, or topo changes -- so an entity
        # that recolors or rotates remains the SAME track.
        track_id = f"{bm_id}_at_{bbox[0]}_{bbox[1]}"
        log.tracks[track_id] = EntityTrack(
            initial_bitmap_id = bm_id,
            initial_bbox      = bbox,
            initial_size      = size,
            initial_palettes  = pals,
            bbox_history      = [bbox],
            bitmap_id_history = [bm_id],
            topo_id_history   = [topo_id],
            shape_id_history  = [shape_id],
            scaled_id_history = [scaled_id],
            palette_history   = [pals],
            size_history      = [size],
            cell_set_history  = [cell_set],
            n_frames_seen     = 1,
        )
    return log


def observe_step(
    log:           InteractionLog,
    frame_pre:     np.ndarray,
    action:        str,
    frame_post:    np.ndarray,
    bg_palettes:   Sequence[int] = (),
) -> None:
    """Record one (frame_pre, action, frame_post) observation.

    Identity continuity through property changes is the core invariant:
    an entity that recolors, rotates, or repaints itself in place is
    STILL THE SAME ENTITY -- it just has new properties.  The matcher
    tries multiple identity tiers, from strict to loose, so the track
    follows the entity through any single-step property change:

      P1 (motion, no property change):    same bitmap_id within radius
      P2 (recolor in place):              same bbox, same size, same shape_id
                                          (palette-normalized -- catches
                                          color swap on same silhouette)
      P3 (recolor with motion OR redraw): same topo_id (silhouette) +
                                          same size, within radius
      P4 (size-tolerant fallback):        same scaled_id, within radius
                                          (catches grow/shrink of same shape)

    A match in P2/P3/P4 produces a change_event labelled by which
    property differs (palette, pattern, both).
    """
    log.n_steps += 1
    log.frame_history.append(frame_post.copy())
    post_cands = _extract_candidates_per_palette(frame_post, bg_palettes=bg_palettes)

    matched_post: set = set()
    MATCH_RADIUS = 6.0

    def _classify_change(prev_pals, prev_topo, prev_shape,
                         new_pals, new_topo, new_shape) -> str:
        pal_changed   = (new_pals  != prev_pals)
        topo_changed  = (new_topo  != prev_topo)
        shape_changed = (new_shape != prev_shape)
        if pal_changed and (topo_changed or shape_changed):
            return "palette+pattern_change"
        if pal_changed:
            return "palette_change"
        if topo_changed or shape_changed:
            return "pattern_change"
        return ""   # bitmap_id changed but neither tier differs -- noise

    for tid, tr in log.tracks.items():
        if tr.disappeared_at_step is not None:
            continue
        last_bbox     = tr.bbox_history[-1]
        last_topo     = tr.topo_id_history[-1]
        last_bm       = tr.bitmap_id_history[-1]
        last_shape    = tr.shape_id_history[-1]
        last_scaled   = tr.scaled_id_history[-1]
        last_pals     = tr.palette_history[-1]
        last_size     = tr.size_history[-1]
        best: Optional[Tuple[int, float, tuple, str]] = None  # (idx, dist, cand, tier)

        # P1: exact bitmap_id within radius (motion, no property change).
        for pi, c in enumerate(post_cands):
            if pi in matched_post:
                continue
            pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c
            if pbm != last_bm:
                continue
            dist = _bbox_distance(last_bbox, pbbox)
            if dist > MATCH_RADIUS:
                continue
            if best is None or dist < best[1]:
                best = (pi, dist, c, "P1_bitmap")

        # P2: same bbox + same size + same SHAPE (recolor in place --
        # palette swap on the same silhouette/footprint).  Exact bbox
        # rules out region-fragmentation artifacts that just happen to
        # share shape at a nearby position.
        if best is None:
            for pi, c in enumerate(post_cands):
                if pi in matched_post:
                    continue
                pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c
                if pbbox != last_bbox:
                    continue
                if psize != last_size:
                    continue
                if pshape != last_shape:
                    continue
                best = (pi, 0.0, c, "P2_recolor_in_place")
                break

        # P3: same topo (silhouette) + same size within radius
        # (recolor with motion, or pattern redraw at same scale).
        if best is None:
            for pi, c in enumerate(post_cands):
                if pi in matched_post:
                    continue
                pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c
                if ptopo != last_topo:
                    continue
                if psize != last_size:
                    continue
                dist = _bbox_distance(last_bbox, pbbox)
                if dist > MATCH_RADIUS:
                    continue
                if best is None or dist < best[1]:
                    best = (pi, dist, c, "P3_silhouette")

        # P4: same scaled (size-normalized) within radius.  Catches an
        # entity that grew or shrank while keeping its essential shape.
        if best is None:
            for pi, c in enumerate(post_cands):
                if pi in matched_post:
                    continue
                pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c
                if pscaled != last_scaled:
                    continue
                dist = _bbox_distance(last_bbox, pbbox)
                if dist > MATCH_RADIUS:
                    continue
                if best is None or dist < best[1]:
                    best = (pi, dist, c, "P4_scaled")

        # P5: SAME PALETTE at OVERLAPPING bbox -- a partial-content
        # change in place (some cells of this palette flipped to another
        # palette, or the partial-loss reverted).  Symmetric: either
        # post is contained in last (shrinking), or last is contained
        # in post (recovering).  Without this, the chamber pal-5 region
        # gets lost when it reverts to its original size at the step
        # AFTER a trigger fires.
        if best is None:
            for pi, c in enumerate(post_cands):
                if pi in matched_post:
                    continue
                pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c
                if ppals != last_pals:
                    continue
                # Bboxes must EITHER fully contain each other OR overlap
                # by at least half the smaller's area.  Same-bbox match
                # is the common case.
                last_contains = (last_bbox[0] <= pbbox[0]
                                 and last_bbox[1] <= pbbox[1]
                                 and last_bbox[2] >= pbbox[2]
                                 and last_bbox[3] >= pbbox[3])
                post_contains = (pbbox[0] <= last_bbox[0]
                                 and pbbox[1] <= last_bbox[1]
                                 and pbbox[2] >= last_bbox[2]
                                 and pbbox[3] >= last_bbox[3])
                if not (last_contains or post_contains):
                    continue
                if best is None:
                    best = (pi, 0.0, c, "P5_partial_palette_change")
                    break

        # P_slide: same bitmap_id (OR same shape_id at same size)
        # anywhere in the frame -- no radius constraint.  Catches an
        # entity that was pushed off from one location and resumed at
        # a non-adjacent one in the same step (bounce-plate slide).
        # Only consult this tier when no positional tier matched.  Also
        # require the source size to be > 1 px and the bitmap to be
        # GLOBALLY UNIQUE among unmatched candidates -- a generic 1-px
        # bitmap can appear inside many entities and would produce
        # spurious "slide" matches between unrelated sprites.
        if best is None and last_size > 1:
            # Count how many remaining candidates share the same
            # bitmap_id; if more than one, the match is ambiguous.
            same_bm_idxs = [pi for pi, c in enumerate(post_cands)
                            if pi not in matched_post and c[3] == last_bm]
            if len(same_bm_idxs) == 1:
                pi = same_bm_idxs[0]
                c = post_cands[pi]
                pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c
                dist = _bbox_distance(last_bbox, pbbox)
                if dist > MATCH_RADIUS:
                    best = (pi, dist, c, "P_slide")

        if best is None:
            # Distinguish OCCLUSION from real disappearance: if the
            # agent (an entity we know moves with actions) now occupies
            # the same bbox the lost track had, the track is most likely
            # hidden by the agent, not gone from the game state.  Mark
            # it as occluded so post-hoc analysis can re-acquire it
            # when the agent moves away, instead of treating it as
            # permanent disappearance.
            occluded = False
            for other_tid, other_tr in log.tracks.items():
                if other_tid == tid:
                    continue
                if not other_tr.moves_with_actions:
                    continue
                if not other_tr.bbox_history:
                    continue
                ob = other_tr.bbox_history[-1]
                if (ob[0] <= last_bbox[2] and last_bbox[0] <= ob[2]
                        and ob[1] <= last_bbox[3] and last_bbox[1] <= ob[3]):
                    occluded = True
                    break
            if occluded:
                # Don't mark disappeared; carry the last known bbox so
                # P1 can re-acquire on a future frame.  The track's
                # state freezes at this step.
                if not hasattr(tr, "occluded_steps"):
                    tr.occluded_steps = set()
                tr.occluded_steps.add(log.n_steps)
            else:
                tr.disappeared_at_step = log.n_steps
            continue

        pi, dist, c_match, tier = best
        pbbox, ppals, psize, pbm, ptopo, pshape, pscaled, pcells = c_match
        matched_post.add(pi)

        # Update the track -- same track_id; bitmap_id and palette
        # may differ from before, but identity continues.
        prev_cells = tr.cell_set_history[-1] if tr.cell_set_history else frozenset()
        tr.bbox_history.append(pbbox)
        tr.bitmap_id_history.append(pbm)
        tr.topo_id_history.append(ptopo)
        tr.shape_id_history.append(pshape)
        tr.scaled_id_history.append(pscaled)
        tr.palette_history.append(ppals)
        tr.size_history.append(psize)
        tr.cell_set_history.append(pcells)
        tr.n_frames_seen += 1
        if dist > 0:
            tr.total_displacement += dist
            if tier != "P_slide":
                # Only within-radius (continuous) matches count
                # toward the honest motion signal.  P_slide is a
                # fallback that can mis-associate vanished entities
                # with same-bitmap candidates elsewhere; its motion
                # contribution stays in total_displacement for
                # bookkeeping but is excluded from the signal that
                # gates agent promotion.
                tr.continuous_displacement += dist
            log.motion_events.append({
                "step": log.n_steps, "action": action,
                "track": tid, "from": last_bbox, "to": pbbox, "dist": dist,
                "match_tier": tier,
            })
            # Teleport tier: record a distinct event tying the action
            # to entrance (last_bbox) and exit (pbbox) for portal
            # discovery downstream.
            if tier == "P_slide":
                log.slide_events.append({
                    "step":      log.n_steps,
                    "action":    action,
                    "tid":       tid,
                    "from_bbox": last_bbox,
                    "to_bbox":   pbbox,
                    "dist":      dist,
                })

        # Property-change accounting.  A change in the raw cell set
        # (pixel positions) is an APPEARANCE property change only
        # when it isn't fully explained by translation.  A translated
        # entity has different cells in the new frame ONLY because
        # its bbox shifted; its shape didn't reshape, its palette
        # didn't swap, its silhouette didn't change.  Counting that
        # as a 'rotation_or_reshape' would conflate translation with
        # genuine in-place mutation.  Three cases to distinguish:
        #
        #   palette / topo / shape differed  -> real appearance change
        #     (palette_change / pattern_change / palette+pattern);
        #     register regardless of bbox shift.
        #   cells differ but palette/topo/shape unchanged AND bbox
        #     stayed the same  -> in-place reshape (rotation, mirror
        #     in same footprint); register as rotation_or_reshape.
        #   cells differ but palette/topo/shape unchanged AND bbox
        #     shifted  -> pure translation; skip (already captured in
        #     motion_events).
        #
        # Teleport tier is excluded -- it's already captured in
        # teleport_events; treating it as a reshape would create a
        # spurious change_event.
        cells_changed = (pcells != prev_cells) and tier != "P_slide"
        if cells_changed:
            kind = _classify_change(
                last_pals, last_topo, last_shape,
                ppals, ptopo, pshape,
            )
            if not kind:
                # No palette/topo/shape difference; cells differ only
                # because of position.  Treat as in-place reshape iff
                # the bbox didn't move.
                if pbbox == last_bbox:
                    kind = "rotation_or_reshape"
                else:
                    kind = ""   # translation; not an appearance change
            if kind:
                if "palette" in kind:
                    tr.n_palette_changes += 1
                if ("pattern" in kind) or kind == "rotation_or_reshape":
                    tr.n_pattern_changes += 1
                log.change_events.append({
                    "step": log.n_steps, "action": action,
                    "track": tid, "kind": kind,
                    "from_bm": tr.bitmap_id_history[-2], "to_bm": pbm,
                    "from_pal": list(last_pals),  "to_pal": list(ppals),
                    "match_tier": tier,
                })


def _spatial_zone(bbox: Tuple[int, int, int, int],
                  frame_h: int = 64, frame_w: int = 64) -> str:
    """Encode a bbox's position semantically.  'lower-left corner',
    'top-right quadrant', 'centre', etc.  Used for cross-level
    transfer: if next level has a similar entity in the same zone,
    it's a strong cue they perform the same function."""
    cr = (bbox[0] + bbox[2]) / 2.0
    cc = (bbox[1] + bbox[3]) / 2.0
    # Thirds: top/middle/bottom and left/centre/right.
    v = "top" if cr < frame_h / 3 else ("bottom" if cr > 2 * frame_h / 3 else "middle")
    h = "left" if cc < frame_w / 3 else ("right" if cc > 2 * frame_w / 3 else "centre")
    # Corner / edge / centre wording.
    if v == "middle" and h == "centre":
        return "centre"
    if v == "middle":
        return f"{h} side"
    if h == "centre":
        return f"{v} centre"
    return f"{v}-{h} corner"


def _bbox_palette_histogram(frame: np.ndarray,
                            bbox: Tuple[int, int, int, int]) -> Dict[int, int]:
    sub = frame[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
    vals, counts = np.unique(sub, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _detect_alignment(
    frame:           np.ndarray,
    working_bbox:    Tuple[int, int, int, int],
    reference_bbox:  Tuple[int, int, int, int],
) -> Tuple[bool, Optional[str]]:
    """Return (aligned, tier) where ``aligned`` is True iff the
    bitmap inside ``working_bbox`` matches the bitmap inside
    ``reference_bbox`` under SOME dihedral transform AND palette
    normalization.  ``tier`` says how strict the match was:
      'bitmap'  -- exact bitmap match (palettes + cells identical)
      'shape'   -- palette-normalized shapes identical
      'topo'    -- binary occupancy / silhouette identical
      'scaled'  -- same shape at canonical size (size-tolerant)

    For alignment puzzles (working_glyph mutated until it visually
    matches the reference) any of these tiers is informative; the
    strictest is 'bitmap', the loosest is 'scaled'.
    """
    from entity_fingerprint import (  # type: ignore
        fingerprint_from_bitmap, masked_bitmap_from_object,
    )
    bm_w = masked_bitmap_from_object(frame, working_bbox, cells=None)
    bm_r = masked_bitmap_from_object(frame, reference_bbox, cells=None)
    fp_w = fingerprint_from_bitmap(bm_w)
    fp_r = fingerprint_from_bitmap(bm_r)
    if fp_w.bitmap_id == fp_r.bitmap_id:
        return True, "bitmap"
    if fp_w.shape_id == fp_r.shape_id:
        return True, "shape"
    if fp_w.topo_id == fp_r.topo_id:
        return True, "topo"
    if fp_w.scaled_id == fp_r.scaled_id:
        return True, "scaled"
    return False, None


def _detect_rotation_direction(
    frame_before: np.ndarray, frame_after: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> Optional[str]:
    """If the bbox region of frame_after is a dihedral transform of
    the bbox region of frame_before, return the transform name
    ('rotated 90° CW', 'rotated 180°', 'reflected horizontal', etc.).
    Otherwise None (the change isn't a simple rotation/reflection)."""
    a = frame_before[bbox[0]:bbox[2]+1, bbox[1]:bbox[3]+1]
    b = frame_after[bbox[0]:bbox[2]+1,  bbox[1]:bbox[3]+1]
    if a.shape != b.shape:
        return None
    if np.array_equal(a, b):
        return "no rotation (identical)"
    # Check the 8 dihedral transforms of `a` against `b`.
    transforms = [
        ("rotated 90° clockwise",         np.rot90(a, k=-1)),
        ("rotated 180°",                  np.rot90(a, k=2)),
        ("rotated 90° counter-clockwise", np.rot90(a, k=1)),
        ("reflected horizontally",        np.fliplr(a)),
        ("reflected vertically",          np.flipud(a)),
        ("reflected along main diagonal", a.T),
        ("reflected along anti-diagonal", np.rot90(a, k=2).T),
    ]
    for name, t in transforms:
        if t.shape == b.shape and np.array_equal(t, b):
            return name
    return None


def learn_action_effects(log: InteractionLog) -> Dict[str, Tuple[float, float]]:
    """From the motion events in ``log``, return each action's average
    (delta_row, delta_col) effect on the agent's bbox centre.

    Game-agnostic: action ids are opaque strings ("ACTION1", "ACTION4",
    or whatever the env enumerates) and their effect is whatever the
    agent's motion under that action turned out to be.
    """
    from collections import defaultdict
    if log.agent_track_id is None:
        return {}
    deltas: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for me in log.motion_events:
        if me.get("track") != log.agent_track_id:
            continue
        prev = me.get("from") or (0, 0, 0, 0)
        new  = me.get("to")   or (0, 0, 0, 0)
        pr = (prev[0] + prev[2]) / 2.0
        pc = (prev[1] + prev[3]) / 2.0
        nr = (new[0]  + new[2])  / 2.0
        nc = (new[1]  + new[3])  / 2.0
        deltas[str(me.get("action", ""))].append((nr - pr, nc - pc))
    avg: Dict[str, Tuple[float, float]] = {}
    for action, ds in deltas.items():
        if not ds:
            continue
        avg[action] = (
            sum(d[0] for d in ds) / len(ds),
            sum(d[1] for d in ds) / len(ds),
        )
    return avg


def unvisited_entity_targets(
    log:                InteractionLog,
    *,
    min_size:           int = 3,
    visit_radius:       float = 4.0,
) -> List[Tuple[str, Tuple[int,int,int,int], Tuple[float,float]]]:
    """Return non-agent, non-tiny entity tracks the agent never reached
    during the recorded motion history.

    visit_radius: an agent bbox whose centre came within this many
    pixels of the target's bbox centre counts as 'visited'.
    """
    if log.agent_track_id is None:
        return []
    agent_tr = log.tracks[log.agent_track_id]
    visited_centres = [
        ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        for bb in agent_tr.bbox_history
    ]
    out: List = []
    for tid, tr in log.tracks.items():
        if tid == log.agent_track_id:
            continue
        if tid.startswith("compound_"):
            continue
        if tr.initial_size < min_size:
            continue
        if tr.disappeared_at_step is not None:
            continue
        bb = tr.initial_bbox
        target_centre = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        ever_visited = any(
            ((vc[0] - target_centre[0]) ** 2 +
             (vc[1] - target_centre[1]) ** 2) ** 0.5 <= visit_radius
            for vc in visited_centres
        )
        if not ever_visited:
            out.append((tid, bb, target_centre))
    # Closest unvisited first.
    if visited_centres:
        last_centre = visited_centres[-1]
        out.sort(key=lambda t:
                 ((t[2][0] - last_centre[0]) ** 2 +
                  (t[2][1] - last_centre[1]) ** 2))
    return out


def pick_action_toward(
    target_centre:  Tuple[float, float],
    agent_centre:   Tuple[float, float],
    action_effects: Dict[str, Tuple[float, float]],
    *,
    exclude: Optional[set] = None,
) -> Optional[str]:
    """Return the action whose learned delta best aligns with the
    direction from agent_centre to target_centre, optionally excluding
    actions that have already failed at this position.  None if no
    eligible action would make positive progress.
    """
    dr_needed = target_centre[0] - agent_centre[0]
    dc_needed = target_centre[1] - agent_centre[1]
    excluded = exclude or set()
    best_action = None
    best_score = 0.0
    for action, (adr, adc) in action_effects.items():
        if action in excluded:
            continue
        score = adr * dr_needed + adc * dc_needed
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def finalise_inferences(log: InteractionLog) -> None:
    """After all steps have been observed, derive cross-frame inferences.

    Idempotent: callable repeatedly (e.g. at each checkpoint of an
    exploration sweep) without accumulating duplicate inference entries.
    """
    # Reset derived state so repeated calls don't accumulate.
    log.static_tracks = []
    log.dynamic_tracks = []
    log.trigger_candidates = []
    log.agent_track_id = None
    log.bounce_plates = []

    # 0. Autonomous tracks: entities that change appearance at more
    # than half the play steps regardless of what the user did.  HUD
    # timers, status bars, animation loops -- the user does not
    # control these, so they cannot be the agent even if their bbox
    # happens to shift step-over-step.  Compute this BEFORE picking
    # agent_track_id so an autonomous animator never wins the
    # top-mover slot.
    log.autonomous_tracks = set()
    autonomous_tracks = log.autonomous_tracks
    for tid, tr in log.tracks.items():
        total_changes = tr.n_palette_changes + tr.n_pattern_changes
        if log.n_steps > 0 and total_changes / float(log.n_steps) > 0.5:
            autonomous_tracks.add(tid)

    # 1. Agent = the track that moved the most -- among non-autonomous
    # AND non-singleton tracks.  The agent moves on USER ACTION; an
    # entity that also "moves" every frame regardless of input is
    # animation, not agency.  A 1-pixel "track" is typically a
    # decorative detail riding on a larger sprite (an eye-dot, a
    # highlight pixel); it isn't a standalone entity even if its
    # motion mirrors the agent's.
    movers = [
        (tid, tr.total_displacement)
        for tid, tr in log.tracks.items()
        if tr.moves_with_actions
            and tid not in autonomous_tracks
            and tr.initial_size > 1
    ]
    movers.sort(key=lambda kv: -kv[1])
    if movers:
        log.agent_track_id = movers[0][0]

    # 2. Static tracks: never moved, never changed appearance.
    for tid, tr in log.tracks.items():
        if tr.stayed_static and tr.disappeared_at_step is None:
            log.static_tracks.append(tid)
        elif tr.appearance_changed and not tr.moves_with_actions:
            # Dynamic-in-place: working glyph / triggered entity.
            log.dynamic_tracks.append(tid)

    # 3. Trigger candidates: for each change event involving a dynamic
    # track, the agent's pre-action bbox is the trigger location.  The
    # CHANGE KIND determines the functional name the trigger gets:
    #   pattern_change (bitmap differs, palette same)  -> shape_changer
    #   palette_change (palette set differs)           -> color_changer
    # These are catalog primitive_ids; we assign them from OBSERVED
    # function, not from prior knowledge of the game.
    #
    # Autonomy filter: an entity that changes appearance at almost
    # EVERY step regardless of agent position is autonomous (a HUD
    # timer, an animation loop) -- NOT agent-triggered.  Triggered
    # entities change rarely and only when the agent is at the right
    # cell.  Skip change events whose track changes > 50% of steps;
    # those are autonomy artifacts, not trigger correlation evidence.
    if log.agent_track_id is None:
        return
    agent_tr = log.tracks[log.agent_track_id]
    # autonomous_tracks was already computed in step 0 above.

    # An entity has a PERMANENT state change iff its current
    # (last-recorded) cell-set differs from its initial cell-set.
    # Transient flashes return to the initial state and contribute
    # nothing to that comparison; only changes that PERSIST register.
    # This is how human vision distinguishes "feedback flash" from
    # "real state change".
    log.permanently_changed_tracks = set()
    permanently_changed_tracks = log.permanently_changed_tracks
    for tid, tr in log.tracks.items():
        if len(tr.cell_set_history) < 2:
            continue
        if tr.cell_set_history[-1] != tr.cell_set_history[0]:
            permanently_changed_tracks.add(tid)

    # Transient flash detection (separate from permanent state change).
    # A track's bbox had a TRANSIENT FLASH if some palette appeared in
    # its bbox at an intermediate step but is NOT in the initial OR
    # final state.  This is the "flash and revert" feedback signal --
    # often co-occurs across multiple entities to signal RELATEDNESS.
    # It does NOT mean the entity itself changed state; the change may
    # be a passing visual effect overlaid on the entity's region.
    log.entities_with_transient_flash: Dict[str, List[int]] = {}  # type: ignore
    if len(log.frame_history) >= 3:
        for tid, tr in log.tracks.items():
            initial_pals = set(
                _bbox_palette_histogram(log.frame_history[0], tr.initial_bbox).keys()
            )
            final_pals = set(
                _bbox_palette_histogram(log.frame_history[-1], tr.initial_bbox).keys()
            )
            intermediate_pals: set = set()
            for f in log.frame_history[1:-1]:
                intermediate_pals.update(
                    _bbox_palette_histogram(f, tr.initial_bbox).keys()
                )
            flash_pals = intermediate_pals - initial_pals - final_pals
            if flash_pals:
                log.entities_with_transient_flash[tid] = sorted(flash_pals)

    # Classify each change_event.  A change_event for an entity that
    # ultimately returned to its initial state is part of a transient
    # cycle; a change_event for an entity whose final state differs
    # is part of a permanent transformation.
    from collections import defaultdict
    transients_by_step: Dict[int, List[str]] = defaultdict(list)
    for ev in log.change_events:
        tid = ev["track"]
        if tid in permanently_changed_tracks:
            ev["permanence"] = "permanent"
        else:
            ev["permanence"] = "transient"
            transients_by_step[ev["step"]].append(tid)

    # Relatedness via transient co-occurrence: entities whose tracks
    # flashed at the same step are likely RELATED (the trigger is
    # signalling this pairing to the player).
    log.relatedness_pairs = []   # type: ignore
    for step, tids in transients_by_step.items():
        unique = sorted(set(tids))
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                log.relatedness_pairs.append((unique[i], unique[j], step))

    # Also build relatedness from bbox-level transient flashes (catches
    # the case where the chamber bbox saw a brief palette injection
    # even if the chamber's own per-palette track was classified
    # permanent due to side-effects from a moving sub-component).
    flash_tids = list(
        getattr(log, "entities_with_transient_flash", {}).keys()
    )
    for i in range(len(flash_tids)):
        for j in range(i + 1, len(flash_tids)):
            pair = (flash_tids[i], flash_tids[j])
            if not any(p[0] == pair[0] and p[1] == pair[1]
                       for p in log.relatedness_pairs):
                log.relatedness_pairs.append((pair[0], pair[1], -1))

    for ev in log.change_events:
        if ev["track"] in autonomous_tracks:
            continue
        # ONLY permanent changes drive trigger-role attribution.
        # Transient flashes are relatedness signals, not state changes.
        if ev.get("permanence") != "permanent":
            continue
        # Use POST-action bbox: the trigger is the cell the agent
        # ENDED UP at, not the cell it started from.  A change_event
        # at step S is observed in frame_post of step S; the agent's
        # bbox in frame_post is bbox_history[S] (initial is [0],
        # after step 1 is [1], etc.).
        step_idx = ev["step"]
        if step_idx < 0 or step_idx >= len(agent_tr.bbox_history):
            continue
        kind = ev["kind"]
        if kind == "rotation_or_reshape":
            # The affected entity's canonical shape stayed the same;
            # only its orientation / cell arrangement changed.  This
            # is a 'rotator' role, not shape_changer (which implies
            # the actual shape mutated to a different shape).
            functional_role = "rotator"
        elif "palette" in kind and "pattern" in kind:
            functional_role = "trigger"
        elif "palette" in kind:
            functional_role = "color_changer"
        elif "pattern" in kind:
            functional_role = "shape_changer"
        else:
            functional_role = "trigger"
        log.trigger_candidates.append({
            "step":          ev["step"],
            "agent_bbox":    agent_tr.bbox_history[step_idx],
            "affected":      ev["track"],
            "change_kind":   ev["kind"],
            "functional_role": functional_role,
        })

    # Bbox-level palette-swap detection.  The per-track tier system
    # misses the case where one palette's pixels are REPLACED by another
    # palette's pixels at the same bbox -- a recolour event where the
    # outgoing palette vanishes entirely and the incoming one grows.
    # We detect this at the parsed-entity level by comparing per-bbox
    # palette histograms between consecutive frames.  Reads the parent
    # bboxes from log.tracks (initial bbox per track).
    def _bboxes_overlap_simple(a, b) -> bool:
        return (a[0] <= b[2] and b[0] <= a[2]
                and a[1] <= b[3] and b[1] <= a[3])

    palette_swap_events: List[dict] = []
    if len(log.frame_history) >= 2:
        agent_tr = (log.tracks.get(log.agent_track_id)
                    if log.agent_track_id else None)
        # Consider LARGE region carriers only -- the working_glyph and
        # similar entities whose contents we expect to mutate.  Skip
        # tiny per-palette components that the agent constantly
        # overlaps as it walks.
        seen_bboxes: set = set()
        for tid, tr in log.tracks.items():
            if tr.initial_size < 20:
                continue
            if tid == log.agent_track_id:
                continue
            # Skip MOVING tracks -- their bbox shifts so a "histogram
            # change in the initial bbox" just reflects them leaving
            # that bbox, not a real palette swap of a stationary region.
            if tr.total_displacement > 0:
                continue
            bbox = tr.initial_bbox
            if bbox in seen_bboxes:
                continue
            seen_bboxes.add(bbox)
            r0, c0, r1, c1 = bbox
            prev_hist = None
            for step_idx, fr in enumerate(log.frame_history):
                if r0 < 0 or c0 < 0 or r1 >= fr.shape[0] or c1 >= fr.shape[1]:
                    prev_hist = None
                    continue
                # Skip frames where the agent overlaps this bbox --
                # changes are then likely just the agent's pixels, not
                # a real palette swap.
                if agent_tr is not None and step_idx < len(agent_tr.bbox_history):
                    if _bboxes_overlap_simple(agent_tr.bbox_history[step_idx], bbox):
                        prev_hist = None
                        continue
                sub = fr[r0:r1+1, c0:c1+1]
                vals, counts = np.unique(sub, return_counts=True)
                hist = {int(v): int(c) for v, c in zip(vals, counts)}
                if prev_hist is not None and hist != prev_hist:
                    diffs = {p: hist.get(p, 0) - prev_hist.get(p, 0)
                             for p in set(prev_hist) | set(hist)}
                    losers  = [p for p, d in diffs.items() if d < 0]
                    gainers = [p for p, d in diffs.items() if d > 0]
                    for lp in losers:
                        for gp in gainers:
                            if abs(diffs[lp] + diffs[gp]) <= 1 \
                                    and abs(diffs[lp]) >= 3:
                                palette_swap_events.append({
                                    "step":        step_idx,
                                    "entity_bbox": list(bbox),
                                    "entity_tid":  tid,
                                    "from_pal":    lp,
                                    "to_pal":      gp,
                                    "n_pixels":    abs(diffs[lp]),
                                })
                prev_hist = hist

    # Keep only PERMANENT swaps -- those whose to_pal is still present
    # in the bbox at the final frame.  Transient swaps (palette flips
    # back) are HUD effects or animations, not state changes.
    if palette_swap_events and len(log.frame_history) > 0:
        final = log.frame_history[-1]
        kept: List[dict] = []
        for sw in palette_swap_events:
            bb = sw["entity_bbox"]
            r0, c0, r1, c1 = bb
            sub = final[r0:r1+1, c0:c1+1]
            if int(sw["to_pal"]) in (int(v) for v in np.unique(sub)):
                kept.append(sw)
        palette_swap_events = kept
    log.palette_swap_events = palette_swap_events

    # Tie palette swaps to the agent's position at the swap step --
    # that cell is the trigger candidate.  ``observe_step`` indexes
    # actions starting at 1 (1-based), so frame_history[k] is the
    # frame AFTER step k.  When a swap is observed between frame
    # k-1 and frame k, the trigger is the cell the agent ENDED at
    # in step k (bbox_history[k]).
    if palette_swap_events and log.agent_track_id:
        agent_tr = log.tracks[log.agent_track_id]
        for sw in palette_swap_events:
            step = sw["step"]   # post-frame index in frame_history
            if step < 1 or step >= len(agent_tr.bbox_history):
                continue
            agent_bbox_at_step = agent_tr.bbox_history[step]
            log.trigger_candidates.append({
                "step":          step,
                "agent_bbox":    agent_bbox_at_step,
                "affected":      "bbox_" + "_".join(str(v) for v in sw["entity_bbox"]),
                "change_kind":   "palette_change",
                "functional_role": "color_changer",
                "from_pal":      sw["from_pal"],
                "to_pal":        sw["to_pal"],
            })

    # Portal endpoint discovery from teleport_events.  Each event ties
    # an action to (entrance, exit) cell centres.  We aggregate by
    # (entrance_centre, exit_centre) so repeated traversals reinforce
    # the endpoint pair and reveal which action drives the portal.
    if log.slide_events:
        portal_agg: Dict[Tuple[Tuple[int, int], Tuple[int, int]], dict] = {}
        for ev in log.slide_events:
            fb = ev["from_bbox"]
            tb = ev["to_bbox"]
            entrance = (int((fb[0] + fb[2]) / 2), int((fb[1] + fb[3]) / 2))
            exit_    = (int((tb[0] + tb[2]) / 2), int((tb[1] + tb[3]) / 2))
            key = (entrance, exit_)
            rec = portal_agg.setdefault(key, {
                "entrance": entrance, "exit": exit_,
                "count": 0, "actions": [], "tids": [],
                "steps": [],
            })
            rec["count"] += 1
            if ev.get("action") and ev["action"] not in rec["actions"]:
                rec["actions"].append(ev["action"])
            if ev["tid"] not in rec["tids"]:
                rec["tids"].append(ev["tid"])
            rec["steps"].append(ev["step"])
        log.bounce_plates = list(portal_agg.values())


# ---------------------------------------------------------------------------
# Apply inferences back to perception's parsed.json output.
# ---------------------------------------------------------------------------


def _is_floor(frame: np.ndarray, r0: int, c0: int, r1: int, c1: int,
              floor_palettes: Sequence[int]) -> bool:
    """True iff every pixel in the rectangle is a floor palette."""
    fh, fw = frame.shape
    if r0 < 0 or c0 < 0 or r1 >= fh or c1 >= fw:
        return False
    sub = frame[r0:r1+1, c0:c1+1]
    fset = set(int(p) for p in floor_palettes)
    return bool(fset) and all(int(v) in fset for v in np.unique(sub))


def _is_blocked(frame: np.ndarray, r0: int, c0: int, r1: int, c1: int,
                floor_palettes: Sequence[int]) -> bool:
    """True if the rectangle is OFF-frame or contains ANY non-floor cell
    (wall, marker, sprite, etc.)."""
    fh, fw = frame.shape
    if r0 < 0 or c0 < 0 or r1 >= fh or c1 >= fw:
        return True
    sub = frame[r0:r1+1, c0:c1+1]
    fset = set(int(p) for p in floor_palettes)
    if not fset:
        return False
    return not all(int(v) in fset for v in np.unique(sub))


def discover_bounce_plates(
    log:                "InteractionLog",
    floor_palettes:     Sequence[int],
    *,
    cell_size:          int = 5,
    cell_origin:        Tuple[int, int] = (0, 0),
    max_marker_size:    int = 8,
    wall_palettes:      Optional[Sequence[int]] = None,
    max_palette_total:  Optional[int] = None,
) -> List[dict]:
    """Frame-wide portal discovery.

    A portal MARKER is a small, static, non-floor sprite that sits
    adjacent (one cell offset) to a floor cell that has wall (or void)
    on the opposite side -- i.e., a sprite at the edge of the play
    area marking a special cell.

    The teleport EXIT DIRECTION is the vector from the marker toward
    the portal cell (continuing into the play area).  From the portal
    cell, we raycast in that direction one cell at a time until we hit
    a wall, another portal marker's cell, or another portal cell.

    Returns a list of portal records with:
      * marker_tid, marker_bbox, marker_palettes, marker_size_px
      * portal_bbox  (cell-sized bbox of the portal floor cell)
      * exit_direction_px (dr, dc) at cell granularity
      * exit_direction_name ("up" | "down" | "left" | "right")
      * predicted_exit_bbox (the cell where the agent would land)
      * predicted_exit_path  (list of bboxes traversed)
    """
    if not log.frame_history:
        return []
    frame = log.frame_history[0]
    fset = set(int(p) for p in floor_palettes)
    wall_set = set(int(p) for p in (wall_palettes or []))

    # Palette rarity: count total pixels per palette in the frame.
    # Portal markers use a palette dedicated to that role (a handful of
    # pixels in total).  Pickups, glyphs, walls all use palettes with
    # many more pixels.
    vals, counts = np.unique(frame, return_counts=True)
    palette_totals = {int(v): int(c) for v, c in zip(vals, counts)}

    # Collect candidate markers: small, static, monochrome, freestanding
    # non-floor non-wall tracks.  Each structural check is necessary;
    # together they distinguish a bounce-plate bar from pickups, glyphs,
    # and sub-pieces of larger multi-palette sprites.
    candidates: List[Tuple[str, EntityTrack]] = []
    for tid, tr in log.tracks.items():
        if tid == log.agent_track_id:
            continue
        if tr.total_displacement > 0.0:
            continue
        if tr.disappeared_at_step is not None:
            continue
        if tr.initial_size > max_marker_size:
            continue
        pals = [int(p) for p in tr.initial_palettes]
        # A bar is monochrome by construction (one palette).
        if len(pals) != 1:
            continue
        if pals[0] in fset or pals[0] in wall_set:
            continue
        # Optional global rarity gate (kept for callers that pass it,
        # but no longer the default -- the structural checks below
        # are stricter).
        if max_palette_total is not None:
            total_pixels = sum(palette_totals.get(p, 0) for p in pals)
            if total_pixels > max_palette_total:
                continue
        # Freestanding: the candidate must NOT sit inside the bbox of a
        # larger multi-palette compound (e.g. a piece of a color_changer
        # sprite isn't itself a bouncer).  We accept candidates that
        # share a bbox with the parent ONLY when the parent is also
        # monochrome and the same palette (the canonical "bar at the
        # edge of the chamber" pattern produces an outer wall component
        # and a bar; we want the bar).
        contained_in_compound = False
        for other_tid, other_tr in log.tracks.items():
            if other_tid == tid or other_tid == log.agent_track_id:
                continue
            if other_tr.initial_size <= tr.initial_size:
                continue
            if len(other_tr.initial_palettes) <= 1:
                continue
            ob = other_tr.initial_bbox
            cb = tr.initial_bbox
            # Strict containment with at least one px slack (so two
            # touching components don't accidentally trigger this).
            if (ob[0] <= cb[0] and ob[1] <= cb[1]
                    and ob[2] >= cb[2] and ob[3] >= cb[3]
                    and (ob[2] - ob[0] + 1) * (ob[3] - ob[1] + 1)
                        > (cb[2] - cb[0] + 1) * (cb[3] - cb[1] + 1)):
                contained_in_compound = True
                break
        if contained_in_compound:
            continue
        candidates.append((tid, tr))

    portals: List[dict] = []
    for tid, tr in candidates:
        r0, c0, r1, c1 = tr.initial_bbox
        # Locate the cell the marker centre falls in.
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        or_r, or_c = int(cell_origin[0]), int(cell_origin[1])
        cell_idx_r = (cr - or_r) // cell_size
        cell_idx_c = (cc - or_c) // cell_size
        cell_r0 = or_r + cell_idx_r * cell_size
        cell_c0 = or_c + cell_idx_c * cell_size
        cell_r1 = cell_r0 + cell_size - 1
        cell_c1 = cell_c0 + cell_size - 1
        # Structural rule (no "wall on back" needed): the bar
        # occupies one EDGE of its containing cell, and the push
        # direction is OUT through that edge.  Horizontal bars sit
        # at the top or bottom edge; vertical bars sit at the left
        # or right edge.  An interior sprite (a pickup, a glyph)
        # touches no edge and is rejected.
        marker_h = r1 - r0 + 1
        marker_w = c1 - c0 + 1
        # Structural property of a real bar: it spans the FULL cell
        # along its long axis (e.g. 5x1 in a cell_size=5 grid).
        # Sub-pieces of other entities that happen to sit at a cell
        # edge but don't span the full side are not standalone bars.
        if max(marker_h, marker_w) != cell_size:
            continue
        if marker_h > marker_w:
            # Vertical bar: meaningful edge is left or right.
            if c0 == cell_c0 and c1 < cell_c1:
                dr, dc, name = 0, -cell_size, "left"
            elif c1 == cell_c1 and c0 > cell_c0:
                dr, dc, name = 0, cell_size, "right"
            else:
                continue
        elif marker_w > marker_h:
            # Horizontal bar: meaningful edge is top or bottom.
            if r0 == cell_r0 and r1 < cell_r1:
                dr, dc, name = -cell_size, 0, "up"
            elif r1 == cell_r1 and r0 > cell_r0:
                dr, dc, name = cell_size, 0, "down"
            else:
                continue
        else:
            # Square (1x1 pixel) marker -- ambiguous direction;
            # require at least one full-cell edge alignment and pick
            # whichever edge is touched (rare; not expected in ls20).
            edges = []
            if r0 == cell_r0 and r1 < cell_r1:
                edges.append((-cell_size, 0, "up"))
            if r1 == cell_r1 and r0 > cell_r0:
                edges.append((cell_size, 0, "down"))
            if c0 == cell_c0 and c1 < cell_c1:
                edges.append((0, -cell_size, "left"))
            if c1 == cell_c1 and c0 > cell_c0:
                edges.append((0, cell_size, "right"))
            if len(edges) != 1:
                continue
            dr, dc, name = edges[0]

        pr0, pc0 = cell_r0 + dr, cell_c0 + dc
        pr1, pc1 = cell_r1 + dr, cell_c1 + dc
        # Activation cell must be floor.
        if not _is_floor(frame, pr0, pc0, pr1, pc1, fset):
            continue
        # Raycast from the activation cell in the push direction
        # one cell at a time until blocked by wall, off-frame, or
        # any non-floor obstacle.
        path: List[List[int]] = []
        cur = [pr0, pc0, pr1, pc1]
        while True:
            nxt = [cur[0] + dr, cur[1] + dc, cur[2] + dr, cur[3] + dc]
            if not _is_floor(frame, nxt[0], nxt[1], nxt[2], nxt[3], fset):
                break
            path.append(nxt)
            cur = nxt
        predicted_exit = path[-1] if path else [pr0, pc0, pr1, pc1]

        # Cell-coord projections.  The planner thinks in cell indices
        # anchored to the level's grid origin -- frame-pixel bboxes
        # don't compose well with BFS or A* over cells.  Each plate
        # record carries:
        #   marker_cell        : (r, c) containing the bar
        #   activation_cell    : the cell the agent must stand on
        #   resting_cell       : where the slide ends
        #   slide_path_cells   : list of cells traversed (including resting)
        def _to_cell(bb):
            ccr = (bb[0] + bb[2]) // 2
            ccc = (bb[1] + bb[3]) // 2
            return [(ccr - or_r) // cell_size, (ccc - or_c) // cell_size]

        marker_cell        = _to_cell(tr.initial_bbox)
        activation_cell    = _to_cell([pr0, pc0, pr1, pc1])
        resting_cell       = _to_cell(predicted_exit)
        slide_path_cells   = [_to_cell(p) for p in path]

        portals.append({
            "marker_tid":          tid,
            "marker_bbox":         list(tr.initial_bbox),
            "marker_palettes":     list(tr.initial_palettes),
            "marker_size_px":      tr.initial_size,
            "marker_cell":         marker_cell,
            "portal_bbox":         [pr0, pc0, pr1, pc1],
            "activation_cell":     activation_cell,
            "exit_direction_px":   [dr, dc],
            "exit_direction_name": name,
            "predicted_exit_bbox": list(predicted_exit),
            "resting_cell":        resting_cell,
            "slide_path_cells":    slide_path_cells,
            "predicted_path_len":  len(path),
        })
    return portals


def derive_cell_system(
    frame:           np.ndarray,
    floor_palettes:  Sequence[int],
    *,
    cell_size:       int = 5,
    log:             Optional["InteractionLog"] = None,
) -> Dict[str, Any]:
    """Derive the level's cell grid from the frame and (optionally)
    motion observations.

    The cell grid is anchored to the top-left of the floor region --
    the first frame pixel whose palette is in ``floor_palettes``.
    Cell `(0, 0)` is rooted there; subsequent cells tile rightward and
    downward at the ``cell_size`` pitch.  This gives consistent,
    structural cell coordinates without per-level hardcoded constants.

    When ``log`` is supplied AND at least one motion event exists,
    ``cell_size`` is sanity-checked against the observed displacement
    (cardinal motion moves the agent by exactly one cell).  A
    mismatch is reported in the returned dict's ``warnings`` list
    rather than overriding the supplied ``cell_size`` -- the caller
    decides whether to trust the observation.
    """
    # Pick the cell-grid origin offset that maximises the number of
    # small static "marker" tracks (potential launcher bars) sitting
    # at a cell edge.  This is the structural signal the launcher
    # detector uses, so optimising for it makes the discovery work
    # without per-level hand-tuning.  Origin equivalence: only the
    # offset mod cell_size matters; we search the 25-element space.
    candidates: Dict[Tuple[int, int], int] = {(or_r, or_c): 0
                                              for or_r in range(cell_size)
                                              for or_c in range(cell_size)}

    def _marker_at_edge(r0: int, c0: int, r1: int, c1: int,
                        or_r: int, or_c: int) -> bool:
        # Compute the containing cell for the marker's centre.
        cr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        cell_idx_r = (cr - or_r) // cell_size
        cell_idx_c = (cc - or_c) // cell_size
        cell_r0 = or_r + cell_idx_r * cell_size
        cell_c0 = or_c + cell_idx_c * cell_size
        cell_r1 = cell_r0 + cell_size - 1
        cell_c1 = cell_c0 + cell_size - 1
        marker_h = r1 - r0 + 1
        marker_w = c1 - c0 + 1
        if marker_h > marker_w:
            # Vertical bar: check left or right edge.
            return ((c0 == cell_c0 and c1 < cell_c1)
                    or (c1 == cell_c1 and c0 > cell_c0))
        if marker_w > marker_h:
            return ((r0 == cell_r0 and r1 < cell_r1)
                    or (r1 == cell_r1 and r0 > cell_r0))
        return False

    if log is not None:
        # Brute force: run the launcher detector for each candidate
        # origin offset and score by the number of plates found.  This
        # is structurally correct -- the right origin is the one that
        # makes the most launcher patterns valid (bar at cell edge +
        # activation cell on floor + raycast lands somewhere reachable).
        # cell_size**2 calls is cheap (25 for ls20).
        wall_pals: Sequence[int] = []
        for or_r in range(cell_size):
            for or_c in range(cell_size):
                try:
                    plates = discover_bounce_plates(
                        log, floor_palettes, cell_size=cell_size,
                        cell_origin=(or_r, or_c), wall_palettes=wall_pals,
                    )
                    candidates[(or_r, or_c)] = len(plates)
                except Exception:
                    candidates[(or_r, or_c)] = 0

    best_count = max(candidates.values()) if candidates else 0
    if best_count > 0:
        # Origin equivalence class with the most edge-aligned markers.
        origin = max(candidates,
                     key=lambda k: (candidates[k], -k[0], -k[1]))
    else:
        # Fallback: floor-palette largest connected component top-left.
        fset = set(int(p) for p in floor_palettes)
        if fset:
            mask = np.isin(frame, list(fset))
            if bool(mask.any()):
                rows, cols = np.where(mask)
                origin = (int(rows.min()), int(cols.min()))
        else:
            origin = (0, 0)

    warnings: List[str] = []
    if log is not None and log.motion_events:
        for me in log.motion_events:
            fb = me.get("from")
            tb = me.get("to")
            if not (fb and tb):
                continue
            dist_r = abs(int(tb[0]) - int(fb[0]))
            dist_c = abs(int(tb[1]) - int(fb[1]))
            observed = max(dist_r, dist_c)
            if observed > 0 and observed != cell_size:
                warnings.append(
                    f"observed motion magnitude {observed}px does not match "
                    f"cell_size={cell_size}px (step={me.get('step')})"
                )
            if observed > 0:
                break
    return {
        "origin":    list(origin),
        "cell_size": cell_size,
        "warnings":  warnings,
    }


def build_launcher_graph(
    parsed: Mapping[str, Any],
) -> Dict[Tuple[int, int], dict]:
    """Build a directed graph of launcher slides for the planner.

    Each entry maps an activation cell to the cell the agent will land
    on after pressing any cardinal action while standing there:

    ::

        graph[(ar, ac)] = {
            "resting_cell":    (rr, rc),
            "push":            "up" | "down" | "left" | "right",
            "plate_idx":       int,
            "observed":        bool,
            "slide_distance":  int,   # cells traversed
            "slide_path":      [(r, c), ...],
            "trigger_actions": [...],  # which actions are known to fire it
        }

    The planner can union these edges with normal walk-edges to compute
    full reachability over both walking and launcher slides.  Chains
    arise naturally when a plate's resting_cell matches another plate's
    activation_cell.
    """
    graph: Dict[Tuple[int, int], dict] = {}
    log = (parsed.get("interaction_log") or {})
    plates = log.get("bounce_plates") or []
    for i, p in enumerate(plates):
        ac = p.get("activation_cell")
        rc = p.get("resting_cell")
        if not (ac and rc):
            continue
        key = (int(ac[0]), int(ac[1]))
        graph[key] = {
            "resting_cell":   (int(rc[0]), int(rc[1])),
            "push":           p.get("exit_direction_name"),
            "plate_idx":      i,
            "observed":       p.get("status") == "observed",
            "slide_distance": int(p.get("predicted_path_len") or 0),
            "slide_path":     [(int(c[0]), int(c[1]))
                               for c in (p.get("slide_path_cells") or [])],
            "trigger_actions": list(p.get("trigger_actions") or []),
        }
    return graph


# Backwards-compatibility aliases for callers that still use the old
# (teleport/portal) names.  The mechanism is unchanged; only the
# vocabulary now reflects the physical analogue (push → slide →
# stop at obstacle), which is how human players naturally read it.
discover_portals = discover_bounce_plates


def validate_slides_against_predictions(
    plates:         Sequence[dict],
    log:            "InteractionLog",
) -> List[dict]:
    """Cross-check predicted bounce-plate slide outcomes against
    observed slide events.  Returns the plates list annotated with
    verification status: ``observed`` (an event matched) or
    ``predicted_only`` (no event yet), plus an ``observed_exit_bbox``
    field when matched.
    """
    out: List[dict] = []
    used_events: set = set()
    for p in plates:
        plate_cell_centre = (
            (p["portal_bbox"][0] + p["portal_bbox"][2]) / 2.0,
            (p["portal_bbox"][1] + p["portal_bbox"][3]) / 2.0,
        )
        # Match ALL slide events whose entry centre lies inside (or
        # touching) this plate's activation cell.  A plate may have
        # been activated multiple times (with different actions);
        # each observation refines the trigger_actions set.
        trigger_actions: List[str] = []
        observed_exit_bbox = None
        last_obs_action = None
        last_pred_error = None
        for i, ev in enumerate(log.slide_events):
            if i in used_events:
                continue
            fb = ev["from_bbox"]
            entry_centre = ((fb[0] + fb[2]) / 2.0, (fb[1] + fb[3]) / 2.0)
            if abs(entry_centre[0] - plate_cell_centre[0]) >= 4 \
                    or abs(entry_centre[1] - plate_cell_centre[1]) >= 4:
                continue
            used_events.add(i)
            act = ev.get("action")
            if act and act not in trigger_actions:
                trigger_actions.append(act)
            observed_exit_bbox = list(ev["to_bbox"])
            last_obs_action = act
            # Compare predicted vs observed centres for THIS event.
            pe = p["predicted_exit_bbox"]
            pred_centre = ((pe[0]+pe[2])/2.0, (pe[1]+pe[3])/2.0)
            obs_centre  = ((observed_exit_bbox[0]+observed_exit_bbox[2])/2.0,
                           (observed_exit_bbox[1]+observed_exit_bbox[3])/2.0)
            last_pred_error = round(
                ((pred_centre[0]-obs_centre[0])**2
                 + (pred_centre[1]-obs_centre[1])**2) ** 0.5, 2
            )
        rec = dict(p)
        if trigger_actions:
            rec["observed_exit_bbox"]  = observed_exit_bbox
            rec["observed_action"]     = last_obs_action
            rec["trigger_actions"]     = trigger_actions
            rec["prediction_error_px"] = last_pred_error
            rec["status"]              = "observed"
        else:
            rec["trigger_actions"]     = []
            rec["status"]              = "predicted_only"
        out.append(rec)
    return out


validate_portals_against_teleports = validate_slides_against_predictions


_BOUNCE_PLATE_BEHAVIOR_CLASS = "push_slide_until_blocked"


def _role_for_behavior_class(catalog: Any,
                             behavior_class: str) -> Optional[str]:
    """Look up the catalog and return the primitive_id of the entity_role
    whose ``interaction_signature.behavior_class`` equals
    ``behavior_class``.  Returns ``None`` when the catalog has no such
    primitive -- callers must handle that explicitly rather than fall
    back to a hardcoded role name.
    """
    if catalog is None:
        return None
    entries = catalog.by_kind.get("entity_role") or []
    for e in entries:
        sig = e.interaction_signature or {}
        if sig.get("behavior_class") == behavior_class:
            return e.primitive_id
    return None


def apply_bounce_plate_roles(
    parsed:   Mapping[str, Any],
    plates:   Sequence[dict],
    *,
    catalog:  Any = None,
) -> int:
    """For each discovered bounce plate, reclassify the entity that
    corresponds to its bar (pusher) using the role primitive_id that
    the catalog declares for the ``push_slide_until_blocked`` behavior
    class.

    The role string is sourced from the catalog, not from a Python
    literal -- so adding/renaming primitives is a catalog edit and the
    perception code does not need to change.

    Mutates ``parsed`` in place; returns the number of overrides
    applied.  Returns ``0`` when the catalog has no primitive declaring
    the bounce-plate behavior class.
    """
    if not plates:
        return 0
    role_id = _role_for_behavior_class(catalog, _BOUNCE_PLATE_BEHAVIOR_CLASS)
    if role_id is None:
        return 0
    n_changes = 0
    # Build a map: marker_tid -> push_direction so we can include
    # direction info in the correction line.
    tid_to_dir = {p["marker_tid"]: p.get("exit_direction_name", "")
                  for p in plates if p.get("marker_tid")}
    # Marker bitmap_ids (strip "_at_R_C" suffix to get the canonical id).
    tid_to_bm = {}
    for tid in tid_to_dir:
        if "_at_" in tid:
            tid_to_bm[tid.split("_at_")[0]] = tid
        else:
            tid_to_bm[tid] = tid

    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        matched = ent.get("_matched_track")
        sub_bms = [s.get("bitmap_id") for s in (ent.get("_sub_bitmaps") or [])
                   if isinstance(s, dict)]
        hit_tid = None
        if matched and matched in tid_to_dir:
            hit_tid = matched
        else:
            # Compare bitmap_ids (canonical, without the "_at_R_C" suffix).
            for bm in sub_bms:
                if not bm:
                    continue
                # Both the bare bitmap_id and the prefix may appear in
                # tid_to_bm; check both.
                for bm_key, full_tid in tid_to_bm.items():
                    if bm == bm_key or bm.startswith(bm_key[:18]):
                        hit_tid = full_tid
                        break
                if hit_tid:
                    break
        if hit_tid is None:
            continue
        direction = tid_to_dir.get(hit_tid, "")
        old_role = ent.get("role")
        # Bounce-plate detection is causal evidence: the catalog
        # primitive matched structurally + the slide can be raycast-
        # predicted + (in many cases) verified against an observed
        # slide event.  Tier it as causal so the resolver picks it
        # over any incidental shape collision.
        try:
            from .level_memory import add_role_candidate
            add_role_candidate(
                ent, role_id, "causal",
                f"Bounce-plate match: pushes agent {direction!r}",
            )
        except Exception:
            pass
        if old_role == role_id:
            continue
        ent["role"] = role_id
        ent.setdefault("_corrections", []).append(
            f"Bounce-plate match: this bar pushes the agent {direction!r} "
            f"-> role={role_id} (catalog primitive with "
            f"behavior_class={_BOUNCE_PLATE_BEHAVIOR_CLASS!r}; "
            f"overrides VLM's {old_role!r})"
        )
        n_changes += 1
    return n_changes


def _enrich_portal_endpoints(
    log:           InteractionLog,
    cell_system:   Optional[Mapping[str, Any]],
) -> None:
    """Enrich each portal_endpoint with (a) cell-coord versions of the
    entrance/exit centres and (b) a candidate visual marker -- a small
    static entity at or adjacent to the entrance/exit cell that may be
    the portal's visible signpost.

    Mutates ``log.bounce_plates`` in place.  No-op when there are no
    teleport events.
    """
    if not log.bounce_plates:
        return
    # Cell-coord conversion (frame_pixel -> cell index).
    cs = cell_system or {}
    cell_size = int(cs.get("cell_size") or 1)
    origin = cs.get("origin") or [0, 0]
    or_r, or_c = int(origin[0]), int(origin[1])

    def to_cell(pt: Tuple[int, int]) -> Tuple[int, int]:
        return ((pt[0] - or_r) // cell_size,
                (pt[1] - or_c) // cell_size)

    # Marker detection: a track is a candidate marker if it is small,
    # static (no motion), and lies within MARKER_RADIUS pixels of the
    # portal cell centre.  Adjacent markers (e.g. a signpost sprite
    # next to the entrance) count; the agent itself is excluded
    # (it occupies the cell at teleport time, but isn't the marker).
    MARKER_RADIUS = 8.0   # ~ 1.5 cells

    def find_marker(centre: Tuple[int, int]) -> Optional[dict]:
        best: Optional[Tuple[float, str, EntityTrack]] = None
        for tid, tr in log.tracks.items():
            # Skip the agent (it visited but isn't the marker).
            if tid == log.agent_track_id:
                continue
            if tr.total_displacement > 0.0:
                # Movers aren't fixed markers.
                continue
            if tr.initial_size > 25:
                # Large entities (walls / regions) aren't markers.
                continue
            r0, c0, r1, c1 = tr.initial_bbox
            cr = (r0 + r1) / 2.0
            cc = (c0 + c1) / 2.0
            dist = ((cr - centre[0]) ** 2 + (cc - centre[1]) ** 2) ** 0.5
            if dist > MARKER_RADIUS:
                continue
            if best is None or dist < best[0]:
                best = (dist, tid, tr)
        if best is None:
            return None
        dist, tid, tr = best
        return {
            "tid":        tid,
            "bitmap_id":  tr.initial_bitmap_id,
            "shape_id":   tr.shape_id_history[0]  if tr.shape_id_history  else None,
            "topo_id":    tr.topo_id_history[0]   if tr.topo_id_history   else None,
            "bbox":       list(tr.initial_bbox),
            "size_px":    tr.initial_size,
            "palettes":   list(tr.initial_palettes),
            "distance":   round(dist, 2),
        }

    for pe in log.bounce_plates:
        entrance = pe["entrance"]
        exit_    = pe["exit"]
        pe["entrance_cell"]   = list(to_cell(entrance))
        pe["exit_cell"]       = list(to_cell(exit_))
        pe["entrance_marker"] = find_marker(entrance)
        pe["exit_marker"]     = find_marker(exit_)


def apply_to_parsed(
    parsed:     Mapping[str, Any],
    log:        InteractionLog,
    *,
    initial_frame: np.ndarray,
    cell_system:   Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Take an initial parsed.json-shaped dict and overwrite role labels
    based on what was observed through interaction.

    Returns a new dict (does not mutate `parsed`).
    """
    out = dict(parsed)
    out["entities"] = [dict(e) for e in (parsed.get("entities") or [])]
    # Enrich portal endpoints with cell-coord locations and candidate
    # visual markers BEFORE serialising the interaction_log block.
    _enrich_portal_endpoints(log, cell_system)
    # Alignment detection: scan every step to see whether the
    # working_glyph and reference_glyph bbox contents EVER matched.
    # Strict 'bitmap' tier = exact bitmap match (palettes + cells
    # identical under dihedral transform); looser tiers tolerate
    # palette / size differences.  For alignment_match win conditions
    # the strict tier is what wins the level.
    alignment_history: List[dict] = []
    working_entity = next(
        (e for e in out["entities"] if e.get("role") == "working_glyph"),
        None,
    )
    reference_entity = next(
        (e for e in out["entities"] if e.get("role") == "reference_glyph"),
        None,
    )
    if (working_entity is not None
            and reference_entity is not None
            and log.frame_history):
        w_bbox = tuple(working_entity.get("bbox_pixels") or (0,0,0,0))
        r_bbox = tuple(reference_entity.get("bbox_pixels") or (0,0,0,0))
        for step, frame in enumerate(log.frame_history):
            aligned, tier = _detect_alignment(frame, w_bbox, r_bbox)
            if aligned:
                alignment_history.append({
                    "step": step, "tier": tier,
                    "working_id": working_entity.get("id"),
                    "reference_id": reference_entity.get("id"),
                })

    out["interaction_log"] = {
        "n_steps":            log.n_steps,
        "agent_track":        log.agent_track_id,
        "static_tracks":      list(log.static_tracks),
        "dynamic_tracks":     list(log.dynamic_tracks),
        "motion_events":      list(log.motion_events),
        "change_events":      list(log.change_events),
        "trigger_candidates": list(log.trigger_candidates),
        "relatedness_pairs":  list(getattr(log, "relatedness_pairs", [])),
        "entities_with_transient_flash":
            dict(getattr(log, "entities_with_transient_flash", {})),
        "alignment_history":  alignment_history,
        "alignment_achieved": len(alignment_history) > 0,
        "slide_events":    list(log.slide_events),
        "bounce_plates":   list(log.bounce_plates),
    }

    # Map perception entities to tracks via mutually-best IoU.  Skip
    # region-role entities (wall, play_area, void) because their bboxes
    # are palette-extents that dominate any IoU calc against a small
    # entity track.
    REGION_ROLES = {"wall", "play_area", "void", "floor", "hud_background"}

    def _bbox_iou(a, b) -> float:
        ar0, ac0, ar1, ac1 = a
        br0, bc0, br1, bc1 = b
        rr0, cc0 = max(ar0, br0), max(ac0, bc0)
        rr1, cc1 = min(ar1, br1), min(ac1, bc1)
        if rr1 < rr0 or cc1 < cc0:
            return 0.0
        inter = (rr1 - rr0 + 1) * (cc1 - cc0 + 1)
        a_area = (ar1 - ar0 + 1) * (ac1 - ac0 + 1)
        b_area = (br1 - br0 + 1) * (br1 - br0 + 1)
        # Note: corrected b_area computation below.
        b_area = (br1 - br0 + 1) * (bc1 - bc0 + 1)
        return inter / max(1.0, a_area + b_area - inter)

    # Greedy mutual best-match: rank all (entity, track) pairs by IoU
    # and pick top scores while ensuring each entity/track is used
    # at most once.
    pairs: list = []
    for i, ent in enumerate(out["entities"]):
        if str(ent.get("role", "")) in REGION_ROLES:
            continue
        ebbox = ent.get("bbox_pixels") or [0, 0, 0, 0]
        for tid, tr in log.tracks.items():
            iou = _bbox_iou(ebbox, tr.initial_bbox)
            if iou > 0.0:
                pairs.append((iou, i, tid))
    pairs.sort(reverse=True)
    used_ent: set = set()
    used_track: set = set()
    matches: Dict[int, Tuple[str, float]] = {}
    for iou, ei, tid in pairs:
        if ei in used_ent or tid in used_track:
            continue
        matches[ei] = (tid, iou)
        used_ent.add(ei)
        used_track.add(tid)
    for i, ent in enumerate(out["entities"]):
        if i in matches:
            tid, iou = matches[i]
            ent["_matched_track"] = tid
            ent["_overlap"]       = iou
            # Propagate the matched track's fingerprint suite onto the
            # entity as its primary _sub_bitmaps record.  This is what
            # downstream KB matchers (bitmap / shape / topo) consume.
            # Without this, VLM-identified entities have no fingerprint
            # attached even though Layer A already computed one.
            tr = log.tracks.get(tid)
            if tr is not None:
                ent.setdefault("_sub_bitmaps", []).append({
                    "bitmap_id": tr.initial_bitmap_id,
                    "shape_id":  tr.shape_id_history[0]  if tr.shape_id_history  else None,
                    "topo_id":   tr.topo_id_history[0]   if tr.topo_id_history   else None,
                    "scaled_id": tr.scaled_id_history[0] if tr.scaled_id_history else None,
                    "bbox":      list(tr.initial_bbox),
                    "size_px":   tr.initial_size,
                    "palettes":  list(tr.initial_palettes),
                    "behaviour": "primary track (matched to VLM entity)",
                    "is_primary": True,
                })
        else:
            ent["_matched_track"] = None
            ent["_overlap"]       = 0.0

    # Build a map: track_id -> functional_role from trigger discoveries.
    # When the agent stood on / next to an entity AT THE MOMENT another
    # entity changed appearance, that entity's role can be inferred from
    # WHAT changed (pattern -> shape_changer, palette -> color_changer).
    # The catalog already defines these as primitive_ids; we just apply
    # them based on observed function.
    # Trigger-role attribution requires REPEATED, CONSISTENT
    # correlation: the same candidate entity must appear adjacent to
    # the agent during MULTIPLE change events.  A single co-occurrence
    # is correlation by passage, not causation -- e.g. the agent
    # moving UP across a row may correlate with an unrelated change
    # in a distant chamber once, but that doesn't make the row-edge
    # entity a trigger.  Below the threshold we surface the
    # correlation as a HYPOTHESIS in the entity's properties (not
    # as a confirmed role).
    MIN_TRIGGER_OBSERVATIONS = 2   # 2 consistent correlations is enough
    MIN_TRIGGER_SIZE         = 2   # avoid 1-px noise but accept genuinely
                                   # tiny triggers like a 2-pixel marker
    correlation_count: Dict[Tuple[str, str], int] = {}
    correlation_kind: Dict[Tuple[str, str], str] = {}

    def _bboxes_overlap(a, b) -> bool:
        return (a[0] <= b[2] and b[0] <= a[2]
                and a[1] <= b[3] and b[1] <= a[3])

    agent_tr_for_mask = (log.tracks.get(log.agent_track_id)
                         if log.agent_track_id else None)
    for tc in log.trigger_candidates:
        agent_bbox = tuple(tc["agent_bbox"])
        step_idx   = tc.get("step")
        agent_mask: frozenset = frozenset()
        if (agent_tr_for_mask is not None
                and step_idx is not None
                and agent_tr_for_mask.cell_set_history
                and 0 <= step_idx < len(agent_tr_for_mask.cell_set_history)):
            agent_mask = agent_tr_for_mask.cell_set_history[step_idx]
        # The agent's pixel MASK must overlap the candidate trigger's
        # pixel mask -- not just their bounding boxes.  For an
        # irregular sprite (e.g. a circular ball in a square bbox) the
        # bbox corners are background pixels that can edge-touch a
        # neighbour's bbox even when the two sprites never share a
        # real pixel.  Mask-overlap rejects that case so a rigid-link
        # diagonal whose bbox kisses the moving ball's bbox isn't
        # mis-credited as a trigger the agent "stepped on".  When
        # mask data isn't available for either side (e.g. region
        # tracks, or the palette-swap pseudo-trigger), fall back to
        # bbox overlap.
        #
        # When multiple candidates overlap, prefer the SMALLEST: a
        # discrete trigger nested inside a chamber should win over
        # the chamber itself.
        best_tid = None
        for tid, tr in log.tracks.items():
            if tid == log.agent_track_id:
                continue
            if tr.initial_size < MIN_TRIGGER_SIZE:
                continue
            cand_mask: frozenset = frozenset()
            if tr.cell_set_history:
                if (step_idx is not None
                        and 0 <= step_idx < len(tr.cell_set_history)):
                    cand_mask = tr.cell_set_history[step_idx]
                else:
                    cand_mask = tr.cell_set_history[0]
            if agent_mask and cand_mask:
                if agent_mask.isdisjoint(cand_mask):
                    continue
            else:
                if not _bboxes_overlap(agent_bbox, tr.initial_bbox):
                    continue
            if best_tid is None or tr.initial_size < log.tracks[best_tid].initial_size:
                best_tid = tid
        if best_tid is not None:
            key = (best_tid, tc["functional_role"])
            correlation_count[key] = correlation_count.get(key, 0) + 1
            correlation_kind[key] = tc["functional_role"]

    track_functional_role:    Dict[str, str] = {}
    track_trigger_hypothesis: Dict[str, dict] = {}
    for (tid, role), n in correlation_count.items():
        if n >= MIN_TRIGGER_OBSERVATIONS:
            track_functional_role[tid] = role
        else:
            track_trigger_hypothesis[tid] = {
                "hypothesised_role": role,
                "observations":      n,
                "needed_for_confirmation": MIN_TRIGGER_OBSERVATIONS,
            }

    # Helper: build a structured properties record for an entity from
    # its matched track and the cross-entity observation log.
    ## Visit-without-effect demotion: track which entities the agent
    ## stepped ON and whether any OTHER entity's appearance changed in
    ## that same step.  A perception-tagged trigger that we visited
    ## without observing any effect should have its role downgraded.
    ## Computed once; consumed in the per-entity correction loop.
    visited_without_effect: set = set()
    visited_with_effect:    set = set()
    if log.agent_track_id is not None:
        agent_tr = log.tracks[log.agent_track_id]
        for tid, tr in log.tracks.items():
            if tid == log.agent_track_id:
                continue
            agent_visited_step: Optional[int] = None
            for step_idx, ag_bb in enumerate(agent_tr.bbox_history):
                # bbox overlap test
                rr0 = max(ag_bb[0], tr.initial_bbox[0])
                cc0 = max(ag_bb[1], tr.initial_bbox[1])
                rr1 = min(ag_bb[2], tr.initial_bbox[2])
                cc1 = min(ag_bb[3], tr.initial_bbox[3])
                if rr1 >= rr0 and cc1 >= cc0:
                    agent_visited_step = step_idx
                    break
            if agent_visited_step is None:
                continue
            # Did any OTHER non-autonomous entity change during this step?
            visit_step_num = agent_visited_step   # 0-indexed bbox history -> step number
            effect_observed = False
            for ev in log.change_events:
                if ev.get("track") == tid:
                    continue   # changes to the visited entity itself don't count
                # Skip autonomous-track changes (HUD ticks).
                changed_tr = log.tracks.get(ev["track"])
                if changed_tr is None:
                    continue
                total_changes = changed_tr.n_palette_changes + changed_tr.n_pattern_changes
                if log.n_steps > 0 and total_changes / float(log.n_steps) > 0.5:
                    continue
                if abs(ev["step"] - visit_step_num) <= 1:
                    effect_observed = True
                    break
            # Also count bbox-level palette swaps as effects.  The
            # per-track tracker misses palette-substitution events that
            # cross palette tracks; the bbox swap detector catches them.
            if not effect_observed:
                for sw in getattr(log, "palette_swap_events", []) or []:
                    if abs(int(sw["step"]) - visit_step_num) <= 1:
                        effect_observed = True
                        break
            if effect_observed:
                visited_with_effect.add(tid)
            else:
                visited_without_effect.add(tid)

    def _properties_for_track(tr: EntityTrack, tid: str) -> dict:
        is_autonomous = (
            log.n_steps > 0 and
            (tr.n_palette_changes + tr.n_pattern_changes) / float(log.n_steps) > 0.5
        )
        # Initial-frame location: pixel-center + bbox + size.  When the
        # ARC adapter supplies a cell_system, the caller can translate
        # pixel-center to cell-coordinate downstream.
        bb = tr.initial_bbox
        center_px = ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)
        # Did this track ever appear as the trigger of a change event?
        triggered_targets: List[dict] = []
        for tc in log.trigger_candidates:
            ab = tuple(tc["agent_bbox"])
            if _bbox_distance(ab, tr.initial_bbox) <= 6.0:
                triggered_targets.append({
                    "step":             tc["step"],
                    "affected_track":   tc["affected"],
                    "change_kind":      tc["change_kind"],
                    "functional_role":  tc["functional_role"],
                })
        # Rotation/reflection direction if this track's cells got
        # rearranged in place.  Only meaningful when the track is
        # in permanently_changed_tracks AND there's a clean dihedral
        # mapping between its initial and final bbox-region.
        rotation_observed = None
        if (tid in log.permanently_changed_tracks
                and len(log.frame_history) >= 2):
            rotation_observed = _detect_rotation_direction(
                log.frame_history[0], log.frame_history[-1],
                tr.initial_bbox,
            )
        h, w = (log.frame_history[0].shape
                if log.frame_history else (64, 64))
        spatial_zone = _spatial_zone(tr.initial_bbox, frame_h=h, frame_w=w)
        return {
            "moves_with_actions":         tr.moves_with_actions,
            "appearance_changes":         tr.appearance_changed,
            "is_autonomous_animation":    is_autonomous,
            "stayed_static":              tr.stayed_static,
            "n_palette_changes":          tr.n_palette_changes,
            "n_pattern_changes":          tr.n_pattern_changes,
            "total_displacement_px":      tr.total_displacement,
            "n_frames_seen":              tr.n_frames_seen,
            "disappeared_at_step":        tr.disappeared_at_step,
            "initial_bbox":               list(tr.initial_bbox),
            "initial_size_px":            tr.initial_size,
            "initial_center_px":          list(center_px),
            "initial_palettes":           list(tr.initial_palettes),
            "all_palettes_seen":          sorted({p for pals in tr.palette_history for p in pals}),
            "is_agent":                   (tid == log.agent_track_id),
            "triggered_changes":          triggered_targets,
            # Trigger hypothesis: the entity correlated with another
            # entity's change SOME times, but below the threshold for
            # confirmation.  Surfaced so the operator (or future
            # exploration) can decide whether to test it deliberately.
            "trigger_hypothesis":         track_trigger_hypothesis.get(tid),
            # Spatial zone for cross-level transfer: if a future level
            # has a similar-signature entity in the same zone, it's
            # likely the same functional role.
            "spatial_zone":               spatial_zone,
            # Rotation/reflection direction iff a dihedral transform
            # maps initial bbox -> final bbox exactly.  None if the
            # change isn't a clean rotation/reflection.
            "rotation_observed":          rotation_observed,
            "is_permanent":               (tid in log.permanently_changed_tracks),
            # Palettes that flashed in this entity's bbox temporarily
            # (appeared at some intermediate step, not in initial or
            # final state).  This is feedback / signal flash -- it
            # does NOT mean the entity itself changed state, but it
            # often co-occurs with other entities to indicate RELATED
            # entities being highlighted to the player.
            "transient_flash_palettes":   getattr(
                log, "entities_with_transient_flash", {}
            ).get(tid, []),
        }

    # Apply role corrections based on the behavior of each matched track.
    # Helper: register a candidate from inside this function without a
    # hard import cycle.  ``add_role_candidate`` is defined in
    # level_memory; we look it up lazily.
    def _cand(ent, role, tier, evidence):
        try:
            from .level_memory import add_role_candidate
            add_role_candidate(ent, role, tier, evidence)
        except Exception:
            pass

    # Pull the catalog's intrinsic-class indices.  These are the
    # substrate's source of truth for "does this role intrinsically
    # move / change appearance".  The heuristic role-overrides below
    # consult them so that a catalog-named role is never downgraded
    # to a generic one whose catalog semantics it already entails.
    # When the catalog isn't available (test harnesses) the sets stay
    # empty and the heuristics fall back to the legacy behaviour.
    #
    # ``agent_controlled`` and ``follower`` are kept separate because
    # they have OPPOSITE relationships to the top-mover semantic:
    # an agent_controlled role IS the user-controlled mover (so the
    # top-mover track confirms the specialisation), while a follower
    # role moves PASSIVELY under another entity's motion (so the
    # top-mover track being labelled "follower" is a VLM error to
    # override).
    roles_agent_controlled: set = set()   # motion_class == agent_controlled
    roles_follower:         set = set()   # motion_class == follower
    roles_static_by_catalog: set = set()  # motion_class == static
    roles_mutable_appearance: set = set() # appearance_class == mutable
    try:
        from .catalog_loader import load_catalog
        _cat = load_catalog()
        for _e in _cat.by_kind.get("entity_role", []):
            mc = getattr(_e, "motion_class", None)
            ac = getattr(_e, "appearance_class", None)
            if mc == "agent_controlled":
                roles_agent_controlled.add(_e.primitive_id)
            if mc == "follower":
                roles_follower.add(_e.primitive_id)
            if mc == "static":
                roles_static_by_catalog.add(_e.primitive_id)
            if ac == "mutable":
                roles_mutable_appearance.add(_e.primitive_id)
    except Exception:
        pass

    for i, ent in enumerate(out["entities"]):
        corrections: List[str] = []
        # Capture the VLM's initial role as the weakest candidate so
        # the resolver has something to fall back on when no stronger
        # matcher fires.
        vlm_role = ent.get("role", "")
        if vlm_role and vlm_role not in REGION_ROLES and vlm_role != "unknown":
            _cand(ent, vlm_role, "vlm", "VLM initial proposal")
        tid = ent.get("_matched_track")
        if tid is None or ent.get("_overlap", 0.0) < 0.2:
            ent["_corrections"] = corrections
            ent["properties"]   = {}
            continue
        tr = log.tracks[tid]
        ent["properties"] = _properties_for_track(tr, tid)
        original_role = ent.get("role", "")
        is_autonomous = tid in getattr(log, "autonomous_tracks", set())
        if (tr.moves_with_actions or tid == log.agent_track_id) \
                and not is_autonomous:
            # The tracker observed motion on this track.  Decide
            # whether to confirm the VLM's catalog claim or override
            # to the generic 'agent' role.  Two structural questions:
            #
            #   1. Is this track the TOP MOVER (``log.agent_track_id``)?
            #      The top mover is the substrate's best guess at
            #      which entity the user controls.
            #   2. What does the catalog say about ``original_role``'s
            #      motion_class?  ``agent_controlled`` roles ARE the
            #      user-controlled mover (e.g. movable_pin).
            #      ``follower`` roles move passively under another
            #      entity's motion -- their motion is real but they
            #      aren't the agent.  ``static`` roles aren't
            #      expected to move at all; observed motion either
            #      contradicts the catalog or is matcher jitter
            #      (which we detect by checking whether the entity's
            #      pixel cells persistently shifted between the first
            #      and last frames).
            cells_first = (tr.cell_set_history[0]
                           if tr.cell_set_history else frozenset())
            cells_last  = (tr.cell_set_history[-1]
                           if tr.cell_set_history else frozenset())
            persistent_shift = bool(cells_first) and (cells_first != cells_last)
            is_top_mover = (tid == log.agent_track_id)

            if original_role in roles_agent_controlled:
                # VLM proposed an agent specialisation and the
                # tracker observed motion -- catalog claim consistent
                # with observation.  Keep the specialisation; it's
                # strictly more informative than the generic 'agent'.
                _cand(ent, original_role, "causal",
                      f"motion {tr.total_displacement:.0f}px; "
                      f"confirms agent specialisation '{original_role}'")
                corrections.append(
                    f"confirmed '{original_role}' (agent specialisation) "
                    f"via motion {tr.total_displacement:.0f}px"
                )
            elif (not is_top_mover) and original_role in roles_follower:
                # Non-top-mover + VLM proposed a passive follower.
                # Consistent: this entity moves under another
                # entity's motion.  Keep the catalog role.
                _cand(ent, original_role, "causal",
                      f"motion {tr.total_displacement:.0f}px (non-top); "
                      f"consistent with follower semantics")
                corrections.append(
                    f"confirmed '{original_role}' (follower) via "
                    f"{tr.total_displacement:.0f}px motion as non-top mover"
                )
            elif (original_role in roles_static_by_catalog
                  and not persistent_shift):
                # Catalog says this role is static and the entity's
                # pixels returned to their starting positions, so the
                # tracker's recorded displacement is matcher jitter,
                # not real motion.
                corrections.append(
                    f"kept '{original_role}' despite "
                    f"{tr.total_displacement:.0f}px tracker motion "
                    f"(catalog: motion_class=static; cell-set returned "
                    f"to initial -- motion was matcher jitter)"
                )
            else:
                # Either: top mover with a non-agent_controlled VLM
                # role (catalog claim contradicted -- VLM was wrong);
                # OR: non-top-mover with a non-follower VLM role
                # (the entity moves but the VLM's role doesn't
                # explain it).  Either way, override to the generic
                # ``agent`` and note the correction.
                _cand(ent, "agent", "causal",
                      f"motion {tr.total_displacement:.0f}px observed")
                ent["role"] = "agent"
                if original_role == "agent":
                    corrections.append(
                        f"confirmed agent role (motion {tr.total_displacement:.0f}px)"
                    )
                elif original_role == "agent_avatar":
                    corrections.append(
                        f"normalised role 'agent_avatar' -> 'agent' "
                        f"(motion {tr.total_displacement:.0f}px confirmed)"
                    )
                else:
                    corrections.append(
                        f"corrected role {original_role!r} -> 'agent' "
                        f"(observed motion {tr.total_displacement:.0f}px over "
                        f"{tr.n_frames_seen} frames -- the user controls "
                        f"this entity's movement; NOT a working_glyph)"
                    )
        elif tid in track_functional_role:
            new_role = track_functional_role[tid]
            _cand(ent, new_role, "causal",
                  "agent visited; another entity changed")
            if original_role != new_role:
                ent["role"] = new_role
                corrections.append(
                    f"corrected role {original_role!r} -> {new_role!r} "
                    f"(visiting this entity caused another entity to change)"
                )
            else:
                corrections.append(
                    f"confirmed {new_role!r} via observed trigger function"
                )
        elif any(
            (s.get("bitmap_id") and any(
                tcid.startswith(s["bitmap_id"][:18])
                for tcid in track_functional_role
            ))
            for s in (ent.get("_sub_bitmaps") or [])
        ):
            # The entity's primary matched track isn't itself in
            # track_functional_role, but one of its sub-bitmaps IS --
            # propagate the role to the parent entity.
            new_role = None
            for s in ent.get("_sub_bitmaps") or []:
                bm = s.get("bitmap_id")
                if not bm:
                    continue
                for tcid, r in track_functional_role.items():
                    if tcid.startswith(bm[:18]):
                        new_role = r
                        break
                if new_role:
                    break
            if new_role:
                old_role = ent.get("role")
                _cand(ent, new_role, "causal",
                      "sub-bitmap was a trigger; agent visit caused change")
                ent["role"] = new_role
                corrections.append(
                    f"corrected role {old_role!r} -> {new_role!r} "
                    f"(a sub-bitmap of this entity was the trigger; "
                    f"visiting it caused another entity to change)"
                )
        elif tr.appearance_changed and not tr.moves_with_actions:
            # Promote to a role OVERRIDE only when:
            #   (a) the change is PERMANENT (not transient HUD ticks)
            #   (b) the track isn't autonomous (HUD timers like the
            #       budget bar change every step but aren't game
            #       entities to be manipulated)
            #   (c) the entity isn't itself moving (the agent)
            # All three filter out the budget_meter and the agent.
            is_permanent = tid in log.permanently_changed_tracks
            is_autonomous = tid in getattr(log, "autonomous_tracks", set())
            # A working_glyph downgrade is REDUNDANT when the VLM
            # already proposed any role the catalog declares as
            # ``appearance_class=mutable`` -- the proposed role
            # already explains "this entity's pixels change during
            # play".  Reading the set from the catalog rather than
            # carrying a hardcoded list keeps new roles automatically
            # exempt as soon as they're added to the catalog.
            if is_permanent and not is_autonomous \
                    and original_role not in roles_mutable_appearance:
                _cand(ent, "working_glyph", "causal",
                      "observed permanent appearance change during play")
                ent["role"] = "working_glyph"
                corrections.append(
                    f"corrected role {original_role!r} -> 'working_glyph' "
                    f"(observed PERMANENT appearance change during play; "
                    f"matches catalog 'entity manipulated by triggers')"
                )
            elif (not is_autonomous
                  and original_role not in roles_mutable_appearance):
                ent["role_hint"] = "working_glyph"
                corrections.append(
                    f"appearance changes during play (palette/pattern) "
                    f"-> likely working_glyph; original role {original_role!r}"
                )
        elif tr.stayed_static and tr.disappeared_at_step is None:
            if original_role in ("agent", "agent_avatar"):
                ent["role"] = "static_observed_not_agent"
                corrections.append(
                    "never moved during play -> NOT the agent (corrected)"
                )
        # Visit-without-effect demotion: if perception's initial Layer-B
        # guess was a TRIGGER role (shape_changer / color_changer / etc.)
        # but the agent stepped on it and nothing else changed, the
        # functional claim is contradicted by observation -- demote.
        TRIGGER_ROLES = {"shape_changer", "color_changer",
                         "target_slot", "movable_pin", "anchor_endpoint"}
        if tid in visited_without_effect and ent.get("role") in TRIGGER_ROLES:
            old = ent["role"]
            ent["role"] = "static_observed_no_effect"
            corrections.append(
                f"perception's initial '{old}' guess CONTRADICTED by "
                f"interaction: agent stepped on this entity, no other "
                f"entity changed -> demoted (no trigger function observed)"
            )
        if tid in visited_with_effect and ent.get("role") in TRIGGER_ROLES:
            corrections.append(
                f"perception's '{ent['role']}' guess CONFIRMED by "
                f"interaction: agent stepped on this entity, other entity "
                f"changed in correlation"
            )
        ent["_corrections"] = corrections

    # Surface EVERY track perception missed.  Each unmatched track is an
    # entity Layer A extracted that perception's VLM step dropped or
    # never assigned a role to.  We classify it from observed behavior:
    #   - moves with actions    -> agent_avatar (high confidence)
    #   - changes appearance    -> mutable / working candidate
    #   - never moved/changed   -> "missed_by_perception" (role unknown,
    #                              but worth surfacing for the operator)
    # This is exactly the data-driven loop: interaction reveals what
    # the one-shot perception step failed to recognise.
    def _bbox_contains(outer, inner) -> bool:
        return (outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3])

    def _bboxes_touch(a, b) -> bool:
        """True iff two bboxes overlap, touch, or one contains the other.

        Row/col gap <= 1 captures edge-touching adjacency and overlap.
        Containment (one bbox entirely inside another) also returns
        True so multi-palette layered sprites (e.g. an inner glyph
        rendered inside an outer body) are merged.  Without this the
        synchronous-motion rule misses pal-inner+pal-outer pairs that
        share an exact-containment relationship rather than touching.
        """
        row_gap = max(a[0], b[0]) - min(a[2], b[2])
        col_gap = max(a[1], b[1]) - min(a[3], b[3])
        if row_gap <= 1 and col_gap <= 1:
            return True
        if _bbox_contains(a, b) or _bbox_contains(b, a):
            return True
        return False

    # Group SMALL unmatched tracks into compound entities when their
    # bboxes touch.  Multiple per-palette components that share an edge
    # (e.g. a palette-0 white core + a palette-1 light-grey accent)
    # are ONE logical sprite, not separate entities.
    SMALL = 10   # px: per-palette components of this size or less are
                 # candidate parts of a multi-palette compound sprite
    # Exclude prior-call synthetic compounds (their tid begins with
    # "compound_") so we don't re-merge them with their own members.
    # apply_to_parsed runs once per checkpoint; without this guard the
    # compound's size doubles each call.
    unmatched_tids = [
        tid for tid, tr in log.tracks.items()
        if tid not in used_track
        and tr.initial_size >= 2
        and not tid.startswith("compound_")
    ]
    # Build union-find over small unmatched tracks.
    parent = {tid: tid for tid in unmatched_tids}
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def _union(a, b):
        parent[_find(a)] = _find(b)
    smalls = [tid for tid in unmatched_tids if log.tracks[tid].initial_size <= SMALL]
    for i in range(len(smalls)):
        for j in range(i + 1, len(smalls)):
            ta = log.tracks[smalls[i]]
            tb = log.tracks[smalls[j]]
            if _bboxes_touch(ta.initial_bbox, tb.initial_bbox):
                _union(smalls[i], smalls[j])
    # Additionally: tracks that MOVED SYNCHRONOUSLY are parts of the
    # same entity, regardless of individual size.  Two parts of a
    # multi-palette sprite (e.g. the agent's inner glyph + outer body)
    # are not both "small", but they move together every step.  Detect
    # this by comparing bbox shift sequences.
    movers = [tid for tid in unmatched_tids
              if log.tracks[tid].total_displacement > 0]

    def _shift_sequence(tr):
        out = []
        for k in range(1, len(tr.bbox_history)):
            prev = tr.bbox_history[k - 1]
            curr = tr.bbox_history[k]
            out.append((curr[0] - prev[0], curr[1] - prev[1]))
        return tuple(out)

    for i in range(len(movers)):
        for j in range(i + 1, len(movers)):
            ta = log.tracks[movers[i]]
            tb = log.tracks[movers[j]]
            if not _bboxes_touch(ta.initial_bbox, tb.initial_bbox):
                continue
            if _shift_sequence(ta) == _shift_sequence(tb) \
                    and len(ta.bbox_history) == len(tb.bbox_history):
                _union(movers[i], movers[j])

    # Same-bitmap chain merge: multiple tracks that share the SAME
    # canonical bitmap_id and are spatially connected (touching,
    # overlapping, or contained) are pieces of ONE multi-segment
    # entity.  Canonical example: a piercer's tail leaves repeated
    # segments behind the head; each segment has the same visual
    # signature but its OWN motion history (it was added at a
    # different step).  The synchronous-motion rule above requires
    # identical step-by-step shifts, which doesn't fit chains where
    # segments appear at different times.  Same-bitmap + spatial
    # contiguity captures it instead.
    bm_groups: Dict[str, List[str]] = {}
    for tid in unmatched_tids:
        bm = log.tracks[tid].initial_bitmap_id
        if bm:
            bm_groups.setdefault(bm, []).append(tid)
    for bm, tids in bm_groups.items():
        if len(tids) < 2:
            continue
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                ta = log.tracks[tids[i]]
                tb = log.tracks[tids[j]]
                if _bboxes_touch(ta.initial_bbox, tb.initial_bbox):
                    _union(tids[i], tids[j])
    # Group representative -> list of member tids.
    groups: Dict[str, List[str]] = {}
    for tid in unmatched_tids:
        rep = _find(tid)
        groups.setdefault(rep, []).append(tid)

    for rep_tid, member_tids in groups.items():
        if len(member_tids) == 1:
            tid = member_tids[0]
            tr  = log.tracks[tid]
        else:
            # Compound entity: union bbox + sum of sizes + union palettes.
            members = [log.tracks[t] for t in member_tids]
            r0 = min(m.initial_bbox[0] for m in members)
            c0 = min(m.initial_bbox[1] for m in members)
            r1 = max(m.initial_bbox[2] for m in members)
            c1 = max(m.initial_bbox[3] for m in members)
            union_pals = sorted({p for m in members for p in m.initial_palettes})
            total_size = sum(m.initial_size for m in members)
            # Build a synthetic EntityTrack representing the compound
            # without mutating the underlying member tracks.
            from copy import copy
            compound_tr = copy(members[0])
            compound_tr.initial_bbox     = (r0, c0, r1, c1)
            compound_tr.initial_size     = total_size
            compound_tr.initial_palettes = tuple(union_pals)
            for m in members[1:]:
                compound_tr.total_displacement = max(
                    compound_tr.total_displacement, m.total_displacement)
                compound_tr.n_palette_changes = max(
                    compound_tr.n_palette_changes, m.n_palette_changes)
                compound_tr.n_pattern_changes = max(
                    compound_tr.n_pattern_changes, m.n_pattern_changes)
            tid = f"compound_{rep_tid[:18]}"
            tr = compound_tr
            log.tracks[tid] = compound_tr   # so _properties_for_track works
        # If this track's bbox is CONTAINED in a NON-REGION perception
        # entity, merge the observed behaviour into THAT entity rather
        # than emitting a duplicate.  The user's invariant: an entity
        # that recolors or rotates stays the SAME entity with different
        # properties -- and equally, a sub-bitmap inside a discrete
        # perception entity's bbox is part of that same entity.
        # Region-role entities (wall, play_area, etc.) have palette-
        # extent bboxes that cover most of the frame; merging into them
        # would absorb every discrete sprite and hide them from the
        # markup.  So skip region parents -- discrete sprites inside
        # the wall's bbox stay as their own entities.
        functional_pre = track_functional_role.get(tid)
        if tid == log.agent_track_id:
            behaviour = "agent_avatar (moves with actions)"
        elif functional_pre:
            behaviour = (f"{functional_pre} (visiting it caused another "
                         "entity to change)")
        elif tr.appearance_changed:
            behaviour = (
                f"appearance changed during play "
                f"({tr.n_palette_changes} palette + "
                f"{tr.n_pattern_changes} pattern changes)"
            )
        else:
            behaviour = (f"static across {tr.n_frames_seen} frames")
        merged = False
        # Structured fingerprint suite for this sub-bitmap (used by the
        # KB role match logic; the human-readable correction string is
        # kept for backwards compat).
        sub_record = {
            "bitmap_id": tr.initial_bitmap_id,
            "shape_id":  tr.shape_id_history[0]  if tr.shape_id_history  else None,
            "topo_id":   tr.topo_id_history[0]   if tr.topo_id_history   else None,
            "scaled_id": tr.scaled_id_history[0] if tr.scaled_id_history else None,
            "bbox":      list(tr.initial_bbox),
            "size_px":   tr.initial_size,
            "palettes":  list(tr.initial_palettes),
            "behaviour": behaviour,
        }
        # A track that BEHAVES distinctly during play deserves its own
        # top-level entity even when its bbox happens to fall inside
        # some larger parsed entity (a common Layer A over-merge):
        #   * The track is the picked agent OR moves with actions
        #     (the agent itself, or a co-moving entity like a
        #     follower / linked_midpoint).
        #   * The track was identified as a functional trigger.
        #   * The track's appearance permanently changed (a working
        #     glyph or similar mutable element).
        # In any of those cases the parent entity's role doesn't
        # predict this track's behavior, so absorbing it as a
        # sub-bitmap would hide a real distinct entity from the
        # downstream pipeline.  Skip tiny (<= 1px) tracks: those are
        # usually a single-pixel detail of a larger sprite, not a
        # standalone entity.
        has_distinct_behaviour = (
            tr.initial_size > 1
            and tid not in getattr(log, "autonomous_tracks", set())
            and (
                tid == log.agent_track_id
                or tr.moves_with_actions
                or tid in track_functional_role
                or (tr.appearance_changed
                    and tid in log.permanently_changed_tracks)
            )
        )
        if not has_distinct_behaviour:
            for ent in out["entities"]:
                parent_role = str(ent.get("role", ""))
                if parent_role in REGION_ROLES:
                    continue   # don't absorb sub-sprites into palette regions
                if _bbox_contains(ent.get("bbox_pixels") or [0,0,0,0],
                                  tr.initial_bbox):
                    ent.setdefault("_corrections", []).append(
                        f"sub-bitmap observation (bm={tid[:18]}, "
                        f"bbox={list(tr.initial_bbox)}, "
                        f"size={tr.initial_size}px, "
                        f"pal={list(tr.initial_palettes)}): {behaviour}"
                    )
                    ent.setdefault("_sub_bitmaps", []).append(sub_record)
                    merged = True
                    break
        if merged:
            continue
        # Functional role discovered through trigger observation takes
        # priority over the generic static/dynamic classification.
        functional = track_functional_role.get(tid)
        if tid == log.agent_track_id:
            role = "agent"
            note = (f"discovered through motion observation "
                    f"(displaced {tr.total_displacement:.0f}px over "
                    f"{tr.n_frames_seen} frames); perception missed this entity")
            correction = ("added by interaction: this bitmap moves with "
                          "actions -> it IS the agent (user-controlled)")
        elif functional:
            role = functional
            note = (f"discovered through trigger interaction: agent visited "
                    f"this entity AT THE MOMENT another entity changed.  "
                    f"Functional role inferred from change kind.")
            correction = (f"added by interaction: visiting this entity "
                          f"caused another entity to change -> {functional}")
        elif tr.appearance_changed:
            # Distinguish autonomous "HUD strip" entities from
            # working_glyph candidates.  An autonomous edge-strip --
            # animates every step regardless of input, spans most of
            # one frame dimension, and sits within a few px of an
            # edge -- is structurally a budget_meter / life_indicator
            # (catalog: motion=static, appearance=mutable).  This is
            # substrate-general: any future game with a HUD strip
            # will be caught.
            is_autonomous = tid in getattr(log, "autonomous_tracks", set())
            bb = tr.initial_bbox
            frame_h = initial_frame.shape[0] if initial_frame is not None else 64
            frame_w = initial_frame.shape[1] if initial_frame is not None else 64
            bb_h = bb[2] - bb[0] + 1
            bb_w = bb[3] - bb[1] + 1
            spans_w = bb_w / frame_w > 0.5
            spans_h = bb_h / frame_h > 0.5
            EDGE_PX = 4
            at_top    = bb[0] <= EDGE_PX
            at_bottom = bb[2] >= frame_h - 1 - EDGE_PX
            at_left   = bb[1] <= EDGE_PX
            at_right  = bb[3] >= frame_w - 1 - EDGE_PX
            at_edge   = at_top or at_bottom or at_left or at_right
            if is_autonomous and at_edge and (spans_w or spans_h):
                role = "budget_meter"
                note = (
                    f"autonomous animation ({tr.n_palette_changes} palette "
                    f"+ {tr.n_pattern_changes} pattern changes over "
                    f"{log.n_steps} steps) on an edge-spanning strip "
                    f"(bbox {bb}); structurally a HUD/budget strip."
                )
                correction = (
                    "added by interaction: autonomous-animated edge-strip "
                    "-> budget_meter (HUD indicator, not a game entity)"
                )
            else:
                role = "unknown"
                note = (
                    f"discovered through interaction: appearance changed "
                    f"({tr.n_palette_changes} palette + "
                    f"{tr.n_pattern_changes} pattern changes); likely a "
                    "mutable / working entity"
                )
                correction = (
                    "added by interaction: this bitmap was operated "
                    "upon during play -> likely working_glyph or "
                    "mutable_block; perception missed it"
                )
        else:
            role = "unknown"
            note = (f"Layer A extracted this candidate (bm={tid[:18]}, "
                    f"size={tr.initial_size}px, pal={list(tr.initial_palettes)}) "
                    "but perception's VLM did not assign it a role.  "
                    "Observed behavior: static across "
                    f"{tr.n_frames_seen} frames.")
            correction = ("added by interaction: Layer A saw this bitmap, "
                          "perception's VLM missed it.  No role yet — "
                          "operator should annotate or run further "
                          "perception pass.")
        # Dedupe: if a previously-emitted entity has substantially
        # the SAME bbox as this discovered track, treat it as the
        # same entity rather than creating a duplicate.  When the
        # new role is more specific than the existing one's (e.g.
        # 'budget_meter' over 'wall' for the autonomous strip), we
        # override; otherwise we just attach as a sub-bitmap.
        dup_target_idx: Optional[int] = None
        for ei, ent in enumerate(out["entities"]):
            ebb = ent.get("bbox_pixels") or [0, 0, 0, 0]
            if (tuple(ebb) == tuple(tr.initial_bbox)
                    or (_bbox_contains(ebb, tr.initial_bbox)
                        and _bbox_contains(tr.initial_bbox, ebb))):
                dup_target_idx = ei
                break
        if dup_target_idx is not None:
            existing = out["entities"][dup_target_idx]
            # Override role if the new role is observationally
            # justified (budget_meter / functional / agent) AND the
            # existing role doesn't predict the observed behaviour.
            existing_role = str(existing.get("role", ""))
            override_roles = {"budget_meter", "agent"}
            if (role in override_roles
                    and existing_role not in override_roles):
                existing["role"] = role
                existing.setdefault("_corrections", []).append(
                    f"role override from '{existing_role}' to '{role}': "
                    f"{correction}"
                )
            else:
                existing.setdefault("_corrections", []).append(correction)
            existing.setdefault("_sub_bitmaps", []).append(sub_record)
            if existing.get("_matched_track") is None:
                existing["_matched_track"] = tid
                existing["_overlap"] = 1.0
                existing["properties"] = _properties_for_track(tr, tid)
            continue
        out["entities"].append({
            "id":            f"discovered_{tid[:24]}",
            "bbox_pixels":   list(tr.initial_bbox),
            "palettes":      list(tr.initial_palettes),
            "role":          role,
            "candidate_ids": (),
            "related_to":    None,
            "member_of_group": None,
            "notes":         note,
            "_matched_track": tid,
            "_overlap":      1.0,
            "_corrections":  [correction],
            "_sub_bitmaps":  [sub_record],
            "properties":    _properties_for_track(tr, tid),
        })

    # Structural promotion of unique-palette sub-bitmaps.
    #
    # When the VLM lumped many distinct things into a single
    # ``unknown`` parent (typical Layer-A over-merge in dense-tile
    # games), action observation alone surfaces only the things
    # that BEHAVED during play (motion, triggers, mutation).  A
    # static landmark inside the same over-merged region -- e.g.
    # the goal cell in a maze -- stays buried as a sub-bitmap
    # because it has no behavioral evidence.
    #
    # Structural cue: a sub-bitmap whose palette is UNIQUE among
    # the parent's sub-bitmaps is, by construction, not part of
    # any repeated texture pattern that defines the parent region.
    # It's a distinct entity sharing space with the parent's bbox.
    # Promote it.
    #
    # Gated by parent_role == 'unknown' so we don't tear apart real
    # multi-palette sprites whose distinct sub-palettes are real
    # parts of a single coherent entity (the VLM gave those a
    # specific role, so we trust the VLM's claim).
    promoted_records: List[dict] = []
    for ent in list(out["entities"]):
        parent_role = str(ent.get("role", ""))
        if parent_role != "unknown":
            continue
        subs = ent.get("_sub_bitmaps") or []
        if len(subs) < 2:
            continue
        pal_to_idxs: Dict[tuple, List[int]] = {}
        for i, s in enumerate(subs):
            pals = tuple(sorted(s.get("palettes") or []))
            if not pals:
                continue
            pal_to_idxs.setdefault(pals, []).append(i)
        promote_idxs: set = set()
        for pals, idxs in pal_to_idxs.items():
            if len(idxs) != 1:
                continue
            s = subs[idxs[0]]
            if int(s.get("size_px") or 0) <= 1:
                continue   # singletons aren't standalone entities
            promote_idxs.add(idxs[0])
        if not promote_idxs:
            continue
        # Emit a top-level entity for each unique-palette sub.
        for i in sorted(promote_idxs):
            s = subs[i]
            promoted_records.append({
                "id":            f"structural_{(s.get('bitmap_id') or '')[:24]}",
                "bbox_pixels":   list(s.get("bbox") or [0, 0, 0, 0]),
                "palettes":      list(s.get("palettes") or []),
                "role":          "unknown",
                "candidate_ids": (),
                "related_to":    None,
                "member_of_group": None,
                "notes": (
                    f"structural promotion: palette "
                    f"{list(s.get('palettes') or [])} is unique within "
                    f"parent '{parent_role}' sub-bitmaps -- this is a "
                    f"distinct entity sharing the parent's bbox, not "
                    f"part of the parent"
                ),
                "_matched_track": None,
                "_overlap":      1.0,
                "_corrections":  [
                    f"structural promotion: unique palette "
                    f"{list(s.get('palettes') or [])} (size "
                    f"{s.get('size_px')}px) inside an 'unknown' parent"
                ],
                "_sub_bitmaps":  [s],
                "properties":    {},
            })
        # Remove promoted subs from parent.
        ent["_sub_bitmaps"] = [s for i, s in enumerate(subs)
                               if i not in promote_idxs]
    out["entities"].extend(promoted_records)

    # Repeated-instance promotion.  Any non-region entity whose
    # bbox contains N >= 3 disconnected sub-CCs of similar size
    # and palette is a "row of items" the seed VLM lumped into one
    # compound -- promote each instance as its own top-level entity
    # so downstream reasoning / probing can target them individually.
    # Substrate-general: catches bp35's 7 collectibles, sk48's 3
    # movable_blocks if Layer A had merged them, etc.
    instance_records: List[dict] = []
    try:
        from .parallel_groups import detect_repeated_instances
        bg_pal = out.get("background_palettes") or []
        for ent in list(out["entities"]):
            role = str(ent.get("role") or "")
            if role in REGION_ROLES:
                continue
            if role in ("agent", "agent_avatar", "movable_pin",
                        "piercer_head", "noise", "budget_meter",
                        "parallel_cell_inlier", "parallel_cell_outlier"):
                continue
            bbox = ent.get("bbox_pixels")
            if not bbox:
                continue
            bw = bbox[3] - bbox[1] + 1
            bh = bbox[2] - bbox[0] + 1
            if bw * bh <= 64:
                # Tiny entity -- not a row-of-items compound.
                continue
            instances = detect_repeated_instances(
                initial_frame, tuple(bbox), bg_pal,
            )
            if len(instances) < 3:
                continue
            ent_id = ent.get("id") or "unknown"
            ent.setdefault("_corrections", []).append(
                f"repeated-instance promotion: detected {len(instances)} "
                f"similar sub-components inside this entity's bbox; each "
                f"emitted as its own top-level entity for individual probing"
            )
            for j, inst in enumerate(instances):
                bbox_i = list(inst["bbox"])
                instance_records.append({
                    "id":              f"instance_{ent_id}_{j}",
                    "bbox_pixels":     bbox_i,
                    "palettes":        list(inst["palettes"]),
                    "role":            "unknown",
                    "candidate_ids":   (),
                    "related_to":      None,
                    "member_of_group": str(ent_id),
                    "notes": (
                        f"repeated-instance #{j} of {len(instances)} found "
                        f"inside parent entity '{ent_id}' (role={role!r}); "
                        f"emitted as own entity by substrate's repeated-"
                        f"instance detector."
                    ),
                    "_matched_track":  None,
                    "_overlap":        1.0,
                    "_corrections": [
                        f"emitted as repeated-instance #{j} of "
                        f"{len(instances)} inside '{ent_id}'"
                    ],
                    "_sub_bitmaps":    [],
                    "properties":      {},
                })
    except Exception:
        pass
    out["entities"].extend(instance_records)

    # Post-VLM compound merge: collapse entities the VLM enumerated
    # separately but that are structurally ONE entity.  Two rules:
    #
    # 1. **Containment + synchronous motion + same role**.  A
    #    multi-palette layered sprite (e.g. piercer outer shell
    #    containing piercer inner glyph) shows up as two VLM
    #    candidates -- the outer body and the inner glyph -- both
    #    tagged with the same role via motion override.  They share
    #    a containment relationship and identical step-by-step
    #    motion.
    #
    # 2. **Same primary bitmap_id + spatial connectivity**.  A
    #    multi-segment chain like a piercer's tail leaves repeated
    #    visual segments at different positions; Layer A extracts
    #    them as separate components (since the segments are
    #    pixel-disconnected) but they share the same canonical
    #    bitmap_id and lie on the same axis.
    out_entities = out["entities"]

    def _bbox_overlap_or_contain(a, b) -> bool:
        if _bbox_contains(a, b) or _bbox_contains(b, a):
            return True
        rr0 = max(a[0], b[0]); cc0 = max(a[1], b[1])
        rr1 = min(a[2], b[2]); cc1 = min(a[3], b[3])
        return rr1 >= rr0 and cc1 >= cc0

    def _shift_seq_from_track(tid_):
        tr_ = log.tracks.get(tid_)
        if tr_ is None:
            return None
        out_seq = []
        for k in range(1, len(tr_.bbox_history)):
            prev = tr_.bbox_history[k - 1]
            curr = tr_.bbox_history[k]
            out_seq.append((curr[0] - prev[0], curr[1] - prev[1]))
        return tuple(out_seq)

    def _is_effectively_static(tid_, max_net_displacement: float = 2.0) -> bool:
        """True iff the track's NET displacement (initial vs final bbox)
        is small enough to be explained by occlusion artefacts.

        Static entities periodically appear to "shrink" when the agent
        overlaps them and "expand back" when the agent moves away.
        That shows up as non-zero ``total_displacement`` in the
        tracker but the NET (first frame -> last frame) position is
        unchanged.  Comparing first-vs-last bbox centres gives a
        reliable static check.
        """
        tr_ = log.tracks.get(tid_)
        if tr_ is None or len(tr_.bbox_history) < 2:
            return True
        b0 = tr_.bbox_history[0]
        bN = tr_.bbox_history[-1]
        cr0 = (b0[0] + b0[2]) / 2.0
        cc0 = (b0[1] + b0[3]) / 2.0
        crN = (bN[0] + bN[2]) / 2.0
        ccN = (bN[1] + bN[3]) / 2.0
        net = ((crN - cr0) ** 2 + (ccN - cc0) ** 2) ** 0.5
        return net <= max_net_displacement

    # Union-find over entity indices.
    n = len(out_entities)
    if n > 1:
        parent_e = list(range(n))
        def _find_e(x):
            while parent_e[x] != x:
                parent_e[x] = parent_e[parent_e[x]]
                x = parent_e[x]
            return x
        def _union_e(a, b):
            parent_e[_find_e(a)] = _find_e(b)

        # Pre-compute primary bitmap_ids and shift sequences.
        prim_bm: List[Optional[str]] = []
        shifts:  List[Optional[tuple]] = []
        for ent in out_entities:
            tid = ent.get("_matched_track")
            shifts.append(_shift_seq_from_track(tid) if tid else None)
            # Primary bitmap = first non-primary sub-bitmap with the
            # largest size, fall back to any sub-bitmap or the
            # matched track's bm.
            subs = ent.get("_sub_bitmaps") or []
            bm = None
            best_size = -1
            for s in subs:
                sz = int(s.get("size_px") or 0)
                if sz > best_size:
                    best_size = sz
                    bm = s.get("bitmap_id")
            prim_bm.append(bm)

        for i in range(n):
            for j in range(i + 1, n):
                a = out_entities[i]
                b = out_entities[j]
                ra = str(a.get("role") or "")
                rb = str(b.get("role") or "")
                ba = a.get("bbox_pixels") or [0]*4
                bb_ = b.get("bbox_pixels") or [0]*4
                if ra in REGION_ROLES or rb in REGION_ROLES:
                    continue
                # Rule 1: spatial connection + identical motion history.
                # Any group of tracks that share a motion history and
                # are spatially connected (touching, overlapping, or
                # contained) form ONE entity -- multi-colour pixels
                # that "stay as one" through play.  The motion history
                # can be:
                #   - matching non-trivial shift sequence (moved
                #     together every frame)
                #   - both empty (stayed put across all frames)
                # The spatial-connection requirement prevents merging
                # unrelated static entities just because both happened
                # to never move.  Roles need not match: a compound's
                # pieces often get heterogeneous role tags from the
                # VLM; the structural fact is they move (or stay) as
                # one unit.
                if (_bbox_overlap_or_contain(ba, bb_)
                        and shifts[i] is not None and shifts[j] is not None
                        and shifts[i] == shifts[j]):
                    _union_e(i, j)
                    continue
                # Rule 3: both effectively-static + spatially related +
                # role-compatible.  A multi-palette static structure
                # (e.g. a striped vertical bar acting as a guide rail)
                # shows up as several sub-bitmaps with different
                # bitmap_ids and small but non-zero
                # ``total_displacement_px`` because the agent
                # occasionally overlaps the bar and the tracker briefly
                # sees the bar "shrink" before it "grows back".  Net
                # displacement (first frame vs last frame) stays near
                # zero.  Rule 1 misses these because the per-step shift
                # sequences differ between the parts; Rule 2 misses
                # these because the parts have different primary
                # bitmap_ids.  This rule catches them by checking that
                # neither side has any real motion and the pair is
                # spatially related.
                tid_i = a.get("_matched_track")
                tid_j = b.get("_matched_track")
                if (tid_i and tid_j
                        and _is_effectively_static(tid_i)
                        and _is_effectively_static(tid_j)):
                    role_compat_r3 = (
                        ra == rb
                        or ra == "unknown"
                        or rb == "unknown"
                    )
                    # Restrict to actual overlap/containment: a
                    # static multi-palette structure has parts whose
                    # bboxes interleave or contain one another. Pure
                    # collinearity (same row or column, no overlap) is
                    # not enough -- two unrelated static items in the
                    # same row of the play area would otherwise merge.
                    if role_compat_r3 and _bbox_overlap_or_contain(ba, bb_):
                        _union_e(i, j)
                        continue
                # Rule 2: same primary bitmap_id + spatial relationship
                # + role compatibility.  Two tracks that share a
                # canonical visual pattern are the same kind of thing,
                # and they're ONE entity when their bboxes are
                # spatially related AND their roles are compatible.
                #
                # Compatibility:
                #   - same role, OR
                #   - both role=="unknown" (perception didn't classify
                #     either side), OR
                #   - one is "unknown" and the other is a
                #     non-reference role (the named one wins).
                # A pair like (agent, reference_arrangement) has
                # different KNOWN roles and is explicitly excluded.
                # References intentionally share visuals with the
                # active sprite they reference; that's their job, not
                # evidence of being the same entity.
                #
                # Skip entities the substrate explicitly emitted as
                # repeated-instance children (``member_of_group``
                # set).  Those were INTENTIONALLY split apart by the
                # detect_repeated_instances pass; re-merging them
                # here would undo that work.
                ent_i = out_entities[i]
                ent_j = out_entities[j]
                if (ent_i.get("member_of_group")
                        or ent_j.get("member_of_group")):
                    continue
                if not (prim_bm[i] and prim_bm[i] == prim_bm[j]):
                    continue
                role_compat = (
                    ra == rb
                    or ra == "unknown"
                    or rb == "unknown"
                )
                if not role_compat:
                    continue
                if _bbox_overlap_or_contain(ba, bb_):
                    _union_e(i, j)
                    continue
                rows_overlap = (max(ba[0], bb_[0]) <= min(ba[2], bb_[2]))
                cols_overlap = (max(ba[1], bb_[1]) <= min(ba[3], bb_[3]))
                if rows_overlap ^ cols_overlap:
                    _union_e(i, j)
                    continue

        # Build merged groups.
        merged_idx: Dict[int, List[int]] = {}
        for i in range(n):
            merged_idx.setdefault(_find_e(i), []).append(i)

        if any(len(v) > 1 for v in merged_idx.values()):
            new_entities = []
            for rep, members in merged_idx.items():
                if len(members) == 1:
                    new_entities.append(out_entities[members[0]])
                    continue
                ents = [out_entities[m] for m in members]
                r0 = min(e["bbox_pixels"][0] for e in ents)
                c0 = min(e["bbox_pixels"][1] for e in ents)
                r1 = max(e["bbox_pixels"][2] for e in ents)
                c1 = max(e["bbox_pixels"][3] for e in ents)
                pals = sorted({int(p) for e in ents
                               for p in (e.get("palettes") or [])})
                # Pick role: prefer the most-confident among members.
                # We don't have credence yet here, so prefer non-unknown
                # roles, ties broken by first occurrence.
                roles_order = [e.get("role") for e in ents]
                role = next((r for r in roles_order
                             if r and r != "unknown"), roles_order[0])
                # Aggregate sub-bitmaps + corrections.
                subs = []
                corrs = []
                for e in ents:
                    subs.extend(e.get("_sub_bitmaps") or [])
                    corrs.extend(e.get("_corrections") or [])
                corrs.append(
                    f"post-VLM compound merge: members={[e.get('id') for e in ents]} "
                    f"role={role!r}"
                )
                new_entities.append({
                    "id":            ents[0].get("id"),
                    "bbox_pixels":   [r0, c0, r1, c1],
                    "palettes":      pals,
                    "role":          role,
                    "candidate_ids": tuple(cid for e in ents
                                            for cid in (e.get("candidate_ids") or ())),
                    "related_to":    None,
                    "member_of_group": None,
                    "notes":         "compound merged from "
                                     + ", ".join(str(e.get("id")) for e in ents),
                    "_matched_track": ents[0].get("_matched_track"),
                    "_overlap":      max(e.get("_overlap", 0) for e in ents),
                    "_corrections":  corrs,
                    "_sub_bitmaps":  subs,
                    "properties":    ents[0].get("properties") or {},
                })
            out["entities"] = new_entities

    return out
