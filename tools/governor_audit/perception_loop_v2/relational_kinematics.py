"""Layer A — Relational Kinematics.

Per-turn temporal visual relations between tracked entities, computed
purely from EntityRecord.bbox_history.  Game-agnostic: knows nothing
about specific game mechanics; only about geometric primitives over
open-vocab entity names.

See docs/SPEC_visual_reasoning_substrate.md.

Outputs a list of RelationRecord per turn, attached to that turn's
DeltaRecord.  Downstream consumers: the enriched mechanic miner (B),
the recoverability fingerprint (C), the strategy prompt surface.

Tier-1 primitives:
  - displacement vector (centroid t-1 -> centroid t)
  - bbox overlap fraction
  - adjacency (gap within `cell_ticks`) and the axis/side of contact

Tier-2 seed relations (composed from primitives):
  - co_displacement(A, B, vec)      — both moved by ~the same vector
  - motion_blocked(A, dir, blocker) — expected motion didn't happen
                                       with a solid on the dir side
  - motion_arrested_at(A, B, dir)   — A's motion stopped on contact
                                       with B
  - penetration(A_tip, B)           — A's extent entered B's bbox
                                       region while B did not displace
  - support_relation(B, S, dir)     — solid S is immediately on B's
                                       far side along dir (geometric;
                                       no action required)

The relation set is OPEN — new kinds can be added without touching
B/C/D.  All thresholds are geometric (tick-space), no game vocabulary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Optional

# Module is sibling to world_knowledge under perception_loop_v2.
from world_knowledge import WorldKnowledge, EntityRecord  # noqa: E402


# ---------------------------------------------------------------------------
# Output record
# ---------------------------------------------------------------------------


@dataclass
class RelationRecord:
    """One observed visual relation between entities at a turn boundary.

    `kind` is open-vocab; the seed set is documented in this module's
    docstring.  Adding new kinds is a matter of writing another detector
    that appends to the output list — no schema migration required.
    """
    kind: str
    entities: list[str]                       # participating entity names
    turn: int                                  # the turn AFTER the transition
    direction: Optional[str] = None            # 'up' / 'down' / 'left' /
                                               #  'right' / None
    evidence: dict = field(default_factory=dict)
        # numeric measurement that fired the relation, so it's a fact
        # not a judgment.  E.g. for co_displacement:
        #   {"dvec_a": [dr_a, dc_a], "dvec_b": [dr_b, dc_b],
        #    "match_tolerance": int}

    def as_dict(self) -> dict:
        return asdict(self)

    def short(self) -> str:
        """Compact one-liner for surfacing in prompts / logs."""
        parts = [self.kind]
        if self.entities:
            parts.append("(" + ",".join(self.entities) + ")")
        if self.direction:
            parts.append(f"dir={self.direction}")
        if self.evidence:
            kv = ",".join(f"{k}={v}" for k, v in self.evidence.items())
            parts.append(f"[{kv}]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Tier-1 primitives over bbox histories
# ---------------------------------------------------------------------------


def _bbox_at_turn(rec: EntityRecord, turn: int) -> Optional[list[int]]:
    """Look up the bbox an EntityRecord had at a specific turn.  Returns
    the last bbox at-or-before `turn`, or None if no such entry."""
    last = None
    for (t, bb) in (rec.bbox_history or []):
        if t <= turn:
            last = bb
        else:
            break
    return last


def _centroid(bb: list[int]) -> tuple[float, float]:
    """bbox = [r1, c1, r2, c2] (tick-space).  Returns (row, col)."""
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _displacement(bb_prev: list[int],
                   bb_curr: list[int]) -> tuple[float, float]:
    """Centroid displacement vector (dr, dc)."""
    pr, pc = _centroid(bb_prev)
    cr, cc = _centroid(bb_curr)
    return (cr - pr, cc - pc)


def _bbox_overlap_area(a: list[int], b: list[int]) -> int:
    """Pixel-area intersection of two bboxes (tick-space)."""
    r1 = max(a[0], b[0])
    c1 = max(a[1], b[1])
    r2 = min(a[2], b[2])
    c2 = min(a[3], b[3])
    if r2 <= r1 or c2 <= c1:
        return 0
    return (r2 - r1) * (c2 - c1)


def _bbox_area(a: list[int]) -> int:
    return max(0, (a[2] - a[0])) * max(0, (a[3] - a[1]))


def _bbox_far_side(a: list[int], direction: str) -> int:
    """Tick coordinate of bbox's edge in the `direction` of motion."""
    if direction == "right": return a[3]
    if direction == "left":  return a[1]
    if direction == "down":  return a[2]
    if direction == "up":    return a[0]
    raise ValueError(direction)


def _is_adjacent(a: list[int], b: list[int],
                  cell_ticks: int = 4,
                  axis_tol: int = 1) -> Optional[str]:
    """Return the side of a on which b touches (a's view: 'right' means
    b is to the right of a), or None if not adjacent.  Adjacency = gap
    within `cell_ticks` along one axis, with overlap on the other.
    """
    # b on a's right
    if (abs(b[1] - a[3]) <= cell_ticks
            and not (b[2] < a[0] - axis_tol or b[0] > a[2] + axis_tol)):
        return "right"
    if (abs(b[3] - a[1]) <= cell_ticks
            and not (b[2] < a[0] - axis_tol or b[0] > a[2] + axis_tol)):
        return "left"
    if (abs(b[0] - a[2]) <= cell_ticks
            and not (b[3] < a[1] - axis_tol or b[1] > a[3] + axis_tol)):
        return "down"
    if (abs(b[2] - a[0]) <= cell_ticks
            and not (b[3] < a[1] - axis_tol or b[1] > a[3] + axis_tol)):
        return "up"
    return None


def _dir_of_vec(dr: float, dc: float,
                  zero_tol: float = 0.5) -> Optional[str]:
    """Cardinal direction of a (dr, dc) vector, or None if ~zero."""
    if abs(dr) < zero_tol and abs(dc) < zero_tol:
        return None
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


# ---------------------------------------------------------------------------
# Playfield boundary
# ---------------------------------------------------------------------------


def _playfield_extent(world: WorldKnowledge) -> Optional[tuple[int, int, int, int]]:
    """Return (r1, c1, r2, c2) of the playfield in tick-space, derived
    from grid_inference.  None if no usable grid info."""
    gi = world.grid_inference
    if gi is None or not getattr(gi, "is_grid_based", False):
        return None
    step = gi.cell_ticks or 0
    rows = gi.rows or 0
    cols = gi.cols or 0
    if step <= 0 or rows <= 0 or cols <= 0:
        return None
    or_, oc = gi.origin_ticks or (0, 0)
    return (or_, oc, or_ + step * rows, oc + step * cols)


def _at_boundary(bb: list[int], extent: tuple[int, int, int, int],
                  direction: str, tol: int = 2) -> bool:
    """Is the bbox flush against the playfield edge on `direction`?"""
    r1, c1, r2, c2 = extent
    if direction == "right": return abs(bb[3] - c2) <= tol
    if direction == "left":  return abs(bb[1] - c1) <= tol
    if direction == "down":  return abs(bb[2] - r2) <= tol
    if direction == "up":    return abs(bb[0] - r1) <= tol
    return False


# ---------------------------------------------------------------------------
# Tier-2 seed relation detectors
# ---------------------------------------------------------------------------


# Geometric thresholds (tick-space, game-agnostic)
_DISP_MATCH_TOL = 1.5          # |d_a - d_b| within this = "same vector"
_ZERO_DISP_TOL = 0.5           # below = "didn't move"
_OVERLAP_MIN_FRAC = 0.1        # min overlap-fraction to count as overlap


def _bbox_motion_signature(prev_bb: list[int],
                              curr_bb: list[int]) -> dict[str, tuple[float, float]]:
    """Compute every meaningful displacement vector for one entity's
    bbox transition: centroid, plus each of the 4 bbox edges treated
    as a representative point.  An entity that grows asymmetrically
    (telescoping extension) has different centroid vs leading-edge
    displacements; matching ANY pair of these between two entities
    captures coupling at the contact face."""
    pr_c = _centroid(prev_bb)
    cu_c = _centroid(curr_bb)
    return {
        "centroid": (cu_c[0] - pr_c[0], cu_c[1] - pr_c[1]),
        "top":      (curr_bb[0] - prev_bb[0], 0.0),
        "bottom":   (curr_bb[2] - prev_bb[2], 0.0),
        "left":     (0.0, curr_bb[1] - prev_bb[1]),
        "right":    (0.0, curr_bb[3] - prev_bb[3]),
    }


def _detect_co_displacement(prev_bb: dict[str, list[int]],
                              curr_bb: dict[str, list[int]],
                              turn: int) -> list[RelationRecord]:
    """B moved by ~the same vector as A.  Signature of contact-coupling,
    carry, chain-push.

    Matches on ANY of {centroid, edge} displacements between the two
    entities — this is what catches a telescoping pusher (whose
    leading edge moves with the pushed object even though its centroid
    moves at half-rate).  At least one of the matching signatures
    must be non-zero.
    """
    out: list[RelationRecord] = []
    names = [n for n in prev_bb if n in curr_bb]
    sigs: dict[str, dict[str, tuple[float, float]]] = {}
    for n in names:
        sigs[n] = _bbox_motion_signature(prev_bb[n], curr_bb[n])
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            # Try every combination of A's and B's motion signatures.
            best: Optional[tuple[str, str, tuple[float, float],
                                   tuple[float, float]]] = None
            for ka, va in sigs[a].items():
                if abs(va[0]) < _ZERO_DISP_TOL and abs(va[1]) < _ZERO_DISP_TOL:
                    continue
                for kb, vb in sigs[b].items():
                    if (abs(vb[0]) < _ZERO_DISP_TOL
                            and abs(vb[1]) < _ZERO_DISP_TOL):
                        continue
                    if (abs(va[0] - vb[0]) <= _DISP_MATCH_TOL
                            and abs(va[1] - vb[1]) <= _DISP_MATCH_TOL):
                        # Prefer centroid-centroid match when available
                        # (it's the strongest evidence of rigid coupling).
                        score = (ka == "centroid") + (kb == "centroid")
                        if best is None or score > (
                            (best[0] == "centroid")
                            + (best[1] == "centroid")
                        ):
                            best = (ka, kb, va, vb)
            if best is None:
                continue
            ka, kb, va, vb = best
            vec = ((va[0] + vb[0]) / 2.0, (va[1] + vb[1]) / 2.0)
            out.append(RelationRecord(
                kind="co_displacement",
                entities=[a, b],
                turn=turn,
                direction=_dir_of_vec(*vec),
                evidence={
                    "match_kind_a": ka,
                    "match_kind_b": kb,
                    "dvec_a": [round(va[0], 1), round(va[1], 1)],
                    "dvec_b": [round(vb[0], 1), round(vb[1], 1)],
                    "match_tol": _DISP_MATCH_TOL,
                },
            ))
    return out


def _detect_motion_blocked(prev_bb: dict[str, list[int]],
                             curr_bb: dict[str, list[int]],
                             action: str,
                             agent_name: Optional[str],
                             extent: Optional[tuple[int, int, int, int]],
                             turn: int) -> list[RelationRecord]:
    """Agent's expected motion didn't happen and a solid is on the
    expected direction.  `action` must be cardinal (UP/DOWN/LEFT/RIGHT
    or sk48-style ACTION1/2 — we look at the agent delta and use that)."""
    out: list[RelationRecord] = []
    if not agent_name or agent_name not in prev_bb or agent_name not in curr_bb:
        return out
    dr, dc = _displacement(prev_bb[agent_name], curr_bb[agent_name])
    if abs(dr) >= _ZERO_DISP_TOL or abs(dc) >= _ZERO_DISP_TOL:
        return out
    # Agent didn't move.  For each cardinal direction, see if a blocker
    # is adjacent on that side.  We don't know which direction the
    # action "intended", so we report the side(s) that have a blocker.
    bb_a = curr_bb[agent_name]
    blockers: list[tuple[str, str]] = []
    for other_name, other_bb in curr_bb.items():
        if other_name == agent_name:
            continue
        side = _is_adjacent(bb_a, other_bb)
        if side:
            blockers.append((side, other_name))
    # Also report playfield-edge blockers
    if extent is not None:
        for d in ("up", "down", "left", "right"):
            if _at_boundary(bb_a, extent, d):
                blockers.append((d, "playfield_boundary"))
    for direction, blocker in blockers:
        out.append(RelationRecord(
            kind="motion_blocked",
            entities=[agent_name, blocker],
            turn=turn,
            direction=direction,
            evidence={
                "action": action,
                "agent_dvec": [round(dr, 1), round(dc, 1)],
            },
        ))
    return out


def _detect_motion_arrested_at(prev_bb: dict[str, list[int]],
                                  curr_bb: dict[str, list[int]],
                                  codisp_pairs: set[frozenset],
                                  turn: int) -> list[RelationRecord]:
    """Entity was moving and stopped just-touching another entity.
    Emitted for an entity whose prev->curr displacement is non-zero and
    whose curr bbox is adjacent to another entity in that direction.

    Suppressed for any pair that already fired co_displacement — those
    are coupled (carrying / chain-pushing), not arrest.  This matters
    for telescoping pushers whose centroid moves at half-rate; without
    suppression they'd register as both pushing AND arresting.

    NOTE: This is a single-step heuristic — we don't have prev-prev to
    confirm continuous motion stopped here.  Treat as a candidate; B's
    contrast pass will refine it.
    """
    out: list[RelationRecord] = []
    names = [n for n in prev_bb if n in curr_bb]
    for a in names:
        dr, dc = _displacement(prev_bb[a], curr_bb[a])
        d = _dir_of_vec(dr, dc)
        if d is None:
            continue
        bb_a = curr_bb[a]
        for b in names:
            if a == b:
                continue
            if frozenset((a, b)) in codisp_pairs:
                continue   # they're coupled, not arrested
            bb_b = curr_bb[b]
            side = _is_adjacent(bb_a, bb_b)
            if side != d:
                continue
            out.append(RelationRecord(
                kind="motion_arrested_at",
                entities=[a, b],
                turn=turn,
                direction=d,
                evidence={
                    "mover_dvec": [round(dr, 1), round(dc, 1)],
                },
            ))
    return out


def _detect_penetration(prev_bb: dict[str, list[int]],
                          curr_bb: dict[str, list[int]],
                          turn: int) -> list[RelationRecord]:
    """One entity's extent entered another's bbox region while the
    other did not displace — pierce/impale signature.

    Implemented as: pair (A, B) where curr-overlap > prev-overlap by
    more than a threshold AND B's displacement is ~zero.  We do not
    require A to be a "tip" — that's the actor's interpretation.
    """
    out: list[RelationRecord] = []
    names = [n for n in prev_bb if n in curr_bb]
    seen: set[tuple[str, str]] = set()
    for a in names:
        for b in names:
            if a == b or (a, b) in seen or (b, a) in seen:
                continue
            seen.add((a, b))
            ov_prev = _bbox_overlap_area(prev_bb[a], prev_bb[b])
            ov_curr = _bbox_overlap_area(curr_bb[a], curr_bb[b])
            min_area = min(_bbox_area(curr_bb[a]), _bbox_area(curr_bb[b]))
            if min_area == 0:
                continue
            gained = ov_curr - ov_prev
            if gained / min_area < _OVERLAP_MIN_FRAC:
                continue
            # Which of the two stayed put?  That's the penetrated party.
            dr_a, dc_a = _displacement(prev_bb[a], curr_bb[a])
            dr_b, dc_b = _displacement(prev_bb[b], curr_bb[b])
            still_a = abs(dr_a) < _ZERO_DISP_TOL and abs(dc_a) < _ZERO_DISP_TOL
            still_b = abs(dr_b) < _ZERO_DISP_TOL and abs(dc_b) < _ZERO_DISP_TOL
            if still_a == still_b:
                # Either both moved or both still — can't determine
                # which penetrated which.  Skip.
                continue
            penetrator, penetrated = (b, a) if still_a else (a, b)
            out.append(RelationRecord(
                kind="penetration",
                entities=[penetrator, penetrated],
                turn=turn,
                direction=_dir_of_vec(
                    *_displacement(prev_bb[penetrator],
                                     curr_bb[penetrator])
                ),
                evidence={
                    "overlap_gained": gained,
                    "min_bbox_area": min_area,
                    "overlap_frac": round(gained / min_area, 2),
                },
            ))
    return out


def _detect_support_relation(curr_bb: dict[str, list[int]],
                                extent: Optional[tuple[int, int, int, int]],
                                turn: int) -> list[RelationRecord]:
    """For each entity B and each cardinal direction d, is there a
    solid (another entity OR the playfield boundary) IMMEDIATELY on B's
    `d` side?  This is the gating condition the physical lens uses to
    predict push-vs-pierce.  Pure geometry; no action required."""
    out: list[RelationRecord] = []
    names = list(curr_bb.keys())
    for b in names:
        bb_b = curr_bb[b]
        for d in ("up", "down", "left", "right"):
            # Find what's on B's d side.
            supporter: Optional[str] = None
            for s in names:
                if s == b:
                    continue
                bb_s = curr_bb[s]
                if _is_adjacent(bb_b, bb_s) == d:
                    supporter = s
                    break
            if supporter is None and extent is not None:
                if _at_boundary(bb_b, extent, d):
                    supporter = "playfield_boundary"
            if supporter is None:
                continue
            out.append(RelationRecord(
                kind="support_relation",
                entities=[b, supporter],
                turn=turn,
                direction=d,
            ))
    return out


# ---------------------------------------------------------------------------
# Static (single-frame) relations — alignment, ordering, clearance
#
# These describe the CURRENT configuration (not a transition).  They are
# the relational replacements for raw coordinates: instead of leaving the
# actor to read bboxes and work out "which is left of which / what's in a
# row / how far to the wall", the substrate states it directly.  All
# quantized in CELLS (game-agnostic) when a grid is locked.
# ---------------------------------------------------------------------------


def _bands_overlap(a: list[int], b: list[int], axis: str,
                    tol: int = 1) -> bool:
    """Do two bboxes overlap on the given axis's extent?  axis='row'
    checks vertical overlap (same horizontal band); axis='col' checks
    horizontal overlap (same vertical band)."""
    if axis == "row":
        return not (a[2] < b[0] - tol or a[0] > b[2] + tol)
    return not (a[3] < b[1] - tol or a[1] > b[3] + tol)


def _ticks_to_cells(ticks: float, cell_ticks: Optional[int]) -> float:
    if not cell_ticks:
        return round(ticks, 1)
    return round(ticks / cell_ticks, 1)


def _detect_aligned(curr_bb: dict[str, list[int]],
                     cell_ticks: Optional[int],
                     turn: int) -> list[RelationRecord]:
    """Pairs sharing a row band (same_row) or column band (same_col),
    with the perpendicular offset quantized in cells.  Replaces the
    actor eyeballing 'are these two at the same height'."""
    out: list[RelationRecord] = []
    names = list(curr_bb.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ba, bb = curr_bb[a], curr_bb[b]
            ca = _centroid(ba)
            cb = _centroid(bb)
            if _bands_overlap(ba, bb, "row"):
                out.append(RelationRecord(
                    kind="same_row", entities=[a, b], turn=turn,
                    direction="horizontal",
                    evidence={"col_gap_cells":
                              abs(_ticks_to_cells(ca[1] - cb[1], cell_ticks))},
                ))
            elif _bands_overlap(ba, bb, "col"):
                out.append(RelationRecord(
                    kind="same_col", entities=[a, b], turn=turn,
                    direction="vertical",
                    evidence={"row_gap_cells":
                              abs(_ticks_to_cells(ca[0] - cb[0], cell_ticks))},
                ))
    return out


def _detect_ordered_along(curr_bb: dict[str, list[int]],
                           turn: int) -> list[RelationRecord]:
    """Group entities by shared band and emit, per group of >=2, the
    members in order along the free axis.  This is the relation that
    directly answers 'in what order are these laid out' — and therefore
    'which one does something approaching from one side reach first'.

    Its absence is exactly what caused the occlusion error: the blocks
    were a left-to-right row green<blue<orange<red, so an arm extending
    rightward reaches green FIRST and it occludes the rest.  Now the
    substrate states the order instead of leaving the actor to derive it.
    """
    out: list[RelationRecord] = []
    names = list(curr_bb.keys())

    def _emit_groups(axis: str, sort_idx: int, direction: str,
                      order_label: str):
        # Build connected groups by shared band along `axis`.
        remaining = list(names)
        used: set[str] = set()
        for seed in names:
            if seed in used:
                continue
            group = [seed]
            used.add(seed)
            for other in names:
                if other in used:
                    continue
                if any(_bands_overlap(curr_bb[g], curr_bb[other], axis)
                       for g in group):
                    group.append(other)
                    used.add(other)
            if len(group) < 2:
                continue
            ordered = sorted(group,
                             key=lambda n: _centroid(curr_bb[n])[sort_idx])
            out.append(RelationRecord(
                kind="ordered_along", entities=ordered, turn=turn,
                direction=direction,
                evidence={
                    "order": order_label,
                    "note": (f"approaching from the {order_label.split('_')[0]} "
                             f"reaches '{ordered[0]}' first; it occludes the "
                             f"rest until cleared"),
                },
            ))

    _emit_groups("row", 1, "horizontal", "left_to_right")
    _emit_groups("col", 0, "vertical", "top_to_bottom")
    return out


def _detect_co_confined(curr_bb: dict[str, list[int]],
                         cell_ticks: Optional[int],
                         turn: int,
                         names: Optional[list[str]] = None) -> list[RelationRecord]:
    """Two entities CONFINED to the same narrow corridor — sharing a band on
    one axis and tightly stacked along the other (within ~one entity-length),
    so one DIRECTLY blocks the other's path along the corridor.

    This is the structural fact a bbox/`same_col` reading does NOT make
    explicit and that an actor needs to plan around: e.g. one block sitting
    directly above another in the same column channel.  Neither can move
    toward the other along the corridor without pushing it; to move the rear
    one PAST the front one, the front one must first be evacuated SIDEWAYS out
    of the shared corridor.  Emitting it makes 'why can't I just go up here'
    representable instead of something the actor has to (mis)deduce.

    Game-agnostic: pure geometry over bboxes; no entity vocabulary.
    """
    out: list[RelationRecord] = []
    # Co-confinement is meaningful only between MANEUVERABLE objects that can
    # block one another. The caller restricts `names` to those (excluding the
    # agent's own arm parts and the fixed reference/HUD strip) — otherwise the
    # relation fires on the HUD targets ("evacuate hud_target_red sideways",
    # nonsense) and on the rigid arm assembly (which is in a line by
    # construction). Default to all bboxes when unrestricted (unit tests).
    if names is None:
        names = list(curr_bb.keys())
    else:
        names = [n for n in names if n in curr_bb]
    seen: set = set()

    def _emit(axis: str, near_idx: int, far_idx: int,
               width_axis_lo: int, width_axis_hi: int, order_label: str):
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if frozenset((a, b)) in seen:
                    continue
                ba, bb = curr_bb[a], curr_bb[b]
                # must share a band on the PERPENDICULAR (corridor-width) axis
                if not _bands_overlap(ba, bb, axis):
                    continue
                # shared-width fraction (how much they overlap across the
                # corridor) — require a substantial corridor, not a corner kiss
                lo = max(ba[width_axis_lo], bb[width_axis_lo])
                hi = min(ba[width_axis_hi], bb[width_axis_hi])
                overlap = hi - lo
                wa = ba[width_axis_hi] - ba[width_axis_lo]
                wb = bb[width_axis_hi] - bb[width_axis_lo]
                if overlap < 0.6 * min(wa, wb):
                    continue
                # gap along the corridor (free axis) between near edges
                a_lo, a_hi = ba[near_idx], ba[far_idx]
                b_lo, b_hi = bb[near_idx], bb[far_idx]
                if a_lo <= b_lo:
                    front, rear = a, b
                    gap = b_lo - a_hi
                    flen = a_hi - a_lo
                else:
                    front, rear = b, a
                    gap = a_lo - b_hi
                    flen = b_hi - b_lo
                if gap < 0:
                    continue                       # overlapping -> not a stack
                # CONFINED iff within ~one entity-length along the corridor
                if gap > max(flen, (cell_ticks or 4) * 1):
                    continue
                seen.add(frozenset((a, b)))
                gap_cells = (round(gap / cell_ticks, 1)
                             if cell_ticks else gap)
                # order near->far along the corridor (front first)
                ordered = [front, rear]
                out.append(RelationRecord(
                    kind="co_confined", entities=ordered, turn=turn,
                    direction=None,
                    evidence={
                        "axis": axis, "order": order_label,
                        "gap_cells": gap_cells,
                        "note": (f"'{front}' sits directly {order_label} '{rear}' "
                                 f"in a shared {axis}-corridor (gap {gap_cells} "
                                 f"cells). '{rear}' cannot move toward '{front}' "
                                 f"without pushing it; to pass, '{front}' must "
                                 f"first be moved sideways out of the corridor."),
                    },
                ))

    # vertical corridor (share columns; stacked in rows): front = upper
    _emit("col", 0, 2, 1, 3, "above")
    # horizontal corridor (share rows; lined up in cols): front = left
    _emit("row", 1, 3, 0, 2, "left_of")
    return out


def _detect_clearance(curr_bb: dict[str, list[int]],
                       extent: Optional[tuple[int, int, int, int]],
                       cell_ticks: Optional[int],
                       turn: int) -> list[RelationRecord]:
    """For each entity and each cardinal direction, how many CELLS of
    empty space before the nearest blocker (another entity in the same
    perpendicular band, or the playfield boundary).  This is the
    quantized metric the actor needs for 'is there room to push right',
    replacing raw-coordinate subtraction.  Only emits when clearance > 0
    (a 0-clearance side is already a support_relation)."""
    if not cell_ticks:
        return []
    out: list[RelationRecord] = []
    names = list(curr_bb.keys())
    for b in names:
        bb = curr_bb[b]
        for d, axis, edge_idx, sign in (
            ("right", "row", 3, +1), ("left", "row", 1, -1),
            ("down", "col", 2, +1), ("up", "col", 0, -1),
        ):
            my_edge = bb[edge_idx]
            # nearest blocker edge in direction d
            best_gap: Optional[float] = None
            blocker: Optional[str] = None
            for s in names:
                if s == b:
                    continue
                bs = curr_bb[s]
                if not _bands_overlap(bb, bs, axis):
                    continue
                # opposite edge of the blocker facing us
                opp = bs[{3: 1, 1: 3, 2: 0, 0: 2}[edge_idx]]
                gap = (opp - my_edge) * sign
                if gap > 0 and (best_gap is None or gap < best_gap):
                    best_gap = gap
                    blocker = s
            # boundary
            if extent is not None:
                bound = {3: extent[3], 1: extent[1],
                         2: extent[2], 0: extent[0]}[edge_idx]
                gap_bound = (bound - my_edge) * sign
                if gap_bound >= 0 and (best_gap is None or gap_bound < best_gap):
                    best_gap = gap_bound
                    blocker = "playfield_boundary"
            if best_gap is None or best_gap <= 0:
                continue
            cells = _ticks_to_cells(best_gap, cell_ticks)
            if cells < 1.0:
                # Sub-cell gaps are adjacency, already covered by
                # same_row / support_relation; not meaningful "room".
                continue
            out.append(RelationRecord(
                kind="clearance", entities=[b, blocker or "?"], turn=turn,
                direction=d,
                evidence={"cells": cells, "blocker": blocker},
            ))
    return out


def _detect_overlapping(curr_bb: dict[str, list[int]], turn: int,
                         min_frac: float = 0.1,
                         gap_tol: int = 2) -> list[RelationRecord]:
    """STANDING current-frame CONTIGUITY between entity pairs: they either
    overlap, OR touch (gap <= gap_tol ticks) along one axis with their
    perpendicular bands overlapping.

    Unlike `penetration` (a one-turn transition), this fires EVERY turn two
    entities are coincident/contiguous -- the persistent 'A is on the same
    shaft / contiguous run as B' signal.  A skewer is exactly such a run
    (gripper-arm-block-block along one row); a block on a different row is
    excluded.  It is what lets the substrate tell whether a block is still on
    the skewer and therefore NOTICE it LEAVING (un-skewering), which a
    change-only channel and the latched HUD both miss.  Contiguity rather than
    strict overlap is required because perceived bboxes of skewered pieces
    TOUCH rather than overlap.  Game-agnostic: pure bbox geometry over
    open-vocab entities; consumers decide what contiguity MEANS."""
    out: list[RelationRecord] = []
    names = list(curr_bb.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ba, bb = curr_bb[a], curr_bb[b]
            connected = False
            frac = 0.0
            ov = _bbox_overlap_area(ba, bb)
            if ov > 0:
                denom = min(_bbox_area(ba), _bbox_area(bb)) or 1
                frac = ov / denom
                connected = frac >= min_frac
            if not connected:
                row_overlap = ba[0] <= bb[2] and bb[0] <= ba[2]
                col_overlap = ba[1] <= bb[3] and bb[1] <= ba[3]
                col_gap = max(bb[1] - ba[3], ba[1] - bb[3])
                row_gap = max(bb[0] - ba[2], ba[0] - bb[2])
                if (row_overlap and 0 < col_gap <= gap_tol) or \
                   (col_overlap and 0 < row_gap <= gap_tol):
                    connected = True
            if connected:
                out.append(RelationRecord(
                    kind="overlapping", entities=[a, b], turn=turn,
                    direction=None, evidence={"overlap_frac": round(frac, 3)},
                ))
    return out


def overlap_reachable(relations: list, start: Optional[str]) -> set:
    """Entities reachable from `start` via `overlapping` relations this turn
    (transitive overlap chain).  The agent's carried/skewered ASSEMBLY:
    agent -> arm/extension -> skewered blocks.  Membership lost between turns
    = that block left the skewer.  Excludes `start` itself.  Accepts relations
    as dicts or RelationRecords."""
    if not start:
        return set()
    adj: dict[str, set] = {}
    for r in (relations or []):
        rd = _rel_as_dict(r)
        if rd.get("kind") != "overlapping":
            continue
        ents = rd.get("entities") or []
        if len(ents) < 2:
            continue
        a, b = ents[0], ents[1]
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    seen: set = set()
    stack = [start]
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(m for m in adj.get(n, ()) if m not in seen)
    seen.discard(start)
    return seen


# ---------------------------------------------------------------------------
# Per-turn entry point
# ---------------------------------------------------------------------------


def _agent_name(world: WorldKnowledge) -> Optional[str]:
    for r in world.entities.values():
        if r.current_role == "agent":
            return r.name
    return None


def compute_relations_for_turn(world: WorldKnowledge,
                                  prev_turn: int,
                                  curr_turn: int,
                                  action: str = "") -> list[RelationRecord]:
    """Compute all seed relations at the transition prev_turn ->
    curr_turn.  Reads bboxes from EntityRecord.bbox_history; pure
    function of (world.entities, world.grid_inference)."""
    prev_bb: dict[str, list[int]] = {}
    curr_bb: dict[str, list[int]] = {}
    for rec in world.entities.values():
        bp = _bbox_at_turn(rec, prev_turn)
        bc = _bbox_at_turn(rec, curr_turn)
        if bp is not None:
            prev_bb[rec.name] = bp
        if bc is not None:
            curr_bb[rec.name] = bc
    agent = _agent_name(world)
    extent = _playfield_extent(world)
    gi = world.grid_inference
    cell_ticks = gi.cell_ticks if gi is not None else None
    relations: list[RelationRecord] = []
    # Temporal (transition) relations
    codisp = _detect_co_displacement(prev_bb, curr_bb, curr_turn)
    relations += codisp
    codisp_pairs = {frozenset(r.entities) for r in codisp}
    relations += _detect_motion_blocked(prev_bb, curr_bb, action,
                                          agent, extent, curr_turn)
    relations += _detect_motion_arrested_at(prev_bb, curr_bb,
                                              codisp_pairs, curr_turn)
    relations += _detect_penetration(prev_bb, curr_bb, curr_turn)
    # Static (current-frame) relations — the relational replacements for
    # raw coordinates: alignment, axis-ordering (occlusion), clearance.
    relations += _detect_support_relation(curr_bb, extent, curr_turn)
    relations += _detect_aligned(curr_bb, cell_ticks, curr_turn)
    relations += _detect_ordered_along(curr_bb, curr_turn)
    # co_confined only over MANEUVERABLE objects: exclude the agent's own arm
    # parts and fixed reference roles (hud / reference / decoration /
    # background) so the relation names real "this blocks that" constraints,
    # not the rigid arm line or the static HUD strip. Role-based, game-agnostic.
    _excl_roles = {"hud", "reference", "decoration", "background"}
    _movable = [r.name for r in world.entities.values()
                if r.name != agent
                and getattr(r, "current_role", None) not in _excl_roles]
    relations += _detect_co_confined(curr_bb, cell_ticks, curr_turn,
                                      names=_movable)
    relations += _detect_clearance(curr_bb, extent, cell_ticks, curr_turn)
    relations += _detect_overlapping(curr_bb, curr_turn)
    return relations


def format_relations_block(relations: list[RelationRecord],
                              max_per_kind: int = 6) -> str:
    """Compact strategy-prompt surface.  Groups by kind; caps per kind
    so long support_relation lists don't dominate."""
    if not relations:
        return "  (no temporal relations this turn — no entity displacements)"
    by_kind: dict[str, list[RelationRecord]] = {}
    for r in relations:
        by_kind.setdefault(r.kind, []).append(r)
    lines: list[str] = []
    # ordered_along first — it is the most decision-relevant (it answers
    # "what's laid out in what order / what does an approach reach first").
    # same_row/same_col are higher-cap because they replace coordinates.
    order = (
        ("ordered_along", max_per_kind),
        ("co_confined", max_per_kind),
        ("clearance", max_per_kind),
        ("co_displacement", max_per_kind),
        ("penetration", max_per_kind),
        ("motion_arrested_at", max_per_kind),
        ("motion_blocked", max_per_kind),
        ("same_row", max_per_kind * 2),
        ("same_col", max_per_kind * 2),
        ("support_relation", max_per_kind),
    )
    for kind, cap in order:
        items = by_kind.get(kind, [])
        if not items:
            continue
        lines.append(f"  {kind}:")
        for r in items[:cap]:
            lines.append(f"    - {r.short()}")
        if len(items) > cap:
            lines.append(f"    ... and {len(items) - cap} more")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relation-based interrupt conditions (for planned-sequence execution)
#
# An actor that commits a multi-step plan can declare interrupt_conditions
# that the substrate evaluates against the CURRENT turn's Layer A relations,
# so an open-loop sequence halts when the geometry says stop — instead of
# overshooting (the turn-21/22 bug: a bulk-extend ran one step past its
# intended column and entered the next block's column band).  Game-agnostic:
# conditions reference relation kinds / directions / entities, never game
# vocabulary.  Non-relational conditions (score_advance, visual_event, ...)
# return False here and are handled by the driver's token checks.
# ---------------------------------------------------------------------------


def _cmp(a: float, op: str, b: float) -> bool:
    if op == "<=": return a <= b
    if op == "<":  return a < b
    if op == ">=": return a >= b
    if op == ">":  return a > b
    if op == "==": return a == b
    return False


def _rel_as_dict(r) -> dict:
    return r if isinstance(r, dict) else r.as_dict()


def evaluate_interrupt_condition(condition: str, relations: list) -> bool:
    """Does a structured, relation-based interrupt condition currently
    hold against `relations` (this turn's Layer A relations, as dicts or
    RelationRecords)?  Supported forms (all game-agnostic):

      relation:<kind>[:<dir>][:<entity>]   e.g. 'relation:same_col',
                                           'relation:penetration:right',
                                           'relation:same_col::block_red'
      clearance:[<entity>:]<dir><op><N>    e.g. 'clearance:right<=1',
                                           'clearance:arm:right<=1'
      adjacent[:<dir>][:<entity>]          (a support_relation is present)

    Returns False for non-relational / unrecognized conditions (the
    driver evaluates score/visual_event/etc. separately)."""
    if not condition:
        return False
    rels = [_rel_as_dict(r) for r in (relations or [])]
    c = condition.strip().lower()

    m = re.match(r"clearance:(?:([a-z0-9_]+):)?(up|down|left|right)"
                 r"\s*(<=|<|==|>=|>)\s*([0-9.]+)$", c)
    if m:
        ent, dir_, op, n = m.group(1), m.group(2), m.group(3), float(m.group(4))
        for r in rels:
            if r.get("kind") != "clearance":
                continue
            if (r.get("direction") or "").lower() != dir_:
                continue
            if ent and ent not in [e.lower() for e in r.get("entities", [])]:
                continue
            cells = (r.get("evidence") or {}).get("cells")
            if cells is not None and _cmp(float(cells), op, n):
                return True
        return False

    if c.startswith("relation:"):
        # Tolerant positional parse of relation:<kind>[:<dir>][:<entity>].
        # Accepts ALL documented forms, including the empty-direction
        # double-colon form 'relation:<kind>::<entity>' (which the previous
        # regex silently rejected, so a stop-condition like
        # 'relation:same_row::block_red' was UNRECOGNIZED -> always False ->
        # the repeat-until executor could never stop and stalled).  Empty
        # segments (from '::') are skipped; a direction word fills the dir
        # slot, any other non-empty token is the entity.
        _dirs = {"up", "down", "left", "right"}
        parts = c.split(":")
        kind = parts[1] if len(parts) > 1 else ""
        dir_ = None
        ent = None
        for p in parts[2:]:
            if not p:
                continue
            if p in _dirs and dir_ is None:
                dir_ = p
            else:
                ent = p
        if kind:
            for r in rels:
                if (r.get("kind") or "").lower() != kind:
                    continue
                if dir_ and (r.get("direction") or "").lower() != dir_:
                    continue
                if ent and ent not in [e.lower() for e in r.get("entities", [])]:
                    continue
                return True
            return False

    m = re.match(r"adjacent(?::(up|down|left|right))?(?::([a-z0-9_]+))?$", c)
    if m:
        dir_, ent = m.group(1), m.group(2)
        for r in rels:
            if r.get("kind") != "support_relation":
                continue
            if dir_ and (r.get("direction") or "").lower() != dir_:
                continue
            if ent and ent not in [e.lower() for e in r.get("entities", [])]:
                continue
            return True
        return False

    return False
