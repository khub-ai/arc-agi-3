"""Per-instance perception + substrate validation oracle.

WHY THIS EXISTS
---------------
The VLM is unreliable at PRECISE visual tasks — exact positions, counting,
alignment, "is this the same block or a different one," "did this change."
The substrate is exact at all of them. So the contract is: the VLM NAMES the
entity types and drives strategy; the substrate measures and validates,
keyed on the VLM's names, and the VLM decides from those FACTS, never from
eyeballing the frame.

This module is the substrate side. It:
  1. SEGMENTS each VLM-named type (block_red, ...) into INSTANCES (connected
     components), with stable ids tracked across turns (block_red#1, #2, ...).
  2. Answers precise factual QUERIES the VLM poses — locate, count,
     same_bitmap, changed, aligned, relation, order_along.
  3. Checks declared INVARIANTS each turn and flags violations
     ("red#1 and red#2 should be identical", "the wall must not change").

It NEVER classifies or decides — it only measures + reports. Geometry is in
tick coords (the raw 64x64 frame, 1px == 1tick), inclusive bboxes
[top,left,bottom,right], matching the rest of the substrate.

Behavioral facts (impaled vs free) come from the co_displacement attachment
classifier, not a single frame; this module exposes the GEOMETRIC proxy
("agent body overlaps the instance") and leaves the behavioral verdict to the
attachment classifier (which a later pass extends to per-instance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import numpy as np
    _OK = True
except Exception:
    _OK = False

# Reuse the existing primitives (single source of truth).
try:
    from agnostic_segmentation import _cca_bboxes
    from frame_bbox_refresh import _load, _background_colors
    _DEPS = True
except Exception:
    _DEPS = False


@dataclass
class Instance:
    """One tracked entity instance with a stable id."""
    inst_id: str                       # 'block_red#1'
    type_name: str                     # 'block_red'
    bbox: Tuple[int, int, int, int]    # (top,left,bottom,right) inclusive ticks
    grid: Tuple                        # bg-removed cropped pixel grid (hashable)
    pixels: int                        # non-bg pixel count

    @property
    def center(self) -> Tuple[float, float]:
        t, l, b, r = self.bbox
        return ((t + b) / 2.0, (l + r) / 2.0)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def _type_mask(region, color, bg, max_dist: int = 90):
    """Boolean mask of pixels that belong to `color` — nearest-color (closer
    to `color` than to any background color, within max_dist). Conflation-safe
    (red vs orange) the same way frame_bbox_refresh is."""
    col = np.array(color, dtype=np.int16)
    dist = np.abs(region.astype(np.int16) - col).sum(axis=2)
    mask = dist <= max_dist
    if bg:
        bg_arr = np.array(list(bg), dtype=np.int16)
        bgd = np.min(np.abs(region[:, :, None, :] - bg_arr[None, None, :, :])
                     .sum(axis=3), axis=2)
        mask &= dist < bgd
    return mask


def _crop_grid(region, bbox_excl, bg) -> Tuple:
    """Hashable cropped pixel grid for the bbox, with background pixels zeroed
    to None — the instance's bitmap signature (shape + colors)."""
    r1, c1, r2, c2 = bbox_excl
    sub = region[r1:r2, c1:c2]
    bg_set = set(bg)
    rows = []
    for rr in range(sub.shape[0]):
        row = []
        for cc in range(sub.shape[1]):
            px = tuple(int(x) for x in sub[rr, cc])
            row.append(None if px in bg_set else px)
        rows.append(tuple(row))
    return tuple(rows)


def segment_instances(frame, name_colors: Dict[str, tuple],
                      play_rows: int = 52, min_pixels: int = 4
                      ) -> Dict[str, List[dict]]:
    """Segment each named type into instance components (no ids yet).
    Returns {type_name: [{bbox(inclusive), grid, pixels}]}, sorted L->R,T->B."""
    if not (_OK and _DEPS):
        return {}
    arr = _load(frame) if not isinstance(frame, np.ndarray) else frame
    if arr is None:
        return {}
    H = arr.shape[0]
    play_rows = min(play_rows, H)
    region = arr[:play_rows]
    bg = _background_colors(arr, play_rows)
    # A color the VLM NAMED as an entity is, by definition, NOT background —
    # drop it from the bg set (otherwise a board with few colors mis-classifies
    # the entity color as background and segments nothing).
    named = list(name_colors.values())
    bg = {c for c in bg
          if not any(sum(abs(int(c[i]) - int(nc[i])) for i in range(3)) <= 60
                     for nc in named)}
    out: Dict[str, List[dict]] = {}
    for tname, color in name_colors.items():
        mask = _type_mask(region, color, bg)
        comps = _cca_bboxes(mask, min_pixels)        # [(r1,c1,r2,c2) exclusive]
        insts = []
        for (r1, c1, r2, c2) in comps:
            grid = _crop_grid(region, (r1, c1, r2, c2), bg)
            pix = sum(1 for row in grid for v in row if v is not None)
            insts.append({"bbox": (r1, c1, r2 - 1, c2 - 1),
                          "grid": grid, "pixels": pix})
        insts.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
        out[tname] = insts
    return out


# ---------------------------------------------------------------------------
# Tracking (stable ids across turns)
# ---------------------------------------------------------------------------

class InstanceTracker:
    """Holds the current tracked instances and assigns stable ids across turns
    by nearest-center matching within a type."""

    def __init__(self, match_dist: float = 8.0):
        self.match_dist = match_dist
        self._counter: Dict[str, int] = {}
        self.instances: Dict[str, Instance] = {}   # inst_id -> Instance
        self.turn: int = -1

    def _new_id(self, tname: str) -> str:
        self._counter[tname] = self._counter.get(tname, 0) + 1
        return f"{tname}#{self._counter[tname]}"

    def update(self, frame, name_colors: Dict[str, tuple],
               play_rows: int = 52, turn: int = 0) -> Dict[str, Instance]:
        """Segment `frame` and match to the previous instances, preserving ids.
        Returns the new {inst_id: Instance}."""
        seg = segment_instances(frame, name_colors, play_rows)
        prev_by_type: Dict[str, List[Instance]] = {}
        for inst in self.instances.values():
            prev_by_type.setdefault(inst.type_name, []).append(inst)

        new_instances: Dict[str, Instance] = {}
        for tname, comps in seg.items():
            prevs = list(prev_by_type.get(tname, []))
            used = set()
            for comp in comps:
                cy = (comp["bbox"][0] + comp["bbox"][2]) / 2.0
                cx = (comp["bbox"][1] + comp["bbox"][3]) / 2.0
                best, best_d = None, self.match_dist
                for p in prevs:
                    if p.inst_id in used:
                        continue
                    d = abs(p.center[0] - cy) + abs(p.center[1] - cx)
                    if d <= best_d:
                        best, best_d = p, d
                inst_id = best.inst_id if best else self._new_id(tname)
                if best:
                    used.add(best.inst_id)
                new_instances[inst_id] = Instance(
                    inst_id=inst_id, type_name=tname, bbox=comp["bbox"],
                    grid=comp["grid"], pixels=comp["pixels"])
        self.instances = new_instances
        self.turn = turn
        return new_instances


# ---------------------------------------------------------------------------
# Validation / measurement oracle (the VLM pulls these; never eyeballs)
# ---------------------------------------------------------------------------

def locate(tr: InstanceTracker, inst_id: str) -> Optional[tuple]:
    inst = tr.instances.get(inst_id)
    return inst.bbox if inst else None


def count(tr: InstanceTracker, type_name: str) -> int:
    return sum(1 for i in tr.instances.values() if i.type_name == type_name)


def list_ids(tr: InstanceTracker, type_name: Optional[str] = None) -> List[str]:
    ids = [i.inst_id for i in tr.instances.values()
           if type_name is None or i.type_name == type_name]
    return sorted(ids)


def same_bitmap(tr: InstanceTracker, a_id: str, b_id: str) -> dict:
    """Are two instances' bitmaps truly identical? Returns
    {identical: bool, reason, n_diff}. Same-size grids compared cell-by-cell;
    different sizes are not identical."""
    a, b = tr.instances.get(a_id), tr.instances.get(b_id)
    if not a or not b:
        return {"identical": False, "reason": "missing instance", "n_diff": -1}
    if len(a.grid) != len(b.grid) or (a.grid and b.grid and
                                      len(a.grid[0]) != len(b.grid[0])):
        return {"identical": False, "reason": "different size",
                "n_diff": -1, "a_size": (len(a.grid), len(a.grid[0]) if a.grid else 0),
                "b_size": (len(b.grid), len(b.grid[0]) if b.grid else 0)}
    n = sum(1 for ra, rb in zip(a.grid, b.grid)
            for va, vb in zip(ra, rb) if va != vb)
    return {"identical": n == 0, "reason": "ok", "n_diff": n}


def changed(tr: InstanceTracker, prev: InstanceTracker, inst_id: str) -> dict:
    """Did this instance's pixels change since the previous turn? Returns
    {changed: bool, n_diff, moved}. Compares the bg-removed grid; also reports
    if it only MOVED (same grid, different bbox)."""
    cur = tr.instances.get(inst_id)
    old = prev.instances.get(inst_id) if prev else None
    if not cur or not old:
        return {"changed": True, "reason": "appeared_or_missing", "n_diff": -1}
    moved = cur.bbox != old.bbox
    if len(cur.grid) != len(old.grid) or (cur.grid and old.grid and
                                          len(cur.grid[0]) != len(old.grid[0])):
        return {"changed": True, "reason": "shape_changed", "n_diff": -1,
                "moved": moved}
    n = sum(1 for ra, rb in zip(cur.grid, old.grid)
            for va, vb in zip(ra, rb) if va != vb)
    return {"changed": n != 0, "reason": "ok", "n_diff": n, "moved": moved}


def aligned(tr: InstanceTracker, a_id: str, b_id: str, kind: str,
            tol: int = 2) -> dict:
    """Geometric alignment check. kind in {on_top_of, below, left_of,
    right_of, same_row, same_col}. Returns {holds: bool, offset, detail}."""
    a, b = tr.instances.get(a_id), tr.instances.get(b_id)
    if not a or not b:
        return {"holds": False, "reason": "missing instance"}
    at, al, ab, ar = a.bbox
    bt, bl, bb, br = b.bbox
    acy, acx = a.center
    bcy, bcx = b.center
    col_overlap = not (ar < bl or al > br)
    row_overlap = not (ab < bt or at > bb)
    if kind == "on_top_of":
        holds = col_overlap and abs(ab - (bt - 1)) <= tol and acy < bcy
        return {"holds": holds, "offset": int(bt - ab),
                "detail": "a's bottom just above b's top, columns overlap"}
    if kind == "below":
        holds = col_overlap and abs(at - (bb + 1)) <= tol and acy > bcy
        return {"holds": holds, "offset": int(at - bb)}
    if kind == "left_of":
        holds = row_overlap and ar < bl
        return {"holds": holds, "offset": int(bl - ar)}
    if kind == "right_of":
        holds = row_overlap and al > br
        return {"holds": holds, "offset": int(al - br)}
    if kind == "same_row":
        return {"holds": abs(acy - bcy) <= tol, "offset": round(acy - bcy, 1)}
    if kind == "same_col":
        return {"holds": abs(acx - bcx) <= tol, "offset": round(acx - bcx, 1)}
    return {"holds": False, "reason": f"unknown kind {kind}"}


def relation(tr: InstanceTracker, a_id: str, b_id: str) -> dict:
    """Full geometric relation between two instances: overlap, adjacency,
    row/col deltas, and left/right/above/below ordering."""
    a, b = tr.instances.get(a_id), tr.instances.get(b_id)
    if not a or not b:
        return {"reason": "missing instance"}
    at, al, ab, ar = a.bbox
    bt, bl, bb, br = b.bbox
    col_overlap = not (ar < bl or al > br)
    row_overlap = not (ab < bt or at > bb)
    gap_h = (bl - ar) if ar < bl else (al - br if al > br else 0)
    gap_v = (bt - ab) if ab < bt else (at - bb if at > bb else 0)
    return {
        "overlap": col_overlap and row_overlap,
        "row_overlap": row_overlap, "col_overlap": col_overlap,
        "horizontal_gap": int(gap_h), "vertical_gap": int(gap_v),
        "a_left_of_b": ar < bl, "a_right_of_b": al > br,
        "a_above_b": ab < bt, "a_below_b": at > bb,
        "adjacent": (row_overlap and abs(gap_h) <= 1) or
                    (col_overlap and abs(gap_v) <= 1),
    }


def order_along(tr: InstanceTracker, type_names, axis: str = "col",
                row: Optional[int] = None, tol: int = 3) -> List[str]:
    """Instances of the given type(s) sorted along an axis ('col' = left->right,
    'row' = top->bottom). If `row` given, only instances whose center row is
    within tol of it (a specific lane). Returns the ordered inst_ids."""
    if isinstance(type_names, str):
        type_names = [type_names]
    items = [i for i in tr.instances.values() if i.type_name in type_names]
    if row is not None:
        items = [i for i in items if abs(i.center[0] - row) <= tol]
    key = (lambda i: i.center[1]) if axis == "col" else (lambda i: i.center[0])
    return [i.inst_id for i in sorted(items, key=key)]


# ---------------------------------------------------------------------------
# Invariant declaration + violation flagging (VLM declares; substrate reports)
# ---------------------------------------------------------------------------

@dataclass
class Invariant:
    """A VLM-declared expectation the substrate checks each turn."""
    kind: str                          # 'identical' | 'unchanged' | 'only_moves'
    args: tuple = ()                   # ids / type involved
    note: str = ""


def check_invariants(tr: InstanceTracker, prev: Optional[InstanceTracker],
                     invariants: List[Invariant]) -> List[dict]:
    """Check declared invariants against the current (and previous) state.
    Returns a list of VIOLATIONS only (empty = all held)."""
    out = []
    for inv in invariants:
        if inv.kind == "identical" and len(inv.args) == 2:
            r = same_bitmap(tr, inv.args[0], inv.args[1])
            if not r.get("identical"):
                out.append({"invariant": "identical", "args": inv.args,
                            "violation": r, "note": inv.note})
        elif inv.kind == "unchanged" and prev is not None and inv.args:
            r = changed(tr, prev, inv.args[0])
            if r.get("changed"):
                out.append({"invariant": "unchanged", "args": inv.args,
                            "violation": r, "note": inv.note})
        elif inv.kind == "only_moves" and prev is not None and inv.args:
            allowed = set(inv.args)
            for iid in tr.instances:
                if iid in allowed:
                    continue
                r = changed(tr, prev, iid)
                if r.get("changed") or r.get("moved"):
                    out.append({"invariant": "only_moves", "unexpected_change": iid,
                                "violation": r, "note": inv.note})
    return out


def instance_attachment(tr: InstanceTracker, carrier_bbox: Optional[tuple],
                        block_types: Optional[List[str]] = None,
                        tol: int = 2,
                        carried_label: str = "carried") -> Dict[str, str]:
    """Per-instance CARRIED/free, GEOMETRICALLY: an object is on the agent's
    carrier if it overlaps the carrier region — the carrier's row band overlaps
    the object's rows AND the object's center lies within the carrier's column
    extent. Otherwise 'free'. ``carrier_bbox`` = (top,left,bottom,right) of the
    agent's carrier / effector / body.

    Game-agnostic: the geometric concept is "is the object on the agent's
    manipulator" (a robot gripper, a skewer, a tray). Callers may pass a
    domain term via ``carried_label`` (e.g. 'impaled' for a threading game),
    but the default is the neutral 'carried'. Geometric proxy; the behavioral
    co_displacement classifier stays the authority where motion history exists.
    """
    if not carrier_bbox:
        return {}
    rt, rl, rb, rr = carrier_bbox
    out: Dict[str, str] = {}
    for iid, inst in tr.instances.items():
        if block_types and inst.type_name not in block_types:
            continue
        t, l, b, r = inst.bbox
        _cy, cx = inst.center
        rows_overlap = not (b < rt - tol or t > rb + tol)
        within_span = (rl - 1) <= cx <= (rr + 1)
        out[iid] = carried_label if (rows_overlap and within_span) else "free"
    return out


def format_instance_factsheet(tr: InstanceTracker,
                              attachment: Optional[Dict[str, str]] = None) -> str:
    """Render the instance-level scene facts for the VLM (replaces eyeballing
    the frame). Positions in tick coords; ordering per type; per-instance
    impaled/free when an attachment map is supplied (with the count)."""
    if not tr.instances:
        return ""
    by_type: Dict[str, List[Instance]] = {}
    for i in tr.instances.values():
        by_type.setdefault(i.type_name, []).append(i)
    lines = ["SCENE FACTS (substrate-measured, per instance; use these, do NOT "
             "eyeball the image):"]
    for tname in sorted(by_type):
        insts = sorted(by_type[tname], key=lambda i: (i.center[0], i.center[1]))
        hdr = f"  {tname}: {len(insts)} instance(s)"
        if attachment:
            # carried = any non-free label (the domain term the caller used);
            # generic, never hardcodes a game-specific word.
            carried = [i for i in insts
                       if attachment.get(i.inst_id) not in (None, "free")]
            lbl = (attachment.get(carried[0].inst_id) if carried else "carried")
            hdr += f" ({len(carried)} {lbl}, {len(insts) - len(carried)} free)"
        lines.append(hdr)
        for i in insts:
            t, l, b, r = i.bbox
            att = (f" [{attachment[i.inst_id]}]"
                   if attachment and i.inst_id in attachment else "")
            lines.append(f"    {i.inst_id}: rows {t}-{b}, cols {l}-{r} "
                         f"({i.pixels}px){att}")
    return "\n".join(lines)
