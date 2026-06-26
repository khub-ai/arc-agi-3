"""Execute parsed probes against a live ARC-AGI-3 env.

Each probe is self-contained: reset the env, apply instructions in
order, then evaluate every observation.  Results are a plain dict that
can be JSON-serialised and compared against the model's outcome_map.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from dsl import (
    DoOne, DoSeq, RepeatDo, Reset,
    ObsRegionDelta, ObsElementMoved, ObsState,
    ObsAvailableActions, ObsScoreDelta, ObsChangeReport,
    ProbeParseResult,
)

ARC_REPO = Path(os.environ.get(
    "ARC_AGI_3_REPO", r"arc-agi-3"
))
sys.path.insert(0, str(ARC_REPO))

from arc_agi import Arcade, OperationMode  # noqa: E402
from arcengine import GameAction  # noqa: E402


def _normalise_frame(raw_frame) -> np.ndarray:
    """Return a 2D grid.  ARC envs sometimes hand back a framestack (list
    of frames) — take the last entry."""
    if isinstance(raw_frame, list):
        if not raw_frame:
            return np.zeros((64, 64), dtype=int)
        inner = raw_frame[-1]
        if isinstance(inner, np.ndarray):
            return inner.astype(int)
        return np.array(inner, dtype=int)
    arr = np.array(raw_frame, dtype=int)
    if arr.ndim == 3:
        return arr[-1]
    return arr


def _action_for_label(label: str) -> GameAction:
    # "ACTION1" -> GameAction.ACTION1
    return GameAction[label]


def _signature_colour(start_grid: np.ndarray, target_bbox) -> int | None:
    """Pick the colour most distinctive to the element's pre-bbox.

    Score each colour inside the bbox by (inside_frac / (outside_frac + eps))
    — i.e. how over-represented it is vs the rest of the grid.  This avoids
    picking the floor colour when the element sits on a large uniform floor.
    """
    r0, c0, r1, c1 = target_bbox
    patch = start_grid[r0:r1 + 1, c0:c1 + 1]
    if patch.size == 0:
        return None
    total_cells    = start_grid.size
    outside_cells  = total_cells - patch.size
    if outside_cells <= 0:
        return int(np.bincount(patch.ravel()).argmax())
    best_c: int | None = None
    best_score = -1.0
    for c in np.unique(patch):
        c = int(c)
        inside  = int(np.sum(patch == c))
        outside = int(np.sum(start_grid == c)) - inside
        in_frac  = inside / patch.size
        out_frac = outside / outside_cells
        score = in_frac / (out_frac + 1e-6)
        if score > best_score:
            best_score = score
            best_c = c
    return best_c


def _bbox_of_value(grid: np.ndarray, target_bbox, tracked_colour: int | None = None):
    """Return (bbox, colour) of the tracked-colour connected region nearest
    to target_bbox.  If tracked_colour is None, derive it from the pre-bbox
    via `_signature_colour` (computed on the pre-probe grid).

    Returns None if the colour has vanished.
    """
    if tracked_colour is None:
        tracked_colour = _signature_colour(grid, target_bbox)
        if tracked_colour is None:
            return None

    mask = grid == tracked_colour
    if not mask.any():
        return None

    # Flood-fill (BFS) connected components of `tracked_colour`, keep the
    # one nearest the target bbox centre.  Avoids returning a huge bbox when
    # the colour also appears in unrelated regions.
    r0, c0, r1, c1 = target_bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    h, w = grid.shape
    visited = np.zeros_like(mask, dtype=bool)
    best: list[int] | None = None
    best_d = float("inf")
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            # BFS
            stack = [(sr, sc)]
            rmin, cmin, rmax, cmax = sr, sc, sr, sc
            while stack:
                rr, cc_ = stack.pop()
                if rr < 0 or rr >= h or cc_ < 0 or cc_ >= w:
                    continue
                if visited[rr, cc_] or not mask[rr, cc_]:
                    continue
                visited[rr, cc_] = True
                if rr < rmin: rmin = rr
                if cc_ < cmin: cmin = cc_
                if rr > rmax: rmax = rr
                if cc_ > cmax: cmax = cc_
                stack.extend([(rr+1, cc_), (rr-1, cc_), (rr, cc_+1), (rr, cc_-1)])
            mcr = (rmin + rmax) / 2.0
            mcc = (cmin + cmax) / 2.0
            d = (mcr - cr) ** 2 + (mcc - cc) ** 2
            if d < best_d:
                best_d = d
                best = [int(rmin), int(cmin), int(rmax), int(cmax)]
    if best is None:
        return None
    return best, tracked_colour


def _connected_components(mask: np.ndarray) -> List[Tuple[Tuple[int,int,int,int], int]]:
    """BFS-flood-fill connected components of a boolean mask.  Returns
    [((r0,c0,r1,c1), cell_count), ...] in no particular order."""
    if not mask.any():
        return []
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    out: List[Tuple[Tuple[int,int,int,int], int]] = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or visited[sr, sc]:
                continue
            stack = [(sr, sc)]
            rmin, cmin, rmax, cmax = sr, sc, sr, sc
            count = 0
            while stack:
                rr, cc_ = stack.pop()
                if rr < 0 or rr >= h or cc_ < 0 or cc_ >= w:
                    continue
                if visited[rr, cc_] or not mask[rr, cc_]:
                    continue
                visited[rr, cc_] = True
                count += 1
                if rr < rmin: rmin = rr
                if cc_ < cmin: cmin = cc_
                if rr > rmax: rmax = rr
                if cc_ > cmax: cmax = cc_
                stack.extend([(rr+1, cc_), (rr-1, cc_), (rr, cc_+1), (rr, cc_-1)])
            out.append(((rmin, cmin, rmax, cmax), count))
    return out


def _bbox_centre(bbox) -> Tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _patch(grid: np.ndarray, bbox, max_side: int = 16) -> list:
    """Return grid slice as python lists; truncates very large regions."""
    r0, c0, r1, c1 = bbox
    if (r1 - r0 + 1) > max_side or (c1 - c0 + 1) > max_side:
        return [[int(x) for x in row[:max_side]]
                for row in grid[r0:r0+max_side, c0:c0+max_side].tolist()]
    return [[int(x) for x in row] for row in grid[r0:r1+1, c0:c1+1].tolist()]


def _build_change_report(
    start_grid:      np.ndarray,
    end_grid:        np.ndarray,
    element_records: Dict[int, dict],
) -> dict:
    """Harness-side semantic summary of start→end change.

    Shape:
      {
        "element_motions":   [{"element_id", "name", "pre_bbox", "post_bbox",
                               "dr", "dc", "moved"}],
        "disappearances":    [{"element_id", "name", "colour"}],
        "appearances":       [{"bbox", "colour", "area"}],
        "counter_changes":   [{"element_id", "name", "before_fill",
                               "after_fill", "direction"}],
        "unexplained_regions": [{"bbox", "cells_changed",
                                 "before_patch", "after_patch"}],
        "full_frame_fallback": null | [[int, ...], ...],
      }
    """
    h, w = start_grid.shape
    diff_mask = (start_grid != end_grid)

    # Mask of cells we have "explained" via element motion / appear / disappear.
    explained = np.zeros_like(diff_mask, dtype=bool)

    element_motions = []
    disappearances  = []
    counter_changes = []

    for eid, rec in element_records.items():
        target = rec.get("bbox")
        if not target or len(target) != 4:
            continue
        sig_col = _signature_colour(start_grid, target)
        if sig_col is None:
            continue
        # Bug fix: use `target` (last-known bbox, updated each frame) as anchor
        # for BOTH pre and post lookups.  Using the stale Round-1 bbox caused the
        # tracker to lock onto trail segments near the origin instead of following
        # the moving element.
        pre  = _bbox_of_value(start_grid, target, sig_col)
        pre_bbox  = pre[0]  if pre  else None
        fn = rec.get("function", "unknown")

        if pre_bbox is None:
            continue  # already gone before probe started — skip

        # For post, anchor to pre_bbox (where we JUST saw the element) not to
        # the Round-1 bbox.  This keeps the tracker following the element as it
        # moves rather than snapping back to the origin on every frame.
        post = _bbox_of_value(end_grid, pre_bbox, sig_col)
        post_bbox = post[0] if post else None

        if post_bbox is None:
            disappearances.append({
                "element_id": int(eid),
                "name":       rec.get("name"),
                "colour":     int(sig_col),
            })
            for r in range(pre_bbox[0], pre_bbox[2]+1):
                for c in range(pre_bbox[1], pre_bbox[3]+1):
                    if 0 <= r < h and 0 <= c < w:
                        explained[r, c] = True
            continue

        pr, pc = _bbox_centre(pre_bbox)
        qr, qc = _bbox_centre(post_bbox)
        dr = round(qr - pr)
        dc = round(qc - pc)
        moved = (pre_bbox != post_bbox)
        # Bug fix: check ACTUAL detected pre_bbox area (not the stale Round-1
        # ref_area) against the 20% frame threshold.  The old code used
        # ref_area_r (the tiny Round-1 bbox area) so the flag never fired even
        # when pre_bbox had grown to cover the whole floor.
        pre_area    = max(1, (pre_bbox[2]  - pre_bbox[0]  + 1) * (pre_bbox[3]  - pre_bbox[1]  + 1))
        post_area   = max(1, (post_bbox[2] - post_bbox[0] + 1) * (post_bbox[3] - post_bbox[1] + 1))
        tracker_unreliable = (
            pre_area  > 0.20 * (h * w)
            or post_area > 0.20 * (h * w)
            or post_area > 10 * pre_area
            or pre_area  > 10 * post_area
        )
        # Advance the stored bbox so the next frame anchors to the current
        # position, not the original Round-1 position.
        rec["bbox"] = list(post_bbox)
        element_motions.append({
            "element_id": int(eid),
            "name":       rec.get("name"),
            "pre_bbox":   list(pre_bbox),
            "post_bbox":  list(post_bbox),
            "dr":         int(dr),
            "dc":         int(dc),
            "moved":      bool(moved),
            "tracker_unreliable": bool(tracker_unreliable),
        })
        # Mark both footprints as explained.
        for bb in (pre_bbox, post_bbox):
            for r in range(bb[0], bb[2]+1):
                for c in range(bb[1], bb[3]+1):
                    if 0 <= r < h and 0 <= c < w:
                        explained[r, c] = True

        # Counter summary: fill = signature-colour pixel count inside pre_bbox.
        if fn in ("counter", "readout"):
            r0, c0, r1, c1 = target
            before_fill = int(np.sum(start_grid[r0:r1+1, c0:c1+1] == sig_col))
            after_fill  = int(np.sum(end_grid  [r0:r1+1, c0:c1+1] == sig_col))
            if before_fill != after_fill:
                counter_changes.append({
                    "element_id": int(eid),
                    "name":       rec.get("name"),
                    "before_fill": before_fill,
                    "after_fill":  after_fill,
                    "direction":   "+" if after_fill > before_fill else "-",
                })

    # Appearances: novel-colour components in end_grid, or components of an
    # existing colour that land on cells which changed AND are unexplained.
    appearances = []
    start_colours = set(int(x) for x in np.unique(start_grid))
    end_colours   = set(int(x) for x in np.unique(end_grid))
    novel_colours = end_colours - start_colours
    for col in novel_colours:
        comps = _connected_components(end_grid == col)
        for bbox, count in comps:
            appearances.append({
                "bbox":   list(bbox),
                "colour": int(col),
                "area":   int(count),
            })
            for r in range(bbox[0], bbox[2]+1):
                for c in range(bbox[1], bbox[3]+1):
                    explained[r, c] = True

    # Unexplained changed cells: diff_mask AND NOT explained.
    leftover = diff_mask & ~explained
    total_changed = int(np.sum(diff_mask))
    leftover_count = int(np.sum(leftover))

    frame_area = h * w
    full_frame_fallback = None
    unexplained_regions: list = []
    if total_changed > 0.30 * frame_area:
        full_frame_fallback = [[int(x) for x in row] for row in end_grid.tolist()]
    else:
        for bbox, count in _connected_components(leftover):
            if count < 1:
                continue
            unexplained_regions.append({
                "bbox":          list(bbox),
                "cells_changed": int(count),
                "before_patch":  _patch(start_grid, bbox),
                "after_patch":   _patch(end_grid,   bbox),
            })

    # Primary motion: the smallest reliably-tracked element that actually
    # moved.  Gives the model a single clean pointer to the likely agent.
    reliable_moved = [m for m in element_motions
                      if m.get("moved") and not m.get("tracker_unreliable")]
    primary_motion = None
    if reliable_moved:
        def _pre_area(m):
            bb = m["pre_bbox"]
            return (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1)
        primary_motion = min(reliable_moved, key=_pre_area)

    # Harness-side agent position: find every element whose function contains
    # "agent" or "cursor" and report its centroid in the POST frame.
    # This gives Oracle a reliable, frame-derived "where is the avatar NOW"
    # without depending on the tracker or model self-reporting.
    agent_pos_after: list[int] | None = None
    for eid, rec in element_records.items():
        fn   = (rec.get("function") or "").lower()
        name = (rec.get("name") or "").lower()
        if not ("agent" in fn or "cursor" in fn or "agent" in name or "cursor" in name):
            continue
        # Use current bbox (already advanced to post_bbox by the fix above).
        cur_bbox = rec.get("bbox")
        if cur_bbox and len(cur_bbox) == 4:
            r0, c0, r1, c1 = cur_bbox
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            # Only trust it if the bbox is small (not the whole floor)
            if area <= 0.15 * h * w:
                agent_pos_after = [round((r0 + r1) / 2), round((c0 + c1) / 2)]
                break

    return {
        "element_motions":      element_motions,
        "primary_motion":       primary_motion,
        "agent_pos_after":      agent_pos_after,
        "disappearances":       disappearances,
        "appearances":          appearances,
        "counter_changes":      counter_changes,
        "unexplained_regions":  unexplained_regions,
        "full_frame_fallback":  full_frame_fallback,
        "totals": {
            "diff_cells":       total_changed,
            "unexplained_cells": leftover_count,
            "frame_area":       frame_area,
        },
    }


def run_probe(
    probe:             ProbeParseResult,
    game_id:           str,
    element_bboxes:    Dict[int, list],   # from the model's ELEMENTS section
    element_records:   Dict[int, dict] | None = None,  # {id: {name, bbox, function}}
) -> Dict[str, Any]:
    """Execute one probe.  Returns a dict with per-observation results
    and a top-level error field if execution blew up."""
    if probe.errors:
        return {
            "probe_id":     probe.probe_id,
            "executed":     False,
            "parse_errors": probe.errors,
        }

    arc = Arcade(
        operation_mode   = OperationMode.OFFLINE,
        environments_dir = str(ARC_REPO / "environment_files"),
    )
    env = arc.make(game_id)
    obs0 = env.reset()
    start_grid  = _normalise_frame(obs0.frame)
    start_score = int(obs0.levels_completed)

    # Apply instructions.
    cur_obs = obs0
    trace: List[Dict[str, Any]] = []
    try:
        for instr in probe.instructions:
            if isinstance(instr, Reset):
                cur_obs = env.reset()
                trace.append({"instr": "RESET"})
            elif isinstance(instr, DoOne):
                cur_obs = env.step(_action_for_label(instr.action))
                trace.append({"instr": f"DO {instr.action}",
                              "state_after": getattr(cur_obs.state, "name", str(cur_obs.state))})
            elif isinstance(instr, DoSeq):
                for a in instr.actions:
                    cur_obs = env.step(_action_for_label(a))
                    trace.append({"instr": f"DO {a} (seq)",
                                  "state_after": getattr(cur_obs.state, "name", str(cur_obs.state))})
            elif isinstance(instr, RepeatDo):
                for _ in range(instr.n):
                    cur_obs = env.step(_action_for_label(instr.action))
                trace.append({"instr": f"REPEAT DO {instr.action} {instr.n}",
                              "state_after": getattr(cur_obs.state, "name", str(cur_obs.state))})
    except Exception as e:  # noqa: BLE001
        return {
            "probe_id":   probe.probe_id,
            "executed":   False,
            "exec_error": f"{type(e).__name__}: {e}",
            "trace":      trace,
        }

    end_grid  = _normalise_frame(cur_obs.frame)
    end_state = getattr(cur_obs.state, "name", str(cur_obs.state))
    end_score = int(cur_obs.levels_completed)
    end_actions = [f"ACTION{int(a)}" for a in cur_obs.available_actions]

    # Evaluate observations.
    obs_results: List[Dict[str, Any]] = []
    for o in probe.observations:
        if isinstance(o, ObsState):
            obs_results.append({"kind": "STATE", "value": end_state})
        elif isinstance(o, ObsAvailableActions):
            obs_results.append({"kind": "AVAILABLE_ACTIONS", "value": end_actions})
        elif isinstance(o, ObsScoreDelta):
            obs_results.append({"kind": "SCORE_DELTA", "value": end_score - start_score})
        elif isinstance(o, ObsRegionDelta):
            r0, c0, r1, c1 = o.bbox
            pre  = start_grid[r0:r1+1, c0:c1+1]
            post = end_grid[r0:r1+1, c0:c1+1]
            delta = int(np.sum(pre != post))
            obs_results.append({
                "kind": "REGION_DELTA",
                "bbox": list(o.bbox),
                "value": delta,
            })
        elif isinstance(o, ObsChangeReport):
            records = element_records
            if records is None:
                records = {int(eid): {"bbox": bb, "name": None, "function": "unknown"}
                           for eid, bb in element_bboxes.items()}
            report = _build_change_report(start_grid, end_grid, records)
            obs_results.append({"kind": "CHANGE_REPORT", **report})
        elif isinstance(o, ObsElementMoved):
            target = element_bboxes.get(o.element_id)
            if target is None:
                obs_results.append({
                    "kind": "ELEMENT_MOVED",
                    "element_id": o.element_id,
                    "error": "element id not in ELEMENTS",
                })
                continue
            sig_col = _signature_colour(start_grid, target)
            before  = _bbox_of_value(start_grid, target, sig_col) if sig_col is not None else None
            before_bbox = before[0] if before is not None else None
            after   = _bbox_of_value(end_grid, target, sig_col) if sig_col is not None else None
            after_bbox = after[0] if after is not None else None
            moved = (after_bbox != before_bbox) if (before_bbox and after_bbox) else None
            obs_results.append({
                "kind": "ELEMENT_MOVED",
                "element_id": o.element_id,
                "pre_bbox":  before_bbox,
                "post_bbox": after_bbox,
                "moved":     bool(moved) if moved is not None else None,
                "tracked_colour": sig_col,
            })

    return {
        "probe_id":     probe.probe_id,
        "hypothesis":   probe.hypothesis,
        "executed":     True,
        "trace":        trace,
        "observations": obs_results,
        "final_state":  end_state,
        "final_score":  end_score,
        "final_actions": end_actions,
    }
