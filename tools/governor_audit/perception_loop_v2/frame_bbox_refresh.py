"""Refresh tracked-entity bboxes from the ACTUAL frame each turn.

WHY THIS EXISTS
---------------
The substrate's downstream geometry (relational kinematics, the mediation
precondition detector, the attachment classifier, the structural producer)
all read ``EntityRecord.bbox_history``. That history is appended from the
PERCEPTION REPLY's bboxes. When the perception layer is sparse, stale, or a
fast-forward placeholder (the lc4 resume responder returns FIXED turn-1
bboxes every turn), the history never reflects reality — every geometry
surface then reasons about a frozen turn-1 board. That silently broke the
mediation precondition ('free intermediary under target' read NOT MET even
when true) and made the attachment classifier default to FREE regardless.

The fix (durable principle P3 — perception geometry comes from the actual
frame, not from trusting the perception text): each turn, segment the live
frame and update each MOVABLE tracked entity's current bbox from it. The
perception VLM still owns SEMANTICS (names, roles, what matters); the
substrate owns GEOMETRY (where things actually are this frame).

APPROACH (game-agnostic)
------------------------
Identity is matched by COLOR, bootstrapped per entity from the start frame
(where the initial bboxes are trustworthy): sample each refreshable
entity's dominant non-background color once, cache it, then each turn set
that entity's current bbox = union of same-colored regions in the live
frame. Aggregate tracks (one ``block_blue`` covering several blue pieces)
are handled naturally — the union spans all matching regions. Static
entities (walls, rails, HUD) are left untouched (they do not move).

No hardcoded colors or game vocabulary: the color is whatever the entity
actually is on screen. Pure numpy + PIL; degrades to a no-op if either is
unavailable.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

try:
    import numpy as np
    from PIL import Image
    _OK = True
except Exception:
    _OK = False


# Roles whose geometry should be refreshed from the frame (they move).
# Static roles (wall/obstacle/rail/hud/reference/background) are skipped.
REFRESHABLE_ROLES = {
    "collectable", "agent", "agent_body", "pushable", "movable",
    "block", "piece", "goal_object",
}

# Module-level color cache: {(game_id, entity_name): (r,g,b)}.
_COLOR_CACHE: dict = {}

# Last level seen, so colors are re-bootstrapped when the level changes. A
# stale color cache across a level boundary is the bug that froze an entity's
# bbox: during a fast-forward the placeholder bbox points at a region that is
# the right color only on the CURRENT level's board, so a color cached from a
# previous level's board never matches and that entity is never re-found.
_LAST_LEVEL: list = [None]


def _load(path) -> Optional["np.ndarray"]:
    try:
        return np.array(Image.open(path).convert("RGB"), dtype=np.int16)
    except Exception:
        return None


def _background_colors(frame, play_rows: int, k: int = 2) -> set:
    """Background COLOURS by STRUCTURE (palette-invariant, Adversarial Test): a
    colour is ground when its pixels are NOT part of any structural foreground
    component (silhouette_track.foreground_components -- ground = large fields +
    repeated lattices/texture).  NOT 'the k most common colours', which assumes a
    single/few-colour background and shatters on a re-skin or a textured board.
    `k` is retained for signature compatibility but unused."""
    import silhouette_track as _ST
    region = np.ascontiguousarray(frame[:play_rows]).astype(int)
    fg_colours = {e["colour"] for e in _ST.foreground_components(region)}
    colors = np.unique(region.reshape(-1, 3), axis=0)
    bg = set()
    for col in colors:
        packed = (int(col[0]) << 16) | (int(col[1]) << 8) | int(col[2])
        if packed not in fg_colours:
            bg.add(tuple(int(x) for x in col))
    return bg


def _saturation(t: tuple) -> int:
    """Crude saturation = max channel - min channel. Saturated colors
    (blue/red blocks) score high; grey (arm rod, rail) scores ~0."""
    return max(t) - min(t)


def _dominant_color(frame, bbox, bg: set, prefer_saturated: bool = True
                    ) -> Optional[tuple]:
    """Most representative NON-background color within a PIXEL bbox.

    With ``prefer_saturated`` (default), each color's vote is its pixel
    count weighted by saturation, so a saturated block colour beats the
    grey arm rod that runs THROUGH the block's bbox — the exact failure
    that made a block bootstrap to the grey rail. If the bbox is genuinely
    grey (e.g. the agent body), the saturated colors simply aren't present
    and the most-common grey still wins via the +1 base weight."""
    r0, c0, r1, c1 = bbox
    r0 = max(0, r0); c0 = max(0, c0)
    sub = frame[r0:r1 + 1, c0:c1 + 1].reshape(-1, 3)
    if sub.size == 0:
        return None
    colors, counts = np.unique(sub, axis=0, return_counts=True)
    best = None; best_score = -1.0
    for col, n in zip(colors, counts):
        t = tuple(int(x) for x in col)
        if t in bg:
            continue
        score = float(n) * (1.0 + (_saturation(t) if prefer_saturated else 0))
        if score > best_score:
            best_score = score; best = t
    return best


def _color_match(a: tuple, b: tuple, tol: int = 40) -> bool:
    return all(abs(int(a[i]) - int(b[i])) <= tol for i in range(3))


def _union_bbox_of_color(frame, color: tuple, play_rows: int,
                         tol: int = 40, min_pixels: int = 3):
    """Bounding box (r0,c0,r1,c1 in PIXELS) of all pixels matching ``color``
    within the playfield, or None. (Single-color thresholding — see
    _assign_nearest_bboxes for the multi-color, conflation-safe version.)"""
    region = frame[:play_rows]
    diff = np.abs(region - np.array(color, dtype=np.int16)).sum(axis=2)
    mask = diff <= (tol * 3)
    if int(mask.sum()) < min_pixels:
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0 = int(np.argmax(rows)); r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols)); c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return (r0, c0, r1, c1)


def _assign_nearest_bboxes(frame, name_colors: dict, play_rows: int, bg: set,
                           max_dist: int = 90, min_pixels: int = 3) -> dict:
    """Conflation-safe multi-color recovery: assign each NON-background
    playfield pixel to the NEAREST cached entity color (within max_dist),
    then return {name: bbox(px)} from each entity's assigned pixels.

    This avoids the single-color-threshold failure where SIMILAR colors
    (e.g. orange vs red, sum-diff ~101) both match a loose per-color
    threshold and merge — a red pixel is strictly closer to the red entity
    than the orange entity, so nearest-assignment separates them. Pixels
    farther than max_dist from every entity color (board/arm/other) are
    unassigned."""
    region = frame[:play_rows].astype(np.int16)
    names = list(name_colors.keys())
    if not names:
        return {}
    palette = np.array([name_colors[n] for n in names], dtype=np.int16)  # (K,3)
    # distance from every pixel to every entity color: (H,W,K)
    d = np.abs(region[:, :, None, :] - palette[None, None, :, :]).sum(axis=3)
    nearest = np.argmin(d, axis=2)             # (H,W) index into names
    nearest_dist = np.min(d, axis=2)           # (H,W)
    bg_arr = np.array(list(bg), dtype=np.int16) if bg else np.zeros((0, 3), np.int16)
    out = {}
    for ki, nm in enumerate(names):
        mask = (nearest == ki) & (nearest_dist <= max_dist)
        # exclude pixels that are actually background-colored
        if bg_arr.shape[0]:
            bgd = np.min(np.abs(region[:, :, None, :] - bg_arr[None, None, :, :]).sum(axis=3), axis=2)
            mask = mask & (nearest_dist < bgd)
        if int(mask.sum()) < min_pixels:
            continue
        rows = np.any(mask, axis=1); cols = np.any(mask, axis=0)
        r0 = int(np.argmax(rows)); r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
        c0 = int(np.argmax(cols)); c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
        out[nm] = (r0, c0, r1, c1)
    return out


def refresh_entities_from_frame(world, curr_frame_path,
                                start_frame_path=None,
                                play_rows_px: int = 52,
                                level=None) -> int:
    """Update movable tracked entities' current bbox from the live frame.

    For each entity whose role is refreshable: ensure its color is cached
    (bootstrap from the start frame, or the current frame if its current
    bbox still looks valid), then set its current bbox to the union of
    same-colored regions in the live frame, appended to bbox_history at
    world.turn. Returns the number of entities refreshed.

    Bboxes in bbox_history are in the SAME tick units the rest of the
    substrate uses (matching the perception bbox_ticks). Static entities are
    untouched. No-op (returns 0) if numpy/PIL unavailable or frame missing.
    """
    if not _OK:
        return 0
    frame = _load(curr_frame_path)
    if frame is None:
        return 0
    # Raw frames are 64x64 (1 px == 1 tick); bbox_history is stored in those
    # tick coords directly (no cell conversion). The producer converts
    # ticks->cells downstream.
    H = frame.shape[0]
    play_rows_px = min(play_rows_px, H)
    bg = _background_colors(frame, play_rows_px)
    gid = getattr(world, "game_id", "?")
    start_frame = _load(start_frame_path) if start_frame_path else None
    start_bg = _background_colors(start_frame, play_rows_px) if start_frame is not None else bg
    # Level change -> drop stale colors so each entity RE-bootstraps its color
    # for the new level. Bootstrap from the PRISTINE level-start frame
    # (start_frame_path), NOT the current frame: the first post-change refresh
    # usually runs AFTER the level's first action, by which point blocks have
    # already moved away from their placeholder bbox positions, so the current
    # frame would mis-bootstrap a moved entity (it was the bug that left
    # block_blue frozen while the un-moved block_red bootstrapped fine). The
    # level-start frame still has every entity at its placeholder position.
    # Fall back to the current frame only if no level-start frame is available.
    if level is not None and level != _LAST_LEVEL[0]:
        _COLOR_CACHE.clear()
        _LAST_LEVEL[0] = level
        if start_frame is None:
            start_frame = frame
            start_bg = bg

    ents = getattr(world, "entities", None) or {}
    values = list(ents.values() if hasattr(ents, "values") else ents)
    # 1) gather refreshable entities + bootstrap each entity's color.
    name_colors: dict = {}
    rec_by_name: dict = {}
    for rec in values:
        role = (getattr(rec, "current_role", None) or "").lower()
        if role not in REFRESHABLE_ROLES:
            continue
        name = getattr(rec, "name", None)
        if not name:
            continue
        key = (gid, name)
        color = _COLOR_CACHE.get(key)
        if color is None:
            bh = getattr(rec, "bbox_history", None) or []
            if start_frame is not None and bh:
                color = _dominant_color(start_frame, bh[0][1], start_bg)
            if color is None and bh:
                color = _dominant_color(frame, bh[-1][1], bg)
            if color is None:
                continue
            _COLOR_CACHE[key] = color
        name_colors[name] = color
        rec_by_name[name] = rec
    if not name_colors:
        return 0
    # 2) joint nearest-color assignment over the playfield (conflation-safe:
    #    separates similar colors like orange vs red).
    bboxes = _assign_nearest_bboxes(frame, name_colors, play_rows_px, bg)
    # 3) write each live bbox (tick coords; raw frame is 64x64, 1px==1tick),
    #    REPLACING the current turn's entry so current_bbox is the live one.
    turn = int(getattr(world, "turn", 0))
    refreshed = 0
    for name, px in bboxes.items():
        rec = rec_by_name[name]
        bbox_ticks = [int(px[0]), int(px[1]), int(px[2]), int(px[3])]
        try:
            bh = rec.bbox_history
            if bh and bh[-1][0] == turn:
                bh[-1] = (turn, bbox_ticks)
            else:
                bh.append((turn, bbox_ticks))
            refreshed += 1
        except Exception:
            pass
    return refreshed
