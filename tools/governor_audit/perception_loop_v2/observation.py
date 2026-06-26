"""Substrate-agnostic per-turn Observations.

The Observation is what the loop sees per turn.  Its fields are
derivable from raw pixels — no harness-specific encoding leaks
into the record:

  - RGB pixel patches per cell (the substrate output, but as colours
    a human-or-camera observer would also see).
  - Visual descriptors per cell (number of distinct colours, dominant
    colour, colour variance) — DERIVED from the patch, NOT looked up
    in a harness palette table.
  - Persistence per cell — has the cell stayed pixel-identical
    across the recent history?
  - Spatial pattern flags — frame-edge, frame-corner, bottom-edge
    strip overlap.
  - Agent position (provided by the harness as a (row, col) tuple,
    not a palette handle).
  - Agent delta — derived from prior turn's agent_position.

The detector / classifier read Observations only.  They never read
truth.json (the operator's answer key) and they never key on
palette indices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# -----------------------------------------------------------------------------
# Constants — the only "fixed" substrate facts the loop is allowed to know:
#   - the playable grid is rows x cols cells (a harness fact, like a chessboard
#     having 8x8 squares; it's geometry, not encoding).
#   - the full-frame raster is `frame_rgb` shape (H, W, 3); each cell is the
#     corresponding raster patch.
# Anything beyond that comes from observation.
# -----------------------------------------------------------------------------


@dataclass
class CellObservation:
    """Per-cell observation for one turn."""
    row: int
    col: int
    rgb_patch: np.ndarray  # shape (cell_h, cell_w, 3), dtype uint8

    # --- visual descriptors (derived from rgb_patch) ---
    n_distinct_colors: int                 # how many distinct colours in patch
    dominant_color: tuple[int, int, int]   # most common RGB
    mean_color: tuple[int, int, int]
    color_variance: float                  # mean over channels of std-dev

    # --- spatial pattern flags ---
    is_frame_corner: bool
    is_frame_edge: bool                    # on the outer ring of the grid
    is_top_edge_row: bool
    is_bottom_edge_row: bool

    # --- persistence (filled in by FrameObservation.build_persistence) ---
    is_static_recent: bool = False         # pixel-identical across recent frames
    same_as_prev: bool = False             # identical to previous turn's patch


@dataclass
class FrameObservation:
    """All observations for a single turn."""
    turn: int
    rgb_frame: np.ndarray                  # full frame, shape (H, W, 3)
    rows: int
    cols: int
    cell_h: int
    cell_w: int

    cells: list[list[CellObservation]]     # rows x cols

    agent_position: tuple[int, int] | None
    agent_delta: tuple[int, int] | None    # (drow, dcol) since prev turn

    # Per-frame derived global features (substrate-agnostic).
    # bottom_strip_rows: which raw-pixel rows form a thin horizontal strip
    # pinned to the bottom edge whose content differs from the background
    # of the rest of the frame.  Determined by visual-similarity scan.
    bottom_strip_rows: tuple[int, int] | None  # (y0_raw, y1_raw) inclusive
                                                 # or None if no such strip
                                                 # was detected.


# -----------------------------------------------------------------------------


def _patch_descriptors(patch: np.ndarray) -> dict:
    """Compute substrate-agnostic descriptors of an (h, w, 3) RGB patch."""
    flat = patch.reshape(-1, 3)
    # Distinct colours.
    unique = np.unique(flat, axis=0)
    n_distinct = int(unique.shape[0])
    # Dominant: tuple-quantise then mode.
    if flat.size:
        # Use np.unique with return_counts.
        keys, counts = np.unique(flat, axis=0, return_counts=True)
        dom_idx = int(counts.argmax())
        dominant = tuple(int(x) for x in keys[dom_idx])
        mean = tuple(int(round(float(v))) for v in flat.mean(axis=0))
    else:
        dominant = (0, 0, 0)
        mean = (0, 0, 0)
    color_variance = float(flat.std(axis=0).mean()) if flat.size else 0.0
    return {
        "n_distinct_colors": n_distinct,
        "dominant_color": dominant,
        "mean_color": mean,
        "color_variance": color_variance,
    }


def _detect_bottom_strip(
    rgb_frame: np.ndarray,
    cell_h: int,
) -> tuple[int, int] | None:
    """Detect a thin horizontal strip pinned to the bottom edge of the frame
    whose contents differ from the surrounding cell content.

    Heuristic (substrate-agnostic): scan upward from the last raw row.  A
    row belongs to "the strip" if its dominant colour differs from the
    interior dominant colour and the row's full-width content forms a
    narrow horizontal band that's pixel-stable across the frame.

    Returns (y0_raw, y1_raw) inclusive of the strip, or None.
    """
    H, W = rgb_frame.shape[:2]
    if H < 2 * cell_h:
        return None
    # Sample the median colour of rows in the bottom cell-row vs the
    # rows just above it.  If the bottom-most raw row is dominated by a
    # colour absent (or rare) in the row just above, mark it as strip.
    bottom_row = rgb_frame[H - 1, :, :]
    above_row = rgb_frame[H - cell_h, :, :]
    # Count colours in each.
    b_colors, b_counts = np.unique(bottom_row.reshape(-1, 3),
                                    axis=0, return_counts=True)
    a_colors, a_counts = np.unique(above_row.reshape(-1, 3),
                                    axis=0, return_counts=True)
    if not b_colors.size or not a_colors.size:
        return None
    b_dom = b_colors[int(b_counts.argmax())]
    a_dom = a_colors[int(a_counts.argmax())]
    # If the bottom row's dominant differs from the row above, treat the
    # bottom raw row as a strip.  Extend upward while dominant stays the
    # same.
    if tuple(b_dom) == tuple(a_dom):
        return None
    y1 = H - 1
    y0 = y1
    for y in range(y1, max(-1, H - cell_h - 1), -1):
        row = rgb_frame[y, :, :]
        cols, counts = np.unique(row.reshape(-1, 3), axis=0,
                                 return_counts=True)
        dom = cols[int(counts.argmax())]
        # Same dominant as the bottom row -> still in strip.
        if tuple(dom) == tuple(b_dom):
            y0 = y
        else:
            break
    return (int(y0), int(y1))


def build_frame_observation(
    rgb_frame: np.ndarray,
    *,
    turn: int,
    rows: int,
    cols: int,
    agent_position: tuple[int, int] | None,
    prev_observation: "FrameObservation | None" = None,
) -> FrameObservation:
    """Build a FrameObservation from an RGB frame and minimal harness
    facts (grid shape + agent_position).  prev_observation is optional;
    persistence flags are filled in when it is supplied."""
    H, W = rgb_frame.shape[:2]
    cell_h = H // rows
    cell_w = W // cols
    bottom_strip = _detect_bottom_strip(rgb_frame, cell_h)
    grid: list[list[CellObservation]] = []
    for r in range(rows):
        row: list[CellObservation] = []
        for c in range(cols):
            y0 = r * cell_h
            y1 = (r + 1) * cell_h
            x0 = c * cell_w
            x1 = (c + 1) * cell_w
            patch = rgb_frame[y0:y1, x0:x1, :]
            d = _patch_descriptors(patch)
            same_as_prev = False
            if prev_observation is not None:
                prev_patch = prev_observation.cells[r][c].rgb_patch
                same_as_prev = bool(
                    prev_patch.shape == patch.shape
                    and np.array_equal(prev_patch, patch)
                )
            row.append(CellObservation(
                row=r, col=c,
                rgb_patch=patch,
                n_distinct_colors=d["n_distinct_colors"],
                dominant_color=d["dominant_color"],
                mean_color=d["mean_color"],
                color_variance=d["color_variance"],
                is_frame_corner=(
                    (r == 0 or r == rows - 1)
                    and (c == 0 or c == cols - 1)
                ),
                is_frame_edge=(
                    r == 0 or r == rows - 1
                    or c == 0 or c == cols - 1
                ),
                is_top_edge_row=(r == 0),
                is_bottom_edge_row=(r == rows - 1),
                same_as_prev=same_as_prev,
            ))
        grid.append(row)

    agent_delta: tuple[int, int] | None = None
    if (prev_observation is not None
            and prev_observation.agent_position is not None
            and agent_position is not None):
        prev_r, prev_c = prev_observation.agent_position
        agent_delta = (
            agent_position[0] - prev_r,
            agent_position[1] - prev_c,
        )

    return FrameObservation(
        turn=turn,
        rgb_frame=rgb_frame,
        rows=rows, cols=cols,
        cell_h=cell_h, cell_w=cell_w,
        cells=grid,
        agent_position=agent_position,
        agent_delta=agent_delta,
        bottom_strip_rows=bottom_strip,
    )


def attach_persistence(
    obs_history: list[FrameObservation],
    static_window: int = 3,
) -> None:
    """Mark cells in the LAST observation as `is_static_recent` if their
    rgb_patch has been pixel-identical for the last `static_window`
    frames (including the last)."""
    if not obs_history:
        return
    last = obs_history[-1]
    window = obs_history[-static_window:]
    for r in range(last.rows):
        for c in range(last.cols):
            patches = [o.cells[r][c].rgb_patch for o in window
                       if r < o.rows and c < o.cols]
            if len(patches) < static_window:
                last.cells[r][c].is_static_recent = False
                continue
            ref = patches[0]
            is_static = all(
                p.shape == ref.shape and np.array_equal(p, ref)
                for p in patches
            )
            last.cells[r][c].is_static_recent = bool(is_static)


# -----------------------------------------------------------------------------
# Convenience: load a fixture turn's frame + agent_position.
# -----------------------------------------------------------------------------


def load_fixture_frame(
    frame_path: Path,
) -> np.ndarray:
    """Read frame.png as RGB ndarray, shape (H, W, 3)."""
    img = Image.open(frame_path).convert("RGB")
    return np.array(img)
