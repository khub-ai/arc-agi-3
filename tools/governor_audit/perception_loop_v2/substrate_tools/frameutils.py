"""Shared, game-agnostic frame helpers for substrate tools.

These are pure pixel utilities every visual tool needs: load a frame into the
logical tick grid, clamp a bbox, hex a colour, find the background / dominant
colour, a font, connected-component counting.  No game knowledge, no state.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Sequence, Union

import numpy as np
from PIL import Image, ImageFont

FrameLike = Union[str, Path, np.ndarray, Image.Image]


def load_logical_frame(frame: FrameLike, n_ticks: int = 64) -> np.ndarray:
    """Return an (n_ticks, n_ticks, 3) uint8 array in TICK space.

    Accepts a path, a numpy array, or a PIL image.  If the source is larger
    (e.g. an already-upscaled render) it is resized with NEAREST so one logical
    tick maps to one array pixel — coordinates stay in [0, n_ticks].  ALWAYS
    take tool/act coordinates from this logical frame, never the upscaled
    render (the upscaled-frame measurement bug — see CONTRIBUTING).
    """
    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = Image.fromarray(arr.astype(np.uint8)[:, :, :3], "RGB")
    elif isinstance(frame, Image.Image):
        img = frame.convert("RGB")
    else:
        img = Image.open(frame).convert("RGB")
    if img.size != (n_ticks, n_ticks):
        # A RAW playfield render is a clean INTEGER upscale of the tick grid
        # (square; width an exact multiple of n_ticks).  A non-square or
        # non-integer ratio signals an ANNOTATED / margined render (axis
        # labels, a legend gutter) whose playfield is INSET — resizing the
        # whole image to the tick grid then mis-scales EVERY coordinate.  Warn
        # loudly (principled divisibility check, no magic threshold); the caller
        # should pass the raw frame.  (su15 lc2: a 1270x1270 labelled render —
        # 1270/64 = 19.84 — read the true tick (23,12) as (25,16), so clicks
        # missed and a whole shot budget burned to a loss before this was found.)
        w, h = img.size
        if w != h or w % n_ticks != 0:
            import warnings
            warnings.warn(
                f"load_logical_frame: source {w}x{h} is not a clean integer "
                f"multiple of n_ticks={n_ticks} (ratio {w / n_ticks:.2f}); this "
                f"looks like an ANNOTATED/margined render, not the raw "
                f"playfield — tick coordinates will be MIS-SCALED. Pass the raw "
                f"frame (e.g. the engine's frame_path), not the labelled image.",
                stacklevel=2)
        img = img.resize((n_ticks, n_ticks), Image.NEAREST)
    return np.asarray(img, dtype=np.uint8)


def clamp_bbox(bbox: Sequence[int], n_ticks: int) -> tuple[int, int, int, int]:
    """Normalise + clamp a [r0, c0, r1, c1] bbox (bottom/right exclusive).

    Tolerates swapped corners and guarantees r1>r0, c1>c0 within [0, n_ticks].
    """
    r0, c0, r1, c1 = (int(round(float(v))) for v in bbox[:4])
    r0, r1 = sorted((r0, r1))
    c0, c1 = sorted((c0, c1))
    r0 = max(0, min(n_ticks, r0))
    r1 = max(0, min(n_ticks, r1))
    c0 = max(0, min(n_ticks, c0))
    c1 = max(0, min(n_ticks, c1))
    if r1 <= r0:
        r1 = min(n_ticks, r0 + 1)
    if c1 <= c0:
        c1 = min(n_ticks, c0 + 1)
    return r0, c0, r1, c1


def hexof(rgb: Sequence[int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def background_rgb(frame: np.ndarray) -> tuple[int, int, int]:
    """Most-common colour over the whole frame (the canonical background)."""
    flat = frame.reshape(-1, 3)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    bg = colours[int(np.argmax(counts))]
    return int(bg[0]), int(bg[1]), int(bg[2])


def dominant_rgb(block: np.ndarray) -> tuple[int, int, int]:
    """Most-common colour in a pixel block (>=1 px)."""
    flat = block.reshape(-1, 3)
    if flat.shape[0] == 0:
        return (0, 0, 0)
    colours, counts = np.unique(flat, axis=0, return_counts=True)
    c = colours[int(np.argmax(counts))]
    return int(c[0]), int(c[1]), int(c[2])


def font(size: int = 11):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def connected_components(mask: np.ndarray) -> int:
    """Count 4-connected True components in a small boolean mask (BFS)."""
    if mask.size == 0:
        return 0
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    n = 0
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            n += 1
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            while q:
                r, c = q.popleft()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w and mask[nr, nc]
                            and not seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return n
