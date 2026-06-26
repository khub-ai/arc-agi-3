"""Layer A — deterministic geometry extraction.

Reads a frame PNG, applies background-palette filtering, groups
non-background pixels into candidate entities by 8-connectivity
across palettes, and emits a numbered list of candidates.

The VLM (Layer B) consumes the candidate list and assigns roles
to each — it does NOT invent positions.  This separation lets
geometry stay accurate (pixel_elements + connectivity) while
semantics stay in the VLM's strong suit (visual + textual reasoning).

Each candidate carries:
* a stable integer ``candidate_id`` (1-based, presented in the prompt)
* the union bbox of all member components
* the union palette set
* total pixel count
* optionally the centroid cell when the game has a cell_system
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

# The harness's pixel_elements module is under usecases/arc-agi-3/python.
_HARNESS_PY = (Path(__file__).resolve().parents[2]
               / "usecases" / "arc-agi-3" / "python")
if str(_HARNESS_PY) not in sys.path:
    sys.path.insert(0, str(_HARNESS_PY))


def _hex_to_rgb(h: str) -> tuple:
    return (int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16))


@dataclass
class GeometryCandidate:
    candidate_id:  int
    bbox_pixels:   tuple                # (r0, c0, r1, c1) inclusive
    palettes:      tuple
    size_pixels:   int
    centroid:      tuple                # (r, c) pixels
    centroid_cell: Optional[tuple] = None   # (cr, cc) when cell_system supplied
    cells_occupied: Optional[tuple] = None  # set of (cr, cc) when cell_system supplied


@dataclass
class GeometryResult:
    frame_size:     tuple
    background_palettes: tuple
    candidates:     List[GeometryCandidate] = field(default_factory=list)
    n_noise_filtered: int = 0


def _frame_to_palette_array(frame_path: Path) -> np.ndarray:
    """Load PNG, downsample to 64x64, map RGB pixels to palette indices."""
    from arc_agi.rendering import COLOR_MAP   # type: ignore
    img = Image.open(frame_path).convert("RGB").resize((64, 64), Image.NEAREST)
    arr = np.array(img)
    rgb_to_idx = {_hex_to_rgb(h): int(i) for i, h in COLOR_MAP.items()}
    frame = np.zeros((64, 64), dtype=np.int32)
    for r in range(64):
        for c in range(64):
            rgb = tuple(int(x) for x in arr[r, c])
            if rgb in rgb_to_idx:
                frame[r, c] = rgb_to_idx[rgb]
    return frame


def extract_geometry(
    frame_path:           Path,
    *,
    background_palettes:  Sequence[int],
    cell_system:          Optional[Mapping[str, Any]] = None,
    min_size_pixels:      int = 2,
) -> GeometryResult:
    """Run Layer A on a frame and return a numbered candidate list.

    ``background_palettes`` is the set of palettes the operator (or
    the prior VLM pass) considers non-entity background.  These are
    excluded from candidate extraction.

    ``cell_system`` — when supplied, each candidate's centroid is
    also translated to cell coordinates for the prompt.

    ``min_size_pixels`` — components smaller than this are noise
    (PNG-conversion artefacts); they're counted in
    ``n_noise_filtered`` but not emitted as candidates.
    """
    import pixel_elements   # type: ignore
    frame = _frame_to_palette_array(frame_path)
    bg_set = set(int(p) for p in background_palettes)

    # Mask out background-palette pixels BEFORE running connected
    # components so the bg doesn't bridge foreground entities into
    # one giant object_id group via diagonal adjacency.
    masked = frame.copy()
    BG_SENTINEL = -1
    for bg_p in bg_set:
        masked[frame == bg_p] = BG_SENTINEL

    # 4-connectivity: only N/S/E/W neighbors merge, so tiles that
    # are visually distinct (separated by even a single pixel of
    # gap) stay as separate components.  This matches the operator
    # intuition that bp35's 7 green launcher tiles should be 7
    # entities, not 2 row-groups.  Earlier code used 8 to keep
    # ls20's shape_changer (diagonal-pixel sprite) merged; that
    # specific case needs a per-entity merge step in Layer B / the
    # role catalog rather than treating ALL diagonal touching as a
    # merge signal.
    comps = pixel_elements.extract_components(
        masked,
        min_size           = 1,           # keep singletons for noise count
        exclude_background = True,        # excludes BG_SENTINEL
        connectivity       = 4,
        return_cells       = True,
    )

    # Drop any component whose palette is -1 (defensive — shouldn't happen).
    fg_comps = [c for c in comps if int(c["palette"]) != BG_SENTINEL]
    # Group by object_id (spatial-adjacency-across-palettes).
    by_obj: dict = {}
    for c in fg_comps:
        by_obj.setdefault(int(c["object_id"]), []).append(c)

    # Build one GeometryCandidate per object_id group.
    candidates: List[GeometryCandidate] = []
    n_noise = 0
    next_id = 1
    o_r = c_offset = cell = None
    if cell_system:
        try:
            o_r, o_c = cell_system.get("origin", (0, 0))
            cell = int(cell_system.get("cell_size", 0)) or None
        except (TypeError, ValueError):
            cell = None
    half = (cell // 2) if cell else 0

    def _emit(all_pixels: list, palettes_override: Optional[tuple] = None) -> None:
        """Emit one candidate from a list of (r, c, palette) tuples."""
        nonlocal next_id, n_noise
        if not all_pixels:
            return
        if len(all_pixels) < min_size_pixels:
            n_noise += 1
            return
        rs = [p[0] for p in all_pixels]
        cs = [p[1] for p in all_pixels]
        cent_r = sum(rs) // len(rs)
        cent_c = sum(cs) // len(cs)
        bbox = (min(rs), min(cs), max(rs), max(cs))
        pals = (palettes_override if palettes_override is not None
                else tuple(sorted({p for _, _, p in all_pixels})))
        centroid_cell = None
        cells_occupied = None
        if cell:
            centroid_cell = ((cent_r - o_r) // cell,
                             (cent_c - (o_c - half)) // cell)
            cells_occupied = tuple(sorted({
                ((pr - o_r) // cell,
                 (pc - (o_c - half)) // cell)
                for pr, pc, _ in all_pixels
            }))
        candidates.append(GeometryCandidate(
            candidate_id   = next_id,
            bbox_pixels    = bbox,
            palettes       = pals,
            size_pixels    = len(all_pixels),
            centroid       = (cent_r, cent_c),
            centroid_cell  = centroid_cell,
            cells_occupied = cells_occupied,
        ))
        next_id += 1

    # Over-merge guard: object_id grouping uses 8-connectivity across
    # palettes, which can chain visually-distinct entities (the sk48
    # striped_column + piercer_head + piercer_tail blob, the r11l bar)
    # into one giant candidate when they happen to touch at the edge.
    # Detect this via the union bbox fill_ratio: a cohesive multi-color
    # sprite fills most of its bbox; an over-merged blob is sparse.
    # When sparse AND many sub-components, emit each per-palette
    # component as its own candidate and let the VLM regroup.
    OVER_MERGE_FILL_RATIO = 0.5
    OVER_MERGE_MIN_COMPS  = 3

    for oid, group in by_obj.items():
        # Aggregate pixels per per-palette component up-front.
        per_comp_pixels: List[list] = []
        for cmp in group:
            pts = [(int(pr), int(pc), int(cmp["palette"]))
                   for pr, pc in (cmp.get("cells") or [])]
            if pts:
                per_comp_pixels.append(pts)
        if not per_comp_pixels:
            continue
        all_pixels = [p for comp in per_comp_pixels for p in comp]
        rs = [p[0] for p in all_pixels]
        cs = [p[1] for p in all_pixels]
        bbox_area = max(1, (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1))
        fill_ratio = len(all_pixels) / bbox_area
        if (fill_ratio < OVER_MERGE_FILL_RATIO
                and len(per_comp_pixels) >= OVER_MERGE_MIN_COMPS):
            # Sparse blob — emit each per-palette component separately.
            for comp_pixels in per_comp_pixels:
                _emit(comp_pixels)
        else:
            _emit(all_pixels)

    return GeometryResult(
        frame_size          = (64, 64),
        background_palettes = tuple(sorted(bg_set)),
        candidates          = candidates,
        n_noise_filtered    = n_noise,
    )


def format_candidates_for_prompt(result: GeometryResult) -> str:
    """Render the candidate list as compact text for the VLM prompt."""
    lines: List[str] = [f"Layer A extracted {len(result.candidates)} candidate "
                        f"entities (background palettes = "
                        f"{list(result.background_palettes)}; "
                        f"{result.n_noise_filtered} single-pixel candidates "
                        f"filtered as noise)."]
    lines.append("Each candidate is a connected-component group "
                 "(8-connectivity across palette boundaries).")
    lines.append("\n# Layer A candidates")
    lines.append("id | bbox (r0,c0,r1,c1) | palettes | size_pixels | centroid_cell")
    for c in result.candidates:
        cc = f"({c.centroid_cell[0]},{c.centroid_cell[1]})" if c.centroid_cell else "—"
        lines.append(
            f"{c.candidate_id:3d} | "
            f"({c.bbox_pixels[0]},{c.bbox_pixels[1]},{c.bbox_pixels[2]},{c.bbox_pixels[3]}) | "
            f"{list(c.palettes)} | {c.size_pixels} | {cc}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    # Smoke test on ls20_4of7.
    sample = (Path(__file__).resolve().parents[2]
              / "tests" / "perception_samples" / "ls20_4of7")
    import json
    with open(sample / "ground_truth.json") as f:
        gt = json.load(f)
    result = extract_geometry(
        sample / "frame.png",
        background_palettes = gt.get("background_palettes", []),
        cell_system         = gt.get("cell_system"),
    )
    print(format_candidates_for_prompt(result))
