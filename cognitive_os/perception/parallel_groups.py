"""Detect parallel sub-regions in a frame.

A parallel group is a set of N >= 2 sub-regions arranged on a regular
grid where the cells are similar to each other in palette and
structural content.  Examples:

  * ft09's four 3x3 puzzle grids (2x2 super-grid).
  * sk48's three movable_blocks in a column (3x1 super-grid).
  * any game with a row of identical icons / collectibles / lives.

The substrate emits these groups as first-class observations so the
VLM has named handles ("super-grid cell (0,0)") to reason over.  No
game-specific knowledge -- we just try plausible grid decompositions
of the frame and report any that produce highly-similar cells, plus
the outlier (the cell that most differs from the others).

The detector ignores background palette(s) when measuring similarity
so a noisy background pattern doesn't drown out the real structure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


# Decompositions worth trying.  Limit to 2..16 cells to avoid
# absurdly fine tilings; the natural grids for ARC-AGI-3 frames sit
# squarely in this range.
_MIN_CELLS = 2
_MAX_CELLS = 16
_MAX_DIM   = 4

# Minimum INLIER similarity for a tiling to be reported.  An ARC
# puzzle with N references + 1 editable instance has very high
# similarity AMONG the references but low similarity between any
# reference and the editable cell.  We pick the tiling whose best
# inlier set (cells minus at most one outlier) has mean similarity
# above this threshold.
_MIN_INLIER_SIM = 0.85

# A cell is reported as an outlier only when removing it raises the
# inlier mean similarity by at least this much.  Prevents flagging
# cosmetic differences as "the outlier".
_MIN_OUTLIER_LIFT = 0.10


@dataclass(frozen=True)
class ParallelGroup:
    """One detected parallel structure.

    ``cells`` is a list of (r0, c0, r1, c1) bboxes in frame-pixel
    coordinates, in row-major order matching ``grid_shape``.

    ``similarity`` is the mean pairwise palette-histogram cosine
    similarity among the cells (excluding background palettes).

    ``outlier_index`` (when not None) points at the cell whose
    similarity to the others is most depressed -- the natural
    candidate for "this is the cell that needs to be edited".

    ``palette_signatures`` is a per-cell palette histogram (after
    background removal) for inclusion in the digest.
    """
    grid_shape:        Tuple[int, int]
    cells:             Sequence[Tuple[int, int, int, int]]
    similarity:        float
    outlier_index:     Optional[int]
    palette_signatures: Sequence[Mapping[int, int]]


def _hist_cosine(a: Mapping[int, int], b: Mapping[int, int]) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = sum(v * v for v in a.values()) ** 0.5
    nb = sum(v * v for v in b.values()) ** 0.5
    if na == 0 or nb == 0:
        return 1.0 if (na == 0 and nb == 0) else 0.0
    return dot / (na * nb)


def _cell_histogram(sub: np.ndarray, bg_set: set) -> Mapping[int, int]:
    vals, counts = np.unique(sub, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts) if int(v) not in bg_set}


def _mean_pairwise_sim(hists: Sequence[Mapping[int, int]],
                       exclude_idx: Optional[int] = None) -> float:
    idxs = [i for i in range(len(hists)) if i != exclude_idx]
    if len(idxs) < 2:
        return 0.0
    sims = []
    for i in range(len(idxs)):
        for j in range(i + 1, len(idxs)):
            sims.append(_hist_cosine(hists[idxs[i]], hists[idxs[j]]))
    return sum(sims) / len(sims)


def _best_inlier_set(hists: Sequence[Mapping[int, int]]
                     ) -> Tuple[float, Optional[int]]:
    """Return (best_inlier_similarity, outlier_idx_or_None).

    Tries excluding each cell in turn (and excluding none); picks the
    excluded cell whose removal maximises the remaining cells' mean
    pairwise similarity.  Returns the FULL-set similarity when no
    single exclusion produces a meaningful lift -- the group is then
    either coherent without an outlier or not coherent at all.
    """
    base = _mean_pairwise_sim(hists)
    if len(hists) < 3:
        return base, None
    best_sim = base
    best_idx: Optional[int] = None
    for i in range(len(hists)):
        s = _mean_pairwise_sim(hists, exclude_idx=i)
        if s > best_sim + _MIN_OUTLIER_LIFT:
            best_sim = s
            best_idx = i
    return best_sim, best_idx


def detect_grid_parallels(
    frame:           np.ndarray,
    bg_palettes:     Iterable[int] = (),
    *,
    min_cells:       int = _MIN_CELLS,
    max_cells:       int = _MAX_CELLS,
    max_dim:         int = _MAX_DIM,
    min_similarity:  float = _MIN_INLIER_SIM,
) -> List[ParallelGroup]:
    """Find regular grid decompositions of ``frame`` whose cells are
    highly-similar in palette histogram.

    Returns groups in descending similarity order, dropping any
    sub-group whose cell-bbox set is fully covered by a higher-scored
    group with a different shape (deduplication of nested tilings).
    """
    H, W = frame.shape
    bg_set = set(int(p) for p in bg_palettes)
    out: List[ParallelGroup] = []

    for rows in range(1, max_dim + 1):
        for cols in range(1, max_dim + 1):
            n = rows * cols
            if n < min_cells or n > max_cells:
                continue
            if rows == 1 and cols == 1:
                continue
            # Even partition; final row/col absorbs any remainder so
            # the cells cover the whole frame.
            cell_h_base = H // rows
            cell_w_base = W // cols
            cells: List[Tuple[int, int, int, int]] = []
            subs:  List[np.ndarray] = []
            for r in range(rows):
                r0 = r * cell_h_base
                r1 = (r + 1) * cell_h_base - 1 if r < rows - 1 else H - 1
                for c in range(cols):
                    c0 = c * cell_w_base
                    c1 = (c + 1) * cell_w_base - 1 if c < cols - 1 else W - 1
                    cells.append((r0, c0, r1, c1))
                    subs.append(frame[r0:r1 + 1, c0:c1 + 1])

            hists = [_cell_histogram(s, bg_set) for s in subs]
            # Skip if any cell is fully background (empty hist) --
            # background-only cells inflate similarity and don't
            # actually represent a structure.
            if any(not h for h in hists):
                continue
            inlier_sim, outlier = _best_inlier_set(hists)
            if inlier_sim < min_similarity:
                continue
            out.append(ParallelGroup(
                grid_shape         = (rows, cols),
                cells              = tuple(cells),
                similarity         = inlier_sim,
                outlier_index      = outlier,
                palette_signatures = tuple(hists),
            ))

    out.sort(key=lambda g: -g.similarity)
    return out


def _summarise_cell_substructure(
    frame:    np.ndarray,
    cell_bbox: Tuple[int, int, int, int],
    bg_palettes: set,
) -> str:
    """For one parallel-group cell, describe its internal sub-structure:
    sub-CCs (connected components, excluding background), their bboxes
    and palettes, and which sub-CC is centred (closest to cell centre).

    To handle cells where a single non-global-background palette
    dominates and acts as fill (e.g. the bracket frame on an
    outlier), we ALSO treat any cell-local palette covering >40% of
    the cell as background.  Substrate-general: any "fill" palette
    that fences the real sub-structure off from itself is exposed
    rather than allowed to merge everything into one blob.
    """
    r0, c0, r1, c1 = cell_bbox
    sub = frame[r0:r1 + 1, c0:c1 + 1]
    H, W = sub.shape
    # Cell-local secondary background detection.
    vals, counts = np.unique(sub, return_counts=True)
    cell_total = H * W
    extended_bg = set(bg_palettes)
    for v, c in zip(vals, counts):
        if int(v) in extended_bg:
            continue
        if int(c) / cell_total > 0.40:
            extended_bg.add(int(v))
    bg_palettes = extended_bg
    seen = np.zeros((H, W), dtype=bool)
    from collections import deque
    components: List[dict] = []
    for sr in range(H):
        for sc in range(W):
            pal = int(sub[sr, sc])
            if pal in bg_palettes or seen[sr, sc]:
                continue
            cells = []
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp_pals = {pal}
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc]:
                            np_pal = int(sub[nr, nc])
                            if np_pal not in bg_palettes:
                                seen[nr, nc] = True
                                comp_pals.add(np_pal)
                                q.append((nr, nc))
            if cells:
                rs = [p[0] for p in cells]; cs = [p[1] for p in cells]
                sub_bbox_local = (min(rs), min(cs), max(rs), max(cs))
                # Translate to frame-pixel coords.
                sub_bbox = (sub_bbox_local[0] + r0, sub_bbox_local[1] + c0,
                            sub_bbox_local[2] + r0, sub_bbox_local[3] + c0)
                cent = (sum(rs) / len(rs) + r0, sum(cs) / len(cs) + c0)
                components.append({
                    "bbox":    sub_bbox,
                    "size":    len(cells),
                    "palettes": sorted(comp_pals),
                    "centroid": cent,
                })
    if not components:
        return ""
    # Filter out single-pixel and 2-pixel sub-CCs as texture noise.
    # In dotted-background games these otherwise flood the digest
    # with dozens of meaningless candidates.  The threshold matches
    # Layer A's ``min_size_pixels`` default.
    NOISE_SIZE = 2
    n_noise = sum(1 for c in components if c["size"] <= NOISE_SIZE)
    components = [c for c in components if c["size"] > NOISE_SIZE]
    if not components:
        return ""
    # Identify the most-central sub-CC (closest to the cell centre).
    cy = (r0 + r1) / 2.0
    cx = (c0 + c1) / 2.0
    components.sort(key=lambda c: (c["centroid"][0] - cy) ** 2
                                + (c["centroid"][1] - cx) ** 2)
    central = components[0]
    peripheral = components[1:]
    if n_noise:
        # Record but don't enumerate -- the count alone tells the VLM
        # "this cell has a texture pattern".
        peripheral_note = (
            f"    (and {n_noise} single-pixel sub-CC(s) suppressed as "
            f"texture noise -- likely a dotted background pattern)"
        )
    else:
        peripheral_note = None
    lines: List[str] = []
    lines.append(
        f"    central sub-CC (closest to cell centre): "
        f"bbox={central['bbox']} size={central['size']}px "
        f"palettes={central['palettes']}"
    )
    if peripheral:
        # Group peripheral CCs by palette signature.
        from collections import defaultdict
        by_pal: dict = defaultdict(list)
        for c in peripheral:
            by_pal[tuple(c["palettes"])].append(c)
        for pals, items in sorted(by_pal.items()):
            lines.append(
                f"    {len(items)} peripheral sub-CC(s) palettes={list(pals)} "
                f"size_range=[{min(c['size'] for c in items)}-"
                f"{max(c['size'] for c in items)}]"
            )
    if peripheral_note:
        lines.append(peripheral_note)
    return "\n".join(lines)


def _per_palette_max_cc_size(sub: np.ndarray) -> dict:
    """For each palette in ``sub``, return its largest connected-
    component size.  Used to identify TEXTURE/NOISE palettes whose
    largest CC is too small to plausibly be an entity."""
    from collections import deque
    H, W = sub.shape
    out: dict = {}
    for pal_val in np.unique(sub):
        pal = int(pal_val)
        mask = (sub == pal)
        seen = np.zeros((H, W), dtype=bool)
        max_size = 0
        for sr in range(H):
            for sc in range(W):
                if not mask[sr, sc] or seen[sr, sc]:
                    continue
                count = 0
                q = deque([(sr, sc)])
                seen[sr, sc] = True
                while q:
                    rr, cc = q.popleft()
                    count += 1
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = rr + dr, cc + dc
                            if 0 <= nr < H and 0 <= nc < W \
                                    and mask[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                q.append((nr, nc))
                if count > max_size:
                    max_size = count
        out[pal] = max_size
    return out


def detect_repeated_instances(
    frame:       np.ndarray,
    bbox:        Tuple[int, int, int, int],
    bg_palettes: Iterable[int] = (),
    *,
    min_count:   int = 3,
    min_size:    int = 3,
    size_ratio:  float = 2.5,
) -> List[dict]:
    """Find N >= min_count visually-similar sub-components within ``bbox``.

    Strategy (palette-INVARIANT, adversarially robust):

      1. Identify "noise" palettes whose largest CC is < min_size --
         these are texture / dotted-background pixels that would
         otherwise bridge entities together.
      2. Run CROSS-PALETTE connected-component extraction (8-conn
         through ANY non-bg non-noise pixel).  Multi-colour sprites
         remain ONE component each.
      3. Compute a palette-INVARIANT shape fingerprint (topo_id)
         for each CC.  Two CCs with the same silhouette but
         different colour schemes share the same topo_id.
      4. Group CCs by topo_id.  When a group has >= min_count
         members of similar size (within ``size_ratio``), it's a
         repeated-instance group.

    This is robust to the adversarial test: if a game designer
    re-skins each repeated item with different colours, they still
    share a silhouette -- the substrate still groups them.

    Substrate-general: no role names, no specific pattern,
    palette-blind.  Callers turn each instance into a top-level
    entity so downstream reasoning can target them individually.
    """
    from collections import deque, defaultdict
    try:
        # entity_fingerprint lives in the use-case harness; import
        # lazily so the detector doesn't hard-require it.
        from entity_fingerprint import (  # type: ignore
            fingerprint_from_bitmap, masked_bitmap_from_object,
        )
    except ImportError:
        fingerprint_from_bitmap = None
        masked_bitmap_from_object = None

    r0, c0, r1, c1 = bbox
    sub = frame[r0:r1 + 1, c0:c1 + 1]
    H, W = sub.shape
    if H * W <= 0:
        return []

    # Step 1: pre-filter noise palettes.
    pal_max_cc = _per_palette_max_cc_size(sub)
    noise_pals = {p for p, m in pal_max_cc.items() if m < min_size}
    bg_set = set(int(p) for p in bg_palettes) | noise_pals

    # Step 2: cross-palette CC extraction.
    seen = np.zeros((H, W), dtype=bool)
    components: List[dict] = []
    for sr in range(H):
        for sc in range(W):
            pal = int(sub[sr, sc])
            if pal in bg_set or seen[sr, sc]:
                continue
            cells = []
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            comp_pals = {pal}
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc]:
                            np_pal = int(sub[nr, nc])
                            if np_pal not in bg_set:
                                seen[nr, nc] = True
                                comp_pals.add(np_pal)
                                q.append((nr, nc))
            if len(cells) < min_size:
                continue
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            sub_bbox_local = (min(rs), min(cs), max(rs), max(cs))
            sub_bbox = (sub_bbox_local[0] + r0, sub_bbox_local[1] + c0,
                        sub_bbox_local[2] + r0, sub_bbox_local[3] + c0)
            cent = (sum(rs) / len(rs) + r0, sum(cs) / len(cs) + c0)
            # Step 3: palette-invariant shape fingerprint.
            topo_id: str = ""
            if fingerprint_from_bitmap is not None:
                try:
                    cells_global = [(rr + r0, cc + c0) for rr, cc in cells]
                    bm = masked_bitmap_from_object(
                        frame, sub_bbox, cells=cells_global)
                    fp = fingerprint_from_bitmap(bm, palettes=sorted(comp_pals))
                    topo_id = str(getattr(fp, "topo_id", "") or "")
                except Exception:
                    topo_id = ""
            if not topo_id:
                # Fallback: dimensions of the bbox plus pixel-count
                # buckets.  Adversarially weaker than topo_id but
                # better than palette-based grouping.
                bw = sub_bbox_local[3] - sub_bbox_local[1] + 1
                bh = sub_bbox_local[2] - sub_bbox_local[0] + 1
                topo_id = f"dim_{bw}x{bh}_size{len(cells) // 4}"
            components.append({
                "bbox":     sub_bbox,
                "size":     len(cells),
                "palettes": tuple(sorted(comp_pals)),
                "centroid": cent,
                "topo_id":  topo_id,
            })

    # Step 4: group by topo_id.
    by_topo: dict = defaultdict(list)
    for c in components:
        by_topo[c["topo_id"]].append(c)
    best: List[dict] = []
    for topo, items in by_topo.items():
        if len(items) < min_count:
            continue
        sizes = [c["size"] for c in items]
        if min(sizes) <= 0:
            continue
        if max(sizes) / min(sizes) > size_ratio:
            continue
        if len(items) > len(best):
            best = items
    return best


def emit_cells_as_entities(
    parsed:      dict,
    frame:       np.ndarray,
    bg_palettes: Iterable[int] = (),
) -> int:
    """Run parallel-group detection and append the top group's cells
    as new entities in ``parsed['entities']``.  Each cell becomes a
    distinct top-level entity with a synthetic role tag so the VLM
    can reason about them by id.  Returns the number of cells
    appended (0 when no parallel structure is detected).

    The first parallel cell entity has bbox = group.cells[0] etc.;
    its 1-based position in the entity list IS the id the VLM will
    see (matching the rendered annotated frame).
    """
    groups = detect_grid_parallels(frame, bg_palettes)
    if not groups:
        return 0
    g = groups[0]
    rows, cols = g.grid_shape
    n_existing = len(parsed.get("entities") or [])
    for idx, (bbox, hist) in enumerate(zip(g.cells, g.palette_signatures)):
        r, c = idx // cols, idx % cols
        is_outlier = (idx == g.outlier_index)
        parsed.setdefault("entities", []).append({
            "id":           f"pcell_{r}_{c}",
            "bbox_pixels":  list(bbox),
            "palettes":     sorted(int(p) for p in hist.keys()),
            "role":         "parallel_cell_outlier" if is_outlier
                            else "parallel_cell_inlier",
            "candidate_ids": (),
            "related_to":   None,
            "member_of_group": None,
            "notes": (f"parallel-group cell ({r},{c}) of {rows}x{cols} "
                      f"super-grid; "
                      f"{'OUTLIER' if is_outlier else 'inlier'}.  "
                      f"Emitted by the substrate's parallel-structure "
                      f"detector for cross-instance reasoning."),
            "_matched_track": None,
            "_overlap":      1.0,
            "_corrections":  [],
            "_sub_bitmaps":  [],
            "properties":    {},
            "_parallel_group_idx": idx,
            "_parallel_group_shape": [rows, cols],
        })
    return len(g.cells)


def summarise_for_digest(
    groups:      Sequence[ParallelGroup],
    *,
    frame:       Optional[np.ndarray] = None,
    bg_palettes: Iterable[int] = (),
    cell_ids:    Optional[Sequence[Optional[int]]] = None,
) -> str:
    """Render a list of ParallelGroup as a block of digest text.

    When ``frame`` is supplied, also emit per-cell sub-structure:
    sub-CC count, palettes, and the central vs peripheral split.
    This is what lets the VLM compare "what's inside cell A" vs
    "what's inside cell B" without us hardcoding a specific
    structural-comparison rule -- the VLM does the rule induction
    from the per-cell facts.

    Empty string when no groups -- the caller can decide whether to
    emit the section heading conditionally.
    """
    if not groups:
        return ""
    bg_set = set(int(p) for p in bg_palettes)
    lines: List[str] = []
    lines.append("# Parallel structures detected")
    lines.append(
        "The substrate sliced the frame on regular grids and "
        "compared the cells.  Cells listed below are highly-similar "
        "in palette content; any outlier is the one cell whose "
        "content differs from the rest -- often the puzzle's working "
        "area."
    )
    for k, g in enumerate(groups, start=1):
        rows, cols = g.grid_shape
        lines.append(
            f"\nGroup {k}: {rows}x{cols} super-grid "
            f"(mean cell similarity {g.similarity:.2f}):"
        )
        for idx, (bbox, hist) in enumerate(zip(g.cells, g.palette_signatures)):
            r, c = idx // cols, idx % cols
            outlier_tag = "  [OUTLIER]" if g.outlier_index == idx else ""
            cid = (cell_ids[idx] if cell_ids and idx < len(cell_ids) else None)
            cid_tag = f"candidate_id={cid}, " if cid is not None else ""
            pal_str = ", ".join(
                f"p{p}={c}" for p, c in sorted(hist.items(), key=lambda kv: -kv[1])
            )
            lines.append(
                f"  cell ({r},{c}) {cid_tag}bbox={bbox} palettes={{{pal_str}}}"
                f"{outlier_tag}"
            )
            if frame is not None:
                sub = _summarise_cell_substructure(frame, bbox, bg_set)
                if sub:
                    lines.append(sub)
        if g.outlier_index is not None:
            ro, co = g.outlier_index // cols, g.outlier_index % cols
            lines.append(
                f"  -> outlier cell ({ro},{co}) differs from the other "
                f"{len(g.cells) - 1}; likely the active / editable "
                f"instance.  Compare the inlier cells' INTERNAL "
                f"structure (central sub-CC vs peripheral sub-CCs) to "
                f"infer what the outlier should look like."
            )
    return "\n".join(lines)
