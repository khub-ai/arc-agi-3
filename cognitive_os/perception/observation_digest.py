"""Build a plain-text observation digest for the VLM.

The digest is the substrate's behavioral summary of an exploration:
which actions were played, what each candidate did in response,
which palettes look like background, etc.  It accompanies the
annotated frame in the ID-grounded prompt scheme so the VLM has
both visual and behavioral grounding for every numbered entity.

Public entry: ``build_digest(parsed, log, actions)``.

The output is human-readable but written to be VLM-friendly --
short lines, candidate IDs reused throughout, no jargon.
"""
from __future__ import annotations

from typing import List, Mapping, Sequence

import numpy as np


def _palette_histogram(frame: np.ndarray) -> List[tuple]:
    """Return [(palette_int, count), ...] sorted by count desc."""
    vals, counts = np.unique(frame, return_counts=True)
    return sorted(((int(v), int(c)) for v, c in zip(vals, counts)),
                  key=lambda kv: -kv[1])


def _summary_for(ent: Mapping) -> str:
    """One-line behavior summary for an entity's properties dict."""
    pr = ent.get("properties") or {}
    disp = pr.get("total_displacement_px") or 0
    npal = pr.get("n_palette_changes", 0)
    npat = pr.get("n_pattern_changes", 0)
    autonomous = pr.get("is_autonomous", False)
    is_agent = pr.get("is_agent", False)
    bits = []
    if is_agent:
        bits.append("identified by tracker as agent (top non-autonomous mover)")
    if autonomous:
        bits.append("animates every step (autonomous; not user-controlled)")
    if disp > 0:
        bits.append(f"moved {disp:.0f}px during play")
    if npal or npat:
        bits.append(f"appearance changed {npal + npat} time(s)")
    if not bits:
        bits.append("static (no motion, no appearance change)")
    return "; ".join(bits)


def build_digest(
    parsed:        Mapping,
    log,
    actions:       Sequence[str],
    frame:         np.ndarray,
) -> str:
    """Build the text digest the VLM will see alongside the annotated frame.

    Parameters
    ----------
    parsed
        The substrate's parsed.json after ``apply_to_parsed`` -- contains
        ``entities`` with 1-based positional ids that match the labels
        rendered onto the annotated frame.
    log
        The ``InteractionLog`` for behavior signals.
    actions
        The sequence of action names that were played.
    frame
        The initial frame (used for palette histogram).
    """
    lines: List[str] = []
    lines.append("# Exploration observations")
    lines.append("")
    lines.append(f"Frame: {frame.shape[0]}x{frame.shape[1]} pixels")
    lines.append(f"Actions played ({len(actions)}): {', '.join(actions)}")
    lines.append("")

    # Per-candidate observations (id matches the annotated-frame labels).
    lines.append("# Candidates")
    lines.append("Each candidate has a numbered label drawn on the frame.")
    lines.append("")
    entities = parsed.get("entities") or []
    for i, ent in enumerate(entities, start=1):
        bbox = ent.get("bbox_pixels") or ent.get("bbox") or [0, 0, 0, 0]
        pals = ent.get("palettes") or []
        role = ent.get("role") or "unknown"
        summary = _summary_for(ent)
        lines.append(
            f"Candidate {i}: bbox={tuple(bbox)}, palettes={list(pals)}, "
            f"size_hint={ent.get('properties', {}).get('initial_size', '?')}px, "
            f"substrate_role_guess='{role}'. {summary}."
        )
    lines.append("")

    # Palette histogram (background-finding signal).
    lines.append("# Frame palette histogram")
    hist = _palette_histogram(frame)
    total = sum(c for _, c in hist)
    for pal, count in hist:
        pct = 100 * count / max(1, total)
        lines.append(f"Palette {pal:2d}: {count:5d} px ({pct:.1f}% of frame)")
    if hist:
        bg_guess = hist[0][0]
        lines.append("")
        lines.append(
            f"Substrate hypothesis: palette {bg_guess} is the background "
            f"({100 * hist[0][1] / max(1, total):.0f}% of frame, dominant)."
        )
    lines.append("")

    # Relationships between candidates the tracker found
    # (slide events, trigger correlations, etc.).
    rel_lines: List[str] = []
    for sl in getattr(log, "slide_events", []) or []:
        rel_lines.append(
            f"Slide: track at {sl['from_bbox']} slid to {sl['to_bbox']} "
            f"during step {sl['step']} ({sl['action']})."
        )
    for tc in getattr(log, "trigger_candidates", []) or []:
        rel_lines.append(
            f"Trigger correlation: at step {tc['step']} ({tc.get('functional_role','?')}), "
            f"agent at bbox {tc.get('agent_bbox')} -- another entity changed."
        )
    if rel_lines:
        lines.append("# Inter-candidate observations")
        lines.extend(rel_lines)
        lines.append("")

    # Repeated-instance groups: when the substrate detected a row /
    # cluster of similar-palette sub-components inside one parent
    # entity (handled in apply_to_parsed.detect_repeated_instances),
    # each child carries ``member_of_group = <parent_id>``.  Surface
    # those groups so the VLM sees "these 7 candidates are the
    # same kind of thing inside one parent" rather than reading
    # each in isolation as "small fragment of the wall".
    try:
        from collections import defaultdict
        by_group: dict = defaultdict(list)
        for i, e in enumerate(entities, start=1):
            grp = e.get("member_of_group")
            if grp and isinstance(grp, str):
                by_group[grp].append((i, e))
        groups_real = {k: v for k, v in by_group.items() if len(v) >= 3}
        if groups_real:
            lines.append("# Repeated-instance groups detected")
            lines.append(
                "The substrate found N >= 3 sub-components inside one "
                "parent entity that share the SAME SHAPE (palette-"
                "invariant silhouette) and similar size.  Each was "
                "emitted as its own top-level candidate.  Same-shape "
                "instances are typically collectibles, life icons, "
                "target markers, launchers, or any row of similar "
                "interactive elements -- not separate wall "
                "fragments.  The grouping is palette-blind, so even "
                "if each instance has different colours, they will "
                "still group together by silhouette.  Reason about "
                "each group AS A GROUP when assigning roles."
            )
            for parent_id, members in groups_real.items():
                ids = [m[0] for m in members]
                centroids = [
                    ((m[1].get("bbox_pixels") or [0]*4)[0]
                     + (m[1].get("bbox_pixels") or [0]*4)[2]) / 2
                    for m in members
                ]
                cols = [
                    ((m[1].get("bbox_pixels") or [0]*4)[1]
                     + (m[1].get("bbox_pixels") or [0]*4)[3]) / 2
                    for m in members
                ]
                # Detect the geometric arrangement.
                if max(centroids) - min(centroids) < 4:
                    arrangement = "horizontal row"
                elif max(cols) - min(cols) < 4:
                    arrangement = "vertical column"
                else:
                    arrangement = "scattered cluster"
                pal_sample = members[0][1].get("palettes") or []
                size_sample = (members[0][1].get("bbox_pixels") or [0]*4)
                bw = size_sample[3] - size_sample[1] + 1
                bh = size_sample[2] - size_sample[0] + 1
                lines.append(
                    f"\nGroup (parent='{parent_id}'): "
                    f"{len(members)} instances arranged as a {arrangement}.  "
                    f"Each ~{bw}x{bh} px, palette {pal_sample}.  "
                    f"Candidate IDs: {ids}."
                )
            lines.append("")
    except Exception:
        pass

    # Parallel-structure detection.  Slice the frame on plausible
    # regular grids and report any decomposition whose cells are
    # highly-similar in palette content.  Flags the outlier cell
    # (the one differing most from the rest) -- typically the
    # puzzle's editable instance amid identical references.  Includes
    # per-cell sub-structure (central vs peripheral sub-CCs) so the
    # VLM has the raw material to induce cross-instance rules
    # without us hardcoding what to compare.
    #
    # If the substrate has already emitted parallel-cells as entities
    # (via ``parallel_groups.emit_cells_as_entities``), we map each
    # cell to its 1-based entity id so the digest references match
    # the annotated frame and the prompt's valid-id list.
    try:
        from .parallel_groups import (
            detect_grid_parallels, summarise_for_digest,
        )
        bg = list(parsed.get("background_palettes") or [])
        # Fallback: if the VLM didn't declare a background, treat the
        # most-frequent palette as background.  Sub-CC extraction
        # needs SOME background to avoid merging every foreground
        # pixel into one giant component.
        if not bg and hist:
            bg = [hist[0][0]]
        groups = detect_grid_parallels(frame, bg)
        if groups:
            g0 = groups[0]
            # Find the entity-ids of the emitted parallel-cell
            # entities, keyed by their _parallel_group_idx.
            cell_ids: List = [None] * len(g0.cells)
            for i, ent in enumerate(entities, start=1):
                pidx = ent.get("_parallel_group_idx")
                if pidx is not None and 0 <= pidx < len(cell_ids):
                    cell_ids[pidx] = i
            block = summarise_for_digest(
                groups[:1], frame=frame, bg_palettes=bg,
                cell_ids=cell_ids,
            )
            if block:
                lines.append(block)
                lines.append("")
    except Exception:
        pass

    return "\n".join(lines)
