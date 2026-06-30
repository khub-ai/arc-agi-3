"""Target-pattern computation -- step 1 of the TARGET-PATTERN PURSUIT (the edit-to-target
executor for configuration / "apply the legend" games, e.g. tr87).

GAME-AGNOSTIC and deterministic.  Given:
  - a LEGEND: example (input -> output) pairs (the rule), as (input_bbox, output_bbox),
  - a QUERY sequence: the input cells to translate, as bboxes,
  - the frame,
it returns, for each query, the TARGET output shape -- by INVARIANT-matching the query to a
legend INPUT key (rotation/scale/colour-invariant via shape_identity) and taking that pair's
OUTPUT.  This turns "the answer is the legend-translation of the input" into a computed table,
the thing a human reads off the legend by eye.

No colour/shape/size keys, no game ids: identity is `shape_identity` (figure-ground silhouette,
8-dihedral orientations, scale-normalised).  The caller supplies which bboxes are legend pairs
vs queries (from the structure-mapping context); this module only does the lookup.
"""
from __future__ import annotations

from typing import List, Dict, Optional, Tuple

try:
    import shape_identity as _si
except ImportError:                                  # imported as a package
    from perception_loop_v2 import shape_identity as _si


def compute_targets(legend_pairs: List[Tuple[list, list]],
                    queries: List[list],
                    frame_rgb,
                    min_score: float = 0.0) -> List[Dict]:
    """For each query bbox, the legend OUTPUT whose INPUT best matches it (invariant).

    legend_pairs: [(input_bbox, output_bbox), ...]  -- inclusive [r0,c0,r1,c1] tick bboxes.
    queries:      [query_bbox, ...]
    Returns one dict per query: {query, target, legend_idx, score} -- target is the matched
    pair's output_bbox (None if no legend / below min_score).  Deterministic; guarded.
    """
    out: List[Dict] = []
    try:
        in_sigs = [_si.shape_signature(frame_rgb, p[0]) for p in legend_pairs]
    except Exception:
        in_sigs = []
    for q in queries:
        best_i, best_s = -1, -1.0
        try:
            qsig = _si.shape_signature(frame_rgb, q)
            for i, isig in enumerate(in_sigs):
                if not isig:
                    continue
                s = _si.similarity(qsig, isig)
                if s > best_s:
                    best_i, best_s = i, s
        except Exception:
            pass
        tgt = legend_pairs[best_i][1] if (best_i >= 0 and best_s >= min_score) else None
        out.append({"query": q, "target": tgt,
                    "legend_idx": (best_i if tgt is not None else -1),
                    "score": round(max(best_s, 0.0), 3)})
    return out


def render_targets(targets: List[Dict], query_names: Optional[List[str]] = None,
                   slot_names: Optional[List[str]] = None) -> str:
    """One-line-per-slot summary of the computed target pattern, for surfacing to the actor:
    'slot <name> -> set to the shape currently at <legend output bbox> (matched legend pair
    #i, conf s)'.  Names optional; falls back to bboxes."""
    lines = []
    for k, t in enumerate(targets):
        if t.get("target") is None:
            continue
        slot = (slot_names[k] if slot_names and k < len(slot_names) else f"slot{k}")
        q = (query_names[k] if query_names and k < len(query_names) else str(t["query"]))
        lines.append(f"  {slot}: make it match the legend output at {t['target']} "
                     f"(its input {q} matches legend pair #{t['legend_idx']}, conf {t['score']})")
    if not lines:
        return ""
    return ("TARGET PATTERN (legend-translation of the input row -- set each editable slot to "
            "the matching legend output shape):\n" + "\n".join(lines))
