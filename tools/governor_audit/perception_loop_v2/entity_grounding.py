"""Ground perception-entity GEOMETRY in the substrate's MEASURED components.

WHY THIS EXISTS
---------------
The perception VLM names and groups entities well, but its hand-estimated
bboxes drift (commonly a few ticks too loose, especially on the trailing
edges) and it can miss a tiny mark or hallucinate a loose box around empty
space.  Every downstream geometry surface -- the entity-analysis view, the
movable-entity bbox refresh bootstrap, relational kinematics -- then inherits
that error.  Durable principle P3: perception geometry comes from the ACTUAL
frame, not from trusting the perception text.

WHAT IT DOES (game-agnostic, the substrate's quality-assurance pass on the
VLM's output)
-------------------------------------------------------------------------
Run the palette-invariant structural figure-ground extractor
(silhouette_track.foreground_components -- ground = large fields + repeated
texture tiles, dropped by STRUCTURE not colour) on the live frame, then SNAP
each VLM entity's bbox to the union of the measured component(s) it covers.
The VLM keeps ownership of SEMANTICS (name, role, grouping); the substrate
owns GEOMETRY (where each thing actually is, to the pixel).

  * An entity that covers one or more measured components -> its bbox becomes
    the tight union of those components.
  * An entity that covers NO measured component (a panel/field/HUD region the
    extractor treats as ground, or a loose box around nothing) -> kept as-is
    and flagged 'unmatched' so the caller can surface it.
  * A measured component that NO entity covers -> reported as 'missed': a real
    structure the VLM did not list, so the quality gap is visible rather than
    silently lost.

Pure numpy; degrades to a no-op (returns the entities unchanged) if numpy is
unavailable.  The bbox convention matches perception: [row_top, col_left,
row_bottom, col_right] with bottom/right EXCLUSIVE.
"""
from __future__ import annotations

from typing import Optional

try:
    import numpy as np
    import silhouette_track as _ST
    _OK = True
except Exception:                                       # pragma: no cover
    _OK = False


_BBOX_KEYS = ("bbox_ticks_turn1", "bbox_ticks", "bbox")


def _bbox_key(ent: dict) -> Optional[str]:
    for k in _BBOX_KEYS:
        v = ent.get(k)
        if isinstance(v, (list, tuple)) and len(v) == 4:
            return k
    return None


def _to_logical(frame, n_ticks: int):
    """Downsample a display frame to the logical (1 px == 1 tick) grid the
    perception bboxes live in.  A 64x64 raw frame (the common case) is already
    logical and returned unchanged."""
    H = frame.shape[0]
    if H == n_ticks or n_ticks <= 0 or H % n_ticks != 0:
        return frame
    scale = H // n_ticks
    off = scale // 2
    return frame[off::scale, off::scale][:n_ticks, :n_ticks]


def ground_entity_bboxes(entities: list[dict], frame_rgb, n_ticks: int = 64,
                         match_tol: int = 1) -> tuple[list[dict], dict]:
    """Snap each entity's bbox to the measured component(s) it covers.

    Returns (entities, report).  ``entities`` is a NEW list of shallow-copied
    dicts with the bbox field replaced by the snapped value where a match was
    found.  ``report`` = {"snapped": [names], "unmatched": [names],
    "missed": [(r0,c0,r1,c1 exclusive), ...]}.
    """
    report = {"snapped": [], "unmatched": [], "missed": []}
    if not _OK or frame_rgb is None or not entities:
        return entities, report

    arr = np.asarray(frame_rgb)[:, :, :3]
    logical = _to_logical(arr, n_ticks)
    comps = _ST.foreground_components(logical)            # inclusive tick bboxes
    # measured components as EXCLUSIVE bboxes + centroids
    measured = [((r0, c0, r1 + 1, c1 + 1), (cr, cc))
                for (r0, c0, r1, c1, cr, cc, npx, sub) in
                ((c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3],
                  c["centroid"][0], c["centroid"][1], c["npix"], c["mask"])
                 for c in comps)]

    # entity exclusive bboxes, by index
    ent_boxes = {}
    for i, e in enumerate(entities):
        k = _bbox_key(e)
        if k is None:
            continue
        r0, c0, r1, c1 = e[k]
        ent_boxes[i] = (k, (int(r0), int(c0), int(r1), int(c1)))

    def _contains(box, cr, cc, tol):
        r0, c0, r1, c1 = box
        return (r0 - tol) <= cr < (r1 + tol) and (c0 - tol) <= cc < (c1 + tol)

    def _center(box):
        r0, c0, r1, c1 = box
        return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

    # assign each measured component to the single entity that best covers it:
    # an entity whose (tolerance-expanded) bbox contains the component centroid;
    # nearest entity-centre breaks ties.  This keeps a loose box from claiming a
    # neighbour's component twice.
    assigned: dict[int, list] = {i: [] for i in ent_boxes}
    for ci, (mbox, (cr, cc)) in enumerate(measured):
        cands = [i for i, (_, box) in ent_boxes.items()
                 if _contains(box, cr, cc, match_tol)]
        if not cands:
            report["missed"].append(mbox)
            continue
        if len(cands) > 1:
            cands.sort(key=lambda i: (abs(_center(ent_boxes[i][1])[0] - cr)
                                      + abs(_center(ent_boxes[i][1])[1] - cc)))
        assigned[cands[0]].append(mbox)

    out = []
    matched_boxes = []                       # grounded boxes of matched entities
    for i, e in enumerate(entities):
        ne = dict(e)
        if i in ent_boxes and assigned.get(i):
            k = ent_boxes[i][0]
            boxes = assigned[i]
            r0 = min(b[0] for b in boxes); c0 = min(b[1] for b in boxes)
            r1 = max(b[2] for b in boxes); c1 = max(b[3] for b in boxes)
            ne[k] = [int(r0), int(c0), int(r1), int(c1)]
            report["snapped"].append(e.get("name", f"#{i}"))
            matched_boxes.append(((int(r0), int(c0), int(r1), int(c1)),
                                  (e.get("color") or "")))
        elif i in ent_boxes:
            report["unmatched"].append(e.get("name", f"#{i}"))
        out.append(ne)

    # QA on the UNMATCHED (ungroundable) boxes -- the boxes that drift / look off:
    #   * one whose CENTRE sits inside a MATCHED entity's grounded box is a
    #     spurious sub-feature or a loose box around part of a real entity (e.g. a
    #     thin strip on a disc's edge) -> DROP it; it is noise, not its own entity.
    #   * an isolated unmatched box (a real region the extractor treated as ground,
    #     or a hallucinated box around nothing) -> KEEP but flag geometry_unverified
    #     so the renderer can mark it and downstream knows the geometry is unproven.
    report["dropped_contained"] = []
    _unmatched_set = set(report["unmatched"])
    kept = []
    for i, ne in enumerate(out):
        nm = ne.get("name", f"#{i}")
        if i in ent_boxes and nm in _unmatched_set:
            k = ent_boxes[i][0]
            r0, c0, r1, c1 = ne[k]
            cr, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
            ecol = (ne.get("color") or "")
            # A contained box is a phantom UNLESS it is a known DIFFERENT colour
            # from the entity that contains it: a small distinct-colour piece (a
            # green square sitting inside a grey platform's bbox) is a real
            # foreground object, not a sub-feature.  Same-or-unknown colour +
            # contained = phantom (drop).  This is what kept the player invisible.
            def _is_phantom():
                for ((mr0, mc0, mr1, mc1), mcol) in matched_boxes:
                    if mr0 <= cr < mr1 and mc0 <= cc < mc1:
                        if ecol and mcol and ecol != mcol:
                            continue          # distinct-colour object -> keep
                        return True
                return False
            if _is_phantom():
                report["dropped_contained"].append(nm)
                report["unmatched"].remove(nm)
                continue                     # drop the phantom
            ne["geometry_unverified"] = True
        kept.append(ne)

    # OVERLAP-CONFLICT CHECK -- catches the wrong-component snap the tight-score is
    # BLIND to: distinct entities should not occupy the same pixels.  If two
    # entities' grounded bboxes overlap by more than a 1-tick abutment in BOTH
    # axes AND neither contains the other (containment = a field/panel holding its
    # contents, which is legitimate), at least one bbox is mis-snapped or
    # mis-localised.  Flagging this would have caught the cup snapping onto the
    # pink-stripe region.  No tuned threshold: the only constant is the 1-tick
    # abutment tolerance (entities may touch but not interpenetrate).
    boxed = []
    for ne in kept:
        k = _bbox_key(ne)
        if k is not None and ne.get(k) is not None:
            boxed.append((ne.get("name", "?"), [int(v) for v in ne[k]]))
    report["overlaps"] = []
    for ai in range(len(boxed)):
        for bi in range(ai + 1, len(boxed)):
            if _overlap_conflict(boxed[ai][1], boxed[bi][1]):
                report["overlaps"].append([boxed[ai][0], boxed[bi][0]])

    n = len(entities)
    snapped, unmatched, dropped, missed, overlaps = (len(report["snapped"]),
        len(report["unmatched"]), len(report["dropped_contained"]),
        len(report["missed"]), len(report["overlaps"]))
    denom = max(1, n - dropped)              # phantoms removed don't count against it
    report["quality"] = {"total": n, "snapped": snapped, "unmatched": unmatched,
                         "dropped": dropped, "missed": missed, "overlaps": overlaps,
                         "score": round(snapped / denom, 2)}
    return kept, report


def _overlap_conflict(a, b, tol: int = 1) -> bool:
    """True if bboxes a,b interpenetrate (overlap > tol ticks in BOTH axes) and
    neither contains the other.  Bottom/right are EXCLUSIVE.  tol = abutment
    tolerance: entities may touch/share a 1-tick border without conflict."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    oh = min(ar1, br1) - max(ar0, br0)       # overlap height
    ow = min(ac1, bc1) - max(ac0, bc0)       # overlap width
    if oh <= tol or ow <= tol:
        return False                          # no real 2D overlap (or just abut)
    a_in_b = br0 <= ar0 and bc0 <= ac0 and ar1 <= br1 and ac1 <= bc1
    b_in_a = ar0 <= br0 and ac0 <= bc0 and br1 <= ar1 and bc1 <= ac1
    return not (a_in_b or b_in_a)             # containment is legitimate


def quality_line(report: dict) -> str:
    """One-line, log-friendly summary of a grounding report's QA."""
    q = report.get("quality", {}) or {}
    parts = [f"grounding QA score={q.get('score', '?')} "
             f"({q.get('snapped', 0)}/{q.get('total', 0)} tight)"]
    if report.get("dropped_contained"):
        parts.append(f"dropped {len(report['dropped_contained'])} phantom-in-entity "
                     f"{report['dropped_contained']}")
    if report.get("unmatched"):
        parts.append(f"{len(report['unmatched'])} UNVERIFIED {report['unmatched']}")
    if report.get("missed"):
        parts.append(f"{len(report['missed'])} missed component(s)")
    if report.get("overlaps"):
        parts.append(f"{len(report['overlaps'])} OVERLAP-CONFLICT(s) "
                     f"{report['overlaps']}")
    return "; ".join(parts)
