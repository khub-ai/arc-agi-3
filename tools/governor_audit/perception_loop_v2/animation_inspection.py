"""Substrate -> VLM detailed-inspection TRIGGER for animations.

WHY THIS EXISTS
---------------
A cheap, narrow read of an animation -- one colour's centroid at the endpoints -- silently hides the
things that actually matter: AREA changes (an entity growing/shrinking), demonstrations drawn in a
NON-primary colour (e.g. a grey preview silhouette), and mid-sequence transients that revert by the
settled frame.  That exact combination once made the acting VLM declare a grow/shrink preview "canned,
no scene effect".  Tracking a single colour's centroid must therefore NOT be the default.

THE BALANCE
-----------
The substrate does a cheap but EXHAUSTIVE pass -- every palette colour, every frame, the whole frame,
tracking AREA as well as centroid -- and then, only when it finds a change of a kind a lazy summary
tends to miss (size change, appearance/disappearance), it emits a DIRECTIVE telling the acting VLM to
LOOK at the raw frames and interpret the demonstration from the pixels.  Substrate does the exhaustive
measuring; the VLM is pulled in just for the flagged cases.  Past mistakes (``error_ledger``) raise the
sensitivity: if the VLM has previously misread animations/demonstrations, the trigger escalates.

DESIGN
------
- Pure + guarded: degrades to ``None`` / ``{}`` rather than raising.
- Threshold-free: kinds are decided by the SIGN of the area trajectory (monotone, or rise-then-fall)
  and by relative ranking of how much each colour changes -- no tuned numeric cutoffs (P: no magic
  thresholds).  The substrate MEASURES and FLAGS a candidate; the VLM looks and INTERPRETS.
"""
from __future__ import annotations

from typing import Optional

try:
    import numpy as np
    _OK = True
except Exception:  # pragma: no cover - environment without numpy
    _OK = False

from animation_analysis import _as_label_frame


def color_tracks(frame_stack) -> dict:
    """Per-colour ``(area, cy, cx)`` trajectory across the framestack -- EVERY colour, EVERY frame, the
    WHOLE frame.  ``None`` at frames where the colour is absent.  This is the exhaustive cheap
    measurement that a centroid-of-one-colour read skips."""
    if not _OK or not frame_stack:
        return {}
    frames = [_as_label_frame(g) for g in frame_stack]
    colours = set()
    for g in frames:
        colours.update(int(v) for v in np.unique(g))
    out = {}
    for c in colours:
        seq = []
        for g in frames:
            m = (g == c)
            if not m.any():
                seq.append(None)
            else:
                ys, xs = np.where(m)
                seq.append((int(m.sum()), float(ys.mean()), float(xs.mean())))
        out[c] = seq
    return out


def _area_kind(areas) -> Optional[str]:
    """Sign-based (threshold-free) classification of an area trajectory: a monotone rise/fall, or a
    rise-then-fall / fall-then-rise (a demonstration that reverts).  ``None`` if the area is flat or
    merely jitters with no consistent trend."""
    if len(areas) < 2 or len(set(areas)) == 1:
        return None
    pk = max(range(len(areas)), key=lambda i: areas[i])
    tr = min(range(len(areas)), key=lambda i: areas[i])
    nondec = all(areas[i] <= areas[i + 1] for i in range(len(areas) - 1))
    noninc = all(areas[i] >= areas[i + 1] for i in range(len(areas) - 1))
    if nondec and areas[-1] > areas[0]:
        return "grew"
    if noninc and areas[-1] < areas[0]:
        return "shrank"
    # rise then fall (peak in the middle) -- a grow demo that reverts to the start
    if (all(areas[i] <= areas[i + 1] for i in range(pk))
            and all(areas[i] >= areas[i + 1] for i in range(pk, len(areas) - 1))
            and areas[pk] > areas[0] and areas[pk] > areas[-1]):
        return "grew-then-reverted"
    # fall then rise -- a shrink demo that reverts
    if (all(areas[i] >= areas[i + 1] for i in range(tr))
            and all(areas[i] <= areas[i + 1] for i in range(tr, len(areas) - 1))
            and areas[tr] < areas[0] and areas[tr] < areas[-1]):
        return "shrank-then-reverted"
    return None


def classify_track(seq) -> set:
    """Tags for one colour's trajectory: ``appeared`` / ``disappeared`` / ``grew`` / ``shrank`` (+
    ``-then-reverted`` variants) / ``moved``.  Sign-based; no magic thresholds."""
    tags: set = set()
    pres = [s for s in seq if s]
    if not pres:
        return tags
    if seq[0] is None and seq[-1] is not None:
        tags.add("appeared")
    if seq[0] is not None and seq[-1] is None:
        tags.add("disappeared")
    ak = _area_kind([s[0] for s in pres])
    if ak:
        tags.add(ak)
    if seq[0] and seq[-1]:
        dr, dc = seq[-1][1] - seq[0][1], seq[-1][2] - seq[0][2]
        if (abs(dr) >= 1 or abs(dc) >= 1) and ak is None:
            tags.add("moved")
    return tags


_NOTABLE = {"grew", "shrank", "grew-then-reverted", "shrank-then-reverted",
            "appeared", "disappeared"}


def _change(seq) -> float:
    ch = 0.0
    for a, b in zip(seq, seq[1:]):
        ch += abs((b[0] if b else 0) - (a[0] if a else 0))
        if a and b:
            ch += abs(a[1] - b[1]) + abs(a[2] - b[2])
        elif (a is None) != (b is None):
            ch += 5.0
    return ch


def _ledger_escalation(error_ledger) -> Optional[str]:
    if error_ledger is None:
        return None
    try:
        areas = error_ledger.areas()
    except Exception:
        return None
    hot = sorted(k for k in areas
                 if any(t in k for t in ("animation", "demonstration", "preview", "frame_read")))
    if hot:
        return ("  NOTE: you have previously MISREAD animations/demonstrations (" + ", ".join(hot)
                + ") -- treat this as error-prone: zoom in, track AREA and NON-primary colours across "
                "ALL frames, and re-measure from the pixels rather than infer.")
    return None


def inspection_directive(frame_stack, error_ledger=None) -> Optional[str]:
    """Tell the acting VLM to LOOK at the raw frames in detail -- but ONLY when the exhaustive pass
    finds a change a lazy summary tends to miss: a colour that GROWS/SHRINKS (area), or APPEARS/leaves.
    A lone clean translation is well served by the motion summary and does NOT trigger this.  Returns
    ``None`` when nothing warrants a close look."""
    if not _OK or not frame_stack or len(frame_stack) <= 1:
        return None
    tracks = color_tracks(frame_stack)
    ranked = sorted(((c, s) for c, s in tracks.items() if _change(s) > 0),
                    key=lambda t: -_change(t[1]))
    flagged = []
    for c, seq in ranked:
        notable = classify_track(seq) & _NOTABLE
        if notable:
            flagged.append((c, sorted(notable), [s[0] if s else 0 for s in seq]))
    if not flagged:
        return None
    parts = ["[INSPECT] the substrate's exhaustive pass (every colour, every frame, whole frame) found "
             "a size/appearance change a single-colour centroid would miss -- LOOK at the raw frames "
             "and read the demonstration from the pixels; do NOT conclude from one colour or the "
             "start/settled frames alone:"]
    for c, tags, areas in flagged[:4]:
        parts.append(f"  - colour {c}: {', '.join(tags)} (area per frame {areas})")
    if len(flagged) >= 2:
        parts.append("  - MULTIPLE regions change size/presence -- inspect each, not just the "
                     "most-obvious mover.")
    esc = _ledger_escalation(error_ledger)
    if esc:
        parts.append(esc)
    return "\n".join(parts)
