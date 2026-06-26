"""Continuous control-law induction (game-agnostic).

WHY
---
For a CONTINUOUS-CONTROL game (an aiming/launch game like su15, a cannon, a
slingshot, a steering game), the actor must learn the CONTROL LAW: how an
action's continuous parameter (e.g. a click's position relative to the
controllable mover) maps to the mover's response (does it launch? in which
direction? how far?). Flailing — changing several things at once and eyeballing —
does NOT converge; the actor burns its action budget. The discipline that works
is the same one a scientist uses: ONE-FACTOR-AT-A-TIME probing with measured
read-back, bisecting to pin a threshold, until the law is fit.

This module is that discipline, made mechanical:
  - probes are recorded in POLAR form relative to the mover (distance d, angle θ
    of the click) with the MEASURED outcome (did it move, and the displacement
    magnitude + angle, read via locate_entity);
  - propose_probe() picks the next OFAT probe — fixing the angle and BISECTING the
    distance to pin the launch threshold(s) — so each shot maximally reduces
    uncertainty;
  - induce_law() fits the law: trigger distance window, direction (toward vs away
    / slingshot), and magnitude (fixed range vs to-the-click vs proportional);
  - predict_step() turns the induced law into the exact click to advance the
    mover toward a target.

Pure geometry/stats; no frame, no colour, no game knowledge — the caller supplies
measured probe outcomes (e.g. from locate_entity) and gets back the next probe and
the fitted law. This is what lets an unfamiliar continuous controller be solved
FROM OBSERVATION, instead of by reading the game's source.
"""
from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import List, Optional, Sequence

MOVE_EPS = 1.5          # displacement (ticks) below this counts as NO launch


def angle_of(drow: float, dcol: float) -> float:
    """Direction of a (row, col) delta in degrees, atan2(-drow, dcol):
    0 = +col (right), 90 = -row (up). Consistent for clicks AND displacements."""
    return math.degrees(math.atan2(-float(drow), float(dcol)))


def _circ_diff(a: float, b: float) -> float:
    """Smallest signed difference a-b folded to (-180, 180]."""
    d = (a - b + 180.0) % 360.0 - 180.0
    return d if d != -180.0 else 180.0


def make_probe(mover, click, after) -> dict:
    """Build a probe record from absolute positions: the mover BEFORE, the
    clicked point, and the mover AFTER (all [row, col]). Polar relative to the
    mover before."""
    mr, mc = float(mover[0]), float(mover[1])
    cr, cc = float(click[0]), float(click[1])
    ar, ac = float(after[0]), float(after[1])
    d = math.hypot(cr - mr, cc - mc)
    th = angle_of(cr - mr, cc - mc)
    dmag = math.hypot(ar - mr, ac - mc)
    dth = angle_of(ar - mr, ac - mc) if dmag >= MOVE_EPS else None
    return {"dist": round(d, 2), "angle": round(th, 1), "moved": dmag >= MOVE_EPS,
            "disp_mag": round(dmag, 2), "disp_angle": (round(dth, 1) if dth is not None else None)}


# -----------------------------------------------------------------------------
# Induction
# -----------------------------------------------------------------------------


def induce_law(records: Sequence[dict]) -> dict:
    """Fit the control law from probe records. Returns a dict with `confidence`
    and the fields it could determine (others None)."""
    recs = list(records)
    launched = [r for r in recs if r["moved"]]
    noop = [r for r in recs if not r["moved"]]
    law: dict = {"n_probes": len(recs), "n_launched": len(launched),
                 "trigger": None, "direction": None, "magnitude": None,
                 "confidence": "low"}
    if not launched:
        law["note"] = ("no launch observed yet — probe a LARGER distance "
                       "(the trigger threshold is above everything tried)")
        return law

    # --- trigger: distance window where launches occur ---
    ld = sorted(r["dist"] for r in launched)
    nd = sorted(r["dist"] for r in noop)
    lo_min, lo_max = ld[0], ld[-1]
    # min threshold: largest no-op strictly below the launch band
    below = [d for d in nd if d < lo_min]
    d_min = (max(below) + lo_min) / 2.0 if below else 0.0
    # max threshold: smallest no-op strictly above the launch band
    above = [d for d in nd if d > lo_max]
    d_max = (min(above) + lo_max) / 2.0 if above else None
    law["trigger"] = {"d_min": round(d_min, 1), "d_max": (round(d_max, 1) if d_max else None),
                      "launch_band": [round(lo_min, 1), round(lo_max, 1)]}

    # --- direction: toward the click vs away (slingshot) ---
    diffs = [_circ_diff(r["disp_angle"], r["angle"]) for r in launched if r["disp_angle"] is not None]
    if diffs:
        md = mean(abs(x) for x in diffs)
        law["direction"] = "toward" if md < 45 else ("away" if md > 135 else "perpendicular")
        law["_dir_resid"] = round(md, 1)

    # --- magnitude: fixed range vs to-the-click vs proportional ---
    mags = [r["disp_mag"] for r in launched]
    dists = [r["dist"] for r in launched]
    if len(mags) >= 1:
        mu = mean(mags)
        cv = (pstdev(mags) / mu) if (len(mags) > 1 and mu > 0) else 0.0
        to_click = mean(abs(m - d) for m, d in zip(mags, dists)) < 0.25 * mu
        if to_click and (len(mags) == 1 or cv > 0.15):
            law["magnitude"] = {"kind": "to_click"}
        elif cv <= 0.2:
            law["magnitude"] = {"kind": "fixed", "range": round(mu, 1)}
        else:
            # linear fit mag = k*dist through the data
            num = sum(m * d for m, d in zip(mags, dists)); den = sum(d * d for d in dists)
            k = num / den if den else 0.0
            law["magnitude"] = {"kind": "proportional", "slope": round(k, 3)}

    # --- confidence ---
    # The MIN threshold is only PINNED when an actual no-op sits below the launch
    # band (a real lower bracket). d_min defaulting to 0 because nothing small was
    # tried is a GUESS, not knowledge -> do NOT treat it as bracketed.
    real_lower = bool(below)
    have_dir = law["direction"] is not None
    have_mag = law["magnitude"] is not None
    n, nl = len(recs), len(launched)
    if nl >= 2 and have_dir and have_mag and real_lower and n >= 4:
        law["confidence"] = "high"
    elif nl >= 2 and have_dir and have_mag and (real_lower or n >= 5):
        law["confidence"] = "medium"
    else:
        law["confidence"] = "low"
    return law


# -----------------------------------------------------------------------------
# Experiment design — the next probe
# -----------------------------------------------------------------------------


def propose_probe(records: Sequence[dict], *, prefer_angle: Optional[float] = None,
                  dmax_search: float = 70.0, tol: float = 4.0) -> Optional[dict]:
    """The next OFAT probe (distance, angle) that most reduces uncertainty.
    Fixes the angle and bisects the distance to pin the launch threshold. Returns
    None when the law is sufficiently determined."""
    recs = list(records)
    launched = [r for r in recs if r["moved"]]
    noop = [r for r in recs if not r["moved"]]
    # fix the probe ANGLE (one factor at a time)
    if prefer_angle is not None:
        ang = float(prefer_angle)
    elif launched:
        ang = launched[0]["angle"]
    elif recs:
        ang = recs[0]["angle"]
    else:
        ang = 0.0

    if not recs:
        return {"dist": 25.0, "angle": ang, "why": "first probe — a moderate distance"}
    if not launched:
        nd = max(r["dist"] for r in recs)
        nxt = min(nd * 1.6 + 4.0, dmax_search)
        if nxt <= nd + 1:
            return None        # exhausted the search range without a launch
        return {"dist": round(nxt, 1), "angle": ang,
                "why": "no launch yet — probe farther to find the min threshold"}

    ld = sorted(r["dist"] for r in launched)
    nd = sorted(r["dist"] for r in noop)
    lo_min, lo_max = ld[0], ld[-1]
    below = max((d for d in nd if d < lo_min), default=None)
    above = min((d for d in nd if d > lo_max), default=None)
    # 1) pin d_min: bisect between the highest no-op-below and the lowest launch
    if below is None:
        target = max(0.5, lo_min - max(6.0, lo_min * 0.4))
        return {"dist": round(target, 1), "angle": ang,
                "why": "bisect downward to find the MIN launch distance"}
    if lo_min - below > tol:
        return {"dist": round((below + lo_min) / 2.0, 1), "angle": ang,
                "why": "bisect to pin the MIN launch distance"}
    # 2) probe for a possible d_max (does it stop launching when very far?)
    if above is None and lo_max < dmax_search - 1:
        return {"dist": round(min(lo_max + max(10.0, lo_max * 0.5), dmax_search), 1),
                "angle": ang, "why": "probe farther to check for a MAX launch distance"}
    if above is not None and above - lo_max > tol:
        return {"dist": round((lo_max + above) / 2.0, 1), "angle": ang,
                "why": "bisect to pin the MAX launch distance"}
    # 3) trigger pinned — verify direction/magnitude at a second angle if needed
    angles = {round(r["angle"] / 30.0) for r in launched}
    if len(angles) < 2:
        return {"dist": round((lo_min + lo_max) / 2.0 if lo_max > lo_min else lo_min + 5, 1),
                "angle": ang + 90.0, "why": "confirm direction/magnitude at a second angle"}
    return None                # law determined


def predict_step(law: dict, mover, target, *, min_clear: float = 2.0) -> Optional[list]:
    """Given the induced law, the click [row,col] that advances the mover toward
    `target` by one step. Returns None if the law isn't determined enough."""
    if not law or not law.get("trigger") or law.get("direction") not in ("toward", "away"):
        return None
    mr, mc = float(mover[0]), float(mover[1])
    tr, tc = float(target[0]), float(target[1])
    dr, dc = tr - mr, tc - mc
    norm = math.hypot(dr, dc) or 1.0
    ur, uc = dr / norm, dc / norm
    if law["direction"] == "away":          # slingshot: click OPPOSITE the target
        ur, uc = -ur, -uc
    d_min = law["trigger"].get("d_min", 0.0) or 0.0
    d_max = law["trigger"].get("d_max")
    d = max(d_min + min_clear + 2.0, d_min * 1.15)
    if d_max:
        d = min(d, d_max - min_clear)
    return [round(mr + ur * d, 1), round(mc + uc * d, 1)]


# -----------------------------------------------------------------------------
# ANISOTROPIC / target-directed controllers
# -----------------------------------------------------------------------------
#
# Not every launcher is isotropic (same in every direction). Some are
# TARGET-DIRECTED: the mover only fires toward a fixed GOAL, and only when the
# click is AT or BEYOND the goal in that direction (you "click past the target to
# fire at it" — a common cannon/aiming pattern). A click in the wrong direction,
# or short of the goal, does nothing. For such a controller the right experiment
# is NOT a distance sweep at one angle but: click PAST the goal and confirm the
# mover steps toward the goal; the aim is then to keep clicking past the goal.


def classify_anisotropy(records: Sequence[dict], target) -> dict:
    """Decide whether launches point at a fixed TARGET (target-directed) or at the
    CLICK (click-directed/isotropic). Each record carries the click + the measured
    displacement; `target` is the goal [row,col]. Returns {kind, confidence, note}.
    The discriminator: does the displacement track the click angle or the
    mover->target angle? When the two agree (a launch happens to point at the
    goal) it is inconclusive — but if launches ONLY occur toward the goal and
    never elsewhere, that itself is the target-directed signature."""
    launched = [r for r in records if r.get("moved")]
    if not launched:
        return {"kind": "unknown", "confidence": "low", "note": "no launch yet"}
    tr, tc = float(target[0]), float(target[1])
    toward_target = 0; toward_click = 0; ambiguous = 0
    for r in launched:
        if r.get("disp_angle") is None or "mover_at" not in r:
            ambiguous += 1; continue
        mr, mc = r["mover_at"]
        ta = angle_of(tr - mr, tc - mc)
        dt = abs(_circ_diff(r["disp_angle"], ta))          # displacement vs target dir
        dc = abs(_circ_diff(r["disp_angle"], r["angle"]))  # displacement vs click dir
        if dt < 25 and dc - dt > 20:
            toward_target += 1
        elif dc < 25 and dt - dc > 20:
            toward_click += 1
        else:
            ambiguous += 1
    if toward_target and toward_target >= toward_click:
        return {"kind": "target_directed", "confidence": "medium" if toward_target >= 2 else "low",
                "note": "launches point at the goal regardless of click angle"}
    if toward_click and toward_click > toward_target:
        return {"kind": "click_directed", "confidence": "medium",
                "note": "launches point at the click (isotropic)"}
    # all launches happened to point at the goal AND the click (same direction):
    # the safe read for a launcher whose ONLY launches are goal-ward is target-directed.
    return {"kind": "target_directed", "confidence": "low",
            "note": "every launch so far was goal-ward; treat as target-directed until refuted"}


def aim_past_target(mover, target, *, clearance: float = 8.0, row_floor: float = 11.0,
                    row_ceil: float = 61.0, col_floor: float = 1.0,
                    col_ceil: float = 62.0) -> list:
    """For a TARGET-DIRECTED launcher: the click just PAST the goal in the
    mover->goal direction (so the click distance exceeds the mover-to-goal
    distance -> it fires), clamped to the open playfield. Each such click advances
    the mover one step toward the goal; repeat to walk it in and OVERLAP it."""
    mr, mc = float(mover[0]), float(mover[1])
    tr, tc = float(target[0]), float(target[1])
    dr, dc = tr - mr, tc - mc
    n = math.hypot(dr, dc) or 1.0
    ur, uc = dr / n, dc / n
    r = max(row_floor, min(row_ceil, tr + ur * clearance))
    c = max(col_floor, min(col_ceil, tc + uc * clearance))
    return [round(r, 1), round(c, 1)]


def make_probe_t(mover, click, after) -> dict:
    """make_probe, but stamps the mover-before position into the record so
    classify_anisotropy can compare displacement to the target direction."""
    p = make_probe(mover, click, after)
    p["mover_at"] = [round(float(mover[0]), 1), round(float(mover[1]), 1)]
    return p


def format_surface(records: Sequence[dict], law: Optional[dict] = None) -> str:
    law = law or induce_law(records)
    lines = [f"CONTROL-LAW INDUCTION ({law['n_probes']} probes, "
             f"{law['n_launched']} launched) — confidence {law['confidence']}:"]
    if law.get("trigger"):
        t = law["trigger"]
        rng = f">= {t['d_min']}" + (f" and <= {t['d_max']}" if t['d_max'] else "")
        lines.append(f"  - LAUNCHES when click distance from the mover is {rng} ticks")
    if law.get("direction"):
        lines.append(f"  - mover travels {law['direction'].upper()} the click")
    if law.get("magnitude"):
        m = law["magnitude"]
        desc = ({"fixed": f"a FIXED range (~{m.get('range')} ticks)",
                 "to_click": "all the way TO the click",
                 "proportional": f"~{m.get('slope')}x the click distance"}).get(m["kind"], m["kind"])
        lines.append(f"  - travel magnitude: {desc}")
    nxt = propose_probe(records)
    if nxt is not None:
        lines.append(f"  NEXT PROBE: click at distance {nxt['dist']} ticks, angle "
                     f"{nxt['angle']} deg from the mover ({nxt['why']}).")
    else:
        lines.append("  law DETERMINED — use predict_step() to aim.")
    return "\n".join(lines)
