"""Routing forward-model (prototype) — predict where a mobile element routes
through a PERCEIVED static field WITHOUT spending a real action.

This generalizes ``maze_model``'s contact relation.  The maze model learns a
BINARY outcome per region-class (a class either STOPS the mover or is PASSABLE)
and rolls the mover forward applying it.  Routing needs a slightly richer, still
game-agnostic, contact vocabulary:

    PASS   — the frontier continues through the static region
    STOP   — the frontier terminates at the region
    SPLIT  — the frontier spawns two child frontiers, one just past each EDGE of
             the region (the edge-offset is the observed branch rule)

Rolling a source frontier forward with these outcomes predicts which target cells
it reaches for an UNTRIED configuration — e.g. "if I place the deflectors here,
does the flow reach all three cups?" — so the in-loop VLM can plan inside a tight
action budget instead of discovering by trial.

Game-agnostic by construction: nothing here names liquid / bar / cup.  A
"frontier" is the mobile element's leading edge (a contiguous span perpendicular
to its travel); a "deflector" is any static region; the per-region OUTCOME and
the SPLIT edge-offset are INDUCED from the animation frontier-trajectory
perception (``_substrate_animation_summary``) — here they are inputs so the
simulator is testable in isolation.  The identical model predicts a bouncing
ball, a reflecting beam, a spreading effect, or flowing water.

SCOPE / KNOWN GAP (grounded in controlled observation).  The SPLIT rule —
children just past each bar EDGE (edge +/- 4) — is ROBUST for a source hitting a
bar: a sweep over bar widths {12, 20} with the nozzle striking at every offset
from one edge to the other gives edge-anchored children EVERY time, independent of
where the stream hits.  So the rule is correct for a source->bar contact, and the
clean, well-separated staircase cascade is predicted correctly end-to-end (induce
once -> predict A,B,C -> matches the real win).

The cross-config misses (2 of 4) are NOT the edge rule being wrong and NOT reach
(reach was ruled out: the field is ~55 rows; a column already traverses it).  They
occur in a FLOODING regime: certain close / wide geometries make the flow spread
broadly (observed ~700px of liquid vs ~450px for clean branching) and fill cups by
POOLING rather than as discrete branches — which this discrete model does not
represent.  Flooding usually fills MORE cups (often itself a win), but can be
partial (a false positive).  So the predictor is reliable for clean cascades —
which winning plans like the staircase are — and its errors cluster in the
flooding regime.  Detecting / modelling that regime (or restricting the planner to
clean-branching configs and flagging close-bar ones as uncertain) is the open
problem, not a fixed-offset fix.
"""
from __future__ import annotations
from dataclasses import dataclass

PASS, STOP, SPLIT = "pass", "stop", "split"


@dataclass(frozen=True)
class Deflector:
    """A static region the frontier may contact.  ``along`` is its coordinate on
    the travel axis (the side the frontier meets first); ``lo``/``hi`` is its
    extent on the perpendicular (span) axis."""
    along: int
    lo: int
    hi: int


def predict_routing(source_span, deflectors, *, travel="up", span_len=64,
                    outcome=lambda d: SPLIT, edge_width=4,
                    max_streams=1024):
    """Roll a source frontier forward through ``deflectors`` and return the list
    of (lo, hi) frontier spans that REACH THE FAR BOUNDARY (the target side).

    ``source_span``  = (lo, hi) of the source frontier on the span axis.
    ``deflectors``   = iterable of Deflector (along on the travel axis).
    ``travel``       = 'up'/'down'/'left'/'right' — direction of travel; only the
                       sign along the travel axis matters, so the same code
                       serves all four (the caller maps rows/cols to along/span).
    ``outcome(d)``   = PASS | STOP | SPLIT for a contacted deflector (induced).
    ``edge_width``   = width of each child frontier spawned past an edge on SPLIT.

    A stream is (lo, hi, frm) where ``frm`` is its current position on the travel
    axis; it advances toward the boundary until it meets the NEAREST deflector
    whose span overlaps [lo,hi].  Deterministic; no game knowledge."""
    # going "up"/"left" the along-coordinate DECREASES toward the boundary;
    # "down"/"right" it increases.  Normalise to "decreasing toward 0" by sign.
    decreasing = travel in ("up", "left")
    start = span_len * 4                       # a position strictly beyond any deflector
    streams = [(source_span[0], source_span[1], start if decreasing else -start)]
    reached, guard = [], 0
    while streams:
        guard += 1
        if guard > max_streams:
            break
        lo, hi, frm = streams.pop()
        # deflectors strictly ahead of frm whose span overlaps [lo,hi]
        ahead = [d for d in deflectors
                 if (d.along < frm if decreasing else d.along > frm)
                 and not (d.hi < lo or d.lo > hi)]
        if not ahead:
            reached.append((lo, hi))           # frontier exits at the far boundary
            continue
        # the FIRST one met = nearest ahead
        hit = (max(ahead, key=lambda d: d.along) if decreasing
               else min(ahead, key=lambda d: d.along))
        oc = outcome(hit)
        if oc == STOP:
            continue
        if oc == PASS:
            streams.append((lo, hi, hit.along))
            continue
        # SPLIT: a child just past each edge of the deflector
        streams.append((hit.lo - edge_width, hit.lo - 1, hit.along))
        streams.append((hit.hi + 1, hit.hi + edge_width, hit.along))
    return sorted(reached)


def _runs(cols):
    """Group ints into contiguous [lo, hi] runs."""
    runs = []
    for v in sorted(set(int(x) for x in cols)):
        if runs and v == runs[-1][1] + 1:
            runs[-1][1] = v
        else:
            runs.append([v, v])
    return [(a, b) for a, b in runs]


def induce_contact_rule(frames, bars, mobile, *, travel="up"):
    """INDUCTION FROM PERCEPTION: read one observed dispense's framestack and emit
    the predictor's rule — the per-bar contact OUTCOME and the split edge-offset —
    so the predictor supplies itself instead of being hand-fed.

    ``frames`` = the animation framestack (list of 2D int grids, e.g.
    StepResult.frame_stack).  ``bars`` = the perceived static bars as
    (rmin,rmax,cmin,cmax) bboxes.  ``mobile`` = the mobile element's colour value
    (the region that travels during the animation — identified by the
    frontier-trajectory perception).  Returns (outcomes, edge_width):
      outcomes  = {bar -> PASS|STOP|SPLIT} for the bars the frontier actually
                  reached (a bar never contacted yields no rule);
      edge_width = the induced split child width (None if no split seen).

    How: union the mobile cells over all frames (branches arrive asynchronously),
    then for each bar compare the APPROACH side (where the frontier meets the bar)
    to the FAR side (just beyond it).  Children that emerge immediately past the
    two EDGES with nothing directly beyond the bar = SPLIT; flow continuing
    straight beyond = PASS; nothing beyond = STOP.  The split offset is the width
    of the edge-adjacent child run (run-adjacency, no tuned window).  Game-agnostic.
    """
    import numpy as np
    grids = [np.asarray(g) for g in frames]
    if not grids:
        return {}, None
    H, W = grids[0].shape
    occ = np.zeros((H, W), dtype=bool)
    for g in grids:                            # union over time (async branches)
        occ |= (g == mobile)
    up = travel == "up"
    outcomes, offsets = {}, []
    for bar in bars:
        rmin, rmax, cmin, cmax = bar
        if up:
            appr_rows = [r for r in (rmax + 1, rmax + 2) if r < H]
            far_rows = [r for r in (rmin - 1, rmin - 2) if r >= 0]
        else:
            appr_rows = [r for r in (rmin - 1, rmin - 2) if r >= 0]
            far_rows = [r for r in (rmax + 1, rmax + 2) if r < H]
        # did the frontier actually reach this bar (within its span)?
        if not any(occ[r, c] for r in appr_rows for c in range(cmin, cmax + 1)):
            continue
        far_cols = sorted({c for r in far_rows for c in range(W) if occ[r, c]})
        fr = _runs(far_cols)
        left_child = next((run for run in fr if run[1] == cmin - 1), None)
        right_child = next((run for run in fr if run[0] == cmax + 1), None)
        mid_present = any(not (hi < cmin or lo > cmax) for lo, hi in fr)
        if (left_child or right_child) and not mid_present:
            outcomes[bar] = SPLIT
            for child in (left_child, right_child):
                if child:
                    offsets.append(child[1] - child[0] + 1)
        elif mid_present:
            outcomes[bar] = PASS
        else:
            outcomes[bar] = STOP
    edge_width = int(round(float(np.median(offsets)))) if offsets else None
    return outcomes, edge_width


def plan_routing(source, targets, paddles, *, edge_width=4, min_gap=8,
                 target_along=12):
    """BACKWARD-CHAIN a placement of movable deflectors so the predicted routing of
    the source frontier covers EVERY target — found in simulation, no dispensing.

    Works from the targets backward: to cover a target, place a paddle on a live
    frontier so one of its edge-children lands on that target; the paddle's two
    children become new live frontiers; recurse until every target is covered, then
    VERIFY the whole placement with predict_routing.

    ``source``  = (lo, hi) source span (e.g. the nozzle columns).
    ``targets`` = {name: (lo, hi)} target spans (e.g. cup openings).
    ``paddles`` = list of {'width':int, 'los':[...], 'alongs':[...]} — each movable
                  deflector's width and the discrete left-columns / contact-rows it
                  can occupy (from the game's movement model).
    ``min_gap`` = minimum along-separation between a bar and the frontier it splits,
                  to keep stages apart and OUT of the flooding regime (grounded: the
                  clean staircase had ~9-12 row separation, flooding ~1).  A planner
                  heuristic, refine when flooding is modelled.
    ``target_along`` = bars must sit beyond this (below the targets).

    Returns a list of Deflector (one per used paddle) or None.  Game-agnostic;
    travel='up'.  The unused paddles are simply not placed (park them out of path)."""
    names = list(targets)

    def covers(span, t):
        lo, hi = span; tl, th = targets[t]
        return not (hi < tl or lo > th)

    def search(frontiers, placements, used):
        done = {t for t in names if any(covers((f[0], f[1]), t) for f in frontiers)}
        if len(done) == len(names):
            reached = predict_routing(source, placements, travel="up",
                                      outcome=lambda d: SPLIT, edge_width=edge_width)
            return placements if set(targets_hit(reached, targets)) == set(names) else None
        for t in names:
            if t in done:
                continue
            tl, th = targets[t]
            for pi, p in enumerate(paddles):
                if pi in used:
                    continue
                w = p["width"]
                # a paddle delivers t as its LEFT child (lo = th+1) or RIGHT child
                # (lo = tl-w); only when an edge-offset child lands exactly on t.
                for lo in (th + 1, tl - w):
                    if lo not in p["los"]:
                        continue
                    hi = lo + w - 1
                    for f in frontiers:
                        flo, fhi, falong = f
                        if not (lo <= flo and hi >= fhi):     # must cover the frontier
                            continue
                        for along in sorted(p["alongs"], reverse=True):
                            if along > falong - min_gap or along <= target_along:
                                continue                      # above f, separated, below targets
                            kids = [(lo - edge_width, lo - 1, along),
                                    (hi + 1, hi + edge_width, along)]
                            nf = [x for x in frontiers if x != f] + kids
                            r = search(nf, placements + [Deflector(along, lo, hi)],
                                       used | {pi})
                            if r is not None:
                                return r
        return None

    return search([(source[0], source[1], 10 ** 9)], [], set())


def targets_hit(reached, targets):
    """Which named targets a reached frontier covers.  ``targets`` = {name:(lo,hi)}.
    A target is hit if any reached span overlaps its span."""
    hit = []
    for name, (lo, hi) in targets.items():
        if any(not (rh < lo or rl > hi) for rl, rh in reached):
            hit.append(name)
    return sorted(hit)
