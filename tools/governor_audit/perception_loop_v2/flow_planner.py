"""Autonomous flow-routing solver — runs the observe-once-then-plan loop through
an action adapter so the driver can solve a cascade-routing level in-loop:

    calibrate (one dispense over the source)  -> induce the contact rule
    plan a winning config (plan_routing)      -> execute it (plan_to_actions)

Game-agnostic: every game specific (which colours are deflectors / the mobile
flow, the source columns, the target spans, each paddle's width + reachable grid,
the action names that select / move / dispense) comes in as a parameter — the
driver discovers them via perception + affordance probing and hands them here.

``plan_to_actions`` is a PURE translation (plan + paddles -> a static action list)
and is unit-tested in isolation.  ``solve_flow`` is the thin orchestrator that
drives the adapter.

KNOWN LIMITATION (execution, not planning).  ``plan_to_actions`` moves each paddle
independently and does NOT do collision-aware motion planning: repositioning
several paddles past one another (or to the SAME columns at different rows) can
make them collide / occlude mid-move and corrupt the layout — observed when three
paddles are shuffled from clustered post-calibration positions.  So calibrate ->
induce -> plan is validated live, but ``solve_flow``'s autonomous EXECUTION is not
yet robust for arbitrary plans.  The remaining piece is collision-free
repositioning (a small motion-planner) and/or an executability constraint in
``plan_routing`` (prefer configs whose paddles sit at distinct columns / clear
lanes).  This is separate from the routing model, which is sound.
"""
from __future__ import annotations
from perception_loop_v2.flow_model import (
    Deflector, predict_routing, targets_hit, induce_contact_rule, plan_routing,
    SPLIT)


def _bbox_centre(bb):
    rmin, rmax, cmin, cmax = bb
    return (cmin + cmax) // 2, (rmin + rmax) // 2     # (col, row) for a CLICK


def plan_to_actions(plan, paddles, actions, *, park=(0, 60)):
    """Translate a plan (list of Deflector) into a STATIC action list that
    positions each paddle and dispenses.  PURE — no game/adapter; deterministic
    move counts from known positions (the active/clicked paddle is the one that
    moves, ``step`` cells per move).

    ``paddles`` = list of {'width':int, 'bbox':(rmin,rmax,cmin,cmax)} (current
    positions).  ``actions`` = {'select','up','down','left','right','dispense',
    'step'}.  Deflectors are assigned to unused paddles by matching width; unused
    paddles are parked at ``park`` (col,row) bottom-left, out of the cascade.

    Returns (action_list, final_bboxes)."""
    step = actions["step"]
    sel, up, dn, lf, rt, disp = (actions[k] for k in
                                 ("select", "up", "down", "left", "right", "dispense"))
    pads = [dict(p, bbox=tuple(p["bbox"])) for p in paddles]
    out = []

    def move(pi, L, T):
        bb = pads[pi]["bbox"]; rmin, rmax, cmin, cmax = bb
        h, w = rmax - rmin + 1, cmax - cmin + 1
        col, row = _bbox_centre(bb)
        out.append(f"{sel}:{col},{row}")                 # select this paddle
        dc = L - cmin
        out.extend([rt if dc > 0 else lf] * (abs(dc) // step))
        dr = T - rmin
        out.extend([dn if dr > 0 else up] * (abs(dr) // step))
        pads[pi]["bbox"] = (T, T + h - 1, L, L + w - 1)

    # assign deflectors to paddles by width
    assign = {}
    for d in plan:
        w = d.hi - d.lo + 1
        pi = next((i for i, p in enumerate(pads)
                   if i not in assign and p["width"] == w), None)
        if pi is None:
            raise ValueError(f"plan_to_actions: no free paddle of width {w}")
        assign[pi] = d
    # park the unused paddles first (clears the cascade), then place the used ones.
    # Stagger parks by column so two parked paddles do not overlap (which would
    # merge them into one component on a re-read).
    k = 0
    for i in range(len(pads)):
        if i not in assign:
            move(i, park[0] + k * 16, park[1]); k += 1
    for pi, d in assign.items():
        h = pads[pi]["bbox"][1] - pads[pi]["bbox"][0] + 1
        move(pi, d.lo, d.along - (h - 1))
    out.append(disp)
    return out, [p["bbox"] for p in pads]


def _calibration_deflector(source, paddles):
    """A single-bar placement that puts the widest paddle over the source so the
    calibration dispense produces a clean split to induce from."""
    widest = max(paddles, key=lambda p: p["width"])
    w = widest["width"]; slo, shi = source
    lo = slo - (w - (shi - slo + 1)) // 2            # centre the bar over the source
    lo = max(0, (lo // 4) * 4)                       # snap to the 4-grid
    h = widest["bbox"][1] - widest["bbox"][0] + 1
    along = widest["bbox"][1]                         # keep its current row
    return Deflector(along, lo, lo + w - 1)


def _components(grid, values):
    """4-connected components whose cells are in ``values`` (a set of colours).
    Returns list of {'width','bbox'} (bbox = rmin,rmax,cmin,cmax).  Each colour is
    grown separately so adjacent different-colour deflectors don't merge."""
    import numpy as np
    H, W = grid.shape
    out = []
    for v in values:
        mask = (grid == v); seen = np.zeros_like(mask, bool)
        for r in range(H):
            for c in range(W):
                if mask[r, c] and not seen[r, c]:
                    st = [(r, c)]; seen[r, c] = True; cs = []
                    while st:
                        y, x = st.pop(); cs.append((y, x))
                        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                                seen[ny, nx] = True; st.append((ny, nx))
                    ys = [p[0] for p in cs]; xs = [p[1] for p in cs]
                    bb = (min(ys), max(ys), min(xs), max(xs))
                    out.append({"width": bb[3] - bb[2] + 1, "bbox": bb})
    return out


def solve_flow(adapter, *, source, targets, paddle_specs, paddles, actions,
               mobile_colour, deflector_colours, travel="up", read_paddles=None):
    """Drive ``adapter`` through calibrate -> induce -> plan -> execute.  Returns
    {'edge_width','plan','predicted','win_state','score'}.  ``adapter`` exposes
    ``step(action)->StepResult`` and ``_frame_stack``/``_obs`` (the live-harness
    adapter).  ``paddle_specs`` = list of {'width','los','alongs'} for the planner;
    ``paddles`` = list of {'width','bbox'} current positions.  ``read_paddles`` is
    an optional ``(grid)->[{'width','bbox'}]`` perception callback (defaults to a
    connected-component reader over ``deflector_colours``) — used to RE-READ paddle
    positions after calibration so execution is robust to move-range clamping."""
    import numpy as np
    read_paddles = read_paddles or (lambda g: _components(g, deflector_colours))

    def run(action_list):
        res = None
        for a in action_list:
            res = adapter.step(a)
        return res

    def grid():
        return np.asarray(adapter._frame_stack(adapter._obs)[-1])

    # 1) CALIBRATE: one paddle over the source, dispense, induce
    calib = _calibration_deflector(source, paddles)
    calib_actions, _ = plan_to_actions([calib], paddles, actions)
    cb = run(calib_actions)
    bars = [p["bbox"] for p in read_paddles(grid())]
    outcomes, edge = induce_contact_rule(cb.frame_stack, bars,
                                         mobile=mobile_colour, travel=travel)
    if not edge:
        return {"edge_width": None, "plan": None, "predicted": [],
                "win_state": cb.win_state, "score": cb.score}

    # 2) PLAN (simulation only)
    plan = plan_routing(source, targets, paddle_specs, edge_width=edge)
    if plan is None:
        return {"edge_width": edge, "plan": None, "predicted": [],
                "win_state": cb.win_state, "score": cb.score}
    predicted = targets_hit(predict_routing(source, plan, travel=travel,
                            outcome=lambda d: SPLIT, edge_width=edge), targets)

    # 3) EXECUTE — re-read actual paddle positions (robust to clamping), then place
    pads_now = read_paddles(grid())
    exec_actions, _ = plan_to_actions(plan, pads_now, actions)
    res = run(exec_actions)
    return {"edge_width": edge, "plan": plan, "predicted": predicted,
            "win_state": res.win_state, "score": res.score}
