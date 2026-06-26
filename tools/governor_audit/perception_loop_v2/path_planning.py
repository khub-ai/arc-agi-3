"""Game-agnostic PATH PLANNING over a scene's cell grid.

The substrate measures a NAVIGABLE MAP -- each cell floor (passable) vs barrier (a
distinct-colour wall, possibly with a gap) -- then BFS a route from the mover cell
to the goal cell as UP/DOWN/LEFT/RIGHT steps.  This is the passability map the
eyeball needs: a VLM reliably traces a route but MISSES walls/gaps at pixel scale
(the tn36 lc2 pink barrier read as "open floor" until measured).  Substrate
measures passability; the route is computed (or eyeballed) on top of it.

Floor vs barrier is figure-ground, not a colour key: floor = the dominant cell
colours (e.g. a checkerboard's two shades); a barrier = a distinct colour whose
cells form a WALL (a line spanning >= half a row/column).  The only structural
constant is "a wall spans at least half the grid"; no per-game tuning.
"""
from __future__ import annotations

from collections import Counter
from typing import Optional

try:
    import numpy as np
    _OK = True
except Exception:                                    # pragma: no cover
    _OK = False


def cell_grid(frame_rgb, scene_bbox, cell_ticks: int, origin_rc):
    """Dominant colour per cell over the scene.  Returns (grid, n_rows, n_cols)
    where grid[i][j] is an (r,g,b) tuple.  origin_rc = (row,col) tick of cell
    (0,0); scene_bbox bottom/right EXCLUSIVE."""
    arr = np.asarray(frame_rgb)[:, :, :3]
    r0o, c0o = int(origin_rc[0]), int(origin_rc[1])
    _, _, R1, C1 = [int(v) for v in scene_bbox]
    n_rows = max(0, (R1 - r0o) // cell_ticks)
    n_cols = max(0, (C1 - c0o) // cell_ticks)
    grid = []
    for i in range(n_rows):
        row = []
        for j in range(n_cols):
            r, c = r0o + i * cell_ticks, c0o + j * cell_ticks
            block = arr[r:r + cell_ticks, c:c + cell_ticks].reshape(-1, 3)
            cols, counts = np.unique(block, axis=0, return_counts=True)
            row.append(tuple(int(x) for x in cols[int(counts.argmax())]))
        grid.append(row)
    return grid, n_rows, n_cols


def _max_run(line) -> int:
    """Longest contiguous run of True in a boolean sequence."""
    best = cur = 0
    for v in line:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return best


def barrier_color(grid, n_rows, n_cols):
    """The colour forming the WALLS.  A wall is a CONTIGUOUS line of one colour
    (it forms straight segments); the floor -- a checkerboard's two alternating
    shades -- never has two same-colour cells adjacent.  So CONTIGUITY is the
    structural discriminator: the wall colour has a run of >= 2 in some row or
    column, a floor shade has a longest run of 1.

    No colour is pre-assumed to be floor (the wall may be as common as the floor),
    and there is NO length gate -- maze walls are often SHORT segments (tn36 lc6's
    pink walls are only ~3 cells, well under half the 7-cell grid; an earlier
    "spans >= half" gate wrongly read them as floor and routed the mover straight
    through them).  Among colours that form a line (run >= 2), the wall is the
    most line-like: the largest (longest-run, then longest-extent).  A pure
    checkerboard (every colour run == 1) yields None.  Returns the barrier colour,
    or None.  N.B. cell colours are per-cell DOMINANT, so small on-floor entities
    (mover/goal/markers) do not appear as their own colour here -- only the floor
    shades and the wall colour do."""
    if n_rows == 0 or n_cols == 0:
        return None
    colours = {grid[i][j] for i in range(n_rows) for j in range(n_cols)}
    best, best_key = None, None
    for col in colours:
        max_run, max_extent = 0, 0
        for j in range(n_cols):                          # scan each column
            line = [grid[i][j] == col for i in range(n_rows)]
            max_run = max(max_run, _max_run(line))
            max_extent = max(max_extent, sum(line))
        for i in range(n_rows):                          # scan each row
            line = [grid[i][j] == col for j in range(n_cols)]
            max_run = max(max_run, _max_run(line))
            max_extent = max(max_extent, sum(line))
        if max_run >= 2:                                 # forms a line (a wall), not a checkerboard/scatter
            key = (max_run, max_extent)
            if best_key is None or key > best_key:
                best, best_key = col, key
    return best


def passability(grid, n_rows, n_cols, barrier):
    """Boolean grid: True = passable (floor), False = barrier cell."""
    return [[grid[i][j] != barrier for j in range(n_cols)] for i in range(n_rows)]


def cell_of(entity_bbox, cell_ticks: int, origin_rc):
    """Cell (row,col) containing an entity's bbox CENTRE."""
    r0, c0, r1, c1 = [int(v) for v in entity_bbox]
    cr, cc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    return (int((cr - origin_rc[0]) // cell_ticks),
            int((cc - origin_rc[1]) // cell_ticks))


def route_cells(color_grid, n_rows, n_cols, barrier, start, goal):
    """Shortest UP/DOWN/LEFT/RIGHT route start->goal, computed by the EXISTING
    maze_model planner (no duplicate BFS).  We build a maze_model StaticField from
    the per-cell colours (barrier cells -> a blocking class; the start & goal
    forced passable), seed the four unit-move rules, and call MazeModel.plan --
    reusing its tested simulate/BFS.  None if unreachable.  Guarded."""
    try:
        import os
        import sys
        _p = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _p not in sys.path:                       # so maze_model's
            sys.path.insert(0, _p)                   # `perception_loop_v2.world_model` import resolves
        from perception_loop_v2 import maze_model as _mm
        from perception_loop_v2.world_model import EntityState, WorldState
        ids: dict = {}
        def cid(col):
            return ids.setdefault(col, len(ids))
        grid = [[cid(color_grid[i][j]) for j in range(n_cols)] for i in range(n_rows)]
        bar_id = cid(barrier) if barrier is not None else None
        field = _mm.StaticField(grid, n_rows, n_cols)
        for cell in (start, goal):                   # mover stands on free ground; goal reachable
            if 0 <= cell[0] < n_rows and 0 <= cell[1] < n_cols:
                field.grid[cell[0]][cell[1]] = _mm.PASSABLE
        m = _mm.MazeModel(coupled=True)
        m.field = field                              # set directly (single-frame blocking, no transitions)
        m.blocking = {bar_id} if bar_id is not None else set()
        m.rules = {"UP": {0: ((-1, 0), False)}, "DOWN": {0: ((1, 0), False)},
                   "LEFT": {0: ((0, -1), False)}, "RIGHT": {0: ((0, 1), False)}}
        state = WorldState.of([EntityState(eid=0, cls=("mover",), pos=tuple(start))])
        gt = tuple(goal)
        return m.plan(state, lambda s: s.get(0) is not None and s.get(0).pos == gt,
                      ["UP", "DOWN", "LEFT", "RIGHT"])
    except Exception:
        return None


def plan_route(passable, start, goal):
    """Route over a boolean passable grid -- a thin adapter that delegates to the
    maze_model-backed route_cells (kept so callers/tests can pass a boolean grid;
    NO separate BFS lives here)."""
    nr = len(passable)
    nc = len(passable[0]) if nr else 0
    floor, wall = (0, 0, 0), (1, 1, 1)
    color_grid = [[floor if passable[i][j] else wall for j in range(nc)]
                  for i in range(nr)]
    return route_cells(color_grid, nr, nc, wall, start, goal)


_OPP = {"UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}
_VEC = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


def shape_opening(mask) -> Optional[str]:
    """The side a silhouette is OPEN on -- its 'mouth' (UP/DOWN/LEFT/RIGHT), or
    None for a closed/solid shape.  A side is a mouth when the shape has walls at
    BOTH ends of that edge but background in the middle (a U opens UP, a ⊓ opens
    DOWN).  The widest mouth wins.  Threshold-light: just 'walls at the ends, gap
    in the middle'.  This is the directional concavity a coupling needs."""
    if not _OK or mask is None:
        return None
    m = np.asarray(mask, dtype=bool)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    if m.shape[0] < 2 or m.shape[1] < 2:
        return None
    sides = {"UP": m[0, :], "DOWN": m[-1, :], "LEFT": m[:, 0], "RIGHT": m[:, -1]}
    best, best_mouth = None, 0
    for k, edge in sides.items():
        if len(edge) < 3 or not (edge[0] and edge[-1]):
            continue                              # need walls at both ends
        mouth = int((~edge[1:-1]).sum())          # background cells in the middle
        if mouth > best_mouth:
            best, best_mouth = k, mouth
    return best


def openings_couple(mover_opening, goal_opening) -> bool:
    """True if a mover with opening ``mover_opening`` can MATE into a goal with
    opening ``goal_opening`` -- their mouths must face each other (opposite
    directions): a cup (opens UP) couples an arch (opens DOWN)."""
    return (mover_opening is not None and goal_opening is not None
            and _OPP.get(mover_opening) == goal_opening)


def infer_cell_ticks(frame_rgb, scene_bbox) -> int:
    """Infer the scene's cell size as the dominant RUN-LENGTH of constant colour
    along the scene's middle row and column (a checkerboard's square size).  The
    mover/goal/barrier are quantised to this grid.  Falls back to 4.  Guarded."""
    try:
        arr = np.asarray(frame_rgb)[:, :, :3]
        r0, c0, r1, c1 = [int(v) for v in scene_bbox]
        runs = []
        for line in (arr[(r0 + r1) // 2, c0:c1], arr[r0:r1, (c0 + c1) // 2]):
            n = 1
            for k in range(1, len(line)):
                if tuple(line[k]) == tuple(line[k - 1]):
                    n += 1
                else:
                    if n > 1:
                        runs.append(n)
                    n = 1
            if n > 1:
                runs.append(n)
        if not runs:
            return 4
        return int(Counter(runs).most_common(1)[0][0])
    except Exception:
        return 4


def route_from_entities(frame_rgb, entities) -> Optional[dict]:
    """Game-agnostic bridge perception -> route.  From the perceived entities,
    identify the COUPLING pair of glyphs (two shapes whose openings mate -- the
    mover and the goal), the scene field that contains them, the barrier, and the
    cell grid, then plan the coupling-aware route.  The mover is the LOWER glyph
    (the traveller that rises into the upper goal -- as the demonstration shows);
    swap is harmless to the geometry.  Returns plan_scene_route's dict augmented
    with {mover, goal} names, or None.  Guarded -- never raises."""
    if not _OK or not entities:
        return None
    try:
        import shape_identity as _si
        # glyphs = entities with a detectable mouth (a U/arch), not panels/fields
        glyphs = []
        for e in entities:
            bb = e.get("bbox_ticks_turn1")
            if not bb:
                continue
            mask = _si._component_mask(frame_rgb, bb)
            if mask is None:
                mask = _si._foreground_mask(frame_rgb, bb)
            op = shape_opening(mask)
            if op is not None:
                glyphs.append((e, op, bb))
        def contains(o, inn):
            return (o[0] <= inn[0] and o[1] <= inn[1]
                    and inn[2] <= o[2] and inn[3] <= o[3] and o != inn)
        # The coupling pair must be a mover+goal in the SAME scene field -- not a
        # demonstration-area glyph paired across regions.  So scan candidate scene
        # fields (entities containing >=2 glyphs), tightest first, and take the
        # first that holds a coupling pair.
        fields = sorted(
            [e.get("bbox_ticks_turn1") for e in entities if e.get("bbox_ticks_turn1")],
            key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        pair, scene_bb = None, None
        for fbb in fields:
            inside = [g for g in glyphs if contains(fbb, g[2])]
            for i in range(len(inside)):
                for j in range(i + 1, len(inside)):
                    if openings_couple(inside[i][1], inside[j][1]):
                        pair, scene_bb = (inside[i], inside[j]), fbb
                        break
                if pair:
                    break
            if pair:
                break
        if not pair:
            return None
        (ea, _oa, bba), (eb, _ob, bbb) = pair
        # mover = lower glyph (larger top-row); goal = upper
        if bba[0] >= bbb[0]:
            mover, goal = (ea, bba), (eb, bbb)
        else:
            mover, goal = (eb, bbb), (ea, bba)
        ct = infer_cell_ticks(frame_rgb, scene_bb)
        origin = (scene_bb[0], scene_bb[1])
        res = plan_scene_route(frame_rgb, scene_bb, ct, origin, mover[1], goal[1])
        if res is not None:
            res["mover"] = mover[0].get("name")
            res["goal"] = goal[0].get("name")
            res["cell_ticks"] = ct
            res["scene_bbox"] = scene_bb
        return res
    except Exception:
        return None


def plan_scene_route(frame_rgb, scene_bbox, cell_ticks, origin_rc,
                     mover_bbox, goal_bbox) -> Optional[dict]:
    """End-to-end: measure the scene's navigable map and route the mover to the
    goal -- COUPLING-AWARE.  The goal is not a point but a SHAPE the mover must
    MATE into: if the two silhouettes' openings are complementary, the route must
    end with the mover entering the goal's mouth from the matching side (the
    docking move), not slide in sideways.  So it BFS-routes to the DOCK cell
    (adjacent to the goal on the mover's opening side) and appends the mating
    move.  Falls back to reaching the goal cell when no coupling is detected.
    Returns {route, coupling, n_rows, n_cols, start, goal, barrier, passable} or
    None.  Guarded -- never raises."""
    if not _OK or frame_rgb is None:
        return None
    try:
        grid, nr, nc = cell_grid(frame_rgb, scene_bbox, cell_ticks, origin_rc)
        if nr == 0 or nc == 0:
            return None
        bar = barrier_color(grid, nr, nc)
        pas = passability(grid, nr, nc, bar)
        start = cell_of(mover_bbox, cell_ticks, origin_rc)
        goal = cell_of(goal_bbox, cell_ticks, origin_rc)
        # Coupling: read each shape's mouth and, if they mate, dock instead of
        # reaching the goal cell head-on.
        mo = go = None
        try:
            import shape_identity as _si
            mm = _si._component_mask(frame_rgb, mover_bbox)
            if mm is None:
                mm = _si._foreground_mask(frame_rgb, mover_bbox)
            gm = _si._component_mask(frame_rgb, goal_bbox)
            if gm is None:
                gm = _si._foreground_mask(frame_rgb, goal_bbox)
            mo, go = shape_opening(mm), shape_opening(gm)
        except Exception:
            pass
        coupling = {"mover_opening": mo, "goal_opening": go,
                    "coupled": openings_couple(mo, go)}
        if coupling["coupled"]:
            dv = _VEC[mo]
            dock = (goal[0] - dv[0], goal[1] - dv[1])      # approach from the mover's open side
            coupling["dock_cell"] = dock
            coupling["mating_move"] = mo
            route = route_cells(grid, nr, nc, bar, start, dock)
            if route is not None:
                route = route + [mo]                        # the docking move that mates the shapes
        else:
            route = route_cells(grid, nr, nc, bar, start, goal)
        return {"route": route, "coupling": coupling, "n_rows": nr, "n_cols": nc,
                "start": start, "goal": goal, "barrier": bar, "passable": pas}
    except Exception:
        return None
