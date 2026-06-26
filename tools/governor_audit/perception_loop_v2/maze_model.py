"""Maze-grounded forward model — see docs/SPEC_maze_grounded_model.md.

Game-agnostic COS fixture: movement is simulated through a PERCEIVED static
field (per-cell class), with the blocking relation LEARNED over region-classes
(not memorized per cell), so the model predicts movement anywhere in the
structure from a few observations.  A lookahead goal-distance (bounded search to
the goal in the model) yields an action-evaluation that correctly values
instrumental moves a one-step block-distance mis-scores.

No per-game branches: the movement primitive, which class blocks, the coupling,
and the goal are all learned/supplied at runtime, never hardcoded.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from collections import deque
import numpy as np
from perception_loop_v2.world_model import WorldState, EntityState


def _sign(x):
    return (x > 0) - (x < 0)


PASSABLE = -1   # sentinel class: always passable (a mover stands on free ground)


@dataclass
class StaticField:
    grid: list      # grid[r][c] -> class int (or PASSABLE)
    R: int
    C: int

    def cls(self, r, c):
        if 0 <= r < self.R and 0 <= c < self.C:
            return self.grid[r][c]
        return None     # out of bounds


def build_field(frame, mover_cells, cell_size=1, offset=0):
    """Static field from a palette-grid `frame` at the movers' cell resolution.

    Cell (r,c) is sampled at the centre of its cell_size block, shifted by
    `offset` (the rendered board's letterbox border), so the field cells align
    with the offset-aware mover quantization round((px-offset)/cell_size).  Mover
    cells are PASSABLE (a mover stands on free ground), so the field is the
    static structure only.  Role-free — just 'what class is at each cell'.
    """
    H, W = frame.shape
    R = (H - offset + cell_size - 1) // cell_size
    C = (W - offset + cell_size - 1) // cell_size
    half = cell_size // 2
    grid = [[0] * C for _ in range(R)]
    for r in range(R):
        for c in range(C):
            y = min(H - 1, offset + r * cell_size + half)
            x = min(W - 1, offset + c * cell_size + half)
            grid[r][c] = int(frame[y, x])
    for (r, c) in mover_cells:
        if 0 <= r < R and 0 <= c < C:
            grid[r][c] = PASSABLE
    return StaticField(grid, R, C)


class MazeModel:
    def __init__(self, coupled=True):
        self.field = None
        self.rules = {}             # action -> {eid: (unit, slide)}
        self.blocking = set()       # learned blocking region-classes
        self._passable_seen = set()
        self._block_seen = {}       # class -> count seen just beyond a stop
        self.coupled = coupled
        self.transitions = []

    def set_field(self, field):
        self.field = field
        # Blocking is field-dependent.  (Re)derive it from ALL recorded
        # transitions so learning is INDEPENDENT of whether observe() ran before
        # or after the field was known — otherwise transitions seen while the
        # field was None are silently lost (they contribute movement rules but
        # never blocking), which empties `blocking` and makes the planner inert.
        self._relearn_blocking()

    def _relearn_blocking(self):
        self._passable_seen = set()
        self._block_seen = {}
        for before, _action, after in self.transitions:
            self._accumulate_blocking(before, after)
        self.blocking = {c for c in self._block_seen if c not in self._passable_seen}

    def _accumulate_blocking(self, before, after):
        """Fold one transition into the passable/block evidence.  Cells passed
        through (incl. start & stop) are passable; the cell just beyond a stop is
        a blocking candidate.  No-op until the field is known."""
        if self.field is None:
            return
        for e in before.entities:
            ae = after.get(e.eid)
            if ae is None:
                continue
            unit = (_sign(ae.pos[0] - e.pos[0]), _sign(ae.pos[1] - e.pos[1]))
            if unit == (0, 0):
                continue
            cur = e.pos
            while cur != ae.pos:
                self._passable_seen.add(self.field.cls(*cur))
                cur = (cur[0] + unit[0], cur[1] + unit[1])
            self._passable_seen.add(self.field.cls(*ae.pos))
            beyond = (ae.pos[0] + unit[0], ae.pos[1] + unit[1])
            bc = self.field.cls(*beyond)
            if bc is not None:
                self._block_seen[bc] = self._block_seen.get(bc, 0) + 1

    # -- learning ---------------------------------------------------------
    def observe(self, before, action, after):
        self.transitions.append((before, action, after))
        rule = self.rules.get(action, {})
        for e in before.entities:
            ae = after.get(e.eid)
            if ae is None:
                continue
            dr = ae.pos[0] - e.pos[0]; dc = ae.pos[1] - e.pos[1]
            unit = (_sign(dr), _sign(dc))
            if unit != (0, 0):
                slid = (abs(dr) + abs(dc)) > 1
                prev = rule.get(e.eid)
                rule[e.eid] = (unit, slid or (prev[1] if prev else False))
        self.rules[action] = rule
        # incremental blocking update (field may still be None -> deferred to
        # set_field via _relearn_blocking); a class blocks iff seen at a
        # stop-boundary and never passed through.
        self._accumulate_blocking(before, after)
        self.blocking = {c for c in self._block_seen if c not in self._passable_seen}

    def _passable(self, cell):
        c = self.field.cls(*cell)
        return c is not None and c not in self.blocking

    # -- simulation -------------------------------------------------------
    def simulate(self, state, action):
        rule = self.rules.get(action)
        if rule is None or self.field is None:
            return None
        pos = {e.eid: e.pos for e in state.entities}
        movers = [eid for eid in rule if eid in pos]
        if not movers:
            return state
        units = {eid: rule[eid][0] for eid in movers}
        sliding = any(rule[eid][1] for eid in movers)
        limit = (self.field.R + self.field.C + 2) if sliding else 1
        steps = 0
        while steps < limit:
            steps += 1
            nxt = {eid: (pos[eid][0] + units[eid][0], pos[eid][1] + units[eid][1])
                   for eid in movers}
            if self.coupled:
                if all(self._passable(nxt[eid]) for eid in movers):
                    pos = {**pos, **nxt}
                else:
                    break
            else:
                moved = False
                for eid in movers:
                    if self._passable(nxt[eid]):
                        pos[eid] = nxt[eid]; moved = True
                if not moved:
                    break
            if len({pos[eid] for eid in movers}) < len(movers):    # collision = arrival
                break
            if not sliding:
                break
        return WorldState.of([replace(e, pos=pos[e.eid]) if e.eid in units else e
                              for e in state.entities])

    # -- planning / evaluation -------------------------------------------
    def plan(self, state, goal, actions, *, max_nodes=20000):
        if goal(state):
            return []
        seen = {state}; q = deque([(state, [])])
        while q:
            st, path = q.popleft()
            for a in actions:
                ns = self.simulate(st, a)
                if ns is None or ns in seen:
                    continue
                if goal(ns):
                    return path + [a]
                seen.add(ns); q.append((ns, path + [a]))
                if len(seen) > max_nodes:
                    return None
        return None

    def goal_distance(self, state, goal, actions, *, max_nodes=20000):
        """Lookahead distance: # moves the model thinks reach the goal, or None
        if unreachable within budget.  The maze-aware heuristic."""
        if goal(state):
            return 0
        seen = {state}; q = deque([(state, 0)])
        while q:
            st, d = q.popleft()
            for a in actions:
                ns = self.simulate(st, a)
                if ns is None or ns in seen:
                    continue
                if goal(ns):
                    return d + 1
                seen.add(ns); q.append((ns, d + 1))
                if len(seen) > max_nodes:
                    return None
        return None

    def evaluate(self, state, action, goal, actions, *, max_nodes=8000):
        """Action-evaluation: +ve => the action brings us CLOSER to the goal
        (through the structure), -ve => farther.  Difference of lookahead
        goal-distances; None if the goal is unreachable in the model from either."""
        ns = self.simulate(state, action)
        if ns is None:
            return None
        d0 = self.goal_distance(state, goal, actions, max_nodes=max_nodes)
        d1 = self.goal_distance(ns, goal, actions, max_nodes=max_nodes)
        if d0 is None or d1 is None:
            return None
        return d0 - d1

    # -- control (MPC) ----------------------------------------------------
    def drive(self, perceive, step, actions, goal, *, budget=200, plan_nodes=20000,
              probe=True, log=None):
        used = 0
        if probe:                                   # learn movement + blocking
            for a in actions:
                if used >= budget:
                    break
                b = perceive(); ns, score, done = step(a); used += 1
                self.observe(b, a, ns)
                if log:
                    log(used, a, 'probe', score, done)
                if done or goal(ns):
                    return True, used
        while used < budget:
            st = perceive()
            if goal(st):
                return True, used
            plan = self.plan(st, goal, actions, max_nodes=plan_nodes)
            if plan:
                for a in plan:
                    before = perceive(); pred = self.simulate(before, a)
                    ns, score, done = step(a); used += 1
                    self.observe(before, a, ns)
                    if log:
                        log(used, a, 'plan', score, done)
                    if done or goal(ns):
                        return True, used
                    if pred is None or ns != pred:    # model wrong here -> refine + replan
                        break
                    if used >= budget:
                        return False, used
            else:
                a = min(actions, key=lambda x: sum(1 for (_, act, _) in self.transitions
                                                   if act == x))
                before = perceive(); ns, score, done = step(a); used += 1
                self.observe(before, a, ns)
                if log:
                    log(used, a, 'explore', score, done)
                if done or goal(ns):
                    return True, used
        return False, used
