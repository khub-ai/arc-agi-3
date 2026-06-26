"""COS World Model — a generic, observation-built forward model over the
PERCEPTION STATE, with planning and model-predictive control.

See docs/SPEC_world_model.md.  This is a COS fixture (part of the player): it
predicts the next perceived state from the current one and an action, learned
ONLY from observed (state, action, next_state) transitions — never from the
game's source, sprite tables, or environment forking.  It lets the solver plan
ahead cheaply instead of poking the live, action-budgeted environment.

State = a snapshot of tracked entities (stable_id, class, cell position).  The
model fits a small, universal vocabulary of transition operators (Translate,
Recolor, CoupledSlide, ...) whose PARAMETERS come from data.  Dynamics it has
not seen are flagged unknown; planning + MPC correction handle model error.
Substrate-agnostic: no role names, no per-game code.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections import deque


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EntityState:
    eid: int                       # stable identity (track id)
    cls: tuple                     # hashable class key (palette sig / role)
    pos: tuple                     # (row, col) cell position
    present: bool = True


@dataclass(frozen=True)
class WorldState:
    entities: tuple                # sorted tuple of EntityState — hashable

    @staticmethod
    def of(ents) -> "WorldState":
        return WorldState(tuple(sorted(ents, key=lambda e: e.eid)))

    def get(self, eid):
        for e in self.entities:
            if e.eid == eid:
                return e
        return None

    def positions(self):
        return {e.eid: e.pos for e in self.entities if e.present}


# --------------------------------------------------------------------------
# Operators — a fixed universal vocabulary; parameters are learned.
# --------------------------------------------------------------------------
class Operator:
    confidence: float = 1.0

    def predict(self, st: WorldState) -> WorldState:
        raise NotImplementedError


@dataclass
class Translate(Operator):
    eids: frozenset
    delta: tuple
    confidence: float = 1.0

    def predict(self, st):
        out = []
        for e in st.entities:
            if e.present and e.eid in self.eids:
                out.append(replace(e, pos=(e.pos[0] + self.delta[0],
                                           e.pos[1] + self.delta[1])))
            else:
                out.append(e)
        return WorldState.of(out)


@dataclass
class Recolor(Operator):
    mapping: tuple                 # tuple of (eid, new_cls)
    confidence: float = 1.0

    def predict(self, st):
        m = dict(self.mapping)
        return WorldState.of([replace(e, cls=m[e.eid]) if e.eid in m else e
                              for e in st.entities])


@dataclass
class CoupledSlide(Operator):
    units: tuple                   # tuple of (eid, (du, dv))
    blocked: frozenset             # learned blocked cells
    bounds: tuple                  # (rmin, cmin, rmax, cmax) inclusive
    lockstep: bool = True
    confidence: float = 0.6

    def predict(self, st):
        units = dict(self.units)
        movers = list(units)
        pos = {e.eid: e.pos for e in st.entities}
        r0, c0, r1, c1 = self.bounds

        def free(p):
            return (r0 <= p[0] <= r1 and c0 <= p[1] <= c1
                    and p not in self.blocked)

        steps = 0
        limit = (r1 - r0) + (c1 - c0) + 2
        while steps < limit:
            steps += 1
            nxt = {eid: (pos[eid][0] + units[eid][0], pos[eid][1] + units[eid][1])
                   for eid in movers}
            if self.lockstep:
                if not all(free(nxt[eid]) for eid in movers):
                    break
                pos = {**pos, **nxt}
            else:
                moved = False
                for eid in movers:
                    if free(nxt[eid]):
                        pos[eid] = nxt[eid]; moved = True
                if not moved:
                    break
            # collision among movers = arrival / merge: stop here.
            mp = [pos[eid] for eid in movers]
            if len(set(mp)) < len(mp):
                break
        return WorldState.of([replace(e, pos=pos[e.eid]) if e.eid in units else e
                              for e in st.entities])


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------
class WorldModel:
    def __init__(self):
        self.transitions: list = []          # (before, action, after)
        self.ops: dict = {}                   # action -> [Operator]
        self.confidence: dict = {}            # action -> float
        self._blocked: dict = {}              # action -> set(cell)
        self.bounds = None

    # -- observation / induction ------------------------------------------
    def observe(self, before: WorldState, action, after: WorldState) -> None:
        self.transitions.append((before, action, after))
        self._grow_bounds(before); self._grow_bounds(after)
        self._induce(action)

    def _grow_bounds(self, st):
        for e in st.entities:
            if not e.present:
                continue
            r, c = e.pos
            if self.bounds is None:
                self.bounds = (r, c, r, c)
            r0, c0, r1, c1 = self.bounds
            self.bounds = (min(r0, r), min(c0, c), max(r1, r), max(c1, c))

    def _induce(self, action):
        trs = [(b, a) for (b, act, a) in self.transitions if act == action]
        translators: dict = {}      # delta -> set(eid)
        recolors: dict = {}         # eid -> new_cls
        sliders: dict = {}          # eid -> unit
        blocked = set(self._blocked.get(action, set()))
        eids = set()
        for b, a in trs:
            eids |= {e.eid for e in b.entities}
        for eid in sorted(eids):
            deltas = []; clschg = []; ok = True
            for b, a in trs:
                be, ae = b.get(eid), a.get(eid)
                if be is None or ae is None or not be.present or not ae.present:
                    ok = False; break
                deltas.append((ae.pos[0] - be.pos[0], ae.pos[1] - be.pos[1]))
                if be.cls != ae.cls:
                    clschg.append(ae.cls)
            if not ok or not deltas:
                continue
            if clschg and all(c == clschg[0] for c in clschg) \
                    and all(d == (0, 0) for d in deltas):
                recolors[eid] = clschg[0]; continue
            if all(d == deltas[0] for d in deltas) and deltas[0] != (0, 0):
                translators.setdefault(deltas[0], set()).add(eid); continue
            nz = [d for d in deltas if d != (0, 0)]
            if nz:
                units = {(_sign(d[0]), _sign(d[1])) for d in nz}
                if len(units) == 1:
                    sliders[eid] = units.pop(); continue
            # else: no consistent behaviour -> leave unmodeled
        # Learn wall cells from where sliders STOP — but only from genuine
        # wall-stops.  Skip transitions where the sliders collide (an arrival /
        # merge, not a wall) so we don't record spurious walls that would block
        # the real solution path.
        for b, a in trs:
            finals = [a.get(e).pos for e in sliders if a.get(e)]
            if len(set(finals)) < len(finals):
                continue
            for eid, u in sliders.items():
                ae = a.get(eid)
                if ae is not None:
                    blocked.add((ae.pos[0] + u[0], ae.pos[1] + u[1]))
        self._blocked[action] = blocked
        ops: list = []
        for delta, es in translators.items():
            ops.append(Translate(frozenset(es), delta))
        if recolors:
            ops.append(Recolor(tuple(sorted(recolors.items()))))
        if sliders:
            ops.append(CoupledSlide(tuple(sorted(sliders.items())),
                                    frozenset(blocked), self.bounds,
                                    lockstep=True))
        self.ops[action] = ops
        self.confidence[action] = self._verify(action)

    def _verify(self, action) -> float:
        """Fraction of observed transitions this action's operators reproduce."""
        trs = [(b, a) for (b, act, a) in self.transitions if act == action]
        if not trs:
            return 0.0
        ok = 0
        for b, a in trs:
            pred, _ = self.simulate(b, action)
            if pred == a:
                ok += 1
        return ok / len(trs)

    # -- prediction / planning --------------------------------------------
    def simulate(self, state: WorldState, action):
        ops = self.ops.get(action)
        if ops is None:
            return None, 0.0                      # unseen action -> unknown
        s = state
        for op in ops:
            s = op.predict(s)
        return s, self.confidence.get(action, 1.0)

    def plan(self, start: WorldState, actions, goal, *, max_nodes=200000):
        if goal(start):
            return []
        seen = {start}; q = deque([(start, [])])
        while q:
            st, path = q.popleft()
            for a in actions:
                ns, _ = self.simulate(st, a)
                if ns is None:
                    continue
                if goal(ns):
                    return path + [a]
                if ns not in seen:
                    seen.add(ns); q.append((ns, path + [a]))
                    if len(seen) > max_nodes:
                        return None
        return None

    def plan_toward_min(self, start, actions, heuristic, *, max_nodes=20000):
        """Best-first search for the action path that MINIMIZES `heuristic` (the
        distance-to-goal) within the model.  Used when the hard goal predicate
        isn't reachable in the model yet: the agent still DESCENDS toward the
        goal, and — unlike greedy single-step descent — can step temporarily
        "away" (uphill) to get around an obstacle, because it searches multi-step
        for the lowest-distance reachable state.  Returns the path to that state,
        or None if no action sequence improves on the start."""
        h0 = heuristic(start)
        seen = {start}
        q = deque([(start, [])])
        best = (h0, [])
        nodes = 0
        while q and nodes < max_nodes:
            st, path = q.popleft(); nodes += 1
            for a in actions:
                ns, _ = self.simulate(st, a)
                if ns is None or ns in seen:
                    continue
                seen.add(ns)
                h = heuristic(ns)
                if h < best[0]:
                    best = (h, path + [a])
                q.append((ns, path + [a]))
        return best[1] if best[1] and best[0] < h0 else None

    def control_rules(self, state):
        """Explicit per-action CONTROL RULES at `state`: what the learned model
        predicts each action does to each entity ("if I do ACTIONx here, block 1
        moves (dr,dc) and block 2 moves (dr,dc)").  This is the inspectable,
        backward-reasoning-ready form of the operators."""
        rules = {}
        for a in sorted(self.ops):
            ns, conf = self.simulate(state, a)
            if ns is None:
                continue
            disp = {}
            for e in state.entities:
                ne = ns.get(e.eid)
                if ne is not None:
                    disp[e.eid] = (ne.pos[0] - e.pos[0], ne.pos[1] - e.pos[1])
            rules[a] = {'displacement': disp, 'confidence': conf}
        return rules

    def pursue(self, perceive, step, actions, goal, heuristic, *, budget=200,
               plan_nodes=20000, max_stuck=6, log=None):
        """Goal-directed control that COMMITS and does not wander.

        Each turn it reasons backward from the goal: using the learned control
        rules it predicts every action's effect and scores it by how much it
        REDUCES the goal-distance (an action that moves both blocks beneficially
        scores higher than one that helps only one).  It takes the most
        beneficial action.  If no single action helps, it plans a multi-step
        descending path.  Only if the model finds NO progress does it explore —
        the least-tried action, to learn a new control rule — and it gives up
        after `max_stuck` consecutive explorations that merely revisit known
        states (genuinely stuck), instead of drifting away from the goal.
        """
        used = 0; stuck = 0; seen = set(); acount = {a: 0 for a in actions}
        while used < budget:
            st = perceive()
            if goal(st):
                return True, used
            seen.add(st)
            h0 = heuristic(st)
            beneficial = []
            for a in actions:                       # backward reasoning: score by benefit
                ns, _ = self.simulate(st, a)
                if ns is not None and h0 - heuristic(ns) > 1e-9:
                    beneficial.append((h0 - heuristic(ns), a))
            if beneficial:
                beneficial.sort(reverse=True); a = beneficial[0][1]; commit = True
            else:
                plan = self.plan_toward_min(st, actions, heuristic, max_nodes=plan_nodes)
                if plan:
                    a = plan[0]; commit = True
                else:
                    a = min(actions, key=lambda x: acount[x]); commit = False
            before = perceive()
            ns, score, done = step(a); used += 1; acount[a] += 1
            self.observe(before, a, ns)
            if log:
                log(used, a, commit, heuristic(ns), score, done)
            if done or goal(ns):
                return True, used
            if commit:
                stuck = 0
            elif ns in seen:
                stuck += 1
                if stuck >= max_stuck:
                    return False, used              # genuinely stuck — don't wander
            else:
                stuck = 0                            # productive exploration (new state)
        return False, used

    # -- model-predictive control -----------------------------------------
    def solve(self, perceive, step, actions, goal, *, budget=300, explore=True,
              plan_nodes=5000, heuristic=None):
        """Drive a real environment to `goal` with bounded real actions.

        perceive() -> WorldState ; step(action) -> (WorldState, score, done).
        Returns (won: bool, actions_used: int).  No environment forking/resets.
        plan_nodes caps each interleaved plan search so a fruitless plan (while
        the model is still incomplete) fails fast and falls through to explore.
        """
        used = 0; visits = {}; tried = set()
        while used < budget:
            st = perceive()
            if goal(st):
                return True, used
            plan = self.plan(st, actions, goal, max_nodes=plan_nodes)
            # If the exact goal isn't reachable in the model, still DESCEND
            # toward it (minimize the heuristic) rather than wandering — the
            # tight goal-following that greedy/novelty exploration lacks.
            if not plan and heuristic is not None:
                plan = self.plan_toward_min(st, actions, heuristic,
                                            max_nodes=plan_nodes)
            if plan:
                for a in plan:
                    before = perceive(); pred, _ = self.simulate(before, a)
                    tried.add((before, a))
                    ns, score, done = step(a); used += 1
                    self.observe(before, a, ns)
                    visits[ns] = visits.get(ns, 0) + 1
                    if done or goal(ns):
                        return True, used
                    if pred is None or ns != pred:     # diverged -> replan
                        break
                    if used >= budget:
                        return False, used
            elif explore:
                # Exploration that can't be fooled by the model's own wrong
                # beliefs: prefer actions NOT YET TRIED from this state (a
                # predicted no-op is exactly where the model may be wrong, e.g.
                # an action blocked at one place but not another), then break
                # ties toward the least-visited predicted state.
                ranked = []
                for a in actions:
                    ns_pred, _ = self.simulate(st, a)
                    untried = 0 if (st, a) not in tried else 1
                    # heuristic (if given) pulls exploration TOWARD the temporary
                    # goal — prefer the action whose predicted state is closer to
                    # it — while still trying untried actions first to learn.
                    h = (heuristic(ns_pred) if heuristic and ns_pred is not None
                         else 0.0)
                    nov = -1 if ns_pred is None else visits.get(ns_pred, 0)
                    ranked.append(((untried, h, nov), a))
                ranked.sort(key=lambda x: x[0])
                a = ranked[0][1]
                tried.add((st, a))
                before = perceive()
                ns, score, done = step(a); used += 1
                self.observe(before, a, ns)
                visits[ns] = visits.get(ns, 0) + 1
                if done or goal(ns):
                    return True, used
            else:
                return False, used
        return False, used


# --------------------------------------------------------------------------
# Adapter: perception snapshot -> model state
# --------------------------------------------------------------------------
def world_state_from_entities(entities, *, id_map=None) -> WorldState:
    """Lift Module Entity objects (from detect_entities) into a WorldState.
    cls = top palette-signature key (a stable identity proxy); pos =
    centroid_cell.  id_map optionally maps entity_id -> temporal-registry
    track_id for cross-frame stable ids.  Backgrounds are dropped."""
    id_map = id_map or {}
    out = []
    for e in entities:
        if getattr(e, "is_background_primary", False) or \
                getattr(e, "is_background_secondary", False):
            continue
        sig = getattr(e, "visual_signature", None)
        cls = (sig[0][0],) if sig else ()
        out.append(EntityState(eid=id_map.get(e.entity_id, e.entity_id),
                               cls=cls, pos=tuple(e.centroid_cell)))
    return WorldState.of(out)


def world_state_from_registry(registry, turn, *, movers_only=True, cell_size=1,
                              max_track_pixels=None, min_move_fraction=None,
                              offset=0, n_cells=None) -> WorldState:
    """Build a STABLE world-model state from a temporal registry's TRACKS
    (persistent ids across frames), not a fresh per-frame detection.

    movers_only keeps only tracks that have ever moved/recolored — the
    CONTROLLABLE entities.  In a dense scene the static background fragments
    fluctuate in count and segmentation, but the agents are exactly the things
    that respond to actions, so requiring a motion history selects them with
    their stable ids.

    min_move_fraction tightens this further: keep a track only if it moves on at
    least that fraction of the turns it has existed.  The controllable agents
    respond on (nearly) every action, while transient maze-reconfiguration
    fragments move once or twice and stop — so a frequency threshold cleanly
    separates the two (agents ~0.95 vs fragments ~0.2), preventing the entity
    count from inflating under free play.

    Positions are quantized by cell_size (camera pixels-per-cell) so a slide is
    an integer cell delta the model can fit.
    """
    out = []
    for tr in registry.active_tracks(turn):
        o = tr.observation_at(turn) or tr.last_observation()
        if o is None:
            continue
        if max_track_pixels is not None and o.n_pixels > max_track_pixels:
            continue
        nmoves = sum(1 for ev in tr.behaviour_events
                     if ev.kind in ("moved", "recolor"))
        if movers_only and nmoves == 0:
            continue
        if min_move_fraction is not None:
            span = max(1, tr.last_seen_turn - tr.first_seen_turn + 1)
            if nmoves / span < min_move_fraction:
                continue
        r, c = o.centroid_cell
        if offset or n_cells is not None:
            # Offset-aware, nearest-cell quantization onto a 0..n_cells-1 grid:
            # accounts for the rendered board's letterbox border and avoids the
            # floor-division overflow/shift that lands entities off the grid.
            rr = round((r - offset) / cell_size)
            cc = round((c - offset) / cell_size)
            if n_cells is not None:
                rr = max(0, min(n_cells - 1, rr))
                cc = max(0, min(n_cells - 1, cc))
            pos = (rr, cc)
        else:
            pos = (r // cell_size, c // cell_size) if cell_size > 1 else (r, c)
        cls = (o.visual_signature[0][0],) if o.visual_signature else ()
        out.append(EntityState(eid=tr.track_id, cls=cls, pos=pos))
    return WorldState.of(out)


class LiveWorldStateSource:
    """Stateful perception bridge for the world model.

    Maintains a temporal registry across frames and exposes the perceive()/step()
    surface WorldModel.solve expects, returning the STABLE mover-state (agents by
    persistent track id) — the fix for feeding a forward model a clean, Markov
    state instead of a fresh, fluctuating per-frame detection.

    get_entities() -> detect_entities output for the current frame.
    do_action(action) -> (score, done).  No environment forking/resets.
    """

    def __init__(self, get_entities, do_action, *, cell_size=1, movers_only=True,
                 max_track_pixels=None, min_move_fraction=None, offset=0,
                 n_cells=None, game_id=""):
        from perception_loop_v2.temporal_registry import TemporalEntityRegistry
        self.get_entities = get_entities
        self.do_action = do_action
        self.cell_size = cell_size
        self.movers_only = movers_only
        self.max_track_pixels = max_track_pixels
        self.min_move_fraction = min_move_fraction
        self.offset = offset
        self.n_cells = n_cells
        self.reg = TemporalEntityRegistry(game_id=game_id)
        self.turn = 0
        self.reg.ingest_frame(0, get_entities())
        self._refresh()

    def _refresh(self):
        self._state = world_state_from_registry(
            self.reg, self.turn, movers_only=self.movers_only,
            cell_size=self.cell_size, max_track_pixels=self.max_track_pixels,
            min_move_fraction=self.min_move_fraction,
            offset=self.offset, n_cells=self.n_cells)

    def perceive(self):
        return self._state

    def step(self, action):
        score, done = self.do_action(action)
        self.turn += 1
        self.reg.ingest_frame(self.turn, self.get_entities())
        self._refresh()
        return self._state, score, done
