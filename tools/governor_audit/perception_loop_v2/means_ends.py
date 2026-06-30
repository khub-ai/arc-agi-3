"""means_ends.py -- general Means-Ends Analysis (MEA): reduce the DIFFERENCE between the current state
and the goal by selecting the operator that shrinks it, attacking preconditions first.

GPS in COS form.  COS's goal instincts were piecemeal MEA, each a corner of the same loop:
  - reach        : POSITION difference  -> move operators
  - match_goal   : SIZE / ORIENTATION   -> grow / shrink / rotate
  - conform      : VALUE / TYPE         -> set / recolour
This unifies them behind one controller:
  1. DIFFERENCE TYPOLOGY -- pluggable DETECTORS, each comparing current<->goal along one dimension and
     emitting a quantified, salience-ranked Difference plus the OPERATOR CLASS(es) that reduce it.
  2. DIFFERENCE->OPERATOR map -- carried on each Difference; the class (GROW/ROTATE/DOWN/...) is
     structural here, and a learned operator-effect map (operator_kb / controlled-experiment) can
     override it.  The actor resolves a class to the game's concrete action.
  3. MEA LOOP -- order the differences so a difference's PRECONDITIONS are reduced first (navigation
     before transform; orient before grow), then emit the reduction plan; recompute as the world moves.

Domain-agnostic: the same loop drives any game, and the robotics manipulate/recognise layer COS is a
restriction of.  The substrate MEASURES differences + names operator classes; the actor VERIFIES (the
score is the judge).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

import match_goal as _mg

# reduction ORDER per dimension: lower = reduce first.  Navigation (reach the goal) precedes
# transforms (become the goal); orient precedes grow (orient while small, then enlarge).  Not a
# match knob -- a precondition ordering (you must be AT/IN the goal before matching its shape/size).
_ORDER = {"reachability": -1, "position": 0, "orientation": 1, "colour": 1, "count": 1, "size": 2, "containment": 0}
_POS_TOL = 2          # ticks: centres within this are "co-located" (documented tolerance, not a detector knob)


@dataclass
class Difference:
    dimension: str
    magnitude: float
    operators: List[str]
    detail: str = ""
    order: int = field(default=99)

    def __post_init__(self):
        if self.order == 99:
            self.order = _ORDER.get(self.dimension, 1)


@dataclass
class OpNode:
    """AND node of the MEA backward chain -- an operator that reduces a goal's
    difference, APPLICABLE only when ALL its preconditions (sub-goals) hold.
    ``action`` is the executable step for a leaf (a primitive move/push).  ``feasible``
    is False for an alternative kept for VISIBILITY but unusable here (e.g. a direct
    walk through a barrier) -- the traversal skips it to the next OR alternative."""
    op: str
    preconds: List["GoalNode"] = field(default_factory=list)
    action: Optional[dict] = None
    feasible: bool = True
    note: str = ""


@dataclass
class GoalNode:
    """OR node of the MEA backward chain -- a (sub)goal reduced by ANY ONE of its
    alternative operators (``ops``).  ``achieved`` when no difference remains.  A tree
    of GoalNode/OpNode IS the backward chain: goal -> [operator alternatives (OR)] ->
    each operator's preconditions (AND) -> ... -> primitive actions.  Because every
    goal carries its alternatives, the substrate can SHOW the chain and EXPLORE a
    different branch when one fails."""
    desc: str
    ops: List["OpNode"] = field(default_factory=list)
    achieved: bool = False


def _center(bbox):
    r0, c0, r1, c1 = [float(v) for v in bbox]
    return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)


# ---- default difference detectors (mover_entity, goal_entity) -> List[Difference] -----------------
def detect_position(mover, goal, tol: float = _POS_TOL) -> List[Difference]:
    mb, gb = mover.get("bbox_ticks_turn1"), goal.get("bbox_ticks_turn1")
    if not (mb and gb):
        return []
    mr, mc = _center(mb)
    gr, gc = _center(gb)
    dr, dc = gr - mr, gc - mc
    ops = []
    if dr > tol:
        ops.append("DOWN")
    elif dr < -tol:
        ops.append("UP")
    if dc > tol:
        ops.append("RIGHT")
    elif dc < -tol:
        ops.append("LEFT")
    if not ops:
        return []
    return [Difference("position", abs(dr) + abs(dc), ops,
                       detail=f"mover centre {(round(mr),round(mc))} vs goal {(round(gr),round(gc))}")]


def detect_transform(mover, goal) -> List[Difference]:
    """SIZE + ORIENTATION differences, via match_goal (reused, not duplicated)."""
    plan = _mg.transform_plan(mover.get("bbox_ticks_turn1"), mover.get("shape_sig"),
                              goal.get("bbox_ticks_turn1"), goal.get("shape_sig"))
    out = []
    if plan["rotate_action"]:
        out.append(Difference("orientation", round(plan["shape_sim"] - plan["oriented_sim"], 3),
                              ["ROTATE"], detail=f"pose-match {plan['shape_sim']:.2f} vs current "
                                                 f"{plan['oriented_sim']:.2f}"))
    if plan["size_action"]:
        out.append(Difference("size", round(abs(1.0 - plan["size_ratio"]), 3), [plan["size_action"]],
                              detail=f"extent ratio {plan['size_ratio']}"))
    return out


def detect_colour(mover, goal) -> List[Difference]:
    mc, gc = (mover.get("color") or ""), (goal.get("color") or "")
    if not (mc and gc and mc.lower() != gc.lower()):
        return []
    # colour-match is the LAST difference: it only applies when the mover is meant to BECOME the goal
    # (already matches its extent).  A tiny mover that FILLS a differently-coloured container keeps its
    # own colour -- so skip colour while a SIZE difference remains.
    if _mg.transform_plan(mover.get("bbox_ticks_turn1"), mover.get("shape_sig"),
                          goal.get("bbox_ticks_turn1"), goal.get("shape_sig")).get("size_action"):
        return []
    return [Difference("colour", 1.0, ["RECOLOUR"], detail=f"{mc} -> {gc}")]


def reachability_detector(walkable, move_step: int = 1) -> Callable:
    """Factory -> a difference detector ``(mover, goal) -> List[Difference]`` that
    runs a PATH SEARCH over ``walkable`` (the set of (row,col) cells the agent may
    occupy) at the agent's move granularity.  If NO path exists -- a barrier blocks
    the agent's CURRENT motion -- it emits a precondition-order REACHABILITY
    difference whose operator classes name the modality-change / intermediary
    MACRO-operators, so the loop does NOT conclude 'impossible' (the seeded
    reachability_is_motion_relative + act_through_intermediary priors).  If a path
    exists it stays silent and detect_position drives.  This is the one piece that
    converts a dead-end into a reducible difference.

    ``walkable`` must be the OPTIMISTIC set (exclude only EMPIRICALLY-bounced cells,
    never weak bbox 'wall' guesses) so a real path is never pre-emptively declared
    blocked -- the detector fires only on a genuine barrier.  Game-agnostic; fully
    guarded (any failure -> no difference, never breaks the loop)."""
    def _detect(mover, goal) -> List[Difference]:
        try:
            mb, gb = mover.get("bbox_ticks_turn1"), goal.get("bbox_ticks_turn1")
            if not (mb and gb) or not walkable:
                return []
            mr = (round((mb[0] + mb[2]) / 2), round((mb[1] + mb[3]) / 2))
            gr = (round((gb[0] + gb[2]) / 2), round((gb[1] + gb[3]) / 2))
            from cell_actor import bfs as _bfs
            step = max(1, int(move_step or 1))
            if _bfs(mr, gr, set(walkable), step=step, goal_tolerance=max(1, step // 2)):
                return []                      # reachable -> let detect_position drive
            mag = float(abs(gr[0] - mr[0]) + abs(gr[1] - mr[1]))
            return [Difference("reachability", mag,
                               ["CHANGE_MODALITY", "VIA_INTERMEDIARY"],
                               detail=f"no path {mr}->{gr} under current motion (barrier)")]
        except Exception:
            return []
    return _detect


_DEFAULT_DETECTORS: List[Callable] = [detect_position, detect_transform, detect_colour]


class MeansEnds:
    def __init__(self, detectors: Optional[List[Callable]] = None):
        self.detectors = list(detectors) if detectors is not None else list(_DEFAULT_DETECTORS)
        self.expanders: dict = {}

    def register(self, detector: Callable) -> None:
        """Add a difference detector (current, goal) -> List[Difference] -- e.g. a count or conform
        detector -- so new difference dimensions plug into the same loop."""
        self.detectors.append(detector)

    def register_expander(self, op_name: str, fn: Callable) -> None:
        """Register a MACRO-operator expander.  When plan() reaches ``op_name`` and a
        scene ``context`` is supplied, it calls ``fn(mea, diff, current, goal, context,
        depth) -> List[str]`` and SPLICES the returned sub-plan in place of the macro
        name -- recursing the MEA for any nested reach.  This is the backward-chaining
        step: a high-level operator (e.g. VIA_INTERMEDIARY) opens a sub-goal."""
        self.expanders[op_name] = fn

    def analyze(self, current, goal) -> List[Difference]:
        diffs: List[Difference] = []
        for d in self.detectors:
            try:
                diffs.extend(d(current, goal) or [])
            except Exception:
                continue
        # MEA ordering: preconditions first (by order), then most-significant difference first
        diffs.sort(key=lambda x: (x.order, -x.magnitude))
        return diffs

    def plan(self, current, goal, context=None, _depth: int = 0) -> List[str]:
        """The ordered operator-class plan that reduces every difference (preconditions
        first).  When a scene ``context`` is supplied, a MACRO operator with a
        registered expander is EXPANDED into its concrete sub-plan (recursing the MEA
        for any nested reach) instead of being emitted as a bare class name -- this is
        the backward-chaining.  Without context the behaviour is unchanged (the raw
        operator classes are returned)."""
        out: List[str] = []
        for d in self.analyze(current, goal):
            for op in d.operators:
                exp = self.expanders.get(op)
                if exp is not None and context is not None and _depth < 4:
                    try:
                        sub = exp(self, d, current, goal, context, _depth) or []
                    except Exception:
                        sub = []
                    for s in sub:
                        if s not in out:
                            out.append(s)
                elif op not in out:
                    out.append(op)
        return out

    def plan_conjunctive(self, deliveries, context=None) -> List[str]:
        """Plan a CONJUNCTIVE goal -- seat each cargo in its slot.  ``deliveries`` is a
        list of {mover, goal} pairs; each is planned via plan() (so each gets the
        macro-expansion).  Then the stranding-ordering prior is applied: a delivery
        that CONSUMES the agent (a direct walk-in -- no PUSH in its sub-plan) is
        ordered AFTER every delivery that leaves the agent free (a PUSH through an
        intermediary), so the agent is not stranded in a slot it still needs to act
        from.  Returns the concatenated, ordered operator-class plan."""
        planned = []
        for dv in deliveries:
            sub = self.plan(dv.get("mover"), dv.get("goal"), context)
            consumes_agent = not any(str(s).startswith("PUSH") for s in sub)
            planned.append((consumes_agent, sub))
        planned.sort(key=lambda t: 1 if t[0] else 0)      # agent-free first, consuming last (stable)
        out: List[str] = []
        for _, sub in planned:
            out.extend(sub)
        return out

    def next_step(self, current, goal, context=None) -> Optional[dict]:
        """The IMMEDIATE concrete step for an EXECUTOR -- the interface that makes MEA
        AUTHORITATIVE.  Resolve the highest-precedence difference to a single action
        dict, RETAINING the target cell that plan() discards so a pursuit can execute
        MEA's decision instead of re-deriving one:
          {kind:'WALK', dir, target_cell, via}  -> walk the mover toward target_cell
          {kind:'PUSH', dir, target_cell, via}  -> mover is at the intermediary's push
                                                   side; move 'dir' to push it on
          {kind:'OP',   dir, target_cell:None}  -> a non-spatial op class (ROTATE/...)
        Returns ``None`` when no difference remains (goal met) or it cannot be resolved.
        Game-agnostic; fully guarded."""
        try:
            diffs = self.analyze(current, goal)
        except Exception:
            return None
        if not diffs:
            return None
        d = diffs[0]
        # reachability -> deliver through the intermediary (the act_through_intermediary
        # prior): WALK to its push side, then PUSH once positioned there.
        if d.dimension == "reachability" and context is not None:
            sel = _select_intermediary(current, goal, context)
            if sel:
                inter, push_side, dirname = sel
                ab = current.get("bbox_ticks_turn1")
                ac = _bbox_centre(ab) if ab else None
                at_push_side = ac is not None and (
                    abs(ac[0] - push_side[0]) + abs(ac[1] - push_side[1]) <= 1)
                if at_push_side:
                    return {"kind": "PUSH", "dir": dirname, "via": inter.get("name"),
                            "target_cell": _bbox_centre(inter["bbox_ticks_turn1"])}
                return {"kind": "WALK", "dir": None, "via": inter.get("name"),
                        "target_cell": push_side}
            # no intermediary -> degrade to walking at the goal (honest; it re-blocks)
        gb = goal.get("bbox_ticks_turn1")
        first = d.operators[0] if d.operators else None
        if d.dimension in ("reachability", "position") and gb:
            return {"kind": "WALK", "dir": first, "via": None,
                    "target_cell": _bbox_centre(gb)}
        if first:
            return {"kind": "OP", "dir": first, "via": None, "target_cell": None}
        return None

    def goal_tree(self, current, goal, context=None, _depth: int = 0, _desc=None,
                  cargo=None):
        """Build the explicit AND-OR backward-chain tree for reducing current -> goal:
        a GoalNode (OR over operator alternatives), each OpNode (AND over its
        preconditions).  At a REACHABILITY goal the OR alternatives are [the DIRECT
        walk -- kept for visibility, marked infeasible because a barrier blocks it] +
        [VIA each pushable INTERMEDIARY -> AND(reach its push side, push it)].  This is
        the backward chain the substrate can SHOW (render_tree) and whose alternatives
        it can EXPLORE when one fails.  Depth-guarded; fully guarded.

        ``cargo`` (a {name, bbox_ticks_turn1} entity) makes this a CARGO-FILL goal:
        the slot is filled by SEATING the cargo, not by the agent walking in.  Then
        the reduction is unconditionally 'push the cargo into the slot' (reach its
        push side, push it) -- the agent walking into the slot is kept only as an
        INFEASIBLE alternative (it would deliver the wrong object).  This is what lets
        MEA pursue 'push the block into the hole' even when the agent COULD reach the
        hole itself; without it the planner walks the agent into the slot and the
        run stalls (the seeded figure_ground_complement / mark-correspondence priors,
        made executable)."""
        desc = _desc or (goal.get("name") if isinstance(goal, dict) else None) or "goal"
        gb0 = goal.get("bbox_ticks_turn1") if isinstance(goal, dict) else None
        if cargo is not None and gb0 and isinstance(cargo, dict):
            cb = cargo.get("bbox_ticks_turn1")
            if cb:
                gc0 = _bbox_centre(gb0)
                cc = _bbox_centre(cb)
                if abs(cc[0] - gc0[0]) + abs(cc[1] - gc0[1]) <= _POS_TOL:
                    return GoalNode(desc, [], achieved=True)     # cargo already seated
                if _depth < 6:
                    move_step = int((context or {}).get("move_step") or 1)
                    push_side, dirname = _push_geometry(cb, gc0, move_step)
                    nm = cargo.get("name") or "cargo"
                    reach_goal = {"name": f"push side of {nm}",
                                  "bbox_ticks_turn1": [push_side[0] - 1, push_side[1] - 1,
                                                       push_side[0] + 1, push_side[1] + 1]}
                    reach_sub = self.goal_tree(current, reach_goal, context, _depth + 1,
                                               _desc=f"reach push side of {nm}")
                    push_sub = GoalNode(
                        f"push {nm} {dirname} into {desc}",
                        [OpNode(f"PUSH_{dirname}",
                                action={"kind": "PUSH", "dir": dirname, "via": nm})])
                    ops = [
                        OpNode("WALK", action={"kind": "WALK", "target_cell": gc0},
                               feasible=False,
                               note=f"the slot is filled by {nm}, not the agent"),
                        OpNode("VIA_INTERMEDIARY", preconds=[reach_sub, push_sub],
                               note=f"seat cargo {nm}"),
                    ]
                    return GoalNode(desc, ops)
        try:
            diffs = self.analyze(current, goal)
        except Exception:
            diffs = []
        if not diffs:
            return GoalNode(desc, [], achieved=True)
        if _depth >= 6:
            return GoalNode(desc, [])
        d = diffs[0]
        gb = goal.get("bbox_ticks_turn1") if isinstance(goal, dict) else None
        gc = _bbox_centre(gb) if gb else None
        ops: List[OpNode] = []
        if d.dimension == "reachability":
            # OR alternative #1 -- the direct walk, KEPT for visibility but infeasible
            # (a barrier blocks it); the chain shows we considered and rejected it.
            ops.append(OpNode("WALK", action={"kind": "WALK", "target_cell": gc},
                              feasible=False, note="direct walk -- blocked by a barrier"))
            # OR alternatives #2.. -- via each candidate intermediary (the
            # act_through_intermediary prior): AND(reach its push side, push it on).
            for inter, push_side, dirname in _all_intermediaries(current, goal, context):
                nm = inter.get("name") or "intermediary"
                reach_goal = {"name": f"push side of {nm}",
                              "bbox_ticks_turn1": [push_side[0] - 1, push_side[1] - 1,
                                                   push_side[0] + 1, push_side[1] + 1]}
                reach_sub = self.goal_tree(current, reach_goal, context, _depth + 1,
                                           _desc=f"reach push side of {nm}")
                push_sub = GoalNode(
                    f"push {nm} {dirname} into {desc}",
                    [OpNode(f"PUSH_{dirname}",
                            action={"kind": "PUSH", "dir": dirname, "via": nm})])
                ops.append(OpNode("VIA_INTERMEDIARY", preconds=[reach_sub, push_sub],
                                  note=f"via {nm}"))
        elif d.dimension == "position":
            ops.append(OpNode(d.operators[0] if d.operators else "MOVE",
                              action={"kind": "WALK", "target_cell": gc},
                              note="walk toward the goal"))
        else:
            first = d.operators[0] if d.operators else "OP"
            ops.append(OpNode(first, action={"kind": "OP", "dir": first}))
        return GoalNode(desc, ops)

    def goal_tree_conjunctive(self, deliveries, context=None, win_desc: str = "WIN"):
        """The full WIN tree: the root is reduced by ONE operator 'fill all slots'
        whose AND-children are the per-delivery subtrees, STRANDING-ORDERED (a delivery
        that keeps the agent free -- a push -- before one that consumes it in a slot).
        ``deliveries`` = [{mover, goal}].  Returns the root GoalNode."""
        subs = []
        for dv in deliveries:
            g = dv.get("goal")
            nm = g.get("name") if isinstance(g, dict) else "slot"
            subs.append(self.goal_tree(dv.get("mover"), g, context, _desc=f"fill {nm}",
                                       cargo=dv.get("cargo")))
        subs.sort(key=lambda gn: 0 if _goal_keeps_agent_free(gn) else 1)
        return GoalNode(win_desc, [OpNode("fill_all_slots", preconds=subs)])

    def directive(self, current, goal, mover_name: str = "mover", goal_name: str = "goal") -> Optional[str]:
        diffs = self.analyze(current, goal)
        if not diffs:
            return None
        lines = [f"[MEANS-ENDS] reduce the differences between {mover_name} and {goal_name} in this "
                 f"order (attack preconditions first), then recompute:"]
        for d in diffs:
            lines.append(f"  - {d.dimension}: {', '.join(d.operators)}  ({d.detail})")
        lines.append("  PLAN: " + " -> ".join(self.plan(current, goal)))
        return "\n".join(lines)


def _bbox_centre(bb):
    return (round((bb[0] + bb[2]) / 2), round((bb[1] + bb[3]) / 2))


def _push_geometry(inter_bbox, goal_centre, move_step: int = 1):
    """Where the agent must STAND and which way to PUSH to drive ``inter_bbox``
    toward ``goal_centre``.  The push direction is the dominant axis from the
    intermediary to the goal; the agent stands one clear cell on the OPPOSITE
    side.  Returns ``(push_side_cell, dirname)``.  The single shared geometry for
    both the all-intermediaries scan and an explicit cargo delivery."""
    oc = _bbox_centre(inter_bbox)
    dr, dc = goal_centre[0] - oc[0], goal_centre[1] - oc[1]
    if abs(dc) >= abs(dr):
        dirname, unit = ("RIGHT", (0, 1)) if dc > 0 else ("LEFT", (0, -1))
    else:
        dirname, unit = ("DOWN", (1, 0)) if dr > 0 else ("UP", (-1, 0))
    half = int(max(inter_bbox[2] - inter_bbox[0], inter_bbox[3] - inter_bbox[1]) / 2.0)
    off = half + max(1, int(move_step or 1))
    push_side = (oc[0] - unit[0] * off, oc[1] - unit[1] * off)
    return push_side, dirname


def expand_change_modality(mea, diff, current, goal, context, depth) -> List[str]:
    """Marker macro for the reachability_is_motion_relative prior: 'a barrier-crossing
    modality exists -> do NOT conclude impossible'.  It contributes no operator of its
    own; the CONCRETE delivery is produced by the intermediary expander."""
    return []


def _all_intermediaries(current, goal, context):
    """EVERY controllable INTERMEDIARY (a pushable / slides-far object) that could
    carry to the goal, nearest the agent first -- each is an OR ALTERNATIVE in the goal
    tree.  Returns a list of ``(intermediary, push_side_cell, dirname)``.  ``context``
    = {objects:[{name, bbox_ticks_turn1, affordances}], move_step}.  Game-agnostic
    (keyed on an affordance, not a colour/id)."""
    objs = (context or {}).get("objects") or []
    move_step = int((context or {}).get("move_step") or 1)
    gb, ab = goal.get("bbox_ticks_turn1"), current.get("bbox_ticks_turn1")
    if not gb:
        return []
    gc = _bbox_centre(gb)
    ac = _bbox_centre(ab) if ab else None

    def _aff(o):
        return o.get("affordances") or {}
    cands = [o for o in objs
             if o.get("bbox_ticks_turn1")
             and (_aff(o).get("pushable") or _aff(o).get("slides_far"))]
    if ac is not None:                                       # nearest intermediary first
        cands.sort(key=lambda o: abs(_bbox_centre(o["bbox_ticks_turn1"])[0] - ac[0])
                   + abs(_bbox_centre(o["bbox_ticks_turn1"])[1] - ac[1]))
    out = []
    for inter in cands:
        push_side, dirname = _push_geometry(inter["bbox_ticks_turn1"], gc, move_step)
        out.append((inter, push_side, dirname))
    return out


def _select_intermediary(current, goal, context):
    """The single best INTERMEDIARY (nearest) -- the head of _all_intermediaries.
    Shared by expand_via_intermediary (PLANNING) and MeansEnds.next_step (EXECUTION)
    so there is exactly ONE selection that cannot drift from the tree's alternatives.
    Returns ``(intermediary, push_side_cell, dirname)`` or ``None``."""
    alls = _all_intermediaries(current, goal, context)
    return alls[0] if alls else None


def _goal_keeps_agent_free(gn) -> bool:
    """True if reducing this GoalNode keeps the agent FREE -- its first FEASIBLE
    operator is a push / via-intermediary -- vs CONSUMING it in a slot (a direct walk).
    Used to stranding-order the AND children of the win tree."""
    for op in getattr(gn, "ops", []) or []:
        if not op.feasible:
            continue
        return op.op == "VIA_INTERMEDIARY" or op.op.startswith("PUSH")
    return False


def render_tree(node, indent: int = 0) -> List[str]:
    """Render the AND-OR backward chain as indented lines -- the VISIBLE reasoning the
    substrate develops (GoalNode goals, their OR alternatives, each operator's AND
    preconditions).  Pass a GoalNode; returns a list of text lines."""
    pad = "  " * indent
    lines: List[str] = []
    if isinstance(node, GoalNode):
        tag = "  [achieved]" if node.achieved else ""
        lines.append(f"{pad}GOAL: {node.desc}{tag}")
        if len(node.ops) > 1:
            lines.append(f"{pad}  OR ({len(node.ops)} alternatives):")
        for op in node.ops:
            lines.extend(render_tree(op, indent + 1))
    elif isinstance(node, OpNode):
        mark = "" if node.feasible else "  (infeasible here)"
        nt = f" -- {node.note}" if node.note else ""
        act = ""
        if node.action:
            tgt = node.action.get("dir") or node.action.get("target_cell") or ""
            act = f"  => {node.action.get('kind')} {tgt}".rstrip()
        lines.append(f"{pad}via {node.op}{mark}{nt}{act}")
        if node.preconds:
            lines.append(f"{pad}  AND (preconditions):")
            for pc in node.preconds:
                lines.extend(render_tree(pc, indent + 2))
    return lines


def first_action(node, _path=None):
    """DFS the AND-OR tree to the first FEASIBLE, not-yet-achieved leaf; return
    ``(action_dict, active_branch_path)``.  Skips infeasible OR alternatives (e.g. a
    direct walk through a barrier) and descends an AND operator's first UNACHIEVED
    precondition -- the tree-driven, alternative-exploring traversal an executor runs.
    ``action_dict`` is None when nothing is actionable (goal met / all branches dead);
    ``active_branch_path`` is the list of goal descriptions on the chosen branch (the
    chain to surface)."""
    path = list(_path or [])
    if isinstance(node, GoalNode):
        if node.achieved:
            return None, path
        path = path + [node.desc]
        for op in node.ops:
            if op.feasible:
                a, p = first_action(op, path)
                if a is not None:
                    return a, p
        return None, path
    if isinstance(node, OpNode):
        for pc in node.preconds:
            if not pc.achieved:
                a, p = first_action(pc, path)
                if a is not None:
                    return a, p
        if node.action:
            return node.action, path
    return None, path


def expand_via_intermediary(mea, diff, current, goal, context, depth) -> List[str]:
    """Expand VIA_INTERMEDIARY (the act_through_intermediary prior) into a concrete
    sub-plan: pick a controllable INTERMEDIARY via _select_intermediary, RECURSE the
    MEA to reach its push side (the cell opposite the goal), then PUSH it toward the
    goal.  This recursion is the backward-chaining.  Returns the sub-plan (reach moves
    + a directional push), or [] if no suitable intermediary -- so the loop degrades to
    the direct move, never crashes."""
    sel = _select_intermediary(current, goal, context)
    if not sel:
        return []
    _inter, push_side, dirname = sel
    push_side_goal = {"name": "push_side",
                      "bbox_ticks_turn1": [push_side[0] - 1, push_side[1] - 1,
                                           push_side[0] + 1, push_side[1] + 1]}
    reach = mea.plan(current, push_side_goal, context, _depth=depth + 1)
    return reach + [f"PUSH_{dirname}"]


def directive_for_entities(entities, mea: Optional["MeansEnds"] = None) -> Optional[str]:
    """Pick the mover (agent) + the best scene goal from a grounded entity list and return the MEA
    difference-reduction directive, or ``None`` if either is missing.  Used by the driver each level."""
    mover = _mg._pick_mover(entities)
    if not mover:
        return None
    goal = _mg._pick_goal(entities, mover)
    if not goal:
        return None
    return (mea or MeansEnds()).directive(mover, goal, mover.get("name", "mover"),
                                          goal.get("name", "goal"))


def directive_for_correspondences(entities, pairs, mea: Optional["MeansEnds"] = None) -> Optional[str]:
    """STRUCTURE MAPPING: per-CORRESPONDENCE difference-reduction directives for the VLM's
    EXPLICIT source<->target match pairs (e.g. each tile to its goal row), instead of the
    single auto-picked mover/goal in ``directive_for_entities``.  Reduces EACH named
    correspondence (position/size/orientation/colour) so the actor knows which source to
    move/transform onto which target.  ``pairs`` is a list of (source_name, target_name).
    Returns a combined directive, or ``None`` if no pair resolves to two known entities."""
    by_name = {e.get("name"): e for e in (entities or []) if e.get("name")}
    m = mea or MeansEnds()
    blocks = []
    for a, b in (pairs or []):
        src, tgt = by_name.get(a), by_name.get(b)
        if src and tgt and src is not tgt:
            d = m.directive(src, tgt, a, b)
            if d:
                blocks.append(d)
    if not blocks:
        return None
    return ("[STRUCTURE MAPPING] make each SOURCE match its TARGET -- reduce every "
            "correspondence's differences:\n" + "\n".join(blocks))
