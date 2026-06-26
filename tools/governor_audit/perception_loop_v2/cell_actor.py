"""Symbolic-state actor that wires the VLM-extracted world model into
the cognitive_os Goal Forest, then picks the next action.

Pipeline:
  symbolic_state  -->  WorldState + GoalForest
                  -->  BFS over walkable cells
                  -->  highest-priority reachable goal
                  -->  first action in shortest path

If no goal is reachable, fall back to a single-step EXPLORATION move:
pick a cell adjacent to the agent that the world model classifies as
UNKNOWN (not in walkable_mask AND not in blocked_mask).  Trying to
move there resolves the unknown — succeeds (new walkable) or stays
put (new blocked).  After N exploration moves, BFS may find the
previously-unreachable goals.

Game-agnostic: knows nothing about bp35 specifically.  Works on any
domain with grid cells, an agent, walkable/blocked masks, and goal
cells.
"""
from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# cognitive_os is in the repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cognitive_os.conditions import AtPosition                 # noqa: E402
from cognitive_os.types import (                                # noqa: E402
    Goal, GoalForest, GoalNode, GoalStatus, NodeType,
    WorldState,
)
from cognitive_os.goal_forest import (                          # noqa: E402
    add_goal, candidates_by_priority, select_active_goal,
)


# ---------------------------------------------------------------------------
# Walkable / blocked mask reconstruction
# ---------------------------------------------------------------------------

# Roles whose bbox cells are walkable.  Floor, collectables, agent —
# all of these are stepped through during navigation.
_WALKABLE_ROLES = {"agent", "collectable", "scenery", "trigger_target"}
_BLOCKED_ROLES  = {"wall", "hud"}


def _bbox_cells(bb: list[int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bb
    return {(r, c) for r in range(r0, r1) for c in range(c0, c1)}


def build_masks(symbolic_state: dict, entities: list[dict],
                 ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    """Build (walkable, blocked) sets from entity bboxes + roles, with
    the symbolic_state's own newly_traversable / blocked_cells layered
    on top (those win over bbox-inferred classification because they
    are empirically confirmed by exploration)."""
    walkable: set[tuple[int, int]] = set()
    blocked:  set[tuple[int, int]] = set()
    # Pass 1: walkable from walkable-role bboxes
    for e in entities:
        bb = e.get("bbox_ticks_turn1")
        role = e.get("role_hypothesis", "unknown")
        if bb and role in _WALKABLE_ROLES:
            walkable |= _bbox_cells(bb)
    # Pass 2: blocked from wall-role bboxes (only if not already walkable)
    for e in entities:
        bb = e.get("bbox_ticks_turn1")
        role = e.get("role_hypothesis", "unknown")
        if bb and role in _BLOCKED_ROLES:
            for cell in _bbox_cells(bb):
                if cell not in walkable:
                    blocked.add(cell)
    # Pass 3: layered overrides from confirmed exploration
    ss = symbolic_state or {}
    for c in (ss.get("traversable_cells") or []):
        walkable.add(tuple(c))
    for c in (ss.get("blocked_cells") or []):
        blocked.add(tuple(c))
        walkable.discard(tuple(c))
    return walkable, blocked


# ---------------------------------------------------------------------------
# BFS
# ---------------------------------------------------------------------------


def bfs(start: tuple[int, int], goal: tuple[int, int],
        walkable: set[tuple[int, int]],
        step: int = 1,
        goal_tolerance: int = 0,
        ) -> Optional[list[tuple[int, int]]]:
    """4-connected BFS at the given `step` size.  Each move increments
    the position by ``+/-step`` along one axis.  A neighbour cell is
    eligible if it is in `walkable`.  `goal_tolerance` lets the
    search succeed when the path reaches any cell within Chebyshev
    distance `goal_tolerance` of the goal (useful when goals snap
    to a different sub-grid than the step).  Returns the shortest
    path (including start and goal-region cell) or None if
    unreachable."""
    def near_goal(cell: tuple[int, int]) -> bool:
        return (abs(cell[0] - goal[0]) <= goal_tolerance
                and abs(cell[1] - goal[1]) <= goal_tolerance)

    if near_goal(start):
        return [start]
    walkable_with_start = walkable | {start}
    prev: dict[tuple[int, int], tuple[int, int]] = {start: start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        cur = q.popleft()
        if near_goal(cur):
            path = [cur]
            while prev[path[-1]] != path[-1]:
                path.append(prev[path[-1]])
            return list(reversed(path))
        r, c = cur
        for dr, dc in ((-step, 0), (step, 0), (0, -step), (0, step)):
            nx = (r + dr, c + dc)
            if nx in walkable_with_start and nx not in prev:
                prev[nx] = cur
                q.append(nx)
    return None


# ---------------------------------------------------------------------------
# Goal Forest integration
# ---------------------------------------------------------------------------


def build_world_state(symbolic_state: dict, entities: list[dict]
                       ) -> tuple[WorldState, set[tuple[int, int]],
                                  set[tuple[int, int]]]:
    """Construct a minimal WorldState with the agent's current
    position, then populate a GoalForest with one Goal per
    goal_candidate_cell.  Goal priority is the inverse of the
    Manhattan distance from the agent — closer goals rank higher.

    Returns (ws, walkable_mask, blocked_mask).
    """
    agent_cell = tuple(symbolic_state.get("agent_cell") or ())
    if not agent_cell:
        raise ValueError("symbolic_state.agent_cell is required")
    walkable, blocked = build_masks(symbolic_state, entities)

    ws = WorldState(
        step=0,
        agent={"position": agent_cell},
        entities={},
        goal_forest=GoalForest(),
    )

    goal_cells = [tuple(c) for c in
                  (symbolic_state.get("goal_candidate_cells") or [])]
    for i, gc in enumerate(goal_cells):
        # Manhattan distance as a cheap priority signal.  BFS will
        # later compute the actual path length.
        manhattan = abs(gc[0] - agent_cell[0]) + abs(gc[1] - agent_cell[1])
        priority = 1.0 / (1.0 + float(manhattan))
        goal_id = f"reach_cell_{gc[0]}_{gc[1]}"
        root = GoalNode(
            id=f"{goal_id}_root",
            node_type=NodeType.ATOM,
            condition=AtPosition(pos=gc, entity_id="agent",
                                  tolerance=0.0),
        )
        goal = Goal(
            id=goal_id,
            root=root,
            priority=priority,
            source="symbolic_actor:goal_candidate",
            tags=frozenset({"task", "pickup"}),
        )
        add_goal(ws, goal)
    return ws, walkable, blocked


# ---------------------------------------------------------------------------
# Action picking
# ---------------------------------------------------------------------------


@dataclass
class ActionChoice:
    action:      str       # UP / DOWN / LEFT / RIGHT / NONE
    rationale:   str
    goal_id:     Optional[str]
    target_cell: Optional[tuple[int, int]]
    path_length: Optional[int]
    plan_kind:   str       # "goal_directed" | "exploration" | "no_move"


def _direction(from_cell: tuple[int, int],
                to_cell: tuple[int, int]) -> str:
    """Cardinal direction of a move.  Treats any pure-vertical or
    pure-horizontal delta (any magnitude) as a single direction —
    callers handle the magnitude via move_step."""
    dr, dc = to_cell[0] - from_cell[0], to_cell[1] - from_cell[1]
    if dr == 0 and dc == 0:
        return "NONE"
    if dc == 0:
        return "UP" if dr < 0 else "DOWN"
    if dr == 0:
        return "LEFT" if dc < 0 else "RIGHT"
    return f"JUMP{dr},{dc}"


def choose_action(symbolic_state: dict, entities: list[dict],
                   move_step: int = 1) -> ActionChoice:
    """Pick the next action by ranking Goal-Forest candidates by
    priority, BFS-pathing to each, and returning the FIRST step of
    the shortest path to the highest-priority REACHABLE goal.

    ``move_step`` is the cell delta produced by one in-game action,
    in TICK units.  The actor cannot know this a priori — the
    appropriate value is GAME-DEPENDENT and is discovered by the
    exploratory driver from the first successful agent move
    (the driver should set it to the observed delta in ticks; e.g.
    if the agent moved 6 ticks between two consecutive frames, set
    move_step=6).  The default value of 1 corresponds to
    "1 tick per action", which is the safest fallback when no
    learning has happened yet.

    If no goal is reachable, pick an unknown neighbour cell as an
    exploration target.
    """
    agent_cell = tuple(symbolic_state["agent_cell"])
    ws, walkable, blocked = build_world_state(symbolic_state, entities)

    # Optimistic walkable set: known-walkable + UNKNOWN cells within
    # the playfield (anything not EMPIRICALLY confirmed blocked).
    # bbox-derived "blocked" labels are WEAK HYPOTHESES — a region
    # the perception VLM labelled as scenery / wall from one frame
    # may turn out to be passable when the actor tries it — so we
    # only EXCLUDE cells listed in symbolic_state.blocked_cells, the
    # ones the actor itself has tried and bounced off.
    n_ticks = 64
    empirical_blocked = {tuple(c) for c
                         in (symbolic_state.get("blocked_cells") or [])}
    optimistic = set(walkable)
    for r in range(n_ticks):
        for c in range(n_ticks):
            if (r, c) not in empirical_blocked:
                optimistic.add((r, c))

    # Find reachable goals by BFS over OPTIMISTIC walkable.  For each,
    # also compute the path that would be taken.
    reachable: list[tuple[Goal, list[tuple[int, int]]]] = []
    unreachable: list[Goal] = []
    for gid in candidates_by_priority(ws):
        goal = ws.goal_forest.goals[gid]
        cond = goal.condition
        if not isinstance(cond, AtPosition):
            continue
        target = tuple(cond.pos)
        # BFS at the in-game move granularity (move_step).  Tolerance
        # is HALF move_step so the agent must actually traverse onto
        # the goal cell's lattice neighbourhood — preventing
        # premature "already at goal" calls when the agent is one
        # full step away (which would otherwise be within
        # tolerance=move_step, blocking real movement toward the
        # remaining goals).
        p = bfs(agent_cell, target, optimistic,
                step=move_step,
                goal_tolerance=max(1, move_step // 2))
        if p is None:
            unreachable.append(goal)
        else:
            reachable.append((goal, p))

    if reachable:
        # Goal Forest already ordered by priority; among reachable
        # ones, take the highest-priority (first one).
        # As a sanity check, also re-rank by ACTUAL path length so
        # an over-confident priority can't beat a much closer goal.
        reachable.sort(key=lambda gp: (
            len(gp[1]),                              # closer wins
            -gp[0].priority,                         # then higher priority
        ))
        # Skip past any "already-at" goals: BFS returns a length-1
        # path when the agent already sits within tolerance of a
        # goal cell.  Those goals are stale (we've already passed
        # through that cell and either collected it or it's
        # unreachable as a distinct tile) — try the next-priority
        # reachable goal whose BFS path is real movement.
        goal, path = next(
            ((g, p) for g, p in reachable if len(p) > 1),
            (None, None),
        )
        if goal is None:
            return ActionChoice(
                action="NONE",
                rationale=("all reachable goals coincide with the "
                           "agent's current cell — nothing else to do"),
                goal_id=reachable[0][0].id,
                target_cell=tuple(reachable[0][0].condition.pos),
                path_length=0, plan_kind="no_move",
            )
        # The simulator moves `move_step` ticks per game action in a
        # cardinal direction.  Take the FIRST cardinal direction the
        # path heads in (path[0] -> path[1]), and report the actual
        # in-game destination cell = agent_cell + move_step * unit.
        direction = _direction(agent_cell, path[1])
        unit = {"UP": (-1, 0), "DOWN": (1, 0),
                "LEFT": (0, -1), "RIGHT": (0, 1)}.get(direction)
        if unit is None:
            return ActionChoice(
                action="NONE",
                rationale=(f"BFS path first step is not cardinal "
                           f"(from {agent_cell} to {path[1]})"),
                goal_id=goal.id, target_cell=tuple(goal.condition.pos),
                path_length=len(path) - 1, plan_kind="no_move",
            )
        ingame_target = (agent_cell[0] + unit[0] * move_step,
                          agent_cell[1] + unit[1] * move_step)
        return ActionChoice(
            action=direction,
            rationale=(f"BFS optimistic path to {goal.id} = "
                       f"{len(path) - 1} ticks; first cardinal "
                       f"step is {direction}; in-game move lands at "
                       f"{ingame_target}"),
            goal_id=goal.id,
            target_cell=tuple(goal.condition.pos),
            path_length=len(path) - 1,
            plan_kind="goal_directed",
        )

    # Exploration fallback: rank the four cardinal directions by
    # how much they reduce Manhattan distance to the nearest
    # unreachable goal, AFTER eliminating directions whose
    # destination cell is already confirmed BLOCKED.  Stepping by
    # ``move_step`` ticks matches the simulator's per-action delta
    # so the blocked-check is accurate.
    if unreachable:
        target_dir = min(unreachable,
                         key=lambda g: (
                             abs(g.condition.pos[0] - agent_cell[0])
                             + abs(g.condition.pos[1] - agent_cell[1])
                         ))
        gr, gc = target_dir.condition.pos
        ar, ac = agent_cell
        candidates = [
            ("UP",    (ar - move_step, ac)),
            ("DOWN",  (ar + move_step, ac)),
            ("LEFT",  (ar, ac - move_step)),
            ("RIGHT", (ar, ac + move_step)),
        ]
        def manhattan(cell: tuple[int, int]) -> int:
            return abs(cell[0] - gr) + abs(cell[1] - gc)
        # Rank: prefer NOT-blocked; among those, prefer cells closer
        # to the goal; among ties, prefer KNOWN-WALKABLE (safe to
        # step into) over UNKNOWN (which gathers info but might
        # block).  In a stale loop the actor would still pick the
        # closest-to-goal known-walkable cell so it can keep making
        # progress instead of bouncing off the same wall.
        def rank(direction_cell):
            _, cell = direction_cell
            is_blocked = cell in blocked
            is_walkable = cell in walkable
            return (
                1 if is_blocked else 0,
                manhattan(cell),
                0 if is_walkable else 1,
            )
        candidates.sort(key=rank)
        # If even the best candidate is blocked, we're truly stuck:
        # report it so the caller can stop the loop.
        direction, next_cell = candidates[0]
        if next_cell in blocked:
            return ActionChoice(
                action="NONE",
                rationale=(f"all four directions blocked from "
                           f"{agent_cell}; cannot make progress "
                           f"toward {target_dir.id}"),
                goal_id=target_dir.id,
                target_cell=target_dir.condition.pos,
                path_length=None, plan_kind="no_move",
            )
        if next_cell in walkable:
            status = "walkable"
        else:
            status = "unknown"
        return ActionChoice(
            action=direction,
            rationale=(f"no goal reachable; explore toward "
                       f"{target_dir.id} at {target_dir.condition.pos}; "
                       f"next cell {next_cell} is {status}"),
            goal_id=target_dir.id,
            target_cell=target_dir.condition.pos,
            path_length=None,
            plan_kind="exploration",
        )

    return ActionChoice(
        action="NONE", rationale="no goals at all",
        goal_id=None, target_cell=None,
        path_length=None, plan_kind="no_move",
    )


# ---------------------------------------------------------------------------
# CLI test harness — load a reply JSON, show what action would be picked
# ---------------------------------------------------------------------------


def _format_summary(symbolic_state: dict, entities: list[dict],
                     choice: ActionChoice) -> str:
    walkable, blocked = build_masks(symbolic_state, entities)
    n_walk = len(walkable)
    n_blk = len(blocked)
    n_goals = len(symbolic_state.get("goal_candidate_cells") or [])
    return (
        f"  agent_cell:        {symbolic_state.get('agent_cell')}\n"
        f"  walkable cells:    {n_walk}\n"
        f"  blocked cells:     {n_blk}\n"
        f"  candidate goals:   {n_goals}\n"
        f"\n"
        f"  ACTION:            {choice.action}\n"
        f"  plan_kind:         {choice.plan_kind}\n"
        f"  goal_id:           {choice.goal_id}\n"
        f"  target_cell:       {choice.target_cell}\n"
        f"  path_length_ticks: {choice.path_length}\n"
        f"  rationale:         {choice.rationale}\n"
    )


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reply_path", type=Path,
                        help="Path to a call_NNN_reply.consumed.txt "
                             "containing entities + symbolic_state")
    parser.add_argument("--move-step", type=int, default=1,
                        help="Tick-cell delta per one in-game action. "
                             "This is GAME-DEPENDENT; pass the value "
                             "observed from the first successful agent "
                             "move (e.g. 6 if the agent moved 6 ticks "
                             "between two consecutive frames).  "
                             "Default 1 is the safest fallback when no "
                             "exploration-derived value is available.")
    args = parser.parse_args()

    data = json.loads(args.reply_path.read_text(encoding="utf-8"))
    entities = data.get("entities", [])
    ss = data.get("symbolic_state") or {}
    if not ss.get("agent_cell"):
        print("no symbolic_state.agent_cell — refusing to run actor")
        return

    print(f"== Actor on {args.reply_path.name} ==")
    choice = choose_action(ss, entities, move_step=args.move_step)
    print(_format_summary(ss, entities, choice))


if __name__ == "__main__":
    main()
