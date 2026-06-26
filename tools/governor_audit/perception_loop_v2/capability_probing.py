"""Layer D — Capability probing (early phase).

Fills in the action-effect table that Layer C (recoverability) consults,
including the inverse-action verdicts that decide one-way-ness.  D is
the part that *invokes* C's probes; C is the part that classifies them.

See docs/SPEC_visual_reasoning_substrate.md (Layer D).

Three probe families:
  1. Action-effect coverage — try each action in each qualitatively
     distinct state-class, record what relations it produces.
  2. Undo-probe — follow a state-changing action with the undo action
     and classify recoverability (delegates to recoverability.py).
  3. Undo-capability calibration — once per trial, determine whether
     the undo action is lossless / destructive / unavailable, so the
     burst knows whether per-move undo-probing is worth the budget.

Structure mirrors C: PURE helpers (coverage, gap selection, burst
gating) that are unit-testable, plus DRIVER functions that execute
real actions via injected step/perceive callables (mock-testable,
driver-ready).

Game-agnostic throughout: state-classes and verdicts come from
recoverability.py, which is itself pure geometry over open-vocab
entities.
"""
from __future__ import annotations

from typing import Callable, Optional

from world_knowledge import WorldKnowledge          # noqa: E402
import recoverability as C                            # noqa: E402


# Defaults (cross-game; per-game-budget can override)
DEFAULT_BURST_TURNS = 8
SAFE_CALIBRATION_PADDING = 0


# ---------------------------------------------------------------------------
# Undo-capability calibration (once per trial)
# ---------------------------------------------------------------------------


def undo_capability(world: WorldKnowledge) -> Optional[str]:
    """The cached per-trial undo verdict, or None if not yet calibrated."""
    return world.probe_state.get("undo_capability")


def calibrate_undo(world: WorldKnowledge,
                    safe_action: str,
                    step_fn: Callable[[str], None],
                    perceive_fn: Callable[[], None],
                    undo_action: str = "ACTION7") -> str:
    """Run ONE calibration probe to determine the global undo behavior
    in this game, cached on world.probe_state['undo_capability'].

    Procedure: snapshot, take a (known-safe) action, snapshot, take the
    undo action, snapshot.  Classify:
      - "unavailable"  — undo produced no observable change OR the
                          safe action itself produced no change (can't
                          tell; conservative)
      - "lossless"     — undo restored the pre-action state exactly
      - "destructive"  — undo changed state but did not restore it, or
                          restored geometry while regressing progress

    Returns the verdict string.  Idempotent: if already calibrated,
    returns the cached value without spending actions.
    """
    cached = world.probe_state.get("undo_capability")
    if cached is not None:
        return cached

    pre = C.exact_state_signature(world)
    step_fn(safe_action)
    perceive_fn()
    post_action = C.exact_state_signature(world)
    step_fn(undo_action)
    perceive_fn()
    post_undo = C.exact_state_signature(world)

    action_changed = not C.signatures_match(pre, post_action)
    undo_changed = not C.signatures_match(post_action, post_undo)
    restored = C.signatures_match(pre, post_undo)

    if not action_changed:
        # The "safe action" did nothing, so we can't observe undo at
        # all.  Conservatively report unavailable until a real probe
        # proves otherwise.
        verdict = "unavailable"
    elif not undo_changed:
        verdict = "unavailable"   # undo is inert
    elif restored:
        verdict = "lossless"
    else:
        verdict = "destructive"

    world.probe_state["undo_capability"] = verdict
    return verdict


# ---------------------------------------------------------------------------
# Coverage tracking (action x state-class cells)
# ---------------------------------------------------------------------------


def _cell_key(action: str, sclass: str) -> str:
    return f"{action}@@{sclass}"


def mark_covered(world: WorldKnowledge, action: str,
                  sclass: Optional[str] = None) -> None:
    """Record that (action, current-state-class) has had its effect
    observed.  Idempotent."""
    if sclass is None:
        sclass = C.state_class(world, action)
    covered = world.probe_state.setdefault("covered_cells", [])
    key = _cell_key(action, sclass)
    if key not in covered:
        covered.append(key)


def is_covered(world: WorldKnowledge, action: str,
                sclass: Optional[str] = None) -> bool:
    if sclass is None:
        sclass = C.state_class(world, action)
    return _cell_key(action, sclass) in world.probe_state.get(
        "covered_cells", [])


def coverage_fraction(world: WorldKnowledge,
                       available_actions: list[str]) -> float:
    """Fraction of action cells covered for the CURRENT state-class.
    A coarse progress signal for the burst's diminishing-returns
    termination."""
    if not available_actions:
        return 1.0
    sclass = None
    covered = 0
    for a in available_actions:
        if sclass is None:
            sclass = C.state_class(world, a)
        if is_covered(world, a, sclass):
            covered += 1
    return covered / len(available_actions)


# ---------------------------------------------------------------------------
# Gap-driven probe selection
# ---------------------------------------------------------------------------


def next_probe_action(world: WorldKnowledge,
                       available_actions: list[str],
                       undo_action: str = "ACTION7") -> Optional[str]:
    """Pick the most informative action to probe from the current
    state: the first available action whose (action, current-state-
    class) cell is not yet covered.  Excludes the undo action itself
    (probing undo-from-rest is calibration, handled separately).
    Returns None when every cell for this state-class is covered."""
    for a in available_actions:
        if a == undo_action or a == "NONE":
            continue
        if not is_covered(world, a):
            return a
    return None


def should_probe_recoverability(world: WorldKnowledge,
                                 action: str) -> bool:
    """Whether a recoverability (undo) probe is worth running for this
    action: only when undo is known to work AND this (action, state-
    class) has not already been given a verdict."""
    if world.probe_state.get("undo_capability") != "lossless":
        # destructive/unavailable undo can't cheaply verify
        # recoverability; skip the extra cost.
        return False
    return C.lookup_verdict(world, action) is None


# ---------------------------------------------------------------------------
# Opening-burst gating
# ---------------------------------------------------------------------------


def opening_burst_active(world: WorldKnowledge,
                         turn: int,
                         available_actions: list[str],
                         budget_turns: int = DEFAULT_BURST_TURNS,
                         coverage_target: float = 0.7) -> bool:
    """Is the opening burst still running?  Ends on EITHER the turn
    budget OR reaching the coverage target (diminishing returns)."""
    start = world.probe_state.get("burst_start_turn")
    if start is None:
        world.probe_state["burst_start_turn"] = turn
        start = turn
    if turn - start >= budget_turns:
        return False
    if coverage_fraction(world, available_actions) >= coverage_target:
        return False
    return True


# ---------------------------------------------------------------------------
# Burst driver — executes real actions via injected callables
# ---------------------------------------------------------------------------


def drive_opening_burst(world: WorldKnowledge,
                        available_actions: list[str],
                        step_fn: Callable[[str], None],
                        perceive_fn: Callable[[], None],
                        undo_action: str = "ACTION7",
                        budget_turns: int = DEFAULT_BURST_TURNS,
                        coverage_target: float = 0.7) -> dict:
    """Run the opening burst against the live game via injected
    step/perceive callables.

    1. Calibrate undo capability once (one safe probe).
    2. Loop: pick an uncovered (action, state-class) cell; execute it;
       mark covered; if undo is lossless, run a recoverability probe
       (which also restores the state, letting the next probe run from
       the same position cheaply).
    3. Stop on budget or coverage target.

    Returns a summary dict for logging.  step_fn executes one action;
    perceive_fn refreshes `world` to reflect the new frame (same path a
    normal turn uses).  Both injected for testability and driver reuse.
    """
    summary = {
        "undo_capability": None,
        "cells_covered": 0,
        "recoverability_verdicts": 0,
        "actions_spent": 0,
    }

    # 1. Calibrate undo with the first available non-undo action.
    safe = next(
        (a for a in available_actions
         if a not in (undo_action, "NONE")),
        None,
    )
    if safe is not None and undo_capability(world) is None:
        verdict = calibrate_undo(world, safe, step_fn, perceive_fn,
                                  undo_action)
        summary["undo_capability"] = verdict
        summary["actions_spent"] += 2   # action + undo
        # The calibration safe-action's effect is now observed.
        mark_covered(world, safe)
    else:
        summary["undo_capability"] = undo_capability(world)

    # 2. Coverage / recoverability loop.
    turn = world.turn
    guard = 0
    max_iters = budget_turns * max(1, len(available_actions))
    while opening_burst_active(world, turn, available_actions,
                                budget_turns, coverage_target):
        guard += 1
        if guard > max_iters:
            break   # safety backstop against a non-advancing loop
        action = next_probe_action(world, available_actions, undo_action)
        if action is None:
            break   # everything covered for this state-class

        if should_probe_recoverability(world, action):
            # Undo is lossless -> probe recoverability (move + undo),
            # which classifies AND restores the state.
            C.probe_recoverability(world, action, step_fn, perceive_fn,
                                    undo_action)
            summary["recoverability_verdicts"] += 1
            summary["actions_spent"] += 2
        else:
            # Just observe the action's effect (no undo available or
            # already have its verdict).
            step_fn(action)
            perceive_fn()
            summary["actions_spent"] += 1
        mark_covered(world, action)
        turn = world.turn

    summary["cells_covered"] = len(
        world.probe_state.get("covered_cells", []))
    summary["recoverability_table_size"] = len(world.inverse_actions)
    return summary


# ---------------------------------------------------------------------------
# Agent reach extent (measured, not eyeballed)
#
# The arm-reach misjudgments (false "can't reach above the row"; false "I'll
# insert the arm below the pair" when the pair is flush to the wall) come
# from the actor JUDGING reach spatially.  The substrate already computes
# clearance-to-boundary, but it never measured the AGENT's own travel range.
# These helpers do: walk the agent to both extremes of an axis and record
# the reachable span as an observed fact, then expose a pure feasibility
# check so "insert the arm on side S of target T" is grounded in
# (reach span) + (clearance on side S), never in eyeballing.
# Game-agnostic: the axis-actions are DISCOVERED and passed in, never
# hardcoded.
# ---------------------------------------------------------------------------


_AXIS_INDEX = {"row": 0, "col": 1}   # bbox = [top, left, bottom, right]


def _ref_coord(bbox: Optional[list], axis: str) -> Optional[int]:
    """The agent's reference coordinate along ``axis`` (top for row,
    left for col).  None if no bbox."""
    if not bbox:
        return None
    return bbox[_AXIS_INDEX[axis]]


def largest_mover(world: WorldKnowledge, axis: str) -> Optional[str]:
    """Name of the entity whose reference coordinate changed most between
    its last two observed bboxes along ``axis`` — i.e. the thing the last
    action actually moved (the agent, when a movement action was taken).
    Returns None if nothing moved."""
    best_name, best_delta = None, 0
    for name, rec in world.entities.items():
        hist = getattr(rec, "bbox_history", None)
        if not hist or len(hist) < 2:
            continue
        prev = _ref_coord(hist[-2][1], axis)
        curr = _ref_coord(hist[-1][1], axis)
        if prev is None or curr is None:
            continue
        d = abs(curr - prev)
        if d > best_delta:
            best_name, best_delta = name, d
    return best_name


def agent_reach(world: WorldKnowledge, axis: str = "row") -> Optional[dict]:
    """The cached measured reach span for ``axis`` ({'min','max','agent'}),
    or None if not yet probed."""
    return world.probe_state.get("agent_reach", {}).get(axis)


def reach_covers_lane(reach: Optional[dict], lane: int) -> bool:
    """Is ``lane`` within the measured reach span?  Unknown reach (None)
    is treated as 'do not assert un-reachability' -> True, so a missing
    measurement never manufactures a false 'cannot reach'."""
    if not reach:
        return True
    return reach["min"] <= lane <= reach["max"]


def insertion_feasible(target_ref: int,
                       side: str,
                       *,
                       clearance_cells: int,
                       reach: Optional[dict]) -> tuple[bool, str]:
    """Can the agent's body be inserted on ``side`` of a target whose
    reference coordinate is ``target_ref``?  PURE.

    Two grounded conditions, both required:
      * ROOM: there must be >=1 free lane on that side — i.e.
        ``clearance_cells >= 1``.  ``clearance_cells == 0`` means the
        target is flush against a boundary/blocker on that side, so there
        is no lane to insert into (this is the bottom-wall case the actor
        got wrong).
      * REACH: the lane just outside the target on that side must lie
        within the agent's measured reach span (this is the top-wall
        case — only a measurement, not arithmetic, may deny reach).

    Returns (feasible, reason).  ``reason`` is a short relation-grounded
    string suitable for surfacing to the actor / the validator.
    """
    if clearance_cells <= 0:
        return False, (
            f"INFEASIBLE insert-{side}: clearance to {side} is 0 "
            f"(target flush against boundary/blocker — no lane to insert)"
        )
    step = -1 if side in ("up", "left") else 1
    lane = target_ref + step          # the lane immediately on that side
    if not reach_covers_lane(reach, lane):
        return False, (
            f"INFEASIBLE insert-{side}: lane is outside the agent's "
            f"MEASURED reach span {reach['min']}..{reach['max']}"
        )
    return True, f"feasible insert-{side}: clearance {clearance_cells}>0 and within reach"


def probe_agent_reach(world: WorldKnowledge,
                      forward_action: str,
                      reverse_action: str,
                      step_fn: Callable[[str], None],
                      perceive_fn: Callable[[], None],
                      axis: str = "row",
                      max_steps: int = 64) -> Optional[dict]:
    """Measure the agent's reachable span along ``axis`` by walking it to
    both extremes, then restoring it.  Idempotent (cached on
    ``world.probe_state['agent_reach'][axis]``).

    ``forward_action`` / ``reverse_action`` are the DISCOVERED actions that
    move the agent along ``axis`` in opposite directions (supplied by the
    caller from burst coverage — never hardcoded here).  Returns the span
    dict, or None if no agent could be identified.
    """
    cache = world.probe_state.setdefault("agent_reach", {})
    if axis in cache:
        return cache[axis]

    # Identify the agent: take one forward step and see what moved.
    step_fn(forward_action)
    perceive_fn()
    agent = largest_mover(world, axis)
    if agent is None:
        return None   # nothing moves on this axis; can't measure

    def coord() -> Optional[int]:
        rec = world.entities.get(agent)
        return _ref_coord(getattr(rec, "current_bbox", None), axis)

    # The agent's coordinate BEFORE the first probe step (for restore).
    start_coord = _ref_coord(world.entities[agent].bbox_history[-2][1], axis)

    # Walk to the forward extreme (until the agent stops changing).
    last = coord()
    fwd_steps = 1   # already took one
    for _ in range(max_steps):
        step_fn(forward_action)
        perceive_fn()
        c = coord()
        if c is None or c == last:
            break
        last = c
        fwd_steps += 1
    extreme_fwd = last

    # Walk to the reverse extreme.
    last = coord()
    rev_steps = 0
    for _ in range(max_steps):
        step_fn(reverse_action)
        perceive_fn()
        c = coord()
        if c is None or c == last:
            break
        last = c
        rev_steps += 1
    extreme_rev = last

    lo, hi = sorted((extreme_fwd, extreme_rev))
    span = {"min": lo, "max": hi, "agent": agent, "axis": axis}
    cache[axis] = span

    # Restore the agent as close to its starting coordinate as the
    # quantized step allows.  Exact restore can be PHYSICALLY impossible
    # (a 6-tick step from an extreme may straddle the start), so move
    # toward the start until the next step would overshoot, then stop.
    if start_coord is not None:
        for _ in range(max_steps):
            cur = coord()
            if cur is None or cur == start_coord:
                break
            toward = forward_action if cur < start_coord else reverse_action
            step_fn(toward)
            perceive_fn()
            new = coord()
            if new is None or abs(new - start_coord) >= abs(cur - start_coord):
                # overshot / no improvement -> undo the last step and stop
                back = reverse_action if toward == forward_action else forward_action
                step_fn(back)
                perceive_fn()
                break
    return span


# ---------------------------------------------------------------------------
# On-demand probe (actor requests a pre-commit recoverability check)
# ---------------------------------------------------------------------------


def probe_on_demand(world: WorldKnowledge,
                    action: str,
                    step_fn: Callable[[str], None],
                    perceive_fn: Callable[[], None],
                    undo_action: str = "ACTION7") -> Optional[dict]:
    """Verify recoverability of one action before the actor commits it.
    Returns the verdict entry, or None if undo is unavailable (can't
    verify).  Skips re-probing if a verdict already exists."""
    existing = C.lookup_verdict(world, action)
    if existing is not None:
        return existing
    if world.probe_state.get("undo_capability") == "unavailable":
        return None
    return C.probe_recoverability(world, action, step_fn, perceive_fn,
                                   undo_action)
