"""GoalManager — unified component for goal-management operations.

Phase 1 of the goal-management consolidation.  Today the operations on
``WorldState.goal_forest`` are spread across three modules:

* :mod:`cognitive_os.goal_forest` — free functions (``add_goal``,
  ``select_active_goal``, ``derive_subgoals_from_causal``,
  ``refresh_status``, ``detect_conflicts``, …).
* :mod:`cognitive_os.episode_runner` — calls a sequence of those each
  step.
* :mod:`cognitive_os.oracle` — calls some of them after declaring goals.
* Per-adapter glue (e.g. ``goal_runtime`` in the ARC adapter) re-exposes
  yet another subset.

Each call site has its own opinion about *which* operations to run and
in what order.  The result is a behaviour that's hard to read, hard to
test in isolation, and non-trivial to upgrade (e.g. swap in a credence-
weighted selector per
:doc:`/SPEC_strategy_hypotheses`).

This module introduces a single :class:`GoalManager` Protocol describing
the contract, and a :class:`DefaultGoalManager` that wraps the existing
free functions without changing their behaviour.  Subsequent phases
update call sites to depend on the Protocol; the free functions remain
available for backward compatibility but are no longer the canonical
surface.

Engineering rationale
---------------------
* **One reading order.**  Anyone learning goal management reads this
  file first.
* **Per-tick canonical entry point.**  ``GoalManager.tick(step)`` runs
  the standard refresh/derive/select sequence.  Every caller that wants
  "do all the per-turn goal maintenance" calls one method.
* **Replaceability.**  A future ``CredenceWeightedGoalManager`` (per
  :doc:`/SPEC_strategy_hypotheses`) implements the same Protocol;
  callers don't change.
* **Testability.**  Mocking ``GoalManager`` in unit tests is cleaner
  than mocking individual free functions.

What this module does NOT own
-----------------------------
* The data shapes (``Goal``, ``GoalNode``, ``GoalStatus``, ``GoalForest``)
  stay in :mod:`cognitive_os.types` — they're the schema, used by other
  subsystems independently.
* ``Condition`` ABC stays in :mod:`cognitive_os.conditions` — used by
  claims AND goals; not specific to goal management.
* Per-adapter ``Condition`` subclasses stay in adapter packages —
  domain vocabulary, the manager doesn't need to know the specific
  predicates.
* Planning a route to a goal (Planner), executing actions (Episode
  Runner / Adapter), and inspecting / declaring claims (Hypothesis
  Store) are sibling responsibilities, NOT GoalManager concerns.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from . import goal_forest as _gf
from .types import (
    Goal,
    GoalConflict,
    GoalStatus,
    WorldState,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GoalManager(Protocol):
    """Public interface for goal management.

    All implementations operate against a :class:`WorldState`'s
    ``goal_forest``.  The state is owned externally (the engine /
    adapter constructs the WorldState); the manager only manipulates
    the goal-forest slice.
    """

    # --- read --------------------------------------------------------

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        """Return the Goal with this id, or None if unknown."""
        ...

    def all_goals(self) -> Iterable[Goal]:
        """Iterate every Goal currently in the forest."""
        ...

    def active_goal(self) -> Optional[str]:
        """Return the highest-priority eligible goal id, or None when
        no live candidates remain.  Side-effect: updates the forest's
        ``active_goal_id`` field."""
        ...

    def candidates(self) -> List[str]:
        """Goal ids in priority-descending order, eligibility filtered
        (achieved / pruned / abandoned removed; conflict-blocked
        removed under FAIL policy)."""
        ...

    def snapshot(self) -> List[Dict[str, Any]]:
        """Compact per-goal snapshot for telemetry / Oracle prompts.
        Each entry: {id, priority, status, node_type, condition_key}."""
        ...

    # --- declarative writes ------------------------------------------

    def declare(self, spec: Mapping[str, Any], *, step: int = 0) -> str:
        """Compile an Oracle-authored or harness-authored goal spec
        (JSON-shaped dict) and add it to the forest.  Re-declaring the
        same id replaces the existing goal.  Returns the goal id."""
        ...

    def retract(self, goal_id: str) -> bool:
        """Mark the goal ABANDONED (kept for audit; no longer competes
        for selection).  Returns True if the id was known."""
        ...

    def amend(self, goal_id: str, **changes: Any) -> None:
        """Adjust mutable attributes of an existing goal — typically
        priority or deadline.  The full re-declaration path is
        ``declare()`` with the same id."""
        ...

    def mark_status(self, goal_id: str, status: GoalStatus) -> None:
        """Force a status transition (e.g. ACHIEVED, ABANDONED).
        Cascades to children where the type calls for it."""
        ...

    # --- per-turn orchestration --------------------------------------

    def tick(self, *, step: int) -> Dict[str, Any]:
        """Run all standard per-turn maintenance:

        1. ``refresh_status`` for every goal (cascades atom evaluation).
        2. ``derive_subgoals_from_causal`` on every active root, so
           causal claims that have crossed the commit threshold expand
           the goal tree this turn.
        3. ``refresh_motion_tolerances`` to lift any frozen-zero
           tolerances after the motion model has committed.
        4. ``detect_conflicts`` to surface MUTEX / TEMPORAL clashes.
        5. ``active_goal`` re-selection.

        Returns a summary dict suitable for telemetry:
        ``{"status_changes": {...}, "expanded": [...], "tolerances_lifted":
        N, "conflicts": [...], "active": <id or None>}``.
        """
        ...

    # --- explicit sub-operations (for caller-driven control) ---------

    def derive_subgoals(
        self,
        goal_id: str,
        *,
        max_depth: int = 3,
        step:      int = 0,
    ) -> List[str]:
        """Walk the goal tree; for each ATOM leaf whose condition
        equals the ``effect`` of any committed CausalClaim, replace it
        with an AND(trigger, effect) subtree (or OR-of-ANDs if multiple
        claims).  Returns ids of newly-expanded nodes."""
        ...

    def refresh_status(self, goal_id: str) -> GoalStatus:
        """Re-evaluate the goal's tree and update its root status."""
        ...

    def refresh_motion_tolerances(self, goal_id: str) -> int:
        """Lift any AtPosition tolerances frozen at zero (created
        before the motion model committed).  Returns the number of
        atoms lifted."""
        ...

    def detect_conflicts(self, *, step: int = 0) -> List[GoalConflict]:
        """Pairwise scan of active goals for logical / temporal /
        resource conflicts."""
        ...


# ---------------------------------------------------------------------------
# Default implementation
# ---------------------------------------------------------------------------


@dataclass
class DefaultGoalManager:
    """Standard implementation backed by the existing free functions
    in :mod:`cognitive_os.goal_forest`.

    Construct with a WorldState; subsequent calls operate on that
    state's goal_forest slice.

    Design note: every method delegates to a free function in
    ``goal_forest``.  This keeps the consolidation a pure refactor —
    no behaviour change — and lets the existing function-level test
    suite continue to verify correctness while call sites migrate to
    the manager interface.
    """

    ws: WorldState

    # --- read --------------------------------------------------------

    def get_goal(self, goal_id: str) -> Optional[Goal]:
        return self.ws.goal_forest.goals.get(goal_id)

    def all_goals(self) -> Iterable[Goal]:
        return self.ws.goal_forest.goals.values()

    def active_goal(self) -> Optional[str]:
        return _gf.select_active_goal(self.ws)

    def candidates(self) -> List[str]:
        return _gf.candidates_by_priority(self.ws)

    def snapshot(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for gid, goal in self.ws.goal_forest.goals.items():
            cond = goal.root.condition
            out.append({
                "id":            gid,
                "priority":      goal.priority,
                "status":        goal.root.status.value,
                "node_type":     goal.root.node_type.value,
                "condition_key": (repr(cond.canonical_key())
                                  if cond is not None else None),
            })
        return out

    # --- declarative writes ------------------------------------------

    def declare(self, spec: Mapping[str, Any], *, step: int = 0) -> str:
        # The spec format is adapter-defined; the engine accepts
        # already-built Goal objects via ``add_goal``.  A real adapter
        # owns the spec compiler (e.g. arc-agi-3's goal_runtime).  This
        # default implementation supports the simplest path: a spec
        # that already carries a Goal instance under the key "goal".
        if isinstance(spec, Goal):
            _gf.add_goal(self.ws, spec)
            return spec.id
        if "goal" in spec and isinstance(spec["goal"], Goal):
            _gf.add_goal(self.ws, spec["goal"])
            return spec["goal"].id
        raise NotImplementedError(
            "DefaultGoalManager.declare() accepts only pre-built Goal "
            "instances or {'goal': Goal(...)}.  Adapters that need a "
            "JSON-spec compiler should subclass DefaultGoalManager and "
            "override declare().  See "
            "usecases/arc-agi-3/python/goal_runtime.py for an example."
        )

    def retract(self, goal_id: str) -> bool:
        if goal_id not in self.ws.goal_forest.goals:
            return False
        _gf.mark_status(self.ws, goal_id, GoalStatus.ABANDONED)
        return True

    def amend(self, goal_id: str, **changes: Any) -> None:
        goal = self.ws.goal_forest.goals.get(goal_id)
        if goal is None:
            return
        if "priority" in changes:
            goal.priority = float(changes["priority"])
            goal.root.priority = float(changes["priority"])
        if "deadline" in changes:
            goal.deadline = (int(changes["deadline"])
                             if changes["deadline"] is not None else None)

    def mark_status(self, goal_id: str, status: GoalStatus) -> None:
        _gf.mark_status(self.ws, goal_id, status)

    # --- per-turn orchestration --------------------------------------

    def tick(self, *, step: int) -> Dict[str, Any]:
        status_changes: Dict[str, GoalStatus] = {}
        # 1. Refresh statuses across all goals.
        for gid, goal in list(self.ws.goal_forest.goals.items()):
            before = goal.root.status
            after = _gf.refresh_status(self.ws, gid)
            if after != before:
                status_changes[gid] = after
        # 2. Derive subgoals from causal claims for every still-OPEN
        #    root.  ACHIEVED / ABANDONED roots don't need expansion.
        expanded: List[str] = []
        for gid, goal in list(self.ws.goal_forest.goals.items()):
            if goal.root.status in (GoalStatus.ACHIEVED,
                                    GoalStatus.ABANDONED,
                                    GoalStatus.PRUNED):
                continue
            try:
                expanded.extend(_gf.derive_subgoals_from_causal(
                    self.ws, gid, step=step,
                ))
            except Exception:
                # Defensive: a single-goal expansion failure must not
                # halt per-turn maintenance.  Logged at the call site
                # if needed.
                pass
            # Trigger-discovery walker for EntitiesEquivalent goals.
            # Sibling pass over the same goal set, different match
            # rule (per-dim property containment vs. canonical-key
            # equality).  See docs/SPEC_trigger_discovery_walker.md.
            try:
                expanded.extend(_gf.derive_subgoals_for_entities_equivalent(
                    self.ws, gid, step=step,
                ))
            except Exception:
                pass
        # 3. Refresh AtPosition tolerances for goals whose atoms were
        #    frozen at zero before motion models committed.
        tolerances_lifted = 0
        for gid in list(self.ws.goal_forest.goals.keys()):
            try:
                tolerances_lifted += _gf.refresh_atposition_tolerances(
                    self.ws, gid,
                )
            except Exception:
                pass
        # 4. Conflict detection.
        try:
            conflicts = _gf.detect_conflicts(self.ws, step=step)
        except Exception:
            conflicts = []
        # 4.5. Goal-progress tracking: per-goal snapshots + events.
        # Decay outstanding recovery bumps first so newly-bumped
        # goals retain their full bump for the next selection pass
        # and fade only on subsequent turns.  Then record current
        # snapshots and emit events for any transitions.
        progress_events: List = []
        regression_claims_minted: int = 0
        try:
            from . import progress as _progress
            _progress.decay_recovery_bumps(self.ws)
            progress_events = list(_progress.record_progress(
                self.ws, turn=step,
            ))
            regression_claims_minted = _progress.mine_regression_claims(
                self.ws, tuple(progress_events), step=step,
            )
        except Exception:
            pass
        # 5. Active goal re-selection.
        active = _gf.select_active_goal(self.ws)
        return {
            "status_changes":     {k: v.value for k, v in status_changes.items()},
            "expanded":           expanded,
            "tolerances_lifted":  tolerances_lifted,
            "conflicts":          [c for c in conflicts],
            "progress_events":    [
                (e.goal_id, e.dimension, e.kind.value, e.turn)
                for e in progress_events
            ],
            "regression_claims_minted": int(regression_claims_minted),
            "active":             active,
        }

    # --- explicit sub-operations -------------------------------------

    def derive_subgoals(
        self,
        goal_id: str,
        *,
        max_depth: int = 3,
        step:      int = 0,
    ) -> List[str]:
        return _gf.derive_subgoals_from_causal(
            self.ws, goal_id, max_depth=max_depth, step=step,
        )

    def refresh_status(self, goal_id: str) -> GoalStatus:
        return _gf.refresh_status(self.ws, goal_id)

    def refresh_motion_tolerances(self, goal_id: str) -> int:
        return _gf.refresh_atposition_tolerances(self.ws, goal_id)

    def detect_conflicts(self, *, step: int = 0) -> List[GoalConflict]:
        return _gf.detect_conflicts(self.ws, step=step)
