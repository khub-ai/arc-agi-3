"""Audit check: does the goal forest currently have at least one
reachable atom-leaf goal at priority >= 0.5 with a concrete cell
target?

When the answer is no, GF has nothing to drive to and the harness
falls through to Oracle every turn — the failure mode that ate
multiple sessions before the trusted_win_conditions seeding landed.

See ``docs/SPEC_pre_run_audit.md`` §"goal_forest_coverage".
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..check import AuditCheck, AuditResult, Severity


class GoalForestCoverageCheck:
    """Verifies the goal forest has actionable cell-target goals.

    "Actionable" here means: priority >= ``MIN_PRIORITY`` AND the
    root condition (or some atom-leaf reachable from it) carries a
    cell target the harness's GF selector can extract.

    A FAIL on this check means GF will return None on every turn
    until either Oracle declares concrete goals or the harness
    seeds them via trusted_* conventions.
    """

    name        = "goal_forest_coverage"
    description = (
        "Goal forest has at least one cell-target goal at priority "
        ">= 0.5 that GF can drive to."
    )

    # Threshold for "high enough priority to drive GF" — matches the
    # default _select_gf_target priority_min in the ARC harness.
    MIN_PRIORITY: float = 0.5

    def run(
        self,
        ws,                                          # WorldState
        kb:            Optional[Mapping[str, Any]]   = None,
        adapter_hooks: Optional[Mapping[str, Any]]   = None,
    ) -> AuditResult:
        del kb, adapter_hooks  # unused in this check
        forest = getattr(ws, "goal_forest", None)
        if forest is None or not getattr(forest, "goals", None):
            return AuditResult(
                check_name = self.name,
                severity   = Severity.FAIL,
                headline   = "goal forest is empty",
                fix_hint   = (
                    "ensure the harness seeds goals at session/level start "
                    "(complete_<level>, trusted_win_conditions, etc.)"
                ),
                metrics    = {"reachable_cell_goals": 0, "max_priority": 0.0},
            )

        cell_goals: list[tuple[str, float, tuple[int, int]]] = []
        max_priority = 0.0
        for gid, goal in forest.goals.items():
            try:
                status = goal.root.status.value
            except Exception:
                status = "unknown"
            if status in ("achieved", "pruned", "abandoned"):
                continue
            priority = float(getattr(goal, "priority", 0.0) or 0.0)
            if priority > max_priority:
                max_priority = priority
            cell = self._extract_cell(goal.root)
            if cell is None:
                continue
            cell_goals.append((str(gid), priority, cell))

        actionable = [
            (gid, p, c) for (gid, p, c) in cell_goals
            if p >= self.MIN_PRIORITY
        ]
        actionable.sort(key=lambda t: -t[1])

        metrics = {
            "reachable_cell_goals":     len(cell_goals),
            "actionable_cell_goals":    len(actionable),
            "max_priority":             max_priority,
            "top_3_goals_by_priority":  [
                {"id": g, "priority": p, "cell": list(c)}
                for (g, p, c) in actionable[:3]
            ],
        }

        if not actionable:
            return AuditResult(
                check_name = self.name,
                severity   = Severity.FAIL,
                headline   = (
                    f"no cell-target goals at priority >= {self.MIN_PRIORITY} "
                    f"(max priority in forest: {max_priority:.2f})"
                ),
                details    = [
                    f"{len(cell_goals)} cell-target goal(s) below threshold; "
                    f"{len(forest.goals)} total goals (incl. abstract leaves "
                    f"like LevelAdvanced or BudgetPressureActive)",
                ],
                fix_hint   = (
                    "seed trusted_win_conditions for this level OR confirm "
                    "Oracle is being called at session start to declare "
                    "interact_<role>/win_cell goals"
                ),
                metrics    = metrics,
            )

        top_id, top_prio, top_cell = actionable[0]
        return AuditResult(
            check_name = self.name,
            severity   = Severity.OK,
            headline   = (
                f"{len(actionable)} reachable cell goal(s) at priority "
                f">= {self.MIN_PRIORITY}; top is {top_id} at {top_prio:.2f} "
                f"-> {list(top_cell)}"
            ),
            metrics    = metrics,
        )

    @staticmethod
    def _extract_cell(node) -> "Optional[tuple[int, int]]":
        """Walk the GoalNode tree looking for the first ATOM whose
        canonical key carries a concrete cell target.

        Recognises three condition shapes:

        * ``AgentAtCell`` and ``TriggerVisitedAtLeast`` — adapter
          conditions whose canonical key is
          ``(name, row, col, ...)``: cell at positions 1, 2 as ints.
        * ``AtPosition`` — the foundational core condition whose
          canonical key is ``(name, entity_id, (row, col), tol)``:
          cell is the tuple at position 2.

        Other ATOM conditions (ResourceAbove, LevelAdvanced, etc.)
        are abstract — they carry no cell — and the walker returns
        None for them so the audit does not double-count them as
        actionable cell goals.
        """
        if node is None:
            return None
        nt = getattr(node, "node_type", None)
        nt_name = getattr(nt, "name", None) or str(nt)
        if nt_name == "ATOM":
            cond = getattr(node, "condition", None)
            if cond is None:
                return None
            try:
                ck = cond.canonical_key()
            except Exception:
                return None
            if not ck:
                return None
            head = ck[0]
            # Adapter-style: (name, row, col, ...)
            if head in ("AgentAtCell", "TriggerVisitedAtLeast") and len(ck) >= 3:
                try:
                    return (int(ck[1]), int(ck[2]))
                except (TypeError, ValueError):
                    return None
            # Core-style: (name, entity_id, (row, col), tol)
            if head == "AtPosition" and len(ck) >= 3:
                pos = ck[2]
                if isinstance(pos, tuple) and len(pos) == 2:
                    try:
                        return (int(pos[0]), int(pos[1]))
                    except (TypeError, ValueError):
                        return None
            return None
        for child in (getattr(node, "children", None) or []):
            r = GoalForestCoverageCheck._extract_cell(child)
            if r is not None:
                return r
        return None
