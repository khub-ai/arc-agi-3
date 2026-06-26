"""Audit check: is ``learned_facts.trusted_win_conditions`` seeded
for the current level (or the game's terminal level)?

The trusted_win_conditions seed is the operator-curated answer to
"how does this level win".  Without it, the harness must rely on
Oracle/Mediator to discover the win condition, which historically
hasn't worked reliably for sub-levels with multi-visit triggers.

Severity tiers:
  OK    — current level seeded with both alignment and win_cell
  WARN  — seeded for some levels but not the current level
  INFO  — not seeded anywhere (game may not need it)

See ``docs/SPEC_pre_run_audit.md`` §"trusted_win_conditions".
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ..check import AuditCheck, AuditResult, Severity


class TrustedWinConditionsCheck:
    """Verifies trusted_win_conditions seed coverage for the current
    level."""

    name        = "trusted_win_conditions"
    description = (
        "learned_facts.trusted_win_conditions is seeded for the current "
        "level with at least an alignment or win_cell entry."
    )

    def run(
        self,
        ws,                                          # WorldState
        kb:            Optional[Mapping[str, Any]]   = None,
        adapter_hooks: Optional[Mapping[str, Any]]   = None,
    ) -> AuditResult:
        del adapter_hooks  # unused
        kb_dict = dict(kb or {})
        twc_all = (kb_dict.get("learned_facts") or {}).get(
            "trusted_win_conditions"
        ) or {}

        # Resolve the "current level" the audit cares about.  Adapters
        # may pass it via ws.agent['_arc_state']['levels_completed']
        # (the ARC convention) or via ws.agent['level'] (generic).
        # When not present, fall back to the lex-max key in the
        # seeded levels — a defensible default that surfaces "you
        # have a seed but it's not for this level" as a WARN.
        cur_level = self._current_level(ws)

        levels_seeded = sorted(twc_all.keys()) if isinstance(twc_all, dict) else []
        cur_seed = (twc_all.get(str(cur_level)) if cur_level is not None else None) \
            if isinstance(twc_all, dict) else None

        metrics = {
            "levels_seeded":    levels_seeded,
            "current_level":    cur_level,
            "current_seeded":   cur_seed is not None,
        }

        if cur_seed is not None and isinstance(cur_seed, dict):
            has_align = bool(cur_seed.get("alignment"))
            has_win   = bool(cur_seed.get("win_cell"))
            metrics["has_alignment"] = has_align
            metrics["has_win_cell"]  = has_win
            if has_win:
                metrics["win_cell"] = list(cur_seed["win_cell"])
            if has_align and has_win:
                head = (
                    f"level {cur_level} seeded with alignment and "
                    f"win_cell {list(cur_seed['win_cell'])}"
                )
                return AuditResult(
                    check_name = self.name,
                    severity   = Severity.OK,
                    headline   = head,
                    metrics    = metrics,
                )
            details = []
            if has_align and not has_win:
                details.append("alignment present but no win_cell")
            elif has_win and not has_align:
                details.append("win_cell present but no alignment")
            return AuditResult(
                check_name = self.name,
                severity   = Severity.WARN,
                headline   = (
                    f"level {cur_level} partially seeded "
                    f"(alignment={has_align}, win_cell={has_win})"
                ),
                details    = details,
                fix_hint   = (
                    "complete the trusted_win_conditions entry for "
                    "this level (both alignment and win_cell)"
                ),
                metrics    = metrics,
            )

        if levels_seeded:
            return AuditResult(
                check_name = self.name,
                severity   = Severity.WARN,
                headline   = (
                    f"trusted_win_conditions seeded for "
                    f"{levels_seeded} but not for current level "
                    f"{cur_level!r}"
                ),
                fix_hint   = (
                    f"add a trusted_win_conditions[{cur_level!r}] entry "
                    f"with alignment + win_cell, or confirm the level "
                    f"truly needs no seed"
                ),
                metrics    = metrics,
            )

        return AuditResult(
            check_name = self.name,
            severity   = Severity.INFO,
            headline   = (
                "trusted_win_conditions not seeded for any level "
                "(may be intentional for games whose win conditions "
                "the system can derive)"
            ),
            fix_hint   = (
                "if this game has known multi-step win mechanics "
                "(rotation triggers, alignment, etc.), seed "
                "trusted_win_conditions"
            ),
            metrics    = metrics,
        )

    @staticmethod
    def _current_level(ws) -> "Optional[str]":
        """Read the current level from the WorldState in a way that
        works for both ARC (levels_completed) and any future adapter
        that uses ws.agent['level']."""
        agent = getattr(ws, "agent", None) or {}
        if isinstance(agent, dict):
            arc_state = agent.get("_arc_state")
            if isinstance(arc_state, dict) and "levels_completed" in arc_state:
                lc = arc_state.get("levels_completed")
                if lc is not None:
                    return str(int(lc))
            level = agent.get("level")
            if level is not None:
                try:
                    return str(int(level))
                except (TypeError, ValueError):
                    return str(level)
        return None
