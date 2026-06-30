"""Action-effectiveness tracking + routing -- learn which actions actually change the board in
THIS game, so COS stops wasting turns on dead actions (e.g. tr87, where ~148 clicks did nothing
while ACTION1-4 were the real editor) -- WITHOUT falsely killing a context-dependent action.

THE su15 GUARDRAIL.  A click can work in a NARROW range (only on a specific target).  So:
  - effectiveness is CONDITIONAL on context.  For position-dependent actions (click) the
    context is the TARGET; a dead result at target T says nothing about an UNTRIED target T'.
  - demotion requires DIVERSE sampling: a position-dependent action is only "likely dead" after
    it has been tried across many DISTINCT targets with no effect; a global action (ACTION1-5)
    after a few issuances with no effect.
  - demotion is SOFT, never a hard kill: a demoted action keeps a non-zero retry priority, and
    an UNTRIED click target is always worth a fair shot (no false dead-end -- 'not found yet',
    not 'impossible').

This is the bookkeeping + status/priority CORE (unit-tested).  The driver wires it by recording
each action's (caused_change, target) and weighting candidate actions by `priority(...)`.
"""
from __future__ import annotations

from typing import Dict, Optional, Hashable


def _is_position_dependent(action: str) -> bool:
    """Click-like actions whose effect depends on WHERE they land (context = target)."""
    a = (action or "").upper()
    return "CLICK" in a or a == "ACTION6"


class ActionEffectiveness:
    def __init__(self, global_dead_after: int = 3, click_diverse_targets: int = 8,
                 dead_priority: float = 0.1, unknown_priority: float = 0.6,
                 effective_priority: float = 1.0, untried_target_priority: float = 0.75):
        # per action: {"issued","effective","targets": {target: effective_bool}}
        self.stats: Dict[str, Dict] = {}
        self.global_dead_after = global_dead_after
        self.click_diverse_targets = click_diverse_targets
        self.dead_priority = dead_priority
        self.unknown_priority = unknown_priority
        self.effective_priority = effective_priority
        self.untried_target_priority = untried_target_priority

    def record(self, action: str, caused_change: bool, target: Optional[Hashable] = None) -> None:
        s = self.stats.setdefault(action, {"issued": 0, "effective": 0, "targets": {}})
        s["issued"] += 1
        if caused_change:
            s["effective"] += 1
        if target is not None:
            # a target counts as effective if it EVER caused a change
            s["targets"][target] = s["targets"].get(target, False) or caused_change

    def status(self, action: str, candidate_target: Optional[Hashable] = None) -> str:
        """'effective' | 'likely_dead' | 'unknown'.  Conservative: only 'likely_dead' after
        adequate sampling; an untried click target is always 'unknown' (deserves a shot)."""
        s = self.stats.get(action)
        if not s:
            return "unknown"
        if s["effective"] > 0:
            return "effective"
        if _is_position_dependent(action):
            # an UNTRIED target hasn't been tested -> keep exploring it (narrow-range guard)
            if candidate_target is not None and candidate_target not in s["targets"]:
                return "unknown"
            if len(s["targets"]) >= self.click_diverse_targets:
                return "likely_dead"          # diverse targets all dead -> probably dead
            return "unknown"                  # not enough coverage yet
        # global action: a few no-effect issuances are enough to demote (still soft)
        if s["issued"] >= self.global_dead_after:
            return "likely_dead"
        return "unknown"

    def priority(self, action: str, candidate_target: Optional[Hashable] = None) -> float:
        """Routing weight in (0, 1].  Effective actions win; dead actions keep a non-zero retry
        budget (soft); an untried click target is boosted above a known-dead one."""
        st = self.status(action, candidate_target)
        if st == "effective":
            return self.effective_priority
        if st == "likely_dead":
            # a click on an untried target still deserves more than a known-dead retry
            if _is_position_dependent(action) and candidate_target is not None \
                    and candidate_target not in (self.stats.get(action, {}).get("targets", {})):
                return self.untried_target_priority
            return self.dead_priority
        return self.unknown_priority

    def is_demoted(self, action: str) -> bool:
        """True if the action is likely dead with NO untried context left to explore (blunt view;
        the driver should prefer priority(action, target) which is target-aware)."""
        return self.status(action) == "likely_dead"

    def summary(self) -> Dict[str, str]:
        return {a: self.status(a) for a in self.stats}
