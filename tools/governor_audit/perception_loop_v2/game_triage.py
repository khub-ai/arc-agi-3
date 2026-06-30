"""Competition game-triage -- give up on a hopeless game and move to the NEXT, so a session of
~200 games doesn't let one stuck game starve the rest.  GAME-AGNOSTIC, pure + testable.

A game is abandoned when ANY of:
  - the per-game TURN BUDGET is exhausted (hard cap);
  - by a CLAIM DEADLINE, COS still cannot even form a credible game-type claim (no committed
    theory AND the best win-hypothesis credence is below a floor) -- it hasn't classified the
    game, so spending more turns is unlikely to pay off;
  - it has STALLED (no lc/score progress for too many turns).

OPT-IN: disabled by default so dev runs (which use --max-turns + want to debug a single game)
are unaffected; the competition harness enables it via COS_GAME_TRIAGE + budget env vars.
This implements the policy in memory/feedback_game_type_is_a_claim_always_have_a_goal (never
flail) at the SESSION level: a game with no claim and no progress is cut, not ground on.
"""
from __future__ import annotations

import os
from typing import Dict


def from_env() -> Dict:
    """Read triage config from the environment (competition sets these).  Defaults are generous so
    that, even when enabled, only a genuinely stuck/unclassifiable game is cut."""
    def _i(name, d):
        try:
            return int(os.environ.get(name, d))
        except Exception:
            return d
    def _f(name, d):
        try:
            return float(os.environ.get(name, d))
        except Exception:
            return d
    return {
        "enabled": os.environ.get("COS_GAME_TRIAGE", "") not in ("", "0", "false", "False"),
        "max_turns": _i("COS_TRIAGE_MAX_TURNS", 200),
        "claim_deadline": _i("COS_TRIAGE_CLAIM_DEADLINE", 12),
        "stall_limit": _i("COS_TRIAGE_STALL_LIMIT", 30),
        "min_claim_credence": _f("COS_TRIAGE_MIN_CLAIM_CREDENCE", 0.35),
    }


def should_abandon(turns: int, turns_since_progress: int, has_committed_claim: bool,
                   max_win_credence: float, max_turns: int = 200, claim_deadline: int = 12,
                   stall_limit: int = 30, min_claim_credence: float = 0.35,
                   enabled: bool = False) -> Dict:
    """Should COS give up on this game and move to the next?  Returns {abandon, reason}.

    NOTE on max_turns: it is an UNCONDITIONAL absolute ceiling -- the ONLY cut that applies to a
    PROGRESSING game (stall/claim only cut non-progressing/unclassified games).  So keep it HIGH (a
    safety backstop, default 200); the real budget for a progressing game is the wall-clock deadline
    (COS_GAME_DEADLINE_EPOCH, enforced in the driver), not this turn count.  A low max_turns would
    cut a winning game prematurely."""
    if not enabled:
        return {"abandon": False, "reason": "triage disabled"}
    if turns >= max_turns:
        return {"abandon": True, "reason": f"turn budget exhausted ({turns} >= {max_turns})"}
    if (turns >= claim_deadline and not has_committed_claim
            and (max_win_credence or 0.0) < min_claim_credence):
        return {"abandon": True,
                "reason": (f"no credible game-type claim by turn {turns} (best credence "
                           f"{(max_win_credence or 0.0):.2f} < {min_claim_credence}) -- cannot "
                           f"classify the game; move on")}
    if turns_since_progress >= stall_limit:
        return {"abandon": True,
                "reason": f"stalled: no lc/score progress for {turns_since_progress} turns"}
    return {"abandon": False, "reason": "within budget / progressing"}
