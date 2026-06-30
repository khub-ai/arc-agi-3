"""Aimlessness detector -- COS must ALWAYS have a goal; figuring out the game type IS a goal, and
churning through disparate pursuits with NO committed theory and NO progress is a FAILURE MODE to
catch and repair (see memory/feedback_game_type_is_a_claim_always_have_a_goal).

GAME-AGNOSTIC, pure + testable.  Aimless = the win is NOT yet understood, NO single game-type
theory is committed (no active context), recent turns span MANY DISTINCT pursuit-kinds (a
grab-bag, not one coherent line), AND there has been no lc/score progress for a while.  The
caller (driver) feeds the recent pursuit-kinds, the no-progress count, and the two state flags;
on a positive verdict it must ESCALATE -- commit the top-ranked game-type claim and pursue it
(refute -> next), or mine the tutorial -- never keep flailing.
"""
from __future__ import annotations

from typing import List, Dict


def assess(recent_plan_kinds: List[str], no_progress_turns: int,
           win_understood: bool, has_active_context: bool,
           distinct_threshold: int = 3, no_progress_threshold: int = 6) -> Dict:
    """Return {aimless, reason}.

    distinct_threshold -- this many DIFFERENT pursuit-kinds in the recent window = a grab-bag,
      not one committed line (a focused run repeats ~1 kind).
    no_progress_threshold -- turns of no lc/score progress that, combined with the grab-bag and
      no committed theory, marks churn rather than deliberate exploration.
    Both are the aimlessness criterion itself, not game-tuned knobs.
    """
    if win_understood:
        return {"aimless": False, "reason": "win understood -> goal is clear"}
    if has_active_context:
        return {"aimless": False, "reason": "a single game-type theory is committed"}
    distinct = len(set(k for k in (recent_plan_kinds or []) if k))
    if distinct >= distinct_threshold and no_progress_turns >= no_progress_threshold:
        return {"aimless": True,
                "reason": (f"{distinct} disparate pursuit-kinds over {no_progress_turns} turns "
                           f"with NO committed game-type theory and no progress")}
    return {"aimless": False,
            "reason": (f"focused or insufficient evidence "
                       f"(distinct={distinct}, no_progress={no_progress_turns})")}
