"""Win-condition hypothesis helpers — commit, update, rank, and
surface the actor's hypotheses about what triggers score / lc /
win_state changes.

Substrate role: hold and rank.  ``game_purpose_guess`` (the
perception layer's initial guess) is just ONE entry — seeded at
trial start at low credence.  The actor authors alternatives and
commits them via the strategy reply's
``commit_win_condition_hypothesis`` field.

Discipline (enforced by the strategy prompt, not the substrate):
the FIRST subgoal of every trial should be to characterize the
win condition.  Every probe is also a probe of the active
hypotheses — observed signals SUPPORT the hypothesis that
predicted them and CONTRADICT the ones that didn't.

See world_knowledge.WinConditionHypothesis and
docs/SPEC_active_subgoals.md.
"""
from __future__ import annotations

import time
from typing import Optional

from world_knowledge import WinConditionHypothesis, WorldKnowledge


_SEED_CREDENCE             = 0.30   # low — every hypothesis is
                                    # provisional at birth
_SUPPORT_BUMP              = 0.10
_CONTRADICTION_DECAY       = 0.25
_PROMOTION_CREDENCE        = 0.85


# ---------------------------------------------------------------------------
# Commit / update
# ---------------------------------------------------------------------------


def _new_id(description: str, turn: int) -> str:
    safe = "".join(
        c if c.isalnum() or c == "_" else "_"
        for c in description.lower()
    )[:40]
    return f"wc_{safe}_t{turn}_{int(time.time()) % 100000}"


def seed_initial_hypothesis_from_perception(
    world: WorldKnowledge,
) -> Optional[WinConditionHypothesis]:
    """At trial start, if perception inferred a ``game_purpose_guess``,
    seed it as the first WinConditionHypothesis at low credence.

    Discipline reminder: this is the perception layer's GUESS, not
    a verified fact.  The actor's first subgoal should probe to
    validate or falsify it."""
    gp = (world.game_purpose_guess or "").strip()
    if not gp:
        return None
    # already seeded?
    for h in world.win_condition_hypotheses:
        if h.description.strip() == gp:
            return h
    wc = WinConditionHypothesis(
        hypothesis_id=_new_id(gp, world.turn),
        description=gp,
        credence=_SEED_CREDENCE,
        created_at_turn=world.turn,
        notes=(
            "Seeded from perception's game_purpose_guess.  "
            "Treat as one of many possible win conditions — "
            "PROBE TO VALIDATE before deriving deliveries."
        ),
    )
    world.win_condition_hypotheses.append(wc)
    return wc


def commit_hypothesis(
    world: WorldKnowledge,
    *,
    description: str,
    notes: str = "",
    credence: float = _SEED_CREDENCE,
) -> WinConditionHypothesis:
    """Author a new win-condition hypothesis from a strategy reply."""
    wc = WinConditionHypothesis(
        hypothesis_id=_new_id(description, world.turn),
        description=description,
        credence=max(0.0, min(1.0, credence)),
        created_at_turn=world.turn,
        notes=notes,
    )
    world.win_condition_hypotheses.append(wc)
    return wc


def record_observation(
    world: WorldKnowledge,
    *,
    hypothesis_id: str,
    delta_index: int,
    kind: str,          # "support" | "contradict"
) -> Optional[WinConditionHypothesis]:
    """Update credence of a hypothesis given a new observation
    (typically: a score / win_state change, or a probed
    no-change observation).  Returns the updated record."""
    target: Optional[WinConditionHypothesis] = None
    for h in world.win_condition_hypotheses:
        if h.hypothesis_id == hypothesis_id:
            target = h
            break
    if target is None:
        return None
    if kind == "support":
        target.supporting_observations.append(delta_index)
        target.credence = min(1.0, target.credence + _SUPPORT_BUMP)
    elif kind == "contradict":
        target.contradicting_observations.append(delta_index)
        target.credence = max(0.0, target.credence - _CONTRADICTION_DECAY)
    if target.credence >= _PROMOTION_CREDENCE:
        target.promoted = True
    return target


# ---------------------------------------------------------------------------
# Retrieval / surface
# ---------------------------------------------------------------------------


def rank_hypotheses(
    world: WorldKnowledge,
) -> list[WinConditionHypothesis]:
    """Return hypotheses ranked by credence × support count
    (desc).  Falsified ones (credence == 0) drop to the bottom
    but stay in the list for provenance."""
    return sorted(
        world.win_condition_hypotheses,
        key=lambda h: (
            -h.credence,
            -len(h.supporting_observations),
            -h.created_at_turn,
        ),
    )


def format_win_condition_surface(
    world: WorldKnowledge,
) -> str:
    """Render the win-condition hypotheses as a strategy-prompt
    block.  Surfaces the discipline that every probe is also a
    probe of these hypotheses."""
    hyps = rank_hypotheses(world)
    if not hyps:
        return (
            "  (no win-condition hypotheses recorded.  PRIORITY: "
            "before deriving any delivery / tactical subgoals, "
            "commit one or more WinConditionHypotheses via "
            "`commit_win_condition_hypothesis` in your reply, "
            "then commit a subgoal whose expected_outcome is to "
            "validate or falsify one of them through observation.)"
        )

    lines: list[str] = []
    lines.append(
        f"  {len(hyps)} WIN-CONDITION HYPOTHESIS/ES recorded.  "
        "These are CLAIMS about what triggers score / lc / "
        "win_state changes.  Every probe is ALSO a probe of "
        "these hypotheses: after each action, check whether the "
        "observed signal matches each hypothesis's prediction "
        "(record support) or contradicts it (record contradict).  "
        "Do NOT derive delivery subgoals from a hypothesis that "
        "is still at low credence — probe to validate it first."
    )
    for h in hyps:
        status = "PROMOTED" if h.promoted else "tentative"
        lines.append("")
        lines.append(
            f"  HYP id={h.hypothesis_id!r}  credence={h.credence:.2f}  "
            f"({status})  created t{h.created_at_turn}"
        )
        lines.append(f"    Description: {h.description}")
        lines.append(
            f"    Evidence:  +{len(h.supporting_observations)} support, "
            f"-{len(h.contradicting_observations)} contradict"
        )
        if h.notes:
            lines.append(f"    Notes:       {h.notes}")
    lines.append("")
    lines.append(
        "  HOW TO USE: in your strategy reply, set "
        "`commit_win_condition_hypothesis` to a JSON object "
        '{ "description": str, "notes": str } to author a new '
        "hypothesis.  When committing a top-level GOAL subgoal, "
        "set `win_condition_hypothesis_id` on it so the subgoal "
        "tracks WHICH hypothesis it's serving.  If the hypothesis "
        "is later contradicted, every subgoal carrying its id is "
        "automatically flagged."
    )
    return "\n".join(lines)
