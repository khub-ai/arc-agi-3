"""Compile a recognized ORDERED objective into Goal-Forest goals.

WHY (the architectural gap this closes)
---------------------------------------
The VLM can RECOGNIZE the objective (e.g. an on-screen strip specifies "achieve
red, then blue, then red, in that order"). But recognizing it is not enough:
this session, the order was *known* yet a second red was pierced anyway,
because the objective lived only as a passive win-relation string — it was
never turned into live, ENFORCED goals. Knowing != enforcing.

THE FIX (game-agnostic — no HUD/sk48 specifics)
-----------------------------------------------
When an ORDERED objective is committed (an `ordered_completion` win-relation),
compile it into a chain of ActiveSubgoals linked by `depends_on`. Then the
EXISTING goal machinery enforces the sequence for free:
  * `depends_on` makes an out-of-order step NON-PURSUABLE / non-achievable
    until its predecessor is done — the gate that would have blocked piercing
    a second red before the blue;
  * the goal-gap surfaces the NEXT-REQUIRED step (first un-achieved whose
    predecessor is done);
  * out-of-order completion is flagged ("the order is the gate; out-of-order
    attempts do not take").

The objective stops being text the VLM must remember and becomes live goals.
This applies to ANY ordered objective from ANY source — a recipe card, an
assembly sequence, a set of waypoints — the HUD is just one instance of "a
scene element whose recognized PURPOSE is an ordered objective".
"""
from __future__ import annotations

from typing import List, Optional


def committed_ordered_hypothesis(world):
    """The highest-credence committed ORDERED win-condition HYPOTHESIS, or None.
    Generic: keys on the relation TYPE (`ordered_completion`), never on the
    game or the HUD."""
    cands = [h for h in (getattr(world, "win_condition_hypotheses", None) or [])
             if getattr(h, "win_relation", None)]
    cands = [h for h in cands
             if ((h.win_relation or {}).get("type") == "ordered_completion")]
    if not cands:
        return None
    return sorted(cands, key=lambda h: getattr(h, "credence", 0.0),
                  reverse=True)[0]


def committed_ordered_relation(world) -> Optional[dict]:
    """The committed ordered win-relation dict (convenience over the hyp)."""
    h = committed_ordered_hypothesis(world)
    return h.win_relation if h else None


def objective_won(world) -> bool:
    """True iff the committed ordered objective is fully satisfied = the level
    is WON. The acceptance predicate behind the terminal `ordered_objective_won`
    goal."""
    rel = committed_ordered_relation(world)
    if not rel:
        return False
    try:
        from knowledge_crystallization import evaluate_win_relation
        return bool(evaluate_win_relation(world, rel).get("satisfied"))
    except Exception:
        return False


def _ordered_len(world, rel) -> int:
    try:
        from knowledge_crystallization import evaluate_win_relation
        det = (evaluate_win_relation(world, rel).get("detail") or {})
        return len(det.get("ordered") or [])
    except Exception:
        return 0


def compile_ordered_objective(world) -> List[str]:
    """Compile the committed ordered objective into a depends_on-linked chain
    of ActiveSubgoals (idempotent). Returns the created subgoal ids. No-op if
    no ordered objective is committed or it has already been compiled."""
    if getattr(world, "_ordered_objective_compiled", False):
        return []
    hyp = committed_ordered_hypothesis(world)
    if not hyp:
        return []
    rel = hyp.win_relation
    n = _ordered_len(world, rel)
    if n < 1:
        return []
    role = (rel.get("roles") or ["target"])[0] or "target"
    # Tie the whole chain to the level's WIN: each step SERVES this win-
    # condition hypothesis, and a terminal goal whose acceptance is the win
    # itself depends on the last step. So achieving the chain == winning the
    # level, and the Goal Forest treats the chain as the path to the win
    # (priority, refute-invalidation) rather than free-floating subgoals.
    wc_id = getattr(hyp, "hypothesis_id", None)
    try:
        from active_subgoals import commit_subgoal
    except Exception:
        return []
    ids: List[str] = []
    prev: Optional[str] = None
    for k in range(n):
        try:
            sg = commit_subgoal(
                world,
                name=f"ordered_step_{k}",
                problem_solved=("enforce the recognized ordered objective as "
                                "Goal-Forest goals so the sequence is gated"),
                expected_outcome=(f"complete the #{k} {role} in the required "
                                  f"order (step {k + 1} of {n})"),
                acceptance_check=f"ordered_step_complete:{k}",
                depends_on=[prev] if prev else None,
                win_condition_hypothesis_id=wc_id,
                derived_from="ordered_objective_compiler",
            )
            ids.append(getattr(sg, "subgoal_id", f"ordered_step_{k}"))
            prev = ids[-1]
        except Exception:
            break
    # Terminal WIN goal: achieving it == the level is won.
    if ids:
        try:
            win = commit_subgoal(
                world,
                name="level_win",
                problem_solved="WIN the level by completing the ordered objective",
                expected_outcome=("WIN the level — the ordered objective is "
                                  "fully satisfied (all steps complete in order)"),
                acceptance_check="ordered_objective_won",
                depends_on=[ids[-1]],
                win_condition_hypothesis_id=wc_id,
                derived_from="ordered_objective_compiler",
            )
            ids.append(getattr(win, "subgoal_id", "level_win"))
        except Exception:
            pass
        world._ordered_objective_compiled = True
    return ids


def ordered_step_satisfied(world, k: int) -> bool:
    """True iff the first k+1 steps of the committed ordered objective are
    completed IN ORDER. The acceptance predicate behind
    `ordered_step_complete:<k>`."""
    rel = committed_ordered_relation(world)
    if not rel:
        return False
    try:
        from knowledge_crystallization import evaluate_win_relation
        det = (evaluate_win_relation(world, rel).get("detail") or {})
        return bool(det.get("in_order", True)) and len(det.get("done") or []) >= (k + 1)
    except Exception:
        return False


def format_ordered_objective_surface(world) -> str:
    """Render the live ordered-objective cursor + next-required + the
    out-of-order gate, for the strategy prompt (the un-ignorable enforcement)."""
    rel = committed_ordered_relation(world)
    if not rel:
        return ""
    try:
        from knowledge_crystallization import evaluate_win_relation
        r = evaluate_win_relation(world, rel)
    except Exception:
        return ""
    det = r.get("detail") or {}
    if r.get("satisfied"):
        return "ORDERED OBJECTIVE (Goal Forest): all steps complete."
    done = det.get("done") or []
    nxt = det.get("next")
    in_order = det.get("in_order", True)
    lines = [
        "ORDERED OBJECTIVE (compiled into the Goal Forest, ENFORCED): "
        f"done={list(done)}; NEXT-REQUIRED = {nxt}. "
        "Steps after it are BLOCKED (depends_on) until it completes — act "
        "only toward NEXT-REQUIRED."]
    if not in_order:
        lines.append("  WARNING: completions are OUT OF ORDER — the order is "
                     "the gate; out-of-order attempts DO NOT TAKE (no-op).")
    return "\n".join(lines)
