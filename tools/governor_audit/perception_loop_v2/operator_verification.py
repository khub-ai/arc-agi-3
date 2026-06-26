"""Replay-verification of mined OPERATORS — a mined operator rule is a
HYPOTHESIS until its stated precondition is set up, its action executed,
and the claimed effect confirmed.

Grounding the REVIEW in reliable signals (see SPEC_playback.md) is
necessary but not sufficient: a rule can be mined from a real signal and
still state the wrong PRECONDITION (e.g. a carry pose read as the engage
precondition). The catch-all is execution: set up the precondition, run
the action, check the effect. This module is the PURE half —
  * replay_spec(rule): turn a mined operator into a checkable
    {precondition, actions, expected_effect};
  * classify_replay(spec, precondition_met, effect_observed): decide
    confirmed / refuted / inconclusive from the replay outcome.
The driver performs the live set-up + execution and records the verdict
through the per-level verification ledger (per_game_lessons /
record_level_verification), exactly like the mechanic-stability prior.
Until a replay confirms it, an operator is surfaced as a hypothesis, not
a fact, and verify-before-depend requires a confirm before a plan leans
on its precondition. See
memory/feedback_mined_rules_are_hypotheses_until_replayed.
"""
from __future__ import annotations

from typing import Optional


def replay_spec(rule: dict) -> Optional[dict]:
    """Turn a mined operator rule into a replay-verification spec, or None
    when the rule has no executable form. The spec names the PRECONDITION
    to establish, the action(s) to execute, and the EFFECT to confirm."""
    if not rule:
        return None
    cat = rule.get("category")
    actions = list((rule.get("actions") or {}).keys())
    if cat == "engage_transition_by_body_sweep":
        off = rule.get("typical_offset")
        return {
            "operator": cat,
            "precondition": (
                f"agent {rule.get('start_side')} the free target "
                f"(~{abs(off) if off is not None else '?'} cells offset on "
                f"the short axis), body spanning PAST it on the long axis"),
            "actions": actions,
            "expected_effect": (
                "the still-free target co-displaces with the agent on the "
                "perpendicular sweep (becomes attached)"),
        }
    if cat == "engage_by_body_sweep":
        return {
            "operator": cat,
            "precondition": ("agent body spans past the target on the long "
                             "axis (WARNING: this is the carry pose; for a "
                             "FREE target use engage_transition_by_body_sweep)"),
            "actions": actions,
            "expected_effect": "target co-displaces with the agent",
        }
    if cat == "grasp_by_pinning":
        return {
            "operator": cat,
            "precondition": "target driven against a barrier (at_extreme)",
            "actions": actions or ["push-into-barrier"],
            "expected_effect": "target attaches to the manipulator",
        }
    if cat == "decouple":
        return {
            "operator": cat,
            "precondition": ("target at the agent's far extreme, pinned "
                             "against a barrier"),
            "actions": actions or ["retract", "perpendicular-move"],
            "expected_effect": "the extreme target is RELEASED (left behind)",
        }
    return None


def classify_replay(spec: Optional[dict], *,
                    precondition_met: bool,
                    effect_observed: bool) -> str:
    """Decide the verdict from a replay attempt.
      - inconclusive: the precondition could not be established, so the rule
        was never actually tested (do NOT refute on a failed set-up).
      - confirmed:    precondition established AND the claimed effect happened.
      - refuted:      precondition established but the effect did NOT happen
        (the rule's stated precondition is wrong/insufficient).
    """
    if spec is None:
        return "inconclusive"
    if not precondition_met:
        return "inconclusive"
    return "confirmed" if effect_observed else "refuted"


def operator_status(rule: dict, level: int,
                    verified_levels: Optional[list] = None,
                    refuted_levels: Optional[list] = None) -> str:
    """A mined operator's trust status on the CURRENT level:
    'confirmed' / 'refuted' if a replay verdict exists for this level,
    else 'hypothesis' (never replayed here -> not a fact)."""
    if level in (refuted_levels or []):
        return "refuted"
    if level in (verified_levels or []):
        return "confirmed"
    return "hypothesis"


def format_operator_hypothesis_surface(rule: dict, status: str) -> str:
    """One-line surface for the strategy prompt: a mined operator and its
    verification status, so the actor knows whether it's a fact or a
    hypothesis it must replay-confirm before depending on it."""
    spec = replay_spec(rule)
    if spec is None:
        return ""
    tag = {"confirmed": "VERIFIED here",
           "refuted": "REFUTED here",
           "hypothesis": "UNVERIFIED here — replay-confirm before depending"}.get(
        status, status)
    return (f"  - operator {spec['operator']}: precondition = "
            f"{spec['precondition']} -> {spec['expected_effect']} [{tag}]")
