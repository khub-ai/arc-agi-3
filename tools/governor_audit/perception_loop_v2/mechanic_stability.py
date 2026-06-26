"""Mechanic-stability prior + level-start semantics verification.

THE PROBLEM THIS SOLVES
-----------------------
When the system enters a new level (or a new game), it already holds
beliefs about what each action does — e.g. "ACTION4 pushes the block
chain rightward", learned on earlier levels of the same game.  Two
failures are possible and the old harness guarded against NEITHER:

  1. It silently ASSUMES those beliefs still hold and commits a plan
     that depends on them — and if a level changed a mechanic, the plan
     misfires with no warning.  (This is exactly the sk48 lc=4 probe
     failure: 8x ACTION4 was committed on the assumption ACTION4 still
     meant "push right"; the result was ambiguous because the
     assumption was never checked.)

  2. Conversely, it could OVER-react and treat every new level as a
     blank slate, re-discovering mechanics from scratch — wasteful,
     because within one game the mechanics almost always persist.

THE PRINCIPLE (asymmetric reusability)
--------------------------------------
Knowledge reuse ACROSS LEVELS OF THE SAME GAME is highly likely;
reuse BETWEEN GAMES is much less so.  So:

  * SAME-GAME priors are STRONG.  A confirmed action->effect rule from
    an earlier level is carried into the next level at HIGH credence
    (``SAME_GAME_PRIOR``).  The default hypothesis is "mechanics are
    unchanged."  A claim that a mechanic CHANGED is the EXPENSIVE one:
    it must overturn a strong belief, so it starts low and must earn
    evidence.

  * CROSS-GAME priors are WEAK.  An action->effect rule seen in OTHER
    games is carried into a NEW game at LOW credence
    (``CROSS_GAME_CEILING``).  Here a "mechanics differ" claim is cheap
    because the prior was weak to begin with.

VERIFY BEFORE YOU DEPEND
------------------------
Strong-but-unverified is still UNVERIFIED.  On level/game open, before
any plan commits to an action's semantics, the substrate emits a
CHEAP, ORDERED verification probe — single-step each load-bearing
action, observe the delta, and CONFIRM or CONTRADICT the carried
belief.  Within-game this is a quick confirm (the belief is strong, so
one reversible step suffices); cross-game it is more thorough.  The
confirm/deny is PERSISTED per level so it is never re-paid.

This module is PURE (no driver/IO coupling beyond reading the lessons
store): it computes the stability claims, the verification plan, and
the prompt surface, and classifies whether a proposed plan assumes an
unverified mechanic change.  The driver wires these into the
level-start path; see docs/ARCHITECTURE_game_priors.md
(§ Mechanic-stability prior).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Credence anchors for the within-game / cross-game asymmetry.
SAME_GAME_PRIOR = 0.85       # a confirmed same-game rule is STRONGLY
                             # assumed to persist on the next level.
CROSS_GAME_CEILING = 0.55    # an other-games rule is only weakly assumed
                             # to apply to a NEW game.
# A plan that assumes a mechanic CHANGED (contradicts a stable claim)
# is penalized in proportion to how strong the contradicted belief is.
# Overturning a strong same-game belief is expensive; overturning a
# weak cross-game belief is cheap.
CHANGE_ASSUMPTION_PENALTY = 1.0   # multiplied by the contradicted
                                  # claim's prior_credence.


@dataclass
class StabilityClaim:
    """A carried-forward belief about what an action does, plus whether
    it has been re-confirmed on the CURRENT level yet."""
    action: str                       # e.g. "ACTION4"
    expected_effect: str              # free-form, from the source rule
    provenance: str                   # 'same_game' | 'cross_game'
    prior_credence: float
    verified_this_level: bool = False
    supporting_levels: list = field(default_factory=list)
    lesson_id: Optional[str] = None

    @property
    def load_bearing(self) -> bool:
        """A claim worth verifying before depending on it: strong enough
        to act on, but not yet confirmed on this level."""
        return (not self.verified_this_level
                and self.prior_credence >= CROSS_GAME_CEILING)


@dataclass
class VerificationStep:
    """One cheap probe that would confirm-or-deny a stability claim."""
    action: str
    expects: str
    reversible: bool
    provenance: str
    why: str


def _normalize_effect(effect: str) -> set:
    """Lowercase keyword set for a conservative effect-equality test.
    Free-form effect strings are compared by keyword overlap, not exact
    match, so 'pushes block right' and 'block moves rightward' agree."""
    if not effect:
        return set()
    toks = "".join(
        c if (c.isalnum() or c.isspace()) else " " for c in effect.lower()
    ).split()
    # Drop trivial stop-tokens that carry no mechanic meaning.
    stop = {"the", "a", "an", "to", "of", "is", "it", "and", "by",
            "with", "on", "in", "at", "effect", "action", "target",
            "role", "entity"}
    return {t for t in toks if t not in stop and len(t) > 2}


def effects_agree(a: str, b: str, *, min_overlap: int = 1) -> bool:
    """Conservative: two free-form effect strings AGREE if their keyword
    sets share at least ``min_overlap`` content tokens.  Used to decide
    whether a plan's asserted effect contradicts a stable claim."""
    sa, sb = _normalize_effect(a), _normalize_effect(b)
    if not sa or not sb:
        return False
    return len(sa & sb) >= min_overlap


def compute_stability_claims(
    *,
    same_game_rules: list,
    current_level: int,
    cross_game_action_effects: Optional[dict] = None,
) -> list:
    """Build the level's stability claims from carried knowledge.

    ``same_game_rules``: list of dicts, one per promoted within-game
    mechanic rule for THIS game.  Each needs:
        {"action": "ACTION4", "effect": "<free-form>",
         "credence": float, "supporting_levels": [int, ...],
         "verified_in_levels": [int, ...],  # optional
         "lesson_id": str}                  # optional
    These produce STRONG (same_game) claims: prior_credence is lifted to
    at least ``SAME_GAME_PRIOR`` because within a game we EXPECT
    persistence — the rule's own credence only raises it further.

    ``cross_game_action_effects`` (optional): {action -> (effect,
    credence)} learned in OTHER games, carried into a NEW game.  These
    produce WEAK (cross_game) claims, capped at ``CROSS_GAME_CEILING``.
    A claim is only added cross-game if the SAME action has no
    same-game rule (a same-game belief always dominates a cross-game
    one).

    ``verified_this_level`` is set from each rule's ``verified_in_levels``
    so a confirm done earlier this level is not re-paid.

    Returns claims ranked by prior_credence desc.
    """
    # Consolidate same-game rules to ONE claim per action.  The lessons
    # store accumulates many rules per action (e.g. several distinct
    # observed effects of ACTION4 across contexts); for a stability
    # surface we want a SINGLE load-bearing belief per action — the
    # one to confirm with a single probe.  Keep the highest-credence
    # rule as the representative effect, but UNION the level ledgers
    # across all rules for that action (a level where ANY of the
    # action's rules was confirmed counts as confirmed for the action).
    by_action: dict = {}
    for r in same_game_rules or []:
        action = r.get("action")
        if not action:
            continue
        cur = by_action.get(action)
        cred = float(r.get("credence", 0.0))
        if cur is None or cred > cur["_raw_cred"]:
            by_action[action] = {
                "_raw_cred": cred,
                "effect": r.get("effect", ""),
                "lesson_id": r.get("lesson_id"),
                "supporting_levels": set(r.get("supporting_levels") or []),
                "verified_in_levels": set(r.get("verified_in_levels") or []),
            }
        else:
            cur["supporting_levels"] |= set(r.get("supporting_levels") or [])
            cur["verified_in_levels"] |= set(r.get("verified_in_levels") or [])

    claims: list = []
    seen_actions: set = set()
    for action, agg in by_action.items():
        cred = max(agg["_raw_cred"], SAME_GAME_PRIOR)
        claims.append(StabilityClaim(
            action=action,
            expected_effect=agg["effect"],
            provenance="same_game",
            prior_credence=cred,
            verified_this_level=(current_level in agg["verified_in_levels"]),
            supporting_levels=sorted(agg["supporting_levels"]),
            lesson_id=agg["lesson_id"],
        ))
        seen_actions.add(action)
    for action, val in (cross_game_action_effects or {}).items():
        if action in seen_actions:
            continue       # same-game belief dominates
        effect, cred = (val if isinstance(val, (list, tuple))
                        else (val, CROSS_GAME_CEILING))
        claims.append(StabilityClaim(
            action=action,
            expected_effect=effect or "",
            provenance="cross_game",
            prior_credence=min(float(cred), CROSS_GAME_CEILING),
            verified_this_level=False,
            supporting_levels=[],
            lesson_id=None,
        ))
    claims.sort(key=lambda c: -c.prior_credence)
    return claims


def verification_plan(
    claims: list,
    *,
    priority_actions: Optional[list] = None,
    reversible_actions: Optional[set] = None,
) -> list:
    """Emit the cheap, ordered probes that would confirm the unverified,
    load-bearing stability claims.

    Ordering: actions an imminent plan DEPENDS ON (``priority_actions``)
    first — confirm the load-bearing assumptions before committing —
    then by prior_credence desc.  An already-verified claim, or one too
    weak to be worth a probe, is skipped.

    ``reversible_actions``: actions known to be safely undoable (e.g.
    via ACTION7=UNDO).  A reversible probe is "free" — try it, observe,
    undo.  An irreversible probe is flagged so the driver can decide
    whether the information is worth the commitment.
    """
    priority = list(priority_actions or [])
    reversible = set(reversible_actions or set())

    def sort_key(c: StabilityClaim):
        try:
            pri = priority.index(c.action)
        except ValueError:
            pri = len(priority)
        return (pri, -c.prior_credence)

    steps: list = []
    for c in sorted([c for c in claims if c.load_bearing], key=sort_key):
        steps.append(VerificationStep(
            action=c.action,
            expects=c.expected_effect,
            reversible=(c.action in reversible),
            provenance=c.provenance,
            why=(f"{c.provenance} prior (cred {c.prior_credence:.2f}"
                 + (f", levels {c.supporting_levels}"
                    if c.supporting_levels else "")
                 + ") — UNVERIFIED on this level."),
        ))
    return steps


@dataclass
class ChangeAssumptionWarning:
    action: str
    asserted_effect: str
    stable_effect: str
    contradicted_credence: float
    provenance: str
    priority_penalty: float


def classify_plan_against_stability(
    plan_action_effects: dict,
    claims: list,
) -> list:
    """Flag where a proposed plan ASSUMES a mechanic change.

    ``plan_action_effects``: {action -> asserted_effect} — the effects a
    plan is COUNTING ON for each action it uses.  (Empty / unspecified
    actions are not flagged: a plan that makes no claim about an action
    isn't assuming a change.)

    For each action the plan asserts an effect for, if a stability claim
    exists for that action whose expected effect DISAGREES with the
    asserted one, AND the claim is unverified this level, emit a warning
    with a priority penalty proportional to the contradicted belief's
    strength.  Overturning a strong same-game belief costs a lot;
    overturning a weak cross-game belief costs little.  This is how a
    "mechanics changed" plan is DE-PRIORITIZED relative to the
    stability default — without forbidding it (evidence can still win).
    """
    by_action = {c.action: c for c in claims}
    warnings: list = []
    for action, asserted in (plan_action_effects or {}).items():
        c = by_action.get(action)
        if c is None or not asserted:
            continue
        if c.verified_this_level:
            continue       # already confirmed on this level — no penalty
        if effects_agree(asserted, c.expected_effect):
            continue       # plan agrees with the stable prior — fine
        warnings.append(ChangeAssumptionWarning(
            action=action,
            asserted_effect=asserted,
            stable_effect=c.expected_effect,
            contradicted_credence=c.prior_credence,
            provenance=c.provenance,
            priority_penalty=CHANGE_ASSUMPTION_PENALTY * c.prior_credence,
        ))
    return warnings


def format_stability_surface(claims: list, vplan: list) -> str:
    """The strategy-prompt block.  Empty string when there is nothing
    carried (first-ever level of a never-seen game), so the prompt does
    not grow uselessly.

    The harness MEASURES and SURFACES; it never decides for the VLM.
    The block states (1) the carried beliefs and their strength, (2)
    which are UNVERIFIED on this level, (3) the cheap confirm probes,
    and (4) the standing rule that a plan assuming a DIFFERENT effect is
    lower-priority until it earns evidence.
    """
    if not claims:
        return ""
    lines = [
        "CARRIED MECHANIC BELIEFS (assume-stable-unless-verified):",
        "Within THIS game, mechanics almost always persist across "
        "levels — treat same-game beliefs as STRONG and DEFAULT to "
        "'unchanged'. Across games they are WEAK. A plan that depends "
        "on an action doing something DIFFERENT from a strong belief "
        "below is LOWER PRIORITY until you verify it.",
    ]
    for c in claims:
        status = ("VERIFIED here" if c.verified_this_level
                  else "UNVERIFIED here")
        lvls = (f", confirmed on levels {c.supporting_levels}"
                if c.supporting_levels else "")
        lines.append(
            f"  - {c.action} -> {c.expected_effect or '?'} "
            f"[{c.provenance}, cred {c.prior_credence:.2f}{lvls}] "
            f"({status})")
    if vplan:
        lines.append(
            "\nCHEAP CONFIRM PROBES (run before committing a plan that "
            "depends on these; reversible ones are free — act, observe, "
            "UNDO):")
        for s in vplan:
            rev = "reversible" if s.reversible else "NOT known-reversible"
            lines.append(f"  - {s.action}: expect '{s.expects or '?'}' "
                         f"[{rev}] — {s.why}")
    return "\n".join(lines)
