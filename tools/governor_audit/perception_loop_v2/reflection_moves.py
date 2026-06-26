"""Reflection moves — a game-agnostic library of META-COGNITIVE REFRAMES,
surfaced to the strategy VLM when it is stuck.

WHY THIS IS DATA, NOT CODE
--------------------------
The substrate has no intelligence and cannot do means-ends reasoning,
constraint re-scoping, or tool-use discovery itself — any attempt to
build that as a symbolic construct on a dumb base is brittle and fails
at exactly the joints that need judgment.  So the *reasoning* stays in
the VLM-in-the-moment.  What lives here is only the PROMPT CONTENT that
reminds whatever VLM is in the loop to apply a reframe — text, editable
without touching code.  The substrate's whole job is to surface this at
a cheap mechanical trigger (no progress for N turns / a blocked goal)
and persist whatever the VLM produces.

These moves lower the task from "ORIGINATE the strategy" (which weak /
swapped VLMs fail) to "APPLY this given reframe to the current
situation" (which far more models clear).  They do not replace VLM
capability; they reduce the capability required, and they degrade
gracefully — once any capable instance turns a reframe into a concrete
breakthrough, that breakthrough is mined + cached and replayed by all
later instances (see per_game_lessons / subroutine_kb).

Each move is GENERALIZED on purpose (no game vocabulary) so it transfers
across games.  When a dev session surfaces a new generalizable insight,
add it here in generalized form (see the coding-agent reminder in
memory: feedback_persist_insights_to_cos_prompt).
"""
from __future__ import annotations


# Ordered roughly by how cheaply they tend to dissolve an impasse.
# Each entry: (short_name, vlm_facing_text).
REFLECTION_MOVES: list[tuple[str, str]] = [
    (
        "re-perceive from scratch",
        "Drop the model you have been building and read the raw frame "
        "fresh. A hard impasse against a well-posed task is usually a "
        "compounding perception/model error you have been building on, "
        "not a broken world. Re-derive entities and relations from the "
        "pixels before concluding anything.",
    ),
    (
        "separate means from ends",
        "Ask: is the goal I am blocked on the actual END, or just a "
        "METHOD I assumed? (e.g. 'reach object X' is usually a method "
        "for 'move object X'.) Climb to the most general statement of "
        "what must become true, then look for OTHER methods that "
        "achieve it. A blocked method is not a blocked goal.",
    ),
    (
        "check the constraint's scope",
        "A limit you discovered for ONE effector or action may be "
        "specific to it, not universal. If 'my arm cannot get there', "
        "ask: is there another body/object/effector NOT subject to that "
        "constraint? A free or detached object often is not bound by "
        "the same rule as the controlled actuator.",
    ),
    (
        "act through an intermediary (tool use)",
        "To affect something you cannot reach or touch directly, use an "
        "object you DO control as the tool: push it, drop it, or propel "
        "it into the target so force transmits at a distance. Direct "
        "contact by your end-effector is one way to cause an effect — "
        "not the only way. CONCRETE TRIGGER: a free intermediary block "
        "sitting between your effector and an unreachable target on the "
        "same axis (same column for vertical stacks; same row for "
        "horizontal chains) is the canonical case — the substrate's "
        "MEDIATION CANDIDATES surface (when populated) names each "
        "candidate by KIND (vertical_lift, horizontal_push, ...) along "
        "with exact intermediaries, targets, and the agent setup, so you "
        "can evaluate one concrete plan instead of re-deriving the "
        "geometry. If the gate refuted your direct plan and a candidate "
        "is surfaced, apply it BEFORE proposing another direct-reach "
        "plan.",
    ),
    (
        "triage doubt by credence (protect invariants)",
        "When backward reasoning hits a contradiction (the goal needs X, "
        "but X seems unreachable), do NOT doubt the highest-credence "
        "belief. Rank the beliefs in the chain by how well-established "
        "they are, and re-examine the WEAKEST first. A goal/win condition "
        "carried consistently across prior levels of THIS game, or a "
        "mechanic confirmed many times, is a strong invariant — distrust "
        "your fresh, few-probe conclusion (e.g. 'this is unreachable') "
        "long before you distrust the invariant. The level is solvable + "
        "the win is fixed => what you just concluded is impossible is the "
        "thing that is actually wrong.",
    ),
    (
        "not-found vs impossible",
        "Exhausting your current ideas means 'not found yet', NOT "
        "'impossible'. Do not fabricate a plausible-but-unchecked move "
        "to escape the discomfort. State precisely what you ruled out, "
        "what would resolve it, and escalate honestly rather than guess.",
    ),
]


def format_reflection_moves(active: bool, *, blocked_goal: str = "") -> str:
    """Render the reflection-move block for the strategy prompt.

    ``active`` is the mechanical stuck-trigger (substrate-set: no
    progress for N turns, or a goal marked blocked). When False, returns
    a single compact pointer line (cheap, always-available) so the moves
    exist in the actor's awareness without dominating the prompt. When
    True, returns the full library with rising prominence and, if known,
    names the specific blocked goal to apply them to.

    The substrate decides ``active`` (a counter); the VLM does the
    reasoning. No judgment lives in this function.
    """
    if not active:
        return ("If you get stuck, reflection moves are available "
                "(separate means from ends; check constraint scope; "
                "act through an intermediary; re-perceive).")
    lines = [
        "YOU APPEAR STUCK — apply these reframes before concluding a "
        "dead-end. The point is to APPLY one to the current situation, "
        "not to admire them:",
    ]
    if blocked_goal:
        lines.append(f"  (blocked goal: {blocked_goal})")
    for name, text in REFLECTION_MOVES:
        lines.append(f"  - {name.upper()}: {text}")
    return "\n".join(lines)
