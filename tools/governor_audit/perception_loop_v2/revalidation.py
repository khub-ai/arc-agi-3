"""revalidation.py -- when a plan fails against expectation, DROP INTO a revalidation loop.

The lc2 and lc3 failures were the same shape: act on a GUESSED premise (a code; a legend button's
direction) that was never validated, then keep going when it silently doesn't work.  This makes
"things didn't work out as expected" an AUTOMATIC trigger to re-test the shakiest beliefs against
GROUND TRUTH -- re-derive each by re-running its discriminating probe (e.g. re-DEMONSTRATE a control
and read its ACTUAL measured effect, not its glyph) -- weakest belief first, before replanning.  It
is NOT "retry the same plan" and NOT "vary a detail blindly".

Pure logic here (rank + directive); the driver/VLM supplies the premises and runs the probes.  Pairs
with [[failure_escalation]] (the stuck/stagnation trigger) and the claim store (the premises).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# A guess is doubted before a measurement; a measurement before a ground-truthed fact.
_PROV_RANK = {"guessed": 0, "eyeballed": 0, "assumed": 0, "glyph": 0,
              "inherited": 1, "carried": 1,
              "demonstrated": 3, "measured": 3, "ground_truthed": 4}


@dataclass
class Premise:
    """A belief a plan depends on, with how to RE-DERIVE it from ground truth."""
    name: str
    value: Any = None
    credence: float = 0.5
    provenance: str = "guessed"
    probe: str = ""              # the discriminating probe that re-derives this from ground truth
    ground_truthed: bool = False


def expectation_violated(expected, observed) -> bool:
    """The plan predicted ``expected``; the world gave ``observed``.  None on either side means
    'no expectation to check' (never a violation)."""
    if expected is None or observed is None:
        return False
    return expected != observed


def revalidation_order(premises):
    """Weakest belief FIRST: lowest credence, then guess-before-measurement, then
    not-yet-ground-truthed.  This is 'triage doubt by credence' made concrete."""
    return sorted(
        premises,
        key=lambda p: (round(float(p.credence), 3),
                       _PROV_RANK.get(p.provenance, 2),
                       1 if p.ground_truthed else 0),
    )


@dataclass
class RevalidationDirective:
    reason: str
    premises: list = field(default_factory=list)

    def to_note(self) -> str:
        head = ("[REVALIDATION] The outcome CONTRADICTS your expectation -- so a premise is wrong. "
                "REASONING is your primary, cheaper tool and is fine to use; the rule is only that "
                "you must NOT push THROUGH a contradiction with more reasoning or trust MEMORY at "
                "that point: LOOK -- drop to the CURRENT RAW IMAGE and RE-MEASURE the relevant state "
                "directly from the pixels to find which belief is false (reasoning misfires "
                "routinely: a stale cache, a read at the wrong position, the GRIDDED display read "
                "instead of the clean frame, a CANNED animation mistaken for the real effect). THEN "
                "re-derive the premises the plan rested on from GROUND TRUTH -- re-run each one's "
                "discriminating probe (re-DEMONSTRATE the control and read its ACTUAL measured "
                "effect; never infer from a glyph/appearance) -- WEAKEST belief first, then replan. "
                "Do NOT re-run the failed plan and do NOT blindly vary a detail.")
        if self.premises:
            body = "\n".join(
                f"  {i + 1}. {p.name} (currently {p.value!r}, credence {p.credence:.2f}, "
                f"{p.provenance}{'' if p.ground_truthed else ', NOT ground-truthed'}) "
                f"-> re-derive by: {p.probe or 'its ground-truth probe'}"
                for i, p in enumerate(self.premises))
            head += "\nPremises to revalidate (weakest first):\n" + body
        else:
            head += (" No premises were registered as claims -- that itself is the bug: register the "
                     "beliefs this plan depends on (guesses marked low-credence) so they are "
                     "doubtable, then re-derive them.")
        return head + f"\n(reason: {self.reason})"


def maybe_revalidate(expected, observed, premises=None, reason=None):
    """If the outcome violated the expectation, return a RevalidationDirective with the premises
    ranked weakest-first; else None.  Game-agnostic entry point."""
    if not expectation_violated(expected, observed):
        return None
    return RevalidationDirective(
        reason=reason or f"expected {expected!r} but observed {observed!r}",
        premises=revalidation_order(list(premises or [])))
