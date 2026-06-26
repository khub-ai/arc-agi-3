"""skill_library.py -- COS's library of high-level, REUSABLE STRATEGIES (skills).

A SKILL is one level above an instinct: not a reactive prior but a multi-step
PROCEDURE (a recipe) that COMPOSES lower substrate capabilities, is ACQUIRED
(authored or induced from a solve), and is REUSED across games via the cross-game
KB.  The recipe is surfaced to the in-loop VLM as a strategic plan it follows with
judgement -- a recipe, not rigid code, so it generalises.

Lifecycle: RECOGNISE (trigger fires on the scene/state) -> APPLY (VLM follows the
recipe, substrate supplies the composed capabilities) -> RECORD (track where it
worked) -> ACQUIRE more (authored here; induced by the end-of-level crystallizer +
governor, which call register()/record_success()).

Game-agnostic; persists to the cross-game KB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Skill:
    name: str
    when: str                                   # human-readable trigger
    recipe: list                                # ordered sub-goal steps (-> VLM)
    composes: list = field(default_factory=list)  # substrate capabilities it uses
    trigger: Optional[Callable] = None          # predicate(ctx) -> bool
    provenance: str = "authored"                # authored | induced
    track_record: list = field(default_factory=list)  # [{game, level}] where it worked
    enabled: bool = True

    def to_dict(self) -> dict:
        return {"name": self.name, "when": self.when, "recipe": list(self.recipe),
                "composes": list(self.composes), "provenance": self.provenance,
                "track_record": list(self.track_record), "enabled": self.enabled}


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: dict = {}

    def register(self, s: Skill, *, override: bool = False) -> Skill:
        if s.name in self.skills and not override:
            return self.skills[s.name]
        self.skills[s.name] = s
        return s

    def applicable(self, ctx: dict) -> list:
        """Skills whose trigger fires for this scene/state context."""
        out = []
        for s in self.skills.values():
            if not s.enabled or s.trigger is None:
                continue
            try:
                if s.trigger(ctx or {}):
                    out.append(s)
            except Exception:
                continue
        # prefer the most-proven (track-record length) when several fire
        return sorted(out, key=lambda s: -len(s.track_record))

    def render(self, ctx: dict) -> str:
        """The recipe block for the applicable skill(s) -- surfaced to the VLM as a
        strategic plan to FOLLOW (with judgement)."""
        blocks = []
        for s in self.applicable(ctx):
            steps = "\n".join(f"  {i+1}. {st}" for i, st in enumerate(s.recipe))
            blocks.append(f"SKILL — {s.name} (applies: {s.when})\n{steps}\n"
                          f"  composes: {', '.join(s.composes)}")
        return "\n".join(blocks)

    def record_success(self, name: str, game: str, level) -> None:
        s = self.skills.get(name)
        if s is not None:
            s.track_record.append({"game": str(game), "level": level})

    # ---- cross-game KB persistence (data only; triggers re-attach on load) ----
    def to_records(self) -> list:
        return [s.to_dict() for s in self.skills.values()]

    def merge_records(self, records) -> None:
        for r in (records or []):
            nm = r.get("name")
            if not nm:
                continue
            if nm in self.skills:                          # keep the live trigger; merge record
                self.skills[nm].track_record = r.get("track_record", self.skills[nm].track_record)
                self.skills[nm].enabled = r.get("enabled", self.skills[nm].enabled)
            else:                                          # an induced skill (no live trigger yet)
                self.skills[nm] = Skill(name=nm, when=r.get("when", ""), recipe=r.get("recipe", []),
                                        composes=r.get("composes", []),
                                        provenance=r.get("provenance", "induced"),
                                        track_record=r.get("track_record", []),
                                        enabled=r.get("enabled", True))


REGISTRY = SkillRegistry()


def skill(**kw) -> Skill:
    reg = kw.pop("registry", None) or REGISTRY
    return reg.register(Skill(**kw))


# -----------------------------------------------------------------------------
# Authored seed skills
# -----------------------------------------------------------------------------

def _follow_instructions_trigger(ctx: dict) -> bool:
    """Fires when the scene PRESENTS its rule: a distinct-symbol KEY/legend group
    plus an EDITABLE target (and usually a reference/example or a demonstration)."""
    return bool(ctx.get("has_distinct_key_group") and ctx.get("has_editable_panel"))


FOLLOW_THE_INSTRUCTIONS = skill(
    name="follow_the_instructions",
    when="the scene shows its rule — a key/legend (distinct symbols) + a reference/"
         "example or a demonstration — and an editable target panel",
    recipe=[
        "Identify the TRIAD: a SELECTOR/key (options you can activate, e.g. a legend), a "
        "REFERENCE that displays the active option's encoding, and an editable PROGRAM "
        "(ordered slots you fill).",
        "For EACH option, ACTIVATE it and read its DEMONSTRATED EFFECT: in the activation "
        "animation the AGENT/mover (a silhouette/ghost in the SCENE) traces a path — its NET "
        "direction + magnitude IS that option's meaning. Distinguish it from the control "
        "panel's own presentation sweep (a column-by-column highlight is HOW the code is "
        "shown, NOT the effect — don't read the sweep's direction as the move). The substrate "
        "flags this as a DEMONSTRATION/PREVIEW + a SALIENT CO-OCCURRENCE: use those, never "
        "dismiss the animation as a transient.",
        "While the option is active, read the REFERENCE's pattern = the CODE for that effect via "
        "panel_config.read_reference_code — a STATIC MEASURED read. The reference DISPLAYS the "
        "answer for the active option: the animation only tells you the DIRECTION (the effect), "
        "the static reference tells you the CODE. NEVER guess/hypothesise a code the reference "
        "shows — MEASURE it (substrate decode, not eyeballed pixels). If the reference's cells "
        "are all IDENTICAL, that is the ACTIVE option's code shown uniformly (a selection-driven "
        "reference) — it is NOT N separate options; confirm by changing the selector and watching "
        "the whole reference change together. If a SOLUTION/ground-truth image is provided, read "
        "IT as authoritative before acting.",
        "BIND effect<->code for every option and REMEMBER it (instruction_vocabulary). "
        "Same-frame changes are RELATED: when the agent steps AND a column lights in the SAME "
        "frame, bind that column to that step (frame_correlation). Demonstrate each option ONCE "
        "— for an effect already learned, REUSE the cached code and SKIP re-demonstrating.",
        "State the GOAL as a SEQUENCE of effects (e.g. the route of moves the agent needs).",
        "For each PROGRAM slot in order, pick the option whose effect that slot needs and "
        "REPLICATE its code into that slot BY ID (one slot per step) — never eyeballed.",
        "Drive each slot to its code with panel_config.toggle_plan — read the slot's CURRENT "
        "bars and toggle ONLY the ones that differ (clean up any contamination); verify it matches "
        "before moving on. Every bar is toggleable, so a slot can always be made to match exactly.",
        "Register your PREMISES as credence-tagged claims (each option's DIRECTION, its CODE, which "
        "glyph is the AGENT) — guesses marked LOW credence — so they are doubtable. Derive each "
        "option's DIRECTION from the DEMONSTRATION (the scene mover's MEASURED net step), NOT from "
        "the legend's static GLYPH/appearance: a button can look more different than what it encodes.",
        "VERIFY by execution (fire/trigger); read the outcome against your EXPECTATION. If a step "
        "misbehaves, RE-MEASURE the reference (read_reference_code) and re-copy — NEVER substitute a "
        "GUESSED code. If the program EXECUTES but the outcome != expectation (e.g. the agent does "
        "not reach the goal), DROP INTO REVALIDATION (revalidation.py auto-fires when stuck): FIRST "
        "LOOK — re-observe the CURRENT RAW IMAGE (the clean substrate frame, NOT the gridded display "
        "render whose grid lines read as set bars) and RE-MEASURE directly; never conclude or give "
        "up from reasoning/memory. THEN re-derive the WEAKEST premises from GROUND TRUTH — direction "
        "from the DEMONSTRATION (and beware a CANNED fire/preview animation that is the same for any "
        "program — only a CORRECT program truly executes), code from the reference, agent from which "
        "glyph actually moves — then replan. Repeated IDENTICAL failures mean a MISREAD to "
        "re-measure, not a guess to vary.",
    ],
    composes=["group_distinctness", "silhouette_track", "demonstration_synthesis",
              "salient_cooccurrence", "settle", "measure_grid/decode_panel",
              "panel_config.read_reference_code", "panel_config.toggle_plan",
              "scene_state.resolve/correct/click_by_id", "click_feedback.change_report"],
    trigger=_follow_instructions_trigger,
)
