"""Instinct registry — COS's game-agnostic priors, organized by TRIGGER.

An *instinct* is a game-agnostic prior the VLM should apply.  Each one fires under
a TRIGGER (a condition over the turn): "always" (ambient), "an animation is
available", "stuck", "win unknown", "non-grid / no agent", etc.  This module is
the single organized home for them — previously they were scattered across the
system prompt, reflection_moves.py, the priors doc, and bespoke enforcement code.

An instinct carries:
  - name, category (perceptual / physics / goal / reasoning / meta)
  - trigger(ctx) -> bool         : WHEN it applies (the organizing axis)
  - when (str)                   : human-readable trigger description
  - content(ctx) -> str | str    : the directive text (DATA, not code — the
                                   reasoning stays in the VLM; this is the prompt
                                   content that reminds it to apply the prior)
  - surfaced_by                  : where its content reaches the VLM —
                                   "registry"      : rendered from `content` here,
                                   "system_prompt" : baked in SYSTEM_PROMPT_TEMPLATE,
                                   "reflection_moves" / "kb_recall" / "validator" /
                                   "driver"        : surfaced by that module.
  - mandatory (bool)             : if True, it is ENFORCED — when it fires and the
                                   reply doesn't satisfy it, the harness re-prompts.
  - satisfied(reply) -> bool     : (mandatory only) did the reply honour it?
  - reprompt (str)               : (mandatory only) the focused re-prompt template
                                   ({n}, {reply_name} substituted by the driver).
  - gate (str)                   : human note on the exemption ("unless win known").

The harness builds a `TurnContext` each turn, asks the registry which instincts
FIRE, surfaces the "registry"-surfaced ones into the prompt, and ENFORCES the
mandatory ones.  The non-"registry" entries are CATALOGUED here (with their
triggers) even though their content lives elsewhere — so this file is the index
of every instinct and when it applies.  See docs/INSTINCTS.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# -----------------------------------------------------------------------------
# Turn context — the signals triggers read
# -----------------------------------------------------------------------------


@dataclass
class TurnContext:
    phase: str = "delta"          # initial | delta | level_start | strategy
    turn_n: int = 0
    has_animation: bool = False    # the last action produced >=2 sub-frames
    n_frames: int = 0
    win_understood: bool = False   # COS holds a credible win-condition hypothesis
    has_agent: bool = True         # a controllable agent is present
    is_grid: bool = True           # the scene is grid/cell based
    stuck: bool = False            # no progress for several turns
    score_gradient: bool = False   # the score is giving feedback this run
    goal_depicted: bool = False    # a region depicts an end-state / target
    last_action_failed: bool = False  # the planned action had no effect


# -----------------------------------------------------------------------------
# Instinct + registry
# -----------------------------------------------------------------------------


@dataclass
class Instinct:
    name: str
    category: str                              # perceptual/physics/goal/reasoning/meta
    when: str                                  # human-readable trigger
    trigger: Callable[[TurnContext], bool]
    surfaced_by: str = "system_prompt"
    content: Optional[Callable[[TurnContext], str]] = None   # for surfaced_by="registry"
    mandatory: bool = False
    satisfied: Optional[Callable[[dict], bool]] = None
    reprompt: Optional[str] = None
    gate: str = ""
    enabled: bool = True
    source: str = "core"


class InstinctRegistry:
    def __init__(self) -> None:
        self._instincts: dict[str, Instinct] = {}

    def register(self, inst: Instinct, *, override: bool = False) -> None:
        if inst.name in self._instincts and not override:
            raise ValueError(f"instinct {inst.name!r} already registered")
        self._instincts[inst.name] = inst

    def get(self, name: str) -> Optional[Instinct]:
        return self._instincts.get(name)

    def enable(self, name: str) -> bool:
        i = self._instincts.get(name)
        if i:
            i.enabled = True
        return bool(i)

    def disable(self, name: str) -> bool:
        i = self._instincts.get(name)
        if i:
            i.enabled = False
        return bool(i)

    def all(self) -> list:
        return sorted(self._instincts.values(), key=lambda i: (i.category, i.name))

    def firing(self, ctx: TurnContext) -> list:
        """Instincts whose trigger holds for this turn (enabled only)."""
        out = []
        for i in self._instincts.values():
            if not i.enabled:
                continue
            try:
                if i.trigger(ctx):
                    out.append(i)
            except Exception:
                continue
        return sorted(out, key=lambda i: (i.category, i.name))

    def render_active(self, ctx: TurnContext) -> str:
        """The prompt block for firing instincts whose content lives HERE
        (surfaced_by='registry').  The rest are surfaced by their own modules."""
        blocks = []
        for i in self.firing(ctx):
            if i.surfaced_by == "registry" and i.content is not None:
                try:
                    blocks.append(i.content(ctx))
                except Exception:
                    continue
        return "\n".join(b for b in blocks if b)

    def mandatory_firing(self, ctx: TurnContext) -> list:
        return [i for i in self.firing(ctx) if i.mandatory]


REGISTRY = InstinctRegistry()


def instinct(**kw) -> Instinct:
    """Build + register an Instinct; returns it."""
    reg = kw.pop("registry", None) or REGISTRY
    inst = Instinct(**kw)
    reg.register(inst)
    return inst


# -----------------------------------------------------------------------------
# Satisfaction check shared with the driver (kept here so the instinct owns it)
# -----------------------------------------------------------------------------


def has_animation_analysis(reply) -> bool:
    """True if a delta reply carries a SUBSTANTIVE animation_analysis (entities /
    movements named, or an explicit win_understood statement)."""
    if not isinstance(reply, dict):
        return False
    aa = reply.get("animation_analysis")
    if not isinstance(aa, dict):
        d = reply.get("delta")
        aa = d.get("animation_analysis") if isinstance(d, dict) else None
    if not isinstance(aa, dict):
        return False
    if isinstance(aa.get("win_understood"), str) and aa["win_understood"].strip():
        return True
    for k in ("entities", "movements", "entity_event_relations"):
        v = aa.get(k)
        if isinstance(v, (list, str)) and len(v) > 0:
            return True
    return False


# -----------------------------------------------------------------------------
# The animation-first instinct — LIVE (content rendered here, ENFORCED)
# -----------------------------------------------------------------------------


_ANIMATION_CONTENT = """
ANIMATION FILMSTRIP (image `animation_filmstrip.png`) — the {n} ACTUAL sub-frames
of this action's animation, in time order (frame 0 = just after the action, the
last = the settled frame), each with the tick grid.

INSTINCT — ANIMATION-FIRST MECHANIC DISCOVERY (MANDATORY unless you can already
STATE the win condition).  The motion across these frames is usually the mechanic,
and the SETTLED frame alone HIDES it — an object can launch, sweep, and return to
rest, so the before/after looks "unchanged" (this exact trap has caused real
mis-reads of "inert" actions).  So UNLESS you already understand the win
condition, you MUST inspect these frames yourself and FILL the `animation_analysis`
field of your reply:
  1. ENTITIES — the objects visible/active across the animation frames.
  2. MOVEMENTS & PROPERTY CHANGES — for EACH, how it moves / grows / shrinks /
     recolours / appears / vanishes across the frames.  Object constancy: a
     region in frame k and a similar region nearby in frame k+1 is the SAME
     entity that moved, NOT two things.
  3. ENTITY<->EVENT RELATIONS — what your action CAUSED, which entity is
     controllable/affected, and what the motion reveals about the rule or win.
Believe the FRAMES over the colour-region text summary when they disagree.  You
may request `animation_zoom` (via visual_queries) to magnify any region across
all frames before answering.
"""

_ANIMATION_REPROMPT = """\
# ANIMATION-FIRST INSTINCT — required animation analysis

Model handle: human:claude

Your last reply did NOT include the `animation_analysis`, but this action played a
{n}-frame ANIMATION and you have not yet established the win condition.  The motion
across those frames is the mechanic and the settled frame hides it — so analysing
it is MANDATORY here.

Re-open the attached `animation_filmstrip.png` (and `animation_zoom` a region via
`visual_queries` if you need detail), then reply with ONLY this JSON:

{{
  "animation_analysis": {{
    "entities":   ["<objects visible/active across the animation frames>"],
    "movements":  ["<entity X: how it moved / grew / shrank / recoloured /
                    appeared / vanished across the frames (object constancy)>"],
    "entity_event_relations": ["<what your action caused; which entity is
                    controllable/affected; what the motion reveals about the
                    rule or win>"]
  }}
}}

If — and only if — you can now plainly STATE the win condition, you may instead
reply {{"animation_analysis": {{"win_understood": "<the win condition in one
sentence>"}}}}.

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


instinct(
    name="animation_first",
    category="perceptual",
    when="an action produced an animation (>=2 sub-frames) AND the win condition is not yet understood",
    trigger=lambda c: c.has_animation and c.n_frames >= 2 and not c.win_understood,
    surfaced_by="registry",
    content=lambda c: _ANIMATION_CONTENT.format(n=c.n_frames),
    mandatory=True,
    satisfied=has_animation_analysis,
    reprompt=_ANIMATION_REPROMPT,
    gate="exempt once COS holds a credible win condition, or the reply states the win (animation_analysis.win_understood)",
)


# -----------------------------------------------------------------------------
# Catalogue of the other instincts — organized by TRIGGER.  Content currently
# lives in the named module (surfaced_by); registered here so this file is the
# single index of every instinct and WHEN it fires.  (Migrating their surfacing
# to surfaced_by="registry" — so they too are trigger-gated rather than always-on
# prose — is the natural follow-up.)
# -----------------------------------------------------------------------------


def _always(_c):  # ambient instincts — always relevant
    return True


for _name, _cat, _when, _trig, _src in [
    # --- perceptual ---
    ("structure_at_all_scales", "perceptual",
     "always (perception)", _always, "system_prompt"),
    ("grid_as_scaffold_pixels_as_truth", "perceptual",
     "a tile grid was inferred", lambda c: c.is_grid, "system_prompt"),
    ("transient_flash_is_state_change", "perceptual",
     "always (perception)", _always, "system_prompt"),
    # --- physics ---
    ("physics_first", "physics",
     "the scene reads as physical (acting)", _always, "system_prompt"),
    ("frame_orientation_can_flip", "physics",
     "the vertical layout looks inverted (source low, targets high)",
     _always, "system_prompt"),
    ("mechanic_stability_verify_on_open", "physics",
     "a new level just opened (carried beliefs unverified here)",
     lambda c: c.phase == "level_start", "driver"),
    # --- goal ---
    ("goal_grounding_diff", "goal",
     "a region depicts the desired end-state",
     lambda c: c.goal_depicted, "system_prompt"),
    ("goal_priors_similarity_to_coincide", "goal",
     "win unknown AND no score gradient (instinct-proposed temporary goals)",
     lambda c: (not c.win_understood) and (not c.score_gradient), "driver"),
    ("controllable_to_target", "goal",
     "an action relocates a controllable entity",
     lambda c: c.has_agent, "system_prompt"),
    # --- reasoning ---
    ("predict_then_falsify", "reasoning",
     "every acting turn", lambda c: c.phase == "strategy", "system_prompt"),
    ("occams_razor", "reasoning",
     "always (ranking interpretations)", _always, "system_prompt"),
    ("configuration_space_enumeration", "reasoning",
     "a planned action failed to achieve its goal",
     lambda c: c.last_action_failed, "system_prompt"),
    ("undo_first_recovery", "reasoning",
     "stuck after forward-direction moves (ACTION7 ~= undo)",
     lambda c: c.stuck, "system_prompt"),
    ("impossibility_needs_relation", "reasoning",
     "the reply claims something is impossible/dead-end (ENFORCED by the validator)",
     _always, "validator"),
    ("look_dont_conclude", "reasoning",
     "a CONTRADICTION appeared (an expectation was violated / the outcome surprised you / stuck). "
     "Reasoning is your primary, cheaper tool and is fine -- the rule is only that you must NOT "
     "push THROUGH a contradiction on reasoning or trust MEMORY there: drop to the CURRENT RAW "
     "IMAGE and RE-MEASURE to find the false belief (reasoning misfires routinely: stale cache, a "
     "read at the wrong position, the gridded display read instead of the clean frame, a canned "
     "animation mistaken for the real effect). On a contradiction, the answer is in the image.",
     lambda c: c.stuck, "system_prompt"),
    ("curiosity_on_novel_element", "reasoning",
     "a salient element is NOVEL or DIFFERENT from a known pattern / a prior level -- e.g. a "
     "legend/control/key ICON that changed shape, a new colour, an unfamiliar glyph. That "
     "difference is a strong signal carrying the new mechanic: AUTOMATICALLY curiosity-probe it in "
     "DETAIL right away (study the icon + guess its meaning, then interact with it ONE at a time "
     "and read its actual effect) BEFORE planning. Do NOT assume it behaves like before, and do "
     "not waste probes elsewhere while an obvious novelty sits unexamined.",
     _always, "system_prompt"),
    ("learn_from_recurring_errors", "reasoning",
     "keep an ERROR LEDGER of your OWN significant mistakes (error_ledger.py) and learn from it. In "
     "areas where you have repeatedly erred -- telling the MOVER from the GOAL, missing a detail on a "
     "LEGEND button, reading a gridded frame as set bars -- invest EXTRA scrutiny and DOUBLE-CHECK "
     "against ground truth, ESPECIALLY when a VARIATION (size / orientation / colour / position) has "
     "appeared, because a variation is exactly what triggers the repeat. For MOVER vs GOAL: the glyph "
     "keeps its SHAPE across levels but varies in size/orientation/position -- so match by SHAPE, "
     "never by size or orientation.",
     _always, "system_prompt"),
    ("decode_identity_before_probe", "reasoning",
     "to learn what a group of similar elements (a candidate key/legend/control "
     "set/palette) ENCODES, READ each member's static content + compare -- distinct "
     "content => an index/key. Do NOT infer identity from an interaction-response, "
     "and NEVER refute 'these encode distinct values' from a probe that produced no "
     "change (a null/HUD-only response is inconclusive, not disconfirming).",
     _always, "system_prompt"),
    # --- meta ---
    ("reflection_moves", "meta",
     "stuck (no progress for N turns / a blocked goal)",
     lambda c: c.stuck, "reflection_moves"),
    ("kb_recall", "meta",
     "stuck — surface stored solutions/lessons",
     lambda c: c.stuck, "kb_recall"),
    ("non_grid_curiosity_probe", "meta",
     "non-grid scene / no controllable agent localized",
     lambda c: (not c.is_grid) or (not c.has_agent), "driver"),
    ("first_level_is_tutorial", "meta",
     "WIN NOT YET UNDERSTOOD — read the frame as INSTRUCTIONS before probing. "
     "ARC-AGI-3 games are DESIGNED to be easy for a human player, so a game that "
     "looks incomprehensible almost always TEACHES its mechanic in the FIRST "
     "level: the designer DRAWS the rule into the scene. Look for the instructive "
     "elements and FOLLOW them first -- a legend/key, a ghost/preview or worked "
     "example, an AIM LINE / dashed trajectory / arrow showing a path, a "
     "highlighted target. (Use detect_trajectory to read an aim line, decode_panel "
     "to read a key.) Treat those marks as the designer's tutorial and act on what "
     "they SHOW; do NOT brute-force the mechanic by blind probing while an "
     "on-screen instruction is sitting unread. This fires from the FIRST turn "
     "(not only once you are stuck) -- read the tutorial BEFORE spending actions.",
     lambda c: not c.win_understood, "registry"),
]:
    instinct(name=_name, category=_cat, when=_when, trigger=_trig,
             surfaced_by=_src,
             # registry-surfaced instincts render their `when` text via
             # render_active() so the guidance actually reaches the VLM; the rest
             # are surfaced by their own modules.
             content=((lambda w: (lambda ctx: w)) (_when)) if _src == "registry" else None)
