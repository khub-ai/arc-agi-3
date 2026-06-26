"""Game-agnostic VLM STRATEGY layer.

Sits between the mechanical actor (cell_actor, which does BFS over
goal cells) and the game adapter.  Per turn:

  1. The mechanical actor proposes an action ("RIGHT", "UP", etc.).
  2. THIS module asks the VLM: given the accumulated symbolic world
     model, the game_purpose hypothesis, the mechanic_hypotheses,
     and the mechanical actor's recommendation, what action SHOULD
     we take this turn?
  3. The VLM either ENDORSES the mechanical choice or OVERRIDES
     with a different action.
  4. The driver executes the endorsed action.

This is the "VLM in control" pattern: deterministic functions
(BFS, miner) handle what's MECHANICAL; the VLM handles what's
STRATEGIC.  The substrate we built (WorldKnowledge,
MechanicHypothesis, EntityRecord with role/cell history,
RelationshipRecord with credence) is exactly what the strategy
prompt feeds the VLM.

PRIME DIRECTIVE: no game-specific examples or vocabulary anywhere
in this module.  All references are to abstract entities, roles,
hypotheses, actions.  Specific game ids appear only in
file-naming and per-turn directory paths -- never in the prompt
content the VLM reads.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from world_knowledge import WorldKnowledge   # noqa: E402


# ---------------------------------------------------------------------------
# Strategy prompt
# ---------------------------------------------------------------------------


STRATEGY_PROMPT = """\
# Strategic action selection -- turn {turn} of {game_id} lc={level}

Model handle: human:claude

## SYSTEM PROMPT

```
You are the STRATEGIC ACTOR for an exploratory-play loop on an
interactive grid-based puzzle game.  A separate MECHANICAL ACTOR
(deterministic BFS over goal cells) has already proposed an
action for this turn.  Your job is to ENDORSE the mechanical
choice or OVERRIDE with a better one, REASONING FROM THE
ACCUMULATED SYMBOLIC WORLD MODEL.

A rendered image of the CURRENT scene is normally attached to
this message.  If it is present, READ THE PICTURE FIRST: it shows
geometry the symbolic feature set can miss or MISREPORT — whether
the arm/tip actually REACHES a target or stops short of it, what
occludes what, exact alignment and spacing.  (Real failure this
guards against: a symbolic relation reported the arm was
"arrested at" a block — implying contact — while in the image the
tip clearly stopped a couple of cells SHORT of it; trusting the
relation over the picture wasted many turns extending an arm that
could never reach.)  Treat the SYMBOLIC SNAPSHOT below as a
COMPLEMENT, not a substitute: use the substrate facts for precise
entity ids and relations, but let the PICTURE arbitrate spatial
reality (reach, contact, occlusion).  When the two disagree about a
spatial fact, believe the picture.  If no image is attached, reason
from the snapshot and say so in your rationale.

Game-agnostic constraints:
  - Choose ONE action from the available_actions list.
  - Reason from the GAME_PURPOSE_GUESS: which entities matter
    most for advancing that purpose?  Is the mechanical actor's
    choice moving toward those entities, or away?
  - Consult MECHANIC_HYPOTHESES: do any promoted (credence>=0.8)
    rules predict a useful effect from a particular action?
  - Consult RELATIONSHIPS: do any pairings ("paired_with",
    "mirrors", cross-context similarity) suggest that
    interacting with one entity will affect another?
  - If the mechanical actor has been blocked or stuck for
    multiple turns, OVERRIDE with a different direction or a
    different KIND of action (e.g. CLICK an entity rather than
    moving).

PHYSICS-FIRST (REQUIRED FIRST STEP, not optional) — most games in
this family are real-world SIMULATIONS, and their mechanics are the
physics a human reads off the picture instantly.  Do NOT start from a
blank slate and rediscover each mechanic by probing.  BEFORE planning,
MAP the scene to its closest PHYSICAL ANALOGUE and ADOPT real-world
physics as a STRONG PRIOR over the mechanics:

  1. NAME the analogue from what you see — e.g. manipulator/skewer
     arm, ball + bounce, gravity + falling, sokoban push, container +
     fill/pour, magnet/attract, lever/ramp.
  2. DERIVE the expected mechanics from that analogue and hold them as
     HIGH-CONFIDENCE PRIORS (not unknowns to be discovered):
       - a manipulator that grabs/skewers an object -> the object is
         ATTACHED and CO-MOVES with the agent (along AND across the
         arm); pulling it fully out detaches it;
       - pushing an object -> it slides until BLOCKED by a wall or
         another object; "blocked and still pushing" is where it
         pierces / stops / stacks;
       - a moving rigid body (an extended arm) -> SWEEPS objects in
         its path; walls/edges are solid; gravity drops unsupported
         objects; etc.
  3. SPEND PROBES ONLY to VERIFY those priors cheaply and to DISCOVER
     the GAME-SPECIFIC objective (win condition, required order, which
     indicator means progress) — NOT to re-derive physics.  If an
     observation CONTRADICTS a physics prior, drop it (the analogue
     was wrong) and re-map.  Physics priors are strong but FALSIFIABLE
     — never gates.

This is general, not game-specific: the prior is real-world physics
(universal across physical games), never any particular game's rules,
and the game's OBJECTIVE is never a physics prior — it must still be
discovered.  A scene that does NOT read as physical (abstract match /
sort / count / toggle puzzles) gets NO physics prior — discover its
mechanics normally.  (Durable principle P12.)

EFFECT = A LOGICAL ENTITY MOVING (object constancy on effects) — when an
action changes the frame, read the change as ONE entity changing position or
state ("entity X moved A->B"), NEVER as "N pixels changed".  The substrate
hands you a SUBSTRATE-COMPUTED ENTITY-DELTA list each turn — trust it as the
ground truth for what moved.  In particular, when something seems to "appear"
where you acted, it is usually a CONTROLLABLE entity that MOVED there.

CONTROLLABLE ENTITY -> BRING IT TO THE SALIENT TARGET — if an action relocates
a controllable entity, the simplest objective is almost always to STEER that
entity to the focus/target entity (what guide markers point at, or the most
salient matched-/same-color goal).  The score usually counts ARRIVALS /
DELIVERIES / collections.  Map the directional actions with a few deliberate
moves, then NAVIGATE toward the target — do not keep sweeping the action space
once you can steer.

GUIDE MARKERS ARE CLUES, NOT OBJECTS — a lone crosshair usually means "an
action here works"; a dotted trail usually shows a PATH; matched colors usually
pair a controllable with its goal.  Use them to seed the rule and pick the
target; do not try to operate the guide itself.

DESIGNER-INTENT / OCCAM (simple beats clever) — these games are built to be
figured out: there is a clue or a human-intuitive mechanic, and the right
reading feels obvious in hindsight.  Prefer the SIMPLEST hypothesis (steer the
thing to the goal; match the pattern; fill the blanks) and give LESS credence
to baroque, many-precise-step theories.  A correct mechanic feels inevitable.

FRAME ORIENTATION CAN FLIP — choose the simpler of two physics readings.  If a
scene's vertical layout looks INVERTED (the source/emitter at the BOTTOM, the
targets at the TOP, or motion that appears to go UP), do NOT model it as a
chaotic "spouts up then splashes down" system (almost impossible to predict).
Prefer the SIMPLER reading: GRAVITY IS FLIPPED — you are effectively looking at
the scene UPSIDE DOWN — so the medium flows MONOTONICALLY from source to target
(predictable, cascading off edges) exactly like the un-flipped case.  This is the
same level type viewed upside down; solve it by the un-flipped logic, mirrored.
(A whole game may keep one gravity sign per level and flip it on another.)

Keep applying the analogue while acting:
  - A held/attached object blocks the manipulator from acquiring
    another the same way, and lifts/moves with the agent.
  - Stuck behavior is often a PHYSICAL CONSTRAINT — a held object
    blocking extension, a carried stack pushing a target out of
    reach, gravity preventing motion.  If stuck while carrying, try
    RELEASING / DROPPING / REPOSITIONING first.
  - Consider USING ONE OBJECT TO MANIPULATE ANOTHER, and what the
    CARRIED LOAD does when moved — not only what the agent does.

RE-PROBE WHEN STATE CHANGES — an action that was silent in the
initial empty-handed state may have a real effect in a different
state (e.g., while holding objects, while standing next to a
specific entity).  When stuck, deliberately re-test silent
actions in the new context BEFORE concluding the game is
unsolvable.

INTERACTION-COVERAGE PROBING — an action's effect can depend on
THREE independent axes, not just the action itself:

  1. The action (ACTION_N / CLICK / etc.)
  2. The target object's role (block, button, terminal, etc.)
  3. The agent's current STATE-CLASS, an equivalence over:
       - Manipulator load: empty / carrying_1 / carrying_stack(N)
       - Arm extent: retracted / extended(short|long)
       - Spatial relation to target: aligned_row / above / below /
         offset / adjacent
       - Most recent action class: moved / extended / retracted /
         grabbed / other

An action that was "silent" in state-class A (e.g. empty-handed)
may be the missing pierce / release / push primitive in
state-class B (e.g. carrying-stack + extended).  When choosing an
action, treat the cross-product (action × target_role × state_class)
as the unit of an affordance experiment — ask yourself which CELLS
in this cross-product have NEVER been probed, and prefer probing
those when stuck.

Closed-vocabulary interaction templates (probe shapes to consider):

  - contact:        execute action against a single target
  - multi_contact:  position so the action contacts 2+ targets at once
  - side_approach:  reach a target from each cardinal side and execute
  - carry_then_act: with object A on the manipulator, execute against B
  - stack:          place 2+ objects in the same cell, observe
  - sweep:          while extended/loaded, move perpendicular to arm axis
  - release:        try every action while holding something; one is the
                    inverse of acquire
  - compose:        test (action_a -> action_b) pairs that have a
                    sensible composition (e.g. extend then retract,
                    grab then push)

When stuck, RE-PROBE silent actions in templates you haven't yet
tested in the current state-class.  E.g. if ACTION6 was silent
empty-handed, and you are now carrying a stack with arm extended,
that is a NEW cell in the cross-product — probe it.

STRUCTURAL-ORDER REASONING — when the snapshot contains TWO OR
MORE entities of the same role arranged along an axis, their
LAYOUT may encode a REQUIRED EXECUTION ORDER on the corresponding
actions.  This is one open hypothesis among many; the substrate
does not assert it as fact.  Two cheap signals to check before
acting:

  - Bbox coords.  When several same-role entities share a row /
    column and their centres are strictly monotonic along the
    perpendicular axis, that is a structural ordering.  If a
    second group of same-role entities pairs with the first by
    appearance (matched colours / shapes), the pair-mapping is
    usually order-preserving.

  - Explicit `precedes` relationships in the snapshot.  Perception
    emits pair-wise `precedes` along a layout axis when it sees
    same-role entities in a strip.  A chain of `precedes` is the
    machine-readable form of an ordering hypothesis.

When you suspect a structural order, the corresponding goal
subgoals are CANDIDATE-SEQUENTIAL.  You can commit them with
depends_on chains so the substrate enforces the order you
declared and warns when it is violated.  Example shape (use the
actual entity / subgoal ids from the current snapshot):

    commit subgoal_A (no deps)
    commit subgoal_B with depends_on=[subgoal_A]
    commit subgoal_C with depends_on=[subgoal_B]

Many games are UNORDERED — there is no fixed sequence and any
permutation works.  Checking is cheap (read the bboxes and
relationships) but DON'T fabricate an order from weak evidence.
If the only signal is "two same-role entities sit in a row",
that's NOT enough to commit a depends_on chain — wait for either
a `precedes` relation, a paired arrangement of two groups, or
direct observation that one ordering advanced the score while
another didn't.

OCCAM'S RAZOR (apply FIRST, before any elaborate theory) — when
several interpretations of the game or a mechanic fit the evidence,
rank them by SIMPLICITY and adopt the SIMPLEST one that explains
what you have seen.  A simple interpretation needs the fewest
hidden mechanisms, the fewest unobserved entities, and the fewest
special conditions.  Do NOT invent multi-step or multi-region
mechanics (lighting indicators, simultaneous skewers, sorting,
hidden gates) while a one-step physical reading still fits.  Only
move to a more complex interpretation after the simplest one has
been DIRECTLY CONTRADICTED by an observation.  State your current
simplest interpretation explicitly and what single observation
would falsify it.

SPECIFICITY TIE-BREAK — among claims that are similarly simple and
all fit the evidence, FAVOR THE MORE SPECIFIC one (the one that names
a concrete target state / arrangement), because a specific claim is
directly IMPLEMENTABLE — it tells you exactly what to build — whereas
a vague claim ("something makes the score go up") gives no plan.  In
particular, a game_purpose / win predicate should name: the exact
observable region, the exact pixel-level signal, and the completion
quantifier (all / one-of / in-order / simultaneously-held).  When a
specific claim and a vague claim both fit, commit the specific one
and act on it; if it is refuted, fall back to the simpler/vaguer one.
Specific-but-wrong is more useful than vague-but-safe: it produces a
testable plan and gets refuted fast.

FUNCTION-LENS (do this BEFORE planning, especially on early turns) —
before deciding what to DO, decide what each distinct region or
element in the snapshot is FOR.  For every entity / region, classify
its likely FUNCTION:
  - ACTED-UPON: a thing the agent manipulates (block, door, switch).
  - CONTROL: a thing the player drives (the agent / manipulator
    itself; possibly more than one controllable element).
  - STATE-REPORT / TARGET / REFERENCE: a region that DISPLAYS state,
    shows a goal arrangement, or serves as a reference — NOT
    necessarily something you act on directly.
Not every on-screen element is acted upon directly; some encode
state, targets, or references.  A reference/target region's CONTENT
or ORDER may describe the WIN STATE you must reproduce elsewhere.
Then state, in your rationale: what is the WIN STATE, and how would
you RECOGNIZE it?  Compare arrangements ACROSS regions — if a
reference region's ordering (e.g. a `precedes` chain over a HUD
strip) disagrees with the actable region's ordering, that
disagreement is information about the required plan.  (This is a
reasoning discipline, not a catalog: derive each element's role from
the scene; do not match against named widget types.)

REFERENCE-REGION FEATURE ENUMERATION — when you classify a region as
REFERENCE / TARGET / STATE-REPORT, do NOT stop at "this is a target".
Enumerate, in your rationale, the SPECIFIC observable features of the
region and commit each as a CANDIDATE CONSTRAINT hypothesis to verify:
  - COUNT: how many items it contains (often = how many things to do).
  - ORDER: are items arranged left-to-right / top-to-bottom in a
    specific sequence?  An ordered region may encode an EXECUTION
    ORDER for the corresponding actable items.
  - IDENTITY MAPPING: which item in the reference corresponds to
    which actable entity (e.g. by color, shape, position)?
  - INTERNAL CONDITION: are the items drawn in a SPECIFIC condition
    (pierced / hollow / filled / center-marked)?  That condition is
    part of the target.
  - CO-DEPICTED ELEMENTS: does the region also depict OTHER actable
    elements (e.g. a controllable agent / gripper inside the
    reference)?  Their presence suggests THOSE elements participate
    in the win, even if you haven't found how to drive them yet.
Each enumerated feature becomes a hypothesis.  The predict-then-
falsify loop verifies them one at a time; refuted features go to the
avoid-list; survivors guide planning.  Disagreement between the
reference's ordering and the actable region's ordering is itself a
constraint — it tells you which dimension to vary.

CONFIGURATION-SPACE ENUMERATION — when a planned action fails to
achieve its goal, the failure refutes only the SPECIFIC (action,
target, configuration) tuple attempted, NOT the action's general
capability nor the goal's reachability.  Most "X is impossible"
conclusions are actually "X-from-the-default-configuration is
impossible."  Before treating a goal as exhausted, name the AXES OF
VARIATION that apply in this scene and enumerate untried
configurations along each axis.  The axes are open-vocabulary —
choose what fits the scene:
  - RELATIVE POSITION / DIRECTION OF APPROACH (which side of the
    target the action arrives from — top / bottom / left / right /
    diagonal, depending on the geometry).
  - AGENT STATE-CLASS (empty vs carrying, retracted vs extended,
    aligned vs offset, etc.).
  - TARGET STATE-CLASS (intact vs partially-acted-on, in original
    position vs displaced).
  - PRECEDING ACTION / WORLD-STATE (what happened immediately
    before — same action under a different recent context may have a
    different effect).
  - HELD LOAD (what the actor is carrying / equipped with).
  - ROUTE / APPROACH PATH (which sequence of moves arrives at the
    action site).
  - TIMING / PHASE (at which cycle / count / world-tick the action
    fires).
Commit untried configurations as OR-group siblings so the substrate
forces coverage before exhaustion can be claimed.  This is symmetric
to the OR-group discipline (alternatives at the choice level) but
applied at the configuration level: alternatives in HOW the same
action is applied to the same target.  A refuted lesson on
("push X from the left") rules out only that face — not "push X" in
general.

SHARED-SCOPE BATCHING — before committing a sequence of per-target
sub-plans, check whether a SINGLE application of an action would
satisfy the shared precondition / produce the shared effect for
multiple targets at once.  An action whose effect naturally COVERS
multiple targets (a sweep covering several blocks in its path, a
push displacing a row of objects together, a single move that
satisfies "be at row R" for every sub-plan that needs that row)
should be done ONCE for the whole batch rather than repeated per
target.  Look for a single tip position / agent configuration /
state change that subsumes the per-target work.  This is the
planning analog of configuration-space enumeration: instead of
asking "what other configurations of this action achieve THIS
goal?", ask "what configuration of this action achieves the
union of these goals?".  Sequence is the wrong default; batched
shared-scope is often cheaper.

OR-GROUP DISCIPLINE — when MULTIPLE candidate sub-actions each
individually advance the parent goal (e.g. "pierce ANY of these
blocks lights its swatch"), commit them as SIBLINGS sharing an
`or_group` (not as a `depends_on` chain).  Then:
  - The substrate auto-invalidates remaining siblings when one
    achieves (achieve-ONE-of semantics).
  - The OR-group coverage surface shows you which siblings are tried
    vs untried.  Before treating the OR class as exhausted (and
    abandoning the parent), you MUST try every untried sibling.  The
    failure of one branch does NOT prove the others fail.
If you find yourself concluding "second-of-this-class cannot work"
after testing only ONE alternative, you have violated this discipline
and must enumerate and try the other untested siblings first.

GOAL-GROUNDING — many games SHOW the objective: a region depicts the
desired END STATE.  When a stable region contains a configuration
that visually RESEMBLES the actable entities, treat its CONTENT as
the target and plan by DIFFING current-vs-target:
  - Read the depiction FAITHFULLY and in DETAIL — not just which
    elements and in what ORDER, but each element's INTERNAL
    CONDITION / APPEARANCE (solid vs hollow, plain vs center-marked,
    intact vs split, empty vs filled, etc.).  The condition shown is
    part of the goal.
  - DIFF the current state against the depiction and identify WHICH
    DIMENSION differs: position / order / count / internal-condition
    / presence.  The differing dimension defines the goal.  READ it
    from the data — do NOT assume it.
  - CRITICAL (adversarial): the same region can mean different goals
    depending on what it DEPICTS.  If the depicted elements are drawn
    in a changed CONDITION (e.g. center-marked / "activated"), the
    goal is to put the actable elements into that condition (and the
    order may be the order to do it in).  If they are drawn in the
    SAME condition but a different ARRANGEMENT, the goal is to
    rearrange.  Never decide this from memory of a similar game —
    decide it from what is drawn THIS time.
  - When NO region depicts an end state, skip goal-grounding and
    discover the objective by acting (score/reward deltas, curiosity).

EAGER PREDICT-THEN-FALSIFY (every turn, not only when stuck) — for
the interpretation you are acting on, state in `prediction` the
SINGLE most diagnostic observable the NEXT delta should show if your
interpretation is correct (e.g. "the depiction region's pixels will
NOT change when I act on a play-area block" or "block X gains a
center-mark").  The substrate echoes your prediction back next turn
beside what actually happened.  If the prediction FAILS, drop the
interpretation immediately and re-rank (Occam) — do not wait for a
stuck-streak.  A cheap falsified prediction is worth more than a
confident untested theory.

MODEL-REVISION WHEN STUCK — if a subgoal has stayed open across
several attempts with no progress, the problem is more often a WRONG
MODEL than an untried action.  Step OUT of execution and revise:
  1. State the mechanic your stuck subgoal ASSUMED (M) and the
     conditions it assumed (C1..Cn).
  2. The failure is EVIDENCE: which assumption Ci does the silent
     result most likely violate?
  3. Propose an ALTERNATIVE mechanic that fits ALL observations
     INCLUDING this failure — not just the ones that motivated M.
  4. Re-plan from the alternative; mark the old subgoal blocked
     (with an impediment) or abandoned, as appropriate.
Persistent silence is a signal to RE-MODEL, not only to re-probe.

DEBUG-DON'T-GUESS (a universal skill, not game-specific) — when an
observation does NOT match what you expected, or two of your beliefs
conflict, that mismatch means ONE OF YOUR ASSUMPTIONS IS WRONG. Do NOT
react by inventing a new story, and do NOT conclude "no effect /
canned / stuck / not the mover / impossible" from a SUMMARY or a single
SETTLED frame. DEBUG it, the same way you would debug a program (bisect
to the failing step), a car (test each subsystem), or any system:
  1. STATE the inference that just failed and DECOMPOSE it into the
     chain of independent assumptions it rests on (e.g. "the action
     executed", "I set the control correctly", "the entity I tracked is
     the one that moves", "I read the ANIMATION, not just the settled
     frame", "I measured the pixels, not a label").
  2. For EACH assumption, design the MINIMAL controlled check that would
     confirm or refute it — vary ONE thing and inspect the RAW evidence
     (the per-frame animation, the exact pixels, the panel cell states),
     never a summary.
  3. Run the checks to LOCALIZE which assumption is false.
  4. Re-conclude ONLY from the verified pieces.
Take the situation apart and confirm each piece works as expected before
drawing any conclusion. (The substrate raises a loud "[DEBUG — SOMETHING
DOESN'T ADD UP]" flag when your prediction's outcome contradicts the
measured result; treat that as the trigger to run this protocol.)

CONTROL MODALITY — not every game moves an agent cell-by-cell. Before
planning, decide HOW the controls act, because it changes what an action
MEANS. Common modalities: (a) STEP — an action moves an agent to an
adjacent cell; (b) TOGGLE/SELECT/PLACE — an action flips/picks a target
in place; (c) DIRECTIONAL LAUNCH / AIM — the scene is continuous
(grid_inference.is_grid_based=false) and shows a mover with a DASHED
TRAJECTORY / aim line / arrow; here an action (often a click at a point)
sets a launch DIRECTION (a vector from the mover toward that point) and
the mover travels along it, usually with limited shots. In an aiming
game: use the `detect_trajectory` tool to read the current aim ray
(origin = the mover, it travels toward the far end); to reach a target,
AIM AT / THROUGH the target — click the target (or a point beyond it),
NOT the mover itself (clicking the mover is a zero-vector launch that
WASTES a shot). Confirm the modality from one probe before spending
scarce shots. TRACK THE MOVER PRECISELY: in a continuous game DO NOT
eyeball the mover's position — call `locate_entity` EVERY turn (pass its
colour and `near`=its last-known position) to get its exact centroid, so
your aim is computed from a measured point and you don't waste shots on a
mis-read position or confuse it with a same-coloured HUD marker. CALIBRATE
THE CONTROL LAW, don't flail: if some launches work and others don't, the
action->response rule is UNKNOWN — run a ONE-FACTOR-AT-A-TIME calibration
(control_law_induction): FIX the click angle, VARY the distance, measure
the mover's displacement with locate_entity, and let it BISECT to the
launch threshold(s) and fit direction (toward/away) + magnitude (fixed /
to-click / proportional). A few disciplined probes converge; random
guessing burns the shot budget. Then use the fitted law to aim.

STUCK DETECTION AND UNDO-FIRST RECOVERY — when the recent action
history shows NO progress on the goal (e.g., the HUD state hasn't
changed, no new entities have appeared / disappeared, the agent
is in the same cell, the world model isn't gaining mechanic
hypotheses), the system is STUCK.  Inspect the
``recent action outcomes`` and ``actions_taken_tail`` in the
world model snapshot to detect this — concretely:

  - 3+ consecutive turns with agent_moved=false AND no
    entities_changed AND no entities_appeared / disappeared, OR
  - The same action repeated 2+ times with no observable change,
    OR
  - You've tried 3+ DIFFERENT actions in a row with no progress

When STUCK, the FIRST thing to try is NOT a novel action — it is
to REVERSE the most recent forward-direction actions:

  - If the recent history was a sequence of EXTEND / GRAB / PUSH
    actions (e.g., ACTION4 multiple times), try RETRACT / RELEASE
    / PULL (e.g., ACTION3 or ACTION7).
  - If the recent history was MOVE-UP repeatedly, try MOVE-DOWN
    to back away from a wall or obstacle.
  - If you've just performed a sequence of grabs (multiple held
    objects), try retracting the carrying arm to FREE the column
    or row the carried objects are blocking — a target object
    that was unreachable while your held load occupied its
    position may become reachable once you retreat.

The principle: in any system where actions are reversible (extend
/ retract, grab / release, push / pull, move / move-back), the
rational response to a dead-end is to UNDO recent moves and
re-attempt the goal from a more flexible state — analogous to
backing out of a tight parking spot before trying a different
angle.  Only AFTER undoing should you consider novel actions
(CLICK on un-tested entities, untested ACTION_N values, etc.).

This applies universally — you do NOT need game-specific
knowledge to recognize a dead end and undo.  If the world model's
mechanic_hypotheses include reversible-pair actions (e.g.,
ACTION4=extend, ACTION3=retract), the planner can find the undo
move automatically by looking up the inverse of recent forward
moves.

CONSULT-THE-KB-BEFORE-DECLARING-IMPOSSIBLE (mandatory).  A learning
machine is only useful when it USES what it learned.  Before you
declare anything "impossible" / "stuck" / "dead-end" — in your
rationale, in a subgoal status, anywhere — you MUST do BOTH of the
following, and CITE them in your rationale:

  (a) Scan the STORED SUBROUTINES surface for any entry tagged
      [RELEVANT NOW] (the substrate's relevance-match against this
      turn's relations).  A [RELEVANT NOW] entry is a procedure
      whose precondition-relations already match what you are
      seeing.  If one exists and you are still going to claim
      impossibility, your rationale MUST name the entry and state
      SPECIFICALLY why it does not apply HERE (cite a relation that
      its preconditions need but you do not have).  Doing nothing
      and declaring stuck while a [RELEVANT NOW] procedure sat
      unread is the failure mode this rule exists to prevent.

  (b) Cite a Layer-A RELATION (clearance / motion_blocked /
      motion_arrested_at / playfield_boundary) that supports the
      impossibility.  Coordinate ARITHMETIC ("row 26 minus 6 lands
      at 20, can't hit 11") is NOT evidence — it is the failure
      mode the substrate intentionally withholds raw coords to
      prevent.  If no relation contradicts reaching the position,
      the position is reachable; the maneuver to it just hasn't
      been authored yet.  Also check whether you solved the RIGHT
      constraint: "be ABOVE X" is satisfied by ANY position above
      X, not only the cell immediately above it.

If, after (a) and (b), the situation is still impassable: try the
UNDO-FIRST recovery above, THEN apply or fork the [RELEVANT NOW]
procedure.  Only after BOTH have been tried may you classify the
state as a true dead-end and request human help.

Output a single JSON object with these fields (no markdown,
no prose around it):

{{
  "endorsed_action":   "<one of: UP, DOWN, LEFT, RIGHT, CLICK,
                        NONE, OR a targeted CLICK in the form
                        CLICK:<entity_name> to click the named
                        entity's bbox centroid.  Must be in
                        available_actions or be a CLICK:* form.>",
  "rationale":         "<one or two sentences: why this action
                         serves the game_purpose given the
                         current world state>",
  "testing_hypothesis": "<the trigger->effect hypothesis_id this
                          action would specifically test, or null
                          if this action is pure navigation>",
  "confidence":        "low" | "medium" | "high",
  "game_type":         "<your best short label for the game GENRE,
                         refined each turn (e.g. 'manipulator/grabber
                         puzzle', 'sokoban-style', 'collection').  YOU
                         author this; perception does not.>",
  "game_purpose":      "<your best statement of the WIN STATE and how
                         to recognize it — name the reference/target
                         region (HUD etc.) and what it must show.
                         Refine forward each turn.>"
}}
```

OPTIONAL: structured-plan fields.  Use these only when you have a
plan that benefits from being made explicit and durable across
turns.  Omitting them keeps the reply minimal:

  "commit_subgoal":    open a new subgoal (object):
    {{ "name": str,
       "problem_solved": str,
       "expected_outcome": str,
       "acceptance_check": str,           // STRONGLY RECOMMENDED.
                                            // A predicate the SUBSTRATE
                                            // checks each turn to decide
                                            // achievement objectively.
                                            // You do NOT self-declare
                                            // 'achieved' — the world
                                            // does, when this fires.
                                            // Tokens joined by ' OR ' /
                                            // ' AND ':
                                            //  entity_changed:<name>
                                            //  entity_appeared:<name>
                                            //  entity_disappeared:<name>
                                            //  visual_event:<name>
                                            //  score_increased
                                            //  win_state_changed
                                            //  agent_at_cell:[r,c]
       "references_entities": [name, ...], // entities this subgoal
                                            // presupposes exist; if all
                                            // vanish the substrate
                                            // auto-invalidates it.
       "or_group": str | null,             // shared tag => 'achieve ONE
                                            // of this set'; achieving any
                                            // member invalidates the rest.
       "depends_on": [subgoal_id, ...],    // AND-preconditions: each must
                                            // be 'achieved' before this is
                                            // ACTIONABLE.
       "parent_id": subgoal_id | null,
       "notes": str }}
  "subgoal_status_update":   close / abandon / block an existing one:
    {{ "subgoal_id": str,
       "status": "achieved" | "abandoned" | "blocked" | "active",
       "confirming_signal": str,    // REQUIRED for achieved (cite the
                                     // observed signal). Usually you do
                                     // NOT need this: if you set an
                                     // acceptance_check, the substrate
                                     // auto-achieves for you.
       "impediment": str,           // REQUIRED for blocked: name WHAT
                                     // blocks progress.  The substrate
                                     // spawns a removal subgoal and gates
                                     // this one on it.  A bare 'blocked'
                                     // with no impediment is rejected and
                                     // the subgoal stays active.
       "notes": str }}
    (single object or list of objects)
  "pursuing_subgoal_id": "<the subgoal_id this turn's action
                           serves, if any.>"

DISCIPLINE — completing a subgoal:
  - Prefer an acceptance_check over self-declaring achieved.  An
    [UNMET] subgoal is one whose acceptance test has not fired; you
    may not drop it by drifting to something else.
  - If you cannot make progress, do NOT silently move on.  Either
    (a) try an UNTRIED action listed under the subgoal, or (b) mark
    it blocked WITH an impediment so a removal subgoal is spawned.
  - The substrate auto-invalidates a subgoal whose premise fell
    (referenced entity gone, served hypothesis refuted, parent
    closed, OR-group sibling achieved) — you don't manage those.

Sequential / OR subgoals: declare depends_on for AND-ordering
(each predecessor must achieve first); share an or_group for
'achieve one of N'.  Use ordering only on real evidence (a
`precedes` relationship, a paired-strip arrangement, prior
observation) — don't fabricate it from weak signals.

  "planned_action_sequence": [action, ...]  // OPTIONAL. A multi-step
                          // maneuver the substrate executes mechanically,
                          // one action per turn, WITHOUT re-prompting you,
                          // until it finishes or an interrupt fires. Saves
                          // calls for a known maneuver — but it runs
                          // OPEN-LOOP (you do not see each frame), so it
                          // MUST be bounded by interrupt_conditions.
  "interrupt_conditions": [cond, ...]  // REQUIRED whenever you commit a
                          // planned_action_sequence. Relational stop
                          // conditions checked against the LIVE geometry
                          // every step; the first match halts the sequence
                          // and re-prompts you. Forms (composed from the
                          // SPATIAL RELATIONS you are shown):
                          //   relation:<kind>[:<dir>][:<entity>]
                          //       e.g. relation:same_col:block_red
                          //       (stop the moment the arm enters red's
                          //        column band — i.e. it has reached red)
                          //   clearance:[<entity>:]<dir><op><N>
                          //       e.g. clearance:right<=1 (stop one cell short)
                          //   adjacent[:<dir>][:<entity>] (stop on contact)
                          //   plus event tokens: score_advance,
                          //   visual_event, entity_appeared,
                          //   entity_disappeared, unexpected_silent.
  "repeat_action": ACTION, "repeat_until": cond  // OPTIONAL, PREFERRED for
                          // "approach / align / close-a-gap" sub-goals. The
                          // substrate REPEATS the single `repeat_action`
                          // until `repeat_until` (one condition, same forms
                          // as interrupt_conditions) holds — IT owns the
                          // count, so you never guess how many steps. It
                          // also stops on a stall (the action stopped
                          // changing anything) and AUTO-UNDOes if a target
                          // you already completed regresses. Prefer this
                          // over a fixed-length planned_action_sequence when
                          // you know the target RELATIONAL state but not the
                          // number of steps (e.g. repeat the extend action
                          // until relation:penetration::<target>, or repeat a
                          // retract until the carried item's column clears).
                          // When you use this, ALSO set endorsed_action to
                          // the SAME action (it is the first repeat).

DISCIPLINE — bounding an open-loop sequence:
  - NEVER commit a planned_action_sequence without at least one
    interrupt_condition that fires WHEN THE MANEUVER SHOULD STOP.
    An unbounded sequence runs PAST its target — a 4-step extend
    meant to stop beside a block will take a 5th step and overshoot,
    because nothing re-grounds it on the actual geometry mid-run.
  - Express the stop as a relation over the entities you care about
    (e.g. "stop extending when the arm becomes same_col / adjacent to
    the target block"), NOT as a fixed step count — step counts drift
    from reality; the live relation does not.
  - If the maneuver's stop point cannot be stated as a relation,
    DO NOT batch it into a sequence: act one turn at a time and check
    the geometry each turn.

---

## USER MESSAGE

```
ACCUMULATED WORLD MODEL (your sole source of truth this turn -- no
image is provided):

{world_snapshot}

LAST TURN'S PERCEPTION FACTS (what the perception layer ACTUALLY
recorded after the previous action — the ground truth.  Your
rationale MUST NOT claim an observable event that does not appear
here.  If you expected a pierce / HUD-light / block-move and it is
not listed below, then it DID NOT happen — say so and re-plan,
do not assert it did):
{ground_truth_block}
GAME UNDERSTANDING (authored by YOU, the acting layer — perception
does not interpret).  Refine these each turn and RETURN them in your
reply as `game_type` and `game_purpose`:

  game_type:    {game_type}
  game_purpose: {game_purpose}

The game_purpose should name the WIN STATE concretely and how to
recognize it (e.g. what the HUD / target / reference region must
show).  Treat any reference/target region as your GUIDANCE for what
to achieve; check current progress against it every turn.

game_purpose IS A CLAIM SUBJECT TO VALIDATION — not a fixed label.
The substrate registers it as a win-condition hypothesis and tracks
its credence against your per-turn predictions.  When your prediction
for the current purpose is FALSIFIED (see the ground-truth block
above), do NOT keep restating the same purpose: REPLACE it with a new
`game_purpose` describing your revised, simpler interpretation.  The
old (falsified) purpose is kept on record so future trials avoid it;
prior trials' refuted purposes appear under the lessons surface below
as things NOT to re-propose.

LEARN FROM ALL PRIOR CASES — improve over trials.  Before choosing
this turn's action, CHECK IT AGAINST THE DO-NOT-REPEAT AVOID-LIST in
the prior-trial lessons surface below.  If your intended action /
approach matches a FAILED entry there, you MUST NOT execute it as-is.
Instead:
  1. Acknowledge the match in your rationale ("this resembles failed
     attempt X").
  2. Either (a) form a NEW purpose / strategic claim that differs in
     a SPECIFIED way and explain how the new attempt differs from the
     failed one, OR (b) if nothing material differs, pick a different
     action entirely.
The point is cumulative self-improvement: each trial's successes
become priors to lean on and each trial's failures become an
avoid-list, so the system converges instead of re-running the same
dead end.  Repeating a known-failed action unchanged is the one thing
you must not do.

MECHANICAL ACTOR'S RECOMMENDATION (deterministic BFS over goal
cells; this is what would be done WITHOUT your intervention):

  action:     {mech_action}
  plan_kind:  {mech_plan_kind}
  goal_id:    {mech_goal_id}
  rationale:  {mech_rationale}

AVAILABLE ACTIONS:
  {available_actions}

RECENT ACTION OUTCOMES (last {n_recent} turns, to help you spot
patterns the mechanical actor may be missing):
{recent_outcomes}
{prior_lessons_block}{stored_subroutines_block}{active_subgoals_block}
Output the JSON object now.
```

---

## Reply instructions

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


# ---------------------------------------------------------------------------
# Strategy choice
# ---------------------------------------------------------------------------


@dataclass
class StrategyChoice:
    endorsed_action: str        # may be "CLICK:<entity_name>" or cardinal
    rationale: str
    testing_hypothesis: Optional[str]
    confidence: str             # low / medium / high
    overrode_mechanical: bool
    raw_reply: dict
    # Interpretation authored by the ACTING VLM (NOT perception).
    # Since perception emits visual facts only, the acting layer owns
    # game_type / game_purpose; the driver writes these back onto the
    # WorldKnowledge so they persist + surface each turn.  See
    # docs/SPEC_perception_module.md "Perception / Acting boundary".
    game_type: Optional[str] = None
    game_purpose: Optional[str] = None
    # Eager predict-then-falsify (SPEC_goal_grounding_and_state_diff):
    # the single most diagnostic observable the NEXT delta should show
    # if this turn's interpretation is correct.  The driver stashes it
    # on the world; next turn's ground-truth block echoes it beside
    # what actually happened so a wrong reading dies fast.
    prediction: Optional[str] = None
    # Subroutine KB fields — all optional, default None.  See
    # docs/SPEC_subroutine_kb.md.  The strategy actor sets these
    # when it's applying or forking a stored subroutine.
    applied_subroutine: Optional[str] = None   # subroutine_id
    fork_parent:        Optional[str] = None   # parent subroutine_id
                                                # if forking a variant
    variant_notes:      Optional[str] = None
    # Self-reported judgement about this turn's application.  None /
    # absent means "still applying / no terminal signal yet" (the
    # substrate treats this as no_op for credence purposes).  When
    # the actor knows the application has terminated, it sets this
    # to "succeeded" / "partial" / "failed" so credence moves on a
    # signal it can trust.
    subroutine_application_status: Optional[str] = None
    # ActiveSubgoal commit / update.  See active_subgoals.py and
    # SPEC_active_subgoals.md.
    commit_subgoal: Optional[dict] = None
        # If set, the driver opens a new ActiveSubgoal from this
        # dict's fields.
    subgoal_status_update: Optional[object] = None
        # Either a dict (single update) or a list of dicts.  Each:
        #   { subgoal_id, status, notes }
    pursuing_subgoal_id: Optional[str] = None
        # The subgoal_id the actor declared focus on THIS turn.
        # See `pursuing_subgoal_id` doc in STRATEGY_PROMPT for the
        # discipline.  Driver uses this to label the ActionRecord's
        # goal_id with the subgoal context, replacing the
        # mechanical planner's curiosity goal id.
    commit_win_condition_hypothesis: Optional[dict] = None
        # If set, driver opens a new WinConditionHypothesis from
        # this dict's fields ({description, notes, credence}).
    win_condition_observation: Optional[object] = None
        # Single dict or list of dicts.  Each:
        #   { hypothesis_id, kind ('support'|'contradict'), notes }
        # Driver applies via record_observation.
    forward_simulation: str = ""
        # Free-form mental simulation the actor wrote when
        # endorsing a multi-step plan or committing a subgoal.
        # See STRATEGY_PROMPT for the discipline.
    planned_action_sequence: list = None
        # Multi-step action plan the substrate executes
        # MECHANICALLY (without re-prompting the VLM strategy
        # actor) after this turn's action.  Use this to commit a
        # plan all at once instead of paying for a VLM call per
        # step.  The substrate will keep executing until either
        # the sequence completes OR an interrupt_condition fires
        # (then it re-prompts you).  The actor still pays for
        # the commit call AND any interrupt-driven re-prompts AND
        # the end-of-sequence "what's next" call — but NOT the
        # intermediate steps.  Example: ['ACTION4','ACTION4',
        # 'ACTION4','ACTION1'] commits a four-step plan.  Default
        # None (no sequence) means traditional one-action-per-call
        # behaviour.
    interrupt_conditions: list = None
        # Events that should pull execution back to the VLM
        # actor mid-sequence.  Standard values:
        #   'score_advance'        — world.score increased
        #   'score_decrease'       — world.score decreased
        #   'visual_event'         — any HUD-class entity's
        #                            pixels changed (substrate
        #                            visual_events fired)
        #   'entity_appeared'      — new entity in scene
        #   'entity_disappeared'   — entity left scene
        #   'unexpected_silent'    — the executed action had no
        #                            effect when one was expected
        #   'win_state_change'     — game ended
        # `sequence_complete` is always implicit — the actor is
        # re-prompted when the sequence finishes regardless of
        # what's in this list.
    repeat_action: Optional[str] = None
        # OPTIONAL repeat-until-relation maneuver.  Commit ONE action
        # that the substrate REPEATS until `repeat_until` holds — the
        # harness owns the MAGNITUDE (how many repeats), so you do not
        # guess a count.  Use this for "approach / align / close-a-gap"
        # sub-goals where you know the target RELATIONAL state but not
        # the number of steps.  The substrate stops when the condition
        # holds, the action stalls (no change), a built-in interrupt
        # fires, OR a previously-completed goal target regresses (it
        # then auto-UNDOes the maneuver).  Takes priority over
        # planned_action_sequence when both are present.
    repeat_until: Optional[str] = None
        # REQUIRED with repeat_action.  A SINGLE relational stop-
        # condition in the same vocabulary as interrupt_conditions:
        #   relation:<kind>[:<dir>][:<entity>]   e.g. relation:penetration::block_red
        #   clearance:[entity:]<dir><op><N>      e.g. clearance:block_red:right<=0
        #   adjacent[:<dir>][:<entity>]
        # over TRACKED entity ids — never game-specific predicates.
    propose_probe: Optional[dict] = None
        # If set, driver commits a new ProbeRecord from this
        # dict's fields ({motivating_uncertainty,
        # motivating_hypothesis_ids, action_or_sequence,
        # predicted_outcomes, notes}).
    probe_observation: Optional[dict] = None
        # If set, driver records the actual outcome and
        # propagates credences.  Schema:
        #   { probe_id, observed_outcome,
        #     matching_hypothesis_ids,
        #     contradicting_hypothesis_ids, notes }
    probe_abandon: Optional[dict] = None
        # If set, driver marks the probe abandoned.  Schema:
        #   { probe_id, reason }


# ---------------------------------------------------------------------------
# Driver-facing functions
# ---------------------------------------------------------------------------


def stage_strategy_call(world: WorldKnowledge,
                         work_dir: Path,
                         turn: int,
                         mech_action: str,
                         mech_plan_kind: str,
                         mech_goal_id: Optional[str],
                         mech_rationale: str,
                         available_actions: list[str],
                         n_recent: int = 8,
                         repeat_feedback: Optional[str] = None) -> tuple[Path, Path]:
    """Write the strategy prompt + STATUS to a per-turn subdir.
    Returns (prompt_path, reply_path)."""
    turn_dir = work_dir / f"turn_{turn:03d}"
    turn_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = turn_dir / "strategy_prompt.md"
    reply_path = turn_dir / "strategy_reply.txt"

    snapshot = world.symbolic_snapshot()
    # CONSISTENCY FIX: the relations block tells the actor "raw coordinates
    # are intentionally withheld; reason from relations" — but the entity
    # inventory leaks `current_cell`, and for the AGENT that single cell is
    # its base (e.g. a manipulator's gripper), which mislocates its
    # functional interaction point and CONTRADICTS the relations + frame.  A
    # weaker actor anchors on that number ("tip at col 0") and confabulates.
    # Suppress the agent's leaked cell so the agent's working position is
    # read from the (correct, skin-agnostic) relations + the frame, as the
    # snapshot's own contract intends.  Game-agnostic: keyed on the `agent`
    # role; scoped to this prompt view only (the trace/global snapshot keep
    # current_cell).  Non-agent (block/target) cells are left intact.
    try:
        ag = snapshot.get("agent")
        if isinstance(ag, dict):
            ag.pop("current_cell", None)
        for e in (snapshot.get("entities") or []):
            if isinstance(e, dict) and e.get("current_role") == "agent":
                e["current_cell"] = None
    except Exception:
        pass
    recent = _format_recent_outcomes(world, n_recent)
    # Stash available actions so the subgoal block can compute the
    # UNTRIED-actions set (component B exhaustion surface).
    try:
        world._available_actions = list(available_actions)
    except Exception:
        pass
    active_subgoals_block = _format_active_subgoals_block(world)

    # Cross-trial knowledge surfaces (slim variants — only the
    # high-confidence promoted entries, no free-form sprawl).
    # The actor is expected to cite which prior lessons /
    # subroutines it LEANED ON when relevant.
    try:
        from per_game_lessons import format_lessons_surface_slim  # noqa: E402
        prior_lessons_block = format_lessons_surface_slim(world)
    except Exception:
        prior_lessons_block = ""
    # Layer B — precondition-qualified mechanic rules (contrastive
    # refinement).  These rules state the CONDITION under which an
    # effect appears, learned by diffing supporting vs contradicting
    # observations; they survive even when the unconditional rule was
    # decayed to credence 0.  Prepended so the actor sees them
    # adjacent to the unconditional lessons surface.
    try:
        from mechanic_miner import (                              # noqa: E402
            format_precondition_qualified_rules,
        )
        b_surface = format_precondition_qualified_rules(world)
        if b_surface:
            prior_lessons_block = b_surface + "\n\n" + prior_lessons_block
    except Exception:
        pass
    # Cross-GAME knowledge: typed game profiles + variation matching.
    # Prepended to the lessons block so the actor sees "have I seen a
    # game like this?" before the within-game lessons.  See
    # docs/SPEC_cross_game_knowledge_store.md.
    try:
        from cross_game_knowledge import load_and_surface  # noqa: E402
        _xg = load_and_surface(
            world, getattr(world, "_available_actions", None),
        )
        if _xg:
            prior_lessons_block = _xg + prior_lessons_block
    except Exception:
        pass
    # MECHANIC-STABILITY PRIOR (prepended LAST so it leads the lessons
    # area).  Carries forward confirmed action->effect rules from
    # earlier levels of THIS game as STRONG, assume-stable beliefs, marks
    # which are UNVERIFIED on the current level, and lists the cheap
    # confirm probes to run before a plan depends on them.  Within-game
    # mechanics almost always persist; a plan assuming otherwise is
    # lower-priority until verified.  This is the guard that the sk48
    # lc=4 probe lacked — it committed 8x ACTION4 on an unverified
    # carried belief.  See mechanic_stability.py +
    # docs/ARCHITECTURE_game_priors.md § Mechanic-stability prior.
    try:
        from mechanic_stability import (                          # noqa: E402
            compute_stability_claims, verification_plan,
            format_stability_surface,
        )
        from per_game_lessons import mechanic_rules_for_stability  # noqa: E402
        _rules = mechanic_rules_for_stability(world.game_id)
        if _rules:
            # Effective current level = score (levels completed so far):
            # this driver launches at --level 0 and leaves world.level
            # pinned there, advancing world.score on each solve. The
            # per-level verification ledger is keyed on score (see
            # _register_solution_on_solve, record_level_verification), so
            # the surface MUST use the same key or 'verified here' never
            # matches the recorded level.
            _cur_level = (world.score if getattr(world, "score", None)
                          is not None else world.level)
            _claims = compute_stability_claims(
                same_game_rules=_rules,
                current_level=_cur_level,
            )
            _reversible = set(
                (getattr(world, "inverse_actions", None) or {}).keys())
            _vplan = verification_plan(
                _claims,
                priority_actions=[mech_action],
                reversible_actions=_reversible,
            )
            _stab = format_stability_surface(_claims, _vplan)
            if _stab:
                prior_lessons_block = _stab + "\n\n" + prior_lessons_block
    except Exception:
        pass
    # RELEVANT OPERATORS — retrieve by GENERALIZED FUNCTION, keyed on the
    # current goal's desired effect, replacing the situation-blind
    # rank_lessons[:8] for operators. This is the fix for the recurring
    # "the operator exists in the KB but never reaches the actor" failure
    # (the decouple rule, cred 0.92, was crowded out by bare action->effect
    # tuples). The query is the active subgoal's desired effect (what must
    # become true), so e.g. a subgoal "release a blue from the agent"
    # retrieves the decouple operator by function regardless of phrasing.
    # Prepended LAST so operators lead the lessons area. Embedding-backed
    # (sentence-transformers) with lexical fallback; precondition is
    # surfaced for the actor to gate. See docs/SPEC_operator_retrieval.md.
    # Current structural signature, used to GATE cross-game recall so a public
    # game's operators / subroutines never leak into an unseen game (see
    # cross_game_knowledge.admits_xgame).
    _cur_sig = None
    try:
        from cross_game_knowledge import compute_signature as _csig  # noqa: E402
        _cur_sig = _csig(world, getattr(world, "_available_actions", None)
                         or getattr(world, "available_actions", None))
    except Exception:
        _cur_sig = None
    try:
        from operator_kb import (                                  # noqa: E402
            load_all_operators as _op_load, retrieve_operators as _op_retrieve,
            format_retrieval_surface as _op_surface,
        )
        # Build the query from the active subgoal's desired effect; fall
        # back to game_purpose. Generalized, role-level text — no instances.
        _q = ""
        try:
            _act = [s for s in (getattr(world, "active_subgoals", None) or [])
                    if getattr(s, "status", "") == "active"]
            if _act:
                _sg = _act[0]
                # Query on the DESIRED EFFECT alone (expected_outcome), not
                # concatenated name+problem prose — clean functional effect
                # text retrieves the right operator sharply (0.81 vs 0.29
                # when noised with prose). The decomposition directive asks
                # the VLM to author expected_outcome as a clean, generalized,
                # instance-free effect for exactly this reason.
                _q = (getattr(_sg, "expected_outcome", "") or "").strip()
                if not _q:
                    _q = (getattr(_sg, "name", "") or "").replace("_", " ")
        except Exception:
            pass
        if not _q:
            _q = getattr(world, "game_purpose_guess", "") or ""
        if _q:
            _ops = _op_load()
            _scored = _op_retrieve(
                _ops, _q,
                scope={"game_id": getattr(world, "game_id", None),
                       "signature": _cur_sig}, k=4)
            _op_block = _op_surface(_scored, _q,
                                    game_id=getattr(world, "game_id", None))
            if _op_block:
                prior_lessons_block = _op_block + "\n\n" + prior_lessons_block
    except Exception:
        pass
    try:
        from subroutine_kb import format_subroutine_surface_slim  # noqa: E402
        # Pass this turn's Layer-A relations so the surface is relevance-
        # boosted: a procedure whose precondition-relations match the
        # current situation surfaces even at low credence (the reader half
        # of knowledge transfer).
        _deltas = getattr(world, "deltas_observed", None) or []
        _cur_rels = []
        if _deltas:
            _last = _deltas[-1]
            _cur_rels = (getattr(_last, "relations", None)
                         or (_last.get("relations") if isinstance(_last, dict)
                             else None) or [])
        stored_subroutines_block = format_subroutine_surface_slim(
            current_relations=_cur_rels,
            game_id=getattr(world, "game_id", None),
            current_sig=_cur_sig)
    except Exception:
        stored_subroutines_block = ""

    ground_truth_block = _format_ground_truth(world)

    # Crystallization reader-side: the live goal gap.  When a checkable
    # win relation has been crystallized, the substrate computes the
    # current-vs-target diff and surfaces it so the actor closes the gap
    # rather than re-deriving the objective.  Empty until a relation
    # exists.  See SPEC_goal_grounding_and_state_diff.md § Substrate-
    # computed goal gap.
    try:
        from knowledge_crystallization import format_goal_gap  # noqa: E402
        _gap = format_goal_gap(world)
        if _gap:
            ground_truth_block = _gap + "\n\n" + ground_truth_block
    except Exception:
        pass

    # Backward-reasoning protocol — fires on a MULTI-STEP goal gap (the
    # load-bearing trigger, SPEC_vlm_backward_reasoning.md §3) or a stall,
    # and leads with the substrate-computed NEXT required target so the
    # actor reasons backward from it instead of acting greedily or out of
    # order.  Only appended when active (off otherwise, to avoid clutter).
    try:
        _nps, _ras = _stuck_streaks(world, n_recent)
        _proto = _format_backward_reasoning_protocol(world, _nps, _ras)
        if _proto and "protocol off" not in _proto:
            ground_truth_block = ground_truth_block + "\n\n" + _proto
    except Exception:
        pass

    # Reflection moves — game-agnostic meta-cognitive reframes (separate
    # means from ends, check constraint scope, act through an
    # intermediary, re-perceive, not-found-vs-impossible).  Surfaced as a
    # one-line pointer normally; the FULL library appears (with the
    # blocked goal named) only when the substrate's dumb stuck-counter
    # trips.  The substrate supplies the WHEN (a no-progress streak) and
    # the stored TEXT; the VLM does the reasoning of applying a reframe.
    # See reflection_moves.py + memory/feedback_persist_insights_to_cos_prompt.
    try:
        from reflection_moves import format_reflection_moves  # noqa: E402
        _nps2, _ = _stuck_streaks(world, n_recent)
        _active = _nps2 >= 3
        _blocked = ""
        if _active:
            # Name a blocked goal if one is surfaced; else the mech goal.
            for sg in (getattr(world, "active_subgoals", None) or []):
                st = getattr(sg, "status", None) or (
                    sg.get("status") if isinstance(sg, dict) else None)
                if st == "blocked":
                    _blocked = (getattr(sg, "name", None)
                                or (sg.get("name") if isinstance(sg, dict)
                                    else "") or "")
                    break
            if not _blocked:
                _blocked = mech_rationale[:80]
        _refl = format_reflection_moves(_active, blocked_goal=_blocked)
        if _refl:
            ground_truth_block = ground_truth_block + "\n\n" + _refl
    except Exception:
        pass

    # PLAN GATE — preventive half of the plan-consistency check.  Always
    # on (not gated on stuck): before committing a plan, the actor must
    # verify it does not repeat a REFUTED approach or contradict a
    # CONFIRMED invariant (both already surfaced in the prior-knowledge
    # block).  This is the single most common planning failure — reverting
    # to first-principles and re-entering a cleared dead-end.  The
    # substrate supplies the directive + the counts; the VLM does the
    # check.  The detective half (lexical flag of the written plan) runs
    # in the driver after the reply.  See plan_consistency.py +
    # memory/feedback_plan_must_consult_established_insights.
    try:
        from plan_consistency import format_plan_gate          # noqa: E402
        from per_game_lessons import load_for_game             # noqa: E402
        _lessons = load_for_game(world.game_id)
        _refuted = [l for l in _lessons if l.kind == "refuted"]
        _invariants = [l for l in _lessons
                       if l.kind == "win_condition" and l.promoted]
        _gate = format_plan_gate(_refuted, _invariants)
        if _gate:
            ground_truth_block = ground_truth_block + "\n\n" + _gate
    except Exception:
        pass
    # POSITIVE half of the gate — surface CONFIRMED strategies (use-this),
    # so the actor applies the confirmed operator for a sub-goal instead of
    # reverting to the naive default (the recurring "forgetfulness about
    # confirmed strategies", e.g. push-instead-of-body-sweep). Only
    # REPLAY-CONFIRMED operators are enforced; until the replay-verification
    # gate (operator_verification + per-level ledger) populates them, this
    # surface is empty.  See plan_consistency.py (positive half) +
    # memory/feedback_plan_must_consult_established_insights.
    try:
        from plan_consistency import (                          # noqa: E402
            format_confirmed_strategy_surface,
        )
        _confirmed = list(getattr(world, "_confirmed_strategies", None) or [])
        _conf_surface = format_confirmed_strategy_surface(_confirmed)
        if _conf_surface:
            ground_truth_block = ground_truth_block + "\n\n" + _conf_surface
    except Exception:
        pass

    # STRUCTURAL + MEDIATION surfaces — built by the PRODUCER from the live
    # world each turn. build_structural_context reads world.entities (dict of
    # EntityRecord, tick bboxes), converts to cell-unit entity-dicts, derives
    # the impassable-obstacle list (entities with an impassable role), and
    # estimates the agent's arm state. It also stashes the results on the
    # world (_structural_obstacles / _rigid_body_kinematics / _arm_state_
    # estimate / _last_entities) so the driver's post-reply projector
    # (simulate_plan_against_structure) can re-use them without recomputing.
    #
    # This closes the turn-164 gap: previously the consumers read world
    # attributes that nothing populated (and passed the entities DICT to a
    # detector expecting a LIST), so both the STRUCTURAL INVARIANTS and
    # MEDIATION CANDIDATES surfaces were dormant. Now the producer feeds them.
    # Game-agnostic: obstacle roles + agent name are constants/params in
    # plan_consistency, not literals here. See plan_consistency.py (PRODUCER)
    # + memory/feedback_ensure_cos_self_sufficiency.
    try:
        from plan_consistency import (                          # noqa: E402
            build_structural_context, format_structural_invariants,
            detect_mediation_candidates, format_mediation_candidates,
            classify_attachment, format_attachment_surface,
            format_mediation_precondition, format_decomposition_directive,
            AgentKinematics,
        )
        # Default kinematics prior: rail-mounted extendable rod (sk48-style).
        # Magnitudes are unit approximations; the structural gate uses them
        # for DIRECTION (does the body cross an obstacle), and the mediation
        # detector is pure geometry (kinematics-independent), so this is a
        # safe default. Other agent topologies pass their own via the world.
        _kin = getattr(world, "_rigid_body_kinematics", None) or AgentKinematics(
            state_action_deltas={
                "head_row": {"ACTION1": -1, "ACTION2": +1},
                "tip_col":  {"ACTION4": +1, "ACTION3": -1},
            },
            body_constraint_text=("rigid horizontal segment, body row == "
                                  "head row at every step (magnitudes "
                                  "approximate; direction is what the gate "
                                  "checks)"),
        )
        _ctx = build_structural_context(world, default_kinematics=_kin)
        # Stash for the driver's post-reply projector + next-turn reuse.
        try:
            world._structural_obstacles = _ctx["obstacles"]
            world._rigid_body_kinematics = _kin
            world._arm_state_estimate = _ctx["arm_state"]
            world._last_entities = _ctx["entities"]
        except Exception:
            pass
        # Surface -3 — PROACTIVE KB RECALL (top-k ranked to THIS situation).
        # The fix for re-deriving what the KB already holds: every turn, retrieve
        # the most relevant lessons/operators for the current goals+relations and
        # surface them FIRST. Ranked by relevance (not just credence) so the
        # situation-specific procedure floats up even in a large KB; a no-progress
        # 'stuck' signal forces recall of stored solutions. See SPEC_kb_recall.
        try:
            sh = getattr(world, "_score_history", None)
            if sh is None:
                sh = world._score_history = []
            sh.append(int(getattr(world, "score", 0) or 0))
            if len(sh) > 30:
                del sh[:-30]
        except Exception:
            pass
        try:
            from kb_recall import format_recall_surface          # noqa: E402
            _rk = format_recall_surface(world, k=6)
            if _rk:
                ground_truth_block = _rk + "\n\n" + ground_truth_block
        except Exception:
            pass
        # Surface -2 — LAST-ACTION NO-OP VERDICT + mined REFUTED APPROACHES.
        # The substrate's own verdict that the previous action changed NOTHING,
        # plus relational approaches it has established as ineffective. This is
        # the recall+verify gate made un-ignorable: no phantom-effect claims,
        # no re-attempting a known no-op (the lc-4 raise-against-the-wall loop).
        try:
            from playback_mining import (                          # noqa: E402
                format_last_action_noop_surface, mine_refuted_approaches)
            _noop = format_last_action_noop_surface(world)
            if _noop:
                ground_truth_block = _noop + "\n\n" + ground_truth_block
            _ref = mine_refuted_approaches(world)
            if _ref:
                _rl = ["REFUTED APPROACHES (mined from your own no-ops — do "
                       "NOT retry these; pick a different mechanism):"]
                for _r in _ref[:5]:
                    _rl.append(f"  - {_r['description']}")
                ground_truth_block = ground_truth_block + "\n\n" + "\n".join(_rl)
        except Exception:
            pass
        # Surface -1 — ORDERED OBJECTIVE (compiled into the Goal Forest +
        # ENFORCED). When the recognized objective is an ordered sequence
        # (an `ordered_completion` win-relation), it is compiled into a
        # depends_on chain of subgoals; this surfaces the live cursor +
        # NEXT-REQUIRED step + the out-of-order gate, so the order is enforced
        # generically (no game/HUD specifics). See SPEC_ordered_objective_goals.
        try:
            from ordered_objective_goals import (                 # noqa: E402
                compile_ordered_objective, format_ordered_objective_surface)
            compile_ordered_objective(world)
            _oo = format_ordered_objective_surface(world)
            if _oo:
                ground_truth_block = ground_truth_block + "\n\n" + _oo
        except Exception:
            pass
        # Surface 0 — PER-INSTANCE SCENE FACTS (substrate-measured). The VLM
        # is unreliable at exact positions/counts/alignment by eye; these are
        # deterministic per-instance facts (block_red#1 at rows..cols..). READ
        # THESE; do not eyeball the frame. See SPEC_perception_contract.md.
        try:
            from instance_perception import (                     # noqa: E402
                format_instance_factsheet, instance_attachment)
            _tr = getattr(world, "_instance_tracker", None)
            if _tr is not None:
                # rod bbox for per-instance impaled/free comes from the KINEMATIC
                # arm_state (head_row/head_col/tip_col), NOT a color-segmented
                # arm_body (the grey rod is not color-separable from the blocks
                # it threads). Best-effort: may be one action stale at the top of
                # the turn; the block positions/counts are the reliable core.
                _rodbb = None
                _as = getattr(world, "_arm_state_estimate", None)
                if _as:
                    try:
                        from plan_consistency import (              # noqa: E402
                            _default_extendable_rod_body_region)
                        _rodbb = _default_extendable_rod_body_region(_as)
                    except Exception:
                        _rodbb = None
                _att = instance_attachment(_tr, _rodbb) if _rodbb else None
                _fs = format_instance_factsheet(_tr, _att)
                if _fs:
                    ground_truth_block = (
                        ground_truth_block + "\n\n" + _fs
                        + "\n  (To check anything else precisely — same-or-"
                          "different, changed, aligned, impaled, lane order — "
                          "ask the substrate; never decide from the picture.)")
                # VLM-TAUGHT recognizer verdicts (substrate-applied, no call).
                # If you can recognize something reliably but it's costly to do
                # every turn, TEACH it once (label instances in your reply's
                # teach_recognizer field) and the substrate runs it from then on.
                try:
                    from taught_recognizers import format_recognizer_surface
                    for _rn, (_rec, _res) in (
                            getattr(world, "_recognizer_results", {}) or {}).items():
                        _rs = format_recognizer_surface(_rec, _res)
                        if _rs:
                            ground_truth_block = ground_truth_block + "\n  " + _rs
                except Exception:
                    pass
        except Exception:
            pass
        # Surface 1 — STRUCTURAL INVARIANTS (obstacles + kinematics prior).
        if _ctx["obstacles"] or _kin:
            _struct_surface = format_structural_invariants(
                _ctx["obstacles"], _kin)
            if _struct_surface:
                ground_truth_block = (
                    ground_truth_block + "\n\n" + _struct_surface)
        # Surface 2 — BLOCK ATTACHMENT + MEDIATION PRECONDITION + CANDIDATES.
        # Attachment (behavioral, from co_displacement) decides which blocks
        # are FREE vs carried, so the mediation detector never proposes an
        # impaled block as a lift intermediary, and the actor gets an
        # explicit precondition verdict (MET / NOT MET + why) instead of
        # guessing from pixels. This is the fix for the rod-rendering-gap
        # trap. See plan_consistency.py (ATTACHMENT CLASSIFIER).
        if _ctx["entities"]:
            _attach = classify_attachment(world)
            _attach_surface = format_attachment_surface(_attach)
            if _attach_surface:
                ground_truth_block = (
                    ground_truth_block + "\n\n" + _attach_surface)
            _candidates = detect_mediation_candidates(
                _ctx["entities"], obstacles=_ctx["obstacles"],
                attachment=_attach)
            _precond = format_mediation_precondition(
                world, _candidates, _attach)
            if _precond:
                ground_truth_block = (
                    ground_truth_block + "\n\n" + _precond)
            # AUTONOMOUS GOAL-CHAIN GENERATION: when the precondition is
            # NOT met, direct the actor to GROW the goal forest (commit the
            # missing enabling state as a depends_on subgoal, recursively)
            # rather than act greedily. This is what lets COS generate the
            # backward chain itself instead of needing it hand-authored —
            # the substrate supplies the grounded trigger + facts, the VLM
            # supplies the means-ends decomposition. See plan_consistency.py
            # (format_decomposition_directive) + SPEC_vlm_backward_reasoning.
            _pursued = ""
            try:
                _ag = [s for s in (getattr(world, "active_subgoals", None) or [])
                       if getattr(s, "status", "") == "active"]
                if _ag:
                    _pursued = getattr(_ag[0], "name", "") or ""
            except Exception:
                pass
            _decomp = format_decomposition_directive(
                world, _candidates, _attach, pursued_goal=_pursued)
            if _decomp:
                ground_truth_block = (
                    ground_truth_block + "\n\n" + _decomp)
            _med_surface = format_mediation_candidates(_candidates)
            if _med_surface:
                ground_truth_block = (
                    ground_truth_block + "\n\n" + _med_surface)
            # Surface 3 — MULTI-GOAL PLAN (side-effect-aware). Wires the
            # engine's conjunctive planner (cognitive_os.plan_search) into the
            # loop for the ordered-collection game class: ONE sequence that
            # satisfies the ordered-collection goals together — a single
            # staging maneuver can satisfy several collects' reachability
            # preconditions at once (cheaper than one goal at a time), and the
            # clean-carrier + HUD-order preconditions are honored automatically
            # (clear-leg inserted when dirty; order discovered, not hardcoded).
            # Surfaces only when the required collection order is known (set on
            # world._required_collection_order by the win-condition / HUD
            # reading); reuses the attachment classification for carrier state.
            try:
                from ordered_collection_planner import (   # noqa: E402
                    build_collection_plan_context,
                    plan_ordered_collection, format_collection_plan,
                )
                _coll = build_collection_plan_context(world, attachment=_attach)
                if _coll:
                    _stack = plan_ordered_collection(
                        generated_at=int(getattr(world, "turn", 0)), **_coll)
                    _plan_surface = format_collection_plan(_stack)
                    if _plan_surface:
                        ground_truth_block = (
                            ground_truth_block + "\n\n" + _plan_surface)
            except Exception:
                pass
    except Exception:
        pass

    # Playback mining — ACTIVATED WHEN NEEDED (stuck / multi-step gap / a
    # recent progress collapse), NOT every turn.  Surfaces what the actor has
    # already achieved THIS level, the genuinely-unsolved sub-goal, and the
    # action-class that has been UN-DOING progress here (blame-assigned from
    # the recorded playback).  Game-agnostic: roles + the committed win
    # relation only.  See SPEC_cumulative_learning_loop.md § Playback mining.
    try:
        from playback_mining import (        # noqa: E402
            format_mining_report, _mining_trigger,
        )
        if _mining_trigger(world, n_recent):
            _mine = format_mining_report(world)
            if _mine:
                ground_truth_block = ground_truth_block + "\n\n" + _mine
    except Exception:
        pass

    # Repeat-until executor feedback — how the LAST committed repeat maneuver
    # ended (condition met / stalled / drop-guard auto-undo / cap), so the
    # actor adapts its next step instead of re-issuing a maneuver that failed.
    if repeat_feedback:
        ground_truth_block = (
            f"  LAST MANEUVER RESULT (repeat-until executor): {repeat_feedback}\n\n"
            + ground_truth_block)

    text = STRATEGY_PROMPT.format(
        game_id=world.game_id, level=world.level, turn=turn,
        world_snapshot=json.dumps(snapshot, indent=2),
        ground_truth_block=ground_truth_block,
        game_type=(world.game_type_guess
                    or "(unset — author it this turn)"),
        game_purpose=(world.game_purpose_guess
                       or "(unset — author it this turn)"),
        mech_action=mech_action,
        mech_plan_kind=mech_plan_kind,
        mech_goal_id=str(mech_goal_id),
        mech_rationale=mech_rationale,
        available_actions=", ".join(available_actions),
        n_recent=n_recent,
        recent_outcomes=recent,
        prior_lessons_block=prior_lessons_block,
        stored_subroutines_block=stored_subroutines_block,
        active_subgoals_block=active_subgoals_block,
        reply_name=reply_path.name,
    )
    prompt_path.write_text(text, encoding="utf-8")
    return prompt_path, reply_path


def _format_ground_truth(world: WorldKnowledge) -> str:
    """Render the previous turn's PERCEPTION-RECORDED facts so the
    acting VLM cannot claim an observable event the world did not
    record.  This is the substrate-side rationale-vs-perception guard:
    it surfaces the last delta's visual_events + summary verbatim.

    Game-agnostic: reads only `deltas_observed` (entity-named events
    sourced from perception), never any game-specific vocabulary.
    """
    deltas = getattr(world, "deltas_observed", None) or []
    if not deltas:
        return "  (no prior action yet — first turn)\n"
    dl = deltas[-1]
    # DeltaRecord may be a dataclass or a dict depending on call site.
    def _get(o, k, default=None):
        if isinstance(o, dict):
            return o.get(k, default)
        return getattr(o, k, default)
    to_turn = _get(dl, "to_turn", "?")
    summary = (_get(dl, "summary", "") or "").strip()
    ves = _get(dl, "visual_events", None) or []
    lines = [f"  turn {to_turn}: {summary or '(no summary)'}"]
    if ves:
        ev_strs = []
        for v in ves:
            ent = _get(v, "entity", "?")
            direction = _get(v, "direction", "changed")
            ev_strs.append(f"{ent}={direction}")
        lines.append("  visual_events recorded: " + ", ".join(ev_strs))
    else:
        lines.append("  visual_events recorded: NONE "
                      "(no HUD/entity pixel change was observed)")
    # Eager predict-then-falsify: echo last turn's prediction beside
    # what actually happened so the actor must reconcile them.
    pred = (getattr(world, "_last_prediction", None) or "").strip()
    if pred:
        lines.append(f"  YOUR PREDICTION last turn was: {pred}")
        # DEBUG-DON'T-GUESS: if the prediction's outcome polarity CONTRADICTS the
        # measured result (predicted an effect, measured nothing -- or vice
        # versa), that surprise means an ASSUMPTION is wrong.  Surface the
        # decompose-and-verify protocol so the actor DEBUGS (inspect the raw
        # animation / pixels, isolate the false piece) instead of re-guessing a
        # new story from the summary.  Falls back to the plain falsify nudge.
        _surprise = None
        try:
            import debugging_discipline as _dbg            # noqa: E402
            _surprise = _dbg.detect_surprise(
                pred, summary=summary, visual_events=ves,
                agent_moved=_get(dl, "agent_moved", None),
                score_increased=_get(dl, "score_increased", None),
                entities_changed=_get(dl, "entities_changed", None))
            if _surprise is not None:
                lines.append("  " + _dbg.protocol_block(_surprise).replace("\n", "\n  "))
        except Exception:
            _surprise = None
        if _surprise is None:
            lines.append("  -> Compare it to what is recorded above.  If it "
                          "did NOT hold, do NOT just re-guess: DECOMPOSE the "
                          "failed step into its assumptions and VERIFY each "
                          "against the RAW animation/pixels before re-ranking.")
    # Layer A — relational kinematics block.  Game-agnostic temporal
    # visual relations between entities computed by the substrate from
    # bbox histories (co_displacement / motion_blocked / penetration /
    # support_relation / motion_arrested_at).  These are FACTS, not
    # interpretations — the actor reads them as the verbs physical
    # and abstract reasoning operate on.  See
    # SPEC_visual_reasoning_substrate.md.
    rels = _get(dl, "relations", None) or []
    if rels:
        try:
            from relational_kinematics import (                   # noqa: E402
                RelationRecord, format_relations_block,
            )
            rel_records = [RelationRecord(**r) for r in rels]
            lines.append("")
            lines.append(
                "  SPATIAL RELATIONS (substrate-computed, game-agnostic "
                "geometry — FACTS, and your PRIMARY basis for spatial "
                "reasoning). Raw pixel/tick coordinates are intentionally "
                "withheld; reason from these relations, not from arithmetic. "
                "`ordered_along` tells you the layout order and what an "
                "approach from one side reaches FIRST (and thus what "
                "occludes what); `clearance` gives room-to-move in CELLS; "
                "`same_row`/`same_col` give alignment; co_displacement / "
                "penetration / motion_blocked describe what just moved:"
            )
            lines.append(format_relations_block(rel_records))
        except Exception:
            pass
    return "\n".join(lines) + "\n"


def _format_active_subgoals_block(world: WorldKnowledge) -> str:
    """Render an ACTIVE SUBGOALS section for the strategy prompt
    iff there are open subgoals.  Otherwise return empty string —
    no section, no nudge to commit, no prompt bloat.

    Each open subgoal is annotated as [ACTIONABLE] or
    [BLOCKED by id1, id2] based on the bridge's is_actionable
    check (cognitive_os Goal Forest depends_on evaluation).
    """
    from active_subgoals import active_subgoals  # noqa: E402

    opens = active_subgoals(world)
    if not opens:
        return ""

    try:
        from subgoal_forest_bridge import actionable_status  # noqa: E402
    except Exception:
        actionable_status = None  # type: ignore[assignment]

    avail = list(getattr(world, "_available_actions", []) or [])
    lines = ["", "ACTIVE SUBGOALS (your prior-turn commitments). "
              "[UNMET] = acceptance test has NOT fired; you may not "
              "drop it until it fires, you mark it blocked with an "
              "impediment, or the substrate invalidates it. "
              "[BLOCKED by X] = a depends_on dependency is still open:"]
    for sg in opens:
        if actionable_status is not None:
            ok, unmet = actionable_status(world, sg.subgoal_id)
        else:
            ok, unmet = True, []
        unmet_tag = "[UNMET]" if (sg.acceptance_check or "").strip() else ""
        dep_tag = ("[ACTIONABLE]" if ok
                   else f"[BLOCKED by {', '.join(unmet)}]")
        age = world.turn - sg.created_at_turn
        og = f" or_group={sg.or_group}" if sg.or_group else ""
        lines.append(
            f"  {unmet_tag}{dep_tag} id={sg.subgoal_id} "
            f"name={sg.name!r} status={sg.status} age={age}t{og}"
        )
        if sg.expected_outcome:
            short = sg.expected_outcome.strip().splitlines()[0][:120]
            lines.append(f"      expected: {short}")
        if (sg.acceptance_check or "").strip():
            lines.append(f"      acceptance_check: {sg.acceptance_check}")
        # Component B: tried vs untried approaches for this subgoal.
        tried = sg.approaches_tried or []
        if tried:
            tried_acts = sorted({e[0] for e in tried if e})
            lines.append(
                f"      tried approaches: {tried_acts}"
            )
            if avail:
                untried = [a for a in avail if a not in tried_acts]
                if untried:
                    lines.append(
                        f"      UNTRIED actions for this subgoal: "
                        f"{untried}"
                    )
    return "\n".join(lines) + "\n"


def _stuck_streaks(world: WorldKnowledge, n: int) -> tuple[int, int]:
    """Compute (no_progress_streak, repeated_action_streak) over the
    last ``n`` deltas.  Mirrors the logic in _format_recent_outcomes
    so the protocol trigger matches what the recent-outcomes block
    shows the actor."""
    if not world.deltas_observed:
        return (0, 0)
    recent = world.deltas_observed[-n:]
    no_progress_streak = 0
    for d in reversed(recent):
        no_progress = (
            not d.agent_moved
            and not d.entities_appeared
            and not d.entities_disappeared
            and not d.entities_changed
        )
        if no_progress:
            no_progress_streak += 1
        else:
            break
    repeated_action_streak = 0
    if recent:
        last_action = recent[-1].action
        for d in reversed(recent):
            if d.action == last_action:
                repeated_action_streak += 1
            else:
                break
    return (no_progress_streak, repeated_action_streak)


def _format_blocking_claims(world: WorldKnowledge) -> str:
    """Render promoted BlockingClaims that match the CURRENT
    state-class — i.e. actions known to be SILENT right now — and
    a short backward-chaining instruction telling the strategy
    layer to derive a removal subgoal.

    A blocking claim's removal subgoal lifts the state-class match
    by changing one of the listed features.  E.g. if a claim says
    "(entity_roles=agent=1,engaged=1,collectable=3) blocks ACTION4",
    then the removal subgoal is "change the entity-role multiset"
    — concretely, find a mechanic that disengages the engaged entity
    OR adds/removes another entity from the role inventory."""
    # Local import to avoid circular dependency at module load.
    from mechanic_miner import (   # noqa: E402
        promoted_blocking_claims, _state_class_fingerprint,
    )

    promoted = promoted_blocking_claims(world)
    if not promoted:
        return ("  (no promoted blocking claims for the current "
                "state — backward chaining not needed yet)")

    current = _state_class_fingerprint(world)
    # Filter to claims whose blocking_state matches the current
    # world state (all features in claim subset of current features).
    matching = [
        c for c in promoted
        if all(current.get(k) == v for k, v in c.blocking_state.items())
    ]

    if not matching:
        lines = ["  (promoted blocking claims exist but none match "
                  "the current state-class)"]
        for c in promoted[:5]:
            lines.append(f"  - dormant: {c.claim_id} "
                          f"(credence={c.credence:.2f})")
        return "\n".join(lines)

    lines = []
    lines.append(
        "  !! HARD CONSTRAINTS for the CURRENT state-class:"
    )
    for c in matching:
        feature_str = ", ".join(
            f"{k}={v}" for k, v in c.blocking_state.items()
        )
        lines.append(
            f"  - action {c.blocked_action!r} is SILENT when "
            f"({feature_str}).  credence={c.credence:.2f}, "
            f"observed {len(c.supporting_observations)} times."
        )
    lines.append(
        "\n  BACKWARD CHAINING — if your candidate action is listed "
        "above, do NOT just retry it.  Instead, recognise that the "
        "constraint must be LIFTED first.  Derive a REMOVAL SUBGOAL "
        "that changes one of the state-class features (e.g., "
        "disengage a carried entity, sweep blocks out of arm range, "
        "move agent to a different row band) BEFORE re-attempting "
        "the blocked action.  Then propose the action that makes "
        "progress on the REMOVAL SUBGOAL this turn."
    )
    return "\n".join(lines)


def _positionally_displaced_entities(
    world: WorldKnowledge, min_turns: int = 5,
    min_displacement_ticks: int = 6,
) -> list[dict]:
    """Entities whose current bbox is displaced from their FIRST
    observed bbox by more than ``min_displacement_ticks`` and have
    been at the displaced location for at least ``min_turns`` turns.

    A complementary durable-state signal to role assignments: when
    the perception layer can't or won't differentiate roles
    ('engaged' vs 'collectable'), a block that's moved far from its
    starting position and stayed there is still a candidate for
    backward-reasoning interrogation."""
    durable: list[dict] = []
    for r in world.entities.values():
        if r.last_seen_turn != world.turn:
            continue
        if r.current_bbox is None:
            continue
        # Need a starting bbox to compare to.  bbox_history is a list
        # of (turn, bbox).  The first entry is the starting bbox.
        history = getattr(r, "bbox_history", None) or []
        if not history:
            continue
        first_turn, first_bbox = history[0]
        # Manhattan displacement of centroid in ticks
        def _centroid(b):
            return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)
        f_r, f_c = _centroid(first_bbox)
        c_r, c_c = _centroid(r.current_bbox)
        dr, dc = abs(c_r - f_r), abs(c_c - f_c)
        displacement = dr + dc
        if displacement < min_displacement_ticks:
            continue
        # How long has the entity been ROUGHLY at its current
        # position?  Walk back and count turns where the centroid is
        # within min_displacement_ticks/2 of the current centroid.
        held_since = world.turn
        for (t, bb) in reversed(history):
            br, bc = _centroid(bb)
            if (abs(br - c_r) + abs(bc - c_c)
                    < min_displacement_ticks / 2.0):
                held_since = t
            else:
                break
        n_turns = world.turn - held_since + 1
        if n_turns < min_turns:
            continue
        durable.append({
            "entity": r.name,
            "from_bbox": list(first_bbox),
            "current_bbox": list(r.current_bbox),
            "displacement_ticks": int(displacement),
            "turns_at_current_pos": n_turns,
            "first_seen_turn": first_turn,
        })
    durable.sort(key=lambda d: -d["turns_at_current_pos"])
    return durable


def _durable_role_assignments(world: WorldKnowledge,
                                min_turns: int = 5) -> list[dict]:
    """Entities whose current_role has been held continuously for at
    least ``min_turns``.  Returns a list of small dicts with
    name, role, and how long the role has been held.

    Used by the BACKWARD-REASONING PROTOCOL: a durable non-default
    role (e.g. 'engaged' for 90 turns) is a candidate for the
    constraint to reason backward from."""
    DEFAULT_ROLES = {
        "scenery", "decoration", "hud", "background", "unknown",
        "collectable",  # the "starting" role of any goal entity
    }
    durable: list[dict] = []
    for r in world.entities.values():
        if r.last_seen_turn != world.turn:
            continue  # not currently present
        if not r.current_role:
            continue
        role_lower = r.current_role.lower()
        if role_lower in DEFAULT_ROLES:
            continue
        # How many turns has the entity held its CURRENT role?
        # role_history is a list of (turn, role, credence) tuples; walk
        # back until the role differs.
        history = getattr(r, "role_history", None) or []
        held_since_turn = world.turn
        for (t, role, _credence) in reversed(history):
            if role and role.lower() == role_lower:
                held_since_turn = t
            else:
                break
        n_turns = world.turn - held_since_turn + 1
        if n_turns < min_turns:
            continue
        durable.append({
            "entity": r.name,
            "role": r.current_role,
            "turns_held": n_turns,
            "since_turn": held_since_turn,
        })
    # Sort by turns_held descending — most durable first.
    durable.sort(key=lambda d: -d["turns_held"])
    return durable


def _turns_since_score_change(world: WorldKnowledge) -> Optional[int]:
    """How many turns since world.score last changed.  None if the
    score history is empty or always the same value (so we can't
    distinguish 'stuck' from 'just started')."""
    if world.score is None:
        return None
    # We don't have a per-turn score history record; approximate via
    # actions_taken length (each ActionRecord is one turn).  This
    # over-counts if score advanced very early and then plateaued —
    # acceptable for a stuck signal.
    return len(world.actions_taken)


def _current_goal_gap_detail(world: WorldKnowledge):
    """Return (win_relation, evaluation) for the highest-credence
    committed win relation, or (None, None) if none is crystallized yet.
    Used to drive the multi-step-gap trigger + goal-first directive."""
    try:
        from knowledge_crystallization import evaluate_win_relation  # noqa: E402
    except Exception:
        return None, None
    cands = [h for h in getattr(world, "win_condition_hypotheses", [])
             if getattr(h, "win_relation", None)]
    if not cands:
        return None, None
    cands.sort(key=lambda h: getattr(h, "credence", 0.0), reverse=True)
    rel = cands[0].win_relation
    try:
        return rel, evaluate_win_relation(world, rel)
    except Exception:
        return rel, None


def _format_backward_reasoning_protocol(world: WorldKnowledge,
                                          no_progress_streak: int,
                                          repeated_action_streak: int
                                          ) -> str:
    """Emit a structured BACKWARD-REASONING PROTOCOL block only when
    the system shows signs of being stuck (no-progress streak,
    repeated-action streak, or long plateau since last score change).

    The block lists durable role assignments — durable states are
    candidates for the constraint to reason backward FROM — and walks
    the strategy actor through a six-step chain:
      1. restate the goal
      2. identify durable state changes
      3. pick one as the constraint
      4. query mechanics that produce / undo it
      5. derive the removal subgoal
      6. choose the next action toward that subgoal
    """
    # Trigger condition — any stuck signal OR a MULTI-STEP goal gap.
    # The multi-step-gap trigger is the load-bearing one
    # (SPEC_vlm_backward_reasoning.md §3): a stall is detected only
    # AFTER several wasted turns, but a goal that is several steps away
    # should recruit backward planning BEFORE acting greedily.  Without
    # it, an actor can confidently commit to the wrong sub-goal on turn 1
    # (the sk48 lc=2 trial: skipped to the 3rd target because nothing
    # told it the 1st was still required).
    score_plateau = _turns_since_score_change(world)
    rel, gap_res = _current_goal_gap_detail(world)
    gap_det = (gap_res or {}).get("detail", {}) if gap_res else {}
    next_target = gap_det.get("next")
    remaining = gap_det.get("remaining") or []
    multi_step_gap = bool(
        gap_res and not gap_res.get("satisfied", False)
        and rel and rel.get("type") == "ordered_completion"
        and len(remaining) >= 2
    )
    hard_stuck = (
        no_progress_streak >= 3
        or repeated_action_streak >= 3
        or (score_plateau is not None and score_plateau >= 20)
    )
    if not (hard_stuck or multi_step_gap):
        return "  (system not in a stuck state — backward-reasoning protocol off)"

    durable_roles = _durable_role_assignments(world)
    displaced = _positionally_displaced_entities(world)

    lines: list[str] = []
    why = "STUCK INDICATORS" if hard_stuck else "MULTI-STEP GOAL GAP"
    lines.append(
        f"  !! {why} — backward-reasoning protocol ACTIVATED.  "
        "Before proposing an action, work through the chain below in "
        "your `rationale` field.  Be brief but explicit at each step."
    )
    # Goal-first directive: name the SUBSTRATE-COMPUTED next required
    # target and forbid skipping the order (the concrete anti-pattern
    # the sk48 trial exhibited — targeting a later item first).
    if next_target is not None:
        lines.append("")
        lines.append(
            f"  GOAL-FIRST (substrate-computed): the win completes its "
            f"members IN ORDER.  The NEXT REQUIRED target is "
            f"{next_target!r}; remaining (in order): {remaining}.  Do NOT "
            f"attempt a later target before {next_target!r} — out-of-order "
            f"attempts DO NOT TAKE and waste turns.  Ask first: is "
            f"completing {next_target!r} one action away?  If yes, do it.  "
            f"If not, reason BACKWARD: what condition enables completing "
            f"{next_target!r}, and what action achieves that condition?"
        )
    lines.append("")
    lines.append("  Durable state assignments worth interrogating:")
    if durable_roles:
        for d in durable_roles[:6]:
            lines.append(
                f"    - entity {d['entity']!r} has held role "
                f"{d['role']!r} for {d['turns_held']} turns "
                f"(since turn {d['since_turn']})."
            )
    else:
        lines.append(
            "    - (no durable non-default role assignments)"
        )
    lines.append("")
    lines.append(
        "  Positionally displaced entities (moved from starting "
        "position and held the new position):"
    )
    if displaced:
        for d in displaced[:6]:
            lines.append(
                f"    - entity {d['entity']!r} displaced "
                f"{d['displacement_ticks']} ticks from its starting "
                f"bbox {d['from_bbox']} to current {d['current_bbox']}; "
                f"held current position for "
                f"{d['turns_at_current_pos']} turns."
            )
    else:
        lines.append(
            "    - (no entities are durably displaced)"
        )
    lines.append("")
    lines.append("  PROTOCOL — six steps to derive a removal subgoal:")
    lines.append("")
    lines.append(
        "  1. RESTATE THE GOAL.  What outcome would advance the score, "
        "given game_purpose_guess?  Name the entity / state change "
        "that delivery / win requires."
    )
    lines.append(
        "  2. IDENTIFY THE DURABLE CHANGE TO QUESTION.  From the list "
        "above, pick ONE durable assignment that may be blocking "
        "further progress.  Name it explicitly."
    )
    lines.append(
        "  3. FRAME IT AS A CONSTRAINT.  'avoid having entity X in "
        "state Y' — write the sentence."
    )
    lines.append(
        "  4. QUERY MECHANICS.  In the mechanic_hypotheses list, find "
        "(a) which actions PRODUCE the durable state (so future calls "
        "of those actions are forbidden here); and (b) which actions "
        "can CHANGE the entity's position / role WITHOUT re-producing "
        "the durable state.  Cite hypothesis_ids."
    )
    lines.append(
        "  5. DERIVE THE REMOVAL SUBGOAL.  In plain language: "
        "'before resuming the goal, achieve <subgoal>.'  The subgoal "
        "is whatever state lifts the constraint from step 3."
    )
    lines.append(
        "  6. CHOOSE THE NEXT ACTION as the FIRST STEP toward the "
        "subgoal.  Don't re-propose actions you just identified as "
        "constraint-producing in step 4(a)."
    )
    lines.append("")
    lines.append(
        "  If the durable list is empty or the chain doesn't yield a "
        "fresh candidate, propose a probe in an UNTESTED state-class "
        "instead (consult BLOCKING CONSTRAINTS above for what's "
        "already known to be silent)."
    )
    return "\n".join(lines)


def _format_recent_outcomes(world: WorldKnowledge, n: int) -> str:
    if not world.deltas_observed:
        return "  (none yet; this is the first turn)"

    # Per-delta lines (chronological, oldest first)
    lines = []
    recent = world.deltas_observed[-n:]
    for d in recent:
        moved = "moved" if d.agent_moved else "stuck"
        chg = (f"+{len(d.entities_appeared)}/"
               f"-{len(d.entities_disappeared)}/"
               f"~{len(d.entities_changed)}")
        lines.append(
            f"  t{d.from_turn}->t{d.to_turn}: action={d.action!r} "
            f"inferred={d.inferred_action!r} {moved} "
            f"agent={d.agent_new_cell} entity_changes={chg}"
        )

    # Stuck-detection summary across the recent window.  A delta
    # counts as "no progress" when the agent did not move AND no
    # entities appeared / disappeared / changed.  Note that an
    # entity that's a CARRIED ITEM moving with the agent counts as
    # 'changed' — so a genuinely silent action shows entity_changes
    # = +0/-0/~0.
    no_progress_streak = 0
    for d in reversed(recent):
        no_progress = (
            not d.agent_moved
            and not d.entities_appeared
            and not d.entities_disappeared
            and not d.entities_changed
        )
        if no_progress:
            no_progress_streak += 1
        else:
            break

    # Repeated-action streak — last K actions all the same
    repeated_action_streak = 0
    if recent:
        last_action = recent[-1].action
        for d in reversed(recent):
            if d.action == last_action:
                repeated_action_streak += 1
            else:
                break

    summary_bits = []
    if no_progress_streak >= 3:
        summary_bits.append(
            f"!! STUCK: {no_progress_streak} consecutive turns with NO "
            f"progress (no agent move, no entity changes)."
        )
    if repeated_action_streak >= 3:
        summary_bits.append(
            f"!! REPETITION: action {last_action!r} repeated "
            f"{repeated_action_streak} times in a row."
        )
    if summary_bits:
        summary_bits.append(
            "STUCK RECOVERY: consider UNDOING recent forward actions "
            "(e.g., retract if you've been extending, move down if you've "
            "been moving up, release if you've been grabbing) BEFORE "
            "trying novel actions.  See STUCK DETECTION AND UNDO-FIRST "
            "RECOVERY in the system prompt."
        )

    if summary_bits:
        return "  " + "\n  ".join(summary_bits) + "\n\n" + "\n".join(lines)
    return "\n".join(lines)


def poll_strategy_reply(reply_path: Path,
                         timeout_s: int = 600,
                         poll_s: float = 2.0) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if reply_path.exists():
            body = reply_path.read_text(encoding="utf-8").strip()
            if body:
                return _consume(reply_path, body)
        time.sleep(poll_s)
    # Strategy VLM did not answer within the timeout — do NOT crash the run.
    # Return an empty reply; apply_strategy() treats it as no endorsement and
    # falls back to the mechanical/probe choice (strict-mode robustness).
    print(f"[strategy] WARNING: strategy reply timed out after {timeout_s}s at "
          f"{reply_path}; falling back to the mechanical choice (no halt).",
          flush=True)
    return {}


def _consume(reply_path: Path, body: str) -> dict:
    if body.startswith("```"):
        body = body.split("\n", 1)[1] if "\n" in body else body
        body = body.rsplit("```", 1)[0]
    parsed = None
    for suffix in ("", "}", '"}', '"}}'):
        try:
            parsed = json.loads(body + suffix)
            break
        except json.JSONDecodeError:
            parsed = None
            continue
    if not isinstance(parsed, dict):
        # STRICT-MODE ROBUSTNESS: a present-but-unparsable strategy reply must
        # NOT crash the run (the old code raise'd RuntimeError here).  Quarantine
        # the bad file and return {}; poll_strategy_reply / apply_strategy treat
        # {} as "no endorsement" and fall back to the mechanical/probe choice.
        print(f"[strategy] WARNING: unparsable/invalid strategy reply at "
              f"{reply_path}; falling back to the mechanical choice (no crash).",
              flush=True)
        try:
            shutil.move(str(reply_path),
                        str(reply_path.with_name("strategy_reply.unparsable.txt")))
        except Exception:
            pass
        return {}
    consumed = reply_path.with_name("strategy_reply.consumed.txt")
    shutil.move(str(reply_path), str(consumed))
    return parsed


# ---------------------------------------------------------------------------
# Reply validator (substrate-boundary check).  See
# memory/session_2026_05_30_drift_failure.md for the failure this catches:
# an actor that authors rationale + structured fields referencing entities
# perception never tracked, or asserts state changes not in the latest
# delta, can otherwise drift from ground truth for many turns and write
# fabricated lessons into the KB.
# ---------------------------------------------------------------------------


_SNAKE_NAME_RE = re.compile(r"\b([a-z][a-z0-9]*)_([a-z][a-z0-9_]*)\b")
_SCORE_CHANGE_RE = re.compile(
    r"\bscore\s+(?:advanced|increased|rose|went\s+up|incremented|"
    r"jumped|climbed)\b",
    re.IGNORECASE,
)
_WIN_STATE_RE = re.compile(
    r"\b(?:reached|advanced\s+to|moved\s+to|entered|transitioned\s+to)\s+"
    r"lc\s*=?\s*\d+|\bwin[-_\s]+state\s+(?:advanced|changed|transitioned)",
    re.IGNORECASE,
)


# --- discipline enforcement (convert load-bearing prose rules into
#     checkable post-conditions; reject + re-prompt on violation) ---
# An "impossibility" claim is the dangerous one: it ends exploration.  It
# is only allowed when backed by a relation AND after consulting any
# situationally-matching ([RELEVANT NOW]) stored procedure.
_IMPOSSIBLE_RE = re.compile(
    r"\b(impossible|unsolvable|un(?:reach|attain)able|dead[- ]?end|"
    r"no\s+way\s+to|cannot\s+(?:be\s+)?(?:reach|reached|pierce|pierced|"
    r"move[d]?|done|solved|completed)|give\s+up|abandon)",
    re.IGNORECASE,
)
# Layer-A relations (the only admissible evidence for an impossibility).
_RELATION_TOKEN_RE = re.compile(
    r"\b(clearance|motion_blocked|motion_arrested(?:_at)?|penetration|"
    r"support_relation|same_row|same_col|co_displacement|ordered_along|"
    r"adjacent|playfield_boundary|boundary|wall)\b",
    re.IGNORECASE,
)
# Raw-coordinate arithmetic — NOT admissible evidence (substrate withholds
# raw coords precisely to prevent this).
_COORD_ARITH_RE = re.compile(
    r"\b(?:row|col)\s*\d{1,3}\b|\bcol\d{1,3}\b|\b\d{1,3}\s*ticks?\b",
    re.IGNORECASE,
)


def _guess_str(value) -> str:
    if isinstance(value, dict):
        value = value.get("guess")
    return value if isinstance(value, str) else ""


def _delta_attr(delta, key, default=None):
    if delta is None:
        return default
    if isinstance(delta, dict):
        return delta.get(key, default)
    return getattr(delta, key, default)


def validate_reply(reply: dict,
                    world: WorldKnowledge) -> list[str]:
    """Scan a strategy reply against perception's tracked entity set and
    the latest delta.  Returns a list of violation strings (empty list
    means the reply passes).

    Catches the narrative-drift failure mode: an actor that hallucinates
    new entities (e.g. ``block_orange`` when only red/green/blue are
    tracked) or asserts progress (score advance, lc transition) that
    never occurred in the world state.

    Three classes of check (all game-agnostic):
      1. Structured ``references_entities`` lists in ``commit_subgoal`` /
         ``subgoal_status_update`` must only name tracked entities.
      2. Prose fields (rationale, prediction, game_purpose, game_type,
         forward_simulation) are scanned for snake_case tokens that
         share a prefix with current entities but are not themselves
         tracked (e.g. ``block_orange`` among ``block_red``,
         ``block_green``, ``block_blue``).
      3. The same prose fields are scanned for score-advance / win-state
         transition claims; flagged if the latest delta does not record
         the asserted change.
    """
    if not isinstance(reply, dict):
        return ["reply is not a dict — cannot validate"]

    violations: list[str] = []
    known: set[str] = set((world.entities or {}).keys())

    # 1. Structured entity references.
    for fld in ("commit_subgoal", "subgoal_status_update"):
        block = reply.get(fld)
        if not isinstance(block, dict):
            continue
        refs = block.get("references_entities") or []
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if isinstance(ref, str) and ref and ref not in known:
                violations.append(
                    f"{fld}.references_entities names '{ref}' but no "
                    f"such entity is tracked by perception. Known "
                    f"entities: {sorted(known) or '(none)'}"
                )

    # 2. Prose entity scanning.  Build a prefix set from current
    # entities ('block_red' -> 'block').  An unknown snake_case token
    # whose prefix matches a tracked-entity prefix is almost certainly
    # a hallucinated sibling.
    prefixes: set[str] = set()
    for name in known:
        if "_" in name:
            prefixes.add(name.split("_", 1)[0])

    prose_fields = {
        "rationale": _guess_str(reply.get("rationale")),
        "prediction": _guess_str(reply.get("prediction")),
        "game_purpose": _guess_str(reply.get("game_purpose")),
        "game_type": _guess_str(reply.get("game_type")),
        "forward_simulation": _guess_str(reply.get("forward_simulation")),
    }
    seen: set[tuple[str, str]] = set()
    for fname, text in prose_fields.items():
        if not text:
            continue
        for m in _SNAKE_NAME_RE.finditer(text):
            tok = m.group(0)
            prefix = m.group(1)
            if tok in known or prefix not in prefixes:
                continue
            key = (fname, tok)
            if key in seen:
                continue
            seen.add(key)
            siblings = sorted(
                n for n in known if n.startswith(prefix + "_")
            )
            violations.append(
                "[soft] "
                f"{fname} references '{tok}' (matches the '{prefix}_*' "
                f"naming pattern of tracked entities {siblings}) but no "
                f"such entity is tracked."
            )

    # 3. State-change assertions vs latest delta.
    deltas = getattr(world, "deltas_observed", None) or []
    last_delta = deltas[-1] if deltas else None
    for fname, text in prose_fields.items():
        if not text:
            continue
        if _SCORE_CHANGE_RE.search(text):
            si = _delta_attr(last_delta, "score_increased", None)
            if not si:
                violations.append(
                    f"{fname} asserts a score advance, but the latest "
                    f"delta has score_increased={si!r}. Substrate score "
                    f"= {getattr(world, 'score', None)!r}."
                )
        if _WIN_STATE_RE.search(text):
            ws = _delta_attr(last_delta, "win_state_changed", None)
            if not ws:
                violations.append(
                    f"{fname} asserts a win-state / level change, but "
                    f"the latest delta has win_state_changed={ws!r}. "
                    f"Substrate level = {getattr(world, 'level', None)!r}."
                )

    # 4. Discipline post-conditions (impossibility needs relation +
    #    KB consultation; relations-not-arithmetic).  Enforced via the
    #    same re-call loop as the entity/state checks.
    violations.extend(validate_disciplines(reply, world))

    return violations


def _relevant_subroutine_names(world: WorldKnowledge) -> list[str]:
    """Names of stored subroutines the substrate would tag [RELEVANT NOW]
    this turn (precondition-relations match the latest delta's relations).
    Best-effort; empty on any failure so validation never crashes."""
    try:
        import subroutine_kb as _S            # noqa: E402
        from knowledge_crystallization import relevance_to_situation, \
            _current_relation_kinds            # type: ignore  # noqa: E402
        deltas = getattr(world, "deltas_observed", None) or []
        rels = []
        if deltas:
            last = deltas[-1]
            rels = (getattr(last, "relations", None)
                    or (last.get("relations") if isinstance(last, dict) else None)
                    or [])
        # relevance_to_situation lives in subroutine_kb; tolerate either home
        try:
            rel_fn = _S.relevance_to_situation
            kinds_fn = _S._current_relation_kinds
        except AttributeError:
            rel_fn = relevance_to_situation
            kinds_fn = _current_relation_kinds
        kinds = kinds_fn(rels)
        cur_game = getattr(world, "game_id", None)
        out = []
        for s in _S.load():
            rel = rel_fn(s, kinds)
            same_game = getattr(s, "game_id", None) == cur_game
            # Game-aware TIGHT gate (prevents cross-game contamination): a
            # SAME-GAME constraint surfaces on a moderate situation match; a
            # constraint that originated in a DIFFERENT game must match the
            # situation STRONGLY before it surfaces, so a procedure never
            # bleeds into an unrelated game on loose relation-kind overlap.
            if rel >= (0.5 if same_game else 0.7):
                out.append(s.name)
        return out
    except Exception:
        return []


def validate_disciplines(reply: dict, world: WorldKnowledge) -> list[str]:
    """Enforce the load-bearing strategy-prompt disciplines as checkable
    post-conditions.  Returns violation strings (hard ones un-tagged,
    soft ones ``[soft]``-prefixed).  Three checks:

      1. consider-relevant-procedure (PROACTIVE, every turn): if a
         [RELEVANT NOW] stored procedure / registered constraint matches
         this turn's situation, the actor must TAKE IT INTO CONSIDERATION
         (name it + apply/adapt or say why it doesn't apply).  This is
         consideration, NOT enforcement — the action is never vetoed —
         and it is TIGHTLY, game-awarely gated so an unrelated game's
         constraints never trigger (no cross-game contamination).
      2. impossibility-needs-relation: an impossibility/blocked claim
         must cite a Layer-A relation as evidence.
      3. relations-not-arithmetic (soft): reason from relations, not raw
         coordinates.

    Together these turn 'reason from relations, consider the KB' from
    advisory prose into a checked boundary — the answer to 'how do I make
    an actor actually weigh its registered strategies'."""
    if not isinstance(reply, dict):
        return []
    parts = [_guess_str(reply.get("rationale")),
             _guess_str(reply.get("prediction"))]
    ssu = reply.get("subgoal_status_update")
    status_blocked = False
    if isinstance(ssu, dict):
        parts += [str(ssu.get("notes") or ""), str(ssu.get("impediment") or "")]
        status_blocked = ssu.get("status") in ("blocked", "abandoned")
    text = " ".join(p for p in parts if p)

    claims_impossible = bool(_IMPOSSIBLE_RE.search(text)) or status_blocked

    violations: list[str] = []

    # PROACTIVE consider-relevant-procedure (EVERY turn, not only when an
    # impossibility is claimed).  If a [RELEVANT NOW] stored procedure /
    # registered constraint matches THIS turn's situation and the reply
    # neither names nor reasons about it, the actor must TAKE IT INTO
    # CONSIDERATION.  This is CONSIDERATION, not enforcement: the chosen
    # action is never vetoed — the actor must only acknowledge the procedure
    # and either apply/adapt it or state the precondition it lacks.  The
    # match is TIGHTLY gated (game-aware relevance in
    # _relevant_subroutine_names) so an unrelated game's constraints never
    # trigger here — no cross-game contamination.
    rel_subs = _relevant_subroutine_names(world)
    if rel_subs:
        low = text.lower()
        considered = ("subroutine" in low or "technique" in low
                      or "relevant now" in low or "constraint" in low
                      or any(any(tok in low for tok in n.lower().split()
                                 if len(tok) > 4) for n in rel_subs))
        if not considered:
            violations.append(
                "DISCIPLINE(consider-relevant-procedure): a [RELEVANT NOW] "
                "stored procedure/constraint matches this turn's situation ("
                + "; ".join(rel_subs) + "). Take it into account before "
                "acting: name it and either apply/adapt it or state the "
                "specific precondition-relation it needs that you lack. "
                "(Consideration is required; your chosen action is NOT vetoed.)"
            )

    # Impossibility-specific discipline (stronger): an impossibility/blocked
    # claim additionally requires a Layer-A relation as evidence.
    if claims_impossible and not _RELATION_TOKEN_RE.search(text):
        violations.append(
            "DISCIPLINE(impossibility-needs-relation): you declared the "
            "situation impossible/blocked/stuck but cited NO Layer-A "
            "relation as evidence. Cite a relation (clearance / "
            "motion_blocked / motion_arrested_at / boundary) that supports "
            "it. Coordinate arithmetic is NOT evidence; if no relation "
            "contradicts the position, it is REACHABLE and the maneuver "
            "just hasn't been authored yet."
        )

    if _COORD_ARITH_RE.search(text):
        violations.append(
            "[soft] DISCIPLINE(relations-not-arithmetic): your reasoning "
            "uses raw coordinates/tick arithmetic; reason from the SPATIAL "
            "RELATIONS instead (the substrate withholds raw coords for "
            "exactly this reason)."
        )
    return violations


def hard_violations(violations: list[str]) -> list[str]:
    """The subset of validate_reply violations that should HARD-BLOCK and
    trigger a re-call: structured entity references and false state-change
    assertions.  Prose-entity mentions (tagged ``[soft]``) are excluded —
    they are surfaced to the actor as warnings but do NOT force an
    (expensive) re-poll, because the prose scan over snake_case tokens
    has a real false-positive rate (descriptive compounds like
    ``block_count`` share a tracked prefix).  Cost-safety: only re-call
    the model for the high-precision checks.  See the cost review in
    the 2026-05-30 session.
    """
    return [v for v in violations if not v.startswith("[soft]")]


def format_validator_rejection_block(violations: list[str],
                                       attempt: int) -> str:
    """Render the validator's verdict for the re-staged prompt + on-disk
    rejection record.  Empty string when no violations."""
    if not violations:
        return ""
    lines = [
        "## REPLY REJECTED BY SUBSTRATE VALIDATOR "
        f"(attempt {attempt})",
        "",
        "Your previous reply named entities perception does not track, "
        "or asserted state changes that are not in the latest delta. "
        "Re-author your reply.  Use ONLY entity names that appear in "
        "the world snapshot below.  Do NOT assert score advances, "
        "level transitions, or other state changes that are not "
        "recorded in the latest delta.",
        "",
        "Violations:",
    ]
    for v in violations:
        lines.append(f"  - {v}")
    lines.append("")
    return "\n".join(lines) + "\n"


def apply_strategy(mech_action: str,
                    strategy_reply: dict,
                    available_actions: list[str]
                    ) -> StrategyChoice:
    """Validate the strategy reply and return a StrategyChoice.  If
    the VLM's endorsed action is invalid (not in available_actions
    and not a CLICK:* form), fall back to the mechanical choice."""
    # str() coercion: a non-string endorsed_action (number/list/dict from a
    # malformed VLM reply) must not raise AttributeError on .strip().
    endorsed = str(strategy_reply.get("endorsed_action") or "").strip()
    rationale = strategy_reply.get("rationale") or ""
    testing = strategy_reply.get("testing_hypothesis")
    confidence = strategy_reply.get("confidence", "low")

    # Acting-VLM-authored interpretation (perception emits none).
    # Accept bare string or {guess: ...} for robustness.
    def _guess(v):
        if isinstance(v, dict):
            v = v.get("guess")
        v = (v or "") if isinstance(v, str) else ""
        return v.strip() or None
    game_type_v    = _guess(strategy_reply.get("game_type"))
    game_purpose_v = _guess(strategy_reply.get("game_purpose"))
    prediction_v   = _guess(strategy_reply.get("prediction"))

    # Optional subroutine-KB fields — see docs/SPEC_subroutine_kb.md
    applied_subroutine = strategy_reply.get("applied_subroutine") or None
    fork_parent        = strategy_reply.get("fork_parent") or None
    variant_notes      = strategy_reply.get("variant_notes") or None
    subroutine_application_status = (
        strategy_reply.get("subroutine_application_status") or None
    )
    # Optional active-subgoal fields — see SPEC_active_subgoals.md
    commit_subgoal_dict     = strategy_reply.get("commit_subgoal") or None
    subgoal_status_update_v = strategy_reply.get(
        "subgoal_status_update"
    ) or None
    pursuing_subgoal_id_v   = (
        strategy_reply.get("pursuing_subgoal_id") or None
    )
    commit_wc_dict          = (
        strategy_reply.get("commit_win_condition_hypothesis") or None
    )
    wc_observation_v        = (
        strategy_reply.get("win_condition_observation") or None
    )
    forward_sim             = (
        strategy_reply.get("forward_simulation") or ""
    )
    propose_probe_dict      = (
        strategy_reply.get("propose_probe") or None
    )
    probe_observation_dict  = (
        strategy_reply.get("probe_observation") or None
    )
    probe_abandon_dict      = (
        strategy_reply.get("probe_abandon") or None
    )
    planned_seq_raw         = (
        strategy_reply.get("planned_action_sequence") or None
    )
    if planned_seq_raw and isinstance(planned_seq_raw, list):
        planned_seq = [str(a) for a in planned_seq_raw if a]
    else:
        planned_seq = None
    interrupt_conditions_raw = (
        strategy_reply.get("interrupt_conditions") or []
    )
    if isinstance(interrupt_conditions_raw, list):
        interrupt_conditions = [
            str(c) for c in interrupt_conditions_raw if c
        ]
    else:
        interrupt_conditions = []
    repeat_action_v = strategy_reply.get("repeat_action") or None
    repeat_until_v = strategy_reply.get("repeat_until") or None
    if repeat_action_v:
        repeat_action_v = str(repeat_action_v)
    if repeat_until_v:
        repeat_until_v = str(repeat_until_v)

    valid = (
        endorsed in available_actions
        or endorsed.startswith("CLICK:")
    )
    if not valid:
        # fall back to mechanical
        return StrategyChoice(
            endorsed_action=mech_action,
            rationale=(f"strategy reply had invalid endorsed_action "
                        f"{endorsed!r}; falling back to mechanical"),
            testing_hypothesis=None,
            confidence="low",
            overrode_mechanical=False,
            raw_reply=strategy_reply,
            applied_subroutine=applied_subroutine,
            fork_parent=fork_parent,
            variant_notes=variant_notes,
            subroutine_application_status=subroutine_application_status,
            commit_subgoal=commit_subgoal_dict,
            subgoal_status_update=subgoal_status_update_v,
            pursuing_subgoal_id=pursuing_subgoal_id_v,
            commit_win_condition_hypothesis=commit_wc_dict,
            win_condition_observation=wc_observation_v,
            forward_simulation=forward_sim,
            propose_probe=propose_probe_dict,
            probe_observation=probe_observation_dict,
            probe_abandon=probe_abandon_dict,
            planned_action_sequence=planned_seq,
            interrupt_conditions=interrupt_conditions,
            repeat_action=repeat_action_v,
            repeat_until=repeat_until_v,
            game_type=game_type_v,
            game_purpose=game_purpose_v,
            prediction=prediction_v,
        )

    return StrategyChoice(
        endorsed_action=endorsed,
        rationale=rationale,
        testing_hypothesis=testing,
        confidence=confidence,
        overrode_mechanical=(endorsed != mech_action),
        raw_reply=strategy_reply,
        applied_subroutine=applied_subroutine,
        fork_parent=fork_parent,
        variant_notes=variant_notes,
        subroutine_application_status=subroutine_application_status,
        commit_subgoal=commit_subgoal_dict,
        subgoal_status_update=subgoal_status_update_v,
        pursuing_subgoal_id=pursuing_subgoal_id_v,
        commit_win_condition_hypothesis=commit_wc_dict,
        win_condition_observation=wc_observation_v,
        forward_simulation=forward_sim,
        propose_probe=propose_probe_dict,
        probe_observation=probe_observation_dict,
        probe_abandon=probe_abandon_dict,
        planned_action_sequence=planned_seq,
        interrupt_conditions=interrupt_conditions,
        repeat_action=repeat_action_v,
        repeat_until=repeat_until_v,
        game_type=game_type_v,
        game_purpose=game_purpose_v,
        prediction=prediction_v,
    )
