"""Game-agnostic exploratory driver.

Drives the perception → actor → action → delta → world-update loop
against ANY game (via a GameAdapter) without baking in any game-
specific assumptions.

Per-turn workflow:

  1. PERCEIVE — stage a perception VLM call (symbolic-first if we
     have prior knowledge, image-supplemented if uncertain).  The
     reply is parsed into the standard schema (entities, groups,
     relationships, grid_inference, symbolic_state).

  2. UPDATE — fold the perception output into the WorldKnowledge
     accumulator.  Entities get persistent IDs (matched by name);
     relationships accumulate confidence; grid_inference refines.

  3. MINE — for the action+delta pair from the LAST turn, run the
     mechanic miner.  Promoted hypotheses become rules the actor
     plans against.

  4. ACT — call the cell_actor (Goal-Forest + BFS) to pick the next
     action.  The actor sees the full WorldKnowledge including
     trusted mechanic rules.

  5. STEP — send the action to the GameAdapter; get the next frame.

  6. DELTA — stage a delta-perception call comparing the new frame
     to the previous one.  Records what moved / appeared /
     disappeared.

Stop conditions: win_state != "playing", turn budget exhausted, or
the actor returns NONE (no move available).

Game-agnostic: the driver knows nothing about specific games.  It
relies on the GameAdapter for I/O and on the prompt's open
vocabulary (action strings, entity names) for everything else.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from cell_actor import choose_action, ActionChoice              # noqa: E402
from game_adapter import (                                       # noqa: E402
    GameAdapter, StepResult, _CARDINAL_ALIASES,
)
from global_priors import (                                      # noqa: E402
    DEFAULT_PRIORS_PATH, load_and_seed, update_and_save,
)
from mechanic_miner import mine_step, trusted_rules             # noqa: E402
from planner_integration import choose_planned_action            # noqa: E402
from vlm_strategy import (                                       # noqa: E402
    StrategyChoice, stage_strategy_call, poll_strategy_reply,
    apply_strategy, validate_reply, hard_violations,
    format_validator_rejection_block,
)
from subroutine_kb import (                                       # noqa: E402
    record_application_outcome as _subroutine_record_outcome,
    promote_chain_as_subroutine as _subroutine_promote,
)
from active_subgoals import (                                     # noqa: E402
    commit_subgoal as _sg_commit,
    update_subgoal_status as _sg_update,
)
from world_knowledge import (                                    # noqa: E402
    WorldKnowledge, ActionRecord, DeltaRecord,
)
# Game-agnostic substrate (prompt templates, grid overlay, bbox
# renderer).  Imported via the neutrally-named re-export module
# so the driver never appears to depend on a specific game.
from perception_substrate import (                              # noqa: E402
    _add_grid_overlay, _fmt_prompts, render_turn1_overlay,
    DEFAULT_N_TICKS, DEFAULT_LABEL_STRIDE,
)
# Refinement prompt template lives in the substrate's underlying
# module (alongside _add_grid_overlay).  Imported via the neutrally-
# named re-export module would be cleaner; for now reach in directly.
from run_bp35_sequence_probe import (                           # noqa: E402
    REFINEMENT_PROMPT_TEMPLATE, ticks_to_playfield_px,
)
import visual_query as _VQ                                       # noqa: E402
import instincts as _INST                                        # noqa: E402
import debugging_discipline as _DBG                              # noqa: E402
from PIL import Image                                            # noqa: E402
import math as _math                                             # noqa: E402

# Bound on VLM-directed visual-query round-trips per perception reply: the VLM
# may ask the substrate for on-demand help (zoom/count/measure/align/...), see
# the answers, then finalise — but only a few times, so a confused VLM cannot
# loop forever requesting tools.  Strict-mode safe: each round is just another
# bounded poll that degrades to the substrate-only fallback on timeout.
_VQ_MAX_ROUNDS = 2


# ---------------------------------------------------------------------------
# Prompt templates — symbolic-first
# ---------------------------------------------------------------------------


INITIAL_PERCEPTION_PROMPT = """\
# Exploratory-loop INITIAL perception — game `{game_id}` level {level}

Model handle: human:claude

## SYSTEM PROMPT

```
{sys_text}
```

---

## USER MESSAGE

Image attachment:
  `image_grid.png` — the turn-1 frame with the {n_ticks}-tick grid
  overlay (label_stride={label_stride}).

This is the FIRST turn of a fresh level.  You have no prior
knowledge to work from — extract entities, groups, relationships,
grid_inference, and symbolic_state per the schema in the system
prompt.

```
{user_text}
```

---

## Reply instructions

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


VISUAL_QUERY_FOLLOWUP_PROMPT = """\
# Exploratory-loop VISUAL-TOOL results — game `{game_id}` level {level}

Model handle: human:claude

You asked the substrate for on-demand visual help.  Here are the measured
answers (the substrate MEASURED only — it did not interpret anything; the
interpretation is yours).

## Tool results

```json
{results_json}
```
{image_block}
## What to do now

Using these precise answers, RE-EMIT your perception for this frame in the
SAME JSON shape you used before (same top-level keys / schema).  Fold the
measurements into your entity descriptions, internal-condition reports,
relationships, and symbolic_state.

You MAY request another round of `visual_queries` in this reply if you still
need a measurement (up to a small bound), but prefer to finalise now.

## Reply instructions

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


LEVEL_START_PERCEPTION_PROMPT = """\
# Exploratory-loop LEVEL-START perception — game `{game_id}` level {level}

Model handle: human:claude

## SYSTEM PROMPT

```
{sys_text}
```

---

## USER MESSAGE — a NEW level of a game you have ALREADY played

Image attachment:
  `image_grid.png` — this level's first frame with the {n_ticks}-tick grid
  overlay (label_stride={label_stride}).

This is a NEW level of the SAME game whose earlier level(s) you already
analyzed.  You are NOT starting from zero.

## PRIOR-LEVEL CONTEXT — strong priors

{prior_context}

---

Now extract entities, groups, relationships, grid_inference, and symbolic_state
for THIS level per the schema in the system prompt, APPLYING the strong priors
above (reuse names + roles for matching entities; add what's new; note what's
absent or changed).

```
{user_text}
```

---

## Reply instructions

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


REFINEMENT_PERCEPTION_PROMPT = """\
# Exploratory-loop REFINEMENT perception — game `{game_id}` level {level}

Model handle: human:claude

## SYSTEM PROMPT

```
{sys_text}
```

---

## USER MESSAGE — SECOND-PASS BBOX REFINEMENT

Image attachments:
  1. `image_grid.png` — the original turn-1 frame with the {n_ticks}-tick
     grid overlay.  Identical to the image you saw on the first pass.
  2. `refinement_overlay.png` — the SAME frame with YOUR previous-pass
     bboxes drawn on top as cyan rectangles, each labeled `#N`.

{ref_text}

---

## Reply instructions

Write the refined JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


COMPLETENESS_PERCEPTION_PROMPT = """\
# Exploratory-loop COMPLETENESS re-look — game `{game_id}` level {level}

Model handle: human:claude

## SYSTEM PROMPT

```
{sys_text}
```

---

## USER MESSAGE — COVERAGE RE-LOOK (you decide; the harness only measured)

Your current entity inventory accounts for only **{covered_pct}%** of the
NON-BACKGROUND pixels in this frame.  The harness has measured which pixels you
have not assigned to any entity and highlighted those regions in RED, labeled
`U1`..`U{n_unc}`.  The harness does NOT know what they are — naming them is YOUR
job.  It is only handing you a measurement so nothing visible is missed.

Image attachments:
  1. `image_grid.png` — the frame with the {n_ticks}-tick grid.
  2. `coverage_overlay.png` — the same frame: your current entities in faint
     cyan, the UNACCOUNTED regions in red (`U1`..`U{n_unc}`).

For EACH red region, decide using the scene and your world knowledge:
  * Is it a REAL entity you missed — a structure (rod / rail / wall / shaft), a
    marker, a cap, a target?  Then ADD it to your entity list with a
    `role_hypothesis`.  Long thin regions are usually structures, which are
    often the most important objects in the scene.
  * Or is it background / decoration / a frame border to IGNORE?  Then put its
    label under `dismissed_regions` with a one-word reason.

Return the COMPLETE corrected inventory (the full entity list, not just the
additions), in the SAME entity schema as before, plus an optional
`dismissed_regions`:

```
{{"entities": [ ... full list, each with name/bbox_ticks_turn1/role_hypothesis/confidence ... ],
  "dismissed_regions": [ {{"bbox_ticks_turn1": [r1,c1,r2,c2], "reason": "frame_border"}} ]}}
```

---

## Reply instructions

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


# The animation-first instinct's content + re-prompt now live in instincts.py
# (_ANIMATION_CONTENT / _ANIMATION_REPROMPT), registered + ENFORCED via the
# instinct registry; the driver surfaces / enforces them generically.


DELTA_PERCEPTION_PROMPT = """\
# Exploratory-loop DELTA perception — turn {turn_n} of game `{game_id}` lc={level}

Model handle: human:claude

## SYSTEM PROMPT

```
{sys_text}
```

---

## USER MESSAGE — DELTA + SYMBOLIC-FIRST REASONING

You have accumulated a rich symbolic model of this level over
{turn_prev} turn(s).  PRIMARILY reason from that symbolic snapshot;
only consult the frame images to RESOLVE AMBIGUITIES the symbolic
state can't answer.

PRIOR WORLD MODEL (your accumulated knowledge — symbolic-first
source of truth):

```json
{world_snapshot}
```

LAST ACTION the actor took: `{last_action}`

SUBSTRATE-COMPUTED ENTITY DELTAS — the object-constancy tracker matched the
prior frame's objects to the current frame and computed what each one did.
Read EACH line as a logical ENTITY moving / appearing / vanishing (object
constancy), NOT as raw pixels: a thing that "appears" where the action landed
is usually a controllable entity that MOVED there. These are substrate facts;
cross-check against the two images and use them to fill `delta` and to reason
about the mechanic (e.g. "this action relocates that entity → I can steer it").

```
{entity_deltas}
```

SUBSTRATE-COMPUTED ANIMATION — the action did not necessarily jump straight to
the settled frame: the engine plays an ANIMATION (a sequence of in-between
frames), and those frames carry the MECHANIC. Animation is usually used to
DISPLAY AN ENTITY MOVING, so the substrate extracts ENTITIES on EVERY sub-frame
and tracks each across them by object constancy. Two views follow.

ENTITY MOVEMENTS (Layer-3/4 — the substrate DETECTED these movements; each is a
FACT, and YOUR job is to CORRELATE it with the other entities/regions: which
entity moved, ALONG what, TOWARD what. E.g. a small mark that appears and sweeps
along a row of targets, then vanishes, is a CURSOR reading them; an entity that
travels into another is a delivery/collision):
```
{animation_entities}
```

COLOUR-REGION DYNAMICS (Layer-2 — a single colour's extent / flow / leading edge
across the frames; complementary low-level detail. E.g. a region that descends,
spreads along a bar, then flows down both edges = a LIQUID poured and routed):
```
{animation_summary}
```
{animation_filmstrip_block}
IMAGES (consult only if needed):
  1. `prev_frame.png` — the gridded frame from turn {turn_prev}.
  2. `curr_frame.png` — the gridded frame from turn {turn_n}.

YOUR TASK — output a JSON object with TWO sections.  Section A is
a delta describing what changed between the two frames.  Section B
is a FULL refreshed perception of the current frame (same schema as
the initial perception).  The driver folds B into the WorldKnowledge
and uses A to drive mechanic discovery.

{{
  "delta": {{
    "agent_moved":             true | false,
    "agent_new_cell":          [row, col] | null,
    "inferred_action":         "<UP | DOWN | LEFT | RIGHT | CLICK | NONE | OTHER>",
    "entities_appeared":       ["<name>", ...],
    "entities_disappeared":    ["<name>", ...],
    "entities_changed":        ["<name>", ...],
       // entities whose bbox / role / appearance changed meaningfully
    "summary":                 "<one paragraph plain-English summary>"
  }},

  "animation_analysis": {{
    // ANIMATION-FIRST INSTINCT — REQUIRED whenever the action ANIMATED (an
    // animation filmstrip is shown above) AND you cannot yet STATE the win
    // condition.  The motion across the sub-frames IS the mechanic; never judge
    // it from the settled frame alone (an object can launch, sweep, and return
    // to rest, looking "unchanged").  Omit this key ONLY if there was no
    // animation, OR you already understand the win condition.
    "entities":   ["<objects visible/active in the animation frames>"],
    "movements":  ["<entity X: how it MOVED / grew / shrank / recoloured /
                    appeared / vanished across the frames (object constancy)>"],
    "entity_event_relations": ["<what your action caused; which entity is
                    controllable/affected; what the motion reveals about the rule
                    or win>"]
  }},

  "perception": {{
    // SAME schema as the initial perception: entities, groups,
    // relationships, grid_inference, symbolic_state, game_type,
    // game_purpose, frame_to_frame_summary, overall_notes.
    //
    // Keep entity NAMES STABLE across turns — if an entity from
    // the prior model is still present, use the same name.  Add
    // new entities for things that newly appeared.  Drop entities
    // that have genuinely disappeared.
    //
    // OPTIONAL per-entity `function` field: once you have LEARNED what an
    // entity DOES (a legend icon's effect, a control's action — e.g.
    // "ROTATE the agent"), set "function" on it. The substrate records that
    // into the cross-game VISUAL CATALOG keyed by the entity's appearance,
    // so a resembling entity in a future level/game recalls it as a prior.
    ...
  }},

  "claims": [
    // OPTIONAL — your EPISTEMIC FRONTIER.  Each claim is a HYPOTHESIS worth
    // resolving (a mechanic rule, an entity PURPOSE / affordance, or a win
    // condition), ideally with a cheap DISCRIMINATING probe that would settle
    // it.  The substrate PERSISTS these (across levels + re-runs), RANKS them by
    // value-of-information, and RUNS the top probe for you — so actions go to the
    // most informative unknown and already-proven claims are not re-probed.
    // Author one for every reasonable OPEN question — ESPECIALLY a clearly-SEEN
    // entity whose PURPOSE is still unknown (perceived is NOT understood).
    {{
      "id": "<stable_snake_case_id>",
      "statement": "<hypothesis, e.g. 'left panel switches are interactive'>",
      "scope": "level" | "cross",   // level = tied to THIS layout; cross = game-wide rule
      "target": ["<entity name(s) it concerns>"],
      "probe": "<ONE discriminating action, e.g. CLICK:8,44 (exact cell), CLICK:<entity_name> (on a thing), CLICK:FLOOR (a random EMPTY cell -- say this when you mean a truly exploratory click on empty space, NOT on any entity), or ACTION6; omit if none>",
      "cost": 1, "importance": 0.8, "credence": 0.5  // credence 0.5 = unknown
    }}
  ],
  "claim_updates": [
    // OPTIONAL — RESOLVE claims from what you just observed (this shrinks the
    // frontier).  {{"id": "<claim id>", "outcome": "proven" | "refuted"}}
  ],
  "errors_emitted": [
    // OPTIONAL — register YOUR OWN significant mistakes into the durable KB error
    // ledger, so a SIMILAR future situation RECALLS them (and the fix) instead of
    // repeating them. Use for a mistake worth guarding against (mover/goal mix-up,
    // misread legend/demo, contaminated-frame read, ...). Add `resolution` (+
    // `solution_id` if it is a replayable win-path) to mark it RESOLVED and LINK the
    // solution that worked — only after the solution actually worked.
    {{"category": "<snake_case area>", "description": "<what went wrong>",
      "variation": "<size|orientation|colour|position — optional>",
      "fix": "<one-line guard — optional>",
      "resolution": "<the solution that worked — optional>",
      "solution_id": "<solutions_kb id — optional>"}}
  ]
}}

Return ONLY the JSON object.

---

## Reply instructions

Write JSON to:
  `{reply_name}`
"""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


@dataclass
class TurnReport:
    """Summary of one turn for the operator log."""
    turn: int
    action: str
    inferred_action: str
    agent_moved: bool
    new_cell: Optional[list[int]]
    entities_changed: int
    new_mechanic_hypotheses: list[str]
    promoted_rules: list[str]


class ExploratoryDriver:
    """Drives the perception / actor / mechanic loop against a game.

    The driver is GAME-AGNOSTIC.  It takes a GameAdapter (fixture
    replay or live harness) and a working directory, plus the
    WorldKnowledge accumulator that gets updated each turn.

    The actual VLM calls are STAGED via the human:claude pending-
    file protocol the project uses elsewhere — the driver writes a
    prompt + image(s) to a per-turn subdirectory and waits for the
    reply file to appear.  This means the driver works with either
    a parallel-subagent operator (this session's pattern) or a
    polling automation script.
    """

    def __init__(self, game: GameAdapter, world: WorldKnowledge,
                  work_dir: Path,
                  n_ticks: int = DEFAULT_N_TICKS,
                  label_stride: int = DEFAULT_LABEL_STRIDE,
                  upscale: int = 16,
                  poll_s: float = 2.0,
                  timeout_s: int = 1200,
                  vlm_timeout_s: Optional[int] = None,
                  use_strategy: bool = True,
                  use_planner: bool = True,
                  grid_line_width_major: int = 1,
                  grid_line_width_minor: int = 1,
                  grid_major_alpha: int = 200,
                  grid_minor_alpha: int = 90):
        self.game = game
        self.world = world
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        # Persistent, scoped, value-of-information-ranked Claim Store -- the
        # epistemic frontier that drives claim-directed probing.  Loaded from
        # the KB per game so a RE-RUN reloads what is already proven and spends
        # its budget on the OPEN claims (claim_store.py).  Guarded -- a missing
        # store never breaks the run.
        try:
            from claim_store import ClaimStore
            self._claim_store = ClaimStore.load_for_game(getattr(game, "game_id", "?"))
        except Exception as e:
            print(f"[claim-store] init skipped ({e})")
            self._claim_store = None
        # Level ORDINAL for claim scoping.  The live adapter never increments
        # self.game.level (levels advance internally; it stays 0), so we count
        # transitions ourselves: 0 = first level, +1 each level-start analysis.
        # Stable across re-runs (the Nth level is always "lc{N}").
        self._level_ordinal = 0
        # Turn the current level started on -- the cx tiers look only at THIS
        # level's deltas (deltas_observed is cross-level history, so a prior
        # level's transient mover must not be read on a new layout).
        self._level_start_turn = 0
        # Scope the RELOADED persisted frontier to the level we are actually
        # STARTING on.  A re-run begins at the first level, but the KB may hold
        # POSITIONED claims settled on a LATER layout (e.g. a persisted lc1
        # 'function_of_blue_ball').  Such an off-level claim is inactive here AND
        # (sharing this level's auto-id) blocks coverage from authoring an active
        # one -- so the entity would never be probed.  carry_to_new_level drops
        # positioned claims from a different layout while KEEPING every
        # cross-level / cross-game claim (the transferable knowledge), exactly as
        # on a normal level transition.  Guarded.
        if self._claim_store is not None:
            try:
                self._claim_store.carry_to_new_level(self._level_signature())
            except Exception as e:
                print(f"[claim-store] start-scope skipped ({e})")
        self.n_ticks = n_ticks
        self.label_stride = label_stride
        # upscale: integer factor applied to the 64×64 game frame before the
        # grid overlay is drawn.  This is the ONLY substrate "help" needed for
        # the VLM to read RECURSIVE / SUB-TILE structure (a thumbnail/key/legend
        # drawn INSIDE a cell): make fine detail legible, then let the flexible
        # VLM decode it — never bake a sub-grid CV decoder (that overfits one
        # game's drawing style and fails the Prime-Directive adversarial test).
        # upscale=8 (634px) was tuned for 4-tick ENTITY localisation but a 2-tick
        # sub-cell is only ~16px there — borderline.  upscale=16 (~1270px) makes a
        # 2-tick sub-cell ~32px, so a 3x3 thumbnail key is plainly readable, while
        # staying under the vision API's ~1568px resample threshold so the grid +
        # labels still reach the model intact.  (At 64 ticks, upscale<=18 stays
        # under the cap.)
        self.upscale = upscale
        # Grid-line styling.  Both lines are 1 px wide so the grid
        # never obscures the game content; the MAJOR-vs-MINOR
        # distinction is carried by alpha (major ~200, minor ~90):
        # major gridlines (every 4 ticks where the labels are) are
        # clearly more visible than the minor tick lines, so the VLM
        # can anchor bbox coordinates by snapping to the prominent
        # major lines and then offsetting by a small minor count.
        # Reverts a 2026-05-27 "variant E" tweak that flattened both
        # alphas to 100 and lost the coordinate-anchoring cue.
        # Pass to _add_grid_overlay() via _gridded_for() below.
        self.grid_line_width_major = grid_line_width_major
        self.grid_line_width_minor = grid_line_width_minor
        self.grid_major_alpha = grid_major_alpha
        self.grid_minor_alpha = grid_minor_alpha
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        # vlm_timeout_s: how long a single VLM-reply poll (perception / strategy)
        # waits before DEGRADING to the substrate-only fallback instead of
        # crashing — see _fallback_perception.  Defaults to timeout_s (good for
        # HITL, where a human/Claude may think for a while).  For strict /
        # competition mode set it short (e.g. 30-60s via --vlm-timeout-s) so a
        # slow/dead autonomous VLM is bypassed quickly and hundreds of games run
        # non-stop.  Robustness is identical either way; this only tunes how long
        # COS waits before falling back.
        self.vlm_timeout_s = (vlm_timeout_s
                              if vlm_timeout_s is not None else timeout_s)
        # use_strategy=True (default) inserts a per-turn VLM strategy
        # call between the mechanical actor and the game step.  Set
        # to False for cheaper / purely-mechanical runs (skips one
        # VLM call per turn).
        self.use_strategy = use_strategy
        # use_planner=True (default) uses the existing cognitive_os
        # explorer + AO* planner via planner_integration as the
        # PRIMARY action source.  Falls back to the mechanical
        # cell_actor BFS when the planner returns no plan.  Set to
        # False to use cell_actor only (pre-Phase-3 behavior).
        self.use_planner = use_planner
        # global_priors_path: cross-session/cross-game store of
        # action-effect priors learned from previous game runs.
        # At driver start: load + seed world.mechanic_hypotheses
        # (so the planner can plan on turn 1 using prior
        # knowledge).  At driver end: extract this run's
        # observations and merge back into the store.  Set to
        # None to disable cross-game transfer entirely.
        # Competition (strict) mode: keep the cross-game store under the FRESH, writable
        # COS_KB_ROOT, so it starts empty/seeded and only ACCUMULATES across the
        # competition's OWN games -- never the dev store of prior public-game priors
        # (that would be knowledge of already-seen games, a rules violation).  Dev mode
        # keeps using the dev store.
        if os.environ.get("COS_STRICT") == "1":
            from kb_paths import kb_path
            self.global_priors_path: Optional[Path] = kb_path("global_priors.json")
        else:
            self.global_priors_path: Optional[Path] = DEFAULT_PRIORS_PATH
        self._global_priors = None  # loaded lazily in run_turn_one
        self._last_frame_path: Optional[Path] = None
        # Most-recent action's animation sub-frame dir (for the filmstrip +
        # the on-demand animation_zoom tool); None when the action didn't animate.
        self._last_anim_dir: Optional[Path] = None
        self._last_anim_events: list = []   # substrate entity-movements, last animated turn
        self._last_filmstrip_path: Optional[str] = None
        # Current level's start frame, used to bootstrap per-entity colors for
        # the live bbox refresh (frame_bbox_refresh). Set at each level start
        # so colors are sampled from the CURRENT level's board, not turn-1.
        self._level_start_frame_path: Optional[Path] = None
        # Per-instance perception (substrate-validated facts the VLM reads
        # instead of eyeballing the frame). See instance_perception + the
        # perception contract (SPEC_perception_contract.md).
        self._instance_tracker = None
        self._prev_instance_tracker = None
        self._reports: list[TurnReport] = []
        # Subroutine-KB auto-promotion staging.  Populated on score
        # advance; consumed opportunistically at next turn boundary.
        self._pending_promotion_prompts: list = []
        # -------- Strategic/mechanical separation --------
        # When the strategy actor commits a planned_action_sequence,
        # the substrate executes the sequence mechanically (without
        # re-prompting the actor) until either (a) the sequence
        # completes, or (b) one of the actor's declared interrupt
        # conditions fires.  Mechanical execution skips the VLM
        # strategy call, dramatically reducing the per-turn cost.
        # The actor still pays for: the initial sequence-committing
        # call, interrupt-driven re-prompts, and the end-of-sequence
        # "what's next" call.
        self._pending_sequence: list[str] = []
        self._sequence_idx: int = 0
        self._interrupt_conditions: list[str] = []
        self._sequence_goal_id: Optional[str] = None
        self._sequence_committed_at_turn: Optional[int] = None
        self._sequence_rationale_prefix: str = ""
        # -------- Repeat-until-relation executor (game-agnostic) --------
        # The VLM commits a sub-maneuver as "repeat ONE opaque action until a
        # RELATIONAL stop-condition holds (or progress stalls)".  The harness
        # owns the MAGNITUDE (how many repeats) + a DROP-GUARD (UNDO if a
        # previously-completed goal target regresses) + failure feedback.  No
        # game semantics here: the action label is opaque and the condition is
        # in the existing skin-agnostic relation vocabulary over tracked ids.
        # SPEC_vlm_backward_reasoning.md (execution half).
        self._repeat_until: Optional[str] = None      # stop-condition string
        self._repeat_count: int = 0                    # repeats executed so far
        self._repeat_cap: int = 25                     # safety bound
        self._repeat_done_baseline: Optional[set] = None  # goal done-set at entry
        self._repeat_skewered_baseline: Optional[set] = None  # skewered blocks at entry
        self._repeat_feedback: Optional[str] = None    # surfaced to next VLM turn
        # SELF-CORRECTING trend monitor (catches a slow drift / stall the
        # single-step surprise check is blind to -- see _monitor_progress).
        self._progress_ledger = _DBG.ProgressLedger()
        self._progress_mover: Optional[str] = None
        # Demonstration-learned click-to-move law (see _learn_move_law_from_demo /
        # _move_law_pursuit): the NUMERIC step a one-shot marker taught.
        self._move_law = None
        self._move_demo: Optional[dict] = None
        self._move_last = None      # last-known mover position (row,col) for tracking
        # Cross-game UNDO convention (P-priors: ACTION7 ~= undo); used by the
        # drop-guard to revert a repeat that regressed a completed target.
        # A setting, not a hardcode in logic; guarded on availability.
        self._undo_action: str = "ACTION7"
        # -------- Proceduralization (System 2 -> System 1) --------
        # When a successful deliberate solve of a goal-segment recurs as
        # the same relational signature, run the compiled skill directly
        # (bypassing the VLM strategy call).  All state guarded; any
        # failure degrades to the normal VLM-driven loop.
        # docs/SPEC_proceduralization.md.
        self.use_proceduralization = True
        self._proc_seg_goal = None          # goal_key of the active segment
        self._proc_seg_sig = None           # entry signature of the segment
        self._proc_seg_actions: list[str] = []
        self._proc_seg_start_turn: Optional[int] = None
        self._proc_autorun_tried: set = set()   # (sig,goal) tried unsuccessfully
        self._proc_autorun_active = None    # (sub_id,sig,goal) currently running
        self._write_baseline_run_info()

    # ---- Proceduralization helpers (all guarded; never raise) --------

    def _proc_sig_goal(self):
        import proceduralization_bridge as _PB   # noqa: E402
        return _PB.relational_signature(self.world), _PB.goal_key(self.world)

    def _proc_maybe_autorun(self):
        """Return a skill action list if a CONFIDENT canonical-KB skill
        matches the current signature+goal (System 1), else None."""
        if not getattr(self, "use_proceduralization", False):
            return None
        try:
            import proceduralization_bridge as _PB     # noqa: E402
            import subroutine_kb as _SK                 # noqa: E402
            subs = _SK.load()
            ctrl = _PB.controller_from_kb(subs)
            if not ctrl.skills:
                return None
            sig, goal = self._proc_sig_goal()
            if goal is None or (sig, goal) in self._proc_autorun_tried:
                return None
            dec = ctrl.decide(sig, goal,
                              reasoner=lambda **k: (), trigger=False)
            if dec.mode == "auto" and dec.actions:
                # remember which record is running, for credence feedback
                sub_id = None
                for s in subs:
                    if (getattr(s, "concrete_chain", None)
                            and tuple(s.concrete_chain) == tuple(dec.actions)):
                        sub_id = s.subroutine_id
                        break
                self._proc_autorun_active = (sub_id, sig, goal)
                print(f"[proc] System-1 auto-run: {list(dec.actions)} "
                      f"for goal={goal}")
                return list(dec.actions)
        except Exception as e:
            print(f"[proc] autorun check failed ({e}); continuing")
        return None

    def _proc_seg_append(self, final_action: str):
        """Pre-step: maintain the current goal-segment (reset when the
        goal changes) and append the executed action."""
        if not getattr(self, "use_proceduralization", False):
            return
        try:
            sig, goal = self._proc_sig_goal()
            if goal != self._proc_seg_goal:
                self._proc_seg_goal = goal
                self._proc_seg_sig = sig
                self._proc_seg_actions = []
                self._proc_seg_start_turn = self.world.turn
            self._proc_seg_actions.append(final_action)
        except Exception as e:
            print(f"[proc] segment append failed ({e})")

    def _proc_on_perception(self):
        """Post-perception: if the active segment's target just COMPLETED
        (goal's next-target advanced), compile the maneuver into the
        canonical subroutine KB and start a fresh segment.  Also closes
        out credence feedback for an auto-run."""
        if not getattr(self, "use_proceduralization", False):
            return
        try:
            import proceduralization_bridge as _PB     # noqa: E402
            new_goal = _PB.goal_key(self.world)
            seg_goal = self._proc_seg_goal
            if seg_goal is None or new_goal == seg_goal:
                return  # segment still open (or nothing to do)
            # the segment's goal changed -> its target was completed
            # (next-target advanced) => the segment SUCCEEDED.
            acts = list(self._proc_seg_actions or [])
            sig = self._proc_seg_sig
            if acts and sig and seg_goal:
                _PB.compile_success_to_kb(
                    actions=acts, signature=sig, goal=seg_goal,
                    game_id=self.world.game_id, level=self.world.level,
                    turn_range=[self._proc_seg_start_turn or self.world.turn,
                                self.world.turn])
                print(f"[proc] compiled skill: goal={seg_goal} "
                      f"({len(acts)} actions)")
            self._proc_autorun_active = None
            # start fresh segment under the new goal
            self._proc_seg_goal = new_goal
            self._proc_seg_sig, _ = self._proc_sig_goal()
            self._proc_seg_actions = []
            self._proc_seg_start_turn = self.world.turn
        except Exception as e:
            print(f"[proc] on_perception failed ({e})")

    def _write_baseline_run_info(self) -> None:
        """Seed run_info.json with what the harness itself knows
        (game/level/harness/perception+strategy channels/start time)
        so the trace's Run-information header is always populated.

        Merge, never clobber: a separate process that actually calls
        the model (e.g. the strategy responder) augments this file
        with acting_model / acting_provider / acting_endpoint.
        """
        import datetime as _dt
        path = self.work_dir / "run_info.json"
        info: dict = {}
        if path.exists():
            try:
                info = json.loads(path.read_text(encoding="utf-8")) or {}
            except Exception:
                info = {}
        info.setdefault("started_at", _dt.datetime.now().isoformat(timespec="seconds"))
        info.update({
            "game_id": getattr(self.game, "game_id", None),
            "level": getattr(self.game, "level", None),
            "harness": "exploratory_driver.py",
            "perception_pipeline": "external reply files (reply.txt / refinement_reply.txt)",
            "strategy_channel": ("external strategy_reply.txt (file handoff)"
                                  if self.use_strategy else "disabled"),
        })
        try:
            path.write_text(json.dumps(info, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[run-info] could not write {path}: {e}")

    # --- prompt staging ---

    def _gridded_for(self, frame_path: Path, out: Path) -> Path:
        img = Image.open(frame_path).convert("RGB")
        gridded, _, _ = _add_grid_overlay(
            img, n_ticks=self.n_ticks, upscale=self.upscale,
            label_stride=self.label_stride,
            line_width_major=self.grid_line_width_major,
            line_width_minor=self.grid_line_width_minor,
            major_alpha=self.grid_major_alpha,
            minor_alpha=self.grid_minor_alpha,
        )
        gridded.save(out)
        return out

    def _ground_perception_geometry(self, parsed: dict, frame_path,
                                     persist_dir: Optional[Path] = None) -> dict:
        """QUALITY-ASSURANCE pass over the perception reply: snap every entity's
        bbox to the substrate's MEASURED connected component(s) on the actual
        frame (entity_grounding), so stored geometry is pixel-accurate instead of
        hand-estimated.  The VLM keeps SEMANTICS (name/role/grouping); the
        substrate owns GEOMETRY.  Logs components the VLM MISSED and boxes that
        matched nothing so the quality gap is visible.

        When ``persist_dir`` is given, the GROUNDED perception is written to
        ``persist_dir/perception_grounded.json`` so the trace's entity-analysis
        view renders the accurate (snapped) bboxes rather than the raw
        hand-estimated reply.  Defensive: returns parsed unchanged on any error
        or when the frame/entities are unavailable.  See entity_grounding +
        durable principle P3 (geometry from the frame)."""
        try:
            ents = parsed.get("entities") or []
            if not ents or frame_path is None or not Path(frame_path).exists():
                return parsed
            import numpy as _np                               # noqa: E402
            from PIL import Image as _Image                   # noqa: E402
            import entity_grounding as _eg                     # noqa: E402
            frame = _np.array(_Image.open(frame_path).convert("RGB"))
            grounded, rep = _eg.ground_entity_bboxes(
                ents, frame, n_ticks=self.n_ticks)
            parsed["entities"] = grounded
            parsed["grounding_qa"] = rep.get("quality")    # persisted for the trace
            parsed["_grounding_report"] = rep              # full report for the VLM gate
            print(f"[grounding] {_eg.quality_line(rep)}")
            # IDENTITY channel: a rotation/scale-invariant SHAPE signature per
            # entity, computed here where frame + grounded bbox are aligned, so a
            # recurring entity can be matched across levels by shape (carried into
            # the level-memory templates).  Guarded; never blocks grounding.
            try:
                import shape_identity as _si
                for e in grounded:
                    bb = e.get("bbox_ticks_turn1")
                    if bb:
                        sig = _si.shape_signature(frame, bb)
                        if sig:
                            e["shape_sig"] = sig
                        crop = _si.color_crop_b64(frame, bb)
                        if crop:
                            e["crop_b64"] = crop
            except Exception as e:
                print(f"[grounding] shape-sig skipped ({e})")
            # VISUAL MEMORY (cross-game): record each salient entity in the catalog so the memory
            # GROWS, and RECALL whether any resembles a salient entity seen in a PRIOR game -- a prior
            # about what it is / does, surfaced to the actor before it probes (e.g. a diamond icon that
            # acted as ROTATE elsewhere).  The catalog measures resemblance; the VLM verifies.
            try:
                import visual_catalog as _vc
                _cat = _vc.VisualCatalog()
                _gid = getattr(getattr(self, "world", None), "game_id", "") or ""
                # Cross-game recall gate: a visual prior from ANOTHER game may
                # surface only if that game is a structural VARIANT of the
                # current one (admits_xgame); same-game / untagged priors pass.
                # Fail closed (cross-game dropped) if the gate is unavailable.
                _vadmit = (lambda games, _g=_gid: (not games) or (_g in games))
                try:
                    from cross_game_knowledge import (compute_signature as _csig,
                                                      load_store as _xload,
                                                      admits_xgame as _xa)
                    _vsig = _csig(self.world,
                                  getattr(self.world, "_available_actions", None)
                                  or getattr(self.world, "available_actions", None))
                    _vstore = _xload()

                    def _vadmit(games, _s=_vsig, _st=_vstore, _g=_gid):
                        if not games:
                            return True
                        return any(_xa(gg, _s, current_game_id=_g, store=_st)
                                   for gg in games)
                except Exception:
                    pass
                # AUTO-RECORD-ON-LEARNING: the VLM sets an entity's `function` once it has learned the
                # effect (from a demonstration / fire); catalog that function with the icon's signature,
                # so the visual memory's FUNCTIONS populate themselves during play (not just appearance).
                _fn_map = {e.get("name"): e.get("function", "") for e in ents
                           if isinstance(e, dict) and e.get("function")}
                _priors = []
                for _e in grounded:
                    _sg = _e.get("shape_sig")
                    if not _sg:
                        continue
                    _nm = _e.get("name", "?")
                    _pr = _cat.recall_prior(_sg, color=_e.get("color") or None, game=_gid, k=1,
                                            admit=_vadmit)
                    if _pr:
                        _priors.append(_nm + " " + _pr.split("\n", 1)[-1].strip())
                    _fn = _fn_map.get(_nm, "")
                    _cat.record(_nm, _sg, color=_e.get("color", ""), crop_b64=_e.get("crop_b64", ""),
                                meaning=_e.get("appearance", ""), function=_fn, game=_gid,
                                level=self._level_signature(),
                                credence=(0.8 if _fn else 0.5),
                                provenance=("learned" if _fn else "observed"))
                if _priors:
                    self._pending_visual_note = ("[VISUAL-MEMORY] entities resembling ones seen before "
                                                 "(prior -- verify, don't assume):\n  - "
                                                 + "\n  - ".join(_priors))
                    print(f"[visual-memory] recalled {len(_priors)} prior(s) from the catalog")
            except Exception as _e:
                print(f"[visual-memory] skipped ({_e})")
            # MEANS-ENDS ANALYSIS (general): compute the DIFFERENCES between the mover and the scene
            # goal across the typology (position / orientation / size / colour) and surface an ORDERED
            # difference-reduction plan -- preconditions first (reach before transform; orient before
            # grow), each difference tagged with the operator class that reduces it.  Subsumes the
            # positional reach instinct AND the transformational match_goal as difference-reducers.
            try:
                import means_ends as _me
                _md = _me.directive_for_entities(grounded)
                if _md:
                    self._pending_match_note = _md
                    print("[means-ends] mover<->goal differences -> reduction plan surfaced")
            except Exception as _e:
                print(f"[means-ends] skipped ({_e})")
            # CLOSURE / completion instinct (Gestalt closure): a shape with a MOUTH is incomplete and
            # "wants" its complement (opposite mouth + MATCHING attribute) so their union is a solid,
            # gap-free figure.  Surface the win goal -- CLOSE EVERY MOUTH.  Game-agnostic: "purple needs
            # purple" emerges from attribute-matching (measured colour), not a colour rule.
            try:
                import shape_closure as _sc
                _ppt = frame.shape[0] / float(getattr(self, "n_ticks", 64) or 64)
                _cents = []
                for _e in grounded:
                    _bb = _e.get("bbox_ticks_turn1")
                    if not _bb:
                        continue
                    _r0, _c0, _r1, _c1 = _bb
                    _pbb = [int(_r0 * _ppt), int(_c0 * _ppt),
                            int((_r1 + 1) * _ppt) - 1, int((_c1 + 1) * _ppt) - 1]
                    _cents.append({"name": _e.get("name", "?"),
                                   "mask": _sc.mask_from_bbox(frame, _pbb),
                                   "color": _e.get("color") or _sc.dominant_color(frame, _pbb)})
                _cd = _sc.closure_directive(_cents)
                if _cd:
                    self._pending_closure_note = _cd
                    print("[closure] incomplete shape(s) -> close-the-mouths goal surfaced")
                # capture a candidate WIN INSTANCE for the per-game win claim (recorded on solve):
                # a closure win is "one solid block"; colour_uniform is recorded ONLY when the mover and
                # goal differ in colour (the win needs a recolour) -- otherwise it is latent and gets
                # back-filled later (win_claims feature-discovery).
                if _sc.incomplete_entities(_cents):
                    _pairs, _ = _sc.find_completions(_cents)
                    _feat = {"combine": "yes", "result": "solid_block",
                             "shape_relation": ("complementary" if _pairs else "matching")}
                    try:
                        import match_goal as _mg2
                        _mv = _mg2._pick_mover(grounded)
                        _gl = _mg2._pick_goal(grounded, _mv) if _mv else None
                        if _mv and _gl and (_mv.get("color") or "").lower() != (_gl.get("color") or "").lower():
                            _feat["colour_uniform"] = "yes"
                    except Exception:
                        pass
                    self._win_instance_features = _feat
            except Exception as _e:
                print(f"[closure] skipped ({_e})")
            # WIN-PRIOR: recall the per-game win condition generalized from PRIOR solved levels and
            # surface it as a high-credence goal prior -- so the win condition transfers across levels
            # without the actor restating it (win_claims).
            try:
                import win_claims as _wc
                _wnote = _wc.directive(getattr(getattr(self, "world", None), "game_id", "") or "")
                if _wnote:
                    self._pending_win_note = _wnote
                    print("[win-prior] surfaced generalized win condition from prior levels")
            except Exception as _e:
                print(f"[win-prior] skipped ({_e})")
            if persist_dir is not None:
                try:
                    Path(persist_dir).mkdir(parents=True, exist_ok=True)
                    (Path(persist_dir) / "perception_grounded.json").write_text(
                        json.dumps(parsed, indent=2), encoding="utf-8")
                except Exception as e:
                    print(f"[grounding] persist skipped ({e})")
        except Exception as e:
            print(f"[grounding] skipped ({e})")
        return parsed

    def _vlm_inspect_grounding(self, parsed: dict, frame_path,
                               ls_dir, poll_fn=None) -> dict:
        """VLM-GATED QUALITY INSPECTION of the grounded entity analysis.

        Substrate metrics (tightness, overlap, coverage) are NECESSARY but not
        SUFFICIENT -- a mislocated bbox can snap onto the WRONG component and
        still score 'tight'.  The only reliable adjudicator of "is this box on
        the right thing" is a VLM that LOOKS at the annotated overlay.  So after
        grounding we render the boxes-on-the-frame overlay and ask the in-loop
        VLM to visually confirm each box, directed by whatever the substrate QA
        flagged (overlap-conflicts, unverified, low tightness).  Any corrections
        the VLM returns are applied and re-grounded.  This GATES the level-start
        perception on VLM visual approval -- the durable guard against the
        recurring 'entity analysis is off' regression.

        Fully guarded + strict-mode safe: on any error or missing reply it keeps
        the substrate result and the run never halts.
        """
        poll_fn = poll_fn or self._poll_path
        try:
            ents = parsed.get("entities") or []
            if not ents or frame_path is None or not Path(frame_path).exists():
                return parsed
            rep = parsed.get("_grounding_report") or {}
            q = rep.get("quality") or {}
            overlaps = rep.get("overlaps") or []
            unmatched = rep.get("unmatched") or []
            flags = []
            if overlaps:
                flags.append(f"OVERLAP-CONFLICT (distinct entities sharing pixels): "
                             f"{overlaps}")
            if unmatched:
                flags.append(f"UNVERIFIED (box matched no measured component): "
                             f"{unmatched}")
            if q.get("score") is not None and q["score"] < 1.0:
                flags.append(f"tightness score {q['score']} (< 1.0)")
            # Render the boxes-on-frame overlay the VLM will inspect.
            ov = Path(ls_dir) / "inspection_overlay.png"
            try:
                from render_exploratory_run import render_turn1_overlay  # noqa: E402
                render_turn1_overlay(
                    frame_path, ents, n_ticks=self.n_ticks, upscale=self.upscale,
                    index_labels=False,
                ).save(ov)
            except Exception as e:
                print(f"[inspect] overlay render failed ({e}); skipping gate")
                return parsed
            slim = [{"name": e.get("name"), "bbox_ticks_turn1": e.get("bbox_ticks_turn1"),
                     "appearance": e.get("appearance", "")} for e in ents]
            prompt = (
                "# Entity-analysis QUALITY INSPECTION (VLM gate)\n\n"
                "Attached image `inspection_overlay.png`: every entity's grounded "
                "bbox drawn + labelled on THIS level's frame.  Your job is to "
                "VISUALLY VERIFY the geometry the substrate produced -- metrics "
                "alone miss a box that snapped onto the WRONG object.\n\n"
                "For EACH entity, look at its drawn box and decide: does it "
                "TIGHTLY and CORRECTLY enclose the named thing (and nothing "
                "else)?  A box is WRONG if it sits on a different object, covers "
                "empty/background space, is shifted, or is too big/small.\n\n"
                + ("Substrate QA flagged these -- inspect them FIRST:\n  - "
                   + "\n  - ".join(flags) + "\n\n" if flags
                   else "Substrate QA found no metric issue, but confirm visually "
                        "anyway.\n\n")
                + "Current entities (name, bbox [r0,c0,r1,c1] bottom/right "
                "EXCLUSIVE, appearance):\n"
                f"{json.dumps(slim, indent=2)}\n\n"
                "Reply ONLY a JSON object:\n"
                "{\"all_correct\": true|false, \"corrections\": "
                "[{\"name\": \"<entity>\", \"bbox_ticks_turn1\": [r0,c0,r1,c1]}], "
                "\"notes\": \"<one line>\"}\n"
                "Include in `corrections` ONLY the entities whose box you are "
                "changing (give the corrected bbox read from the overlay's grid). "
                "If every box is already correct, set all_correct=true and "
                "corrections=[]."
            )
            (Path(ls_dir) / "inspection_prompt.md").write_text(prompt, encoding="utf-8")
            (Path(ls_dir) / "STATUS.txt").write_text(
                f"WAITING for GROUNDING-INSPECTION reply at "
                f"{Path(ls_dir) / 'inspection_reply.txt'}\n", encoding="utf-8")
            print(f"[inspect] VLM grounding-inspection gate: "
                  f"{len(flags)} substrate flag(s); awaiting VLM verdict")
            reply = poll_fn(Path(ls_dir) / "inspection_reply.txt")
            if not isinstance(reply, dict):
                return parsed
            corr = reply.get("corrections") or []
            if corr:
                by = {e.get("name"): e for e in ents}
                applied = []
                for c in corr:
                    nm, bb = c.get("name"), c.get("bbox_ticks_turn1")
                    if nm in by and bb and len(bb) == 4:
                        by[nm]["bbox_ticks_turn1"] = [int(v) for v in bb]
                        applied.append(nm)
                if applied:
                    print(f"[inspect] VLM corrected {len(applied)} box(es): {applied}; "
                          f"re-grounding")
                    parsed = self._ground_perception_geometry(
                        parsed, frame_path, persist_dir=ls_dir)
            else:
                print("[inspect] VLM confirmed the entity analysis (no corrections)")
        except Exception as e:
            print(f"[inspect] grounding-inspection gate skipped ({e})")
        return parsed

    def stage_initial_perception(self, frame_path: Path) -> Path:
        turn_dir = self.work_dir / "turn_001"
        turn_dir.mkdir(parents=True, exist_ok=True)
        grid_img = self._gridded_for(
            frame_path, turn_dir / "image_grid.png",
        )
        sys_text, user_text = _fmt_prompts(
            self.n_ticks, label_stride=self.label_stride,
            single_frame=True, game_id=self.game.game_id,
        )
        prompt = INITIAL_PERCEPTION_PROMPT.format(
            game_id=self.game.game_id, level=self.game.level,
            sys_text=sys_text, user_text=user_text,
            n_ticks=self.n_ticks, label_stride=self.label_stride,
            reply_name="reply.txt",
        )
        prompt_path = turn_dir / "prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        (turn_dir / "STATUS.txt").write_text(
            f"WAITING for initial perception reply at {turn_dir/'reply.txt'}\n",
            encoding="utf-8",
        )
        return prompt_path

    def stage_refinement(self, frame_path: Path,
                          prev_entities: list[dict]) -> Path:
        """Stage a SECOND-PASS bbox refinement.  Writes a refinement
        prompt that shows the VLM (a) the original gridded frame and
        (b) the same frame with its FIRST-PASS bboxes drawn as cyan
        #N rectangles, asking it to correct any misaligned bboxes.

        This is critical when the source frame is small / dense
        relative to entity scale: the VLM often gets the bbox edges
        +/- 1-2 ticks off on the first pass, and the second pass
        with self-overlay-as-feedback usually fixes most of them.
        """
        turn_dir = self.work_dir / "turn_001"
        turn_dir.mkdir(parents=True, exist_ok=True)
        # Image 1: the original gridded frame (same as initial pass)
        # is already at turn_001/image_grid.png; do not overwrite it.
        # Image 2: overlay of prev bboxes on the same gridded frame,
        # rendered with the driver's current grid styling so the
        # refinement input visually matches the initial input.
        overlay_img = render_turn1_overlay(
            frame_path, prev_entities, n_ticks=self.n_ticks,
            upscale=self.upscale,
            bbox_line_width=2,
            grid_line_width_major=self.grid_line_width_major,
            grid_line_width_minor=self.grid_line_width_minor,
            grid_major_alpha=self.grid_major_alpha,
            grid_minor_alpha=self.grid_minor_alpha,
        )
        overlay_path = turn_dir / "refinement_overlay.png"
        overlay_img.save(overlay_path)

        sys_text, _ = _fmt_prompts(
            self.n_ticks, label_stride=self.label_stride,
            single_frame=True, game_id=self.game.game_id,
        )
        ref_text = REFINEMENT_PROMPT_TEMPLATE.format(n_ticks=self.n_ticks)
        prompt = REFINEMENT_PERCEPTION_PROMPT.format(
            game_id=self.game.game_id, level=self.game.level,
            sys_text=sys_text, ref_text=ref_text,
            n_ticks=self.n_ticks,
            reply_name="refinement_reply.txt",
        )
        prompt_path = turn_dir / "refinement_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        (turn_dir / "STATUS.txt").write_text(
            f"WAITING for refinement reply at "
            f"{turn_dir / 'refinement_reply.txt'}\n",
            encoding="utf-8",
        )
        return prompt_path

    def poll_refinement_reply(self) -> dict:
        """Poll for the turn-1 refinement reply.  Same parsing as
        poll_reply but uses the refinement filename so it doesn't
        collide with the initial-perception reply."""
        turn_dir = self.work_dir / "turn_001"
        reply_path = turn_dir / "refinement_reply.txt"
        deadline = time.time() + self.vlm_timeout_s
        print(f"[exploratory-driver] waiting for refinement reply at "
              f"{reply_path}", flush=True)
        while time.time() < deadline:
            if reply_path.exists():
                body = reply_path.read_text(encoding="utf-8").strip()
                if body:
                    # Use the shared hardened parser: it degrades (returns None)
                    # on an unparsable reply instead of raising, so a malformed
                    # autonomous-VLM refinement does not crash the run.
                    parsed = self._consume_reply(reply_path, body)
                    if parsed is None:
                        return self._fallback_perception(1)
                    return self._fulfill_visual_queries(
                        parsed, turn_dir, self._last_frame_path, "refinement")
            time.sleep(self.poll_s)
        # VLM did not refine within the timeout — keep the substrate-only
        # perception and continue (strict-mode robustness; no crash).
        return self._fallback_perception(1)

    @staticmethod
    def _classify_object_deltas(prev, curr, cen, npix, colour):
        """Object-constancy across one step.  ``prev``/``curr`` are entity lists;
        ``cen``/``npix``/``colour`` read an entity's centroid (r,c) / pixel count /
        dominant colour.  Returns (moved, recoloured, appeared, vanished) where
        moved & recoloured are (prev,curr) pairs and appeared/vanished are entities.

        SAME-colour pairs match at any distance (track a moved object by
        appearance).  CROSS-colour pairs match ONLY when co-located (centroid
        d<=2): an entity that changed colour WITHOUT moving is a RECOLOR IN PLACE
        = a state change (e.g. an active/selected toggle), not a vanish+appear or
        a phantom swap.  Nearest-first assignment; at EQUAL distance a same-colour
        match wins, so static / cursor-ring cases are untouched and a recolor only
        wins when nothing same-colour is co-located.  Pure + game-agnostic."""
        pairs = []
        for ci, c in enumerate(curr):
            for pi, p in enumerate(prev):
                if not (0.4 <= (npix(p) + 1) / (npix(c) + 1) <= 2.5):
                    continue
                d = abs(cen(c)[0] - cen(p)[0]) + abs(cen(c)[1] - cen(p)[1])
                same = colour(p) == colour(c)
                if same or d <= 2:
                    pairs.append((d, 0 if same else 1, ci, pi))
        pairs.sort(key=lambda t: (t[0], t[1]))
        m_c, m_p, moved, recoloured = {}, set(), [], []
        for d, cdiff, ci, pi in pairs:
            if ci in m_c or pi in m_p:
                continue
            m_c[ci] = (pi, d, cdiff)
            m_p.add(pi)
        for ci, (pi, d, cdiff) in m_c.items():
            if cdiff:                           # same place, different colour
                recoloured.append((prev[pi], curr[ci]))
            elif d > 2:                         # <=2 cells of centroid jitter = same place
                moved.append((prev[pi], curr[ci]))
        appeared = [c for ci, c in enumerate(curr) if ci not in m_c]
        vanished = [p for pi, p in enumerate(prev) if pi not in m_p]
        return moved, recoloured, appeared, vanished

    def _substrate_response_facts(self) -> str:
        """Measured RESPONSE-ASYMMETRY facts over the delta history (pixel stage ->
        symbolic stage feed): which entities are SETTABLE (own click changed them
        in place, no animation) vs TRIGGERS (own click produced an animation).
        Purely factual (keys on the response to an action, not colour/position);
        the symbolic layer types program/scene from it, the VLM interprets.  Also
        persisted on self.world.response_facts for that symbolic layer.  Guarded."""
        try:
            import response_asymmetry as _ra
            bboxes = {n: r.current_bbox
                      for n, r in (getattr(self.world, "entities", {}) or {}).items()
                      if getattr(r, "current_bbox", None) is not None}
            if not bboxes:
                return ""
            facts = _ra.classify_responses(
                getattr(self.world, "deltas_observed", []) or [], bboxes)
            try:
                self.world.response_facts = facts      # for the symbolic layer
            except Exception:
                pass
            return _ra.response_narration(facts)
        except Exception as e:
            print(f"[response-facts] skipped ({e})")
            return ""

    def _substrate_entity_deltas(self, prev_frame: Path, curr_frame: Path) -> str:
        """Object-constancy tracker over the raw prev/curr frames: detect
        entities in both, match them across the step, and report each as
        MOVED / CHANGED-COLOUR-IN-PLACE (a state change, e.g. a select/active
        toggle) / APPEARED / VANISHED with its A->B position.  Hands the VLM
        "entity X moved A->B" as a substrate
        FACT so an action's effect is read as a logical-entity change, not a
        pixel diff.  Pure substrate CV (works for any in-loop VLM, zero model
        cost); degrades to a short note on any error."""
        try:
            import os, sys
            import numpy as np
            from PIL import Image
            # entity_detector uses package-relative imports; ensure the parent
            # of perception_loop_v2 is importable so the package form resolves.
            _gov = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if _gov not in sys.path:
                sys.path.insert(0, _gov)
            from perception_loop_v2.entity_detector import detect_entities
            from perception_loop_v2.observation import build_frame_observation

            def ents(path):
                im = Image.open(path).convert("RGB")
                if im.size != (64, 64):
                    im = im.resize((64, 64), Image.NEAREST)
                obs = build_frame_observation(np.array(im), turn=0, rows=64,
                                              cols=64, agent_position=None)
                return [e for e in detect_entities(obs)
                        if getattr(e, "bitmap", None) is not None
                        and not getattr(e, "is_background_primary", False)
                        and not getattr(e, "is_background_secondary", False)]

            cen = lambda e: tuple(e.centroid_cell)
            bb = lambda e: tuple(e.bbox_logical)
            def dim(e):
                r0, c0, r1, c1 = bb(e)
                return f"{r1 - r0}x{c1 - c0}"
            def npix(e):
                m = np.asarray(e.bitmap)
                return int((m != -1).sum())
            def colour(e):
                m = np.asarray(e.bitmap)
                v = m[m != -1]
                if not v.size:
                    return -1
                # np.unique is bounded by the number of DISTINCT values; the old
                # np.bincount(v) allocated an array of size max(v)+1 — for packed
                # RGB (values up to 0xFFFFFF) that is a ~16M-element array PER
                # call, which (called O(n^2) times in the matcher) made a single
                # 40-entity step take ~115s.
                vals, counts = np.unique(v, return_counts=True)
                return int(vals[counts.argmax()])

            # Match by DOMINANT COLOUR + size + nearest centroid (coarse object
            # constancy).  Colour cleanly separates the movable agents from the
            # ball / cursor / guides, and is immune to a white cursor-ring
            # overlapping a coloured square.  Drop sub-5px specks (dotted-line
            # guides) so they don't flood the deltas.
            big = lambda L: [e for e in L if npix(e) >= 5
                             and (bb(e)[2] - bb(e)[0]) < 40
                             and (bb(e)[3] - bb(e)[1]) < 40]
            _T = os.environ.get("COS_TIMING")
            _ta = time.time()
            _ep = ents(prev_frame)
            _tb = time.time()
            _ec = ents(curr_frame)
            _tc = time.time()
            prev, curr = big(_ep), big(_ec)
            _td = time.time()
            if _T:
                print(f"[timing]   ents(prev)={_tb-_ta:.1f}s ents(curr)={_tc-_tb:.1f}s "
                      f"big={_td-_tc:.1f}s n_prev={len(prev)} n_curr={len(curr)}",
                      file=sys.stderr, flush=True)
            # Object constancy across frames = OPTIMAL (nearest-first) assignment,
            # NOT arbitrary-order greedy.  Build every same-colour, size-compatible
            # (prev,curr) pair, sort by centroid distance, and assign the smallest
            # distances first.  This makes STATIC entities self-match at distance 0
            # BEFORE any neighbour can be mis-claimed -- which is exactly what stops
            # a maze / tile lattice from dissolving into a cascade of phantom
            # "moves" (the earlier greedy loop paired a shifted neighbour whenever a
            # few cells changed).  No instance-count threshold: a repeated structure
            # stays put at d=0, and only the genuinely changed cells surface (an
            # agent that hops one cell shows up as the one APPEARED/MOVED that it is).
            # Object constancy across frames: nearest-first matching with
            # recolor-in-place detection (a colour change at the same spot is a
            # STATE change, not a move/appear/vanish).  See _classify_object_deltas.
            _tcl = time.time()
            # Precompute per-entity metrics ONCE.  The matcher calls cen/npix/
            # colour O(n^2) times in its inner loop; recomputing them from the
            # bitmap each time (esp. colour()) made a 40-entity step take ~115s.
            # Cache by id() — prev/curr hold the same entity objects throughout.
            _np_c = {id(e): npix(e) for e in prev + curr}
            _cn_c = {id(e): cen(e) for e in prev + curr}
            _co_c = {id(e): colour(e) for e in prev + curr}
            moved, recoloured, appeared, vanished = \
                ExploratoryDriver._classify_object_deltas(
                    prev, curr,
                    lambda e: _cn_c[id(e)],
                    lambda e: _np_c[id(e)],
                    lambda e: _co_c[id(e)])
            if os.environ.get("COS_TIMING"):
                print(f"[timing]   classify_object_deltas={time.time()-_tcl:.1f}s",
                      file=sys.stderr, flush=True)

            # A whole-scene re-render (level start, reset, menu) changes dozens of
            # entities at once; listing each is noise that floods the prompt and
            # buries any incremental move.  When the change count exceeds a small
            # PROMPT-DISPLAY budget, summarise it as the scene transition it is and
            # tell the VLM to re-read the scene rather than track per-object deltas.
            # (This is a readability cap on the report, not a behavioural threshold:
            # it changes nothing about the matching, only how a large diff is shown.)
            display_budget = 12
            if len(moved) + len(recoloured) + len(appeared) + len(vanished) > display_budget:
                return (f"(tracker: the scene was SUBSTANTIALLY REDRAWN this step — "
                        f"{len(moved)} object(s) moved, {len(appeared)} appeared, "
                        f"{len(vanished)} vanished.  This is a full re-render / "
                        f"scene transition (level change, reset, or menu), not an "
                        f"incremental move — re-read the new scene rather than "
                        f"tracking individual deltas.)")

            lines = []
            for p, c in moved:
                lines.append(f"- a {dim(c)} object MOVED  centroid {cen(p)} -> "
                             f"{cen(c)}  (bbox {bb(p)} -> {bb(c)})")
            for p, c in recoloured:
                lines.append(f"- a {dim(c)} object at centroid {cen(c)} CHANGED "
                             f"COLOUR #{colour(p) & 0xFFFFFF:06x} -> "
                             f"#{colour(c) & 0xFFFFFF:06x} in place (did NOT move; "
                             f"same location + size = the same object in a new "
                             f"state, e.g. a select/active toggle)")
            for c in appeared:
                lines.append(f"- a {dim(c)} object APPEARED at centroid "
                             f"{cen(c)} (bbox {bb(c)})")
            for p in vanished:
                lines.append(f"- a {dim(p)} object VANISHED from centroid "
                             f"{cen(p)} (was bbox {bb(p)})")
            if not lines:
                return ("(tracker: no object moved / appeared / vanished between "
                        "the two frames — the action produced no entity-level "
                        "change.)")
            return "\n".join(lines)
        except Exception as e:
            return f"(substrate entity-delta tracker unavailable: {e})"

    def _substrate_animation_summary(self, frame_stack) -> str:
        """Narrate the MOTION across an action's animation framestack so the VLM
        reads the effect's DYNAMICS (how things moved / grew / spread / flowed),
        not just the settled end state.  Delegates to the shared, unit-testable
        animation_analysis module (one implementation for live play + tests).
        Palette-invariant; guarded."""
        import animation_analysis as _aa
        return _aa.animation_summary(frame_stack)

    def _substrate_animation_entities(self, anim_dir):
        """ENTITY-LEVEL movement events ACROSS an action's animation (object
        constancy across sub-frames -> appeared/moved/grew/shrank/recoloured/
        vanished, each with trajectory + net + span).  Delegates to the shared,
        unit-testable animation_analysis module (one implementation for live play
        + tests).  Returns (events, narration).  Palette-invariant; guarded."""
        import animation_analysis as _aa
        return _aa.animation_entities(_aa.load_frames(anim_dir))

    def _render_animation_filmstrip(self, anim_dir, out_path):
        """Render the action's saved animation sub-frames as ONE gridded,
        frame-labelled montage the VLM can VISUALLY INSPECT.  The colour-region
        text summary is a substrate hint; this filmstrip lets the VLM read the
        actual frames itself — extract entities and infer their motion (object
        constancy across frames is the mechanic).  Pure rendering, game-agnostic,
        guarded.  Returns out_path, or None when there is no animation (<2
        frames) or rendering fails (the run never depends on it)."""
        try:
            from PIL import ImageDraw
            from substrate_tools.frameutils import font as _ft_font
            if anim_dir is None:
                return None
            files = sorted(Path(anim_dir).glob("frame_*.png"))
            if len(files) < 2:
                return None
            n = len(files)
            cols = min(n, 4)
            rows = (n + cols - 1) // cols
            # per-tile upscale so the whole montage stays ~<=1500px wide (under
            # the vision-API resample cap) yet each frame is still readable.
            tile_target = max(180, 1500 // cols)
            up = max(3, int(round((tile_target - 56) / (self.n_ticks * 1.24))))
            up = min(up, self.upscale)
            tiles = []
            for f in files:
                img = Image.open(f).convert("RGB")
                gridded, _, _ = _add_grid_overlay(
                    img, n_ticks=self.n_ticks, upscale=up,
                    label_stride=self.label_stride,
                    line_width_major=self.grid_line_width_major,
                    line_width_minor=self.grid_line_width_minor,
                    major_alpha=self.grid_major_alpha,
                    minor_alpha=self.grid_minor_alpha,
                )
                tiles.append(gridded)
            tw, th = tiles[0].size
            lab, gap = 22, 8
            W = cols * tw + (cols + 1) * gap
            H = rows * (th + lab) + (rows + 1) * gap
            canvas = Image.new("RGB", (W, H), (18, 18, 24))
            d = ImageDraw.Draw(canvas)
            fnt = _ft_font(16)
            for i, tile in enumerate(tiles):
                r, c = divmod(i, cols)
                x = gap + c * (tw + gap)
                y = gap + r * (th + lab + gap)
                tag = " (settled)" if i == n - 1 else ""
                d.text((x + 2, y + 2), f"frame {i}{tag}", fill=(240, 240, 140),
                       font=fnt)
                canvas.paste(tile, (x, y + lab))
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            canvas.save(out_path)
            return out_path
        except Exception as e:
            print(f"[animation] filmstrip render failed ({e}); skipping",
                  flush=True)
            return None

    def _render_entity_analysis_filmstrip(self, anim_dir, out_path, known_entities=None):
        """Render each animation sub-frame with the substrate's PER-FRAME ENTITY
        ANALYSIS drawn on it (the verification view: exactly what the extractor
        found per frame, scene cuts marked).  Delegates to the shared
        animation_analysis module.  Returns out_path or None."""
        import animation_analysis as _aa
        return _aa.render_entity_analysis_filmstrip(
            _aa.load_frames(anim_dir), out_path, known_entities)

    def _substrate_silhouette_track(self, anim_dir):
        """Track a MOVING SILHOUETTE by novelty (diff vs the static first frame)
        and identify it with a known static entity by scale-normalised SHAPE.
        Delegates to the shared animation_analysis module, passing this world's
        entity bboxes as the known shapes.  Returns identified movers ([] on
        miss).  Guarded."""
        import animation_analysis as _aa
        known = {n: r.current_bbox
                 for n, r in (getattr(self.world, "entities", {}) or {}).items()
                 if getattr(r, "current_bbox", None) is not None}
        return _aa.silhouette_movers(_aa.load_frames(anim_dir), known)

    def _substrate_scene_cuts(self, anim_dir):
        """Frame indices where the animation CHANGES VIEW (a majority-of-frame
        repaint: zoom / overlay / different screen).  Motion analysis is
        unreliable across such a cut, so they must be surfaced.  Delegates to the
        shared animation_analysis module.  Returns cut indices ([] none)."""
        import animation_analysis as _aa
        return _aa.scene_cuts(_aa.load_frames(anim_dir))

    def _demonstration_narration(self, anim_events, last_action, silhouettes=None) -> str:
        """Recognise a DEMONSTRATION/PREVIEW from the animation (an entity that
        traces a path and RETURNS, previewing a motion the settled frame hides).
        Delegates to the shared animation_analysis module; records demos on
        self._demonstrations for later use.  Returns the narration ('' when
        none).  Guarded."""
        import animation_analysis as _aa
        store = getattr(self, "_demonstrations", None)
        if store is None:
            store = self._demonstrations = {}
        narr = _aa.demonstration_narration(
            anim_events, last_action, silhouettes, store=store)
        if narr:
            # Record the action that PRODUCED this demonstration so it is NOT
            # mistaken for the TRIGGER later: a legend/preview click animates a
            # mover-PREVIEW, which would otherwise be picked as "the most recent
            # click that animated" by cx-OFAT (the tn36 lc2 mis-fire).
            try:
                srcs = getattr(self, "_demo_source_actions", None)
                if srcs is None:
                    srcs = self._demo_source_actions = set()
                if last_action:
                    srcs.add(last_action)
            except Exception:
                pass
            # A preview was recognised this turn -> LEARN the mechanic from it:
            # seed the previewed goal + bind the demonstrated magnitude to the
            # structure that parameterises it.  Grounded in the observation, so
            # it outranks guessed priors and drives the autonomous solve.
            try:
                self._synthesize_demonstrations()
            except Exception as e:
                print(f"[demo-synth] skipped ({e})")
        return narr

    def _structure_column_counts(self) -> dict:
        """For each panel-scale STRUCTURE entity, its column count -- by
        decomposing it into its lattice on the clean level-start frame.  Used to
        bind a demonstrated travel-magnitude to the structure whose unit count
        equals it.  Guarded -> {}."""
        out: dict = {}
        try:
            import structural_grid
            frame = (getattr(self, "_level_start_frame_path", None)
                     or getattr(self, "_last_frame_path", None))
            for name, rec in (getattr(self.world, "entities", {}) or {}).items():
                bb = getattr(rec, "current_bbox", None)
                if not bb or not self._is_panel_scale(bb):
                    continue
                d = structural_grid.read_path(frame, list(bb))
                if d and d.get("n_cols"):
                    out[name] = int(d["n_cols"])
        except Exception:
            pass
        return out

    def _synthesize_demonstrations(self) -> None:
        """Turn the extracted DEMONSTRATIONS into mechanic knowledge: a previewed
        WIN (the mover should reach the demonstrated displacement) + a demo-
        GROUNDED parameterisation claim (the structure whose unit count == the
        demonstrated magnitude controls that motion).  Game-agnostic; see
        demonstration_synthesis.  Guarded -- never breaks the turn."""
        import demonstration_synthesis as _ds
        from world_knowledge import WinConditionHypothesis
        demos = [x for v in (getattr(self, "_demonstrations", {}) or {}).values()
                 for x in (v or [])]
        if not demos:
            return
        gi = getattr(self.world, "grid_inference", None)
        cell_ticks = int(getattr(gi, "cell_ticks", 0) or 0) or 4
        res = _ds.synthesize(demos, self._structure_column_counts(),
                             cell_ticks=cell_ticks)
        turn = int(getattr(self.world, "turn", 0) or 0)
        wch = self.world.win_condition_hypotheses
        existing = {getattr(h, "description", "") for h in (wch or [])}
        for w in res.get("win", []):
            if w["description"] in existing:
                continue
            wch.append(WinConditionHypothesis(
                hypothesis_id=f"demo_win_{len(wch)}", description=w["description"],
                credence=float(w["credence"]), created_at_turn=turn,
                notes="grounded in a demonstration/preview",
                win_relation=w.get("win_relation")))
            existing.add(w["description"])
        cs = getattr(self, "_claim_store", None)
        if cs is not None and res.get("claims"):
            try:
                cs.ingest(res["claims"], turn=turn,
                          level_signature=self._level_signature())
            except Exception:
                pass
        if res.get("win") or res.get("claims"):
            print(f"[demo-synth] from {len(demos)} demonstration(s): "
                  f"+{len(res.get('win', []))} win hyp, "
                  f"+{len(res.get('claims', []))} grounded per-step claim(s).")

    def _substrate_salient_correlation(self, anim_events, silhouettes,
                                       last_action) -> str:
        """Bind the two most-salient CO-OCCURRING animation events (the marker's
        motion + the control that highlights with it) with a CORRELATION claim.
        The substrate MEASURES the co-occurrence (a fact); delegates that to
        animation_analysis.  It surfaces a narration for the VLM to author the
        binding AND auto-files a low-credence (GUESSED) correlation claim into the
        Claim Store -- so even unprompted, the prober TESTS it (reproduce the
        control pattern, check the marker + score follow).  The causal binding is
        a hypothesis to verify, never asserted.  Returns the narration ('' if no
        salient co-occurrence).  Fully guarded."""
        import animation_analysis as _aa
        try:
            known = {n: r.current_bbox
                     for n, r in (getattr(self.world, "entities", {}) or {}).items()
                     if getattr(r, "current_bbox", None) is not None}
            cooc = _aa.salient_cooccurrence(anim_events, silhouettes, known)
        except Exception:
            return ""
        if not cooc:
            return ""
        cs = getattr(self, "_claim_store", None)
        claim = _aa.correlation_claim(cooc, str(last_action))
        if cs is not None and claim is not None:
            try:
                is_new = claim["id"] not in cs.claims
                cs.ingest([claim], turn=int(getattr(self.world, "turn", 0) or 0),
                          level_signature=self._level_signature())
                cs.save_for_game()
                if is_new:
                    print(f"[correlation] filed claim {claim['id']} "
                          f"(credence {claim['credence']:.2f}, to verify)")
                # STRUCTURE MAPPING: immediately transfer this freshly-filed
                # correlation across any similar-structure mapping (e.g. the LEFT
                # reference panel it was grounded to -> the RIGHT control panel),
                # so the actionable analogue ('set the right switches to the shown
                # pattern') is filed for the prober.
                self._map_claims_across_structures()
            except Exception as e:
                print(f"[correlation] claim ingest skipped ({e})")
        return _aa.correlation_narration(cooc)

    def _substrate_frame_correlation(self, frame_stack) -> str:
        """PER-FRAME co-occurrence: regions that change in the SAME animation frame are
        likely RELATED (a designer's temporal binding -- the finer complement to the
        across-frames salient version; generalises to any synchronized-observation domain,
        e.g. robotics).  This is how a demonstration ties an agent's step to the CODE that
        drives it: when the mover steps AND one switch column highlights in the same frame,
        bind them.  Surfaces the co-occurring regions (scene vs panel + x) + auto-files a
        GUESSED correlation claim per pair so the prober tests it.  Fully guarded."""
        try:
            import frame_correlation as _fc
            cooc = _fc.frame_cooccurrence(frame_stack, min_px=3)
        except Exception:
            return ""
        if not cooc:
            return ""

        def _lab(c):
            cx, cy = int(c["center"][0]), c["center"][1]
            return f"scene-mover@x{cx}" if cy < 32 else f"panel-col@x{cx}"

        try:
            pairs = _fc.correlation_pairs(cooc, _lab)
        except Exception:
            return ""
        pairs = [p for p in pairs if p["a"] != p["b"]]
        if not pairs:
            return ""
        cs = getattr(self, "_claim_store", None)
        filed = 0
        for p in pairs:
            claim = {"id": f"framecorr::{p['a']}~{p['b']}",
                     "statement": (f"FRAME CO-OCCURRENCE: '{p['a']}' and '{p['b']}' changed in the "
                                   f"SAME animation frame (frame {p['frame']}) -> likely RELATED; "
                                   f"TEST by reproducing one and checking the other."),
                     "kind": "correlation", "scope": "level", "target": [p["a"], p["b"]],
                     "plan": f"reproduce '{p['b']}' (set the panel pattern) and check '{p['a']}' + score",
                     "provenance": "guessed", "credence": 0.35}
            if cs is not None:
                try:
                    cs.ingest([claim], turn=int(getattr(self.world, "turn", 0) or 0),
                              level_signature=self._level_signature())
                    filed += 1
                except Exception:
                    pass
        if cs is not None and filed:
            try:
                cs.save_for_game()
            except Exception:
                pass
            print(f"[frame-corr] filed {filed} per-frame co-occurrence claim(s) to verify")
        lines = "\n".join(f"  frame {p['frame']}: {p['a']} <-> {p['b']} changed together"
                          for p in pairs[:8])
        return ("[SUBSTRATE] FRAME CO-OCCURRENCE — MEASURED FACT: regions that change in the SAME "
                "animation frame are likely RELATED (bind them; this is how a demonstration ties an "
                "agent's move to the CODE driving it -- read the SCENE mover's move, NOT the panel "
                "sweep's direction):\n" + lines)

    def _substrate_agent_move_guard(self, prev_frame: Path, curr_frame: Path) -> str:
        """GROUND-TRUTH GUARD (the picture arbitrates over symbolic claims): measure,
        from the raw frames, whether the agent's silhouette LEFT its pre-action cell.
        Stashes the measurement on ``self._pending_gt_agent_move`` so the post-reply
        DeltaRecord reconcile can CORRECT a wrong ``agent_moved=false`` (the tn36 lc5
        failure: a fire moved the mover several cells but the symbolic flag said it
        hadn't, and a successful action was called a no-op).  Returns a loud note to
        surface in the delta prompt when the agent moved, else ''.  Fully guarded."""
        self._pending_gt_agent_move = None
        try:
            import numpy as _np
            from PIL import Image as _Im
            import perception_ground_truth as _pgt
            gi = getattr(self.world, "grid_inference", None)
            agent = self.world._find_agent() if hasattr(self.world, "_find_agent") else None
            if gi is None or agent is None or agent.current_cell is None:
                return ""
            ct = int(getattr(gi, "cell_ticks", 0) or 0)
            org = getattr(gi, "origin_ticks", None)
            if not ct or not org:
                return ""
            r, c = int(agent.current_cell[0]), int(agent.current_cell[1])
            bbox = [org[0] + r * ct, org[1] + c * ct,
                    org[0] + (r + 1) * ct, org[1] + (c + 1) * ct]
            prev = _np.array(_Im.open(prev_frame).convert("RGB"))
            curr = _np.array(_Im.open(curr_frame).convert("RGB"))
            m = _pgt.measure_agent_move(prev, curr, bbox)
            if not m or not m.get("moved"):
                return ""
            self._pending_gt_agent_move = {"m": m, "origin": tuple(org),
                                           "ct": ct, "prev_cell": [r, c]}
            dest = ""
            nc = m.get("new_centroid_tick")
            if nc is not None:
                dr, dc = int((nc[0] - org[0]) // ct), int((nc[1] - org[1]) // ct)
                dest = f" and a matching silhouette is now near cell [{dr}, {dc}]"
            return (f"[SUBSTRATE GROUND-TRUTH — MEASURED] The agent '{agent.name}' has "
                    f"VACATED its cell [{r}, {c}] (only {m['retained_fraction']:.0%} of "
                    f"its silhouette remains there){dest}. It MOVED -- this action HAD an "
                    f"effect. Do NOT report agent_moved=false / 'no-op' / 'canned' / "
                    f"'stuck'; the PICTURE is ground truth. Read the frame and report the move.")
        except Exception as _e:
            print(f"[ground-truth] agent-move guard skipped ({_e})")
            return ""

    def _reconcile_agent_move(self, delta) -> None:
        """Apply the ground-truth guard's measurement to a freshly-built DeltaRecord:
        if the substrate measured the agent moving but the VLM reported it did NOT,
        CORRECT the record (picture wins) + log loudly + surface the mismatch next
        turn.  Asymmetric: never flips a reported move to a non-move.  Guarded."""
        try:
            import perception_ground_truth as _pgt
            gt = getattr(self, "_pending_gt_agent_move", None)
            if not gt or not gt.get("m"):
                return
            rec = _pgt.reconcile(delta.agent_moved, delta.agent_new_cell,
                                 gt["m"], gt["origin"], gt["ct"])
            if rec.get("corrected"):
                delta.agent_moved = True
                if rec.get("new_cell") is not None:
                    delta.agent_new_cell = list(rec["new_cell"])
                self._pending_gt_note = rec.get("note")
                print("[ground-truth] CORRECTED agent_moved false->true "
                      f"(picture wins): agent left cell {gt['prev_cell']}"
                      + (f" -> {rec['new_cell']}" if rec.get("new_cell") else ""))
        except Exception as _e:
            print(f"[ground-truth] reconcile skipped ({_e})")
        finally:
            self._pending_gt_agent_move = None

    def stage_delta_perception(self, prev_frame: Path, curr_frame: Path,
                                turn_n: int, last_action: str,
                                frame_stack=None, anim_dir=None) -> Path:
        turn_dir = self.work_dir / f"turn_{turn_n:03d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        _T = os.environ.get("COS_TIMING")
        _t0 = time.time()
        self._gridded_for(prev_frame, turn_dir / "prev_frame.png")
        self._gridded_for(curr_frame, turn_dir / "curr_frame.png")
        _t1 = time.time()
        sys_text, _ = _fmt_prompts(
            self.n_ticks, label_stride=self.label_stride,
            single_frame=True, game_id=self.game.game_id,
        )
        snapshot = self.world.symbolic_snapshot()
        _t2 = time.time()
        _entity_deltas = self._substrate_entity_deltas(prev_frame, curr_frame)
        # GROUND-TRUTH GUARD: measure whether the agent's silhouette LEFT its cell and
        # surface it LOUD (so a real move is never read as a no-op); the post-reply
        # reconcile then auto-corrects a wrong agent_moved=false.  Prepended so it leads.
        _gt_guard = self._substrate_agent_move_guard(prev_frame, curr_frame)
        if _gt_guard:
            _entity_deltas = (_gt_guard + "\n" + (_entity_deltas or "")).strip()
        # MEASURED RESPONSE-ASYMMETRY FACTS (pixel stage -> symbolic stage feed):
        # which entities are SETTABLE (own click changed them in place) vs TRIGGERS
        # (own click animated) -- the click-response asymmetry that, downstream,
        # lets the symbolic layer type sections as program vs scene and tell an
        # active control from a fixed reference. Neutral fact; the VLM interprets.
        _response_facts = self._substrate_response_facts()
        if _response_facts:
            _entity_deltas = ((_entity_deltas or "") + "\n" + _response_facts).strip()
        _t3 = time.time()
        _animation_summary = self._substrate_animation_summary(frame_stack)
        # ENTITY-LEVEL inter-frame motion (Layer-3/4): detect entities per
        # sub-frame and track them across the animation -> movement FACTS.  This
        # is the primary animation view; the colour-region summary is complementary.
        _anim_events, _animation_entities = self._substrate_animation_entities(anim_dir)
        self._last_anim_events = _anim_events
        # DEMONSTRATION / PREVIEW recognition (game-agnostic): if the animation
        # shows an entity tracing a path and RETURNING to ~its start (so the
        # settled frame is unchanged), the action did not commit a change AND is
        # not inert -- it DEMONSTRATED/PREVIEWED a motion (e.g. a legend/preview
        # control showing what it would do).  Surface it so the acting VLM can
        # MATCH the previewed motion to the transformation its goal needs (means-
        # ends), instead of dismissing the control as functionless.
        _silhouettes = self._substrate_silhouette_track(anim_dir)
        _demo_block = self._demonstration_narration(_anim_events, last_action, _silhouettes)
        if _demo_block:
            _animation_summary = (_demo_block + "\n" + (_animation_summary or "")).strip()
        # SALIENT CO-OCCURRENCE -> correlation claim (game-agnostic): two salient
        # events that change TOGETHER across multiple sub-frames (e.g. the marker
        # moving up WHILE the switches that drive it highlight) are a designer's
        # clue that one is bound to the other.  Surface the co-occurrence FACT +
        # prompt the VLM to bind them, AND auto-file a low-credence (GUESSED)
        # correlation claim so the prober TESTS it (reproduce B; check A + score).
        _corr_block = self._substrate_salient_correlation(
            _anim_events, _silhouettes, last_action)
        if _corr_block:
            _animation_summary = ((_animation_summary or "") + "\n" + _corr_block).strip()
        # PER-FRAME co-occurrence: regions changing in the SAME frame are likely related.
        _frame_corr = self._substrate_frame_correlation(frame_stack)
        if _frame_corr:
            _animation_summary = ((_animation_summary or "") + "\n" + _frame_corr).strip()
        # SETTLE-THEN-READ (game-agnostic): an action that animates SETTLES into a new
        # state.  The ground truth is the LAST framestack frame; if it differs from the
        # PREVIOUS settled frame, a PERSISTENT state change occurred (a selection / toggle
        # / commit) -- the VLM must READ the settled state, not dismiss the animation as a
        # transient.  Flag it loudly + remember this settled frame for the next turn.
        try:
            import settle as _settle
            _curr_settled = _settle.settled_frame(frame_stack)
            _trans = _settle.classify_transition(getattr(self, "_prev_settled", None), _curr_settled)
            _snote = _settle.settle_note(_trans) or _settle.settle_note(
                _settle.classify_animation(frame_stack))
            if _snote:
                _animation_summary = ("[SETTLE] " + _snote + "\n" + (_animation_summary or "")).strip()
            if _trans.get("kind") == "state_change":
                print(f"[settle] PERSISTENT state change vs prev settled: "
                      f"{_trans['settled_change_bbox']} ({_trans['n_settled_changed']} cells)")
            if _curr_settled is not None:
                self._prev_settled = _curr_settled
        except Exception as _e:
            print(f"[settle] skipped ({_e})")
        # DETAILED-INSPECTION TRIGGER (substrate -> VLM): the substrate just did the cheap EXHAUSTIVE
        # pass over the framestack (every colour, every frame, whole frame).  If it found a
        # size/appearance change a single-colour centroid read would miss -- an entity growing/shrinking
        # or a colour appearing (e.g. a grey preview silhouette demonstrating an action) -- pull the VLM
        # in to LOOK at the raw frames rather than let a lazy summary stand.  Registered past misreads
        # of animations/demonstrations escalate it.  This is the balance: substrate measures
        # exhaustively, the VLM is triggered only for the flagged cases.
        try:
            import os as _os2
            import animation_inspection as _ainsp
            from error_ledger import ErrorLedger as _EL2
            _root2 = _os2.environ.get("COS_KB_ROOT") or str(
                Path(__file__).resolve().parents[3] / ".tmp/kb")
            _idir = _ainsp.inspection_directive(
                frame_stack, _EL2(Path(_root2) / "error_ledger.json"))
            if _idir:
                _animation_summary = (_idir + "\n" + (_animation_summary or "")).strip()
                print("[anim-inspect] flagged size/appearance change -> VLM detailed look")
        except Exception as _e:
            print(f"[anim-inspect] skipped ({_e})")
        # VISUAL-MEMORY priors: if grounding recalled that a salient entity resembles one seen in a
        # prior game (cross-game visual catalog), surface that prior to the actor (once).
        _vnote = getattr(self, "_pending_visual_note", None)
        if _vnote:
            _animation_summary = (_vnote + "\n" + (_animation_summary or "")).strip()
            self._pending_visual_note = None
        # MATCH-GOAL conform: if the mover differs in size/orientation from the scene goal, surface the
        # transform actions (grow/shrink/rotate) to fill/match it -- not just reach it (once).
        _mnote = getattr(self, "_pending_match_note", None)
        if _mnote:
            _animation_summary = (_mnote + "\n" + (_animation_summary or "")).strip()
            self._pending_match_note = None
        # CLOSURE: if any shape is incomplete (has a mouth), surface the close-every-mouth win goal --
        # pair each with its complement (opposite mouth + matching attribute) into a solid figure (once).
        _cnote = getattr(self, "_pending_closure_note", None)
        if _cnote:
            _animation_summary = (_cnote + "\n" + (_animation_summary or "")).strip()
            self._pending_closure_note = None
        # WIN-PRIOR: the win condition generalized from prior solved levels of this game (the goal to
        # pursue + verify), surfaced once at level start.
        _wnote = getattr(self, "_pending_win_note", None)
        if _wnote:
            _animation_summary = (_wnote + "\n" + (_animation_summary or "")).strip()
            self._pending_win_note = None
        # GROUND-TRUTH CORRECTION carried from last turn: the guard overrode a wrong
        # agent_moved=false; tell the VLM its prior read disagreed with the pixels so
        # it re-grounds rather than building on the false state.
        _gtnote = getattr(self, "_pending_gt_note", None)
        if _gtnote:
            _animation_summary = (_gtnote + "\n" + (_animation_summary or "")).strip()
            self._pending_gt_note = None
        # ERROR-LEDGER: surface the system's own recurring-error areas (heightened scrutiny), once
        # per run -- so it does not repeat a kind of mistake (mover/goal mix-up, missed legend detail).
        try:
            if not getattr(self, "_ledger_surfaced", False):
                import os as _os
                from error_ledger import ErrorLedger as _EL
                _root = _os.environ.get("COS_KB_ROOT") or str(
                    Path(__file__).resolve().parents[3] / ".tmp/kb")
                _ln = _EL(Path(_root) / "error_ledger.json").scrutiny_note()
                if _ln:
                    _animation_summary = (_ln + "\n" + (_animation_summary or "")).strip()
                    print("[error-ledger] surfaced recurring-error scrutiny")
                self._ledger_surfaced = True
        except Exception as _e:
            print(f"[error-ledger] skipped ({_e})")
        # FAILURE ESCALATION (game-agnostic backstop): did THIS action advance the world
        # (score/lc up, or a genuinely NEW settled state)?  If the loop keeps acting with no
        # progress -- retrying an action, or varying the same approach turn after turn -- surface
        # a STOP-and-MEASURE-the-displayed-ground-truth directive so the VLM breaks the guess loop.
        try:
            import hashlib as _hl
            import numpy as _np
            if not hasattr(self, "_escalator"):
                from failure_escalation import FailureEscalator
                self._escalator = FailureEscalator()
                self._esc_last_score, self._esc_state_hashes = None, []
            _sc = self.world.score
            _score_up = (_sc is not None and self._esc_last_score is not None
                         and _sc > self._esc_last_score)
            _new_state = False
            if frame_stack is not None and len(frame_stack):
                _h = _hl.md5(_np.asarray(frame_stack[-1]).tobytes()).hexdigest()
                _new_state = _h not in self._esc_state_hashes
                self._esc_state_hashes.append(_h)
                self._esc_state_hashes = self._esc_state_hashes[-self._escalator.window:]
            if _sc is not None:
                self._esc_last_score = _sc
            self._escalator.record(last_action, bool(_score_up or _new_state))
            _edir = self._escalator.directive()
            if _edir:
                _animation_summary = ("[ESCALATION] " + _edir + "\n"
                                      + (_animation_summary or "")).strip()
                print("[escalation] fired -> directive surfaced to VLM")
                # AUTO-REVALIDATION: stuck/surprised -> DROP INTO a revalidation loop. Re-derive the
                # doubtable premises (the claim store's OPEN / needs-recheck claims) from GROUND TRUTH,
                # weakest-credence first (re-demonstrate the control, read its measured effect), rather
                # than retrying the failed plan.  If nothing is registered, the directive says so --
                # that itself is the bug (premises a plan rests on must be filed as doubtable claims).
                import revalidation as _rv
                _cs = getattr(self, "_claim_store", None)
                _prems = []
                if _cs is not None:
                    try:
                        _opens = _cs.open_claims(None)
                    except Exception:
                        _opens = []
                    for _c in _opens:
                        _cr = float(getattr(_c, "credence", 0.0) or 0.0)
                        _prems.append(_rv.Premise(
                            name=getattr(_c, "statement", "") or getattr(_c, "claim_id", "?"),
                            credence=_cr, provenance=getattr(_c, "provenance", "guessed"),
                            probe=getattr(_c, "probe", "") or "",
                            ground_truthed=(not getattr(_c, "needs_recheck", False)) and _cr >= 0.8))
                _rdir = _rv.RevalidationDirective(
                    reason="stuck / repeated failure -- a premise the plan rests on is probably wrong",
                    premises=_rv.revalidation_order(_prems))
                _animation_summary = (_rdir.to_note() + "\n" + _animation_summary).strip()
                print(f"[revalidation] auto-dropped in ({len(_prems)} doubtable premise(s))")
        except Exception as _e:
            print(f"[escalation] skipped ({_e})")
        # SCENE-CUT QUALITY GATE: if the animation changes view partway, the
        # automated motion analysis spans unrelated screens and is UNRELIABLE.
        # Surface it loudly and make the acting VLM RESPONSIBLE for reading the
        # filmstrip per phase rather than trusting the (garbage) cross-cut events.
        _cuts = self._substrate_scene_cuts(anim_dir)
        if _cuts:
            _flag = (
                f"\n[SUBSTRATE QUALITY WARNING] This animation CHANGES VIEW "
                f"{len(_cuts)} time(s) (scene cut at sub-frame(s) {_cuts}): part of "
                f"it is a DIFFERENT screen (a zoom / overlay / preview), not the "
                f"main scene moving.  The automated entity-motion + silhouette "
                f"analysis above is computed against the FIRST view and is "
                f"UNRELIABLE across a cut (it can invent large phantom movers from "
                f"the view change).  YOU must read the FILMSTRIP yourself, phase by "
                f"phase (before the cut vs after), and report what each phase "
                f"actually shows -- do not trust cross-cut motion numbers.\n")
            _animation_summary = (_flag + (_animation_summary or "")).strip()
            print(f"[substrate] animation has {len(_cuts)} scene cut(s) at "
                  f"{_cuts}; flagged analysis UNRELIABLE -> VLM must verify.")
        # Persist the substrate-detected movements per turn (so they are
        # inspectable + renderable, not write-only).
        if _anim_events:
            try:
                (turn_dir / "animation_entities.json").write_text(
                    json.dumps(_anim_events, indent=2), encoding="utf-8")
            except Exception:
                pass
        # Render the animation as a VIEWABLE filmstrip (the actual sub-frames),
        # so the VLM can inspect the motion itself, not just the text narration.
        # Stash anim_dir for the on-demand animation_zoom tool.
        self._last_anim_dir = anim_dir
        # Persist the RAW 64-grid sub-frames (not just the rendered filmstrip) so
        # the silhouette tracker + any offline analysis can re-read them.
        if anim_dir is not None:
            try:
                import shutil
                raw_dir = turn_dir / "anim_raw"
                raw_dir.mkdir(parents=True, exist_ok=True)
                for f in sorted(Path(str(anim_dir)).glob("frame_*.png")):
                    shutil.copy2(f, raw_dir / f.name)
            except Exception:
                pass
        self._last_filmstrip_path = None
        self._last_entanalysis_filmstrip = None
        _film_block = ""
        if anim_dir is not None:
            # Per-frame ENTITY ANALYSIS filmstrip (the verification view): each
            # sub-frame with the substrate's detected-entity boxes drawn on it, so
            # the analysis can be SEEN/checked rather than trusted blind.
            _known = {n: r.current_bbox
                      for n, r in (getattr(self.world, "entities", {}) or {}).items()
                      if getattr(r, "current_bbox", None) is not None}
            _ea = self._render_entity_analysis_filmstrip(
                anim_dir, turn_dir / "animation_entities_filmstrip.png", _known)
            if _ea is not None:
                self._last_entanalysis_filmstrip = str(
                    turn_dir / "animation_entities_filmstrip.png")
            _film = self._render_animation_filmstrip(
                anim_dir, turn_dir / "animation_filmstrip.png")
            if _film is not None:
                self._last_filmstrip_path = str(turn_dir / "animation_filmstrip.png")
                # Surface the firing registry-instincts (e.g. the mandatory
                # animation-first directive) from instincts.py.  When none fires
                # (e.g. the win is already understood), still REFERENCE the
                # attached filmstrip so the VLM can consult it.
                _ctx = self._instinct_context("delta", anim_dir)
                _film_block = _INST.REGISTRY.render_active(_ctx)
                if not _film_block:
                    _film_block = (
                        f"\nANIMATION FILMSTRIP (`animation_filmstrip.png`) — the "
                        f"{_ctx.n_frames} sub-frames of this action's animation; "
                        f"consult if useful.\n")
        _t4 = time.time()
        if _T:
            print(f"[timing] turn {turn_n}: gridded={_t1-_t0:.1f}s "
                  f"prompt_prep={_t2-_t1:.1f}s entity_deltas={_t3-_t2:.1f}s "
                  f"animation={_t4-_t3:.1f}s (frames={len(frame_stack) if frame_stack else 0})",
                  file=sys.stderr, flush=True)
        prompt = DELTA_PERCEPTION_PROMPT.format(
            game_id=self.game.game_id, level=self.game.level,
            turn_n=turn_n, turn_prev=turn_n - 1,
            sys_text=sys_text,
            world_snapshot=json.dumps(snapshot, indent=2),
            last_action=last_action,
            entity_deltas=_entity_deltas,
            animation_entities=_animation_entities,
            animation_summary=_animation_summary,
            animation_filmstrip_block=_film_block,
            reply_name="reply.txt",
        )
        prompt_path = turn_dir / "prompt.md"
        # Defensive: re-ensure the turn dir exists right before writing.  A
        # concurrent mirror/cleanup or a level-advance reset can race away the
        # dir created at the top of this method; recreating it here is a no-op
        # when present and prevents a FileNotFoundError that would kill the run.
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(prompt, encoding="utf-8")
        (turn_dir / "STATUS.txt").write_text(
            f"WAITING for delta perception reply at {turn_dir/'reply.txt'}\n",
            encoding="utf-8",
        )
        return prompt_path

    # --- reply polling ---

    def poll_reply(self, turn_n: int) -> dict:
        turn_dir = self.work_dir / f"turn_{turn_n:03d}"
        reply_path = turn_dir / "reply.txt"
        deadline = time.time() + self.vlm_timeout_s
        print(f"[exploratory-driver] waiting for reply at {reply_path}",
              flush=True)
        while time.time() < deadline:
            if reply_path.exists():
                body = reply_path.read_text(encoding="utf-8").strip()
                if body:
                    parsed = self._consume_reply(reply_path, body)
                    if parsed is None:  # unparsable -> degrade like a timeout
                        return self._fallback_reply(turn_n)
                    parsed = self._fulfill_visual_queries(
                        parsed, turn_dir, self._last_frame_path, "reply")
                    # INSTINCT ENFORCEMENT: re-prompt once for any MANDATORY
                    # instinct that fires this turn but the reply didn't satisfy
                    # (e.g. animation-first on an animated, win-unknown turn).
                    # No-op on non-animated/initial turns.
                    return self._enforce_instincts(parsed, turn_dir, "reply")
            time.sleep(self.poll_s)
        # VLM did not answer within the timeout — do NOT crash the run.  Degrade
        # to a substrate-only perception and keep going (strict-mode robustness).
        return self._fallback_reply(turn_n)

    def _consume_reply(self, reply_path: Path, body: str) -> Optional[dict]:
        # tolerate ```json fences
        if body.startswith("```"):
            body = body.split("\n", 1)[1] if "\n" in body else body
            body = body.rsplit("```", 1)[0]
        # tolerate truncation by appending closing braces
        parsed = None
        for suffix in ("", "}", '"}', '"}}'):
            try:
                parsed = json.loads(body + suffix)
                break
            except json.JSONDecodeError:
                parsed = None
                continue
        if not isinstance(parsed, dict):
            # STRICT-MODE ROBUSTNESS: a present-but-unparsable (or non-object)
            # reply must NOT crash the run.  An autonomous VLM emitting malformed
            # JSON is the single most likely strict-mode event; the old code
            # raise'd RuntimeError here, which propagated out of the (unguarded)
            # run loop and killed the game.  Quarantine the bad file and signal
            # the caller to DEGRADE to the substrate-only fallback, exactly like
            # a timeout.  Returns None; every caller treats None as "no reply".
            print(f"[driver] WARNING: unparsable/invalid reply at {reply_path}; "
                  f"degrading to substrate-only perception (no crash).",
                  flush=True)
            self._last_perception_ss = {}
            try:
                bad = reply_path.with_name(reply_path.stem + ".unparsable.txt")
                shutil.move(str(reply_path), str(bad))
            except Exception:
                pass
            return None
        # Stash the perception's OWN symbolic_state (initial replies carry it
        # at top level; delta replies nest it under "perception") so the
        # directed-intent gate in run_one_step can read perception's localized
        # goal_candidate_cells + confidence — the synthesized `ss` there is
        # built from world inference and carries no confidence.
        try:
            perc = (parsed.get("perception")
                    if isinstance(parsed.get("perception"), dict) else parsed)
            self._last_perception_ss = (perc or {}).get("symbolic_state") or {}
        except Exception:
            self._last_perception_ss = {}
        # Ingest any VLM-authored claims + claim resolutions (claim-directed
        # probing).  Guarded so a malformed `claims` block never breaks the run.
        self._ingest_claims(parsed)
        consumed = reply_path.with_name(reply_path.stem + ".consumed.txt")
        shutil.move(str(reply_path), str(consumed))
        return parsed

    def _level_signature(self) -> str:
        """Layout signature for claim scoping -- a level's claims are valid for
        re-runs of THAT level, not a different one.  Uses a driver-maintained
        transition ORDINAL (not self.game.level, which the live adapter leaves at
        0): the Nth level reached is 'lc{N}', stable across re-runs."""
        return f"lc{getattr(self, '_level_ordinal', 0)}"

    def _ingest_claims(self, parsed) -> None:
        """Fold the VLM's authored claims + resolutions into the persistent
        Claim Store: `claims` (a list of hypothesis records, each optionally
        carrying a discriminating `probe` action + `importance`) and
        `claim_updates` (a list of {id, outcome:'proven'|'refuted'}).  The VLM
        WRITES; the substrate scopes/ranks/persists.  Fully guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None or not isinstance(parsed, dict):
            return
        try:
            turn = int(getattr(self.world, "turn", 0) or 0)
            aq = parsed.get("action_queue")
            # Load on first sight, OR REFILL when the queue has drained -- so a
            # closed-loop responder can feed the next action each turn (e.g. drive a
            # toggleable panel to a target: read state -> emit next toggle/fire).
            if (isinstance(aq, list) and aq
                    and (not getattr(self, "_action_queue_initialized", False)
                         or not getattr(self, "_action_queue", None))):
                self._action_queue = [str(a) for a in aq]
                self._action_queue_initialized = True
                print(f"[action-queue] (re)loaded {len(self._action_queue)} action(s)")
            authored = parsed.get("claims")
            if isinstance(authored, list) and authored:
                cs.ingest(authored, turn=turn, level_signature=self._level_signature())
            _guard = getattr(self, "_refute_is_discriminating", None)
            for upd in (parsed.get("claim_updates") or []):
                if isinstance(upd, dict) and upd.get("id"):
                    outcome = upd.get("outcome", "proven")
                    disc = _guard(cs, upd["id"], outcome) if _guard else True
                    cs.close(upd["id"], outcome, turn=turn, discriminating=disc)
            if authored or parsed.get("claim_updates"):
                cs.save_for_game()
            # ERRORS_EMITTED: the in-loop VLM registers its OWN significant mistakes -- and optionally
            # the SOLUTION that resolved one -- into the durable KB error ledger, mirroring claims /
            # lessons.  Each item: {category, description, variation?, fix?, resolution?, solution_id?}.
            # A `resolution` marks the error RESOLVED and links the working solution; otherwise it is a
            # registered pitfall recalled when a similar situation recurs (learning_from_error loop).
            _emitted = parsed.get("errors_emitted")
            if isinstance(_emitted, list) and _emitted:
                try:
                    from learning_loop import LearningLoop as _LL
                    _loop = _LL()
                    _lvl = self._level_signature()
                    _n = 0
                    for e in _emitted:
                        if not isinstance(e, dict):
                            continue
                        # accept common synonyms so a valid report is never silently dropped over
                        # key naming (category/situation/area; description/error/what)
                        _cat = e.get("category") or e.get("situation") or e.get("area")
                        _desc = e.get("description") or e.get("error") or e.get("what") or ""
                        if not _cat:
                            continue
                        if e.get("resolution"):
                            _loop.register_solution(str(_cat), str(e["resolution"]),
                                                    solution_id=str(e.get("solution_id", "")),
                                                    description=str(_desc))
                        else:
                            _loop.register_error(str(_cat), str(_desc),
                                                 variation=str(e.get("variation", "")),
                                                 fix=str(e.get("fix", "")), level=_lvl)
                        _n += 1
                    if _n:
                        print(f"[error-ledger] registered {_n} VLM-emitted error(s)/resolution(s)")
                except Exception as _ee:
                    print(f"[error-ledger] emit skipped ({_ee})")
            # SCENE STATE refinement loop: apply the VLM's `scene_ops` (resolve/
            # tighten/correct/...) + refresh+persist the state for the live views.
            # getattr-guarded so mock-based unit tests (no real driver) are unaffected.
            _so = getattr(self, "_apply_scene_ops", None)
            if _so:
                _so(parsed)
            _sy = getattr(self, "_sync_scene", None)
            if _sy:
                _sy()
        except Exception as e:
            print(f"[claim-store] ingest skipped ({e})")

    def _ensure_function_coverage(self) -> None:
        """COMPLETENESS guarantee: every perceived, plausibly-functional entity
        gets an OPEN 'what is its function?' claim auto-authored (with a click-
        centre discriminating probe), so the value-of-information selector
        necessarily works through ALL of them and nothing salient stays unprobed
        because the VLM forgot to write a claim for it.  A perceived role is a
        HYPOTHESIS, not a confirmation, so an entity is covered only once a claim
        TARGETING it exists (authored or auto) -- and 'confirmed' only once that
        claim is resolved.  Fully guarded; runs just before claim selection.

        CONTEXT roles below are the ones perception already treats as
        non-interactive backdrop; everything else (incl. 'unknown', a hypothesised
        'legend'/'scenery', a control, a marker) is plausibly functional and is
        probed until its function is observed."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return
        context_roles = {"hud", "background", "bg", "wall", "floor",
                         "border", "frame"}
        # SAMPLE, don't exhaust: infer the remaining members of a homogeneous
        # similarity group from the few already probed, so they are covered and
        # not seeded/probed one-by-one.
        try:
            self._generalize_homogeneous_group_functions()
        except Exception as e:
            print(f"[generalise] skipped ({e})")
        try:
            sig = self._level_signature()
            turn = int(getattr(self.world, "turn", 0) or 0)
            # WIN-STATE-GATED PRIORITY: the value of resolving an unknown entity's
            # function is the COMPLEMENT of how well the win is already understood.
            # No win in hand -> exploring unknowns is maximally valuable (one might
            # BE the key; never give up with unknowns unprobed).  A high-credence
            # win already known -> a path is being pursued, so probing unknowns is
            # deferred (efficiency in competition mode).  A 0.1 floor keeps them
            # non-zero, so they are still eventually probed when nothing else is
            # pressing.  Principled (the win-credence itself), no tuned threshold.
            max_win = 0.0
            for h in (getattr(self.world, "win_condition_hypotheses", []) or []):
                max_win = max(max_win, float(getattr(h, "credence", 0) or 0))
            explore_imp = round(max(0.1, 1.0 - max_win), 2)
            # STUCK -> CURIOSITY PRIORITY (game-agnostic): when recent actions made
            # no progress, an UNEXPLORED but SALIENT element (a distinct control /
            # button that is NOT one of many identical marks -- it "looks like it
            # implies a solution") should jump the probe queue.  Salience proxy:
            # the entity is a SINGLETON (not a member of any sizeable similarity
            # group), so a lone bordered box/button outranks the 12th identical
            # switch.  Only boosts UNPROBED salient claims; once probed they fall
            # back, so it can't thrash.
            try:
                stuck = self._is_stuck()
            except Exception:
                stuck = False
            grouped = set()
            if stuck:
                _groups = getattr(self.world, "groups", {}) or {}
                _giter = _groups.values() if isinstance(_groups, dict) else _groups
                for g in _giter:
                    mem = list(getattr(g, "members", []) or [])
                    if len(mem) >= 3:
                        grouped.update(mem)

            def _imp_for(name: str) -> float:
                if stuck and name not in grouped:
                    return max(explore_imp, 0.85)        # salient unknown -> jump queue
                return explore_imp
            # apply the current policy to existing function-coverage claims
            for cid, c in cs.claims.items():
                if not cid.startswith("function_of_"):
                    continue
                if c.status == "open":
                    nm = cid[len("function_of_"):]
                    c.importance = (_imp_for(nm)
                                    if (stuck and int(getattr(c, "times_probed", 0) or 0) == 0)
                                    else explore_imp)
                elif c.status == "proven" and getattr(c, "needs_recheck", False):
                    # CARRIED / GENERALISED function (assume-stable): keep it
                    # re-probeable -- a level may redesign an entity's function --
                    # but at a FRACTION of the unknown-exploration importance, so a
                    # re-verify ALWAYS ranks below probing a genuinely-unknown
                    # entity (esp. when the win is not yet in sight, when explore_imp
                    # is high).  It is reached only once the unknowns are exhausted.
                    c.importance = round(0.25 * explore_imp, 3)
            seed, uncovered = [], 0
            for name, rec in (getattr(self.world, "entities", {}) or {}).items():
                role = (getattr(rec, "current_role", "") or "").lower()
                if role in context_roles or getattr(rec, "current_bbox", None) is None:
                    continue
                if not cs.has_confirmed_function(name, sig):
                    uncovered += 1
                if cs.is_covered(name, sig):
                    continue
                r0, c0, r1, c1 = rec.current_bbox
                col, row = int((c0 + c1) // 2), int((r0 + r1) // 2)
                seed.append({
                    "id": f"function_of_{name}",
                    "statement": f"what is '{name}'s function in the mechanic? "
                                 f"(perceived role guess: {role or 'unknown'}, unconfirmed)",
                    "scope": "level", "target": [name],
                    "probe": f"CLICK:{col},{row}", "cost": 1,
                    "importance": _imp_for(name), "credence": 0.5})
            if seed or uncovered:
                if seed:
                    cs.ingest(seed, turn=turn, level_signature=sig)
                cs.save_for_game()
                mode = ("DEFERRED (win understood)"
                        if explore_imp <= self._WIN_UNDERSTOOD_EXPLORE_CUTOFF
                        else "PRIORITISED (win unknown)")
                print(f"[claim-coverage] {len(seed)} new function-claim(s); "
                      f"{uncovered} entity(ies) lack a CONFIRMED function; "
                      f"exploration importance={explore_imp} -> {mode}.")
        except Exception as e:
            print(f"[claim-coverage] skipped ({e})")

    def _seed_structural_claims(self) -> None:
        """COS's GENERALISATION step: form higher-level STRUCTURAL hypotheses from
        the perceived scene (two same-shape panels -> one may be the instruction
        for the other; marker-distance == control-count -> per-step program) and
        file them as CLAIMS to be VERIFIED -- strong priors at guessed credence,
        never hardcoded facts.  Idempotent (ingest dedups by id).  Guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return
        try:
            import structural_claims as _sc
            ents = {n: {"role": (getattr(r, "current_role", "") or ""),
                        "bbox": getattr(r, "current_bbox", None)}
                    for n, r in (getattr(self.world, "entities", {}) or {}).items()
                    if getattr(r, "current_bbox", None)}
            grps = {n: {"members": list(getattr(g, "members", []) or []),
                        "criterion": getattr(g, "criterion", "")}
                    for n, g in (getattr(self.world, "groups", {}) or {}).items()}
            proposed = _sc.propose_structural_claims(ents, grps)
            new = [c for c in proposed if c["id"] not in cs.claims]
            if proposed:
                cs.ingest(proposed, turn=int(getattr(self.world, "turn", 0) or 0),
                          level_signature=self._level_signature())
                cs.save_for_game()
            if new:
                print(f"[structural] formed {len(new)} structural claim(s) "
                      f"(priors, to verify): {[c['id'] for c in new]}")
        except Exception as e:
            print(f"[structural] skipped ({e})")

    def _seed_program_map_claims(self) -> None:
        """SYMBOLIC stage (program/scene): orient each geometric similar-structure
        pair by the MEASURED response facts -- the side whose members are SETTABLE
        is the active PROGRAM, its inert twin is the REFERENCE -- and file a
        program-mapping claim (MATCH:active~reference+TRIGGER).  The active/
        reference roles are DISCOVERED from behaviour, not from a label the VLM
        typed, so this is the rung above the bitmap correspondence.  Verified only
        by the SCORE.  Idempotent; guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return
        try:
            import symbolic_structure as _sym
            rf = getattr(self.world, "response_facts", None) or {}
            if not rf:
                return
            ents = {n: {"role": (getattr(r, "current_role", "") or ""),
                        "bbox": getattr(r, "current_bbox", None)}
                    for n, r in (getattr(self.world, "entities", {}) or {}).items()
                    if getattr(r, "current_bbox", None)}
            grps = {n: {"members": list(getattr(g, "members", []) or [])}
                    for n, g in (getattr(self.world, "groups", {}) or {}).items()}
            claims = _sym.propose_program_map_claims(ents, grps, rf)
            new = [c for c in claims if c["id"] not in cs.claims]
            if claims:
                cs.ingest(claims, turn=int(getattr(self.world, "turn", 0) or 0),
                          level_signature=self._level_signature())
                cs.save_for_game()
            if new:
                print(f"[program-map] oriented {len(new)} program/reference pair(s) "
                      f"by response: {[c['id'] for c in new]}")
        except Exception as e:
            print(f"[program-map] skipped ({e})")

    def _map_claims_across_structures(self) -> None:
        """STRUCTURE MAPPING (analogical transfer): for every pair of SIMILAR
        structures, take what is already CLAIMED about one and hypothesise the
        analogous claim about the other under the member correspondence -- 'what
        happens in the left panel applies to the right panel'.  This is the core
        of human analogy (Gentner), wired generally: it transfers ANY claim (a
        function, a correlation, a per-step program) across ANY similar-structure
        mapping.  Each transferred claim is a GUESSED hypothesis to VERIFY on the
        target structure (credence = source x mapping score), filed so the prober
        tests it.  Idempotent (ingest dedups by id).  Guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return
        try:
            import structural_claims as _sc
            ents = {n: {"role": (getattr(r, "current_role", "") or ""),
                        "bbox": getattr(r, "current_bbox", None)}
                    for n, r in (getattr(self.world, "entities", {}) or {}).items()
                    if getattr(r, "current_bbox", None)}
            grps = {n: {"members": list(getattr(g, "members", []) or [])}
                    for n, g in (getattr(self.world, "groups", {}) or {}).items()}
            existing = [{"id": c.claim_id, "statement": c.statement,
                         "target": list(c.target or []), "credence": c.credence,
                         "importance": c.importance, "plan": c.plan, "kind": c.kind}
                        for c in cs.claims.values()]
            transferred = _sc.transfer_across_mappings(ents, grps, existing)
            new = [t for t in transferred if t["id"] not in cs.claims]
            if transferred:
                cs.ingest(transferred, turn=int(getattr(self.world, "turn", 0) or 0),
                          level_signature=self._level_signature())
                cs.save_for_game()
            if new:
                print(f"[structure-mapping] transferred {len(new)} claim(s) across "
                      f"similar structures: {[t['id'] for t in new]}")
        except Exception as e:
            print(f"[structure-mapping] skipped ({e})")

    def _has_unexamined_affordance(self) -> bool:
        """True while some perceived, plausibly-functional entity has NEVER been
        probed -- its function-claim is OPEN with zero probes.  Used to keep
        exploration BREADTH-FIRST: a single cheap click on an unexamined control
        may reveal the mechanic, so it should be spent before any expensive,
        committed plan.  Context roles (hud/background/wall/...) don't count.
        Fully guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return False
        try:
            sig = self._level_signature()
            context = {"hud", "background", "bg", "wall", "floor", "border", "frame"}
            for name, rec in (getattr(self.world, "entities", {}) or {}).items():
                if (getattr(rec, "current_role", "") or "").lower() in context:
                    continue
                if getattr(rec, "current_bbox", None) is None:
                    continue
                c = cs.claims.get(f"function_of_{name}")
                if (c is not None and c.status == "open" and c.times_probed == 0
                        and cs._active_for(c, sig)):
                    return True
            return False
        except Exception:
            return False

    # The explore->exploit crossover: the SAME cutoff function-coverage uses to
    # DEFER blind probing (exploration value 1 - max_win_credence has fallen this
    # low).  Defined once here so both sites share it (no second magic number).
    _WIN_UNDERSTOOD_EXPLORE_CUTOFF = 0.3

    def _max_win_credence(self) -> float:
        """Highest credence among the current win-condition hypotheses (0.0 if
        none).  A carried/confirmed win seeds this high."""
        return max([float(getattr(h, "credence", 0) or 0)
                    for h in (getattr(self.world, "win_condition_hypotheses", []) or [])],
                   default=0.0)

    def _win_understood(self) -> bool:
        """True when the win is UNDERSTOOD enough to EXPLOIT (pursue it) rather
        than EXPLORE (scout).  Uses the same crossover as function-coverage:
        exploration value (1 - max_win_credence) has fallen to the deferred
        cutoff.  Game-agnostic; reuses the existing win-credence signal."""
        return (1.0 - self._max_win_credence()) <= self._WIN_UNDERSTOOD_EXPLORE_CUTOFF

    def _kb_has_prior_guidance(self) -> bool:
        """True when the KB already holds SUBSTANTIVE prior knowledge for the
        current situation: this game's own lessons, or a relevant cross-game
        technique / win-condition.  Used to CONSULT THE KB BY DEFAULT before
        falling back to a blind curiosity probe -- so COS acts on what it (or a
        related game) already learned instead of re-deriving from scratch.
        Game-agnostic; for a first-ever game with an empty KB this is False, so
        the blind-probe discovery path is unchanged."""
        try:
            from kb_recall import recall                       # noqa: E402
            for h in recall(self.world, k=4):
                if (h.get("source") == "lesson"
                        or h.get("kind") in ("technique", "win_condition", "summary")):
                    return True
        except Exception:
            pass
        return False

    def _consult_before_blind_probe(self) -> bool:
        """Whether to DEFER a blind curiosity probe to the prior-guided STRATEGY
        layer instead of auto-sweeping. True when COS should think before
        spending an action: either the KB holds guidance for this game, OR the
        WIN IS NOT YET UNDERSTOOD -- in which case the actor should READ THE FRAME
        AS A TUTORIAL (first_level_is_tutorial: ARC-AGI-3 first levels teach the
        mechanic; act on the legend / aim line / preview the designer drew)
        rather than blind-probe past it. The strategy layer can still choose to
        probe; this just routes the decision through the INFORMED actor. The
        whole point of su15's failure: COS blind-probed for thousands of clicks
        while the tutorial sat unread."""
        if not self.use_strategy:
            return False
        return self._kb_has_prior_guidance() or (not self._win_understood())

    def _select_probe(self, entities_for_actor):
        """Choose the next probe.  EXPLORE while the win is unknown, EXPLOIT once
        it is understood.

          1. cx single-trial -- evidence-driven, dormant when no mover exists;
             keep it first.
          2. WIN-PURSUIT (exploit): once the win is UNDERSTOOD (a high-credence
             win hypothesis -- e.g. carried from a prior level), ATTEMPT an
             available committed pursuit (structural-template match / OFAT)
             BEFORE more curiosity scouting.  This is what makes COS actually TRY
             to solve instead of re-probing entity functions; if no pursuit is
             available yet it falls through to scouting.
          3. EXPLORE (win unknown): while an UNEXAMINED affordance remains, a
             CHEAP one-action scout BEFORE any committed multi-action plan (don't
             pour resource into a complex path a cheap scout could make
             unnecessary); once everything is examined, the committed plans run.
        Game-agnostic; the only numeric gate is the shared win-understood cutoff
        (reused from function-coverage).  Fully guarded."""
        # First: resolve the PREVIOUS function-probe from its observed outcome and
        # CLOSE it (one-shot) -- so an inert entity drops out of the open pool and
        # the prober escalates instead of re-clicking it forever.
        self._resolve_prev_function_probe()
        # Resolve a pending structural disambiguation: measure whether the panel
        # we just clicked CHANGED (settable) by a direct pre/post diff of its own
        # region -> records _panel_settable, which orders active vs reference.
        self._resolve_prev_disambiguation()
        # INSTINCT: a mover implies a target -- auto-file target-seeking claims so
        # the prober commits to EXPLOIT instead of looping on blind curiosity.  The
        # competition perception prompt never elicits these, so the substrate does.
        _mst = getattr(self, "_substrate_mover_implies_target", None)
        if _mst:
            try:
                _mst(entities_for_actor)
            except Exception:
                pass
        # TOP PRIORITY: a VLM-authored DETERMINISTIC action queue runs verbatim,
        # one action per turn, before any autonomous tier -- so a grounded program
        # (set these switches, then fire the trigger) executes CLEANLY, without the
        # curiosity / cx tiers interleaving or looping on it.  Empty by default.
        _aq = getattr(self, "_action_queue_probe", None)
        q = _aq(entities_for_actor) if _aq else None
        if q is not None:
            return q
        # A committed structural pursuit, ONCE STARTED, runs to completion (fire)
        # or refutation before anything else -- not preempted by a transient
        # win-credence dip or a cheap scout, which would otherwise scatter a
        # half-configured panel across many interleaved turns.  Only an
        # in-progress match (probe_state['struct'] mid-lifecycle) gets this
        # priority; detecting a NEW match still goes through the normal gating
        # below, so this never commits blindly.
        st = (getattr(self.world, "probe_state", {}) or {}).get("struct")
        if st and st.get("phase") in ("match", "fire", "fired"):
            cont = self._structural_match_probe(entities_for_actor)
            if cont is not None:
                return cont
        cx = self._controlled_experiment_probe(entities_for_actor)
        if cx is not None:
            return cx
        # EXPLOIT: the win is understood -> attempt a committed pursuit first.
        if self._win_understood():
            # A solved-mechanic pursuit (the click-to-move law, or a self-calibrating
            # merge-and-deliver) IS the committed plan -- run it before ANY claim-
            # directed scouting, else a CONFIRMED mechanic gets re-verified
            # indefinitely instead of executed (that burned su15 lc1 into GAME_OVER:
            # the merge was confirmed turn 11, then 30 more claim-probes till the
            # budget ran out).  Each pursuit returns non-None only when applicable.
            # MERGE precedes the click-to-move law: on a merge level you must COMBINE
            # pieces before delivering, and a move-law carried from a prior level would
            # otherwise deliver an un-merged piece and LOSE (su15 lc1).  Each returns
            # non-None only when applicable, so a pure move-law level is unaffected.
            _ml = getattr(self, "_move_law_pursuit", None)
            _mg = getattr(self, "_merge_pursuit", None)
            pursue = (self._structural_match_probe(entities_for_actor)
                      or self._cx_ofat_probe(entities_for_actor)
                      or (_mg(entities_for_actor) if _mg else None)
                      or (_ml(entities_for_actor) if _ml else None))
            if pursue is not None:
                return pursue
            # No pursuit available yet -- but if a match is only blocked because
            # active/reference is undetermined, a directed disambiguation probe
            # ENABLES it (beats scouting unrelated entities).
            disambig = self._structural_disambiguation_probe(entities_for_actor)
            if disambig is not None:
                return disambig
            # otherwise scout to enable a pursuit.
            return (self._claim_directed_probe(entities_for_actor)
                    or self._exploratory_probe_choice(entities_for_actor))
        # EXPLORE: win unknown -> VERIFY ON-SCREEN INSTRUCTIONS first (designer
        # guidance is the highest-value lead).
        _instr_fn = getattr(self, "_instruction_verification_probe", None)
        instr = _instr_fn(entities_for_actor) if _instr_fn else None
        if instr is not None:
            return instr
        # Then the COMMITTED solved-mechanic pursuits, BEFORE the claim-directed scouts
        # below (so a confirmed mechanic is EXECUTED, not re-verified forever -- su15
        # lc1).  MERGE precedes the click-to-move law: on a merge level you must COMBINE
        # pieces before delivering, and a move-law carried from a prior level would
        # otherwise deliver an un-merged piece and LOSE.  Each returns non-None only
        # when applicable, so a pure move-law level (no same-appearance group ->
        # merge None) still runs the click-to-move law exactly as before.
        _merge_fn = getattr(self, "_merge_pursuit", None)
        merge = _merge_fn(entities_for_actor) if _merge_fn else None
        if merge is not None:
            return merge
        _pursue_fn = getattr(self, "_move_law_pursuit", None)
        pursue = _pursue_fn(entities_for_actor) if _pursue_fn else None
        if pursue is not None:
            return pursue
        # EXPLORE: win unknown -> breadth-first (cheap scouts before committed).
        if self._has_unexamined_affordance():
            return (self._claim_directed_probe(entities_for_actor)
                    or self._structural_match_probe(entities_for_actor)
                    or self._cx_ofat_probe(entities_for_actor)
                    or self._exploratory_probe_choice(entities_for_actor))
        return (self._structural_match_probe(entities_for_actor)
                or self._cx_ofat_probe(entities_for_actor)
                or self._claim_directed_probe(entities_for_actor)
                or self._exploratory_probe_choice(entities_for_actor))

    def _resolve_prev_disambiguation(self) -> None:
        """If a structural disambiguation probe clicked a panel last turn, MEASURE
        whether that panel changed -- a direct per-region pixel diff between the
        snapshotted PRE-click raw frame and the current POST-click raw frame.
        Records _panel_settable[name] (True=settable/active, False=inert/reference).
        Diffing the panel's OWN bbox is HUD-immune (the budget bar is elsewhere)
        and sidesteps the visual_events watch/threshold/persistence pipeline that
        was dropping a single-cell toggle.  Guarded -- never breaks the run."""
        pend = getattr(self, "_disambig_pending", None)
        self._disambig_pending = None
        if not pend:
            return
        try:
            import numpy as np
            pre = self._cx_read_sprite(pend.get("pre_frame"), pend["bbox"])
            post = self._cx_read_sprite(getattr(self, "_last_frame_path", None),
                                        pend["bbox"])
            if pre is None or post is None or pre.shape != post.shape:
                return
            changed = bool((np.asarray(pre) != np.asarray(post)).any())
            store = getattr(self, "_panel_settable", None)
            if store is None:
                store = self._panel_settable = {}
            store[pend["name"]] = changed
            print(f"[struct] disambiguation MEASURED {pend['name']}: "
                  f"{'SETTABLE (active)' if changed else 'inert (reference)'} "
                  f"-- direct pre/post diff of its own region.")
        except Exception as e:
            print(f"[struct] disambiguation resolve skipped ({e})")

    def _resolve_prev_function_probe(self) -> None:
        """A curiosity FUNCTION-probe is ONE-SHOT: clicking an entity and watching
        the result ANSWERS 'what does it do' -- a visible change confirms it has a
        function (proven), no change at all confirms it is inert to that click
        (refuted).  Either way the claim is RESOLVED after a single observation and
        must never be re-probed.  Without this, an inert marker's function-claim
        stays OPEN at high value-of-information and the prober re-clicks it forever
        (observed: 52x on tn36's arch), starving the cx / structural solver.
        Closes the previous function-probe's claim from its observed delta.
        Guarded -- never breaks the run."""
        cs = getattr(self, "_claim_store", None)
        lp = getattr(self, "_last_fn_probe", None)
        self._last_fn_probe = None
        if cs is None or not lp:
            return
        try:
            from claim_store import OPEN as _OPEN
            c = cs.claims.get(lp["claim_id"])
            if c is None or getattr(c, "status", "") != _OPEN:
                return
            deltas = getattr(self.world, "deltas_observed", []) or []
            d = next((x for x in reversed(deltas)
                      if lp["action"] in ((getattr(x, "action", "") or ""),
                                          (getattr(x, "inferred_action", "") or ""))),
                     deltas[-1] if deltas else None)
            if d is None:
                return
            changed = bool((getattr(d, "entities_changed", None) or [])
                           or (getattr(d, "animation_events", None) or [])
                           or getattr(d, "agent_moved", False))
            turn = int(getattr(self.world, "turn", 0) or 0)
            cs.close(lp["claim_id"], "proven" if changed else "refuted", turn=turn)
            cs.save_for_game()
            print(f"[claim] one-shot function probe '{lp['claim_id']}' RESOLVED "
                  f"({'function observed' if changed else 'inert -- no change'}) "
                  f"-> closed; will not re-probe.")
        except Exception as e:
            print(f"[claim] function-probe resolve skipped ({e})")

    def _action_queue_probe(self, entities_for_actor):
        """Pop the next action from a VLM-authored DETERMINISTIC queue (loaded once
        from a reply's `action_queue`).  Returns a probe that executes it verbatim,
        or None when the queue is empty.  Top-priority so a grounded program runs
        uninterrupted by the autonomous tiers.  Guarded."""
        try:
            from types import SimpleNamespace
            q = getattr(self, "_action_queue", None)
            if not q:
                return None
            action = self._resolve_action(str(q.pop(0)))
            print(f"[action-queue] executing {action} ({len(q)} left)")
            return SimpleNamespace(
                action=action, rationale="VLM action-queue (grounded program)",
                plan_kind="action_queue", goal_id="action_queue",
                target_cell=None, full_plan_actions=[action], is_probe=True)
        except Exception as e:
            print(f"[action-queue] skipped ({e})")
            return None

    def _claim_directed_probe(self, entities_for_actor):
        """Claim-directed probe: run the highest value-of-information OPEN claim's
        discriminating action (ranked by the Claim Store).  This is the unified
        prober between the specialised cx tier and blind curiosity coverage --
        it spends each action on the cheapest most-informative UNKNOWN, and skips
        what is already proven.  Returns a probe choice or None.  Guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return None
        try:
            from types import SimpleNamespace
            cover = getattr(self, "_ensure_function_coverage", None)
            if cover:
                cover()                             # no salient entity un-authored
            seedstruct = getattr(self, "_seed_structural_claims", None)
            if seedstruct:
                seedstruct()                        # form higher-level structural claims
            progmap = getattr(self, "_seed_program_map_claims", None)
            if progmap:
                progmap()                           # SYMBOLIC: orient similar structures by response (program vs reference)
            mapstruct = getattr(self, "_map_claims_across_structures", None)
            if mapstruct:
                mapstruct()                         # analogically transfer claims across similar structures
            # scope to the CURRENT level so a persisted claim from a DIFFERENT
            # level (reloaded on a re-run) is never probed on the wrong layout.
            c = cs.next_probe(self._level_signature())
            if c is None or not c.probe:
                return None
            act = self._resolve_action(c.probe)
            if act is None:
                return None
            cs.note_probed(c.claim_id, turn=int(getattr(self.world, "turn", 0) or 0))
            # A function-probe is ONE-SHOT: remember it so its outcome is read +
            # the claim CLOSED next turn (see _resolve_prev_function_probe), so an
            # inert entity is never re-probed in a loop.
            if str(c.claim_id).startswith("function_of_"):
                self._last_fn_probe = {"claim_id": c.claim_id, "action": c.probe}
            print(f"[claim] probing '{c.claim_id}' (VoI={c.value_of_information():.2f}, "
                  f"cost={c.cost}): {c.statement!r} -> {c.probe}")
            return SimpleNamespace(
                action=act, plan_kind="claim_probe",
                rationale=f"claim-directed probe: resolve {c.claim_id!r} ({c.statement})",
                goal_id="claim_probe", target_cell=None,
                full_plan_actions=[act], is_probe=True)
        except Exception as e:
            print(f"[claim] probe skipped ({e})")
            return None

    def _substrate_mover_implies_target(self, entities_for_actor):
        """INSTINCT -- a mover implies a target.  The moment perception yields a
        controllable MOVER, the win almost always routes through that mover
        REACHING or CONTACTING something (a goal, a slot, or any not-yet-understood
        object -- perceived is not understood).  So the substrate AUTO-FILES a
        guessed target-seeking claim per candidate target -- WITHOUT waiting for the
        (small competition) VLM to author it -- each carrying a discriminating
        CLICK-at-the-target probe.  That moves the prober off blind curiosity and
        toward EXPLOIT: drive the mover at the target and watch win_state/score.

        Why this exists: the competition perception prompt (cos_responder.PERC_SYS)
        elicits ENTITIES + a one-line purpose but never GROUPS, RELATIONSHIPS, or
        behaviour/target CLAIMS (the responder hardcodes those empty), so the
        explore->exploit hand-off had nothing to commit to.  This instinct supplies
        the missing claims structurally.  Game-agnostic, idempotent (stable ids),
        bounded fan-out, fully guarded."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return
        ents = [e for e in (entities_for_actor or [])
                if isinstance(e, dict) and e.get("name")]
        if not ents:
            return

        def tags(e):
            return ((e.get("role_hypothesis") or "") + " "
                    + (e.get("appearance") or "") + " "
                    + (e.get("name") or "")).lower()

        MOVER = ("mover", "agent", "player", "controll")
        STRUCT = ("wall", "hud", "status", "decor", "background", "field", "floor",
                  "border", "legend", "scenery", "room", "platform", "barrier")
        TARGETY = ("goal", "target", "slot", "box", "hole", "trigger", "exit",
                   "door", "blob", "ball", "win", "switch", "button", "gate",
                   "portal", "pad", "socket", "cup")
        movers = [e for e in ents if any(w in tags(e) for w in MOVER)]
        if not movers:
            return                                   # no mover -> nothing to seek
        mover_names = {e.get("name") for e in movers}

        def is_struct(e):
            return any(w in tags(e) for w in STRUCT)
        # target-ish objects first; else ANY non-mover non-structural object (an
        # unexplained perceived thing is itself a candidate target).
        targety = [e for e in ents if e.get("name") not in mover_names
                   and not is_struct(e) and any(w in tags(e) for w in TARGETY)]
        cands = targety or [e for e in ents if e.get("name") not in mover_names
                            and not is_struct(e)]
        if not cands:
            return
        primary = movers[0].get("name")
        filed = 0
        for t in cands[:6]:                          # bound the fan-out
            tn = t.get("name")
            cid = f"mover_seek::{primary}~{tn}"
            if cid in getattr(cs, "claims", {}):
                continue                             # idempotent
            claim = {
                "id": cid,
                "statement": (f"MOVER->TARGET: '{primary}' is controllable, so "
                              f"reaching/contacting '{tn}' may trigger progress or "
                              f"the win."),
                "kind": "win_hypothesis", "scope": "level",
                "target": [primary, tn],
                "plan": f"drive '{primary}' onto '{tn}', then check win_state/score",
                "probe": f"CLICK:{tn}",
                "cost": 1, "importance": 0.8, "credence": 0.35,
                "provenance": "guessed",
            }
            try:
                cs.ingest([claim], turn=int(getattr(self.world, "turn", 0) or 0),
                          level_signature=self._level_signature())
                filed += 1
            except Exception:
                pass
        if filed:
            try:
                cs.save_for_game()
            except Exception:
                pass
            print(f"[mover-seek] mover '{primary}' present -> filed {filed} "
                  f"target-seeking claim(s) (exploit frontier).", flush=True)

    # --- VLM-directed visual tools (general-purpose substrate help) ----------
    #
    # A perception reply MAY include a top-level `visual_queries` list — the VLM
    # directing the substrate to MEASURE something it can't reliably eyeball
    # (exact count, alignment/collinearity, distance, a magnified read of small/
    # nested structure, exact colour).  The harness fulfils each query against
    # the CURRENT logical frame (visual_query.py), shows the answers + any
    # rendered images back, and lets the VLM finalise its perception.  Keyed
    # ONLY on the presence of `visual_queries` (no game knowledge), bounded to
    # _VQ_MAX_ROUNDS round-trips, and fully guarded so a malformed request or a
    # timeout degrades to the original perception instead of crashing the loop.

    def _poll_named_reply(self, reply_path: Path) -> Optional[dict]:
        """Poll one specific reply file until vlm_timeout_s.  Returns the parsed
        reply (consuming the file) or None on timeout.  Does NOT itself trigger
        visual-query fulfilment — chaining is handled by the caller's recursion."""
        deadline = time.time() + self.vlm_timeout_s
        while time.time() < deadline:
            if reply_path.exists():
                body = reply_path.read_text(encoding="utf-8").strip()
                if body:
                    return self._consume_reply(reply_path, body)
            time.sleep(self.poll_s)
        return None

    def _stage_visual_query_followup(self, turn_dir: Path, vq_dir: Path,
                                     results: list, reply_stub: str,
                                     round_idx: int, reply_name: str) -> Path:
        """Write the follow-up prompt that shows the tool results (+ any rendered
        images) and asks the VLM to re-emit its perception."""
        images = [r["image"] for r in results
                  if isinstance(r, dict) and r.get("image")]
        rel = vq_dir.name  # vq dir is a child of turn_dir
        if images:
            lines = ["\n## Image attachments\n"]
            for im in images:
                lines.append(f"  `{rel}/{im}`")
            image_block = "\n".join(lines) + "\n"
        else:
            image_block = ""
        prompt = VISUAL_QUERY_FOLLOWUP_PROMPT.format(
            game_id=self.game.game_id, level=self.game.level,
            results_json=json.dumps(results, indent=2),
            image_block=image_block, reply_name=reply_name,
        )
        prompt_path = turn_dir / f"{reply_stub}_requery_r{round_idx}_prompt.md"
        prompt_path.write_text(prompt, encoding="utf-8")
        (turn_dir / f"{reply_stub}_requery_r{round_idx}_STATUS.txt").write_text(
            f"WAITING for visual-tool requery reply at {turn_dir/reply_name}\n",
            encoding="utf-8")
        return prompt_path

    def _fulfill_visual_queries(self, parsed, turn_dir: Path, frame_path,
                                reply_stub: str, round_n: int = 0):
        """If `parsed` requested visual tools, fulfil them and return the VLM's
        finalised re-perception; otherwise return `parsed` unchanged."""
        if not isinstance(parsed, dict):
            return parsed
        queries = parsed.get("visual_queries")
        if not queries:
            return parsed
        if round_n >= _VQ_MAX_ROUNDS:
            print(f"[visual-tools] reached max {_VQ_MAX_ROUNDS} rounds; "
                  f"proceeding with current perception", flush=True)
            return parsed
        if frame_path is None or not Path(frame_path).exists():
            print("[visual-tools] no current frame to query; ignoring "
                  "visual_queries", flush=True)
            return parsed
        try:
            vq_dir = Path(turn_dir) / f"vq_{reply_stub}_r{round_n + 1}"
            # Pass the last action's animation sub-frames so animation_* tools
            # (e.g. animation_zoom) can inspect the motion on demand.
            _anim = None
            _ad = getattr(self, "_last_anim_dir", None)
            if _ad is not None:
                _fr = sorted(Path(_ad).glob("frame_*.png"))
                _anim = _fr if len(_fr) >= 2 else None
            results = _VQ.run_visual_queries(
                frame_path, queries, vq_dir, n_ticks=self.n_ticks,
                anim_frames=_anim)
            (vq_dir / "results.json").write_text(
                json.dumps(results, indent=2), encoding="utf-8")
            n_err = sum(1 for r in results
                        if isinstance(r, dict) and r.get("error"))
            print(f"[visual-tools] round {round_n + 1}: fulfilled "
                  f"{len(results)} quer{'y' if len(results) == 1 else 'ies'} "
                  f"({n_err} error(s)) -> {vq_dir.name}/results.json", flush=True)
            reply_name = f"{reply_stub}_requery_r{round_n + 1}_reply.txt"
            self._stage_visual_query_followup(
                Path(turn_dir), vq_dir, results, reply_stub,
                round_n + 1, reply_name)
            parsed2 = self._poll_named_reply(Path(turn_dir) / reply_name)
            if not isinstance(parsed2, dict):
                print("[visual-tools] no requery reply (timeout); keeping "
                      "prior perception", flush=True)
                return parsed
            # the re-perception may itself ask for more tools (bounded)
            return self._fulfill_visual_queries(
                parsed2, turn_dir, frame_path, reply_stub, round_n + 1)
        except Exception as e:
            print(f"[visual-tools] fulfilment failed ({e}); keeping prior "
                  f"perception", flush=True)
            return parsed

    # --- INSTINCT REGISTRY integration (triggers + mandatory enforcement) -----
    #
    # COS's game-agnostic priors are organized in instincts.py by TRIGGER.  The
    # driver builds a TurnContext, surfaces the firing registry-instincts into the
    # delta prompt (e.g. the animation-first instinct), and ENFORCES the mandatory
    # firing ones — re-prompting ONCE if the reply doesn't satisfy them.  Adding a
    # new mandatory instinct needs no driver edit: register it in instincts.py.
    # Fully guarded + bounded — strict-mode safe (a re-prompt timeout keeps the
    # prior reply; never crashes or hangs).

    def _win_understood(self) -> bool:
        """COS 'thinks it understands the win' when it holds a credible
        win-condition hypothesis (promoted, or credence >= 0.6)."""
        try:
            for h in getattr(self.world, "win_condition_hypotheses", []) or []:
                if getattr(h, "promoted", False) or \
                        float(getattr(h, "credence", 0.0) or 0.0) >= 0.6:
                    return True
        except Exception:
            pass
        return False

    def _is_stuck(self, n: int = 7, threshold: int = 3) -> bool:
        """True when the last several actions produced no board progress -- the
        signal that COS should pivot from grinding to EXPLORING salient unknowns.
        Uses the same no-progress-streak the strategy layer computes."""
        try:
            from vlm_strategy import _stuck_streaks               # noqa: E402
            no_progress, _ = _stuck_streaks(self.world, n)
            return no_progress >= threshold
        except Exception:
            return False

    def _monitor_progress(self) -> None:
        """SELF-CORRECTING trend monitor.  Each turn, measure progress scalars the
        actor expects to TREND -- distance-to-goal should SHRINK, and a mover you
        are driving toward a goal should travel ~STRAIGHT from its start to that
        goal (perpendicular deviation ~0).  When the TREND is violated (a stall,
        or a systematic drift), AUTO-surface the decompose-and-verify protocol
        into the next turn -- WITHOUT being asked.  This is the gap su15 exposed:
        every step had the right polarity (the mover DID move), so the single-step
        surprise check never fired while the mover quietly veered off the aim
        line.  Game-agnostic (centroids + a straight-line expectation; no game
        specifics) and fully guarded -- runs no matter how the actor is driven."""
        try:
            turn = int(getattr(self.world, "turn", 0) or 0)
            ents = getattr(self.world, "entities", {}) or {}

            def ctr(bbox):
                r0, c0, r1, c1 = bbox[:4]
                return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

            info = {}
            for name, rec in ents.items():
                bh = getattr(rec, "bbox_history", None)
                if not bh:
                    continue
                now, start = ctr(bh[-1][1]), ctr(bh[0][1])
                moved_now = (len(bh) >= 2 and bh[-1][0] == turn
                             and ctr(bh[-1][1]) != ctr(bh[-2][1]))
                info[name] = {"now": now, "start": start, "moved": moved_now,
                              "disp": _math.hypot(now[0] - start[0], now[1] - start[1])}
            if len(info) < 2:
                return
            # MOVER: the entity moving this turn (persist once chosen); else max-displaced.
            mover = self._progress_mover
            if mover not in info:
                moved = [n for n, i in info.items() if i["moved"]]
                mover = moved[0] if moved else max(info, key=lambda n: info[n]["disp"])
            if info[mover]["disp"] < 1.0:        # nothing has really moved yet
                return
            self._progress_mover = mover
            # TARGET: the most STATIC other entity (a goal you move toward stays put).
            others = [(n, i) for n, i in info.items() if n != mover]
            tname, tinfo = min(others, key=lambda ni: ni[1]["disp"])
            if tinfo["disp"] > 2.0:              # no clearly-static target -> no metrics
                return
            m, s, tg = info[mover]["now"], info[mover]["start"], tinfo["now"]
            self._progress_ledger.record(
                "goal_distance", turn, _math.hypot(m[0] - tg[0], m[1] - tg[1]))
            vx, vy = tg[0] - s[0], tg[1] - s[1]
            L = _math.hypot(vx, vy)
            if L > 1e-6:                         # signed perp distance from the start->goal line
                dev = ((m[0] - s[0]) * vy - (m[1] - s[1]) * vx) / L
                self._progress_ledger.record("path_deviation", turn, dev)
            hits = _DBG.scan_progress(
                self._progress_ledger,
                {"goal_distance": "decrease", "path_deviation": "zero"})
            if hits:
                name, verdict = hits[0]
                block = _DBG.progress_protocol_block(name, verdict)
                # Apply the self-correction FIRST (the critical work), so a logging
                # hiccup can never block it.
                self._repeat_feedback = (
                    ((self._repeat_feedback + "\n\n") if self._repeat_feedback else "")
                    + block)
                # keep only the latest sample so the window must REFILL before this
                # metric can fire again -- one loud nudge, not every turn.
                self._progress_ledger.series[name] = self._progress_ledger.series[name][-1:]
                try:                              # log line is best-effort: the em-dash
                    print(f"[self-correct] {block.splitlines()[0]}", flush=True)
                except Exception:                 # in [DEBUG —] can't encode on cp1252 stdout
                    pass
        except Exception:
            pass

    def _instinct_context(self, phase: str, anim_dir=None):
        """Build the TurnContext the instinct triggers read."""
        ad = anim_dir if anim_dir is not None else getattr(
            self, "_last_anim_dir", None)
        n = 0
        if ad is not None:
            try:
                n = len(sorted(Path(ad).glob("frame_*.png")))
            except Exception:
                n = 0
        gi = getattr(self.world, "grid_inference", None) or {}
        is_grid = bool(gi.get("is_grid_based", True)) if isinstance(gi, dict) else True
        has_agent = getattr(self.world, "agent", None) is not None
        return _INST.TurnContext(
            phase=phase, turn_n=getattr(self.world, "turn", 0),
            has_animation=(n >= 2), n_frames=n,
            win_understood=self._win_understood(),
            is_grid=is_grid, has_agent=has_agent,
            stuck=self._is_stuck())

    def _stage_instinct_reprompt(self, turn_dir, inst, ctx, reply_name) -> Path:
        prompt = (inst.reprompt or "").format(n=ctx.n_frames, reply_name=reply_name)
        p = Path(turn_dir) / f"instinct_{inst.name}_prompt.md"
        p.write_text(prompt, encoding="utf-8")
        (Path(turn_dir) / f"instinct_{inst.name}_STATUS.txt").write_text(
            f"WAITING for {inst.name} reply at {turn_dir/reply_name}\n",
            encoding="utf-8")
        return p

    def _enforce_instincts(self, parsed, turn_dir, reply_stub="reply"):
        """Enforce the MANDATORY instincts that FIRE this turn: if the reply does
        not satisfy one, re-prompt ONCE for it.  Generic over instincts.py — a new
        mandatory instinct is enforced automatically once registered."""
        try:
            if not isinstance(parsed, dict):
                return parsed
            ctx = self._instinct_context("delta")
            for inst in _INST.REGISTRY.mandatory_firing(ctx):
                if inst.satisfied is not None and inst.satisfied(parsed):
                    continue                      # already honoured
                if not inst.reprompt:
                    continue
                reply_name = f"{reply_stub}_{inst.name}_reply.txt"
                print(f"[instinct:{inst.name}] reply did not satisfy this "
                      f"mandatory instinct (firing: {inst.when}); re-prompting "
                      f"once.", flush=True)
                self._stage_instinct_reprompt(turn_dir, inst, ctx, reply_name)
                parsed2 = self._poll_named_reply(Path(turn_dir) / reply_name)
                if isinstance(parsed2, dict):
                    parsed = parsed2              # adopt; keep checking others
                else:
                    print(f"[instinct:{inst.name}] no reply (timeout); keeping "
                          f"the prior reply (no halt).", flush=True)
            return parsed
        except Exception as e:
            print(f"[instinct] enforcement skipped ({e})", flush=True)
            return parsed

    def _detect_entities_as_perception(self, frame_path) -> list:
        """Substrate figure-ground -> a minimal perception entity list (no VLM
        interpretation).  Used only by the VLM-timeout fallback so COS keeps
        tracking the scene when the in-loop VLM does not answer.

        Uses the PALETTE-INVARIANT STRUCTURAL extractor
        (silhouette_track.foreground_components), NOT the colour-keyed
        entity_detector: the structural one drops large fields AND repeated
        texture (a checkerboard board) by their spatial footprint, so the
        fallback does not shatter a textured background into dozens of tile
        'entities' (the colour-keyed detector did exactly that on tn36)."""
        import numpy as np
        from PIL import Image
        import silhouette_track as _ST                     # noqa: E402
        im = Image.open(frame_path).convert("RGB")
        if im.size != (64, 64):
            im = im.resize((64, 64), Image.NEAREST)
        out = []
        for i, c in enumerate(_ST.foreground_components(np.array(im))):
            r0, c0, r1, c1 = c["bbox"]                     # inclusive
            out.append({"name": f"obj_{i}",
                        "bbox_ticks_turn1": [int(r0), int(c0),
                                             int(r1) + 1, int(c1) + 1],
                        "role_hypothesis": "unknown", "confidence": "low"})
        return out

    def _fallback_perception(self, turn_n: int) -> dict:
        """Degraded, VLM-free perception returned when the in-loop VLM does not
        answer within the poll timeout.  Strict/competition mode must NOT hang or
        crash on a slow/dead VLM: COS keeps tracking the scene via the substrate
        entity_detector and the never-halt explorer drives the next action.
        goal_candidate_cells is empty so no stale directed plan fires.  Logged
        loudly so a stalling VLM is visible."""
        ents = []
        try:
            if getattr(self, "_last_frame_path", None) is not None:
                ents = self._detect_entities_as_perception(self._last_frame_path)
        except Exception as e:
            print(f"[driver] fallback entity-detect failed ({e}); "
                  f"reusing tracked world entities", flush=True)
        if not ents:
            ents = [{"name": r.name, "bbox_ticks_turn1": list(r.current_bbox),
                     "role_hypothesis": r.current_role or "unknown",
                     "confidence": "low"}
                    for r in self.world.entities.values()
                    if r.current_bbox is not None]
        perc = {"entities": ents, "groups": [], "relationships": [],
                "frame_to_frame_summary": [], "grid_inference": {},
                "symbolic_state": {"agent_cell": None, "goal_candidate_cells": [],
                                   "confidence": "low"},
                "game_type": {}, "game_purpose": {},
                "overall_notes": "(VLM reply timed out -- substrate-only fallback)"}
        self._last_perception_ss = perc["symbolic_state"]
        print(f"[driver] WARNING: VLM reply timed out at turn {turn_n} after "
              f"{self.vlm_timeout_s}s -- continuing with substrate-only perception "
              f"({len(ents)} entities); the run does NOT halt (strict-mode "
              f"robustness).", flush=True)
        return perc

    def _fallback_reply(self, turn_n: int) -> dict:
        """Shape the timeout fallback for poll_reply: the initial poll (turn 1)
        expects the perception schema directly; a delta poll expects
        {delta, perception}."""
        perc = self._fallback_perception(turn_n)
        if turn_n <= 1:
            return perc
        return {"delta": {"agent_moved": False, "agent_new_cell": None,
                          "inferred_action": "?", "entities_appeared": [],
                          "entities_disappeared": [], "entities_changed": [],
                          "summary": "(VLM reply timed out -- no delta observed)"},
                "perception": perc}

    def _poll_path(self, reply_path: Path) -> dict:
        """Generic reply poller for an arbitrary reply file path (used by
        the level-start analysis, which writes into a per-turn
        ``level_start/`` subdir rather than the canonical turn dir)."""
        deadline = time.time() + self.vlm_timeout_s
        print(f"[exploratory-driver] waiting for reply at {reply_path}",
              flush=True)
        while time.time() < deadline:
            if reply_path.exists():
                body = reply_path.read_text(encoding="utf-8").strip()
                if body:
                    parsed = self._consume_reply(reply_path, body)
                    if parsed is None:  # unparsable -> degrade like a timeout
                        return self._fallback_perception(self.world.turn)
                    return self._fulfill_visual_queries(
                        parsed, reply_path.parent, self._last_frame_path,
                        reply_path.stem)
            time.sleep(self.poll_s)
        # VLM did not answer within the timeout — degrade to substrate-only
        # perception rather than crash (strict-mode robustness).
        return self._fallback_perception(self.world.turn)

    # --- the run loop ---

    def run_turn_one(self) -> None:
        frame = self.game.turn_one_frame()
        self._last_frame_path = frame
        self._level_start_frame_path = frame  # bootstrap colors for the live
                                              # bbox refresh from the initial board
        self.world.turn = 1

        # CROSS-GAME PRIORS: load global priors and seed the new
        # game's world.mechanic_hypotheses with promoted
        # observations from prior games BEFORE perception runs.
        # This way the planner can plan on turn 1 using priors
        # (e.g. ACTION1=UP if that was true in N prior games),
        # only re-verifying when the new game contradicts.
        if self.global_priors_path is not None:
            try:
                available = self.game.available_actions()
                self._global_priors, n_seeded = load_and_seed(
                    self.world, available,
                    priors_path=self.global_priors_path,
                )
                if n_seeded:
                    print(f"[turn 1] CROSS-GAME PRIORS: seeded "
                          f"{n_seeded} hypotheses from "
                          f"{self._global_priors.n_games_contributed} "
                          f"prior game(s) -- "
                          f"{self.world.inherited_from}")
                else:
                    n_known = len(self._global_priors.priors)
                    print(f"[turn 1] CROSS-GAME PRIORS: 0 seeded "
                          f"({n_known} priors in store, none "
                          f"matched this game's action set or met "
                          f"the support threshold)")
            except Exception as e:
                print(f"[turn 1] CROSS-GAME PRIORS: load failed "
                      f"({e}); continuing without priors")
                self._global_priors = None

        prompt_path = self.stage_initial_perception(frame)
        print(f"[turn 1] initial-perception prompt: {prompt_path}")
        first_pass = self.poll_reply(1)

        # SECOND PASS — bbox refinement.  Show the VLM its own
        # first-pass bboxes overlaid on the same frame and ask it
        # to correct edge misalignments.  Critical at the upscale
        # the driver currently uses: even at upscale=8, a 4-tick
        # entity is only 32 image pixels wide and the first pass
        # often gets edges +/- 1-2 ticks off.  The refined entities
        # REPLACE the first-pass entities in WorldKnowledge — we do
        # NOT call ingest_perception() on the first pass.
        prev_entities = first_pass.get("entities") or []
        if prev_entities:
            ref_prompt = self.stage_refinement(frame, prev_entities)
            print(f"[turn 1] refinement prompt: {ref_prompt}")
            refined = self.poll_refinement_reply()
            print(f"[turn 1] refinement returned "
                  f"{len(refined.get('entities') or [])} entities")
            refined = self._ground_perception_geometry(
                refined, frame, persist_dir=self.work_dir / "turn_001")
            self.world.ingest_perception(refined)
        else:
            print("[turn 1] no entities in first-pass reply; "
                  "skipping refinement")
            first_pass = self._ground_perception_geometry(
                first_pass, frame, persist_dir=self.work_dir / "turn_001")
            self.world.ingest_perception(first_pass)
        # Save world snapshot
        self.world.save(self.work_dir / "world_knowledge.json")
        # Mirror the trace immediately after the INITIAL perception so a
        # slow interactive run (e.g. human-in-the-loop VLM) shows turn-1
        # analysis right away.  The per-turn loop (run_one_step) only
        # mirrors from turn 2 onward, which leaves the canonical trace
        # bookmark stale/empty for the whole of turn 1 -- long enough to
        # look like a different (previous) run.  Force past the debounce.
        try:
            self._maybe_mirror_trace(min_interval_s=0.0)
        except Exception as e:
            print(f"[trace-mirror] init refresh failed: {e}")

    # ------------------------------------------------------------------
    # Cross-level entity consistency (strong same-game priors)
    # ------------------------------------------------------------------
    def _level_memory_path(self) -> Path:
        # LevelMemory is per-GAME, not per-level, so it lives in the work-dir's
        # PARENT (shared across every <game>_lcN run) rather than under one
        # level's work-dir.  This is what lets a fast-forwarded / relaunched
        # run on a LATER level load the digest + templates crystallised when an
        # EARLIER level was solved in a previous session.
        return self.work_dir.parent / f"level_memory_{self.world.game_id}.json"

    def _legacy_level_memory_path(self) -> Path:
        # Where LevelMemory used to be written (under the per-level work-dir);
        # read as a fallback so pre-existing data is not lost on upgrade.
        return self.work_dir / f"level_memory_{self.world.game_id}.json"

    def _ensure_level_memory(self):
        """Lazy-init the per-game LevelMemory (cross-trial if a file exists).
        Prefers the shared per-game path; falls back to the legacy per-work-dir
        file so data written before the path change still loads."""
        if getattr(self, "_level_memory", None) is not None:
            return self._level_memory
        from level_memory import LevelMemory                  # noqa: E402
        for cand in (self._level_memory_path(), self._legacy_level_memory_path()):
            try:
                if cand.exists():
                    self._level_memory = LevelMemory.load(cand)
                    return self._level_memory
            except Exception:
                continue
        self._level_memory = LevelMemory(game_id=self.world.game_id)
        return self._level_memory

    def _snapshot_entity_inventory(self) -> list:
        """The current world's entity inventory (name/role/appearance/bbox) —
        captured at level start it is the JUST-FINISHED level's set, the prior
        the next level inherits."""
        inv = []
        for rec in self.world.entities.values():
            inv.append({
                "name": rec.name,
                "role": rec.current_role or "unknown",
                "appearance": (rec.appearance or "").strip(),
                "bbox": rec.current_bbox,
            })
        return inv

    def _format_prior_entity_context(self, prior_inv: list) -> str:
        """Build the strong-priors block: prior-level inventory + accumulated
        cross-level templates + the reuse instruction. Empty string if there is
        nothing to carry (the harness REMEMBERS and SURFACES; the VLM decides
        how to apply it)."""
        lm = getattr(self, "_level_memory", None)
        templates = list(getattr(lm, "templates", []) or [])
        digest = lm.latest_digest() if lm is not None else None
        if not prior_inv and not templates and digest is None:
            return ""
        lines = [
            "Within THIS game, levels are STRONGLY related: an entity that looks "
            "the same almost always plays the SAME ROLE. Treat the following as "
            "strong priors — reuse the prior name and role for an entity that "
            "matches by appearance; deviate only with clear visual evidence.",
        ]
        if digest is not None:
            lines.append(
                "\nCARRIED MECHANIC DIGEST from the previous level (a STRONG, "
                "assume-stable prior — verify cheaply on THIS level before trusting; "
                "the win condition is almost certainly the SAME mechanic re-skinned):")
            if digest.game_purpose:
                lines.append(f"  - game purpose: {digest.game_purpose}")
            if digest.win_condition:
                lines.append(f"  - WIN CONDITION (conf {digest.win_condition_confidence:.2f}): "
                             f"{digest.win_condition}")
            if digest.entity_roles:
                lines.append("  - roles to look for (role-pattern -> role): "
                             + "; ".join(f"{k} -> {v}" for k, v in list(digest.entity_roles.items())[:12]))
            for h in (digest.mechanic_hints or [])[:8]:
                lines.append(f"  - hint (conf {h.get('confidence',0):.2f}): {h.get('hint','')}")
            lines.append(
                "  -> MAP the new scene onto these roles, then pursue the carried win "
                "plan directly; only fall back to broad probing if the carried "
                "mechanic clearly does NOT hold here.")
        if prior_inv:
            lines.append("\nEntities on the PREVIOUS level "
                         "(name — role — appearance — last bbox):")
            for e in prior_inv:
                lines.append(f"  - {e['name']} — {e['role']} — "
                             f"{e['appearance'] or '?'} — {e['bbox']}")
        if templates:
            lines.append("\nCross-level entity templates "
                         "(appearance-signature -> role; seen in level(s)):")
            for t in templates[:24]:
                lines.append(f"  - {t.appearance_signature or '?'} -> "
                             f"{t.canonical_role} (name like "
                             f"{t.canonical_name_pattern}; levels "
                             f"{t.observed_in_levels})")
        lines.append(
            "\nWhen perceiving THIS level: REUSE the prior name+role for each "
            "matching entity; ADD any genuinely new entity; and NOTE in "
            "overall_notes which prior entities are absent or changed.")
        return "\n".join(lines)

    def _crystallize_level_digest(self, ls_dir: Path) -> None:
        """END-OF-LEVEL CRYSTALLISATION (VLM-authored).  A level was just solved;
        before the world is reset for the next level, ask the VLM to author ONE
        considered, reusable MECHANIC DIGEST -- roles, the win condition, and
        free-form mechanic hints (uncertainty OK).  A dedicated end-of-level
        summary is stable, where mid-play claim credences fluctuate.  Stored in
        LevelMemory (and carried as a strong, assume-stable prior into the next
        level: surfaced in the prompt + seeded as a high-credence win hypothesis).
        Role-keyed / pixel-free so it also primes structurally-similar games.
        Defensive: never breaks the level transition."""
        try:
            w = self.world
            ents = {n: (getattr(r, "current_role", "") or "unknown")
                    for n, r in (getattr(w, "entities", {}) or {}).items()}
            _wg = getattr(w, "groups", {}) or {}
            grps = {getattr(g, "name", "?"): list(getattr(g, "members", []) or [])
                    for g in (_wg.values() if isinstance(_wg, dict) else _wg)}
            rels = [f"{getattr(r,'from_entity','?')} {getattr(r,'relation','?')} "
                    f"{getattr(r,'to_entity','?')}" for r in (getattr(w, "relationships", []) or [])]
            mechs = [f"{getattr(h,'trigger','?')} -> {getattr(h,'effect','?')} "
                     f"(cred {float(getattr(h,'credence',0) or 0):.2f})"
                     for h in (getattr(w, "mechanic_hypotheses", []) or [])]
            wins = [f"{getattr(h,'description','')} (cred {float(getattr(h,'credence',0) or 0):.2f})"
                    for h in (getattr(w, "win_condition_hypotheses", []) or [])]
            lm = self._ensure_level_memory()
            wpath = lm.winning_paths[-1].actions if lm.winning_paths else []
            summary = json.dumps({
                "finished_level_ordinal": getattr(self, "_level_ordinal", 0),
                "entities_by_role": ents, "groups": grps, "relationships": rels,
                "game_type_guess": getattr(w, "game_type_guess", ""),
                "game_purpose_guess": getattr(w, "game_purpose_guess", ""),
                "mechanic_hypotheses": mechs, "win_condition_hypotheses": wins,
                "winning_action_sequence": wpath,
            }, indent=2)
            prompt = (
                f"# END-OF-LEVEL CRYSTALLISATION -- game `{self.game.game_id}`\n\n"
                "Model handle: human:claude\n\n"
                "You just SOLVED a level.  Levels of one ARC-AGI-3 game are ALWAYS "
                "related, so before the next level, CRYSTALLISE the reusable "
                "understanding so the next level (and structurally-similar games) "
                "start INFORMED, not from scratch.  Author ONE considered summary.  "
                "It is fine to be UNCERTAIN -- a low-confidence hint still PRIMES the "
                "next level (it is a prior to verify cheaply, not an asserted fact).\n\n"
                "Keep everything ROLE-KEYED and PIXEL-FREE (name things by their ROLE "
                "and function, never by exact colour/position) so it survives a "
                "re-skin and transfers across similar games.\n\n"
                "What you learned this level (substrate summary):\n```json\n"
                f"{summary}\n```\n\n"
                "Return ONLY this JSON object (no prose outside it):\n"
                "{\n"
                '  "game_purpose": "<one line: what the game is fundamentally about>",\n'
                '  "win_condition": {"statement": "<what makes the level advance/win, '
                "named by ROLE -- e.g. \\'configure the active program so the fired "
                "mover reaches the goal\\'>\", \"confidence\": 0.0},\n"
                '  "entity_roles": {"<role pattern / appearance phrase>": "<program|'
                'trigger|scene|goal|reference|control|hud|...>"},\n'
                '  "mechanic_hints": [{"hint": "<a reusable observation worth carrying '
                "-- e.g. 'a panel of repeated marks is a settable program; each column "
                "= one program step; firing the trigger runs it and the scene mover "
                'responds\'>", "confidence": 0.0}]\n'
                "}\n"
            )
            (ls_dir / "crystallize_prompt.md").write_text(prompt, encoding="utf-8")
            (ls_dir / "STATUS.txt").write_text(
                f"WAITING for END-OF-LEVEL crystallisation reply at "
                f"{ls_dir / 'crystallize_reply.txt'}\n", encoding="utf-8")
            print(f"[crystallise] end-of-level digest prompt: "
                  f"{ls_dir / 'crystallize_prompt.md'}")
            reply = self._poll_path(ls_dir / "crystallize_reply.txt")
            dg = lm.ingest_digest(reply, level=getattr(self, "_level_ordinal", 0) - 1,
                                  turn=int(getattr(w, "turn", 0) or 0))
            print(f"[crystallise] digest authored: win='{dg.win_condition[:60]}' "
                  f"(conf {dg.win_condition_confidence:.2f}), "
                  f"{len(dg.entity_roles)} roles, {len(dg.mechanic_hints)} hints")
            # transferable (cross-game/domain) store: symbolic, role-keyed digest
            self._persist_digest_cross_game(dg)
        except Exception as e:
            print(f"[crystallise] skipped ({e})")

    def _generalize_homogeneous_group_functions(self) -> int:
        """Characterise a SIMILARITY GROUP by a SAMPLE, not by exhaustion.  Once
        >= 2 members of a same-appearance group have RESOLVED function-probes and
        they AGREE (all settable, or all inert), generalise that outcome to the
        group's remaining members -- pre-confirm their function so they are NOT
        probed one-by-one.  Game-agnostic: a homogeneous group needs sampling, not
        a full sweep.  This is what stops COS grinding all 15 reference switches
        and all 12 control switches individually.  2 is the minimum sample that can
        evidence agreement (1 cannot); a member that later differs simply re-opens
        via the normal delta path, and for an identical-appearance group homogeneity
        is a strong prior.  Returns the count generalised."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return 0
        sig = self._level_signature()
        turn = int(getattr(self.world, "turn", 0) or 0)
        _g = getattr(self.world, "groups", {}) or {}
        n = 0
        for g in (_g.values() if isinstance(_g, dict) else _g):
            mem = list(getattr(g, "members", []) or [])
            if len(mem) < 3:
                continue
            statuses = {}
            for m in mem:
                c = cs.claims.get(f"function_of_{m}")
                if c is not None and c.status in ("proven", "refuted"):
                    statuses[m] = c.status
            if len(statuses) < 2 or len(set(statuses.values())) != 1:
                continue                       # too few resolved, or heterogeneous
            outcome = next(iter(set(statuses.values())))
            for m in mem:
                c = cs.claims.get(f"function_of_{m}")
                if c is not None and c.status in ("proven", "refuted"):
                    continue
                cs.ingest([{"id": f"function_of_{m}",
                            "statement": f"generalised from homogeneous group "
                                         f"'{getattr(g, 'name', '?')}' ({outcome})",
                            "scope": "level", "target": [m], "cost": 1,
                            "importance": 0.1, "provenance": "generalized",
                            "credence": 0.7}], turn=turn, level_signature=sig)
                cc = cs.claims.get(f"function_of_{m}")
                if cc is not None:
                    cc.status = outcome
                    # re-probeable but low-priority (see the re-pricing in
                    # _ensure_function_coverage): a sampled generalisation can be
                    # wrong, so allow a cheap re-verify below unknown exploration.
                    cc.needs_recheck = True
                    cc.credence = 0.7
                    n += 1
        if n:
            cs.save_for_game()
            print(f"[generalise] {n} group-member function(s) inferred from a "
                  f"homogeneous sample (not probed individually).")
        return n

    def _snapshot_proven_functions(self) -> set:
        """Entity names whose ``function_of_<name>`` claim is PROVEN right now --
        captured BEFORE the level reset drops level-scoped claims, so the per-entity
        carry can recognise a recurring entity by name on the next level."""
        cs = getattr(self, "_claim_store", None)
        out: set = set()
        if cs is None:
            return out
        for cid, c in cs.claims.items():
            if cid.startswith("function_of_") and getattr(c, "status", "") == "proven":
                for t in (getattr(c, "target", None) or []):
                    out.add(t)
        return out

    def _preconfirm_carried_functions(self) -> None:
        """Carry per-entity FUNCTION knowledge across levels.  An entity whose
        function was established on a PRIOR level -- the trigger (disc/button), the
        mover, the goal -- must NOT be re-discovered from scratch on the new level
        (levels of one game are related).  Pre-confirm its ``function_of_<name>``
        claim as PROVEN + needs_recheck (assume-stable; the win-pursuit re-verifies
        it cheaply in context) so the prober does not waste turns re-probing it.

        Carried-function signals (game-agnostic): the same entity NAME recurs and
        had a proven function before; OR its appearance matches a cross-level
        TEMPLATE with a known role; OR perception assigned it a recognised mechanic
        ROLE (mapped from the carried digest).  CRUCIAL EXCEPTION: an entity that
        is a member of a sizeable similarity group (a repeated-mark control panel)
        is LEFT to be probed -- its settable-vs-inert RESPONSE is exactly the
        active-vs-reference signal the new level still needs (do not suppress it).
        Guarded; only runs when a digest was carried."""
        try:
            cs = getattr(self, "_claim_store", None)
            lm = getattr(self, "_level_memory", None)
            if cs is None or lm is None or lm.latest_digest() is None:
                return
            sig = self._level_signature()
            turn = int(getattr(self.world, "turn", 0) or 0)
            context_roles = {"hud", "background", "bg", "wall", "floor", "border", "frame"}
            mechanic_roles = {"trigger", "program", "control", "scene", "mover",
                              "goal", "reference", "switch", "button", "trigger_target"}
            grouped = set()
            _g = getattr(self.world, "groups", {}) or {}
            for g in (_g.values() if isinstance(_g, dict) else _g):
                m = list(getattr(g, "members", []) or [])
                if len(m) >= 3:
                    grouped.update(m)
            n = 0
            for name, rec in (getattr(self.world, "entities", {}) or {}).items():
                role = (getattr(rec, "current_role", "") or "").lower()
                if role in context_roles or getattr(rec, "current_bbox", None) is None:
                    continue
                if name in grouped:
                    continue                       # repeated-mark panel: must be probed
                if cs.has_confirmed_function(name, sig):
                    continue
                prior = cs.claims.get(f"function_of_{name}")
                name_recurs = (name in getattr(self, "_carried_proven_functions", set())
                               or (prior is not None and prior.status == "proven"))
                tpl = lm.match_template(getattr(rec, "appearance", "") or "")
                appr = tpl is not None and (getattr(tpl, "canonical_role", "") or "").lower() not in ("", "unknown")
                role_carried = role in mechanic_roles
                if not (name_recurs or appr or role_carried):
                    continue
                why = ("name-recurs" if name_recurs else
                       ("appearance->" + tpl.canonical_role if appr else f"role={role}"))
                cs.ingest([{
                    "id": f"function_of_{name}",
                    "statement": f"carried function of '{name}' ({why}); assume-stable, verify cheaply",
                    "scope": "level", "target": [name], "cost": 1,
                    "importance": 0.15, "provenance": "carried", "credence": 0.8,
                }], turn=turn, level_signature=sig)
                c = cs.claims.get(f"function_of_{name}")
                if c is not None:
                    # CROSS scope so the carried function is active on THIS level
                    # regardless of which level first proved it (ingest leaves an
                    # existing claim's original level_signature untouched).
                    c.scope = "cross"
                    c.status = "proven"
                    # Assume-stable but NOT immutable: a later level may redesign an
                    # entity's function, so keep it re-probeable (needs_recheck) --
                    # the re-pricing above puts that re-verify well BELOW exploring
                    # unknown entities, so it never crowds out new discovery.
                    c.needs_recheck = True
                    c.credence = 0.8
                    n += 1
            if n:
                cs.save_for_game()
                print(f"[carry] pre-confirmed {n} entity function(s) from prior-level "
                      f"knowledge (assume-stable; not re-probed). Repeated-mark panels "
                      f"left to probe for the active-vs-reference signal.")
        except Exception as e:
            print(f"[carry] function pre-confirm skipped ({e})")

    def _persist_digest_cross_game(self, dg) -> None:
        """Store the role-keyed digest in the cross-game KB so a structurally
        similar FUTURE game can recall it as hints (foundation for cross-game /
        cross-domain transfer).  Best-effort; never breaks the run."""
        try:
            from kb_paths import kb_root                       # noqa: E402
            import json as _json
            root = Path(kb_root())
            root.mkdir(parents=True, exist_ok=True)
            p = root / "cross_game_digests.json"
            store = {}
            if p.exists():
                store = _json.loads(p.read_text(encoding="utf-8"))
            rec = {"game_id": self.game.game_id, "level": dg.level,
                   "game_purpose": dg.game_purpose, "win_condition": dg.win_condition,
                   "win_condition_confidence": dg.win_condition_confidence,
                   "entity_roles": dg.entity_roles, "mechanic_hints": dg.mechanic_hints}
            store.setdefault(self.game.game_id, [])
            store[self.game.game_id].append(rec)
            p.write_text(_json.dumps(store, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[crystallise] cross-game persist skipped ({e})")

    def _surface_coupling_route(self, parsed, frame_path) -> None:
        """Compute the COUPLING-AWARE route via path_planning (perception ->
        navigable map -> mover-mates-goal route) and surface it as the target the
        program must realise -- the 'encoding list' (the ordered move sequence).
        Stored on self._coupling_route for the planner / cx-OFAT to aim at, and
        logged.  Game-agnostic; fully guarded -- never blocks the run."""
        self._coupling_route = None
        try:
            import numpy as _np                            # noqa: E402
            from PIL import Image as _Image                # noqa: E402
            import path_planning as _pp                     # noqa: E402
            if frame_path is None or not Path(frame_path).exists():
                return
            frame = _np.array(_Image.open(frame_path).convert("RGB"))
            res = _pp.route_from_entities(frame, parsed.get("entities") or [])
            if not res or not res.get("route"):
                return
            self._coupling_route = res
            cp = res.get("coupling") or {}
            print(f"[path] coupling route {res.get('mover')} -> {res.get('goal')} "
                  f"(couple={cp.get('coupled')}, dock={cp.get('dock_cell')}, "
                  f"mate={cp.get('mating_move')}); ENCODING LIST the program must "
                  f"realise = {res['route']}")
        except Exception as e:
            print(f"[path] coupling-route skipped ({e})")

    def _surface_group_distinctness(self, parsed, frame_path) -> None:
        """For each perceived GROUP of similar elements (a candidate key / legend /
        control set / palette), decode each member's STATIC content and report
        whether they are mutually DISTINCT -- the IDENTITY evidence to read BEFORE
        probing.  Distinct members => the group ENCODES distinct values (a
        key/index); a click that produces no change can never make them
        equivalent.  Stored on self._group_distinctness for the refute guard, and
        logged.  Game-agnostic; fully guarded -- never blocks the run."""
        self._group_distinctness = {}
        try:
            import numpy as _np                            # noqa: E402
            from PIL import Image as _Image                # noqa: E402
            import identity_evidence as _ie                 # noqa: E402
            if frame_path is None or not Path(frame_path).exists():
                return
            groups = parsed.get("groups") or []
            ents = {e.get("name"): e for e in (parsed.get("entities") or [])}
            if not groups or not ents:
                return
            frame = _np.array(_Image.open(frame_path).convert("RGB"))
            for g in groups:
                members = []
                for nm in (g.get("members") or []):
                    e = ents.get(nm)
                    bb = e.get("bbox_ticks_turn1") if e else None
                    if bb:
                        members.append({"name": nm, "bbox": bb})
                if len(members) < 2:
                    continue
                r = _ie.group_distinctness(frame, members)
                self._group_distinctness[g.get("name")] = r
                if r.get("all_distinct"):
                    print(f"[identity] group '{g.get('name')}' -> {r['n_members']} members "
                          f"ALL STATICALLY DISTINCT (differ in {r.get('differs_in')}) => it "
                          f"ENCODES distinct values (a key/index). Read THOSE channel(s); do "
                          f"NOT infer from click-responses or one channel alone.")
                elif r.get("differs_in"):
                    print(f"[identity] group '{g.get('name')}' -> members differ in "
                          f"{r.get('differs_in')} (read THOSE channel(s), not just shape).")
        except Exception as e:
            print(f"[identity] group-distinctness skipped ({e})")

    def _refute_is_discriminating(self, cs, claim_id, outcome) -> bool:
        """A refutation must be DISCRIMINATING.  Block (return False) a REFUTE of a
        claim whose target is a member of a group the substrate measured as
        statically DISTINCT -- you cannot refute 'these encode distinct values'
        from an interaction-response when the static decode shows they do.  All
        other closes are discriminating (True).  Guarded."""
        if outcome != "refuted":
            return True
        try:
            c = cs.claims.get(claim_id)
            gd = getattr(self, "_group_distinctness", {}) or {}
            if c is None or not gd:
                return True
            tgt = set(c.target or [])
            for gname, r in gd.items():
                members = {m.get("name") for m in (r.get("members") or [])}
                if r.get("all_distinct") and (tgt & members):
                    print(f"[identity] BLOCKED a non-discriminating REFUTE of '{claim_id}': "
                          f"group '{gname}' members are statically DISTINCT (a key) -- read "
                          f"the static content, not a click-response. Claim stays OPEN.")
                    return False
        except Exception:
            return True
        return True

    def _surface_skills(self) -> None:
        """RECOGNISE which high-level skills apply to the current scene and surface
        the recipe to the VLM (logged + stored on the Scene State for the views).
        The trigger context is read from the measured scene -- a distinct-symbol KEY
        group + an editable panel => follow-the-instructions.  Game-agnostic; guarded."""
        try:
            import skill_library as _sl                    # noqa: E402
            gd = getattr(self, "_group_distinctness", {}) or {}
            ents = (getattr(self, "_scene", None).entities
                    if getattr(self, "_scene", None) else {}) or {}
            ctx = {
                "has_distinct_key_group": any((r or {}).get("all_distinct") for r in gd.values()),
                "has_editable_panel": any(
                    "panel" in str(getattr(e, "role", "") or getattr(e, "id", "")).lower()
                    or "switch" in str(getattr(e, "id", "")).lower()
                    or getattr(e, "kind", None) == "control"
                    for e in ents.values()),
            }
            skills = _sl.REGISTRY.applicable(ctx)
            if not skills:
                return
            print(f"[skill] applicable: {[s.name for s in skills]} (ctx={ctx})")
            if getattr(self, "_scene", None) is not None:
                self._scene.active_skill = _sl.REGISTRY.render(ctx)
                self._persist_scene()
        except Exception as e:
            print(f"[skill] surface skipped ({e})")

    def _sync_scene(self, frame_path=None) -> None:
        """Populate/refresh the canonical Scene State from the world's grounded
        entities and PERSIST it (scene_state.json + frame.png) for the live views.
        Top-level bboxes refresh each turn; resolved children + VLM refinements are
        preserved.  Game-agnostic; guarded."""
        try:
            import scene_state as _ss                      # noqa: E402
            if getattr(self, "_scene", None) is None:
                self._scene = _ss.SceneState()
            sc = self._scene
            sc.frame_id = int(getattr(self.world, "turn", 0) or 0)
            for name, rec in (getattr(self.world, "entities", {}) or {}).items():
                bb = getattr(rec, "current_bbox", None)
                if bb is None:
                    continue
                bb = [int(v) for v in bb]
                role = (getattr(rec, "role", None)
                        or (getattr(rec, "role_history", None) or [None])[-1])
                click = [int((bb[1] + bb[3]) // 2), int((bb[0] + bb[2]) // 2)]
                if name in sc.entities:
                    e = sc.entities[name]
                    # refresh top-level geometry -- but NEVER clobber a VLM
                    # `correct` (provenance='corrected'), else the refinement loop's
                    # bbox fixes are reverted every turn.
                    if e.parent is None and e.provenance != "corrected":
                        e.bbox, e.click = bb, click
                        if role:
                            e.role = role
                else:
                    kind = ("field" if role and any(k in str(role)
                            for k in ("field", "scenery", "background")) else "region")
                    sc.add(_ss.Entity(id=name, bbox=bb, click=click, kind=kind,
                                      role=role, provenance="measured"))
            self._persist_scene(frame_path)
        except Exception as e:
            print(f"[scene] sync skipped ({e})")

    def _persist_scene(self, frame_path=None) -> None:
        try:
            import json as _json, shutil as _sh          # noqa: E402
            if getattr(self, "_scene", None) is None:
                return
            snap = _json.dumps(self._scene.snapshot())
            (self.work_dir / "scene_state.json").write_text(snap, encoding="utf-8")
            # per-turn STATE-SPACE SNAPSHOT history (archived next to the trace).
            hist = self.work_dir / "scene_history"
            hist.mkdir(exist_ok=True)
            (hist / f"scene_{int(self._scene.frame_id):04d}.json").write_text(snap, encoding="utf-8")
            fp = frame_path or getattr(self, "_last_frame_path", None)
            if fp and Path(fp).exists():
                _sh.copyfile(fp, self.work_dir / "frame.png")
        except Exception as e:
            print(f"[scene] persist skipped ({e})")

    def _apply_scene_ops(self, parsed, frame_path=None) -> None:
        """Apply the VLM's REFINEMENT ops (reply `scene_ops`) to the Scene State:
        resolve (lazy detail), tighten (re-measure), correct/relabel/verify/
        flag_inspect/dismiss.  The VLM drives quality + detail + recovery by id;
        it never supplies pixels.  Guarded."""
        try:
            ops = parsed.get("scene_ops") if isinstance(parsed, dict) else None
            sc = getattr(self, "_scene", None)
            if not ops or sc is None:
                return
            import numpy as _np                            # noqa: E402
            from PIL import Image as _Image                # noqa: E402
            fp = frame_path or getattr(self, "_last_frame_path", None)
            frame = (_np.array(_Image.open(fp).convert("RGB"))
                     if (fp and Path(fp).exists()) else None)
            n = 0
            for op in ops:
                if not isinstance(op, dict):
                    continue
                k, i = op.get("op"), op.get("id")
                if k == "resolve" and frame is not None:
                    sc.resolve(i, frame, into=op.get("into", "grid"))
                elif k == "tighten" and frame is not None:
                    sc.tighten(i, frame)
                elif k == "correct":
                    sc.correct(i, bbox=op.get("bbox"), role=op.get("role"))
                elif k == "relabel":
                    sc.relabel(i, op.get("role"))
                elif k == "verify":
                    sc.verify(i)
                elif k == "flag_inspect":
                    sc.flag_inspect(i, op.get("why", ""))
                elif k == "dismiss":
                    sc.dismiss(i)
                else:
                    continue
                n += 1
            if n:
                print(f"[scene] applied {n} VLM refinement op(s)")
                self._persist_scene(fp)
        except Exception as e:
            print(f"[scene] ops skipped ({e})")

    def run_level_start_analysis(self, frame_path: Path,
                                  crystallize: bool = True) -> None:
        """Fresh ENTITY ANALYSIS at the start of a NEW level.

        ``crystallize=False`` is used for a FAST-FORWARD entry (replay_prefix):
        there is no in-session just-finished level to crystallise / snapshot /
        reset, so those steps are skipped; the carried digest is already on disk
        (loaded LevelMemory) and is surfaced + seeded below exactly as normal.

        A level advance resets the board to a new layout — new structures
        (walls / rails / shafts) and a new piece arrangement.  Without a fresh
        discovery pass the entity inventory is stale (it keeps tracking the
        previous level's set), so structures the new level introduces are never
        tracked — and a structure that is never an entity can never be reasoned
        about (e.g. as the *cause* of a blocked move; see
        SPEC_vlm_backward_reasoning.md § Causal attribution).

        Runs the same two-pass discovery (extract → refine) as turn one, but on
        the new level's frame, into a per-turn ``level_start/`` subdir so it
        does not collide with the turn's delta-perception files.  Re-grounds
        ``world.entities`` and writes ``level_start_analysis.json`` for the
        trace.  Defensive callers wrap this; it must not break the step."""
        # A level advance reached a NEW level -> bump the scoping ordinal BEFORE
        # the reset/carry below, so claims are scoped + carried against the new
        # level's signature (the live adapter leaves self.game.level at 0).
        self._level_ordinal = getattr(self, "_level_ordinal", 0) + 1
        # one PAST the transition turn, so the transition delta (the prior
        # level's winning action) is excluded from this level's cx delta scan.
        self._level_start_turn = int(getattr(self.world, "turn", 0) or 0) + 1
        ls_dir = (self.work_dir / f"turn_{self.world.turn:03d}" / "level_start")
        ls_dir.mkdir(parents=True, exist_ok=True)
        # Save the RAW new-level frame so the trace renders the per-level entity
        # overlay on THIS level's board.  Without it the renderer falls back to
        # the transition turn's frame (the JUST-FINISHED level), so e.g. the lc=1
        # entity image showed the lc=0 board.
        try:
            import shutil as _sh
            _sh.copy2(frame_path, ls_dir / "frame.png")
        except Exception:
            pass

        # Record this level's start frame for the live bbox refresh, and
        # clear the per-entity color cache so colors are re-bootstrapped from
        # THIS level's board (a new level can recolor/relayout pieces). See
        # frame_bbox_refresh.
        self._level_start_frame_path = frame_path
        try:
            import frame_bbox_refresh as _fbr                     # noqa: E402
            _fbr._COLOR_CACHE.clear()
        except Exception:
            pass

        # --- END-OF-LEVEL CRYSTALLISATION: VLM authors the reusable digest of the
        # just-finished level (roles + win condition + mechanic hints) BEFORE the
        # world is reset, so it can be carried forward as a strong prior. ---
        # Skipped on a fast-forward entry (no in-session level just finished; the
        # digest is already on disk from the session that solved the prior level).
        if crystallize:
            self._crystallize_level_digest(ls_dir)
            # Patch the freshly-crystallised mechanic digest into the just-finished
            # level's durable archive record (it was written at solve, before the
            # digest existed).  just-finished level ordinal = score - 1.  Guarded.
            try:
                import game_archive, dataclasses as _dc
                dg = self._ensure_level_memory().latest_digest()
                if dg is not None and getattr(self.world, "score", None):
                    dgd = _dc.asdict(dg) if _dc.is_dataclass(dg) else dict(getattr(dg, "__dict__", {}))
                    game_archive.update_digest(self.world.game_id,
                                               int(self.world.score) - 1, dgd)
            except Exception as e:
                print(f"[archive] digest patch skipped ({e})")
            # Snapshot which entities had a PROVEN function on the just-finished
            # level NOW -- before _reset_level_scoped_state drops those
            # level-scoped claims -- so the per-entity carry can recognise a
            # recurring entity (e.g. the disc) by name and not re-probe it.
            self._carried_proven_functions = self._snapshot_proven_functions()

        # --- CROSS-LEVEL CONSISTENCY: carry strong same-game priors ---
        # At this point world.entities still holds the JUST-FINISHED level's
        # inventory (the new frame is not yet perceived). Snapshot it + promote
        # its entity templates into the per-game LevelMemory, then surface both
        # to the VLM as strong priors. The harness REMEMBERS + SURFACES; the VLM
        # decides how to reuse names/roles (consistent declarations across
        # levels). Defensive — never break the analysis.
        prior_ctx = ""
        try:
            self._ensure_level_memory()
            prior_inv = self._snapshot_entity_inventory()
            try:
                self._level_memory.promote_from_world(self.world, win=True)
            except Exception as e:
                print(f"[level-start] template promotion skipped ({e})")
            prior_ctx = self._format_prior_entity_context(prior_inv)
            if prior_ctx:
                print(f"[level-start] carrying {len(prior_inv)} prior entit(ies)"
                      f" + {len(self._level_memory.templates)} template(s) as "
                      f"strong priors into level {self.game.level}")
        except Exception as e:
            print(f"[level-start] prior-entity context skipped ({e})")

        # --- pass 1: initial extraction (prior-aware when we have history) ---
        self._gridded_for(frame_path, ls_dir / "image_grid.png")
        sys_text, user_text = _fmt_prompts(
            self.n_ticks, label_stride=self.label_stride,
            single_frame=True, game_id=self.game.game_id,
        )
        if prior_ctx:
            prompt = LEVEL_START_PERCEPTION_PROMPT.format(
                game_id=self.game.game_id, level=self.game.level,
                sys_text=sys_text, user_text=user_text,
                n_ticks=self.n_ticks, label_stride=self.label_stride,
                prior_context=prior_ctx, reply_name="reply.txt",
            )
        else:
            prompt = INITIAL_PERCEPTION_PROMPT.format(
                game_id=self.game.game_id, level=self.game.level,
                sys_text=sys_text, user_text=user_text,
                n_ticks=self.n_ticks, label_stride=self.label_stride,
                reply_name="reply.txt",
            )
        (ls_dir / "prompt.md").write_text(prompt, encoding="utf-8")
        (ls_dir / "STATUS.txt").write_text(
            f"WAITING for LEVEL-START perception reply at "
            f"{ls_dir / 'reply.txt'}\n", encoding="utf-8")
        print(f"[level-start] new-level entity analysis prompt: "
              f"{ls_dir / 'prompt.md'}")
        first = self._poll_path(ls_dir / "reply.txt")

        # --- pass 2: bbox refinement (same as turn one) ---
        prev_entities = first.get("entities") or []
        final = first
        if prev_entities:
            overlay = render_turn1_overlay(
                frame_path, prev_entities, n_ticks=self.n_ticks,
                upscale=self.upscale, bbox_line_width=2,
                grid_line_width_major=self.grid_line_width_major,
                grid_line_width_minor=self.grid_line_width_minor,
                grid_major_alpha=self.grid_major_alpha,
                grid_minor_alpha=self.grid_minor_alpha,
            )
            overlay.save(ls_dir / "refinement_overlay.png")
            ref_text = REFINEMENT_PROMPT_TEMPLATE.format(n_ticks=self.n_ticks)
            ref_prompt = REFINEMENT_PERCEPTION_PROMPT.format(
                game_id=self.game.game_id, level=self.game.level,
                sys_text=sys_text, ref_text=ref_text,
                n_ticks=self.n_ticks, reply_name="refinement_reply.txt",
            )
            (ls_dir / "refinement_prompt.md").write_text(
                ref_prompt, encoding="utf-8")
            (ls_dir / "STATUS.txt").write_text(
                f"WAITING for LEVEL-START refinement reply at "
                f"{ls_dir / 'refinement_reply.txt'}\n", encoding="utf-8")
            final = self._poll_path(ls_dir / "refinement_reply.txt")

        # COMPLETENESS RE-LOOK: the harness MEASURES which non-background pixels
        # the VLM left unaccounted (a fact — it has no memory or world knowledge
        # to name anything), shows the VLM those regions, and lets the VLM decide
        # whether each is a missed entity or background to dismiss.  The VLM
        # drives; the harness only measures + surfaces.  Closes the lc=3 failure
        # (rods/caps/track dropped -> 27% covered) WITHOUT the harness inventing
        # or classifying any entity.
        try:
            final = self._complete_perception_with_vlm(final, frame_path, ls_dir)
        except Exception as e:
            print(f"[level-start] completeness re-look skipped ({e})")

        # LEVEL-SCOPED STATE RESET (game-agnostic): a new level is a NEW scene
        # with its own layout, so POSITIONED state belongs to the level that
        # produced it.  Carrying the prior level's entities/groups forward (the
        # merge in ingest_perception only ever appends/updates by name) makes
        # the exploration planner probe GHOST positions from the OLD layout, and
        # leaves one-shot probe tiers stuck in their finished phase (e.g. the
        # controlled-experiment tier ends a solved level in phase "done" and
        # would never re-engage on the next one).  Reset positioned + scratch
        # state here; cross-level KNOWLEDGE is preserved elsewhere and untouched:
        # the entity TEMPLATES were just promoted into LevelMemory (position-free
        # roles + mechanics), the prior inventory is already baked into the text
        # priors surfaced to the VLM above (prior_ctx), and the mechanic /
        # blocking / win-condition hypotheses + action/delta history stay on
        # `world`.  The fresh per-level perception below re-grounds every current
        # entity (the VLM reuses names where an entity recurs, so identity that
        # SHOULD persist still does -- via the name, not a stale coordinate).
        if self._snapshot_entity_inventory():        # a prior level existed
            self._reset_level_scoped_state()

        # Re-ground the entity inventory for the new level, and persist the
        # analysis artifact the trace renders per level.
        final = self._ground_perception_geometry(
            final, frame_path, persist_dir=ls_dir)
        # VLM-GATED QUALITY INSPECTION: the VLM visually confirms the grounded
        # boxes on the overlay (directed by the substrate QA flags) and corrects
        # any that snapped onto the wrong object -- the durable guard against the
        # recurring 'entity analysis is off' regression.
        final = self._vlm_inspect_grounding(final, frame_path, ls_dir)
        self.world.ingest_perception(final)
        # COUPLING-AWARE ROUTE: from the perceived scene, compute the path the
        # mover must travel to MATE into the goal (around any barrier, through its
        # gap, entering the goal's mouth) -- the 'encoding list' the program must
        # realise.  Surfaced as a target; never blocks the run.
        self._surface_coupling_route(final, frame_path)
        # IDENTITY-BEFORE-PROBE: decode each similarity-group's STATIC content so a
        # distinct-encoding group (a key/legend) is a measured fact BEFORE probing
        # -- never inferred (or wrongly refuted) from a null click-response.
        self._surface_group_distinctness(final, frame_path)
        # SCENE STATE (state-as-medium): populate the canonical entity graph from the
        # grounded entities + persist it for the live views (VLM + user).
        self._sync_scene(frame_path)
        # SKILL RECOGNITION: surface any high-level strategy whose trigger fires on
        # this scene (e.g. follow-the-instructions when a key/legend + editable panel
        # are present) -- a reusable recipe the VLM follows, not a one-off.
        self._surface_skills()
        # CARRY THE WIN CONDITION (the lever): seed the prior level's VLM-authored
        # win condition as a high-credence, assume-stable hypothesis on the new
        # level.  This drops function-coverage exploration importance (1 - max_win)
        # so COS DEFERS blind per-entity re-probing and instead maps the new scene
        # onto the carried roles and pursues the win -- verifying it cheaply.
        try:
            if self._level_memory.seed_win_condition(self.world):
                print("[level-start] carried prior-level WIN CONDITION as a strong "
                      "prior (defers blind re-probing; verify cheaply)")
        except Exception as e:
            print(f"[level-start] win-condition carry skipped ({e})")
        # Carry per-entity FUNCTION knowledge so already-known entities (the
        # trigger, mover, goal) are not re-probed from scratch on the new level.
        self._preconfirm_carried_functions()
        (ls_dir / "level_start_analysis.json").write_text(
            json.dumps(final, indent=2), encoding="utf-8")
        self.world.save(self.work_dir / "world_knowledge.json")
        print(f"[level-start] re-grounded {len(final.get('entities') or [])} "
              f"entities for the new level (turn {self.world.turn})")
        # Persist the accumulated cross-level templates (this level's entities
        # are promoted at the NEXT level start, or here for cross-trial reuse).
        try:
            self._ensure_level_memory().save(self._level_memory_path())
        except Exception as e:
            print(f"[level-start] level-memory save skipped ({e})")

    def _reset_level_scoped_state(self) -> None:
        """Drop the layout-tied state of the just-finished level so the next
        level is re-grounded purely from its own fresh perception, while
        PRESERVING cross-level knowledge (the benefit of having solved the
        prior level).

        The LEVEL vs CROSS-LEVEL classification + the actual reset live on
        ``WorldKnowledge.reset_for_new_level`` (the single source of truth, next
        to the fields it governs); this just invokes it and logs.  Cross-level
        knowledge is preserved both there (mechanic / blocking / win-condition
        hypotheses, durable subgoals, recoverability, capability calibration,
        history) and via LevelMemory (position-free entity TEMPLATES promoted
        above) + the prior inventory baked into the surfaced text priors.
        Fully guarded -- never break the run.
        """
        try:
            reset = self.world.reset_for_new_level()
            print(f"[level-start] reset level-scoped world state {reset}; "
                  f"cross-level knowledge preserved")
        except Exception as e:                                   # never break
            print(f"[level-start] level-scoped reset skipped ({e})")
        # The cached per-entity cell lattices + measured settable verdicts are
        # layout-tied -> drop them so the new level re-decomposes + re-measures.
        self._struct_cell_cache = {}
        self._panel_scale_cache = {}
        self._panel_settable = {}
        self._disambig_pending = None
        # Scope the persistent Claim Store to the new level the same way: drop
        # positioned claims from the old layout, keep cross-level mechanic claims
        # but flag them for one cheap reconfirmation (assume-but-recheck), then
        # persist so a re-run reloads the settled frontier.
        cs = getattr(self, "_claim_store", None)
        if cs is not None:
            try:
                cs.carry_to_new_level(self._level_signature())
                cs.save_for_game()
            except Exception as e:
                print(f"[claim-store] level carry skipped ({e})")

    @staticmethod
    def _entity_bboxes(entities) -> list:
        """Pull [r1,c1,r2,c2] boxes out of an entity list (skips malformed)."""
        out = []
        for e in entities or []:
            b = e.get("bbox_ticks_turn1")
            if isinstance(b, (list, tuple)) and len(b) == 4:
                out.append(tuple(int(v) for v in b))
        return out

    def _complete_perception_with_vlm(self, final, frame_path, ls_dir,
                                      max_passes: int = 2,
                                      poll_fn=None) -> dict:
        """VLM-DRIVEN completeness loop. The harness MEASURES the non-background
        pixels the VLM's inventory left unaccounted (a deterministic fact — the
        harness has no memory/world-knowledge to name anything) and SURFACES
        those regions to the VLM, which decides whether each is a missed entity
        (it adds it) or background to ignore (it dismisses it). Loops on the
        VLM's revised inventory until everything visible is accounted for, or a
        safety budget of `max_passes` re-looks is hit.

        Division of labor (SPEC_perception_module / VLM-in-control): the harness
        is a measuring HELPER; every naming/keep/ignore decision is the VLM's.
        The harness NEVER invents or classifies an entity.

        Records ``coverage_fraction`` / ``perception_complete`` /
        ``n_completeness_passes`` (facts) on the artifact. ``poll_fn`` overrides
        the reply source for tests."""
        import agnostic_segmentation as AS                    # noqa: E402
        poll_fn = poll_fn or self._poll_path
        dismissed = []                  # regions the VLM judged non-entities
        passes = 0
        cov = None
        for passes in range(1, max_passes + 1):
            bboxes = self._entity_bboxes(final.get("entities")) + dismissed
            cov = AS.coverage(frame_path, bboxes, play_rows=64,
                              min_uncovered=4, gap=2)
            if cov["complete"]:
                passes -= 1
                break
            unc = cov["uncovered_regions"]
            print(f"[level-start] completeness re-look {passes}: VLM accounts "
                  f"for {cov['covered_fraction']:.0%} of non-bg pixels; "
                  f"surfacing {len(unc)} unaccounted region(s) to the VLM")
            ov = ls_dir / f"coverage_overlay_p{passes}.png"
            try:
                self._render_coverage_overlay(
                    frame_path, final.get("entities") or [], unc, ov)
            except Exception as e:
                print(f"[level-start] coverage overlay render failed ({e})")
            sys_text, _ = _fmt_prompts(
                self.n_ticks, label_stride=self.label_stride,
                single_frame=True, game_id=self.game.game_id)
            reply_name = f"completeness_reply_p{passes}.txt"
            prompt = COMPLETENESS_PERCEPTION_PROMPT.format(
                game_id=self.game.game_id, level=self.game.level,
                sys_text=sys_text, n_ticks=self.n_ticks,
                covered_pct=int(round(cov["covered_fraction"] * 100)),
                n_unc=len(unc), reply_name=reply_name)
            (ls_dir / f"completeness_prompt_p{passes}.md").write_text(
                prompt, encoding="utf-8")
            (ls_dir / "STATUS.txt").write_text(
                f"WAITING for COMPLETENESS re-look reply at "
                f"{ls_dir / reply_name}\n", encoding="utf-8")
            reply = poll_fn(ls_dir / reply_name)
            # The VLM returns the COMPLETE revised inventory; adopt it verbatim.
            if isinstance(reply, dict) and reply.get("entities") is not None:
                final["entities"] = reply["entities"]
            # The VLM may explicitly dismiss regions as non-entities; honor that
            # so they are not re-surfaced (the VLM made the call, not us).
            for dr in (reply.get("dismissed_regions") or []) if isinstance(reply, dict) else []:
                b = dr.get("bbox_ticks_turn1") if isinstance(dr, dict) else dr
                if isinstance(b, (list, tuple)) and len(b) == 4:
                    dismissed.append(tuple(int(v) for v in b))
        # PIXEL-CONTENT AUDIT — catches PHANTOM / APPEARANCE-MISMATCH /
        # NO-UNIQUE-CONTENT entities the coverage check is blind to.  The
        # harness MEASURES (deterministic, frame-only); the VLM is then
        # surfaced the suspect list in a final re-look pass so it can
        # revise or drop each flagged entity.  Catches the trial-13
        # chain_connector_red pattern: bbox spans only pixels other
        # entities already own, so the entity adds no information.
        suspects = AS.audit_entity_pixel_content(
            frame_path, final.get("entities") or [], play_rows=64)
        if suspects:
            print(f"[level-start] pixel-content audit flagged "
                  f"{len(suspects)} suspect entity(s): "
                  f"{[s['name'] for s in suspects]}")
            ov2 = ls_dir / "audit_overlay.png"
            try:
                self._render_coverage_overlay(
                    frame_path, final.get("entities") or [], [], ov2)
            except Exception as e:
                print(f"[level-start] audit overlay render failed ({e})")
            reply_name = "audit_reply.txt"
            prompt = (
                "# Pixel-content audit re-look\n\n"
                "The harness ran a deterministic per-entity pixel audit on "
                "your inventory and flagged the following entities as suspect. "
                "For each, EITHER revise (correct bbox / appearance) OR remove "
                "the entity. Return the FULL revised entities array. Per "
                "[spec/SPEC_perception_module.md] the harness never edits "
                "your inventory itself; this is your decision.\n\n"
                "Flag meanings:\n"
                "  - phantom_background_only: bbox interior is essentially all "
                "background colour — no real feature there.\n"
                "  - appearance_rgbs_absent: bbox interior contains none of "
                "the RGB tuples named in your appearance string.\n"
                "  - no_unique_content: every non-bg pixel in this bbox is "
                "ALREADY inside another smaller-or-similar entity's bbox — "
                "this entity adds no pixels to the inventory.\n\n"
                "Suspect list (JSON):\n"
                f"{json.dumps(suspects, indent=2)}\n\n"
                "Reply ONLY a JSON object: "
                "`{\"entities\": [...revised full list...]}`"
            )
            (ls_dir / "audit_prompt.md").write_text(prompt, encoding="utf-8")
            (ls_dir / "STATUS.txt").write_text(
                f"WAITING for AUDIT re-look reply at {ls_dir / reply_name}\n",
                encoding="utf-8")
            try:
                reply = poll_fn(ls_dir / reply_name)
                if isinstance(reply, dict) and reply.get("entities") is not None:
                    final["entities"] = reply["entities"]
                    final["audit_revised"] = True
                    print(f"[level-start] audit re-look applied; inventory "
                          f"now has {len(final['entities'])} entities")
            except Exception as e:
                print(f"[level-start] audit re-look skipped ({e})")
        final["audit_suspects"] = suspects
        # DETERMINISTIC BBOX TIGHTENING — VLM-supplied bboxes routinely drift
        # by 1-3 ticks (the void in lc=4 was declared at [25,28,30,36] when
        # its actual extent was [24,29,30,35]).  This pass snaps every bbox
        # whose appearance text contains an explicit (R,G,B) tuple to the
        # tight connected-component of those pixels inside (+/- a few px of)
        # the declared bbox.  Mechanical, frame-only, no extra VLM call.
        try:
            tightened = AS.tighten_entity_bboxes(
                frame_path, final.get("entities") or [])
            n_tight = sum(1 for e in tightened if e.get("_bbox_tightened"))
            if n_tight:
                names = [e.get("name") for e in tightened
                         if e.get("_bbox_tightened")]
                print(f"[level-start] bbox tightener snapped "
                      f"{n_tight} bbox(es): {names}")
            final["entities"] = tightened
        except Exception as e:
            print(f"[level-start] bbox tightener skipped ({e})")
        # final fact recompute
        bboxes = self._entity_bboxes(final.get("entities")) + dismissed
        cov = AS.coverage(frame_path, bboxes, play_rows=64,
                          min_uncovered=4, gap=2)
        final["coverage_fraction"] = round(float(cov["covered_fraction"]), 3)
        final["perception_complete"] = bool(cov["complete"])
        final["n_completeness_passes"] = passes
        if cov["complete"]:
            print(f"[level-start] perception COMPLETE after {passes} re-look(s) "
                  f"({cov['covered_fraction']:.0%} covered; VLM accounted for "
                  f"every visible region)")
        else:
            print(f"[level-start] perception still {cov['covered_fraction']:.0%} "
                  f"after {passes} re-look(s); {len(cov['uncovered_regions'])} "
                  f"region(s) remain unaccounted (flagged for the strategy VLM, "
                  f"NOT auto-named by the harness)")
        return final

    def _render_coverage_overlay(self, frame_path, entities, uncovered, out):
        """Draw the VLM's current entities (faint cyan) + the harness-measured
        unaccounted regions (red, labeled U1..Un) on the gridded frame, so the
        VLM can SEE exactly what it has not yet named. Pure visualization."""
        from PIL import ImageDraw, ImageFont
        base = render_turn1_overlay(
            frame_path, entities, n_ticks=self.n_ticks, upscale=self.upscale,
            bbox_line_width=1,
            grid_line_width_major=self.grid_line_width_major,
            grid_line_width_minor=self.grid_line_width_minor,
            grid_major_alpha=self.grid_major_alpha,
            grid_minor_alpha=self.grid_minor_alpha,
        ).convert("RGB")
        d = ImageDraw.Draw(base)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        # The base image wraps the logical playfield in SYMMETRIC label margins
        # (see _add_grid_overlay), and render_turn1_overlay draws entity bboxes
        # via ticks_to_playfield_px + those margin offsets.  The uncovered
        # regions are in the SAME [row_top, col_left, row_bottom, col_right]
        # tick space, so they MUST be mapped the same way.  The old code drew at
        # raw (col*upscale, row*upscale) with NO margin offset, so every red
        # "U#" box was shifted up-and-left by the full margin (~123px at
        # upscale 16) -- the upscale 8->16 bump doubled that error, misleading
        # the VLM about what it had not yet named.  Derive pf size + margin from
        # the actual rendered image so it cannot drift from _add_grid_overlay.
        _img = Image.open(frame_path).convert("RGB")
        pf_w = _img.width * self.upscale
        pf_h = _img.height * self.upscale
        margin_left = max(0, (base.width - pf_w) // 2)
        margin_top = max(0, (base.height - pf_h) // 2)
        for i, region in enumerate(uncovered):
            py0, px0, py1, px1 = ticks_to_playfield_px(
                list(region), self.n_ticks, pf_w, pf_h)
            x0, y0 = margin_left + px0, margin_top + py0
            x1, y1 = margin_left + px1, margin_top + py1
            d.rectangle([x0, y0, x1, y1], outline=(255, 40, 40), width=3)
            d.text((x0 + 1, y0 + 1), f"U{i + 1}", fill=(255, 80, 80), font=font)
        base.save(out)
        return out

    def _playback_review_text(self, outcome: str) -> str:
        """Compact REVIEW of this level's play for the end-of-trial reflection: the
        action sequence + how the moves were spent + whether it advanced.  Lets the
        VLM SEE what it actually did and DIAGNOSE a failure (playback mining extended
        to losses), instead of authoring generic lessons blind.  Game-agnostic."""
        try:
            from collections import Counter
            i0 = int(getattr(self, "_level_start_action_idx", 0) or 0)
            acts = [str(getattr(a, "action", a)) for a in self.world.actions_taken[i0:]]
            if not acts:
                return ""
            kinds = Counter(a.split(":")[0] for a in acts)
            lost = ("win" not in str(outcome).lower()
                    and "advance" not in str(outcome).lower())
            verdict = ("This level was NOT completed -- the trial ended (outcome="
                       f"{outcome}) before the goal was reached."
                       if lost else "This level was completed.")
            return (f"{verdict}\nYou took {len(acts)} actions this level "
                    f"(by kind: {dict(kinds)}).\nAction sequence (oldest->newest, "
                    f"last 60):\n  " + " -> ".join(acts[-60:]))
        except Exception:
            return ""

    def finalize(self, outcome: str = "manual_close") -> None:
        """Called at end of run.  Persists:
          1. Cross-GAME learned action-semantic observations to
             the global priors store (global_priors.py).
          2. Within-GAME lessons to the per-game lessons store
             (per_game_lessons.py).  Auto-distills promoted
             MechanicHypotheses / BlockingClaims /
             WinConditionHypotheses, then OPTIONALLY stages an
             end-of-trial author prompt so the actor can add
             free-form lessons and refuted alternatives.
        """
        # 1. Cross-game priors (existing behaviour)
        if self.global_priors_path is not None and self._global_priors is not None:
            try:
                n_new, n_updated = update_and_save(
                    self._global_priors, self.world,
                    priors_path=self.global_priors_path,
                )
                if n_new or n_updated:
                    print(f"[finalize] CROSS-GAME PRIORS: +{n_new} new "
                          f"priors, +{n_updated} updated observations "
                          f"saved to {self.global_priors_path}")
                else:
                    print(f"[finalize] CROSS-GAME PRIORS: no new "
                          f"observations to merge (no hypotheses "
                          f"reached promoted=True this run)")
            except Exception as e:
                print(f"[finalize] CROSS-GAME PRIORS: save failed: {e}")

        # 2. Per-game lessons (within-GAME knowledge for future
        # trials of the same game_id, including in competition
        # mode where games interleave).
        try:
            from per_game_lessons import (                  # noqa: E402
                auto_distill_from_world as _lessons_distill,
                stage_end_of_trial_prompt as _lessons_stage,
                apply_end_of_trial_reply as _lessons_apply,
            )
            trial_id = (
                f"{self.world.game_id}_lc{self.world.level}_"
                f"t{self.world.turn}_"
                f"{int(time.time()) % 100000}"
            )
            distilled = _lessons_distill(
                self.world, trial_id=trial_id, outcome=outcome,
            )
            print(
                f"[finalize] PER-GAME LESSONS: distilled "
                f"{len(distilled)} lesson(s) for game "
                f"{self.world.game_id!r} (trial={trial_id})"
            )
            # Optionally stage end-of-trial author prompt.
            # The prompt is a one-shot VLM-as-Claude call asking
            # the actor to add free-form lessons + refuted
            # alternatives.  If the reply isn't filled in within
            # a short window (or the actor returns an empty
            # object), substrate proceeds without crashing.
            if getattr(self, "use_strategy", False):
                prompt_path, reply_path = _lessons_stage(
                    self.world, self.work_dir,
                    trial_id=trial_id, outcome=outcome,
                    auto_distilled=distilled,
                    playback_review=self._playback_review_text(outcome),
                )
                print(
                    f"[finalize] PER-GAME LESSONS: end-of-trial "
                    f"prompt staged at {prompt_path}"
                )
                # Best-effort poll for the reply.  Short timeout —
                # this is optional; the trial is already over.
                try:
                    reply = poll_strategy_reply(
                        reply_path,
                        timeout_s=min(60, getattr(self, "vlm_timeout_s", 60)),
                        poll_s=2.0,
                    )
                    if isinstance(reply, dict):
                        self._end_of_trial_reply = reply
                        written = _lessons_apply(
                            self.world, reply, trial_id=trial_id,
                        )
                        print(
                            f"[finalize] PER-GAME LESSONS: actor "
                            f"contributed {len(written)} "
                            f"additional lesson(s)"
                        )
                except Exception as e:
                    print(
                        f"[finalize] PER-GAME LESSONS: end-of-trial "
                        f"reply skipped ({e}); auto-distilled "
                        f"lessons persisted regardless"
                    )
        except Exception as e:
            print(f"[finalize] PER-GAME LESSONS: save failed: {e}")

        # 2b. Playback-mining DURABLE findings -> per-game lessons.
        # The within-trial mining trigger surfaces findings live; this
        # crystallizes the stable ones so they persist + resurface next trial.
        try:
            self._persist_mining_findings(
                trial_id=f"mining_{self.world.game_id}_t{self.world.turn}")
        except Exception as e:
            print(f"[finalize] MINING PERSIST: skipped ({e})")

        # 3. Render + mirror trace to the canonical bookmark URL so
        # the user's saved file:///.../trace.html always reflects
        # the most recent run.  Best-effort — if rendering fails we
        # log but don't crash the trial.
        try:
            self._render_and_mirror_trace()
        except Exception as e:
            print(f"[finalize] TRACE MIRROR: failed: {e}")

    def _render_and_mirror_trace(self) -> None:
        """Render the work_dir to index.html and copy it (plus the
        views/ directory) into .tmp/training_data/latest/ so the
        canonical bookmark URL at
        file:///.../.tmp/training_data/latest/trace.html always
        shows the latest run.

        views/ must travel with the HTML because the rendered page
        uses relative <img src="views/...">; a stale views/ from a
        previous trial produces a loading page with the wrong
        frames.
        """
        from render_exploratory_run import render  # noqa: E402
        frame_dir = getattr(self.game, "_work_frame_dir", None)
        if frame_dir is None:
            print("[trace-mirror] adapter has no _work_frame_dir; "
                   "skipping render")
            return
        out = render(self.work_dir, Path(frame_dir))

        repo_root = Path(__file__).resolve().parents[3]
        dest = repo_root / ".tmp" / "training_data" / "latest"
        dest.mkdir(parents=True, exist_ok=True)
        dest_html = dest / "trace.html"
        dest_views = dest / "views"
        src_views = self.work_dir / "views"

        shutil.copyfile(str(out), str(dest_html))
        if dest_views.exists():
            shutil.rmtree(str(dest_views))
        if src_views.exists():
            shutil.copytree(str(src_views), str(dest_views))
            n_frames = sum(1 for _ in dest_views.iterdir())
        else:
            n_frames = 0
        self._last_trace_mirror_ts = time.time()
        print(f"[trace-mirror] -> {dest_html} ({n_frames} frames)")

    def _maybe_mirror_trace(self,
                              min_interval_s: float = 10.0) -> None:
        """Debounced wrapper around _render_and_mirror_trace.
        Called from per-turn paths so the bookmark URL refreshes
        during a long run, but never more often than once every
        ``min_interval_s`` seconds (the renderer scales linearly
        with turn count and isn't free)."""
        last = getattr(self, "_last_trace_mirror_ts", 0.0)
        if (time.time() - last) < min_interval_s:
            return
        try:
            self._render_and_mirror_trace()
        except Exception as e:
            print(f"[trace-mirror] periodic refresh failed: {e}")

    # ---- Repeat-until-relation executor helpers (game-agnostic) --------

    def _goal_done_set(self):
        """Set of goal targets currently COMPLETED, read from the substrate's
        crystallized win relation (game-agnostic).  None when no relation is
        crystallized or the relation type carries no done-set (then the
        drop-guard is simply inactive)."""
        try:
            from knowledge_crystallization import evaluate_win_relation  # noqa: E402
            cands = [h for h in self.world.win_condition_hypotheses
                     if getattr(h, "win_relation", None)]
            if not cands:
                return None
            cands.sort(key=lambda h: getattr(h, "credence", 0.0), reverse=True)
            det = (evaluate_win_relation(self.world, cands[0].win_relation)
                   .get("detail") or {})
            done = det.get("done")
            return set(done) if done is not None else None
        except Exception:
            return None

    def _skewered_collectables(self):
        """Collectable entities currently overlap-reachable from the agent —
        i.e. on the skewer / carry assembly (agent -> arm -> blocks).  Losing
        one between turns means it LEFT the skewer (un-skewered), which the
        latched HUD done-set cannot see.  Game-agnostic: overlap geometry +
        open-vocab roles.  None on error."""
        try:
            from relational_kinematics import overlap_reachable  # noqa: E402
            deltas = getattr(self.world, "deltas_observed", None) or []
            if not deltas:
                return set()
            rels = getattr(deltas[-1], "relations", None) or []
            agent = None
            coll = set()
            for rec in self.world.entities.values():
                role = getattr(rec, "current_role", None)
                if role == "agent" and agent is None:
                    agent = rec.name
                elif role == "collectable":
                    coll.add(rec.name)
            return overlap_reachable(rels, agent) & coll
        except Exception:
            return None

    def _handle_repeat_step(self, delta, interrupt) -> None:
        """Decide, after one repeat step, whether to STOP or CONTINUE the
        repeat-until-relation maneuver.  Stop reasons: the relational
        stop-condition (or a built-in interrupt) fired; a completed goal
        target regressed (DROP-GUARD -> UNDO); the action stalled; or the
        safety cap.  Otherwise re-queue the same action."""
        self._repeat_count += 1
        # DROP-GUARDS ARE CHECKED FIRST — losing progress (a completed target
        # regressing, or a skewered block leaving the skewer) must win over
        # reaching the stop-condition, because a single step can BOTH meet the
        # stop-condition AND drop a block; in that case we must undo, not
        # accept the stop.
        # 1. drop-guard: a previously-completed goal target regressed.
        base, cur = self._repeat_done_baseline, self._goal_done_set()
        if base is not None and cur is not None and not (base <= cur):
            dropped = sorted(base - cur)
            try:
                from causal_attribution import Anomaly
                self._run_causal_attribution(
                    Anomaly("regression", getattr(delta, "action", ""),
                            frozenset(dropped)))
            except Exception as e:
                print(f"[causal] hook skipped ({e})")
            return self._end_repeat(
                f"DROP-GUARD: completed target(s) {dropped} regressed",
                undo_steps=self._repeat_count)
        # 1b. skewer drop-guard: a previously-skewered block LEFT the skewer
        # (un-skewered).  Catches the loss the latched HUD done-set above is
        # blind to (the orange-un-skewer hazard).  Top-down board, no gravity:
        # the risk is un-skewering, not dropping.
        sk_base = getattr(self, "_repeat_skewered_baseline", None)
        sk_cur = self._skewered_collectables()
        if sk_base and sk_cur is not None and not (sk_base <= sk_cur):
            unskewered = sorted(sk_base - sk_cur)
            try:
                from causal_attribution import Anomaly
                self._run_causal_attribution(
                    Anomaly("regression", getattr(delta, "action", ""),
                            frozenset(unskewered)))
            except Exception as e:
                print(f"[causal] hook skipped ({e})")
            return self._end_repeat(
                f"DROP-GUARD: block(s) {unskewered} left the skewer "
                f"(un-skewered)", undo_steps=self._repeat_count)
        # 2. the repeat's OWN relational stop-condition, evaluated directly
        # against this step's relations (NOT in _interrupt_conditions, so
        # _check_sequence_interrupts does not see it), plus any built-in
        # interrupt (score advance, visual_event, win, ...).
        cond_met = False
        if self._repeat_until:
            try:
                from relational_kinematics import (              # noqa: E402
                    evaluate_interrupt_condition,
                )
                cond_met = evaluate_interrupt_condition(
                    self._repeat_until, getattr(delta, "relations", None) or [])
            except Exception as e:
                print(f"[driver] repeat stop-condition eval failed ({e})")
        if cond_met or interrupt is not None:
            why = self._repeat_until if cond_met else interrupt
            return self._end_repeat(f"stop-condition met ({why})")
        # 3. stall: the action produced no observable change this step
        changed = bool(delta.agent_moved or delta.entities_appeared
                       or delta.entities_disappeared or delta.entities_changed
                       or getattr(delta, "visual_events", None))
        if not changed:
            return self._end_repeat("stalled (action produced no change)")
        # 4. safety cap
        if self._repeat_count >= self._repeat_cap:
            return self._end_repeat(f"hit repeat cap ({self._repeat_cap})")
        # else: continue repeating (keep _pending_sequence + _repeat_until)
        self._sequence_idx = 0

    def _end_repeat(self, reason: str, undo_steps: int = 0) -> None:
        """Terminate a repeat-until maneuver; stash feedback for the next VLM
        turn.  On a drop-guard abort, queue UNDO x N (the cross-game undo
        action) to revert the maneuver before re-prompting."""
        self._repeat_feedback = (
            f"repeat-until [{self._repeat_until}] ended after "
            f"{self._repeat_count} repeat(s): {reason}")
        print(f"[driver] REPEAT-UNTIL ended: {self._repeat_feedback}")
        self._repeat_until = None
        self._repeat_done_baseline = None
        undo = self._undo_action
        if undo_steps > 0 and undo in self.game.available_actions():
            # revert the regressing maneuver, then re-prompt the VLM
            self._pending_sequence = [undo] * undo_steps
            self._sequence_idx = 0
            self._interrupt_conditions = []
            self._sequence_rationale_prefix = (
                f"DROP-GUARD revert: {undo} x{undo_steps}")
            self._repeat_feedback += (
                f" — auto-reverting {undo_steps} step(s) via {undo}")
        else:
            self._pending_sequence = []
            self._sequence_idx = 0
            self._interrupt_conditions = []
            self._sequence_goal_id = None
            self._sequence_committed_at_turn = None
            self._sequence_rationale_prefix = ""
        self._repeat_count = 0

    # ------------------------------------------------------------------
    # Causal attribution hook (SPEC_vlm_backward_reasoning.md §7)
    # ------------------------------------------------------------------
    def _run_causal_attribution(self, anomaly, provider=None,
                                 attribute_fn=None) -> object:
        """On an anomaly (a drop-guard regression / a block), run the
        causal-attribution loop to name the CAUSE and a way around it, then
        register the avoidance as a DURABLE ActiveSubgoal (so it is not
        re-derived/dropped each turn — the drift interlock, §8).

        Safe by default: needs a perception `observe` to build a live provider;
        until one is wired (`self._causal_observe`), this is a logged no-op so
        live runs are unaffected. A caller/test may inject `provider` +
        `attribute_fn` directly to exercise the wiring."""
        try:
            from causal_attribution import attribute as _default_attr
            attribute_fn = attribute_fn or _default_attr
            if provider is None:
                provider = self._build_causal_provider(anomaly)
            if provider is None:
                print(f"[causal] anomaly: {anomaly.describe()} — "
                      f"attribution observe not wired; skipped")
                self._last_anomaly = anomaly
                return None
            claim = attribute_fn(provider, anomaly)
            if (claim and claim.confidence in (
                    "confirmed", "twin_confirmed", "predicted")
                    and claim.remedy_action):
                self._register_avoidance_subgoal(claim)
                # Persist + crystallize so the diagnosis survives the
                # trial (the "remembered, generalized, reused" loop).
                self._crystallize_causal_claim(claim)
                self._repeat_feedback = (
                    (self._repeat_feedback or "")
                    + f"  CAUSE FOUND: {claim.effect} is caused by "
                      f"{claim.gating_relation}(agent,{claim.culprit}); "
                      f"avoid via {claim.remedy_action!r} first "
                      f"({claim.confidence}).")
                print(f"[causal] {claim.confidence}: cause={claim.culprit} "
                      f"via {claim.gating_relation}; remedy={claim.remedy_action}")
            else:
                print(f"[causal] no confirmed cause for {anomaly.describe()} "
                      f"({getattr(claim, 'confidence', 'none')})")
            return claim
        except Exception as e:
            print(f"[causal] attribution skipped ({e})")
            return None

    def _build_causal_provider(self, anomaly):
        """Build a live ARC provider if a perception `observe` is wired.
        Returns None otherwise (the perception slot — segmenting standing
        structures like a shaft — is the dependency the loop needs; see
        arc_causal_binding)."""
        observe = getattr(self, "_causal_observe", None)
        if observe is None:
            return None
        try:
            from arc_causal_binding import build_arc_provider
            return build_arc_provider(
                self.game, observe,
                project_path=getattr(self, "_causal_project_path", None))
        except Exception as e:
            print(f"[causal] provider build failed ({e})")
            return None

    def _register_avoidance_subgoal(self, claim) -> None:
        """Turn a confirmed CausalClaim into a durable ActiveSubgoal commitment
        (closed explicitly by the actor, not silently re-derived)."""
        try:
            from world_knowledge import ActiveSubgoal
            sid = f"avoid_{claim.culprit}_{self.world.turn}"
            if any(getattr(s, "subgoal_id", None) == sid
                   for s in self.world.active_subgoals):
                return
            self.world.active_subgoals.append(ActiveSubgoal(
                subgoal_id=sid,
                name=f"avoid {claim.gating_relation} with {claim.culprit}",
                problem_solved=claim.effect,
                expected_outcome=(
                    f"after {claim.remedy_action!r}, {claim.action!r} no longer "
                    f"causes: {claim.effect}"),
                created_at_turn=self.world.turn,
                status="active",
                forward_simulation=claim.note,
                derived_from="causal_attribution (confirmed cause + remedy)",
            ))
            print(f"[causal] registered durable avoidance subgoal {sid!r}")
        except Exception as e:
            print(f"[causal] subgoal registration failed ({e})")

    def _crystallize_causal_claim(self, claim) -> None:
        """Persist a confirmed CausalClaim CROSS-TRIAL so the diagnosis is
        remembered, generalized, and reused — closing the loop the
        within-trial ActiveSubgoal cannot (it dies with the trial).

        Two role/relation-keyed stores (NEVER instance/colour-keyed, so the
        knowledge transfers to recoloured / resized variants in competition):

          1. per_game_lessons (kind='blocking') — the avoid-list surfaced at
             the START of every future trial of this game.  Credence is graded
             by the attribution tier (confirmed > twin_confirmed > predicted),
             matching the sandbox that earned it.
          2. subroutine_kb — the remedy compiled as an EARNED, reusable
             avoidance macro: when ``gating_relation(agent, <structure>)`` recurs
             and blocks the goal, apply ``remedy_action × repeats`` to clear it
             first.  Keyed by a structured relational ``signature`` so the
             proceduralization bridge can auto-surface it on the same relation.

        Fully defensive: a persistence failure never breaks the live run."""
        tier = getattr(claim, "confidence", "predicted")
        repeats = max(1, int(getattr(claim, "remedy_repeats", 1) or 1))
        cred_by_tier = {
            "confirmed": 0.85, "twin_confirmed": 0.7, "predicted": 0.55,
        }
        trial_id = f"causal_t{getattr(self.world, 'turn', 0)}"
        # role-level (skin-agnostic) description — names the RELATION + the
        # structure's ROLE/behaviour, not its colour or pixel identity.
        desc = (
            f"BLOCKING CAUSE ({tier}): action {claim.action!r} causes "
            f"{claim.effect} while {claim.gating_relation}"
            f"(agent, {claim.culprit}). REMEDY: apply "
            f"{claim.remedy_action!r} x{repeats} FIRST to break "
            f"{claim.gating_relation} before {claim.action!r}."
        )
        # 1. durable avoid-list lesson
        try:
            from per_game_lessons import (                      # noqa: E402
                commit_lesson_from_actor as _commit,
            )
            _commit(
                game_id=self.world.game_id,
                kind="blocking",
                description=desc,
                notes=("auto-crystallized from causal_attribution "
                       f"(two-arm counterfactual, tier={tier}"
                       + (f", fidelity={claim.fidelity}"
                          if getattr(claim, "fidelity", None) else "")
                       + "). " + (getattr(claim, "note", "") or "")),
                credence=cred_by_tier.get(tier, 0.55),
                trial_id=trial_id,
            )
            print(f"[causal] persisted blocking lesson ({tier}) -> "
                  f"per_game_lessons")
        except Exception as e:
            print(f"[causal] lesson persist failed ({e})")
        # 2. earned avoidance macro in the subroutine KB (keyed on the
        # gating relation so the bridge can retrieve it by relation, not skin)
        try:
            from subroutine_kb import (                         # noqa: E402
                promote_chain_as_subroutine as _promote,
            )
            signature = [[claim.gating_relation,
                          ["agent", "blocking_structure"], None]]
            relational_steps = [
                f"if {claim.gating_relation}(agent, blocking_structure) holds "
                f"and it blocks the goal, apply {claim.remedy_action} "
                f"until {claim.gating_relation} no longer holds",
                f"then resume {claim.action} toward the goal",
            ]
            _promote(
                name=f"clear_{claim.gating_relation}_before_act",
                description=(
                    f"Break a blocking {claim.gating_relation} between the "
                    f"agent and a standing structure before acting, by "
                    f"applying {claim.remedy_action!r}; learned because "
                    f"{claim.action!r} otherwise causes {claim.effect}."),
                problem_solved=(
                    f"goal blocked by {claim.gating_relation}"
                    f"(agent, structure); naive {claim.action!r} regresses"),
                concrete_chain=[claim.remedy_action] * repeats,
                expected_outcome=(
                    f"{claim.gating_relation}(agent, structure) no longer "
                    f"holds; {claim.action!r} can proceed without "
                    f"{claim.effect}"),
                game_id=self.world.game_id,
                level=getattr(self.world, "level", 0),
                turn_range=[getattr(self.world, "turn", 0)] * 2,
                original_goal=claim.effect,
                notes=f"earned via causal_attribution ({tier})",
                signature=signature,
                relational_steps=relational_steps,
            )
            print(f"[causal] crystallized avoidance macro "
                  f"clear_{claim.gating_relation}_before_act -> subroutine_kb")
        except Exception as e:
            print(f"[causal] macro crystallize failed ({e})")

    def _persist_mining_findings(self, trial_id: str = "") -> None:
        """Crystallize the STABLE playback-mining findings of this trial into
        per_game_lessons so they survive the trial and resurface next time.

        Persisted (role/relation-keyed, skin-agnostic — transfers across
        recoloured/resized variants):
          * catastrophe action-classes — actions whose steps coincided with
            committed completions being lost (a 'blocking' avoid-list entry);
          * the un-skewer credit model — if un-skewering REVERTS credit, the
            simultaneity requirement is a durable 'mechanic' of this game.

        Defensive: never raises into finalize."""
        try:
            import playback_mining as _PM                       # noqa: E402
        except Exception as e:
            print(f"[mining] module import failed ({e}); skipped")
            return
        try:
            r = _PM.mine_current_level(self.world)
        except Exception as e:
            print(f"[mining] mine_current_level failed ({e}); skipped")
            return
        from per_game_lessons import (                          # noqa: E402
            commit_lesson_from_actor as _commit,
        )
        n = 0
        # movement rules: mine + ACCUMULATE into the persistent KB (refines by
        # data alone across trials/levels, no code change); promote high-support
        # ones to durable mechanic lessons the backward reasoner can query.
        try:
            kb = _PM.accumulate_movement_rules(self.world)
            for cat, rule in kb.items():
                if rule.get("support", 0) >= 2:
                    _commit(
                        game_id=self.world.game_id,
                        kind="mechanic",
                        description=_PM.describe_movement_rule(cat, rule),
                        notes="auto-mined movement rule (geometric primitives)",
                        credence=min(0.85, 0.4 + 0.05 * rule["support"]),
                        trial_id=trial_id,
                    )
                    n += 1
            # column-control (push) rule: how to bring an object to a chosen
            # position along the free axis -- data-mined, refinable by later
            # levels.  Promote when seen with reasonable directional purity.
            cc = _PM.mine_column_control(self.world)
            if cc and cc.get("support", 0) >= 2 and cc.get("purity", 0) >= 0.7:
                _commit(
                    game_id=self.world.game_id,
                    kind="mechanic",
                    description=_PM.describe_column_control(cc),
                    notes="auto-mined column-control rule (pusher-behind-pushee)",
                    credence=min(0.85, 0.4 + 0.08 * cc["support"]),
                    trial_id=trial_id,
                )
                n += 1
            # grasp-by-pinning: pin an object against a barrier to attach it to
            # the manipulator, then carry it -- the repositioning operator for an
            # otherwise-immovable isolated object.  Data-mined; promote on a clean
            # contrastive gap (carried-while-pinned >> carried-while-free).
            gp = _PM.mine_grasp_by_pinning(self.world)
            if gp and gp.get("support", 0) >= 2 and gp.get("gap", 0) >= 0.5:
                _commit(
                    game_id=self.world.game_id,
                    kind="mechanic",
                    description=_PM.describe_grasp_by_pinning(gp),
                    notes="auto-mined grasp-by-pinning rule (pin-to-barrier attaches; then carry)",
                    credence=min(0.85, 0.4 + 0.06 * gp["support"]),
                    trial_id=trial_id,
                )
                n += 1
            # decouple / separation: retract releases a movable; a subsequent UP
            # separates the released one (stays) from the coupled ones (rise) --
            # the inverse of grasp-by-pinning, and the operator that parks one
            # block of a pair onto its structure while leaving the partner below.
            dc = _PM.mine_decouple(self.world)
            if dc and dc.get("support", 0) >= 2:
                _commit(
                    game_id=self.world.game_id,
                    kind="mechanic",
                    description=_PM.describe_decouple(dc),
                    notes="auto-mined decouple/separation rule (retract releases; UP separates a pair)",
                    credence=min(0.85, 0.4 + 0.06 * dc["support"]),
                    trial_id=trial_id,
                )
                n += 1
            # engage-by-body-sweep: agent's extended body sweeping perpendicular
            # to its long axis DRAGS movables already in its bbox extent.  Third
            # independent engagement operator (distinct from grasp-by-pinning and
            # decouple).  Game-agnostic; surfaced from playback when contrastive
            # gap shows the spanning condition correlates with co-displacement.
            bs = _PM.mine_engagement_by_body_sweep(self.world)
            if bs and bs.get("support", 0) >= 2 and bs.get("gap", 0) >= 0.1:
                _commit(
                    game_id=self.world.game_id,
                    kind="mechanic",
                    description=_PM.describe_engagement_by_body_sweep(bs),
                    notes="auto-mined engage-by-body-sweep (extended body + perpendicular sweep)",
                    credence=min(0.85, 0.4 + 0.06 * bs["support"]),
                    trial_id=trial_id,
                )
                n += 1
        except Exception as e:
            print(f"[mining] movement-rule accumulation failed ({e})")
        # catastrophe action-classes -> avoid-list
        for act in (r.get("catastrophe_actions") or []):
            try:
                _commit(
                    game_id=self.world.game_id,
                    kind="blocking",
                    description=(
                        f"CATASTROPHE ACTION (mined): action {act!r} "
                        f"coincided with losing already-secured "
                        f"completions this level. Avoid {act!r} while "
                        f"holding committed members; it reverts progress."),
                    notes="auto-mined from playback (membership-loss "
                          "credit-assignment)",
                    credence=0.6,
                    trial_id=trial_id,
                )
                n += 1
            except Exception as e:
                print(f"[mining] catastrophe persist failed ({e})")
        # un-skewer credit model -> durable mechanic
        if r.get("unskewer_verdict") == "reverts":
            try:
                _commit(
                    game_id=self.world.game_id,
                    kind="mechanic",
                    description=(
                        "WIN CREDIT MODEL (mined): un-skewering a secured "
                        "member REVERTS its credit. All committed members "
                        "must be held SIMULTANEOUSLY to win; do not release "
                        "one to free the agent."),
                    notes="auto-mined from playback (membership-loss events)",
                    credence=0.65,
                    trial_id=trial_id,
                )
                n += 1
            except Exception as e:
                print(f"[mining] verdict persist failed ({e})")
        print(f"[finalize] MINING PERSIST: crystallized {n} durable "
              f"finding(s) for {self.world.game_id!r} "
              f"(catastrophe_actions={r.get('catastrophe_actions')}, "
              f"unskewer={r.get('unskewer_verdict')})")

    def run_one_step(self) -> Optional[TurnReport]:
        # Consume any auto-promotion replies left from prior turns
        # before assembling the next strategy prompt — that way a
        # newly-promoted subroutine appears in the surface ASAP.
        try:
            self._consume_pending_promotions()
        except Exception as e:
            print(f"[driver] consume_pending_promotions failed "
                  f"({e}); continuing")

        # ------------------------------------------------------------------
        # MECHANICAL EXECUTION MODE — if a planned_action_sequence
        # was committed by a prior strategy call and not yet
        # exhausted/interrupted, execute its next action WITHOUT
        # re-prompting the VLM.  Strategic / discipline updates
        # (subgoal commits, probe proposals, WC observations) are
        # NOT processed in this mode — they only fire on real
        # strategy calls.  Substrate-computed observations
        # (visual_events, score deltas, perception delta) still
        # happen.  After the step, _check_interrupts decides
        # whether the next call should be strategic (sequence
        # complete, interrupt fired) or mechanical (more
        # actions queued).
        # ------------------------------------------------------------------
        if self._repeat_until is not None or (
                self._pending_sequence
                and self._sequence_idx < len(self._pending_sequence)):
            return self._run_mechanical_step()

        # pick action
        snapshot = self.world.symbolic_snapshot()
        # the cell_actor wants the symbolic_state shape from a
        # perception reply; build a synthesized one from the snapshot.
        # CRITICAL: feed BACK the cells we've empirically discovered
        # are blocked (from prior failed moves) and traversable
        # (from prior successful moves), so the BFS doesn't keep
        # retrying actions that don't work.
        ss = {
            "agent_cell": (snapshot["agent"]["current_cell"]
                            if snapshot["agent"] else None),
            "goal_candidate_cells":
                self._infer_goal_cells_from_world(),
            "traversable_cells":
                self._infer_traversable_cells_from_deltas(),
            "blocked_cells":
                self._infer_blocked_cells_from_deltas(),
        }
        entities_for_actor = [
            {
                "name": r.name,
                "bbox_ticks_turn1": r.current_bbox,
                "role_hypothesis": r.current_role or "unknown",
            }
            for r in self.world.entities.values()
            if r.current_bbox is not None
        ]
        # NOTE: a null agent_cell (non-grid / continuous-arena game) no longer
        # halts the run.  The grid planner + cell_actor below only run when a
        # grid agent_cell exists; otherwise the autonomous curiosity probe
        # further down drives discovery so COS can still learn the mechanic.
        move_step = (self.world.grid_inference.cell_ticks
                     if self.world.grid_inference
                     and self.world.grid_inference.cell_ticks
                     else 1)

        # TRACK THE CONTROLLABLE: the world flags controllable_found but does not
        # record WHICH entity it is for a non-grid scene -- so the cardinal pursuit
        # has no mover to steer.  Identify it purely by MOTION: the entity that
        # moves most CONSISTENTLY in response to actions (a count over turns, so a
        # pushed block that slides once never outvotes the player that steps every
        # turn).  Fully game-agnostic (no entity-name / game assumptions, no tuned
        # threshold -- the only test is 'did it move at all'); guarded.
        try:
            _cur_ctr = {e["name"]: ((e["bbox_ticks_turn1"][0] + e["bbox_ticks_turn1"][2]) / 2.0,
                                    (e["bbox_ticks_turn1"][1] + e["bbox_ticks_turn1"][3]) / 2.0)
                        for e in entities_for_actor
                        if e.get("name") and e.get("bbox_ticks_turn1")
                        and len(e["bbox_ticks_turn1"]) >= 4}
            _prev_ctr = getattr(self, "_prev_entity_centers", None) or {}
            _acts = getattr(self.world, "actions_taken", []) or []
            _last = str(getattr(_acts[-1], "action", "")) if _acts else ""
            if _prev_ctr and _last and _last not in ("NONE", "RESET"):
                self._move_counts = getattr(self, "_move_counts", {})
                for n in _cur_ctr:
                    if n in _prev_ctr:
                        if (abs(_cur_ctr[n][0] - _prev_ctr[n][0])
                                + abs(_cur_ctr[n][1] - _prev_ctr[n][1])) > 0:
                            self._move_counts[n] = self._move_counts.get(n, 0) + 1
                if self._move_counts:
                    self._controllable_name = max(self._move_counts,
                                                  key=self._move_counts.get)
            self._prev_entity_centers = _cur_ctr
        except Exception:
            pass

        # Action source — try the cognitive_os planner first (it
        # produces curiosity-action-trial goals for untested
        # actions and primary "reach-target" goals for known
        # collectables; smooth credence-based transition with no
        # explicit probe phase).  Fall back to the mechanical
        # cell_actor when the planner returns no plan.
        planned_choice = None
        if self.use_planner:
            try:
                available = self.game.available_actions()
                planned_choice = choose_planned_action(
                    self.world, available,
                )
                if planned_choice.action_string == "NONE":
                    planned_choice = None
                else:
                    print(f"[turn {self.world.turn}] PLANNER chose "
                          f"{planned_choice.action_string!r} "
                          f"(goal={planned_choice.goal_id}, "
                          f"kind={planned_choice.plan_kind}, "
                          f"plan={planned_choice.full_plan_actions})")
            except Exception as e:
                print(f"[driver] planner failed ({e}); falling back "
                      f"to mechanical actor")
                planned_choice = None

        if planned_choice is not None:
            # Adapt the PlannedActionChoice to the ActionChoice
            # shape the rest of the loop expects (so the VLM
            # strategy layer can still override).
            class _AC:
                action = planned_choice.action_string
                rationale = planned_choice.rationale
                plan_kind = planned_choice.plan_kind
                goal_id = planned_choice.goal_id
                target_cell = None
                full_plan_actions = list(
                    planned_choice.full_plan_actions or []
                )
            choice = _AC()
        elif ss["agent_cell"] is not None:
            choice = choose_action(
                ss, entities_for_actor, move_step=move_step,
            )
            # mechanical fallback emits a single action; pad the
            # plan-trace to match the new ActionRecord field.
            if not hasattr(choice, "full_plan_actions"):
                choice.full_plan_actions = [choice.action]
        else:
            # Non-grid scene (no agent_cell) and the planner produced no
            # plan -> leave choice unset so the curiosity probe below fires.
            choice = None
        # A "blind" choice = no plan, a NONE, or a generic curiosity/explore probe
        # -- these are the cases where a solved-mechanic pursuit or an on-screen
        # instruction should take over.
        def _is_blind_choice(ch):
            if ch is None or getattr(ch, "action", "NONE") == "NONE":
                return True
            gid = str(getattr(ch, "goal_id", ""))
            pk = str(getattr(ch, "plan_kind", ""))
            # a generic explore/curiosity action-trial is "blind" -- preempt it
            # with a solved-mechanic pursuit or an on-screen instruction (those
            # only fire when one actually exists, so this stays safe).
            return ("explore" in gid or "curiosity" in gid or "curiosity" in pk)
        # SOLVED-MECHANIC PURSUIT — once the click-to-move step is learned from the
        # one-shot demo, WALK the piece to the goal by the numeric next-click; this
        # is the solid plan, so it preempts everything below.
        if _is_blind_choice(choice):
            _pursue = self._move_law_pursuit(entities_for_actor)
            if _pursue is not None:
                choice = _pursue
                print(f"[turn {self.world.turn}] MOVE-LAW PURSUIT "
                      f"{choice.action!r} ({choice.rationale})", flush=True)
        # HIGHEST-VALUE LEAD — when COS has NO committed plan and the win is not
        # understood, an on-screen DESIGNER INSTRUCTION (aim line, legend,
        # highlighted marker/cursor) is the thing to act on FIRST: these games
        # teach the player on screen.  Verify it RIGHT AWAY, ahead of blind probing
        # AND ahead of deferring to a generic strategy turn (it IS the solid idea).
        if _is_blind_choice(choice):
            _instr = self._instruction_verification_probe(entities_for_actor)
            if _instr is not None:
                choice = _instr
                print(f"[turn {self.world.turn}] VERIFY-INSTRUCTION "
                      f"{choice.action!r} ({choice.rationale})", flush=True)
        # SELF-CALIBRATING MERGE PURSUIT — several same-appearance pieces (not a
        # collinear strip) + a distinct goal: the substrate chains the merges and
        # delivers (score-driven), deriving every parameter at run time.  Ordered
        # AFTER instruction-verification so an on-screen tutorial keeps priority, and
        # gated on a blind choice so it never overrides a real plan.
        if _is_blind_choice(choice):
            _merge = self._merge_pursuit(entities_for_actor)
            if _merge is not None:
                choice = _merge
                print(f"[turn {self.world.turn}] MERGE PURSUIT "
                      f"{choice.action!r} ({choice.rationale})", flush=True)
        # MOVER->TARGET EXPLOIT — on a blind curiosity choice, file target-seeking
        # claims (the mover-implies-target instinct) AND form structural claims
        # (similar_entities relates look-alike pieces), then run the highest-VoI
        # claim's probe.  This is what makes COS COMMIT to reaching a perceived
        # target instead of cycling the action space forever once the mover is
        # found -- the explore->exploit hand-off that _select_probe used to own but
        # which is skipped under use_strategy.  Gated on a blind choice (never
        # overrides a real plan); returns None until a probe-able claim exists.
        if _is_blind_choice(choice) and self.world.probe_state.get("controllable_found"):
            # gated on controllable_found so the cheap action-trials DISCOVER the
            # controls first (which action moves the mover); only THEN does the
            # exploit pursue a target -- explore controls, then exploit.
            _mst = getattr(self, "_substrate_mover_implies_target", None)
            if _mst:
                try:
                    _mst(entities_for_actor)
                except Exception:
                    pass
            # MEA-AUTHORITATIVE FIRST -- the planner (MeansEnds.next_step) decides the
            # step: WALK toward a target, or PUSH an intermediary ACROSS A BARRIER that
            # blocks walking; ordered so the agent fills a pushed/far slot before
            # stranding itself in a near one.  This is the single-planner consolidation;
            # _cardinal_pursuit becomes the fallback for the cases MEA abstains on (and
            # can be retired once MEA is proven), then the claim-directed CLICK probe.
            _mea = self._mea_pursuit(ss, entities_for_actor, move_step)
            if _mea is not None:
                choice = _mea
                print(f"[turn {self.world.turn}] MEA PURSUIT "
                      f"{choice.action!r} ({choice.rationale})", flush=True)
            else:
                _card = self._cardinal_pursuit(ss, entities_for_actor, move_step)
                if _card is not None:
                    choice = _card
                    print(f"[turn {self.world.turn}] CARDINAL PURSUIT "
                          f"{choice.action!r} ({choice.rationale})", flush=True)
                else:
                    _exploit = self._claim_directed_probe(entities_for_actor)
                    if _exploit is not None:
                        choice = _exploit
                        print(f"[turn {self.world.turn}] CLAIM-DIRECTED EXPLOIT "
                              f"{choice.action!r} ({choice.rationale})", flush=True)
        # AUTONOMOUS CURIOSITY PROBE — when grid planning yielded no action
        # (a non-grid / continuous-arena game, or the grid planner stalled),
        # fall back to a coverage-driven probe of the action space so COS can
        # LEARN what each action does instead of halting.  Game-agnostic: try
        # each coordinate-free action once, then CLICK each perceived entity
        # once; the effect is recorded by the normal delta-perception pass.
        # For a NON-GRID scene the autonomous probe DRIVES (so action-effect
        # discovery is hands-off — no per-probe strategy call); it also covers
        # the grid-stalled case (choice None / NONE).  Once probe coverage is
        # exhausted, a planner/cell_actor choice (if any) is kept as the
        # fallback so higher-level strategy can take over.
        if ((ss["agent_cell"] is None or choice is None
                or getattr(choice, "action", "NONE") == "NONE")
                and not self.world.probe_state.get("controllable_found")
                # KB-FIRST: if the KB already holds prior guidance for this game,
                # don't blind-probe -- fall through to the strategy handoff below
                # so the prior-guided strategy layer (with the recall surface) acts
                # on what is already known.  No-op for a game with an empty KB.
                and not self._consult_before_blind_probe()):
            # Active controlled-experiment tier: once coverage has surfaced a
            # transient mover + toggleable controls, this takes over and runs the
            # OFAT experiment (vary one control, fire the trigger, induce the
            # control->effect law, apply the proposed solve).  Fully guarded:
            # returns None until detectable / on any error, so coverage drives
            # until then and the run never breaks.
            probe = self._select_probe(entities_for_actor)
            if probe is not None:
                choice = probe
                print(f"[turn {self.world.turn}] CURIOSITY PROBE "
                      f"{choice.action!r} ({choice.rationale})")
            elif choice is None:
                # The probe escalates rather than exhausting, so it returns None
                # ONLY when the game advertises no actions at all.
                print("[driver] game advertises no actions; stopping.")
                return None

        # STRATEGY HANDOFF — once a controllable entity has been identified
        # from the probe deltas, the gate above stops blind probing.  If
        # neither the planner nor the probe produced a choice at that point,
        # synthesize a neutral, non-probe seed so the prior-guided strategy
        # layer is consulted (it will steer the controllable entity toward the
        # goal rather than continue sweeping the action space).
        if choice is None:
            _kb_first = (not self.world.probe_state.get("controllable_found")
                         and self._consult_before_blind_probe())
            if ((self.world.probe_state.get("controllable_found") or _kb_first)
                    and self.use_strategy):
                from types import SimpleNamespace
                seed_act = next(
                    (a for a in self.game.available_actions()
                     if a not in _CARDINAL_ALIASES and a != "NONE"),
                    "NONE",
                )
                _why = ("KB-first: the KB already holds prior knowledge for this "
                        "game; consult the recall surface and act on it rather "
                        "than blind-probing"
                        if _kb_first and not self.world.probe_state.get("controllable_found")
                        else "strategy handoff: a controllable entity is "
                             "identified; prior-guided strategy chooses the move")
                choice = SimpleNamespace(
                    action=seed_act, rationale=_why,
                    plan_kind="strategy_handoff", goal_id="strategy_handoff",
                    target_cell=None, full_plan_actions=[seed_act],
                    is_probe=False,
                )
                print(f"[turn {self.world.turn}] "
                      f"{'KB-FIRST HANDOFF' if _kb_first else 'STRATEGY HANDOFF'} "
                      f"(seed {seed_act!r}); strategy layer will choose.")
            else:
                # No planner/probe choice and no strategy to consult — but do
                # NOT halt while the game is playable (strict/competition mode:
                # getting stuck is not an option).  Fall to the never-halt
                # explorer; the run ends only on win/loss/turn-budget.
                choice = self._exploratory_probe_choice(entities_for_actor)
                if choice is None:
                    print("[driver] game advertises no actions; stopping.")
                    return None
                print(f"[turn {self.world.turn}] NEVER-STUCK EXPLORE "
                      f"{choice.action!r} ({choice.rationale})")

        # DIRECTED-INTENT OVERRIDE OF THE BLIND PROBE — once perception has
        # CONFIDENTLY localized the goal in a NON-GRID scene (no controllable
        # agent, but specific goal_candidate_cells), the VLM strategy layer
        # should DIRECT the action toward that goal instead of letting the
        # autonomous curiosity probe keep sweeping blindly.  This is what lets
        # COS ACT on a found win condition (e.g. click the tiles a matching
        # puzzle's key specifies) rather than only perceive it.  Game-agnostic:
        # keyed solely on perception's OWN goal_candidate_cells + confidence,
        # never on any game's rules.  The blind probe remains the fallback for
        # pure discovery (no localized goal yet).
        pss = getattr(self, "_last_perception_ss", {}) or {}
        directed_intent = (
            pss.get("agent_cell") is None
            and bool(pss.get("goal_candidate_cells"))
            and str(pss.get("confidence", "")).lower() in ("high", "medium")
        )
        if directed_intent and getattr(choice, "is_probe", False):
            print(f"[turn {self.world.turn}] DIRECTED INTENT: perception "
                  f"localized {len(pss.get('goal_candidate_cells'))} goal "
                  f"cell(s) (conf={pss.get('confidence')}); consulting strategy "
                  f"to ACT toward the goal instead of blind probing.")

        # Optionally consult the VLM strategy layer.  The strategy
        # call sees the full world snapshot + mechanical actor's
        # choice and returns an endorsed action.  If endorsed_action
        # differs from the mechanical choice, the driver uses the
        # endorsed one.  CLICK:<entity_name> endorsements get
        # translated to CLICK:px,py via the entity's bbox centroid.
        final_action = choice.action
        rationale = choice.rationale
        plan_kind = choice.plan_kind
        goal_id = choice.goal_id
        strategy_choice: Optional[StrategyChoice] = None
        # PROCEDURALIZATION (System 1): a confident compiled skill whose
        # relational signature matches the current situation runs directly,
        # bypassing the VLM strategy call.  Verified per-turn against the
        # delta; decays + falls back to System 2 if it stops working.
        proc_actions = (self._proc_maybe_autorun()
                        if choice.action != "NONE"
                        and not getattr(choice, "is_probe", False) else None)
        if proc_actions:
            final_action = proc_actions[0]
            rationale = "proceduralization: System-1 auto-run of a compiled skill"
            plan_kind = "skill_autorun"
            goal_id = "skill_autorun"
            if len(proc_actions) > 1:
                self._pending_sequence = list(proc_actions)
                self._sequence_idx = 1
                self._interrupt_conditions = []
                self._sequence_goal_id = goal_id
                self._sequence_committed_at_turn = self.world.turn
                self._sequence_rationale_prefix = (
                    "mechanical execution of auto-run skill")
        elif (self.use_strategy and choice.action != "NONE"
              and (not getattr(choice, "is_probe", False) or directed_intent)):
            available = self.game.available_actions()
            stage_strategy_call(
                self.world, self.work_dir,
                turn=self.world.turn,
                mech_action=choice.action,
                mech_plan_kind=choice.plan_kind,
                mech_goal_id=choice.goal_id,
                mech_rationale=choice.rationale,
                available_actions=available,
                repeat_feedback=self._repeat_feedback,
            )
            self._repeat_feedback = None  # surfaced once, then cleared
            print(f"[turn {self.world.turn}] strategy prompt: "
                  f"{self.work_dir / f'turn_{self.world.turn:03d}' / 'strategy_prompt.md'}")
            turn_dir = (self.work_dir
                         / f"turn_{self.world.turn:03d}")
            # Surface the firing instincts at the TOP of the strategy prompt so the
            # actor sees design-level guidance BEFORE choosing an action -- most
            # importantly first_level_is_tutorial (when the win is not yet
            # understood: READ THE FRAME AS INSTRUCTIONS / the first level is a
            # tutorial) so COS consults the on-screen tutorial instead of blind-
            # probing. Prepended (not appended) so it leads.
            try:
                _iblock = _INST.REGISTRY.render_active(
                    self._instinct_context("strategy"))
                _sp = turn_dir / "strategy_prompt.md"
                if _iblock and _sp.exists():
                    _sp.write_text(
                        "INSTINCTS (consult FIRST):\n" + _iblock + "\n\n"
                        + _sp.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
            # OPERATOR MENU: when the substrate built a merge/deliver menu for this
            # turn, append it so the VLM SELECTS the operator (it sees the scene).
            try:
                _sp = turn_dir / "strategy_prompt.md"
                if (getattr(choice, "plan_kind", "") == "merge_pursuit"
                        and getattr(self, "_merge_menu_text", None) and _sp.exists()):
                    _sp.write_text(_sp.read_text(encoding="utf-8")
                                   + "\n\n" + self._merge_menu_text + "\n",
                                   encoding="utf-8")
            except Exception:
                pass
            reply_path = turn_dir / "strategy_reply.txt"
            reply = poll_strategy_reply(
                reply_path,
                timeout_s=self.vlm_timeout_s, poll_s=self.poll_s,
            )
            # Substrate-boundary reply validator.  Rejects replies that
            # name entities perception doesn't track or assert state
            # changes not in the latest delta.  See
            # memory/session_2026_05_30_drift_failure.md.
            prompt_path = turn_dir / "strategy_prompt.md"
            original_prompt = (prompt_path.read_text(encoding="utf-8")
                                if prompt_path.exists() else "")
            attempt = 0
            max_attempts = 3
            # Only HARD violations (structured entity refs + false
            # state-change assertions) trigger an expensive re-poll.
            # Soft prose-entity mentions are surfaced but do not force a
            # re-call — the prose scan has a false-positive rate that
            # would otherwise multiply model usage (see cost review).
            violations = hard_violations(validate_reply(reply, self.world))
            while violations and attempt < max_attempts:
                print(f"[turn {self.world.turn}] reply validator "
                      f"REJECTED (attempt {attempt}): "
                      f"{len(violations)} hard violation(s)")
                for v in violations:
                    print(f"  - {v}")
                (turn_dir
                 / f"strategy_reply.rejected_{attempt}.json").write_text(
                    json.dumps(reply, indent=2), encoding="utf-8",
                )
                rejection_block = format_validator_rejection_block(
                    violations, attempt,
                )
                (turn_dir
                 / f"validator_violations_{attempt}.txt").write_text(
                    rejection_block, encoding="utf-8",
                )
                if prompt_path.exists():
                    prompt_path.write_text(
                        rejection_block + "\n" + original_prompt,
                        encoding="utf-8",
                    )
                reply = poll_strategy_reply(
                    reply_path,
                    timeout_s=self.vlm_timeout_s, poll_s=self.poll_s,
                )
                attempt += 1
                violations = hard_violations(validate_reply(reply, self.world))
            if violations:
                print(f"[turn {self.world.turn}] reply validator "
                      f"EXHAUSTED {max_attempts} retries; falling back "
                      f"to mechanical action {choice.action!r}")
                (turn_dir
                 / "strategy_reply.rejected_final.json").write_text(
                    json.dumps(reply, indent=2), encoding="utf-8",
                )
                reply = {
                    "endorsed_action": choice.action,
                    "rationale": (
                        f"VALIDATOR FALLBACK after {max_attempts} "
                        f"rejected attempts: "
                        + "; ".join(violations[:3])
                    ),
                    "confidence": "low",
                }
            # PLAN-CONSISTENCY detective check (soft) — flag when the
            # committed plan lexically resembles a REFUTED approach, so the
            # actor does not silently re-enter a cleared dead-end.  Kept
            # soft (surface next turn via repeat_feedback, no forced
            # re-poll) to avoid hangs/cost; the always-on PLAN GATE in the
            # prompt is the primary, preventive mechanism.  See
            # plan_consistency.py + memory/feedback_plan_must_consult_established_insights.
            try:
                from plan_consistency import (                     # noqa: E402
                    check_plan_against_refuted, plan_text_of,
                    format_refuted_match_warning,
                )
                from per_game_lessons import load_for_game         # noqa: E402
                _refuted = [
                    {"lesson_id": l.lesson_id, "description": l.description}
                    for l in load_for_game(self.world.game_id)
                    if l.kind == "refuted"
                ]
                _matches = check_plan_against_refuted(
                    plan_text_of(reply), _refuted)
                if _matches:
                    _warn = format_refuted_match_warning(_matches)
                    print(f"[turn {self.world.turn}] plan-consistency: "
                          f"plan resembles {len(_matches)} refuted "
                          f"approach(es) "
                          f"{[m.shared_terms for m in _matches[:2]]}; "
                          f"surfacing warning next turn")
                    (turn_dir / "plan_consistency_warning.txt").write_text(
                        _warn, encoding="utf-8")
                    # Surface on the NEXT strategy prompt (soft nudge).
                    self._repeat_feedback = (
                        (self._repeat_feedback + "\n\n")
                        if self._repeat_feedback else "") + _warn
            except Exception as e:
                print(f"[driver] plan-consistency check skipped ({e})")
            # STRUCTURAL detective check — project the committed
            # planned_action_sequence forward through the rigid-body
            # kinematics prior and flag any step whose arm_body bbox
            # overlaps an impassable obstacle. This is the gate that
            # catches "bend the body through the wall" plans that the
            # textual refuted matcher misses (the wall has no textual
            # avoid-list entry — it's a geometric fact). Same wiring as
            # the textual check above: soft, surfaced as repeat_feedback.
            try:
                from plan_consistency import (                  # noqa: E402
                    simulate_plan_against_structure,
                    format_structural_violations,
                )
                # VLM TEACHES a recognizer: reply may carry
                # {"teach_recognizer": {"name","target_types","labels":{id:lab}}}.
                # Capture exemplars from the CURRENT frame + persist, so the
                # substrate applies it cheaply every turn thereafter.
                try:
                    _teach = (reply.get("teach_recognizer")
                              if isinstance(reply, dict) else None)
                    if (_teach and self._instance_tracker is not None
                            and self._last_frame_path):
                        import taught_recognizers as _trc          # noqa: E402
                        _gid = getattr(self.world, "game_id", "?")
                        _recs = _trc.load_recognizers(_gid)
                        _nm = _teach.get("name")
                        _rec = _trc.teach(
                            _nm, _teach.get("target_types") or [],
                            str(self._last_frame_path),
                            self._instance_tracker.instances,
                            _teach.get("labels") or {},
                            existing=_recs.get(_nm))
                        _recs[_nm] = _rec
                        _trc.save_recognizers(_gid, _recs)
                        print(f"[driver] taught recognizer '{_nm}' "
                              f"({len(_rec.exemplars)} exemplars)")
                except Exception as _te:
                    print(f"[driver] teach_recognizer skipped ({_te})")
                # Operator the actor declares it is APPLYING this turn (optional
                # reply field). Its per-game verification is judged by the
                # outcome on the next delta (no state change -> refuted; the
                # claimed effect -> confirmed). This is the replay-verification
                # gate: a mined-generic operator is trusted for a game only once
                # it actually works there.
                try:
                    self.world._pending_operator = (
                        reply.get("applied_operator_id")
                        if isinstance(reply, dict) else None)
                except Exception:
                    self.world._pending_operator = None
                _seq = reply.get("planned_action_sequence") if isinstance(reply, dict) else None
                if isinstance(_seq, list) and _seq:
                    _obstacles = list(getattr(self.world,
                                              "_structural_obstacles", None) or [])
                    _kinematics = getattr(self.world, "_rigid_body_kinematics", None)
                    _init_state = getattr(self.world, "_arm_state_estimate", None)
                    if _obstacles and _kinematics and _init_state:
                        _violations = simulate_plan_against_structure(
                            _seq, initial_state=_init_state,
                            kinematics=_kinematics, obstacles=_obstacles,
                        )
                        if _violations:
                            _struct_warn = format_structural_violations(_violations)
                            print(f"[turn {self.world.turn}] plan-consistency "
                                  f"STRUCTURAL: {len(_violations)} step(s) "
                                  f"project arm_body into an impassable "
                                  f"obstacle; first violation step "
                                  f"{_violations[0].step_index} action "
                                  f"{_violations[0].action} -> "
                                  f"{_violations[0].overlap_with}")
                            (turn_dir / "plan_consistency_structural_warning.txt"
                             ).write_text(_struct_warn, encoding="utf-8")
                            self._repeat_feedback = (
                                (self._repeat_feedback + "\n\n")
                                if self._repeat_feedback else "") + _struct_warn
            except Exception as e:
                print(f"[driver] plan-consistency STRUCTURAL check skipped "
                      f"({e})")
            strategy_choice = apply_strategy(
                choice.action, reply, available,
            )
            # Acting VLM owns interpretation (perception emits none):
            # persist its game_type / game_purpose onto the world so
            # they surface each turn + appear in the trace.  Refine
            # forward — only overwrite when the actor supplied a value.
            if strategy_choice.game_type:
                self.world.game_type_guess = strategy_choice.game_type
            if strategy_choice.game_purpose:
                self.world.game_purpose_guess = strategy_choice.game_purpose
                # game_purpose is a CLAIM subject to validation, not a
                # free string: sync it into the WinConditionHypothesis
                # lifecycle so a materially-changed purpose REFUTES the
                # prior claim and mints a new one.  The trial-end
                # distiller persists refuted/confirmed purpose claims
                # cross-trial (per_game_lessons), so the next trial
                # avoids refuted purposes and builds toward a better
                # guess.  See SPEC_goal_grounding_and_state_diff.
                try:
                    self._sync_game_purpose_claim(
                        strategy_choice.game_purpose,
                    )
                except Exception as e:
                    print(f"[driver] game_purpose claim sync failed "
                          f"({e}); continuing")
            # Stash this turn's prediction so next turn's ground-truth
            # block can echo it beside what actually happened (eager
            # predict-then-falsify).  See SPEC_goal_grounding.
            try:
                self.world._last_prediction = strategy_choice.prediction
            except Exception:
                pass
            print(f"[turn {self.world.turn}] strategy endorsed "
                  f"{strategy_choice.endorsed_action!r} "
                  f"(override={strategy_choice.overrode_mechanical} "
                  f"conf={strategy_choice.confidence})")
            # Translate CLICK:<entity_name> -> CLICK:px,py via bbox
            final_action = self._resolve_action(
                strategy_choice.endorsed_action,
            )
            rationale = (strategy_choice.rationale
                          + f"  [mechanical was {choice.action}]")
            plan_kind = (f"vlm_strategy_override"
                         if strategy_choice.overrode_mechanical
                         else "vlm_strategy_endorse")
            # Apply commit_subgoal NOW (before the ActionRecord)
            # so the new subgoal_id is available for the goal_id
            # override below.  Status updates wait until after the
            # action executes so the actor's achieved/abandoned
            # verdict can reflect post-action state.
            try:
                committed_sg_id = self._apply_subgoal_commits(
                    strategy_choice,
                )
            except Exception as e:
                print(f"[driver] subgoal commit failed "
                      f"({e}); continuing")
                committed_sg_id = None
            # Override goal_id with subgoal context when the actor
            # named one (pursuing_subgoal_id) or just committed
            # one.  Otherwise fall back to mechanical's goal_id.
            goal_id = self._resolve_goal_id_from_subgoal(
                strategy_choice,
                committed_id=committed_sg_id,
                fallback_goal_id=goal_id,
            )
            # Capture planned_action_sequence — if the actor
            # committed a multi-step plan, queue actions 2..N for
            # mechanical execution on subsequent turns.  Action 1
            # of the sequence is the one being executed THIS turn
            # (the endorsed_action that should already match
            # sequence[0]).  Subsequent run_one_step calls will
            # short-circuit into _run_mechanical_step until the
            # sequence completes or an interrupt fires.
            planned_seq = getattr(
                strategy_choice, "planned_action_sequence", None,
            )
            if planned_seq and isinstance(planned_seq, list):
                # Sanity check: first action in sequence should
                # match the endorsed action.  If not, log and
                # use the endorsed action as the prefix.
                if planned_seq and planned_seq[0] != strategy_choice.endorsed_action:
                    print(
                        f"[driver] planned_action_sequence[0]="
                        f"{planned_seq[0]!r} doesn't match "
                        f"endorsed_action={strategy_choice.endorsed_action!r}; "
                        "executing the endorsed action and queuing the rest"
                    )
                # Queue actions 2..N (index 1 onward) for mechanical
                # mode.  After this turn's strategic step executes
                # action 0, the next run_one_step call will execute
                # action 1 mechanically, then action 2, etc.
                self._pending_sequence = list(planned_seq)
                self._sequence_idx = 1
                self._interrupt_conditions = list(
                    getattr(strategy_choice, "interrupt_conditions",
                            None) or []
                )
                self._sequence_goal_id = goal_id
                self._sequence_committed_at_turn = self.world.turn
                self._sequence_rationale_prefix = (
                    f"mechanical execution of sequence committed "
                    f"at t{self.world.turn}"
                )
                if len(planned_seq) > 1:
                    print(
                        f"[driver] planned_action_sequence queued: "
                        f"{len(planned_seq)} steps, interrupts="
                        f"{self._interrupt_conditions!r}"
                    )

        # REPEAT-UNTIL-RELATION (game-agnostic magnitude executor).  Takes
        # priority over a literal planned_action_sequence when present.  The
        # harness repeats ONE opaque action until the relational stop-condition
        # holds, the action stalls, or a completed goal target regresses
        # (drop-guard -> UNDO).  All repeats (including the first) run through
        # _run_mechanical_step so they share its perceive/mine/guard machinery.
        ra = getattr(strategy_choice, "repeat_action", None) if strategy_choice else None
        ru = getattr(strategy_choice, "repeat_until", None) if strategy_choice else None
        if ra and ru and ra in self.game.available_actions():
            self._pending_sequence = [ra]
            self._sequence_idx = 0
            self._repeat_until = ru
            self._repeat_count = 0
            self._repeat_done_baseline = self._goal_done_set()
            self._repeat_skewered_baseline = self._skewered_collectables()
            self._repeat_feedback = None
            self._interrupt_conditions = []
            self._sequence_goal_id = goal_id
            self._sequence_committed_at_turn = self.world.turn
            self._sequence_rationale_prefix = (
                f"repeat {ra} until [{ru}] (harness-owned magnitude)")
            print(f"[driver] REPEAT-UNTIL armed: repeat {ra} until "
                  f"[{ru}] (cap {self._repeat_cap}, drop-guard on goal "
                  f"done-set {sorted(self._repeat_done_baseline or [])})")
            return self._run_mechanical_step()

        action_record = ActionRecord(
            turn=self.world.turn, action=final_action,
            rationale=rationale,
            actor_chose_from=plan_kind,
            goal_id=goal_id,
            target_cell=(list(choice.target_cell)
                         if choice.target_cell else None),
            full_plan_actions=list(
                getattr(choice, "full_plan_actions", []) or []
            ),
        )
        self.world.record_action(action_record)
        self._proc_seg_append(final_action)
        print(f"[turn {self.world.turn}] executing {final_action} "
              f"(plan={plan_kind}, goal={goal_id})")
        if final_action == "NONE":
            return None

        # step the game
        try:
            _gs=time.time()
            step_result = self.game.step(final_action)
            if os.environ.get("COS_TIMING"):
                print(f"[timing] game.step={time.time()-_gs:.1f}s", file=sys.stderr, flush=True)
        except StopIteration:
            print("[driver] adapter has no more steps.")
            return None
        prev_frame = self._last_frame_path
        self._last_frame_path = step_result.frame_path
        prev_score = self.world.score
        self.world.turn = self.game.current_turn
        if step_result.win_state != "playing":
            self.world.win_state = step_result.win_state
        if step_result.lives is not None:
            self.world.lives = step_result.lives
        if step_result.score is not None:
            self.world.score = step_result.score

        # ---- subroutine credence + auto-promotion hooks ----------
        # If the strategy actor recorded an applied_subroutine on the
        # last reply, update its credence based on whether the score
        # advanced (success), held (partial), or did the action chain
        # backfire (failure — heuristic).  And on score advance,
        # auto-promote the most recent action window as a candidate
        # new Subroutine via a separate VLM call.
        score_advanced = (
            self.world.score is not None
            and prev_score is not None
            and self.world.score > prev_score
        )
        # Credence policy.  In order of priority:
        #   1. score_advanced            -> hard win signal: success
        #   2. actor self-reported       -> use the actor's value
        #      (succeeded/partial/failed)
        #   3. neither                   -> no_op (provenance only)
        # This protects credence from inflating on every mid-
        # application turn just because nothing failed.
        actor_status = (
            getattr(strategy_choice, "subroutine_application_status",
                    None)
            if strategy_choice else None
        )
        if score_advanced:
            outcome = "success"
        elif actor_status in {"succeeded", "success"}:
            outcome = "success"
        elif actor_status == "partial":
            outcome = "partial"
        elif actor_status in {"failed", "failure"}:
            outcome = "failure"
        else:
            outcome = "no_op"

        self._update_subroutine_credence(
            applied_subroutine=getattr(
                strategy_choice, "applied_subroutine", None,
            ) if strategy_choice else None,
            fork_parent=getattr(
                strategy_choice, "fork_parent", None,
            ) if strategy_choice else None,
            variant_notes=getattr(
                strategy_choice, "variant_notes", None,
            ) if strategy_choice else None,
            outcome=outcome,
            action_record=action_record,
        )
        # ---- ActiveSubgoal status update (commits already done
        # before ActionRecord) — status verdicts may reflect the
        # post-action world state, so they fire AFTER the step.
        try:
            self._apply_subgoal_status_updates(strategy_choice)
        except Exception as e:
            print(f"[driver] subgoal status update failed "
                  f"({e}); continuing")
        if score_advanced:
            self._schedule_subroutine_auto_promotion(
                window_size=12, advanced_to_score=self.world.score,
            )
            # Register the solved level's win-path into the canonical
            # solutions_kb (replayable; shortest stays canonical).
            try:
                self._register_solution_on_solve()
            except Exception as e:
                print(f"[driver] solutions_kb registration failed "
                      f"({e}); continuing")

        # stage delta perception — pass the ACTUALLY-EXECUTED
        # action (not the mechanical planner's proposal) so the
        # perception layer interprets the frame transition
        # correctly when the strategy actor overrode.
        prompt_path = self.stage_delta_perception(
            prev_frame, step_result.frame_path,
            turn_n=self.world.turn, last_action=final_action,
            frame_stack=getattr(step_result, "frame_stack", None),
            anim_dir=getattr(step_result, "anim_dir", None),
        )
        print(f"[turn {self.world.turn}] delta-perception prompt: {prompt_path}")
        reply = self.poll_reply(self.world.turn)
        delta_json = reply.get("delta") or {}
        perception_json = reply.get("perception") or {}

        # ingest perception
        perception_json = self._ground_perception_geometry(
            perception_json, step_result.frame_path)
        self.world.ingest_perception(perception_json)
        self._refresh_bboxes_from_frame()
        self._verify_pending_operator()
        # build DeltaRecord — use the executed action, not the
        # planner's proposal.  The trace's "delta to t{N+1}" line
        # was previously showing the planner's curiosity choice
        # (ACTION6) even when the strategy actor overrode, which
        # looked like a turn-to-turn disjunction.
        delta = DeltaRecord(
            from_turn=self.world.turn - 1,
            to_turn=self.world.turn,
            action=final_action,
            agent_moved=bool(delta_json.get("agent_moved")),
            agent_new_cell=(list(delta_json["agent_new_cell"])
                            if delta_json.get("agent_new_cell") else None),
            inferred_action=delta_json.get("inferred_action"),
            entities_appeared=list(delta_json.get("entities_appeared") or []),
            entities_disappeared=list(
                delta_json.get("entities_disappeared") or []
            ),
            entities_changed=list(delta_json.get("entities_changed") or []),
            summary=delta_json.get("summary", ""),
        )
        # GROUND-TRUTH GUARD: the picture wins -- if the pixels show the agent
        # vacated its cell but the VLM reported agent_moved=false, correct it.
        self._reconcile_agent_move(delta)
        # ESCALATION SIGNAL — the moment a COORDINATE-FREE action is observed
        # to relocate a controllable entity (object constancy on effects:
        # agent_moved), record it so the loop STOPS blind probing and hands
        # off to the prior-guided strategy layer (see the gate + handoff in
        # run_one_step).  One confirmed move is enough to know the game has a
        # steerable entity; the strategy VLM maps the rest and navigates.
        if (delta.agent_moved
                and final_action not in (_CARDINAL_ALIASES | {"NONE"})
                and not str(final_action).startswith("CLICK")
                and not self.world.probe_state.get("controllable_found")):
            self.world.probe_state["controllable_found"] = True
            print(f"[turn {self.world.turn}] CONTROLLABLE ENTITY IDENTIFIED "
                  f"(action {final_action!r} relocated it) -> halting blind "
                  f"probing; escalating to prior-guided strategy.")
        # Substrate-computed visual events — pixel-diff inside
        # watched bboxes (default: role=hud entities).  Game-
        # agnostic.  Errors are swallowed; events stay empty.
        try:
            from visual_events import (                           # noqa: E402
                compute_internal_pixel_events as _viz_compute,
            )
            delta.visual_events = _viz_compute(
                prev_frame, step_result.frame_path, self.world,
            )
            if delta.visual_events:
                print(
                    f"[driver] visual events: "
                    f"{len(delta.visual_events)} entit(ies) "
                    f"showed internal pixel changes: "
                    f"{[e.get('entity') for e in delta.visual_events]}"
                )
        except Exception as e:
            print(f"[driver] visual-events compute failed "
                  f"({e}); continuing")
        # Substrate ENTITY-LEVEL animation movements (+ filmstrip) for THIS
        # turn, carried on the delta so the trace can render what was found in
        # the animation frames -- not just consumed by the prompt.
        delta.animation_events = list(getattr(self, "_last_anim_events", []) or [])
        delta.animation_filmstrip = getattr(self, "_last_filmstrip_path", None)
        delta.animation_entities_filmstrip = getattr(
            self, "_last_entanalysis_filmstrip", None)
        delta.score_increased = bool(score_advanced)
        delta.win_state_changed = (
            self.world.win_state != "playing"
        )
        self.world.ingest_delta(delta)
        # SELF-CORRECTION: watch the progress TREND (distance-to-goal shrinking,
        # mover travelling straight to the goal) and auto-raise the debug protocol
        # on a drift/stall the single-step surprise check cannot see.
        self._monitor_progress()
        # DEMONSTRATION LEARNING: if the instruction-verification click just
        # relocated a piece to where a one-shot marker was, extract the numeric
        # click-to-move step so COS can reapply it itself once the marker is gone.
        self._learn_move_law_from_demo()
        # PROCEDURALIZATION: if this delta completed the active goal
        # segment's target (the goal's next-target advanced), compile the
        # maneuver into the canonical subroutine KB so it can auto-run next
        # time the same relational signature recurs.
        self._proc_on_perception()

        # ---- Crystallization: event-triggered win-condition credit
        # assignment.  On a score / lc / win advance, derive a CHECKABLE,
        # role-keyed win relation by contrasting the advancing transition
        # against non-advancing ones, and attach it to a
        # WinConditionHypothesis so the next turn (and next trial) can
        # diff against it instead of re-deriving the objective.  See
        # SPEC_cumulative_learning_loop.md § Crystallization.  Defensive:
        # never let it break the turn; quality is gated by perception
        # quality (a trace with bad HUD perception yields a bad relation,
        # so this is only as good as the live perception path).
        if delta.score_increased or delta.win_state_changed:
            try:
                from knowledge_crystallization import (         # noqa: E402
                    derive_win_condition, commit_derived_win_relation,
                )
                rel = derive_win_condition(self.world)
                if rel is not None:
                    commit_derived_win_relation(self.world, rel)
                    print(f"[driver] crystallized win relation on advance: "
                          f"{rel}")
                    # Populate the actor's objective fields from the
                    # SUBSTRATE-DERIVED relation when they are still empty,
                    # so game_type/game_purpose carry a concrete, grounded
                    # objective from the moment it is known (not blank until
                    # the VLM guesses).  Only fills when empty — the actor
                    # is free to refine it on its next strategy reply.
                    try:
                        from knowledge_crystallization import (   # noqa: E402
                            describe_win_relation, game_type_from_relation,
                        )
                        if not (self.world.game_purpose_guess or "").strip():
                            self.world.game_purpose_guess = \
                                describe_win_relation(rel)
                        if not (self.world.game_type_guess or "").strip():
                            self.world.game_type_guess = \
                                game_type_from_relation(rel)
                    except Exception as e:
                        print(f"[driver] objective-field fill skipped ({e})")
            except Exception as e:
                print(f"[driver] win-condition crystallization "
                      f"skipped ({e})")

        # ---- WinConditionHypothesis commits + observations.
        # Called AFTER ingest_delta so the actor's observations
        # (recorded via win_condition_observation in the strategy
        # reply) reference the just-ingested delta index.
        try:
            self._apply_win_condition_updates(strategy_choice)
        except Exception as e:
            print(f"[driver] win-condition update failed "
                  f"({e}); continuing")
        # Write side of the DO-NOT-REPEAT avoid-list: a hypothesis-
        # tagged action that produced a no-op delta is a falsified
        # tactical assumption — auto-register it as 'refuted' so the
        # actor won't silently repeat it.  Mechanical; not dependent
        # on the actor authoring it at trial end.
        try:
            self._register_failed_assumption(strategy_choice)
        except Exception as e:
            print(f"[driver] failed-assumption registration "
                  f"skipped ({e})")
        # ---- Probe ledger commits / observations / abandons.
        # Observations propagate credences to referenced WC and
        # Mechanic hypotheses automatically.
        try:
            self._apply_probe_updates(strategy_choice)
        except Exception as e:
            print(f"[driver] probe-ledger update failed "
                  f"({e}); continuing")

        # ---- Subgoal Completion Contract: substrate-authority
        # transitions.  Runs AFTER ingest_delta + WC updates so
        # acceptance tests see this turn's delta and invalidation
        # sees current WC credences.  Records the pursued subgoal's
        # (action, state-class) into its exhaustion ledger first.
        try:
            self._record_subgoal_approach(action_record, delta)
            self._run_subgoal_contract(delta)
        except Exception as e:
            print(f"[driver] subgoal contract eval failed "
                  f"({e}); continuing")

        # mine mechanic hypotheses
        touched = mine_step(self.world, action_record, delta)
        promoted_rules = [r.hypothesis_id for r in trusted_rules(self.world)]

        # ---- Sequence interrupt check.  If a planned_action_sequence
        # is in flight, check whether any interrupt condition fired
        # this turn; if so, clear the pending sequence so the next
        # run_one_step call goes back to the VLM strategy actor.
        if self._pending_sequence:
            interrupt = self._check_sequence_interrupts(
                delta, prev_score,
            )
            if interrupt is not None:
                print(
                    f"[driver] sequence INTERRUPT fired: "
                    f"{interrupt!r} after step "
                    f"{self._sequence_idx}/"
                    f"{len(self._pending_sequence)}; "
                    "clearing pending sequence; next call "
                    "will re-prompt VLM"
                )
                self._pending_sequence = []
                self._sequence_idx = 0
                self._interrupt_conditions = []
                self._sequence_goal_id = None
                self._sequence_committed_at_turn = None
                self._sequence_rationale_prefix = ""
            elif self._sequence_idx >= len(self._pending_sequence):
                # Sequence finished normally
                print(
                    f"[driver] sequence COMPLETE; "
                    "next call will re-prompt VLM"
                )
                self._pending_sequence = []
                self._sequence_idx = 0
                self._interrupt_conditions = []
                self._sequence_goal_id = None
                self._sequence_committed_at_turn = None
                self._sequence_rationale_prefix = ""

        # LEVEL-START ENTITY ANALYSIS: a score advance means the board is now
        # a NEW level — run a fresh discovery pass on the new frame so the new
        # level's structures get tracked (and become reasoning targets), and
        # record it for the trace.  End-of-step so all per-step processing ran
        # on the consistent prior entity set first.  Defensive — never break.
        if score_advanced:
            try:
                self.run_level_start_analysis(step_result.frame_path)
            except Exception as e:
                print(f"[driver] level-start analysis skipped ({e})")
            # Force an un-debounced trace mirror NOW so the canonical
            # bookmark trace reflects the new level's entity analysis
            # immediately (the debounced per-turn mirror could otherwise
            # leave the bookmarked trace one level behind).
            try:
                self._render_and_mirror_trace()
            except Exception as e:
                print(f"[driver] post-level-start trace mirror skipped ({e})")

        # save world
        self.world.save(self.work_dir / "world_knowledge.json")

        report = TurnReport(
            turn=self.world.turn, action=choice.action,
            inferred_action=delta.inferred_action or "UNKNOWN",
            agent_moved=delta.agent_moved,
            new_cell=delta.agent_new_cell,
            entities_changed=(
                len(delta.entities_appeared)
                + len(delta.entities_disappeared)
                + len(delta.entities_changed)
            ),
            new_mechanic_hypotheses=touched,
            promoted_rules=promoted_rules,
        )
        self._reports.append(report)
        self._maybe_mirror_trace()
        return report

    def _run_mechanical_step(self) -> Optional[TurnReport]:
        """Execute the next action in the pending planned sequence
        WITHOUT calling the VLM strategy actor.  Substrate-side
        observations (delta, visual_events, mechanic mining) still
        happen normally.  Returns a TurnReport like run_one_step."""
        if not self._pending_sequence:
            return None
        repeat_mode = self._repeat_until is not None
        if not repeat_mode and self._sequence_idx >= len(self._pending_sequence):
            return None

        # In repeat mode the single action is always _pending_sequence[0];
        # the index/length bookkeeping is bypassed (the stop decision is made
        # in the end-block below from the relational condition, not the count).
        action_str = self._pending_sequence[0 if repeat_mode else self._sequence_idx]
        next_idx_after = self._sequence_idx + 1
        # Translate CLICK:<entity_name> if present
        final_action = self._resolve_action(action_str)

        # ActionRecord — tag goal_id with the sequence's
        # committing-turn so the trace shows which strategic
        # decision this mechanical step belongs to.
        action_record = ActionRecord(
            turn=self.world.turn,
            action=final_action,
            rationale=(
                f"{self._sequence_rationale_prefix} "
                f"(step {next_idx_after}/"
                f"{len(self._pending_sequence)})"
            ),
            actor_chose_from="mechanical_sequence_step",
            goal_id=self._sequence_goal_id or "mechanical_sequence",
            target_cell=None,
            full_plan_actions=[final_action],
        )
        self.world.record_action(action_record)
        print(
            f"[turn {self.world.turn}] MECHANICAL step "
            f"{next_idx_after}/{len(self._pending_sequence)}: "
            f"executing {final_action} "
            f"(goal={action_record.goal_id})"
        )
        if final_action == "NONE":
            return None

        # step the game
        try:
            _gs=time.time()
            step_result = self.game.step(final_action)
            if os.environ.get("COS_TIMING"):
                print(f"[timing] game.step={time.time()-_gs:.1f}s", file=sys.stderr, flush=True)
        except StopIteration:
            print("[driver] adapter has no more steps.")
            return None
        prev_frame = self._last_frame_path
        self._last_frame_path = step_result.frame_path
        prev_score = self.world.score
        self.world.turn = self.game.current_turn
        if step_result.win_state != "playing":
            self.world.win_state = step_result.win_state
        if step_result.lives is not None:
            self.world.lives = step_result.lives
        if step_result.score is not None:
            self.world.score = step_result.score

        # Advance sequence pointer
        self._sequence_idx = next_idx_after

        # Perception (delta) — same flow as the strategic path.
        prompt_path = self.stage_delta_perception(
            prev_frame, step_result.frame_path,
            turn_n=self.world.turn, last_action=final_action,
            frame_stack=getattr(step_result, "frame_stack", None),
            anim_dir=getattr(step_result, "anim_dir", None),
        )
        print(
            f"[turn {self.world.turn}] delta-perception prompt: "
            f"{prompt_path}"
        )
        reply = self.poll_reply(self.world.turn)
        delta_json = reply.get("delta") or {}
        perception_json = reply.get("perception") or {}
        perception_json = self._ground_perception_geometry(
            perception_json, step_result.frame_path)
        self.world.ingest_perception(perception_json)
        self._refresh_bboxes_from_frame()
        self._verify_pending_operator()
        delta = DeltaRecord(
            from_turn=self.world.turn - 1,
            to_turn=self.world.turn,
            action=final_action,
            agent_moved=bool(delta_json.get("agent_moved")),
            agent_new_cell=(list(delta_json["agent_new_cell"])
                            if delta_json.get("agent_new_cell") else None),
            inferred_action=delta_json.get("inferred_action"),
            entities_appeared=list(delta_json.get("entities_appeared") or []),
            entities_disappeared=list(
                delta_json.get("entities_disappeared") or []
            ),
            entities_changed=list(delta_json.get("entities_changed") or []),
            summary=delta_json.get("summary", ""),
        )
        # GROUND-TRUTH GUARD: picture wins over a wrong agent_moved=false.
        self._reconcile_agent_move(delta)
        try:
            from visual_events import (                           # noqa: E402
                compute_internal_pixel_events as _viz_compute,
            )
            delta.visual_events = _viz_compute(
                prev_frame, step_result.frame_path, self.world,
            )
            if delta.visual_events:
                print(
                    f"[driver] visual events: "
                    f"{len(delta.visual_events)} entit(ies) "
                    f"showed internal pixel changes: "
                    f"{[e.get('entity') for e in delta.visual_events]}"
                )
        except Exception as e:
            print(f"[driver] visual-events compute failed "
                  f"({e}); continuing")
        delta.score_increased = (
            self.world.score is not None and prev_score is not None
            and self.world.score > prev_score
        )
        delta.win_state_changed = (self.world.win_state != "playing")
        self.world.ingest_delta(delta)

        # Subgoal Completion Contract — same substrate-authority
        # pass as the strategic path (acceptance + invalidation),
        # plus recording the pursued subgoal's approach.  Mechanical
        # steps inherit the committing turn's goal_id.
        try:
            self._record_subgoal_approach(action_record, delta)
            self._run_subgoal_contract(delta)
        except Exception as e:
            print(f"[driver] subgoal contract eval failed "
                  f"({e}); continuing")

        # Mine mechanics
        touched = mine_step(self.world, action_record, delta)
        promoted_rules = [r.hypothesis_id for r in trusted_rules(self.world)]

        # Interrupt check
        interrupt = self._check_sequence_interrupts(
            delta, prev_score,
        )
        if repeat_mode:
            # REPEAT-UNTIL: decide stop (condition met / drop-guard / stall /
            # cap) vs. continue (re-queue the same action).  All decisions are
            # from the relational delta + the goal done-set; the harness owns
            # the magnitude.
            self._handle_repeat_step(delta, interrupt)
        elif interrupt is not None:
            print(
                f"[driver] sequence INTERRUPT fired: "
                f"{interrupt!r} after mechanical step "
                f"{next_idx_after}/{len(self._pending_sequence)}; "
                "clearing pending sequence; next call "
                "will re-prompt VLM"
            )
            self._pending_sequence = []
            self._sequence_idx = 0
            self._interrupt_conditions = []
            self._sequence_goal_id = None
            self._sequence_committed_at_turn = None
            self._sequence_rationale_prefix = ""
        elif self._sequence_idx >= len(self._pending_sequence):
            print(
                f"[driver] sequence COMPLETE; "
                "next call will re-prompt VLM"
            )
            self._pending_sequence = []
            self._sequence_idx = 0
            self._interrupt_conditions = []
            self._sequence_goal_id = None
            self._sequence_committed_at_turn = None
            self._sequence_rationale_prefix = ""

        # LEVEL-START ENTITY ANALYSIS on score advance (mechanical path).
        # Symmetric to the strategic path's trigger (see ~line 2480): a score
        # advance means the board is now a NEW level, so run a fresh entity
        # discovery on the new frame.  Previously this path SKIPPED the
        # trigger, so any trial whose level-crossing action lived inside a
        # planned_action_sequence (i.e. most of them) silently bypassed
        # discovery and inherited the prior level's entities.
        if (delta is not None and prev_score is not None
                and self.world.score is not None
                and self.world.score > prev_score):
            try:
                self.run_level_start_analysis(step_result.frame_path)
            except Exception as e:
                print(f"[driver] mech-path level-start analysis skipped ({e})")
            try:
                self._render_and_mirror_trace()
            except Exception as e:
                print(f"[driver] mech-path post-level-start trace mirror skipped ({e})")

        self.world.save(self.work_dir / "world_knowledge.json")

        report = TurnReport(
            turn=self.world.turn, action=final_action,
            inferred_action=delta.inferred_action or "UNKNOWN",
            agent_moved=delta.agent_moved,
            new_cell=delta.agent_new_cell,
            entities_changed=(
                len(delta.entities_appeared)
                + len(delta.entities_disappeared)
                + len(delta.entities_changed)
            ),
            new_mechanic_hypotheses=touched,
            promoted_rules=promoted_rules,
        )
        self._reports.append(report)
        self._maybe_mirror_trace()
        return report

    def _check_sequence_interrupts(
        self, delta, prev_score,
    ) -> Optional[str]:
        """Return the name of the first triggered interrupt or
        None.  Interrupts only fire when the matching event is
        listed in self._interrupt_conditions OR is structural
        (win_state_change always interrupts; sequence_complete is
        handled separately by the caller).
        """
        conditions = set(self._interrupt_conditions or [])

        # Score signals
        if (delta is not None and prev_score is not None
                and self.world.score is not None):
            if (self.world.score > prev_score
                    and "score_advance" in conditions):
                return "score_advance"
            if (self.world.score < prev_score
                    and "score_decrease" in conditions):
                return "score_decrease"

        # Visual events
        if "visual_event" in conditions:
            ve = getattr(delta, "visual_events", None) or []
            if ve:
                return f"visual_event ({len(ve)} fired)"

        # Entity appear / disappear
        if ("entity_appeared" in conditions
                and (delta.entities_appeared or [])):
            return "entity_appeared"
        if ("entity_disappeared" in conditions
                and (delta.entities_disappeared or [])):
            return "entity_disappeared"

        # Unexpected silent — action had no observable effect at
        # all (no movement, no entity changes).  Coarse heuristic.
        if "unexpected_silent" in conditions:
            silent = (
                not delta.agent_moved
                and not delta.entities_changed
                and not delta.entities_appeared
                and not delta.entities_disappeared
            )
            if silent:
                return "unexpected_silent"

        # Relation-based interrupts — evaluate the actor's declared
        # conditions against THIS turn's Layer A relations, so an
        # open-loop sequence halts when the geometry says stop (e.g.
        # 'relation:same_col' fires when the extending arm enters the
        # next block's column band; 'clearance:right<=1' when it closes
        # to one cell).  This is the fix for the turn-21/22 overshoot:
        # the bulk-extend would stop on adjacency instead of running a
        # step past its target.  Game-agnostic; conditions reference
        # relation kinds/dirs/entities only.
        rels = getattr(delta, "relations", None) or []
        if rels:
            try:
                from relational_kinematics import (              # noqa: E402
                    evaluate_interrupt_condition,
                )
                for cond in (self._interrupt_conditions or []):
                    if evaluate_interrupt_condition(cond, rels):
                        return f"relation_interrupt:{cond}"
            except Exception:
                pass

        # win_state_change is always an interrupt regardless of
        # what the actor declared — losing or winning is not
        # something the actor should run past mechanically.
        if self.world.win_state != "playing":
            return f"win_state_change ({self.world.win_state})"

        return None

    def replay_prefix(self, target_level: int) -> bool:
        """FAST-FORWARD the live game to ``target_level`` by deterministically
        replaying the canonical recorded solutions for levels
        ``0 .. target_level-1`` -- with NO VLM calls (budget-free).  Reuses the
        canonical ``solutions_kb`` artifacts saved on every ``lc++``
        (``_register_solution_on_solve``); it is a pure CONSUMER, so it never
        creates a duplicate solution.

        Returns True if the game reached ``target_level``; False if any prior
        level's recorded solution is missing or stale (the caller then falls
        back to full play from level 0).  On success the driver's frame /
        turn / score / level-ordinal are advanced so live play continues
        seamlessly from the working level, and the next solve registers ONLY
        the working level's win-path (``_level_start_action_idx`` reset)."""
        if not target_level or target_level <= 0:
            return False
        try:
            import sys as _sys
            from pathlib import Path as _P
            _adapter = (_P(__file__).resolve().parents[3]
                        / "usecases" / "arc-agi-3" / "python")
            if _adapter.exists() and str(_adapter) not in _sys.path:
                _sys.path.insert(0, str(_adapter))
            import solutions_kb
            import solution_replay
        except Exception as e:
            print(f"[replay] unavailable ({e}); cannot fast-forward")
            return False

        gid = self.world.game_id

        def _recall(level: int):
            return solutions_kb.recall_solution(gid, level, only_canonical=True)

        res = solution_replay.replay_to_level(
            game=self.game, recall=_recall, target_level=int(target_level),
        )
        if not res.reached:
            return False
        # Adopt the fast-forwarded state so live play resumes here.
        self._last_frame_path = res.final_frame
        self._level_start_frame_path = res.final_frame
        self.world.turn = int(res.final_turn or self.game.current_turn or 1)
        if res.final_score is not None:
            self.world.score = int(res.final_score)
        # run_level_start_analysis(crystallize=False) will bump the ordinal by one
        # when it records the working-level entry, so set it ONE BELOW the target.
        self._level_ordinal = int(target_level) - 1
        # Record each replayed level's score-advance as a real boundary in the
        # delta log, so the trace attributes the replayed turns to lc 0..N-1 and
        # the live turns to lc N (otherwise every turn renders as the working
        # level and the replayed-prefix frames masquerade as it).  These are at
        # turns BEFORE the working level's first live action, so the per-level cx
        # (which scans from _level_start_turn) never sees them.
        try:
            from world_knowledge import DeltaRecord  # noqa: E402
            for lr in res.levels:
                et = getattr(lr, "end_turn", None)
                if not lr.advanced or et is None:
                    continue
                self.world.deltas_observed.append(DeltaRecord(
                    from_turn=int(et) - 1, to_turn=int(et),
                    action="REPLAY", agent_moved=False,
                    inferred_action="REPLAY",
                    summary=(f"fast-forward: replayed the recorded lc={lr.level} "
                             f"solution ({lr.n_acts_played} acts) -> score "
                             f"{lr.final_score}"),
                    score_increased=True))
        except Exception as e:
            print(f"[replay] boundary-delta record skipped ({e})")
        # The replay did NOT record into world.actions_taken, so the next
        # solve's win-path = the working level's live acts only.
        self._level_start_action_idx = len(self.world.actions_taken)
        return True

    def run(self, max_turns: int = 40, start_level: int = 0) -> list[TurnReport]:
        # STRICT-MODE ROBUSTNESS: each turn is wrapped so a single failing turn
        # (a malformed-perception shape that trips ingest_perception, an
        # unsupported action raising ValueError in the adapter, a stray
        # AttributeError, etc.) is LOGGED and SKIPPED rather than crashing the
        # whole game.  An autonomous run over hundreds of games must survive one
        # bad turn.  A run of consecutive failures (the world genuinely wedged)
        # still stops cleanly after a bound, so we never spin forever.
        import traceback as _tb
        # SELF-REAP: on exit, kill any worker subprocesses this driver spawned, so a trial
        # leaves nothing behind (a hung driver is killed by the launcher via proc_cleanup.kill_tree).
        try:
            import atexit as _atexit
            import proc_cleanup as _pc
            _atexit.register(_pc.reap_self_children)
        except Exception:
            pass
        _MAX_CONSEC_ERRORS = 5
        # FAST-FORWARD: if asked to start on a later level, replay the recorded
        # solutions for the earlier levels to reach it (resume-after-death /
        # budget-free seek).  On failure, fall back to full play from level 0.
        fast_forwarded = False
        if start_level and start_level > 0:
            fast_forwarded = self.replay_prefix(start_level)
            if fast_forwarded:
                print(f"[driver] fast-forwarded to level {start_level} via "
                      f"recorded solutions; live play begins on the working "
                      f"level (no VLM spent on the solved prefix).")
            else:
                print(f"[driver] could not fast-forward to level {start_level} "
                      f"(missing/stale recorded solution); playing from level 0.")
        try:
            try:
                if fast_forwarded:
                    # The env already sits on the working level (replay_prefix).
                    # Record it as a proper LEVEL START (correct turn + its own
                    # level_start/ dir with the working-level frame), reusing the
                    # normal level-start path but skipping crystallisation (no
                    # in-session level just finished).
                    self.run_level_start_analysis(
                        self._last_frame_path, crystallize=False)
                else:
                    self.run_turn_one()
            except Exception as e:
                print(f"[driver] WARNING: turn-one failed ({e}); continuing "
                      f"(later turns re-perceive each frame).", flush=True)
                _tb.print_exc()
            consec_errors = 0
            for _ in range(max_turns - 1):
                if self.world.win_state != "playing":
                    print(f"[driver] win_state={self.world.win_state}, stop")
                    break
                try:
                    report = self.run_one_step()
                    consec_errors = 0
                except Exception as e:
                    consec_errors += 1
                    print(f"[driver] WARNING: turn failed ({e}); skipping it "
                          f"(consecutive failures={consec_errors}/"
                          f"{_MAX_CONSEC_ERRORS}).", flush=True)
                    _tb.print_exc()
                    if consec_errors >= _MAX_CONSEC_ERRORS:
                        print("[driver] too many consecutive turn failures; "
                              "stopping this game (no crash).", flush=True)
                        break
                    continue
                if report is None:
                    break
        finally:
            # Persist this run's learnings to the cross-game priors
            # store EVEN if the run errored out or stopped early --
            # any promoted hypotheses are still worth saving.
            self.finalize()
        return self._reports

    # --- helpers ---

    def _infer_goal_cells_from_world(self) -> list[list[int]]:
        """Pick out goal candidates from the world model.  Heuristic
        (game-agnostic): any entity currently with role in
        {collectable, trigger_target, goal} and a known cell is a
        goal candidate."""
        GOAL_ROLES = {"collectable", "trigger_target", "goal", "target"}
        out: list[list[int]] = []
        for r in self.world.entities.values():
            if r.current_role in GOAL_ROLES and r.current_cell is not None:
                out.append(list(r.current_cell))
        return out

    def _direction_unit(self, action: str) -> Optional[tuple[int, int]]:
        return {
            "UP":    (-1, 0),
            "DOWN":  (1, 0),
            "LEFT":  (0, -1),
            "RIGHT": (0, 1),
        }.get(action)

    def _infer_blocked_cells_from_deltas(self) -> list[list[int]]:
        """Walk the delta history.  For every delta where
        agent_moved=False and the action was a cardinal move, the
        cell in that direction from the agent's then-position is
        blocked.  This is empirical evidence the actor can use to
        avoid retrying the same blocked move."""
        step = (self.world.grid_inference.cell_ticks
                if self.world.grid_inference
                and self.world.grid_inference.cell_ticks
                else 1)
        # We need the agent's position at the moment OF EACH ACTION,
        # not after.  Walk actions+deltas together; for each action,
        # the agent's cell_history entry whose turn == action.turn
        # tells us where the agent was before stepping.
        agent_rec = self.world._find_agent()
        if agent_rec is None:
            return []
        cell_by_turn = {turn: cell
                         for (turn, cell) in agent_rec.cell_history}
        blocked: set[tuple[int, int]] = set()
        for action_rec in self.world.actions_taken:
            unit = self._direction_unit(action_rec.action)
            if unit is None:
                continue
            # find the delta that happened AFTER this action
            matching = [d for d in self.world.deltas_observed
                        if d.from_turn == action_rec.turn]
            if not matching:
                continue
            delta = matching[0]
            if delta.agent_moved:
                continue
            # blocked: pre-action cell + unit*step is impassable
            pre_cell = cell_by_turn.get(action_rec.turn)
            if pre_cell is None:
                continue
            blocked_cell = (pre_cell[0] + unit[0] * step,
                            pre_cell[1] + unit[1] * step)
            blocked.add(blocked_cell)
        return [list(c) for c in blocked]

    def _infer_traversable_cells_from_deltas(self) -> list[list[int]]:
        """Cells the agent has actually entered are confirmed
        traversable."""
        agent_rec = self.world._find_agent()
        if agent_rec is None:
            return []
        seen = {tuple(cell) for (_, cell) in agent_rec.cell_history}
        return [list(c) for c in seen]

    def _cx_ofat_probe(self, entities_for_actor):
        """ACTIVE multi-trial experimentation for the NO-INITIAL-MOVER case
        (game-agnostic).  The single-trial cx tier needs the initial trigger to
        already move the mover; when it does NOT (e.g. tn36 lc=1's right board
        starts with every switch in the inert state, so the GO sweeps but the
        marker never descends), there is no demonstrated law to attribute.  This
        tier BOOTSTRAPS one: it sets all toggle CONTROLS to their active state
        (the cheapest informative experiment -- the uniform-active config that
        an "every control contributes one step toward the goal" mechanic needs),
        then fires the TRIGGER and gates on the win subgoal.  Apply-first; on no
        score it hands back (-> explore more / deeper experimentation).

        Detects from what COS already perceives: a goal_marker SOURCE + its
        goal_target/twin TARGET + >=2 toggle controls + a repeatable trigger (the
        most recent CLICK that ANIMATED something).  FULLY GUARDED -- returns None
        on any miss so coverage/claim-probing continues.

        COS_DISABLE_CX_OFAT (env) bypasses this tier so a more grounded program
        (decoded from the reference, not blind uniform-active) can drive."""
        import os as _os
        if _os.environ.get("COS_DISABLE_CX_OFAT"):
            return None
        try:
            from types import SimpleNamespace
            cx = self.world.probe_state.get("cx_ofat")
            if not cx or cx.get("phase") == "unavailable":
                cx = self._cx_detect_ofat(entities_for_actor) or {"phase": "unavailable"}
                self.world.probe_state["cx_ofat"] = cx
            ph = cx.get("phase")
            if ph in ("unavailable", "done"):
                return None
            if ph == "set_all":
                # click each control POINT to set it active, one per turn, so the
                # next fire tests the uniform-active config.
                pts = cx.get("points") or []
                while cx["idx"] < len(pts):
                    col, row = pts[cx["idx"]]
                    cx["idx"] += 1
                    act = self._resolve_action(f"CLICK:{int(col)},{int(row)}")
                    if act is None:
                        continue
                    return SimpleNamespace(
                        action=act, plan_kind="controlled_experiment",
                        rationale=f"OFAT: set unit active ({cx['idx']}/{len(pts)}) "
                                  f"for the uniform-active trial",
                        goal_id="controlled_experiment", target_cell=None,
                        full_plan_actions=[act], is_probe=True)
                cx["phase"] = "fire"
            if cx["phase"] == "fire":
                cx["phase"] = "fired"
                act = self._resolve_action(cx["trigger"])
                print(f"[cx-ofat] all {len(cx.get('points') or [])} units set active; "
                      f"firing trigger {cx['trigger']} (uniform-active win attempt).")
                return SimpleNamespace(
                    action=act, plan_kind="controlled_experiment",
                    rationale="OFAT: fire trigger on the uniform-active config",
                    goal_id="controlled_experiment", target_cell=None,
                    full_plan_actions=[act], is_probe=True)
            if cx["phase"] == "fired":
                won = getattr(self.world, "win_state", "playing") != "playing"
                last = list(getattr(self.world, "deltas_observed", []) or [])
                scored = won or (bool(last) and getattr(last[-1], "score_increased", False))
                print("[cx-ofat] uniform-active " + ("SCORED -- subgoal achieved."
                      if scored else "did not score; the law is not uniform-active "
                      "-> hand back for deeper experimentation / explore more."))
                cx["phase"] = "done"
            return None
        except Exception as e:
            print(f"[cx-ofat] probe error ({e}); falling through")
            return None

    # Roles that mark an entity as a candidate program TRIGGER / fire button.
    # Closed, game-agnostic vocabulary (a button you CLICK to run the program),
    # distinct from the SETTABLE controls (the program's cells) and the scene
    # mover/goal.  Used only as ONE signal; the picker also requires the entity
    # not be a control, not be a decomposable settable panel, and not be a
    # demonstration source.
    _TRIGGER_ROLES = {"trigger", "trigger_target", "button", "go", "fire",
                      "fire_button", "activator", "launcher"}

    def _cx_fire_button_names(self, controls, demo_srcs):
        """Perceived FIRE-BUTTON entities -- a distinct button that RUNS the
        program when clicked (tn36's blue disc) -- ranked best-first.  Game-
        agnostic: a fire button (a) has a trigger-ish role OR the substrate
        measured its own click as a trigger (animated), (b) is NOT a settable
        control (the detected program cells), and (c) is NOT a demonstration/
        preview source.

        Ranking (strongest evidence first) -- 'prefer a LARGE DISTINCT button':
          1. a MEASURED trigger (its own click animated the program) outranks a
             role-only guess;
          2. a STANDALONE button outranks a member of a multi-entity similarity
             group -- a scene GLYPH cluster (tn36's yellow_glyphs = arch+cup) is
             the mover/goal, and a switch-PANEL group is the settable board, NOT
             the fire button, even when a member carries a 'trigger_target' role;
             the button stands alone;
          3. larger area (a distinct button is big; a glyph is small).
        A settable PANEL is excluded by (b) -- once OFAT has its >=2 controls the
        program board IS in `controls` -- and de-ranked by (2) when it is a group;
        deliberately NOT by a structural-lattice test, which false-positives on a
        solid disc (the disc's circle reads as a pseudo-grid) and would drop the
        very button we want.  Guarded -> []."""
        try:
            demo_ents = set()
            for d in (getattr(self.world, "deltas_observed", []) or []):
                if (getattr(d, "action", "") or "") in (demo_srcs or set()):
                    for nm in (getattr(d, "entities_changed", []) or []):
                        demo_ents.add(nm)
            # entities that belong to a multi-member similarity/cluster group
            # (scene-glyph clusters, repeated-mark panels) -- NOT a lone button.
            grouped = set()
            _g = getattr(self.world, "groups", {}) or {}
            for gv in (_g.values() if isinstance(_g, dict) else (_g or [])):
                mem = list(getattr(gv, "members", None)
                           or (gv.get("members") if isinstance(gv, dict) else []) or [])
                if len(mem) >= 2:
                    grouped.update(mem)
            rfacts = getattr(self.world, "response_facts", {}) or {}
            out = []
            for nm, rec in (getattr(self.world, "entities", {}) or {}).items():
                bb = getattr(rec, "current_bbox", None)
                if bb is None or nm in controls or nm in demo_ents:
                    continue
                role = (getattr(rec, "current_role", "") or "").lower()
                measured = bool((rfacts.get(nm) or {}).get("is_trigger"))
                if not (measured or role in self._TRIGGER_ROLES):
                    continue
                r0, c0, r1, c1 = bb
                out.append((nm, (1 if measured else 0,
                                 0 if nm in grouped else 1,
                                 abs((r1 - r0) * (c1 - c0)))))
            out.sort(key=lambda t: t[1], reverse=True)
            return [nm for nm, _ in out]
        except Exception:
            return []

    def _cx_click_hits(self, action, names):
        """If a ``CLICK:col,row`` action lands inside one of ``names``' bboxes,
        return that name (else None).  Lets trigger selection accept an OBSERVED
        click that landed ON the fire button -- even one that produced no
        animation (firing an unset program) -- while rejecting a stray click on
        an inert reference panel.  Guarded."""
        try:
            payload = (action or "")[len("CLICK:"):]
            parts = payload.split(",")
            col, row = int(parts[0]), int(parts[1])
        except Exception:
            return None
        for nm in (names or []):
            rec = self.world.entities.get(nm)
            bb = getattr(rec, "current_bbox", None)
            if bb is None:
                continue
            r0, c0, r1, c1 = bb
            if r0 <= row <= r1 and c0 <= col <= c1:
                return nm
        return None

    def _cx_detect_ofat(self, entities_for_actor):
        """Detect the NO-MOVER OFAT preconditions: a repeatable trigger (an
        animated CLICK, an observed click on the fire button, or -- if unfired --
        the fire button itself), >=2 toggle controls (clicked -> changed in
        place; expands to the full similarity group; a panel-scale control is
        DECOMPOSED into its cell units and the inert reference dropped), and a
        goal_marker SOURCE whose firing did NOT move it.  Returns a cx_ofat state
        dict or None.  Only fires AFTER the single-trial tier found no toward-goal
        mover."""
        try:
            # only THIS level's deltas: _level_start_turn is set one PAST the
            # transition turn, so the transition delta (the prior level's winning
            # action) is excluded but every in-level delta is kept.
            _lst = int(getattr(self, "_level_start_turn", 0) or 0)
            deltas = [d for d in (getattr(self.world, "deltas_observed", []) or [])
                      if int(getattr(d, "to_turn", 0) or 0) >= _lst]
            # toggle controls (clicked -> changed in place, not on a non-click turn).
            # Built FIRST (before trigger selection) so the fire-button picker can
            # exclude them: a settable control is never the program's fire button.
            non_click_changed = set()
            for d in deltas:
                if not (getattr(d, "action", "") or "").startswith("CLICK"):
                    for nm in (getattr(d, "entities_changed", []) or []):
                        non_click_changed.add(nm)
            controls = {}
            for d in deltas:
                if not (getattr(d, "action", "") or "").startswith("CLICK") \
                        or getattr(d, "animation_events", None):
                    continue
                for nm in (getattr(d, "entities_changed", []) or []):
                    if nm in non_click_changed:
                        continue
                    rec = self.world.entities.get(nm)
                    if rec is None or rec.current_bbox is None \
                            or (getattr(rec, "current_role", "") or "") == "hud":
                        continue
                    controls[nm] = True
            # DIRECTLY-OBSERVED controls = the ones a click changed in place on THIS
            # level (the truly settable units).  Snapshot them BEFORE the similarity-
            # group expansion below, because that expansion ASSUMES every member of a
            # group containing a control is itself a control -- which over-reaches for
            # a two-panel similarity group where one panel is the settable ACTIVE and
            # the other is an inert REFERENCE (clicking the reference does nothing).
            # Keeping the direct set lets the panel path pick the settable panel and
            # drop the inert reference (the tn36 lc2 wasted-click).
            direct_controls = set(controls)
            for g in (getattr(self.world, "groups", {}) or {}).values():
                mem = list(getattr(g, "members", []) or [])
                if any(m in controls for m in mem):
                    for m in mem:
                        rec = self.world.entities.get(m)
                        if rec is not None and rec.current_bbox is not None \
                                and (getattr(rec, "current_role", "") or "") != "hud":
                            controls[m] = True

            # TRIGGER selection -- the action that RUNS the program.  Three passes,
            # most-direct evidence first:
            #   1. the most recent CLICK that ANIMATED (the program actually ran),
            #      excluding demonstration/preview sources (a legend/preview click
            #      also animates -- it previews a mover -- and would otherwise be
            #      mis-picked as the trigger; the tn36 lc2 legend mis-fire).
            #   2a. an OBSERVED click that LANDED ON the fire button -- a distinct
            #       program button (tn36's blue disc) -- even if it did NOT animate
            #       (firing an UNSET program is net-0, no mover); this is preferred
            #       over a stray non-control click, which would grab a click on an
            #       inert reference (the live tn36 lc2 wasted CLICK on the left
            #       panel that only ticked the HUD).
            #   2b. if the button has not been fired yet, SYNTHESISE a click on it
            #       (prefer a large distinct button wired to the program), so OFAT
            #       can run the experiment without waiting for blind exploration to
            #       happen to click it.
            # No catch-all "any non-control click" pass: with no animated trigger
            # AND no identifiable fire button, there is nothing sound to fire, so
            # detection declines (None) and exploration continues -- better than
            # firing a spurious trigger and burning the one OFAT engagement.
            demo_srcs = getattr(self, "_demo_source_actions", set()) or set()
            fire_buttons = self._cx_fire_button_names(controls, demo_srcs)

            trigger = None
            for d in reversed(deltas):                       # pass 1: animated, non-demo
                act = getattr(d, "action", "") or ""
                if (act.startswith("CLICK") and act not in demo_srcs
                        and (getattr(d, "animation_events", None) or [])):
                    trigger = act
                    break
            if not trigger and fire_buttons:                 # pass 2a: observed click ON the button
                for d in reversed(deltas):
                    act = getattr(d, "action", "") or ""
                    if (act.startswith("CLICK:") and act not in demo_srcs
                            and self._cx_click_hits(act, fire_buttons)):
                        trigger = act
                        break
            if not trigger and fire_buttons:                 # pass 2b: synthesise a click on the button
                trigger = f"CLICK:{fire_buttons[0]}"
            if not trigger:
                return None
            # NOTE: there is NO transient-mover deferral here.  Single-trial cx
            # (_controlled_experiment_probe) runs FIRST in _select_probe every
            # turn; a transient mover it can use (a toward-goal preview it can
            # imitate) makes it return a probe, so OFAT is never reached.  Reaching
            # OFAT therefore means single-trial cx already DECLINED -- e.g. the
            # only transient mover moved LATERALLY (tn36: the arch nudges sideways
            # while its cup sits straight below), which single-trial cx does not
            # treat as toward-goal.  So OFAT must OWN that case and run the
            # controlled experiment; deferring again would deadlock (neither tier
            # acts), the exact wedge observed on tn36 lc=0.

            def _center(bb):
                r0, c0, r1, c1 = bb
                return [int((c0 + c1) // 2), int((r0 + r1) // 2)]   # (col, row)

            # PANEL-SCALE case: one or more "controls" are actually multi-cell
            # STRUCTURES (a switch panel/board), not atomic cell-marks.  The program
            # is set per CELL, so clicking a whole panel ONCE (its centre) is the
            # wrong granularity; and when both a settable panel and an inert reference
            # panel were caught (the reference rode in on the group expansion), the
            # reference click is wasted.  So pick the single ACTIVE (settable) panel,
            # DECOMPOSE it into its cell units, and run per-cell uniform-active; the
            # inert reference panel(s) drop out by construction.  (tn36 lc2: decompose
            # right_switch_panel into cells, exclude the inert left_switch_panel.)
            # Fully guarded -> on any miss the atomic path below still runs, so the
            # lc-0 individual-switch case and the fake-self unit tests are unaffected.
            try:
                panel_controls = [nm for nm in controls
                                  if self._is_structure_member(nm)]
            except Exception:
                panel_controls = []
            if panel_controls:
                try:
                    # most-active panel: strongest control-score, tie-broken toward
                    # the one a click was DIRECTLY observed to change in place.
                    active = max(panel_controls, key=lambda nm: (
                        self._entity_control_score(nm), nm in direct_controls))
                    cells = self._resolve_structure_units(active) or []
                    pts = [_center(b) for b in cells]
                    if len(pts) < 2:           # fall back to the bbox-based splitter
                        pts = self._cx_panel_components(
                            self.world.entities[active].current_bbox) or []
                    excluded = [nm for nm in panel_controls if nm != active]
                    _exc = f"; excluded inert reference {excluded}" if excluded else ""
                    if len(pts) >= 2:
                        print(f"[cx-ofat] DETECTED no-mover case: trigger {trigger}; "
                              f"ACTIVE panel {active!r} decomposed into {len(pts)} "
                              f"cell units -> per-cell uniform-active{_exc}.")
                        return {"phase": "set_all", "trigger": trigger,
                                "points": pts, "idx": 0, "active_panel": active}
                    # decomposition unavailable -> still act on the ACTIVE panel only
                    # (never the inert reference): one coarse click + fire.
                    print(f"[cx-ofat] DETECTED no-mover case: trigger {trigger}; "
                          f"ACTIVE panel {active!r} could not be decomposed -> single "
                          f"coarse click on it{_exc}.")
                    return {"phase": "set_all", "trigger": trigger,
                            "points": [_center(self.world.entities[active].current_bbox)],
                            "idx": 0, "active_panel": active}
                except Exception as e:
                    print(f"[cx-ofat] panel-decompose skipped ({e}); "
                          f"falling back to individual controls")

            if len(controls) >= 2:
                names = sorted(controls, key=lambda nm: (
                    self.world.entities[nm].current_bbox[1],
                    self.world.entities[nm].current_bbox[0]))
                points = [_center(self.world.entities[nm].current_bbox) for nm in names]
                print(f"[cx-ofat] DETECTED no-mover case: trigger {trigger}, "
                      f"{len(points)} individual controls {names} -> uniform-active.")
                return {"phase": "set_all", "trigger": trigger, "points": points, "idx": 0}

            # SELF-SUFFICIENT FALLBACK: no individual controls were clicked, but
            # the VLM perceived a control_group PANEL as one region -> split it
            # into its switch UNITS with the `components` op and use those as the
            # control points (so OFAT bootstraps the controls itself).
            region = None
            for nm, rec in (self.world.entities or {}).items():
                if rec.current_bbox is None:
                    continue
                if (getattr(rec, "current_role", "") or "") == "control_group":
                    region = rec.current_bbox
                    break
            pts = self._cx_panel_components(region) if region else []
            if len(pts) >= 2:
                print(f"[cx-ofat] DETECTED no-mover case: trigger {trigger}; "
                      f"auto-enumerated {len(pts)} units from a control_group panel "
                      f"-> uniform-active experiment.")
                return {"phase": "set_all", "trigger": trigger, "points": pts, "idx": 0}
            return None
        except Exception as e:
            print(f"[cx-ofat] detect failed ({e})")
            return None

    def _cx_region_bg(self, bboxes):
        """A panel's BACKGROUND colour = the colour that SURROUNDS its units --
        the dominant colour on the unit sprites' BORDER rings.  This is a
        structural figure-ground (the field surrounds the marks), PALETTE-
        INVARIANT (Adversarial Test): it does NOT assume 'the most common colour
        overall is background', which breaks on a re-skin where a mark colour
        dominates a sparse panel.  Returns an [r,g,b] list or None.  Guarded."""
        try:
            import numpy as np
            sprs = [self._cx_read_sprite(getattr(self, "_last_frame_path", None), bb)
                    for bb in (bboxes or [])]
            sprs = [s for s in sprs if s is not None and s.size]
            if not sprs:
                return None
            rings = []
            for s in sprs:
                rings.append(s[0, :].reshape(-1, s.shape[-1]))
                rings.append(s[-1, :].reshape(-1, s.shape[-1]))
                rings.append(s[:, 0].reshape(-1, s.shape[-1]))
                rings.append(s[:, -1].reshape(-1, s.shape[-1]))
            flat = np.concatenate(rings, axis=0)
            vals, counts = np.unique(flat, axis=0, return_counts=True)
            return vals[counts.argmax()].tolist()
        except Exception:
            return None

    def _group_control_fraction(self, gname) -> float:
        """Fraction of a group's members with a control-like role -- how
        'changeable' the panel is (the panel you can act on vs a static
        reference).  Mirrors structural_claims._group_role_mix over the live
        world.  Guarded."""
        control = {"control_group", "trigger_target", "control", "switch"}
        try:
            grp = (getattr(self.world, "groups", {}) or {}).get(gname)
            members = list(getattr(grp, "members", []) or [])
            ents = self.world.entities
            roles = [(getattr(ents.get(m), "current_role", "") or "").lower()
                     for m in members]
            roles = [r for r in roles if r]
            if not roles:
                return 0.0
            return sum(1 for r in roles if r in control) / len(roles)
        except Exception:
            return 0.0

    def _struct_claim_panels(self, c):
        """Extract the (ACTIVE, REFERENCE) group pair a structural / similarity /
        structure-mapped claim implies, or None.  This is what lets the SAME
        match-execute-verify machinery run from a SIMILARITY claim (the user's
        principle: two similar structures alone justify trying to make one match
        the other), not only an explicit MATCH:A~B+TRIGGER template claim.

          - MATCH:A~B           -> (A, B): the template generator already names
                                   active~reference, so keep the order.
          - COMPARE:A~B         -> the two similar groups (similar_structures);
                                   ACTIVE = the one with the higher control-role
                                   fraction (the panel you can change), REFERENCE
                                   = the other.  Skipped if neither is clearly more
                                   interactive (don't guess which to change).
          - else (structural /  -> two GROUP names among the claim's targets,
            structure_mapped)      ordered the same way by interactivity.

        Only returns groups that exist with members.  Guarded."""
        try:
            groups = getattr(self.world, "groups", {}) or {}

            def present(g):
                return bool(g in groups and (getattr(groups[g], "members", None) or []))

            plan = getattr(c, "plan", "") or ""
            kind = getattr(c, "kind", "") or ""
            if plan.startswith("MATCH:"):
                a, _, b = plan[len("MATCH:"):].split("+", 1)[0].partition("~")
                a, b = a.strip(), b.strip()
                return (a, b) if present(a) and present(b) else None
            pair = None
            if plan.startswith("COMPARE:"):
                a, _, b = plan[len("COMPARE:"):].split(" ", 1)[0].partition("~")
                a, b = a.strip(), b.strip()
                if present(a) and present(b):
                    pair = (a, b)
            if pair is None and kind in ("structural", "structure_mapped"):
                gs = [t for t in (getattr(c, "target", []) or []) if present(t)]
                if len(gs) >= 2:
                    pair = (gs[0], gs[1])
            if pair is None:
                return None
            A, B = pair
            fa, fb = self._group_control_fraction(A), self._group_control_fraction(B)
            if fa == fb:
                return None                     # ambiguous which panel is the active one
            return (A, B) if fa > fb else (B, A)
        except Exception:
            return None

    _CONTROL_ROLES = {"control_group", "trigger_target", "control", "switch"}

    def _entity_control_score(self, name):
        """Ordered evidence that this entity is the ACTIVE (settable) structure to
        CHANGE, vs an inert REFERENCE to match.  Returns a comparable tuple so the
        strongest, most DIRECT measurement dominates -- measurement beats a carried
        label.  Priority:

          1. DIRECT disambiguation diff (_panel_settable, set by
             _resolve_prev_disambiguation): a clean pre/post pixel diff of the
             panel's OWN region across the disambiguation click -- changed=settable
             (+1), unchanged=inert (-1), not-yet-measured (0).  This is the most
             reliable signal (independent of the visual_events watch/threshold/
             timing); one measurement also breaks the tie by ELIMINATION (the
             non-inert panel is the active one).
          2. response_facts (substrate response-asymmetry): settable / is_trigger.
          3. the role prior (tiebreak only)."""
        direct = (getattr(self, "_panel_settable", None) or {}).get(name)
        rf = (getattr(self.world, "response_facts", {}) or {}).get(name) or {}
        rec = self.world.entities.get(name)
        role = (getattr(rec, "current_role", "") or "").lower()
        return (
            (1 if direct is True else (-1 if direct is False else 0)),  # DIRECT diff
            1 if rf.get("settable") else 0,          # substrate: changed in place
            -1 if rf.get("is_trigger") else 0,       # substrate: it's a trigger
            1 if role in self._CONTROL_ROLES else 0,  # weak role prior (tiebreak)
        )

    def _structure_bbox(self, name):
        """Bounding box of a structure named by an ENTITY or a GROUP (the union
        of its members' boxes).  None if not locatable.  Guarded."""
        ents = self.world.entities
        groups = getattr(self.world, "groups", {}) or {}
        rec = ents.get(name)
        if rec is not None and rec.current_bbox is not None:
            return [int(v) for v in rec.current_bbox]
        g = groups.get(name) if isinstance(groups, dict) else None
        if g is not None:
            bbs = [ents[m].current_bbox for m in (getattr(g, "members", []) or [])
                   if m in ents and ents[m].current_bbox is not None]
            if bbs:
                return [min(b[0] for b in bbs), min(b[1] for b in bbs),
                        max(b[2] for b in bbs), max(b[3] for b in bbs)]
        return None

    # A region is "panel-scale" (a multi-cell structure, not one cell-mark) when
    # it spans more than 1/8 of the 64-tick frame in BOTH dimensions.  A single
    # repeated cell-mark is a few ticks; a panel/board/legend is many.  Frame-
    # relative (not a tuned pixel constant) and FRAME-INDEPENDENT, so it is the
    # robust signal for "group of cells vs group of panels" even when the
    # decomposition read is unreliable (e.g. a transient / mis-grounded frame).
    _PANEL_SCALE_MIN_TICKS = 8

    def _is_panel_scale(self, bbox) -> bool:
        try:
            r0, c0, r1, c1 = [int(v) for v in bbox]
            return (min(r1 - r0, c1 - c0) >= self._PANEL_SCALE_MIN_TICKS)
        except Exception:
            return False

    def _is_structure_member(self, name) -> bool:
        """Cached verdict: is this entity a multi-cell STRUCTURE (panel / board)
        rather than an atomic cell-mark?  Computed ONCE on the entity's first-seen
        (clean, level-start) bbox and cached by NAME, so frame_bbox_refresh's
        per-turn bbox jitter cannot later flip a panel into being treated as a
        cell (which produced the bogus 2-unit panel<->legend match).  A panel
        never becomes a cell within a level."""
        cache = getattr(self, "_panel_scale_cache", None)
        if cache is None:
            cache = self._panel_scale_cache = {}
        if cache.get(name):
            return True                        # MONOTONIC: once a panel, always a panel
        rec = self.world.entities.get(name)
        bb = getattr(rec, "current_bbox", None)
        if not bb:
            return False
        v = bool(self._is_panel_scale(bb) or self._is_decomposable(bb))
        if v:
            cache[name] = True                 # cache only the TRUE verdict, so a
        return v                               # transient small/bad bbox cannot poison it

    def _is_decomposable(self, bbox) -> bool:
        """True if a region itself contains a repeating lattice of >= 2 cells
        (a PANEL / board), vs an atomic cell-mark.  Distinguishes a group of
        cells from a group of panels.  Guarded -> False."""
        try:
            import structural_grid
            d = structural_grid.read_path(getattr(self, "_last_frame_path", None),
                                          list(bbox))
            return bool(d and len(d.get("cells") or []) >= 2)
        except Exception:
            return False

    def _resolve_structure_units(self, name):
        """Resolve a SINGLE structure (GROUP or ENTITY name) to its list of
        ATOMIC CELL-UNIT bounding boxes, sorted by (col, row).  Game/domain-
        agnostic:

          - a GROUP whose members are ATOMIC cells (each does not itself
            decompose into a sub-lattice) -> the member boxes (the lc-0
            behaviour where cells were perceived separately);
          - a single ENTITY perceived as one monolithic region (a panel /
            board / keypad) -> DECOMPOSE it into its lattice of cells
            (structural_grid), so the executor has per-cell units to diff.

        Returns None when ``name`` is NOT a single cell-structure -- in
        particular a group whose members are THEMSELVES decomposable panels
        (e.g. a 'switch_panels' group of two whole panels, or a 'legend_boxes'
        group of two whole legends).  Such a group is a similarity SET, not one
        structure; matching its members pairwise would compare whole panels to
        whole legends at incompatible granularity.  (The panel<->panel match
        runs via the 2-member similarity-group candidate, which resolves each
        MEMBER entity to cells.)"""
        ents = self.world.entities
        groups = getattr(self.world, "groups", {}) or {}
        g = groups.get(name) if isinstance(groups, dict) else None
        if g is not None:
            members = [m for m in (getattr(g, "members", []) or [])
                       if m in ents and ents[m].current_bbox is not None]
            if len(members) >= 2:
                if self._is_structure_member(members[0]):
                    return None      # members are panels, not cells -> not one structure
                members.sort(key=lambda m: (ents[m].current_bbox[1],
                                            ents[m].current_bbox[0]))
                return [list(ents[m].current_bbox) for m in members]
            if len(members) == 1:
                name = members[0]
        rec = ents.get(name)
        if rec is not None and rec.current_bbox is not None:
            # STABILITY: decompose ONCE, on the clean LEVEL-START raw frame, and
            # CACHE the cell lattice per entity.  A static panel's lattice does
            # not change between turns (only the cell STATES toggle, read fresh by
            # the diff); re-decomposing a bbox that frame_bbox_refresh re-grounds
            # slightly every turn was giving an unstable cell count (24 one turn,
            # 2 the next).  Cache is cleared on level reset.
            cache = getattr(self, "_struct_cell_cache", None)
            if cache is None:
                cache = self._struct_cell_cache = {}
            if name in cache:
                return [list(b) for b in cache[name]]
            frame = (getattr(self, "_level_start_frame_path", None)
                     or getattr(self, "_last_frame_path", None))
            try:
                import structural_grid
                d = structural_grid.read_path(frame, rec.current_bbox)
            except Exception:
                d = None
            cells = (d or {}).get("cells") or []
            if len(cells) >= 2:
                cells = sorted(cells, key=lambda b: (b[1], b[0]))
                cache[name] = [list(b) for b in cells]
                return [list(b) for b in cells]
        return None

    def _structural_candidates(self):
        """Yield (claim_id, active_name, reference_name) match candidates from
        BOTH the claim frontier (explicit MATCH / structural / structure-mapped
        claims, via _struct_claim_panels) AND any 2-member SIMILARITY group --
        because two similar structures alone justify trying to make one match
        the other (the active = the more settable member; skip if it is a tie,
        i.e. don't guess which to change)."""
        out = []
        cs = getattr(self, "_claim_store", None)
        if cs is not None:
            try:
                for c in cs.rank(self._level_signature()):
                    p = self._struct_claim_panels(c)
                    if p:
                        out.append((getattr(c, "claim_id", None)
                                    or getattr(c, "id", None) or "struct", p[0], p[1]))
            except Exception:
                pass
        ents = self.world.entities
        groups = getattr(self.world, "groups", {}) or {}
        for gname, g in (groups.items() if isinstance(groups, dict) else []):
            crit = (getattr(g, "criterion", "") or "").lower()
            members = [m for m in (getattr(g, "members", []) or [])
                       if m in ents and ents[m].current_bbox is not None]
            if "similar" in crit and len(members) == 2:
                fa = self._entity_control_score(members[0])
                fb = self._entity_control_score(members[1])
                if fa == fb:
                    continue                       # ambiguous which to change
                a, b = (members[0], members[1]) if fa > fb else (members[1], members[0])
                out.append((f"match::{gname}", a, b))
        return out

    def _structural_disambiguation_probe(self, entities_for_actor):
        """A directed scout that ENABLES a structural match when which of two
        similar structures is ACTIVE (settable) vs REFERENCE is undetermined.

        The match needs the active/reference order, which comes from the MEASURED
        response-asymmetry -- but that requires having clicked one of them.  When
        the win is understood and a 2-member similarity group is still a role TIE
        with no measured response, this clicks ONE member (a real cell of it, via
        decomposition, so the click lands on a switch) to measure whether it is
        settable.  Next turn response_facts resolves the order and the match
        fires.  Probes each member at most once (then gives up if still tied), so
        it cannot thrash.  Returns a probe action or None.  Game-agnostic."""
        try:
            from types import SimpleNamespace
            ents = self.world.entities
            groups = getattr(self.world, "groups", {}) or {}

            def _clicked_already(m) -> bool:
                # measured == a CLICK has actually landed inside this entity (not
                # merely present in response_facts, which initialises EVERY entity)
                bb = ents[m].current_bbox
                for d in (getattr(self.world, "deltas_observed", []) or []):
                    act = (getattr(d, "action", "") or
                           getattr(d, "inferred_action", "") or "")
                    if not act.startswith("CLICK:"):
                        continue
                    try:
                        col, row = (int(x) for x in act[len("CLICK:"):].split(",")[:2])
                    except Exception:
                        continue
                    if bb[0] <= row < bb[2] and bb[1] <= col < bb[3]:
                        return True
                return False

            for gname, g in (groups.items() if isinstance(groups, dict) else []):
                crit = (getattr(g, "criterion", "") or "").lower()
                members = [m for m in (getattr(g, "members", []) or [])
                           if m in ents and ents[m].current_bbox is not None]
                if "similar" not in crit or len(members) != 2:
                    continue
                if self._entity_control_score(members[0]) != self._entity_control_score(members[1]):
                    continue                      # order already determinable
                unmeasured = [m for m in members if not _clicked_already(m)]
                if not unmeasured:
                    continue                      # both already clicked, still tied
                m = unmeasured[0]
                bb = ents[m].current_bbox
                click = None
                try:
                    import structural_grid
                    d = structural_grid.read_path(getattr(self, "_last_frame_path", None),
                                                  list(bb))
                    cells = (d or {}).get("cells") or []
                    if cells:
                        cell = cells[0]
                        click = f"CLICK:{int((cell[1]+cell[3])//2)},{int((cell[0]+cell[2])//2)}"
                except Exception:
                    pass
                if click is None:
                    r0, c0, r1, c1 = bb
                    click = f"CLICK:{int((c0+c1)//2)},{int((r0+r1)//2)}"
                act = self._resolve_action(click)
                # Snapshot the pre-click raw frame + the panel's region so the
                # NEXT turn can MEASURE settable directly (pre/post pixel diff of
                # this panel), independent of the substrate visual_events pipeline.
                self._disambig_pending = {
                    "name": m, "bbox": [int(v) for v in bb],
                    "pre_frame": getattr(self, "_last_frame_path", None)}
                print(f"[struct] active/reference undetermined for group '{gname}'; "
                      f"probing {m} ({click}) to measure which similar panel is settable.")
                return SimpleNamespace(
                    action=act, plan_kind="structural_disambiguation",
                    rationale=(f"structural: click {m} to measure which similar panel "
                               f"is settable (active) -> enables the match-to-reference"),
                    goal_id="structural", target_cell=None,
                    full_plan_actions=[act], is_probe=True)
            return None
        except Exception as e:
            print(f"[struct] disambiguation probe failed ({e})")
            return None

    def _structural_detect(self):
        """Build a 'make the ACTIVE structure match the REFERENCE structure, then
        fire the trigger' plan from the top-ranked structural / similarity
        candidate whose two structures resolve to the SAME number of cell-units.

        Each structure is resolved to per-cell units by _resolve_structure_units
        -- which DECOMPOSES a monolithic panel entity into its lattice of cells
        when the cells were not perceived as separate entities -- so the executor
        can diff and click individual differing cells instead of one block.
        Returns a state dict or None."""
        try:
            ents = self.world.entities
            for claim_id, A, B in self._structural_candidates():
                au = self._resolve_structure_units(A)   # active units (to change)
                bu = self._resolve_structure_units(B)   # reference units (template)
                if not au or not bu or len(au) != len(bu):
                    continue
                pairs = list(zip(au, bu))               # (active_cell, reference_cell)
                # the trigger is a trigger_target entity OUTSIDE both structures
                # (the GO ball), so a cell is never mistaken for the trigger.
                exclude = [bb for bb in (self._structure_bbox(A), self._structure_bbox(B))
                           if bb is not None]

                def _inside(bb, r, c):
                    return bb[0] <= r < bb[2] and bb[1] <= c < bb[3]
                trig = None
                for n, r in ents.items():
                    if n in (A, B) or r.current_bbox is None:
                        continue
                    if (getattr(r, "current_role", "") or "") != "trigger_target":
                        continue
                    r0, c0, r1, c1 = r.current_bbox
                    rr, cc = (r0 + r1) // 2, (c0 + c1) // 2
                    if any(_inside(bb, rr, cc) for bb in exclude):
                        continue
                    trig = f"CLICK:{int(cc)},{int(rr)}"
                    break
                if trig is None:
                    continue
                # each structure's own BACKGROUND (stable, whole-region) so the
                # match is background-invariant across e.g. a white vs grey field.
                ch_bg = self._cx_region_bg([p[0] for p in pairs])   # active
                ex_bg = self._cx_region_bg([p[1] for p in pairs])   # reference
                print(f"[struct] DETECTED plan for '{claim_id}': match {A} "
                      f"({len(au)} cell-units) to reference {B}, then fire {trig}.")
                return {"phase": "match", "claim_id": claim_id, "pairs": pairs,
                        "trigger": trig, "idx": 0, "budget": 6, "progress": {},
                        "ex_bg": ex_bg, "ch_bg": ch_bg}
            return None
        except Exception as e:
            print(f"[struct] detect failed ({e})")
            return None

    def _structural_match_probe(self, entities_for_actor):
        """EXECUTE a structural 'make the active panel match the template panel,
        then fire' plan -- the step that turns a structural template CLAIM into a
        solve, and VERIFIES it (win -> proven, else refuted).  Drives each active
        unit to its paired template unit with the same best-shift diff + progress
        guard as the cx match loop.  Guarded -> None so other tiers still run."""
        cs = getattr(self, "_claim_store", None)
        if cs is None:
            return None
        try:
            from types import SimpleNamespace
            st = (self.world.probe_state or {}).get("struct")
            if not st or st.get("phase") == "unavailable":
                st = self._structural_detect() or {"phase": "unavailable"}
                self.world.probe_state["struct"] = st
            ph = st.get("phase")
            if ph in ("unavailable", "done"):
                return None
            if ph == "match":
                pairs = st["pairs"]
                prog = st.setdefault("progress", {})
                while st["idx"] < len(pairs):
                    a_bb, b_bb = pairs[st["idx"]]
                    tmpl = self._cx_read_sprite(getattr(self, "_last_frame_path", None), b_bb)
                    regions = (self._cx_diff_regions(tmpl, a_bb, ex_bg=st.get("ex_bg"),
                                                     ch_bg=st.get("ch_bg"))
                               if tmpl is not None else [])
                    if not regions:
                        prog.pop(st["idx"], None); st["idx"] += 1
                        continue
                    total = sum(r[2] for r in regions)
                    s = prog.setdefault(st["idx"], {"last_total": None, "tried": [],
                                                    "clicks": 0, "last_click": None})
                    if (s["last_total"] is not None and total >= s["last_total"]
                            and s["last_click"] and s["last_click"] not in s["tried"]):
                        s["tried"].append(s["last_click"])
                    pick = next(([r[0], r[1]] for r in regions
                                 if [int(r[0]), int(r[1])] not in s["tried"]), None)
                    if pick is None or s["clicks"] >= st["budget"]:
                        print(f"[struct] cannot match unit {st['idx']} -> structural "
                              f"claim '{st['claim_id']}' failing; refuting.")
                        cs.close(st["claim_id"], "refuted",
                                 turn=int(getattr(self.world, "turn", 0) or 0))
                        st["phase"] = "done"; return None
                    col, row = int(pick[0]), int(pick[1])
                    s["last_total"] = total; s["last_click"] = [col, row]; s["clicks"] += 1
                    act = self._resolve_action(f"CLICK:{col},{row}")
                    return SimpleNamespace(
                        action=act, plan_kind="structural",
                        rationale=f"structural: match the active panel to the template "
                                  f"(unit {st['idx'] + 1}/{len(pairs)})",
                        goal_id="structural", target_cell=None,
                        full_plan_actions=[act], is_probe=True)
                st["phase"] = "fire"
            if st["phase"] == "fire":
                st["phase"] = "fired"
                act = self._resolve_action(st["trigger"])
                print(f"[struct] active panel matches the template; firing the trigger "
                      f"to verify '{st['claim_id']}'.")
                return SimpleNamespace(
                    action=act, plan_kind="structural",
                    rationale="structural: fire the trigger after matching the template",
                    goal_id="structural", target_cell=None,
                    full_plan_actions=[act], is_probe=True)
            if st["phase"] == "fired":
                won = getattr(self.world, "win_state", "playing") != "playing"
                last = list(getattr(self.world, "deltas_observed", []) or [])
                scored = won or (bool(last) and getattr(last[-1], "score_increased", False))
                cs.close(st["claim_id"], "proven" if scored else "refuted",
                         turn=int(getattr(self.world, "turn", 0) or 0))
                print("[struct] match+fire " + ("SCORED -- structural claim CONFIRMED "
                      "(the template solved the panel)." if scored else
                      "did not score -- structural claim refuted; explore more."))
                st["phase"] = "done"
            return None
        except Exception as e:
            print(f"[struct] probe error ({e})")
            return None

    def _cx_panel_components(self, region_bbox):
        """Split a PANEL region into its switch UNITS' click points [col,row], so
        OFAT can bootstrap its controls from a panel the VLM perceived as one
        region.  Guarded -> [] on any miss.

        Prefers structural_grid's FULL LATTICE -- every cell position, INCLUDING
        currently-empty ones -- because OFAT must be able to poke (toggle) every
        switch to compose a configuration, not only the cells that are already
        filled.  Falls back to the `components` op (filled marks only) when no
        regular lattice is found."""
        if not region_bbox:
            return []
        # 1) full lattice (all cells): the right control set for calibration.
        try:
            import structural_grid
            d = structural_grid.read_path(getattr(self, "_last_frame_path", None),
                                          list(region_bbox))
            cells = (d or {}).get("cells") or []
            if len(cells) >= 2:
                return [[int((c[1] + c[3]) // 2), int((c[0] + c[2]) // 2)]  # [col,row]
                        for c in cells]
        except Exception as e:
            print(f"[cx-ofat] lattice decompose skipped ({e})")
        # 2) fallback: filled connected components only.
        try:
            import tempfile
            import numpy as np
            from PIL import Image
            from substrate_tools.registry import run_queries
            fp = getattr(self, "_last_frame_path", None)
            if not fp or not Path(str(fp)).exists():
                return []
            im = Image.open(str(fp)).convert("RGB")
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.NEAREST)
            arr = np.asarray(im, dtype=np.uint8)
            res = run_queries(arr, [{"op": "components", "id": "panel",
                                     "bbox": list(region_bbox), "min_cells": 2}],
                              tempfile.gettempdir(), n_ticks=64)[0]
            return [[int(c["centroid"][1]), int(c["centroid"][0])]   # (col, row)
                    for c in (res.get("components") or [])]
        except Exception as e:
            print(f"[cx-ofat] panel-components skipped ({e})")
            return []

    def _cx_peak(self, ev):
        """The mover's PEAK position (farthest from its start) across the
        animation trajectory -- the right response variable for a transient mover
        that returns to rest (net ~0)."""
        try:
            traj = ev.get("trajectory") or []
            start = tuple(ev.get("from") or (traj[0][1] if traj else (0, 0)))
            peak, best = start, -1.0
            for t in traj:
                p = tuple(t[1])
                d = abs(p[0] - start[0]) + abs(p[1] - start[1])
                if d > best:
                    best, peak = d, p
            return (round(peak[0], 1), round(peak[1], 1))
        except Exception:
            return None

    def _cx_find_twin(self, mover_hex, mover_start):
        """The mover's same-colour FIXED twin = the natural TARGET (tn36's cup is
        the same yellow as the descending arch).  Per-colour CC on the current
        frame; returns the farthest same-colour blob centroid, or None."""
        try:
            import numpy as np
            from PIL import Image
            fp = getattr(self, "_last_frame_path", None)
            if not fp or not Path(str(fp)).exists():
                return None
            im = Image.open(str(fp)).convert("RGB")
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.NEAREST)
            arr = np.asarray(im, dtype=int)
            packed = (arr[:, :, 0] << 16) | (arr[:, :, 1] << 8) | arr[:, :, 2]
            want = int(str(mover_hex).lstrip("#"), 16)
            mask = (packed == want)
            if int(mask.sum()) < 5:
                return None
            H, W = mask.shape
            seen = np.zeros_like(mask, bool)
            blobs = []
            ys, xs = np.where(mask)
            for y0, x0 in zip(ys.tolist(), xs.tolist()):
                if seen[y0, x0]:
                    continue
                st = [(y0, x0)]; seen[y0, x0] = True; cl = []
                while st:
                    y, x = st.pop(); cl.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
                rr = [p[0] for p in cl]; cc = [p[1] for p in cl]
                blobs.append((sum(rr) / len(rr), sum(cc) / len(cc)))
            far, best = None, -1.0
            for (r, c) in blobs:
                d = abs(r - mover_start[0]) + abs(c - mover_start[1])
                if d > best:
                    best, far = d, (round(r, 1), round(c, 1))
            return far if best > 4 else None
        except Exception:
            return None

    def _cx_detect(self, entities_for_actor):
        """Detect controlled-experiment preconditions from what COS has ALREADY
        observed: a TRANSIENT MOVER + the TRIGGER that animated it (substrate
        animation_events), >=2 toggle CONTROLS (clicked -> changed in place, no
        animation, and NOT changing on non-click turns -- excludes a budget
        meter), and a same-colour fixed TARGET.  Returns a cx state dict, or None
        (not yet detectable -> coverage keeps gathering the deltas it needs)."""
        try:
            # only THIS level's deltas: _level_start_turn is set one PAST the
            # transition turn, so the transition delta (the prior level's winning
            # action) is excluded but every in-level delta is kept.
            _lst = int(getattr(self, "_level_start_turn", 0) or 0)
            deltas = [d for d in (getattr(self.world, "deltas_observed", []) or [])
                      if int(getattr(d, "to_turn", 0) or 0) >= _lst]
            mover = trigger = mstart = mhex = None
            go_evs = []
            go_turn = None
            for d in reversed(deltas):
                aes = list(getattr(d, "animation_events", None) or [])
                for ev in aes:
                    if "transient" in " ".join(ev.get("verbs", [])).lower():
                        mover, trigger = ev, getattr(d, "action", None)
                        mstart = tuple(ev.get("from") or (0, 0))
                        mhex = ev.get("colour_hex")
                        go_evs = aes
                        go_turn = getattr(d, "to_turn", None)
                        break
                if mover is not None:
                    break
            if mover is None or not trigger:
                return None
            non_click_changed = set()
            for d in deltas:
                if not (getattr(d, "action", "") or "").startswith("CLICK"):
                    for nm in (getattr(d, "entities_changed", []) or []):
                        non_click_changed.add(nm)
            controls = {}
            for d in deltas:
                act = getattr(d, "action", "") or ""
                if not act.startswith("CLICK") or getattr(d, "animation_events", None):
                    continue
                for nm in (getattr(d, "entities_changed", []) or []):
                    if nm in non_click_changed:
                        continue
                    rec = self.world.entities.get(nm)
                    if rec is None or rec.current_bbox is None:
                        continue
                    if (getattr(rec, "current_role", "") or "") == "hud":
                        continue
                    r0, c0, r1, c1 = rec.current_bbox
                    controls[nm] = (int((c0 + c1) // 2), int((r0 + r1) // 2))
            if len(controls) < 2:
                return None
            # Probe the WHOLE reasonable control set, not only the few toggled so
            # far: if a confirmed control belongs to a similarity GROUP of like
            # entities, every member is likely a control too.  Game-agnostic --
            # like entities share a role.
            for g in (getattr(self.world, "groups", {}) or {}).values():
                mem = list(getattr(g, "members", []) or [])
                if not any(m in controls for m in mem):
                    continue
                for m in mem:
                    if m in controls:
                        continue
                    rec = self.world.entities.get(m)
                    if rec is None or rec.current_bbox is None \
                            or (getattr(rec, "current_role", "") or "") == "hud":
                        continue
                    r0, c0, r1, c1 = rec.current_bbox
                    controls[m] = (int((c0 + c1) // 2), int((r0 + r1) // 2))
            order = sorted(controls, key=lambda nm: controls[nm][0])
            target = self._cx_find_twin(mhex, mstart)
            if target is None:
                return None
            # GOAL-DIRECTED single-trial induction (the experiment is a win
            # SUBGOAL, not an end): the trigger's CURSOR visits the controls in
            # sequence, so ONE trigger fire already attributes each mover step to
            # the control the cursor is over -- no multi-trial OFAT needed.  Find
            # the cursor = the OTHER animated entity whose path rides the controls'
            # row band.
            ctrl_row = sum(controls[c][1] for c in controls) / len(controls)
            mover_traj = [(t[0], (t[1][0], t[1][1]))
                          for t in (mover.get("trajectory") or [])]
            cursor_traj, best = None, 1e9
            for ev in go_evs:
                if ev is mover:
                    continue
                tr = ev.get("trajectory") or []
                if len(tr) < 2:
                    continue
                mr = sum(t[1][0] for t in tr) / len(tr)
                if abs(mr - ctrl_row) < best:
                    best = abs(mr - ctrl_row)
                    cursor_traj = [(t[0], (t[1][0], t[1][1])) for t in tr]
            if not cursor_traj:
                return None
            import controlled_experiment as _CX
            steps = _CX.attribute_steps(mover_traj, cursor_traj, controls)
            # raw displacement toward the target -- imitation_plan's cosine
            # weights by magnitude, so a large 'down' dominates a negligible
            # sideways offset (no sign-thresholding / magic number needed).
            tdir = (target[0] - mstart[0], target[1] - mstart[1])
            exemplars, _tc, exid = _CX.imitation_plan(steps, tdir)
            if not exemplars or exid not in self.world.entities:
                return None
            # Capture the proven DOWN-STATE sprite as the imitation TEMPLATE.
            # Read it from the RAW current frame (`_last_frame_path`): the gridded
            # per-turn PNGs carry axis-label margins and do NOT map to 64-tick
            # space.  To stay immune to a coverage click that mutated an exemplar
            # AFTER the proving GO, pick an exemplar whose switch was NOT clicked
            # since that GO -- its current sprite still IS the proven down-state.
            clicked_since_go = set()
            for d in deltas:
                if ((getattr(d, "to_turn", -1) or -1) > (go_turn or -1)
                        and (getattr(d, "action", "") or "").startswith("CLICK")):
                    for nm in (getattr(d, "entities_changed", []) or []):
                        clicked_since_go.add(nm)
            pristine = [e for e in exemplars if e not in clicked_since_go]
            exid = (pristine or list(exemplars))[0]
            ex_rec = self.world.entities.get(exid)
            if ex_rec is None or ex_rec.current_bbox is None:
                return None
            ex_sprite = self._cx_read_sprite(
                getattr(self, "_last_frame_path", None), ex_rec.current_bbox)
            if ex_sprite is None:
                return None
            # EVERY control must end in the proven down-state -> match them ALL to
            # the template.  Controls already in that state are skipped instantly;
            # a coverage-corrupted exemplar gets RESTORED too (it is just another
            # control that differs from its own proven template).
            to_change = list(order)
            # APPLY by IMITATION via a CLOSED-LOOP MATCH: click where a control
            # still differs from the template, re-perceive, repeat until it
            # matches -- but only while each click REDUCES the diff (a click that
            # does not is not repeated; if none help, abandon that control fast).
            # Then fire the trigger = the win attempt.
            cx = {"phase": "matching", "controls": controls, "order": order,
                  "trigger": trigger, "mhex": mhex, "mstart": list(mstart),
                  "target": list(target),
                  "steps": {k: list(v) for k, v in steps.items()},
                  "exemplars": list(exemplars), "to_change": list(to_change),
                  "exid": exid, "ex_sprite": ex_sprite.tolist(),
                  "match_idx": 0, "progress": {}, "budget": 6}
            print(f"[cx] single-trial law {steps}; target_dir {tdir}; exemplars "
                  f"{exemplars}; match ALL controls {to_change} to the proven "
                  f"down-state of {exid!r} (best-shift diff + progress-guarded "
                  f"click-reverify) then fire the trigger.")
            return cx
        except Exception as e:
            print(f"[cx] detect failed ({e})")
            return None

    def _cx_read_sprite(self, frame_path, bbox):
        """Read the RGB sprite (np array h x w x 3) at ``bbox`` from a 64x64
        frame; None on any miss.  Guarded."""
        try:
            import numpy as np
            from PIL import Image
            if not frame_path or not Path(str(frame_path)).exists():
                return None
            im = Image.open(str(frame_path)).convert("RGB")
            if im.size != (64, 64):
                im = im.resize((64, 64), Image.NEAREST)
            arr = np.asarray(im, dtype=int)
            r0, c0, r1, c1 = [int(v) for v in bbox]
            r0, c0 = max(0, r0), max(0, c0)
            r1, c1 = min(arr.shape[0], r1), min(arr.shape[1], c1)
            if r1 <= r0 or c1 <= c0:
                return None
            return arr[r0:r1, c0:c1]
        except Exception:
            return None

    def _cx_diff_regions(self, template, ch_bbox, ex_bg=None, ch_bg=None):
        """Connected regions where the control at ``ch_bbox`` (read on the CURRENT
        frame) differs from the proven down-state ``template`` (a stored sprite),
        AFTER a small best-shift alignment that absorbs +/-2px bbox error -- so a
        1px-misplaced thin feature (e.g. a switch stem) is not reported as a
        phantom forever-difference that the match loop re-clicks forever.  Returns
        [(col, row, size), ...] centroids in ABSOLUTE frame coords, LARGEST-first;
        [] when the control already matches the template.  Guarded."""
        try:
            import numpy as np
            ex = template if isinstance(template, np.ndarray) \
                else np.asarray(template, dtype=int)
            if ex is None or ex.size == 0:
                return []
            ch = self._cx_read_sprite(getattr(self, "_last_frame_path", None), ch_bbox)
            if ch is None or ch.size == 0:
                return []
            cr0, cc0 = int(ch_bbox[0]), int(ch_bbox[1])

            # BACKGROUND-INVARIANT compare (game-agnostic), used only when the
            # caller passes the two panels' BACKGROUND colours and they DIFFER
            # (cross-panel template matching: a white-backed panel vs a grey-backed
            # one).  A cell that is background in BOTH panels matches regardless of
            # the two background colours; a real difference is a presence mismatch
            # (mark vs no mark) or a foreground-colour difference.  When no
            # backgrounds are supplied (within-panel matching) it is the plain
            # pixel diff, so level-one matching is unchanged.  The backgrounds come
            # from the whole PANEL region (stable), never a tight per-unit dominant.
            _cross_bg = (ex_bg is not None and ch_bg is not None
                         and not np.array_equal(np.asarray(ex_bg), np.asarray(ch_bg)))
            _exb = np.asarray(ex_bg) if _cross_bg else None
            _chb = np.asarray(ch_bg) if _cross_bg else None

            def _diff(e, c):
                if not _cross_bg:
                    return (e != c).any(axis=2)
                e_bg = (e == _exb).all(axis=2)
                c_bg = (c == _chb).all(axis=2)
                presence = e_bg != c_bg            # mark vs no-mark
                both_fg = (~e_bg) & (~c_bg)
                return presence | (both_fg & (e != c).any(axis=2))
            # Best (dy,dx) shift of the template over the control, minimizing the
            # mismatch on the overlap, to absorb small bbox misalignment.  Prefer
            # fewer diffs, then larger overlap.
            # Try only small shifts (+/-1px) and, on ties, PREFER the smallest
            # shift + largest overlap -- so a feature is never slid onto
            # background to spuriously zero a real colour difference.
            best = None
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    er0, ec0 = max(0, -dy), max(0, -dx)   # template offset
                    hr0, hc0 = max(0, dy), max(0, dx)     # control offset
                    h = min(ex.shape[0] - er0, ch.shape[0] - hr0)
                    w = min(ex.shape[1] - ec0, ch.shape[1] - hc0)
                    if h < 1 or w < 1:
                        continue
                    d = _diff(ex[er0:er0 + h, ec0:ec0 + w],
                              ch[hr0:hr0 + h, hc0:hc0 + w])
                    score = (int(d.sum()), abs(dy) + abs(dx), -(h * w))
                    if best is None or score < best[0]:
                        best = (score, hr0, hc0, h, w, d)
            if best is None:
                return []
            _, hr0, hc0, h, w, diff = best
            if not diff.any():
                return []
            seen = np.zeros_like(diff, bool)
            clicks = []
            ys, xs = np.where(diff)
            for y0, x0 in zip(ys.tolist(), xs.tolist()):
                if seen[y0, x0]:
                    continue
                st = [(y0, x0)]; seen[y0, x0] = True; cells = []
                while st:
                    y, x = st.pop(); cells.append((y, x))
                    for dyy, dxx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dyy, x + dxx
                        if 0 <= ny < h and 0 <= nx < w and diff[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True; st.append((ny, nx))
                rr = [c[0] for c in cells]; ccc = [c[1] for c in cells]
                clicks.append((cc0 + hc0 + int(round(sum(ccc) / len(ccc))),
                               cr0 + hr0 + int(round(sum(rr) / len(rr))), len(cells)))
            clicks.sort(key=lambda r: r[2], reverse=True)
            return clicks
        except Exception:
            return []

    def _controlled_experiment_probe(self, entities_for_actor):
        """Goal-directed controlled-experiment probe (game-agnostic).  The
        experiment is a WIN SUBGOAL: once a transient mover + a sequential-scan
        trigger + toggle controls + a target are detected, ONE trigger fire
        attributes each mover step to the control the cursor visits, the
        toward-goal exemplar is identified, and the other controls are made to
        IMITATE it (click where they differ) before firing the trigger = the win
        attempt.  Apply first; only on no-score does COS fall back to exploring
        more claims (this probe returns None -> coverage/escalation resumes).
        FULLY GUARDED -- returns None on any miss/error, never breaks the run."""
        try:
            from types import SimpleNamespace
            cx = self.world.probe_state.get("cx")
            if not cx or cx.get("phase") == "unavailable":
                # re-detect each turn while not yet detectable (the mover only
                # appears after the trigger fires; the controls after they toggle)
                cx = self._cx_detect(entities_for_actor) or {"phase": "unavailable"}
                self.world.probe_state["cx"] = cx
            ph = cx.get("phase")
            if ph in ("unavailable", "done"):
                return None
            if ph == "matching":
                tmpl = cx.get("ex_sprite")
                if tmpl is None:
                    cx["phase"] = "done"; return None
                prog = cx.setdefault("progress", {})
                # CLOSED-LOOP MATCH with a PROGRESS GUARD: for the first control
                # that still differs from the proven down-state, click its largest
                # differing region -- but if the previous click did NOT reduce the
                # total diff, mark that region tried and pick another, and if no
                # region makes progress, abandon this control fast (no re-clicking
                # the same spot until the budget runs out).  One click/turn so the
                # next re-diff sees a fresh frame.
                while cx["match_idx"] < len(cx["to_change"]):
                    nm = cx["to_change"][cx["match_idx"]]
                    rec = self.world.entities.get(nm)
                    if rec is None or rec.current_bbox is None:
                        cx["match_idx"] += 1; prog.pop(nm, None)
                        continue
                    regions = self._cx_diff_regions(tmpl, rec.current_bbox)
                    if not regions:
                        if prog.get(nm, {}).get("clicks"):
                            print(f"[cx] {nm} now matches the proven down-state.")
                        cx["match_idx"] += 1; prog.pop(nm, None)
                        continue
                    total = sum(r[2] for r in regions)
                    st = prog.setdefault(nm, {"last_total": None, "tried": [],
                                              "clicks": 0, "last_click": None})
                    # the previous click did not shrink the diff -> that spot is
                    # not the controllable difference; don't click it again.
                    if (st["last_total"] is not None and total >= st["last_total"]
                            and st["last_click"] and st["last_click"] not in st["tried"]):
                        st["tried"].append(st["last_click"])
                    pick = next(([r[0], r[1]] for r in regions
                                 if [int(r[0]), int(r[1])] not in st["tried"]), None)
                    if pick is None or st["clicks"] >= cx["budget"]:
                        print(f"[cx] cannot drive {nm} to the proven down-state "
                              f"(tried {len(st['tried'])} region(s), {st['clicks']} "
                              f"click(s)) -> imitation failing; explore more.")
                        cx["phase"] = "done"; return None
                    col, row = int(pick[0]), int(pick[1])
                    st["last_total"] = total
                    st["last_click"] = [col, row]
                    st["clicks"] += 1
                    act = self._resolve_action(f"CLICK:{col},{row}")
                    return SimpleNamespace(
                        action=act, plan_kind="controlled_experiment",
                        rationale=f"controlled-experiment: match {nm} to the proven "
                                  f"down-state of {cx['exid']} (click {st['clicks']})",
                        goal_id="controlled_experiment", target_cell=None,
                        full_plan_actions=[act], is_probe=True)
                # every control matches the proven down-state -> fire the trigger
                cx["phase"] = "fired"
                act = self._resolve_action(cx["trigger"])
                print("[cx] all controls imitate the exemplar; firing the trigger "
                      "(win attempt).")
                return SimpleNamespace(
                    action=act, plan_kind="controlled_experiment",
                    rationale="controlled-experiment: fire trigger after imitation",
                    goal_id="controlled_experiment", target_cell=None,
                    full_plan_actions=[act], is_probe=True)
            if ph == "fired":
                won = getattr(self.world, "win_state", "playing") != "playing"
                last = list(getattr(self.world, "deltas_observed", []) or [])
                scored = won or (bool(last) and getattr(last[-1], "score_increased", False))
                print("[cx] imitate+fire " + ("SCORED -- subgoal achieved."
                      if scored else "did not score; handing back to explore more claims."))
                cx["phase"] = "done"
            return None
        except Exception as e:
            print(f"[cx] probe error ({e}); falling through")
            return None

    def _instruction_verification_probe(self, entities_for_actor):
        """When the win/mechanic is NOT understood, an on-screen DESIGNER
        INSTRUCTION (an aim/trajectory line, a legend/key, a highlighted marker or
        cursor) is the HIGHEST-VALUE lead -- ARC-AGI-3 games are built to be easy
        for a human, so the first level TEACHES its mechanic on screen.  So, unless
        COS already holds a committed plan, VERIFY the instruction RIGHT AWAY:
        act on the instructive element to learn what it does, AHEAD of blind
        action-space scouting.  Game-agnostic -- keys on the perception layer
        flagging an element as instruction-like (role/appearance), then clicks it.
        Fires once per element (tracked in probe_state).  Fully guarded: returns
        None when the win is understood, nothing reads as instructive, or clicking
        is unsupported."""
        try:
            if self._win_understood():
                return None
            if "CLICK" not in self.game.available_actions():
                return None
            INSTR = ("instruction", "tutorial", "aim", "trajectory", "cursor",
                     "reticle", "marker", "pointer", "legend", "key", "arrow",
                     "guide", "preview", "hint", "indicator", "target line",
                     "dashed", "dotted")
            done = self.world.probe_state.setdefault("instruction_probed", [])
            for e in entities_for_actor:
                nm = e.get("name")
                if not nm or nm in done or not e.get("bbox_ticks_turn1"):
                    continue
                tags = ((e.get("role_hypothesis") or "") + " "
                        + (e.get("appearance") or "")).lower()
                if not any(w in tags for w in INSTR):
                    continue
                done.append(nm)
                # Record the demo context so _learn_move_law_from_demo can detect a
                # click-to-relocate (a piece ends up WHERE this marker was) and
                # extract the NUMERIC step.  Also stash the AGENT's pre-click centre
                # so the step can be learned as |marker - mover_before| even if the
                # grounding later mis-tracks the relocated piece (same-colour HUD
                # distractor).
                bb = e.get("bbox_ticks_turn1") or e.get("bbox_ticks")
                if bb:
                    mb = None
                    ag = next((x for x in entities_for_actor
                               if "agent" in (x.get("role_hypothesis") or "").lower()
                               and x.get("bbox_ticks_turn1")), None)
                    if ag:
                        ab = ag["bbox_ticks_turn1"]
                        mb = ((ab[0] + ab[2]) / 2.0, (ab[1] + ab[3]) / 2.0)
                    self._move_demo = {
                        "marker": ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0),
                        "marker_name": nm, "mover_before": mb,
                        "turn": int(getattr(self.world, "turn", 0) or 0)}
                from types import SimpleNamespace
                action = self._resolve_action(f"CLICK:{nm}")
                return SimpleNamespace(
                    action=action,
                    rationale=(f"VERIFY ON-SCREEN INSTRUCTION: act on '{nm}' "
                               f"(designer guidance) to learn its function before "
                               f"blind probing"),
                    plan_kind="instruction_verify",
                    goal_id=f"verify_instruction:{nm}",
                    target_cell=None, full_plan_actions=[action], is_probe=True,
                )
            return None
        except Exception:
            return None

    def _locate_via_substrate(self, near, *, color=None):
        """Locate an entity on the RAW frame via the substrate's locate_entity
        (near a last-known point), bypassing the entity-grounding's same-colour
        confusion (e.g. a HUD swatch the same colour as the mover).  Uses colour
        when known, else the 'moved' mode (the piece that moved in the last
        action's animation).  Returns (row,col) or None.  Guarded."""
        try:
            fp = getattr(self, "_last_frame_path", None)
            if not fp or not Path(fp).exists() or near is None:
                return None
            q = {"op": "locate_entity", "id": "mv",
                 "near": [float(near[0]), float(near[1])]}
            if color:
                q["color"] = color
            else:
                q["match"] = "moved"
            ad = getattr(self, "_last_anim_dir", None)
            anim = None
            if ad is not None:
                fr = sorted(Path(ad).glob("frame_*.png"))
                anim = fr if len(fr) >= 2 else None
            out = Path(self.work_dir) / "_movelaw_vq"
            res = _VQ.run_visual_queries(str(fp), [q], out,
                                         n_ticks=self.n_ticks, anim_frames=anim)
            r = (res or [{}])[0]
            c = r.get("centroid")
            return (float(c[0]), float(c[1])) if r.get("found") and c else None
        except Exception:
            return None

    def _vacated_centroid_from_anim(self):
        """Before-position of the piece that MOVED in the last action's animation:
        the largest PLAYFIELD blob that was foreground in the FIRST sub-frame and
        vacated (-> background) in the LAST.  Lets the move-law learn the step from
        the DEMONSTRATED MOTION even when the perception model mislabelled which
        static object is the mover (a small-model failure mode: tagging the
        cursor/marker as the mover and the real piece as 'decoration').  Returns
        (row,col) in tick space or None.  Guarded."""
        try:
            import numpy as np
            from PIL import Image
            ad = getattr(self, "_last_anim_dir", None)
            if not ad:
                return None
            fr = sorted(Path(ad).glob("frame_*.png"))
            if len(fr) < 2:
                return None
            a = np.asarray(Image.open(fr[0]).convert("RGB")).astype(int)
            b = np.asarray(Image.open(fr[-1]).convert("RGB")).astype(int)
            if a.shape != b.shape:
                return None
            sc = a.shape[0] / 64.0
            vac = (a.max(2) > 40) & (b.max(2) <= 40)      # was foreground, now bg
            vac[int(58 * sc):, :] = False                 # drop the bottom HUD / shot bar
            ys, xs = np.where(vac)
            if len(ys) < 1:
                return None
            pts = set(zip(ys.tolist(), xs.tolist()))      # largest connected vacated blob
            seen, best = set(), []
            for p0 in list(pts):
                if p0 in seen:
                    continue
                st = [p0]; seen.add(p0); comp = []
                while st:
                    y, x = st.pop(); comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        q = (y + dy, x + dx)
                        if q in pts and q not in seen:
                            seen.add(q); st.append(q)
                if len(comp) > len(best):
                    best = comp
            if not best:
                return None
            return (sum(p[0] for p in best) / len(best) / sc,
                    sum(p[1] for p in best) / len(best) / sc)
        except Exception:
            return None

    def _learn_move_law_from_demo(self) -> None:
        """After the instruction-verification click on a one-shot marker, learn the
        NUMERIC step = |marker - mover_before| -- the generalization the marker
        taught: keep the NUMBER, not the marker, so once it's gone COS recomputes
        the next exact click itself (mover + step*unit(mover->goal)) and on the NEXT
        level re-measures step from that level's own demo.  Robust to the grounding
        mis-tracking the relocated piece (same-colour distractor): it learns from
        the PRE-click agent centre + the marker, confirming the demo fired (the
        one-shot marker disappeared, a piece is now at the marker, or the substrate
        locates the moved piece there).  Game-agnostic; runs each turn; guarded."""
        try:
            if getattr(self, "_move_law", None) is not None or not getattr(self, "_move_demo", None):
                return
            import move_law as _ML
            d = self._move_demo
            marker, mb = d["marker"], d.get("mover_before")
            # ROBUST MOVER IDENTITY (small-model compensation): a weak perception
            # model may not label the mover (no 'agent' role) or may MISLABEL which
            # static object is the mover.  The DEMONSTRATED MOTION is ground truth:
            # derive the before-position from the animation (the cell the piece
            # VACATED as it relocated onto the marker), so the numeric step is
            # learned regardless of the static labels.
            if mb is None or _ML._dist(mb, marker) < 3.0:
                mb = self._vacated_centroid_from_anim() or mb
            if mb is None or _ML._dist(mb, marker) < 3.0:
                self._move_demo = None
                return
            ents = getattr(self.world, "entities", {}) or {}

            def ctr(bb):
                r0, c0, r1, c1 = bb[:4]
                return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

            marker_gone = d.get("marker_name") not in ents
            at_marker = any(_ML._dist(ctr(r.bbox_history[-1][1]), marker) <= 2.5
                            for r in ents.values() if getattr(r, "bbox_history", None))
            loc = None if (marker_gone or at_marker) else self._locate_via_substrate(marker)
            if not (marker_gone or at_marker
                    or (loc is not None and _ML._dist(loc, marker) <= 3.0)):
                return                                   # the demo has not fired yet
            law = _ML.learn_step_from_demo(mb, marker, marker)   # the piece went to the marker
            if law is not None:
                self._move_law = law
                self._move_last = marker                 # the mover is now at the marker
                self._move_demo = None
                print(f"[move-law] learned click-to-move step={law.step:.1f} "
                      f"from the one-shot demo (mover {tuple(round(x) for x in mb)}"
                      f"->{tuple(round(x) for x in marker)})", flush=True)
        except Exception:
            pass

    def _move_law_pursuit(self, entities_for_actor):
        """Once the click-to-move step is learned, WALK the piece to the goal by
        computing the EXACT next click numerically: mover + step*unit(mover->goal),
        clamped so it never overshoots.  Tracks the mover via the substrate
        (locate_entity near last-known) so the entity-grounding's same-colour
        confusion can't derail it.  Highest-priority action once learned.  Returns
        a CLICK probe, or None (no law / no goal / already overlapping).  Guarded."""
        try:
            law = getattr(self, "_move_law", None)
            if law is None or getattr(self, "_move_law_disabled", False):
                return None
            import move_law as _ML

            def ctr(bb):
                r0, c0, r1, c1 = bb[:4]
                return ((r0 + r1) / 2.0, (c0 + c1) / 2.0)

            near = getattr(self, "_move_last", None)
            mover = self._locate_via_substrate(near) if near else None
            if mover is None:
                mover = near                             # fall back to last-known/expected
            goal = goal_name = None
            for e in entities_for_actor:
                nm = e.get("name")
                tags = ((e.get("role_hypothesis") or "") + " "
                        + (e.get("appearance") or "")).lower()
                if (nm and e.get("bbox_ticks_turn1")
                        and any(w in tags for w in ("goal", "target", "blob",
                                                    "ball", "exit", "win"))):
                    goal = ctr(e["bbox_ticks_turn1"])
                    goal_name = nm
                    break
            if mover is None or goal is None:
                return None
            # PURSUIT-PROGRESS WATCHDOG: the learned step is a HYPOTHESIS.  A wrong step or
            # a mis-tracked mover makes the click-walk OSCILLATE instead of converge (so a
            # previously-solved game can fail on a noisy fresh demo).  Verify the mover is
            # actually getting CLOSER to the goal each step; if it stalls (no decrease) for a
            # few consecutive steps, the law is wrong -> DROP it (a fresh demo can re-learn).
            # After a couple of such failures, DISABLE it for the run so COS hands off to
            # exploration/strategy instead of clicking forever to no effect.  Mirrors the
            # merge mechanism's score watchdog; game-agnostic (the gate is structural --
            # 'not getting closer' -- not a tuned magnitude).
            _dist = ((mover[0] - goal[0]) ** 2 + (mover[1] - goal[1]) ** 2) ** 0.5
            _last = getattr(self, "_move_law_last_dist", None)
            self._move_law_stall = (getattr(self, "_move_law_stall", 0) + 1
                                    if (_last is not None and _dist >= _last) else 0)
            self._move_law_last_dist = _dist
            if self._move_law_stall >= 3:
                self._move_law = None
                self._move_demo = None
                self._move_law_stall = 0
                self._move_law_last_dist = None
                self._move_law_fails = getattr(self, "_move_law_fails", 0) + 1
                if self._move_law_fails >= 2:
                    self._move_law_disabled = True
                print(f"[move-law] NOT CONVERGING (mover not approaching goal); dropped the "
                      f"learned step (unverified hypothesis)"
                      + (" -- DISABLED for the run; handing off to exploration/strategy"
                         if getattr(self, "_move_law_disabled", False)
                         else " -- will re-learn from a fresh demo"), flush=True)
                return None
            if _ML.reached(mover, goal, tol=5.0):
                return None                              # overlapping -> done
            r, c = _ML.next_click(law, mover, goal)
            self._move_last = (r, c)                     # the piece relocates to the click
            action = "CLICK:%d,%d" % (int(round(c)), int(round(r)))   # x=col, y=row
            from types import SimpleNamespace
            return SimpleNamespace(
                action=action,
                rationale=(f"click-to-move pursuit: step {law.step:.1f} from "
                           f"{tuple(round(x) for x in mover)} toward '{goal_name}' "
                           f"-> {action}"),
                plan_kind="move_law_pursuit", goal_id=f"reach:{goal_name}",
                target_cell=None, full_plan_actions=[action], is_probe=True)
        except Exception:
            return None

    def _mea_pursuit(self, ss, entities_for_actor, move_step):
        """MEA-AUTHORITATIVE exploit: route action selection through
        ``MeansEnds.next_step`` so the PLANNER decides each step (WALK toward a target,
        or PUSH an intermediary across a barrier) and this pursuit EXECUTES it -- the
        consolidation that makes MEA the single planner and the pursuits its executors
        (collapsing the advisory-MEA / acting-pursuit split).  The cardinal walk and
        the push share one cell_actor BFS.  Supersedes _cardinal_pursuit for the cases
        MEA resolves; returns None (fall back to the legacy pursuits) on any gap.

        Multiple goals are ordered by the STRANDING prior (a delivery that keeps the
        agent FREE -- a push, or a walk toward an intermediary -- before one that
        CONSUMES it in a slot).  Intermediaries are grounded in OBSERVED motion (an
        entity that moved but is not the controllable = a pushable hypothesis), so the
        macro-delivery fires only on evidence; the convergence watchdog refutes it if
        the push does not progress.  Fully guarded."""
        try:
            from cell_actor import bfs, _direction
            from types import SimpleNamespace
            import means_ends as _me
            cn = getattr(self, "_controllable_name", None)
            if not cn:
                return None
            mover_e = next((e for e in (entities_for_actor or [])
                            if e.get("name") == cn), None)
            mb = (mover_e or {}).get("bbox_ticks_turn1") or (mover_e or {}).get("bbox")
            if not (mb and len(mb) >= 4):
                return None
            ac = (int(round((mb[0] + mb[2]) / 2)), int(round((mb[1] + mb[3]) / 2)))
            # convergence watchdog (separate from _cardinal_pursuit's): if the mover
            # has not moved across consecutive MEA steps, the action isn't working ->
            # drop so it falls back instead of looping.
            if getattr(self, "_mea_prev_pos", None) == ac:
                self._mea_stuck = getattr(self, "_mea_stuck", 0) + 1
            else:
                self._mea_stuck = 0
            self._mea_prev_pos = ac
            if self._mea_stuck >= 2:
                self._mea_stuck = 0
                return None
            TARGETY = ("goal", "target", "slot", "box", "hole", "exit", "door",
                       "pad", "socket", "cup", "gate", "portal")

            def _tags(e):
                return ((e.get("role_hypothesis") or "") + " "
                        + (e.get("name") or "") + " "
                        + (e.get("appearance") or "")).lower()
            goals = []
            for e in (entities_for_actor or []):
                gb = e.get("bbox_ticks_turn1") or e.get("bbox")
                if gb and len(gb) >= 4 and any(w in _tags(e) for w in TARGETY):
                    gc = (round((gb[0] + gb[2]) / 2), round((gb[1] + gb[3]) / 2))
                    goals.append((abs(gc[0] - ac[0]) + abs(gc[1] - ac[1]), e))
            if not goals:
                return None
            goals.sort(key=lambda t: t[0])
            # intermediaries grounded in OBSERVED motion + the optimistic walkable
            # (every cell not EMPIRICALLY bounced off -- weak wall labels never block).
            moved = getattr(self, "_move_counts", {}) or {}
            objects = []
            for e in (entities_for_actor or []):
                nm = e.get("name")
                bb = e.get("bbox_ticks_turn1") or e.get("bbox")
                if nm != cn and bb and len(bb) >= 4 and moved.get(nm, 0) > 0:
                    objects.append({"name": nm,
                                    "bbox_ticks_turn1": [int(x) for x in bb],
                                    "affordances": {"pushable": True, "slides_far": True}})
            empirical = {tuple(c) for c in (ss.get("blocked_cells") or [])}
            optimistic = {(r, c) for r in range(64) for c in range(64)
                          if (r, c) not in empirical}
            step = max(1, int(move_step or 1))
            ctx = {"objects": objects, "walkable": optimistic, "move_step": step}
            mea = _me.MeansEnds()
            mea.register(_me.reachability_detector(optimistic, move_step=step))
            mea.register_expander("CHANGE_MODALITY", _me.expand_change_modality)
            mea.register_expander("VIA_INTERMEDIARY", _me.expand_via_intermediary)
            cur = {"bbox_ticks_turn1": [int(x) for x in mb], "name": cn}
            # build the full AND-OR WIN tree over all deliveries (stranding-ordered),
            # render it for the trace, and traverse to the first FEASIBLE, not-yet-
            # achieved leaf -- the tree-driven executor skips infeasible alternatives
            # (a direct walk through a barrier) and descends an AND op's first unmet
            # precondition.  This is MEA developing + showing the backward chain.
            deliveries = []
            for _dist, g in goals:
                gbx = g.get("bbox_ticks_turn1") or g.get("bbox")
                deliveries.append({"mover": cur,
                                   "goal": {"bbox_ticks_turn1": [int(x) for x in gbx],
                                            "name": g.get("name")}})
            root = mea.goal_tree_conjunctive(deliveries, ctx)
            try:
                self._mea_tree = "\n".join(_me.render_tree(root))
                print(f"[turn {getattr(self.world, 'turn', '?')}] MEA BACKWARD CHAIN:\n"
                      + self._mea_tree, flush=True)
            except Exception:
                self._mea_tree = None
            info, path = _me.first_action(root)
            if not info:
                return None
            chain = " <- ".join(path) if path else ""
            kind, direction = info.get("kind"), info.get("dir")
            if kind == "PUSH" and direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                return SimpleNamespace(
                    action=direction, plan_kind="mea_push",
                    rationale=f"MEA chain [{chain}] -> push '{info.get('via')}' {direction}",
                    goal_id=f"push:{info.get('via')}",
                    target_cell=info.get("target_cell"),
                    full_plan_actions=[direction], is_probe=True)
            if kind == "WALK" and info.get("target_cell"):
                tc = (int(info["target_cell"][0]), int(info["target_cell"][1]))
                p = bfs(ac, tc, optimistic, step=step, goal_tolerance=max(1, step // 2))
                if p and len(p) > 1:
                    dd = _direction(ac, p[1])
                    if dd in ("UP", "DOWN", "LEFT", "RIGHT"):
                        return SimpleNamespace(
                            action=dd, plan_kind="mea_walk",
                            rationale=f"MEA chain [{chain}] -> walk {dd} to {tc}",
                            goal_id=f"reach:{info.get('via') or 'goal'}", target_cell=tc,
                            full_plan_actions=[dd], is_probe=True)
            return None
        except Exception:
            return None

    def _cardinal_pursuit(self, ss, entities_for_actor, move_step):
        """Cardinal-grid EXPLOIT: BFS-walk the mover to the nearest reachable
        target-ish entity and emit ONE cardinal step.  The cardinal analogue of
        _move_law_pursuit (which is click-only): reuses cell_actor's BFS over the
        optimistic walkable set, so it solves the same explore->exploit gap for
        STEP/grid games that move_law solves for click games.  Returns an
        ActionChoice (UP/DOWN/LEFT/RIGHT) or None (no agent cell / no target /
        unreachable / first step not cardinal).  Fully guarded."""
        try:
            from cell_actor import bfs, _direction
            # Mover position: prefer the MOTION-tracked controllable entity (the
            # only source that works for a non-grid scene, where agent_cell is
            # None); else the grid agent cell; else the tracked agent's bbox.  BFS
            # runs in TICK space, so the cardinal DIRECTION is correct regardless
            # of cell granularity.
            ac = None
            cn = getattr(self, "_controllable_name", None)
            if cn:
                for e in (entities_for_actor or []):
                    if e.get("name") == cn:
                        bb = e.get("bbox_ticks_turn1") or e.get("bbox")
                        if bb and len(bb) >= 4:
                            ac = (round((bb[0] + bb[2]) / 2),
                                  round((bb[1] + bb[3]) / 2))
                        break
            if ac is None:
                ac = ss.get("agent_cell") if isinstance(ss, dict) else None
            if ac is None:
                try:
                    snap = self.world.symbolic_snapshot()
                    ag = snap.get("agent") if isinstance(snap, dict) else None
                    bb = ((ag or {}).get("current_bbox")
                          or (ag or {}).get("bbox"))
                    if bb and len(bb) >= 4:
                        ac = (round((bb[0] + bb[2]) / 2),
                              round((bb[1] + bb[3]) / 2))
                except Exception:
                    ac = None
            if ac is None:
                return None
            ac = (int(ac[0]), int(ac[1]))
            # CONVERGENCE WATCHDOG: if the tracked mover's position has NOT changed
            # across consecutive pursuit steps, the emitted cardinal action isn't
            # moving it (wrong controllable, or fully blocked) -> drop so it falls
            # back instead of re-emitting the same step forever.  Game-agnostic.
            if getattr(self, "_card_prev_pos", None) == ac:
                self._card_stuck = getattr(self, "_card_stuck", 0) + 1
            else:
                self._card_stuck = 0
            self._card_prev_pos = ac
            if self._card_stuck >= 2:
                self._card_stuck = 0
                return None
            TARGETY = ("goal", "target", "slot", "box", "hole", "exit", "door",
                       "pad", "socket", "cup", "gate", "portal")

            def _tags(e):
                return ((e.get("role_hypothesis") or "") + " "
                        + (e.get("name") or "") + " "
                        + (e.get("appearance") or "")).lower()
            cands = []
            for e in (entities_for_actor or []):
                if any(w in _tags(e) for w in TARGETY):
                    bb = e.get("bbox_ticks_turn1") or e.get("bbox")
                    if bb and len(bb) >= 4:
                        ctr = (round((bb[0] + bb[2]) / 2), round((bb[1] + bb[3]) / 2))
                        cands.append((abs(ctr[0] - ac[0]) + abs(ctr[1] - ac[1]),
                                      ctr, e.get("name")))
            if not cands:
                return None
            cands.sort()
            # optimistic walkable: every cell not EMPIRICALLY bounced off (matches
            # cell_actor's philosophy; avoids build_world_state, which requires a
            # grid agent_cell that a non-grid scene lacks).  Wall/scenery bbox
            # labels are weak hypotheses -- only cells the mover actually bounced
            # off are excluded, so a real path is never pre-emptively blocked.
            empirical = {tuple(c) for c in (ss.get("blocked_cells") or [])}
            optimistic = {(r, c) for r in range(64) for c in range(64)
                          if (r, c) not in empirical}
            step = max(1, int(move_step or 1))
            for _, tcell, tname in cands:
                p = bfs(ac, tcell, optimistic, step=step,
                        goal_tolerance=max(1, step // 2))
                if p and len(p) > 1:
                    direction = _direction(ac, p[1])
                    if direction in ("UP", "DOWN", "LEFT", "RIGHT"):
                        from types import SimpleNamespace
                        return SimpleNamespace(
                            action=direction, plan_kind="cardinal_pursuit",
                            rationale=(f"cardinal pursuit: BFS-walk mover to "
                                       f"'{tname}' ({len(p) - 1} ticks), first "
                                       f"step {direction}"),
                            goal_id=f"reach:{tname}", target_cell=tcell,
                            full_plan_actions=[direction], is_probe=True)
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    # SELF-CALIBRATING COMBINE / MERGE mechanism.
    #
    # The substrate owns the chaining/decomposition a small reasoner can't author
    # (means_ends); the VLM only supplies perception.  Everything it needs is
    # DERIVED or OBSERVED at run time -- no constant is tuned to a specific game,
    # so it runs unattended on an unseen game (competition mode):
    #   - it ACTIVATES only on a confirmed-by-observation merge (a same-appearance
    #     group's count drops after a walk), never on an assumed rule;
    #   - it never assumes a move STEP: it clicks just inside a piece's half of the
    #     gap so the piece (not the target) is grabbed, converging over turns;
    #   - delivery is SCORE-DRIVEN: it pushes the leftover at the goal and lets the
    #     game's score decide success -- no baked overlap threshold;
    #   - a structural watchdog disables it if nothing merges and the score never
    #     moves, so it is a safe no-op on a non-merge game.
    # ------------------------------------------------------------------
    @staticmethod
    def _mg_dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    @staticmethod
    def _mg_color_name(hexc):
        """Nearest common colour name for a hex -- so the operator menu reads
        'merge two cyan pieces' (which the VLM can match to the scene), not a hex."""
        s = str(hexc).lstrip("#")
        try:
            rgb = (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        except Exception:
            return str(hexc)
        named = {"red": (220, 40, 40), "green": (40, 200, 40), "blue": (30, 110, 230),
                 "cyan": (120, 210, 235), "magenta": (225, 60, 140),
                 "yellow": (240, 200, 40), "purple": (160, 90, 210),
                 "orange": (240, 150, 40), "white": (240, 240, 240),
                 "grey": (128, 128, 128), "black": (20, 20, 20)}
        return min(named, key=lambda n: sum((c - x) ** 2 for c, x in zip(rgb, named[n])))

    def _mg_group_by_appearance(self, blobs, tol=45.0):
        """Group blobs by appearance using the substrate's colour-match TOLERANCE
        (the same ~45 RGB distance locate_entity uses), NOT a fixed grid -- a grid
        splits two near-identical shades the game itself produces (a merge result vs
        the original), which strands a piece and breaks the chain.  Greedy clustering
        in deterministic colour order, so the cluster keys are stable across turns.

        Each colour cluster is then SPLIT by chain LEVEL, where level is read from the
        piece SIZE: two pieces only count as the same mergeable type if BOTH their
        colour matches AND their sizes are within ~1.5x.  A merge yields a markedly
        larger piece (each level renders at a distinct size -- 1,4,9,16 ... here, gaps
        >=1.78x), while perception noise within one level is <1.3x; 1.5x is the gap
        between those two scales, so it is structural, not a game constant.  Crucially
        this is robust to a COLOUR MISREAD: a level-16 piece the detector renders as a
        neighbouring hue still carries its true size, so it never groups with a real
        smaller-level piece of that hue and the two are never wrongly merged."""
        def rgb(h):
            s = str(h).lstrip("#")
            return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
        reps = []                                         # [(rgb, ckey)]
        colour_groups = {}
        for b in sorted(blobs, key=lambda b: b["color"]):
            c = rgb(b["color"]); ckey = None
            for rr, kk in reps:
                if sum((x - y) ** 2 for x, y in zip(c, rr)) ** 0.5 <= tol:
                    ckey = kk; break
            if ckey is None:
                ckey = tuple(v // 16 for v in c)
                reps.append((c, ckey))
            colour_groups.setdefault(ckey, []).append(b)
        # split each colour cluster by size LEVEL (smallest-first -> deterministic keys)
        groups = {}
        for ckey, members in colour_groups.items():
            levels = []                                   # [(ref_size, lkey)]
            for b in sorted(members, key=lambda b: b["cells"]):
                lkey = None
                for ref, kk in levels:
                    if max(b["cells"], ref) / max(1, min(b["cells"], ref)) < 1.5:
                        lkey = kk; break
                if lkey is None:
                    lkey = (ckey, b["cells"]); levels.append((b["cells"], lkey))
                groups.setdefault(lkey, []).append(b)
        return groups

    @staticmethod
    def _mg_collinear(pts):
        """True if the points lie ~on a line (an aim line / row of dashes, not a
        scatter of merge pieces).  Structural shape test (minor vs major spread),
        no game-specific constant -- guards the merge mechanism against firing on a
        trajectory/legend strip whose marks share one appearance."""
        n = len(pts)
        if n < 3:
            return False
        mr = sum(p[0] for p in pts) / n
        mc = sum(p[1] for p in pts) / n
        srr = scc = src = 0.0
        for r, c in pts:
            dr, dc = r - mr, c - mc
            srr += dr * dr; scc += dc * dc; src += dr * dc
        srr /= n; scc /= n; src /= n
        tr = srr + scc
        disc = max(0.0, (tr * tr / 4.0) - (srr * scc - src * src)) ** 0.5
        major = tr / 2.0 + disc
        minor = tr / 2.0 - disc
        return major > 1e-6 and (minor / major) < 0.06   # minor spread tiny -> a line

    def _mg_blobs(self):
        """Substrate-measured foreground blobs in the playfield band (drops the top
        legend/HUD and the bottom status bar): {color, center, cells, radius}."""
        try:
            fp = getattr(self, "_last_frame_path", None)
            if not fp or not Path(fp).exists():
                return []
            from substrate_tools.registry import run_queries
            band = [int(round(0.17 * self.n_ticks)), 0,
                    int(round(0.91 * self.n_ticks)), self.n_ticks]   # exclude HUD margins
            r = run_queries(str(fp), [{"op": "components", "id": "m", "bbox": band,
                                       "min_cells": 1, "max_return": 48}],
                            str(Path(self.work_dir) / "_merge_vq"), n_ticks=self.n_ticks)
            out = []
            for c in (r[0].get("components") or []):
                r0, c0, r1, c1 = c["bbox"]
                out.append({"color": c["color"], "cells": c["cells"],
                            "center": (c["centroid"][0], c["centroid"][1]),
                            "radius": max(r1 - r0, c1 - c0) / 2.0})
            return out
        except Exception:
            return []

    def _mg_safe_click(self, mover, target, static_target=False):
        """A click toward the target.  The control law (read off the merge trace) is:
        a click moves the NEAREST MOVABLE piece TO the click location -- but ONLY if the
        click lands within the GRAB RANGE of a movable piece; a click on empty space or
        on a static goal grabs nothing and is a no-op.  So every click is computed
        ENTITY-RELATIVE: anchor on the mover, step toward the target by an offset bounded
        to the grab range (never an absolute far point).  The mover is re-resolved each
        turn, so the click can never go stale.

        MERGE: both endpoints are movable, so the offset also stays nearer the mover than
        the OTHER piece (the d/2 cap) or the wrong piece moves; reach adapts on no-progress.

        DELIVERY (static_target): the target is a large STATIC goal -- not movable -- so
        the mover stays the nearest movable piece the whole way (no d/2 cap).  But the
        click must STILL sit within grab range of the PIECE: clicking ON the goal grabs
        nothing (the (9,50)/(23,50) no-ops).  So step the piece toward the goal by the
        grab range -- a near piece (d<=reach) lands on the goal in one click, a far one
        walks over in a few -- using the base reach so a merge-phase shrink can't strand
        it."""
        d = self._mg_dist(mover, target)
        if d < 1e-6:
            return mover
        if static_target:
            off = max(1.0, min(self.n_ticks * 0.1, d))    # toward goal, bounded to grab range
        else:
            reach = getattr(self, "_mg_reach", None) or (self.n_ticks * 0.1)
            off = max(1.0, min(reach, d / 2.0 - 0.5))     # within reach AND nearer the mover
        ur, uc = (target[0] - mover[0]) / d, (target[1] - mover[1]) / d
        r = max(1.0, min(self.n_ticks - 2.0, mover[0] + ur * off))
        c = max(1.0, min(self.n_ticks - 2.0, mover[1] + uc * off))
        return (r, c)

    def _mg_reset(self):
        for a in ("_mg_active", "_mg_prev_counts", "_mg_prev_score",
                  "_mg_confirmed", "_mg_stuck", "_mg_best", "_mg_reach",
                  "_mg_n_goals", "_mg_deliver_assign",
                  "_mg_blacklist", "_mg_last_action", "_mg_action_stall"):
            setattr(self, a, None)

    def _mg_goals_by_size(self, blobs):
        """Goal(s) = the blob(s) markedly larger than the merge pieces, found by the
        largest RATIO break in the sorted sizes (structural, no fixed threshold: the
        pieces are small + many, a delivery goal is large).  Handles ONE goal (lc1)
        or SEVERAL (lc2's two balls).  Returns the goal blobs (possibly empty)."""
        if len(blobs) < 2:
            return []
        sz = sorted(blobs, key=lambda b: -b["cells"])
        cut, best = -1, 1.8                               # need a clear >1.8x size jump
        for i in range(len(sz) - 1):
            if sz[i]["cells"] / max(1, sz[i + 1]["cells"]) > best:
                best, cut = sz[i]["cells"] / max(1, sz[i + 1]["cells"]), i
        if cut < 0 or cut + 1 > max(1, len(blobs) // 3):  # no break, or 'goals' are the bulk
            return []
        return sz[:cut + 1]

    def _mg_assign(self, pieces, goals):
        """Assign each finished piece to a DISTINCT goal and COMMIT it, keyed by the
        piece's SIZE (its chain level).  Size is STABLE even when the colour read flips
        between neighbouring hues; a colour key is not -- and a flipped key sends both
        finished pieces to one goal, where the first one delivered then BLOCKS the
        second's approach (measured on lc2: the purple landed in the left ball and the
        yellow, mis-routed to that same full ball, froze a step short while the right
        ball sat empty).  Committing distinct goals up front, by a stable key, keeps the
        two pieces on two paths to two different balls.  Returns {piece_index: goal}."""
        cache = getattr(self, "_mg_deliver_assign", None)
        if cache is None:
            cache = []; self._mg_deliver_assign = cache       # [(size, goal)]
        out = {}
        # EVERY already-committed goal stays claimed -- even for a piece that has since
        # been DELIVERED and is no longer on the board.  The two finished pieces here
        # are delivered at DIFFERENT times (the first finishes and is delivered before
        # the second is even merged), so without this a later piece re-picks the goal
        # the first one already filled, and a full ball blocks it (the lc2 failure).
        claimed = set(tuple(g) for _, g in cache)
        # honour existing commitments first, matched by SIZE (same chain level, <1.5x)
        for i, p in enumerate(pieces):
            for sz, g in cache:
                if max(p["cells"], sz) / max(1, min(p["cells"], sz)) < 1.5:
                    out[i] = g; claimed.add(tuple(g)); break
        # assign the rest to the nearest still-unclaimed goal, closest piece first, COMMIT
        rest = sorted((i for i in range(len(pieces)) if i not in out),
                      key=lambda i: min(self._mg_dist(pieces[i]["center"], g) for g in goals))
        for i in rest:
            avail = [g for g in goals if tuple(g) not in claimed] or list(goals)
            g = min(avail, key=lambda gg: self._mg_dist(pieces[i]["center"], gg))
            out[i] = g; claimed.add(tuple(g))
            cache.append((pieces[i]["cells"], g))
        return out

    def _merge_pursuit(self, entities_for_actor):
        try:
            score = int(getattr(self.world, "score", 0) or 0)
            # a score advance = the level was won / changed -> re-evaluate fresh
            if getattr(self, "_mg_prev_score", None) is not None and score != self._mg_prev_score:
                self._mg_reset()
            blobs = self._mg_blobs()
            if not blobs:
                return None
            # GOAL(S) by size (1 or several); the merge PIECES are everything else.
            # The size-ratio detector needs MANY pieces for a clean break; as pieces
            # merge away it eventually sees only goals+a few pieces and mis-rejects them
            # (a static goal does not stop being a goal just because the board emptied).
            # Goals are STATIC + always the largest blobs, so LOCK their count on first
            # clean detection and thereafter take the N largest -- robust through the
            # endgame, where delivery (and thus winning) actually happens.
            goal_blobs = self._mg_goals_by_size(blobs)
            if goal_blobs:
                self._mg_n_goals = len(goal_blobs)
            elif getattr(self, "_mg_n_goals", None):
                goal_blobs = sorted(blobs, key=lambda b: -b["cells"])[:self._mg_n_goals]
            if not goal_blobs:
                return None
            goals = [g["center"] for g in goal_blobs]
            gset = {id(g) for g in goal_blobs}
            non_goal = [b for b in blobs if id(b) not in gset]
            groups = self._mg_group_by_appearance(non_goal)
            counts = {k: len(v) for k, v in groups.items()}

            # ACTIVATION: a repeated, non-collinear appearance among the pieces (an
            # aim line / legend strip shares an appearance but is collinear -> excluded)
            if not getattr(self, "_mg_active", False):
                reps = [v for v in groups.values() if len(v) >= 2
                        and not self._mg_collinear([b["center"] for b in v])]
                if not reps:
                    return None
                self._mg_active = True
                self._mg_confirmed = False
                self._mg_stuck = 0
                self._mg_best = None
                self._mg_prev_counts = dict(counts)
                self._mg_prev_score = score
                print(f"[merge] candidate: {len(goals)} goal(s); probing whether "
                      f"same-appearance pieces merge (unattended)", flush=True)

            # OBSERVED MERGE: a piece-appearance count fell since last turn
            prev = getattr(self, "_mg_prev_counts", {}) or {}
            if any(counts.get(k, 0) < prev.get(k, 0) for k in prev) and not self._mg_confirmed:
                self._mg_confirmed = True
                print("[merge] CONFIRMED by observation: a same-appearance pair merged "
                      "-> chaining the rest", flush=True)
            metric = sum(counts.values()) * 1000.0
            self._mg_prev_counts = dict(counts)
            self._mg_prev_score = score

            # OPERATOR MENU (the VLM SELECTS; the substrate measures).  Merge ops while
            # same-appearance pairs exist; DELIVER ops appear only once no pair remains
            # (the chain is finished -- so a mid-chain piece is never offered for
            # delivery), each routed to a DISTINCT goal.
            mergeable = {k: v for k, v in groups.items() if len(v) >= 2}
            pieces = [b for v in groups.values() for b in v]
            ops = []
            for k, v in mergeable.items():
                pts = [b["center"] for b in v]
                pa = pb = None; pd = 1e18
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        dd = self._mg_dist(pts[i], pts[j])
                        if dd < pd:
                            pa, pb, pd = pts[i], pts[j], dd
                nm = self._mg_color_name(v[0]["color"]); oc = self._mg_safe_click(pa, pb)
                # expose the COST (closest-pair gap) so a "merge close pairs first to
                # save moves" lesson, learned from a prior budget loss, is actionable.
                ops.append((f"merge two {nm} pieces ({len(v)} {nm} on the board; "
                            f"closest pair ~{int(round(pd))} ticks apart)",
                            "CLICK:%d,%d" % (int(round(oc[1])), int(round(oc[0])))))
            # A piece with NO same-appearance partner cannot merge further this turn, so
            # delivering it is a legal move -- offer it ALONGSIDE the merge ops (operator
            # completeness) instead of withholding ALL delivery until every colour is a
            # singleton.  Otherwise a finished piece (e.g. a merged yellow) is stranded
            # while another colour is still mid-merge and nothing ever reaches a goal.  The
            # VLM decides whether the piece is actually finished (matches a HUD target).
            finished = [v[0] for k, v in groups.items() if len(v) == 1]
            if finished:
                # Route each finished piece to a DISTINCT goal via _mg_assign, which
                # claims a goal persistently (even after its piece is delivered) so a
                # later piece avoids the already-filled ball.  Pass ALL goals: an earlier
                # version handed it only the len(finished) SMALLEST goals as a filled-goal
                # guess, but that collapses to ONE goal when a single piece is left and
                # then FORCES it onto the very ball the first piece already filled (the
                # lc2 last-delivery failure).  The claim, not a size guess, picks the
                # empty ball.
                assign = self._mg_assign(finished, [g["center"] for g in goal_blobs])
                # Deliver ONE piece at a time, NEAREST-to-its-goal first: it completes
                # fastest and stays nearest as it approaches, so the substrate keeps
                # driving the SAME piece until it lands, then the next.  Only one piece
                # moves at a time, so two finished pieces never travel together and
                # collide into one blob (the lc2 yellow+purple-onto-one-goal failure).
                for i in sorted(range(len(finished)),
                                key=lambda j: self._mg_dist(finished[j]["center"], assign[j])):
                    p = finished[i]; nm = self._mg_color_name(p["color"])
                    oc = self._mg_safe_click(p["center"], assign[i], static_target=True)
                    ops.append((f"deliver the finished {nm} piece onto its goal (no other "
                                f"{nm} piece left to merge it with)",
                                "CLICK:%d,%d" % (int(round(oc[1])), int(round(oc[0])))))
            if not ops:
                return None

            # metric distance component (for the no-progress watchdog) + recommendation
            if mergeable:
                pts = [b["center"] for v in mergeable.values() for b in v]
                metric += min(self._mg_dist(pts[i], pts[j])
                              for i in range(len(pts)) for j in range(i + 1, len(pts)))
            elif pieces:
                metric += min(self._mg_dist(p["center"], g) for p in pieces for g in goals)
            action = ops[0][1]

            # WATCHDOG + ADAPTIVE REACH: no progress this turn -> shrink the reach so
            # the next click lands within the grab range (self-calibrates; no step).
            best = getattr(self, "_mg_best", None)
            if best is None or metric < best - 0.5:
                self._mg_best = metric
                self._mg_stuck = 0
            else:
                self._mg_stuck = (getattr(self, "_mg_stuck", 0) or 0) + 1
                # shrink toward the grab range, but FLOOR it: a click too CLOSE to a
                # piece is a no-op too (no room to move), so keep a minimum useful step.
                self._mg_reach = max(self.n_ticks * 0.05, (getattr(self, "_mg_reach", None)
                                                           or self.n_ticks * 0.1) * 0.7)
            if self._mg_stuck > self.n_ticks // 2:
                print("[merge] no progress for a long run -> disabling (not a merge game)", flush=True)
                self._mg_reset()
                return None

            # --- ABANDON A STALLED TARGET (bug tn36) -------------------------------------
            # The substrate kept RE-RECOMMENDING the same click while the actor wasn't
            # moving (CLICK:23,12 on 59/60 turns).  Consume the stall signal: emit the best
            # operator whose click is NOT blacklisted; count consecutive stalls on the
            # emitted click; after a few, BLACKLIST that click and re-target to the next
            # operator; if EVERY operator is blacklisted, disable (the menu can't progress
            # here, so hand off to exploration instead of looping a dead click).
            bl = self._mg_blacklist = getattr(self, "_mg_blacklist", None) or set()
            action = next((a for _, a in ops if a not in bl), None)
            if action is None:
                print("[merge] every operator stalled (blacklisted) -> disabling", flush=True)
                self._mg_reset()
                return None
            if self._mg_stuck and action == getattr(self, "_mg_last_action", None):
                self._mg_action_stall = (getattr(self, "_mg_action_stall", 0) or 0) + 1
            else:
                self._mg_action_stall = 0
            self._mg_last_action = action
            if self._mg_action_stall >= 3:
                bl.add(action)
                self._mg_action_stall = 0
                print(f"[merge] target {action} stalled 3x -> BLACKLISTED; re-targeting", flush=True)
                action = next((a for _, a in ops if a not in bl), None)
                if action is None:
                    print("[merge] every operator stalled -> disabling", flush=True)
                    self._mg_reset()
                    return None
                self._mg_last_action = action

            self._merge_menu_actions = [a for _, a in ops]
            self._merge_menu_text = (
                "MERGE OPERATOR MENU — the substrate measured these candidate operators "
                "(each is ONE click).  Look at the scene and PICK the single best one for "
                "THIS turn; reply with its exact CLICK as endorsed_action.  Rule: merge "
                "same-coloured pieces up the legend chain FIRST; deliver a finished piece "
                "onto a goal blob only when no same-coloured partner is left for it.\n"
                + "\n".join(f"  - {d}  ->  {a}" for d, a in ops)
                + f"\n  Substrate recommendation: {action}")

            from types import SimpleNamespace
            # SMALL-MODEL COMPENSATION: normally the actor PICKS from the menu (is_probe
            # =False -> strategy call).  But if the merge has not progressed for a couple
            # of turns, a weak actor is overriding the grounded recommendation with an
            # unproductive click (observed: it repeated one dead click while the magenta
            # waited to be merged).  Assert the substrate's grounded move DIRECTLY
            # (is_probe=True bypasses the strategy call) to break the derail; control
            # returns to the actor the moment a merge lands (stuck resets to 0).
            # Drive directly (is_probe bypasses the actor) when the actor STALLS, or
            # whenever we are in the PURE-DELIVERY phase (no same-type pair left to
            # merge).  Routing N distinct finished objects to N distinct destinations is
            # deterministic; letting a weak actor micro-manage it sends two objects to
            # one destination and collides them.  ops[0] is the nearest piece's delivery,
            # so the substrate lands one piece, then the next -- one mover at a time.
            pure_delivery = (not mergeable) and bool(finished)
            stalled = (getattr(self, "_mg_stuck", 0) or 0) >= 2
            assert_move = pure_delivery or stalled
            return SimpleNamespace(
                action=action, plan_kind="merge_pursuit", goal_id="merge_and_deliver",
                rationale=(f"merge-menu ({len(ops)} ops) -> {action}"
                           + (" [substrate: sequential delivery]" if pure_delivery
                              else " [substrate-asserted: actor stalled]" if stalled else "")),
                target_cell=None, full_plan_actions=[action], is_probe=assert_move)
        except Exception:
            return None

    def _exploratory_probe_choice(self, entities_for_actor):
        """Game-agnostic CURIOSITY PROBE, used when grid planning yields no
        action (a non-grid / continuous-arena game, or a stalled planner).

        Systematically covers the action space so COS can DISCOVER what each
        action does on a game its grid actor cannot drive:

          1. each coordinate-free action once (ACTION_N that is not the click);
          2. then a CLICK on each perceived entity once (click the things we
             can see — the most informative coordinate probes).

        Coverage is tracked in ``world.probe_state['explore_coverage']`` so the
        probe advances one new (action | entity-click) per turn and STOPS
        (returns None) once everything has been tried — at which point the
        run ends unless a grid plan has since become available.  The effect of
        each probe is recorded by the normal delta-perception pass, exactly as
        for any other action.  Returns a choice-like object with
        ``is_probe=True`` (so the strategy call is bypassed and the probe runs
        autonomously) or None when coverage is exhausted.
        """
        from types import SimpleNamespace
        available = [a for a in self.game.available_actions() if a != "NONE"]
        click_supported = "CLICK" in available
        cov = self.world.probe_state.setdefault("explore_coverage", [])

        # Ordered probe targets: coordinate-free raw actions first (the bare
        # click alias is excluded — entity clicks below cover clicking better),
        # then a click on each perceived entity that has a bbox.
        targets: list[tuple[str, str]] = []
        for a in available:
            # Skip the convenience cardinal aliases (UP/DOWN/LEFT/RIGHT/CLICK):
            # available_actions() advertises a raw ACTION_N name for EVERY
            # supported action_id, so the aliases map to ids already probed via
            # their ACTION_N name -- probing them too just burns duplicate turns.
            if a in _CARDINAL_ALIASES:
                continue
            targets.append(("act", a))
        if click_supported:
            for e in entities_for_actor:
                nm = e.get("name")
                if nm and e.get("bbox_ticks_turn1"):
                    targets.append(("click", nm))
            # ALSO probe clicking NEAR each element (a few ticks toward the field
            # centre), not only ON it.  Many effects -- click-to-move, aim, select,
            # merge -- trigger only on a click ADJACENT to a thing, so 'clicking ON
            # it did nothing' is NOT evidence the element is inert.  This is the
            # readily-visible clue that, un-probed, made COS wrongly conclude
            # su15 lc1 was unwinnable.  Exhaust the visible affordances before
            # surrendering.
            for e in entities_for_actor:
                nm = e.get("name")
                if nm and e.get("bbox_ticks_turn1"):
                    targets.append(("click_near", nm))

        for kind, ref in targets:
            key = f"{kind}:{ref}"
            if key in cov:
                continue
            cov.append(key)
            if kind == "act":
                action = ref
                desc = f"curiosity probe: try {ref} to observe its effect"
            elif kind == "click":
                action = self._resolve_action(f"CLICK:{ref}")
                desc = f"curiosity probe: click entity {ref} to observe its effect"
            else:                                    # click_near
                e = next((x for x in entities_for_actor
                          if x.get("name") == ref), None)
                bb = e.get("bbox_ticks_turn1") if e else None
                if not bb:
                    continue
                r = (bb[0] + bb[2]) / 2.0; c = (bb[1] + bb[3]) / 2.0
                dr, dc = 37.0 - r, 32.0 - c          # toward the playfield centre
                L = _math.hypot(dr, dc) or 1.0
                nr = max(11.0, min(62.0, r + 5.0 * dr / L))
                nc = max(1.0, min(62.0, c + 5.0 * dc / L))
                action = "CLICK:%d,%d" % (int(round(nc)), int(round(nr)))
                desc = (f"curiosity probe: click NEAR entity {ref} (~5 ticks "
                        f"toward centre) -- tests click-to-move / aim / select "
                        f"effects that only fire ADJACENT to a thing")
            return SimpleNamespace(
                action=action, rationale=desc,
                plan_kind="curiosity_probe", goal_id="curiosity_probe",
                target_cell=None, full_plan_actions=[action], is_probe=True,
            )
        # Single-pass coverage is exhausted — but getting stuck is NOT an option
        # in strict/competition mode (no user to nudge COS, hundreds of games
        # non-stop).  Escalate instead of returning None, so the run ends ONLY
        # on win/loss/turn-budget, never on "I have run out of ideas".
        return self._escalated_explore(available, entities_for_actor)

    def _escalated_explore(self, available, entities_for_actor):
        """Never-halt exploration, reached when the single-pass curiosity probe
        has tried every action + entity-click once.  Returns a probe-style
        choice and NEVER returns None while the game advertises any action.
        Escalating, game-agnostic tiers (each chosen action's effect is recorded
        by the normal delta pass, so COS keeps LEARNING the whole time):

          T2  re-try each action / entity-click ONCE PER DISTINCT perceived
              STATE — an action that did nothing in one state may act in another
              (effects are state-dependent); keyed on a coarse state signature so
              this is genuine re-exploration, not blind repetition of a no-op.
          T3  a CLICK grid across the playfield at the perceived cell pitch —
              covers click targets that are not entity centroids.
          T4  diversify to reach a NEW state (which reopens T2/T3): UNDO
              (ACTION7) right after a no-op to back out of a dead branch, else
              round-robin the action set.

        The run loop is turn-bounded, so this consumes the remaining budget
        exploring rather than quitting early — the correct behaviour when no
        confident plan exists.
        """
        from types import SimpleNamespace
        if not available:
            return None
        ps = self.world.probe_state
        acts = [a for a in available if a not in _CARDINAL_ALIASES]
        click_ok = "CLICK" in available
        clickable = [e["name"] for e in entities_for_actor
                     if e.get("name") and e.get("bbox_ticks_turn1")]

        def mk(action, desc, kind):
            return SimpleNamespace(
                action=action, rationale=desc, plan_kind=f"explore:{kind}",
                goal_id="curiosity_probe", target_cell=None,
                full_plan_actions=[action], is_probe=True)

        # T2 — per-state re-exploration (state-dependent effects)
        sig = self._state_signature()
        seen = ps.setdefault("state_coverage", {}).setdefault(sig, [])
        for a in acts:
            if f"act:{a}" not in seen:
                seen.append(f"act:{a}")
                return mk(a, f"re-try {a} in this state (effects are "
                          f"state-dependent)", "restate")
        for nm in clickable:
            if f"click:{nm}" not in seen:
                seen.append(f"click:{nm}")
                return mk(self._resolve_action(f"CLICK:{nm}"),
                          f"re-click {nm} in this state", "restate")

        # T3 — coordinate-grid CLICK sweep at the perceived cell pitch
        if click_ok:
            gi = self.world.grid_inference
            pitch = gi.cell_ticks if gi and getattr(gi, "cell_ticks", None) else 8
            pitch = max(2, int(pitch))
            swept = ps.setdefault("coord_sweep", [])
            off = pitch // 2
            for r in range(off, self.n_ticks, pitch):
                for c in range(off, self.n_ticks, pitch):
                    if f"{r},{c}" not in swept:
                        swept.append(f"{r},{c}")
                        return mk(f"CLICK:{c},{r}",
                                  f"coordinate-sweep click ({c},{r})", "sweep")

        # T4 — diversify to reach a new state; NEVER halt while playable
        i = ps.get("diversify_idx", 0)
        ps["diversify_idx"] = i + 1
        if self._last_action_was_noop() and "ACTION7" in available:
            return mk("ACTION7", "diversify: UNDO to back out of a dead branch",
                      "undo")
        a = acts[i % len(acts)] if acts else available[0]
        return mk(a, f"diversify: cycle {a} to reach a new state", "cycle")

    def _state_signature(self) -> str:
        """Coarse, cheap signature of the current situation, used by the never-
        halt explorer to re-try actions once PER DISTINCT state instead of
        looping on an identical no-op.  Keyed on the tracked entities'
        positions + roles (semantic, stable to render jitter); falls back to a
        downsampled-frame hash, then a constant."""
        try:
            items = tuple(sorted(
                (r.name, tuple(r.current_bbox or ()), r.current_role or "")
                for r in self.world.entities.values()
                if r.current_bbox is not None))
            if items:
                return str(hash(items) & 0xFFFFFFFF)
        except Exception:
            pass
        try:
            import numpy as np
            from PIL import Image
            im = Image.open(self._last_frame_path).convert("RGB").resize(
                (16, 16), Image.NEAREST)
            return str(hash(np.asarray(im).tobytes()) & 0xFFFFFFFF)
        except Exception:
            return "0"

    def _last_action_was_noop(self) -> bool:
        """True when the most recent step produced no observable change — the
        signal to diversify (UNDO / change tack) rather than keep poking a dead
        branch."""
        d = self.world.deltas_observed or []
        if not d:
            return False
        last = d[-1]
        return (not getattr(last, "agent_moved", False)
                and not getattr(last, "entities_changed", None)
                and not getattr(last, "entities_appeared", None)
                and not getattr(last, "entities_disappeared", None)
                and not getattr(last, "score_increased", False))

    # Click coordinates are consumed by the env in the SAME coordinate
    # frame as the perception grid (tick space), so a CLICK:<entity_name>
    # resolves to the entity's bbox centroid with NO rescaling.  An earlier
    # value of 8 assumed a 512x512 native pixel space (8 = 512/64); that
    # assumption is wrong for these envs — scale 8 sent every entity-click
    # out of bounds, where it SILENTLY no-ops, which hid the fact that some
    # "scenery" entities are actually selectable/controllable by a click
    # (e.g. sp80's paddles).  That made affordance discovery report inert
    # entities and wrongly conclude they are fixed.  If a future env ever
    # uses a different native click resolution, that rescaling belongs in
    # the ADAPTER (which knows the env), not here — the driver works in the
    # perception coordinate frame.  (Verified on sp80: only scale 1 selects.)
    CLICK_PIXEL_SCALE = 1

    def _resolve_action(self, action: str) -> str:
        """Translate VLM strategy-layer action strings into the
        form the GameAdapter expects.  Specifically: turn
        ``CLICK:<entity_name>`` into ``CLICK:px,py`` by looking up
        the named entity's current bbox in the world model and
        taking its centroid, scaled from tick coords to the
        game's native pixel coords (by ``CLICK_PIXEL_SCALE``).  If
        the named entity doesn't exist or has no bbox, fall back
        to bare CLICK (adapter uses playfield center)."""
        if not action.startswith("CLICK:"):
            return action
        payload = action[len("CLICK:"):]
        # Already in px,py form? Pass through.
        if "," in payload and payload.replace(",", "").replace(
            " ", "").replace("-", "").isdigit():
            return action
        # DECLARED random-floor click (explicit exploratory intent): resolve to a cell
        # NOT covered by any structural foreground entity -- never modal-colour keyed,
        # never snapped to an entity.  Lets a probe (or the VLM) say "click empty floor"
        # so the outcome is a clean exploratory click, not a silent centre no-op.
        if payload.strip().upper() == "FLOOR":
            return self._resolve_floor_click()
        # SCENE-STATE id (incl. sub-cells like 'panel/r0c0') -> measured click.
        # The substrate owns the coordinate; the VLM only names the id.
        _sc = getattr(self, "_scene", None)
        if _sc is not None and payload.strip() in getattr(_sc, "entities", {}):
            _e = _sc.entities[payload.strip()]
            if _e.click:
                return (f"CLICK:{int(_e.click[0]) * self.CLICK_PIXEL_SCALE},"
                        f"{int(_e.click[1]) * self.CLICK_PIXEL_SCALE}")
        # Treat as entity name
        entity_name = payload.strip()
        rec = self.world.entities.get(entity_name)
        if rec is None or rec.current_bbox is None:
            print(f"  [driver] CLICK target {entity_name!r} not found "
                  f"or has no bbox; using a declared FLOOR click")
            return self._resolve_floor_click()
        bb = rec.current_bbox
        # bbox is in tick coords; scale to game's native pixel space
        # before sending to the adapter (which forwards px,py to the
        # game's action_id=6 data payload).
        cx = ((bb[1] + bb[3]) // 2) * self.CLICK_PIXEL_SCALE
        cy = ((bb[0] + bb[2]) // 2) * self.CLICK_PIXEL_SCALE
        return f"CLICK:{cx},{cy}"

    def _resolve_floor_click(self) -> str:
        """Resolve a DECLARED random-floor click -> a cell inside the playfield extent
        but NOT covered by any known entity's footprint (the structural foreground
        components, via their grounded bboxes).  Game-agnostic and palette-invariant:
        the floor is 'everywhere no entity is', never a modal background colour.  The
        cell varies per call (deterministically, no RNG -- replay-safe) so successive
        floor probes explore different empty locations instead of re-clicking one spot.
        Falls back to bare centre CLICK only if no entity footprints are known yet."""
        occ: set[tuple[int, int]] = set()
        rs: list[int] = []
        cs: list[int] = []
        for rec in self.world.entities.values():
            bb = getattr(rec, "current_bbox", None)
            if not bb:
                continue
            r0, c0, r1, c1 = int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])
            rs += [r0, r1]; cs += [c0, c1]
            for r in range(r0, r1 + 1):
                for c in range(c0, c1 + 1):
                    occ.add((r, c))
        if not rs:
            return "CLICK"               # no footprints yet -> adapter centre default
        free = [(r, c)
                for r in range(min(rs), max(rs) + 1)
                for c in range(min(cs), max(cs) + 1)
                if (r, c) not in occ]
        if not free:
            return "CLICK"
        n = getattr(self, "_floor_probe_n", 0)
        self._floor_probe_n = n + 1
        r, c = free[(n * 7919) % len(free)]   # 7919 (prime) spreads successive picks out
        return f"CLICK:{c * self.CLICK_PIXEL_SCALE},{r * self.CLICK_PIXEL_SCALE}"

    # ------------------------------------------------------------------
    # Subroutine KB hooks — credence updates + auto-promotion
    # ------------------------------------------------------------------

    def _update_subroutine_credence(
        self,
        *,
        applied_subroutine: Optional[str],
        fork_parent: Optional[str],
        variant_notes: Optional[str],
        outcome: str,
        action_record: ActionRecord,
    ) -> None:
        """When the strategy actor recorded an applied_subroutine on
        this turn's reply, bump the stored subroutine's credence
        based on whether the action visibly advanced progress.
        Also fork a variant record when the actor supplied
        ``fork_parent`` + ``variant_notes``.

        ``outcome`` is the heuristic substrate-computed verdict —
        the strategy actor can override on later turns if it has
        more context.
        """
        if not applied_subroutine and not fork_parent:
            return
        try:
            game_id = self.world.game_id
            level   = self.world.level
            turn    = action_record.turn
            turn_range = [turn, turn]
            orig_goal = action_record.goal_id or "(no goal id)"

            if applied_subroutine:
                _subroutine_record_outcome(
                    subroutine_id=applied_subroutine,
                    outcome=outcome,
                    game_id=game_id, level=level,
                    turn_range=turn_range,
                    original_goal=orig_goal,
                    notes=action_record.rationale[:240],
                )
            if fork_parent and variant_notes:
                # Fork only on successful outcomes — failed variants
                # don't deserve their own record.  The strategy
                # actor may reuse the same fork_parent later if it
                # wants the variant promoted then.
                if outcome in ("success", "partial"):
                    _subroutine_promote(
                        name=variant_notes[:60],
                        description=variant_notes,
                        problem_solved=("forked variant — see "
                                         "parent for original "
                                         "problem context"),
                        concrete_chain=list(
                            getattr(action_record, "full_plan_actions",
                                     None) or [action_record.action]
                        ),
                        expected_outcome=("see parent; this variant "
                                           "claims to behave per "
                                           "variant_notes"),
                        game_id=game_id, level=level,
                        turn_range=turn_range,
                        original_goal=orig_goal,
                        parent_id=fork_parent,
                        variant_notes=variant_notes,
                    )
        except Exception as e:
            # Subroutine-KB writes are bookkeeping; never let them
            # break the action loop.
            print(f"[driver] subroutine credence update failed "
                  f"({e}); continuing")

    def _prune_winning_chain(self, actions) -> list:
        """Produce a shorter candidate from a winning action window
        by dropping signal-free oscillation pairs.

        Heuristic, game-agnostic: a step "contributes" if its delta
        carried an acceptance-relevant signal (visual_event, score
        increase, entity appeared/disappeared, or any entity change).
        A pair of ADJACENT steps that (a) both carried no such signal
        and (b) are mutual inverses (extend/retract, up/down, or an
        ACTION7 undo) nets to no progress and is removed.  The result
        is a PLAUSIBLE shorter chain; correctness is verified by the
        actor trying it in a future trial (the KB keeps it only if it
        wins).  Never drops a signal-bearing step.
        """
        # Map action -> its delta (by to_turn).
        delta_by_turn = {
            d.to_turn: d for d in self.world.deltas_observed
        }

        def _signal(a) -> bool:
            d = delta_by_turn.get(a.turn + 1) or delta_by_turn.get(a.turn)
            if d is None:
                return False
            if getattr(d, "score_increased", False):
                return True
            if getattr(d, "win_state_changed", False):
                return True
            if getattr(d, "visual_events", None):
                return True
            if (getattr(d, "entities_appeared", None)
                    or getattr(d, "entities_disappeared", None)):
                return True
            return False

        # Known inverse pairs (game-agnostic action semantics; the
        # specific ids are this game's, learned at runtime — but the
        # pairing is structural: extend<->retract, up<->down, and
        # ACTION7=undo reverses whatever preceded it).
        inverses = {
            ("ACTION4", "ACTION3"), ("ACTION3", "ACTION4"),
            ("ACTION1", "ACTION2"), ("ACTION2", "ACTION1"),
        }
        items = list(actions)
        changed = True
        while changed:
            changed = False
            out = []
            i = 0
            while i < len(items):
                if i + 1 < len(items):
                    a, b = items[i], items[i + 1]
                    pair = (a.action, b.action)
                    undo = b.action == "ACTION7"
                    if ((pair in inverses or undo)
                            and not _signal(a) and not _signal(b)):
                        # drop both — a no-progress oscillation
                        i += 2
                        changed = True
                        continue
                out.append(items[i])
                i += 1
            items = out
        return [a.action for a in items]

    def _register_solution_on_solve(self) -> None:
        """On a score advance (level solved), register the win-path
        into the canonical solutions_kb so it can be replayed later
        (and shortened over future re-solves — solutions_kb keeps the
        shortest path as canonical).  win_path = the acts executed
        since this level began.  Reuses the existing feature; does not
        reinvent storage."""
        try:
            import sys as _sys
            from pathlib import Path as _P
            _adapter = (_P(__file__).resolve().parents[3]
                        / "usecases" / "arc-agi-3" / "python")
            if _adapter.exists() and str(_adapter) not in _sys.path:
                _sys.path.insert(0, str(_adapter))
            import solutions_kb
        except Exception as e:
            print(f"[driver] solutions_kb unavailable ({e}); skip")
            return
        start = getattr(self, "_level_start_action_idx", 0)
        acts = self.world.actions_taken[start:]
        if not acts:
            return
        # solutions_kb._act_signature does int(action_id), so it needs
        # NUMERIC ids.  The exploratory stack uses string action names
        # ("ACTION4"); map them to the game's numeric ids via the
        # adapter's action_map.  Keep the readable name in `name`.
        action_map = getattr(self.game, "_action_map", {}) or {}
        win_path = []
        for a in acts:
            num_id = action_map.get(a.action)
            if num_id is None:
                base = a.action.split(":", 1)[0]
                num_id = action_map.get(base, 0)
            rec = {"action_id": int(num_id),
                   "name": a.action,
                   "purpose": (a.rationale or "")[:80]}
            tc = getattr(a, "target_cell", None)
            if tc and len(tc) == 2:
                rec["click_xy"] = [int(tc[1]), int(tc[0])]
            win_path.append(rec)
        # The level JUST solved = score-1 (score N means N levels
        # completed).  Falls back to world.level if score is unset.
        solved_level = (
            (self.world.score - 1) if self.world.score is not None
            else self.world.level
        )
        sol = solutions_kb.save_solution(
            game_id=self.world.game_id,
            level=int(solved_level),
            win_path=win_path,
            source_session="auto:exploratory_driver",
            notes=(self.world.game_purpose_guess or "")[:200],
            tags=[self.world.game_id],
        )
        # Next level's win-path starts after this solve.
        self._level_start_action_idx = len(self.world.actions_taken)
        print(
            f"[driver] solutions_kb: registered {len(win_path)}-act "
            f"win-path for {self.world.game_id} lc={solved_level} "
            f"(id={sol.get('id')}, canonical={sol.get('is_canonical')})"
        )
        # DURABLE ARCHIVE: persist an organized, long-term record of this solve
        # (trace + detailed log + win-path + solution id) under <repo>/.archive,
        # keyed by game/level, for AI/human/miner perusal.  The mechanic digest is
        # crystallised at the NEXT level start -> patched in later (update_digest).
        try:
            import game_archive
            entry = game_archive.archive_solve(
                game_id=self.world.game_id, level=int(solved_level),
                work_dir=self.work_dir, win_path=win_path,
                solution_id=sol.get("id"), score=self.world.score,
                source_session="exploratory_driver",
                frame_dir=getattr(self.game, "_work_frame_dir", None))
            if entry:
                print(f"[archive] recorded {self.world.game_id} lc{solved_level} -> {entry}")
        except Exception as e:
            print(f"[archive] solve-record skipped ({e})")
        # WIN-CLAIM: record this level's win INSTANCE into the per-game win claim, so the win condition
        # generalizes across levels and is auto-recalled at future level starts (win_claims).  The
        # candidate features were captured by the closure hook at this level's start.
        try:
            import win_claims
            _feats = getattr(self, "_win_instance_features", None)
            if _feats:
                _pat = win_claims.record(self.world.game_id, f"lc{solved_level}", _feats)
                self._win_instance_features = None
                print(f"[win-claim] recorded lc{solved_level} win instance; claim now {_pat}")
        except Exception as e:
            print(f"[win-claim] record skipped ({e})")

    def _schedule_subroutine_auto_promotion(
        self,
        *,
        window_size: int,
        advanced_to_score: Optional[int],
        achieved_subgoal=None,
    ) -> None:
        """Offer the last ``window_size`` actions to a separate VLM call
        for auto-promotion into a reusable subroutine/technique.

        Triggered by EITHER a score advance (``advanced_to_score`` set) OR
        the achievement of an intermediate subgoal (``achieved_subgoal``
        set, ``advanced_to_score`` None) — the latter captures
        score-neutral techniques (e.g. 'move reds up to make them
        accessible') that the score-only trigger missed.

        V1: writes a `subroutine_promotion_prompt.md` into the work
        directory and lets the strategy actor (or a tool) supply a
        promotion reply at `subroutine_promotion_reply.txt`.  The
        driver does NOT block on this — the reply is processed on
        a best-effort basis at next turn boundary.
        """
        try:
            actions = list(self.world.actions_taken[-window_size:])
            if not actions:
                return
            first_turn = actions[0].turn
            last_turn  = actions[-1].turn
            prompt_dir = self.work_dir / "subroutine_promotion"
            prompt_dir.mkdir(parents=True, exist_ok=True)
            prompt_path = (
                prompt_dir
                / f"promotion_t{first_turn}_t{last_turn}.md"
            )
            chain_actions = [a.action for a in actions]
            chain_rationales = [
                f"  t{a.turn}: {a.action!r} — {a.rationale[:120]}"
                for a in actions
            ]
            if advanced_to_score is not None:
                _trigger_hdr = (
                    f"# Subroutine promotion — score advanced to "
                    f"{advanced_to_score} after turn {last_turn}\n\n"
                    f"The system just observed a score advance.  The "
                    f"action chain in the {window_size}-turn window "
                    f"leading up to it is below.  If this chain is "
                    f"worth remembering as a reusable subroutine, "
                    f"return a JSON object describing it.  Otherwise "
                    f"return null.\n\n")
            else:
                _sg_name = (getattr(achieved_subgoal, "name", "")
                            or getattr(achieved_subgoal, "subgoal_id", "")
                            or "an intermediate subgoal")
                _sg_prob = getattr(achieved_subgoal, "problem_solved", "") or ""
                _trigger_hdr = (
                    f"# Subroutine promotion — intermediate subgoal "
                    f"ACHIEVED after turn {last_turn} (score-neutral)\n\n"
                    f"The system just achieved a non-scoring subgoal: "
                    f"{_sg_name!r}"
                    + (f" — {_sg_prob[:160]}" if _sg_prob else "") + ".\n"
                    f"This kind of maneuver (repositioning / enabling a "
                    f"future move without itself advancing the score) is "
                    f"exactly the reusable technique that the score-only "
                    f"trigger used to miss.  The {window_size}-turn action "
                    f"window that achieved it is below.  If it is worth "
                    f"remembering as a reusable technique, return a JSON "
                    f"object describing it (state its PRECONDITION "
                    f"explicitly).  Otherwise return null.\n\n")
            prompt_text = (
                _trigger_hdr
                + f"Game: {self.world.game_id} lc={self.world.level}\n"
                f"Goal context (last action's goal_id): "
                f"{actions[-1].goal_id}\n\n"
                f"Chain (in execution order):\n"
                + "\n".join(chain_rationales)
                + "\n\nReply schema (write JSON to "
                  "`subroutine_promotion_reply.txt`):\n"
                  "{\n"
                  '  "name":             "<free-form short name>",\n'
                  '  "description":      "<free-form: what it does, '
                  'when to use it>",\n'
                  '  "problem_solved":   "<free-form: the problem '
                  'this addressed (goal + obstacle + constraint)>",\n'
                  '  "expected_outcome": "<free-form: what should '
                  'be true after this subroutine runs>",\n'
                  '  "parent_id":        "<subroutine_id if this is '
                  'a variant of an existing one, else null>",\n'
                  '  "variant_notes":   "<one-line description of '
                  'how this variant differs from parent, or null>"\n'
                  "}\n\n"
                  "Or, if the chain is not worth promoting, write "
                  "`null` (the literal JSON null)."
            )
            prompt_path.write_text(prompt_text, encoding="utf-8")
            print(f"[driver] subroutine auto-promotion prompt "
                  f"staged at {prompt_path}")

            # The reply file is checked opportunistically at the
            # next turn boundary — see _consume_pending_promotions.
            # We also track the turn at which the prompt was staged
            # so the consumer can auto-author after K turns of
            # silence (the score-advance chain is too valuable to
            # lose just because the actor never replied).
            self._pending_promotion_prompts.append((
                prompt_path, actions, advanced_to_score,
                self.world.turn,
            ))
        except Exception as e:
            print(f"[driver] subroutine promotion staging failed "
                  f"({e}); continuing")

    def _apply_subgoal_commits(self, strategy_choice
                               ) -> Optional[str]:
        """Apply a commit_subgoal from the strategy actor's reply.
        Called BEFORE the ActionRecord is written so the driver can
        tag the action's goal_id with the newly-committed
        subgoal_id.  Returns the new subgoal_id if one was created,
        else None.  Substrate just persists what the actor said;
        no validation or judgement."""
        if not strategy_choice:
            return None
        commit = getattr(strategy_choice, "commit_subgoal", None)
        if not (commit and isinstance(commit, dict)):
            return None
        name           = str(commit.get("name") or "(unnamed)")
        problem_solved = str(commit.get("problem_solved") or "")
        expected       = str(commit.get("expected_outcome") or "")
        parent_id      = commit.get("parent_id") or None
        related        = commit.get("related_subroutine_id") or None
        notes          = str(commit.get("notes") or "")
        # Discipline fields: forward_simulation may live either on
        # the commit dict OR (more commonly) on the top-level reply
        # — accept either source.  Same for derived_from /
        # win_condition_hypothesis_id.
        forward_sim    = str(
            commit.get("forward_simulation")
            or getattr(strategy_choice, "forward_simulation", "")
            or ""
        )
        derived_from   = str(commit.get("derived_from") or "")
        wc_hyp_id      = commit.get("win_condition_hypothesis_id") or None
        # Sequential-subgoals (2026-05-29): flat list of subgoal_ids
        # the actor declares as preconditions.  Bridged to
        # cognitive_os Goal.depends_on for engine-level gating.
        depends_on_raw = commit.get("depends_on") or []
        if isinstance(depends_on_raw, str):
            depends_on_raw = [depends_on_raw]
        depends_on = [str(d) for d in depends_on_raw if d]
        # Subgoal Completion Contract fields.
        acceptance_check = str(commit.get("acceptance_check") or "")
        refs_raw = commit.get("references_entities") or []
        if isinstance(refs_raw, str):
            refs_raw = [refs_raw]
        references_entities = [str(r) for r in refs_raw if r]
        or_group = commit.get("or_group") or None
        sg = _sg_commit(
            self.world,
            name=name,
            problem_solved=problem_solved,
            expected_outcome=expected,
            parent_id=parent_id,
            related_subroutine_id=related,
            notes=notes,
            forward_simulation=forward_sim,
            derived_from=derived_from,
            win_condition_hypothesis_id=wc_hyp_id,
            depends_on=depends_on,
            acceptance_check=acceptance_check,
            references_entities=references_entities,
            or_group=or_group,
        )
        dep_str = (f" depends_on={depends_on}" if depends_on else "")
        ac_str = (" +acceptance_check" if acceptance_check else "")
        og_str = (f" or_group={or_group}" if or_group else "")
        print(f"[driver] committed subgoal {sg.subgoal_id} "
              f"({sg.name!r}){dep_str}{ac_str}{og_str}")
        return sg.subgoal_id

    def _sync_game_purpose_claim(self, purpose: str) -> None:
        """Treat game_purpose as a VALIDATABLE CLAIM, not a free
        string: mirror it into the WinConditionHypothesis lifecycle.

        A win_condition hypothesis already carries credence,
        supporting/contradicting observations, promotion + refutation,
        and is persisted cross-trial by the per_game_lessons distiller
        (as 'win_condition' or, if refuted, 'refuted').  By keeping the
        current game_purpose registered as a WC hypothesis, a
        materially-changed purpose leaves the OLD one on record (with
        whatever contradicting observations it accrued) and registers
        the NEW one — so the next trial sees which purposes were tried
        and refuted, and can author a better guess instead of
        re-deriving a known-bad one.

        Idempotent: if the latest WC hypothesis already carries this
        exact purpose text, do nothing (avoids duplicate spam when the
        actor restates the same purpose every turn).
        """
        purpose = (purpose or "").strip()
        if not purpose:
            return
        hyps = getattr(self.world, "win_condition_hypotheses", None)
        if hyps is None:
            return
        # Already the most-recent claim? no-op.
        for h in hyps:
            if (h.description or "").strip().lower() == purpose.lower():
                return
        try:
            from win_condition_hypotheses import (        # noqa: E402
                commit_hypothesis,
            )
        except Exception as e:
            print(f"[driver] WC module unavailable for purpose-sync "
                  f"({e})")
            return
        commit_hypothesis(
            self.world,
            description=purpose,
            notes=(f"auto-registered from game_purpose at turn "
                   f"{self.world.turn} (validatable claim)"),
        )

    def _apply_win_condition_updates(self, strategy_choice) -> None:
        """Apply commit_win_condition_hypothesis and
        win_condition_observation from the strategy reply.
        Substrate-side bookkeeping; no validation."""
        if not strategy_choice:
            return
        try:
            from win_condition_hypotheses import (        # noqa: E402
                commit_hypothesis as _wc_commit,
                record_observation as _wc_record,
            )
        except Exception as e:
            print(f"[driver] win-condition module unavailable "
                  f"({e}); skipping")
            return

        # Commit a new hypothesis
        commit = getattr(strategy_choice,
                         "commit_win_condition_hypothesis", None)
        if commit and isinstance(commit, dict):
            desc       = str(commit.get("description") or "").strip()
            notes      = str(commit.get("notes") or "")
            credence   = commit.get("credence")
            try:
                credence = float(credence) if credence is not None \
                    else 0.30
            except Exception:
                credence = 0.30
            if desc:
                wc = _wc_commit(
                    self.world, description=desc, notes=notes,
                    credence=credence,
                )
                print(f"[driver] committed win-condition hyp "
                      f"{wc.hypothesis_id!r} "
                      f"(credence={wc.credence:.2f})")

        # Apply observations (single dict or list)
        obs = getattr(strategy_choice,
                      "win_condition_observation", None)
        if obs:
            obs_list = obs if isinstance(obs, list) else [obs]
            # Compute the delta index this observation references
            delta_idx = len(self.world.deltas_observed) - 1
            for entry in obs_list:
                if not isinstance(entry, dict):
                    continue
                hid  = entry.get("hypothesis_id")
                kind = entry.get("kind")
                if not hid or kind not in ("support", "contradict"):
                    continue
                result = _wc_record(
                    self.world,
                    hypothesis_id=hid,
                    delta_index=delta_idx,
                    kind=kind,
                )
                if result:
                    print(
                        f"[driver] wc-hyp {hid} {kind} "
                        f"(credence -> {result.credence:.2f}, "
                        f"promoted={result.promoted})"
                    )
                else:
                    print(f"[driver] wc-hyp {hid} not found "
                          f"for observation")

    def _register_failed_assumption(self, strategy_choice) -> None:
        """Write side of the DO-NOT-REPEAT avoid-list.  When this turn's
        action tested a hypothesis (``testing_hypothesis``) and the
        resulting delta was a NO-OP (no agent move, no entity
        change/appear/disappear, no visual event), the tactical
        assumption is FALSIFIED.  Persist it immediately as a 'refuted'
        per-game lesson so future turns/trials surface it on the
        avoid-list — without depending on the actor to remember.

        Conservative: only fires on a genuine no-op of a real,
        hypothesis-tagged action.  Best-effort; never raises.
        """
        if not strategy_choice:
            return
        hyp = (getattr(strategy_choice, "testing_hypothesis", None)
               or "").strip()
        action = (getattr(strategy_choice, "endorsed_action", None)
                  or "").strip()
        if not hyp or not action or action.upper() == "NONE":
            return
        deltas = getattr(self.world, "deltas_observed", None) or []
        if not deltas:
            return
        d = deltas[-1]

        def _g(o, k, default=None):
            return (o.get(k, default) if isinstance(o, dict)
                    else getattr(o, k, default))
        no_op = (
            not _g(d, "agent_moved", False)
            and not (_g(d, "entities_changed", None) or [])
            and not (_g(d, "entities_appeared", None) or [])
            and not (_g(d, "entities_disappeared", None) or [])
            and not (_g(d, "visual_events", None) or [])
        )
        if not no_op:
            return
        desc = (
            f"TACTIC FAILED: hypothesis {hyp!r} via action {action} "
            f"produced NO observable change at turn "
            f"{getattr(self.world, 'turn', '?')}. Do not repeat this "
            f"action in this configuration expecting that effect; form "
            f"a different approach."
        )
        try:
            from per_game_lessons import (                  # noqa: E402
                commit_lesson_from_actor as _commit,
            )
            _commit(
                game_id=self.world.game_id,
                kind="refuted",
                description=desc,
                notes="auto-registered from no-op delta of a "
                      "hypothesis-tagged action",
                credence=0.6,
                trial_id=f"auto_t{getattr(self.world, 'turn', 0)}",
            )
            print(f"[driver] AUTO-REFUTED tactic: {hyp!r} via "
                  f"{action} (no-op) -> avoid-list")
        except Exception as e:
            print(f"[driver] auto-refute persist failed ({e})")

    def _apply_subgoal_status_updates(self, strategy_choice) -> None:
        """Apply subgoal_status_update from the strategy actor's
        reply.  Called AFTER the action has executed so the actor's
        achieved / abandoned / blocked verdicts reflect the
        post-action world state.

        Substrate gate: status='achieved' requires a
        ``confirming_signal``.  Without it the substrate
        downgrades to 'inferred_satisfied' and keeps the subgoal
        in the open set."""
        if not strategy_choice:
            return
        updates = getattr(strategy_choice, "subgoal_status_update",
                          None)
        if not updates:
            return
        update_list = (
            updates if isinstance(updates, list) else [updates]
        )
        for upd in update_list:
            if not isinstance(upd, dict):
                continue
            sgid   = upd.get("subgoal_id")
            status = upd.get("status")
            notes  = str(upd.get("notes") or "")
            confirming = str(upd.get("confirming_signal") or "")
            impediment = str(upd.get("impediment") or "")
            if not sgid or not status:
                continue
            result = _sg_update(
                self.world,
                subgoal_id=sgid, status=status,
                notes_append=notes,
                confirming_signal=confirming,
                impediment=impediment,
            )
            if result:
                effective_status = result.status
                if effective_status != status:
                    print(
                        f"[driver] subgoal {sgid} requested "
                        f"status={status!r} but gate not met — "
                        f"now {effective_status!r}"
                    )
                else:
                    print(f"[driver] subgoal {sgid} -> "
                          f"status={effective_status!r}")
                # Component C: a successfully-blocked subgoal spawns
                # a removal subgoal and depends_on itself on it.
                if (effective_status == "blocked"
                        and (result.impediment or "").strip()):
                    self._spawn_removal_subgoal(result)
                # ACHIEVED-SUBGOAL -> TECHNIQUE capture.  Previously the
                # ONLY trigger for distilling a reusable maneuver was a
                # SCORE ADVANCE (_schedule_subroutine_auto_promotion is
                # called solely under score_advanced).  A technique that
                # achieves an INTERMEDIATE, score-neutral subgoal — e.g.
                # 'move the reds up to make them accessible', which
                # repositions blocks without piercing anything, so the
                # score does not change — produced NO capture signal and
                # was lost (had to be hand-authored later).  This is the
                # 'achieved-subgoal -> technique' half of the credit-
                # assignment that the spec calls for but was unbuilt.
                # Now: a genuinely-achieved subgoal (gate met, real
                # confirming signal) ALSO schedules promotion over the
                # window since it was committed, so the maneuver that
                # achieved it is offered for distillation.  See
                # memory/feedback_ensure_cos_self_sufficiency +
                # SPEC_cumulative_learning_loop.md.
                if effective_status == "achieved":
                    try:
                        committed_turn = (
                            getattr(result, "created_at_turn", None)
                            or getattr(result, "committed_turn", None))
                        win = 12
                        if isinstance(committed_turn, int):
                            win = max(2, min(
                                40, self.world.turn - committed_turn + 1))
                        self._schedule_subroutine_auto_promotion(
                            window_size=win, advanced_to_score=None,
                            achieved_subgoal=result,
                        )
                        print(f"[driver] subgoal {sgid} ACHIEVED "
                              f"(score-neutral ok) -> scheduled technique "
                              f"capture over last {win} action(s)")
                    except Exception as e:
                        print(f"[driver] achieved-subgoal technique "
                              f"capture skipped ({e})")
                # Write side of the avoid-list: an ABANDONED subgoal is
                # a failed strategic assumption — persist it as a
                # 'refuted' lesson so it isn't silently re-attempted.
                if effective_status == "abandoned":
                    try:
                        from per_game_lessons import (      # noqa: E402
                            commit_lesson_from_actor as _commit,
                        )
                        prob = (getattr(result, "problem_solved", "")
                                or getattr(result, "name", "")
                                or sgid)
                        _commit(
                            game_id=self.world.game_id,
                            kind="refuted",
                            description=(
                                f"ABANDONED STRATEGY: {prob} "
                                f"(subgoal {sgid} abandoned at turn "
                                f"{getattr(self.world,'turn','?')}; "
                                f"notes: {notes[:160]})"
                            ),
                            notes="auto-registered from abandoned "
                                  "subgoal",
                            credence=0.6,
                            trial_id=f"auto_t"
                                     f"{getattr(self.world,'turn',0)}",
                        )
                        print(f"[driver] AUTO-REFUTED abandoned "
                              f"strategy {sgid} -> avoid-list")
                    except Exception as e:
                        print(f"[driver] abandoned-refute persist "
                              f"failed ({e})")
            else:
                print(f"[driver] subgoal {sgid} not found "
                      f"for status update")

    def _spawn_removal_subgoal(self, blocked_sg) -> None:
        """Component C: turn an impediment into an actionable child
        subgoal and gate the blocked subgoal on it (depends_on edge
        via the bridge).  The blocked subgoal re-activates when the
        removal achieves."""
        removal = _sg_commit(
            self.world,
            name=f"remove_impediment_for_{blocked_sg.name}"[:60],
            problem_solved=(
                f"Subgoal {blocked_sg.subgoal_id} is blocked by: "
                f"{blocked_sg.impediment}.  Achieve this removal "
                f"subgoal to lift the block."
            ),
            expected_outcome=(
                f"The impediment '{blocked_sg.impediment}' no longer "
                f"holds, so {blocked_sg.subgoal_id} can resume."
            ),
            parent_id=blocked_sg.subgoal_id,
            derived_from=(
                f"to advance parent {blocked_sg.subgoal_id}: remove "
                f"impediment '{blocked_sg.impediment}'"
            ),
        )
        # Gate the blocked subgoal on the removal: add removal id to
        # its depends_on and re-register with the bridge so the
        # engine hides it until removal achieves.
        try:
            blocked_sg.depends_on = list(
                set((blocked_sg.depends_on or []) + [removal.subgoal_id])
            )
            from subgoal_forest_bridge import register_subgoal  # noqa: E402
            # Re-register: drop the stale Goal, re-add with new deps.
            from subgoal_forest_bridge import _get_or_init_bridge  # noqa: E402
            ws = _get_or_init_bridge(self.world)
            ws.goal_forest.goals.pop(blocked_sg.subgoal_id, None)
            register_subgoal(self.world, blocked_sg)
        except Exception as e:
            print(f"[driver] removal depends_on wiring failed "
                  f"({e}); removal exists but gate not enforced")
        print(
            f"[driver] component C: spawned removal subgoal "
            f"{removal.subgoal_id} for blocked {blocked_sg.subgoal_id} "
            f"(impediment: {blocked_sg.impediment[:60]})"
        )

    def _record_subgoal_approach(self, action_record, delta) -> None:
        """Component B: append the pursued subgoal's (action,
        state-class) to its exhaustion ledger."""
        gid = getattr(action_record, "goal_id", None) or ""
        if not gid.startswith("subgoal:"):
            return
        sgid = gid.split("subgoal:", 1)[1]
        sc = self._state_class(delta)
        for sg in self.world.active_subgoals:
            if sg.subgoal_id == sgid:
                entry = [action_record.action, sc]
                if entry not in sg.approaches_tried:
                    sg.approaches_tried.append(entry)
                break

    def _state_class(self, delta) -> str:
        """Compact, game-agnostic state-class label for the approach
        ledger: agent row band + whether the last action moved the
        agent.  Coarse on purpose — it's a probe-coverage key, not a
        full state encoding."""
        cell = getattr(delta, "agent_new_cell", None)
        row = cell[0] if (cell and len(cell) > 0) else None
        gi = self.world.grid_inference
        rows = getattr(gi, "rows", None) if gi else None
        if row is None or not rows:
            band = "unknown"
        elif row <= rows // 3:
            band = "top"
        elif row <= 2 * rows // 3:
            band = "mid"
        else:
            band = "bottom"
        moved = "moved" if getattr(delta, "agent_moved", False) else "static"
        return f"row_band={band},last={moved}"

    def _verify_pending_operator(self) -> None:
        """Per-game replay verification: judge the operator the actor declared
        it applied (``applied_operator_id``) by the OUTCOME this turn. A pure
        no-op (no movable moved + no score) REFUTES it for the game (the
        impale-against-wall failure); a score advance CONFIRMS it. A weak signal
        leaves it unconfirmed. So a mined-generic operator is trusted for a game
        only once it actually works there. Defensive; never breaks the step."""
        op_id = getattr(self.world, "_pending_operator", None)
        if not op_id:
            return
        self.world._pending_operator = None
        try:
            import operator_kb as _ok                            # noqa: E402
            gid = getattr(self.world, "game_id", "?")
            moved = False
            _roles = {"collectable", "block", "pushable", "movable",
                      "piece", "goal_object"}
            for rec in (getattr(self.world, "entities", {}) or {}).values():
                if (getattr(rec, "current_role", "") or "").lower() not in _roles:
                    continue
                bh = getattr(rec, "bbox_history", None) or []
                if len(bh) >= 2 and bh[-1][1] != bh[-2][1]:
                    moved = True
                    break
            cur = int(getattr(self.world, "score", 0) or 0)
            prev = getattr(self, "_last_verify_score", cur)
            self._last_verify_score = cur
            confirmed = True if cur > prev else (False if not moved else None)
            if confirmed is None:
                return                       # weak signal -> stay unconfirmed
            recs = _ok.load_kb()
            if _ok.note_operator_outcome(recs, op_id, gid, confirmed):
                _ok.save_kb(recs)
                print(f"[driver] operator {op_id} -> "
                      f"{'CONFIRMED' if confirmed else 'REFUTED'} for {gid}")
        except Exception:
            pass

    def _refresh_bboxes_from_frame(self) -> None:
        """Update movable entities' current bbox from the LIVE frame, so the
        substrate's geometry (relations, mediation precondition, attachment,
        structural producer) reflects reality instead of the perception
        reply's possibly-stale/placeholder bboxes. Perception owns semantics;
        the frame owns geometry (durable principle P3). Runs AFTER
        ingest_perception so the frame-derived bbox is authoritative for the
        turn. Defensive — never breaks the step. See frame_bbox_refresh."""
        try:
            import frame_bbox_refresh as _fbr                     # noqa: E402
            n = _fbr.refresh_entities_from_frame(
                self.world, self._last_frame_path,
                start_frame_path=self._level_start_frame_path,
                level=getattr(self.world, "score", None),
            )
            if n:
                # one-line, low-noise confirmation
                pass
        except Exception as e:
            print(f"[driver] frame bbox refresh skipped ({e})")
        # Per-instance perception: track each VLM-named type's instances from
        # the live frame, so the strategy surface can carry exact per-instance
        # facts (counts, positions, relations) instead of the VLM eyeballing.
        # Colors are the ones frame_bbox_refresh bootstrapped per named entity.
        try:
            import frame_bbox_refresh as _fbr2                    # noqa: E402
            import instance_perception as _ip                     # noqa: E402
            gid = getattr(self.world, "game_id", "?")
            # Track ONLY the manipulable blocks (collectables). Color-based
            # per-instance segmentation is reliable for saturated, distinctly
            # colored blocks; the agent parts (grey rod, gripper) are NOT
            # color-separable (their placeholder regions overlap blocks/rail)
            # and their geometry comes from the kinematic structural producer,
            # not color. Excluding them keeps the fact-sheet clean + correct.
            _coll_roles = {"collectable", "block", "pushable", "movable",
                           "piece", "goal_object"}
            _coll_names = {
                getattr(rec, "name", None)
                for rec in (getattr(self.world, "entities", {}) or {}).values()
                if (getattr(rec, "current_role", "") or "").lower() in _coll_roles}
            name_colors = {nm: col for (g, nm), col in _fbr2._COLOR_CACHE.items()
                           if g == gid and nm in _coll_names}
            if name_colors:
                # Reset instance ids on a level change (fresh #1.. per level).
                _lvl = getattr(self.world, "score", None)
                if (self._instance_tracker is None
                        or getattr(self, "_instance_tracker_level", None) != _lvl):
                    self._instance_tracker = _ip.InstanceTracker()
                    self._instance_tracker_level = _lvl
                import copy as _copy
                self._prev_instance_tracker = _copy.deepcopy(
                    self._instance_tracker)
                self._instance_tracker.update(
                    str(self._last_frame_path), name_colors,
                    turn=int(getattr(self.world, "turn", 0)))
                self.world._instance_tracker = self._instance_tracker
                self.world._prev_instance_tracker = self._prev_instance_tracker
                # Apply any VLM-TAUGHT recognizers (e.g. 'impaled') cheaply,
                # every turn, with no VLM call. Verdicts (+ abstentions) are
                # surfaced so the VLM reads the count from the substrate.
                try:
                    import taught_recognizers as _trc              # noqa: E402
                    recs = _trc.load_recognizers(gid)
                    if recs:
                        applied = {}
                        for rname, rec in recs.items():
                            applied[rname] = (rec, _trc.apply_all(
                                rec, str(self._last_frame_path),
                                self._instance_tracker.instances))
                        self.world._recognizer_results = applied
                except Exception:
                    pass
        except Exception as e:
            print(f"[driver] instance perception skipped ({e})")

    def _run_subgoal_contract(self, delta) -> None:
        """Component A + invalidation: run the substrate-authority
        subgoal transitions for this turn and log them."""
        from active_subgoals import substrate_evaluate_subgoals  # noqa: E402
        transitions = substrate_evaluate_subgoals(self.world, delta)
        for (sgid, new_status, reason) in transitions:
            print(
                f"[subgoal-contract] {sgid} -> {new_status} "
                f"({reason})"
            )
            # ACHIEVED-SUBGOAL -> TECHNIQUE capture (substrate-authority
            # path).  The ONLY prior capture trigger was a SCORE ADVANCE,
            # so score-neutral enabling techniques (e.g. 'move reds up to
            # make them accessible') were never distilled.  A subgoal whose
            # acceptance_check fires here is an achieved intermediate goal;
            # offer the action window that achieved it for promotion. This
            # is the substrate-side counterpart to the actor-driven hook in
            # _apply_subgoal_status_updates (auto-achievement happens HERE,
            # not there). See memory/feedback_ensure_cos_self_sufficiency.
            if new_status == "achieved":
                try:
                    sg = self._find_subgoal(sgid)
                    committed_turn = (
                        getattr(sg, "created_at_turn", None)
                        if sg else None)
                    win = 12
                    if isinstance(committed_turn, int):
                        win = max(2, min(
                            40, self.world.turn - committed_turn + 1))
                    self._schedule_subroutine_auto_promotion(
                        window_size=win, advanced_to_score=None,
                        achieved_subgoal=sg,
                    )
                    print(f"[driver] subgoal {sgid} ACHIEVED "
                          f"(score-neutral) -> scheduled technique capture "
                          f"over last {win} action(s)")
                except Exception as e:
                    print(f"[driver] achieved-subgoal technique capture "
                          f"skipped ({e})")

    def _find_subgoal(self, sgid: str):
        """Return the ActiveSubgoal record by id, or None. Tolerant of the
        world's storage shape (dict or list)."""
        sgs = (getattr(self.world, "active_subgoals", None)
               or getattr(self.world, "subgoals", None) or [])
        items = sgs.values() if hasattr(sgs, "values") else sgs
        for s in items:
            if getattr(s, "subgoal_id", None) == sgid or (
                    isinstance(s, dict) and s.get("subgoal_id") == sgid):
                return s
        return None

    def _apply_probe_updates(self, strategy_choice) -> None:
        """Apply propose_probe / probe_observation / probe_abandon
        from the strategy reply.  Substrate maintains the probe
        ledger; observations auto-propagate credences to the
        referenced WC + Mechanic hypotheses."""
        if not strategy_choice:
            return
        try:
            from probes import (                          # noqa: E402
                propose_probe as _probe_propose,
                record_probe_observation as _probe_record,
                abandon_probe as _probe_abandon,
            )
        except Exception as e:
            print(f"[driver] probes module unavailable "
                  f"({e}); skipping")
            return

        # Commit a new probe
        prop = getattr(strategy_choice, "propose_probe", None)
        if prop and isinstance(prop, dict):
            uncertainty = str(
                prop.get("motivating_uncertainty") or ""
            )
            hyp_ids = list(
                prop.get("motivating_hypothesis_ids") or []
            )
            action_seq = list(
                prop.get("action_or_sequence") or []
            )
            preds = dict(prop.get("predicted_outcomes") or {})
            notes = str(prop.get("notes") or "")
            new_p = _probe_propose(
                self.world,
                motivating_uncertainty=uncertainty,
                motivating_hypothesis_ids=hyp_ids,
                action_or_sequence=action_seq,
                predicted_outcomes=preds,
                notes=notes,
            )
            if new_p is not None:
                print(
                    f"[driver] proposed probe "
                    f"{new_p.probe_id!r} "
                    f"(discriminating {len(hyp_ids)} hyps)"
                )

        # Record observation on an executed probe
        obs = getattr(strategy_choice, "probe_observation", None)
        if obs and isinstance(obs, dict):
            pid       = obs.get("probe_id")
            outcome   = str(obs.get("observed_outcome") or "")
            matches   = list(
                obs.get("matching_hypothesis_ids") or []
            )
            contras   = list(
                obs.get("contradicting_hypothesis_ids") or []
            )
            notes     = str(obs.get("notes") or "")
            if pid:
                result = _probe_record(
                    self.world,
                    probe_id=pid,
                    observed_outcome=outcome,
                    matching_hypothesis_ids=matches,
                    contradicting_hypothesis_ids=contras,
                    notes=notes,
                )
                if result:
                    print(
                        f"[driver] probe {pid} observed "
                        f"(status={result.status!r}, "
                        f"+{len(matches)}/-{len(contras)} "
                        f"credences propagated)"
                    )
                else:
                    print(f"[driver] probe {pid} not found "
                          f"for observation")

        # Abandon a pending probe
        ab = getattr(strategy_choice, "probe_abandon", None)
        if ab and isinstance(ab, dict):
            pid    = ab.get("probe_id")
            reason = str(ab.get("reason") or "")
            if pid:
                result = _probe_abandon(
                    self.world, probe_id=pid, reason=reason,
                )
                if result:
                    print(f"[driver] probe {pid} abandoned")
                else:
                    print(f"[driver] probe {pid} not found "
                          f"for abandon")

    def _resolve_goal_id_from_subgoal(
        self, strategy_choice, committed_id: Optional[str],
        fallback_goal_id: str,
    ) -> str:
        """Decide the ActionRecord's goal_id given subgoal
        commitments / focus from the strategy actor's reply.
        Priority:
          1. pursuing_subgoal_id (if it points to a real active or
             blocked subgoal)
          2. just-committed subgoal_id (committed_id) — only set
             if commit_subgoal fired this turn
          3. fallback to the mechanical planner's goal_id
        """
        if strategy_choice is None:
            return fallback_goal_id
        pursuing = getattr(strategy_choice, "pursuing_subgoal_id",
                           None)
        if pursuing:
            # validate it points to an actual active subgoal
            for sg in self.world.active_subgoals:
                if (sg.subgoal_id == pursuing
                        and sg.status in ("active", "blocked")):
                    # Sequential-subgoals gate (2026-05-29):
                    # if the actor is pursuing a subgoal whose
                    # depends_on isn't satisfied, log a
                    # warn-and-proceed line so the trace shows
                    # the gate fired.  Action still runs per
                    # actor's endorsed_action.
                    try:
                        from subgoal_forest_bridge import (    # noqa: E402
                            actionable_status,
                        )
                        ok, unmet = actionable_status(
                            self.world, sg.subgoal_id,
                        )
                        if not ok:
                            action = getattr(
                                strategy_choice, "endorsed_action",
                                "<unknown>",
                            )
                            print(
                                f"[blocked-subgoal] turn "
                                f"{self.world.turn}: actor pursued "
                                f"{sg.subgoal_id!r} "
                                f"({sg.name!r}) while depends_on="
                                f"{unmet} unmet; action {action} "
                                f"proceeds (warn-and-proceed "
                                f"policy)."
                            )
                    except Exception:
                        pass
                    return f"subgoal:{sg.subgoal_id}"
            # fall through if id was stale
        if committed_id:
            return f"subgoal:{committed_id}"
        # Actor OVERRODE the mechanical planner but named no subgoal:
        # attribute the goal to the ACTOR's intent, never the planner's
        # discarded curiosity goal (the bug where 36/43 turns all read
        # 'explore:action:ACTION6' even though the actor was running a
        # deliberate maneuver — the real intent lived only in rationale).
        # Prefer the testing_hypothesis; else a generic strategy tag.
        # An ENDORSE (no override) correctly keeps the mechanical goal.
        if getattr(strategy_choice, "overrode_mechanical", False):
            hyp = getattr(strategy_choice, "testing_hypothesis", None)
            if hyp:
                return f"hypothesis:{hyp}"
            action = getattr(strategy_choice, "endorsed_action", None)
            if action:
                return f"vlm_strategy:{action}"
        return fallback_goal_id

    # ------------------------------------------------------------------
    # Auto-author fallback config — if the actor doesn't reply to a
    # subroutine-promotion prompt within this many turns, the
    # consumer authors a minimal Subroutine record from the chain
    # so the winning sequence isn't lost.  See task #1 in the
    # 2026-05-29 session — promotion prompts fire correctly on
    # score advance but historically the reply slot stayed empty,
    # leaving the winning chain undistilled.
    # ------------------------------------------------------------------
    _PROMOTION_AUTO_AUTHOR_AFTER_TURNS = 5

    def _consume_pending_promotions(self) -> None:
        """Walk pending promotion prompts.  Three paths per entry:

          1. Reply file exists → parse JSON, build full Subroutine
             from actor-supplied name / description / etc.
          2. No reply file, age < auto-author threshold → keep
             pending; actor may still respond.
          3. No reply file, age >= threshold → AUTO-AUTHOR a
             minimal Subroutine from the action chain.  Better to
             persist a placeholder we can review than to lose the
             winning sequence to actor silence.
        """
        pending = getattr(self, "_pending_promotion_prompts", None)
        if not pending:
            return
        still: list = []
        for entry in pending:
            # Back-compat with 3-tuple entries from older sessions.
            if len(entry) == 4:
                prompt_path, actions, advanced_to_score, staged_at = entry
            else:
                prompt_path, actions, advanced_to_score = entry
                staged_at = self.world.turn  # treat as just-staged
            reply_path = prompt_path.with_suffix(".reply.txt")

            if not reply_path.exists():
                age = self.world.turn - staged_at
                if age < self._PROMOTION_AUTO_AUTHOR_AFTER_TURNS:
                    still.append((prompt_path, actions,
                                   advanced_to_score, staged_at))
                    continue
                # Auto-author from the chain.  Prune redundant
                # oscillation steps first so the stored solution is
                # closer to minimal (ARC-AGI-3 scores step-efficiency).
                try:
                    raw_chain = [a.action for a in actions]
                    chain = self._prune_winning_chain(actions)
                    turn_range = [actions[0].turn, actions[-1].turn]
                    if len(chain) < len(raw_chain):
                        print(
                            f"[driver] pruned winning chain "
                            f"{len(raw_chain)} -> {len(chain)} steps "
                            f"(removed signal-free oscillations)"
                        )
                    auto_name = (
                        f"auto_score_advance_chain_t"
                        f"{turn_range[0]}_t{turn_range[1]}"
                    )
                    auto_desc = (
                        f"Auto-distilled subroutine: the {len(chain)}"
                        f"-action chain that preceded the score "
                        f"advance to {advanced_to_score} at turn "
                        f"{turn_range[1]}.  No actor reply was "
                        f"supplied within "
                        f"{self._PROMOTION_AUTO_AUTHOR_AFTER_TURNS} "
                        f"turns; substrate captured the sequence so "
                        f"the winning trajectory isn't lost.  "
                        f"Description is a placeholder — review the "
                        f"per-turn rationales in the trace and "
                        f"refine on next observation."
                    )
                    auto_problem = (
                        f"Advance score from "
                        f"{(advanced_to_score or 1) - 1} to "
                        f"{advanced_to_score} on {self.world.game_id} "
                        f"lc={self.world.level}."
                    )
                    auto_outcome = (
                        f"Score advances to {advanced_to_score} "
                        f"within the chain's execution window."
                    )
                    # COMPLIANCE (Tier 1): do NOT store a literal action-chain
                    # replay in the cross-game subroutine_kb.  An auto-captured
                    # score-advance chain is a MEMORIZED, non-transferable solution
                    # (overfit by definition) that the ARC-AGI-3 rules forbid
                    # carrying onto an unseen game -- and it is redundant, since the
                    # winning trajectory is already saved in solutions_kb (which
                    # lives under dev_only/).  Skip the promote; just resolve the
                    # pending prompt below so it doesn't loop.
                    _ = (auto_name, auto_desc, auto_problem, auto_outcome)
                    print(
                        f"[driver] auto-distill SKIPPED (Tier-1 compliance): the "
                        f"{len(chain)}-action score-advance chain "
                        f"t{turn_range[0]}-{turn_range[1]} is a literal replay; "
                        f"not stored in subroutine_kb (the win-path is in the "
                        f"dev-only solutions_kb)."
                    )
                    # Mark the prompt as auto-resolved.
                    consumed = prompt_path.with_suffix(
                        ".auto-authored.md",
                    )
                    try:
                        prompt_path.rename(consumed)
                    except Exception:
                        pass
                except Exception as e:
                    print(f"[driver] auto-author failed ({e}); "
                           f"leaving pending one more turn")
                    still.append((prompt_path, actions,
                                   advanced_to_score, staged_at))
                continue

            try:
                body = reply_path.read_text(encoding="utf-8").strip()
                if not body or body.lower() in ("null", '"null"'):
                    print(f"[driver] auto-promotion declined "
                          f"({reply_path.name})")
                else:
                    reply = json.loads(body)
                    chain = [a.action for a in actions]
                    turn_range = [actions[0].turn, actions[-1].turn]
                    sub = _subroutine_promote(
                        name=str(reply.get("name", "(unnamed)")),
                        description=str(reply.get("description", "")),
                        problem_solved=str(
                            reply.get("problem_solved", ""),
                        ),
                        concrete_chain=chain,
                        expected_outcome=str(
                            reply.get("expected_outcome", ""),
                        ),
                        game_id=self.world.game_id,
                        level=self.world.level,
                        turn_range=turn_range,
                        original_goal=(actions[-1].goal_id
                                        or "(no goal)"),
                        parent_id=reply.get("parent_id") or None,
                        variant_notes=str(
                            reply.get("variant_notes") or "",
                        ),
                    )
                    print(f"[driver] auto-promoted subroutine "
                          f"{sub.subroutine_id} ({sub.name!r}) "
                          f"from t{turn_range[0]}-{turn_range[1]}")
                # consumed: rename
                consumed = reply_path.with_suffix(".reply.consumed.txt")
                reply_path.replace(consumed)
            except Exception as e:
                print(f"[driver] auto-promotion reply parse failed "
                      f"({e}); leaving pending")
                still.append((prompt_path, actions,
                              advanced_to_score, staged_at))
        self._pending_promotion_prompts = still


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse
    # Force UTF-8 console output so --help / logs (which contain Unicode like
    # arrows and box chars) don't crash under the Windows cp1252 console.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-id", required=True,
                        help="game identifier (e.g. bp35)")
    parser.add_argument("--level", type=int, default=0,
                        help="Working level to play.  For the LIVE adapter, "
                             "levels >0 are reached by FAST-FORWARDING: the "
                             "driver replays the canonical recorded solutions "
                             "(solutions_kb) for levels 0..N-1 with no VLM "
                             "calls, then hands control to live play on level "
                             "N.  Requires those earlier levels to have been "
                             "solved before (recorded); if a recorded solution "
                             "is missing/stale the driver falls back to playing "
                             "from level 0.  For the fixture adapter it is the "
                             "fixture's entry level (no seek).")
    parser.add_argument("--fixture-dir", type=Path,
                        help="fixture directory for FixtureReplayAdapter; "
                             "if omitted, use LiveHarnessAdapter")
    parser.add_argument("--env-dir", type=Path, default=None,
                        help="environment_files directory for "
                             "LiveHarnessAdapter (defaults to repo's "
                             "<root>/environment_files)")
    parser.add_argument("--work-dir", type=Path,
                        default=Path(""
                                      ".tmp/exploratory_play"))
    parser.add_argument("--max-turns", type=int, default=40)
    parser.add_argument("--timeout-s", type=int, default=1200,
                        help="Overall per-poll wait budget (s). HITL default; "
                             "a human/Claude may think for a while.")
    parser.add_argument("--vlm-timeout-s", type=int, default=None,
                        help="How long a single VLM-reply poll (perception / "
                             "strategy) waits before DEGRADING to substrate-only "
                             "perception + the mechanical/explorer fallback "
                             "(never crashes). Defaults to --timeout-s. For "
                             "strict/competition mode set short (e.g. 30-60) so a "
                             "slow/dead autonomous VLM is bypassed fast.")
    parser.add_argument("--no-strategy", action="store_true",
                        help="Skip the per-turn VLM strategy call "
                             "(use mechanical actor only).  Saves "
                             "one VLM call per turn at the cost of "
                             "pure-BFS-style decisions.")
    parser.add_argument("--no-planner", action="store_true",
                        help="Skip the cognitive_os explorer + "
                             "planner; use cell_actor BFS only.")
    parser.add_argument("--no-global-priors", action="store_true",
                        help="Skip the cross-session/cross-game "
                             "action-semantic priors load+seed at "
                             "start and the save at end.  Useful "
                             "for measuring the WITHOUT-transfer "
                             "baseline.")
    parser.add_argument("--global-priors-path", type=Path,
                        default=None,
                        help="Override the default global priors "
                             "JSON path "
                             "(.tmp/global_action_priors.json).")
    args = parser.parse_args()

    if args.level and args.fixture_dir is None:
        print(f"[driver] --level {args.level}: the live game starts at level 0 "
              f"and will be FAST-FORWARDED to level {args.level} by replaying the "
              f"recorded solutions for the earlier levels (no VLM cost). If a "
              f"recorded solution is missing/stale, it falls back to full play "
              f"from level 0.", flush=True)

    from game_adapter import FixtureReplayAdapter, LiveHarnessAdapter
    if args.fixture_dir is not None:
        game = FixtureReplayAdapter(
            args.fixture_dir, game_id=args.game_id, level=args.level,
        )
    else:
        game = LiveHarnessAdapter(
            args.game_id, level=args.level,
            environments_dir=args.env_dir,
        )

    work_dir = args.work_dir / f"{args.game_id}_lc{args.level}"
    world = WorldKnowledge(game_id=args.game_id, level=args.level)
    driver = ExploratoryDriver(
        game, world, work_dir, timeout_s=args.timeout_s,
        vlm_timeout_s=args.vlm_timeout_s,
        use_strategy=not args.no_strategy,
        use_planner=not args.no_planner,
    )
    if args.no_global_priors:
        driver.global_priors_path = None
    elif args.global_priors_path is not None:
        driver.global_priors_path = args.global_priors_path

    print(f"Starting exploratory play: game={args.game_id} "
          f"level={args.level} work_dir={work_dir}")
    # For the live adapter, --level >0 fast-forwards via recorded solutions.
    # The fixture adapter replays its own frames, so it starts at its entry
    # level directly (no solution-replay seek).
    start_level = args.level if args.fixture_dir is None else 0
    reports = driver.run(max_turns=args.max_turns, start_level=start_level)
    print()
    print(f"=== END OF RUN ({len(reports)} action turns) ===")
    print(f"Final world_state: turn={world.turn} win_state={world.win_state} "
          f"entities={len(world.entities)} "
          f"groups={len(world.groups)} "
          f"relationships={len(world.relationships)} "
          f"mechanic_hypotheses={len(world.mechanic_hypotheses)}")
    print(f"WorldKnowledge saved to: "
          f"{work_dir / 'world_knowledge.json'}")


if __name__ == "__main__":
    main()
