"""Persistent, game-agnostic world-knowledge accumulator.

The ``WorldKnowledge`` object is the per-LEVEL memory: an inventory
of every entity ever observed, the relationships and groups
discovered, all actions taken, all deltas observed between
consecutive frames, and the mechanic hypotheses the actor is
building up about cause-and-effect.

Design principles:

  - GAME-AGNOSTIC: no field anywhere mentions a specific game,
    sprite, or mechanic.  All vocabularies are open ended
    (entity-name strings, action strings).

  - PERSISTENT: a single instance covers an entire LEVEL.  Turn-by-
    turn updates accumulate; nothing gets cleared mid-level.  At
    level end the rich subset (promoted hypotheses, entity
    templates, solution path) gets handed off to ``LevelMemory``
    for cross-level transfer.

  - SERIALIZABLE: ``to_json()`` / ``from_json()`` round-trip via
    plain dicts so a level's full memory can be saved and resumed.

  - CONFIDENCE-WEIGHTED: every relationship, every mechanic
    hypothesis carries a numeric credence in [0, 1].  Observations
    that survive multiple turns gain credence; contradicted ones
    decay.  This is the substrate the planner reads to decide
    whether a "rule" is firm enough to plan against.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class EntityRecord:
    """Persistent record for one entity, tracked across turns.

    ``name`` is the canonical ID used by all references (in groups,
    relationships, action deltas).  ``first_seen_turn`` / ``last_seen_turn``
    track lifespan; an entity can disappear and reappear.

    ``role_history`` records role hypotheses across turns, with
    credence — a strong way to handle "the VLM thought this was a
    wall on turn 1, then revised to floor on turn 5 after stepping
    on it without being blocked."
    """
    name: str
    first_seen_turn: int
    last_seen_turn: int
    bbox_history: list[tuple[int, list[int]]] = field(default_factory=list)
        # [(turn, bbox), ...]
    appearance: str = ""
    role_history: list[tuple[int, str, float]] = field(default_factory=list)
        # [(turn, role, credence), ...]
    cell_history: list[tuple[int, list[int]]] = field(default_factory=list)
        # [(turn, [row, col]), ...]
    notes: list[str] = field(default_factory=list)
    shape_sig: str = ""
        # rotation/scale-invariant SHAPE signature (shape_identity.shape_signature)
        # of the entity's crop -- the IDENTITY channel for matching a recurring
        # entity across levels by shape (kept separate from ROLE).
    crop_b64: str = ""
        # base64 PNG of the entity's COLOUR crop (shape_identity.color_crop_b64):
        # the actual cropped image, carried so a human/VLM can visually compare
        # identity ("keep a cropped image of the entity" -- text is lossy).

    @property
    def current_bbox(self) -> Optional[list[int]]:
        return self.bbox_history[-1][1] if self.bbox_history else None

    @property
    def current_role(self) -> Optional[str]:
        return self.role_history[-1][1] if self.role_history else None

    @property
    def current_cell(self) -> Optional[list[int]]:
        return self.cell_history[-1][1] if self.cell_history else None


@dataclass
class GroupRecord:
    name: str
    members: list[str]
    criterion: str
    evidence: str
    confidence: float
    first_seen_turn: int


@dataclass
class RelationshipRecord:
    """One observed structural relationship.  Confidence accumulates
    across turns if the same relationship is re-observed."""
    from_name: str
    to_name: str
    relation: str
    evidence: str
    confidence: float
    first_seen_turn: int
    last_seen_turn: int
    times_observed: int = 1


@dataclass
class ActionRecord:
    turn: int
    action: str
    rationale: str
    actor_chose_from: str       # e.g. "goal_directed", "exploration"
    goal_id: Optional[str] = None
    target_cell: Optional[list[int]] = None
    # Optional: full multi-step plan the planner produced, of which
    # `action` is the first step.  Empty when the action came from
    # the mechanical fallback (no symbolic plan).  Persisted so the
    # trace can show WHAT THE PLANNER INTENDED, not just the next
    # step it executed.
    full_plan_actions: list[str] = field(default_factory=list)


@dataclass
class DeltaRecord:
    """Frame-to-frame delta as reported by the VLM after one action."""
    from_turn: int
    to_turn: int
    action: str
    agent_moved: bool
    agent_new_cell: Optional[list[int]] = None
    inferred_action: Optional[str] = None
        # what the VLM thinks happened (may differ from intended `action`
        # if the move was blocked or had a side effect)
    entities_appeared: list[str] = field(default_factory=list)
    entities_disappeared: list[str] = field(default_factory=list)
    entities_changed: list[str] = field(default_factory=list)
        # entities whose appearance/bbox/role changed significantly
    summary: str = ""
    visual_events: list[dict] = field(default_factory=list)
        # Substrate-computed visual change events, surfaced per turn
        # for entities the perception layer flagged as worth
        # watching (role=hud, or `watch_internal_pixels=true` on
        # the entity).  Each event is a small open dict:
        #   { entity: str,
        #     kind: 'internal_pixel_change',
        #     pixel_diff_fraction: float,  # 0..1
        #     bbox: [r1,c1,r2,c2],
        #   }
        # Computed by visual_events.compute_internal_pixel_events.
        # GAME-AGNOSTIC: the diff is a pure function of two
        # frames + a bbox; the trigger ("which entities to
        # watch") is perception's role inference, not a
        # hardcoded entity list.
    score_increased: bool = False
        # Set by the driver when world.score rose on this turn.
        # Read by the subgoal acceptance evaluator (token
        # 'score_increased').
    win_state_changed: bool = False
        # Set by the driver when world.win_state changed on this
        # turn.  Read by token 'win_state_changed'.
    relations: list[dict] = field(default_factory=list)
        # Game-agnostic temporal visual relations between entities
        # computed by Layer A (relational_kinematics).  Each entry is
        # a RelationRecord.as_dict() with keys
        # {kind, entities, turn, direction, evidence}.  Populated by
        # ingest_delta after EntityRecord.bbox_history has the new
        # turn's bboxes.  See SPEC_visual_reasoning_substrate.md.
    animation_events: list[dict] = field(default_factory=list)
        # Substrate ENTITY-LEVEL movements detected ACROSS the action's
        # animation sub-frames (Layer-3/4: detect_entities per sub-frame +
        # object-constancy tracking).  Each entry is an
        # _substrate_animation_entities() event dict {colour_hex, size, verbs,
        # first_frame, last_frame, from, to, net, path_span_rows/cols,
        # trajectory}.  Pure substrate FACTS; the acting VLM interprets the
        # correlation.  Empty when the action produced a single settled frame.
    animation_filmstrip: Optional[str] = None
        # Filename of the rendered animation filmstrip for this turn (under the
        # turn dir) when the action animated; else None.  Lets the trace embed
        # the actual sub-frames next to the detected movements.
    animation_entities_filmstrip: Optional[str] = None
        # Filename of the PER-FRAME ENTITY ANALYSIS filmstrip -- each sub-frame
        # with the substrate's detected-entity boxes drawn on it -- so the
        # animation analysis can be SEEN and verified per frame, not trusted blind.

    def __post_init__(self):
        # The VLM may report changed/appeared/disappeared entities either as bare
        # name STRINGS or as richer DICTS like {"name": ..., "change": ...}.
        # Normalise all three to plain string names at the SOURCE so every
        # consumer (mechanic_miner, cx-OFAT, structural reasoning) can use them as
        # hashable dict/set keys without crashing on an unhashable dict.
        for _attr in ("entities_appeared", "entities_disappeared",
                      "entities_changed"):
            _norm = []
            for _it in (getattr(self, _attr) or []):
                _nm = _it.get("name") if isinstance(_it, dict) else _it
                if _nm is not None:
                    _norm.append(_nm)
            setattr(self, _attr, _norm)


@dataclass
class MechanicHypothesis:
    """A causal claim about how the game responds to an action or
    state.  Hypotheses are formed by the mechanic miner from
    DeltaRecords, then refined as evidence accumulates.

    ``trigger`` and ``effect`` are open strings (game-agnostic).
    Examples (NOT injected — discovered):
      - trigger="action=DOWN from cell with neighbor of role=collectable",
        effect="entity at that cell disappears"
      - trigger="action=CLICK on entity of role=hud_button",
        effect="agent appearance.color changes"

    ``credence`` updates as evidence accumulates:
      - +0.1 per supporting observation
      - -0.3 per contradicting observation (asymmetric: easier to
        falsify than to confirm)
    """
    hypothesis_id: str
    trigger: str
    effect: str
    credence: float
    supporting_observations: list[int]      # delta indices that support
    contradicting_observations: list[int]   # delta indices that contradict
    promoted: bool = False
        # True once credence crosses a threshold and the hypothesis
        # becomes a "rule" the planner reads with high trust
    precondition: Optional[dict] = None
        # Layer B (contrastive refutation) output.  When a hypothesis
        # has both supporting and contradicting observations and a
        # relational feature cleanly discriminates the two sets, B
        # annotates that feature here.  Shape:
        #   {"feature": [kind, direction], "correlation": "positive"|
        #     "negative", "support_presence": float (0..1),
        #     "contradict_presence": float (0..1), "discovered_at_turn": int}
        # The actor reads this as "this rule applies only when the
        # named relation is present (positive) or absent (negative)."
        # See SPEC_visual_reasoning_substrate.md.
    parent_hypothesis_id: Optional[str] = None
        # Set if this hypothesis was created by B as a precondition-
        # qualified child of another.  Currently unused (B annotates in
        # place); reserved for a future split-on-divergence policy.


@dataclass
class BlockingClaim:
    """A relational claim that an action is BLOCKED (silent / no-op)
    when the world is in a particular state-class.

    Complements ``MechanicHypothesis``: where MechanicHypothesis is
    forward (trigger -> effect), BlockingClaim is a CONSTRAINT
    (state-class -> action is forbidden / silent).  The planner and
    decomposer consume BlockingClaims to derive REMOVAL SUBGOALS:
    "to enable blocked_action, change the state so it no longer
    matches blocking_state_class."  See
    ``docs/SPEC_blocked_goal_subgoal_spawning.md`` for the role this
    plays in the impediment pipeline.

    Mined by ``mechanic_miner.mine_blocking_claims`` when the same
    (action, state_class_fingerprint) pair is observed silent ≥ N
    times.  ``blocking_state`` is a small dict of game-agnostic
    state-class features the miner can extract from WorldKnowledge:
      - "entity_roles": canonical sorted role-count string
        (e.g. "agent=1,collectable=3,engaged=1,swept-up=1")
      - "agent_row_band": qualitative band ("top"/"mid"/"bottom")
      - "arm_extent_class": qualitative arm length
        ("none"/"short"/"long")

    Game-agnostic: features are derived from generic WorldKnowledge
    primitives (entity roles, bboxes, grid_inference).  Adapters can
    add additional features by extending the extractor without
    changing the substrate.
    """
    claim_id: str
    blocked_action: str
    blocking_state: dict[str, str]
    credence: float
    supporting_observations: list[int]
        # delta indices where action was silent in matching state-class
    contradicting_observations: list[int] = field(default_factory=list)
        # delta indices where action had an effect despite matching
        # state-class — falsifies the claim
    promoted: bool = False
        # True once credence crosses threshold; the planner then
        # treats this state-class as a hard precondition violation
        # for the blocked_action.


@dataclass
class ProbeRecord:
    """A single logged experiment.

    Every probe explicitly identifies AT LEAST TWO competing
    hypotheses and a prediction per hypothesis for the probe's
    action sequence.  When the probe executes, the substrate
    compares the observed outcome to each prediction and updates
    the matching / mismatching hypothesis credences automatically.

    Why this discipline matters: single-hypothesis "let me try X
    and see what happens" probes don't reduce uncertainty, they
    just generate data.  Forcing the actor to enumerate ≥2
    hypotheses with differing predictions per probe turns each
    action into a real experiment.

    Status lifecycle:
      pending     -- committed but not yet executed
      executed    -- action taken; outcome recorded
      inconclusive -- executed but observation didn't discriminate
                      among hypotheses (all predictions equally
                      consistent or no measurable signal)
      resolved    -- executed and one hypothesis was uniquely
                      supported / contradicted
      abandoned   -- actor explicitly dropped the probe
    """
    probe_id: str
    motivating_uncertainty: str
        # free-form: what gap in understanding this probe targets
    motivating_hypothesis_ids: list[str]
        # ids of the ≥2 WinConditionHypotheses or MechanicHypotheses
        # this probe discriminates among.  Strategy prompt rejects
        # single-id probes — they're not experiments.
    action_or_sequence: list[str]
        # one or more actions; for multi-step probes, the actor
        # commits the full sequence up front.
    predicted_outcomes: dict
        # mapping hypothesis_id -> free-form prediction text.
        # The actor MUST predict per hypothesis BEFORE executing,
        # so the outcome can be compared against each prediction.
    status: str = "pending"
        # "pending" | "executed" | "inconclusive" | "resolved" |
        # "abandoned"
    proposed_at_turn: int = 0
    executed_at_turn: Optional[int] = None
    observed_at_delta_index: Optional[int] = None
    observed_outcome: str = ""
        # free-form description of what actually happened, as the
        # actor reported when applying the probe observation.
    notes: str = ""


@dataclass
class WinConditionHypothesis:
    """The actor's current hypothesis about what triggers a score
    advance, lc-advance, or win_state change in the game.

    Substrate-side discipline: ``game_purpose_guess`` (the
    perception layer's initial guess) is just one entry here, at
    low starting credence.  Every action's outcome should
    correlate with predictions some hypothesis makes — score
    advances are SUPPORTING observations for hypotheses that
    predicted advance, CONTRADICTING for those that didn't.

    Open-vocabulary by design.  The actor authors each hypothesis
    as free-form text; the substrate holds and ranks.  Multiple
    hypotheses can co-exist (one per credible alternative);
    actor's job is to design probes that discriminate among them.

    Lifecycle:
      - Seeded at trial start from perception's
        ``game_purpose_guess`` (low credence).
      - New ones added by the actor via the strategy reply's
        ``commit_win_condition_hypothesis`` field.
      - Credence updates: +bump on supporting observation
        (predicted outcome matched actual signal), -decay on
        contradicting observation.
      - PROMOTED once credence >= threshold; the actor can then
        derive concrete delivery subgoals against it with high
        confidence.
    """
    hypothesis_id: str
    description: str           # free-form actor text, e.g.
                                # "<condition over entities> advances
                                #  the score"
    credence: float
    supporting_observations: list[int] = field(default_factory=list)
        # delta indices where prediction matched signal
    contradicting_observations: list[int] = field(default_factory=list)
        # delta indices where prediction was wrong (e.g. predicted
        # advance, score didn't change)
    promoted: bool = False
        # True once credence crosses threshold; actor's delivery
        # subgoals may now reference this hypothesis as foundation.
    created_at_turn: int = 0
    notes: str = ""            # actor's free-form notes
    win_relation: Optional[dict] = None
        # Crystallized, CHECKABLE form of the win condition (the
        # representation bridge between writer and reader — see
        # SPEC_cumulative_learning_loop.md § Crystallization and
        # SPEC_goal_grounding_and_state_diff.md § Substrate-computed
        # goal gap).  Open schema; the seed type is:
        #   {"type": "ordered_match", "roles": [roleA, roleB],
        #    "axis": "col"|"row", "trigger": <optional action/relation>}
        # Keyed on ROLES (not entity names) so it transfers across
        # level/game variants.  Populated by
        # knowledge_crystallization.derive_win_condition (event-
        # triggered credit assignment); evaluated each turn by
        # evaluate_win_relation to produce the goal gap.  None until
        # derived.


@dataclass
class GridInferenceRecord:
    is_grid_based: bool
    cell_ticks: Optional[int]
    origin_ticks: Optional[list[int]]
    rows: Optional[int]
    cols: Optional[int]
    confidence: float
    locked: bool = False
        # locked=True once the actor has observed enough successful
        # moves to confirm the cell pitch; future re-perception
        # cannot downgrade it without strong contradictory evidence.


@dataclass
class ActiveSubgoal:
    """A subgoal the strategy actor has explicitly committed to.

    Substrate role: be a DURABLE CONTAINER for the actor's
    commitments.  The substrate does NOT classify or interpret —
    every descriptive field is free-form text written by the
    actor.  The substrate just remembers what the actor said,
    surfaces it on subsequent turns, and forces the actor to
    consciously close or abandon it.

    This is the ``rationale``-as-text-only gap fix: turning a
    transient prose field into a persistent named commitment so
    the next turn's actor can see it AND must act on it.

    Subgoals can nest via ``parent_id``: an actor working toward
    "move <entity> to <cell>" can spawn a sub-subgoal "verify the
    effect once <entity> arrives" under it.

    A subgoal's lifecycle:
      open      -- actor commits via the strategy reply's
                   `commit_subgoal` slot.
      active    -- surfaced in the prompt on every subsequent turn
                   until closed.
      achieved  -- actor sets `subgoal_status_update.status =
                   "achieved"` when the expected_outcome is now
                   observed.
      abandoned -- actor sets status="abandoned" with a notes
                   reason; the substrate stops surfacing it but
                   keeps the record for post-hoc analysis.
      blocked   -- actor sets status="blocked"; substrate keeps
                   surfacing it but flags blocked status.

    No closed-vocabulary preconditions, no machine-checked
    expected_outcome.  The next-turn actor reads the free-form
    description and judges.
    """
    subgoal_id:        str            # auto-generated
    name:              str            # actor-given, free-form
    problem_solved:    str            # why this subgoal exists
                                        # (goal / obstacle / constraint
                                        # it addresses), free-form
    expected_outcome:  str            # what should be true when
                                        # the subgoal closes (free-form)
    parent_id:         Optional[str] = None
                                        # id of an enclosing subgoal,
                                        # if any
    created_at_turn:   int = 0
    status:            str = "active"
        # "active" | "achieved" | "inferred_satisfied" |
        # "abandoned" | "blocked" | "invalidated"
        # "inferred_satisfied" is the substrate downgrade for
        # status="achieved" without a confirming_signal — the
        # actor THINKS it's done but cited no observed evidence.
        # Stays in the open set; the prompt prods the actor to
        # find a confirming signal or re-classify.
        # "invalidated" is a SUBSTRATE-AUTOMATIC transition (not an
        # actor choice): the subgoal's premise evaporated — its
        # referenced entity is gone, the win-condition hypothesis it
        # served was refuted, its parent closed by another route, or
        # an OR-group sibling was achieved.  Distinct from
        # "abandoned" (actor gave up on a still-valid goal).  Leaves
        # the open set; never subject to the completion/exhaustion
        # gate.
    closed_at_turn:    Optional[int] = None
    notes:             str = ""        # actor's running notes,
                                        # appended on status changes
    related_subroutine_id: Optional[str] = None
        # if the actor is applying a stored subroutine in service
        # of this subgoal, the subroutine_id
    forward_simulation: str = ""
        # Free-form mental simulation the actor wrote when
        # committing this subgoal: a step-by-step trace of expected
        # world states under the planned action sequence,
        # grounded in promoted MechanicHypotheses + physics priors.
        # Substrate holds the text; the actor's discipline is to
        # write it before committing so the plan isn't hand-wavy.
        # See vlm_strategy.STRATEGY_PROMPT for the discipline.
    derived_from: str = ""
        # Free-form text linking this subgoal to its source:
        #   "from win_condition_hypothesis: <id>" (top-level), or
        #   "to advance parent <id>: <reasoning>" (child)
        # Forces the actor to name what this subgoal serves; no
        # orphan tactics survive the round-trip.
    win_condition_hypothesis_id: Optional[str] = None
        # If this is a top-level delivery / goal subgoal, points
        # at the WinConditionHypothesis it serves.  When the
        # underlying hypothesis is contradicted, every subgoal
        # carrying its id is flagged as "premise under question."
    confirming_signal: str = ""
        # Free-form citation of the signal that confirms the
        # subgoal's expected_outcome.  REQUIRED when status
        # transitions to "achieved" — without it, the substrate
        # downgrades the transition to "inferred_satisfied" and
        # keeps the subgoal in the open set.  Format examples
        # (entity names are placeholders, sourced from perception):
        #   "delta[N] entities_disappeared: ['<entity>']"
        #   "visual_event entity=<indicator> on delta[N]"
        #   "score advanced from X to Y at turn N"
        #   "win_state changed to 'won' at turn N"
    depends_on: list = field(default_factory=list)
        # List of subgoal_ids (strings) that must reach status
        # ACHIEVED before this subgoal becomes ACTIONABLE.  Bridged
        # to a cognitive_os Goal with DepAll(DepRef(...)) so the
        # engine's is_actionable check gates pursuit.  Empty list
        # (default) means the subgoal is unconditionally
        # actionable.  See subgoal_forest_bridge.py.
    # --- Subgoal Completion Contract (2026-05-29) -----------------------
    acceptance_check: str = ""
        # Closed-vocab predicate the SUBSTRATE evaluates against each
        # DeltaRecord to decide achievement objectively (component A).
        # Grammar: space-free tokens joined by " OR " / " AND ".
        # Tokens (entity names sourced from perception, never hardcoded):
        #   entity_changed:<name> | entity_appeared:<name> |
        #   entity_disappeared:<name> | visual_event:<name> |
        #   score_increased | win_state_changed |
        #   agent_at_cell:[r,c]
        # When the predicate fires, the substrate auto-sets
        # status="achieved" and fills confirming_signal with the
        # delta reference.  Empty string = no substrate acceptance
        # test (actor must close it manually with a cited signal).
    references_entities: list = field(default_factory=list)
        # Entity names this subgoal presupposes exist.  If ALL are
        # gone for a multi-frame persistence window, the substrate
        # auto-invalidates (premise: referenced-entity-gone).  Uses
        # last_seen_turn, NOT single-frame absence (occlusion !=
        # disappearance).
    approaches_tried: list = field(default_factory=list)
        # Component B exhaustion ledger.  Each entry a [action,
        # state_class] pair recorded by the driver on every turn this
        # subgoal was the pursued one.  Surfaced as tried/untried so
        # the actor varies approach instead of repeating a dead action
        # or abandoning after one try.
    or_group: Optional[str] = None
        # OR-achievement group id.  Subgoals sharing an or_group are
        # an "achieve ONE of" set: when ANY member reaches achieved,
        # the substrate auto-invalidates the other open members
        # (reason: superseded-by-or-sibling).  Distinct from
        # depends_on (which is AND-precondition gating).
    impediment: str = ""
        # Set when status="blocked": the actor's free-form description
        # of WHAT blocks progress.  Required for the blocked
        # transition (component C); the substrate spawns a removal
        # subgoal from it and links this subgoal via depends_on.


# ---------------------------------------------------------------------------
# WorldKnowledge — the per-level container
# ---------------------------------------------------------------------------


@dataclass
class WorldKnowledge:
    game_id: str
    level: int
    turn: int = 0
    win_state: str = "playing"   # playing / won / lost / unknown
    lives: Optional[int] = None
    score: Optional[int] = None

    entities:       dict[str, EntityRecord]       = field(default_factory=dict)
    groups:         dict[str, GroupRecord]        = field(default_factory=dict)
    relationships:  list[RelationshipRecord]      = field(default_factory=list)
    grid_inference: Optional[GridInferenceRecord] = None

    actions_taken:  list[ActionRecord]            = field(default_factory=list)
    deltas_observed: list[DeltaRecord]            = field(default_factory=list)
    mechanic_hypotheses: list[MechanicHypothesis] = field(default_factory=list)
    blocking_claims: list[BlockingClaim]          = field(default_factory=list)
        # Constraint claims of form "action A is silent in state-class S"
        # — mined when the same (action, state_class) pair is observed
        # silent ≥ N times.  Drives backward-chaining subgoal derivation
        # in the planner ("to enable A, leave state-class S").
    active_subgoals: list[ActiveSubgoal]          = field(default_factory=list)
        # Durable container for the actor's explicit subgoal
        # commitments.  See ActiveSubgoal docstring and
        # docs/SPEC_active_subgoals.md.
    win_condition_hypotheses: list[WinConditionHypothesis] = field(default_factory=list)
        # The actor's hypotheses about what triggers score / lc /
        # win_state changes.  ``game_purpose_guess`` is seeded as
        # an initial low-credence entry; actor authors alternatives
        # and the substrate ranks by credence.  See
        # WinConditionHypothesis docstring.
    probes: list[ProbeRecord] = field(default_factory=list)
        # The probe ledger — every committed experiment.  See
        # ProbeRecord docstring.  Drives systematic probing
        # discipline: pending probes surface in the strategy
        # prompt; executed probes update hypothesis credences
        # automatically from prediction-vs-observation comparison.
    probe_state: dict = field(default_factory=dict)
        # Layer D — capability-probing bookkeeping.  Keys:
        #   "undo_capability": "lossless"|"destructive"|"unavailable"|None
        #     (set once per trial by calibrate_undo)
        #   "burst_start_turn": int (turn the opening burst began)
        #   "covered_cells": list[str] of "{action}@@{state_class}"
        #     action-effect cells already observed (for coverage %)
        # See SPEC_visual_reasoning_substrate.md (Layer D).
    inverse_actions: dict = field(default_factory=dict)
        # Layer C — recoverability verdicts keyed by a composite
        # "{action}@@{state_class}" string.  Each value is a dict:
        #   {"verdict": "reversible"|"irreversible"|
        #     "progress_destructive"|"undo_unavailable",
        #    "action": str, "state_class": str,
        #    "undo_action": str, "evidence": dict,
        #    "discovered_at_turn": int, "confirmations": int}
        # Populated by recoverability.probe_recoverability; read by
        # recoverability.format_recoverability_vetoes for the strategy
        # prompt.  See SPEC_visual_reasoning_substrate.md.

    # Free-form game-type / purpose guesses (refined over turns)
    game_type_guess: str = ""
    game_purpose_guess: str = ""

    # Per-turn `overall_notes` from each perception call.  The
    # perception VLM uses this field to capture its narrative
    # understanding — what changed, what the next plan is, what
    # hypothesis to test.  Stored per-turn (not overwritten) so the
    # trace can render the system's evolving reasoning.
    perception_notes_by_turn: list[list] = field(default_factory=list)
        # list of [turn:int, notes:str].  Stored as list (not tuple)
        # for clean JSON round-trip.

    # Cross-level inheritance — when this level was started fresh, this
    # is None; when resuming from a previous level's promoted rules,
    # this carries the inherited knowledge tag (e.g. "lc=0:promoted").
    inherited_from: Optional[str] = None

    # ------------------------------------------------------------------
    # LEVEL vs CROSS-LEVEL state partition — the SINGLE source of truth.
    #
    # A level advance resets the board to a NEW layout, so state tied to
    # THIS level's layout must be dropped: otherwise the prior level's
    # positioned entities / grid leak forward and the planner probes GHOST
    # coordinates (the bug this fixes).  But state that is LEARNED KNOWLEDGE
    # about the game — mechanics, win conditions, recoverability, durable
    # subgoals, capability calibration — MUST persist: that knowledge is the
    # entire benefit of having solved the previous level; dropping it forces
    # COS to re-discover the game from scratch every level.
    #
    # Historically each subsystem cleared its own slice in its own place,
    # so every newly-added piece of level-scoped state became a fresh leak.
    # Instead, EVERY dataclass field is classified into exactly one set
    # below, `reset_for_new_level` drops only the level-scoped set, and
    # `test_level_scoped_reset` asserts the union is exhaustive + disjoint —
    # so adding a field without classifying it FAILS the test (it can neither
    # silently leak nor silently drop knowledge).  These are plain class
    # attributes (un-annotated), so the dataclass machinery ignores them.
    # ------------------------------------------------------------------
    _LEVEL_SCOPED_FIELDS = (
        "entities",        # positioned identities of THIS level's pieces
        "groups",          # observation-time groupings of those pieces
        "relationships",   # relations between positioned pieces
        "grid_inference",  # THIS level's grid origin / cell size / extent
    )
    _CROSS_LEVEL_FIELDS = (
        # identity / live status (refreshed by the harness each step)
        "game_id", "level", "turn", "win_state", "lives", "score",
        # cumulative trial history (append-only logs; sliced per level by index)
        "actions_taken", "deltas_observed", "perception_notes_by_turn",
        # LEARNED KNOWLEDGE — the benefit of solving prior levels, preserved
        "mechanic_hypotheses", "blocking_claims", "win_condition_hypotheses",
        "active_subgoals", "probes", "inverse_actions",
        "game_type_guess", "game_purpose_guess", "inherited_from",
        # preserved as a whole, but its level-scoped keys are pruned in
        # reset_for_new_level (see _PROBE_STATE_CROSS_LEVEL_KEYS)
        "probe_state",
    )
    # probe_state mixes level-scoped exploration scratch (coverage cells, the
    # controlled-experiment 'cx' state, the opening-burst marker) with
    # GAME-scoped capability findings.  Reset the former; KEEP the latter so
    # the next level need not re-calibrate (e.g. whether ACTION7 is a lossless
    # undo is a property of the engine, stable across levels).
    _PROBE_STATE_CROSS_LEVEL_KEYS = ("undo_capability",)

    # Level-scoped scratch attached to the world DYNAMICALLY (via setattr, not
    # declared fields) — so the exhaustiveness guard above cannot see them.
    # PREFER declaring new level-scoped scratch as a real field (then it is
    # guard-covered + reset automatically); anything that must stay a dynamic
    # attr is listed here so it is still reset.  (`_vis_state` is intentionally
    # NOT here: it is owned by visual_events, which self-invalidates its
    # perception baselines whenever the level/score changes.)
    _LEVEL_SCOPED_DYNAMIC_ATTRS = ("_pending_operator", "_last_prediction")

    def reset_for_new_level(self) -> list:
        """Drop level-scoped (layout-tied) state at a level boundary while
        PRESERVING all cross-level knowledge.  The single authoritative
        world-state reset for a level transition; returns the names of the
        fields actually reset (for logging / tests).
        """
        import dataclasses
        spec = {f.name: f for f in dataclasses.fields(self)}
        for name in self._LEVEL_SCOPED_FIELDS:
            f = spec[name]
            if f.default_factory is not dataclasses.MISSING:   # dict / list
                setattr(self, name, f.default_factory())
            else:                                              # scalar default
                setattr(self, name, f.default)
        # prune level-scoped probe_state keys; keep game-scoped capability
        self.probe_state = {
            k: v for k, v in self.probe_state.items()
            if k in self._PROBE_STATE_CROSS_LEVEL_KEYS
        }
        # clear level-scoped dynamic scratch (reads are getattr(...None)-safe)
        cleared_dyn = []
        for attr in self._LEVEL_SCOPED_DYNAMIC_ATTRS:
            if getattr(self, attr, None) is not None:
                setattr(self, attr, None)
                cleared_dyn.append(attr)
        return (list(self._LEVEL_SCOPED_FIELDS) + ["probe_state(pruned)"]
                + cleared_dyn)

    # ------------------------------------------------------------------
    # Update operations called by the ExploratoryDriver
    # ------------------------------------------------------------------

    def ingest_perception(self, perception_json: dict) -> None:
        """Fold a per-turn perception output (entities + groups +
        relationships + grid_inference + symbolic_state) into the
        persistent record.  Existing entities get UPDATED with new
        bboxes/roles; new entities get APPENDED.  Re-observed
        relationships get their credence and times_observed bumped."""
        turn = self.turn
        # Entities — match by name; append to history
        for e in perception_json.get("entities") or []:
            name = e.get("name", "")
            if not name:
                continue
            bbox = e.get("bbox_ticks_turn1") or e.get("bbox_ticks")
            role = e.get("role_hypothesis", "unknown")
            conf = _confidence_to_float(e.get("confidence"))
            cell = e.get("cell")  # optional — symbolic_state mapping
            rec = self.entities.get(name)
            if rec is None:
                rec = EntityRecord(
                    name=name, first_seen_turn=turn, last_seen_turn=turn,
                    appearance=e.get("appearance", ""),
                )
                self.entities[name] = rec
            rec.last_seen_turn = turn
            if e.get("shape_sig"):                 # carry the shape-identity signature
                rec.shape_sig = e["shape_sig"]
            if e.get("crop_b64"):                  # carry the colour crop image
                rec.crop_b64 = e["crop_b64"]
            if bbox is not None:
                rec.bbox_history.append((turn, list(bbox)))
            # Preserve a known prior role rather than overwriting with
            # "unknown" or null.  Delta-perception VLM calls
            # frequently emit `role_hypothesis=null` for unchanged
            # entities (meaning "still there, no role change");
            # downgrading them loses the actor's ability to find the
            # agent on subsequent turns.
            if role and role.lower() != "unknown":
                rec.role_history.append((turn, role, conf))
            elif not rec.role_history and role:
                rec.role_history.append((turn, role, conf))
            if cell is not None:
                rec.cell_history.append((turn, list(cell)))

        # Groups — overwrite-by-name (groups are observation-time,
        # not persistent identities the way entities are)
        for g in perception_json.get("groups") or []:
            name = g.get("name", "")
            if not name:
                continue
            self.groups[name] = GroupRecord(
                name=name,
                members=list(g.get("members") or []),
                criterion=g.get("criterion", ""),
                evidence=g.get("evidence", ""),
                confidence=_confidence_to_float(g.get("confidence")),
                first_seen_turn=self.groups[name].first_seen_turn
                                  if name in self.groups else turn,
            )

        # Relationships — match (from, to, relation); bump confidence
        # and times_observed if re-observed.
        existing_keys = {
            (r.from_name, r.to_name, r.relation): r
            for r in self.relationships
        }
        for rel in perception_json.get("relationships") or []:
            # Relationships may be free-form STRINGS (a perfectly valid VLM
            # output, e.g. "aim_line connects mover -> ball") rather than
            # {from,to,relation} dicts.  Those carry no structured key to match
            # on, so skip them instead of crashing the whole turn on .get().
            if not isinstance(rel, dict):
                continue
            key = (rel.get("from"), rel.get("to"), rel.get("relation"))
            if any(k is None for k in key):
                continue
            r = existing_keys.get(key)
            if r is None:
                r = RelationshipRecord(
                    from_name=key[0], to_name=key[1], relation=key[2],
                    evidence=rel.get("evidence", ""),
                    confidence=_confidence_to_float(rel.get("confidence")),
                    first_seen_turn=turn, last_seen_turn=turn,
                )
                self.relationships.append(r)
                existing_keys[key] = r
            else:
                r.last_seen_turn = turn
                r.times_observed += 1
                r.confidence = min(1.0, r.confidence + 0.05)
                # only overwrite evidence if newer is more specific
                if (rel.get("evidence")
                        and len(rel["evidence"]) > len(r.evidence)):
                    r.evidence = rel["evidence"]

        # Grid inference — keep the highest-confidence (or locked)
        gi = perception_json.get("grid_inference") or {}
        if gi:
            new_gi = GridInferenceRecord(
                is_grid_based=bool(gi.get("is_grid_based")),
                cell_ticks=gi.get("cell_ticks"),
                origin_ticks=gi.get("origin_ticks"),
                rows=gi.get("rows"), cols=gi.get("cols"),
                confidence=_confidence_to_float(gi.get("confidence")),
            )
            if self.grid_inference is None or (
                not self.grid_inference.locked
                and new_gi.confidence >= self.grid_inference.confidence
            ):
                self.grid_inference = new_gi

        # Top-level guesses — refine on each turn (refinement passes
        # tend to produce more specific phrases).  VLM replies have
        # been observed to emit `game_type` as EITHER a dict
        # ({guess, evidence, confidence}) OR a bare string; handle
        # both defensively.
        gt_raw = perception_json.get("game_type")
        gt = (gt_raw.get("guess") if isinstance(gt_raw, dict)
              else gt_raw if isinstance(gt_raw, str) else None)
        if gt:
            self.game_type_guess = gt
        gp_raw = perception_json.get("game_purpose")
        gp = (gp_raw.get("guess") if isinstance(gp_raw, dict)
              else gp_raw if isinstance(gp_raw, str) else None)
        if gp:
            self.game_purpose_guess = gp

        # Capture this turn's free-form perception notes.  If the
        # refinement pass re-emits notes for the same turn, replace
        # the prior entry rather than appending a duplicate (so the
        # final stored note is the post-refinement view).
        notes = (perception_json.get("overall_notes") or "").strip()
        if notes:
            self.perception_notes_by_turn = [
                entry for entry in self.perception_notes_by_turn
                if entry[0] != turn
            ]
            self.perception_notes_by_turn.append([turn, notes])

        # symbolic_state — fold agent_cell + entity_cells into the
        # entity records' cell_history.  The VLM produces these as
        # separate top-level fields rather than per-entity `cell`
        # values, so we have to do the projection here.
        ss = perception_json.get("symbolic_state") or {}
        agent_cell = ss.get("agent_cell")
        if agent_cell:
            agent_rec = self._find_agent()
            if agent_rec is not None:
                agent_rec.cell_history.append((turn, list(agent_cell)))
        for ent_name, cell in (ss.get("entity_cells") or {}).items():
            rec = self.entities.get(ent_name)
            if rec is None or cell is None:
                continue
            # entity_cells values can be a single [r, c] or a list of
            # cells [[r1, c1], [r2, c2], ...] — for the per-entity
            # cell_history we only record the centroid (single cell).
            if (isinstance(cell, list) and cell
                    and isinstance(cell[0], list)):
                rs = [c[0] for c in cell]
                cs = [c[1] for c in cell]
                cell = [int(sum(rs) / len(rs)),
                        int(sum(cs) / len(cs))]
            rec.cell_history.append((turn, list(cell)))

    def record_action(self, action_record: ActionRecord) -> None:
        self.actions_taken.append(action_record)

    def ingest_delta(self, delta: DeltaRecord) -> None:
        self.deltas_observed.append(delta)
        # Mark disappeared entities as no longer "current"
        for name in delta.entities_disappeared:
            rec = self.entities.get(name)
            if rec is not None:
                rec.notes.append(
                    f"disappeared at turn {delta.to_turn} after action "
                    f"{delta.action}"
                )
        # Propagate agent's new position from the delta into the
        # agent entity's cell_history.  The delta is the authoritative
        # source for "what happened" between two frames; the
        # perception's symbolic_state often omits agent_cell when the
        # VLM is doing a refresh rather than a full re-extraction.
        if delta.agent_moved and delta.agent_new_cell:
            agent_rec = self._find_agent()
            if agent_rec is not None:
                agent_rec.cell_history.append(
                    (delta.to_turn, list(delta.agent_new_cell))
                )
        # Layer A — relational kinematics.  Compute temporal visual
        # relations between entities (co_displacement, motion_blocked,
        # penetration, support_relation, motion_arrested_at) from
        # already-populated bbox_history and attach to the delta.
        # Lazy import to avoid a circular dependency (the relational
        # kinematics module imports WorldKnowledge).
        try:
            from relational_kinematics import (                   # noqa: E402
                compute_relations_for_turn,
            )
            rels = compute_relations_for_turn(
                self, delta.from_turn, delta.to_turn, action=delta.action,
            )
            delta.relations = [r.as_dict() for r in rels]
        except Exception as e:
            # Never let A's failure block delta ingestion — the rest of
            # the substrate must work even if relations are unavailable.
            delta.relations = []
            print(f"[world] relational kinematics failed at "
                  f"t{delta.from_turn}->t{delta.to_turn}: {e}; "
                  f"continuing without relations")

    # ------------------------------------------------------------------
    # Symbolic-first export — what the VLM/actor sees when reasoning
    # without the image
    # ------------------------------------------------------------------

    def symbolic_snapshot(self) -> dict:
        """Compact, JSON-serializable view of the world the actor can
        reason from WITHOUT needing the image.  Used by the symbolic-
        first VLM call as the primary input; the image is only
        supplemented when the symbolic snapshot is insufficient."""
        agent = self._find_agent()
        return {
            "game_id": self.game_id,
            "level": self.level,
            "turn": self.turn,
            "win_state": self.win_state,
            "lives": self.lives,
            "score": self.score,
            "game_type_guess": self.game_type_guess,
            "game_purpose_guess": self.game_purpose_guess,
            "grid_inference": (
                asdict(self.grid_inference) if self.grid_inference else None
            ),
            "agent": (
                {
                    "name": agent.name,
                    "current_cell": agent.current_cell,
                    "role_history_tail":
                        agent.role_history[-3:] if agent.role_history else [],
                }
                if agent else None
            ),
            # Coordinates are DEMOTED: raw tick bboxes are deliberately
            # omitted from the actor-facing snapshot.  Spatial structure
            # is carried by the relational layer (same_row / same_col /
            # ordered_along / clearance / support_relation / co_displacement)
            # surfaced in the ground-truth block, with the coarse grid
            # `current_cell` kept as the only positional anchor.  Reasoning
            # over relations is far less error-prone than over tick
            # arithmetic.  See SPEC_visual_reasoning_substrate.md.
            "entity_inventory_count": len(self.entities),
            "entities": [
                {
                    "name": r.name,
                    "current_role": r.current_role,
                    "current_cell": r.current_cell,
                    "first_seen_turn": r.first_seen_turn,
                    "last_seen_turn": r.last_seen_turn,
                    "still_present":
                        r.last_seen_turn == self.turn,
                    "appearance": r.appearance,
                    "notes": r.notes[-3:],
                }
                for r in self.entities.values()
            ],
            "groups": [asdict(g) for g in self.groups.values()],
            "relationships": [
                {
                    "from": r.from_name, "to": r.to_name,
                    "relation": r.relation, "evidence": r.evidence,
                    "confidence": r.confidence,
                    "times_observed": r.times_observed,
                }
                for r in self.relationships
            ],
            "actions_taken_tail": [
                asdict(a) for a in self.actions_taken[-10:]
            ],
            "deltas_observed_tail": [
                # Strip the verbose `relations` list from the JSON tail:
                # Layer A's relations are surfaced separately (and
                # capped) via _format_ground_truth's TEMPORAL RELATIONS
                # block, so dumping the full per-delta relation records
                # here would just duplicate them at ~3.9k tokens/call.
                # The records remain on the DeltaRecord for B/C to read
                # programmatically; they simply don't go in the prompt.
                {k: v for k, v in asdict(d).items() if k != "relations"}
                for d in self.deltas_observed[-5:]
            ],
            "mechanic_hypotheses": [
                asdict(h) for h in self.mechanic_hypotheses
                if h.credence >= 0.2
            ],
        }

    def _find_agent(self) -> Optional[EntityRecord]:
        for r in self.entities.values():
            if r.current_role == "agent":
                return r
        return None

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def to_json(self) -> dict:
        d = asdict(self)
        # asdict handles dataclasses recursively; nothing extra needed
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_json(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "WorldKnowledge":
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, d: dict) -> "WorldKnowledge":
        ents = {
            n: EntityRecord(**rd) for n, rd in (d.get("entities") or {}).items()
        }
        grps = {
            n: GroupRecord(**rd) for n, rd in (d.get("groups") or {}).items()
        }
        rels = [RelationshipRecord(**rd) for rd in d.get("relationships") or []]
        acts = [ActionRecord(**rd) for rd in d.get("actions_taken") or []]
        deltas = [DeltaRecord(**rd) for rd in d.get("deltas_observed") or []]
        hyps = [MechanicHypothesis(**rd) for rd in d.get("mechanic_hypotheses") or []]
        bclaims = [BlockingClaim(**rd) for rd in d.get("blocking_claims") or []]
        asubs = [ActiveSubgoal(**rd) for rd in d.get("active_subgoals") or []]
        wcs = [WinConditionHypothesis(**rd)
               for rd in d.get("win_condition_hypotheses") or []]
        probes = [ProbeRecord(**rd) for rd in d.get("probes") or []]
        gi = d.get("grid_inference")
        gi_rec = GridInferenceRecord(**gi) if gi else None
        return cls(
            game_id=d["game_id"], level=d["level"],
            turn=d.get("turn", 0),
            win_state=d.get("win_state", "playing"),
            lives=d.get("lives"), score=d.get("score"),
            entities=ents, groups=grps, relationships=rels,
            grid_inference=gi_rec,
            actions_taken=acts, deltas_observed=deltas,
            mechanic_hypotheses=hyps,
            blocking_claims=bclaims,
            active_subgoals=asubs,
            win_condition_hypotheses=wcs,
            probes=probes,
            game_type_guess=d.get("game_type_guess", ""),
            game_purpose_guess=d.get("game_purpose_guess", ""),
            perception_notes_by_turn=[
                list(entry) for entry in
                (d.get("perception_notes_by_turn") or [])
            ],
            inherited_from=d.get("inherited_from"),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _confidence_to_float(c: Any) -> float:
    """Map 'low'/'medium'/'high' or a numeric value to [0, 1]."""
    if isinstance(c, (int, float)):
        return max(0.0, min(1.0, float(c)))
    if isinstance(c, str):
        return {"low": 0.33, "medium": 0.66, "high": 0.9}.get(
            c.lower(), 0.5
        )
    return 0.5
