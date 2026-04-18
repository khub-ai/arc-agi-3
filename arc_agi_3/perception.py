"""Frame → symbolic :class:`Observation` translation.

The engine is told nothing about pixels.  Perception's job is to
turn the ARC-AGI-3 frame (a 60×60 integer palette grid plus state
metadata) into a typed :class:`Observation` populated with:

* ``events`` — ``EntityAppeared`` / ``EntityDisappeared`` /
  ``EntityStateChanged`` / ``AgentMoved`` / ``AgentDied`` /
  ``GoalConditionMet`` / ``SurpriseEvent``.
* ``entity_snapshots`` — per-component property bag for every
  tracked region.
* ``agent_state`` — a thin dict recording the state-machine status
  ("PLAYING" / "WIN" / "GAME_OVER"), levels completed, and step
  number.  The engine uses these as evidence but does not interpret
  them semantically.

What perception does NOT do:

* It does not decide which component is the agent.  That is a
  claim the engine forms from behavioural evidence (which region
  moves in response to inputs).
* It does not assign semantic labels (wall / coin / door).  Those
  are hypotheses the engine generates via miners.
* It does not compute plans or goals.

Keeping perception strictly bottom-up ensures that every top-down
assertion in the system is an auditable hypothesis, not a hidden
assumption baked into the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from cognitive_os import (
    AgentDied,
    EntityAppeared,
    EntityDisappeared,
    EntityStateChanged,
    EntityVisualPatternChanged,
    Event,
    GoalConditionMet,
    Observation,
    SurpriseEvent,
)

from .tools.components import Region, extract_regions
from .tools.diff import cell_diff, is_identical, motion_vectors

Grid = Sequence[Sequence[int]]


# ARC-AGI-3 state-name strings emitted by the SDK.  We don't branch
# on them for semantics; we only use them to decide when to emit
# terminal events (AgentDied / GoalConditionMet).
_STATE_WIN       = "WIN"
_STATE_GAME_OVER = "GAME_OVER"
_STATE_PLAYING   = "PLAYING"


@dataclass
class PerceptionState:
    """Between-step state held by the adapter.

    Perception is stateful only in the bookkeeping sense: it needs
    the previous frame to compute diffs, and it needs a stable
    mapping from "region found in frame N" to "region found in
    frame N+1" so entity identity survives motion.

    Identity tracking is intentionally minimal here: components are
    matched across frames by ``(colour, normalised_shape)``.  This is
    the same criterion the :func:`motion_vectors` tool uses, so the
    two are guaranteed consistent.  Ambiguous matches fall back to
    treating the previous region as disappeared and the new one as
    appeared — the engine's miners then have to re-establish the
    link from behaviour, which is the correct epistemic stance.
    """

    prev_frame:           Optional[Grid]                = None
    # Stable entity id per (colour, normalised_shape) key observed
    # in the previous frame.  Reset at episode boundary.
    entity_ids_by_key:    Dict[Tuple[int, Tuple[Tuple[int, int], ...]], str] = field(default_factory=dict)
    prev_levels_completed: int                          = 0
    prev_state_name:      str                           = _STATE_PLAYING
    step_counter:         int                           = 0
    next_entity_seq:      int                           = 0

    def reset_for_new_episode(self) -> None:
        self.prev_frame            = None
        self.entity_ids_by_key     = {}
        self.prev_levels_completed = 0
        self.prev_state_name       = _STATE_PLAYING
        self.step_counter          = 0
        self.next_entity_seq       = 0


def build_observation(
    *,
    frame:            Grid,
    state_name:       str,
    levels_completed: int,
    state:            PerceptionState,
    background:       int = 0,
) -> Observation:
    """Produce an :class:`Observation` from a raw frame plus metadata.

    Mutates ``state`` (bumps the step counter, updates the identity
    map, stores the frame as the new ``prev_frame``).
    """
    state.step_counter += 1
    current_step = state.step_counter

    events: List[Event] = []

    # -- Terminal-state events ---------------------------------------
    if state_name == _STATE_GAME_OVER and state.prev_state_name != _STATE_GAME_OVER:
        events.append(AgentDied(step=current_step, cause="GAME_OVER"))
    if state_name == _STATE_WIN and state.prev_state_name != _STATE_WIN:
        # ``goal_id="episode"`` is a conventional top-level identifier;
        # adapter-level goal wiring in ``adapter.initialize`` uses the
        # same id so the engine's goal machinery sees the match.
        events.append(GoalConditionMet(step=current_step, goal_id="episode"))
    if levels_completed > state.prev_levels_completed:
        # Mid-episode level-up — surface as a distinct signal even
        # though the episode may still be running.  The engine can
        # form CausalClaims ("action X preceded level-up") over this.
        for lvl in range(state.prev_levels_completed + 1, levels_completed + 1):
            events.append(GoalConditionMet(step=current_step, goal_id=f"level_{lvl}"))

    # -- Region extraction and identity matching ---------------------
    regions = extract_regions(frame, background=background)
    key_to_region = {_key_for(reg): reg for reg in regions}

    new_entity_ids_by_key: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], str] = {}
    entity_snapshots: Dict[str, Dict[str, Any]] = {}

    # Entities present in both frames → either unchanged or moved
    # (motion_vectors surfaces the translation; we synthesise
    # EntityStateChanged for position-bearing properties).
    for key, reg in key_to_region.items():
        prev_id = state.entity_ids_by_key.get(key)
        if prev_id is None:
            state.next_entity_seq += 1
            ent_id = f"e{state.next_entity_seq}"
            events.append(EntityAppeared(
                step          = current_step,
                entity_id     = ent_id,
                initial_state = _entity_properties(reg),
            ))
        else:
            ent_id = prev_id
        new_entity_ids_by_key[key] = ent_id
        entity_snapshots[ent_id] = _entity_properties(reg)

    # Entities present previously but not now → disappeared.
    for old_key, old_id in state.entity_ids_by_key.items():
        if old_key not in new_entity_ids_by_key:
            events.append(EntityDisappeared(step=current_step, entity_id=old_id))

    # -- Motion / state-change events for persistent entities --------
    if state.prev_frame is not None and not is_identical(state.prev_frame, frame):
        vectors = motion_vectors(state.prev_frame, frame, background=background)
        # motion_vectors is keyed on region labels within a single
        # frame, not on our cross-frame entity ids.  We re-key via the
        # region keys (both frames' regions share the same key space).
        prev_regions = {reg.label: reg for reg in extract_regions(state.prev_frame, background=background)}
        cur_regions  = {reg.label: reg for reg in regions}
        for vec in vectors:
            if vec.dr == 0 and vec.dc == 0:
                continue
            prev_reg = prev_regions.get(vec.before_label)
            cur_reg  = cur_regions.get(vec.after_label)
            if prev_reg is None or cur_reg is None:
                continue
            key = _key_for(cur_reg)
            ent_id = new_entity_ids_by_key.get(key)
            if ent_id is None:
                continue
            # Emit a single EntityStateChanged with the new centroid.
            # The engine's TransitionMiner picks these up and forms
            # TransitionClaims parameterised by the preceding action.
            old_centroid = prev_reg.centroid
            new_centroid = cur_reg.centroid
            events.append(EntityStateChanged(
                step      = current_step,
                entity_id = ent_id,
                property  = "centroid",
                old       = old_centroid,
                new       = new_centroid,
            ))

    # -- Cell-diff count as a coarse surprise signal -----------------
    # A frame that changes without any matched motion is surprising
    # at the perception level (colour swaps, teleports, regeneration).
    # The SurpriseMiner will refine this; perception only emits the
    # raw signal.
    if state.prev_frame is not None:
        raw_changes = cell_diff(state.prev_frame, frame)
        # Only emit if there is change that motion_vectors did not
        # account for.  This is a heuristic — overly eager surprise
        # emission floods the hypothesis store.
        if raw_changes and not _all_cells_accounted_for(raw_changes, state.prev_frame, frame):
            events.append(SurpriseEvent(
                step     = current_step,
                expected = "frame explained by motion alone",
                actual   = f"{len(raw_changes)} cells changed without full motion match",
            ))

    # -- In-place visual mutation detection --------------------------
    # Emit EntityVisualPatternChanged when a same-colour entity
    # disappears and a new same-colour entity appears with an
    # overlapping bounding box in the SAME step — the signature of a
    # glyph rotation or in-place colour-pattern change.
    #
    # We need the bboxes of the PREVIOUS frame's regions, so we
    # build that map here (only when the previous frame exists and
    # the current frame has changed).
    if state.prev_frame is not None and not is_identical(state.prev_frame, frame):
        prev_key_to_region: Dict[Tuple[int, Tuple[Tuple[int, int], ...]], Region] = {
            _key_for(reg): reg
            for reg in extract_regions(state.prev_frame, background=background)
        }

        # Disappeared entities: key was in previous frame but not current.
        # Map colour → [(entity_id, bbox)] for disappeared entities.
        gone: Dict[Any, List[Tuple[str, Tuple[int, int, int, int]]]] = {}
        for old_key, old_id in state.entity_ids_by_key.items():
            if old_key in key_to_region:
                continue  # still present — not disappeared
            old_reg = prev_key_to_region.get(old_key)
            if old_reg is None:
                continue
            gone.setdefault(old_key[0], []).append((old_id, old_reg.bbox))

        # Appeared entities: key is in current frame but was not in previous.
        # Map colour → [(entity_id, bbox)] for newly-appeared entities.
        came: Dict[Any, List[Tuple[str, Tuple[int, int, int, int]]]] = {}
        for key, reg in key_to_region.items():
            if key in state.entity_ids_by_key:
                continue  # was already present — not newly appeared
            new_id = new_entity_ids_by_key.get(key)
            if new_id is None:
                continue
            came.setdefault(key[0], []).append((new_id, reg.bbox))

        # Pair same-colour disappear+appear with overlapping bboxes.
        # Each old entity may match at most one new entity (greedy).
        for colour in set(gone.keys()) & set(came.keys()):
            matched_new: set = set()
            for old_id, old_bbox in gone[colour]:
                for new_id, new_bbox in came[colour]:
                    if new_id in matched_new:
                        continue
                    if _bboxes_overlap(old_bbox, new_bbox):
                        events.append(EntityVisualPatternChanged(
                            step             = current_step,
                            entity_id_before = old_id,
                            entity_id_after  = new_id,
                            colour           = colour,
                            bbox             = _bbox_union(old_bbox, new_bbox),
                        ))
                        matched_new.add(new_id)
                        break  # each old entity matches at most one new

    # -- Commit state for next step ----------------------------------
    state.prev_frame             = _freeze(frame)
    state.entity_ids_by_key      = new_entity_ids_by_key
    state.prev_levels_completed  = levels_completed
    state.prev_state_name        = state_name

    # The engine reads goal conditions over ``ws.agent["resources"]``,
    # and ``ws.agent`` is assigned from ``obs.agent_state`` each step.
    # Surface the terminal-state flags as resources so the
    # ``ResourceAbove`` condition machinery can evaluate them:
    #
    #   * ``episode_won``  — 1.0 on WIN, 0.0 otherwise.
    #   * ``episode_lost`` — 1.0 on GAME_OVER, 0.0 otherwise.
    #   * ``levels_completed`` — raw counter; useful for mid-episode
    #     progress tracking.
    resources: Dict[str, float] = {
        "levels_completed": float(levels_completed),
        "episode_won":      1.0 if state_name == _STATE_WIN       else 0.0,
        "episode_lost":     1.0 if state_name == _STATE_GAME_OVER else 0.0,
    }

    return Observation(
        step             = current_step,
        agent_state      = {
            "state_name":       state_name,
            "levels_completed": levels_completed,
            "resources":        resources,
        },
        events           = events,
        entity_snapshots = entity_snapshots,
        raw_frame        = frame,
        metadata         = {
            "background":     background,
            "region_count":   len(regions),
        },
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bboxes_overlap(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> bool:
    """True iff two ``(r_min, c_min, r_max, c_max)`` bounding boxes overlap."""
    r_min_a, c_min_a, r_max_a, c_max_a = a
    r_min_b, c_min_b, r_max_b, c_max_b = b
    return (r_min_a <= r_max_b and r_max_a >= r_min_b and
            c_min_a <= c_max_b and c_max_a >= c_min_b)


def _bbox_union(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> Tuple[int, int, int, int]:
    """Return the smallest bbox enclosing both ``a`` and ``b``."""
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _key_for(region: Region) -> Tuple[int, Tuple[Tuple[int, int], ...]]:
    r0, c0 = region.bbox[0], region.bbox[1]
    shape = tuple((r - r0, c - c0) for (r, c) in region.cells)
    return (region.colour, shape)


def _entity_properties(region: Region) -> Dict[str, Any]:
    return {
        "colour":   region.colour,
        "bbox":     region.bbox,
        "centroid": region.centroid,
        "area":     region.area,
        "height":   region.height,
        "width":    region.width,
    }


def _all_cells_accounted_for(
    changes:  Sequence[Any],
    _before:  Grid,
    _after:   Grid,
) -> bool:
    """Placeholder: with the current implementation we cannot cheaply
    decide whether every changed cell is part of a motion-matched
    region.  Returning ``False`` conservatively emits a SurpriseEvent
    whenever there is any change that was not trivially explained by
    motion.  The SurpriseMiner downstream deduplicates against
    expected transitions so this is safe."""
    return False


def _freeze(frame: Grid) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(row) for row in frame)
