"""Temporal entity registry: stable identities for detected entities
across frames.

WHY THIS LAYER EXISTS
=====================

The entity detector emits a list of per-frame Entity records.  Each
Entity is a snapshot — a geometric description of pixels in one
frame, with no notion of "this is the same thing I saw last turn."
Downstream consumers need that notion:

  - The VLM-based role classifier needs to ask "has this entity
    been there before?" to distinguish a static UI element from a
    transient sprite.
  - Behaviour matchers need to track how an entity changed across
    turns (moved, grew, disappeared).
  - The tool API exposes `entity_history(handle)` which only works
    if the handle is stable.

The registry assigns a stable `track_id` to each entity that
persists across frames, by matching this-frame entities against
recent-frame tracks.


HOW MATCHING WORKS
==================

For each new-frame entity, score it against every still-recent
track using a HIERARCHICAL matcher (no weighted sums of dubious
weights):

  1. Filter: require signature_similarity >= SIG_THRESHOLD.  Two
     entities with very different colour signatures aren't the
     same thing regardless of where they sit.

  2. Among signature-compatible candidates, require centroid_cell
     distance <= MAX_CELL_DELTA cells.  An entity in cell (0,0)
     can't be the same thing as a separate one in cell (7,7).

  3. Among position-compatible candidates, pick the one with the
     CLOSEST centroid (manhattan distance).  Closest = same.

  4. Tiebreak by pixel-count ratio (closer-to-1.0 wins).

Match assignments are GREEDY in one pass through the new-frame
entities ordered by track-affinity (most-confident matches first),
so two new entities can't both claim the same prior track.


TRACK LIFECYCLE
===============

A track has three states relative to the current turn N:

  ACTIVE   — observed in turn N (the most recent observation).
  MISSING  — last observed at some turn < N, within RECENCY_WINDOW
             turns ago.  Still a match candidate.
  LOST     — last observed > RECENCY_WINDOW turns ago.  No longer
             a match candidate; new tracks will be minted for any
             reappearance.

A track can flip ACTIVE → MISSING → ACTIVE (an entity briefly
disappears then comes back at the same position with the same
signature — e.g. a threat cloud that pulses).  But once LOST it
stays lost — if the same visual reappears much later it gets a
fresh track id.  This is a conservative choice; the consequence
is rare double-counting of "same entity, came back later" which is
preferable to incorrect long-range merges.


WHAT THIS DOES NOT DO
=====================

  - No role assignment.  Tracks carry an optional `role_hint` that
    a downstream classifier (VLM or template matcher) sets, but the
    registry itself doesn't decide what a track IS.
  - No segmentation refinement.  If the detector mis-grouped two
    entities into one in one frame, the registry tracks the
    mis-grouped blob; fixing detector segmentation is the
    detector's job.
  - No multi-track-from-one-blob splits.  If a single connected
    component contains pixels from two different tracks (e.g. the
    bp35 lc=0 counter + threat cloud case), the registry sees one
    entity.  Splitting it is a job for the detector + VLM.


CONSTANTS — operator-tunable but not magic
==========================================

SIG_THRESHOLD = 0.5     A loose floor: two entities sharing < 50%
                         of their colour signature are clearly
                         different.  Higher would miss legitimate
                         appearance shifts (a sprite that lost a
                         pixel of detail); lower would allow
                         spurious merges.
MAX_CELL_DELTA = 3      An entity that moved more than 3 cells in
                         one turn is almost certainly a different
                         entity that happens to share signature.
                         Sprites in ARC-AGI-3 typically move 0-1
                         cells per turn.
RECENCY_WINDOW = 15     A track that hasn't been observed in 15
                         turns is considered lost.  Threats that
                         re-appear after long gaps will get new
                         track ids — this is fine; downstream can
                         still note "track X looks like prior
                         track Y" if it wants.

These three values are the only tunables; everything else is
algorithmic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .entity_detector import Entity
from .template_store import signature_similarity
from .perceptual_equivalence import Lens, canonical, score_association


SIG_THRESHOLD = 0.5
MAX_CELL_DELTA = 3
RECENCY_WINDOW = 15


@dataclass
class EntityObservation:
    """Per-frame snapshot for one entity in a track.  Carries the
    same geometric facts as Entity but pins down WHICH turn it was
    seen in and which track it belongs to.
    """

    turn: int
    centroid_cell: tuple[int, int]
    bbox_logical: tuple[int, int, int, int]
    n_pixels: int
    visual_signature: tuple
    cells: list[tuple[int, int]]
    # Whether the entity was actually detected this turn (True) or
    # this track is just being marked missing (False — not currently
    # used; missing turns are implicit gaps in observations).
    present: bool = True
    # Background-removed coloured mask (cell coords, bg = -1), carried from the
    # Entity so the object-constancy matcher can canonicalise it for pose-factored
    # association.  None when the source entity had no mask (e.g. VLM gestalts).
    bitmap: Optional["np.ndarray"] = None


@dataclass
class BehaviourEvent:
    """An observed event involving a track.

    These are FACTS — observations of what happened, with full
    credence.  They are NOT hypotheses about why something happened
    (those live in the existing hypothesis_store / WorldState with
    proper credence tracking).

    `kind` values are a closed vocabulary the registry emits
    deterministically inside ingest_frame:

      appeared     - track first observed at this turn
      disappeared  - track was present last turn but absent this turn
      reappeared   - track had been missing for >=1 turn and is back
      moved        - centroid_cell changed since last observation
      grew         - n_pixels increased materially (>=20%)
      shrank       - n_pixels decreased materially (>=20%)
      drifted      - centroid moved by <1 cell (sub-cell drift)
    """

    turn: int
    kind: str
    details: Optional[str] = None
    # Other tracks involved in this event (e.g. the agent track id
    # when a sediment disappears after the agent enters its cell).
    related_track_ids: list[int] = field(default_factory=list)


@dataclass
class EntityTrack:
    """A persistent entity identity across many frames.

    The track is the unit of identity downstream consumers refer to.
    Two different frames' Entity records that share a track_id are
    "the same thing" by the registry's matching judgement.

    Carries TWO kinds of information:

      OBSERVED FACTS (registry-owned, full credence, never decay):
        - geometric observations (one per turn it was present)
        - behaviour events (appeared/disappeared/moved/grew/shrank)
        - derived stats (presence_fraction, pixel_count_variance)
        - description (one-line text, may be set by VLM but it's
          recording observed visual content, not a hypothesis)
        - properties (typed key-value pairs about colour, shape,
          signature class -- all derivable from observations)

      HYPOTHESIS POINTERS (cross-reference to the hypstore):
        - hypothesis_ids: list of Hypothesis ids in WorldState that
          claim something about this track ("track 1 is the agent",
          "track 5's disappearance is caused by track 1 contact").
          Credence updates happen on those hypotheses, not here.

    This split keeps observed facts from being polluted by uncertain
    inferences.  When the VLM wants to know what the registry KNOWS
    about a track it reads the fact fields; when it wants to know
    what's been HYPOTHESISED it follows hypothesis_ids.
    """

    track_id: int
    first_seen_turn: int
    last_seen_turn: int
    observations: list[EntityObservation] = field(default_factory=list)

    # Behaviour event log -- observed events involving this track,
    # added automatically by the registry inside ingest_frame.
    behaviour_events: list[BehaviourEvent] = field(default_factory=list)

    # Descriptive metadata.  description and category_labels are
    # typically set by a VLM probe; properties are derivable.
    description: Optional[str] = None
    # category_labels: list of (label, confidence_word) pairs that the
    # VLM proposes (e.g. [("avatar", "high"), ("controllable", "high")]).
    # These are *labels* the VLM has chosen to attach -- treated as
    # facts that the VLM made this judgement, not as ground truth
    # about what the entity actually is.  Quasi-hypotheses; richer
    # claims should go through hypothesis_store.
    category_labels: list[tuple[str, str]] = field(default_factory=list)
    # Typed key-value properties.  Recommended keys: "color",
    # "shape", "size_class", "signature_class", "position_kind".
    # All derivable from observations or set externally.
    properties: dict[str, str] = field(default_factory=dict)

    # Cross-references to the hypothesis_store.  The registry never
    # touches credences itself; following one of these ids gives the
    # caller a Hypothesis with credence + evidence.  Unused until the
    # hypstore wiring lands; included now so downstream code can
    # depend on the contract.
    hypothesis_ids: list[str] = field(default_factory=list)

    # Legacy compatibility fields (older callers).  role_hint is
    # superseded by category_labels but kept for backward-compat.
    role_hint: Optional[str] = None
    role_confidence: Optional[str] = None

    # ---------- presence / observation queries ----------

    def is_present_at(self, turn: int) -> bool:
        return any(o.turn == turn for o in self.observations)

    def last_observation(self) -> Optional[EntityObservation]:
        return self.observations[-1] if self.observations else None

    def observation_at(self, turn: int) -> Optional[EntityObservation]:
        for o in self.observations:
            if o.turn == turn:
                return o
        return None

    def n_turns_observed(self) -> int:
        return len(self.observations)

    def n_turns_in_window(
        self, turn_now: int, window: int,
    ) -> int:
        cutoff = turn_now - window + 1
        return sum(
            1 for o in self.observations
            if cutoff <= o.turn <= turn_now
        )

    def turns_in_window(
        self, turn_now: int, window: int,
    ) -> list[int]:
        cutoff = turn_now - window + 1
        return [
            o.turn for o in self.observations
            if cutoff <= o.turn <= turn_now
        ]

    # ---------- derived stats (facts computed from observations) ----------

    def presence_fraction(self, turn_now: int, window: int) -> float:
        n_possible = min(window, turn_now - self.first_seen_turn + 1)
        if n_possible <= 0:
            return 0.0
        return self.n_turns_in_window(turn_now, window) / n_possible

    def pixel_count_min(self) -> int:
        return min((o.n_pixels for o in self.observations), default=0)

    def pixel_count_max(self) -> int:
        return max((o.n_pixels for o in self.observations), default=0)

    def pixel_count_variance_ratio(self) -> float:
        """max / min ratio of pixel counts across observations.  A
        stable entity has ratio ~1.0; a track that's been merging
        and un-merging with other entities (lc=0 counter+cloud case)
        has a huge ratio.  This is the signal that "one track" might
        really represent multiple entities.
        """
        lo = self.pixel_count_min()
        hi = self.pixel_count_max()
        if lo <= 0:
            return float("inf") if hi > 0 else 0.0
        return hi / lo

    def canonical_signature(self) -> tuple:
        """The most-common top-key of the visual_signatures across
        observations.  Substrate-agnostic "what colour is this thing
        usually".
        """
        if not self.observations:
            return tuple()
        # Take the modal first-key across observations.
        from collections import Counter
        first_keys = Counter()
        for o in self.observations:
            if o.visual_signature:
                first_keys[o.visual_signature[0][0]] += 1
        if not first_keys:
            return tuple()
        # Return the full signature from the most-recent observation
        # whose first-key matches the modal value.  Picks a "typical"
        # signature, not an average.
        modal_key, _ = first_keys.most_common(1)[0]
        for o in reversed(self.observations):
            if o.visual_signature and o.visual_signature[0][0] == modal_key:
                return o.visual_signature
        return self.observations[-1].visual_signature

    def position_drift_cells(self) -> int:
        """Manhattan distance between earliest and latest centroid
        cells.  0 = never moved.  Useful for distinguishing pinned
        UI elements from mobile sprites.
        """
        if len(self.observations) < 2:
            return 0
        a = self.observations[0].centroid_cell
        b = self.observations[-1].centroid_cell
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def is_stationary(self) -> bool:
        return self.position_drift_cells() == 0


@dataclass
class TemporalEntityRegistry:
    """Cross-frame entity tracking for one game session.

    Caller pattern:
      reg = TemporalEntityRegistry(game_id="bp35")
      for turn_n in turns:
          entities = detect_entities(observation_n)
          mapping = reg.ingest_frame(turn_n, entities)
          # mapping[entity.entity_id] = track_id assigned to that entity
    """

    game_id: str
    tracks: dict[int, EntityTrack] = field(default_factory=dict)
    _next_track_id: int = 1
    # Lens for the object-constancy matcher (spec: shape identity tolerant of
    # recolour and rotation; position plausibility disambiguates).  Game-agnostic
    # default; the reasoning layer may override per game.
    equivalence_lens: Lens = field(default_factory=lambda: Lens(
        rotations=True, reflection=False, scale="none", color="agnostic"))

    # Identity residual a same-object association may have (cell-mismatch fraction
    # under the lens).  Principled value: 0.0 — frames are palette-indexed (no
    # antialiasing), so the SAME object under the lens canonicalises EXACTLY; any
    # nonzero residual is a genuine shape change (grew/shrank), not the same
    # identity.  No magic fudge factor.
    OC_RESIDUAL_TAU = 0.0

    def ingest_frame(
        self,
        turn: int,
        entities: list[Entity],
    ) -> dict[int, int]:
        """Match `entities` against existing tracks; mint new tracks
        for unmatched entities.  Returns {entity.entity_id: track_id}.
        Backgrounds (is_background_primary / is_background_secondary)
        are NOT tracked — they're not entities in the perception
        sense, they're the frame's static scene.
        """
        mapping: dict[int, int] = {}
        # Candidate tracks: any track whose last observation is within
        # RECENCY_WINDOW of the current turn.
        candidates: list[EntityTrack] = [
            t for t in self.tracks.values()
            if t.last_seen_turn >= turn - RECENCY_WINDOW
            and t.last_seen_turn < turn  # not already matched this turn
        ]
        used_entity_ids: set[int] = set()
        used_track_ids: set[int] = set()

        # --- PRIMARY pass: OBJECT CONSTANCY (when masks are available) ----
        # One principled estimator that subsumes the move / recolor / long-jump
        # passes below: it associates by pose-factored identity (shape, under the
        # lens) scored by pose-change plausibility, so a moved, recoloured, or
        # rotated object stays the same track — near or far.  Entities/tracks
        # without masks fall through to the legacy passes unchanged.
        self._object_constancy_pass(
            turn, entities, candidates, mapping, used_entity_ids, used_track_ids)

        # Score every remaining (entity, candidate) pair using the hierarchical
        # matcher.  Score=None means filtered out (incompatible).
        scored: list[tuple[float, Entity, EntityTrack]] = []
        for e in entities:
            if e.is_background_primary or e.is_background_secondary:
                continue
            for c in candidates:
                last = c.last_observation()
                if last is None:
                    continue
                sim = signature_similarity(
                    e.visual_signature, last.visual_signature,
                )
                if sim < SIG_THRESHOLD:
                    continue
                # Position filter
                dr = abs(e.centroid_cell[0] - last.centroid_cell[0])
                dc = abs(e.centroid_cell[1] - last.centroid_cell[1])
                if dr > MAX_CELL_DELTA or dc > MAX_CELL_DELTA:
                    continue
                # Score: signature similarity dominates; position
                # closeness and size-ratio are tiebreakers.  Single
                # composite score so the best-first selection is a
                # simple sort.
                pos_closeness = 1.0 - (dr + dc) / (2.0 * MAX_CELL_DELTA)
                if last.n_pixels > 0 and e.n_pixels > 0:
                    size_ratio = (
                        min(e.n_pixels, last.n_pixels)
                        / max(e.n_pixels, last.n_pixels)
                    )
                else:
                    size_ratio = 0.5
                score = (sim, pos_closeness, size_ratio)
                # Use tuple ordering so signature similarity dominates,
                # then position, then size — same hierarchy as the
                # filter, with no weighted-sum magic.
                scored.append((score, e, c))

        # Greedy best-first assignment: sort by descending score,
        # assign in order, skipping pairs whose entity or candidate is
        # already taken.
        scored.sort(key=lambda x: x[0], reverse=True)
        for _score, e, c in scored:
            if e.entity_id in used_entity_ids:
                continue
            if c.track_id in used_track_ids:
                continue
            used_entity_ids.add(e.entity_id)
            used_track_ids.add(c.track_id)
            mapping[e.entity_id] = c.track_id
            self._record_observation(c, turn, e)

        # --- Secondary pass: RECOLOR-IN-PLACE ----------------------------
        # The signature gate above reads a same-position palette swap as
        # disappear+appear (the swapped entity's signature flips, failing
        # SIG_THRESHOLD).  But an entity of the SAME shape at the SAME cell is
        # the SAME object that recolored.  Match those by position + size alone
        # (ignoring signature) so coupled toggles (r11l's star pair) are seen
        # as one entity changing colour, not two replacing each other.  Strict
        # gates -- same cell (<=1 delta), size within 30% -- keep this from
        # merging genuinely different entities that briefly share a location.
        RECOLOR_MAX_DELTA = 1
        RECOLOR_MIN_SIZE_RATIO = 0.7
        for e in entities:
            if e.is_background_primary or e.is_background_secondary:
                continue
            if e.entity_id in used_entity_ids:
                continue
            best: Optional[tuple[tuple[int, float], EntityTrack]] = None
            for c in candidates:
                if c.track_id in used_track_ids:
                    continue
                last = c.last_observation()
                if last is None or last.n_pixels <= 0 or e.n_pixels <= 0:
                    continue
                dr = abs(e.centroid_cell[0] - last.centroid_cell[0])
                dc = abs(e.centroid_cell[1] - last.centroid_cell[1])
                if dr + dc > RECOLOR_MAX_DELTA:
                    continue
                sr = (min(e.n_pixels, last.n_pixels)
                      / max(e.n_pixels, last.n_pixels))
                if sr < RECOLOR_MIN_SIZE_RATIO:
                    continue
                key = (dr + dc, 1.0 - sr)
                if best is None or key < best[0]:
                    best = (key, c)
            if best is None:
                continue
            c = best[1]
            used_entity_ids.add(e.entity_id)
            used_track_ids.add(c.track_id)
            mapping[e.entity_id] = c.track_id
            self._record_observation(c, turn, e)  # fires the recolor event

        # --- Tertiary pass: LONG-JUMP MOVE -------------------------------
        # The primary matcher gates position at MAX_CELL_DELTA, so an entity
        # that JUMPS many cells in one turn (m0r0's agents slide to a wall;
        # any fast sprite) fails the gate and reads as disappear+appear.  But
        # an unmatched entity that LOOKS the same (signature) and is the same
        # SIZE as an unmatched track is that track having moved.  Match the
        # same-appearance unmatched pairs greedily by MINIMUM displacement
        # (so two identical agents each bind to their nearest new position),
        # ignoring distance, and fire a "moved" event.  Appearance + size
        # gates keep this from linking genuinely distinct sprites.
        jump_pairs: list[tuple[float, Entity, EntityTrack]] = []
        for e in entities:
            if e.is_background_primary or e.is_background_secondary:
                continue
            if e.entity_id in used_entity_ids:
                continue
            for c in candidates:
                if c.track_id in used_track_ids:
                    continue
                last = c.last_observation()
                if last is None or last.n_pixels <= 0 or e.n_pixels <= 0:
                    continue
                if signature_similarity(e.visual_signature,
                                        last.visual_signature) < SIG_THRESHOLD:
                    continue
                sr = (min(e.n_pixels, last.n_pixels)
                      / max(e.n_pixels, last.n_pixels))
                if sr < 0.7:
                    continue
                disp = (abs(e.centroid_cell[0] - last.centroid_cell[0])
                        + abs(e.centroid_cell[1] - last.centroid_cell[1]))
                jump_pairs.append((disp, e, c))
        jump_pairs.sort(key=lambda x: x[0])
        for _disp, e, c in jump_pairs:
            if e.entity_id in used_entity_ids or c.track_id in used_track_ids:
                continue
            used_entity_ids.add(e.entity_id)
            used_track_ids.add(c.track_id)
            mapping[e.entity_id] = c.track_id
            self._record_observation(c, turn, e)  # fires the moved event

        # Mint new tracks for unmatched entities (excluding backgrounds).
        for e in entities:
            if e.is_background_primary or e.is_background_secondary:
                continue
            if e.entity_id in mapping:
                continue
            track = EntityTrack(
                track_id=self._next_track_id,
                first_seen_turn=turn,
                last_seen_turn=turn,
            )
            self._next_track_id += 1
            self.tracks[track.track_id] = track
            self._record_observation(track, turn, e)
            mapping[e.entity_id] = track.track_id

        # Emit "disappeared" events for tracks that were present at
        # the previous turn but didn't match anything this turn.  This
        # is the canonical signal for consumption / despawn.
        for t in self.tracks.values():
            if t.track_id in used_track_ids:
                continue
            if t.last_seen_turn == turn - 1:
                # Was present last turn; absent now.
                # Avoid double-emit if we just emitted one for this turn.
                already_emitted = any(
                    e.turn == turn and e.kind == "disappeared"
                    for e in t.behaviour_events
                )
                if not already_emitted:
                    t.behaviour_events.append(BehaviourEvent(
                        turn=turn, kind="disappeared",
                    ))

        return mapping

    def _object_constancy_pass(
        self, turn, entities, candidates, mapping,
        used_entity_ids, used_track_ids,
    ) -> None:
        """Associate entities to tracks by pose-factored identity + plausibility.

        For each (entity, track) with masks available, score the entity against the
        track's last mask under `equivalence_lens`: identity must match within
        OC_RESIDUAL_TAU, and the implied pose change must be possible
        (plausibility > 0).  Greedy best-first by (plausibility, identity), so near
        matches bind before far ones — which makes a long jump match only when no
        nearer same-identity entity competes.  Subsumes move / recolor / long-jump
        in one estimator; masks-absent entities are left for the legacy passes."""
        lens = self.equivalence_lens
        scored = []
        for e in entities:
            if e.is_background_primary or e.is_background_secondary:
                continue
            if getattr(e, "bitmap", None) is None:
                continue
            for c in candidates:
                last = c.last_observation()
                if last is None or getattr(last, "bitmap", None) is None:
                    continue
                cm = canonical(last.bitmap, -1, lens)
                r = score_association(cm["descriptor"], cm["pose"], e.bitmap, lens,
                                      obs_bg=-1, threshold=self.OC_RESIDUAL_TAU)
                m = r["match"]; plaus = r["pose_change_plausibility"]
                if m.residual <= self.OC_RESIDUAL_TAU and plaus > 0.0:
                    scored.append(((plaus, 1.0 - m.residual), e, c))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _s, e, c in scored:
            if e.entity_id in used_entity_ids or c.track_id in used_track_ids:
                continue
            used_entity_ids.add(e.entity_id)
            used_track_ids.add(c.track_id)
            mapping[e.entity_id] = c.track_id
            self._record_observation(c, turn, e)

    # ------------------------------------------------------------------
    # Behaviour event thresholds.  All substrate-agnostic.
    # ------------------------------------------------------------------

    # A pixel-count change of >= 20% counts as "grew" / "shrank".
    # Smaller changes are noise; this threshold reflects the sensitivity
    # we want without per-game tuning.
    _SIZE_CHANGE_THRESHOLD = 0.20

    def _record_observation(
        self, track: EntityTrack, turn: int, e: Entity,
    ) -> None:
        """Append an observation and emit behaviour events for the
        change from the previous observation, if any.
        """
        prev = track.last_observation()
        track.observations.append(EntityObservation(
            turn=turn,
            centroid_cell=e.centroid_cell,
            bbox_logical=e.bbox_logical,
            n_pixels=e.n_pixels,
            visual_signature=e.visual_signature,
            cells=list(e.cells),
            present=True,
            bitmap=getattr(e, "bitmap", None),
        ))
        track.last_seen_turn = turn

        if prev is None:
            track.behaviour_events.append(BehaviourEvent(
                turn=turn, kind="appeared",
            ))
            return

        # Reappearance: track was missing for some intermediate turns
        # then came back.  Distinct from a continuous "moved" event.
        if turn - prev.turn > 1:
            gap = turn - prev.turn - 1
            track.behaviour_events.append(BehaviourEvent(
                turn=turn, kind="reappeared",
                details=f"absent for {gap} turn(s)",
            ))

        # Movement (cell-level).
        if e.centroid_cell != prev.centroid_cell:
            dr = e.centroid_cell[0] - prev.centroid_cell[0]
            dc = e.centroid_cell[1] - prev.centroid_cell[1]
            track.behaviour_events.append(BehaviourEvent(
                turn=turn, kind="moved",
                details=f"by ({dr:+d},{dc:+d}) cells",
            ))

        # Size change.
        if prev.n_pixels > 0:
            ratio = e.n_pixels / prev.n_pixels
            if ratio >= 1.0 + self._SIZE_CHANGE_THRESHOLD:
                track.behaviour_events.append(BehaviourEvent(
                    turn=turn, kind="grew",
                    details=f"{prev.n_pixels} -> {e.n_pixels} px",
                ))
            elif ratio <= 1.0 - self._SIZE_CHANGE_THRESHOLD:
                track.behaviour_events.append(BehaviourEvent(
                    turn=turn, kind="shrank",
                    details=f"{prev.n_pixels} -> {e.n_pixels} px",
                ))

        # Recolour: dominant colour changed in place (same track, same shape).
        # The signature matcher gates on colour similarity, so a swap normally
        # reads as disappear+appear; the recolor-in-place pass in ingest_frame
        # routes it here instead, where this records the colour transition.
        prev_top = prev.visual_signature[0][0] if prev.visual_signature else None
        cur_top = e.visual_signature[0][0] if e.visual_signature else None
        if prev_top is not None and cur_top is not None and prev_top != cur_top:
            track.behaviour_events.append(BehaviourEvent(
                turn=turn, kind="recolor",
                details=f"signature {prev_top} -> {cur_top}",
            ))

    # ---------------------------------------------------------------
    # Query helpers — what downstream consumers (VLM tools, behaviour
    # matchers, the trace renderer) call to introspect the registry.
    # ---------------------------------------------------------------

    def get_track(self, track_id: int) -> Optional[EntityTrack]:
        return self.tracks.get(track_id)

    def active_tracks(self, turn: int) -> list[EntityTrack]:
        """Tracks observed at the given turn."""
        return [t for t in self.tracks.values() if t.is_present_at(turn)]

    def all_tracks(self) -> list[EntityTrack]:
        return list(self.tracks.values())

    def stable_tracks(
        self,
        turn_now: int,
        window: int = RECENCY_WINDOW,
        min_presence: float = 0.8,
    ) -> list[EntityTrack]:
        """Tracks present in at least `min_presence` fraction of the
        last `window` turns.  These are the candidates for "persistent
        UI" or "always-there scenery" entities — the budget counter,
        the win marker, etc.  No magic threshold beyond the fraction
        the caller picks.
        """
        out: list[EntityTrack] = []
        for t in self.tracks.values():
            n_present = t.n_turns_in_window(turn_now, window)
            n_possible = min(window, turn_now - t.first_seen_turn + 1)
            if n_possible <= 0:
                continue
            if n_present / n_possible >= min_presence:
                out.append(t)
        return out

    def intermittent_tracks(
        self,
        turn_now: int,
        window: int = RECENCY_WINDOW,
        max_presence: float = 0.4,
    ) -> list[EntityTrack]:
        """Tracks that have appeared but are present in at most
        `max_presence` fraction of recent turns.  Candidates for
        "intermittent sprites" like the threat cloud — entities that
        come and go.
        """
        out: list[EntityTrack] = []
        for t in self.tracks.values():
            n_present = t.n_turns_in_window(turn_now, window)
            n_possible = min(window, turn_now - t.first_seen_turn + 1)
            if n_possible <= 0:
                continue
            frac = n_present / n_possible
            if frac > 0 and frac <= max_presence:
                out.append(t)
        return out


# -----------------------------------------------------------------------------
# Spatial relationships -- observed geometric facts between two tracks at one
# turn.  Computed deterministically every frame.  Distinct from causal /
# behavioural HYPOTHESES which live in the existing hypothesis_store with
# credence.
# -----------------------------------------------------------------------------


@dataclass
class SpatialRelationship:
    """An observed geometric relationship between two tracks at a
    specific turn.  Facts only (full credence).
    """

    type: str  # closed vocabulary, see RELATION_KINDS below
    a: int     # track_id
    b: int     # track_id
    turn: int
    extra: dict = field(default_factory=dict)


# Closed vocabulary of spatial relationship kinds.  Substrate-agnostic.
RELATION_KINDS = (
    "same_cell",        # a and b share their centroid cell
    "adjacent_cell",    # centroids are 4-neighbours
    "aligned_row",      # same row but different columns
    "aligned_col",      # same column but different rows
    "bbox_overlap",     # bboxes overlap (in logical pixels)
    "a_contains_b",     # a's bbox fully contains b's bbox
)


def _compute_relationships_at(
    tracks: list[EntityTrack], turn: int,
) -> list[SpatialRelationship]:
    """For every pair of tracks present at `turn`, emit each applicable
    spatial relationship (closed vocabulary).  O(n^2) in entities per
    frame -- fine for ARC-AGI-3 scale (~20-30 entities per frame).
    """
    present: list[tuple[EntityTrack, EntityObservation]] = []
    for t in tracks:
        o = t.observation_at(turn)
        if o is not None:
            present.append((t, o))

    out: list[SpatialRelationship] = []
    for i in range(len(present)):
        for j in range(len(present)):
            if i == j:
                continue
            t_a, o_a = present[i]
            t_b, o_b = present[j]
            ar, ac = o_a.centroid_cell
            br, bc = o_b.centroid_cell

            # same cell
            if (ar, ac) == (br, bc) and i < j:
                # emit only one direction to avoid duplicates
                out.append(SpatialRelationship(
                    type="same_cell", a=t_a.track_id, b=t_b.track_id,
                    turn=turn,
                ))
                continue

            # adjacent (4-neighbour) -- emit only one direction
            if i < j:
                dr = abs(ar - br)
                dc = abs(ac - bc)
                if (dr == 1 and dc == 0) or (dr == 0 and dc == 1):
                    out.append(SpatialRelationship(
                        type="adjacent_cell",
                        a=t_a.track_id, b=t_b.track_id,
                        turn=turn,
                    ))
                # aligned row (same row, distance >= 2 cols)
                elif ar == br and dc >= 2:
                    out.append(SpatialRelationship(
                        type="aligned_row",
                        a=t_a.track_id, b=t_b.track_id,
                        turn=turn, extra={"col_gap": dc},
                    ))
                # aligned column (same col, distance >= 2 rows)
                elif ac == bc and dr >= 2:
                    out.append(SpatialRelationship(
                        type="aligned_col",
                        a=t_a.track_id, b=t_b.track_id,
                        turn=turn, extra={"row_gap": dr},
                    ))

            # bbox-level (directional: a_contains_b is asymmetric)
            ay0, ax0, ay1, ax1 = o_a.bbox_logical
            by0, bx0, by1, bx1 = o_b.bbox_logical
            # Strict containment (a's bbox strictly contains b's)
            if (ay0 <= by0 and ax0 <= bx0
                    and ay1 >= by1 and ax1 >= bx1
                    and (ay0, ax0, ay1, ax1) != (by0, bx0, by1, bx1)):
                out.append(SpatialRelationship(
                    type="a_contains_b",
                    a=t_a.track_id, b=t_b.track_id, turn=turn,
                ))
            # General overlap (emit only one direction)
            elif i < j:
                overlap = not (
                    ay1 < by0 or by1 < ay0
                    or ax1 < bx0 or bx1 < ax0
                )
                if overlap:
                    out.append(SpatialRelationship(
                        type="bbox_overlap",
                        a=t_a.track_id, b=t_b.track_id, turn=turn,
                    ))
    return out


# -----------------------------------------------------------------------------
# Action history -- record of agent actions and observed scene changes.
# -----------------------------------------------------------------------------


@dataclass
class ActionRecord:
    """One agent action and the changes observed in the frame that
    followed.  Facts only -- the connection between action and change
    is correlation, not causation (causation is a hypothesis in the
    hypothesis_store).
    """

    turn: int
    action: str                       # action label as the harness reports
    agent_pos_before: Optional[tuple[int, int]] = None
    agent_pos_after: Optional[tuple[int, int]] = None
    # Tracks that changed state on the following frame: each entry is
    # (track_id, kind, details).  Sourced from BehaviourEvents at that
    # next turn so the data isn't duplicated.
    observed_changes: list[tuple[int, str, Optional[str]]] = field(
        default_factory=list,
    )


# -----------------------------------------------------------------------------
# Scene -- the top-level symbolic state the VLM reasons over.  Owns the
# registry, derives spatial relationships every frame, holds action history,
# carries hypothesis pointers into the hypothesis_store.
# -----------------------------------------------------------------------------


@dataclass
class Scene:
    """Scene-level symbolic state for one game session.

    Owns the TemporalEntityRegistry and adds three things:

      - per-turn spatial relationships (derived facts)
      - action history (recorded facts)
      - hypothesis_ids cross-referencing the existing hypothesis_store
        (for inferences about the scene that have credence)

    The Scene is the structured input the action VLM reads.  All its
    fields are FACTS -- observed by the detector + registry, or
    recorded actions, or geometric derivations.  Hypotheses (causal
    chains, role labels above mere category labels, strategy claims)
    live in WorldState.hypotheses and are referenced by id.
    """

    game_id: str
    registry: TemporalEntityRegistry = field(
        default_factory=lambda: TemporalEntityRegistry(game_id=""),
    )

    # Per-turn observed spatial relationships, computed inside
    # ingest_frame().
    relationships_by_turn: dict[int, list[SpatialRelationship]] = field(
        default_factory=dict,
    )

    # Agent action history.
    actions: list[ActionRecord] = field(default_factory=list)

    # Scene-level hypothesis pointers (into WorldState.hypotheses).
    # Holds claims that aren't naturally pinned to a single entity --
    # "the game is a maze", "actions cost from a budget", etc.
    hypothesis_ids: list[str] = field(default_factory=list)

    # Short text notes the operator or VLM has attached to the scene
    # as observed mechanics.  These are observed *patterns* the system
    # has noticed, not credence-bearing claims.
    notes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Keep the registry's game_id in sync with the scene's.
        if not self.registry.game_id:
            self.registry.game_id = self.game_id

    # ------------------------------------------------------------------
    # Frame ingest
    # ------------------------------------------------------------------

    def ingest_frame(
        self, turn: int, entities: list[Entity],
    ) -> dict[int, int]:
        """Ingest one frame's detected entities.  Delegates to the
        registry for identity tracking and behaviour events, then
        derives spatial relationships at this turn.

        Returns the {entity_id: track_id} mapping the registry
        produced.
        """
        mapping = self.registry.ingest_frame(turn, entities)
        # Compute spatial relationships from tracks active this turn.
        rels = _compute_relationships_at(
            list(self.registry.tracks.values()), turn,
        )
        self.relationships_by_turn[turn] = rels
        return mapping

    # ------------------------------------------------------------------
    # Action recording
    # ------------------------------------------------------------------

    def record_action(
        self,
        turn: int,
        action: str,
        agent_pos_before: Optional[tuple[int, int]] = None,
        agent_pos_after: Optional[tuple[int, int]] = None,
    ) -> ActionRecord:
        """Record that the agent took `action` at `turn` (i.e. between
        the observation at `turn` and the observation at `turn+1`).
        observed_changes is filled in lazily by reading the next
        turn's BehaviourEvents.
        """
        rec = ActionRecord(
            turn=turn,
            action=action,
            agent_pos_before=agent_pos_before,
            agent_pos_after=agent_pos_after,
        )
        self.actions.append(rec)
        return rec

    def attach_observed_changes(self, action_turn: int) -> None:
        """After ingesting the post-action frame, populate the action
        record's observed_changes from the BehaviourEvents at turn
        action_turn + 1.
        """
        next_turn = action_turn + 1
        rec = next(
            (r for r in self.actions if r.turn == action_turn), None,
        )
        if rec is None:
            return
        rec.observed_changes = []
        for t in self.registry.tracks.values():
            for ev in t.behaviour_events:
                if ev.turn == next_turn:
                    rec.observed_changes.append(
                        (t.track_id, ev.kind, ev.details),
                    )

    # ------------------------------------------------------------------
    # VLM-facing serialization
    # ------------------------------------------------------------------

    def serialize_for_vlm(
        self, turn: int, *, history_window: int = 5,
    ) -> dict:
        """Produce a curated JSON-ready dict describing the scene at
        `turn`, with `history_window` turns of recent context.  This is
        what the VLM reads to reason about the next action.

        Includes:
          entities       - active tracks with descriptive facts +
                           recent behaviour events
          relationships  - spatial relationships at this turn
          recent_actions - last `history_window` action records with
                           observed changes
          notes          - operator/VLM scene notes
          hypothesis_ids - scene-level hypstore pointers

        Per-entity hypothesis_ids are NOT inlined here -- the VLM is
        expected to fetch them from the hypstore directly when needed.
        """
        recent_cutoff = turn - history_window + 1
        entities_out: list[dict] = []
        for t in self.registry.tracks.values():
            obs_at = t.observation_at(turn)
            # Include tracks present this turn, plus those that were
            # present in the recent window (so "what disappeared
            # recently" is visible).
            recent_observed = t.last_seen_turn >= recent_cutoff
            if obs_at is None and not recent_observed:
                continue
            recent_events = [
                {"turn": e.turn, "kind": e.kind, "details": e.details}
                for e in t.behaviour_events
                if recent_cutoff <= e.turn <= turn
            ]
            entry: dict = {
                "track_id": t.track_id,
                "present_now": obs_at is not None,
                "description": t.description,
                "category_labels": t.category_labels,
                "properties": dict(t.properties),
                "first_seen_turn": t.first_seen_turn,
                "last_seen_turn": t.last_seen_turn,
                "n_turns_observed": t.n_turns_observed(),
                "presence_fraction_window": round(
                    t.presence_fraction(turn, history_window), 2,
                ),
                "pixel_count_variance_ratio": round(
                    t.pixel_count_variance_ratio(), 2,
                ),
                "canonical_signature_first_key": (
                    t.canonical_signature()[0][0]
                    if t.canonical_signature() else None
                ),
                "position_drift_cells": t.position_drift_cells(),
                "recent_behaviour_events": recent_events,
                "hypothesis_ids": list(t.hypothesis_ids),
            }
            if obs_at is not None:
                entry["current_position"] = obs_at.centroid_cell
                entry["current_n_pixels"] = obs_at.n_pixels
                entry["current_bbox_logical"] = obs_at.bbox_logical
            entities_out.append(entry)

        rels_now = [
            {
                "type": r.type, "a": r.a, "b": r.b, "extra": r.extra,
            }
            for r in self.relationships_by_turn.get(turn, [])
        ]

        recent_actions = [
            {
                "turn": a.turn, "action": a.action,
                "agent_pos_before": a.agent_pos_before,
                "agent_pos_after": a.agent_pos_after,
                "observed_changes": [
                    {"track_id": tid, "kind": k, "details": d}
                    for (tid, k, d) in a.observed_changes
                ],
            }
            for a in self.actions
            if recent_cutoff <= a.turn <= turn
        ]

        return {
            "game_id": self.game_id,
            "turn": turn,
            "entities": entities_out,
            "relationships_now": rels_now,
            "recent_actions": recent_actions,
            "notes": list(self.notes),
            "hypothesis_ids": list(self.hypothesis_ids),
        }
