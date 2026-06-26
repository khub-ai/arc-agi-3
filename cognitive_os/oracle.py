"""Oracle triggers — the engine-side dispatch of ObserverQuery /
MediatorQuery.

Phase 4 built the *answering* side: an adapter can forward typed
queries to a VLM (``observer_query``) or text LLM (``mediator_query``)
and receive typed answers back.  But nothing in the engine ever
constructed a query.  Cached visual claims went stale with no
re-check; planner impasses produced no consultation; surprises
piled up with no explanation.

This module supplies the missing half: **triggers**.  A trigger is a
narrow bundle of

* a **detector** — a predicate over :class:`WorldState` that flips
  true when oracle help is the right next move,
* a **builder** — code that packages the situation into a typed
  :class:`ObserverQuery` / :class:`MediatorQuery`,
* a **handler** — code that integrates the typed answer back into
  the hypothesis store at appropriate source-prior credence.

Triggers are injected as a list into :func:`run_episode` and fire
once per step in declared order, after observation ingest and
before goal / plan work.  An empty list reproduces Phase 4 behaviour
exactly (no oracle traffic) — so every existing test and adapter
continues to pass untouched.

Standing-invariant compliance
-----------------------------
The triggers here are **generic over domain**.  A trigger fires on
"first observation of a new level" — not on "ls20 level 1"; on
"plan is None after N consecutive steps" — not on "ls20 impasse".
Game-specific triggering would violate invariant #1 and is
explicitly not done here.

Budget
------
The engine's :class:`LLMBudget` is enforced inside the adapter's
backend (see ``ChatBackend.answer_*``), so the trigger does not
re-check budget itself; if the budget is exhausted the backend
returns a zero-confidence answer, which the trigger's handler
treats as "no information" and moves on.

Capability audit
----------------
* **Problem-solving** — PRIMARY.  Roles and positions identified at
  level start give the planner something to reach toward long
  before symbolic miners can derive equivalent beliefs.
* **Debugging** — secondary.  Future triggers (``EXPLAIN_SURPRISE``)
  turn mispredicted events into abductive hypotheses.
* **Tool creation** — not yet.  The ``PROPOSE_TOOL`` trigger lives
  in a future phase.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set

from .claims import CausalClaim, PropertyClaim
from .conditions import (
    Condition,
    EntitiesVisuallyMatch,
    ResourceAbove,
    ResourceBelow,
)
from .types import (
    EntityModel,
    EntityVisualPatternChanged,
    Goal,
    GoalNode,
    GoalStatus,
    MediatorAnswer,
    MediatorQuery,
    MediatorQuestion,
    NodeType,
    ObserverAnswer,
    ObserverQuery,
    QuestionType,
    Scope,
    ScopeKind,
    WorldState,
    WorldStateSummary,
)
from . import goal_forest as _goal_forest
from . import hypothesis_store as _store

if TYPE_CHECKING:  # pragma: no cover
    from .adapters import Adapter
    from .config import EngineConfig


# ---------------------------------------------------------------------------
# WorldStateSummary construction — shared by Mediator-side triggers
# ---------------------------------------------------------------------------


def build_world_summary(
    ws:              WorldState,
    *,
    step:            int,
    focus_entities:  Optional[List[str]] = None,
    impasse_context: Optional[str]       = None,
) -> WorldStateSummary:
    """Construct a :class:`WorldStateSummary` for Mediator consumption.

    Curates the WorldState: trims entities to focus set (or all if
    ``focus_entities is None``), keeps only committed + contested
    hypotheses, truncates recent events to the last 20 steps.
    """
    if focus_entities is None:
        entities = dict(ws.entities)
    else:
        focus_set = set(focus_entities)
        entities = {eid: ent for eid, ent in ws.entities.items() if eid in focus_set}

    cfg = ws.config
    cred_cfg = getattr(cfg, "credence", None) if cfg is not None else None

    committed: List[Any] = []
    contested: List[Any] = []
    for h in ws.hypotheses.values():
        if cred_cfg is not None and h.credence.is_committed(cred_cfg):
            committed.append(h)
        elif cred_cfg is not None and (
            not h.credence.is_committed(cred_cfg)
            and not h.credence.is_abandoned(cred_cfg)
        ):
            contested.append(h)

    recent_events: List[Any] = []
    for obs in list(ws.observation_history)[-20:]:
        recent_events.extend(obs.events)

    # Surface non-terminal goals so Mediator questions like
    # PROPOSE_GOAL_LINKAGE can reason about their leaf conditions.
    # "Terminal" here means achieved/pruned/abandoned — the Mediator
    # should only be asked about goals still in play.
    active_goals: List[Any] = []
    terminal = {"achieved", "pruned", "abandoned"}
    for goal in ws.goal_forest.goals.values():
        if goal.root.status.value in terminal:
            continue
        active_goals.append(goal)

    return WorldStateSummary(
        step                 = step,
        agent                = dict(ws.agent),
        entities             = entities,
        committed_hypotheses = committed,
        contested_hypotheses = contested,
        active_goals         = active_goals,
        recent_events        = recent_events,
        impasse_context      = impasse_context,
        available_tools      = getattr(ws, "tool_registry", None),
    )


# ---------------------------------------------------------------------------
# Stable summary for EpisodeGoalLinkageTrigger
# ---------------------------------------------------------------------------


def _stable_linkage_summary(
    ws:     "WorldState",
    target: "Goal",
) -> WorldStateSummary:
    """Build a **cache-stable** :class:`WorldStateSummary` for
    :class:`EpisodeGoalLinkageTrigger`.

    The question "what concrete conditions cause the episode-win resource to
    become true?" depends only on *which entities exist* and *what visual
    properties they have* — it does not change with the agent's position,
    the current step number, which hypotheses have committed so far, or which
    events were recently observed.  All of those volatile fields produce a
    new SHA-1 cache key on every run, forcing live API calls even when the
    answer is already cached from a prior trajectory.

    This helper eliminates the volatility by zeroing every trajectory-specific
    field:

    * ``step`` → 0  (fixed sentinel; step is irrelevant to the causal question)
    * ``agent`` → {} (agent position irrelevant)
    * ``committed_hypotheses`` → [] (question is pre-evidence; what commits
      elsewhere cannot change which entity *causes* the win flag)
    * ``contested_hypotheses`` → []
    * ``recent_events`` → []
    * Entity ``first_seen_step`` / ``last_seen_step`` → 0 (timestamps differ
      across trajectories even for the same entity in the same level)
    * ``active_goals`` → [target] only (other goals are trajectory-specific)
    * ``impasse_context`` → fixed literal string (no step or id embedding)

    The resulting summary serialises identically across all runs on the same
    level layout → one live API call per level, all subsequent runs use the
    cache.

    This satisfies the Tier-1 principle (mechanical / stable queries → tool /
    cache, not repeated LLM calls).
    """
    import dataclasses as _dc

    # Entity copies with timestamps zeroed so they don't perturb the hash.
    stable_entities: Dict[str, EntityModel] = {}
    for eid, ent in ws.entities.items():
        try:
            stable_entities[eid] = _dc.replace(
                ent,
                first_seen_step=0,
                last_seen_step=0,
            )
        except TypeError:
            # Defensive fallback if EntityModel ever becomes non-dataclass.
            stable_entities[eid] = ent

    return WorldStateSummary(
        step                 = 0,
        agent                = {},
        entities             = stable_entities,
        committed_hypotheses = [],
        contested_hypotheses = [],
        active_goals         = [target],
        recent_events        = [],
        impasse_context      = (
            "The episode-success goal has an atomic resource-predicate leaf "
            "with no committed causal explanation.  "
            "Identify the concrete condition(s) whose occurrence makes this "
            "resource flag become true.  "
            "Return each as a CausalClaim with trigger=<concrete atomic "
            "condition> and effect=<resource predicate>.  "
            "Base your answer only on what the visible entities suggest; "
            "ignore the agent's current position and step count."
        ),
        available_tools      = getattr(ws, "tool_registry", None),
    )


# ---------------------------------------------------------------------------
# Trigger base class
# ---------------------------------------------------------------------------


class OracleTrigger(ABC):
    """Base class for all oracle-dispatch triggers.

    A trigger is stateful across steps within an episode — e.g. a
    trigger may record "I already scanned level L, don't re-scan".
    Reset is called at episode boundaries via :meth:`reset`.

    Subclasses must implement :meth:`maybe_dispatch`, which is
    invoked once per engine step.  The implementation is responsible
    for its own detector, query construction, adapter invocation,
    and answer handling.  Adapter exceptions MUST be swallowed: an
    oracle failure never takes down the episode loop.
    """

    #: Human-readable identifier, used in logging and in the
    #: ``source`` tag of any hypotheses the handler proposes.
    name: str = "oracle_trigger"

    def reset(self) -> None:
        """Called at episode start.  Subclasses override to zero
        per-episode state; the default is a no-op for stateless
        triggers."""

    @abstractmethod
    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        """If the detector is satisfied, build a query, dispatch it
        via the adapter, and handle the answer.  Otherwise return
        without side effects."""


# ---------------------------------------------------------------------------
# InitialFrameScanTrigger — the bootstrap Observer trigger
# ---------------------------------------------------------------------------


def _level_key(ws: WorldState) -> Any:
    """Extract a stable key identifying the current *level* from the
    agent state.

    Domains that do not have a level concept report ``0`` (or omit
    the key), so in that case the key degenerates to a single value
    for the whole episode and the trigger fires exactly once at step
    zero.  Domains that *do* have levels (ARC-AGI-3's ``levels_completed``,
    a sim environment's ``scene_id``, etc.) naturally cause the
    trigger to refire whenever the level changes.
    """
    # Prefer an explicit level marker if the adapter surfaced one.
    for key in ("level_idx", "level", "levels_completed", "scene_id"):
        if key in ws.agent:
            return (key, ws.agent[key])
    return ("level", 0)


class InitialFrameScanTrigger(OracleTrigger):
    """Bootstrap trigger: ask the Observer to enumerate every object
    visible in the initial frame of each game level.

    Rationale
    ---------
    Without a prior on what the objects in the scene *are*, the
    engine's symbolic miners start from zero: every coloured blob is
    just an entity with observed colours, no semantics.  The
    planner has nothing to plan *toward* and the explorer wanders.

    Letting a VLM look at the very first frame of a level — and only
    the first frame — gives the WorldState a one-shot injection of
    structured object-level knowledge (positions, coarse roles) that
    the symbolic layer can then confirm, contradict, and refine.
    The injection cost is bounded: at most one call per level per
    episode, cacheable across attempts by the adapter.

    Detector
    --------
    Fires exactly once per *level*, on the first step the level is
    observed.  Level identity is read from :func:`_level_key`; the
    trigger keeps a set of already-scanned level keys and skips
    any repeat.

    Handler
    -------
    Parses the Observer answer's ``result`` as a list of objects
    (each a dict with ``position``, ``role``, ``description``, and
    optional ``shape``/``colour``).  For each:

    * If an EntityModel already exists matching the reported
      position (within a tolerance), attach role to it.
    * Else register a new EntityModel seeded with the LLM's
      position and colour/shape hints.

    A ``PropertyClaim(entity_id, "role", role)`` is emitted at
    :attr:`SourcePriors.llm_proposer` credence, and a
    ``PropertyClaim(entity_id, "position", (x, y))`` is emitted
    similarly.  Both are scoped to :attr:`ScopeKind.LEVEL` — the
    role of an entity in level N is not, in general, the same as
    the role of the same-looking entity in level N+1.

    Tolerance
    ---------
    LLM position estimates are inexact.  ``position_tolerance`` is
    the grid-cell-equivalent radius within which a reported
    position "matches" an existing entity's position property.  The
    default (10 units) is generous for a 64×64 ARC frame; tighten
    for finer-grained domains.
    """

    name = "initial_frame_scan"

    def __init__(
        self,
        *,
        position_tolerance:  float = 10.0,
        urgency:             float = 0.6,
        min_verify_fraction: float = 0.5,
    ) -> None:
        self.position_tolerance  = position_tolerance
        self.urgency             = urgency
        self.min_verify_fraction = min_verify_fraction
        self._scanned_levels: Set[Any] = set()
        self._counter = 0

    def reset(self) -> None:
        self._scanned_levels.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _should_fire(self, ws: WorldState) -> bool:
        """True iff we have not yet scanned the current level."""
        key = _level_key(ws)
        if key in self._scanned_levels:
            return False
        # Need at least one observation to have a frame to scan.
        if not ws.observation_history:
            return False
        return True

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        if not self._should_fire(ws):
            return

        level_key = _level_key(ws)
        obs = ws.observation_history[-1]
        raw_frame = obs.raw_frame

        if not raw_frame:
            # Nothing visual to scan; mark as "scanned" anyway so we
            # don't re-try this level on later empty frames.
            self._scanned_levels.add(level_key)
            return

        self._counter += 1
        query = ObserverQuery(
            query_id = f"initial_frame_scan::level{level_key}::n{self._counter}",
            question = QuestionType.ENUMERATE_OBJECTS,
            targets  = [],                      # whole frame
            frames   = [raw_frame],
            urgency  = self.urgency,
            context  = (
                "Initial-frame scan for a new game level.  "
                "Enumerate every distinct visible object (coloured shape, "
                "sprite, region).  For each, report an estimated "
                "(x, y) position in frame coordinates, a short "
                "lowercase role label (e.g. \"agent\", \"wall\", "
                "\"target\", \"hazard\", \"resource\", \"counter\"), "
                "and a brief description.  Prefer lower confidence "
                "over guessing."
            ),
        )

        try:
            answer = adapter.observer_query(query)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "observer_query raised %r; continuing without answer", exc,
            )
            # Do NOT mark as scanned — we want to retry next step if
            # the failure was transient.
            return

        # Mark scanned regardless of confidence: an oracle that
        # returned "I don't know" is not going to do better on the
        # identical frame next step.
        self._scanned_levels.add(level_key)

        if answer is None or answer.confidence <= 0.0:
            return

        self._integrate_enumeration(ws, answer, step, raw_frame=raw_frame)

    # ------------------------------------------------------------------
    # Handler: integrate enumeration into WorldState
    # ------------------------------------------------------------------

    def _integrate_enumeration(
        self,
        ws:     WorldState,
        answer: ObserverAnswer,
        step:   int,
        *,
        raw_frame: Any = None,
    ) -> None:
        """Install the Observer's object list into the hypothesis
        store and entity map.

        The ``result`` is expected to be a list of dicts.  Each dict
        should have ``position`` (tuple / list of two numbers) and
        ``role`` (str).  Extra fields (``description``, ``shape``,
        ``colour``, ``bbox``) are stored as entity properties for
        later symbolic confirmation.

        Matching policy
        ---------------
        The LLM's object list may disagree with the engine's
        segmentation in both directions:

        * the engine may have aggregated what the LLM sees as
          multiple distinct objects (under-segmentation — common
          when same-colour regions are topologically connected);
        * the LLM may have hallucinated an object in a region the
          engine correctly sees as background or noise.

        We favour the LLM's segmentation on conflict.  An engine
        entity whose colour agrees with the LLM's object and whose
        bbox plausibly contains the LLM's position is matched; once
        matched, it is **consumed** — subsequent LLM objects in
        this pass cannot re-use the same engine entity.  Everything
        else mints a fresh ``scan_NN`` entity, optionally tagged
        with a ``_parent`` pointer to the engine entity that likely
        absorbed this sub-region.
        """
        objects = _coerce_object_list(answer.result)
        if not objects:
            return

        rationale_base = (
            f"InitialFrameScanTrigger @ step {step}; "
            f"confidence={answer.confidence:.2f}; "
            f"explanation={answer.explanation[:120]!r}"
        )

        # Consumed engine entity ids for this enumeration pass.
        # Each engine entity can absorb at most one LLM object's
        # role/initial_position claim — otherwise a single entity
        # ends up with contradictory role claims (see ls20 L1
        # attempt 1: e1 got both role=target and role=decor).
        consumed: Set[str] = set()

        for idx, obj in enumerate(objects):
            pos  = _coerce_position(obj.get("position"))
            role = obj.get("role")
            if not isinstance(role, str) or not role.strip():
                continue
            role = role.strip().lower()

            entity_id = self._match_or_mint(
                ws, pos, obj, step, idx, consumed=consumed,
                raw_frame=raw_frame,
            )
            if entity_id is None:
                # LLM-only mint was rejected by pixel verification —
                # the claimed region does not exist in the raw frame.
                # Drop this object silently; no role or position claims
                # should be derived from an unverified hallucination.
                continue

            # Property: role (scoped to LEVEL — roles may differ
            # across levels of the same game).
            _store.propose(
                ws,
                claim     = PropertyClaim(
                    entity_id = entity_id, property = "role", value = role),
                source    = "llm_proposer",
                scope     = Scope(kind=ScopeKind.LEVEL),
                step      = step,
                rationale = rationale_base,
            )

            # Property: position (scoped to STEP — position changes
            # freely; this is an initial-snapshot claim only).
            if pos is not None:
                _store.propose(
                    ws,
                    claim     = PropertyClaim(
                        entity_id = entity_id,
                        property  = "initial_position",
                        value     = pos),
                    source    = "llm_proposer",
                    scope     = Scope(kind=ScopeKind.LEVEL),
                    step      = step,
                    rationale = rationale_base,
                )

            # Optional: colour / shape / description as per-entity
            # property claims.  These are the LLM's hints; symbolic
            # miners confirm / contradict them.
            for propname in ("colour", "color", "shape", "description"):
                val = obj.get(propname)
                if isinstance(val, (str, int, float)) and val != "":
                    # Normalise "color" -> "colour" for consistency.
                    canonical = "colour" if propname in ("color", "colour") else propname
                    _store.propose(
                        ws,
                        claim     = PropertyClaim(
                            entity_id = entity_id,
                            property  = canonical,
                            value     = str(val)[:80],
                        ),
                        source    = "llm_proposer",
                        scope     = Scope(kind=ScopeKind.LEVEL),
                        step      = step,
                        rationale = rationale_base,
                    )

    def _match_or_mint(
        self,
        ws:       WorldState,
        pos:      Optional[Any],
        obj:      Dict[str, Any],
        step:     int,
        idx:      int,
        *,
        consumed:  Optional[Set[str]] = None,
        raw_frame: Any = None,
    ) -> Optional[str]:
        """Decide which engine entity this LLM object refers to,
        minting a fresh scan-sourced entity when no engine entity
        fits.

        Match precedence (first rule that fires wins):

        1. **Colour + bbox-contains-pos match.**  If the LLM object
           specifies a colour and some engine entity with that
           colour has a bbox containing the LLM's position, and
           that entity has not been consumed this pass, take it.
        2. **Colour + centroid distance.**  Same colour + L-inf
           centroid distance within ``self.position_tolerance``,
           not consumed.
        3. **Bbox-contains-pos without colour check.**  Only when
           the LLM gave no colour.
        4. **Mint new**.  On mint, if some same-colour engine
           entity's bbox *does* contain the LLM's position but was
           rejected by size/consume rules, record it as ``_parent``
           on the new scan entity.  That preserves the
           under-segmentation signal for later refinement.

        Pixel verification (mint path only).  The LLM routinely
        hallucinates small objects at plausible-sounding coordinates
        — a "target" at (32, 8) in an area that is actually empty
        backdrop.  When ``raw_frame`` is supplied AND the LLM gave
        both a ``colour`` and a ``bbox``, we verify the claim against
        the pixels: if less than :attr:`min_verify_fraction` of cells
        in the bbox match the claimed colour, return ``None``.  The
        caller drops such objects without minting an entity or
        emitting role/position claims.  Rules 1-3 are *not* gated on
        this check: a successful match against an engine entity is
        already behaviourally grounded.
        """
        consumed = consumed if consumed is not None else set()
        llm_colour = _coerce_colour(obj.get("colour") or obj.get("color"))
        llm_bbox   = _coerce_bbox(obj.get("bbox"))
        llm_size   = _bbox_size(llm_bbox)

        # Gather candidate engine entities with basic per-entity info.
        candidates: List[Dict[str, Any]] = []
        for eid, ent in ws.entities.items():
            if eid in consumed:
                continue
            cand = {
                "id":       eid,
                "colour":   _entity_colour(ent),
                "bbox":     _entity_bbox(ent),
                "pos":      _entity_position(ent),
                "area":     _coerce_number(ent.properties.get("area")),
            }
            candidates.append(cand)

        def _colour_ok(c: Dict[str, Any]) -> bool:
            return (
                llm_colour is None
                or c["colour"] is None
                or c["colour"] == llm_colour
            )

        def _size_ok(c: Dict[str, Any]) -> bool:
            # Reject candidates that are ≥10× the LLM's implied
            # object area.  Without an LLM bbox we cannot detect
            # under-segmentation; default to accept so the old
            # (bbox-less) behaviour is preserved.
            if llm_size is None or c["area"] is None:
                return True
            return c["area"] <= llm_size * 10.0

        # Rule 1: colour match + bbox contains pos.
        if pos is not None:
            for c in candidates:
                if not _colour_ok(c):
                    continue
                if c["bbox"] is None:
                    continue
                if not _bbox_contains(c["bbox"], pos):
                    continue
                if not _size_ok(c):
                    continue
                consumed.add(c["id"])
                return c["id"]

        # Rule 2: colour match + centroid distance.
        if pos is not None:
            best: Optional[Dict[str, Any]] = None
            best_dist = float("inf")
            for c in candidates:
                if not _colour_ok(c):
                    continue
                if c["pos"] is None:
                    continue
                if not _size_ok(c):
                    continue
                d = _l_inf(c["pos"], pos)
                if d < best_dist:
                    best_dist = d
                    best       = c
            if best is not None and best_dist <= self.position_tolerance:
                consumed.add(best["id"])
                return best["id"]

        # Rule 3: bbox-contains-pos without colour check (only when
        # the LLM gave no colour — otherwise this would re-introduce
        # the cross-colour merges we were trying to prevent).
        if pos is not None and llm_colour is None:
            for c in candidates:
                if c["bbox"] is None:
                    continue
                if not _bbox_contains(c["bbox"], pos):
                    continue
                if not _size_ok(c):
                    continue
                consumed.add(c["id"])
                return c["id"]

        # Pixel verification.  Before minting a fresh entity from
        # an LLM-only claim, require the pixels at the claimed bbox
        # to actually contain the claimed colour.  Skipped when the
        # caller has no raw frame (unit tests that exercise the
        # matching algebra in isolation) or when the LLM omitted
        # either bbox or colour (verification has nothing to check).
        if raw_frame is not None and llm_bbox is not None and llm_colour is not None:
            if not _verify_bbox_colour(
                raw_frame, llm_bbox, llm_colour, self.min_verify_fraction,
            ):
                return None

        # Rule 4: mint new.  Record an under-segmentation parent if
        # any same-colour engine entity's bbox overlaps this object's
        # footprint.  We prefer bbox-overlap (when the LLM gave a
        # bbox) over pos-containment because the LLM's reported
        # position is often the top-left corner rather than a point
        # inside the object's mass.
        parent_id: Optional[str] = None
        for eid, ent in ws.entities.items():
            col = _entity_colour(ent)
            ebbox = _entity_bbox(ent)
            if ebbox is None:
                continue
            if llm_colour is not None and col is not None and col != llm_colour:
                continue
            if llm_bbox is not None:
                if _bbox_overlap(ebbox, llm_bbox):
                    parent_id = eid
                    break
            elif pos is not None and _bbox_contains(ebbox, pos):
                parent_id = eid
                break

        entity_id = f"scan_{idx:02d}"
        suffix = 0
        while entity_id in ws.entities:
            suffix += 1
            entity_id = f"scan_{idx:02d}_{suffix}"

        properties: Dict[str, Any] = {"_source": "initial_frame_scan"}
        if parent_id is not None:
            properties["_parent"] = parent_id
        if llm_bbox is not None:
            properties["bbox"] = list(llm_bbox)
        if llm_colour is not None:
            properties["colour"] = llm_colour

        ws.entities[entity_id] = EntityModel(
            id              = entity_id,
            properties      = properties,
            first_seen_step = step,
            last_seen_step  = step,
            kind            = str(obj.get("role", "unknown")),
        )
        return entity_id


# ---------------------------------------------------------------------------
# EpisodeGoalLinkageTrigger — explain adapter-seeded resource goals
# ---------------------------------------------------------------------------


class EpisodeGoalLinkageTrigger(OracleTrigger):
    """Mediator trigger: given a seed goal whose leaf condition is an
    abstract resource predicate, ask common sense which concrete
    triggering conditions plausibly make the resource become true.

    Why the engine needs this
    -------------------------
    Adapters commonly seed a top-level "episode-success" goal whose
    only content is ``ResourceAbove("<win-flag>", …)``.  The engine
    knows that reaching this predicate is the goal — but, on its own,
    it has no committed ``CausalClaim`` whose ``effect`` matches, so
    :func:`cognitive_os.goal_forest.derive_subgoals_from_causal`
    cannot expand the opaque leaf into actionable subgoals.  The
    planner therefore sees an ATOM condition it has no way to reach,
    and all concrete goals an Observer may have proposed separately
    (e.g. "stand on the target") are *siblings* rather than
    *subgoals* of the episode goal — they neither inherit its
    priority nor disappear when it is achieved.

    Asking the Mediator once per episode to name the causal links
    from concrete atomic conditions to the abstract effect closes
    this gap without teaching the engine any game rules.  The
    resulting committed :class:`CausalClaim`\\(s) let the generic
    subgoal-derivation machinery expand the opaque leaf exactly as
    it would for any other causally-linked resource.

    Robotics analogue
    -----------------
    A task spec may define success as ``ResourceAbove("delivered", 0.5)``
    with no indication of how delivery is achieved.  Common sense
    supplies candidate triggers — "the package is at the drop-off
    position", "the recipient has signed", "a handover pose is
    reached" — each as a ``CausalClaim`` seeding a subgoal that the
    planner can then pursue through normal motion/pose primitives.

    Detector
    --------
    Fires at most once per level-episode boundary (same convention as
    :class:`InitialFrameScanTrigger` via :func:`_level_key`) AND
    only when:

    1. the goal forest contains at least one active :class:`Goal`
       whose root is an ATOM node, whose condition is a
       :class:`ResourceAbove` or :class:`ResourceBelow` predicate,
       and
    2. no committed :class:`CausalClaim` currently has an ``effect``
       matching that condition's ``canonical_key`` (i.e. the leaf is
       still opaque to ``derive_subgoals_from_causal``).

    Once it fires for a level — whether or not the Mediator returns
    useful claims — it does not refire for the same level in the
    same episode.  An adapter whose backend or budget refuses the
    query will simply have an unexpanded leaf; the rest of the
    engine continues to run.

    Builder
    -------
    Packs the offending goal's id into ``focus_goals`` so the
    Mediator knows exactly which leaf it is being asked about, and
    places ``MediatorQuestion.PROPOSE_GOAL_LINKAGE`` on the query.
    The :class:`WorldStateSummary` is built with no entity focus
    (whole-world context) because the trigger's question is about a
    goal, not a specific entity.

    Handler
    -------
    Iterates ``answer.proposed_claims`` and accepts each
    :class:`CausalClaim` whose effect matches the target leaf's
    canonical key.  Non-matching claims (and non-CausalClaim objects)
    are logged and dropped — we want this trigger to seed
    goal-linkage specifically, not absorb every claim the Mediator
    felt like volunteering.

    For each accepted claim the handler calls
    :func:`hypothesis_store.propose` with an ``initial_credence``
    derived from ``answer.confidence``:

        initial_credence = min(0.95, max(commit_threshold + 0.01,
                                         answer.confidence))

    The floor at ``commit_threshold + 0.01`` is a **bootstrap
    necessity**, not a normal-case override.  The llm_proposer
    source prior (0.30) is well below the commit threshold (0.85),
    so a plain ``propose`` call would register the causal claim as
    merely *contested* — and ``derive_subgoals_from_causal`` only
    walks *committed* hypotheses.  Until the claim is committed
    there is no subgoal, so the planner never drives the agent to a
    state that could confirm or contradict the trigger; the credence
    can never move.  Forcing commit on confident-enough Mediator
    answers breaks this deadlock.  The engine's normal symbolic
    pipelines still update the credence — an observed contradiction
    (action leads to trigger without effect firing) pulls the claim
    back below the commit line; an observed confirmation pushes it
    toward 1.0.  The LLM gets *one* bootstrap bet; symbolic evidence
    owns every subsequent update.

    By default (``low_confidence_only=False``), confidence ≥ 0.5 is
    enough to commit — the Mediator volunteering the claim with
    even moderate confidence is treated as sufficient bootstrap
    evidence.  Zero-confidence answers skip commit entirely
    (no claim was ever proposed).

    After ingestion the handler calls
    :func:`goal_forest.derive_subgoals_from_causal` on the target
    goal so the newly-committed link is reflected in the forest
    immediately (rather than waiting for the planner's next tick).

    Configuration
    -------------
    ``commit_min_confidence`` — minimum ``answer.confidence`` at which
    an accepted claim is force-committed via the credence floor.
    Defaults to 0.5, reflecting the bootstrap-necessity described
    above: if the Mediator was confident enough to name a candidate
    trigger, the engine acts on it rather than deadlock.  Raise
    (e.g. to 0.85) to require high-confidence answers only; lower
    (not below abandon_threshold) to let even hedged guesses
    bootstrap.  Below the threshold, claims are registered at their
    raw confidence — contested, awaiting symbolic evidence.

    ``low_confidence_floor`` — legacy alias retained for backwards
    compatibility.  If True, forces commit regardless of
    confidence.  Prefer ``commit_min_confidence`` for new code.
    """

    name = "episode_goal_linkage"

    def __init__(
        self,
        *,
        urgency:                float = 0.7,
        # Default is intentionally LOW (0.25) to reflect the bootstrap
        # deadlock: the LLM is reporting an honest prior over "is this
        # game-mechanic actually real?" while the engine is asking
        # "what should I bet on to explore?" — different questions.
        # Real calibration improves when the LLM has symbolic evidence
        # to reason from; that evidence can only accumulate once the
        # agent acts, which requires subgoal expansion, which requires
        # commit.  Raise to 0.5 or 0.85 to prefer more confident
        # answers once the first level has been solved and future
        # oracle calls can be primed with memoir-style context.
        commit_min_confidence:  float = 0.25,
        low_confidence_floor:   bool  = False,
    ) -> None:
        self.urgency               = urgency
        self.commit_min_confidence = float(commit_min_confidence)
        self.low_confidence_floor  = low_confidence_floor
        self._scanned_levels: Set[Any] = set()
        self._counter = 0

    def reset(self) -> None:
        self._scanned_levels.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _find_opaque_resource_goal(
        self, ws: WorldState,
    ) -> Optional[Goal]:
        """Return the first active goal whose root-ATOM condition is a
        resource predicate with no committed matching CausalClaim.
        """
        cfg = ws.config
        cred_cfg = getattr(cfg, "credence", None) if cfg is not None else None
        if cred_cfg is None:
            return None

        # Gather canonical-keys of committed causal effects so we can
        # check each candidate leaf in O(1).
        committed_effect_keys: Set[Any] = set()
        for h in _store.committed(ws):
            if isinstance(h.claim, CausalClaim):
                committed_effect_keys.add(h.claim.effect.canonical_key())

        for goal in ws.goal_forest.goals.values():
            root = goal.root
            if root.node_type != NodeType.ATOM:
                continue
            cond = root.condition
            if cond is None:
                continue
            if not isinstance(cond, (ResourceAbove, ResourceBelow)):
                continue
            # Skip already-achieved / pruned / abandoned goals.
            if root.status.value in ("achieved", "pruned", "abandoned"):
                continue
            if cond.canonical_key() in committed_effect_keys:
                continue
            return goal
        return None

    def _should_fire(
        self, ws: WorldState,
    ) -> Optional[Goal]:
        """Return the offending goal if the trigger should fire; else
        None.  Combines the per-level dedup with the opacity check so
        callers get a single decision + payload."""
        if not ws.observation_history:
            return None
        level_key = _level_key(ws)
        if level_key in self._scanned_levels:
            return None
        return self._find_opaque_resource_goal(ws)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        target = self._should_fire(ws)
        if target is None:
            return

        level_key = _level_key(ws)
        self._counter += 1

        # Build a cache-stable summary: volatile fields (step, agent,
        # committed_hypotheses, recent_events, entity timestamps) are
        # zeroed so the SHA-1 cache key is identical across trajectories
        # with the same level layout.  One live API call per level; all
        # subsequent runs hit the cache.  See _stable_linkage_summary.
        summary = _stable_linkage_summary(ws, target)
        query = MediatorQuery(
            query_id      = f"episode_goal_linkage::level{level_key}::n{self._counter}",
            question      = MediatorQuestion.PROPOSE_GOAL_LINKAGE,
            world_summary = summary,
            focus_goals   = [target.id],
            urgency       = self.urgency,
            context       = (
                "The engine has a top-level goal whose only leaf is an "
                "abstract resource predicate (typically an episode-win "
                "flag).  To decompose this into actionable subgoals we "
                "need CausalClaims whose `effect` matches the leaf and "
                "whose `trigger` is a concrete atomic condition — e.g. "
                "AtPosition(entity, pos), EntityInState(entity, state). "
                "Return one or more CausalClaims in `proposed_claims`; "
                "avoid bundling unrelated observations."
            ),
        )

        try:
            answer = adapter.mediator_query(query)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "mediator_query raised %r; continuing without answer", exc,
            )
            # Do NOT mark as scanned — retry next step if transient.
            return

        # Mark scanned regardless of outcome: we asked, we accept the
        # answer (or silence) as final for this level.
        self._scanned_levels.add(level_key)

        if answer is None or answer.confidence <= 0.0:
            return

        self._integrate_linkage(ws, target, answer, step)

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _integrate_linkage(
        self,
        ws:     WorldState,
        target: Goal,
        answer: MediatorAnswer,
        step:   int,
    ) -> None:
        """Commit matching CausalClaims from the Mediator's answer and
        re-run subgoal derivation on the target goal."""
        target_cond = target.root.condition
        if target_cond is None:
            return
        target_key = target_cond.canonical_key()

        cred_cfg = getattr(ws.config, "credence", None)
        commit_threshold = (
            cred_cfg.commit_threshold if cred_cfg is not None else 0.85
        )

        rationale_base = (
            f"EpisodeGoalLinkageTrigger @ step {step}; "
            f"target_goal={target.id!r}; "
            f"confidence={answer.confidence:.2f}; "
            f"explanation={answer.explanation[:120]!r}"
        )

        accepted = 0
        for claim in answer.proposed_claims:
            if not isinstance(claim, CausalClaim):
                continue
            if claim.effect.canonical_key() != target_key:
                continue

            # Derive initial_credence so that sufficiently-confident
            # answers commit immediately — enabling
            # derive_subgoals_from_causal to see them and breaking the
            # bootstrap deadlock described in the class docstring.
            cred = float(answer.confidence)
            should_commit = (
                self.low_confidence_floor
                or cred >= self.commit_min_confidence
            )
            if should_commit:
                cred = min(0.95, max(commit_threshold + 0.01, cred))
            # else: leave cred as the raw confidence; claim will be
            # contested and wait for corroborating evidence.

            _store.propose(
                ws,
                claim            = claim,
                source           = "llm_proposer",
                scope            = Scope(kind=ScopeKind.LEVEL),
                step             = step,
                rationale        = rationale_base,
                initial_credence = cred,
            )
            accepted += 1

        if accepted == 0:
            return

        # Re-derive subgoals on the target goal so the planner sees
        # the expansion this same step.  Routed through GoalManager so
        # alternate manager implementations (e.g. credence-weighted)
        # take effect uniformly.  See cognitive_os/goal_manager.py.
        try:
            from .goal_manager import DefaultGoalManager as _GM
            _GM(ws).derive_subgoals(target.id, step=step)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "GoalManager.derive_subgoals raised %r on %s",
                exc, target.id,
            )


# ---------------------------------------------------------------------------
# CausalClaimGoalLinkageTrigger — the dual of EpisodeGoalLinkageTrigger
# ---------------------------------------------------------------------------


class CausalClaimGoalLinkageTrigger(OracleTrigger):
    """Mediator trigger: when a newly-committed CausalClaim has no
    active goal whose precondition its ``effect`` would satisfy, ask
    the Mediator whether to introduce intermediate causal link(s) that
    bridge this orphan claim to a top-level goal.

    Why this exists
    ---------------
    :class:`EpisodeGoalLinkageTrigger` fires from the *goal* side:
    *"this goal has an opaque leaf — propose causal links pointing
    at it."*  But causal claims also arrive *bottom-up* — symbolic
    miners (:class:`~cognitive_os.miners.EntityActionCausalMiner` and
    its kin) discover that ``action A causes effect E`` from raw
    observation, with no prior reason to link E to any goal.

    The dual situation arises naturally on real games: the agent
    stumbles into a recolor / state-change tile, the miner records the
    causal link, but the engine has no goal whose precondition is
    *"the recolored entity has the matching colour."*  Without an
    interpretive nudge, the discovered claim sits in the hypstore
    forever as orphaned knowledge — the goal forest's backward-
    chaining (``derive_subgoals_from_causal``) needs a goal whose
    leaf condition matches a claim's ``effect`` to do anything, and
    no such goal exists.

    This trigger fills the gap.  It detects orphan causal claims and
    asks the Mediator whether each one is relevant to any active
    top-level goal — and, if so, what intermediate causal link(s)
    should be added.  The Mediator's typed answer feeds claims back
    through the standard hypothesis-store + goal-forest pipeline; no
    bespoke goal-declaration path is needed.

    Detector
    --------
    Per step:

    1. Snapshot the canonical-keys of every active-goal ATOM-leaf
       condition.
    2. Iterate committed CausalClaims.  A claim is *orphan* when its
       ``effect.canonical_key()`` is NOT in that snapshot — no current
       goal has a precondition this claim could directly close.
    3. Among orphan claims, skip those already asked-about this episode
       (per-key dedup via ``self._asked_keys``).
    4. If any orphan remains, fire one Mediator query about the
       highest-credence one.

    A run typically discovers many orphan claims (motion-related ones,
    irrelevant scene flux, etc.).  Per-key dedup means each candidate
    claim is queried at most once per episode, bounding cost.

    Builder
    -------
    Reuses :data:`MediatorQuestion.PROPOSE_GOAL_LINKAGE` — the
    question is structurally the same ("propose CausalClaim(s) that
    link …") just framed from the claim side rather than the goal side.
    The world summary carries the orphan claim in
    ``contested_hypotheses`` (so the Mediator sees it explicitly) and
    the candidate goals in ``active_goals``.  ``focus_goals`` lists
    every active top-level goal so the Mediator picks which (if any)
    the orphan claim is relevant to.  Context string explains the
    orphan-claim → goal pairing the Mediator should reason about.

    Handler
    -------
    Same shape as :class:`EpisodeGoalLinkageTrigger`'s
    ``_integrate_linkage``: each returned CausalClaim is committed at
    a credence floor that breaks the bootstrap deadlock when
    ``answer.confidence >= commit_min_confidence``; the goal-manager's
    ``derive_subgoals`` is then re-run on every active goal so the
    new link expands the relevant tree the same step.

    Domain-agnosticism
    ------------------
    Generic over claim contents, condition language, and goal shapes.
    A robotics adapter that mines ``EntityStateChanged`` causal claims
    benefits identically — orphan claims about gripper state, joint
    pose, sensor activation all flow through the same trigger.

    Configuration
    -------------
    ``commit_min_confidence`` — same role as in EpisodeGoalLinkageTrigger
    (default 0.25, intentionally low to break the bootstrap deadlock;
    raise once corroborating evidence accumulates across episodes).
    ``min_claim_credence`` — gates which committed claims the trigger
    considers orphan candidates.  Defaults to the engine's standard
    commit threshold (read from ``ws.config.credence``); only claims
    at or above commit are considered, so we don't ask the Mediator
    about contested low-credence claims.
    ``max_per_step`` — at most this many Mediator queries per step
    (default 1; each query has cost, and orphan claims will be
    discovered on subsequent steps anyway).
    """

    name = "causal_claim_goal_linkage"

    def __init__(
        self,
        *,
        urgency:               float = 0.6,
        commit_min_confidence: float = 0.25,
        max_per_step:          int   = 1,
    ) -> None:
        self.urgency               = urgency
        self.commit_min_confidence = float(commit_min_confidence)
        self.max_per_step          = int(max_per_step)
        # Per-episode dedup: canonical keys of orphan claims we've
        # already asked the Mediator about.
        self._asked_keys: Set[Any] = set()
        self._counter = 0

    def reset(self) -> None:
        self._asked_keys.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _orphan_claims(self, ws: WorldState) -> List["Hypothesis"]:
        """Return committed CausalClaim hypotheses whose effect doesn't
        match any active-goal ATOM leaf condition.

        Orphan = claim represents knowledge with no goal that uses it.
        """
        # Collect every leaf condition canonical key from active goals.
        active_leaf_keys: Set[Any] = set()

        def _walk(node: GoalNode) -> None:
            if node.status.value in ("achieved", "pruned", "abandoned"):
                return
            if node.node_type == NodeType.ATOM and node.condition is not None:
                active_leaf_keys.add(node.condition.canonical_key())
                return
            for child in (node.children or []):
                _walk(child)

        for goal in ws.goal_forest.goals.values():
            if goal.root.status.value in ("achieved", "pruned", "abandoned"):
                continue
            _walk(goal.root)

        # Iterate committed CausalClaims.  Sort by credence desc so the
        # highest-credence orphan is asked about first.
        candidates: List["Hypothesis"] = []
        for h in _store.committed(ws):
            if not isinstance(h.claim, CausalClaim):
                continue
            effect_key = h.claim.effect.canonical_key()
            if effect_key in active_leaf_keys:
                continue
            if effect_key in self._asked_keys:
                continue
            candidates.append(h)
        candidates.sort(
            key=lambda h: (h.credence.point if h.credence else 0.0),
            reverse=True,
        )
        return candidates

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        candidates = self._orphan_claims(ws)
        if not candidates:
            return

        # Only ask about claims whose target-goal context exists — if
        # there are no active top-level goals, there's no Mediator
        # question to pose.
        active_goals = [
            g for g in ws.goal_forest.goals.values()
            if g.root.status.value not in ("achieved", "pruned", "abandoned")
        ]
        if not active_goals:
            return

        for orphan in candidates[: self.max_per_step]:
            self._counter += 1
            self._asked_keys.add(orphan.claim.effect.canonical_key())

            # Build a minimal world summary; rely on the adapter's
            # mediator backend to serialise as the LLM expects.
            summary = WorldStateSummary(
                step                 = step,
                agent                = dict(ws.agent),
                committed_hypotheses = [orphan],
                contested_hypotheses = [],
                active_goals         = list(active_goals),
                impasse_context      = (
                    "A new CausalClaim has been mined whose effect does "
                    "not match any active goal's precondition.  The "
                    "engine asks: is this claim relevant to any of the "
                    "listed active goals?  If yes, propose intermediate "
                    "CausalClaim(s) bridging the orphan claim's effect "
                    "to a goal's leaf condition (or directly to the "
                    "goal's root condition).  If no, return zero "
                    "claims and confidence 0."
                ),
            )

            query = MediatorQuery(
                query_id      = (
                    f"causal_claim_goal_linkage::n{self._counter}::"
                    f"{repr(orphan.claim.effect.canonical_key())[:80]}"
                ),
                question      = MediatorQuestion.PROPOSE_GOAL_LINKAGE,
                world_summary = summary,
                focus_goals   = [g.id for g in active_goals],
                urgency       = self.urgency,
                context       = (
                    "Orphan CausalClaim discovered via symbolic mining. "
                    "Its effect is reproducible (committed credence) but "
                    "no active goal evaluates against it.  Propose "
                    "CausalClaim(s) only if the orphan effect is "
                    "instrumental to closing one of the listed goals — "
                    "otherwise the orphan is genuinely irrelevant and "
                    "the correct answer is zero claims with confidence 0. "
                    "When proposing, prefer concrete EntityInState / "
                    "AtPosition triggers chained directly to the active "
                    "goal's leaf condition."
                ),
            )

            try:
                answer = adapter.mediator_query(query)
            except Exception as exc:  # pragma: no cover — defensive
                import logging
                logging.getLogger("cognitive_os.oracle").warning(
                    "mediator_query raised %r; continuing without answer",
                    exc,
                )
                continue

            if answer is None or answer.confidence <= 0.0:
                continue
            if not answer.proposed_claims:
                continue

            self._integrate_orphan_linkage(ws, orphan, answer, step)

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _integrate_orphan_linkage(
        self,
        ws:     WorldState,
        orphan: "Hypothesis",
        answer: MediatorAnswer,
        step:   int,
    ) -> None:
        """Commit any returned CausalClaim and re-run subgoal derivation
        on every active goal."""
        cred_cfg = getattr(ws.config, "credence", None)
        commit_threshold = (
            cred_cfg.commit_threshold if cred_cfg is not None else 0.85
        )

        rationale_base = (
            f"CausalClaimGoalLinkageTrigger @ step {step}; "
            f"orphan_effect={orphan.claim.effect.canonical_key()!r}; "
            f"confidence={answer.confidence:.2f}; "
            f"explanation={answer.explanation[:120]!r}"
        )

        accepted = 0
        for claim in answer.proposed_claims:
            if not isinstance(claim, CausalClaim):
                continue
            cred = float(answer.confidence)
            if cred >= self.commit_min_confidence:
                cred = min(0.95, max(commit_threshold + 0.01, cred))
            _store.propose(
                ws,
                claim            = claim,
                source           = "llm_proposer",
                scope            = Scope(kind=ScopeKind.LEVEL),
                step             = step,
                rationale        = rationale_base,
                initial_credence = cred,
            )
            accepted += 1

        if accepted == 0:
            return

        # Re-run subgoal derivation on every active goal — any of them
        # might now match the freshly-committed bridging claim.
        try:
            from .goal_manager import DefaultGoalManager as _GM
            gm = _GM(ws)
            for g in list(ws.goal_forest.goals.values()):
                if g.root.status.value in ("achieved", "pruned", "abandoned"):
                    continue
                gm.derive_subgoals(g.id, step=step)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "GoalManager.derive_subgoals raised %r after orphan-claim "
                "linkage", exc,
            )


# ---------------------------------------------------------------------------
# VisualOrientationTrigger — compare goal-entity glyph to reference glyph
# ---------------------------------------------------------------------------


class VisualOrientationTrigger(OracleTrigger):
    """Observer trigger: check whether a goal entity's internal glyph is
    in the same orientation as a spatially-separate reference entity.

    Why the engine needs this
    -------------------------
    Some games (e.g. ARC-AGI-3 ls20) embed a *win condition* that is a
    visual conjunction: the agent must reach a target cell AND the glyph
    inside that cell must match a reference indicator's orientation.
    The Mediator's causal-linkage trigger only seeds a positional
    subgoal; the orientation half is invisible to it because there are
    no committed CausalClaims linking orientation to episode_won.

    This trigger supplies the missing half: it fires once per level,
    asks the Observer to compare the two glyphs, and if they differ
    seeds an :class:`~cognitive_os.conditions.EntitiesVisuallyMatch`
    goal that the explorer must satisfy before the navigation goal can
    finish the episode.

    Detector
    --------
    Fires at most once per level AND only when:

    1. At least two entities share the same dominant colour AND are
       spatially separated (non-overlapping bboxes).
    2. At least one of those entities has a committed ``role="target"``
       PropertyClaim.
    3. No agent slot ``ws.agent[f"_vm:{match_slot}"]`` has yet been
       written for this level (i.e. the trigger has not already fired).

    The detector fires after :class:`InitialFrameScanTrigger` has had a
    chance to label entity roles — so it needs to be placed after that
    trigger in the list returned by :func:`default_triggers`.

    Handler
    -------
    Builds an ``ObserverQuery(COMPARE_VISUAL_STATES, targets=[target_id,
    ref_id], frames=[current_frame])`` and dispatches it.  The answer
    writes ``ws.agent[f"_vm:{match_slot}"]`` to ``True`` or ``False``
    (or leaves it absent on zero-confidence).

    If the answer is ``False`` (mismatch) with sufficient confidence, the
    handler seeds a new top-level goal whose root condition is
    ``EntitiesVisuallyMatch(match_slot)`` at priority
    ``orientation_goal_priority`` (default 0.55 — above role-goal 0.5,
    below episode-goal 1.0, so the planner falls through to it when the
    episode goal is unplannable).

    Robotics analogue
    -----------------
    A manipulation task that requires the gripper to be in the same
    orientation as the part before insertion.  The trigger fires when
    an "insert-part" goal is seeded and there are two same-colour visual
    markers — the part and the gripper alignment indicator.  If the
    Observer says "orientations differ", the trigger seeds a
    ``RotateGripper`` goal that the motion planner can pursue.
    """

    name = "visual_orientation"

    def __init__(
        self,
        *,
        urgency:                   float = 0.65,
        min_confidence:            float = 0.3,
        orientation_goal_priority: float = 0.55,
    ) -> None:
        self.urgency                   = urgency
        self.min_confidence            = min_confidence
        self.orientation_goal_priority = orientation_goal_priority
        self._scanned_levels: Set[Any] = set()
        self._counter = 0

    def reset(self) -> None:
        self._scanned_levels.clear()
        self._counter = 0

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _find_pair(
        self, ws: WorldState,
    ) -> Optional[tuple]:
        """Return ``(target_id, ref_id, match_slot)`` or ``None``.

        Scans entities for a same-colour pair (target + non-target) with
        non-overlapping bboxes.  Prefers entities with committed
        ``role="target"`` PropertyClaims; falls back to those with
        ``kind="target"``.
        """
        # Group by colour
        by_colour: Dict[Any, List[str]] = {}
        for eid, ent in ws.entities.items():
            col = _entity_colour(ent)
            if col is None:
                continue
            by_colour.setdefault(col, []).append(eid)

        for col, eids in by_colour.items():
            if len(eids) < 2:
                continue

            # Identify target vs reference candidates by role/kind.
            target_id: Optional[str] = None
            ref_candidates: List[str] = []
            for eid in eids:
                ent = ws.entities[eid]
                role = ent.properties.get("role") or ent.kind
                if role == "target":
                    target_id = eid
                else:
                    ref_candidates.append(eid)

            if target_id is None or not ref_candidates:
                continue

            target_bbox = _entity_bbox(ws.entities[target_id])
            for ref_id in ref_candidates:
                ref_bbox = _entity_bbox(ws.entities[ref_id])
                if target_bbox is None or ref_bbox is None:
                    continue
                # Require spatial separation — identical bboxes mean the
                # same entity appeared under two keys, not two entities.
                if not _bbox_overlap(target_bbox, ref_bbox):
                    level_val = _level_key(ws)[1] if len(_level_key(ws)) > 1 else _level_key(ws)[0]
                    match_slot = f"lvl{level_val}_col{col}"
                    return (target_id, ref_id, match_slot)

        return None

    def _should_fire(self, ws: WorldState) -> Optional[tuple]:
        if not ws.observation_history:
            return None
        level_key = _level_key(ws)
        if level_key in self._scanned_levels:
            return None
        return self._find_pair(ws)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        pair = self._should_fire(ws)
        if pair is None:
            return

        target_id, ref_id, match_slot = pair
        level_key = _level_key(ws)
        self._counter += 1

        # Stash the target bbox under a sibling slot so
        # VisualPatternChangedTrigger can gate future re-checks to
        # mutations that actually occur at the target site (rather
        # than anywhere in the frame).  The key uses prefix
        # ``_vm_target_bbox:`` so it does NOT match the
        # ``_vm:``-boolean-slot filter in the downstream trigger.
        target_bbox = _entity_bbox(ws.entities[target_id])
        if target_bbox is not None:
            ws.agent[f"_vm_target_bbox:{match_slot}"] = target_bbox

        obs     = ws.observation_history[-1]
        raw_frame = obs.raw_frame
        if not raw_frame:
            self._scanned_levels.add(level_key)
            return

        query = ObserverQuery(
            query_id = f"visual_orientation::level{level_key}::n{self._counter}",
            question = QuestionType.COMPARE_VISUAL_STATES,
            targets  = [target_id, ref_id],
            frames   = [raw_frame],
            urgency  = self.urgency,
            context  = (
                f"Compare the internal glyph pattern of entity {target_id!r} "
                f"(the goal/target entity) with entity {ref_id!r} "
                f"(a reference indicator).  Determine whether the glyph "
                f"inside {target_id!r} is in the same orientation as the "
                f"glyph inside {ref_id!r}.  Report same_glyph=true if the "
                f"same type of pictogram appears in both, and "
                f"orientation_match=true only if same_glyph AND the "
                f"orientations are identical (not just similar)."
            ),
        )

        try:
            answer = adapter.observer_query(query)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "VisualOrientationTrigger: observer_query raised %r", exc,
            )
            return

        self._scanned_levels.add(level_key)

        if answer is None or answer.confidence < self.min_confidence:
            return

        self._integrate(ws, answer, match_slot, step)

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _integrate(
        self,
        ws:         WorldState,
        answer:     ObserverAnswer,
        match_slot: str,
        step:       int,
    ) -> None:
        """Write match result to agent slot; seed orientation goal on mismatch."""
        result = answer.result
        if not isinstance(result, dict):
            return

        orientation_match: bool = bool(result.get("orientation_match", False))

        # Write the stable slot — this is what EntitiesVisuallyMatch reads.
        slot_key = f"_vm:{match_slot}"
        ws.agent[slot_key] = orientation_match

        if orientation_match:
            # Already matched — no new goal needed.
            return

        # Mismatch: seed a goal to fix the orientation.
        goal_id = f"visual_match:{match_slot}"
        if goal_id in ws.goal_forest.goals:
            # Already seeded (e.g. by a prior VisualPatternChangedTrigger
            # call that re-checked and still found mismatch).
            return

        import uuid as _uuid
        node_id = f"{goal_id}::atom"
        goal = Goal(
            id         = goal_id,
            root       = GoalNode(
                id         = node_id,
                node_type  = NodeType.ATOM,
                condition  = EntitiesVisuallyMatch(match_slot=match_slot),
                status     = GoalStatus.OPEN,
                source     = "oracle:visual_orientation",
                priority   = self.orientation_goal_priority,
                created_at = step,
            ),
            priority   = self.orientation_goal_priority,
            source     = "oracle:visual_orientation",
            created_at = step,
        )
        _goal_forest.add_goal(ws, goal)


# ---------------------------------------------------------------------------
# VisualPatternChangedTrigger — re-check orientation after glyph mutation
# ---------------------------------------------------------------------------


class VisualPatternChangedTrigger(OracleTrigger):
    """Observer trigger: re-query COMPARE_VISUAL_STATES after a glyph
    rotation (EntityVisualPatternChanged) is detected in the current step.

    Why
    ---
    After the explorer tries an action at the scan entity and the glyph
    rotates, perception emits :class:`~cognitive_os.types.EntityVisualPatternChanged`.
    The orientation slot set by :class:`VisualOrientationTrigger` is now
    stale — it reflects the OLD orientation.  This trigger fires as soon
    as the mutation event appears, re-queries the Observer, and updates
    the slot.  If the new orientation matches the reference, the slot
    becomes ``True`` and the :class:`~cognitive_os.conditions.EntitiesVisuallyMatch`
    condition evaluates to ``True``, which achieves the orientation goal
    and unblocks the episode.

    Detector
    --------
    Fires when the LATEST observation's event list contains at least one
    :class:`~cognitive_os.types.EntityVisualPatternChanged` event AND
    a ``_vm:{match_slot}`` slot exists in ``ws.agent`` (which means
    :class:`VisualOrientationTrigger` has already fired and we have a
    reference pair to compare against).

    Robustness
    ----------
    Entity IDs are not preserved across the mutation (the old maroon
    entity disappeared and a new one appeared).  The trigger uses the
    ``bbox`` field of the ``EntityVisualPatternChanged`` event to
    identify the mutation region, then finds the new entity overlapping
    that region in the current WorldState and uses it as ``target_id``
    for the fresh COMPARE_VISUAL_STATES query.

    There is no per-level dedup: the trigger fires every time a mutation
    is detected, so that multiple rotations (needed when the glyph has
    four distinct orientations) each trigger a fresh check.

    Robotics analogue
    -----------------
    After a gripper rotation command, the vision sub-system re-checks
    the alignment indicator.  This trigger is the structured "re-check
    after actuator command" loop that avoids drifting on stale oracle
    state.
    """

    name = "visual_pattern_changed"

    def __init__(
        self,
        *,
        urgency:        float = 0.7,
        min_confidence: float = 0.3,
    ) -> None:
        self.urgency        = urgency
        self.min_confidence = min_confidence
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _get_mutation_events(self, ws: WorldState) -> List[EntityVisualPatternChanged]:
        """Return EntityVisualPatternChanged events from the latest obs."""
        if not ws.observation_history:
            return []
        obs = ws.observation_history[-1]
        return [e for e in obs.events if isinstance(e, EntityVisualPatternChanged)]

    def _find_active_match_slots(self, ws: WorldState) -> List[str]:
        """Return all ``_vm:*`` slot keys currently in ``ws.agent``."""
        return [k for k in ws.agent if k.startswith("_vm:")]

    def _should_fire(
        self, ws: WorldState,
    ) -> Optional[tuple]:
        """Return (mutation_event, slot_key, new_entity_id, ref_entity_colour)
        or None if the trigger should not fire.

        Locality gate: for each active ``_vm:`` slot we require the
        mutation's bbox to overlap the stored target bbox.  This
        prevents the trigger from firing on unrelated same-colour
        mutations elsewhere in the frame (e.g. background region
        shape churn).  The target bbox was stashed by
        :class:`VisualOrientationTrigger` under
        ``_vm_target_bbox:{match_slot}`` when it fired.
        """
        mutations = self._get_mutation_events(ws)
        if not mutations:
            return None
        slots = self._find_active_match_slots(ws)
        if not slots:
            return None  # VisualOrientationTrigger hasn't fired yet

        # Find a (mutation, slot) pair where the mutation bbox overlaps
        # the slot's stored target bbox.  If no slot has a bbox (back-
        # compat for pre-gate state) we fall back to the prior
        # first-mutation-first-slot pairing.
        chosen: Optional[tuple] = None  # (evt, slot_key, match_slot)
        for slot_key in slots:
            match_slot = slot_key[len("_vm:"):]
            target_bbox_key = f"_vm_target_bbox:{match_slot}"
            target_bbox = ws.agent.get(target_bbox_key)
            if target_bbox is None:
                continue
            for evt in mutations:
                if _bbox_overlap(evt.bbox, target_bbox):
                    chosen = (evt, slot_key, match_slot)
                    break
            if chosen is not None:
                break
        if chosen is None:
            # No mutation landed at a known target site — suppress.
            return None
        evt, slot_key, match_slot = chosen

        # Find current entity at the mutation bbox (the "after" entity).
        # It may have a new entity ID if perception assigned one.
        new_entity_id = evt.entity_id_after
        if new_entity_id not in ws.entities:
            # Try to find a same-colour entity near the mutation bbox.
            new_entity_id = self._find_entity_near_bbox(
                ws, evt.colour, evt.bbox,
            )
        if new_entity_id is None:
            return None

        # Find reference entity: look for a different entity that is NOT
        # new_entity_id but has the same colour and a non-overlapping bbox.
        ref_id = self._find_reference(ws, new_entity_id, evt.colour, evt.bbox)
        if ref_id is None:
            return None

        return (evt, slot_key, match_slot, new_entity_id, ref_id)

    def _find_entity_near_bbox(
        self,
        ws:     WorldState,
        colour: Any,
        bbox:   tuple,
    ) -> Optional[str]:
        """Find the entity (by colour + overlapping bbox) nearest the
        mutation site.  Returns entity ID or None."""
        for eid, ent in ws.entities.items():
            if _entity_colour(ent) != colour:
                continue
            ebbox = _entity_bbox(ent)
            if ebbox is None:
                continue
            if _bbox_overlap(ebbox, bbox):
                return eid
        return None

    def _find_reference(
        self,
        ws:       WorldState,
        skip_id:  str,
        colour:   Any,
        near_bbox: tuple,
    ) -> Optional[str]:
        """Find a same-colour entity that is spatially separate from the
        mutation site — the reference indicator."""
        for eid, ent in ws.entities.items():
            if eid == skip_id:
                continue
            if _entity_colour(ent) != colour:
                continue
            ebbox = _entity_bbox(ent)
            if ebbox is None:
                continue
            if not _bbox_overlap(ebbox, near_bbox):
                return eid
        return None

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def maybe_dispatch(
        self,
        ws:      WorldState,
        adapter: "Adapter",
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        payload = self._should_fire(ws)
        if payload is None:
            return

        evt, slot_key, match_slot, new_entity_id, ref_id = payload
        self._counter += 1

        obs       = ws.observation_history[-1]
        raw_frame = obs.raw_frame
        if not raw_frame:
            return

        query = ObserverQuery(
            query_id = f"visual_pattern_changed::step{step}::n{self._counter}",
            question = QuestionType.COMPARE_VISUAL_STATES,
            targets  = [new_entity_id, ref_id],
            frames   = [raw_frame],
            urgency  = self.urgency,
            context  = (
                f"The glyph in entity {new_entity_id!r} just changed (colour "
                f"{evt.colour}, bbox {evt.bbox}).  Re-compare it to the "
                f"reference entity {ref_id!r}.  Report same_glyph and "
                f"orientation_match based on the CURRENT frame only."
            ),
        )

        try:
            answer = adapter.observer_query(query)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "VisualPatternChangedTrigger: observer_query raised %r", exc,
            )
            return

        if answer is None or answer.confidence < self.min_confidence:
            return

        self._integrate(ws, answer, slot_key, match_slot, step)

    # ------------------------------------------------------------------
    # Handler
    # ------------------------------------------------------------------

    def _integrate(
        self,
        ws:         WorldState,
        answer:     ObserverAnswer,
        slot_key:   str,
        match_slot: str,
        step:       int,
    ) -> None:
        result = answer.result
        if not isinstance(result, dict):
            return

        orientation_match = bool(result.get("orientation_match", False))
        ws.agent[slot_key] = orientation_match

        if orientation_match:
            # Goal achieved: mark the EntitiesVisuallyMatch goal as achieved
            # so the engine doesn't keep spinning on it.
            goal_id = f"visual_match:{match_slot}"
            goal = ws.goal_forest.goals.get(goal_id)
            if goal is not None:
                try:
                    _goal_forest.mark_status(
                        ws, goal_id=goal_id, status=GoalStatus.ACHIEVED,
                    )
                except Exception:  # pragma: no cover — defensive
                    pass


# ---------------------------------------------------------------------------
# Default trigger suite
# ---------------------------------------------------------------------------


class VisualRecognitionTrigger(OracleTrigger):
    """LLM escape hatch for the harness-side VisualStore.

    Fires when a freshly-observed entity has no shared multi-key
    fingerprint with anything in the store at any abstraction level
    (bitmap_id / shape_id / topo_id / scaled_id) — i.e. the cheap
    deterministic perception tier returned no match.

    Detector
    --------
    Scans ``ws.agent['_visual_store']`` (when present) for entities
    flagged as "novel" by the harness.  An entity is novel when:

      * It was first observed within the last
        ``recency_window`` steps, AND
      * No record at any abstraction level shares any of its keys with
        a previously-known role (i.e. ``role_hypotheses`` is empty
        across all coarser-key matches), AND
      * The entity has been asked about at most ``max_per_episode``
        times this episode (per-bitmap_id dedup via
        ``self._asked_keys``).

    Builder
    -------
    :data:`MediatorQuestion.RECOGNIZE_ENTITY`.  The world summary
    carries the new entity's bitmap (encoded as PNG bytes via the
    adapter's payload mechanism) plus low-confidence candidates
    surfaced by the store's coarser-key lookups.  Context string asks
    the VLM:

      "Here is an entity we just observed.  Below are the closest
       harness-side matches at progressively coarser abstraction
       levels (exact pixels, palette-permutation, topology, scale-
       normalised).  Is this similar to something you recognise?
       What role might it play?"

    Handler
    -------
    Reply yields:

      * :class:`PropertyClaim` ``role`` for the new entity (e.g.
        ``role=rotation_trigger``) at credence derived from
        ``answer.confidence``.  The store's ``annotate_role`` method
        is called so the role hypothesis persists alongside the
        bitmap record.
      * Optional :class:`EntityEquivalenceClaim`s linking the new
        entity to known ones at the inferred semantic-similarity
        tier.  (Future work — current implementation just commits
        role claims.)

    Phase status
    ------------
    Skeleton trigger.  Adapter ``mediator_query()`` for the ARC
    adapter currently returns zero-confidence answers (same status
    as :class:`CausalClaimGoalLinkageTrigger` until the LLM bridge
    lands), so the trigger fires correctly and is no-op until that
    bridge is wired.  The detector + builder logic is in place so
    when the bridge is added, recognition queries flow without
    further integration work.
    """

    name = "visual_recognition"

    def __init__(
        self,
        *,
        urgency: float = 0.5,
        max_per_episode: int = 6,
        commit_min_confidence: float = 0.30,
        recency_window: int = 3,
    ) -> None:
        self.urgency               = float(urgency)
        self.max_per_episode       = int(max_per_episode)
        self.commit_min_confidence = float(commit_min_confidence)
        self.recency_window        = int(recency_window)
        self._asked_keys: Set[Any] = set()
        self._counter:    int      = 0

    def maybe_dispatch(
        self,
        ws,
        adapter,
        step:    int,
        cfg:     "EngineConfig",
    ) -> None:
        store = ws.agent.get("_visual_store") if ws.agent else None
        if store is None:
            return
        if len(self._asked_keys) >= self.max_per_episode:
            return

        # Find a novel entity: recently first-seen AND no role
        # hypothesis attached to any of its abstraction-level peers.
        try:
            all_entities = store.all()
        except Exception:
            return

        novel = None
        for rec in all_entities:
            if int(step) - int(rec.first_seen_step) > self.recency_window:
                continue
            if rec.bitmap_id in self._asked_keys:
                continue
            # Coarse-key peers — anything sharing shape/topo/scaled keys.
            peers: list = []
            for ab in ("shape_id", "topo_id", "scaled_id"):
                key = getattr(rec, ab, None)
                if key:
                    peers.extend(store.lookup(key))
            if any(p.role_hypotheses for p in peers):
                # A peer at some abstraction has a known role — skip;
                # the harness can already chain via that peer.
                continue
            novel = rec
            break

        if novel is None:
            return

        self._counter += 1
        self._asked_keys.add(novel.bitmap_id)

        # Build the world summary.  Carrying the bitmap rendering is
        # the adapter's responsibility (it encodes payload as PNG
        # base64 in to_dict / mediator backend serialisation).
        summary = WorldStateSummary(
            step                 = step,
            agent                = dict(ws.agent),
            committed_hypotheses = [],
            contested_hypotheses = [],
            active_goals         = list(
                g for g in ws.goal_forest.goals.values()
                if g.root.status.value not in ("achieved", "pruned", "abandoned")
            ),
            impasse_context      = (
                f"A new entity {novel.bitmap_id} {novel.annotation} was "
                f"observed at step {novel.first_seen_step}.  Harness-side "
                f"fingerprint matching at every abstraction level "
                f"(bitmap, shape, topology, scaled) returned no peer "
                f"with a known role.  Asking the VLM whether this entity "
                f"is similar to anything previously recognised, and "
                f"what role it might play."
            ),
        )

        query = MediatorQuery(
            query_id      = f"visual_recognition::n{self._counter}::{novel.bitmap_id}",
            question      = MediatorQuestion.RECOGNIZE_ENTITY,
            world_summary = summary,
            focus_goals   = [],
            urgency       = self.urgency,
            context       = (
                "Identify or describe the visual entity above.  If you "
                "recognise it as a kind of object you've seen elsewhere "
                "(trigger, indicator, hazard, goal, ...), report that "
                "as a role hypothesis.  If you don't recognise it, "
                "return zero claims with confidence 0."
            ),
        )

        try:
            answer = adapter.mediator_query(query)
        except Exception as exc:  # pragma: no cover — defensive
            import logging
            logging.getLogger("cognitive_os.oracle").warning(
                "mediator_query (RECOGNIZE_ENTITY) raised %r; continuing",
                exc,
            )
            return

        if answer is None or answer.confidence <= 0.0:
            return
        if not answer.proposed_claims:
            return

        # Handler: commit role hypotheses.  The current minimal
        # implementation walks proposed PropertyClaim(role=...) and
        # passes them to the VisualStore's annotate_role.  Future
        # work: also commit EntityEquivalenceClaims here.
        for claim in answer.proposed_claims:
            try:
                if not hasattr(claim, "property") or claim.property != "role":
                    continue
                store.annotate_role(
                    bitmap_id = novel.bitmap_id,
                    role      = str(getattr(claim, "value", "")),
                    credence  = float(answer.confidence),
                )
            except Exception:
                continue


def default_triggers() -> List[OracleTrigger]:
    """The default trigger set used by :func:`run_episode` when the
    caller does not pass an explicit list.

    Phase 5b ships:
      * :class:`InitialFrameScanTrigger` — Observer-side bootstrap.
      * :class:`EpisodeGoalLinkageTrigger` — Mediator-side causal
        linkage for opaque adapter-seeded resource goals.

    GAP 21 adds:
      * :class:`VisualOrientationTrigger` — fires after
        ``InitialFrameScanTrigger`` has labelled entity roles; compares
        goal-entity glyph to reference-indicator glyph; seeds an
        :class:`~cognitive_os.conditions.EntitiesVisuallyMatch` goal on
        mismatch so the explorer is directed to fix the orientation.
      * :class:`VisualPatternChangedTrigger` — re-fires the COMPARE
        query whenever a glyph mutation is detected; updates the slot
        immediately so that orientation-match status is always current.

    Trigger order matters: ``VisualOrientationTrigger`` must come after
    ``InitialFrameScanTrigger`` so entity roles are available; both must
    come before ``EpisodeGoalLinkageTrigger`` so the orientation goal is
    seeded before the navigation subgoal is planned.
    ``VisualPatternChangedTrigger`` can be anywhere after the orientation
    trigger has had a chance to seed the match slot.

    Future phases add impasse-driven (``SUGGEST_STRATEGY``),
    surprise-driven (``EXPLAIN_SURPRISE``), and cached-claim
    revalidation (``ObserverQuery.STILL_SIMILAR``) triggers.
    """
    return [
        InitialFrameScanTrigger(),
        VisualOrientationTrigger(),
        VisualPatternChangedTrigger(),
        EpisodeGoalLinkageTrigger(),
        CausalClaimGoalLinkageTrigger(),
        VisualRecognitionTrigger(),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_object_list(result: Any) -> List[Dict[str, Any]]:
    """The Observer's ``result`` field for ENUMERATE_OBJECTS should
    be a list of dicts, but real LLM replies can be anything.  We
    accept several plausible shapes and return ``[]`` on anything
    unusable."""
    if isinstance(result, list):
        return [x for x in result if isinstance(x, dict)]
    if isinstance(result, dict):
        # Common LLM shapes: {"objects": [...]} or {"entities": [...]}
        for key in ("objects", "entities", "items", "result"):
            value = result.get(key)
            if isinstance(value, list):
                return [x for x in value if isinstance(x, dict)]
    return []


def _coerce_position(pos: Any) -> Optional[tuple]:
    """Accept a (x, y) / [x, y] / {"x": x, "y": y} / {"row": r, "col": c}
    form and return a 2-tuple of floats, else None."""
    if pos is None:
        return None
    if isinstance(pos, (list, tuple)) and len(pos) >= 2:
        try:
            return (float(pos[0]), float(pos[1]))
        except (TypeError, ValueError):
            return None
    if isinstance(pos, dict):
        x = pos.get("x", pos.get("col", pos.get("cx")))
        y = pos.get("y", pos.get("row", pos.get("cy")))
        if x is not None and y is not None:
            try:
                return (float(x), float(y))
            except (TypeError, ValueError):
                return None
    return None


def _entity_position(ent: EntityModel) -> Optional[tuple]:
    """Pull a 2-D position out of an EntityModel's properties, if any."""
    for key in ("position", "pos", "initial_position", "centroid"):
        p = _coerce_position(ent.properties.get(key))
        if p is not None:
            return p
    # Some adapters store x / y separately.
    x = ent.properties.get("x")
    y = ent.properties.get("y")
    if x is not None and y is not None:
        try:
            return (float(x), float(y))
        except (TypeError, ValueError):
            return None
    return None


def _l_inf(a: tuple, b: tuple) -> float:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _coerce_number(x: Any) -> Optional[float]:
    """Accept an int/float or a numeric string; else None."""
    if isinstance(x, bool):  # bool is subclass of int, but not what we want
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x)
        except ValueError:
            return None
    return None


def _coerce_colour(x: Any) -> Optional[int]:
    """Coerce the LLM / engine's colour field to a palette index.

    ARC-AGI-3 palettes are 0-15.  The LLM sometimes stringifies
    ("\"3\"") and sometimes gives a name ("red").  We accept ints
    and numeric strings; named colours degrade to None so we don't
    guess a palette mapping and silently mis-match.
    """
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float) and x.is_integer():
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            try:
                return int(s)
            except ValueError:
                return None
    return None


def _coerce_bbox(x: Any) -> Optional[tuple]:
    """Accept ``[x_min, y_min, x_max, y_max]`` (4 numbers) and return
    a 4-tuple; else None.

    Engines and LLMs use subtly different bbox conventions (inclusive
    vs exclusive max, x/y vs row/col).  We treat both endpoints as
    *inclusive* because that is what the ARC-AGI-3 adapter's
    perception reports.  An LLM that returns exclusive max is off by
    one at the boundary — tolerable for matching purposes.
    """
    if x is None:
        return None
    if isinstance(x, (list, tuple)) and len(x) >= 4:
        try:
            vals = tuple(float(v) for v in x[:4])
        except (TypeError, ValueError):
            return None
        return vals
    if isinstance(x, dict):
        keys_by_shape = [
            ("x_min", "y_min", "x_max", "y_max"),
            ("xmin",  "ymin",  "xmax",  "ymax"),
            ("x",     "y",     "w",     "h"),
            ("left",  "top",   "right", "bottom"),
        ]
        for keys in keys_by_shape:
            if all(k in x for k in keys):
                try:
                    a, b, c, d = (float(x[k]) for k in keys)
                except (TypeError, ValueError):
                    return None
                # For (x, y, w, h) convert to (x_min, y_min, x_max, y_max).
                if keys == ("x", "y", "w", "h"):
                    return (a, b, a + c, b + d)
                return (a, b, c, d)
    return None


def _bbox_size(bbox: Optional[tuple]) -> Optional[float]:
    """Area of a bbox in the same units as bbox coordinates.

    Uses ``(x_max - x_min + 1) * (y_max - y_min + 1)`` to match the
    engine's convention that bbox extremes are inclusive."""
    if bbox is None:
        return None
    w = bbox[2] - bbox[0] + 1
    h = bbox[3] - bbox[1] + 1
    if w < 0 or h < 0:
        return None
    return float(w * h)


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    """Inclusive-boundary bbox overlap (non-empty intersection)."""
    return not (
        a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]
    )


def _verify_bbox_colour(
    raw_frame:    Any,
    bbox:         tuple,
    colour:       int,
    min_fraction: float,
) -> bool:
    """True iff the LLM's (bbox, colour) claim is grounded in pixels.

    We look at every cell inside ``bbox`` and count how many carry
    the claimed ``colour``.  If the fraction clears ``min_fraction``
    AND at least 4 cells match, the claim is considered grounded and
    we allow minting.  Otherwise the LLM has hallucinated an object
    at this location and we reject.

    Tolerance of bbox-edge slop.  The LLM sometimes reports a bbox
    that overshoots the real object by a pixel or two.  A fraction
    threshold (not a strict "all cells match") absorbs that while
    still catching the 25–30%-match hallucinations observed on ls20.

    Robust to ``raw_frame`` shape issues (non-rectangular, None
    rows, short rows) — any cell whose index is out of range is
    simply counted as a mismatch.
    """
    if raw_frame is None or bbox is None or colour is None:
        return True          # nothing to verify against — don't block minting
    try:
        r0 = int(bbox[0]); c0 = int(bbox[1])
        r1 = int(bbox[2]); c1 = int(bbox[3])
    except (TypeError, ValueError):
        return True
    if r0 > r1 or c0 > c1:
        return False
    total = 0
    matches = 0
    nrows = len(raw_frame)
    for r in range(r0, r1 + 1):
        if r < 0 or r >= nrows:
            total += (c1 - c0 + 1)
            continue
        row = raw_frame[r]
        ncols = len(row)
        for c in range(c0, c1 + 1):
            total += 1
            if 0 <= c < ncols and row[c] == colour:
                matches += 1
    if total <= 0:
        return False
    if matches < 4:
        return False
    return (matches / total) >= min_fraction


def _bbox_contains(bbox: tuple, pos: tuple) -> bool:
    """Inclusive-boundary point-in-bbox test with 1-unit slack on
    each side.  The slack absorbs the LLM's ±1-pixel rounding on
    position estimates — which is otherwise a common source of
    false negatives at box edges."""
    x, y = pos[0], pos[1]
    return (
        bbox[0] - 1.0 <= x <= bbox[2] + 1.0
        and bbox[1] - 1.0 <= y <= bbox[3] + 1.0
    )


def _entity_colour(ent: EntityModel) -> Optional[int]:
    """Pull a palette-index colour from an EntityModel's properties."""
    for key in ("colour", "color"):
        v = _coerce_colour(ent.properties.get(key))
        if v is not None:
            return v
    return None


def _entity_bbox(ent: EntityModel) -> Optional[tuple]:
    """Pull a 4-number bbox from an EntityModel's properties."""
    for key in ("bbox", "bounding_box"):
        b = _coerce_bbox(ent.properties.get(key))
        if b is not None:
            return b
    return None
