"""Game characterization — LLM-sourced scenario priors.

The engine core has no prior on *what kind of scenario* it is solving.
A level begins, and the hypothesis store has only pixel observations;
there is no reason yet to prefer any concrete decomposition of the
top-level ``episode_won`` goal.  :class:`InitialFrameScanTrigger`
gives the LLM one chance to enumerate objects, but that trigger asks
for precise pixel coordinates — the thing the LLM is *worst* at — and
its hallucinations must be pixel-verified away.

This module answers a different, higher-level question the LLM is
actually good at: **what kind of game is this?**  World knowledge
about maze navigation, collection tasks, chases, puzzles, etc. lives
naturally in the LLM.  We ask:

* What genre does the scene resemble?
* What is the typical win condition?
* What roles / characters are likely present?
* How does the player usually act?

The reply combines a free-form ``narrative`` (human-readable, also
used as context when priming future levels) with a small set of
open-string structured fields consumed by the decomposer (not yet
built — piece 2) to synthesise intermediate goals under
``_EPISODE_GOAL_ID``.

Responsibilities split
----------------------
* :class:`GameCharacterizationTrigger` — the ``OracleTrigger``
  subclass that fires once per new level, dispatches a
  ``CHARACTERIZE_GAME`` Observer query, and writes the characterisation
  into ``ws.hypotheses`` as ``PropertyClaim("_game", k, v)`` claims.
* :class:`CharacterizationStore` — a tag-indexed JSON store of
  *confirmed* characterisations (those that were active when a level
  was completed).  The trigger queries this store before asking the
  LLM, passes relevant priors into the prompt, and writes back on
  level-complete.

Both live adapter-side; the engine core is unchanged beyond a single
``QuestionType.CHARACTERIZE_GAME`` enum label.  They will generalise
into the core when a second domain adopts them.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from cognitive_os import (
    ObserverQuery,
    WorldState,
)
from cognitive_os.claims import PropertyClaim
from cognitive_os.oracle import OracleTrigger, _level_key
from cognitive_os.types import QuestionType, Scope, ScopeKind
from cognitive_os import hypothesis_store as _store

if TYPE_CHECKING:  # pragma: no cover
    from cognitive_os.adapters import Adapter
    from cognitive_os.config import EngineConfig


_LOG = logging.getLogger(__name__)

# Synthetic entity id used to carry game-level (as opposed to
# per-object) property claims.  Chosen to not collide with any real
# entity id the engine's segmentation might produce — real entities
# start with ``e`` or ``scan_``.
GAME_ENTITY_ID = "_game"

# Characterisation fields persisted as PropertyClaim values.  Kept
# here so the decomposer (piece 2) and any audit tool can agree on
# the schema.  All fields are free-form strings (or list[str] for
# ``characters``).
CHARACTERIZATION_FIELDS: Tuple[str, ...] = (
    "narrative",
    "genre",
    "win_pattern",
    "characters",
    "mechanics",
)


# ---------------------------------------------------------------------------
# Persistence: tag-indexed store of confirmed characterisations
# ---------------------------------------------------------------------------


@dataclass
class CharacterizationEntry:
    """One persisted characterisation — what the LLM said AND which
    tags it applied under.  The tags are what the retrieval layer
    matches; the hypothesis is what gets injected into new-level
    prompts and claim emissions.
    """

    tags:         Dict[str, Any]
    hypothesis:   Dict[str, Any]
    confirmed_at: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "tags":         dict(self.tags),
            "hypothesis":   dict(self.hypothesis),
            "confirmed_at": dict(self.confirmed_at),
        }

    @classmethod
    def from_json(cls, obj: Dict[str, Any]) -> "CharacterizationEntry":
        return cls(
            tags         = dict(obj.get("tags",         {}) or {}),
            hypothesis   = dict(obj.get("hypothesis",   {}) or {}),
            confirmed_at = dict(obj.get("confirmed_at", {}) or {}),
        )


class CharacterizationStore:
    """Tag-indexed JSON store of confirmed game characterisations.

    Entries are matched by **tag overlap** rather than exact keys, so
    a new level's ``{game_id=ls20, level_id=2}`` query can surface
    everything tagged ``game_id=ls20`` from prior levels plus anything
    tagged with the same ``genre`` from other games.

    The persisted file is a plain JSON list — human-readable, diffable,
    trivially mergeable by hand if two runs are stitched together.

    Scoring policy
    --------------
    For each entry, overlap score = count of ``(key, value)`` matches
    between query tags and entry tags.  List-valued tags
    (``characters: [...]``) contribute one point per shared element.
    Entries with score 0 are dropped; the rest are returned sorted by
    descending score, capped at ``limit``.
    """

    def __init__(self, path: Path) -> None:
        self.path: Path = Path(path)
        self._entries: List[CharacterizationEntry] = []
        self._loaded: bool = False

    # ---- file I/O ----------------------------------------------------

    def load(self) -> None:
        """Idempotent load.  Safe to call multiple times."""
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _LOG.warning("characterization store %s unreadable: %r", self.path, exc)
            return
        if not isinstance(raw, list):
            _LOG.warning("characterization store %s: top-level must be a list", self.path)
            return
        self._entries = [
            CharacterizationEntry.from_json(obj)
            for obj in raw
            if isinstance(obj, dict)
        ]

    def save(self) -> None:
        """Write the current entry list atomically."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(
            json.dumps([e.to_json() for e in self._entries], indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    # ---- query / persist --------------------------------------------

    def query(
        self,
        tags:  Dict[str, Any],
        *,
        limit: int = 3,
    ) -> List[Tuple[int, CharacterizationEntry]]:
        """Return entries ranked by tag overlap with ``tags``.

        Each returned tuple is ``(score, entry)``.  Score 0 entries
        are omitted.  Ties break by insertion order (older wins),
        so early-confirmed priors surface first.
        """
        self.load()
        scored: List[Tuple[int, CharacterizationEntry]] = []
        for entry in self._entries:
            score = _tag_overlap(tags, entry.tags)
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda s_e: s_e[0], reverse=True)
        return scored[:limit]

    def persist(
        self,
        tags:         Dict[str, Any],
        hypothesis:   Dict[str, Any],
        *,
        confirmed_at: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a confirmed entry and save.

        Duplicates (exact same ``tags`` AND same ``hypothesis``) are
        suppressed — replaying the same level doesn't bloat the file.
        """
        self.load()
        existing_keys = {(_tag_key(e.tags), _tag_key(e.hypothesis)) for e in self._entries}
        key = (_tag_key(tags), _tag_key(hypothesis))
        if key in existing_keys:
            return
        self._entries.append(CharacterizationEntry(
            tags         = dict(tags),
            hypothesis   = dict(hypothesis),
            confirmed_at = dict(confirmed_at or {}),
        ))
        self.save()

    # ---- introspection ----------------------------------------------

    def __len__(self) -> int:
        self.load()
        return len(self._entries)


def _tag_overlap(query: Dict[str, Any], entry: Dict[str, Any]) -> int:
    """Count matching ``(key, value)`` pairs.

    List-valued tags contribute one point per shared element.
    Scalar values contribute one point on exact match.
    """
    score = 0
    for key, qval in query.items():
        eval_ = entry.get(key)
        if eval_ is None:
            continue
        if isinstance(qval, list) and isinstance(eval_, list):
            score += len(set(map(_norm, qval)) & set(map(_norm, eval_)))
        elif isinstance(qval, list):
            # query is a list, entry is scalar: hit if entry in query list
            if _norm(eval_) in set(map(_norm, qval)):
                score += 1
        elif isinstance(eval_, list):
            if _norm(qval) in set(map(_norm, eval_)):
                score += 1
        else:
            if _norm(qval) == _norm(eval_):
                score += 1
    return score


def _norm(v: Any) -> Any:
    """Normalise values for comparison: lowercase strings, pass-through others."""
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _tag_key(d: Dict[str, Any]) -> str:
    """Stable string key for a dict (for duplicate detection)."""
    return json.dumps(d, sort_keys=True, default=str)


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------


class GameCharacterizationTrigger(OracleTrigger):
    """Adapter-side trigger: ask the LLM to characterise each new level.

    Detector
    --------
    Fires exactly once per level (keyed by :func:`_level_key`, same as
    :class:`InitialFrameScanTrigger`).  On the first step of a level
    we have not yet characterised, dispatches one ``CHARACTERIZE_GAME``
    Observer query.

    Handler
    -------
    Parses the Observer answer's ``result`` as an object with
    ``narrative``, ``genre``, ``win_pattern``, ``characters``, and
    ``mechanics``.  Each non-empty field becomes a
    :class:`PropertyClaim` on the synthetic ``_game`` entity, scoped
    to :class:`ScopeKind.LEVEL` — characterisation is per-level; the
    same game at level 2 may play differently than at level 1.

    Persistence
    -----------
    When the trigger observes a level transition (``levels_completed``
    increment), the characterisation that was active during the
    just-completed level is persisted to the
    :class:`CharacterizationStore` with tags derived from
    ``ws.agent`` (``game_id``, ``level_id``) plus the hypothesis's own
    ``genre`` and ``characters``.  Future levels' trigger calls query
    the store first and include top matches in the prompt context.
    """

    name = "game_characterization"

    def __init__(
        self,
        *,
        store:   Optional[CharacterizationStore] = None,
        urgency: float = 0.5,
        top_k:   int = 3,
    ) -> None:
        self.store    = store
        self.urgency  = urgency
        self.top_k    = top_k

        # Per-episode state (reset on episode boundaries)
        self._characterised_levels: Set[Any] = set()
        # Map level_key -> hypothesis dict, so we can persist on
        # level transition.
        self._active_hypothesis: Dict[Any, Dict[str, Any]] = {}
        # Track which level key was last seen, so a transition can be
        # detected even when the detector skips firing (e.g. already
        # characterised).
        self._last_level_key: Optional[Any] = None
        # Levels completed counter we last saw — used to decide
        # whether to persist on transition.
        self._last_levels_completed: int = 0
        self._counter = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._characterised_levels.clear()
        self._active_hypothesis.clear()
        self._last_level_key = None
        self._last_levels_completed = 0
        self._counter = 0

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _should_fire(self, ws: WorldState) -> bool:
        key = _level_key(ws)
        if key in self._characterised_levels:
            return False
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
        # First: check for a level transition and persist the prior
        # characterisation if appropriate.  Do this before firing the
        # new-level query so the completed level's hypothesis is
        # captured even when characterise + transition happen on
        # back-to-back steps.
        self._maybe_persist_on_transition(ws, step)

        if not self._should_fire(ws):
            return

        level_key = _level_key(ws)
        obs = ws.observation_history[-1]
        raw_frame = getattr(obs, "raw_frame", None)

        if not raw_frame:
            # No frame to reason about; mark characterised anyway so
            # we don't re-attempt on every empty frame.
            self._characterised_levels.add(level_key)
            return

        priors = self._gather_priors(ws) if self.store is not None else []
        context = self._build_context(ws, priors)

        self._counter += 1
        query = ObserverQuery(
            query_id = f"characterize_game::level{level_key}::n{self._counter}",
            question = QuestionType.CHARACTERIZE_GAME,
            targets  = [],
            frames   = [raw_frame],
            urgency  = self.urgency,
            context  = context,
        )

        try:
            answer = adapter.observer_query(query)
        except Exception as exc:  # pragma: no cover — defensive
            _LOG.warning("characterize_game observer_query raised %r", exc)
            return

        # Mark characterised regardless of confidence — a zero-confidence
        # answer is a valid "I don't know", not a reason to retry the
        # identical frame.
        self._characterised_levels.add(level_key)

        if answer is None or answer.confidence <= 0.0:
            return

        hypothesis = self._integrate(ws, answer, step, level_key=level_key)
        if hypothesis:
            self._active_hypothesis[level_key] = hypothesis

    # ------------------------------------------------------------------
    # Prior retrieval and prompt context
    # ------------------------------------------------------------------

    def _gather_priors(self, ws: WorldState) -> List[Dict[str, Any]]:
        """Ask the store for entries matching tags from ws.agent."""
        assert self.store is not None
        tags = _query_tags_from_ws(ws)
        hits = self.store.query(tags, limit=self.top_k)
        return [entry.hypothesis for _score, entry in hits]

    @staticmethod
    def _build_context(ws: WorldState, priors: List[Dict[str, Any]]) -> str:
        base = (
            "Characterise this new level at a conceptual level — "
            "what kind of game/task is this, what is the likely win "
            "condition, what roles/characters are present, how does "
            "the player act."
        )
        if not priors:
            return base
        lines = [base, "", "PREVIOUSLY-CONFIRMED CHARACTERISATIONS THAT MAY APPLY:"]
        for i, p in enumerate(priors, start=1):
            lines.append(
                f"  [{i}] genre={p.get('genre','?')!r} "
                f"win_pattern={p.get('win_pattern','?')!r} "
                f"characters={p.get('characters', [])} "
                f"narrative={p.get('narrative','')!r}"
            )
        lines.append(
            "If one of the above describes this level, you may re-use "
            "its fields verbatim.  If none fit, propose a new "
            "characterisation."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Claim emission
    # ------------------------------------------------------------------

    def _integrate(
        self,
        ws:        WorldState,
        answer:    Any,
        step:      int,
        *,
        level_key: Any,
    ) -> Dict[str, Any]:
        """Install the characterisation as PropertyClaims on ``_game``.

        Returns the hypothesis dict (the same fields written as claims)
        so the caller can cache it for later persistence.  Empty fields
        are skipped — a partial characterisation is better than no
        characterisation, and absent claims simply don't exist rather
        than asserting an empty value.
        """
        result = getattr(answer, "result", None)
        if not isinstance(result, dict):
            return {}

        rationale = (
            f"GameCharacterizationTrigger @ step {step}; "
            f"level={level_key}; confidence={answer.confidence:.2f}"
        )

        hypothesis: Dict[str, Any] = {}
        for field_name in CHARACTERIZATION_FIELDS:
            value = result.get(field_name)
            if isinstance(value, list):
                # Characters list — persist as a list, one claim total.
                cleaned = [str(v).strip().lower() for v in value
                           if isinstance(v, (str, int, float)) and str(v).strip()]
                if not cleaned:
                    continue
                hypothesis[field_name] = cleaned
                _store.propose(
                    ws,
                    claim     = PropertyClaim(
                        entity_id = GAME_ENTITY_ID,
                        property  = field_name,
                        value     = tuple(cleaned),
                    ),
                    source    = "llm_proposer",
                    scope     = Scope(kind=ScopeKind.LEVEL),
                    step      = step,
                    rationale = rationale,
                )
            elif isinstance(value, str):
                cleaned = value.strip()
                if not cleaned:
                    continue
                hypothesis[field_name] = cleaned
                _store.propose(
                    ws,
                    claim     = PropertyClaim(
                        entity_id = GAME_ENTITY_ID,
                        property  = field_name,
                        value     = cleaned,
                    ),
                    source    = "llm_proposer",
                    scope     = Scope(kind=ScopeKind.LEVEL),
                    step      = step,
                    rationale = rationale,
                )
        return hypothesis

    # ------------------------------------------------------------------
    # Persistence on level transition
    # ------------------------------------------------------------------

    def _maybe_persist_on_transition(self, ws: WorldState, step: int) -> None:
        """If ``levels_completed`` incremented since the last tick, the
        level we were on is confirmed — persist its hypothesis."""
        current_key    = _level_key(ws)
        current_levels = int(ws.agent.get("levels_completed", 0) or 0)

        if self._last_level_key is None:
            self._last_level_key = current_key
            self._last_levels_completed = current_levels
            return

        transitioned = (current_levels > self._last_levels_completed)
        if transitioned and self.store is not None:
            prev_key = self._last_level_key
            hypothesis = self._active_hypothesis.get(prev_key)
            if hypothesis:
                tags = _persist_tags_from_ws_plus_hypothesis(
                    ws, hypothesis, previous_levels_completed=self._last_levels_completed,
                )
                self.store.persist(
                    tags         = tags,
                    hypothesis   = hypothesis,
                    confirmed_at = {
                        "step":             step,
                        "levels_completed": current_levels,
                    },
                )

        self._last_level_key = current_key
        self._last_levels_completed = current_levels


# ---------------------------------------------------------------------------
# Tag helpers — what tags do we query under and persist under?
# ---------------------------------------------------------------------------


def _query_tags_from_ws(ws: WorldState) -> Dict[str, Any]:
    """Tags used to look up priors at level start.

    At level start we know the game id and the level we are about to
    play (the incoming level).  We don't yet know ``genre`` or
    ``characters`` (that's what we're asking the LLM about), so the
    prior match is necessarily coarse — driven by game_id and
    level_id.  That's fine: retrieval is one input among several, and
    the LLM makes the final call.
    """
    agent = ws.agent
    tags: Dict[str, Any] = {}
    game_id = (
        agent.get("game_id")
        or agent.get("env_id")
        or agent.get("_env_id")
    )
    if game_id:
        tags["game_id"] = str(game_id)
    level_id = agent.get("level_idx")
    if level_id is None:
        level_id = agent.get("levels_completed")
    if level_id is not None:
        tags["level_id"] = int(level_id)
    return tags


def _persist_tags_from_ws_plus_hypothesis(
    ws:                      WorldState,
    hypothesis:              Dict[str, Any],
    *,
    previous_levels_completed: int,
) -> Dict[str, Any]:
    """Tags used to persist a confirmed characterisation.

    Combines the situational tags from ``_query_tags_from_ws`` (with
    ``level_id`` pinned to the level that *was just completed*, not
    the one we are entering) and the hypothesis's own
    vocabulary-free tags (``genre``, ``win_pattern``, ``characters``).
    This lets later retrieval match by situation OR by semantic
    similarity.
    """
    agent = ws.agent
    tags: Dict[str, Any] = {}
    game_id = (
        agent.get("game_id")
        or agent.get("env_id")
        or agent.get("_env_id")
    )
    if game_id:
        tags["game_id"] = str(game_id)
    # The just-completed level is the prior levels_completed value.
    tags["level_id"] = int(previous_levels_completed)
    for field_name in ("genre", "win_pattern"):
        val = hypothesis.get(field_name)
        if isinstance(val, str) and val:
            tags[field_name] = val
    characters = hypothesis.get("characters")
    if isinstance(characters, list) and characters:
        tags["characters"] = list(characters)
    return tags
