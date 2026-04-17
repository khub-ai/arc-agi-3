"""ArcAdapter — the ARC-AGI-3 implementation of
:class:`cognitive_os.Adapter`.

This class is the single point of contact between the engine and the
ARC-AGI-3 SDK.  It is deliberately thin: all reasoning (hypothesis
formation, planning, exploration, post-mortem analysis) lives in the
engine.  The adapter's responsibilities are:

1. Environment lifecycle — ``reset`` / ``step`` against the SDK.
2. Perception — turn the raw frame into an :class:`Observation`.
3. Action translation — engine :class:`Action` ↔ arc_agi native.
4. Tool provision — register the grid primitives and dispatch
   :class:`ToolInvocation`\\s.
5. Oracle delegation — ``observer_query`` / ``mediator_query``
   forward to a pluggable LLM backend (Phase 5b; Phase 5a defaults
   to "unsupported").

The class supports two construction modes:

* **Live**: ``ArcAdapter(raw_env=env, env_id="ls20")`` — the SDK env
  is injected.  Used by the harness.
* **Replay**: ``ArcAdapter.from_replay(frames, actions, env_id=...)``
  — wraps a recorded frame sequence as a fake env implementing
  ``reset`` and ``step``.  Used by tests so CI never hits the live
  API.

Both modes present the same :class:`Adapter` surface; the engine
cannot tell them apart.  That is the whole point of the seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from cognitive_os import (
    Action,
    Adapter,
    Goal,
    GoalNode,
    MediatorAnswer,
    MediatorQuery,
    NodeType,
    Observation,
    ObserverAnswer,
    ObserverQuery,
    ResourceAbove,
    ToolInvocation,
    ToolResult,
    WorldState,
    add_goal,
)

from .action_mapping import (
    engine_action_for,
    engine_action_space,
    native_action_for,
)
from .backends import LLMBackend, NullBackend
from .perception import PerceptionState, build_observation
from .tools.registry import build_registry, dispatch


# The conventional top-level goal id perception uses when emitting
# ``GoalConditionMet`` on a ``WIN`` state.  Keeping this synchronised
# with :mod:`perception` is important; an off-by-one name would
# cause the engine's goal to never fire.
_EPISODE_GOAL_ID = "episode"


def _to_list_2d(frame: Any) -> Any:
    """Normalise a raw game frame into a plain ``list[list[int]]``.

    The ARC-AGI-3 SDK returns frames in multiple shapes depending on
    version and game: ``list[list[int]]``, 2-D ``numpy.ndarray``,
    ``list[numpy.ndarray]`` (one element per viewport), and
    occasionally deeper nesting.  This helper canonicalises all of
    them so the engine's perception layer only ever sees a 2-D list
    of ints.

    Ported from the proven parity logic in the predecessor adapter
    (khub-knowledge-fabric .private/usecases/arc-agi-3/agents.py
    ``_to_list_2d``).  Kept in the adapter (rather than perception)
    because it is an SDK-shape concern, not a pixel-semantics one.
    """
    # Numpy array (or anything ndarray-like): .tolist() handles any
    # dimensionality in a single shot.
    if hasattr(frame, "tolist"):
        result = frame.tolist()
        if isinstance(result, list) and result and isinstance(result[0], list):
            return result
        return result if isinstance(result, list) else []

    if not isinstance(frame, list) or not frame:
        return [] if frame in (None, []) else frame

    first = frame[0]

    # Outer wrapper with a numpy array inside (the live-SDK shape).
    if hasattr(first, "tolist"):
        inner = first.tolist()
        if isinstance(inner, list) and inner and isinstance(inner[0], list):
            return inner
        return inner if isinstance(inner, list) else []

    # Already flat list-of-lists (rows of ints): pass through.
    if isinstance(first, list) and first and not isinstance(first[0], list):
        return frame

    # Deeper nesting (3-D list-of-list-of-list): recurse one level.
    inner = _to_list_2d(first)
    return inner if inner else frame


def _normalise_state_name(raw_state: Any) -> str:
    """Coerce the SDK's ``state`` field to a plain string.

    The live ARC-AGI-3 SDK returns a ``GameState`` StrEnum; equality
    against ``"WIN"`` etc. already works (StrEnum hashes identically
    to its string value), but ``str(GameState.WIN)`` surprisingly
    yields ``"GameState.WIN"`` rather than ``"WIN"``.  Callers that
    pass the state name through ``str()`` (logs, JSON dumps of
    ``agent_state``) would see the ugly form.  Flatten here so every
    downstream consumer sees the plain value.

    Uses ``.name`` (per the predecessor adapter's proven parity
    logic) — works uniformly for every ``Enum`` subclass, not only
    ``StrEnum`` where ``.value`` happens to be a string.
    """
    if raw_state is None:
        return "UNKNOWN"
    name = getattr(raw_state, "name", None)
    if isinstance(name, str):
        return name
    return str(raw_state)


@dataclass
class _ReplayEnv:
    """Fake env that feeds a pre-recorded frame sequence.

    Replay fixtures pair each step's frame with the metadata the
    real SDK provides (``state``, ``levels_completed``,
    ``available_actions``).  Steps past the recorded length are
    treated as ``GAME_OVER`` so the episode runner terminates
    cleanly rather than hanging.
    """

    frames:              List[Any]
    states:              List[str]
    levels:              List[int]
    available_actions:   List[List[Any]]
    _index:              int = 0

    @property
    def action_space(self) -> List[Any]:
        return self.available_actions[min(self._index, len(self.available_actions) - 1)]

    def reset(self) -> Any:
        self._index = 0
        return self._frame_obj()

    def step(self, _action: Any) -> Any:
        self._index += 1
        return self._frame_obj()

    def _frame_obj(self) -> Any:
        if self._index < len(self.frames):
            return _FrameObj(
                frame             = self.frames[self._index],
                state             = self.states[self._index],
                levels_completed  = self.levels[self._index],
                available_actions = self.available_actions[self._index],
            )
        # Past the end — synthesise a GAME_OVER frame so the runner
        # exits normally rather than asserting.
        last_frame = self.frames[-1] if self.frames else [[0]]
        return _FrameObj(
            frame             = last_frame,
            state             = "GAME_OVER",
            levels_completed  = self.levels[-1] if self.levels else 0,
            available_actions = [],
        )


@dataclass
class _FrameObj:
    """Shape matches the real arc_agi SDK frame just well enough for
    perception and action-space extraction.  Only the attributes
    the adapter reads from are present."""
    frame:              Any
    state:              str
    levels_completed:   int
    available_actions:  List[Any]


class ArcAdapter(Adapter):
    """The ARC-AGI-3 domain adapter.

    Parameters
    ----------
    raw_env
        An object providing ``reset()``, ``step(action)``, and
        ``action_space``.  Passing the SDK env gives live behaviour;
        passing a :class:`_ReplayEnv` gives recorded-fixture behaviour.
    env_id
        Stable identifier for knowledge-accumulation scoping.
    background_colour
        Palette value treated as background during component
        extraction.  Defaults to ``0`` (the ARC-AGI-3 canvas colour)
        but is exposed in case a given game re-colours the canvas.
    """

    def __init__(
        self,
        raw_env:           Any,
        env_id:            str,
        *,
        background_colour: int = 0,
        backend:           Optional[LLMBackend] = None,
    ) -> None:
        self.env_id        = env_id
        self._env          = raw_env
        self._bg           = background_colour
        self._perception   = PerceptionState()
        self._handlers:    Dict[str, Any] = {}
        self._last_frame:  Optional[_FrameObj] = None
        self._current_step = 0
        # Backend defaults to NullBackend: observer / mediator queries
        # return zero-confidence, matching the engine's "no oracle
        # configured" convention.  Tests and Phase-5a paths use this.
        self.backend: LLMBackend = backend if backend is not None else NullBackend()

    # ------------------------------------------------------------------
    # Alternate constructor for tests
    # ------------------------------------------------------------------

    @classmethod
    def from_replay(
        cls,
        frames:            Sequence[Any],
        states:            Sequence[str],
        levels_completed:  Sequence[int],
        available_actions: Sequence[Sequence[Any]],
        *,
        env_id:            str = "replay",
        background_colour: int = 0,
        backend:           Optional[LLMBackend] = None,
    ) -> "ArcAdapter":
        if not (len(frames) == len(states) == len(levels_completed) == len(available_actions)):
            raise ValueError("replay arrays must be same length")
        env = _ReplayEnv(
            frames            = list(frames),
            states            = list(states),
            levels            = list(levels_completed),
            available_actions = [list(a) for a in available_actions],
        )
        return cls(
            raw_env           = env,
            env_id            = env_id,
            background_colour = background_colour,
            backend           = backend,
        )

    # ------------------------------------------------------------------
    # Adapter ABC — lifecycle
    # ------------------------------------------------------------------

    def initialize(self, ws: WorldState) -> None:
        """Populate the tool registry and seed the top-level goal.

        The seed goal is deliberately minimal — just "reach a state
        where the ``episode`` goal condition is met".  Concrete
        subgoals are formed by the engine's goal-derivation machinery
        as transition claims accumulate.  We do NOT inject
        game-specific subgoals here; doing so would inject exactly
        the game-specific knowledge the extraction was meant to
        eliminate.
        """
        registry, handlers = build_registry()
        ws.tool_registry = registry
        self._handlers = handlers

        # Seed the top-level goal: "WIN the episode".  Perception
        # populates ``agent_state["resources"]["episode_won"]`` with
        # 1.0 on the WIN frame and 0.0 otherwise; the runner copies
        # that sub-dict into ``ws.agent["resources"]`` each step;
        # :class:`ResourceAbove` evaluates the goal's truth from
        # there.  This keeps the adapter from hard-coding a concrete
        # success pose — any WIN state reached through any gameplay
        # fires the goal.
        add_goal(ws, Goal(
            id       = _EPISODE_GOAL_ID,
            root     = GoalNode(
                id         = _EPISODE_GOAL_ID,
                node_type  = NodeType.ATOM,
                condition  = ResourceAbove("episode_won", 0.5),
            ),
            priority = 1.0,
        ))

    def reset(self) -> Observation:
        self._perception.reset_for_new_episode()
        # Per-episode budget and usage counters reset on every reset.
        # The engine's ResourceTracker (when it lands) will enforce
        # the budget at a higher level; the backend enforces it per
        # call regardless.
        self.backend.reset_usage()
        raw = self._env.reset()
        self._last_frame = raw
        return self._observe_from(raw)

    def observe(self) -> Observation:
        # After ``execute`` the adapter holds the post-step frame
        # obtained from ``step()``.  If no step has been taken yet
        # (observe called before reset), fall back to the last frame
        # of a fresh reset.
        if self._last_frame is None:
            return self.reset()
        return self._observe_from(self._last_frame)

    def execute(self, action: Action) -> None:
        raw_space = self._current_action_space()
        try:
            raw_action = native_action_for(action, raw_space)
        except KeyError:
            # Action unavailable — the engine will see no state
            # change on the next observe; the SurpriseMiner /
            # FutilePatternMiner form the appropriate claims.  We
            # deliberately do NOT raise here; raising would make the
            # engine's action space / execute contract brittle.
            return
        self._last_frame = self._env.step(raw_action)

    def action_space(self) -> List[Action]:
        return engine_action_space(self._current_action_space())

    def is_done(self) -> bool:
        if self._last_frame is None:
            return False
        state = getattr(self._last_frame, "state", None)
        return state in {"WIN", "GAME_OVER"}

    # ------------------------------------------------------------------
    # Adapter ABC — oracle delegation
    # ------------------------------------------------------------------

    def observer_query(self, query: ObserverQuery) -> ObserverAnswer:
        """Delegate to :attr:`backend`.  The default :class:`NullBackend`
        returns zero confidence; an :class:`AnthropicBackend` (or any
        other :class:`LLMBackend`) invokes the underlying LLM.  Budget
        exhaustion is surfaced by the backend as a zero-confidence
        answer with a clear explanation; the engine treats that the
        same as "oracle unavailable" and proceeds symbolically.
        """
        return self.backend.answer_observer_query(query)

    def mediator_query(self, query: MediatorQuery) -> MediatorAnswer:
        """Delegate to :attr:`backend`.  See :meth:`observer_query` for
        the backend semantics."""
        return self.backend.answer_mediator_query(query)

    # ------------------------------------------------------------------
    # Adapter ABC — tool dispatch
    # ------------------------------------------------------------------

    def invoke_tool(self, invocation: ToolInvocation) -> ToolResult:
        return dispatch(invocation, self._handlers, current_step=self._current_step)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_action_space(self) -> List[Any]:
        # Prefer the per-frame ``available_actions`` list (the ARC-AGI-3
        # SDK exposes this; it can shrink / grow mid-episode).  Fall
        # back to the env-level ``action_space`` when the frame object
        # does not expose it.
        if self._last_frame is not None:
            per_frame = getattr(self._last_frame, "available_actions", None)
            if per_frame:
                return list(per_frame)
        return list(getattr(self._env, "action_space", []))

    def _observe_from(self, raw: Any) -> Observation:
        frame             = _to_list_2d(getattr(raw, "frame", None))
        state_name       = _normalise_state_name(getattr(raw, "state", "PLAYING"))
        levels_completed = getattr(raw, "levels_completed", 0)
        obs = build_observation(
            frame            = frame or [[0]],
            state_name       = state_name,
            levels_completed = levels_completed,
            state            = self._perception,
            background       = self._bg,
        )
        self._current_step = obs.step
        return obs
