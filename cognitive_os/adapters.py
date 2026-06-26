"""Adapter protocol — the engine/domain boundary.

The cognitive engine is domain-agnostic.  To run against any specific
domain — an ARC-AGI-3 game, an AI2-THOR scene, a real robot — that
domain supplies an :class:`Adapter`.  The adapter is responsible for:

1. **Environment lifecycle**: reset, observe, execute, is_done.
2. **Action translation**: turning an engine :class:`Action` (a name
   + parameters) into a domain-native invocation.
3. **Perceptual grounding**: producing symbolic :class:`Observation`\\s
   (typed events + entity snapshots) from whatever raw data the
   domain provides — pixels, video, sensor streams.
4. **Tool provision**: registering domain primitives (grid BFS,
   motion planning, etc.) and dispatching invocations.
5. **Oracle delegation**: forwarding :class:`ObserverQuery` /
   :class:`MediatorQuery` to whichever implementation the adapter
   chose (VLM, text LLM, classical pipeline, human-in-the-loop).

The engine itself imports this module to **consume** the protocol —
i.e. to hold a reference to "some adapter" whose methods it calls on
its main loop.  Adapters themselves live under ``usecases/<domain>/``
and import the engine, never the other way around.  This import
direction is enforced by standing invariant #1.

Design notes
------------
* Adapter is an :class:`abc.ABC`, not a :class:`typing.Protocol`.  We
  want subclass discovery and a clear "implement these N methods to
  build an adapter" surface.  Protocols are structurally-typed,
  which is fine but less discoverable for future contributors.

* **Stateful vs stateless**: the adapter is inherently stateful (it
  holds the environment / SDK handle).  The engine treats that
  state as opaque.  The adapter MUST NOT mutate :class:`WorldState`
  directly; it returns observations, and the engine integrates them.

* **Thread safety**: not required.  The engine calls the adapter
  serially.  Async tool invocations are signalled via the tool
  callback pattern in :class:`ToolInvocation`, not via threads.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .types import (
    Action,
    MediatorAnswer,
    MediatorQuery,
    Observation,
    ObserverAnswer,
    ObserverQuery,
    ToolInvocation,
    ToolResult,
    WorldState,
)


class Adapter(ABC):
    """Abstract base class for all domain adapters.

    Subclasses MUST implement:

    * :meth:`initialize` — populate ``ws.tool_registry``, seed the
      primary :class:`Goal`, run a full Observer scan of the initial
      frame if appropriate.
    * :meth:`reset` — reset the environment and return the initial
      :class:`Observation`.
    * :meth:`observe` — produce an :class:`Observation` for the
      current step without executing anything.  Called after
      :meth:`execute` to see the result.
    * :meth:`execute` — execute the engine-supplied :class:`Action`
      in the domain, returning when the action is complete (for
      synchronous domains) or when the command has been dispatched
      (for asynchronous domains — in which case the subsequent
      :meth:`observe` will catch up).
    * :meth:`action_space` — list of :class:`Action`\\s available in
      the current state.  May change between steps for domains with
      context-dependent action sets.
    * :meth:`is_done` — boolean: episode terminated?

    Subclasses SHOULD implement (default: neutral no-op):

    * :meth:`observer_query` — visual oracle call.
    * :meth:`mediator_query` — common-sense oracle call.
    * :meth:`invoke_tool` — synchronous tool dispatch.

    The default implementations of the three optional methods return
    an "unsupported" response so engines that call them without an
    implementing adapter get a clean failure signal rather than a
    silent wrong answer.
    """

    #: Stable identifier for this adapter instance — distinct episodes
    #: of the same game / task can share an env_id for knowledge
    #: accumulation purposes.  Set by the subclass on construction.
    env_id: str = "unknown"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def initialize(self, ws: WorldState) -> None:
        """One-time setup called once per adapter instance before the
        first episode.

        Typical responsibilities:

        * Populate ``ws.tool_registry`` with domain primitives.
        * Seed the primary :class:`Goal` via
          :func:`goal_forest.add_goal`.
        * Optionally run ``observer_query`` on the initial frame to
          seed :class:`RelationalClaim`\\s (identical / similar /
          distinct groupings).
        * Set ``self.env_id`` to a stable identifier.
        """

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment to its start state and return the
        first :class:`Observation`.  Called at the top of every
        episode.
        """

    @abstractmethod
    def observe(self) -> Observation:
        """Return the current :class:`Observation` without advancing
        the environment.  Used by the runner after each
        :meth:`execute` call to integrate the outcome into the
        WorldState.
        """

    @abstractmethod
    def execute(self, action: Action) -> None:
        """Execute ``action`` in the domain.  May block for
        synchronous environments or dispatch and return for async
        ones.  Any exception should be caught by the adapter and
        surfaced as a ``SurpriseEvent`` or ``AgentDied`` event on
        the next :meth:`observe`.
        """

    @abstractmethod
    def action_space(self) -> List[Action]:
        """Return the list of :class:`Action`\\s available in the
        current state.  May vary between steps.

        The engine filters this list through
        :func:`planner.apply_rules_filter` before BFS so rule-
        forbidden actions are never considered.
        """

    @abstractmethod
    def is_done(self) -> bool:
        """True when the episode has terminated (success, failure,
        timeout, or explicit termination).
        """

    # ------------------------------------------------------------------
    # Oracle delegation (default: unsupported)
    # ------------------------------------------------------------------

    def observer_query(self, query: ObserverQuery) -> ObserverAnswer:
        """Default: unsupported.  Adapters supporting a visual oracle
        should override this method and forward to their VLM / vision
        pipeline.

        The engine guards against unsupported oracles by checking the
        returned ``confidence``; a value of 0 signals "no answer",
        and the engine must proceed without the visual verdict.
        """
        return ObserverAnswer(
            query_id    = query.query_id,
            result      = None,
            confidence  = 0.0,
            explanation = "adapter does not support Observer queries",
        )

    def mediator_query(self, query: MediatorQuery) -> MediatorAnswer:
        """Default: unsupported.  Adapters supporting a common-sense
        oracle should override this method and forward to their
        text LLM.  Like :meth:`observer_query`, a zero ``confidence``
        signals "no answer".
        """
        return MediatorAnswer(
            query_id    = query.query_id,
            confidence  = 0.0,
            explanation = "adapter does not support Mediator queries",
        )

    # ------------------------------------------------------------------
    # Tool dispatch (default: unsupported)
    # ------------------------------------------------------------------

    def invoke_tool(self, invocation: ToolInvocation) -> ToolResult:
        """Default: unsupported.  Adapters registering tools in
        ``initialize`` should override this to dispatch by
        ``invocation.tool_name`` and return the result.

        Async tools return a :class:`ToolResult` with
        ``success=False, error="pending"`` and rely on the callback
        pattern (or polling of ``ws.pending_tool_calls``) to deliver
        the actual result later.
        """
        return ToolResult(
            invocation_id = invocation.invocation_id,
            success       = False,
            error         = "adapter does not support invoke_tool",
        )

    # ------------------------------------------------------------------
    # Optional hooks
    # ------------------------------------------------------------------

    def on_episode_start(self, ws: WorldState) -> None:
        """Called at the start of each episode (after :meth:`reset`).
        Default no-op; override to inject per-episode preparation
        (e.g. load level-specific :class:`CachedSolution`\\s for this
        specific level).
        """

    def on_episode_end(self, ws: WorldState) -> None:
        """Called at the end of each episode (after PostMortem).
        Default no-op; override for cleanup or persistence steps
        the adapter needs to run (e.g. close SDK handles, flush
        logs).
        """
