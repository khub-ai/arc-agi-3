"""Tool system — adapter-provided domain primitives the engine can invoke.

The cognitive engine is code-centric and domain-agnostic.  Many tasks
within an episode require domain-specific computation — BFS on a 2-D
grid, motion planning in a 3-D workspace, connected-component labelling,
symmetry detection, pattern matching — that are deterministic and well
suited to hand-written code.  Doing these through an LLM is slow,
expensive, and error-prone; doing them in the engine directly would
contaminate the engine with domain knowledge.

The solution is a **Tool registry**: the adapter declares a set of typed
callable primitives it provides, and the engine invokes them through a
generic request/response protocol.  Miners, the Planner, the Explorer,
and the Mediator can all issue ToolInvocations; the adapter executes them
and returns ToolResults.

Two invocation modes are supported:

* **Synchronous** — ``ToolSignature.is_async == False``.  The adapter
  returns the result immediately.  Used for fast deterministic tools
  (grid BFS on a 100×100 grid, pattern match, diff).

* **Asynchronous** — ``ToolSignature.is_async == True``.  The engine
  issues the invocation and continues; the adapter posts the result
  later via callback or a pull-style pending-result queue.  Used for
  slow tools (e.g. a motion planner that may take seconds).

Tool latency is exposed on the signature so the Planner can budget
real-time costs when selecting among alternatives.  Determinism is
exposed so the engine may memoise results and avoid redundant calls.

This module defines only the data types for the registry and protocol.
Invocation dispatch lives in the adapter boundary
(``cognitive_os.adapters`` in a later phase); memoisation and budget
tracking live in the runtime (``cognitive_os.episode_runner``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Signatures and registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSignature:
    """Description of a tool the adapter makes available.

    The signature is the engine's entire view of a tool: it does NOT
    contain the implementation.  That lives in the adapter.  The engine
    invokes tools by name, passing arguments that match
    ``input_schema``; the adapter is responsible for dispatch and for
    returning a ``ToolResult`` whose ``result`` field matches
    ``output_schema``.

    Parameters
    ----------
    name
        Unique identifier used in ToolInvocation.
    description
        Human- and Mediator-readable description.  Shown to the Mediator
        in WorldStateSummary so it knows what tools already exist before
        proposing new ones.
    input_schema
        Tuple of ``(param_name, type_hint)`` pairs.  Type hints are
        strings for now (e.g. ``"int"``, ``"tuple[int,int]"``,
        ``"Grid"``) rather than Python types, to keep the registry
        trivially serialisable for Mediator consumption.
    output_schema
        Type description of the result, same format as parameter type
        hints.
    cost
        Unitless cost used by the Planner to compare alternatives.
        Calibrated so that a "typical inexpensive primitive" is 1.0.
    typical_latency_ms
        Expected wall-clock latency for a call.  The Planner uses this
        to budget real-time expenditure — important in robotics where
        latency matters.  Set realistically from profiling.
    determinism
        True if repeated calls with identical arguments always return
        identical results.  Deterministic tools may be memoised by the
        runtime; non-deterministic tools (e.g. a sampled motion planner
        or a stochastic perception pipeline) must not be.
    side_effects
        True if calling the tool mutates world state.  Most adapter
        primitives are pure queries (side_effects=False); some
        robotics tools (e.g. "actuate gripper") are actions disguised
        as tool calls and do have side effects.  The engine treats
        side-effectful tool calls the same way it treats normal
        actions (expected_effects generate TransitionClaims).
    is_async
        If True, the tool's invocation returns a pending handle; the
        result arrives later via callback or a pull-style queue.  If
        False, the invocation is synchronous and the result is
        available immediately.
    """

    name:                str
    description:         str
    input_schema:        Tuple[Tuple[str, str], ...] = ()
    output_schema:       str = "Any"
    cost:                float = 1.0
    typical_latency_ms:  float = 1.0
    determinism:         bool  = True
    side_effects:        bool  = False
    is_async:            bool  = False


@dataclass
class ToolRegistry:
    """The set of tools an adapter exposes.

    Populated by the adapter at initialisation; read by the engine for
    planning, miner configuration, and Mediator summaries.  New tools
    may be added mid-episode (e.g. when an Option is synthesised and
    promoted, or when a Mediator-proposed tool is adopted) — the
    registry is mutable.
    """

    tools: Dict[str, ToolSignature] = field(default_factory=dict)

    def register(self, sig: ToolSignature) -> None:
        """Add a tool signature.  Raises ValueError on name collision."""
        if sig.name in self.tools:
            raise ValueError(f"tool already registered: {sig.name}")
        self.tools[sig.name] = sig

    def has(self, name: str) -> bool:
        return name in self.tools

    def get(self, name: str) -> Optional[ToolSignature]:
        return self.tools.get(name)

    def list_available(self) -> List[ToolSignature]:
        return list(self.tools.values())

    def names(self) -> List[str]:
        return sorted(self.tools.keys())


# ---------------------------------------------------------------------------
# Invocation and result
# ---------------------------------------------------------------------------


# Callback type: a function taking a ToolResult and returning None.
# Kept as a free type alias rather than a Protocol so that any callable
# can be passed without structural-subtyping ceremony.
ToolCallback = Callable[["ToolResult"], None]


@dataclass
class ToolInvocation:
    """A request to invoke a tool.

    ``invocation_id`` must be unique within an episode.  ``requester``
    identifies the subsystem making the call (e.g. ``"miner:Symmetry"``,
    ``"planner"``, ``"mediator:q_42"``) for audit and budget attribution.

    For synchronous tools, ``callback`` should be None and the runtime
    returns the ToolResult directly.  For asynchronous tools, the
    caller may supply a callback to be invoked when the result is
    ready; alternatively the runtime maintains a pull-style pending
    queue keyed by ``invocation_id``.
    """

    invocation_id: str
    tool_name:     str
    arguments:     Dict[str, Any]
    requester:     str
    requested_at:  int                              # step number
    callback:      Optional[ToolCallback] = None
    timeout_ms:    Optional[float]        = None
    urgency:       float                  = 0.5


@dataclass
class ToolResult:
    """The outcome of a ToolInvocation.

    ``success == False`` means the tool failed or was unavailable; the
    ``error`` string carries an adapter-provided reason.  ``cost_consumed``
    and ``latency_ms`` are measured values (as opposed to the
    ``ToolSignature`` defaults, which are estimates); they feed the
    ResourceTracker and may be used to recalibrate signature estimates
    over time.
    """

    invocation_id: str
    success:       bool
    result:        Any = None
    error:         Optional[str] = None
    cost_consumed: float = 0.0
    latency_ms:    float = 0.0
    completed_at:  int = -1     # step number at which the result became available


# ---------------------------------------------------------------------------
# Tool proposals (Mediator-assisted tool creation)
# ---------------------------------------------------------------------------


@dataclass
class ToolProposal:
    """A proposed new tool, typically returned by the Mediator when
    asked ``MediatorQuestion.PROPOSE_TOOL``.

    The adapter decides whether to implement and register the proposal.
    Implementations may be hand-written, code-synthesised, or
    delegated to a human operator.  A proposed tool is NOT admitted
    to the ``ToolRegistry`` until it has passed the adapter's
    adoption tests (e.g. regression on past observations, type-
    checking, sandbox safety evaluation).

    ``safety_notes`` lets the Mediator communicate constraints the
    adapter must respect (e.g. ``"read-only; must not mutate grid"``,
    ``"do not call with len(region) > 10_000"``).  These notes are
    advisory input to the adapter's adoption gate.
    """

    signature:           ToolSignature
    implementation_hint: str                          # algorithm description or pseudocode
    expected_use_case:   str                          # when this tool would help
    safety_notes:        Tuple[str, ...] = ()
    rationale:           str = ""
