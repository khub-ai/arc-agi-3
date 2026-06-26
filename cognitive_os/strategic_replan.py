"""Strategic replan — record dead-ends and offer them to future planners.

When a resource-constrained planner returns
:class:`~cognitive_os.resource_planner.RouteStatus.NO_VIABLE`, the agent
is in a position from which it cannot reach its goal even with all
known refuels available.  In game-playing terms: the agent took a
sequence of decisions that drove itself into a no-win situation.  The
right response is **not** to keep banging at the dead-end — it's to
remember WHAT the dead-end looked like (level state, position,
budget, refuels, target) and HOW the agent got here (recent decision
chain) so the *next* attempt can avoid it.

This module is the persistence layer for that:

* :class:`StrandedEvent` — structured record of one no-win observation.
* :func:`record_stranded_event` — append to ``kb["strategic_replan"]
  ["stranded_events_by_level"][level_key]``.
* :func:`stranded_events_for_level` — query interface for future
  planners.  Stable read shape so consumers don't reach into the kb
  dict directly.

The module is domain-agnostic.  Levels are identified by an opaque
``level_key`` string; positions and targets are opaque hashables; the
caller decides how to encode "decision history" (a list of dicts; the
recorder is structurally agnostic).

The actual avoidance logic — *what* a planner does with these records —
lives in the consumer.  This module is the recording substrate; future
work plugs in postmortem-style analysis (e.g., "every time we tried
target T from agent A with refuels R, we stranded — avoid that
configuration") as a separate concern.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, List, Optional


@dataclass(frozen=True)
class StrandedEvent:
    """One observation of a NO_VIABLE_PLAN dead-end.

    Captures enough to (a) reproduce the planner's reasoning, and
    (b) enable hindsight credit-assignment by the strategic-replan
    consumer.

    Fields:
      level_key:        Opaque level identifier (e.g. ``"2"`` for the
                        ARC sub-level index).
      turn:             Trial-level turn counter (caller's clock).
      agent_pos:        Where the agent is when the planner gave up.
      target:           What the agent was trying to reach.
      refuel_positions: Refuels the planner considered (list, not set,
                        for stable JSON serialisation).
      current_budget:   Steps remaining at observation time.
      full_budget:      Steps after a (hypothetical) refuel.
      direct_cost:      Distance ``agent_pos → target``, or None when
                        the target is unreachable on the known graph.
      reason:           Human-readable explanation from the planner.
      decision_history: Caller-supplied chain of recent commitments
                        (goal picks, branch choices, sub-plan steps).
                        Free-form list of dicts; the consumer
                        interprets.
      extra:            Domain-specific telemetry (component ids,
                        wall observations, etc.).
    """
    level_key:        str
    turn:             int
    agent_pos:        Any
    target:           Any
    refuel_positions: List[Any]
    current_budget:   int
    full_budget:      int
    direct_cost:      Optional[int]      = None
    reason:           str                = ""
    decision_history: List[dict]         = field(default_factory=list)
    extra:            dict               = field(default_factory=dict)

    def to_record(self) -> dict:
        """JSON-safe dict for kb persistence.

        Hashable opaque positions are coerced to lists when they are
        tuple-shaped (typical for ARC cell coords); other shapes pass
        through unchanged.  The recorder doesn't assume a coordinate
        system — callers that store complex objects should pre-
        serialise them.
        """
        d = asdict(self)
        d["agent_pos"]        = _to_jsonable(self.agent_pos)
        d["target"]           = _to_jsonable(self.target)
        d["refuel_positions"] = [_to_jsonable(r) for r in self.refuel_positions]
        return d


def _to_jsonable(pos: Any) -> Any:
    """Coerce a position to a JSON-serialisable shape.

    Tuples become lists (so json.dump round-trips); everything else
    passes through.  Callers that store complex hashables are
    responsible for their own serialisation.
    """
    if isinstance(pos, tuple):
        return list(pos)
    return pos


def record_stranded_event(kb: dict, event: StrandedEvent) -> dict:
    """Append ``event`` to ``kb["strategic_replan"]
    ["stranded_events_by_level"][level_key]``.

    Creates the bucket if absent.  Returns the persisted record dict
    (post-serialisation), useful for logging.

    Idempotency: the recorder does NOT dedupe — a repeated
    observation at the same configuration is a meaningful signal
    (the agent keeps hitting the same dead-end), and the consumer
    decides whether to fold duplicates.
    """
    if not isinstance(kb, dict):
        raise TypeError("kb must be a dict")
    bucket = kb.setdefault("strategic_replan", {})
    by_level = bucket.setdefault("stranded_events_by_level", {})
    lst = by_level.setdefault(str(event.level_key), [])
    record = event.to_record()
    lst.append(record)
    return record


def stranded_events_for_level(
    kb: dict,
    level_key: str,
) -> List[dict]:
    """Return the persisted stranded events for ``level_key``.

    Stable read shape so consumers (future replanners) don't depend
    on the precise nesting of the kb dict.  Returns an empty list
    when the bucket is absent — callers don't need to defend against
    KeyError.
    """
    if not isinstance(kb, dict):
        return []
    return list(
        ((kb.get("strategic_replan") or {})
         .get("stranded_events_by_level") or {})
        .get(str(level_key), [])
    )


def clear_stranded_events_for_level(kb: dict, level_key: str) -> int:
    """Drop all stranded-event records for ``level_key``.

    Used on full-game reset, when the agent's accumulated decision
    history is no longer the right context for current play.
    Returns the number of records removed.
    """
    if not isinstance(kb, dict):
        return 0
    by_level = ((kb.get("strategic_replan") or {})
                .get("stranded_events_by_level") or {})
    lst = by_level.get(str(level_key))
    if not lst:
        return 0
    n = len(lst)
    by_level[str(level_key)] = []
    return n
