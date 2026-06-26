"""learning_loop.py -- the closed learning-from-error loop.

Ties together the four pieces so a mistake becomes durable, recallable experience:

  1. REGISTER the error            -> error_ledger (a KNOWN_STORE under the unified KB root)
  2. RECALL on a similar situation -> error_ledger.recall_surface (ranked to NOW, with the fix)
  3. FORMULATE a new solution      -> when no fix exists yet: playback-mine prior history, or run a
                                      caller-supplied formulator (a discriminating probe, etc.)
  4. REGISTER the solution         -> error_ledger.resolve links the error to the SOLUTION (and, if
                                      replayable, its solutions_kb id) -- a future recurrence reuses it

Design:
  - All durable state lives in the default KB: the error ledger (with its solution links) is federated
    via kb_paths; replayable win-paths live in solutions_kb and are referenced by id from the ledger.
  - Pure orchestration + dependency-injected formulator, so it is unit-testable without a live game and
    has no hard dependency on the heavyweight miners.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from error_ledger import ErrorLedger, ErrorRecord


class LearningLoop:
    def __init__(self, ledger: Optional[ErrorLedger] = None):
        self.ledger = ledger if ledger is not None else ErrorLedger()

    # 1. register -----------------------------------------------------------
    def register_error(self, category: str, description: str, variation: str = "",
                       fix: str = "", level: str = "") -> ErrorRecord:
        return self.ledger.record(category, description, variation=variation, fix=fix, level=level)

    # 2. recall -------------------------------------------------------------
    def recall(self, situation, k: int = 4) -> Tuple[Optional[str], List[ErrorRecord]]:
        """Surface prior experience for a situation: a recall directive (with the working solution to
        reuse, or a 'formulate one' flag) plus the matching records."""
        return self.ledger.recall_surface(situation, k=k), self.ledger.recall(situation, k=k)

    def has_solution(self, category: str) -> bool:
        return self.ledger.resolution_for(category) is not None

    # 3. formulate ----------------------------------------------------------
    def formulate(self, category: str, situation="",
                  formulator: Optional[Callable[[str], Optional[str]]] = None) -> Optional[str]:
        """Propose a NEW solution for an unresolved error area.  If a ``formulator`` is supplied it is
        used (e.g. a closure over playback mining of recorded history, or a discriminating probe);
        otherwise the default playback-mining formulator is tried.  Returns a solution string or
        ``None`` if none could be formulated (caller then falls back to live probing)."""
        if self.has_solution(category):
            r = self.ledger.resolution_for(category)
            return r.resolution if r else None
        fn = formulator or _default_playback_formulator()
        if fn is None:
            return None
        try:
            return fn(str(situation))
        except Exception:
            return None

    # 4. register the solution ---------------------------------------------
    def register_solution(self, category: str, resolution: str, solution_id: str = "",
                          description: str = "") -> int:
        """Attach the working solution to the error area (links error<->solution in the KB).  Returns
        the number of error records updated (0 if the area was never registered -- in which case record
        it first)."""
        n = self.ledger.resolve(category, resolution, solution_id=solution_id, description=description)
        if n == 0:
            # area not previously registered: register it already-resolved so it is recallable
            self.ledger.record(category, description or f"resolved area {category}",
                               resolution=resolution, solution_id=solution_id)
            n = 1
        return n

    # convenience: the full step for a recurring situation -----------------
    def on_situation(self, situation, formulator: Optional[Callable[[str], Optional[str]]] = None,
                     k: int = 4) -> Optional[str]:
        """One call for a live decision point: recall matching past errors; for any area without a
        working solution, attempt to formulate one and register it; return a combined directive for the
        actor (reuse the recalled solutions; act on the freshly formulated ones)."""
        surface, hits = self.recall(situation, k=k)
        extra = []
        for r in hits:
            if not (r.resolved and r.resolution):
                sol = self.formulate(r.category, situation=situation, formulator=formulator)
                if sol:
                    self.register_solution(r.category, sol, description=r.description)
                    extra.append(f"  - {r.category}: FORMULATED solution -> {sol}")
        if not surface and not extra:
            return None
        return "\n".join([s for s in [surface] if s] + extra)


def _default_playback_formulator() -> Optional[Callable[[str], Optional[str]]]:
    """A formulator that mines recorded history for an answer, if playback_mining is importable.
    Returns ``None`` when unavailable so the loop degrades to live probing.  The concrete query is
    caller-driven; here we only expose the capability hook (callers with a live ``world`` pass a
    closure that runs the specific miner)."""
    try:
        import playback_mining  # noqa: F401  (capability present)
    except Exception:
        return None
    # Without a live world/query this hook cannot mine; return None so callers supply a bound closure.
    return None
