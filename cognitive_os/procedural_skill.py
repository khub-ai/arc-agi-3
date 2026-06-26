"""Proceduralization: compile a successful deliberate solve into a cached,
retrievable SKILL (System 2 -> System 1). See docs/SPEC_proceduralization.md.

Engine-clean: a Skill is keyed on a *relational signature* (a set of opaque
precondition fact keys) + the goal it achieves, holds the maneuver, and tracks
credence from outcomes. Retrieval is signature-subset match; a skill applies
when its preconditions hold in the current situation, so it transfers to richer
situations and same-shape variants (P14). No game specifics, no pixel
coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Set, Tuple


@dataclass
class Skill:
    """A compiled, reusable maneuver.

    ``goal`` and ``signature`` entries are opaque hashable keys (e.g.
    ``Condition.canonical_key()`` tuples); the engine never interprets them.
    """
    goal: tuple
    signature: frozenset            # precondition fact keys that held when it worked
    actions: Tuple[str, ...]
    uses: int = 0
    successes: int = 0
    provenance: str = "compiled"

    @property
    def success_rate(self) -> float:
        return self.successes / self.uses if self.uses else 0.0

    def record(self, success: bool) -> "Skill":
        self.uses += 1
        if success:
            self.successes += 1
        return self

    def is_automatic(self, *, min_uses: int = 3, min_rate: float = 0.8) -> bool:
        """Confident enough to run as System 1 (bypass the System-2 trigger)."""
        return self.uses >= min_uses and self.success_rate >= min_rate


def compile_skill(goal: tuple,
                  signature: Iterable[tuple],
                  actions: Sequence[str],
                  *,
                  provenance: str = "compiled") -> Skill:
    """Crystallize a successful (goal, entry-signature, action-chain) episode
    into a Skill. ``signature`` should be the precondition facts that held at
    entry (ideally minimal); consolidation widens/narrows it over time."""
    return Skill(goal=goal, signature=frozenset(signature),
                 actions=tuple(actions), provenance=provenance)


def applicable(skill: Skill, situation_keys: Set[tuple]) -> bool:
    """A skill applies when ALL its precondition facts hold now (signature
    ⊆ situation). Subset match -> transfers to richer situations/variants."""
    return skill.signature <= set(situation_keys)


def retrieve(skills: Iterable[Skill],
             situation_keys: Set[tuple],
             goal: Optional[tuple] = None) -> List[Skill]:
    """Applicable skills for the current situation (and goal, if given),
    ranked best-first by (success_rate, uses)."""
    keys = set(situation_keys)
    cands = [s for s in skills
             if (goal is None or s.goal == goal) and applicable(s, keys)]
    return sorted(cands, key=lambda s: (s.success_rate, s.uses), reverse=True)


def choose(skills: Iterable[Skill],
           situation_keys: Set[tuple],
           goal: tuple) -> Tuple[Optional[Skill], str]:
    """Dual-process arbitration for a goal in the current situation:

      ("auto", skill)    — a confident matching skill: run it as System 1.
      ("suggest", skill) — a matching but not-yet-confident skill: surface it
                           to the VLM as remembered know-how (System 2 biased).
      ("reason", None)   — no applicable skill: recruit VLM backward reasoning.
    """
    ranked = retrieve(skills, situation_keys, goal)
    if not ranked:
        return None, "reason"
    best = ranked[0]
    return (best, "auto") if best.is_automatic() else (best, "suggest")
