"""Dual-process executive controller: the loop that ties proceduralization
(System 1) to VLM backward reasoning (System 2). See
docs/SPEC_vlm_backward_reasoning.md + docs/SPEC_proceduralization.md.

Per turn:
  * a CONFIDENT cached skill that matches -> run it (System 1), no reasoning;
  * else recruit the VLM reasoner, DELIBERATE iff the harness trigger fires
    (multi-step goal gap / stall / blocked), passing any unproven matching
    skill as a hint;
  * on a successful DELIBERATE solve -> compile the maneuver into a skill
    (System 2 -> System 1), so the System-2 trigger fires less next time.

Engine-clean: the VLM is an injected ``reasoner`` callable and the harness's
escalation cue is an injected ``trigger`` boolean — no perception, VLM, or game
specifics here. The live packet-builder / signature-builder / verifier are the
harness's job (game-side); this is the executive logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

from .procedural_skill import Skill, choose, compile_skill


@dataclass
class Decision:
    mode: str                       # "auto" (System 1) | "reason" (S2 deliberate) | "greedy" (S2 shallow)
    actions: Tuple[str, ...]
    skill: Optional[Skill]          # the cached skill used (for credence update), if any
    deliberate: bool
    rationale: str = ""


# A reasoner is the action VLM: given the goal + situation (+ optional skill
# hint + whether to deliberate), it returns an ordered action list.
Reasoner = Callable[..., Sequence[str]]


@dataclass
class DualProcessController:
    skills: List[Skill] = field(default_factory=list)
    auto_min_uses: int = 3
    auto_min_rate: float = 0.8

    def decide(self,
               situation_keys: Set[tuple],
               goal: tuple,
               *,
               reasoner: Reasoner,
               trigger: bool) -> Decision:
        """Arbitrate System 1 (cached skill) vs System 2 (VLM reasoning)."""
        skill, mode = choose(self.skills, situation_keys, goal)
        # re-evaluate "automatic" against this controller's thresholds
        if skill is not None and skill.is_automatic(min_uses=self.auto_min_uses,
                                                    min_rate=self.auto_min_rate):
            return Decision("auto", skill.actions, skill, False,
                            f"System1: cached skill (rate={skill.success_rate:.2f}, uses={skill.uses})")
        suggested = skill if (skill is not None and mode != "reason") else None
        actions = tuple(reasoner(goal=goal, situation=situation_keys,
                                 deliberate=bool(trigger), suggested=suggested) or ())
        return Decision("reason" if trigger else "greedy", actions, None, bool(trigger),
                        ("System2 deliberate" if trigger else "System2 shallow")
                        + (f" (+skill hint)" if suggested else ""))

    def observe(self,
                decision: Decision,
                *,
                success: bool,
                goal: tuple,
                signature: Iterable[tuple]) -> Optional[Skill]:
        """Close the loop: update the used skill's credence, or — on a
        successful DELIBERATE solve — compile/strengthen a skill. Returns the
        affected skill (for inspection)."""
        if decision.skill is not None:
            decision.skill.record(success)              # verify cached skill (P13/P1)
            return decision.skill
        if success and decision.deliberate and decision.actions:
            sig = frozenset(signature)
            existing = next((s for s in self.skills
                             if s.goal == goal and s.signature == sig), None)
            if existing is not None:
                return existing.record(True)            # consolidate repeats
            sk = compile_skill(goal, sig, decision.actions)
            sk.record(True)                             # first successful use
            self.skills.append(sk)
            return sk
        return None
