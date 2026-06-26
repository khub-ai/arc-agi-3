"""Proceduralization bridge — connect the engine-clean dual-process /
procedural-skill modules (cognitive_os) to the harness's live world model.

See docs/SPEC_proceduralization.md + docs/SPEC_vlm_backward_reasoning.md.

The committed ``cognitive_os.procedural_skill`` / ``cognitive_os.dual_process``
modules are deliberately game-agnostic: a Skill is keyed on an OPAQUE relational
signature + goal, and the controller arbitrates System-1 (run a confident cached
skill) vs System-2 (recruit the VLM reasoner, deliberate iff the harness trigger
fires) and compiles successful deliberate solves.

This module supplies the two harness-side pieces those modules leave abstract:

  1. ``relational_signature(world)`` — the skin-agnostic SIGNATURE of the
     current situation, built from Layer-A relations over tracked entities'
     ROLES (not entity ids / colours / counts), so a compiled skill transfers
     across colour / count / and same-shape variants (P11, P14).
  2. ``goal_key(world)`` — the current goal, from the substrate-computed win
     relation + the next-required target, so skills compile at the
     sub-maneuver granularity ("how to complete the next target here").

Plus thin ``arbitrate`` / ``learn`` wrappers the per-turn loop calls.  This
module owns NO persistence — the caller holds the controller and persists its
skills through the canonical KB; keeping the bridge stateless avoids a second
unreconciled store.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from world_knowledge import WorldKnowledge          # noqa: E402
import knowledge_crystallization as _K               # noqa: E402
from cognitive_os.dual_process import DualProcessController, Decision  # noqa: E402
from cognitive_os.procedural_skill import Skill       # noqa: E402

# Relation kinds whose participants are too generic to discriminate a
# situation (always-present scaffolding); excluded from the signature.
_SIG_INERT_ROLES = {"scenery", "decoration", "background"}


def _role(world: WorldKnowledge, name: str) -> str:
    rec = world.entities.get(name)
    return _K._role_of(rec) if rec is not None else "unknown"


def relational_signature(world: WorldKnowledge,
                         *,
                         turn: Optional[int] = None,
                         kinds: Optional[Iterable[str]] = None) -> frozenset:
    """Skin-agnostic signature of the world's current relational facts.

    Each key is ``(kind, sorted(role_pair...), direction)`` over the ROLES of
    the participating entities — never their names — so the same relational
    shape yields the same signature regardless of colour / count / game.
    Built from the most recent delta that carries Layer-A relations at or
    before ``turn`` (default: latest)."""
    kinds = set(kinds) if kinds is not None else None
    deltas = getattr(world, "deltas_observed", None) or []
    rels: list = []
    for d in reversed(deltas):
        if turn is not None and getattr(d, "to_turn", 0) > turn:
            continue
        r = getattr(d, "relations", None) or []
        if r:
            rels = r
            break
    sig = set()
    for r in rels:
        kind = r.get("kind")
        if kinds is not None and kind not in kinds:
            continue
        roles = [_role(world, e) for e in (r.get("entities") or [])]
        if not roles or any(rl in _SIG_INERT_ROLES for rl in roles):
            continue
        sig.add((kind, tuple(sorted(roles)), r.get("direction")))
    return frozenset(sig)


def goal_key(world: WorldKnowledge) -> Optional[tuple]:
    """The current goal as an opaque key: the highest-credence committed win
    relation + the substrate-computed NEXT required target.  None when no win
    relation is crystallized yet (the VLM then reasons unbiased)."""
    cands = [h for h in getattr(world, "win_condition_hypotheses", [])
             if getattr(h, "win_relation", None)]
    if not cands:
        return None
    cands.sort(key=lambda h: getattr(h, "credence", 0.0), reverse=True)
    rel = cands[0].win_relation
    nxt = None
    try:
        nxt = (_K.evaluate_win_relation(world, rel).get("detail") or {}).get("next")
    except Exception:
        nxt = None
    return (rel.get("type"), tuple(rel.get("roles") or []),
            rel.get("axis"), f"next:{nxt}")


def arbitrate(controller: DualProcessController,
              world: WorldKnowledge,
              *,
              reasoner,
              trigger: bool,
              sig_kinds: Optional[Iterable[str]] = None
              ) -> Tuple[Decision, frozenset, Optional[tuple]]:
    """Decide System-1 (auto-run a confident cached skill) vs System-2 (recruit
    the VLM reasoner) for the current situation.  Returns (decision, entry
    signature, goal_key) — the caller keeps the signature/goal to pass to
    ``learn`` after the outcome is known."""
    sig = relational_signature(world, kinds=sig_kinds)
    goal = goal_key(world)
    if goal is None:
        actions = tuple(reasoner(goal=None, situation=sig,
                                 deliberate=bool(trigger), suggested=None) or ())
        return (Decision("reason" if trigger else "greedy", actions, None,
                         bool(trigger), "no grounded goal yet"), sig, goal)
    return controller.decide(sig, goal, reasoner=reasoner, trigger=trigger), sig, goal


def learn(controller: DualProcessController,
          decision: Decision,
          *,
          success: bool,
          goal: Optional[tuple],
          signature: Iterable[tuple]):
    """Close the loop: verify a used skill, or compile a new one on a
    successful deliberate solve.  ``signature`` must be the ENTRY signature
    captured by ``arbitrate`` (not the post-outcome one)."""
    if goal is None:
        return None
    return controller.observe(decision, success=success, goal=goal,
                              signature=signature)


# ---------------------------------------------------------------------------
# Canonical-store projection (subroutine KB is the single home for skills)
# ---------------------------------------------------------------------------

def _tuplify(x):
    """Recursively turn JSON lists back into tuples so signature/goal keys
    are hashable (JSON round-trips tuples as lists)."""
    if isinstance(x, list):
        return tuple(_tuplify(e) for e in x)
    return x


def controller_from_kb(subroutines: Iterable,
                       *,
                       auto_min_uses: int = 2,
                       auto_min_rate: float = 0.75) -> DualProcessController:
    """Build a DualProcessController whose skills are the subroutine-KB
    records that carry a structured ``signature`` (the compiled-policy
    records).  Legacy / dialogic records (no signature) are left to the
    existing relevance-surfacing path and are not auto-run."""
    skills: List[Skill] = []
    for s in subroutines:
        sig = getattr(s, "signature", None) or []
        gk = getattr(s, "goal_key", None) or []
        chain = getattr(s, "concrete_chain", None) or []
        if not sig or not gk or not chain:
            continue
        att = getattr(s, "attempts", None)
        uses = getattr(att, "n_applied", 0) if att else 0
        succ = getattr(att, "n_succeeded", 0) if att else 0
        skills.append(Skill(
            goal=_tuplify(gk),
            signature=frozenset(_tuplify(f) for f in sig),
            actions=tuple(chain),
            uses=max(uses, 1), successes=max(succ, 0),
            provenance="kb",
        ))
    return DualProcessController(skills=skills, auto_min_uses=auto_min_uses,
                                 auto_min_rate=auto_min_rate)


def compile_success_to_kb(*,
                          actions: Iterable[str],
                          signature: Iterable[tuple],
                          goal: tuple,
                          game_id: str,
                          level: int,
                          turn_range: list,
                          path: Optional[Path] = None) -> None:
    """Persist a successful deliberate maneuver as a signature-keyed
    subroutine-KB record (the canonical store), so it can auto-run next
    time the same relational signature recurs.  No-op on empty inputs."""
    acts = list(actions or [])
    sig = [list(f) for f in (signature or [])]
    if not acts or not sig or goal is None:
        return
    import subroutine_kb as _S  # local import to avoid cycle at module load
    kwargs = dict(
        name=f"skill::{goal[0]}::{goal[-1]}" if isinstance(goal, (tuple, list)) and goal else "skill",
        description="auto-compiled skill (proceduralization): achieves the "
                    "goal under the recorded relational signature.",
        problem_solved=f"goal={goal}",
        concrete_chain=acts,
        expected_outcome="goal-relation progresses / completes for the next target",
        game_id=game_id, level=level, turn_range=list(turn_range),
        original_goal=str(goal),
        signature=sig, goal_key=list(goal),
    )
    if path is not None:
        kwargs["path"] = path
    _S.promote_chain_as_subroutine(**kwargs)
