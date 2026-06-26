"""Efficiency heuristics: typed, declarative, DISCOVERED — not hand-coded.

WHY THIS EXISTS
---------------
A planner that only minimizes a base cost still walks into churn: it will
greedily commit an irreversible op (impale a block) before the rest of the
targets are arranged, then pay to undo it. A human learns rules-of-thumb that
avoid this ("stage everything before you commit"; "keep each not-yet-placed
target in its own lane for control"; "when you place a block, leave it where
it'll be usable"). We want COS to DISCOVER such heuristics from its own
playback and apply them, game-agnostically.

DESIGN (two halves share one engine — the plan search)
  * DISCOVERY: mine CHURN SIGNATURES from playback — work that was wasted
    (an op later undone, a block repositioned twice, two targets interfering
    in a shared lane). Each signature instantiates a typed Heuristic
    (P9: heuristics as typed declarative data, never branchy code). A mined
    heuristic is a HYPOTHESIS until a replay shows it actually prefers the
    shorter trajectory (mined-rules-are-hypotheses-until-replayed).
  * APPLICATION: an INTERPRETER scores a candidate plan against the active
    heuristics, returning a soft penalty the planner adds to its ranking.
    So "lookahead that avoids churn" is just the existing min-cost search
    with these penalties folded in.

Everything here is ROLE/RELATION level (a plan is an abstract step list of
{op, target, lane, reversible}); no game vocabulary, so heuristics transfer
across games by signature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional


# ---------------------------------------------------------------------------
# Abstract plan representation (game-agnostic)
# ---------------------------------------------------------------------------
# A PlanStep is a dict with:
#   op:         'stage' | 'commit' | 'move' | 'release'
#   target:     role-typed id of the entity acted on (e.g. 'red', 'blue')
#   lane:       optional int lane/row the target occupies after the step
#   reversible: bool — is the op cheap to undo? (commit/impale = False)
# The interpreter reads ONLY these fields, so it is domain-independent.
PlanStep = Dict[str, object]


# ---------------------------------------------------------------------------
# Anti-pattern templates: penalty functions over an abstract plan.
# Each returns a non-negative count of anti-pattern occurrences. The KEY of
# this registry is a heuristic `kind`; discovery instantiates a Heuristic of
# that kind, the interpreter dispatches on it.
# ---------------------------------------------------------------------------

def _ap_defer_irreversible_until_ready(steps: List[PlanStep]) -> int:
    """Penalize committing an irreversible op on a target before EVERY other
    committed target has been staged. (= 'stage all before you commit'.)"""
    committed_targets = {s["target"] for s in steps if s.get("op") == "commit"}
    staged_before: set = set()
    penalty = 0
    for s in steps:
        op, tgt = s.get("op"), s.get("target")
        if op == "stage":
            staged_before.add(tgt)
        elif op == "commit":
            # other targets that will be committed but aren't staged yet
            pending = {t for t in committed_targets
                       if t != tgt and t not in staged_before}
            penalty += len(pending)
    return penalty


def _ap_avoid_undo_redo(steps: List[PlanStep]) -> int:
    """Penalize a target whose committed/placed state is later undone
    (commit ... release, or moved again after a commit)."""
    penalty = 0
    committed: set = set()
    for s in steps:
        op, tgt = s.get("op"), s.get("target")
        if op == "commit":
            committed.add(tgt)
        elif op in ("release", "move") and tgt in committed:
            penalty += 1
            committed.discard(tgt)
    return penalty


def _ap_separate_lanes(steps: List[PlanStep]) -> int:
    """Penalize two not-yet-committed targets sharing a lane (interference;
    = 'keep each pending target in its own row for control')."""
    lanes: Dict[object, object] = {}      # target -> lane (staged, not committed)
    penalty = 0
    for s in steps:
        op, tgt, lane = s.get("op"), s.get("target"), s.get("lane")
        if op == "stage" and lane is not None:
            # does another pending target already sit in this lane?
            for other, ol in lanes.items():
                if other != tgt and ol == lane:
                    penalty += 1
            lanes[tgt] = lane
        elif op == "commit":
            lanes.pop(tgt, None)
    return penalty


def _ap_place_in_ready_state(steps: List[PlanStep]) -> int:
    """Penalize repositioning a target more than once (it should have been
    placed in its usable/impalable spot the first time)."""
    moves: Dict[object, int] = {}
    for s in steps:
        if s.get("op") in ("stage", "move"):
            moves[s["target"]] = moves.get(s["target"], 0) + 1
    return sum(max(0, c - 1) for c in moves.values())


ANTIPATTERN_TEMPLATES: Dict[str, Callable[[List[PlanStep]], int]] = {
    "defer_irreversible_until_ready": _ap_defer_irreversible_until_ready,
    "avoid_undo_redo": _ap_avoid_undo_redo,
    "separate_lanes": _ap_separate_lanes,
    "place_in_ready_state": _ap_place_in_ready_state,
}

# Which churn signature (mined from playback) implies which heuristic kinds.
CHURN_TO_HEURISTICS: Dict[str, List[str]] = {
    "undo_redo": ["avoid_undo_redo", "defer_irreversible_until_ready"],
    "repeated_reposition": ["place_in_ready_state"],
    "shared_lane_interference": ["separate_lanes"],
}

_HUMAN = {
    "defer_irreversible_until_ready":
        "stage ALL targets into a ready state before committing any "
        "irreversible op (don't commit, then have to undo it)",
    "avoid_undo_redo":
        "don't put a target into a state you'll have to undo later",
    "separate_lanes":
        "keep each not-yet-committed target in its own lane/row for control",
    "place_in_ready_state":
        "when you place a target, leave it in the spot it'll be used from "
        "(don't reposition it twice)",
}


@dataclass
class Heuristic:
    """A typed, declarative efficiency heuristic. `kind` selects an
    anti-pattern penalty template; `trigger_signature` keys it to a relational
    context for transfer; `status` is 'hypothesis' until replay-verified."""
    heuristic_id: str
    kind: str
    trigger_signature: dict = field(default_factory=dict)
    weight: float = 1.0
    credence: float = 0.5
    support: int = 1
    status: str = "hypothesis"          # 'hypothesis' | 'verified' | 'refuted'
    provenance: dict = field(default_factory=dict)

    def describe(self) -> str:
        return _HUMAN.get(self.kind, self.kind)


# ---------------------------------------------------------------------------
# Interpreter: score a plan against active heuristics
# ---------------------------------------------------------------------------

def plan_penalty(steps: List[PlanStep], heuristics: List[Heuristic],
                 *, include_hypotheses: bool = True) -> float:
    """Soft cost a plan incurs under the active heuristics. The planner adds
    this to its base cost so churny plans rank worse. Verified heuristics
    apply at full credence; hypotheses apply down-weighted (so an unproven
    heuristic nudges but never dominates)."""
    total = 0.0
    for h in heuristics:
        if h.status == "refuted":
            continue
        if h.status == "hypothesis" and not include_hypotheses:
            continue
        fn = ANTIPATTERN_TEMPLATES.get(h.kind)
        if not fn:
            continue
        count = fn(steps)
        if not count:
            continue
        w = h.credence * h.weight
        if h.status == "hypothesis":
            w *= 0.5
        total += w * count
    return total


def explain_penalty(steps: List[PlanStep], heuristics: List[Heuristic]) -> str:
    parts = []
    for h in heuristics:
        if h.status == "refuted":
            continue
        fn = ANTIPATTERN_TEMPLATES.get(h.kind)
        c = fn(steps) if fn else 0
        if c:
            parts.append(f"{h.kind} x{c} ({h.describe()})")
    return "; ".join(parts) if parts else "no anti-patterns"


# ---------------------------------------------------------------------------
# Discovery: churn signatures -> typed Heuristic records (hypotheses)
# ---------------------------------------------------------------------------

def _new_heuristic_id(kind: str, sig: dict) -> str:
    base = "_".join(str(v) for v in sig.values()) if sig else "global"
    return f"heur_{kind}_{base}"[:80]


def discover_heuristics(churn: dict, existing: Optional[List[Heuristic]] = None,
                        signature: Optional[dict] = None) -> List[Heuristic]:
    """Turn mined churn signatures into typed heuristics. Each signature kind
    maps to one or more anti-pattern heuristics (CHURN_TO_HEURISTICS). New
    ones are minted as hypotheses; repeats bump support/credence
    (consolidation). Returns the full (updated) heuristic list."""
    out = list(existing or [])
    by_key = {(h.kind, _new_heuristic_id(h.kind, h.trigger_signature)): h
              for h in out}
    sig = signature or {}
    for churn_kind, info in (churn or {}).items():
        support = int(info.get("support", 0)) if isinstance(info, dict) else int(info)
        if support <= 0:
            continue
        for hkind in CHURN_TO_HEURISTICS.get(churn_kind, []):
            hid = _new_heuristic_id(hkind, sig)
            key = (hkind, hid)
            if key in by_key:
                h = by_key[key]
                h.support += support
                h.credence = min(0.5 + 0.1 * h.support, 0.95)
            else:
                h = Heuristic(
                    heuristic_id=hid, kind=hkind, trigger_signature=dict(sig),
                    credence=min(0.5 + 0.1 * support, 0.95), support=support,
                    status="hypothesis",
                    provenance={"derived_from": f"churn:{churn_kind}"})
                out.append(h)
                by_key[key] = h
    return out


# ---------------------------------------------------------------------------
# Replay-verification: a heuristic is a hypothesis until it demonstrably
# prefers the SHORTER real trajectory.
# ---------------------------------------------------------------------------

def verify_against_trajectories(h: Heuristic, churny: List[PlanStep],
                                efficient: List[PlanStep]) -> Heuristic:
    """Promote/refute a heuristic by replay: if `efficient` truly has fewer
    steps AND this heuristic penalizes `churny` MORE than `efficient`, the
    heuristic correctly discriminates -> verified (credence up). If it gets
    the preference backwards, refute it. Otherwise leave it a hypothesis."""
    fn = ANTIPATTERN_TEMPLATES.get(h.kind)
    if not fn:
        return h
    pc, pe = fn(churny), fn(efficient)
    shorter_is_efficient = len(efficient) < len(churny)
    if shorter_is_efficient and pc > pe:
        h.status = "verified"
        h.credence = min(h.credence + 0.2, 0.98)
    elif shorter_is_efficient and pe > pc:
        h.status = "refuted"
        h.credence = max(h.credence - 0.3, 0.05)
    return h


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _default_path() -> Path:
    try:
        from subroutine_kb import DEFAULT_SUBROUTINE_KB_PATH as _p
        return Path(_p).parent / "efficiency_heuristics.json"
    except Exception:
        return Path("efficiency_heuristics.json")


def load_heuristics(path: Optional[Path] = None) -> List[Heuristic]:
    p = Path(path or _default_path())
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return [Heuristic(**d) for d in data]
    except Exception:
        return []


def save_heuristics(heuristics: List[Heuristic], path: Optional[Path] = None) -> None:
    p = Path(path or _default_path())
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps([asdict(h) for h in heuristics], indent=1))
    except Exception:
        pass


def format_heuristics_surface(heuristics: List[Heuristic]) -> str:
    """Render active heuristics for the actor (so the discovered rules-of-thumb
    are visible guidance, not just hidden plan biases)."""
    act = [h for h in heuristics if h.status != "refuted"]
    if not act:
        return ""
    lines = ["EFFICIENCY HEURISTICS (discovered from your own churn; "
             "verified ones are reliable, hypotheses are tentative):"]
    for h in sorted(act, key=lambda h: (h.status != "verified", -h.credence)):
        tag = "VERIFIED" if h.status == "verified" else "hypothesis"
        lines.append(f"  - [{tag}, cred={h.credence:.2f}] {h.describe()}")
    return "\n".join(lines)
