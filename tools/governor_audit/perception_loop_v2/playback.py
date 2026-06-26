"""Playback — offline consolidation / review (deterministic core).

See docs/SPEC_playback.md.  This first build provides:
  * a RECORD READER over a recorded run directory, and
  * three deterministic analyzers: efficiency, kb_validation, perception.

It is DRY by default (``commit=False``): it emits findings, each tied to
a RELIABLE recorded signal, and does NOT write the KB.  Per the spec's
reliability rule, every finding rests on a recorded ground-truth signal
— the outcome (``score_increased``), the detector's ``visual_events``, or
the substrate delta (``summary`` / ``agent_moved`` / ``entities_changed``)
— never a fresh VLM re-reading of pixels.  Pure-Python, deterministic,
no model calls.

Input format: a run directory containing ``world_knowledge.json`` (with
``deltas_observed``) and per-turn ``turn_NNN/`` directories (with
``strategy_prompt.md`` and a consumed ``strategy_reply``).  Game-agnostic.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# Reversible action pairs (game-agnostic: up/down, retract/extend).  Used
# only to spot oscillations; the labels are the harness's action ids.
_INVERSE = {"ACTION1": "ACTION2", "ACTION2": "ACTION1",
            "ACTION3": "ACTION4", "ACTION4": "ACTION3",
            "UP": "DOWN", "DOWN": "UP", "LEFT": "RIGHT", "RIGHT": "LEFT"}

_UNDO_ACTIONS = ("ACTION7",)   # the undo action (per the sk48 KB; configurable)

# Generic words excluded when measuring whether a rationale reflects a
# procedure's CONTENT (so overlap is carried by domain terms, not "block"/
# "move"/"row" boilerplate).
_STOP = {"this", "that", "with", "from", "into", "your", "you", "will",
         "when", "then", "each", "must", "only", "not", "are", "its", "for",
         "but", "before", "after", "the", "and", "of", "to", "in", "on", "it",
         "is", "be", "or", "by", "block", "blocks", "action", "agent", "move",
         "turn", "step", "steps", "row", "left", "right", "down", "side",
         "one", "two", "all", "any", "per", "via", "so", "this", "next",
         "here", "now", "them", "they", "their", "what", "have", "has"}


def _salient(text: str) -> set:
    return {w for w in __import__("re").findall(r"[a-z]{4,}", (text or "").lower())
            if w not in _STOP}


def _considers(rationale: str, relevant_ids: list, sub_tokens: dict,
               min_overlap: int = 3) -> bool:
    """Did the rationale take a surfaced [RELEVANT NOW] procedure into
    account?  True if it names the KB explicitly OR reflects the
    procedure's CONTENT (>= min_overlap domain terms shared with the
    surfaced procedure's name/steps) — so describing the maneuver counts,
    not only saying the word 'technique'."""
    low = (rationale or "").lower()
    if any(k in low for k in ("subroutine", "technique", "relevant now",
                              "constraint", "recovery")):
        return True
    rt = _salient(rationale)
    return any(sub_tokens.get(sid) and len(rt & sub_tokens[sid]) >= min_overlap
               for sid in relevant_ids)


@dataclass
class TurnRecord:
    turn: int
    action: str
    summary: str = ""
    agent_moved: bool = False
    entities_changed: list = field(default_factory=list)
    visual_events: list = field(default_factory=list)
    score_increased: bool = False
    rationale: str = ""
    relevant_surfaced: bool = False        # a [RELEVANT NOW] procedure was shown this turn
    considered: Optional[bool] = None      # did the reply reference it (None if none surfaced)
    relations: list = field(default_factory=list)      # substrate relations the actor saw
    surfaced_ids: list = field(default_factory=list)   # KB subroutine ids shown this turn
    relevant_ids: list = field(default_factory=list)   # subset tagged [RELEVANT NOW]

    @property
    def no_change(self) -> bool:
        s = (self.summary or "").lower()
        if "no change" in s:
            return True
        return (not self.agent_moved and not self.entities_changed
                and not self.visual_events)

    def dir_events(self, direction: str) -> list:
        return [e.get("entity") for e in (self.visual_events or [])
                if e.get("direction") == direction]


@dataclass
class Finding:
    analyzer: str
    kind: str
    turns: list
    detail: str
    ground_truth: str          # the reliable signal this finding rests on
    severity: str = "info"     # "info" | "warn"


@dataclass
class ReviewReport:
    run_ref: str
    n_turns: int
    findings: list = field(default_factory=list)
    committed: bool = False

    def by_analyzer(self) -> dict:
        out: dict = {}
        for f in self.findings:
            out.setdefault(f.analyzer, []).append(f)
        return out


# ---------------------------------------------------------------------------
# Record reader
# ---------------------------------------------------------------------------

def load_trajectory(run_dir) -> list[TurnRecord]:
    """Read a run directory into a list of TurnRecords (ordered by turn).

    Pulls the per-turn delta from ``world_knowledge.json`` (the reliable
    recorded signal), then enriches each turn with the actor's rationale
    and whether a [RELEVANT NOW] procedure was surfaced + referenced."""
    run = Path(run_dir)
    recs: list[TurnRecord] = []
    wk = run / "world_knowledge.json"
    if wk.exists():
        try:
            blob = json.loads(wk.read_text(encoding="utf-8"))
        except Exception:
            blob = {}
        for d in (blob.get("deltas_observed") or []):
            recs.append(TurnRecord(
                turn=int(d.get("to_turn") or d.get("from_turn") or 0),
                action=str(d.get("action") or ""),
                summary=str(d.get("summary") or ""),
                agent_moved=bool(d.get("agent_moved")),
                entities_changed=list(d.get("entities_changed") or []),
                visual_events=list(d.get("visual_events") or []),
                score_increased=bool(d.get("score_increased")),
                relations=list(d.get("relations") or []),
            ))
    # KB content tokens per subroutine, for content-based 'considered'.
    sub_tokens: dict = {}
    try:
        import subroutine_kb as _S
        for s in _S.load():
            blob = " ".join(filter(None, [
                str(getattr(s, "name", "") or ""),
                str(getattr(s, "description", "") or ""),
                " ".join(getattr(s, "relational_steps", []) or [])]))
            sub_tokens[getattr(s, "subroutine_id", None)] = _salient(blob)
    except Exception:
        sub_tokens = {}

    # Enrich from per-turn files (best-effort; never raises).
    for r in recs:
        td = run / f"turn_{r.turn:03d}"
        for fn in ("strategy_reply.consumed.txt", "strategy_reply.txt"):
            p = td / fn
            if p.exists():
                try:
                    r.rationale = str(json.loads(
                        p.read_text(encoding="utf-8")).get("rationale", "") or "")
                except Exception:
                    pass
                break
        pp = td / "strategy_prompt.md"
        if pp.exists():
            try:
                txt = pp.read_text(encoding="utf-8")
            except Exception:
                txt = ""
            # Parse the surfaced KB subroutine ids and which were tagged
            # [RELEVANT NOW] (one entry per line: "  sub_<id>: '<name>' ...").
            for line in txt.splitlines():
                m = re.match(r"\s+(sub_\w+):", line)
                if m:
                    sid = m.group(1)
                    r.surfaced_ids.append(sid)
                    if "[RELEVANT NOW]" in line:
                        r.relevant_ids.append(sid)
            if "[RELEVANT NOW]" in txt:
                r.relevant_surfaced = True
                # 'considered' = the rationale reflects the surfaced
                # procedure's CONTENT (its maneuver), not just the word
                # 'technique'.
                r.considered = _considers(r.rationale, r.relevant_ids, sub_tokens)
    return recs


# ---------------------------------------------------------------------------
# Analyzers (deterministic; each finding cites a ground-truth signal)
# ---------------------------------------------------------------------------

def analyze_efficiency(traj: list[TurnRecord],
                       no_progress_min: int = 3) -> list[Finding]:
    """Find wasted steps: undo runs, no-progress streaks, oscillations.
    Grounds in the action log + substrate deltas."""
    f: list[Finding] = []
    n = len(traj)
    # Undo runs (>=2 consecutive undo actions => the rewound actions + the
    # undos were all wasted steps).
    i = 0
    while i < n:
        if traj[i].action in _UNDO_ACTIONS:
            j = i
            while j < n and traj[j].action in _UNDO_ACTIONS:
                j += 1
            if j - i >= 2:
                f.append(Finding(
                    "efficiency", "undo_run", [t.turn for t in traj[i:j]],
                    f"{j - i} consecutive UNDO steps — the actions they rewound, "
                    f"plus the undos, were wasted; a better plan avoids the move "
                    f"that had to be undone.",
                    "action log", "warn"))
            i = j
        else:
            i += 1
    # No-progress streaks (>= no_progress_min turns with no observable effect).
    i = 0
    while i < n:
        if traj[i].no_change:
            j = i
            while j < n and traj[j].no_change:
                j += 1
            if j - i >= no_progress_min:
                f.append(Finding(
                    "efficiency", "no_progress_streak",
                    [t.turn for t in traj[i:j]],
                    f"{j - i} consecutive turns produced no observable change "
                    f"(action did nothing in this state).",
                    "substrate delta (no change)", "warn"))
            i = j
        else:
            i += 1
    # Oscillations: an effective action immediately followed by its inverse.
    for a, b in zip(traj, traj[1:]):
        if (_INVERSE.get(a.action) == b.action
                and not a.no_change and not b.no_change):
            f.append(Finding(
                "efficiency", "oscillation", [a.turn, b.turn],
                f"{a.action} then its inverse {b.action} — motion likely "
                f"cancelled; candidate for removal.",
                "action log + deltas", "info"))
    return f


def analyze_perception(traj: list[TurnRecord],
                       transient_window: int = 3) -> list[Finding]:
    """Validate the detector against outcome ground truth: misses (score
    advanced with no detector activation) and transient activations
    (activated then reverted with no score change between)."""
    f: list[Finding] = []
    for r in traj:
        if r.score_increased and not r.dir_events("activated"):
            f.append(Finding(
                "perception", "possible_miss", [r.turn],
                "Score advanced but the detector reported no activation this "
                "turn — the progress signal may have been missed.",
                "outcome (score_increased) vs detector", "warn"))
    # transient activation: activated then reverted within the window, with
    # no score advance in between => possibly a false/transient activation.
    last_act: dict = {}
    for idx, r in enumerate(traj):
        for e in r.dir_events("activated"):
            last_act[e] = idx
        for e in r.dir_events("reverted"):
            if e in last_act and idx - last_act[e] <= transient_window:
                span = traj[last_act[e]:idx + 1]
                if not any(x.score_increased for x in span):
                    f.append(Finding(
                        "perception", "transient_activation",
                        [traj[last_act[e]].turn, r.turn],
                        f"{e} activated then reverted within "
                        f"{idx - last_act[e]} turns with no score change — "
                        f"possibly a transient/false activation to verify.",
                        "detector activated->reverted vs outcome", "info"))
                last_act.pop(e, None)
    return f


def analyze_kb_validation(traj: list[TurnRecord]) -> list[Finding]:
    """Check that surfaced [RELEVANT NOW] procedures were taken into
    account.  Grounds in the prompt's [RELEVANT NOW] tag vs the reply.
    AGGREGATED: one finding per run (not per turn) so a procedure that was
    surfaced-but-ignored across many turns reads as a single signal."""
    surfaced = [r for r in traj if r.relevant_surfaced]
    if not surfaced:
        return []
    ignored = [r for r in surfaced if r.considered is False]
    if not ignored:
        return []
    eg = ", ".join(f"t{r.turn}" for r in ignored[:6])
    more = "..." if len(ignored) > 6 else ""
    return [Finding(
        "kb_validation", "relevant_not_considered",
        [r.turn for r in ignored],
        f"A [RELEVANT NOW] procedure/constraint was surfaced on {len(surfaced)} "
        f"turn(s) but the rationale referenced it on only "
        f"{len(surfaced) - len(ignored)}; it was IGNORED on {len(ignored)} "
        f"(e.g. {eg}{more}). The consider-relevant-procedure discipline targets "
        f"exactly these; if the procedure was genuinely irrelevant, its "
        f"surfacing trigger is too loose.",
        "prompt [RELEVANT NOW] vs reply rationale", "warn")]


def _run_game_id(run_dir) -> Optional[str]:
    try:
        blob = json.loads((Path(run_dir) / "world_knowledge.json").read_text(
            encoding="utf-8"))
        return blob.get("game_id")
    except Exception:
        return None


def analyze_kb_triggering(traj: list[TurnRecord],
                          game_id: Optional[str] = None) -> list[Finding]:
    """Validate the KB's TRIGGERING mechanism: an entry that SHOULD be
    invoked must actually be invoked.

    For each KB subroutine that the prompt surfaced this turn (so it
    existed at run time — no KB-drift false positives), re-run the LIVE
    relevance gate over the recorded ``relations`` and compare to whether
    the prompt actually tagged it ``[RELEVANT NOW]``:
      * relevant by the gate but NOT tagged  -> trigger_miss (the entry
        should have been invoked but the pipeline didn't tag it),
      * tagged but NOT relevant by the gate  -> trigger_spurious
        (over-triggering — the trigger fired when it shouldn't).
    Grounds in the recorded delta.relations + the KB's own relevance gate
    + the prompt's tag.  Returns [] (and validates nothing) if the
    relevance machinery or KB can't be loaded."""
    try:
        import subroutine_kb as _S
        subs = {getattr(s, "subroutine_id", None): s for s in _S.load()}
        try:
            rel_fn = _S.relevance_to_situation
            kinds_fn = _S._current_relation_kinds
        except AttributeError:
            from knowledge_crystallization import (  # noqa: E402
                relevance_to_situation as rel_fn,
                _current_relation_kinds as kinds_fn)
    except Exception:
        return []

    miss: dict = {}
    spurious: dict = {}
    for r in traj:
        if not r.relations or not r.surfaced_ids:
            continue
        kinds = kinds_fn(r.relations)
        for sid in r.surfaced_ids:
            s = subs.get(sid)
            if s is None:
                continue                      # changed since the run; skip
            thr = 0.5 if getattr(s, "game_id", None) == game_id else 0.7
            should = rel_fn(s, kinds) >= thr
            tagged = sid in r.relevant_ids
            if should and not tagged:
                miss.setdefault(sid, []).append(r.turn)
            elif tagged and not should:
                spurious.setdefault(sid, []).append(r.turn)

    f: list[Finding] = []
    for sid, turns in miss.items():
        f.append(Finding(
            "kb_triggering", "trigger_miss", turns,
            f"KB entry '{sid}' matched the relevance gate but was NOT tagged "
            f"[RELEVANT NOW] on {len(turns)} surfaced turn(s) (e.g. t{turns[0]}) "
            f"— an entry that should have been invoked was not. Investigate the "
            f"surfacing/relevance pipeline.",
            "delta.relations vs KB relevance gate vs prompt [RELEVANT NOW]", "warn"))
    for sid, turns in spurious.items():
        f.append(Finding(
            "kb_triggering", "trigger_spurious", turns,
            f"KB entry '{sid}' was tagged [RELEVANT NOW] on {len(turns)} turn(s) "
            f"(e.g. t{turns[0]}) but does NOT meet the relevance gate — the "
            f"trigger is firing too loosely.",
            "delta.relations vs KB relevance gate vs prompt [RELEVANT NOW]", "warn"))
    return f


_ANALYZERS = {
    "efficiency": analyze_efficiency,
    "kb_validation": analyze_kb_validation,
    "perception": analyze_perception,
}
_DEFAULT_ANALYZERS = ("efficiency", "kb_validation", "perception", "kb_triggering")


def review(run_ref, *, turns: Optional[tuple] = None,
           analyzers: tuple = _DEFAULT_ANALYZERS,
           commit: bool = False) -> ReviewReport:
    """Review a run for the chosen analyzers over the chosen turn range.

    ``commit=False`` (default) is a DRY review: it returns findings without
    writing anything.  ``commit=True`` is intentionally a no-op in this
    first build — committing findings will route through
    ContextMemory.commit_knowledge (reconciled, verifiable, idempotent)
    once that path is wired; until then Playback only reports."""
    traj = load_trajectory(run_ref)
    if turns:
        lo, hi = turns
        traj = [r for r in traj if lo <= r.turn <= hi]
    game_id = _run_game_id(run_ref)
    findings: list[Finding] = []
    for name in analyzers:
        if name == "kb_triggering":
            findings += analyze_kb_triggering(traj, game_id=game_id)
        else:
            fn = _ANALYZERS.get(name)
            if fn:
                findings += fn(traj)
    findings.sort(key=lambda x: (x.turns[0] if x.turns else 0))
    return ReviewReport(run_ref=str(run_ref), n_turns=len(traj),
                        findings=findings, committed=False)


def format_report(rep: ReviewReport) -> str:
    lines = [f"PLAYBACK REVIEW  run={rep.run_ref}  turns={rep.n_turns}  "
             f"findings={len(rep.findings)}  (DRY)"]
    for analyzer, fs in rep.by_analyzer().items():
        lines.append(f"  [{analyzer}] {len(fs)} finding(s):")
        for x in fs:
            tr = (f"t{x.turns[0]}" if len(x.turns) == 1
                  else f"t{x.turns[0]}-{x.turns[-1]}" if x.turns else "-")
            lines.append(f"    - ({x.severity}) {x.kind} @ {tr}: {x.detail} "
                         f"[ground: {x.ground_truth}]")
    return "\n".join(lines)


if __name__ == "__main__":   # pragma: no cover
    import sys
    rep = review(sys.argv[1] if len(sys.argv) > 1 else ".")
    print(format_report(rep))
