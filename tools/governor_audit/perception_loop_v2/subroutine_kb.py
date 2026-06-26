"""Subroutine knowledge base — VLM-described, dynamically
generalised, cross-level / cross-game.

ARCHITECTURE PRINCIPLE — the substrate holds and ranks records.
It does NOT classify them.  Every field that describes WHAT a
subroutine does, WHEN it applies, or WHAT it expects is free-form
text supplied by the strategy actor (a VLM).  The substrate
treats those strings as opaque.

This is a deliberate departure from the predecessor `tactic_kb.py`,
which carried closed-vocabulary preconditions (e.g. an
``agent_has_extension_tool`` predicate).  Those closed-vocab
predicates leaked game-specific concepts ("arm", "tool", "swipe")
into the substrate, defeating the goal of dealing with hundreds
of different games autonomously.  See
``docs/SPEC_subroutine_kb.md`` for the full design discussion.

What the substrate DOES guarantee:

  * Atomic persistence to a single JSON file.
  * Provenance tracking — every application records (game_id,
    level, turn_range, original_goal, created_at).
  * Credence tracking — asymmetric updates (failures decay faster
    than successes raise) and per-application attempt counters.
  * Lineage tracking — a forked variant records its parent and
    the VLM-supplied note describing what was generalised or
    specialised.
  * Retrieval ranking — by credence × success ratio × recency,
    with the VLM as the final filter (it sees the top-K records
    and picks one to apply, or none).

A Subroutine is "a tool an LLM agent uses."  The substrate's job
is to remember tools without telling the agent what they mean.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


try:
    from kb_paths import kb_path as _kb_path
except ImportError:  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path
# Unified KB root (see kb_paths.py + docs/SPEC_knowledge_base.md).
DEFAULT_SUBROUTINE_KB_PATH = _kb_path("subroutine_kb.json")


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceRecord:
    """One application or fork of a subroutine — when, where, and
    under what original goal it ran."""
    game_id:           str
    level:             int
    turn_range:        list[int]      # [first_turn, last_turn]
    original_goal:     str            # free-form goal description
    created_at:        float          # unix time
    outcome:           str            # "success", "partial", "failure",
                                       # "seeded" (for the initial
                                       # promotion entry)
    notes:             str = ""       # free-form, optional
    source:            str = "earned"  # WHERE this knowledge came from
                                       # (orthogonal to ``outcome``):
                                       #   "earned"   — the system's own
                                       #                play produced it;
                                       #   "dialogic" — a human authored
                                       #                it in dialogue
                                       #                (rare; must stay
                                       #                honestly flagged so
                                       #                it is never mistaken
                                       #                for earned, and so
                                       #                the system knows it
                                       #                is still unverified
                                       #                by its own play).
                                       # Defaulting to "earned" is honest
                                       # for the overwhelming bulk of
                                       # records (self-play); the dialogic
                                       # ingestion path sets "dialogic"
                                       # explicitly.


@dataclass
class AttemptsRecord:
    """Per-subroutine outcome counters.  Used by the retrieval
    ranker and by the credence update rule."""
    n_applied:    int = 0
    n_succeeded:  int = 0
    n_failed:     int = 0
    n_partial:    int = 0


@dataclass
class Subroutine:
    """A reusable procedure the strategy actor has identified as
    worth remembering.  All descriptive fields are free-form text;
    the substrate treats them as opaque strings.

    The concrete_chain is what the system actually ran the first
    (or canonical) time, in the original game's action vocabulary.
    It's both proof-of-concept and a fallback "literal replay" the
    strategy actor can paraphrase or adapt when applying.
    """
    subroutine_id:     str            # auto-generated
    name:              str            # VLM-given, free-form
    description:       str            # VLM-given, free-form —
                                       # what it does, when to use it,
                                       # what to watch out for
    problem_solved:    str            # VLM-given, free-form — the
                                       # problem this addressed in the
                                       # originating context (goal +
                                       # obstacle + constraint)
    concrete_chain:    list[str]      # literal action ids as observed
    expected_outcome:  str            # VLM-given, free-form — what
                                       # should be true after the
                                       # subroutine runs (used as a
                                       # rough success check)
    provenance:        list[ProvenanceRecord]
    attempts:          AttemptsRecord = field(
                            default_factory=AttemptsRecord)
    credence:          float = 0.6     # base for a newly promoted
                                       # subroutine; asymmetric updates
    parent_id:         Optional[str] = None
                                       # id of the subroutine this was
                                       # forked from, if any
    variant_notes:     str = ""        # VLM annotation: what this
                                       # variant specialised /
                                       # generalised vs. its parent
    step_count:        int = 0         # len(concrete_chain) — the cost
                                       # of this solution.  Lower is
                                       # better (ARC-AGI-3 scores
                                       # step-efficiency).  On merge,
                                       # the SHORTER winning chain
                                       # replaces the longer one so the
                                       # KB always holds the best-known
                                       # solution, never a stale long one.
    raw_chain:         list = field(default_factory=list)
                                       # the unpruned chain as literally
                                       # executed (kept for provenance
                                       # when concrete_chain is a pruned
                                       # candidate).
    relational_steps:  list[str] = field(default_factory=list)
                                       # the GENERALIZED, game-agnostic
                                       # procedure: an ordered list of
                                       # steps phrased in ROLES and
                                       # RELATIONS (not colours, not
                                       # counts, not literal action ids),
                                       # so the procedure transfers across
                                       # piece colour / piece count / and
                                       # (when the roles map) other games.
                                       # ``concrete_chain`` is the literal
                                       # replay; ``relational_steps`` is
                                       # what the procedure MEANS.  A
                                       # dialogic (human-authored) recipe
                                       # typically carries relational_steps
                                       # and an EMPTY concrete_chain — there
                                       # is no single literal chain that
                                       # generalises over count.
    signature:         list = field(default_factory=list)
                                       # STRUCTURED relational precondition
                                       # signature captured at the entry of
                                       # the maneuver that earned this
                                       # record: a list of [kind, [roles],
                                       # direction] facts (skin-agnostic,
                                       # role-keyed).  Empty for legacy /
                                       # dialogic records.  When present it
                                       # enables signature-⊆ retrieval and
                                       # System-1 auto-run via the
                                       # proceduralization bridge (the
                                       # compiled-policy contract of
                                       # docs/SPEC_proceduralization.md).
    goal_key:          list = field(default_factory=list)
                                       # the goal this record achieves, as a
                                       # structured key (win-relation type +
                                       # roles + axis + next-target).  Pairs
                                       # with ``signature`` for keyed
                                       # retrieval.


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load(path: Path = DEFAULT_SUBROUTINE_KB_PATH) -> list[Subroutine]:
    """Read the subroutine KB from disk.  Empty list on first run."""
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[Subroutine] = []
    for rd in data.get("subroutines", []):
        provenance = [ProvenanceRecord(**p)
                      for p in rd.get("provenance", [])]
        attempts_d = rd.get("attempts") or {}
        attempts = AttemptsRecord(**attempts_d)
        rd2 = {**rd, "provenance": provenance, "attempts": attempts}
        out.append(Subroutine(**rd2))
    return out


def save(subroutines: list[Subroutine],
         path: Path = DEFAULT_SUBROUTINE_KB_PATH) -> None:
    """Atomic write via tempfile rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"subroutines": [asdict(s) for s in subroutines]}
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


# Credence policy.  Asymmetric — failures decay faster than
# successes raise.  Failed application drops credence ≥ promotion
# threshold drops a subroutine out of the top-K surface.
_BASE_CREDENCE = 0.6
_SUCCESS_BUMP  = 0.08
_PARTIAL_BUMP  = 0.03
_FAILURE_DECAY = 0.15


def _new_id(name: str, ts: float) -> str:
    """Stable-ish id from name + timestamp.  Not cryptographic; just
    a way to give every record a unique handle."""
    safe = "".join(c if c.isalnum() or c == "_" else "_"
                   for c in name.lower())[:40]
    return f"sub_{safe}_{int(ts)}"


def promote_chain_as_subroutine(
    *,
    name: str,
    description: str,
    problem_solved: str,
    concrete_chain: list[str],
    expected_outcome: str,
    game_id: str,
    level: int,
    turn_range: list[int],
    original_goal: str,
    notes: str = "",
    parent_id: Optional[str] = None,
    variant_notes: str = "",
    signature: Optional[list] = None,
    goal_key: Optional[list] = None,
    relational_steps: Optional[list] = None,
    path: Path = DEFAULT_SUBROUTINE_KB_PATH,
) -> Subroutine:
    """Create or merge a Subroutine and persist.

    A subroutine with the SAME ``name`` and matching ``parent_id``
    is treated as the same record; its provenance grows and its
    credence bumps.  Different names (or a different parent_id under
    the same name) create a new record — the VLM is the namer, so
    it's free to fork variants with new names whenever it judges
    them meaningfully different.
    """
    existing = load(path)
    ts = time.time()

    # Find a same-name same-parent existing record to merge into
    match: Optional[Subroutine] = None
    for s in existing:
        if s.name == name and s.parent_id == parent_id:
            match = s
            break

    new_prov = ProvenanceRecord(
        game_id=game_id, level=level,
        turn_range=list(turn_range),
        original_goal=original_goal,
        created_at=ts,
        outcome="success" if match is None else "success",
        notes=notes,
    )

    chain = list(concrete_chain)
    if match is None:
        sub = Subroutine(
            subroutine_id=_new_id(name, ts),
            name=name,
            description=description,
            problem_solved=problem_solved,
            concrete_chain=chain,
            expected_outcome=expected_outcome,
            provenance=[ProvenanceRecord(
                game_id=game_id, level=level,
                turn_range=list(turn_range),
                original_goal=original_goal,
                created_at=ts,
                outcome="seeded", notes=notes,
            )],
            attempts=AttemptsRecord(
                n_applied=1, n_succeeded=1, n_failed=0, n_partial=0,
            ),
            credence=_BASE_CREDENCE,
            parent_id=parent_id,
            variant_notes=variant_notes,
            step_count=len(chain),
            signature=list(signature) if signature else [],
            goal_key=list(goal_key) if goal_key else [],
            relational_steps=list(relational_steps) if relational_steps else [],
        )
        existing.append(sub)
    else:
        match.provenance.append(new_prov)
        match.attempts.n_applied   += 1
        match.attempts.n_succeeded += 1
        match.credence = min(1.0, match.credence + _SUCCESS_BUMP)
        # backfill the structured signature/goal if the record lacked them
        if signature and not match.signature:
            match.signature = list(signature)
        if goal_key and not match.goal_key:
            match.goal_key = list(goal_key)
        if relational_steps and not match.relational_steps:
            match.relational_steps = list(relational_steps)
        # KEEP-SHORTEST: if this winning chain is strictly shorter
        # than the stored one, it replaces it.  The KB always holds
        # the best-known (shortest) solution for the task, so the
        # next trial is asked to beat a tighter target — not the
        # original wasteful path.
        if chain and (match.step_count == 0
                      or len(chain) < match.step_count):
            old = match.step_count
            match.concrete_chain = chain
            match.step_count = len(chain)
            match.provenance[-1].notes = (
                (match.provenance[-1].notes + " | ") if match.provenance[-1].notes else ""
            ) + f"SHORTER solution: {old} -> {len(chain)} steps; replaced canonical chain."
        sub = match

    save(existing, path)
    return sub


def is_dialogic(sub: Subroutine) -> bool:
    """True if this subroutine entered the KB via human dialogue
    rather than being earned by the system's own play.

    Keyed on the seed provenance record's ``source`` tag.  Used by the
    surface to render an honest HUMAN-AUTHORED marker so the actor never
    mistakes expert advice for a battle-tested procedure, and so a later
    pass can tell which records still owe the system a self-play
    confirmation."""
    return any(
        p.outcome == "seeded" and getattr(p, "source", "earned") == "dialogic"
        for p in sub.provenance
    )


# Credence prior for a human-authored (dialogic) subroutine.  High
# enough to surface and be taken seriously (expert advice), but BELOW
# what a repeatedly-self-confirmed earned subroutine reaches — it has
# not yet been verified by the system's own play.
_DIALOGIC_CREDENCE = 0.7


def ingest_dialogic_subroutine(
    *,
    name: str,
    description: str,
    problem_solved: str,
    expected_outcome: str,
    relational_steps: list[str],
    game_id: str,
    level: int,
    original_goal: str,
    concrete_chain: Optional[list[str]] = None,
    turn_range: Optional[list[int]] = None,
    credence: float = _DIALOGIC_CREDENCE,
    notes: str = "",
    path: Path = DEFAULT_SUBROUTINE_KB_PATH,
) -> Subroutine:
    """Register a HUMAN-AUTHORED subroutine into the KB via dialogue.

    This is the rare path: normally the system earns every subroutine
    from its own play.  Here a human hands it a generalised procedure.
    The entry is kept scrupulously honest so it is never confused with
    earned knowledge (the prior "human-confirmed" lessons that turned
    out to be a cheat are the cautionary tale):

      * the seed provenance record is tagged ``source="dialogic"``;
      * attempt counters are ``0/0`` — the system has NOT run it, so we
        do not fabricate a success;
      * credence is a human-input PRIOR (``_DIALOGIC_CREDENCE``), not a
        self-confirmed value — it ranks the record sensibly without
        pretending it has been validated;
      * the procedure is stored GENERALISED in ``relational_steps``
        (roles + relations, no colour / count / literal action ids);
        ``concrete_chain`` is left empty unless a literal replay is
        genuinely available, because no single literal chain generalises
        over piece count.

    Idempotent on ``name`` (a top-level, non-forked record): re-ingesting
    refreshes the descriptive / step fields in place (so a human can
    revise the wording) WITHOUT touching earned counters, credence, or
    provenance history that real play may have accrued since."""
    existing = load(path)
    ts = time.time()
    steps = list(relational_steps)
    chain = list(concrete_chain or [])

    for s in existing:
        if s.name == name and s.parent_id is None:
            # Idempotent refresh of human-authored content; preserve
            # everything the system itself has since earned.
            s.description = description
            s.problem_solved = problem_solved
            s.expected_outcome = expected_outcome
            s.relational_steps = steps
            if concrete_chain is not None:
                s.concrete_chain = chain
                s.step_count = len(chain)
            save(existing, path)
            return s

    sub = Subroutine(
        subroutine_id=_new_id(name, ts),
        name=name,
        description=description,
        problem_solved=problem_solved,
        concrete_chain=chain,
        expected_outcome=expected_outcome,
        provenance=[ProvenanceRecord(
            game_id=game_id, level=level,
            turn_range=list(turn_range or [0, 0]),
            original_goal=original_goal,
            created_at=ts,
            outcome="seeded",
            source="dialogic",
            notes=(notes or "Human-authored via dialogue; generalised "
                   "for any piece colour/count; UNVERIFIED by the "
                   "system's own play."),
        )],
        attempts=AttemptsRecord(),   # 0/0 — honestly never run by the system
        credence=credence,
        parent_id=None,
        variant_notes="",
        step_count=len(chain),
        relational_steps=steps,
    )
    existing.append(sub)
    save(existing, path)
    return sub


def record_application_outcome(
    *,
    subroutine_id: str,
    outcome: str,
        # Allowed values:
        #   "success"  — strong positive signal (e.g. score advanced
        #                while applying); bumps credence + counter.
        #   "partial"  — explicit partial-progress self-report from
        #                the strategy actor; small credence bump.
        #   "failure"  — explicit failure self-report from the
        #                strategy actor; larger credence decay.
        #   "no_op"    — provenance entry only; credence / counters
        #                unchanged.  This is the default for
        #                "applied this turn, no terminal signal yet"
        #                so applications mid-window don't inflate
        #                credence just by repeating.
    game_id: str,
    level: int,
    turn_range: list[int],
    original_goal: str,
    notes: str = "",
    path: Path = DEFAULT_SUBROUTINE_KB_PATH,
) -> None:
    """Update credence + attempts after the system applies a stored
    subroutine and observes the result.

    Credence-affecting policy is INTENTIONALLY TIGHT.  Default
    behaviour for a mid-application turn is ``no_op`` — provenance
    grows, but credence and attempt counters do not change.
    Credence only moves on:
      * a hard win signal (substrate-computed: score advance),
      * an explicit partial or failure self-report from the
        strategy actor.
    This protects credence from inflation by repeated mid-
    application turns where the substrate has no real evidence the
    subroutine is working."""
    subs = load(path)
    target: Optional[Subroutine] = None
    for s in subs:
        if s.subroutine_id == subroutine_id:
            target = s
            break
    if target is None:
        return

    target.provenance.append(ProvenanceRecord(
        game_id=game_id, level=level,
        turn_range=list(turn_range),
        original_goal=original_goal,
        created_at=time.time(),
        outcome=outcome, notes=notes,
    ))
    if outcome == "success":
        target.attempts.n_applied   += 1
        target.attempts.n_succeeded += 1
        target.credence = min(1.0, target.credence + _SUCCESS_BUMP)
    elif outcome == "partial":
        target.attempts.n_applied   += 1
        target.attempts.n_partial   += 1
        target.credence = min(1.0, target.credence + _PARTIAL_BUMP)
    elif outcome == "failure":
        target.attempts.n_applied   += 1
        target.attempts.n_failed    += 1
        target.credence = max(0.0, target.credence - _FAILURE_DECAY)
    # "no_op": provenance only; counters and credence unchanged.

    save(subs, path)


# ---------------------------------------------------------------------------
# Retrieval ranking
# ---------------------------------------------------------------------------


def _success_ratio(s: Subroutine) -> float:
    """Bayes-smoothed success ratio.  +1 success / +2 trials prior
    so a 0/0 record sits at 0.5 rather than dropping to 0 / nan."""
    n_app = s.attempts.n_applied + 2
    n_suc = s.attempts.n_succeeded + 1
    return n_suc / n_app


def _recency_score(s: Subroutine, now: float) -> float:
    """Decay an entry by how long since its most recent provenance
    record.  Half-life of 30 days (in seconds)."""
    if not s.provenance:
        return 1.0
    last = max(p.created_at for p in s.provenance)
    age_days = max(0.0, (now - last) / 86400.0)
    return 0.5 ** (age_days / 30.0)


# Game-agnostic relation kinds the substrate computes (Layer A).  A
# subroutine that NAMES these in its steps/description is stating the
# precondition-relations it operates on.  Matching them against the
# CURRENT situation's relations is the reader-half of knowledge transfer:
# it lets a situationally-relevant procedure surface even when its
# credence is low, instead of being buried under high-credence but
# irrelevant records.  Relations are game-agnostic, so this transfers
# across games (and, in principle, to a robotics domain whose perception
# emits the same relational vocabulary).
_RELATION_KINDS = frozenset({
    "co_displacement", "motion_blocked", "motion_arrested_at",
    "penetration", "support_relation", "same_row", "same_col",
    "ordered_along", "clearance", "adjacent",
})

# How hard relevance lifts a record's rank.  At weight W a fully-relevant
# subroutine (every stated relation active now) scores up to (1+W)x its
# credence-base.  For transfer, a clearly-matching procedure must DOMINATE
# credence so it reliably reaches the actor: a fresh, unproven-but-on-point
# technique (credence-base ~0.35) has to clear a maxed-out but off-point
# chain (credence-base ~0.9).  W=2.5 gives that margin (0.35*3.5 > 0.9)
# while relevance still only boosts when the precondition actually matches
# (an unrelated situation leaves the record credence-ranked and buried).
_RELEVANCE_WEIGHT = 2.5


def _stated_relation_kinds(s: "Subroutine") -> set:
    """The relation kinds this subroutine NAMES, scanned from its
    generalized steps + free-form description/problem_solved.  Empty for
    a free-form earned chain that names no relations (it then ranks on
    credence alone — relevance never penalizes, only boosts)."""
    text = " ".join([
        s.problem_solved or "",
        s.description or "",
        " ".join(s.relational_steps or []),
    ]).lower()
    return {kind for kind in _RELATION_KINDS if kind in text}


def relevance_to_situation(s: "Subroutine", current_kinds: set) -> float:
    """[0,1] — fraction of the subroutine's stated relation-kinds that are
    ACTIVE in the current situation.  0 when either side names no
    relations.  This is a checkable, game-agnostic match (no free-form
    classification by the substrate); the VLM still does final
    applicability judgment by reading the surfaced descriptions."""
    if not current_kinds:
        return 0.0
    stated = _stated_relation_kinds(s)
    if not stated:
        return 0.0
    return len(stated & current_kinds) / len(stated)


def _current_relation_kinds(current_relations) -> set:
    """Extract the set of relation kinds present this turn from the
    Layer-A relations (dicts or RelationRecords)."""
    kinds = set()
    for r in current_relations or []:
        kind = r.get("kind") if isinstance(r, dict) else getattr(r, "kind", None)
        if kind:
            kinds.add(kind)
    return kinds


def rank_subroutines(
    subroutines: list[Subroutine],
    *,
    current_relations=None,
    now: Optional[float] = None,
    k: int = 5,
) -> list[Subroutine]:
    """Return top-k subroutines, ranked by
    (credence × success_ratio × recency) BOOSTED by relevance to the
    current situation when ``current_relations`` is given.

    The credence base answers "has this worked before?"; the relevance
    boost answers "does this fit the situation in front of me NOW?".
    Transfer needs both: without relevance, a low-credence but
    situationally-critical procedure (e.g. a freshly-seeded recovery
    technique) is buried under high-credence but off-point records and
    never reaches the actor — which is exactly how the applicable
    knowledge failed to surface at sk48 lc=1."""
    if not subroutines:
        return []
    now = now if now is not None else time.time()
    current_kinds = _current_relation_kinds(current_relations)
    scored = []
    for s in subroutines:
        base = (
            s.credence
            * _success_ratio(s)
            * _recency_score(s, now)
        )
        rel = relevance_to_situation(s, current_kinds)
        score = base * (1.0 + _RELEVANCE_WEIGHT * rel)
        scored.append((score, s))
    scored.sort(key=lambda kv: -kv[0])
    return [s for (_score, s) in scored[:k]]


# ---------------------------------------------------------------------------
# Surface for the strategy prompt
# ---------------------------------------------------------------------------


def _subroutine_origin_game(s) -> Optional[str]:
    """Origin game id of a subroutine (from its provenance), or None when
    game-agnostic. Used by the cross-game recall gate."""
    for p in (getattr(s, "provenance", None) or []):
        g = getattr(p, "game_id", None)
        if g is None and isinstance(p, dict):
            g = p.get("game_id")
        if g:
            return g
    return None


def format_subroutine_surface_slim(
    subroutines: Optional[list[Subroutine]] = None,
    k: int = 3,
    current_relations=None,
    game_id: Optional[str] = None,
    current_sig: Optional[dict] = None,
) -> str:
    """Slim variant for the trimmed strategy prompt.

    One line per subroutine: id, name, concrete chain, success
    ratio.  No problem_solved / expected_outcome / fork notes —
    just enough for the actor to recognize an applicable pattern
    and decide whether to apply.  Returns "" when KB is empty so
    the prompt doesn't carry empty boilerplate.

    When ``current_relations`` (this turn's Layer-A relations) is given,
    ranking is relevance-boosted so a procedure that matches the CURRENT
    situation surfaces even at low credence, and matching entries are
    flagged [RELEVANT NOW] — the reader half of knowledge transfer.
    """
    if subroutines is None:
        subroutines = load()
    if not subroutines:
        return ""
    # Cross-game recall gate: when the current game is known, admit a subroutine
    # only if its origin game is agnostic, the same game, or a signature variant
    # (see cross_game_knowledge.admits_xgame). Fails closed for cross-game.
    if game_id is not None:
        def _admit(origin) -> bool:
            if not origin or origin == game_id:
                return True
            try:
                from cross_game_knowledge import admits_xgame as _xadmit
                return _xadmit(origin, current_sig, current_game_id=game_id)
            except Exception:
                return False
        subroutines = [s for s in subroutines
                       if _admit(_subroutine_origin_game(s))]
        if not subroutines:
            return ""
    top = rank_subroutines(subroutines, k=k, current_relations=current_relations)
    if not top:
        return ""
    current_kinds = _current_relation_kinds(current_relations)

    # Auto-audit the surfaced subroutines for execution-gap defects
    # (hardcoded coords / missing-precondition), so a flawed technique
    # warns the actor at the point of use — the same defect classes the KB
    # auditor flags for lessons, applied here where the side-swipe
    # technique actually lives.
    defect_map: dict[str, list[str]] = {}
    try:
        from knowledge_crystallization import detect_subroutine_defects  # noqa: E402
        for _d in detect_subroutine_defects(top):
            defect_map.setdefault(_d.get("label", ""), []).append(_d["type"])
    except Exception:
        pass

    lines: list[str] = [""]
    lines.append(
        f"STORED SUBROUTINES ({len(top)} top-ranked by credence AND "
        f"relevance to the CURRENT situation; chains validated in prior "
        f"trials).  Cite the id if you apply one.  A [RELEVANT NOW] tag "
        f"means the procedure's precondition-relations match what you are "
        f"seeing this turn — strongly consider it even if its credence is "
        f"low.  IMPORTANT: a stored chain is a known-good solution, NOT a "
        f"target to replay verbatim.  ARC-AGI-3 scores STEP-EFFICIENCY, so "
        f"when you recognize the same situation, treat the chain's length "
        f"as a BUDGET TO BEAT: drop redundant moves and aim for the same "
        f"acceptance signals in FEWER steps.  A shorter winning chain "
        f"replaces the stored one automatically."
    )
    for s in top:
        a = s.attempts
        dialogic = is_dialogic(s)
        rel = relevance_to_situation(s, current_kinds)
        reltag = " [RELEVANT NOW]" if rel >= 0.5 else ""
        _defs = defect_map.get(s.name)
        deftag = (f" [FLAGGED: {', '.join(sorted(set(_defs)))} — verify "
                  f"before applying]" if _defs else "")
        tag = ((" [HUMAN-AUTHORED, unverified by own play]" if dialogic else "")
               + reltag + deftag)
        if s.concrete_chain:
            n = s.step_count or len(s.concrete_chain)
            cost = f"best={n} steps -> BEAT IT"
        else:
            cost = "generalized recipe (no literal chain)"
        lines.append(
            f"  {s.subroutine_id}: {s.name!r}{tag} "
            f"(c={s.credence:.2f}, {a.n_succeeded}/{a.n_applied} OK, {cost})"
        )
        if s.concrete_chain:
            chain = ", ".join(s.concrete_chain[:14])
            if len(s.concrete_chain) > 14:
                chain += ", ..."
            lines.append(f"    chain: {chain}")
        elif s.relational_steps:
            lines.append(
                "    steps: " + " | ".join(s.relational_steps[:6])
            )
        lines.append(f"    what: {s.description[:120]}")
    return "\n".join(lines) + "\n"


def format_subroutine_surface(
    subroutines: Optional[list[Subroutine]] = None,
    k: int = 5,
    current_relations=None,
) -> str:
    """Render top-k subroutines as a natural-language block for the
    strategy prompt.  The substrate writes them out as free-form
    descriptions; the VLM decides relevance and adaptation.  When
    ``current_relations`` is given, ranking is relevance-boosted toward
    procedures whose precondition-relations match the current turn."""
    if subroutines is None:
        subroutines = load()
    if not subroutines:
        return ("  (subroutine KB is empty — the system has not yet "
                "identified any reusable procedures.)")
    top = rank_subroutines(subroutines, k=k, current_relations=current_relations)
    if not top:
        return "  (no subroutines surfaced)"

    lines: list[str] = []
    lines.append(
        f"  TOP {len(top)} subroutines from the KB (ranked by "
        "credence × success ratio × recency).  Each is a tool you "
        "may APPLY (with adaptation), FORK (create a variant), or "
        "IGNORE.  Treat the descriptions as VLM-authored and "
        "free-form — the substrate makes no claims about their "
        "applicability beyond the credence numbers."
    )
    for s in top:
        lines.append("")
        a = s.attempts
        lines.append(
            f"  SUBROUTINE id={s.subroutine_id!r}  "
            f"credence={s.credence:.2f}  "
            f"({a.n_succeeded}/{a.n_applied} succeeded, "
            f"{a.n_partial} partial, {a.n_failed} failed)"
        )
        lines.append(f"    Name:           {s.name}")
        if is_dialogic(s):
            lines.append(
                "    SOURCE:         HUMAN-AUTHORED via dialogue — "
                "expert advice, NOT yet verified by the system's own "
                "play.  Treat the steps as a strong prior; confirm by "
                "the outcome and let the result move its credence."
            )
        lines.append(f"    What it does:   {s.description}")
        lines.append(f"    Problem solved: {s.problem_solved}")
        lines.append(f"    Expected outcome: {s.expected_outcome}")
        if s.relational_steps:
            lines.append(
                "    Generalized steps (game-agnostic; roles + relations, "
                "independent of piece colour/count):"
            )
            for i, st in enumerate(s.relational_steps, 1):
                lines.append(f"      {i}. {st}")
        if s.concrete_chain:
            lines.append(
                f"    Concrete chain (first observed):  "
                f"{', '.join(s.concrete_chain)}"
            )
        if s.parent_id:
            lines.append(
                f"    Fork of {s.parent_id!r}: {s.variant_notes}"
            )
        if s.provenance:
            recent = s.provenance[-1]
            lines.append(
                f"    Last applied: {recent.game_id} lc={recent.level} "
                f"t{recent.turn_range[0]}-{recent.turn_range[1]}, "
                f"outcome={recent.outcome}"
            )
    lines.append("")
    lines.append(
        "  HOW TO USE: in your strategy reply, set "
        "`applied_subroutine` to a subroutine_id from this list "
        "when you're applying it (with whatever adaptation you "
        "describe in `rationale`).  If you're creating a new "
        "variant, set `fork_parent` to the parent id and "
        "`variant_notes` to a one-line description of how this "
        "variant differs.  Otherwise leave both fields null."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience: build a Subroutine from an explicit dict.  Used by the
# auto-promotion VLM call when its reply is parsed.
# ---------------------------------------------------------------------------


def build_subroutine_from_vlm_reply(
    reply: dict,
    *,
    game_id: str,
    level: int,
    turn_range: list[int],
    original_goal: str,
    concrete_chain: list[str],
) -> Subroutine:
    """Turn the VLM's auto-promotion reply into a Subroutine.

    Expected reply schema (free-form except for these slots):
      {
        "name":             str,
        "description":      str,
        "problem_solved":   str,
        "expected_outcome": str,
        "parent_id":        Optional[str],
        "variant_notes":    Optional[str],
      }
    """
    return promote_chain_as_subroutine(
        name=str(reply.get("name", "(unnamed)")),
        description=str(reply.get("description", "")),
        problem_solved=str(reply.get("problem_solved", "")),
        concrete_chain=list(concrete_chain),
        expected_outcome=str(reply.get("expected_outcome", "")),
        game_id=game_id,
        level=level,
        turn_range=list(turn_range),
        original_goal=original_goal,
        parent_id=reply.get("parent_id"),
        variant_notes=str(reply.get("variant_notes", "") or ""),
    )
