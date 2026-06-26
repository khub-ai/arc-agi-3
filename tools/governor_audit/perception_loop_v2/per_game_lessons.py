"""Per-game lessons — the WITHIN-game knowledge that persists
across trials of the SAME game, keyed by ``game_id``.

Distinct from existing cross-trial stores:
  - subroutine_kb:   cross-GAME tactics (reusable in any game)
  - global_priors:   cross-GAME action-effect aggregator
  - physics_priors:  cross-GAME real-world-physics seeds
  - PER-GAME LESSONS (this module): within-GAME knowledge that
    DOES NOT generalize across games — sk48's "ACTION4 pushes
    blocks rightward" is about sk48's specific mechanics, not a
    universal property of ACTION4.

Designed for COMPETITION-MODE INTERLEAVING: when a session plays
game1.lc0, then game2.lc0, then game1.lc1, the driver loads
game1's lessons fresh on the return visit — they were persisted
when game1.lc0 closed, untouched while game2 ran.

Schema:
  {
    "games": {
      "<game_id>": {
        "game_id":          str,
        "last_updated_iso": iso8601,
        "n_trials_contributing": int,
        "trial_provenance": [
          {trial_id, level, score_reached, turns_played,
           outcome, distilled_at_iso},
          ...
        ],
        "lessons": [
          {
            "lesson_id":            str,
            "kind":                 'mechanic' | 'blocking' |
                                    'win_condition' | 'refuted' |
                                    'free_form',
            "description":          str,
              # free-form actor text
            "credence":             float [0..1],
            "n_trials_supporting":  int,
            "n_trials_contradicting": int,
            "first_observed_trial": str,
            "last_observed_trial":  str,
            "promoted":             bool,
              # >= promotion threshold
            "source_hypothesis_id": str | null,
              # if auto-distilled from a WorldKnowledge hyp, the
              # original id for provenance
            "notes":                str,
          },
          ...
        ]
      },
      ...
    },
    "schema_version": 1,
    "last_updated_iso": iso8601
  }

Operations:
  load_for_game(game_id) -> list[Lesson]
    Lessons persisted for this game (empty if first run).
  format_lessons_surface(world) -> str
    Strategy-prompt block surfaced at trial start.
  commit_lesson_from_actor(game_id, ...) -> Lesson
    Actor authors a new free-form lesson (typically at trial
    close via end_of_trial_reply).
  update_lesson_credence(game_id, lesson_id, kind) -> Lesson
    Substrate bumps credence on revisit.
  auto_distill_from_world(world, trial_id, outcome) -> list[Lesson]
    Walk world.mechanic_hypotheses + blocking_claims +
    win_condition_hypotheses for promoted ones, convert to
    candidate lessons, merge into persisted store.
  end_of_trial_summary(world, trial_id, outcome) -> dict
    Aggregate everything (distilled + actor-authored) into a
    single persisted update.

GAME-AGNOSTIC: all fields are open-vocabulary; substrate just
persists and ranks.  No game-specific code anywhere.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from world_knowledge import WorldKnowledge


try:
    from kb_paths import kb_path as _kb_path
except ImportError:  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path
# Unified KB root (see kb_paths.py + docs/SPEC_knowledge_base.md); legacy flat
# .tmp path is auto-migrated in and kept as a fallback.
DEFAULT_LESSONS_PATH = _kb_path("per_game_lessons.json")

_SCHEMA_VERSION         = 1
_PROMOTION_CREDENCE     = 0.80
_DEFAULT_CREDENCE       = 0.60   # actor-authored lessons
_DISTILLED_CREDENCE     = 0.70   # auto-distilled from promoted
                                 # hypotheses (slight bias because
                                 # they already crossed the
                                 # within-trial credence threshold)
_SUPPORT_BUMP           = 0.08
_CONTRADICT_DECAY       = 0.20


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class Lesson:
    """A single piece of within-game knowledge."""
    lesson_id: str
    kind: str
        # 'mechanic' | 'blocking' | 'win_condition' |
        # 'refuted' | 'free_form'
    description: str
    credence: float = _DEFAULT_CREDENCE
    n_trials_supporting: int = 1
    n_trials_contradicting: int = 0
    first_observed_trial: str = ""
    last_observed_trial: str = ""
    promoted: bool = False
    source_hypothesis_id: Optional[str] = None
    notes: str = ""
    # Per-LEVEL verification ledger for mechanic lessons.  A mechanic
    # rule learned on level N is ASSUMED to hold on level N+1 (strong
    # same-game prior), but that assumption is UNVERIFIED until a probe
    # confirms it ON that level.  These lists record where the rule was
    # re-confirmed vs where it was contradicted, so the confirm/deny is
    # never re-paid and a level that genuinely changed a mechanic is
    # remembered.  See mechanic_stability.py.
    verified_in_levels: list = field(default_factory=list)
    contradicted_in_levels: list = field(default_factory=list)

    def __post_init__(self):
        if self.credence >= _PROMOTION_CREDENCE:
            self.promoted = True


@dataclass
class TrialProvenance:
    trial_id: str
    level: Optional[int] = None
    score_reached: Optional[int] = None
    turns_played: int = 0
    outcome: str = ""        # 'lc_advance' | 'lc_fail' |
                              # 'max_turns' | 'manual_close' | 'unknown'
    distilled_at_iso: str = ""


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(description: str, kind: str) -> str:
    safe = "".join(
        c if c.isalnum() or c == "_" else "_"
        for c in description.lower()
    )[:40]
    return f"lesson_{kind}_{safe}_{int(time.time()) % 100000}"


def _load_blob(path: Path) -> dict:
    if not path.exists():
        return {"games": {}, "schema_version": _SCHEMA_VERSION,
                "last_updated_iso": _now_iso()}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"games": {}, "schema_version": _SCHEMA_VERSION,
                "last_updated_iso": _now_iso()}


def _save_blob(blob: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    blob["last_updated_iso"] = _now_iso()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    tmp.replace(path)


def _game_record(blob: dict, game_id: str) -> dict:
    games = blob.setdefault("games", {})
    rec = games.setdefault(game_id, {})
    rec.setdefault("game_id", game_id)
    rec.setdefault("last_updated_iso", _now_iso())
    rec.setdefault("n_trials_contributing", 0)
    rec.setdefault("trial_provenance", [])
    rec.setdefault("lessons", [])
    return rec


# ---------------------------------------------------------------------------
# Load / surface
# ---------------------------------------------------------------------------


def load_for_game(
    game_id: str, *, path: Path = DEFAULT_LESSONS_PATH,
) -> list[Lesson]:
    """Return all persisted lessons for this game (empty if
    first-ever run)."""
    blob = _load_blob(path)
    rec = (blob.get("games") or {}).get(game_id) or {}
    lessons_raw = rec.get("lessons") or []
    out: list[Lesson] = []
    for d in lessons_raw:
        try:
            out.append(Lesson(**d))
        except TypeError:
            # forward-compat: skip records with unknown fields
            continue
    return out


def load_all_games(
    *, path: Path = DEFAULT_LESSONS_PATH, min_credence: float = 0.0,
    exclude_game: Optional[str] = None,
) -> list[tuple[str, Lesson]]:
    """Every persisted lesson across ALL games as (game_id, Lesson) pairs, for
    CROSS-GAME recall — so the actor can be guided by a relevant idea learned in
    another game, not only this game's own lessons. Filtered by credence;
    optionally excludes one game (typically the current one, loaded separately)."""
    blob = _load_blob(path)
    out: list[tuple[str, Lesson]] = []
    for gid, rec in (blob.get("games") or {}).items():
        if exclude_game and gid == exclude_game:
            continue
        for d in (rec.get("lessons") or []):
            try:
                lesson = Lesson(**d)
            except TypeError:
                continue
            if lesson.credence >= min_credence:
                out.append((gid, lesson))
    return out


def rank_lessons(lessons: list[Lesson]) -> list[Lesson]:
    """Sort by credence desc, then by n_trials_supporting desc."""
    return sorted(
        lessons,
        key=lambda l: (-l.credence, -l.n_trials_supporting),
    )


import re as _re

_ACTION_RE = _re.compile(r"ACTION\s*\d+", _re.IGNORECASE)


def _parse_mechanic_rule(description: str) -> tuple:
    """Extract (action, effect) from a mechanic lesson's free-form
    description.  Handles both the auto-distilled shape
    ('trigger=action=ACTION4 -> effect=effect=entity_changed, ...') and
    plain prose ('ACTION4 pushes blocks rightward').  Returns
    (None, '') when no action token is present (not an action-keyed
    mechanic, so not a stability claim)."""
    if not description:
        return (None, "")
    m = _ACTION_RE.search(description)
    if not m:
        return (None, "")
    action = m.group(0).upper().replace(" ", "")
    # Effect = the part after '->' if present, else the description with
    # the action token and bookkeeping prefixes stripped.
    if "->" in description:
        effect = description.split("->", 1)[1]
    else:
        effect = description[:m.start()] + description[m.end():]
    effect = (effect.replace("effect=", "").replace("trigger=", "")
              .replace("action=", "").strip(" =,;:-"))
    return (action, effect)


def mechanic_rules_for_stability(
    game_id: str, *, path: Path = DEFAULT_LESSONS_PATH,
) -> list:
    """Promoted, action-keyed mechanic lessons for THIS game, in the
    dict shape ``mechanic_stability.compute_stability_claims`` expects:
        {action, effect, credence, supporting_levels,
         verified_in_levels, lesson_id}

    Only mechanic lessons whose description names an ACTION are returned
    (those are the ones whose semantics a plan can DEPEND on).  Used at
    level start to build the carried-belief stability surface."""
    out: list = []
    for l in load_for_game(game_id, path=path):
        if l.kind != "mechanic" or not l.promoted:
            continue
        action, effect = _parse_mechanic_rule(l.description)
        if action is None:
            continue
        out.append({
            "action": action,
            "effect": effect,
            "credence": l.credence,
            "supporting_levels": list(l.verified_in_levels or []),
            "verified_in_levels": list(l.verified_in_levels or []),
            "lesson_id": l.lesson_id,
        })
    return out


def record_level_verification(
    game_id: str, lesson_id: str, level: int, *, confirmed: bool,
    path: Path = DEFAULT_LESSONS_PATH,
) -> Optional[Lesson]:
    """Persist that a mechanic lesson was CONFIRMED or CONTRADICTED on a
    specific level — the durable result of a level-start verification
    probe, so it is never re-paid and a genuine mechanic change is
    remembered.

    A confirmation appends ``level`` to ``verified_in_levels`` (and
    removes it from ``contradicted_in_levels`` if present).  A
    contradiction does the inverse AND decays credence by
    ``_CONTRADICT_DECAY`` — a level that changed a mechanic is strong
    evidence the rule is not as universal as believed.  Returns the
    updated Lesson, or None if not found.
    """
    blob = _load_blob(path)
    rec = _game_record(blob, game_id)
    target = None
    for d in rec.get("lessons") or []:
        if d.get("lesson_id") == lesson_id:
            target = d
            break
    if target is None:
        return None
    vset = set(target.get("verified_in_levels") or [])
    cset = set(target.get("contradicted_in_levels") or [])
    if confirmed:
        vset.add(level)
        cset.discard(level)
    else:
        cset.add(level)
        vset.discard(level)
        target["credence"] = max(
            0.0, float(target.get("credence", 0.0)) - _CONTRADICT_DECAY)
        target["promoted"] = target["credence"] >= _PROMOTION_CREDENCE
    target["verified_in_levels"] = sorted(vset)
    target["contradicted_in_levels"] = sorted(cset)
    _save_blob(blob, path)
    try:
        return Lesson(**target)
    except TypeError:
        return None


def format_lessons_surface_slim(
    world: WorldKnowledge,
    *,
    path: Path = DEFAULT_LESSONS_PATH,
    max_mechanic_lessons: int = 8,
) -> str:
    """Slim variant for the trimmed strategy prompt.

    Surfaces ONLY the highest-signal prior knowledge:
      * promoted mechanic lessons (action -> effect rules established
        across prior trials of the SAME game)
      * win-condition lessons that have reached promoted credence

    Surfaces, in addition, a DO-NOT-REPEAT avoid-list of REFUTED
    alternatives.  Negative evidence is surfaced AGGRESSIVELY (no
    promotion threshold): a single clean refutation is enough to warn
    the actor off repeating a demonstrated failure, because the cost
    of repeating a known-bad action is high and the bar for "stop
    doing what didn't work" should be LOWER than the bar for "trust
    what did work".  This closes the self-improvement loop: the actor
    forms NEW claims instead of re-running refuted ones.

    Skips: free-form sprawl and blocking claims (already in
    snapshot's mechanic_hypotheses).  Empty surface returned when no
    qualifying lessons exist so the prompt doesn't grow uselessly on
    first-trial play.
    """
    lessons = load_for_game(world.game_id, path=path)
    if not lessons:
        return ""

    promoted_mechanic = [
        l for l in lessons
        if l.kind == "mechanic" and l.promoted
    ]
    promoted_wc = [
        l for l in lessons
        if l.kind == "win_condition" and l.promoted
    ]
    # Refuted alternatives — surfaced with NO promotion gate (negative
    # evidence matters even from a single trial).  Sorted worst-first
    # (most contradicted) so the clearest failures lead.
    refuted = [l for l in lessons if l.kind == "refuted"]
    refuted = sorted(
        refuted,
        key=lambda l: (-(l.n_trials_contradicting or 0),
                       -(l.n_trials_supporting or 0)),
    )[:8]
    # Curated onboarding channels: a one-shot game SUMMARY (read first)
    # and reusable TECHNIQUEs (how-to maneuvers).  These are the
    # crystallized, promoted knowledge a future actor needs to solve the
    # game (and its sibling levels) without rediscovering — see
    # SPEC_cumulative_learning_loop.md § Crystallization.  Kept out of the
    # free-form sprawl precisely so they surface.
    promoted_summary = [
        l for l in lessons if l.kind == "summary" and l.promoted
    ]
    promoted_technique = [
        l for l in lessons if l.kind == "technique" and l.promoted
    ]
    # Ineffective actions: those observed to produce NO effect (globally,
    # across all probed state-classes).  Surfaced as an explicit
    # DO-NOT-PROPOSE list — a vocab fact buried in a mechanic ("CLICK is
    # silent") is not actionable enough; an actor kept trying CLICK
    # despite it.  Promotion-gated like other established knowledge.
    promoted_inert = [
        l for l in lessons if l.kind == "inert_action" and l.promoted
    ]
    if not (promoted_mechanic or promoted_wc or refuted
            or promoted_summary or promoted_technique or promoted_inert):
        return ""

    promoted_mechanic = rank_lessons(promoted_mechanic)[:max_mechanic_lessons]
    promoted_wc = rank_lessons(promoted_wc)[:3]
    promoted_summary = rank_lessons(promoted_summary)[:1]
    promoted_technique = rank_lessons(promoted_technique)[:4]

    n_trials = len(set(
        l.first_observed_trial for l in lessons if l.first_observed_trial
    ))
    lines: list[str] = [""]
    lines.append(
        f"PRIOR-TRIAL LESSONS for game {world.game_id!r} "
        f"(promoted lessons + a DO-NOT-REPEAT avoid-list; "
        f"observation-validated across {n_trials} prior trial(s)):"
    )
    if promoted_summary:
        lines.append("  GAME SUMMARY (read first — how this game works "
                     "and how to win it):")
        for l in promoted_summary:
            lines.append(f"    {l.description}")
    if promoted_inert:
        lines.append("  INEFFECTIVE ACTIONS — observed to do NOTHING in "
                     "this game; DO NOT PROPOSE them (it wastes a turn):")
        for l in promoted_inert:
            lines.append(f"    {l.description}")
    if promoted_technique:
        lines.append("  TECHNIQUES established by prior trials "
                     "(reusable maneuvers):")
        for l in promoted_technique:
            lines.append(
                f"    {l.description}  "
                f"(+{l.n_trials_supporting}/-{l.n_trials_contradicting})"
            )
    if promoted_mechanic:
        lines.append("  Action mechanics established by prior trials:")
        for l in promoted_mechanic:
            lines.append(
                f"    {l.description}  "
                f"(+{l.n_trials_supporting}/-{l.n_trials_contradicting})"
            )
    if promoted_wc:
        lines.append("  Win-condition hypotheses established by prior trials:")
        for l in promoted_wc:
            lines.append(
                f"    {l.description}  "
                f"(+{l.n_trials_supporting}/-{l.n_trials_contradicting})"
            )
    if refuted:
        lines.append(
            "  DO NOT REPEAT — these were TRIED and FAILED in prior "
            "trials (avoid-list).  If your intended action matches one "
            "of these, do NOT execute it as-is: form a NEW purpose / "
            "strategic claim that differs in a specified way, and say "
            "in your rationale how it differs from the failed attempt:"
        )
        for l in refuted:
            lines.append(
                f"    [FAILED x{l.n_trials_contradicting or 1}] "
                f"{l.description}"
            )
    if (promoted_mechanic or promoted_wc or promoted_summary
            or promoted_technique or promoted_inert):
        lines.append(
            "  Treat the above established lessons as STRONG PRIORS but "
            "validate early; report which you LEAN ON in your rationale."
        )
    return "\n".join(lines) + "\n"


def format_lessons_surface(
    world: WorldKnowledge,
    *,
    path: Path = DEFAULT_LESSONS_PATH,
    max_per_kind: int = 5,
) -> str:
    """Render a strategy-prompt block summarizing what's known
    about THIS game from prior trials.  Skips kinds with no
    entries to keep noise low when the game is new."""
    lessons = load_for_game(world.game_id, path=path)
    if not lessons:
        return ("  (no prior-trial lessons for this game yet.  "
                "On first trial, the actor's job is to discover; "
                "at trial close, lessons will be persisted here "
                "for future trials of the same game.)")

    # Group by kind
    by_kind: dict[str, list[Lesson]] = {}
    for l in lessons:
        by_kind.setdefault(l.kind, []).append(l)

    lines: list[str] = []
    n_total = len(lessons)
    n_promoted = sum(1 for l in lessons if l.promoted)
    lines.append(
        f"  {n_total} prior-trial lesson(s) for game "
        f"{world.game_id!r}, {n_promoted} promoted (credence "
        f">= {_PROMOTION_CREDENCE:.2f}).  These are within-game "
        "claims established by prior trials.  Treat them as "
        "high-prior starting hypotheses BUT VALIDATE EARLY in "
        "this trial — credence may not transfer perfectly across "
        "levels.  Refuted lessons are listed so you don't waste "
        "turns re-testing dead ends."
    )

    kind_labels = [
        ("mechanic",      "MECHANICS — within-game action/effect rules"),
        ("blocking",      "BLOCKING CONSTRAINTS — known-silent (action, state-class) pairs"),
        ("win_condition", "WIN-CONDITION HYPOTHESES — what triggers score / lc / win"),
        ("free_form",     "FREE-FORM LESSONS — actor-authored at prior trial close"),
        ("refuted",       "REFUTED — alternatives ruled out; don't re-test"),
    ]
    for kind_key, label in kind_labels:
        bucket = rank_lessons(by_kind.get(kind_key) or [])
        if not bucket:
            continue
        lines.append("")
        lines.append(f"  {label}")
        for l in bucket[:max_per_kind]:
            promo = "PROMOTED" if l.promoted else f"c={l.credence:.2f}"
            lines.append(
                f"    [{promo}] +{l.n_trials_supporting}"
                f"/-{l.n_trials_contradicting}  "
                f"{l.description}"
            )
            if l.notes:
                lines.append(f"      notes: {l.notes}")
        if len(bucket) > max_per_kind:
            lines.append(
                f"    ... ({len(bucket) - max_per_kind} more)"
            )

    lines.append("")
    lines.append(
        "  HOW TO USE: read these as STRONG PRIORS for this game.  "
        "Probe each promoted lesson early to confirm it still "
        "holds in this trial; if confirmed, skip re-exploring it "
        "and proceed directly to building on it.  If a promoted "
        "lesson is contradicted, commit a "
        "win_condition_observation / probe_observation that "
        "flags the contradiction — the substrate will decay its "
        "credence at trial close."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Author / merge
# ---------------------------------------------------------------------------


def commit_lesson_from_actor(
    *,
    game_id: str,
    kind: str,
    description: str,
    notes: str = "",
    credence: Optional[float] = None,
    trial_id: str = "",
    source_hypothesis_id: Optional[str] = None,
    general: bool = False,
    path: Path = DEFAULT_LESSONS_PATH,
) -> Lesson:
    """Add a new actor-authored lesson to the persisted store.

    Set ``general=True`` only when the AUTHOR judges the lesson genuinely
    game-agnostic (a technique/process insight, no game-specific entities,
    actions, colours or win conditions); it is then also re-homed into the
    always-surfaced general_knowledge store -- but a text safety-net still
    vetoes anything carrying a game-specific token, so a mislabel can't leak."""
    blob = _load_blob(path)
    game_rec = _game_record(blob, game_id)
    if credence is None:
        credence = _DEFAULT_CREDENCE
    credence = max(0.0, min(1.0, float(credence)))
    new = Lesson(
        lesson_id=_new_id(description, kind),
        kind=kind,
        description=description,
        credence=credence,
        first_observed_trial=trial_id,
        last_observed_trial=trial_id,
        source_hypothesis_id=source_hypothesis_id,
        notes=notes,
    )
    # De-dup against existing identical descriptions
    existing = [l for l in game_rec["lessons"]
                if l.get("description", "").strip().lower() ==
                description.strip().lower()
                and l.get("kind") == kind]
    if existing:
        # Bump credence on revisit instead of duplicating
        rec = existing[0]
        rec["n_trials_supporting"] = int(rec.get(
            "n_trials_supporting", 0
        )) + 1
        rec["credence"] = min(
            1.0,
            float(rec.get("credence", _DEFAULT_CREDENCE))
            + _SUPPORT_BUMP,
        )
        rec["last_observed_trial"] = trial_id or rec.get(
            "last_observed_trial", ""
        )
        if rec["credence"] >= _PROMOTION_CREDENCE:
            rec["promoted"] = True
        if notes:
            rec["notes"] = (
                (rec.get("notes") + "\n") if rec.get("notes")
                else ""
            ) + notes
        out_lesson = Lesson(**rec)
    else:
        game_rec["lessons"].append(asdict(new))
        out_lesson = new
    game_rec["last_updated_iso"] = _now_iso()
    _save_blob(blob, path)
    # Re-home into the always-surfaced game-agnostic store ONLY when the author
    # marked this lesson general (explicit judgement) -- the text safety-net then
    # still vetoes a mislabel.  Guarded; never blocks the commit.
    if general:
        try:
            try:
                import general_knowledge as _gk
            except ImportError:
                from perception_loop_v2 import general_knowledge as _gk
            _gk.consider_promote(out_lesson.description, kind=out_lesson.kind,
                                 credence=out_lesson.credence, from_game=game_id,
                                 lesson_id=out_lesson.lesson_id)
        except Exception:
            pass
    return out_lesson


def contradict_lesson(
    *, game_id: str, lesson_id: str,
    trial_id: str = "", note: str = "",
    path: Path = DEFAULT_LESSONS_PATH,
) -> Optional[Lesson]:
    """Decay credence on a lesson when this trial observed a
    contradiction."""
    blob = _load_blob(path)
    rec = (blob.get("games") or {}).get(game_id)
    if not rec:
        return None
    for l in rec.get("lessons") or []:
        if l.get("lesson_id") == lesson_id:
            l["n_trials_contradicting"] = int(l.get(
                "n_trials_contradicting", 0
            )) + 1
            l["credence"] = max(
                0.0,
                float(l.get("credence", _DEFAULT_CREDENCE))
                - _CONTRADICT_DECAY,
            )
            l["last_observed_trial"] = trial_id or l.get(
                "last_observed_trial", ""
            )
            if l["credence"] < _PROMOTION_CREDENCE:
                l["promoted"] = False
            if note:
                l["notes"] = (
                    (l.get("notes") + "\n") if l.get("notes")
                    else ""
                ) + f"contradicted at {trial_id or 'unknown'}: {note}"
            _save_blob(blob, path)
            return Lesson(**l)
    return None


# ---------------------------------------------------------------------------
# Auto-distill at trial close
# ---------------------------------------------------------------------------


def auto_distill_from_world(
    world: WorldKnowledge,
    *,
    trial_id: str,
    outcome: str = "manual_close",
    path: Path = DEFAULT_LESSONS_PATH,
) -> list[Lesson]:
    """Walk the world's promoted MechanicHypotheses + BlockingClaims
    + WinConditionHypotheses, convert each to a candidate lesson,
    and merge into the persisted store.  Returns the lessons
    written (or revisited)."""
    written: list[Lesson] = []
    blob = _load_blob(path)
    game_rec = _game_record(blob, world.game_id)

    def _merge(kind: str, description: str, src_id: Optional[str],
               credence: float, notes: str = ""):
        nonlocal written
        existing = [
            l for l in game_rec["lessons"]
            if l.get("description", "").strip().lower() ==
                description.strip().lower()
            and l.get("kind") == kind
        ]
        if existing:
            rec = existing[0]
            rec["n_trials_supporting"] = int(rec.get(
                "n_trials_supporting", 0
            )) + 1
            rec["credence"] = min(
                1.0,
                float(rec.get("credence", credence))
                + _SUPPORT_BUMP,
            )
            rec["last_observed_trial"] = trial_id
            if rec["credence"] >= _PROMOTION_CREDENCE:
                rec["promoted"] = True
            written.append(Lesson(**rec))
        else:
            new = Lesson(
                lesson_id=_new_id(description, kind),
                kind=kind,
                description=description,
                credence=max(0.0, min(1.0, credence)),
                first_observed_trial=trial_id,
                last_observed_trial=trial_id,
                source_hypothesis_id=src_id,
                notes=notes,
            )
            game_rec["lessons"].append(asdict(new))
            written.append(new)

    # Mechanic hypotheses
    for h in (world.mechanic_hypotheses or []):
        if not getattr(h, "promoted", False):
            continue
        desc = f"trigger={h.trigger} -> effect={h.effect}"
        _merge(
            "mechanic", desc, h.hypothesis_id,
            credence=_DISTILLED_CREDENCE,
        )

    # Blocking claims
    for b in (world.blocking_claims or []):
        if not getattr(b, "promoted", False):
            continue
        state_str = ",".join(
            f"{k}={v}" for k, v in
            sorted((b.blocking_state or {}).items())
        )
        desc = f"action={b.blocked_action} is silent in state-class [{state_str}]"
        _merge(
            "blocking", desc, b.claim_id,
            credence=_DISTILLED_CREDENCE,
        )

    # Win-condition hypotheses
    for w in (world.win_condition_hypotheses or []):
        if w.credence >= _PROMOTION_CREDENCE:
            kind = "win_condition"
        elif w.credence <= 0.05 and w.contradicting_observations:
            kind = "refuted"
        else:
            continue
        _merge(
            kind, w.description, w.hypothesis_id,
            credence=w.credence,
            notes=(f"+{len(w.supporting_observations)} support / "
                    f"-{len(w.contradicting_observations)} contradict"),
        )

    # Refuted approaches mined from RELATIONAL no-ops (e.g. "raising a
    # manipulator that spans an obstacle does nothing"). These feed the
    # existing refuted-lesson gate so the substrate -- not the VLM's memory --
    # carries 'do not retry X in this situation'. Treated as hypotheses
    # (lower credence) until re-observed, per the mined-rules discipline.
    try:
        from playback_mining import mine_refuted_approaches  # noqa: E402
        for r in mine_refuted_approaches(world):
            _merge("refuted", r["description"], None,
                   credence=min(0.4 + 0.1 * (r["support"] - 2), _DISTILLED_CREDENCE),
                   notes=f"auto-mined no-op; support={r['support']}")
    except Exception:
        pass

    # Record trial provenance
    prov = {
        "trial_id": trial_id,
        "level": world.level,
        "score_reached": world.score,
        "turns_played": world.turn,
        "outcome": outcome,
        "distilled_at_iso": _now_iso(),
    }
    # de-dup by trial_id
    if not any(
        p.get("trial_id") == trial_id
        for p in game_rec["trial_provenance"]
    ):
        game_rec["trial_provenance"].append(prov)
        game_rec["n_trials_contributing"] = int(
            game_rec.get("n_trials_contributing", 0)
        ) + 1
    game_rec["last_updated_iso"] = _now_iso()
    _save_blob(blob, path)
    return written


# ---------------------------------------------------------------------------
# End-of-trial author hook
# ---------------------------------------------------------------------------


_END_OF_TRIAL_PROMPT = """\
# End-of-trial lessons authoring — game {game_id} trial {trial_id}

This trial just ended (outcome={outcome}, score reached={score},
turns played={turns}).  Authoring lessons NOW is how knowledge
accumulates for future trials of THIS GAME.  In competition
mode, levels from many games interleave — so anything you write
here is what your future self will see first when the substrate
re-encounters {game_id} later in the session.

## What's already been auto-distilled

{auto_distilled}

## What you actually did this level (REVIEW the playback)

{playback_review}

A frame of the END state is attached.  If the level was NOT completed,
this is the most valuable moment to learn: REVIEW the playback above
and the end frame, and work out WHY it failed and HOW to do better next
time -- e.g. "I spent most of my moves shuttling pieces around and ran
out before finishing; next time merge pairs that are ALREADY close
first and never move a piece more than needed", or "I kept re-doing a
move that never changed anything".  Be concrete and ACTIONABLE: your
future self gets `corrective_strategy` first when it replays this game,
so it must say what to DO differently, not just what went wrong.

## Your task

Write a small JSON object summarizing the trial.  Substrate
persists what you write at the per-game key.  Schema:

{{
  "free_form_lessons": [
    "<short prose lesson — a within-game claim that wasn't already
     captured by the auto-distilled mechanic/blocking/
     win-condition records.  Things like: 'the win condition is X
     (cred 0.85) — column-alignment hypothesis is REFUTED'; or
     'side-swipe pattern works when ...'; or 'ACTION_N has alias
     ACTION_M; do not re-discover'.  One lesson per array entry.
     Free-form text; substrate just stores it.>",
    ...
  ],
  "refuted_alternatives": [
    "<a hypothesis the actor explored this trial and falsified;
     persist so the next trial of this game doesn't re-test.>",
    ...
  ],
  "trial_notes": "<one-paragraph summary of what worked and what
                   didn't.  Used in the trace and surfaced at next
                   trial start.>",
  "game_summary": "<one or two sentences: WHAT THIS GAME IS — genre,
                   the core objects, and the central mechanic.  e.g.
                   'a manipulator puzzle: a horizontal arm pierces
                   colored blocks; the bottom strip is a live
                   per-block indicator.'  Stored in the CROSS-GAME
                   knowledge store and matched to future games by
                   structure, so a VARIANT of this game can reuse it.>",
  "strategic_approach": "<one or two sentences: the BEST STRATEGIC
                   APPROACH discovered — how to actually make progress.
                   e.g. 'pierce each block at its home row; full-retract
                   before repositioning to avoid pushing the column.'
                   Also cross-game; keep it transferable.>",
  "failure_diagnosis": "<if the level was NOT completed: one sentence on
                   the ROOT CAUSE of the failure, grounded in the playback
                   (e.g. 'ran out of the level's move budget because most
                   moves were spent walking scattered pieces together').
                   Empty string if the level was completed.>",
  "corrective_strategy": "<if it failed: ONE concrete, actionable change
                   to apply on the NEXT attempt to avoid that failure
                   (e.g. 'merge the closest same-colour pair first to
                   minimise walking, and deliver a finished piece the
                   moment it is made'). This is surfaced FIRST next time,
                   so make it directly usable. Empty string otherwise.>"
}}

If you have nothing additional to add, return an empty object:
{{"free_form_lessons": [], "refuted_alternatives": [],
  "trial_notes": ""}}.

Write the JSON to:
  `{reply_name}`

Plain JSON, no markdown fences, no prose.
"""


def stage_end_of_trial_prompt(
    world: WorldKnowledge,
    work_dir: Path,
    *,
    trial_id: str,
    outcome: str = "manual_close",
    auto_distilled: Optional[list[Lesson]] = None,
    playback_review: str = "",
) -> tuple[Path, Path]:
    """Write the end-of-trial authoring prompt + a status file.
    Returns (prompt_path, reply_path).  Driver polls for the
    reply same as other VLM-as-Claude calls."""
    turn_dir = work_dir / f"turn_end_of_trial"
    turn_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = turn_dir / "end_of_trial_prompt.md"
    reply_path  = turn_dir / "end_of_trial_reply.txt"

    auto_lines: list[str] = []
    if auto_distilled:
        for l in auto_distilled[:15]:
            auto_lines.append(
                f"  - [{l.kind}] {l.description} "
                f"(credence {l.credence:.2f})"
            )
    auto_text = "\n".join(auto_lines) if auto_lines else (
        "  (nothing auto-distilled — no promoted mechanic / "
        "blocking / win-condition hypotheses at trial close)"
    )

    text = _END_OF_TRIAL_PROMPT.format(
        game_id=world.game_id,
        trial_id=trial_id,
        outcome=outcome,
        score=world.score,
        turns=world.turn,
        auto_distilled=auto_text,
        playback_review=(playback_review or "(no playback summary available)"),
        reply_name=reply_path.name,
    )
    prompt_path.write_text(text, encoding="utf-8")
    return prompt_path, reply_path


def apply_end_of_trial_reply(
    world: WorldKnowledge,
    reply_json: dict,
    *,
    trial_id: str,
    path: Path = DEFAULT_LESSONS_PATH,
) -> list[Lesson]:
    """Persist actor-authored lessons from the end-of-trial reply.
    Returns the list of lessons committed."""
    written: list[Lesson] = []
    free_form = reply_json.get("free_form_lessons") or []
    refuted = reply_json.get("refuted_alternatives") or []
    trial_notes = str(reply_json.get("trial_notes") or "")

    for desc in free_form:
        desc = str(desc or "").strip()
        if not desc:
            continue
        l = commit_lesson_from_actor(
            game_id=world.game_id,
            kind="free_form",
            description=desc,
            trial_id=trial_id,
            path=path,
        )
        written.append(l)
    for desc in refuted:
        desc = str(desc or "").strip()
        if not desc:
            continue
        l = commit_lesson_from_actor(
            game_id=world.game_id,
            kind="refuted",
            description=desc,
            trial_id=trial_id,
            path=path,
        )
        written.append(l)

    # CORRECTIVE STRATEGY from a failure review -> a high-credence lesson so
    # kb_recall surfaces it FIRST on the next attempt of this game (the
    # learn-from-the-loss loop; the actionable output of the playback review).
    corrective = str(reply_json.get("corrective_strategy") or "").strip()
    diagnosis = str(reply_json.get("failure_diagnosis") or "").strip()
    if corrective:
        desc = (f"CORRECTIVE (last attempt failed: {diagnosis}) -> {corrective}"
                if diagnosis else f"CORRECTIVE: {corrective}")
        written.append(commit_lesson_from_actor(
            game_id=world.game_id, kind="corrective", description=desc,
            credence=0.85, trial_id=trial_id, path=path,
        ))

    if trial_notes:
        # Attach trial notes to the most recent provenance entry
        blob = _load_blob(path)
        rec = (blob.get("games") or {}).get(world.game_id) or {}
        prov = rec.get("trial_provenance") or []
        for p in reversed(prov):
            if p.get("trial_id") == trial_id:
                p["actor_notes"] = trial_notes
                break
        _save_blob(blob, path)
    return written
