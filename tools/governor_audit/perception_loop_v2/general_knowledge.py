"""Game-agnostic general knowledge store.

The cross-game recall gate (cross_game_knowledge.admits_xgame) admits a
game-keyed lesson only for a structural VARIANT of its origin game. That is
correct for game-specific knowledge, but it would also stop a GENUINELY GENERAL
lesson (one that holds on any game) from transferring to an unseen game. The fix
is to re-home such lessons here: entries in this store carry NO origin game id,
so the gate treats them as game-agnostic and surfaces them everywhere.

This is the compliant home for "general technique" knowledge: the origin game,
if any, is kept only as private provenance (never used for recall or matching),
so the knowledge itself carries no game-specific reference.

Schema (JSON): {"schema_version": 1, "entries": [
    {"id", "text", "kind", "credence", "provenance": {"from_game", "lesson_id",
     "note"}}, ...]}

Promotion is deliberate (a lesson must be judged genuinely general before it is
moved here); this module provides the mechanism, not an automatic migration.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Optional

try:
    from kb_paths import kb_path as _kb_path
except ImportError:  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path

DEFAULT_GENERAL_PATH = _kb_path("general_knowledge.json")
_SCHEMA_VERSION = 1


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def load_general(path: Path = DEFAULT_GENERAL_PATH) -> list:
    """All game-agnostic entries as a list of dicts (empty when absent)."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        blob = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(blob, list):
        return blob
    return blob.get("entries") or []


def save_general(entries: list, path: Path = DEFAULT_GENERAL_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    blob = {"schema_version": _SCHEMA_VERSION, "entries": entries,
            "updated": int(time.time())}
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(p)


def add_general(text: str, *, kind: str = "technique", credence: float = 0.7,
                from_game: str = "", lesson_id: str = "", note: str = "",
                path: Path = DEFAULT_GENERAL_PATH) -> bool:
    """Add one game-agnostic entry. Deduplicates by normalized text (keeps the
    higher credence). Returns True if the store changed. The origin game is
    recorded only as provenance - never used for recall."""
    text = (text or "").strip()
    if not text:
        return False
    entries = load_general(path)
    key = _norm(text)
    for e in entries:
        if _norm(e.get("text", "")) == key:
            if float(credence) > float(e.get("credence", 0)):
                e["credence"] = float(credence)
                save_general(entries, path)
                return True
            return False
    entries.append({
        "id": f"gen_{abs(hash(key)) % 10_000_000}",
        "text": text,
        "kind": kind,
        "credence": float(credence),
        "provenance": {"from_game": from_game, "lesson_id": lesson_id,
                       "note": note},
    })
    save_general(entries, path)
    return True


def promote_from_lesson(game_id: str, lesson_id: str,
                        path: Path = DEFAULT_GENERAL_PATH) -> bool:
    """Copy a per-game lesson into the general store (game stripped; origin kept
    as provenance only). Non-destructive: the per-game lesson is left in place
    (now harmlessly gated). Returns True on success.

    Use ONLY for a lesson judged genuinely general - that judgement is the
    human/VLM step, not this function's."""
    try:
        from per_game_lessons import load_for_game
    except ImportError:
        from perception_loop_v2.per_game_lessons import load_for_game
    for l in load_for_game(game_id):
        if getattr(l, "lesson_id", "") == lesson_id:
            return add_general(
                getattr(l, "description", ""),
                kind=getattr(l, "kind", "technique"),
                credence=float(getattr(l, "credence", 0.7)),
                from_game=game_id, lesson_id=lesson_id,
                note="promoted from per_game_lessons", path=path)
    return False


# ---------------------------------------------------------------------------
# Auto-promotion POLICY (this module was otherwise mechanism-only).
#
# The store stays empty unless something decides a lesson is genuinely general
# and calls add_general/promote_from_lesson.  Nothing did, so the always-surfaced
# store never filled.  These give COS a CONSERVATIVE judgement so a learned
# technique is re-homed automatically — strict on purpose: a false negative just
# leaves a lesson game-gated; a false positive injects game-specific knowledge
# into every unseen game (a compliance leak).
# ---------------------------------------------------------------------------

# Kinds that describe a game's OWN behaviour — never general by nature.
_GAME_SPECIFIC_KINDS = frozenset({
    "mechanic", "blocking", "win_condition", "refuted", "inert_action",
})
# Kinds that CAN be a general technique / process lesson (still token-checked).
_GENERAL_KINDS = frozenset({
    "technique", "strategy", "tactic", "perception", "discipline", "process",
})

_ACTION_TOKEN = re.compile(r"\bACTION\s*\d", re.I)   # ACTION1.. = game action semantics
_HEX_COLOR = re.compile(r"#[0-9a-fA-F]{6}\b")        # a specific sprite colour
_GAMEID_SHAPE = re.compile(r"\b[a-z]{2}\d{2}\b")     # game-id shape: su15 / tn36 / sk48 ...
_PROVENANCE_MARK = re.compile(                       # a specific level/trial/win provenance
    r"\blc\s*=?\s*\d|\bwas won\b|\btrial\b|\bphase\s*\d|score\s*\d\s*-?>", re.I)


def game_id_vocab() -> set:
    """Known game ids (base, lowercased) from the lessons store — the same
    provenance the compliance scanner derives its vocabulary from.  Empty on any
    error (the regex shape check still guards)."""
    try:
        from per_game_lessons import load_all_games
    except ImportError:
        from perception_loop_v2.per_game_lessons import load_all_games
    try:
        return {str(gid).split("-")[0].lower() for gid, _ in load_all_games()}
    except Exception:
        return set()


def is_general_lesson(description: str, kind: str = "", *, vocab=None) -> tuple:
    """Conservatively decide whether a per-game lesson is GENUINELY game-agnostic
    and so safe to surface on every game.  Returns (ok, reason).

    Rejects a game-specific kind, and any text carrying a game-specific token: a
    known game id or a game-id-shaped token, an ACTION<n> action-semantics
    reference, or a specific sprite colour (#rrggbb).  Accepts only a
    technique/strategy/process-style lesson whose text is clean."""
    text = (description or "").strip()
    if not text:
        return (False, "empty")
    k = (kind or "").strip().lower()
    if k in _GAME_SPECIFIC_KINDS:
        return (False, f"game-specific kind '{k}'")
    if k and k not in _GENERAL_KINDS:
        return (False, f"kind '{k}' is not a general-technique kind")
    if _ACTION_TOKEN.search(text):
        return (False, "names a specific ACTION<n> (game action semantics)")
    if _HEX_COLOR.search(text):
        return (False, "names a specific sprite colour (#rrggbb)")
    if _PROVENANCE_MARK.search(text):
        return (False, "references a specific level/trial/win provenance")
    low = text.lower()
    known = vocab if vocab is not None else game_id_vocab()
    hit = next((g for g in known if g and g in low), None)
    if hit:
        return (False, f"names a specific game id '{hit}'")
    if _GAMEID_SHAPE.search(low):
        return (False, "contains a game-id-shaped token")
    return (True, "general technique, no game-specific tokens")


def consider_promote(description: str, *, kind: str = "technique",
                     credence: float = 0.7, from_game: str = "",
                     lesson_id: str = "", min_credence: float = 0.6,
                     vocab=None, path: Path = DEFAULT_GENERAL_PATH) -> bool:
    """Auto-promotion hook: re-home a per-game lesson into the game-agnostic
    store IFF it is judged genuinely general AND credible enough.  Guarded +
    idempotent (add_general dedups).  Returns True iff the store grew.  Call this
    wherever a lesson is committed; it self-filters."""
    try:
        if float(credence) < float(min_credence):
            return False
        ok, _why = is_general_lesson(description, kind, vocab=vocab)
        if not ok:
            return False
        return add_general(description, kind=kind, credence=float(credence),
                           from_game=from_game, lesson_id=lesson_id,
                           note="auto-promoted: judged game-agnostic", path=path)
    except Exception:
        return False


def general_promotion_candidates(*, min_credence: float = 0.6) -> list:
    """List per-game lessons that PASS the text safety-net and so are CANDIDATES
    for promotion -- for VLM/human REVIEW, deliberately NOT auto-promoted.

    The safety-net (is_general_lesson) is necessary but NOT sufficient: a
    game-flavoured PROCEDURE (game-specific nouns like 'pierce'/'skewer', but no
    game-id / ACTION / hex token) passes it yet must NOT be surfaced on every
    game.  Telling such a procedure from a genuine technique needs MEANING, not a
    regex, so a judgement step confirms generality before promote_from_lesson is
    called.  Returns [{game_id, lesson_id, kind, credence, text}], newest-credence
    first."""
    try:
        from per_game_lessons import load_all_games
    except ImportError:
        from perception_loop_v2.per_game_lessons import load_all_games
    vocab = game_id_vocab()
    out = []
    for gid, l in load_all_games():
        cred = float(getattr(l, "credence", 0.0))
        if cred < min_credence:
            continue
        ok, _why = is_general_lesson(getattr(l, "description", ""),
                                     getattr(l, "kind", ""), vocab=vocab)
        if ok:
            out.append({"game_id": str(gid), "lesson_id": getattr(l, "lesson_id", ""),
                        "kind": getattr(l, "kind", ""), "credence": cred,
                        "text": getattr(l, "description", "")})
    out.sort(key=lambda d: -d["credence"])
    return out


if __name__ == "__main__":  # tiny CLI: promote a lesson / list candidates
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "promote":
        ok = promote_from_lesson(sys.argv[2], sys.argv[3])
        print("promoted" if ok else "not found / no change")
    elif len(sys.argv) == 2 and sys.argv[1] == "candidates":
        for c in general_promotion_candidates():
            print(f"  [{c['kind']} {c['credence']:.2f}] {c['game_id']}/"
                  f"{c['lesson_id']}: {c['text'][:90]}")
    else:
        print("usage: python general_knowledge.py promote <game_id> <lesson_id>"
              "  |  python general_knowledge.py candidates")
