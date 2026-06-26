"""Situation-keyed KB recall: surface the RIGHT knowledge at every decision,
ranked to the live situation, cheaply even when the KB is large.

WHY
---
The recurring failure: I re-derive from scratch while the answer sits in the KB
(e.g. the lc-4 win procedure at credence 0.85 among 261 lessons). A slim dump by
credence buries the situation-specific entry. The fix is RETRIEVAL, not a dump:
each turn, build a query from the current situation, rank ALL knowledge sources
by relevance to it, and surface the top-k prominently — so recall is proactive
and the actor doesn't have to remember to look.

EFFICIENT AT SCALE
------------------
- SCOPE pre-filter first (per-game lessons are already game-scoped; cross-game
  knowledge is filtered by scope), so we never score the whole store.
- KEYWORD pre-filter: entries with zero term overlap are skipped before scoring
  (cheap set intersection), so scoring is O(matching subset), not O(KB).
- TOP-K only; results are cached by a situation signature so an unchanged
  situation reuses the last retrieval. For a very large KB an inverted index /
  embedding ANN slots in behind `_candidate_pool` without changing callers.
"""
from __future__ import annotations

import re
from typing import List, Optional

_STOP = set(
    "the a an of to in on at is are be it its this that with for and or not no "
    "you your i we they he she as by from into out up down so if then than can "
    "will would should could one two three when where what which who how do does "
    "per via etc each all any only also more most less few here there now".split())


def _terms(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9_]+", (text or "").lower())
            if len(t) > 2 and t not in _STOP}


def _referenced_actions(text: str) -> set:
    """ACTION indices a piece of knowledge names (e.g. {1,2} for an ACTION1/2
    rule)."""
    return {int(m) for m in re.findall(r"action\s*(\d+)", (text or "").lower())}


def action_scope_ok(text: str, available_actions) -> bool:
    """KB HYGIENE: a knowledge entry that names an ACTIONk NOT in this game's
    action space is cross-game-mining contamination (e.g. an ACTION1 'impale'
    rule recalled for a game whose only actions are 6/7). Drop it. Entries that
    name no action, or when the action space is unknown, pass through."""
    if not available_actions:
        return True
    refs = _referenced_actions(text)
    if not refs:
        return True
    try:
        avail = {int(a) for a in available_actions}
    except Exception:
        return True
    return refs.issubset(avail)


# Kinds that ARE answers — boosted, especially when stuck.  'corrective' is the
# fix derived from REVIEWING a prior failed attempt at this game (the playback
# review); it is the single most-actionable thing to consult on a retry, so it
# also gets a dedicated top-priority tier in the sort below.
_ANSWER_KINDS = {"win_condition", "technique", "summary", "operator",
                 "subroutine", "resolution", "corrective"}

# DECLARATIVE how-to knowledge — safe to surface proactively at level start even
# with no situation to match (a technique/win-condition/summary is useful before
# you act).  Situation-specific PROCEDURES (operator/subroutine) are deliberately
# excluded: without a matching situation they are noise (e.g. a skewer-game
# 'impale' operator surfacing for an aiming game), so they wait for keyword match.
_DECLARATIVE_KINDS = {"win_condition", "technique", "summary"}


def situation_query(world) -> set:
    """Terms describing what the actor needs NOW: active goals, current
    relations, the objective, and recent intent."""
    parts: List[str] = []
    for sg in (getattr(world, "active_subgoals", None) or []):
        if getattr(sg, "status", "") in ("active", "blocked"):
            parts.append(getattr(sg, "expected_outcome", "") or "")
            parts.append(getattr(sg, "name", "") or "")
    deltas = getattr(world, "deltas_observed", None) or []
    if deltas:
        rels = (getattr(deltas[-1], "relations", None)
                or (deltas[-1].get("relations") if isinstance(deltas[-1], dict) else None) or [])
        for r in rels:
            parts.append(r if isinstance(r, str) else str(r))
    parts.append(getattr(world, "game_purpose", "") or "")
    parts.append(getattr(world, "_last_intent", "") or "")
    return _terms(" ".join(parts))


def is_stuck(world, window: int = 6) -> bool:
    """No score progress over the last `window` turns -> recall of answers is
    forced (this is exactly when re-deriving happens)."""
    hist = getattr(world, "_score_history", None)
    if hist and len(hist) >= window:
        return len(set(hist[-window:])) == 1
    return False


def _candidate_pool(world) -> List[dict]:
    """All knowledge entries in scope: this game's lessons/operators/errors PLUS
    high-credence lessons from OTHER games (so a relevant idea learned elsewhere
    can guide the actor even when nothing game-specific exists). The action-scope
    hygiene guard drops cross-game-mined entries that name actions this game does
    not have. (Swap in an ANN index here for huge KBs.)"""
    pool: List[dict] = []
    gid = getattr(world, "game_id", "") or ""
    avail = (getattr(world, "_available_actions", None)
             or getattr(world, "available_actions", None))
    # Cross-game recall gate: compute the current structural signature once so
    # cross-game knowledge is admitted only from a signature VARIANT of this
    # game (game-agnostic + same-game always pass; fail closed otherwise). See
    # cross_game_knowledge.admits_xgame.
    _xadmit = None
    _cur_sig = None
    _xstore = None
    try:
        from cross_game_knowledge import (compute_signature as _csig,
                                          load_store as _xload,
                                          admits_xgame as _xa)
        _xadmit = _xa
        _cur_sig = _csig(world, avail)
        _xstore = _xload()
    except Exception:
        _xadmit = None

    def _admit_origin(origin) -> bool:
        if not origin or origin == gid:
            return True               # game-agnostic or same game always pass
        if _xadmit is None:
            return False              # gate unavailable -> fail closed
        try:
            return _xadmit(origin, _cur_sig, current_game_id=gid, store=_xstore)
        except Exception:
            return False

    try:
        from per_game_lessons import load_for_game            # noqa: E402
        for l in load_for_game(gid):
            if float(getattr(l, "credence", 0)) < 0.2:
                continue
            pool.append({"source": "lesson", "kind": l.kind,
                         "text": l.description, "credence": float(l.credence),
                         "id": getattr(l, "lesson_id", "")})
    except Exception:
        pass
    # CROSS-GAME: relevant ideas from OTHER games (general knowledge), discounted
    # so this game's own lessons rank ahead.  Ranked to the situation by recall(),
    # so an irrelevant cross-game lesson simply never surfaces.
    try:
        from per_game_lessons import load_all_games           # noqa: E402
        for ogid, l in load_all_games(min_credence=0.7, exclude_game=gid):
            if not _admit_origin(ogid):
                continue              # not a signature variant -> would leak
            pool.append({"source": f"xgame:{ogid}", "kind": l.kind,
                         "text": f"[seen in {ogid}] {l.description}",
                         "credence": float(l.credence) * 0.85,
                         "id": getattr(l, "lesson_id", "")})
    except Exception:
        pass
    # GAME-AGNOSTIC general knowledge: no origin game -> the gate admits it for
    # every game (the compliant home for lessons that hold universally).
    try:
        from general_knowledge import load_general            # noqa: E402
        for g in load_general():
            txt = g.get("text", "")
            if not txt:
                continue
            pool.append({"source": "general", "kind": g.get("kind", "technique"),
                         "text": txt,
                         "credence": float(g.get("credence", 0.6)),
                         "id": g.get("id", "")})
    except Exception:
        pass
    # GAME-AGNOSTIC planning priors: structured means-ends / goal / perception
    # priors (act-through-intermediary, figure-ground-complement = delivery, a
    # mark = correspondence/role).  Forward priors are declarative so they
    # front-load a win hypothesis at level start; backward operators surface on
    # match / when stuck.  No origin game -> admitted on every game.
    try:
        from planning_priors import as_pool_entries as _planning_entries  # noqa: E402
        pool.extend(_planning_entries())
    except Exception:
        pass
    try:
        from operator_kb import load_all_operators, record_origin_game  # noqa: E402
        for r in load_all_operators():
            if not _admit_origin(record_origin_game(r)):
                continue              # cross-game operator, not a variant
            pool.append({"source": "operator", "kind": "operator",
                         "text": f"{r.effect_key} {getattr(r, 'description', '')}",
                         "credence": float(getattr(r, "credence", 0.6)),
                         "id": getattr(r, "operator_id", "")})
    except Exception:
        pass
    # the system's OWN errors + the solutions that resolved them -- so a SIMILAR situation recalls
    # prior experience (reuse the working solution; heed the guard) rather than repeating the mistake.
    try:
        from error_ledger import ErrorLedger                  # noqa: E402
        for r in ErrorLedger().records:
            if r.resolved and r.resolution:
                txt = (f"prior error '{r.category}': {r.description} -> SOLUTION that worked: "
                       f"{r.resolution}")
                kind, cred = "resolution", 0.85
            else:
                txt = (f"error-prone '{r.category}': {r.description}"
                       + (f"; guard: {r.fix}" if r.fix else ""))
                kind, cred = "pitfall", 0.5
            pool.append({"source": "error_ledger", "kind": kind, "text": txt,
                         "credence": cred, "id": r.solution_id or r.category})
    except Exception:
        pass
    # KB HYGIENE: never surface a rule naming an action this game does not have.
    pool = [e for e in pool if action_scope_ok(e["text"], avail)]
    return pool


def recall(world, k: int = 6, query: Optional[set] = None) -> List[dict]:
    """Top-k knowledge entries relevant to the current situation. Cheap:
    scope-filtered pool, keyword pre-filter, partial top-k."""
    q = query if query is not None else situation_query(world)
    stuck = is_stuck(world)
    # cache by (query, stuck) signature
    sig = (frozenset(q), stuck, k)
    cache = getattr(world, "_recall_cache", None)
    if cache and cache.get("sig") == sig:
        return cache["results"]
    # A SPARSE query (no goals/relations/purpose yet — e.g. LEVEL START) means
    # there is nothing to keyword-match against, so fall back to default recall:
    # surface this game's own lessons, stored answers, and high-credence ideas by
    # credence.  With a RICH query (mid-game) the keyword filter still applies, so
    # only situation-relevant entries surface (recall stays cheap + on-point).
    sparse = len(q) < 3
    scored = []
    for e in _candidate_pool(world):
        et = _terms(e["text"])
        overlap = len(q & et)
        base_keep = sparse and (e["source"] == "lesson"
                                or e["kind"] in _DECLARATIVE_KINDS)
        if overlap == 0 and not stuck and not base_keep:
            continue                       # keyword pre-filter: skip irrelevant noise
        denom = (len(q) + 3)
        score = (2.0 * overlap / denom
                 + 0.5 * e["credence"]
                 + (0.15 if e["source"] == "lesson" else 0.0)   # prefer this game
                 + (0.6 if (stuck and e["kind"] in _ANSWER_KINDS) else 0.0)
                 + (0.2 if e["kind"] in _ANSWER_KINDS else 0.0))
        scored.append((score, e))
    # A 'corrective' (the fix learned from a prior failed attempt at THIS game) is
    # an imperative for the retry, so it forms a top-priority tier ahead of any
    # credence-ranked prior — otherwise a stale high-credence cross-game win-
    # condition can crowd the freshly-learned fix out of the top-k.
    scored.sort(key=lambda x: (x[1]["kind"] != "corrective", -x[0]))
    results = [e for _, e in scored[:k]]
    try:
        world._recall_cache = {"sig": sig, "results": results}
    except Exception:
        pass
    return results


def format_recall_surface(world, k: int = 6) -> str:
    """The proactive recall surface: top-k situation-relevant knowledge, shown
    FIRST so the actor consults it before deriving. Names the stuck-trigger."""
    results = recall(world, k=k)
    if not results:
        return ""
    stuck = is_stuck(world)
    head = ("RECALLED KNOWLEDGE — top matches for THIS situation (consult these "
            "BEFORE deriving anything; if one is a stored solution/technique for "
            "this configuration, USE it rather than re-deriving):")
    if stuck:
        head = ("NO PROGRESS detected — FORCED RECALL. The KB very likely "
                "already holds the procedure for this situation. Use it:\n" + head)
    lines = [head]
    for e in results:
        tag = f"{e['kind']}/{e['source']}" if e["source"] != "lesson" else e["kind"]
        lines.append(f"  - [{tag} c={e['credence']:.2f}] {e['text'][:240]}")
    return "\n".join(lines)
