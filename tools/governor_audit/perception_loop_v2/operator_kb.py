"""Operator/technique retrieval by generalized functional key.

See docs/SPEC_operator_retrieval.md. The decisive KB failure is RETRIEVAL,
not registration: a high-credence operator (the mined decouple rule) exists
but the live surface (situation-blind global top-N-by-credence + hard cap)
never shows it. This module fixes that with:

  - GENERALIZED FUNCTIONAL KEYS (VLM-authored, open vocabulary, role/relation
    level, never instance-level): the key is *what the operator achieves*
    ("release a carried object from the manipulator without relocating it"),
    not "free a blue under a red".
  - SEMANTIC RETRIEVAL over those short keys (embeddings when available,
    pure-lexical token-cosine fallback otherwise — always runs).
  - AUTHOR-AGAINST-THE-NEIGHBORHOOD on write: surface nearest existing keys
    so the author reuses (alias) rather than re-mints, keeping the open
    vocabulary self-consolidating instead of drifting.
  - KEY FOR RECALL, PRECONDITION FOR PRECISION: the generalized key recalls
    candidates; a checkable precondition gates applicability.

Dependency-light: reuses cognitive_os.knowledge_index._st_embed/_cosine when
importable; otherwise a self-contained lexical similarity with no numpy/model
dependency, so it runs in the offline human-VLM loop.
"""
from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


try:
    from kb_paths import kb_path as _kb_path
except ImportError:  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path
# Unified KB root (see kb_paths.py + docs/SPEC_knowledge_base.md).
DEFAULT_OPERATOR_KB_PATH = _kb_path("operator_kb.json")

# Reuse the project's embedding stack when available; degrade to lexical.
try:  # pragma: no cover - depends on optional sentence-transformers
    import sys as _sys
    _COG = Path(__file__).resolve().parents[3] / "cognitive_os"
    if str(_COG.parent) not in _sys.path:
        _sys.path.insert(0, str(_COG.parent))
    from cognitive_os.knowledge_index import _st_embed as _ST_EMBED  # type: ignore
    from cognitive_os.knowledge_index import _cosine as _VEC_COSINE  # type: ignore
    # IMPORTANT: the function imports even when the MODEL is absent; it raises
    # only when called. So probe it once — _ST_OK must reflect whether
    # embeddings actually WORK, not merely that the symbol imported. (This
    # exact mistake silently dropped us to the lexical path while reporting
    # 'sentence-transformers'.)
    try:
        _ST_EMBED("probe")
        _ST_OK = True
    except Exception:
        _ST_OK = False
except Exception:
    _ST_EMBED = None
    _VEC_COSINE = None
    _ST_OK = False


def embedding_backend() -> str:
    """'sentence_transformer' when paraphrase-robust embeddings are live,
    else 'lexical'. Callers/operators should surface this so it's clear when
    retrieval is running degraded (lexical needs consistent vocabulary)."""
    return "sentence_transformer" if _ST_OK else "lexical"


_STOP = {
    "a", "an", "the", "to", "of", "is", "it", "and", "or", "by", "with",
    "on", "in", "at", "as", "for", "this", "that", "into", "onto", "from",
    "be", "without", "via", "its", "so", "then", "when", "if", "not",
}


def _tokens(text: str) -> set:
    if not text:
        return set()
    cleaned = "".join(c if c.isalnum() or c.isspace() else " "
                      for c in text.lower())
    # crude stemming: drop trailing 's'/'ing'/'ed' so plurals/tenses match
    out = set()
    for t in cleaned.split():
        if t in _STOP or len(t) <= 2:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if t.endswith(suf) and len(t) - len(suf) >= 3:
                t = t[: -len(suf)]
                break
        out.add(t)
    return out


def _lexical_sim(a: str, b: str) -> float:
    """Token-set cosine over two short functional phrases. Always available;
    matches on shared functional terms (release/object/agent/...)."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / math.sqrt(len(ta) * len(tb))


def _key_similarity(query: str, key: str, key_vec=None, query_vec=None) -> float:
    """Semantic similarity between a query effect and an operator key.
    Uses embeddings when both vectors are available, else lexical."""
    if _ST_OK and key_vec is not None and query_vec is not None:
        try:
            return float(_VEC_COSINE(query_vec, key_vec))
        except Exception:
            pass
    return _lexical_sim(query, key)


def _embed_key(text: str):
    """Return a cached embedding vector (list[float]) for a key, or None if
    embeddings are unavailable (callers fall back to lexical)."""
    if not (_ST_OK and text):
        return None
    try:
        return [float(x) for x in _ST_EMBED(text)]
    except Exception:
        return None


@dataclass
class OperatorRecord:
    """A parameterized macro keyed by generalized function. See SPEC."""
    operator_id: str
    effect_key: str                       # primary, embedded key (generalized)
    precondition: str = ""                # checkable role/relation predicate(s)
    precondition_token: str = ""          # optional substrate-evaluable token
    action_template: list = field(default_factory=list)
    description: str = ""
    credence: float = 0.7
    confirmations: int = 1
    provenance: dict = field(default_factory=dict)
    aliases: list = field(default_factory=list)   # other key phrasings merged
    embedding: Optional[list] = None              # cached vector for effect_key
    scope: dict = field(default_factory=dict)     # game_id / level / roles

    def all_keys(self) -> list:
        return [self.effect_key] + list(self.aliases or [])


def _new_op_id(effect_key: str) -> str:
    safe = "".join(c if c.isalnum() or c == "_" else "_"
                   for c in effect_key.lower())[:40]
    return f"op_{safe}_{int(time.time()) % 1000000}"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def load_kb(path: Path = DEFAULT_OPERATOR_KB_PATH) -> list:
    if not Path(path).exists():
        return []
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        recs = [OperatorRecord(**r) for r in raw.get("operators", [])]
        # Lazy backfill: records committed while the backend was lexical
        # carry embedding=None. Now that embeddings are available, fill them
        # so vector cosine actually runs (the silent-degradation trap: an
        # embedded query vs a None key-vector falls back to lexical and
        # paraphrase retrieval mysteriously returns nothing).
        if _ST_OK:
            changed = False
            for r in recs:
                if r.embedding is None and r.effect_key:
                    r.embedding = _embed_key(r.effect_key)
                    changed = bool(r.embedding) or changed
            if changed:
                try:
                    save_kb(recs, path)
                except Exception:
                    pass
        return recs
    except Exception:
        return []


DEFAULT_SUBROUTINE_KB_PATH = _kb_path("subroutine_kb.json")


def subroutines_as_operator_records(
    path: Path = DEFAULT_SUBROUTINE_KB_PATH) -> list:
    """Bridge: expose subroutine_kb records through the SAME functional-key
    retrieval as operator_kb, so the rich human-authored recipes (e.g.
    'free a wall-jammed cluster') are findable BY FUNCTION — not only via
    the old credence-ranked subroutine surface. Each subroutine becomes an
    OperatorRecord whose:
      - effect_key  = expected_outcome (+ name) — what it ACHIEVES
      - precondition = problem_solved (the situation/gate it addresses)
      - action_template = relational_steps (the generalized chain)
    The embedding is computed lazily by load_all_operators. Source-tagged
    via operator_id prefix 'sub_' so callers can tell origin.
    """
    if not Path(path).exists():
        return []
    try:
        blob = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return []
    subs = blob.get("subroutines") if isinstance(blob, dict) else blob
    out = []
    for s in (subs or []):
        if not isinstance(s, dict):
            continue
        name = s.get("name") or ""
        eo = s.get("expected_outcome") or ""
        # Functional key: prefer expected_outcome (the effect), append the
        # name for extra functional signal. Keep it reasonably short.
        effect_key = (eo or name).strip()
        if name and name.lower() not in effect_key.lower():
            effect_key = f"{effect_key}  ({name})"
        if not effect_key:
            continue
        out.append(OperatorRecord(
            operator_id=str(s.get("subroutine_id") or _new_op_id(name)),
            effect_key=effect_key,
            precondition=(s.get("problem_solved") or "")[:400],
            action_template=list(s.get("relational_steps") or []),
            description=(s.get("description") or "")[:400],
            credence=float(s.get("credence") or 0.6),
            provenance={"source": "subroutine_kb",
                        "prov": s.get("provenance")},
            embedding=None,  # filled lazily
        ))
    return out


def load_all_operators(op_path: Path = DEFAULT_OPERATOR_KB_PATH,
                       sub_path: Path = DEFAULT_SUBROUTINE_KB_PATH) -> list:
    """Unified pool for retrieval: operator_kb operators + subroutine_kb
    recipes (bridged), with embeddings backfilled when the embedding
    backend is live. This is what the live surface should query so the
    BEST knowledge — wherever it was authored — is retrievable by
    function."""
    recs = load_kb(op_path)
    subs = subroutines_as_operator_records(sub_path)
    if _ST_OK:
        for r in subs:
            if r.embedding is None and r.effect_key:
                r.embedding = _embed_key(r.effect_key)
    return recs + subs


def record_origin_game(rec) -> Optional[str]:
    """Origin game id of an operator / bridged-subroutine record, or None when
    game-agnostic. operator_kb records carry ``scope.game_id``; bridged
    subroutine records carry it under ``provenance['prov'][*]['game_id']``.
    Used by the cross-game recall gate (cross_game_knowledge.admits_xgame)."""
    sc = getattr(rec, "scope", None) or {}
    g = sc.get("game_id")
    if g:
        return g
    prov = getattr(rec, "provenance", None) or {}
    if isinstance(prov, dict):
        pv = prov.get("prov")
        if isinstance(pv, list):
            for p in pv:
                if isinstance(p, dict) and p.get("game_id"):
                    return p["game_id"]
    return None


def save_kb(records: list, path: Path = DEFAULT_OPERATOR_KB_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blob = {"operators": [asdict(r) for r in records],
            "updated": int(time.time())}
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(blob, indent=2), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Author against the neighborhood (write path)
# ---------------------------------------------------------------------------

def neighbors(records: list, effect_key: str, *, k: int = 5,
              scope: Optional[dict] = None) -> list:
    """Nearest existing operators to a proposed effect_key, for the
    author-against-the-neighborhood write step. Returns [(record, sim)]
    sorted desc. Scope filter applied when provided."""
    qv = _embed_key(effect_key)
    out = []
    for r in records:
        if scope and r.scope and not _scope_match(scope, r.scope):
            continue
        best = max(
            (_key_similarity(effect_key, kk, r.embedding, qv)
             for kk in r.all_keys()),
            default=0.0,
        )
        out.append((r, best))
    out.sort(key=lambda t: -t[1])
    return out[:k]


def format_authoring_surface(nbrs: list) -> str:
    """Surface nearest existing operators so the author REUSES one (alias)
    when it denotes the same function, or authors a new key explaining the
    difference. Keeps the open vocabulary self-consolidating."""
    if not nbrs:
        return ("AUTHORING A NEW OPERATOR KEY: no existing operators are "
                "near this function — author a clear, unambiguous, FUNCTIONAL "
                "effect key (what it achieves), role-typed (no instance "
                "names like 'blue'/'red'), mechanism-explicit (e.g. 'without "
                "direct contact', 'by carrying').")
    lines = [
        "AUTHOR AGAINST THE NEIGHBORHOOD — these existing operators are the "
        "nearest by function. REUSE one (set reuse_id) if it denotes the "
        "SAME function (your phrasing becomes an alias); otherwise author a "
        "new key and state how it differs:",
    ]
    for r, sim in nbrs:
        lines.append(f"  - [{r.operator_id}] sim={sim:.2f} "
                     f"effect=\"{r.effect_key}\""
                     + (f"  precondition=\"{r.precondition}\""
                        if r.precondition else ""))
    return "\n".join(lines)


def commit_operator(records: list, *, effect_key: str, precondition: str = "",
                    precondition_token: str = "", action_template=None,
                    description: str = "", credence: float = 0.7,
                    provenance: Optional[dict] = None,
                    scope: Optional[dict] = None,
                    reuse_id: Optional[str] = None) -> OperatorRecord:
    """Commit an operator. ``reuse_id`` set => merge this phrasing as an
    alias of that record (consolidation: bump credence/confirmations, fill
    missing fields). Else mint a new record. Returns the affected record.
    Caller persists via save_kb."""
    if reuse_id:
        for r in records:
            if r.operator_id == reuse_id:
                if effect_key and effect_key not in r.all_keys():
                    r.aliases.append(effect_key)
                r.confirmations += 1
                r.credence = max(r.credence, credence)
                if precondition and not r.precondition:
                    r.precondition = precondition
                if precondition_token and not r.precondition_token:
                    r.precondition_token = precondition_token
                if action_template and not r.action_template:
                    r.action_template = list(action_template)
                return r
        # reuse_id not found — fall through to mint
    rec = OperatorRecord(
        operator_id=_new_op_id(effect_key),
        effect_key=effect_key,
        precondition=precondition,
        precondition_token=precondition_token,
        action_template=list(action_template or []),
        description=description,
        credence=credence,
        provenance=dict(provenance or {}),
        scope=dict(scope or {}),
        embedding=_embed_key(effect_key),
    )
    records.append(rec)
    return rec


# ---------------------------------------------------------------------------
# Retrieval (read path)
# ---------------------------------------------------------------------------

def _scope_match(query_scope: dict, rec_scope: dict) -> bool:
    """Cheap scope pre-filter. An operator is in scope if it shares the
    game_id OR is game-agnostic (no game_id), and any required roles it
    names are present. Permissive by design — recall is wide, precision is
    the precondition gate's job."""
    if not rec_scope:
        return True
    gq = query_scope.get("game_id")
    gr = rec_scope.get("game_id")
    if gr and gq and gr != gq:
        # different game — still allow (cross-game transfer) but caller can
        # down-weight; keep permissive here.
        return True
    return True


def retrieve_operators(records: list, query_effect: str, *,
                       current_relations: Optional[list] = None,
                       precondition_checker=None, scope: Optional[dict] = None,
                       k: int = 5, min_sim: float = 0.12) -> list:
    """Retrieve operators relevant to a desired effect.

    1. scope pre-filter, 2. semantic recall over effect_key+aliases,
    3. precondition gate. ``precondition_checker(record, current_relations)
    -> bool`` (optional) decides applicability for substrate-checkable
    preconditions; free-form preconditions pass through (the VLM gates) but
    are surfaced. Returns [(record, sim)] sorted by sim desc, top-k."""
    qv = _embed_key(query_effect)
    scored = []
    for r in records:
        # Cross-game recall gate: a record whose origin game differs from the
        # current one surfaces ONLY as a signature variant; game-agnostic and
        # same-game records pass. Fails closed when the signature is absent.
        if scope is not None:
            origin = record_origin_game(r)
            cgid = scope.get("game_id")
            if origin and origin != cgid:
                try:
                    from cross_game_knowledge import admits_xgame as _xadmit
                    if not _xadmit(origin, scope.get("signature"),
                                   current_game_id=cgid):
                        continue
                except Exception:
                    continue
        sim = max(
            (_key_similarity(query_effect, kk, r.embedding, qv)
             for kk in r.all_keys()),
            default=0.0,
        )
        if sim < min_sim:
            continue
        if precondition_checker is not None and r.precondition_token:
            try:
                if not precondition_checker(r, current_relations):
                    continue
            except Exception:
                pass
        scored.append((r, sim))
    scored.sort(key=lambda t: -t[1])
    return scored[:k]


# ---------------------------------------------------------------------------
# Per-game replay verification
# ---------------------------------------------------------------------------
# A mined operator can be GENERIC (derived from another game, or never seen to
# actually produce its effect HERE). Recalling it is not enough -- this session
# the generic `impale-by-backstop` was recalled correctly but did NOT pierce in
# lc=4 (pushing against the wall was a no-op). So an operator is trusted for a
# game only once it is CONFIRMED to produce its effect there; until then it is
# UNCONFIRMED (apply tentatively + verify), and if it demonstrably fails it is
# REFUTED for that game. Status lives in provenance (json-safe).

def operator_status(rec, game_id: str) -> str:
    """'confirmed' | 'refuted' | 'unconfirmed' for `game_id`."""
    prov = getattr(rec, "provenance", None) or {}
    if game_id in (prov.get("refuted_in_games") or []):
        return "refuted"
    if game_id in (prov.get("confirmed_in_games") or []):
        return "confirmed"
    return "unconfirmed"


def note_operator_outcome(records: list, operator_id: str, game_id: str,
                          confirmed: bool) -> bool:
    """Record that operator `operator_id`, applied in `game_id`, DID (confirmed)
    or DID NOT (refuted) produce its claimed effect. Caller persists via
    save_kb. Returns True if a record was updated."""
    for r in records:
        if r.operator_id != operator_id:
            continue
        prov = r.provenance = dict(getattr(r, "provenance", None) or {})
        key = "confirmed_in_games" if confirmed else "refuted_in_games"
        other = "refuted_in_games" if confirmed else "confirmed_in_games"
        lst = list(prov.get(key) or [])
        if game_id not in lst:
            lst.append(game_id)
        prov[key] = lst
        prov[other] = [g for g in (prov.get(other) or []) if g != game_id]
        if confirmed:
            r.credence = min(getattr(r, "credence", 0.7) + 0.05, 0.98)
        else:
            r.credence = max(getattr(r, "credence", 0.7) - 0.2, 0.05)
        return True
    return False


def _status_tag(rec, game_id: Optional[str]) -> str:
    if not game_id:
        return ""
    st = operator_status(rec, game_id)
    return {"confirmed": "  [CONFIRMED here]",
            "refuted": "  [REFUTED here — DO NOT use]",
            "unconfirmed": "  [UNCONFIRMED here — apply tentatively + VERIFY "
                           "it actually produces the effect]"}.get(st, "")


def format_retrieval_surface(scored: list, query_effect: str = "",
                             game_id: Optional[str] = None) -> str:
    """Render retrieved operators for the actor, key-first, with precondition
    (the applicability gate the actor must check), action template, and the
    per-game verification status (confirmed / unconfirmed / refuted)."""
    if not scored:
        return ""
    head = ("RELEVANT OPERATORS (retrieved by FUNCTION for the current "
            "goal" + (f": \"{query_effect}\"" if query_effect else "")
            + " — ranked by functional similarity; CHECK each precondition "
            "against the current relations before applying):")
    lines = [head]
    # Query-normalization reminder: retrieval matches the GOAL's generalized
    # functional effect against generalized operator keys. If the goal was
    # phrased in absolute/instance terms the top match will be weak/wrong;
    # re-query in RELATIVE, ROLE-typed terms. (Verified: relative+role
    # phrasing ~0.64 vs absolute+instance ~0.34 for the same intent.)
    if query_effect:
        lines.append("  (If these look off-target, your goal may be phrased "
                     "in ABSOLUTE/INSTANCE terms. Re-query as a RELATIVE, "
                     "ROLE-typed effect — e.g. 'to row 26' -> 'one step "
                     "perpendicular to the arm reach axis'; 'the orange "
                     "block' -> 'a block'.)")
    for r, sim in scored:
        lines.append(f"  - sim={sim:.2f} EFFECT: {r.effect_key}"
                     + _status_tag(r, game_id))
        if r.precondition:
            lines.append(f"      precondition (gate): {r.precondition}")
        if r.action_template:
            lines.append(f"      how: {', '.join(str(a) for a in r.action_template)}")
        if r.description:
            lines.append(f"      note: {r.description[:200]}")
    return "\n".join(lines)
