"""Cross-game knowledge store — game profiles + variation matching.

Strategic-layer analog of ``global_priors.py``.  Where global_priors
transfers low-level action->effect primitives across games, THIS store
transfers high-level TYPED knowledge about what a game IS and how to
approach it, and matches it to new games by a structural SIGNATURE so
that minor VARIANTS of a known game benefit immediately.

See docs/SPEC_cross_game_knowledge_store.md.

PRIME DIRECTIVE: signature + knowledge flow from open-vocabulary
perception output and actor text only.  No hardcoded game list, no
per-game branch.  Every game is one data point treated identically.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from world_knowledge import WorldKnowledge

try:
    from kb_paths import kb_path as _kb_path
except ImportError:  # imported as a package
    from perception_loop_v2.kb_paths import kb_path as _kb_path
# Unified KB root (see kb_paths.py + docs/SPEC_knowledge_base.md).
DEFAULT_STORE_PATH = _kb_path("cross_game_knowledge.json")

_SCHEMA_VERSION = 1

# Signature-similarity weighting (must sum to ~1.0).  Each component is
# a structural, game-agnostic feature of the scene.
_W_ROLE_MULTISET = 0.45    # which entity roles + counts are present
_W_GRID_SHAPE    = 0.20    # grid rows/cols/cell size
_W_COLOR_COUNT   = 0.15    # number of distinct entity colors
_W_INDICATOR     = 0.10    # presence of a candidate reference/indicator region
_W_ACTION_ARITY  = 0.10    # number of available actions

# A different-game profile must reach this signature similarity to be
# surfaced as a VARIANT.  Conservative on purpose: a false "variant"
# prior misleads on turn 1.  Verify-early + predict-then-falsify will
# still correct it, but we prefer precision.
_VARIANT_MATCH_THRESHOLD = 0.80

# Credence policy (mirrors per_game_lessons asymmetry).
_BASE_CREDENCE     = 0.55
_SUPPORT_BUMP      = 0.10
_CONTRADICT_DECAY  = 0.25


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class KnowledgeEntry:
    """One typed piece of knowledge about a game.  ``type`` is an OPEN
    set meant to grow; the substrate does not interpret ``content``."""
    type: str            # 'summary' | 'win_condition' | 'strategic_approach'
                          # | 'key_mechanic' | 'avoid' | 'open_question' | ...
    content: str         # free-form actor/distilled text
    credence: float = _BASE_CREDENCE
    n_supporting: int = 1
    n_contradicting: int = 0
    first_trial: str = ""
    last_trial: str = ""
    provenance: str = ""  # how derived (e.g. 'actor', 'distilled:win_cond')


@dataclass
class GameProfile:
    game_id: str
    signature: dict = field(default_factory=dict)
    knowledge: list = field(default_factory=list)  # list[KnowledgeEntry|dict]
    n_trials_contributing: int = 0
    last_updated_iso: str = ""


# ---------------------------------------------------------------------------
# Signature computation (game-agnostic structural fingerprint)
# ---------------------------------------------------------------------------


def compute_signature(world: "WorldKnowledge",
                      available_actions: Optional[list] = None) -> dict:
    """Structural fingerprint from open-vocab perception output.

    Game-agnostic: role names + counts + geometry only.  Two games
    with the same fingerprint are very likely the same puzzle with
    cosmetic changes (a VARIANT).
    """
    # Role multiset
    role_counts: dict = {}
    colors = set()
    indicator_present = False
    for name, ent in (getattr(world, "entities", None) or {}).items():
        role = "unknown"
        rh = getattr(ent, "role_history", None) or []
        if rh:
            last = rh[-1]
            if isinstance(last, (list, tuple)) and len(last) >= 2:
                role = str(last[1])
        role_counts[role] = role_counts.get(role, 0) + 1
        if role in ("hud", "indicator", "target", "reference"):
            indicator_present = True
        # color proxy: appearance string if present
        ap = getattr(ent, "appearance", None)
        if ap:
            colors.add(str(ap)[:12])

    gi = getattr(world, "grid_inference", None)
    grid = {}
    if gi is not None:
        grid = {
            "rows": getattr(gi, "rows", None),
            "cols": getattr(gi, "cols", None),
            "cell": getattr(gi, "cell_ticks", None),
        }

    return {
        "role_multiset": dict(sorted(role_counts.items())),
        "grid": grid,
        "n_colors": len(colors),
        "indicator_present": bool(indicator_present),
        "action_arity": (len(available_actions)
                         if available_actions is not None else None),
    }


def _multiset_similarity(a: dict, b: dict) -> float:
    """Jaccard-like over role->count multisets."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    keys = set(a) | set(b)
    inter = sum(min(a.get(k, 0), b.get(k, 0)) for k in keys)
    union = sum(max(a.get(k, 0), b.get(k, 0)) for k in keys)
    return (inter / union) if union else 1.0


def _grid_similarity(a: dict, b: dict) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    score, n = 0.0, 0
    for k in ("rows", "cols", "cell"):
        va, vb = a.get(k), b.get(k)
        if va is None and vb is None:
            continue
        n += 1
        if va is not None and vb is not None and va == vb:
            score += 1.0
    return (score / n) if n else 1.0


def signature_similarity(sig_a: dict, sig_b: dict) -> float:
    """Weighted structural similarity in [0,1].  Pure geometry/roles."""
    s_role = _multiset_similarity(
        sig_a.get("role_multiset") or {}, sig_b.get("role_multiset") or {}
    )
    s_grid = _grid_similarity(sig_a.get("grid") or {}, sig_b.get("grid") or {})
    na, nb = sig_a.get("n_colors"), sig_b.get("n_colors")
    s_color = 1.0 if (na is None or nb is None) else (
        1.0 - min(1.0, abs(na - nb) / max(1, max(na, nb)))
    )
    s_ind = 1.0 if (sig_a.get("indicator_present")
                    == sig_b.get("indicator_present")) else 0.0
    aa, ab = sig_a.get("action_arity"), sig_b.get("action_arity")
    s_act = 1.0 if (aa is None or ab is None) else (
        1.0 if aa == ab else 1.0 - min(1.0, abs(aa - ab) / max(1, max(aa, ab)))
    )
    return (
        _W_ROLE_MULTISET * s_role
        + _W_GRID_SHAPE * s_grid
        + _W_COLOR_COUNT * s_color
        + _W_INDICATOR * s_ind
        + _W_ACTION_ARITY * s_act
    )


# ---------------------------------------------------------------------------
# Cross-game recall gate (the ONE place every cross-game seam asks "may this
# knowledge surface for the current game?")
# ---------------------------------------------------------------------------


def admits_xgame(origin_game_id: Optional[str],
                 current_sig: Optional[dict],
                 *,
                 current_game_id: Optional[str] = None,
                 store: Optional[dict] = None,
                 threshold: float = _VARIANT_MATCH_THRESHOLD) -> bool:
    """May knowledge whose ORIGIN is ``origin_game_id`` surface for a game whose
    current structural signature is ``current_sig``?

    Every cross-game recall seam (per-game lessons, operators, subroutines,
    visual priors) routes its decision through this one function, so the policy
    lives in a single place. Rules, in order:

      1. ``origin_game_id`` falsy        -> True  (game-agnostic knowledge
         transfers freely; it carries no game-specific reference).
      2. same game (``current_game_id``) -> True.
      3. different game                  -> True ONLY if the origin game's
         stored signature is a structural VARIANT of the current one
         (``signature_similarity >= threshold``).
      4. current / origin signature unknown -> False (FAIL CLOSED).

    Consequence: a genuinely unseen game (whose signature matches nothing in
    the store) receives ONLY game-agnostic knowledge - the required competition
    behavior. Structural decision; no hardcoded game list.
    """
    if not origin_game_id:
        return True
    if current_game_id and origin_game_id == current_game_id:
        return True
    if not current_sig:
        return False
    st = store if store is not None else load_store()
    prof = (st.get("games") or {}).get(origin_game_id)
    osig = (prof or {}).get("signature") or {}
    if not osig:
        return False
    return signature_similarity(current_sig, osig) >= threshold


# ---------------------------------------------------------------------------
# Store load / save
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_store(path: Path = DEFAULT_STORE_PATH) -> dict:
    if not Path(path).exists():
        return {"schema_version": _SCHEMA_VERSION, "games": {}}
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {"schema_version": _SCHEMA_VERSION, "games": {}}


def save_store(store: dict, path: Path = DEFAULT_STORE_PATH) -> None:
    store["last_updated_iso"] = _now_iso()
    tmp = Path(path).with_suffix(".json.tmp")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(json.dumps(store, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Surface (start of game)
# ---------------------------------------------------------------------------


def load_and_surface(world: "WorldKnowledge",
                     available_actions: Optional[list] = None,
                     path: Path = DEFAULT_STORE_PATH) -> str:
    """Compute the current game's signature, find exact + variant
    matches in the store, and return a strategy-prompt block.  Empty
    string when there is nothing to surface (true cold start)."""
    store = load_store(path)
    games = store.get("games") or {}
    if not games:
        return ""
    sig = compute_signature(world, available_actions)
    gid = world.game_id

    exact = games.get(gid)
    variants = []
    for other_id, prof in games.items():
        if other_id == gid:
            continue
        sim = signature_similarity(sig, prof.get("signature") or {})
        if sim >= _VARIANT_MATCH_THRESHOLD:
            variants.append((sim, other_id, prof))
    variants.sort(reverse=True, key=lambda t: t[0])

    if not exact and not variants:
        return ""

    def _fmt_knowledge(prof, top=6):
        ks = prof.get("knowledge") or []
        ks = sorted(
            ks,
            key=lambda k: (-(k.get("credence") or 0),
                           -(k.get("n_supporting") or 0)),
        )[:top]
        out = []
        for k in ks:
            out.append(f"      [{k.get('type','?')} "
                       f"c={k.get('credence',0):.2f}] {k.get('content','')}")
        return out

    lines = ["", "CROSS-GAME KNOWLEDGE (typed profiles from prior games):"]
    if exact:
        lines.append(f"  EXACT MATCH — you have played {gid!r} before "
                     f"({exact.get('n_trials_contributing',0)} trials).  "
                     f"Established knowledge (strong priors, still verify):")
        lines.extend(_fmt_knowledge(exact, top=8))
    if variants:
        for sim, other_id, prof in variants[:2]:
            lines.append(
                f"  VARIANT MATCH — this game RESEMBLES {other_id!r} "
                f"(structural signature {sim*100:.0f}% similar).  The "
                f"following MAY TRANSFER — treat as HYPOTHESES to verify "
                f"EARLY, not facts (an adversarial variant may look "
                f"similar but behave differently):")
            lines.extend(_fmt_knowledge(prof, top=6))
    lines.append(
        "  Use these to jump-start: plan from them, but predict + check "
        "each against this game's actual deltas; drop any that fail.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Update (end of game)
# ---------------------------------------------------------------------------


def _upsert_entry(profile: dict, etype: str, content: str,
                  trial_id: str, provenance: str) -> None:
    content = (content or "").strip()
    if not content:
        return
    ks = profile.setdefault("knowledge", [])
    for k in ks:
        if (k.get("type") == etype
                and (k.get("content", "").strip().lower()
                     == content.lower())):
            k["n_supporting"] = int(k.get("n_supporting", 0)) + 1
            k["credence"] = min(1.0, float(k.get("credence", _BASE_CREDENCE))
                                + _SUPPORT_BUMP)
            k["last_trial"] = trial_id
            return
    ks.append(asdict(KnowledgeEntry(
        type=etype, content=content, first_trial=trial_id,
        last_trial=trial_id, provenance=provenance,
    )))


def update_from_world(world: "WorldKnowledge",
                      *,
                      trial_id: str,
                      available_actions: Optional[list] = None,
                      actor_summary: str = "",
                      actor_strategic_approach: str = "",
                      path: Path = DEFAULT_STORE_PATH) -> int:
    """Upsert this game's profile at end-of-game.  Distills typed
    knowledge from the world + actor-authored narrative.  Returns the
    number of knowledge entries written/updated."""
    store = load_store(path)
    games = store.setdefault("games", {})
    prof = games.setdefault(world.game_id, asdict(GameProfile(
        game_id=world.game_id,
    )))
    prof["signature"] = compute_signature(world, available_actions)
    prof["n_trials_contributing"] = int(
        prof.get("n_trials_contributing", 0)) + 1
    prof["last_updated_iso"] = _now_iso()

    before = len(prof.get("knowledge", []))

    # Distill win-condition from highest-credence WC hypothesis.
    wcs = getattr(world, "win_condition_hypotheses", None) or []
    if wcs:
        best = max(wcs, key=lambda h: getattr(h, "credence", 0.0))
        if getattr(best, "description", ""):
            _upsert_entry(prof, "win_condition", best.description,
                          trial_id, "distilled:win_condition")

    # Actor-authored narrative.
    _upsert_entry(prof, "summary", actor_summary, trial_id, "actor")
    _upsert_entry(prof, "strategic_approach", actor_strategic_approach,
                  trial_id, "actor")

    save_store(store, path)
    return len(prof.get("knowledge", [])) - before + 1
