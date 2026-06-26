"""win_claims.py -- persistent, per-game WIN-CONDITION claim (the win condition as a generalizing claim).

Each solved level contributes a win INSTANCE; the induced claim (claim_generalization) is the
high-credence GOAL prior for the NEXT level.  This module persists that claim per game in the KB and
feeds it to the driver, so the win condition transfers across levels AUTOMATICALLY -- the actor no longer
restates it each level.  Feature discovery + back-fill come for free: a feature that is latent early
(e.g. colour-uniformity across same-colour levels) and salient later (a level needing a recolour) is
back-filled into the prior wins via the measure callback, so the claim keeps it.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

from claim_generalization import GeneralizedClaim, Taxonomy, _fmt
from kb_paths import kb_path

# shared win abstractions; extend as new games need (values -> parent)
_WIN_TAX = Taxonomy({
    "complementary": "compatible", "matching": "compatible",
    "closed_box": "unified_figure", "solid_block": "unified_figure", "merged": "unified_figure",
})


def _default_measure(instance_id: str, feature: str):
    """Back-fill prior wins for a latently-constant feature.  A closure win is colour-uniform (the
    union is one solid block of one colour), so colour_uniform back-fills to 'yes'."""
    return "yes" if feature == "colour_uniform" else None


def _store_path():
    return kb_path("win_claims.json")


def _load() -> Dict:
    p = _store_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(data: Dict) -> None:
    p = _store_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _claim_for(game: str, data: Dict, measure=_default_measure) -> GeneralizedClaim:
    c = GeneralizedClaim(f"{game}_win", _WIN_TAX, measure=measure)
    for rec in data.get(game, {}).get("instances", []):
        c.add(rec["id"], dict(rec["features"]))
    return c


def _serialise_pattern(pattern: Optional[Dict]) -> Dict:
    if not pattern:
        return {}
    return {k: (sorted(v) if isinstance(v, frozenset) else v) for k, v in pattern.items()}


def record(game: str, instance_id: str, features: Dict, measure=_default_measure) -> Optional[Dict]:
    """Record a solved level's win instance; re-induce (with back-fill) + persist.  Idempotent per id.
    Returns the induced win pattern."""
    data = _load()
    g = data.setdefault(game, {"instances": []})
    if not any(r["id"] == instance_id for r in g["instances"]):
        g["instances"].append({"id": instance_id, "features": dict(features)})
    c = _claim_for(game, data, measure)
    g["instances"] = [{"id": i, "features": f} for i, f in c.instances]  # persist back-filled features
    g["pattern"] = _serialise_pattern(c.pattern)
    _save(data)
    return c.pattern


def recall(game: str, measure=_default_measure) -> Optional[Dict]:
    """The induced win-condition pattern for a game (the goal prior), or None if no wins recorded."""
    data = _load()
    if game not in data or not data[game].get("instances"):
        return None
    return _claim_for(game, data, measure).pattern


def directive(game: str) -> Optional[str]:
    """A one-line WIN-PRIOR note for the actor, from the generalized win of prior solved levels."""
    p = recall(game)
    if not p:
        return None
    body = ", ".join(f"{k}={_fmt(v)}" for k, v in sorted(p.items()))
    return (f"[WIN-PRIOR] generalized from solved levels of {game}, the win condition is: {body}. "
            f"High-credence goal -- verify against this level, generalize if it doesn't fit.")
