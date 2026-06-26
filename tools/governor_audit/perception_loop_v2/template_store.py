"""Cross-trial entity template store.

A template is a corroborated mapping from a visual signature
(top-K quantised-RGB histogram tuple) to a role name.  Templates
get written by the aggregator after corroboration; the resolver
consumes them via the `visual_template` matcher.

Storage format: one JSON file per game at
`knowledge/templates/<game_id>.json`:

    {
      "game_id": "bp35",
      "templates": [
        {
          "role":         "consumable",
          "signature":    [[8401664, 0.85], [0, 0.15], ...],
          "level":        "0",
          "first_seen_trial": "...",
          "corroborations":  3
        },
        ...
      ]
    }

Templates are SCOPED AT THE GAME LEVEL — the `level` field on each
template records WHERE it was first observed (provenance), but the
resolver considers every template for the game regardless of which
level the runtime is currently in.

Rationale: levels of the same game share a lot — the same role
vocabulary (consumable / hazard / hud / win_marker), often similar
sprite geometry, often similar behaviour grammar.  Visual
signatures may differ across levels (lc=0 green vs lc=1 pink
consumables) but the cross-level matches that DON'T fire are just
the best-match resolver doing its job — similarity + size gates
reject them naturally.  Filtering by level upfront prevents
transfer that should happen (an entity that visually resembles a
prior-level template can still hint at its role, even if it doesn't
match perfectly).

This module is intentionally simple: load / save / lookup.  The
aggregator (governor side) writes; the resolver (perception side)
reads.  Templates are DATA; the engine code never authors a
specific (signature → role) pair — only learning loops do.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


TEMPLATES_DIR = Path(__file__).resolve().parent / "knowledge" / "templates"


@dataclass
class Template:
    """One corroborated (signature → role) entry.

    `pixel_count_min` / `pixel_count_max` capture the observed
    geometric range of source entities that produced this template.
    The visual_template matcher uses these to reject candidates whose
    size is far outside the template's typical range — no global
    magic threshold needed; the threshold comes from observation.
    """

    role: str
    signature: tuple                # top-K (rgb_key, fraction) pairs
    level: Optional[str] = None     # None = applies to all levels
    first_seen_trial: Optional[str] = None
    corroborations: int = 1
    pixel_count_min: int = 0        # smallest source entity pixel count
    pixel_count_max: int = 0        # largest source entity pixel count


@dataclass
class GameTemplates:
    """Templates for one game.  Templates are GAME-scoped — every
    template in `.templates` is a candidate for the resolver
    regardless of which level the runtime is currently in.  The
    `level` field on each template is informational provenance
    only.

    Callers used to invoke `for_level(lc)` to narrow the candidate
    set; that method has been removed.  Just iterate `.templates`
    directly, or use `all_for_game()` for symmetry with future
    accessor methods.
    """

    game_id: str
    templates: list[Template] = field(default_factory=list)

    def all_for_game(self) -> list[Template]:
        """All templates for this game, regardless of first-seen
        level.  The resolver uses this; level metadata is preserved
        on each template for provenance but is not used as a filter.
        """
        return list(self.templates)


def _parse_template(d: dict) -> Template:
    sig_raw = d.get("signature", [])
    # Allow either list-of-pairs or tuple-of-tuples; normalise to tuples.
    sig = tuple(
        (int(k), float(v)) for k, v in sig_raw
    )
    return Template(
        role=str(d["role"]),
        signature=sig,
        level=(str(d["level"]) if "level" in d else None),
        first_seen_trial=d.get("first_seen_trial"),
        corroborations=int(d.get("corroborations", 1)),
        pixel_count_min=int(d.get("pixel_count_min", 0)),
        pixel_count_max=int(d.get("pixel_count_max", 0)),
    )


def load_templates(game_id: str) -> GameTemplates:
    short_id = game_id.split("-", 1)[0]
    path = TEMPLATES_DIR / f"{short_id}.json"
    if not path.exists():
        return GameTemplates(game_id=short_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    return GameTemplates(
        game_id=str(data.get("game_id", short_id)),
        templates=[
            _parse_template(t) for t in (data.get("templates") or [])
        ],
    )


def save_templates(gt: GameTemplates) -> Path:
    short_id = gt.game_id.split("-", 1)[0]
    TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
    path = TEMPLATES_DIR / f"{short_id}.json"
    data = {
        "game_id": short_id,
        "templates": [
            {
                "role": t.role,
                "signature": [list(s) for s in t.signature],
                **(
                    {"level": t.level} if t.level is not None else {}
                ),
                **(
                    {"first_seen_trial": t.first_seen_trial}
                    if t.first_seen_trial is not None else {}
                ),
                "corroborations": t.corroborations,
                "pixel_count_min": t.pixel_count_min,
                "pixel_count_max": t.pixel_count_max,
            }
            for t in gt.templates
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Similarity primitive — same Bhattacharyya-like overlap used by the
# validator's template-check inspector, so the matcher and the
# inspector stay consistent.
# -----------------------------------------------------------------------------


def signature_similarity(a: tuple, b: tuple) -> float:
    """Bhattacharyya-like overlap of two top-K signatures in [0, 1].

    The signature is a tuple of (key, fraction) pairs.  Overlap is
    the sum of min-fraction for shared keys.  Identity signatures
    return 1.0; disjoint return 0.0.
    """
    if not a or not b:
        return 0.0
    d_a = dict(a)
    d_b = dict(b)
    overlap = 0.0
    for k, fa in d_a.items():
        fb = d_b.get(k, 0.0)
        overlap += min(fa, fb)
    return overlap
