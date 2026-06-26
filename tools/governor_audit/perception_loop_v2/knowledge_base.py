"""Knowledge base loader and accessor.

Reads per-game JSON knowledge files from `knowledge/` (sibling to
the perception_loop_v2 package) and provides a typed API for the
role resolver and fixture scorer.

Knowledge files are DATA, not CODE.  A new game ships by writing
a new JSON file; no engine code change.  Schema:

    {
      "game_id": "bp35",
      "universal_roles":  [ {role, matcher, projection_mode?,
                             cardinality?}, ... ],
      "game_roles":       [ {role, matcher, projection_mode?,
                             cardinality?}, ... ],
      "levels": {
        "0": {
          "background_projection": "bbox_membership" | "pixel_majority",
          "projection_overrides":  { <role_name>: <projection_mode>, ... },
          "truth_codes":           { <role_name>: <fixture_code>, ... },
          "roles":                 [ {role, matcher, ...}, ... ]   // legacy
        },
        ...
      }
    }

`universal_roles` are the same across every game (the agent and
background detectors).  `game_roles` are game-wide — declared once,
they apply to every level of the game with optional per-level
`projection_overrides` for roles whose projection genuinely differs
across levels (e.g. lc=0 HUD is a small counter projecting on its
centroid, lc=1 HUD is a wide bar projecting across its bbox row).

Per-level `roles` is the legacy slot from before game_roles existed
— still parsed for back-compat, but new knowledge files should
declare every cross-level-shared rule in `game_roles` instead.

`matcher` is a typed dict:
    { "type": "<matcher_kind>", "<arg_name>": <value>, ... }

The engine knows a closed set of matcher kinds (see
role_resolver.py).  Operators DO NOT add new matcher kinds; if a
game's behaviour needs something new, that's an engine extension.
Operators DO add new role/matcher COMBINATIONS — that's the
substrate-agnostic data-driven extension surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"


@dataclass
class RoleRule:
    """One role definition: how to identify entities playing this role.

    `projection_mode` controls how matched entities project onto the
    cell grid:
      - "centroid"  (default): only the entity's centroid cell gets
                    the role's truth code.  Right for compact sprites
                    (agent, sediment, win-marker, hazard).
      - "all_cells": every cell the entity touches gets the code.
                    Right for strips and other entities whose extent
                    is the role's meaning (HUD bar, danger bar).
      - "bbox_strip_row": the entity's bbox row(s) get coded the
                    role across the full frame width.  Useful for
                    HUD entities that visually occupy a strip even
                    if the entity itself is narrower.

    `cardinality`:
      - "multi" (default): claim every entity the matcher accepts.
                Right for roles that legitimately appear multiple
                times per frame (sediment, hazard, win-marker).
      - "single": claim AT MOST ONE entity (the best — i.e. smallest —
                  matching entity).  Right for singleton roles where
                  multiple matches indicate ambiguity, not multiplicity.
                  The agent is the canonical example: there is one
                  agent per frame, and if several entities overlap
                  the harness's agent_position signal, only the
                  smallest (most specific) should be claimed.
    """

    role: str            # role name (e.g., "agent", "consumable")
    matcher: dict        # typed dict; role_resolver dispatches on matcher["type"]
    projection_mode: str = "centroid"
    cardinality: str = "multi"


@dataclass
class LevelKnowledge:
    """Per-level overlay on top of the game-wide rule set.

    `background_projection` selects how cells are assigned to
    backgrounds:
      - "bbox_membership"  - cells overlapping a bg's bbox; smaller
                             bbox wins (good when truth uses region
                             semantics — \"cell inside playfield = bg\").
      - "pixel_majority"   - cells with the most bg pixels (good when
                             truth uses pixel content — \"cell mostly
                             water = water\").
    Both rules are substrate-agnostic; the choice is operator
    metadata about which labelling convention the fixture follows.

    `projection_overrides` lets a level redefine the projection_mode
    for any role declared in `game_roles` whose projection genuinely
    differs at this level.  Sparse: declare only roles that change.

    `roles` is the legacy per-level rule list, still parsed for
    back-compat.  New knowledge files should declare cross-level
    rules in `game_roles` and use `projection_overrides` here.
    """

    truth_codes: dict[str, str] = field(default_factory=dict)
    background_projection: str = "bbox_membership"
    projection_overrides: dict[str, str] = field(default_factory=dict)
    roles: list[RoleRule] = field(default_factory=list)


@dataclass
class GameKnowledge:
    """All knowledge about one game.

    `universal_roles` are cross-game (agent + background detectors).
    `game_roles` are cross-level (the role vocabulary the game has,
    declared once and reused across every level with optional
    per-level projection overrides).  `levels` carries only what
    truly varies per level.
    """

    game_id: str
    universal_roles: list[RoleRule] = field(default_factory=list)
    game_roles: list[RoleRule] = field(default_factory=list)
    levels: dict[str, LevelKnowledge] = field(default_factory=dict)

    def for_level(self, lc: int) -> tuple[list[RoleRule], dict[str, str], str]:
        """Return the active role rules (universal + game-wide +
        level-specific), the truth-code mapping, and the background-
        projection style for the given level.

        Game-wide rules get any per-level projection_overrides applied
        before being handed to the resolver — the matcher and
        cardinality stay, only the projection_mode is rebound.

        The order universal -> game_roles -> level.roles matters for
        first-match-wins categorical matchers (the agent rule should
        always fire before any visual_template rule, for instance).
        """
        lvl = self.levels.get(str(lc)) or LevelKnowledge()
        effective_game_rules: list[RoleRule] = []
        for r in self.game_roles:
            override = lvl.projection_overrides.get(r.role)
            if override is not None and override != r.projection_mode:
                effective_game_rules.append(RoleRule(
                    role=r.role,
                    matcher=dict(r.matcher),
                    projection_mode=override,
                    cardinality=r.cardinality,
                ))
            else:
                effective_game_rules.append(r)
        rules = (
            list(self.universal_roles)
            + effective_game_rules
            + list(lvl.roles)
        )
        return rules, dict(lvl.truth_codes), lvl.background_projection


def _parse_rule(d: dict) -> RoleRule:
    return RoleRule(
        role=str(d["role"]),
        matcher=dict(d["matcher"]),
        projection_mode=str(d.get("projection_mode", "centroid")),
        cardinality=str(d.get("cardinality", "multi")),
    )


def _parse_level(d: dict) -> LevelKnowledge:
    return LevelKnowledge(
        truth_codes=dict(d.get("truth_codes", {})),
        background_projection=str(
            d.get("background_projection", "bbox_membership")
        ),
        projection_overrides={
            str(k): str(v)
            for k, v in (d.get("projection_overrides") or {}).items()
        },
        roles=[_parse_rule(r) for r in d.get("roles", [])],
    )


def load_game(game_id: str) -> GameKnowledge:
    """Load knowledge for one game.  Strips the harness's hash suffix
    (e.g., 'bp35-0a0ad940' → 'bp35') so the same knowledge file applies
    across different harness builds of the same game.

    Returns an empty GameKnowledge if no file is found — the resolver
    will fall back to its universal matchers (agent_position +
    backgrounds) and the rest of the entities stay unclassified.
    That's the cold-start path: no knowledge yet, system still runs.
    """
    short_id = game_id.split("-", 1)[0]
    path = KNOWLEDGE_DIR / f"{short_id}.json"
    if not path.exists():
        return GameKnowledge(game_id=short_id)
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_levels = data.get("levels") or {}
    return GameKnowledge(
        game_id=str(data.get("game_id", short_id)),
        universal_roles=[
            _parse_rule(r) for r in data.get("universal_roles", [])
        ],
        game_roles=[
            _parse_rule(r) for r in data.get("game_roles", [])
        ],
        levels={
            str(k): _parse_level(v)
            for k, v in raw_levels.items()
            # Skip schema doc-comments like "_comment": "string"
            if isinstance(v, dict)
        },
    )


def all_games() -> list[str]:
    """Game ids with on-disk knowledge files (no harness suffix)."""
    if not KNOWLEDGE_DIR.exists():
        return []
    return sorted(p.stem for p in KNOWLEDGE_DIR.glob("*.json"))


# -----------------------------------------------------------------------------
# Writeback — the aggregator writes back to knowledge files when
# cross-trial evidence corroborates a new role rule or visual
# template.  Operators don't write to this; only the engine does.
# -----------------------------------------------------------------------------


def _rule_to_json(r: RoleRule) -> dict:
    out = {"role": r.role, "matcher": r.matcher}
    if r.projection_mode != "centroid":
        out["projection_mode"] = r.projection_mode
    if r.cardinality != "multi":
        out["cardinality"] = r.cardinality
    return out


def save_game(kb: GameKnowledge) -> Path:
    """Persist a knowledge entry to disk.  Called by the aggregator
    after a promotion event.  Operators editing knowledge files
    directly should write the file by hand and reload."""
    short_id = kb.game_id.split("-", 1)[0]
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    path = KNOWLEDGE_DIR / f"{short_id}.json"
    levels_out: dict = {}
    for lc, lvl in kb.levels.items():
        entry: dict = {
            "background_projection": lvl.background_projection,
        }
        if lvl.projection_overrides:
            entry["projection_overrides"] = dict(lvl.projection_overrides)
        entry["truth_codes"] = dict(lvl.truth_codes)
        if lvl.roles:
            entry["roles"] = [_rule_to_json(r) for r in lvl.roles]
        levels_out[lc] = entry
    data = {
        "game_id": short_id,
        "universal_roles": [
            _rule_to_json(r) for r in kb.universal_roles
        ],
        "game_roles": [
            _rule_to_json(r) for r in kb.game_roles
        ],
        "levels": levels_out,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
