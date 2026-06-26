"""Substrate-agnostic perception rule schema and persistent store.

A perception rule binds a substrate-agnostic feature (a quantised
RGB colour, a composite-sprite signature, a HUD-strip position)
to one of the system's entity codes (W / B / P / A / H / U).

Rules NEVER mention palette indices, tile IDs, byte channels, or any
other harness-implementation handle.  Their bodies are expressed in
the vocabulary described in DURABLE_PRINCIPLES.md P10:

  - Visual signature   — quantised RGB tuples
  - Spatial pattern    — frame-edge proximity, strip position, bbox dims
  - Persistence        — static-across-frames flag

Lifecycle (sandbox → trial → established) matches the spec.  See
SPEC_governor.md for the design intent and the retraction note for
why the old palette-keyed body shapes are gone.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# -----------------------------------------------------------------------------
# Rule bodies — all substrate-agnostic.
# -----------------------------------------------------------------------------


@dataclass
class ColorBindingBody:
    """`type=color_binding`: a quantised RGB key plays a particular
    structural role in this (game, lc).

    Example: in bp35 lc=1, the primary background colour is
    quantised (64, 192, 48) -> role W.  No mention of which palette
    index that came from."""
    role: str                              # "W" | "B" | "P"
    rgb_key: tuple[int, int, int]          # quantised RGB
    structural_rank: str = "primary"       # "primary" | "secondary" | "minor"


@dataclass
class HudStripBody:
    """`type=hud_strip`: a thin horizontal band pinned to a frame
    edge is the non-game-area display.  Cells overlapping its
    y-range -> U."""
    y_range_logical: tuple[int, int]       # inclusive raw-pixel rows
    edge: str = "bottom"                   # "top" | "bottom"
    dominant_rgb_key: tuple[int, int, int] | None = None


@dataclass
class CompositeSpriteBody:
    """`type=composite_sprite`: a small bbox containing pixels of N
    distinct colours from this colour set.  Cells touching such a
    sprite -> H."""
    role: str = "H"
    color_set: list[tuple[int, int, int]] = field(default_factory=list)
    min_distinct_colors: int = 3
    bbox_max_logical: tuple[int, int] = (8, 8)


# -----------------------------------------------------------------------------


@dataclass
class Rule:
    """One committed rule in the per-(game, lc) store."""
    id: str
    type: str                              # "color_binding" | "hud_strip" | ...
    body: dict                             # serialised body dataclass
    status: str                            # "sandbox" | "trial" | "established" | "deprecated"
    evidence_count: int                    # turns this rule was supported
    confidence: float
    added_at: str                          # ISO timestamp
    added_in_trial: str                    # trial id where first observed
    last_trial: str                        # most recent trial supporting it
    # Trials this rule has been independently corroborated in.
    supporting_trials: list[str] = field(default_factory=list)
    # Trials with contradicting evidence.
    contradicting_trials: list[str] = field(default_factory=list)


def _stable_id(rule_type: str, body: dict) -> str:
    """Hash of (type, body) so the same logical rule gets the same id
    across runs."""
    blob = json.dumps({"type": rule_type, "body": body}, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


# -----------------------------------------------------------------------------
# Persistent store.
# -----------------------------------------------------------------------------


class RuleStore:
    """JSON-backed store of perception rules for one (game, lc).

    File layout: <root>/<game>_lc<N>.json
    Schema:
      {
        "game_id": str,
        "lc":      int,
        "rules":   [Rule.asdict, ...],
      }
    """

    def __init__(self, file_path: Path, game_id: str, lc: int):
        self.file_path = file_path
        self.game_id = game_id
        self.lc = lc
        self._rules: dict[str, Rule] = {}
        self._dirty = False
        self._load()

    @classmethod
    def for_level(cls, root: Path, game_id: str, lc: int) -> "RuleStore":
        return cls(root / f"{game_id}_lc{lc}.json", game_id, lc)

    def _load(self) -> None:
        if not self.file_path.exists():
            return
        try:
            data = json.loads(
                self.file_path.read_text(encoding="utf-8")
            )
        except Exception:
            return
        for r in data.get("rules", []):
            try:
                rule = Rule(**r)
                self._rules[rule.id] = rule
            except TypeError:
                continue

    def save(self) -> None:
        if not self._dirty:
            return
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_path.write_text(
            json.dumps({
                "game_id": self.game_id, "lc": self.lc,
                "rules": [asdict(r) for r in self._rules.values()],
            }, indent=2),
            encoding="utf-8",
        )
        self._dirty = False

    def upsert(self, rule: Rule) -> None:
        self._rules[rule.id] = rule
        self._dirty = True

    def get(self, rule_id: str) -> Rule | None:
        return self._rules.get(rule_id)

    def all(self) -> list[Rule]:
        return list(self._rules.values())

    def by_status(self, *statuses: str) -> list[Rule]:
        return [r for r in self._rules.values() if r.status in statuses]

    def by_type(self, *types: str) -> list[Rule]:
        return [r for r in self._rules.values() if r.type in types]

    def active(self, *, allow_sandbox: bool = False) -> list[Rule]:
        """Rules that should be applied at perception time.  Sandbox
        excluded by default."""
        return [
            r for r in self._rules.values()
            if r.status in (
                ("sandbox", "trial", "established")
                if allow_sandbox else ("trial", "established")
            )
        ]


# -----------------------------------------------------------------------------
# Candidate emitted by the detector before the aggregator commits.
# -----------------------------------------------------------------------------


@dataclass
class Candidate:
    """A proposed rule before it has been committed."""
    type: str
    body: dict
    evidence_count: int = 1
    supporting_turns: list[int] = field(default_factory=list)

    @property
    def signature(self) -> str:
        return _stable_id(self.type, self.body)
