"""Cross-level knowledge promotion.

At END OF LEVEL: take the WorldKnowledge accumulated during the
level and EXTRACT the subset worth carrying to the next level:

  - PROMOTED MECHANIC RULES: hypotheses whose credence crossed
    threshold and were never contradicted.  These become "rules"
    the next level's actor trusts immediately, without needing to
    re-discover them via exploration.

  - ENTITY TEMPLATES: visual signature (appearance) + observed
    role(s) for each canonical entity kind.  In the next level,
    new entities whose appearance matches a known template get
    their role pre-filled.

  - WINNING PATH (if the level was won): the exact sequence of
    actions that solved the level, saved for replay or for solving
    "similar" later levels via transfer.

  - LEVEL-LEVEL OBSERVATIONS: things that span multiple turns,
    e.g. "agent identity persists across all levels", "HUD layout
    is consistent across levels".

At START OF NEXT LEVEL: load the promoted knowledge, pre-populate
the new WorldKnowledge's mechanic_hypotheses (at high credence) and
provide entity-template lookups so first-pass perception can use
the templates as priors.

GAME-AGNOSTIC: nothing in this module references specific games,
sprites, or mechanics.  Templates are stored as appearance-string
+ role-string pairs; rules are stored as trigger-string +
effect-string pairs.  The downstream consumer matches by string
similarity at the moment of use.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from world_knowledge import (
    WorldKnowledge, MechanicHypothesis, EntityRecord,
    WinConditionHypothesis,
)


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class PromotedRule:
    """A mechanic hypothesis that crossed credence threshold during a
    level and is now treated as a near-fact for future levels of
    the same game."""
    trigger: str
    effect: str
    promoted_at_level: int
    promoted_at_turn: int
    credence: float
    times_observed: int
    contradicted_in_levels: list[int] = field(default_factory=list)


@dataclass
class EntityTemplate:
    """A reusable description of an entity kind, indexed by its
    visual appearance signature.  When a new level shows an entity
    with matching appearance, the role can be pre-filled."""
    appearance_signature: str
    canonical_role: str
    canonical_name_pattern: str
        # e.g. "green_tile_*" — used to suggest naming for matching
        # instances in new levels
    observed_in_levels: list[int]
    shape_sig: str = ""
        # rotation/scale-invariant SHAPE signature of the entity's crop
        # (shape_identity).  The IDENTITY channel: match a recurring entity
        # across levels by SHAPE (rotation/scale-tolerant) -- complementary to
        # the text appearance_signature, and kept SEPARATE from the ROLE.
    crop_b64: str = ""
        # base64 PNG of the entity's COLOUR crop -- the actual cropped image,
        # carried so a human/VLM can visually compare identity.


@dataclass
class WinningPath:
    level: int
    actions: list[str]
    turns_used: int
    final_world_summary: dict


@dataclass
class MechanicDigest:
    """A VLM-AUTHORED, end-of-level summary of the level's reusable mechanic
    understanding -- the knowledge worth carrying to the NEXT level (and, because
    it is ROLE-KEYED and pixel-free, eventually to other games / domains).

    Authored by a dedicated end-of-level CRYSTALLISATION prompt (NOT harvested
    mid-play): a single considered summary is stable, whereas per-claim credences
    fluctuate turn to turn.  UNCERTAINTY-TOLERANT by design -- a hint is worth
    carrying even at low confidence (it primes the next level rather than asserting
    a fact).  The reader treats every field as a STRONG PRIOR to verify cheaply
    (mechanic-stability prior), never as ground truth.

    Fields:
      - game_purpose / win_condition: what winning the level requires, in
        free-form role-keyed text (e.g. 'configure the active program so the fired
        mover reaches the goal'), with a confidence.
      - entity_roles: role per ROLE-PATTERN (a name pattern or appearance phrase,
        not an instance id) -> role string (program / trigger / scene / goal /
        reference / control / ...).
      - mechanic_hints: free-form observations the VLM thinks generalise, each with
        its own confidence (e.g. 'a panel of repeated marks is a settable program;
        each column = one program step; firing the trigger runs the program and the
        scene element responds')."""
    level: int
    game_purpose: str = ""
    win_condition: str = ""
    win_condition_confidence: float = 0.0
    entity_roles: dict = field(default_factory=dict)
    mechanic_hints: list = field(default_factory=list)  # [{"hint": str, "confidence": float}]
    win_relation: Optional[dict] = None                 # crystallized checkable form, role-keyed
    authored_at_turn: int = 0


@dataclass
class LevelMemory:
    """Persistent cross-level memory for ONE game (game_id) across
    its sequence of levels.  Loaded at level start, updated at
    level end, saved to disk so a subsequent session can resume."""
    game_id: str
    rules: list[PromotedRule] = field(default_factory=list)
    templates: list[EntityTemplate] = field(default_factory=list)
    winning_paths: list[WinningPath] = field(default_factory=list)
    cross_level_notes: list[str] = field(default_factory=list)
    digests: list[MechanicDigest] = field(default_factory=list)

    # ------------------------------------------------------------------
    # VLM-authored mechanic digest (end-of-level crystallisation)
    # ------------------------------------------------------------------

    def ingest_digest(self, reply: dict, level: int, turn: int = 0) -> "MechanicDigest":
        """Store a VLM-authored mechanic digest (the end-of-level crystallisation
        reply).  Also mirrors the human-readable lines into cross_level_notes so a
        plain reader (and the next-level prompt) sees them.  Returns the digest."""
        reply = reply or {}
        wc = reply.get("win_condition") or {}
        if isinstance(wc, str):
            wc = {"statement": wc, "confidence": 0.5}
        hints = []
        for h in (reply.get("mechanic_hints") or []):
            if isinstance(h, str):
                hints.append({"hint": h, "confidence": 0.4})
            elif isinstance(h, dict) and h.get("hint"):
                hints.append({"hint": str(h["hint"]),
                              "confidence": float(h.get("confidence", 0.4) or 0.4)})
        dg = MechanicDigest(
            level=level,
            game_purpose=str(reply.get("game_purpose", "") or ""),
            win_condition=str(wc.get("statement", "") or ""),
            win_condition_confidence=float(wc.get("confidence", 0.5) or 0.5),
            entity_roles={str(k): str(v) for k, v in (reply.get("entity_roles") or {}).items()},
            mechanic_hints=hints,
            win_relation=reply.get("win_relation") if isinstance(reply.get("win_relation"), dict) else None,
            authored_at_turn=int(turn or 0),
        )
        self.digests.append(dg)
        note = f"[lc{level} digest] WIN: {dg.win_condition} (conf {dg.win_condition_confidence:.2f})"
        if note not in self.cross_level_notes:
            self.cross_level_notes.append(note)
        for h in hints:
            ln = f"[lc{level} hint] {h['hint']} (conf {h['confidence']:.2f})"
            if ln not in self.cross_level_notes:
                self.cross_level_notes.append(ln)
        return dg

    def latest_digest(self) -> Optional["MechanicDigest"]:
        return self.digests[-1] if self.digests else None

    # ------------------------------------------------------------------
    # End-of-level extraction
    # ------------------------------------------------------------------

    def promote_from_world(self, world: WorldKnowledge,
                            win: bool, promotion_credence: float = 0.8
                            ) -> tuple[int, int, bool]:
        """Take the WorldKnowledge from a finished level and pull the
        worth-promoting bits out.  Returns
        (n_rules_added, n_templates_added, winning_path_saved)."""
        n_rules_added = self._promote_rules(
            world, promotion_credence,
        )
        n_templates_added = self._promote_templates(world)
        winning_path_saved = False
        if win and world.actions_taken:
            self.winning_paths.append(WinningPath(
                level=world.level,
                actions=[a.action for a in world.actions_taken],
                turns_used=world.turn,
                final_world_summary=self._summarize_world(world),
            ))
            winning_path_saved = True
        return (n_rules_added, n_templates_added, winning_path_saved)

    def _promote_rules(self, world: WorldKnowledge,
                        threshold: float) -> int:
        n = 0
        existing = {(r.trigger, r.effect): r for r in self.rules}
        for h in world.mechanic_hypotheses:
            if h.credence < threshold:
                continue
            key = (h.trigger, h.effect)
            if key in existing:
                # already known across levels — update times_observed
                existing[key].times_observed += h.supporting_observations.__len__()
                existing[key].credence = max(
                    existing[key].credence, h.credence,
                )
                continue
            self.rules.append(PromotedRule(
                trigger=h.trigger, effect=h.effect,
                promoted_at_level=world.level,
                promoted_at_turn=world.turn,
                credence=h.credence,
                times_observed=len(h.supporting_observations),
            ))
            n += 1
        return n

    def _promote_templates(self, world: WorldKnowledge) -> int:
        """Walk world's entities; for each entity that ended the
        level with a stable role and a non-trivial appearance, add
        a template (or update an existing one)."""
        n = 0
        existing = {t.appearance_signature: t for t in self.templates}
        for rec in world.entities.values():
            if not rec.appearance or not rec.current_role:
                continue
            if rec.current_role == "unknown":
                continue
            sig = _appearance_signature(rec.appearance)
            if sig in existing:
                tpl = existing[sig]
                if world.level not in tpl.observed_in_levels:
                    tpl.observed_in_levels.append(world.level)
                if not tpl.shape_sig and getattr(rec, "shape_sig", ""):
                    tpl.shape_sig = rec.shape_sig          # fill the crop-shape if missing
                if not tpl.crop_b64 and getattr(rec, "crop_b64", ""):
                    tpl.crop_b64 = rec.crop_b64            # fill the colour crop if missing
                continue
            self.templates.append(EntityTemplate(
                appearance_signature=sig,
                canonical_role=rec.current_role,
                canonical_name_pattern=_name_pattern(rec.name),
                observed_in_levels=[world.level],
                shape_sig=getattr(rec, "shape_sig", ""),
                crop_b64=getattr(rec, "crop_b64", ""),
            ))
            n += 1
        return n

    def _summarize_world(self, world: WorldKnowledge) -> dict:
        return {
            "turns": world.turn,
            "n_entities": len(world.entities),
            "n_groups": len(world.groups),
            "n_relationships": len(world.relationships),
            "game_type_guess": world.game_type_guess,
            "game_purpose_guess": world.game_purpose_guess,
        }

    # ------------------------------------------------------------------
    # Start-of-level inheritance
    # ------------------------------------------------------------------

    def seed_new_level(self, world: WorldKnowledge,
                        seed_credence: float = 0.8) -> int:
        """Pre-populate ``world.mechanic_hypotheses`` with the rules
        promoted from prior levels.  Returns the number of rules
        seeded.  Templates remain in this LevelMemory for matching
        as new entities are perceived."""
        if not self.rules:
            return 0
        n = 0
        for r in self.rules:
            # already there from a previous seed?  Skip.
            if any(h.trigger == r.trigger and h.effect == r.effect
                    for h in world.mechanic_hypotheses):
                continue
            world.mechanic_hypotheses.append(MechanicHypothesis(
                hypothesis_id=f"H[{r.trigger} -> {r.effect}]",
                trigger=r.trigger, effect=r.effect,
                credence=min(0.95, max(seed_credence, r.credence)),
                supporting_observations=[],
                contradicting_observations=[],
                promoted=True,
            ))
            n += 1
        world.inherited_from = f"{self.game_id}:level-memory:{n}-rules"
        return n

    def seed_win_condition(self, world: WorldKnowledge,
                           min_seed_credence: float = 0.7) -> bool:
        """Carry the prior level's VLM-authored WIN CONDITION into the new level as
        a STRONG, assume-stable prior.  This is the lever that stops COS from
        re-discovering the win from scratch: a high-credence win hypothesis drops
        the function-coverage exploration importance (1 - max_win_credence), so the
        prober DEFERS blind per-entity probing and the solver pursues the carried
        win plan instead -- to be verified cheaply, not trusted blindly (mechanic-
        stability prior).  Returns True if a win hypothesis was seeded.

        Levels of one ARC-AGI-3 game are always related, so the win condition is
        the single most valuable thing to carry; it is role-keyed text so it also
        survives a re-skin and primes structurally-similar games."""
        dg = self.latest_digest()
        if dg is None or not dg.win_condition:
            return False
        wc = getattr(world, "win_condition_hypotheses", None)
        if wc is None:
            return False
        desc = dg.win_condition
        if any((getattr(h, "description", "") or "") == desc for h in wc):
            return True
        cred = min(0.85, max(min_seed_credence, float(dg.win_condition_confidence or 0.0)))
        wc.append(WinConditionHypothesis(
            hypothesis_id=f"carried-win[lc{dg.level}]",
            description=desc,
            credence=cred,
            promoted=True,
            notes=(f"CARRIED from level {dg.level} (assume-stable; verify cheaply on "
                   f"this level before trusting). Mechanic hints also carried."),
            win_relation=dg.win_relation,
        ))
        return True

    def match_template(self, appearance: str) -> Optional[EntityTemplate]:
        sig = _appearance_signature(appearance)
        for tpl in self.templates:
            if tpl.appearance_signature == sig:
                return tpl
        return None

    def match_by_shape(self, shape_sig: str) -> tuple:
        """Match a new entity to a carried template by ROTATION/SCALE-INVARIANT
        SHAPE (the identity channel) -- recognises a recurring entity even when
        it is rotated or rescaled, which the text appearance_signature misses.
        Returns (template, score) of the best match, or (None, 0.0).  ROLE is
        NOT decided here (mover and goal can be rotations of each other); the
        caller separates identity from role."""
        if not shape_sig:
            return None, 0.0
        try:
            import shape_identity as _si
        except Exception:
            return None, 0.0
        best, best_s = None, 0.0
        for tpl in self.templates:
            if not getattr(tpl, "shape_sig", ""):
                continue
            sc = _si.similarity(shape_sig, tpl.shape_sig)
            if sc > best_s:
                best, best_s = tpl, sc
        return best, best_s

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LevelMemory":
        if not path.exists():
            raise FileNotFoundError(path)
        d = json.loads(path.read_text(encoding="utf-8"))
        rules = [PromotedRule(**r) for r in d.get("rules") or []]
        templates = [EntityTemplate(**t) for t in d.get("templates") or []]
        paths = [WinningPath(**p) for p in d.get("winning_paths") or []]
        digests = [MechanicDigest(**g) for g in d.get("digests") or []]
        return cls(
            game_id=d["game_id"],
            rules=rules, templates=templates, winning_paths=paths,
            cross_level_notes=d.get("cross_level_notes") or [],
            digests=digests,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _appearance_signature(appearance: str) -> str:
    """Normalize a free-text appearance string into a comparable
    signature.  Lowercases, strips punctuation, sorts comma-separated
    tokens — so 'bright green tile with rounded corners' and 'rounded-
    corner bright green tile' map to the same signature."""
    if not appearance:
        return ""
    cleaned = "".join(c for c in appearance.lower()
                       if c.isalnum() or c.isspace() or c == ",")
    tokens = sorted(t.strip() for t in cleaned.replace(",", " ").split() if t)
    return " ".join(tokens)


def _name_pattern(name: str) -> str:
    """Generalize an instance name to a pattern.  e.g.
    'green_tile_top_3' -> 'green_tile_top_*'.  Used to suggest
    naming for new instances of a known template."""
    if not name:
        return ""
    parts = name.split("_")
    # strip trailing numeric suffix
    while parts and (parts[-1].isdigit() or
                      (parts[-1].startswith("c") and parts[-1][1:].isdigit())):
        parts.pop()
    return "_".join(parts) + "_*" if parts else name
