"""Persistent, scoped, value-of-information-ranked Claim Store (game-agnostic).

The UNIFIED epistemic frontier that makes exploration claim-directed instead of
blind coverage.  The VLM AUTHORS claims -- free-form hypotheses about mechanics,
entity purpose, or the win condition, EACH optionally carrying a DISCRIMINATING
PROBE (an action that would resolve it) + a goal-relevance importance.  The
SUBSTRATE scopes, persists, and RANKS them by value-of-information so the prober
attacks the cheapest most-informative unknown first and skips what is already
proven.  Same division of labor as subroutine_kb (VLM writes, substrate ranks).

It composes the existing pieces rather than duplicating them: credence +
provenance ceilings come from ``claim_credence`` (a carried claim sits at the
GUESSED ceiling until an OBSERVATION re-promotes it -- that IS "assume-stable but
verify"), and a chosen probe is meant to be logged via the ``probes`` ledger.

The four properties the design calls for:
  (1) SCOPE -- each claim is LEVEL-scoped (positioned: valid only for the layout
      that produced it, keyed by a level signature) or CROSS-level (a mechanic
      rule valid across this game's levels).  Mirrors the world's level-vs-cross
      field partition, so positioned claims never leak onto a new layout.
  (2) ASSUME-BUT-RECHECK -- on a level transition a proven CROSS claim is kept
      but dropped to GUESSED credence + flagged needs_recheck, so it is believed
      (a prior) yet scheduled for ONE cheap reconfirmation on the new level.
  (3) PERSIST -- the store round-trips to the KB keyed by game, so a RE-RUN
      reloads the settled frontier and spends its budget on the OPEN claims.
  (4) VALUE-OF-INFORMATION -- open claims are ranked by importance * uncertainty
      / cost (uncertainty = Bernoulli variance 4c(1-c), peaking at c=0.5), with
      un-probed-first as the only tiebreak.  No tuned thresholds.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import claim_credence as cc

# Scope of a claim.
LEVEL = "level"            # positioned -- valid only for the same level layout
CROSS = "cross"            # mechanic / rule -- valid across this game's levels
CROSS_GAME = "cross_game"  # structural pattern -- a prior across DIFFERENT games

# Resolution status.
OPEN = "open"
PROVEN = "proven"
REFUTED = "refuted"


@dataclass
class StoredClaim:
    claim_id: str
    statement: str
    scope: str = CROSS
    level_signature: str = ""              # for LEVEL claims: the layout they belong to
    kind: str = "function"                 # 'function' | 'structural' | ...
    target: list = field(default_factory=list)   # entity names the claim concerns
    probe: Optional[str] = None            # a discriminating ACTION, e.g. "CLICK:8,44"
    plan: Optional[str] = None             # a multi-step verification, e.g. "MATCH:A~B+TRIGGER"
    cost: int = 1                          # actions to run the probe
    importance: float = 0.5                # VLM-judged goal-relevance / value, [0,1]
    provenance: str = cc.GUESSED
    credence: float = 0.0
    status: str = OPEN
    needs_recheck: bool = False            # carried into a new level -> verify cheaply
    times_probed: int = 0
    last_turn: int = 0

    def uncertainty(self) -> float:
        """Bernoulli variance of the belief -- maximal (1.0) at credence 0.5,
        zero at 0 or 1.  A claim we are sure about (either way) is not worth a
        probe; one at 50/50 is."""
        c = max(0.0, min(1.0, self.credence))
        return 4.0 * c * (1.0 - c)

    def value_of_information(self) -> float:
        """VLM importance x substrate uncertainty, per unit cost.  No tuned
        constants -- importance and cost are VLM-supplied, uncertainty is
        derived, and un-probed-first is handled as a structural tiebreak in
        ``ClaimStore.rank`` rather than a magic weight."""
        return self.importance * self.uncertainty() / max(1, int(self.cost))


@dataclass
class ClaimStore:
    game_id: str = ""
    claims: dict = field(default_factory=dict)     # claim_id -> StoredClaim

    # ---- authoring (VLM writes) ------------------------------------------
    def ingest(self, authored, turn: int = 0, level_signature: str = "") -> "ClaimStore":
        """Fold VLM-authored claim records into the store.  Each record:
        {id, statement, scope?('level'|'cross'), target?[], probe?(action),
         cost?, importance?, provenance?, credence?}.  New claims start OPEN;
        existing claims get their mutable fields refreshed but are never
        silently downgraded from PROVEN/REFUTED back to OPEN."""
        for a in authored or []:
            cid = a.get("id") or a.get("claim_id")
            if not cid:
                continue
            raw = str(a.get("scope", "")).lower()
            scope = LEVEL if raw.startswith("lev") else (raw or CROSS)
            prov = a.get("provenance") or cc.GUESSED
            cur = self.claims.get(cid)
            if cur is None:
                cred = float(a.get("credence", cc.CEILING.get(prov, cc.CEILING[cc.GUESSED])))
                self.claims[cid] = StoredClaim(
                    claim_id=cid, statement=a.get("statement", ""),
                    scope=scope,
                    level_signature=(level_signature if scope == LEVEL else ""),
                    kind=a.get("kind", "function"),
                    target=list(a.get("target") or []),
                    probe=a.get("probe"), plan=a.get("plan"),
                    cost=int(a.get("cost", 1) or 1),
                    importance=float(a.get("importance", 0.5)),
                    provenance=prov, credence=cc.clamp(prov, cred),
                    status=OPEN, last_turn=turn)
            else:
                if a.get("statement"):
                    cur.statement = a["statement"]
                if a.get("probe"):
                    cur.probe = a["probe"]
                if a.get("importance") is not None:
                    cur.importance = float(a["importance"])
                if a.get("cost") is not None:
                    cur.cost = int(a["cost"])
                if a.get("target"):
                    cur.target = list(a["target"])
                cur.last_turn = turn
        return self

    # ---- resolution (closing is first-class) -----------------------------
    def close(self, claim_id: str, outcome: str, turn: int = 0,
              observed_credence: float = 0.9,
              discriminating: bool = True) -> Optional[StoredClaim]:
        """Resolve a claim from an OBSERVATION (proven or refuted).  Closing is
        first-class so the frontier actually shrinks; an OBSERVED resolution
        clears needs_recheck and promotes credence.

        A REFUTATION requires a DISCRIMINATING observation -- one whose result
        actually bears on the hypothesis.  A NON-discriminating probe (no signal
        that could decide the claim -- e.g. only the HUD ticked, or an action
        probe was used to settle an IDENTITY question that is a static fact) is
        INCONCLUSIVE: it records that a probe was spent but does NOT refute and
        leaves the claim OPEN.  Absence of an interaction-effect is never proof
        that distinct things are equivalent.  (Proving is unaffected -- a positive
        observation is self-discriminating.)"""
        c = self.claims.get(claim_id)
        if c is None:
            return None
        if outcome == REFUTED and not discriminating:
            self.note_probed(claim_id, turn)       # inconclusive -> stays OPEN
            return c
        c.times_probed += 1
        c.last_turn = turn
        c.provenance = cc.OBSERVED
        c.credence = cc.clamp(cc.OBSERVED, observed_credence)
        c.needs_recheck = False
        c.status = REFUTED if outcome == REFUTED else PROVEN
        return c

    def note_probed(self, claim_id: str, turn: int = 0) -> None:
        """Record that a probe was spent on a claim even if it stayed OPEN
        (inconclusive) -- so the un-probed-first tiebreak rotates on."""
        c = self.claims.get(claim_id)
        if c is not None:
            c.times_probed += 1
            c.last_turn = turn

    # ---- coverage (no salient entity left un-authored) -------------------
    def is_covered(self, entity_name: str, level_signature=None) -> bool:
        """True if a claim ACTIVE for the current level targets this entity -- so
        the function-coverage layer never seeds a duplicate.  Scope-aware: a
        positioned claim from a DIFFERENT level (e.g. a persisted lc1 claim
        reloaded while playing lc0) does NOT count as covering the entity here,
        so coverage still authors a fresh, ACTIVE claim for the current layout
        instead of being silently suppressed by stale off-level state."""
        return any(entity_name in (c.target or [])
                   and self._active_for(c, level_signature)
                   for c in self.claims.values())

    def has_confirmed_function(self, entity_name: str, level_signature=None) -> bool:
        """True if a claim ACTIVE for the current level AND targeting this entity
        has been RESOLVED (proven or refuted) -- i.e. its function/affordance is
        observed on THIS layout, not just guessed or proven on a different one."""
        return any(entity_name in (c.target or []) and c.status != OPEN
                   and self._active_for(c, level_signature)
                   for c in self.claims.values())

    # ---- ranking (substrate decides what to attack) ----------------------
    @staticmethod
    def _active_for(c, level_signature) -> bool:
        """A claim is active for the CURRENT level if it is CROSS-level, or it is
        a LEVEL claim whose signature matches (or no level filter is given).  So
        a positioned claim from a DIFFERENT level (e.g. a persisted lc=1 claim
        reloaded while playing lc=0) is NOT probed on the wrong layout.  Any
        non-LEVEL scope (cross-level, cross-game structural) is always active."""
        return (level_signature is None or c.scope != LEVEL
                or c.level_signature == level_signature)

    def open_claims(self, level_signature=None) -> list:
        """OPEN claims active for this level, plus PROVEN ones flagged
        needs_recheck (assume-but-verify) -- the things still worth a probe."""
        return [c for c in self.claims.values()
                if (c.status == OPEN or (c.status == PROVEN and c.needs_recheck))
                and self._active_for(c, level_signature)]

    def rank(self, level_signature=None) -> list:
        """Open (level-active) claims by descending value-of-information;
        un-probed first on ties."""
        return sorted(self.open_claims(level_signature),
                      key=lambda c: (c.value_of_information(), c.times_probed == 0),
                      reverse=True)

    def next_probe(self, level_signature=None) -> Optional[StoredClaim]:
        """The highest-value open claim (active for this level) that carries a
        discriminating probe action, or None when none is actionable."""
        for c in self.rank(level_signature):
            if c.probe:
                return c
        return None

    # ---- scoping on a level transition -----------------------------------
    def carry_to_new_level(self, new_signature: str) -> "ClaimStore":
        """Scope the store to a new level: DROP level-scoped claims from a
        different layout (positioned -- meaningless here); KEEP cross-level
        claims, but knock any PROVEN one down to GUESSED + needs_recheck so it
        is believed yet scheduled for one cheap reconfirmation on the new
        level.  REFUTED and OPEN claims carry unchanged."""
        drop = []
        for cid, c in self.claims.items():
            if c.scope == LEVEL and c.level_signature != new_signature:
                drop.append(cid)
            elif c.scope != LEVEL and c.status == PROVEN:
                # cross-level + cross-game structural priors: believed but
                # UNVERIFIED on the new level -> recheck before relying on them.
                c.needs_recheck = True
                c.provenance = cc.GUESSED
                c.credence = cc.clamp(cc.GUESSED, c.credence)
        for cid in drop:
            del self.claims[cid]
        return self

    # ---- persistence (KB-backed, so a re-run reloads the frontier) -------
    def to_dict(self) -> dict:
        return {"game_id": self.game_id,
                "claims": {k: asdict(v) for k, v in self.claims.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "ClaimStore":
        store = cls(game_id=d.get("game_id", ""))
        for k, rd in (d.get("claims") or {}).items():
            store.claims[k] = StoredClaim(**rd)
        return store

    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "ClaimStore":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            return cls()

    @staticmethod
    def kb_file(game_id: str) -> Path:
        """KB location for a game's persistent claim frontier."""
        try:
            import kb_paths
            root = kb_paths.kb_root()
        except Exception:
            root = Path(".tmp/kb")
        return Path(root) / "claim_store" / f"{game_id}.json"

    @classmethod
    def load_for_game(cls, game_id: str) -> "ClaimStore":
        store = cls.load(cls.kb_file(game_id))
        store.game_id = game_id
        return store

    def save_for_game(self) -> None:
        if self.game_id:
            self.save(self.kb_file(self.game_id))
