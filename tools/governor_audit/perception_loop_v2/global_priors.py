"""Global cross-game action-semantic priors.

Cross-game transfer is the highest-leverage shortcut for "similar
but different" games.  When the system has played bp35 and learned
that ACTION1 moves the agent UP, that knowledge should jump-start
sk48: on turn 1, the system can already PLAN multi-step motion
using a UP-direction prior, only needing to verify (not re-derive)
that the new game obeys the same mapping.

This module persists action-effect mappings across SESSIONS and
GAMES.  It writes to a single file (default
``<repo>/.tmp/global_action_priors.json``) that survives any number
of game runs.

DATA FLOW

  END OF LEVEL  (driver explicitly calls)
    1. extract_observations(world) walks promoted MechanicHypothesis
       records; for each one with a parseable
       ``action=ACTION_N -> effect=...`` shape, records
       (action_name, effect_signature) as observed in this game.
    2. Observations are MERGED into the global file: a (action,
       effect) pair seen in N out of M games carries credence
       N/M; a pair that's been contradicted in a game (i.e. the
       action was promoted with a DIFFERENT effect in that game)
       decays the credence further.

  START OF LEVEL  (driver explicitly calls)
    1. load() reads the global file (returns empty store on
       first-ever run).
    2. seed_world_from_priors(world, available_actions) walks the
       store; for every (action, effect) prior with sufficient
       support, creates a MechanicHypothesis with credence
       proportional to its cross-game support, marks it
       seeded=True (vs promoted=True), and appends to
       world.mechanic_hypotheses.

CREDENCE TRANSFER POLICY

  A prior is seeded at credence MIN(base_credence, support_ratio)
  where support_ratio = n_games_observed / max(1,
  n_games_observed + n_games_contradicted).  Default
  base_credence is 0.7 (high enough to bias the planner toward
  expected behavior, low enough to be falsified within one or
  two turns if the new game disagrees -- the miner's -0.3 per
  contradiction quickly demotes it).

GAME-AGNOSTIC: action names + effect signatures flow through as
opaque strings.  No hardcoded action semantics; no per-game
overrides.  The store treats every game equally as one
"observation source" and aggregates by string match.

PRIME DIRECTIVE: nothing in this module references a specific
game.  All inputs come from the WorldKnowledge open
vocabulary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from world_knowledge import WorldKnowledge, MechanicHypothesis


DEFAULT_PRIORS_PATH = Path(
    ".tmp/global_action_priors.json"
)


# ---------------------------------------------------------------------------
# Record types
# ---------------------------------------------------------------------------


@dataclass
class ActionSemanticPrior:
    """One (action, effect) observation aggregated across games.

    ``games_observed`` is the set of game_ids where this exact
    (action, effect) was promoted.  ``games_contradicted`` is the
    set of game_ids where the SAME action was promoted but with a
    DIFFERENT effect (one game can appear in many priors -- it's
    contradicted for some, supports others).
    """
    action_name: str
    effect: str
    games_observed: list[str] = field(default_factory=list)
    games_contradicted: list[str] = field(default_factory=list)
    first_observed_iso: str = ""
    last_observed_iso: str = ""

    @property
    def support_ratio(self) -> float:
        n_obs = len(self.games_observed)
        n_contra = len(self.games_contradicted)
        if n_obs + n_contra == 0:
            return 0.0
        return n_obs / (n_obs + n_contra)

    @property
    def aggregate_credence(self) -> float:
        """Credence at which this prior should be seeded into a
        new game.  Caps at 0.95 even with perfect cross-game
        support so the miner can still demote if the new game
        truly behaves differently.

        Single-game perfect-support priors seed at 0.85 — high
        enough to be promoted=True for the bridge to consume on
        turn 1, low enough that a single contradicting
        observation (which the miner penalises at -0.3) drops it
        below the planner-trust threshold.  Multi-game perfect
        support climbs toward 0.95.

        Contradicted priors are penalised proportionally:
        ``support_ratio * confidence_factor`` where
        confidence_factor scales from 0.85 (one observation) to
        0.95 (many observations)."""
        n_obs = len(self.games_observed)
        if n_obs == 0:
            return 0.0
        # confidence_factor: 0.85 at n=1, asymptote toward 0.95
        confidence_factor = 0.85 + 0.10 * (1.0 - 0.9 ** (n_obs - 1))
        return min(0.95, self.support_ratio * confidence_factor)


@dataclass
class GlobalActionPriors:
    """Persistent cross-game store of action-semantic priors."""
    priors: list[ActionSemanticPrior] = field(default_factory=list)
    last_updated_iso: str = ""
    n_games_contributed: int = 0
    contributing_games: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def find(self, action: str, effect: str
              ) -> Optional[ActionSemanticPrior]:
        for p in self.priors:
            if p.action_name == action and p.effect == effect:
                return p
        return None

    def for_action(self, action: str) -> list[ActionSemanticPrior]:
        return [p for p in self.priors if p.action_name == action]

    # ------------------------------------------------------------------
    # Updating from a finished game's world model
    # ------------------------------------------------------------------

    def update_from_world(self, world: WorldKnowledge) -> tuple[int, int]:
        """Walk world.mechanic_hypotheses; for every PROMOTED one with
        a parseable action+effect shape, record an observation in
        the store.  Also marks contradictions for previously-seen
        (action, effect) pairs that did NOT promote in this game.

        Returns (n_new_priors, n_updated_priors)."""
        game_id = world.game_id
        observed_in_game: dict[str, set[str]] = {}
            # action -> set of effects promoted in THIS game

        for h in world.mechanic_hypotheses:
            if not h.promoted:
                continue
            action = _parse_action_from_trigger(h.trigger)
            if action is None:
                continue
            observed_in_game.setdefault(action, set()).add(h.effect)

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        n_new = 0
        n_updated = 0
        for action, effects in observed_in_game.items():
            for effect in effects:
                p = self.find(action, effect)
                if p is None:
                    self.priors.append(ActionSemanticPrior(
                        action_name=action, effect=effect,
                        games_observed=[game_id],
                        first_observed_iso=now,
                        last_observed_iso=now,
                    ))
                    n_new += 1
                else:
                    if game_id not in p.games_observed:
                        p.games_observed.append(game_id)
                        p.last_observed_iso = now
                        n_updated += 1
                    # If this game previously contradicted this
                    # prior, REMOVE the contradiction now that we
                    # see it observed.
                    if game_id in p.games_contradicted:
                        p.games_contradicted.remove(game_id)

            # Mark contradictions: for any prior with this action
            # but a DIFFERENT effect, the current game saw this
            # action with effects {effects}; if a prior effect is
            # NOT in {effects} for the same action, this game is
            # a contradiction for that prior.
            for p in self.for_action(action):
                if p.effect in effects:
                    continue
                if game_id in p.games_contradicted:
                    continue
                if game_id in p.games_observed:
                    continue  # game observed THIS effect, doesn't contradict
                p.games_contradicted.append(game_id)
                n_updated += 1

        if game_id not in self.contributing_games:
            self.contributing_games.append(game_id)
            self.n_games_contributed += 1
        self.last_updated_iso = now
        return n_new, n_updated

    # ------------------------------------------------------------------
    # Seeding a fresh game's WorldKnowledge
    # ------------------------------------------------------------------

    def seed_world_from_priors(self, world: WorldKnowledge,
                                 available_actions: list[str],
                                 *,
                                 max_credence: float = 0.9,
                                 min_support_ratio: float = 0.5
                                 ) -> int:
        """Pre-populate ``world.mechanic_hypotheses`` with cross-game
        priors.  Returns the number of priors seeded.

        Only seeds priors:
          - whose action is in ``available_actions`` (no point
            seeding ACTION5 if the new game doesn't expose it)
          - whose support_ratio is at least ``min_support_ratio``
            (avoid seeding contested mappings the new game will
            just have to re-test)
          - that aren't already present in
            world.mechanic_hypotheses (idempotent seeding)

        Seeded hypotheses are marked promoted=True when their
        credence reaches 0.85+ so the bridge_promoted_hypotheses
        path picks them up immediately and the planner can use
        them on turn 1.

        ``max_credence`` is a UPPER bound on the seeded credence
        — set it lower than 0.85 to deliberately keep priors as
        soft hints that won't auto-promote (useful for measuring
        the prior-only baseline).
        """
        n_seeded = 0
        action_set = set(available_actions)
        existing_keys = {
            (h.trigger, h.effect)
            for h in world.mechanic_hypotheses
        }
        for p in self.priors:
            if p.action_name not in action_set:
                continue
            if p.support_ratio < min_support_ratio:
                continue
            trigger = f"action={p.action_name}"
            key = (trigger, p.effect)
            if key in existing_keys:
                continue
            # Seed at the prior's aggregate credence, capped above
            # by max_credence (default 0.9) so the planner can
            # demote even high-confidence priors when the new
            # game contradicts.
            cred = min(max_credence, p.aggregate_credence)
            if cred < 0.5:
                continue
            world.mechanic_hypotheses.append(MechanicHypothesis(
                hypothesis_id=f"H[seeded:{p.action_name} -> {p.effect}]",
                trigger=trigger,
                effect=p.effect,
                credence=cred,
                supporting_observations=[],
                contradicting_observations=[],
                promoted=(cred >= 0.85),
            ))
            n_seeded += 1
        if n_seeded:
            inh = (world.inherited_from or "")
            world.inherited_from = (
                f"{inh}; " if inh else ""
            ) + f"global_priors:{n_seeded}-actions"
        return n_seeded

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path = DEFAULT_PRIORS_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_PRIORS_PATH
              ) -> "GlobalActionPriors":
        if not path.exists():
            return cls()
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return cls()
        priors = [ActionSemanticPrior(**p) for p in d.get("priors", [])]
        return cls(
            priors=priors,
            last_updated_iso=d.get("last_updated_iso", ""),
            n_games_contributed=d.get("n_games_contributed", 0),
            contributing_games=d.get("contributing_games", []),
        )


# ---------------------------------------------------------------------------
# Helpers (mirror the parsing used in planner_integration)
# ---------------------------------------------------------------------------


def _parse_action_from_trigger(trigger: str) -> Optional[str]:
    """Extract the action name from a MechanicHypothesis trigger
    string.  Mirrors planner_integration._parse_action_from_trigger
    but duplicated here so this module has no dependency on the
    planner bridge (the priors store is independent of whether
    the bridge runs)."""
    parts = [p.strip() for p in trigger.split(",")]
    for part in parts:
        if part.startswith("action="):
            return part[len("action="):]
    return None


# ---------------------------------------------------------------------------
# Convenience helpers for the driver
# ---------------------------------------------------------------------------


def load_and_seed(world: WorldKnowledge,
                   available_actions: list[str],
                   priors_path: Path = DEFAULT_PRIORS_PATH,
                   *,
                   max_credence: float = 0.9,
                   ) -> tuple[GlobalActionPriors, int]:
    """One-shot: load the global priors and seed the new game's
    world model.  Returns (the loaded priors object, number of
    seeded hypotheses).

    The returned priors object is mutable -- the driver can update
    it at end-of-game and call save() to persist.
    """
    priors = GlobalActionPriors.load(priors_path)
    n_seeded = priors.seed_world_from_priors(
        world, available_actions, max_credence=max_credence,
    )
    return priors, n_seeded


def update_and_save(priors: GlobalActionPriors,
                     world: WorldKnowledge,
                     priors_path: Path = DEFAULT_PRIORS_PATH
                     ) -> tuple[int, int]:
    """One-shot end-of-game: extract observations from world,
    merge into priors, persist.  Returns (n_new_priors, n_updated_priors)."""
    n_new, n_updated = priors.update_from_world(world)
    priors.save(priors_path)
    return n_new, n_updated
