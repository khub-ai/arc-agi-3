"""Substrate-agnostic action-semantics registry.

Observes agent-position deltas across turns and learns the most-
likely cell delta for each action_id.  Replaces the hardcoded
ACTION1-4 → cardinal table that bp35-tuned the validator's V1
inspector and missed every other game's movement model.

This is the action-semantics-discovery sub-pass of the
aggregative role per SPEC_governor.md § Aggregative role.  Each
recorded observation is one piece of evidence; the registry
promotes a candidate to "confirmed" only when corroboration
clears the threshold.

No assumptions about which action_ids exist or what they do.
Cold start: every action_id is unknown; V1 silently records
observations without firing.  After CORROBORATION_THRESHOLD
matching observations, the action_id is "confirmed" with that
delta and V1 starts using it to check subsequent motion.

The registry is per-session.  Cross-session promotion to a
contract entry happens through the aggregative role's standard
machinery once a corroborated delta survives a second trial.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional


# Number of matching (action_id, delta) observations required before
# a delta is treated as confirmed.  Small enough to learn quickly
# from a 20-turn sweep, large enough that a single noisy observation
# can't poison the registry.
CORROBORATION_THRESHOLD = 3


# Maximum delta magnitude considered.  Most ARC-AGI-3 games move by
# at most 1 cell per action; anything larger is treated as a
# multi-cell slide or a respawn and is recorded as a separate
# delta bucket.
MAX_DELTA_MAGNITUDE = 7


@dataclass
class ActionStats:
    """Per-action accumulator."""

    action_id: int
    observations: Counter = field(default_factory=Counter)   # (dr,dc) → count
    total_seen: int = 0
    confirmed_delta: Optional[tuple[int, int]] = None

    def record(self, delta: tuple[int, int]) -> None:
        if abs(delta[0]) > MAX_DELTA_MAGNITUDE or \
                abs(delta[1]) > MAX_DELTA_MAGNITUDE:
            return
        self.observations[delta] += 1
        self.total_seen += 1
        # Promote if the modal delta has at least CORROBORATION_THRESHOLD
        # observations AND outweighs all others.
        if self.observations:
            top_delta, top_count = self.observations.most_common(1)[0]
            second_count = (
                self.observations.most_common(2)[1][1]
                if len(self.observations) > 1 else 0
            )
            if top_count >= CORROBORATION_THRESHOLD \
                    and top_count > second_count:
                self.confirmed_delta = top_delta

    def to_dict(self) -> dict:
        return {
            "action_id": self.action_id,
            "total_seen": self.total_seen,
            "confirmed_delta": (
                list(self.confirmed_delta)
                if self.confirmed_delta is not None else None
            ),
            "observations": [
                {"delta": list(k), "count": v}
                for k, v in self.observations.most_common()
            ],
        }


class ActionSemanticsRegistry:
    """Substrate-agnostic learner for per-action cell deltas.

    The registry never assumes any particular action_id means
    anything specific.  It only records correlations between
    observed `(action_id, agent_pos_delta)` pairs.

    Lifecycle:

      - record_motion(action_id, prev_pos, curr_pos)  — call after
        every env.step where the agent's position is known both
        before and after.  The registry computes the delta and
        updates per-action statistics.
      - best_delta(action_id) → optional (dr, dc) — returns the
        confirmed delta if corroboration threshold has been met,
        else None.
      - is_translate(action_id) — True if the confirmed delta is
        non-zero; the action moves the agent.

    Designed to live alongside Validator: one Registry per session,
    shared by V1 and the action-record builder in sequence.py.
    """

    def __init__(self) -> None:
        self._stats: dict[int, ActionStats] = {}

    def record_motion(
        self,
        action_id: int,
        prev_pos: Optional[tuple[int, int]],
        curr_pos: Optional[tuple[int, int]],
    ) -> Optional[tuple[int, int]]:
        """Record one observation.  Returns the observed delta, or
        None if either position is missing.  Caller is responsible
        for ensuring the call corresponds to a single env.step.
        """
        if prev_pos is None or curr_pos is None:
            return None
        delta = (
            curr_pos[0] - prev_pos[0],
            curr_pos[1] - prev_pos[1],
        )
        stats = self._stats.setdefault(action_id, ActionStats(action_id))
        stats.record(delta)
        return delta

    def best_delta(
        self, action_id: int,
    ) -> Optional[tuple[int, int]]:
        """Confirmed delta for action_id, or None if not yet learned.
        """
        stats = self._stats.get(action_id)
        if stats is None:
            return None
        return stats.confirmed_delta

    def is_translate(self, action_id: int) -> bool:
        """True iff the confirmed delta is non-zero."""
        d = self.best_delta(action_id)
        return d is not None and d != (0, 0)

    def all_confirmed(self) -> dict[int, tuple[int, int]]:
        """Return every action_id with a confirmed delta."""
        return {
            aid: s.confirmed_delta
            for aid, s in self._stats.items()
            if s.confirmed_delta is not None
        }

    def snapshot(self) -> dict:
        """Diagnostic snapshot for tracing.  Includes every action_id
        seen, even those without a confirmed delta yet.
        """
        return {
            "n_actions_seen": len(self._stats),
            "n_confirmed": sum(
                1 for s in self._stats.values()
                if s.confirmed_delta is not None
            ),
            "per_action": [s.to_dict() for s in
                            sorted(self._stats.values(),
                                   key=lambda s: s.action_id)],
        }
