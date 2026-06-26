"""Credence — heuristic belief-strength tracking for hypotheses.

A :class:`Credence` bundles together four related quantities:

* ``point`` — 0..1 best-estimate probability that the hypothesis is true.
  What the planner reads to decide if a hypothesis is committed.
* ``evidence_weight`` — unbounded counter of supporting observations.
  Distinguishes "confirmed three times" from "confirmed three hundred times";
  slows demotion for well-supported claims.
* ``last_confirmed`` / ``last_contradicted`` — freshness markers used by
  the staleness decay rule.
* ``competing`` — list of hypothesis IDs sharing the same canonical key.
  The :class:`HypothesisStore` populates this; evidence relevant to one
  competitor is always relevant to the others.

The update rules below are deliberately simple and heuristic (not
Bayesian).  Using a point estimate with a learning rate is well-suited
to a system that accumulates many hypotheses from cheap sources and
retires most of them — a full Bayesian model would require per-claim-type
likelihood functions that we don't yet have calibrated.

All functions are PURE — they return a new :class:`Credence` rather than
mutating in place.  This keeps the WorldState trivially snapshottable and
makes the update rules easier to test.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional

from .config import CredenceConfig


# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Credence:
    """Immutable credence snapshot.  Use the module-level update functions
    to produce new Credences rather than mutating an existing one."""

    point:             float             = 0.5
    evidence_weight:   float             = 0.0
    last_confirmed:    int               = -1
    last_contradicted: Optional[int]     = None
    competing:         tuple = ()        # tuple of hypothesis IDs (immutable)

    # --- derived predicates (use config thresholds, not globals) ---

    def is_committed(self, cfg: CredenceConfig) -> bool:
        return self.point >= cfg.commit_threshold

    def is_abandoned(self, cfg: CredenceConfig) -> bool:
        return self.point <= cfg.abandon_threshold


# ---------------------------------------------------------------------------
# Update rules
# ---------------------------------------------------------------------------


def update_on_support(c: Credence,
                      step: int,
                      cfg: CredenceConfig,
                      source_strength: float = 1.0) -> Credence:
    """Return a new Credence reflecting supporting evidence seen at ``step``.

    ``source_strength`` modulates the learning rate.  Strong evidence
    (e.g. an observation directly matching the claim's effect condition)
    yields ``source_strength=1.0``; weaker circumstantial evidence should
    pass a smaller value.  Values outside [0,1] are clamped.
    """
    s = max(0.0, min(1.0, source_strength))
    delta = (1.0 - c.point) * cfg.learning_rate * s
    return replace(
        c,
        point=min(1.0, c.point + delta),
        evidence_weight=c.evidence_weight + s,
        last_confirmed=step,
    )


def update_on_contradict(c: Credence,
                         step: int,
                         cfg: CredenceConfig,
                         strength: float = 1.0) -> Credence:
    """Return a new Credence reflecting contradicting evidence at ``step``.

    Contradiction lowers ``point`` multiplicatively: well-supported
    hypotheses (high point) drop more on each contradiction, but take
    many contradictions to collapse entirely.  ``evidence_weight`` is
    intentionally NOT reset — a well-evidenced claim should not instantly
    abandon on a single surprise, but it should start losing credibility.
    """
    s = max(0.0, min(1.0, strength))
    delta = c.point * cfg.learning_rate * s
    return replace(
        c,
        point=max(0.0, c.point - delta),
        last_contradicted=step,
    )


def apply_decay(c: Credence, current_step: int, cfg: CredenceConfig) -> Credence:
    """Apply freshness decay.

    Hypotheses confirmed long ago but never recently re-confirmed slowly
    lose credence, reflecting that the world may have changed.  Decay
    only begins after ``cfg.staleness_window`` steps of silence, so
    recently-confirmed claims are unaffected.
    """
    if c.last_confirmed < 0:
        return c
    staleness = current_step - c.last_confirmed
    if staleness <= cfg.staleness_window:
        return c
    decay = cfg.decay_per_step * (staleness - cfg.staleness_window)
    if decay <= 0.0:
        return c
    return replace(c, point=max(0.0, c.point - decay))


def link_competitor(c: Credence, competitor_id: str) -> Credence:
    """Return a new Credence with ``competitor_id`` added to the competing
    list (deduplicated)."""
    if competitor_id in c.competing:
        return c
    return replace(c, competing=tuple(list(c.competing) + [competitor_id]))


def unlink_competitor(c: Credence, competitor_id: str) -> Credence:
    """Return a new Credence with ``competitor_id`` removed from the
    competing list."""
    if competitor_id not in c.competing:
        return c
    return replace(c, competing=tuple(x for x in c.competing if x != competitor_id))
