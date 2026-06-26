"""Provenance-tagged credence + propagation + verify-before-gating.

WHY
---
Impossibility / dead-end claims kept being treated as load-bearing facts when
they were actually GUESSES from an unverified model ("the right trio is
immovable" -> wrongly pruned a winning plan; symmetrically a guessed no-op got
blindly trusted). The fix: a claim's credence is set by its PROVENANCE, an
inferred claim can never out-credence its premises (and a guess-on-a-guess
decays), and -- crucially -- only an OBSERVED claim may hard-gate (prune/block).
Everything weaker SURFACES but must be PROBED before it is load-bearing.

This is game-agnostic pure logic; stores (per_game_lessons, win-condition
hypotheses, operator_kb, blocking claims) tag their claims with a provenance and
route gating decisions through `gate_decision`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Provenance, strongest -> weakest.
OBSERVED = "observed"   # read directly from a frame/delta this instance
INFERRED = "inferred"   # derived by reasoning from observed premises
GUESSED = "guessed"     # derived from a model/assumption, not grounded

# A claim of a given provenance cannot exceed this credence until an OBSERVATION
# promotes it. Inference never manufactures certainty.
CEILING = {OBSERVED: 1.0, INFERRED: 0.70, GUESSED: 0.40}

# Per-inference-step reliability discount (so a chain decays geometrically).
DEFAULT_INFERENCE_RELIABILITY = 0.85

# A claim may HARD-GATE (prune a plan / block an action) only if it is OBSERVED
# and at least this credent. Anything else can surface but must be probed.
GATING_THRESHOLD = 0.80

# Gate decisions.
HARD = "hard"            # trusted enough to prune/block
PROBE = "probe_first"    # surface it, but verify before relying on it
IGNORE = "ignore"        # too weak to even surface prominently
PROBE_FLOOR = 0.15       # below this an unverified claim is barely worth raising


@dataclass
class Claim:
    claim_id: str
    statement: str
    provenance: str = GUESSED
    credence: float = 0.0
    premises: List[str] = field(default_factory=list)
    verified: bool = False          # observation-confirmed on the current instance


def ceiling_for(provenance: str) -> float:
    return CEILING.get(provenance, CEILING[GUESSED])


def clamp(provenance: str, credence: float) -> float:
    return max(0.0, min(float(credence), ceiling_for(provenance)))


def provenance_of_inference(premise_provenances: List[str]) -> str:
    """An inference is only as grounded as its weakest premise: if any premise
    is a GUESS, the conclusion is guess-grade; otherwise it is INFERRED."""
    if any(p == GUESSED for p in premise_provenances):
        return GUESSED
    return INFERRED


def propagate_credence(premises: List[Claim],
                       reliability: float = DEFAULT_INFERENCE_RELIABILITY) -> float:
    """Credence of a claim inferred from `premises`.
    = min(premise credences) * reliability, then capped by the ceiling of the
    inference's provenance. With no stated premises the inference is a bare
    guess. Inferred-from-guess therefore lands EVEN LOWER (guess ceiling)."""
    if not premises:
        return CEILING[GUESSED] * reliability
    base = min(p.credence for p in premises)
    prov = provenance_of_inference([p.provenance for p in premises])
    return clamp(prov, base * reliability)


def observed_claim(claim_id: str, statement: str, credence: float = 0.9) -> Claim:
    return Claim(claim_id, statement, OBSERVED, clamp(OBSERVED, credence),
                 premises=[], verified=True)


def guess_claim(claim_id: str, statement: str, credence: float = CEILING[GUESSED]) -> Claim:
    return Claim(claim_id, statement, GUESSED, clamp(GUESSED, credence))


def inferred_claim(claim_id: str, statement: str, premises: List[Claim],
                   reliability: float = DEFAULT_INFERENCE_RELIABILITY) -> Claim:
    prov = provenance_of_inference([p.provenance for p in premises]) if premises else GUESSED
    return Claim(claim_id, statement, prov,
                 propagate_credence(premises, reliability),
                 premises=[p.claim_id for p in premises])


def mark_verified(claim: Claim, observed_credence: float = 0.9) -> Claim:
    """An observation confirmed this claim on the current instance -> promote it
    to OBSERVED so it may now gate. This is the ONLY way credence rises to a
    gating level."""
    claim.provenance = OBSERVED
    claim.credence = clamp(OBSERVED, observed_credence)
    claim.verified = True
    return claim


def can_gate(claim: Claim) -> bool:
    """May this claim HARD-prune/block? Only if OBSERVED + verified + credent."""
    return (claim.provenance == OBSERVED and claim.verified
            and claim.credence >= GATING_THRESHOLD)


def gate_decision(claim: Claim) -> str:
    """HARD (trusted), PROBE (surface but verify first), or IGNORE (too weak)."""
    if can_gate(claim):
        return HARD
    if claim.credence >= PROBE_FLOOR:
        return PROBE
    return IGNORE


def needs_verification(claim: Claim) -> bool:
    """A claim being used as load-bearing (a dead-end / impossibility) but not
    yet gating-eligible -> a cheap probe is required before relying on it."""
    return gate_decision(claim) == PROBE


def format_gate_surface(claim: Claim) -> str:
    """How a dead-end / impossibility claim should be surfaced given its
    provenance — hard avoid vs. PROBE-FIRST."""
    d = gate_decision(claim)
    if d == HARD:
        return (f"ESTABLISHED (observed, credence {claim.credence:.2f}): "
                f"{claim.statement} — treat as a hard constraint.")
    if d == PROBE:
        return (f"UNVERIFIED {claim.provenance} claim (credence "
                f"{claim.credence:.2f}) — {claim.statement}. PROBE this cheaply "
                f"before relying on it; do NOT prune a plan or assert "
                f"impossibility on its basis until an observation confirms it.")
    return ""
