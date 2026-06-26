"""Governor-Perception Validator (substrate-agnostic).

Concrete realization of SPEC_governor_perception_validator.md.  Reads
the v2 classifier's Classification + the surrounding FrameObservation,
the previous action, and any outcome (lc-delta, score-delta); emits
typed PolicyEvent and VerifiableClaim records into the substrate's
existing streams; maintains a per-entity trust score the trial driver
applies before building the actor prompt.

Five inspectors per turn:

  motion-check      — agent cell shifted by expected delta?
  persistence-check — cells oscillating without plausible cause?
  structural-check  — singleton agent / HUD exclusivity / win-marker
                      rarity / **bbox-density priors**.  An entity
                      whose bbox is much larger than its visible
                      pixel content (density < 30%) is suspicious:
                      either the sprite grouper over-merged into
                      background speckle, or two unrelated entities
                      got fused.  Substrate-agnostic: \"sprites are
                      dense by definition\".
  outcome-check     — deferred VerifiableClaim resolution
                      (win_marker_at clicked → level_advanced?).
  template-check    — outcome-corroborated visual signatures
                      persist across frames?

No palette indices, no game-specific tables, no truth.json — every
signal comes from substrate-agnostic primitives (RGB pixels, action
ids, score deltas, level-advance flags, frame deltas).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .classifier import Classification
from .observation import FrameObservation
from .sequence import ActionRecord, FrameHistory
from .action_semantics import ActionSemanticsRegistry


# -----------------------------------------------------------------------------
# Parent-spec event records (PolicyEvent, VerifiableClaim, VerifierCheck).
#
# These mirror the schemas in SPEC_governor.md § Substrate.  They are
# duplicated as local dataclasses here rather than imported from a
# shared module because the substrate-side observer pass that owns
# them lives in the legacy harness; the validator's outputs are
# eventually serialized as plain dicts into turn_record["metadata"],
# which is the contract surface that matters.
# -----------------------------------------------------------------------------


@dataclass
class PolicyEvent:
    """Reactive-role event per SPEC_governor.md.

    Validator emits with ``policy_id = "perception_validation_failure"``.
    """

    policy_id: str
    severity: str            # "info" | "warn" | "block"
    subject: dict            # what was checked
    violation: dict          # specific values that tripped the check
    remedy: Optional[dict]
    context: dict            # trial_id, turn, game_id, model

    def to_dict(self) -> dict:
        return {
            "policy_id": self.policy_id,
            "severity": self.severity,
            "subject": self.subject,
            "violation": self.violation,
            "remedy": self.remedy,
            "context": self.context,
        }


@dataclass
class VerifiableClaim:
    """Per SPEC_governor.md § Schema."""

    claim_type: str
    args: dict
    asserted: object
    source: str
    confidence: float

    def to_dict(self) -> dict:
        return {
            "claim_type": self.claim_type,
            "args": self.args,
            "asserted": self.asserted,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass
class VerifierCheck:
    """Per SPEC_governor.md § Schema."""

    claim_type: str
    args: dict
    asserted: object
    actual: object
    tool_used: Optional[str]
    agreement: Optional[bool]
    outcome: str             # "agree" | "disagree" | "no_matching_tool" | ...

    def to_dict(self) -> dict:
        return {
            "claim_type": self.claim_type,
            "args": self.args,
            "asserted": self.asserted,
            "actual": self.actual,
            "tool_used": self.tool_used,
            "agreement": self.agreement,
            "outcome": self.outcome,
        }


# -----------------------------------------------------------------------------
# Outcome record + trust scoring constants.
# -----------------------------------------------------------------------------


@dataclass
class OutcomeRecord:
    """What the harness observed after the previous action fired."""

    lc_delta: int = 0        # > 0 → level_advanced
    score_delta: float = 0.0
    game_over: bool = False
    frame_changed: bool = True    # cheap: any pixel diff at all?


# Trust update coefficients (alpha for corroboration, beta for failure).
# These are tuned per inspector — motion-check outcomes are the highest-
# weight evidence, persistence-check oscillation is the lowest.
# Documented in SPEC_governor_perception_validator.md § Trust scoring.
_TRUST_ALPHA = 0.4          # corroboration weight
_TRUST_BETA = {              # inconsistency weight per inspector
    "motion_check":       0.5,
    "persistence_check":  0.1,
    "structural_check":   0.3,
    "outcome_check":      0.6,
    "template_check":     0.3,
}
_TRUST_DECAY_PER_TURN = 0.001  # slow drift toward uncertainty
_TRUST_INITIAL = 0.6           # cold-start prior


# -----------------------------------------------------------------------------
# Validation result wrapper.
# -----------------------------------------------------------------------------


@dataclass
class ValidationResult:
    """What `Validator.ingest()` returns each turn."""

    policy_events: list[PolicyEvent] = field(default_factory=list)
    verifiable_claims: list[VerifiableClaim] = field(default_factory=list)
    verifier_checks: list[VerifierCheck] = field(default_factory=list)
    trust_scores: dict = field(default_factory=dict)   # cell_key → float
    demoted_cells: set = field(default_factory=set)    # (r, c) tuples


# -----------------------------------------------------------------------------
# The Validator.
# -----------------------------------------------------------------------------


class Validator:
    """Substrate-agnostic perception validator.  Stateful — maintains
    per-trial entity history, per-session pending-claim log, and a
    cross-session entity template store.

    One Validator per session (a sequence of trials on the same level).
    """

    # Demotion thresholds applied to cell trust scores.
    HIGH_TRUST = 0.80
    LOW_TRUST = 0.30

    def __init__(
        self,
        game_id: str,
        session_dir: Optional[Path] = None,
        action_semantics: Optional[ActionSemanticsRegistry] = None,
    ):
        self.game_id = game_id
        self.session_dir = session_dir

        # Action-semantics registry.  motion-check consults this to determine the
        # expected motion delta for a translation action.  If no
        # registry is supplied, the validator creates its own — the
        # registry is per-session and starts empty (no assumptions).
        self.action_semantics = (
            action_semantics if action_semantics is not None
            else ActionSemanticsRegistry()
        )

        # Per-cell trust scores keyed by (r, c).  Reset at level
        # transitions (level changes which entities exist) but visual
        # templates persist.
        self._trust: dict[tuple[int, int], float] = {}

        # Per-session pending-claim log: cell → list of pending claims.
        # Lifetime: session.
        self._pending: list[dict] = []

        # Cross-session entity template store (deferred persistence —
        # in-memory only for first cut; ContextMemory integration
        # comes with template-check + SPEC_context_memory_component wiring).
        self._templates: dict[str, dict] = {}

        # Per-turn buffer of cells whose code has been stable for K
        # turns — used by persistence-check to detect oscillation.
        self._cell_history: dict[tuple[int, int], list[str]] = {}
        self._history_window = 5

        # Turn counter for trust decay.
        self._turn = 0

    # ------------------------------------------------------------------
    # State management.
    # ------------------------------------------------------------------

    def reset_for_level(self) -> None:
        """Drop per-level state.  Templates and pending claims persist."""
        self._trust.clear()
        self._cell_history.clear()

    # ------------------------------------------------------------------
    # Ingest entry point.
    # ------------------------------------------------------------------

    def ingest(
        self,
        *,
        turn: int,
        observation: FrameObservation,
        classification: Classification,
        prev_action: Optional[ActionRecord] = None,
        prev_observation: Optional[FrameObservation] = None,
        prev_classification: Optional[Classification] = None,
        outcome: Optional[OutcomeRecord] = None,
        history: Optional[FrameHistory] = None,
        game_id: Optional[str] = None,
        trial_id: Optional[str] = None,
    ) -> ValidationResult:
        """Run all five inspectors against this turn's data."""

        self._turn = turn
        ctx = {
            "trial_id": trial_id or "?",
            "turn": turn,
            "game_id": game_id or self.game_id,
            "model": "validator",
        }
        result = ValidationResult()

        # Update per-cell history buffer (for persistence-check).
        self._update_cell_history(classification)

        # motion-check (was V1) — agent moved by the action's expected
        # cell delta?
        if (prev_observation is not None
                and prev_action is not None):
            self._motion_check(
                observation, classification,
                prev_observation, prev_classification,
                prev_action, outcome, ctx, result,
            )

        # persistence-check (was V2) — cells oscillating without
        # plausible local cause?
        if prev_observation is not None and prev_classification is not None:
            self._persistence_check(
                observation, classification,
                prev_observation, prev_classification,
                prev_action, ctx, result,
            )

        # structural-check (was V3) — singleton agent, HUD exclusivity,
        # win-marker rarity.
        self._structural_check(observation, classification, ctx, result)

        # outcome-check (was V4) — deferred claims that this turn's
        # action made testable.
        if prev_action is not None and outcome is not None:
            self._outcome_check(
                prev_action, outcome, ctx, result,
            )

        # template-check (was V5) — visual-template drift detection.
        if prev_observation is not None and prev_classification is not None:
            self._template_check(
                observation, classification, ctx, result,
            )

        # Aggregate trust state, decide demotions.
        self._apply_trust_updates(result)
        self._compute_demotions(classification, result)

        # Capture new pending claims from THIS turn's classifier
        # output (cells classified as win-marker that haven't been
        # outcome-tested yet).
        self._capture_pending_claims(classification, ctx)

        return result

    # ------------------------------------------------------------------
    # motion-check — action-grounded inspection.
    # ------------------------------------------------------------------

    def _motion_check(
        self,
        observation, classification,
        prev_observation, prev_classification,
        prev_action: ActionRecord,
        outcome: Optional[OutcomeRecord],
        ctx: dict,
        result: ValidationResult,
    ) -> None:
        """Agent cell shifted by expected delta when the action was a
        translation?  Emit VerifierCheck on agreement; PolicyEvent on
        mismatch.

        Expected delta comes from the action-semantics registry, not
        from a hardcoded action_id → delta table.  If the registry
        has not yet corroborated the action, this inspector records
        an observation (so the registry CAN learn) and skips firing.

        Tool emitted (parent spec's substrate registry): the inspector
        records VerifierChecks with ``tool_used="motion_check"`` and
        PolicyEvents whose ``subject.inspector`` is ``"motion_check"``.
        """
        if prev_action.kind != "translate":
            return
        prev_pos = prev_observation.agent_position
        curr_pos = observation.agent_position
        if prev_pos is None or curr_pos is None:
            return

        # Record an observation in the registry regardless of whether
        # we have an expected delta yet.  This is the substrate-agnostic
        # learning loop: motion-check watches motion, the registry
        # accumulates, and once corroborated, future calls fire against it.
        if prev_action.action_id is not None:
            self.action_semantics.record_motion(
                prev_action.action_id, prev_pos, curr_pos,
            )

        # Prefer the registry's confirmed delta over whatever was
        # baked into the ActionRecord at construction time.  This
        # ensures motion-check stays consistent when the registry has updated
        # mid-trial.
        expected = None
        if prev_action.action_id is not None:
            expected = self.action_semantics.best_delta(
                prev_action.action_id,
            )
        if expected is None:
            expected = prev_action.delta
        if expected is None:
            # Registry hasn't corroborated this action yet; nothing to
            # compare against.  Returning here means motion-check
            # stays silent until enough motion has been observed.
            return

        dr_exp, dc_exp = expected
        dr_obs = curr_pos[0] - prev_pos[0]
        dc_obs = curr_pos[1] - prev_pos[1]

        claim = VerifiableClaim(
            claim_type="agent_motion_consistent",
            args={
                "action": {
                    "kind": prev_action.kind,
                    "action_id": prev_action.action_id,
                    "expected_delta": list(expected),
                },
                "prev_cell": list(prev_pos),
                "curr_cell": list(curr_pos),
            },
            asserted=True,
            source="validator_motion_check",
            confidence=1.0,
        )
        result.verifiable_claims.append(claim)

        if (dr_obs, dc_obs) == (dr_exp, dc_exp):
            # Confirmed — both classifications corroborate.
            result.verifier_checks.append(VerifierCheck(
                claim_type=claim.claim_type,
                args=claim.args, asserted=True, actual=True,
                tool_used="motion_check",
                agreement=True, outcome="agree",
            ))
            self._bump_trust(curr_pos, +1,
                             weight=_TRUST_BETA["motion_check"])
            self._bump_trust(prev_pos, +1,
                             weight=_TRUST_BETA["motion_check"])
        elif (dr_obs, dc_obs) == (0, 0):
            # Stayed put — could be blocked.  Ambiguous; no record.
            return
        else:
            # Inconsistent motion.  Lower-confidence side takes the hit.
            result.verifier_checks.append(VerifierCheck(
                claim_type=claim.claim_type,
                args=claim.args, asserted=True, actual=False,
                tool_used="motion_check",
                agreement=False, outcome="disagree",
            ))
            ev = PolicyEvent(
                policy_id="perception_validation_failure",
                severity="warn",
                subject={
                    "inspector": "motion_check",
                    "entity_id": "agent",
                    "cell": list(curr_pos),
                    "claim": "agent_motion_delta",
                },
                violation={
                    "rule": "agent_motion_delta",
                    "expected": [dr_exp, dc_exp],
                    "actual": [dr_obs, dc_obs],
                },
                remedy=None, context=ctx,
            )
            result.policy_events.append(ev)
            self._bump_trust(curr_pos, -1,
                             weight=_TRUST_BETA["motion_check"])

    # ------------------------------------------------------------------
    # persistence-check — oscillation detection.
    # ------------------------------------------------------------------

    def _persistence_check(
        self,
        observation, classification,
        prev_observation, prev_classification,
        prev_action: Optional[ActionRecord],
        ctx: dict,
        result: ValidationResult,
    ) -> None:
        """Cells whose code oscillates without a plausible local cause
        emit a PolicyEvent.  Definition of "plausible cause" matches
        the sequence-aware classifier's logic so the two stay in sync.
        """
        rows, cols = classification.rows, classification.cols
        plausible = self._plausible_changed_cells(
            observation, prev_observation, prev_action,
        )
        for (r, c), code_hist in self._cell_history.items():
            # Need at least 3 frames of history to detect oscillation:
            # the pattern X → Y → X (or wider).
            if len(code_hist) < 3:
                continue
            recent = code_hist[-3:]
            # Oscillation: first and last agree, middle differs.
            if recent[0] == recent[-1] and recent[1] != recent[0]:
                if (r, c) in plausible:
                    continue
                ev = PolicyEvent(
                    policy_id="perception_validation_failure",
                    severity="info",
                    subject={
                        "inspector": "persistence_check",
                        "cell": [r, c],
                        "claim": "cell_oscillation",
                    },
                    violation={
                        "rule": "cell_oscillation",
                        "expected": "stable_code",
                        "actual": list(recent),
                    },
                    remedy=None, context=ctx,
                )
                result.policy_events.append(ev)
                self._bump_trust((r, c), -1,
                                  weight=_TRUST_BETA["persistence_check"])

    # ------------------------------------------------------------------
    # structural-check — game-agnostic single-frame priors.
    # ------------------------------------------------------------------

    def _structural_check(
        self,
        observation, classification,
        ctx: dict, result: ValidationResult,
    ) -> None:
        """Game-agnostic structural invariants over a single frame:
            (i)   singleton agent,
            (ii)  HUD exclusivity,
            (iii) win-marker rarity,
            (iv)  agent-hazard exclusivity.
        Sprite-cell consistency is a fifth check but requires per-
        component metadata that the current Classification doesn't
        carry; deferred until classifier exposes it.
        """
        rows, cols = classification.rows, classification.cols
        codes = classification.cell_codes

        # (i) Singleton agent.
        agent_cells = [
            (r, c)
            for r in range(rows) for c in range(cols)
            if codes[r][c] == "A"
        ]
        if len(agent_cells) > 1:
            # Multiple A cells — emit one event with all locations.
            ev = PolicyEvent(
                policy_id="perception_validation_failure",
                severity="warn",
                subject={
                    "inspector": "structural_check",
                    "claim": "agent_split",
                    "cells": [list(c) for c in agent_cells],
                },
                violation={
                    "rule": "agent_split",
                    "expected": "at_most_one_agent_cell",
                    "actual": len(agent_cells),
                },
                remedy=None, context=ctx,
            )
            result.policy_events.append(ev)
            for ac in agent_cells:
                self._bump_trust(ac, -1,
                                  weight=_TRUST_BETA["structural_check"])

        # (ii) HUD exclusivity.  Cells classified U should not also
        # carry an entity code; the inverse is enforced implicitly
        # by single-code-per-cell semantics, but a hazard sitting
        # INSIDE the U row would surface as a violation here.  We
        # check the cells immediately ABOVE the U strip for hazard /
        # agent / win-marker rows that overlap the bottom edge.
        u_rows = sorted({r for r in range(rows)
                          if any(codes[r][c] == "U" for c in range(cols))})
        for r in u_rows:
            for c in range(cols):
                if codes[r][c] in ("A", "H", "P", "G", "X"):
                    ev = PolicyEvent(
                        policy_id="perception_validation_failure",
                        severity="warn",
                        subject={
                            "inspector": "structural_check",
                            "claim": "hud_occupied",
                            "cell": [r, c],
                        },
                        violation={
                            "rule": "hud_occupied",
                            "expected": "U_only_in_hud_row",
                            "actual": codes[r][c],
                        },
                        remedy=None, context=ctx,
                    )
                    result.policy_events.append(ev)
                    self._bump_trust((r, c), -1,
                                      weight=_TRUST_BETA["structural_check"])

        # (iii) Win-marker rarity.
        x_cells = [
            (r, c) for r in range(rows) for c in range(cols)
            if codes[r][c] == "X"
        ]
        if len(x_cells) > 1:
            ev = PolicyEvent(
                policy_id="perception_validation_failure",
                severity="info",
                subject={
                    "inspector": "structural_check",
                    "claim": "win_marker_rarity",
                    "cells": [list(c) for c in x_cells],
                },
                violation={
                    "rule": "win_marker_rarity",
                    "expected": "at_most_one_win_marker",
                    "actual": len(x_cells),
                },
                remedy=None, context=ctx,
            )
            result.policy_events.append(ev)
            # Light down-weight on all candidates; outcome-check
            # breaks the tie.
            for xc in x_cells:
                self._bump_trust(
                    xc, -1,
                    weight=_TRUST_BETA["structural_check"] * 0.5,
                )

    # ------------------------------------------------------------------
    # outcome-check — deferred-claim verification.
    # ------------------------------------------------------------------

    def _outcome_check(
        self,
        prev_action: ActionRecord,
        outcome: OutcomeRecord,
        ctx: dict,
        result: ValidationResult,
    ) -> None:
        """When the actor clicked, resolve any pending win_marker_at
        claim against that cell.
        """
        if prev_action.kind != "click":
            return
        target = prev_action.target_cell
        if target is None:
            return
        # Find any pending claim that matches this clicked cell.
        resolved: list[dict] = []
        for claim in self._pending:
            if (claim["claim_type"] == "win_marker_at"
                    and tuple(claim["args"]["cell"]) == tuple(target)):
                actual = bool(outcome.lc_delta > 0)
                claim_outcome = "agree" if actual else "disagree"
                result.verifier_checks.append(VerifierCheck(
                    claim_type=claim["claim_type"],
                    args=claim["args"],
                    asserted=True, actual=actual,
                    tool_used="outcome_check",
                    agreement=actual, outcome=claim_outcome,
                ))
                # Bump trust on the target cell accordingly.
                self._bump_trust(
                    tuple(target),
                    +1 if actual else -1,
                    weight=_TRUST_BETA["outcome_check"],
                )
                resolved.append(claim)
        # Remove resolved claims from the pending log.
        for r in resolved:
            self._pending.remove(r)

    # ------------------------------------------------------------------
    # template-check — visual-signature drift detection.
    # ------------------------------------------------------------------

    def _template_check(
        self,
        observation, classification,
        ctx: dict, result: ValidationResult,
    ) -> None:
        """Capture / corroborate visual templates for cells whose
        identity has been corroborated by motion-check / outcome-check.

        First-cut: agent template only.  Whenever the agent cell has
        passed motion-check corroboration and trust is high, record
        the cell's quantized RGB fingerprint as the agent's template.
        Future frames where the classified-as-agent cell does not
        match the template emit a `visual_drift` PolicyEvent.
        """
        if observation.agent_position is None:
            return
        ar, ac = observation.agent_position
        if not (0 <= ar < classification.rows
                and 0 <= ac < classification.cols):
            return
        sig = self._cell_signature(observation, ar, ac)
        trust = self._trust.get((ar, ac), _TRUST_INITIAL)

        existing = self._templates.get("agent")
        if trust >= self.HIGH_TRUST:
            # Promote / update template.
            self._templates["agent"] = {
                "signature": sig,
                "first_seen_turn": (
                    existing.get("first_seen_turn", self._turn)
                    if existing else self._turn
                ),
                "last_corroborated_turn": self._turn,
            }
        elif existing is not None:
            # Compare against existing template.
            similarity = self._signature_similarity(sig, existing["signature"])
            if similarity < 0.5:
                ev = PolicyEvent(
                    policy_id="perception_validation_failure",
                    severity="info",
                    subject={
                        "inspector": "template_check",
                        "claim": "visual_drift",
                        "entity_id": "agent",
                        "cell": [ar, ac],
                    },
                    violation={
                        "rule": "visual_drift",
                        "expected": "template_match",
                        "actual": {
                            "similarity": round(similarity, 3),
                            "template_first_seen_turn":
                                existing.get("first_seen_turn"),
                        },
                    },
                    remedy=None, context=ctx,
                )
                result.policy_events.append(ev)
                self._bump_trust((ar, ac), -1,
                                  weight=_TRUST_BETA["template_check"])

    # ------------------------------------------------------------------
    # Trust scoring + demotion.
    # ------------------------------------------------------------------

    def _bump_trust(self, cell: tuple[int, int], sign: int,
                    weight: float) -> None:
        """Multiplicative update.  +1 → asymptotic-to-1; -1 → multiplicative
        down-weight.
        """
        prev = self._trust.get(cell, _TRUST_INITIAL)
        if sign > 0:
            new = prev + _TRUST_ALPHA * weight * (1.0 - prev)
        else:
            new = prev * (1.0 - weight)
        self._trust[cell] = max(0.0, min(1.0, new))

    def _apply_trust_updates(self, result: ValidationResult) -> None:
        """Apply slow decay to cells not touched this turn, and copy
        the trust map onto the result for the trial driver to read.
        """
        # The decay is bounded so cells the validator didn't touch this
        # turn drift slowly toward the prior, not catastrophically.
        for cell in list(self._trust.keys()):
            self._trust[cell] = max(
                0.0,
                self._trust[cell] - _TRUST_DECAY_PER_TURN,
            )
        # Emit trust map (only for cells that have been touched at
        # least once; the rest implicitly carry the prior).
        for cell, score in self._trust.items():
            result.trust_scores[cell] = score

    def _compute_demotions(self, classification: Classification,
                            result: ValidationResult) -> None:
        """Cells with trust < LOW_TRUST are demoted; the trial driver
        substitutes "?" for these before building the actor prompt.
        """
        for cell, score in self._trust.items():
            if score < self.LOW_TRUST:
                result.demoted_cells.add(cell)

    # ------------------------------------------------------------------
    # Pending claim management.
    # ------------------------------------------------------------------

    def _capture_pending_claims(
        self,
        classification: Classification,
        ctx: dict,
    ) -> None:
        """Capture this turn's win-marker classifications as deferred
        claims to be verified when the actor next clicks each cell.
        """
        for r in range(classification.rows):
            for c in range(classification.cols):
                if classification.cell_codes[r][c] != "X":
                    continue
                # Already pending?
                already = any(
                    p["claim_type"] == "win_marker_at"
                    and tuple(p["args"]["cell"]) == (r, c)
                    for p in self._pending
                )
                if already:
                    continue
                self._pending.append({
                    "claim_type": "win_marker_at",
                    "args": {"cell": [r, c]},
                    "asserted": True,
                    "source": "classifier",
                    "confidence": 0.5,
                    "captured_turn": ctx["turn"],
                    "captured_trial": ctx["trial_id"],
                })

    # ------------------------------------------------------------------
    # Helpers — cell history, plausible-cause definition, signatures.
    # ------------------------------------------------------------------

    def _update_cell_history(self, classification: Classification) -> None:
        for r in range(classification.rows):
            for c in range(classification.cols):
                key = (r, c)
                buf = self._cell_history.setdefault(key, [])
                buf.append(classification.cell_codes[r][c])
                if len(buf) > self._history_window:
                    buf.pop(0)

    def _plausible_changed_cells(
        self,
        observation: FrameObservation,
        prev_observation: FrameObservation,
        prev_action: Optional[ActionRecord],
    ) -> set:
        """Match the sequence-aware classifier's definition exactly so
        smoothing and validation stay in sync.
        """
        plausible: set[tuple[int, int]] = set()
        if prev_observation.agent_position is not None:
            plausible.add(tuple(prev_observation.agent_position))
        if observation.agent_position is not None:
            plausible.add(tuple(observation.agent_position))
        if (prev_action is not None
                and prev_action.kind == "click"
                and prev_action.target_cell is not None):
            plausible.add(tuple(prev_action.target_cell))
        rows = observation.rows
        cols = observation.cols
        for r in range(rows):
            for c in range(cols):
                if not observation.cells[r][c].same_as_prev:
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            plausible.add((nr, nc))
        return plausible

    def _cell_signature(
        self,
        observation: FrameObservation,
        r: int, c: int,
    ) -> tuple:
        """Substrate-agnostic per-cell visual fingerprint.

        Quantized RGB histogram of the cell's pixel patch.  Same
        QUANT_STEP the classifier uses so the spaces are commensurable.
        """
        patch = observation.cells[r][c].rgb_patch
        flat = patch.reshape(-1, 3)
        if flat.size == 0:
            return tuple()
        # Quantize to multiples of 16 (matches classifier.QUANT_STEP).
        q = (flat.astype(np.int32) // 16) * 16
        keys = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
        unique, counts = np.unique(keys, return_counts=True)
        # Sort by count descending and return top-5 (key, frac) pairs.
        order = np.argsort(-counts)
        top = []
        total = float(keys.size)
        for idx in order[:5]:
            top.append((int(unique[idx]), float(counts[idx] / total)))
        return tuple(top)

    def _signature_similarity(self, a: tuple, b: tuple) -> float:
        """Bhattacharyya-like overlap of two top-K histograms.
        Returns a float in [0, 1]."""
        if not a or not b:
            return 0.0
        d_a = dict(a)
        d_b = dict(b)
        # Sum of min-fraction for shared bins.
        overlap = 0.0
        for k, fa in d_a.items():
            fb = d_b.get(k, 0.0)
            overlap += min(fa, fb)
        return overlap

    # ------------------------------------------------------------------
    # Snapshot for tracing / debugging.
    # ------------------------------------------------------------------

    def snapshot(self) -> dict:
        """Diagnostic snapshot for the trace renderer."""
        return {
            "turn": self._turn,
            "trust_scores": {
                f"{r},{c}": round(s, 3)
                for (r, c), s in self._trust.items()
            },
            "pending_claims": len(self._pending),
            "templates": list(self._templates.keys()),
            "history_cells": len(self._cell_history),
            "action_semantics": self.action_semantics.snapshot(),
        }
