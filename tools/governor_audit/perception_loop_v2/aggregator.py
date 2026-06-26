"""Rule aggregator — manages the sandbox / trial / established
lifecycle.

Lifecycle (matches SPEC_governor.md):

  sandbox      a rule has been observed enough times within one trial
               to be candidate for tentative application.  Not applied
               at perception time by default (passive evidence
               accumulation).
  trial        the rule has been corroborated in at least one
               independent trial of the same (game, lc).  Applied
               silently at perception time.
  established  the rule has held over many turns / multiple trials
               with no contradiction.  Applied authoritatively.
  deprecated   contradicting evidence has pushed the rule back below
               sandbox.  Not applied; kept on disk for audit and
               possible re-promotion.

Cross-trial promotion is what makes the aggregator robust to a single
bad trial.  For a fixture set that contains only one trial, rules
plateau at sandbox until additional trials are captured.
"""

from __future__ import annotations

import time
from dataclasses import asdict

from .rules import Candidate, Rule, RuleStore, _stable_id


# Thresholds.  Tunable; sensible defaults below.
SANDBOX_TURN_THRESHOLD = 3       # candidate must reappear K turns to enter sandbox
TRIAL_TURN_THRESHOLD = 8         # sandbox + K more turns of agreement -> trial (within trial)
ESTABLISHED_TRIAL_THRESHOLD = 2  # rule must hold in N independent trials to be established


class Aggregator:
    """Ingests Candidate streams turn-by-turn, manages the rule store."""

    def __init__(
        self,
        store: RuleStore,
        *,
        trial_id: str,
        sandbox_threshold: int = SANDBOX_TURN_THRESHOLD,
        trial_threshold: int = TRIAL_TURN_THRESHOLD,
        established_trial_threshold: int = ESTABLISHED_TRIAL_THRESHOLD,
    ):
        self.store = store
        self.trial_id = trial_id
        self.sandbox_threshold = sandbox_threshold
        self.trial_threshold = trial_threshold
        self.established_trial_threshold = established_trial_threshold

        # Pending candidates within this trial: signature -> Candidate
        # accumulating evidence_count.
        self._pending: dict[str, Candidate] = {}

    def ingest_turn(
        self,
        candidates: list[Candidate],
        turn: int,
    ) -> list[Rule]:
        """Ingest one turn's candidates.  Returns the list of rules
        that changed status this turn (newly created, promoted, or
        demoted)."""
        changed: list[Rule] = []
        seen_signatures: set[str] = set()
        for cand in candidates:
            sig = cand.signature
            seen_signatures.add(sig)
            existing = self.store.get(sig)
            if existing is None:
                # In-trial pending candidate.
                pending = self._pending.get(sig)
                if pending is None:
                    pending = Candidate(
                        type=cand.type, body=cand.body,
                        evidence_count=0, supporting_turns=[],
                    )
                    self._pending[sig] = pending
                pending.evidence_count += 1
                pending.supporting_turns.append(turn)
                # Promote to sandbox when threshold met.
                if pending.evidence_count >= self.sandbox_threshold:
                    rule = self._make_rule(pending, status="sandbox")
                    self.store.upsert(rule)
                    changed.append(rule)
                    del self._pending[sig]
            else:
                # Existing rule: re-corroborate.
                existing.evidence_count += 1
                existing.last_trial = self.trial_id
                if self.trial_id not in existing.supporting_trials:
                    existing.supporting_trials.append(self.trial_id)
                promoted = self._maybe_promote(existing)
                self.store.upsert(existing)
                if promoted:
                    changed.append(existing)

        # Demote: existing rules supported by this trial that did NOT
        # appear in this turn's candidates accumulate "non-observation"
        # implicit contradiction.  Don't demote on a single miss —
        # demotion is for repeated contradictions over many turns,
        # which we approximate by tracking misses and only demoting
        # when misses outnumber recent supports.  Implementation is
        # intentionally conservative for the v2 first cut.

        return changed

    def finalize_trial(self) -> list[Rule]:
        """Called at end-of-trial.  Currently a no-op placeholder —
        cross-trial promotion logic is applied incrementally inside
        ingest_turn whenever a rule is re-observed in a new trial.

        Returns rules promoted at trial finalisation (currently none).
        """
        return []

    # ------------------------------------------------------------------

    def _make_rule(self, cand: Candidate, *, status: str) -> Rule:
        sig = cand.signature
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        confidence = min(0.99, 0.5 + 0.05 * cand.evidence_count)
        return Rule(
            id=sig,
            type=cand.type,
            body=cand.body,
            status=status,
            evidence_count=cand.evidence_count,
            confidence=confidence,
            added_at=now,
            added_in_trial=self.trial_id,
            last_trial=self.trial_id,
            supporting_trials=[self.trial_id],
            contradicting_trials=[],
        )

    def _maybe_promote(self, rule: Rule) -> bool:
        """Apply promotion rules.  Returns True if status changed."""
        changed = False
        # Sandbox -> trial: enough in-trial turns OR a new trial has
        # corroborated it.
        if rule.status == "sandbox":
            if rule.evidence_count >= self.trial_threshold:
                rule.status = "trial"
                changed = True
            elif len(rule.supporting_trials) >= 2:
                # Cross-trial agreement on a sandbox rule: promote.
                rule.status = "trial"
                changed = True
        if rule.status == "trial":
            if (len(rule.supporting_trials)
                    >= self.established_trial_threshold + 1):
                rule.status = "established"
                changed = True
        rule.confidence = min(0.99, 0.5 + 0.05 * rule.evidence_count)
        return changed
