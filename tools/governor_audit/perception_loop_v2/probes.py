"""Probe ledger — systematic-experimentation discipline.

Every probe is a logged experiment that names ≥2 competing
hypotheses and predicts the outcome under each before executing.
On execution, the substrate compares the observed outcome to
each prediction and bumps / decays hypothesis credences
automatically.

Substrate role:
  * persist ProbeRecords on WorldKnowledge
  * surface pending probes in the strategy prompt
  * apply commit + observation hooks from the strategy reply
  * detect "epistemic stuck" (N turns since any hypothesis
    credence moved)

The actor's discipline:
  * Before any action whose purpose is to learn, commit a
    ProbeRecord with ≥2 hypothesis ids and per-hypothesis
    predictions
  * After executing, record a probe_observation with the actual
    outcome — substrate updates credences automatically
  * Don't leave probes pending indefinitely (drift detection)

GAME-AGNOSTIC: all fields are open-vocabulary; the substrate
holds and compares strings.  No game-specific code anywhere.
"""
from __future__ import annotations

import time
from typing import Optional

from world_knowledge import ProbeRecord, WorldKnowledge


_MIN_HYPOTHESES_PER_PROBE = 2
    # < 2 hypotheses = not a probe, it's just an action.  The
    # discipline rejects single-hypothesis "let me try X and see"
    # — those don't discriminate.

_SUPPORT_BUMP_FROM_PROBE     = 0.10
_CONTRADICT_DECAY_FROM_PROBE = 0.25
_PENDING_PROBE_AGE_WARNING   = 5
    # turns; if a probe has been pending this long without
    # execution, surface a drift warning


# ---------------------------------------------------------------------------
# Id generation
# ---------------------------------------------------------------------------


def _new_id(uncertainty: str, turn: int) -> str:
    safe = "".join(
        c if c.isalnum() or c == "_" else "_"
        for c in uncertainty.lower()
    )[:40]
    return f"probe_{safe}_t{turn}_{int(time.time()) % 100000}"


# ---------------------------------------------------------------------------
# Commit / observe
# ---------------------------------------------------------------------------


def propose_probe(
    world: WorldKnowledge,
    *,
    motivating_uncertainty: str,
    motivating_hypothesis_ids: list[str],
    action_or_sequence: list[str],
    predicted_outcomes: dict,
    notes: str = "",
) -> Optional[ProbeRecord]:
    """Commit a new probe to the ledger.  Rejects single-hypothesis
    probes (not experiments).  Returns the new record on success,
    None on validation failure."""
    if not motivating_uncertainty or not action_or_sequence:
        return None
    if len(motivating_hypothesis_ids) < _MIN_HYPOTHESES_PER_PROBE:
        print(
            f"[probes] rejecting probe with "
            f"{len(motivating_hypothesis_ids)} hypotheses "
            f"(need >={_MIN_HYPOTHESES_PER_PROBE}): not an experiment"
        )
        return None
    # Each hypothesis must have a prediction
    missing = [h for h in motivating_hypothesis_ids
                if h not in (predicted_outcomes or {})]
    if missing:
        print(
            f"[probes] rejecting probe missing predictions for "
            f"hypotheses: {missing}"
        )
        return None
    probe = ProbeRecord(
        probe_id=_new_id(motivating_uncertainty, world.turn),
        motivating_uncertainty=motivating_uncertainty,
        motivating_hypothesis_ids=list(motivating_hypothesis_ids),
        action_or_sequence=list(action_or_sequence),
        predicted_outcomes=dict(predicted_outcomes),
        status="pending",
        proposed_at_turn=world.turn,
        notes=notes,
    )
    world.probes.append(probe)
    return probe


def record_probe_observation(
    world: WorldKnowledge,
    *,
    probe_id: str,
    observed_outcome: str,
    matching_hypothesis_ids: list[str],
    contradicting_hypothesis_ids: list[str],
    notes: str = "",
) -> Optional[ProbeRecord]:
    """Record the actual outcome of an executed probe.  The actor
    self-reports which hypothesis predictions matched the observed
    outcome and which were contradicted.  Substrate updates the
    referenced WC and Mechanic hypothesis credences accordingly
    via the hooks the driver wires.

    Returns the updated probe record, or None if not found."""
    target: Optional[ProbeRecord] = None
    for p in world.probes:
        if p.probe_id == probe_id:
            target = p
            break
    if target is None:
        return None
    target.status = (
        "resolved"
        if (matching_hypothesis_ids or contradicting_hypothesis_ids)
        else "inconclusive"
    )
    target.executed_at_turn = world.turn
    target.observed_at_delta_index = (
        len(world.deltas_observed) - 1
        if world.deltas_observed else None
    )
    target.observed_outcome = observed_outcome
    if notes:
        target.notes = (
            target.notes + "\n" if target.notes else ""
        ) + f"t{world.turn}: {notes}"
    # Apply credence updates to WC + Mechanic hypotheses
    _propagate_credence(
        world,
        matching=matching_hypothesis_ids,
        contradicting=contradicting_hypothesis_ids,
        delta_index=target.observed_at_delta_index,
    )
    return target


def abandon_probe(
    world: WorldKnowledge, *, probe_id: str, reason: str = "",
) -> Optional[ProbeRecord]:
    for p in world.probes:
        if p.probe_id == probe_id:
            p.status = "abandoned"
            if reason:
                p.notes = (p.notes + "\n" if p.notes else "") + (
                    f"t{world.turn}: abandoned — {reason}"
                )
            return p
    return None


# ---------------------------------------------------------------------------
# Credence propagation
# ---------------------------------------------------------------------------


def _propagate_credence(
    world: WorldKnowledge,
    *,
    matching: list[str],
    contradicting: list[str],
    delta_index: Optional[int],
) -> None:
    """Update WinConditionHypothesis and MechanicHypothesis
    credences from a probe observation.  Matching hypotheses get
    +support; contradicting get +contradict."""
    if delta_index is None:
        delta_index = -1
    # WinConditionHypotheses
    for h in world.win_condition_hypotheses:
        if h.hypothesis_id in matching:
            h.supporting_observations.append(int(delta_index))
            h.credence = min(1.0, h.credence + _SUPPORT_BUMP_FROM_PROBE)
        if h.hypothesis_id in contradicting:
            h.contradicting_observations.append(int(delta_index))
            h.credence = max(0.0, h.credence - _CONTRADICT_DECAY_FROM_PROBE)
        if h.credence >= 0.85:
            h.promoted = True
    # MechanicHypotheses
    for m in world.mechanic_hypotheses:
        if m.hypothesis_id in matching:
            m.supporting_observations.append(int(delta_index))
            m.credence = min(1.0, m.credence + _SUPPORT_BUMP_FROM_PROBE)
        if m.hypothesis_id in contradicting:
            m.contradicting_observations.append(int(delta_index))
            m.credence = max(0.0, m.credence - _CONTRADICT_DECAY_FROM_PROBE)


# ---------------------------------------------------------------------------
# Retrieval / surface
# ---------------------------------------------------------------------------


def pending_probes(world: WorldKnowledge) -> list[ProbeRecord]:
    return [p for p in world.probes if p.status == "pending"]


def executed_probes(world: WorldKnowledge) -> list[ProbeRecord]:
    return [p for p in world.probes
            if p.status in ("executed", "resolved", "inconclusive")]


def detect_epistemic_stuck(
    world: WorldKnowledge,
    *,
    window_turns: int = 5,
) -> Optional[str]:
    """Return a free-form drift warning if no WC hypothesis credence
    has moved in the last ``window_turns`` turns AND probes have
    been executed during that window.  None if the system is
    learning.

    This is the 'acting but not learning' detector — the system
    is taking actions but no hypothesis is shifting, which means
    either the probes aren't discriminating or the actor isn't
    recording observations.
    """
    if not world.win_condition_hypotheses:
        return None
    # Check if any WC hypothesis got a new observation in the
    # last window
    threshold = max(0, len(world.deltas_observed) - window_turns)
    moved_recently = False
    for h in world.win_condition_hypotheses:
        all_obs = (
            (h.supporting_observations or [])
            + (h.contradicting_observations or [])
        )
        if any(int(o) >= threshold for o in all_obs):
            moved_recently = True
            break
    if moved_recently:
        return None
    # Count recently-executed probes
    n_recent_executed = sum(
        1 for p in world.probes
        if p.status in ("resolved", "inconclusive")
        and p.executed_at_turn is not None
        and p.executed_at_turn >= (world.turn - window_turns)
    )
    n_action_turns = min(window_turns, len(world.actions_taken))
    if n_action_turns < window_turns:
        return None
    if n_recent_executed == 0:
        # Not probing at all — different warning
        return (
            f"EPISTEMIC STUCK: {window_turns} action turns without "
            "any executed probe AND no WC hypothesis credence has "
            "moved.  You are acting but not learning.  Commit a "
            "ProbeRecord that tests ≥2 hypotheses against each "
            "other before your next action."
        )
    return (
        f"EPISTEMIC STUCK: {n_recent_executed} probe(s) executed in "
        f"the last {window_turns} turns but NO WinConditionHypothesis "
        "credence has moved.  Probable causes: (a) probes aren't "
        "discriminating — alternative hypotheses predict the same "
        "outcome; (b) actor isn't recording probe_observation "
        "matches / contradicts.  Redesign next probe to predict "
        "DIFFERENT outcomes under each hypothesis."
    )


def format_probes_surface(world: WorldKnowledge) -> str:
    """Render the probe ledger as a strategy-prompt block.
    Pending probes get top billing (the actor should execute or
    abandon them); recently executed probes are summarized for
    learning context."""
    pending = pending_probes(world)
    executed = executed_probes(world)
    lines: list[str] = []
    stuck = detect_epistemic_stuck(world)
    if stuck:
        lines.append(f"  !! {stuck}")
        lines.append("")
    if pending:
        lines.append(
            f"  {len(pending)} PENDING PROBE(s).  Each was "
            "committed but not yet executed.  THIS TURN you must "
            "either execute one (by setting `endorsed_action` to "
            "the next step in its `action_or_sequence` AND setting "
            "`probe_observation` after the resulting delta) OR "
            "abandon one (via `probe_abandon`).  Drifting away "
            "from a pending probe wastes its commitment."
        )
        for p in pending:
            age = world.turn - p.proposed_at_turn
            age_warn = (
                "  !! STALE"
                if age >= _PENDING_PROBE_AGE_WARNING else ""
            )
            lines.append("")
            lines.append(
                f"  PROBE id={p.probe_id!r}  age={age} turn(s)"
                f"{age_warn}"
            )
            lines.append(
                f"    Motivating uncertainty: {p.motivating_uncertainty}"
            )
            lines.append(
                f"    Discriminating among:   "
                f"{', '.join(p.motivating_hypothesis_ids)}"
            )
            lines.append(
                f"    Action sequence:        "
                f"{' -> '.join(p.action_or_sequence)}"
            )
            for hid, pred in (p.predicted_outcomes or {}).items():
                lines.append(f"    Predict under {hid}: {pred}")
    else:
        lines.append(
            "  (no pending probes.  If you are uncertain about "
            "ANY mechanic or win-condition hypothesis, commit a "
            "probe via `propose_probe` — at least 2 hypotheses, "
            "with a prediction under each.  Untested uncertainty "
            "is debt.)"
        )
    if executed:
        lines.append("")
        recent = sorted(
            executed,
            key=lambda p: -(p.executed_at_turn or 0),
        )[:5]
        lines.append(
            f"  Recently executed (last {len(recent)}): "
        )
        for p in recent:
            lines.append(
                f"    - {p.probe_id} ({p.status}) at "
                f"t{p.executed_at_turn}: "
                f"{p.observed_outcome[:80]}"
            )
    lines.append("")
    lines.append(
        "  HOW TO USE: in your strategy reply, "
        "`propose_probe` is a small JSON object: "
        '{ "motivating_uncertainty": str, '
        '"motivating_hypothesis_ids": [str, str, ...], '
        '"action_or_sequence": [str, ...], '
        '"predicted_outcomes": { hyp_id: str, ... }, '
        '"notes": str }.  '
        "`probe_observation` is { probe_id, observed_outcome, "
        "matching_hypothesis_ids: [str,...], "
        "contradicting_hypothesis_ids: [str,...], notes }.  "
        "`probe_abandon` is { probe_id, reason }."
    )
    return "\n".join(lines)
