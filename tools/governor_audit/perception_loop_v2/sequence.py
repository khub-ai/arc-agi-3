"""Substrate-agnostic sequence-aware classification wrapper.

Implements the contract specified in
`docs/SPEC_perception_module.md` § Substrate-agnostic sequence-aware
classification.  The single-frame `classify()` in `classifier.py` is
the baseline (Pass 1: component extraction + code commitment); this
module adds the cross-frame passes:

  Pass 2 — Component matching (cell-level approximation in this
           cut: per-cell `same_as_prev` flag from FrameObservation).
  Pass 3 — Identity propagation: cells with unchanged RGB patches
           inherit the previous classification.
  Pass 4 — Code commitment with three coupling guards:
             (a) inheritance does not increase confidence;
             (b) visual-signature contradiction wins over inheritance;
             (c) agent-motion mismatch flagged for the validator
                 (handled out-of-band in agent_detect.py).

Cold-start (no `history` / no `prev_action`) falls through to the
single-frame baseline unchanged.  Worst-case behaviour is the v2
baseline — never below it.

No game-specific knowledge, no palette indices, no harness-encoding
dependence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .classifier import classify as _single_frame_classify, Classification
from .observation import FrameObservation


# -----------------------------------------------------------------------------
# Substrate-agnostic action vocabulary.
# -----------------------------------------------------------------------------


@dataclass
class ActionRecord:
    """Substrate-agnostic record of the action that fired between two
    consecutive observations.

    `kind` values:
      - ``"translate"``: agent slides; ``delta`` carries the expected
        ``(dr, dc)`` cell shift if known.  ``delta=None`` means the
        action-semantics registry has not yet learned what this
        action does (cold start).  Validator V1 then records the
        observation rather than firing.
      - ``"click"``: actor clicked a specific cell; ``target_cell``
        carries ``(r, c)``.  Click semantics (consumes, triggers,
        wins, no-op) are not part of the record — those are
        outcomes the validator grades against.
      - ``"noop"``: no expected world effect (start-of-trial, idle
        turn, end-state).

    ``action_id`` is the harness's action id (e.g. 1-6 for
    ARC-AGI-3).  Stored so the validator can consult the action-
    semantics registry for an up-to-date confirmed delta even when
    the caller built the record before the registry had learned
    anything.
    """

    kind: str
    delta: Optional[tuple[int, int]] = None
    target_cell: Optional[tuple[int, int]] = None
    action_id: Optional[int] = None


# -----------------------------------------------------------------------------
# Rolling frame history.
# -----------------------------------------------------------------------------


@dataclass
class FrameHistory:
    """Rolling buffer of recent ``(observation, classification, action)``
    triples for one trial.

    The buffer stores at most ``max_frames`` entries.  ``action_to_next``
    on entry ``i`` is the action that fired between entry ``i`` and
    entry ``i+1``; on the latest entry it is ``None`` until the next
    turn appends.

    Persistence boundary: ``FrameHistory`` is per-trial; reset at level
    transitions and trial restarts.
    """

    max_frames: int = 5
    frames: list[dict] = field(default_factory=list)

    def append(
        self,
        turn: int,
        observation: FrameObservation,
        classification: Classification,
        action_taken: Optional[ActionRecord] = None,
    ) -> None:
        """Record this turn.

        ``action_taken`` is the action that fired between the previous
        turn's frame and this turn's frame — i.e. the action that
        produced ``observation``.  It is recorded onto the previous
        entry's ``action_to_next`` slot (so each entry's
        ``action_to_next`` describes "what came after this frame").
        """
        if self.frames and action_taken is not None:
            self.frames[-1]["action_to_next"] = action_taken
        self.frames.append({
            "turn": turn,
            "observation": observation,
            "classification": classification,
            "action_to_next": None,
        })
        while len(self.frames) > self.max_frames:
            self.frames.pop(0)

    def previous(self) -> Optional[dict]:
        """Return the most-recently-appended entry's *previous* sibling,
        or ``None`` if the buffer has fewer than two entries.
        """
        if len(self.frames) >= 2:
            return self.frames[-2]
        return None

    def reset(self) -> None:
        """Drop all entries.  Called at level transitions or trial
        restarts where cross-frame inheritance is no longer valid.
        """
        self.frames.clear()


# -----------------------------------------------------------------------------
# Sequence-aware classifier.
# -----------------------------------------------------------------------------


# Diagnostic counters per call, surfaced via `classify_sequence_aware`'s
# returned classification.  Kept as a dict on the side rather than baked
# into the Classification dataclass to keep the dataclass stable for
# existing call-sites.
_LAST_DIAGNOSTICS: dict = {
    "cells_smoothed": 0,
    "cells_overridden_by_visual_change": 0,
    "cells_overridden_by_plausible_cause": 0,
    "cold_start": False,
}


def last_diagnostics() -> dict:
    """Return diagnostics from the most-recent
    ``classify_sequence_aware`` call.  Operator/test introspection."""
    return dict(_LAST_DIAGNOSTICS)


def classify_sequence_aware(
    obs: FrameObservation,
    *,
    history: Optional[FrameHistory] = None,
    prev_action: Optional[ActionRecord] = None,
) -> Classification:
    """Sequence-aware perception.

    Runs the single-frame classifier, then applies persistence
    smoothing per
    ``docs/SPEC_perception_module.md`` § Substrate-agnostic
    sequence-aware classification.

    Behaviour matrix:

      * ``history is None`` or empty           → cold start; return
        single-frame classification as-is.
      * ``history.frames[-1]`` exists, but its ``classification`` has a
        different grid shape than ``obs``                            →
        full reset (no smoothing); return single-frame classification.
      * For every cell where the new code disagrees with the previous
        cell's code:
          - If the cell's RGB patch has changed since the previous
            frame (``cell_obs.same_as_prev == False``) → keep the new
            code (visual-signature contradiction wins).
          - Else if the cell could plausibly have changed due to a
            local cause (the agent left / entered, a click landed
            here) → keep the new code.
          - Else → inherit the previous cell's code (single-frame
            heuristic flicker; pixels say nothing happened so the
            classifier's switch is noise).
    """
    global _LAST_DIAGNOSTICS
    _LAST_DIAGNOSTICS = {
        "cells_smoothed": 0,
        "cells_overridden_by_visual_change": 0,
        "cells_overridden_by_plausible_cause": 0,
        "cold_start": False,
    }

    # Pass 1: single-frame classification — unchanged v2 baseline.
    cls = _single_frame_classify(obs)

    # Cold start: no previous frame to draw on.
    if history is None or not history.frames:
        _LAST_DIAGNOSTICS["cold_start"] = True
        return cls

    prev_entry = history.frames[-1]
    prev_cls = prev_entry.get("classification")
    prev_obs = prev_entry.get("observation")
    if prev_cls is None or prev_obs is None:
        _LAST_DIAGNOSTICS["cold_start"] = True
        return cls

    rows, cols = cls.rows, cls.cols
    if prev_cls.rows != rows or prev_cls.cols != cols:
        # Grid shape changed (level transition?).  No smoothing possible.
        _LAST_DIAGNOSTICS["cold_start"] = True
        return cls

    # Identify cells where a local cause could plausibly explain a
    # change between the previous and current frame.  The spec lists
    # three substrate-agnostic causes:
    #   (a) the agent left or entered the cell,
    #   (b) the actor's click targeted it,
    #   (c) a neighbor cell's pixels changed (which can shift the
    #       cell's own quantization boundary even if its pixels look
    #       identical — a small but real coupling).
    plausible: set[tuple[int, int]] = set()
    if prev_obs.agent_position is not None:
        plausible.add(tuple(prev_obs.agent_position))
    if obs.agent_position is not None:
        plausible.add(tuple(obs.agent_position))
    if (prev_action is not None
            and prev_action.kind == "click"
            and prev_action.target_cell is not None):
        plausible.add(tuple(prev_action.target_cell))
    # Neighbor-change check: any cell whose 4-neighbor's pixels
    # changed is plausibly affected by the surrounding shift.  This
    # catches the failure mode where a quantization threshold flips
    # for a cell that is *itself* pixel-identical but whose
    # surroundings changed.
    for r in range(rows):
        for c in range(cols):
            if not obs.cells[r][c].same_as_prev:
                # All 4-neighbors of this changed cell are plausibly
                # affected.
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        plausible.add((nr, nc))

    # Pass 2-4 (cell-level approximation): per-cell smoothing.
    for r in range(rows):
        for c in range(cols):
            curr_code = cls.cell_codes[r][c]
            prev_code = prev_cls.cell_codes[r][c]
            if prev_code == curr_code:
                continue

            # Coupling guard #2: visual-signature contradiction wins.
            cell_obs = obs.cells[r][c]
            if not cell_obs.same_as_prev:
                _LAST_DIAGNOSTICS["cells_overridden_by_visual_change"] += 1
                continue

            # Plausible local cause: keep new code rather than smoothing.
            if (r, c) in plausible:
                _LAST_DIAGNOSTICS["cells_overridden_by_plausible_cause"] += 1
                continue

            # Patch unchanged + no plausible cause → single-frame
            # heuristic flicker; inherit previous code.
            cls.cell_codes[r][c] = prev_code
            _LAST_DIAGNOSTICS["cells_smoothed"] += 1

    return cls


# -----------------------------------------------------------------------------
# Action-record convenience builders for the trial driver.
# -----------------------------------------------------------------------------


def translate_action(
    dr: int, dc: int,
    *,
    action_id: Optional[int] = None,
) -> ActionRecord:
    """Build a translate ActionRecord from the expected cell delta.

    When the caller does not yet know the delta (cold start), pass
    None for the delta via ``action_record_unknown``.
    """
    return ActionRecord(
        kind="translate", delta=(int(dr), int(dc)),
        action_id=action_id,
    )


def click_action(
    target_cell: tuple[int, int],
    *,
    action_id: Optional[int] = None,
) -> ActionRecord:
    """Build a click ActionRecord from the targeted cell."""
    return ActionRecord(
        kind="click",
        target_cell=(int(target_cell[0]), int(target_cell[1])),
        action_id=action_id,
    )


def noop_action(action_id: Optional[int] = None) -> ActionRecord:
    """Build a noop ActionRecord (start-of-trial, idle turn, etc.)."""
    return ActionRecord(kind="noop", action_id=action_id)


def action_record_unknown(action_id: int) -> ActionRecord:
    """Build an ActionRecord whose semantics are not yet known.

    The action's effect on the agent has not yet been corroborated by
    the action-semantics registry.  Validator V1 reads this as "skip
    the inspection; record an observation instead."
    """
    return ActionRecord(
        kind="translate", delta=None, action_id=action_id,
    )


# -----------------------------------------------------------------------------
# Action-id → ActionRecord resolver (substrate-agnostic at point of call).
# -----------------------------------------------------------------------------


def action_record_from_action_id(
    action_id: int,
    *,
    click_target: Optional[tuple[int, int]] = None,
    registry: object = None,
) -> ActionRecord:
    """Resolve an action id to an ActionRecord using the action-
    semantics registry.

    With ``registry`` supplied, the function asks the registry for
    the confirmed delta.  If the registry has corroborated the
    action's effect, the returned record carries that delta; if
    not, the record is "translate but delta unknown" (validator V1
    skips the inspection and records an observation instead).

    With ``registry=None`` the function returns a noop record for
    any non-click action — there is no hardcoded translation table
    any more.  Each game's deltas are learned from observation.

    A click target turns the record into a click action regardless
    of registry state.
    """
    if click_target is not None:
        return click_action(click_target, action_id=action_id)
    if registry is not None:
        try:
            confirmed = registry.best_delta(action_id)
        except Exception:
            confirmed = None
        if confirmed is not None:
            # Even a confirmed (0, 0) delta is information V1 should
            # check on each occurrence — "this action does nothing"
            # is a verifiable claim, and a single observed motion
            # against that prior is a strong disagreement signal.
            return translate_action(
                confirmed[0], confirmed[1], action_id=action_id,
            )
        # Registry exists but hasn't yet corroborated this action.
        return action_record_unknown(action_id)
    # No registry — cold start without any learning surface.  Mark
    # as unknown rather than guessing a delta.
    return action_record_unknown(action_id)
