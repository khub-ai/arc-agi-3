"""Controlled-experiment causal induction (game-agnostic).

The substrate now measures an OBSERVABLE MOVER's full trajectory per animated
action (``_substrate_animation_entities``).  This module turns *active*
variation of a set of CONTROL entities into a control->effect map and proposes
the control configuration that drives the mover to a TARGET -- the piece COS was
missing (its operator induction was passive: operators only emerged from
whatever the actor happened to do).

The design is one-factor-at-a-time (OFAT): fire the trigger once per single
control change and attribute the mover's response delta to the control that
changed.  OFAT is the simplest *sound* causal-attribution design -- it needs no
model of the mechanic, only the ability to vary one thing and measure one thing,
which is exactly what perception (controls) + substrate (mover trajectory)
already provide.

Nothing here is game-specific: a "control" is any entity observed to change
state in place when acted on; the "mover" is any transient/animated entity whose
trajectory varies; the "target" is any salient fixed entity the mover should
reach.  Pure data + functions; the live driver supplies the trials (sets a
config, fires the trigger, hands over the substrate-measured trajectory) and
verifies the proposal by running it -- the score stays the judge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Trial:
    """One controlled-experiment trial.

    config:   control_id -> value (any hashable) -- the control configuration set
              before the trigger fired.
    endpoint: (dr, dc) NET displacement of the mover, measured by the substrate
              (``animation_events`` -> the mover's ``net``).  This is the
              response variable.
    path:     the mover's full per-frame trajectory (optional, for richer
              attribution / diagnostics).
    """
    config: dict
    endpoint: tuple
    path: list = field(default_factory=list)


@dataclass
class ControlEffect:
    """The attributed effect of flipping one control: how the mover's endpoint
    changed when this control went from ``from_value`` to ``to_value``."""
    control_id: object
    from_value: object
    to_value: object
    endpoint_delta: tuple        # (d_dr, d_dc) change in the mover's net displacement
    n: int = 1                   # supporting trials (OFAT can repeat for confidence)


def induce_ofat(baseline: Trial, variations: list) -> dict:
    """One-factor-at-a-time attribution.

    Each variation that differs from ``baseline`` in EXACTLY ONE control
    attributes its mover-endpoint change to that control.  Variations differing
    in 0 or >1 controls are not OFAT-attributable and are skipped (the live
    driver should schedule single-control flips, but this stays robust if it
    doesn't).  Returns {control_id: ControlEffect}.
    """
    effects: dict = {}
    b = baseline.config
    for v in variations:
        diff = [k for k in (set(b) | set(v.config)) if b.get(k) != v.config.get(k)]
        if len(diff) != 1:
            continue
        k = diff[0]
        de = (round(v.endpoint[0] - baseline.endpoint[0], 2),
              round(v.endpoint[1] - baseline.endpoint[1], 2))
        if k in effects and effects[k].to_value == v.config.get(k):
            effects[k].n += 1          # repeated confirmation
        else:
            effects[k] = ControlEffect(k, b.get(k), v.config.get(k), de)
    return effects


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def propose_config(baseline: Trial, effects: dict, target_endpoint: tuple) -> Optional[dict]:
    """Greedily compose control flips so the mover's endpoint reaches
    ``target_endpoint``, under the OFAT/additivity assumption (each control's
    effect is independent and adds to the endpoint).  Applies, in greedy order,
    each not-yet-applied control flip that most reduces the Manhattan distance to
    the target; stops at a local optimum.

    Returns the proposed config (only the controls whose flip helped are
    changed), or None if no flip improves on the baseline.  This is the simplest
    sound planner under the model; the driver MUST verify it by running it (the
    score is the judge), since real effects may not be perfectly additive.
    """
    cfg = dict(baseline.config)
    cur = (float(baseline.endpoint[0]), float(baseline.endpoint[1]))
    applied = set()
    improved = True
    while improved:
        improved = False
        best = None
        for k, eff in effects.items():
            if k in applied:
                continue
            cand = (cur[0] + eff.endpoint_delta[0], cur[1] + eff.endpoint_delta[1])
            if _dist(cand, target_endpoint) < _dist(cur, target_endpoint) - 1e-9:
                if best is None or _dist(cand, target_endpoint) < _dist(best[1], target_endpoint):
                    best = (k, cand)
        if best is not None:
            k, cand = best
            cfg[k] = effects[k].to_value
            cur = cand
            applied.add(k)
            improved = True
    if _dist(cur, target_endpoint) >= _dist(baseline.endpoint, target_endpoint):
        return None
    return cfg


def attribute_steps(mover_traj, cursor_traj, controls) -> dict:
    """SINGLE-TRIAL per-step causal attribution -- the cheap, goal-directed
    alternative to multi-trial OFAT.

    When the TRIGGER sequentially VISITS the controls (a cursor / scan sweeps
    them) and the mover STEPS once per visit, one trigger fire already reveals
    each control's effect: attribute the mover's step at frame f to the control
    the cursor is over at frame f.  No varying of controls needed -- the initial
    frame's own settings demonstrate the law (e.g. tn36: the marker steps down
    exactly at the black-bar switches, so a single GO proves "black bar -> down
    step").

    mover_traj, cursor_traj: [(frame, (r, c)), ...].  controls: {cid: (col,row)}.
    Returns {cid: (dr, dc)} -- the step each visited control produced.
    """
    mpos = {int(f): (float(p[0]), float(p[1])) for f, p in mover_traj}
    cpos = {int(f): (float(p[0]), float(p[1])) for f, p in cursor_traj}
    eff = {}
    for f in sorted(mpos):
        if (f - 1) not in mpos or f not in cpos:
            continue
        step = (round(mpos[f][0] - mpos[f - 1][0], 1),
                round(mpos[f][1] - mpos[f - 1][1], 1))
        if abs(step[0]) + abs(step[1]) < 1:
            continue                       # the mover did not move this frame
        ccol = cpos[f][1]
        cid = min(controls, key=lambda k: abs(controls[k][0] - ccol))
        eff[cid] = step
    return eff


def imitation_plan(per_control_steps, target_dir):
    """Goal-directed imitation (not exhaustive search).  Given each control's
    step + the direction TOWARD the target, pick the EXEMPLAR controls (those
    already stepping cleanly toward the target) and the controls TO CHANGE (the
    rest, to be made to imitate an exemplar so they too step toward the target).

    This is the "the initial frame already shows the answer" move: replicate the
    toward-goal exemplar across the controls that need it, rather than learning
    what every other setting does.  Returns (exemplars, to_change, exemplar_id),
    or (None, None, None) when no control steps toward the target.
    """
    import math
    n = math.hypot(target_dir[0], target_dir[1]) or 1.0
    td = (target_dir[0] / n, target_dir[1] / n)

    def cos_to_target(step):
        s = math.hypot(step[0], step[1]) or 1.0
        return (step[0] * td[0] + step[1] * td[1]) / s

    exemplars = [c for c, st in per_control_steps.items() if cos_to_target(st) > 0.9]
    if not exemplars:
        return None, None, None
    to_change = [c for c in per_control_steps if c not in exemplars]
    return exemplars, to_change, exemplars[0]


def summarize(effects: dict) -> str:
    """One-line-per-control human/VLM-readable summary of the induced map."""
    if not effects:
        return "(no single-control effects attributable yet)"
    lines = ["controlled-experiment: induced control -> mover-endpoint effects (OFAT):"]
    for k, e in effects.items():
        lines.append(f"  - control {k!r}: {e.from_value!r}->{e.to_value!r} "
                     f"shifts the mover's endpoint by {e.endpoint_delta} "
                     f"(n={e.n})")
    return "\n".join(lines)
