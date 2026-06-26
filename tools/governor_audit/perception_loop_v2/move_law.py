"""Learn a click-to-relocate MOVE LAW from a SINGLE on-screen demonstration, then
reapply it NUMERICALLY.

Some games TEACH the move once: a marker / cursor shows the next position; you
click exactly on it and the controllable piece RELOCATES there; the marker then
disappears and never returns.  The transferable thing is a NUMBER -- the STEP the
demo showed:

    step = |marker - mover_before|        (and the piece moves TOWARD the click)

After the marker is gone you reapply it yourself -- to advance toward a goal G
from the current mover M you click

    M + step * unit(M -> G)

stepping until the piece overlaps G (never past it).  This is the numeric
generalization: keep the measured STEP, recompute the exact next click from the
CURRENT mover and the goal each turn -- so it transfers to the next level (where
mover and goal sit elsewhere) by re-measuring `step` from that level's own demo
and aiming at that level's goal.  Pure geometry; game-agnostic; no model cost.
"""
from __future__ import annotations
import math
from dataclasses import dataclass

Point = tuple


@dataclass
class MoveLaw:
    step: float                       # demonstrated click-to-move magnitude (ticks)
    kind: str = "click_to_point"      # the piece moves TOWARD the clicked point
    source: str = "demonstration"


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def learn_step_from_demo(mover_before, marker, mover_after, *, tol: float = 2.5):
    """From ONE demonstration: the actor clicked exactly on `marker` and the
    piece went from `mover_before` to `mover_after`.  If it LANDED on the marker
    (click-to-relocate confirmed within `tol`), return a MoveLaw whose numeric
    `step` is the demonstrated magnitude |marker - mover_before|.  Returns None
    when the move does not fit a click-to-point relocation (didn't land on the
    marker, or a zero-length step)."""
    if mover_before is None or marker is None or mover_after is None:
        return None
    if _dist(mover_after, marker) > tol:
        return None                                   # didn't relocate to the click
    step = _dist(marker, mover_before)
    if step < 1.0:
        return None
    return MoveLaw(step=float(step))


def next_click(law: MoveLaw, mover, goal):
    """The EXACT next click to advance the mover one step toward the goal:
    mover + step * unit(mover -> goal), clamped so it never lands PAST the goal
    (when the goal is within a step, click the goal itself).  Returns (row, col)
    in tick space."""
    dr, dc = goal[0] - mover[0], goal[1] - mover[1]
    d = math.hypot(dr, dc)
    if d <= 1e-6:
        return (float(mover[0]), float(mover[1]))
    s = min(law.step, d)                              # don't overshoot the goal
    return (mover[0] + s * dr / d, mover[1] + s * dc / d)


def reached(mover, goal, *, tol: float) -> bool:
    """True when the mover overlaps the goal within `tol` ticks."""
    return _dist(mover, goal) <= tol
