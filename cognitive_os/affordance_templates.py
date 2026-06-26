"""Affordance program templates — the small library of well-known
function programs the system can match observations against.

Each template builds a fresh :class:`AffordanceClaim` AST keyed on a
provided ``entity_class``.  Templates are pure constructors; they
embed no game-specific tokens.  The launcher template, for example,
describes the bouncer/launcher mechanic ("agent steps adjacent to a
bar on its attached side and gets traversed perpendicular to the
bar's axis until a barrier") using only the affordance DSL's
primitives; it applies equally to ls20 bouncer-bar entities, to any
arc-agi-3 game with the same mechanic, and to robotics analogues like
slide-plates on factory floors.

When new mechanics appear across multiple games and prove worth typing
as templates, they get added here.  Observation-mined affordances that
don't fit a named template are still perfectly valid claims; the
template registry is a convenience for the inference and prediction
paths, not a gate on what the system can learn.

See :doc:`docs/SPEC_entity_affordances.md` for the design.
"""

from __future__ import annotations

from typing import Optional

from .affordance_claim import AffordanceClaim
from .affordance_dsl import (
    AgentAction,
    AgentAt,
    And,
    AttachedSideOf,
    CellInDirection,
    CellOf,
    IsBarrier,
    NextCellInDir,
    OrientationOf,
    PerpendicularTo,
    SelfRef,
    Traverse,
)


def launcher_program(
    *,
    entity_class: str,
    action_id:    Optional[str] = None,
) -> AffordanceClaim:
    """Build a launcher AffordanceClaim for the given visual class.

    Semantics:
      Launch direction = perpendicular to ``self``'s orientation
        axis, pointing AWAY from the attached side (i.e. into open
        space, away from the wall the bar is fused to).
      Trigger: agent stands on the cell one step from ``self`` in
        the launch direction (the bar's free side), having
        dispatched action ``action_id`` (or any action when
        ``action_id`` is ``None``).
      Effect:  agent traverses in the launch direction until the
        next cell is a barrier.

    No magnitude is hardcoded.  The traversal evaluates against the
    current frame's ``passable_grid`` to produce the per-level concrete
    destination.

    Both ls20 L2 bouncers (which throw e.g. +5 east) and L3 bouncers
    (which throw e.g. -3 north) fit this template; the only difference
    is the binding of ``self`` and the per-level wall geometry the
    traversal evaluates against.
    """
    self_ref       = SelfRef()
    launch_dir     = PerpendicularTo(
        axis      = OrientationOf(self_ref),
        away_from = AttachedSideOf(self_ref),
    )
    return AffordanceClaim(
        entity_class = str(entity_class),
        trigger = And(
            AgentAt(CellInDirection(CellOf(self_ref), launch_dir, 1)),
            AgentAction(action_id=action_id),
        ),
        effect = Traverse(
            direction = launch_dir,
            until     = IsBarrier(NextCellInDir()),
        ),
        labels = {
            "template": "launcher",
        },
    )
