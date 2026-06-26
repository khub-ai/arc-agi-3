"""Default audit-check inventory.

Adapters extend this set by passing additional checks to
:func:`cognitive_os.audit.run_audit` — they don't mutate the default
list directly so different runs can use different inventories.

Two representative checks ship in the foundational implementation;
more land incrementally as failure modes warrant.
"""
from __future__ import annotations

from .goal_forest_coverage import GoalForestCoverageCheck
from .trusted_win_conditions import TrustedWinConditionsCheck

DEFAULT_CHECKS = (
    GoalForestCoverageCheck(),
    TrustedWinConditionsCheck(),
)

__all__ = [
    "DEFAULT_CHECKS",
    "GoalForestCoverageCheck",
    "TrustedWinConditionsCheck",
]
