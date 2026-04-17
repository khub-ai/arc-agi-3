"""Synthetic fixtures for Phase-5a tests.

Phase 5b will add recorded-frame fixtures captured from the live
ARC-AGI-3 SDK (one episode per game; compressed JSON).  For now,
everything here is hand-built so the tests have zero external
dependencies.
"""

from .synthetic import (
    blank_grid,
    moving_agent_episode,
    static_symmetric_grid,
    two_object_collision,
)

__all__ = [
    "blank_grid",
    "moving_agent_episode",
    "static_symmetric_grid",
    "two_object_collision",
]
