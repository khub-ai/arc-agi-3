"""arc_agi_3 — ARC-AGI-3 domain adapter for the Cognitive OS Engine.

This package is deliberately thin.  All reasoning — hypothesis
formation, AND-OR-CHANCE planning, curiosity-driven exploration,
post-mortem analysis, Option synthesis — lives in the domain-agnostic
engine at :mod:`cognitive_os`.  This package contributes only the
domain boundary:

* :class:`ArcAdapter` — :class:`cognitive_os.Adapter` implementation.
* :mod:`arc_agi_3.perception` — frame → symbolic events / entities.
* :mod:`arc_agi_3.action_mapping` — engine Action ↔ arc_agi native.
* :mod:`arc_agi_3.tools` — grid primitives registered in the engine's
  :class:`cognitive_os.ToolRegistry`.
* :mod:`arc_agi_3.observer` — *(Phase 5b)* VLM-backed visual oracle.
* :mod:`arc_agi_3.mediator` — *(Phase 5b)* LLM-backed common-sense oracle.
* :mod:`arc_agi_3.harness` — *(Phase 5b)* CLI for live competition runs.

The directive is strict: **no game-specific heuristics**.  The
adapter exposes what ARC-AGI-3 provides and lets the engine learn
what things mean.
"""

from .adapter import ArcAdapter
from . import action_mapping, perception, tools

__all__ = [
    "ArcAdapter",
    "action_mapping",
    "perception",
    "tools",
]

__version__ = "0.1.0"
