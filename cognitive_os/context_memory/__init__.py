"""Context & Memory component.

See ``docs/SPEC_context_memory_component.md`` for the full design.

Today this package contains the read-side bridge that translates
perception-substrate output into engine claims via the existing
``hypothesis_store``.  Future steps wire up the markdown-shaped
persistence encoding, hierarchical recall over scopes, and the
operator-facing purge surface.
"""
from __future__ import annotations

from .markdown_io import (
    load_committed_hierarchy,
    load_committed_markdown,
    migrate_jsonl_to_markdown,
    save_committed_hierarchy,
    save_committed_markdown,
)
from .perception_bridge import propose_from_parsed
from .purge import PurgeRecord, PurgeResult, purge
from .session import SessionResult, run_perception_session

__all__ = [
    "propose_from_parsed",
    "SessionResult",
    "run_perception_session",
    "load_committed_markdown",
    "save_committed_markdown",
    "load_committed_hierarchy",
    "save_committed_hierarchy",
    "migrate_jsonl_to_markdown",
    "purge",
    "PurgeRecord",
    "PurgeResult",
]
