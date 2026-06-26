"""Dialogic component — see ``docs/SPEC_dialogic_component.md``.

Two-way prose↔structure interface for the engine.  Foundational
implementation ships only the outbound (verbalize) direction; the
inbound (listen) direction is specced and deferred.

Public API:

    from cognitive_os.dialogic import verbalize, register_template
"""
from __future__ import annotations

from .verbalize import verbalize, register_template

__all__ = [
    "verbalize",
    "register_template",
]
