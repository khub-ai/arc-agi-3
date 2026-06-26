"""Backward-compatible shim.

The VLM-directed visual tools were refactored into the `substrate_tools/`
registry package (a consistent interface + add/remove/enable/disable management +
auto-generated prompt vocabulary + an open contribution path — see
docs/CONTRIBUTING_substrate_tools.md).  This module is kept so existing
`import visual_query` call sites keep working; new code should import from
`substrate_tools` directly.
"""

from __future__ import annotations

from substrate_tools import REGISTRY, render_vocabulary  # noqa: F401
from substrate_tools.frameutils import (                 # noqa: F401
    load_logical_frame,
    clamp_bbox as _clamp_bbox,
    hexof as _hex,
    background_rgb as _background_rgb,
    dominant_rgb as _dominant_rgb,
)
from substrate_tools.registry import run_queries as run_visual_queries  # noqa: F401


def _tool_names():
    return REGISTRY.names(enabled_only=True)


# Kept as a module attribute for callers that referenced visual_query.TOOL_NAMES.
TOOL_NAMES = _tool_names()
