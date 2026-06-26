"""VLM-directed substrate tools — the contribution surface.

Importing this package gives you the REGISTRY and auto-registers every bundled
tool (importing each tool module runs its @tool decorators).  Env-var disables
($COS_DISABLED_TOOLS) are applied last.

  from substrate_tools import run_queries, render_vocabulary, REGISTRY
  results = run_queries(frame, queries, out_dir, n_ticks=64)
  prompt_block = render_vocabulary(n_ticks=64)

To CONTRIBUTE a tool: add a module here and decorate a handler with @tool(...),
then import it below so it self-registers.  See
docs/CONTRIBUTING_substrate_tools.md.
"""

from __future__ import annotations

from .registry import (   # noqa: F401
    REGISTRY,
    ToolContext,
    ToolSpec,
    ToolRegistry,
    tool,
    run_queries,
    render_vocabulary,
)

# --- bundled tool families (importing self-registers their @tool handlers) ---
from . import visual     # noqa: F401  -> zoom, highlight, count, measure, align,
#                                          palette, grid_readout
from . import animation  # noqa: F401  -> animation_zoom (inspect the framestack)
from . import trajectory  # noqa: F401  -> detect_trajectory (aim/launch ray)
from . import locate      # noqa: F401  -> locate_entity (reliable mover tracking)

# Apply run-time disables ($COS_DISABLED_TOOLS=zoom,palette) after registration.
REGISTRY.apply_env_disables()


def list_tools(**kw):
    """Convenience: REGISTRY.list(...)."""
    return REGISTRY.list(**kw)


def enable(name: str) -> bool:
    return REGISTRY.enable(name)


def disable(name: str) -> bool:
    return REGISTRY.disable(name)


def tool_names(*, enabled_only: bool = True):
    return REGISTRY.names(enabled_only=enabled_only)
