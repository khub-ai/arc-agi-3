"""Game-agnostic perception substrate — prompt templates, grid
overlay utilities, and bbox rendering shared by every driver.

This module exists so the cross-game ExploratoryDriver and per-game
probes (like the bp35-specific run_bp35_sequence_probe.py) BOTH
import their substrate from a NEUTRALLY-NAMED source.  The actual
implementation lives in run_bp35_sequence_probe.py for historical
reasons (it grew up as a bp35 probe), but the SUBSTRATE itself —
prompt templates, gridding, overlay rendering — is game-agnostic.

PRIME DIRECTIVE: nothing imported here may reference a specific
game.  If a bp35-isms slips into one of the underlying definitions,
fix it there; this module is just a clean re-export point.

When a refactor moves the substrate into its OWN module body (so
the bp35 script becomes a thin caller), the re-exports here stay
stable — callers of perception_substrate don't need to change.
"""
from __future__ import annotations

# Re-export the substrate so callers can import "from
# perception_substrate import X" instead of "from
# run_bp35_sequence_probe import X".  The underlying source is
# considered an internal detail.
from run_bp35_sequence_probe import (   # noqa: F401
    # constants
    DEFAULT_N_TICKS,
    DEFAULT_LABEL_STRIDE,

    # grid overlay (adds dark margin + tick labels + minor/major
    # gridlines around any RGB frame)
    _add_grid_overlay,

    # bbox-on-grid renderer (cyan rectangles labeled #N, with
    # large-region and empty_cell filtering)
    render_turn1_overlay,

    # prompt formatting (substitutes n_ticks + label_stride into
    # SYSTEM_PROMPT_TEMPLATE and the single-frame / multi-frame
    # USER_PROMPT_TEMPLATE; all rules in the templates are
    # game-agnostic — composite exception, completeness, grid as
    # scaffold, group + relationship mining)
    _fmt_prompts,

    # tick-coord -> playfield-pixel conversion
    ticks_to_playfield_px,
)
