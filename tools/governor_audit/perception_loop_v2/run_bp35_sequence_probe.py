"""human:claude sequence probe for bp35 lc=0.

Goal: ask a VLM (Claude via the project's existing human:claude
pending-file protocol — see trial_driver.call_planner_human and
[[feedback_use_human_claude_pending_protocol]]) to look at a sample
of frames from one complete bp35 lc=0 trial and produce a single
structured analysis covering:

  - distinct entities + bbox on turn 1 + role hypotheses
  - relationships between entities (what triggers what, what contains
    what, what aligns with what)
  - the game's *type* (collection / maze / puzzle / platformer / ...)
  - the game's likely *purpose* (what wins it)
  - a per-transition summary of what changed between consecutive
    sample frames

This script BLOCKS, polling for the reply file.  It does not require
an API key.

Output layout (.tmp/bp35_sequence_probe/):

  pending/
    call_001_prompt.md           - paste-ready prompt (system + user)
    call_001_image_grid.png      - 3x3 composite (the upload)
    call_001_image_turn_NNN.png  - individual upscaled frames
    call_001_reply.txt           - operator writes Claude's JSON here
    call_001_reply.consumed.txt  - renamed after the script consumes it
    STATUS.txt                   - one-line state for external watchers

  response.json    - parsed reply (mirror of consumed reply, for re-rendering)
  turn1_overlay.png - turn-1 frame with VLM bboxes overlaid
  index.html       - viewer page

Usage:
    python -m perception_loop_v2.run_bp35_sequence_probe
    # script writes pending/call_001_prompt.md and call_001_image_grid.png,
    # then blocks waiting for pending/call_001_reply.txt to appear.
    # ...operator pastes Claude's JSON into pending/call_001_reply.txt...
    # script consumes it, renders, exits.
    #
    # To force a fresh call (new call_NNN), pass --new-call:
    python -m perception_loop_v2.run_bp35_sequence_probe --new-call
    # To just re-render from the last consumed reply (no waiting):
    python -m perception_loop_v2.run_bp35_sequence_probe --render-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PACKAGE_PARENT = HERE.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


FIXTURE_NAME = "bp35_lc0_seq01"
FIXTURE_DIR = HERE.parent / "perception_tests" / "fixtures" / FIXTURE_NAME
OUTPUT_DIR = Path(
    ".tmp/bp35_sequence_probe"
)

# 9 turns spaced ~every 5 across the 40-turn trial.  Picks include the
# first frame (initial state) and the final frame (terminal state) so
# the VLM sees the trajectory's endpoints.
SAMPLE_TURNS = [1, 6, 11, 16, 21, 26, 31, 36, 40]

UPSCALE_PER_FRAME = 2   # 512x512 -> 1024x1024 raw before compositing
GRID_CELL_PX = 512      # each cell in the 3x3 composite (downscale)

# Tick-coord overlay (substrate-agnostic viewing aid; NOT a game grid).
# N integer ticks per axis (default 64 = one tick per logical pixel for
# bp35, so truth and VLM bbox coords are always whole integers).
# Major labels appear only every LABEL_STRIDE ticks (default 4), so the
# labelled tick numbers are 0, 4, 8, 12, ..., N.  Between labels the
# intermediate integer positions get faint un-labelled gridlines.
# Bbox schema uses INTEGER coords [row_min, col_min, row_max, col_max]
# in range [0, N].
DEFAULT_N_TICKS = 64
DEFAULT_LABEL_STRIDE = 4
GRID_MARGIN_FRAC = 0.12   # ~12% of playfield dim per axis for labels
GRID_MARGIN_BG = (28, 28, 40)
GRID_LABEL_COLOR = (245, 245, 245)
GRID_ORIGIN_COLOR = (255, 200, 0)   # amber marker for (0,0) corner

# Both MAJOR and MINOR gridlines are drawn as TWO-TONE STRIPE PAIRS
# (dark pixels adjacent to light pixels) so they stay visible against
# ANY game background — single-tone grey blends invisibly with
# light-blue or pastel regions, and pure dark blends with walls.  The
# two-tone trick guarantees that at least one side has high contrast
# everywhere.
#
# MAJOR (labeled) lines use NEAR-BLACK + WHITE at high alpha so they
# read as dense, prominent stripes.  MINOR lines use the same
# structure but with lower alpha and a slightly softer light side,
# so they appear visibly FAINTER than majors — useful as sub-tick
# counting aids without competing with the labeled lines for
# attention.
#
# Width is 4 pixels (2 dark + 2 light) for both, large enough to
# survive any image-resampling step (browser downsampling on the
# trace page, or model-side resampling in the vision API).
GRID_LINE_DARK_RGBA   = (40, 40, 60, 240)    # MAJOR dark stripe
GRID_LINE_LIGHT_RGBA  = (255, 255, 255, 240) # MAJOR light stripe
GRID_LINE_WIDTH = 4                          # 2 dark + 2 light pixels

GRID_MINOR_DARK_RGBA  = (40, 40, 60, 110)    # MINOR dark stripe (fainter)
GRID_MINOR_LIGHT_RGBA = (235, 235, 245, 110) # MINOR light stripe (fainter)
GRID_MINOR_WIDTH = 4                          # 2 dark + 2 light pixels

# Kept for back-compat with anything still importing the old names.
GRID_LINE_RGBA = GRID_LINE_DARK_RGBA
GRID_MINOR_RGBA = GRID_MINOR_LIGHT_RGBA


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are a careful vision-language analyst observing {frame_intro} from
an interactive grid-based pixel-art puzzle game.{frame_extra}

Operational context (true for any game in this family, NOT game-specific):
  - The scene MAY contain a small AGENT the player controls (it moves
    between turns in response to ACTIONS) — OR the player's actions may
    DIRECTLY MODIFY the scene (toggle / select / place / cycle a target,
    flip a tile's colour) with NO moving agent at all.  Do NOT assume an
    agent exists; many games (pattern / matching / rule puzzles) have none.
  - The scene contains one or more BACKGROUND regions (floor / wall /
    interior / exterior) that fill most of the frame.
  - The scene may contain other distinct visual elements: collectables,
    threats, on-screen status indicators (HUD bars, counters),
    scenery, decorations.

PHYSICAL vs ABSTRACT INTERPRETATION (weigh BOTH; neither is the
default).  Some games emulate physical-world mechanics; others are
purely ABSTRACT / INFORMATIONAL (pattern, matching, rule, or
configuration puzzles) with no physics at all.  Do not force a
physical reading onto an abstract scene or vice-versa — pick whichever
the visual evidence supports.  When the scene DOES read as physical,
consider whether it fits a PHYSICAL system:

  - Is the AGENT a manipulator / gripper / robot arm tip, with rigid
    links connecting it to an anchor?  (Long thin entities that
    appear "attached to" the agent are often arm segments / rope /
    chain / hose, not decorations.)
  - Are the colored/distinct objects PHYSICAL BLOCKS that can be
    PICKED UP, PUSHED, STACKED, DROPPED, or used as TOOLS to push
    other blocks?
  - Does the scene obey GRAVITY (objects fall when unsupported) or
    INERTIA (things slide when pushed and stop against walls)?
  - When the agent "collects" an item, is the item GRABBED and now
    physically attached, occupying space on the agent's body / arm?
    A grabbed item still has physical extent and can collide with
    other objects.
  - Stacked blocks behave like a stack: lifting the bottom one lifts
    everything above; pulling the top one off requires reaching it
    from above or pushing it sideways.

Standard interaction primitives that show up across games (both UI
and PHYSICS varieties):

  UI-style primitives:
    - "cycle"  : an indicator advances through a list of options
    - "select" : an entity becomes the active target
    - "trigger": an entity activates a downstream effect when touched

  Physics-style primitives (use these when the scene reads as
  physical rather than UI):
    - "grab"   : the agent picks up a block (block now attached)
    - "release": the agent drops a held block (block falls / stays
                  in place depending on support)
    - "push"   : the agent or a held block displaces another block
                  in the direction of motion
    - "stack"  : a block placed on top of another stays atop it;
                  the whole stack can be lifted as a unit
    - "extend / retract": an arm or rope grows/shrinks in length;
                  longer arms can reach farther but also collide
                  with more things

When you observe an effect (a delta after an action), prefer the
PHYSICAL interpretation if it fits — it usually predicts later
turns better than a UI-style one.

Coordinate system — EVERY frame has a numbered INTEGER tick grid:
  - {n_ticks} integer positions per axis (0 through {n_ticks}).
  - LABELS appear every {label_stride} ticks: {label_values}.  Labels
    are in dark margin strips on ALL FOUR SIDES of the playfield (top,
    left, bottom, right) — the same label "k" appears on opposite
    sides so you can count from the nearest edge.
  - Each label "k" sits DIRECTLY ABOVE (or beside) the gridline at
    integer coordinate k.  A small tick mark in the margin connects
    each label to its gridline.
  - In between labels (positions 1, 2, 3 between labels "0" and "4",
    for example), the gridlines are still drawn but VERY FAINTLY.
    These are STILL integer positions — count them visually using
    the gridlines as anchors.
  - The top-left corner of the playfield is coordinate (0, 0); the
    bottom-right corner is ({n_ticks}, {n_ticks}).
  - The grid is purely a VIEWING AID — a counting tool so you can
    read positions off the labels.  It is NOT the game's native cell
    grid.

How to read coordinates:
  - Find the nearest labels and count integer gridlines between them
    and the entity edge.  All coordinates are WHOLE INTEGERS in
    [0, {n_ticks}].
  - Example: if the entity's left edge sits between labels "4" and
    "8", count the faint minor gridlines: if it sits at the 3rd
    minor gridline past label "4" (after positions 5, 6, 7), the
    bbox left coord is 7.

Bbox values are INTEGERS in [0, {n_ticks}].  Each integer position
corresponds to one logical pixel boundary in the game frame; the
{n_ticks} positions per axis match the game's underlying pixel
resolution, so integer coords are sufficient for exact bboxes.

Expected behaviour:
  - Look at where the entity's actual pixel edges fall RELATIVE to
    the nearest labeled gridlines and the faint minor gridlines in
    between.
  - Pick the nearest integer position for each edge.  Do NOT use
    fractional values — coordinates are always whole integers.

One entity per visible instance:
  - If you see N (>=2) distinct repeating instances — in a row, column,
    rectangular grid, diagonal, ring, or ANY arrangement — return EACH
    as its own entity (e.g. <kind>_1, <kind>_2, ... where <kind> is a
    short visual descriptor) with its own precise bbox.  DO NOT collapse
    them into one entity with a bbox spanning all of them.
  - The same applies to any repeating sprite or tile.  Per-instance
    bboxes are how the downstream pipeline knows where each one is.

COMPOSITE EXCEPTION — the per-instance rule has ONE exception, for
LINEAR PIXEL PATTERNS that are visually decorative or structural
rather than independent game units.  When you see 3 or more small
pixel clusters that meet ALL of the following criteria:
  (a) co-linear — they lie on a single straight line (horizontal,
      vertical, or diagonal)
  (b) evenly spaced — gaps between consecutive clusters are within
      +/-1 tick of each other
  (c) each cluster is <=2 ticks wide on at least one axis (the
      shape reads as a thin line, not a tile)
  (d) the line either CONNECTS two clearly-distinct larger entities,
      OR FORMS A CONTINUOUS BORDER / PERIMETER / TRAIL
treat the WHOLE set as ONE composite entity (e.g. names like
``dashed_connecting_line``, ``stippled_rod``, ``dotted_border_top``,
``particle_trail``, ``laser_beam``) with a single bbox spanning the
full extent of all clusters.  Do NOT enumerate the individual
clusters as separate entities.

Two-color alternation is a STYLING choice, not a structural
distinction — alternating clusters of two different colors that
otherwise meet (a)-(d) still form ONE composite entity.

If the criteria are only partially met and you're uncertain, prefer
the composite (one entity) over enumeration.  Downstream code can
DECOMPOSE a composite later when temporal evidence reveals the
clusters are actually independent (e.g. agent steps onto them
individually).  But N false collectables in the entity list is
harder to recover from than one slightly-too-broad composite.

Tiles vs composites — the criteria naturally separate the two cases:
collectable tiles in a row are typically THICK (>=3 ticks on both
axes) and not anchored to two distinct objects, so they fail
criterion (c) and/or (d) and remain per-instance.  Thin scattered
dots forming a line between two anchors satisfy all four and
collapse into one composite.

REPETITIVE-GRID COMPLETENESS — when the scene contains a regular
grid of similar tiles (rows x cols repeating pattern, like a
palette, sokoban floor, color-matching board, or pattern puzzle),
visual scanning alone tends to MISS one or two tiles AND
HALLUCINATE one or two phantom tiles at off-grid positions.  After
you list your tile entities, run these consistency checks before
returning:

  1. PER-ROW / PER-COLUMN COUNT — tally how many tile entities you
     placed in each row of the pattern and each column.  If one
     row or column has FEWER entities than the others, you
     probably MISSED a tile in that row/column — re-examine each
     expected cell and add the missing entity.  If one has MORE
     than the established pattern allows (e.g., a tile in a column
     that's otherwise empty), the extra one is likely a PHANTOM
     and should be removed.

  2. POSITION ON GRID — every tile entity in the regular pattern
     should have its bbox aligned to a cell on the inferred grid:
     bbox_left should equal ``origin_c + cell_size * N`` for some
     integer N, and bbox_top should equal ``origin_r + cell_size * M``
     for some integer M.  If a tile entity's bbox falls BETWEEN grid
     cells (off-pattern position), it's almost certainly either a
     misplaced bbox (snap to the nearest cell) or a phantom (remove).

  3. EXPLICIT EMPTIES — if your grid has cells that are genuinely
     empty (only background visible at that cell), it is OK and
     useful to list them as e.g. ``empty_cell_r<R>_c<C>`` entities
     with role ``scenery``.  This lets downstream code know those
     cells were CHECKED and intentionally empty, rather than just
     missing from your output.

  4. EXPECTED COUNT — before returning, state to yourself the
     expected tile count (rows * cols, minus any explicitly-gap
     cells) and verify your entity count matches.  Mismatch is a
     signal to re-examine.

These checks apply to TILES in a regular grid.  They do NOT apply
to free-form sprites, agents, or composites — only to entities
that fit the "repeating-tile-on-grid" pattern.

ORDERED ARRANGEMENTS (emit these as VISUAL FACTS) — whenever you
see TWO OR MORE same-role entities laid out in a regular line
(a row of HUD slots, a column of blocks, a strip of swatches),
emit the SEQUENCE as relationships so downstream reasoning can use
the order.  For each consecutive pair in the line, emit a
`precedes` relationship (from_entity = earlier-along-axis,
to_entity = next), and note the axis + any color/label sequence in
the evidence.  Do this for EVERY ordered line you see, INCLUDING
ones in different regions of the frame (e.g. a status/HUD strip at
the bottom AND a stack of objects in the play area).  Reporting the
ORDER is a visual fact; do NOT interpret what the order MEANS or
whether one region is a "target" for another — that is the acting
layer's job.  Your job is only to make the orderings visible.

DEPICTION REGIONS (report as visual facts) — if a STABLE region of
the frame contains a configuration of small elements that VISUALLY
RESEMBLE the main interactive elements (same colors / shapes, often
in a strip, panel, or corner), describe its CONTENT precisely in
`overall_notes`: which elements, in what ORDER, and CRUCIALLY each
element's INTERNAL CONDITION (solid/hollow, plain/center-marked,
intact/split, empty/filled).  Example phrasing (illustrative, not
game-specific): "bottom strip depicts, left-to-right: a gripper,
then three blocks each with a white center-mark, in colors A, B, C."
Report what it DEPICTS; do NOT say it is a goal/target/HUD or what
it means — the acting layer decides that.  Reporting the internal
condition is essential: the SAME region drawn with elements in a
different condition implies a different objective, and only a
faithful condition-level description lets the acting layer tell
those apart.

STRUCTURE AT ALL SCALES — recursion, transformation, drawn specifications
(this is the GENERAL principle; the specific cases above — rows of tiles,
linear connectors, depiction strips — are just instances of it).  Apply it to
EVERY scene, not only grid/tile games:

  - LOOK INSIDE elements, not only at them.  A single detected element is NOT
    necessarily the smallest unit of meaning.  A cell, token, marker, swatch, or
    "cursor" may ITSELF contain a smaller structure — a sub-grid, an icon, a
    glyph, a miniature picture, an internal arrangement of marks.  For any element
    that is not plainly uniform (ESPECIALLY small white/marked "cursor"/"label"
    cells), describe its INTERNAL structure (e.g. "this cell is itself a 3x3 of
    white/grey sub-cells with a dot at its centre"), not just its outline and
    dominant colour.

  - SAME STRUCTURE UP TO A TRANSFORMATION.  Two things can be "the same" or
    "related" even when they differ by SCALE (one is a smaller or larger copy),
    ROTATION, REFLECTION, RECOLOUR, partial occlusion, or other MINOR
    modification.  Do NOT require identical size/colour to call two structures
    related.  In particular, surface when a SMALL structured region mirrors the
    LAYOUT of a LARGER one (a miniature / thumbnail / minimap / key of it) — this
    cross-scale correspondence is a high-value visual fact.

  - A SPECIFICATION may be DRAWN somewhere in the scene.  A region — possibly
    small, possibly a legend / key / panel / sample, possibly NESTED INSIDE
    another element, possibly at a DIFFERENT SCALE than the thing it describes —
    may depict what some part of the scene should become.  Describe such a
    region's content and the internal CONDITION of each depicted element
    precisely (solid/hollow, which colour, which sub-cells filled).  Report it as
    a visual fact and, where one structure appears to specify another, say so as a
    relationship; do NOT decide it is "the goal" — the acting layer interprets it.

  - ANY ARRANGEMENT, not just rows and rectangular grids.  Repetition, ordering,
    regular structure, and completeness checks apply to whatever regular layout is
    present: a row, column, rectangular grid, DIAGONAL, RING / LOOP, RADIAL or
    HEX layout, NESTED structure, or a scattered set of marks that together form
    ONE shape / outline.  Use counts and pitch to detect missing or phantom
    instances in WHATEVER arrangement you see.

{visual_tools_block}

GRID AS SCAFFOLD, PIXELS AS TRUTH — when you've inferred a tile
grid (origin_ticks, cell_ticks, rows, cols), USE IT AS A SCAFFOLD
for KNOWING WHERE TO LOOK for tiles, but for EACH TILE'S BBOX
derive the edges from the tile's actual visible pixel edges, NOT
from grid extrapolation (i.e., NOT from
``origin_c + col_idx * cell_ticks``).

The grid is a guess — it can be slightly wrong, especially when:

  - The layout has TWO OR MORE CLUSTERS of tiles separated by
    NON-UNIFORM GAPS (e.g. left cluster, then a big gap, then a
    right cluster).  A single ``cell_ticks`` cannot describe both
    the within-cluster pitch AND the cross-cluster gap.

  - The grid origin you guessed is off by 1-2 ticks but the
    within-cluster pitch is right (so far-from-origin tiles
    accumulate small errors).

  - The cell pitch you guessed is fractional in reality (e.g. some
    rows are 7-tick-pitch, others 8-tick-pitch).

When a tile you can SEE doesn't match the grid prediction within
+/-1 tick on any edge, the GRID is wrong, not the tile.  Either:

  a) Correct the grid (re-examine origin / cell_ticks), OR

  b) Recognize that the layout is multi-cluster and describe it
     accordingly — set ``is_grid_based=true`` but use the grid as a
     loose framing only, and ensure every tile's bbox is anchored
     by its own pixel edges.

The completeness checks above (per-row/col counts, position on
grid) are still valuable — they help you NOTICE when a tile's
position contradicts the grid.  But the FIX when they fire is to
adjust either the GRID or the TILE-POSITION; the source of truth
is always the actual pixels in the frame.

Bbox must TIGHTLY enclose the entity's own pixels — exclude
background/floor/wall:
  - The bbox of an entity must NOT include any adjacent background,
    floor, or wall pixels that are visually separate from the entity.
  - If a green tile sits on a black background with a 1-pixel dark
    gap between the tile and the next tile, the bbox of each tile
    must END at the tile's own edge.  The gap pixels are NOT part
    of either entity.
  - The bbox should NEVER span a whole row or column of pure
    background or floor.  Background and floor are SEPARATE entities
    (large region entities with their own bboxes), not contributors
    to a sprite's bbox.

Identical entities have identical dimensions:
  - When you return multiple entities of the same kind (e.g. 7 green
    tiles that all look the same size), they MUST have the same bbox
    DIMENSIONS — same width and same height in tick units.  Only
    their position (row/col offsets) differs.
  - If your visual estimate gives slightly different sizes for what
    should be identical sprites, pick the most-confident size and
    apply it uniformly to all instances.  Consistency is more
    important than capturing every pixel-level variation.

PRESERVE GAPS between repeating tiles:
  - When repeating tiles are arranged in a row or column, look for
    visible GAP pixels between them — a thin strip of background
    (often 1 pixel wide) separating each tile from the next.
  - The bbox positions MUST respect these gaps.  Two adjacent tiles
    with a 1-pixel gap between them must have bboxes that do NOT
    touch: tile_1's right edge ends, then 1 gap pixel of background,
    then tile_2's left edge begins.
  - PITCH = tile_size + gap_size.  If tiles are 5 pixels wide with
    1-pixel gaps, the pitch is 6 pixels.

READ EACH EDGE'S ACTUAL PIXEL POSITION — do NOT snap to major labels:

The major labels ({label_values}) are visual counting anchors, NOT
allowed values for bbox edges.  An entity's edge can fall at ANY
integer position 0-{n_ticks}, including positions BETWEEN major
labels (e.g. 21, 22, 23, 37, 41, 45 are all valid bbox edges, not
just 20, 24, 28, 36, 40, 44).

ANTI-SNAP RULE: if your bbox edges happen to land on ALL FOUR major
labels for a small entity (e.g. [20, 44, 24, 48] when the entity
might actually be at [21, 44, 23, 47]), pause and re-examine.  Real
sprites are placed by the artist at arbitrary pixel positions; only
truly cell-aligned tiles in a uniform grid will have all-major edges.

HOW TO READ AN EDGE:
  1. Find the nearest major label to the edge.
  2. Look at the faint minor gridlines between that label and the
     edge.  Each minor gridline is ONE integer position.
  3. Count the minor lines from the major label to the entity's
     actual pixel edge.  Do not round up or down to the nearest
     major — report the exact integer position you land on.
  4. If the entity edge sits BETWEEN two minor gridlines, pick the
     gridline CLOSER to the actual pixel edge (not the closer major
     label).

The faint minor gridlines are integer positions.  They are dim by
design (so they don't dominate visually) but they are RELIABLE
counting marks.  Use them.

FIND THE WIDEST ROW / TALLEST COLUMN for each sprite:
  - For each sprite, examine EVERY row it occupies.  Some rows have
    more visible coloured pixels than others.  The bbox WIDTH equals
    the count of consecutive integer columns covered by the WIDEST
    row (the row with the most coloured pixels), NOT a narrower row.
  - Same for height: examine every column the sprite occupies; the
    bbox HEIGHT equals the count of integer rows covered by the
    TALLEST column.
  - This catches a common case: a rounded-corner sprite has narrow
    rows at the top and bottom (corner rows with the corner pixels
    missing), and a wider row in the middle.  Pick the WIDER row
    to set the bbox left/right extent.
  - Example: a 5x5 rounded sprite has rows [3, 5, 5, 5, 3] wide.
    The bbox WIDTH is 5 (taken from the middle row), NOT 3 (taken
    from the top/bottom row).  The bbox LEFT EDGE extends ONE
    POSITION beyond the visible top-row pixels to include the column
    where the middle row's leftmost pixel sits.

HYPOTHESIS TESTING for tile bbox edges (especially LEFT and TOP):
  - Once you've identified the visible solid-colour block of a tile,
    consider TWO possibilities for the bbox left edge:
      H1: bbox left = visible solid colour's left column.
      H2: bbox left = H1 - 1 (one column earlier — accounts for a
          possible empty/dark CORNER pixel that's part of the sprite
          shape but invisible against a dark background).
  - Test each hypothesis against the IMAGE: under H1, the leftmost
    "dark band" before the tile must be a TRUE background gap of
    arbitrary width.  Under H2, the column at H2's position is a
    sprite-corner position, and the GAP between this tile and the
    previous (if any) is only 1 pixel wide on average.
  - For tiles arranged in a regular row, the gap between tiles is
    usually 1 pixel.  If under H1 your tiles end up TOUCHING the next
    tile with no gap, switch to H2 — H1 was missing the corner column.
  - Apply the same hypothesis test to the TOP edge.

VERIFICATION PASS — run this AFTER placing all entities:
  - For each row or column of repeating tiles, measure TWO independent
    things on the IMAGE:
      A. The pixel position of the LEFTMOST tile's left edge (read it
         directly from labels).
      B. The pixel position of the RIGHTMOST tile's right edge (read
         it directly from labels — DO NOT compute it from A + pitch).
  - Then check: does (B - A) match what your placed bboxes give you
    when computed as (N * tile_size + (N-1) * gap_size)?
  - If there's a mismatch, your initial reads of A or B were off.
    Re-read both endpoints CAREFULLY from the labels — typical errors
    are off-by-1 or off-by-2 because the leftmost tile's left edge is
    very close to but NOT exactly at the major label.
  - It is BETTER to adjust both A and B than to anchor on a single
    endpoint.  Find the consistent placement that matches BOTH
    endpoints in the image.

For SMALL SPRITES (agents, single tokens — anything that looks
roughly 3-5 pixels per side):
  - DO NOT apply the rounded-corner / border-pixel extension rule.
    Small sprites are usually drawn as a fully-filled compact shape
    with no empty bounding pixels; the bbox matches the visible
    pixel extent EXACTLY.
  - Count the visible coloured pixels along each edge.  Width and
    height should be reported as the EXACT visible extent — do not
    add 1 for a hypothetical corner.
  - For a sprite that has two color regions (e.g. blue body + yellow
    accent merged into one entity), the bbox bounds the FULL extent
    of both regions combined — but no further.  If the yellow extends
    just 1 pixel beyond the blue, the bbox right edge = (last yellow
    column) + 1.  Do NOT extend by an extra pixel.
  - Pick TWO reference labeled gridlines (e.g. the major labels above
    and to the left of the sprite).
  - Count the EXACT number of integer positions from each label to
    each edge of the sprite.  Use the minor gridlines as counting aids.
  - Example: a sprite occupying cols 20, 21, 22 (3 columns of visible
    colour) has bbox right = 23 (exclusive), NOT 24 or 25.

Identify what's happening in the sequence using visual evidence and
your world knowledge of pixel-art puzzle games.  Be CONCRETE about
what you see and HONEST about uncertainty — a small sample of frames
gives you strong evidence for some things and weak evidence for
others.

Do NOT speculate about ANY mechanic, win condition, or rule unless
the frames give you specific visual evidence for it.

Return ONLY a JSON object with the exact shape described below.  No
prose, no markdown fences.
"""

USER_PROMPT_TEMPLATE = """\
You are looking at a 3x3 grid composite image.  The 9 cells, in
row-major order (top-left first), show the frames at turns 1, 6, 11,
16, 21, 26, 31, 36, 40 of one trial.  Each cell has a "turn N" header
strip at the top, then the gridded game frame underneath.

EVERY frame has the same {n_ticks}x{n_ticks} tick grid overlay:
column labels 0..{n_minus_1} run across the top margin, row labels
0..{n_minus_1} run down the left margin.  Coordinate (0.0, 0.0) is
the top-left CORNER of the playfield (NOT the top-left of the entire
composite cell — the labels are in dark margin strips outside the
playfield).  Coordinate ({n_ticks}.0, {n_ticks}.0) is the bottom-right
corner.

Return a single JSON object with EXACTLY these top-level keys:

{{
  "entities": [
    {{
      "name":               "<short snake_case slug, unique>",
      "first_seen_turn":    <int — first turn this entity is visible>,
      "still_present_turn40": <true|false>,
      "bbox_ticks_turn1":   [row_top, col_left, row_bottom, col_right] | null,
            // INTEGERS in [0, {n_ticks}] representing the entity's
            // visual extent on the turn-1 frame.  Each integer position
            // corresponds to one logical pixel boundary.  Examples
            // (illustrative; not from any specific game):
            //   - a 4x3 sprite at rows R..R+3 and cols C..C+2:
            //       [R, C, R+4, C+3]   (note: bottom/right exclusive)
            //   - a 5x5 tile at rows R..R+4 and cols C..C+4:
            //       [R, C, R+5, C+5]
            // null if the entity is not visible at turn 1.
            //
            // PER-INSTANCE: if you see N distinct repeating tiles,
            // return N separate entities (e.g. <kind>_1, <kind>_2,
            // ... where <kind> is a short visual descriptor).  DO
            // NOT return one entity with a bbox that spans multiple
            // tiles.
      "appearance":         "<short visual description, INCLUDING the
                             entity's INTERNAL CONDITION: solid vs
                             hollow, plain vs center-marked / pierced,
                             intact vs split, empty vs filled, outlined
                             vs flat.  This internal condition is a
                             VISUAL FACT and is often what a goal
                             depiction differs by — always report it.>",
      "behavior_observed":  "<one short phrase based on what you saw
                             across the sequence: 'static', 'moves
                             around', 'appears then disappears', etc.>",
      "role_hypothesis":    "<one of: agent, collectable, threat,
                             hud, wall, scenery, decoration,
                             trigger_target, unknown>",
      "confidence":         "low" | "medium" | "high"
    }},
    ...
  ],

  "relationships": [
    {{
      "from":       "<entity name>",
      "to":         "<entity name>",
      "relation":   "<one short phrase: 'agent collects target',
                     'target disappears when agent adjacent',
                     'HUD counter decrements with each collected',
                     'wall blocks agent', etc.>",
      "evidence":   "<one sentence pointing to a specific frame or
                     pair of frames>",
      "confidence": "low" | "medium" | "high"
    }},
    ...
  ],

  "frame_to_frame_summary": [
    {{
      "from_turn":    1,
      "to_turn":      6,
      "what_changed": "<short summary of the deltas you can see>"
    }},
    ... (one entry per consecutive sample pair, so 8 entries total)
  ],

  "overall_notes": "<one paragraph: things you are confident about,
                    things you are uncertain about, and any
                    observation that doesn't fit the structured
                    fields above>"
}}

Return ONLY the JSON object.  No prose, no markdown fences.
"""


REFINEMENT_PROMPT_TEMPLATE = """\
This is a SECOND-PASS bbox refinement.  You previously identified
entities in a game frame and gave each a bbox in integer tick coords.
Here are TWO images:

  1. The original gridded frame.
  2. The same frame with YOUR PREVIOUS BBOXES drawn on top as cyan
     rectangles, each labeled with its entity number "#N".

YOUR TASK is to detect and fix BBOX EDGE DRIFT — bboxes that are
off by 1-3 ticks on one or more edges.  Do NOT change entity
identities, names, or counts.  Only adjust bbox numbers.

PER-EDGE CHECKLIST — for EVERY entity (not just the obviously
wrong ones), inspect ALL FOUR edges of its cyan rectangle:

  TOP edge:
    - Does the cyan line sit on the row JUST ABOVE the entity's
      topmost coloured pixel?  (Remember: top is INCLUSIVE — a
      pixel at row 20 has bbox.row_top = 20.)
    - If you see the entity's top pixel below the cyan line by 1-2
      rows, MOVE the cyan line DOWN by that amount.
    - If you see the cyan line cuts INTO the entity (entity pixels
      visible ABOVE the cyan line), MOVE the cyan line UP.

  BOTTOM edge:
    - Bottom is EXCLUSIVE — a pixel at row 23 (last filled row) has
      bbox.row_bottom = 24.
    - Check: does the cyan line sit on the row JUST BELOW the
      entity's bottom-most coloured pixel?
    - If there's a 1-2 row gap between entity bottom and cyan line,
      move the cyan UP.  If the cyan cuts INTO the entity, move DOWN.

  LEFT and RIGHT edges:
    - Same rules as TOP/BOTTOM, but for columns.

GAPS BETWEEN ADJACENT ENTITIES — visible BACKGROUND pixels between
two entities MUST be respected:
  - If two repeating tiles in a row/column have 1+ background pixel
    between them, their bboxes MUST NOT touch.  e.g., red tile ends
    at col 47 (bbox.col_right=48), blue tile next to it starts at
    col 49 (bbox.col_left=49) — NOT col 48 (which would mean they
    touch).
  - DO NOT collapse a visible gap.  The presence of a gap is more
    reliable than the precise edge position.
  - When the cyan rect of tile_A and tile_B share a coordinate
    (e.g., A.right = B.left), check the IMAGE: is there a visible
    background gap between them?  If yes, ADD 1-2 ticks of
    separation by moving one bbox.

HUD AND SMALL SPRITES — these often have 1-tick drift on multiple
edges because they're only 4-6 ticks wide.  Be especially careful
about:
  - HUD strip background height — does the strip's actual visible
    grey region match the bbox rows?  (NOT including the cells/
    sprites that sit ON the strip — the background bbox is the
    strip itself.)
  - HUD cells (small 4-tick squares) — verify ALL four edges.
  - Agent sprites with hollow outlines — the bbox bounds the
    OUTER outline, not the hollow centre.

ANTI-SNAP CHECK — bboxes commonly drift to MAJOR LABEL positions
(the labeled gridlines every 4 ticks: 0, 4, 8, ...) when the actual
entity edge is 1-2 ticks AWAY from a major label.  This happens
because major labels are visually prominent while minor sub-tick
gridlines are dim.

For EACH bbox edge in your first-pass output:
  - If the edge value is a multiple of 4 (sits on a major label),
    verify by carefully examining the IMAGE at that position.
  - Check the entity's actual pixel edge: does it sit ON the major
    gridline, or 1-3 ticks to the side?
  - If 1-3 ticks away, MOVE the bbox edge to the actual integer
    position (e.g. 21, 22, 23 are all valid — they are NOT to be
    rounded to 20 or 24).

A small sprite whose bbox came out as ALL FOUR major-label edges
(e.g. [20, 44, 24, 48]) is especially suspect.  Re-read each
edge against actual pixels.  The correct bbox often looks like
[21, 44, 24, 47] or [20, 45, 24, 48] — mixing major and non-major
positions.

DIRECTION OF CORRECTION — when uncertain whether to expand or
shrink a bbox by 1 tick, prefer:
  - TIGHTER bbox if extra background pixels are clearly visible
    inside the cyan rectangle around the entity.
  - LARGER bbox if the entity pixels touch or cross the cyan line.

Output the FULL JSON object same as the first pass (entities,
groups, relationships, grid_inference, symbolic_state, game_type,
game_purpose, frame_to_frame_summary, overall_notes).

ALSO re-fill the relationships, grid_inference, and symbolic_state
fields based on the refined bboxes — if a bbox correction shifts an
entity into a different cell, update entity_cells, agent_cell,
traversable_cells, blocked_cells, and goal_candidate_cells
accordingly.

KEEP overall_notes brief: list which bboxes you adjusted and which
edges, e.g. "tile_blue: row_top 24->26 (gap with tile_red restored);
hud_cell_red: col_right 32->31 (right edge tightened)."
"""


SINGLE_FRAME_USER_PROMPT_TEMPLATE = """\
You are looking at a SINGLE frame: the very first frame (turn 1) of
a trial.  No sequence is provided.

The frame has a {n_ticks}-tick INTEGER grid overlay: major labels
({label_values}) appear in the dark margin strips on all four sides,
with each label "k" anchored to the gridline at integer coordinate k.
A small tick mark in the margin connects each label to its gridline.
Between major labels the intermediate integer positions (1, 2, 3
between labels "0" and "4", etc.) have faint minor gridlines.
Coordinate (0, 0) is the top-left CORNER of the playfield;
({n_ticks}, {n_ticks}) is the bottom-right corner.  ALL coordinates
are integers.

Return a single JSON object with EXACTLY these top-level keys (the
sequence-related fields are kept but can be left as empty arrays /
"unknown" strings since you only see one frame):

{{
  "entities": [
    {{
      "name":               "<short snake_case slug, unique>",
      "first_seen_turn":    1,
      "still_present_turn40": "unknown",
      "bbox_ticks_turn1":   [row_top, col_left, row_bottom, col_right] | null,
            // INTEGERS in [0, {n_ticks}] representing the entity's
            // visual extent.  Bottom/right are EXCLUSIVE (a 1-pixel
            // sprite at row 5 col 3 has bbox [5, 3, 6, 4]).
            // PER-INSTANCE: if you see N distinct repeating tiles,
            // return N separate entities (e.g. <kind>_1, <kind>_2,
            // ... where <kind> is a short visual descriptor) each
            // with its own precise bbox.
            // null only if the entity is genuinely not visible.
      "appearance":         "<short visual description>",
      "behavior_observed":  "unknown - single frame",
      "role_hypothesis":    "<one of: agent, collectable, threat,
                             hud, wall, scenery, decoration,
                             trigger_target, unknown>",
      "confidence":         "low" | "medium" | "high"
    }},
    ...
  ],

  "groups": [
    // ACTIVELY MINE for groups — every multi-entity scene has at
    // least two kinds of groupings worth surfacing:
    //
    //   (A) SIMILARITY GROUPS — entities that share visual signature
    //       (palette + shape + size).  If you see N>=2 entities that
    //       look the same, group them.  These probably play the same
    //       role in the game mechanic (collectables, walls, targets,
    //       etc.).  Use criterion "similar_appearance".
    //
    //   (B) SPATIAL GROUPS — entities clustered in a region of the
    //       playfield, OR aligned along a row / column / diagonal,
    //       EVEN IF they don't share appearance.  Use criterion
    //       "colinear_row", "colinear_col", or "spatial_cluster".
    //
    // BOTH types of grouping are independent — an entity often
    // belongs to one similarity group AND one spatial group, and
    // surfacing both makes downstream reasoning much richer.  E.g.,
    // tiles arranged in two rows might form:
    //     - one similarity group: "all_collectable_tiles"
    //     - two spatial groups:   "top_row_tiles", "bottom_row_tiles"
    //
    // Groups must have at least 2 members.  Group members must be a
    // subset of the entities array above.  Naming convention:
    // snake_case describing the SHARED PROPERTY.
    {{
      "name":      "<group name>",
      "members":   ["<entity name>", ...],
      "criterion": "<one of: similar_appearance, colinear_row,
                    colinear_col, colinear_diagonal, spatial_cluster>",
      "evidence":  "<one sentence>",
      "confidence": "low" | "medium" | "high"
    }},
    ...
  ],

  "relationships": [
    // ACTIVELY MINE for STRUCTURAL / SPATIAL relationships you can
    // READ off the single frame.  Do not invent causal / mechanic
    // relationships — those need temporal evidence.  For every
    // distinct pair of entities OR groups that has a meaningful
    // spatial relationship, return an entry.  Relation must be one of:
    //
    //   "contains"               — A's bbox fully encloses B's
    //   "adjacent_to"            — A and B share a touching edge
    //   "aligned_horizontally"   — A and B share the same row band
    //   "aligned_vertically"     — A and B share the same column band
    //   "similar_appearance"     — A and B share visual signature
    //                              (same palette / shape / size)
    //   "separated_by"           — A and B are non-adjacent but
    //                              separated by a clearly-visible third
    //                              entity (specify the separator in
    //                              evidence)
    //   "mirrors"                — A and B are visual mirror images
    //                              (reflected horizontally, vertically,
    //                              or about a centerline)
    //   "paired_with"            — A and B are clearly a matched pair
    //                              (same color highlighted in two
    //                              places, two halves of a shape, etc.)
    //   "precedes"               — A is positionally BEFORE B along a
    //                              named axis.  Specify the axis in
    //                              the evidence field (one of
    //                              "horizontal", "vertical",
    //                              "left_to_right", "top_to_bottom").
    //                              A chain of `precedes` relations
    //                              encodes a structural ORDER among
    //                              the entities — this is a
    //                              HYPOTHESIS about a possible
    //                              execution order, not an assertion
    //                              that the order is required.  Emit
    //                              whenever two or more same-role
    //                              entities lie along an axis; emit
    //                              pair-wise between successive
    //                              neighbours rather than only the
    //                              endpoints, so the substrate has
    //                              the full chain.
    //
    // "from" and "to" can name EITHER an individual entity OR a group
    // (from the groups array above).  PREFER group-level
    // relationships when the relationship holds for the whole group
    // (e.g. "<tile_group> adjacent_to <region>" is better than
    // enumerating each tile-region adjacency separately).
    //
    // What to look for, exhaustively:
    //   - which entities CONTAIN other entities (rooms, frames, HUD strips)
    //   - which entities are ADJACENT (touching) to which
    //   - which groups (similarity or spatial) align with which other groups
    //   - any mirror / pair / symmetry observations
    //   - any HUD / status indicator entity that mirrors a game-state
    //     entity (e.g., a HUD counter color-matched to a target tile)
    //   - any ORDERED ARRANGEMENT of same-role entities — two or
    //     more entities sharing a role and lying along a common
    //     axis (a row, a column, a diagonal).  When detected, emit
    //     pair-wise `precedes` relations capturing the layout
    //     order between successive neighbours.  This is a
    //     structural observation; whether the order is meaningful
    //     to gameplay is for downstream layers to decide.
    //
    // Many games have a tell where two entities of the same color
    // are matched (e.g., color-coded selection puzzles, paired
    // sokoban targets).  ACTIVELY look for such pairings and surface
    // them as "paired_with" or "similar_appearance" relationships.
    {{
      "from":       "<entity name OR group name>",
      "to":         "<entity name OR group name>",
      "relation":   "<one of the values above>",
      "evidence":   "<one sentence based on the single frame>",
      "confidence": "low" | "medium" | "high"
    }},
    ...
  ],

  "frame_to_frame_summary": [],

  "grid_inference": {{
    // Does the scene show a repeating cell pattern (tile grid /
    // checkerboard / sokoban-style cells)?  This is what lets a
    // symbolic planner treat positions as discrete cell indices
    // rather than raw pixels.
    //
    // If yes, identify:
    //   - cell_ticks:  side length of one cell in INTEGER ticks
    //                  (e.g. if cells are 8 ticks wide, cell_ticks=8)
    //   - origin_ticks: [row_offset, col_offset] of the TOP-LEFT
    //                   corner of cell (0, 0) in tick coords.  Often
    //                   non-zero because the visible playfield has
    //                   a border / margin / wall of cells.
    //   - rows, cols:  the size of the cell grid covering the
    //                  visible playfield
    //
    // If the scene is NOT cell-based (continuous arena, free-form
    // shapes, platformer with arbitrary positions), set
    // is_grid_based=false and leave the other fields null.
    "is_grid_based":   true | false,
    "cell_ticks":      <int> | null,
    "origin_ticks":    [row_offset, col_offset] | null,
    "rows":            <int> | null,
    "cols":            <int> | null,
    "evidence":        "<one sentence describing the repeating pattern
                        you used to infer the cell size and origin>",
    "confidence":      "low" | "medium" | "high"
  }},

  "symbolic_state": {{
    // Project every entity onto the inferred cell grid so a planner
    // can reason in cell coordinates.  Only fill these fields if
    // grid_inference.is_grid_based is true.  Cell indices are
    // [row, col] zero-based, with (0, 0) at the top-left of the
    // grid (NOT the top-left of the playfield in tick coords —
    // grid origin_ticks already accounts for that).
    //
    // - agent_cell:           which cell the controlled agent
    //                         currently occupies (one cell).
    // - entity_cells:         map of entity name -> cell index, OR
    //                         a list of cell indices for multi-cell
    //                         entities (rooms, walls spanning many
    //                         cells).  Use the SAME entity names as
    //                         in the entities list above.
    // - traversable_cells:    cells the agent could plausibly step
    //                         into based on visual evidence (open
    //                         floor / interior).  Includes
    //                         goal_candidate_cells.
    // - blocked_cells:        cells visually filled by walls / solid
    //                         obstacles the agent cannot enter.
    // - goal_candidate_cells: cells containing entities that look
    //                         like things the agent would interact
    //                         with (collectables, targets, exits).
    //                         Game-agnostic — just "things that
    //                         visually stand out as interactive".
    //
    // For non-grid games (is_grid_based=false), set agent_cell=null
    // and leave the cell lists empty.
    "agent_cell":            [row, col] | null,
    "entity_cells":          {{
      "<entity name>":       [row, col] | [[r1,c1], [r2,c2], ...]
    }},
    "traversable_cells":     [[row, col], ...],
    "blocked_cells":         [[row, col], ...],
    "goal_candidate_cells":  [[row, col], ...],
    "confidence":            "low" | "medium" | "high"
  }},

  // OPTIONAL — include ONLY if you want the substrate to MEASURE something you
  // cannot reliably eyeball (see "ON-DEMAND VISUAL TOOLS" in the system
  // prompt).  Each item: an op (zoom|grid_readout|count|measure|align|palette|
  // highlight) plus its args, in tick coordinates.  Omit this key entirely when
  // you don't need help.  When present, the substrate fulfils the queries and
  // asks you to re-emit this perception with the answers folded in.
  "visual_queries": [
    {{ "op": "zoom", "id": "<short>", "bbox": [r0, c0, r1, c1] }},
    ...
  ],

  "overall_notes": "<one paragraph>"
}}

Return ONLY the JSON object.  No prose, no markdown fences.
"""


# How many top-credence promoted lessons to surface in the perception
# prompt's prior-knowledge block.  Bounded so the prompt stays compact.
_PERCEPTION_PRIOR_TOP_K = 8
# Minimum credence for a lesson to be surfaced.  Below this it's still
# noisy enough that re-deriving from the frame is preferable.
_PERCEPTION_PRIOR_MIN_CREDENCE = 0.7


def _build_prior_knowledge_block(game_id: str | None) -> str:
    """Render a 'PRIOR-TRIAL KNOWLEDGE' section for the perception
    system prompt.

    Two modes:

    * **Warm** — `per_game_lessons.json` has promoted high-credence
      lessons for `game_id`.  Surface the top-K and instruct the VLM
      to REFINE rather than regenerate (treat known mechanics as
      reliable; treat refuted hypotheses as ruled out).
    * **Cold** — no prior knowledge for this game (or no game_id
      supplied).  Emit a MASTER-GAMER cold-start framing that leans on
      the VLM's own knowledge of puzzle genres and on real-world
      physics analogues (gravity, leverage, magnetism, container/
      contents, lock-and-key, etc.).  The substrate stamps NO
      game-specific guess — the VLM commits one from its own priors.

    Returns a string that drops cleanly into the system prompt; never
    raises (a missing KB or import failure degrades to cold).
    """
    cold_block = (
        "PRIOR-TRIAL KNOWLEDGE — none on file for this game.  This is "
        "a COLD START: make a MASTER-GAMER guess for `game_type` and "
        "`game_purpose` from this frame alone, drawing on:\n"
        "  (a) your world knowledge of puzzle-game GENRES (sokoban, "
        "collection, navigation, cycle-and-select, platformer, lock-"
        "and-key, sorting, pattern-matching, manipulator/grabber, "
        "etc.) — recognise familiar shapes early.\n"
        "  (b) REAL-WORLD PHYSICS ANALOGUES (gravity, leverage, "
        "friction, magnetism, container/contents, lock-and-key, "
        "lever-arm, gear-train, pierce-and-hold).  Many games in this "
        "family map directly to physical-world mechanics; that "
        "mapping is usually the right one.\n"
        "Frame your guess in those terms.  Mark `confidence` honestly "
        "— `low` is fine on turn 1.\n"
    )
    if not game_id:
        return cold_block
    try:
        from per_game_lessons import (    # noqa: WPS433
            load_for_game, rank_lessons,
        )
    except Exception:
        return cold_block
    try:
        lessons = load_for_game(game_id)
    except Exception:
        return cold_block
    if not lessons:
        return cold_block
    ranked = rank_lessons(lessons)
    promoted = [
        l for l in ranked
        if getattr(l, "promoted", False)
        and getattr(l, "credence", 0.0) >= _PERCEPTION_PRIOR_MIN_CREDENCE
        and getattr(l, "kind", "") != "refuted"
    ]
    refuted = [
        l for l in ranked
        if getattr(l, "kind", "") == "refuted"
        or getattr(l, "credence", 1.0) <= 0.2
    ]
    if not promoted and not refuted:
        return cold_block
    parts: list[str] = [
        f"PRIOR-TRIAL KNOWLEDGE for game `{game_id}` — REFINE, do NOT "
        "regenerate from scratch.  Treat the items below as the "
        "actor's own accumulated experience across past trials; use "
        "them to ground your `game_type` / `game_purpose` guesses.",
        "",
    ]
    if promoted:
        parts.append(
            f"  KNOWN (promoted, credence>={_PERCEPTION_PRIOR_MIN_CREDENCE}) — "
            f"treat as reliable:"
        )
        for l in promoted[:_PERCEPTION_PRIOR_TOP_K]:
            desc = (getattr(l, "description", "") or "").strip()
            if not desc:
                continue
            cred = getattr(l, "credence", 0.0)
            kind = getattr(l, "kind", "") or "lesson"
            parts.append(f"    - [{kind} c={cred:.2f}] {desc}")
    if refuted:
        parts.append("")
        parts.append(
            "  REFUTED — these hypotheses were tried and FAILED in "
            "past trials.  Do NOT propose them again as the game's "
            "purpose or mechanic:"
        )
        for l in refuted[:_PERCEPTION_PRIOR_TOP_K]:
            desc = (getattr(l, "description", "") or "").strip()
            if not desc:
                continue
            parts.append(f"    - {desc}")
    parts.append("")
    parts.append(
        "Use the above to inform `game_type` and `game_purpose`: "
        "refine the current best understanding, tag anything still "
        "uncertain, and (per the PHYSICS-FIRST block above) prefer "
        "physical analogues over UI ones when the scene supports it."
    )
    return "\n".join(parts) + "\n"


def _fmt_prompts(n_ticks: int,
                  label_stride: int = DEFAULT_LABEL_STRIDE,
                  single_frame: bool = False,
                  game_id: str | None = None,   # accepted, unused:
                  # perception is interpretation-free; prior-trial
                  # knowledge is surfaced on the ACTING-VLM side only.
                  ) -> tuple[str, str]:
    """Substitute n_ticks + label_stride into the system + user prompts.
    With label_stride > 1, major labels appear only every Nth tick (e.g.
    0, 4, 8, ..., n_ticks).  In-between integer positions get faint
    minor gridlines.  All coordinates are integers."""
    if single_frame:
        frame_intro = "a single frame"
        frame_extra = (
            "  The frame is the first state of an unspecified trial; "
            "no prior or subsequent frames are provided.")
    else:
        frame_intro = "a sequence of 9 frames"
        frame_extra = (
            "  The frames are samples from a single complete trial; "
            "they are NOT consecutive — between frame N and frame N+1 "
            "several game turns have passed during which a human or "
            "model agent took ACTIONS.")
    # The on-demand visual-tool vocabulary is GENERATED from the substrate-tool
    # registry, so a newly-contributed tool is advertised to the VLM
    # automatically (no prompt hand-edit).  See substrate_tools/ +
    # docs/CONTRIBUTING_substrate_tools.md.  Guarded: if the registry can't be
    # imported, fall back to a minimal hint rather than break prompt assembly.
    try:
        import substrate_tools as _ST
        visual_tools_block = _ST.render_vocabulary(n_ticks=n_ticks)
    except Exception:
        visual_tools_block = (
            "ON-DEMAND VISUAL TOOLS — include a top-level `visual_queries` array "
            "in your reply to have the substrate MEASURE what you can't eyeball "
            "(zoom / grid_readout / count / measure / align / palette / "
            "highlight). The substrate measures; you interpret.")
    fmt_args = {
        "n_ticks": n_ticks,
        "n_minus_1": n_ticks - 1,
        "label_stride": label_stride,
        "label_values": ", ".join(
            str(k) for k in range(0, n_ticks + 1, label_stride)
        ),
        "frame_intro": frame_intro,
        "frame_extra": frame_extra,
        "visual_tools_block": visual_tools_block,
    }
    sys_text = SYSTEM_PROMPT_TEMPLATE.format(**fmt_args)
    if single_frame:
        user_text = SINGLE_FRAME_USER_PROMPT_TEMPLATE.format(**fmt_args)
    else:
        user_text = USER_PROMPT_TEMPLATE.format(**fmt_args)
    return sys_text, user_text


# ---------------------------------------------------------------------------
# Image preparation
# ---------------------------------------------------------------------------


def _font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _upscale_pil(img: Image.Image, factor: int) -> Image.Image:
    if factor == 1:
        return img
    return img.resize(
        (img.width * factor, img.height * factor),
        Image.Resampling.NEAREST,
    )


def _draw_alt_dash_line(draw: ImageDraw.ImageDraw,
                         p0: tuple[int, int],
                         p1: tuple[int, int],
                         dash_px: int,
                         color_a: tuple[int, int, int, int],
                         color_b: tuple[int, int, int, int],
                         width: int) -> None:
    """Draw a straight line as alternating-colour dashes (color_a on
    even segments, color_b on odd).  Used for sub-tick gridlines so
    they stay visible against both dark and light backgrounds: dark
    dashes show against light areas, light dashes show against dark
    areas."""
    x0, y0 = p0
    x1, y1 = p1
    dx = x1 - x0
    dy = y1 - y0
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    n = max(2, int(round(length / dash_px)))
    for i in range(n):
        t0 = i / n
        t1 = (i + 1) / n
        sx = x0 + dx * t0
        sy = y0 + dy * t0
        ex = x0 + dx * t1
        ey = y0 + dy * t1
        color = color_a if (i % 2 == 0) else color_b
        draw.line([(sx, sy), (ex, ey)], fill=color, width=width)


def _add_grid_overlay(rgb: Image.Image, n_ticks: int,
                       upscale: int = 1,
                       label_stride: int = DEFAULT_LABEL_STRIDE,
                       line_width_major: int = GRID_LINE_WIDTH,
                       line_width_minor: int = GRID_MINOR_WIDTH,
                       major_alpha: Optional[int] = None,
                       minor_alpha: Optional[int] = None,
                       label_size_override: Optional[int] = None,
                       upscale_resample: int = Image.Resampling.NEAREST,
                       ) -> tuple[Image.Image, int, int]:
    """Take an RGB game frame, upscale it, then surround it with dark
    margin strips bearing tick-axis labels (top = columns, left = rows)
    and overlay faint grey cell-boundary lines on the playfield.

    Returns (composite_image, playfield_y_offset, playfield_x_offset).
    The offsets tell callers where the playfield starts inside the
    composite — needed to draw bbox overlays on top.

    Substrate-agnostic: ticks are a viewing aid, NOT a claim about the
    game's native grid.  Works for any rectangular RGB image."""
    if upscale != 1:
        rgb = rgb.resize(
            (rgb.width * upscale, rgb.height * upscale),
            upscale_resample,
        )
    pw, ph = rgb.size

    # 4-sided margins: labels + tick marks on top/left AND right/bottom
    # so the VLM can count from the nearest edge instead of always from
    # the top-left.  Halves the maximum distance for cumulative
    # counting errors.
    margin_top = max(28, int(round(ph * GRID_MARGIN_FRAC)))
    margin_left = max(28, int(round(pw * GRID_MARGIN_FRAC)))
    margin_right = margin_left
    margin_bottom = margin_top

    total_w = margin_left + pw + margin_right
    total_h = margin_top + ph + margin_bottom
    canvas = Image.new("RGB", (total_w, total_h), GRID_MARGIN_BG)
    canvas.paste(rgb, (margin_left, margin_top))

    # Integer gridlines on the playfield: MAJOR (labeled, every
    # label_stride-th tick) and MINOR (non-labeled integer ticks)
    # are both drawn as 4-pixel-wide TWO-TONE STRIPE PAIRS
    # (2 dark + 2 light pixels).  The width survives image
    # resampling (browser zoom, vision-API downscaling).  MAJOR uses
    # high-alpha near-black + pure-white so the stripe is bold;
    # MINOR uses lower alpha and a softer light tone so it's
    # visibly fainter — the labeled lines stay dominant.
    #
    # Stripe layout (vertical, at integer x):
    #   x+0 : DARK
    #   x+1 : DARK
    #   x+2 : LIGHT
    #   x+3 : LIGHT
    # The anchor pixel (x+0) is the integer position the label
    # points to; the dark side is the primary visual anchor.
    playfield_rgba = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    pdraw = ImageDraw.Draw(playfield_rgba)

    # Resolve stripe colors — start from the module constants and
    # apply the caller's alpha overrides if any.  Color channels
    # (RGB) are always taken from the module constants; only the
    # alpha component is overridable.
    def _with_alpha(rgba: tuple[int, int, int, int],
                     alpha_override: Optional[int]) -> tuple[int, int, int, int]:
        if alpha_override is None:
            return rgba
        return (rgba[0], rgba[1], rgba[2], max(0, min(255, int(alpha_override))))
    _major_dark  = _with_alpha(GRID_LINE_DARK_RGBA,   major_alpha)
    _major_light = _with_alpha(GRID_LINE_LIGHT_RGBA,  major_alpha)
    _minor_dark  = _with_alpha(GRID_MINOR_DARK_RGBA,  minor_alpha)
    _minor_light = _with_alpha(GRID_MINOR_LIGHT_RGBA, minor_alpha)

    def _stripe_pair(anchor: int, dim_max: int, horizontal: bool,
                      dark_rgba: tuple[int, int, int, int],
                      light_rgba: tuple[int, int, int, int],
                      total_width: int = 4) -> None:
        """Draw a two-tone stripe pair: first half DARK pixels,
        second half LIGHT pixels, starting at `anchor`.  Clamps
        within [0, dim_max).  `horizontal=True` means the stripe
        runs left-to-right (a horizontal gridline); False means
        top-to-bottom (a vertical gridline)."""
        half = total_width // 2
        for i in range(total_width):
            offset = anchor + i - (half - 1)  # roughly center stripe on anchor
            if offset < 0 or offset >= dim_max:
                continue
            color = dark_rgba if i < half else light_rgba
            if horizontal:
                pdraw.line([(0, offset), (pw - 1, offset)],
                            fill=color, width=1)
            else:
                pdraw.line([(offset, 0), (offset, ph - 1)],
                            fill=color, width=1)

    for k in range(1, n_ticks):
        is_major = (k % label_stride == 0)
        x = int(round(k * pw / n_ticks))
        y = int(round(k * ph / n_ticks))
        if is_major:
            _stripe_pair(x, pw, horizontal=False,
                          dark_rgba=_major_dark,
                          light_rgba=_major_light,
                          total_width=line_width_major)
            _stripe_pair(y, ph, horizontal=True,
                          dark_rgba=_major_dark,
                          light_rgba=_major_light,
                          total_width=line_width_major)
        else:
            _stripe_pair(x, pw, horizontal=False,
                          dark_rgba=_minor_dark,
                          light_rgba=_minor_light,
                          total_width=line_width_minor)
            _stripe_pair(y, ph, horizontal=True,
                          dark_rgba=_minor_dark,
                          light_rgba=_minor_light,
                          total_width=line_width_minor)
    pf_region = canvas.crop(
        (margin_left, margin_top, margin_left + pw, margin_top + ph),
    ).convert("RGBA")
    pf_region = Image.alpha_composite(pf_region, playfield_rgba)
    canvas.paste(pf_region.convert("RGB"), (margin_left, margin_top))

    # Axis labels on the dark margins, aligned with gridlines.
    # Label "k" sits directly above (or beside) gridline k = float k.0.
    # The right/bottom edge label "n_ticks" is also drawn so the
    # full coord range [0.0, n_ticks.0] is unambiguous.
    draw = ImageDraw.Draw(canvas)
    # Scale the label font so 2-digit labels fit within one tick cell.
    # Cell width (in raw playfield px) = pw / n_ticks * label_stride;
    # font height ~ 0.7 * cell width gives 2-digit width ~ font * 1.1
    # = ~0.77 * cell width — fits comfortably with some breathing room.
    cell_px = max(1, int(round(pw / n_ticks * label_stride)))
    if label_size_override is not None:
        label_size = max(6, int(label_size_override))
    else:
        label_size = max(8, min(
            int(round(min(margin_top, margin_left) * 0.14)),  # margin cap
            int(round(cell_px * 0.70)),                        # cell cap
        ))
    font = _font(label_size)

    # Short tick marks in the margin pointing INTO the playfield direction,
    # visually anchoring each label to its gridline.
    tick_mark_len = max(3, int(round(min(margin_top, margin_left) * 0.18)))

    pf_right = margin_left + pw   # right edge of playfield in canvas px
    pf_bottom = margin_top + ph   # bottom edge of playfield in canvas px

    # Labels are positioned RIGHT NEXT TO the playfield edge (just
    # outside the tick marks) so the VLM associates the label number
    # with the gridline below it directly.
    label_gap_px = 2   # tiny gap between label text and tick mark
    for k in range(0, n_ticks + 1, label_stride):
        text = str(k)
        col_x = margin_left + int(round(k * pw / n_ticks))
        row_y = margin_top + int(round(k * ph / n_ticks))

        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]

        # --- TOP margin: column label + tick mark pointing DOWN ---
        label_x = max(0, min(total_w - tw, col_x - tw // 2))
        # Position label just above the tick mark, against the
        # playfield edge.
        top_label_y = margin_top - tick_mark_len - th - label_gap_px - tb[1]
        draw.text((label_x, top_label_y),
                   text, fill=GRID_LABEL_COLOR, font=font)
        draw.line(
            [(col_x, margin_top - tick_mark_len), (col_x, margin_top - 1)],
            fill=GRID_LABEL_COLOR, width=2,
        )

        # --- LEFT margin: row label + tick mark pointing RIGHT ---
        label_y = max(0, min(total_h - th, row_y - th // 2 - tb[1] // 2))
        # Position label just to the left of the tick mark, against
        # the playfield edge.
        left_label_x = (margin_left - tick_mark_len - tw - label_gap_px)
        draw.text((max(0, left_label_x), label_y),
                   text, fill=GRID_LABEL_COLOR, font=font)
        draw.line(
            [(margin_left - tick_mark_len, row_y), (margin_left - 1, row_y)],
            fill=GRID_LABEL_COLOR, width=2,
        )

        # --- BOTTOM margin: column label + tick mark pointing UP ---
        draw.text(
            (label_x, pf_bottom + tick_mark_len + label_gap_px - tb[1]),
            text, fill=GRID_LABEL_COLOR, font=font,
        )
        draw.line(
            [(col_x, pf_bottom), (col_x, pf_bottom + tick_mark_len)],
            fill=GRID_LABEL_COLOR, width=2,
        )

        # --- RIGHT margin: row label + tick mark pointing LEFT ---
        right_label_x = pf_right + tick_mark_len + label_gap_px
        draw.text((right_label_x, label_y),
                   text, fill=GRID_LABEL_COLOR, font=font)
        draw.line(
            [(pf_right, row_y), (pf_right + tick_mark_len, row_y)],
            fill=GRID_LABEL_COLOR, width=2,
        )

    # Subdivision marks are now drawn directly on the playfield
    # gridlines (see playfield_rgba block above), not in the margins.

    # Origin marker "(0,0)" stamped in the top-left margin cross — makes
    # axis orientation unambiguous even if labels are hard to read.
    origin_font = _font(max(10, int(round(label_size * 0.6))))
    draw.text(
        (4, 2), "(0,0)", fill=GRID_ORIGIN_COLOR, font=origin_font,
    )

    return canvas, margin_top, margin_left


def ticks_to_playfield_px(
    bbox_ticks: list[float] | tuple[float, ...],
    n_ticks: int, playfield_w: int, playfield_h: int,
) -> tuple[int, int, int, int]:
    """Convert [row_top, col_left, row_bottom, col_right] tick coords
    (FLOATS in [0.0, n_ticks], representing the entity's actual visual
    extent — NOT snapped to tick boundaries) into a pixel rect INSIDE
    the playfield (no margin offset).  Caller adds the margin offset
    before drawing.

    The grid is a viewing aid; bbox coords are continuous.  Coordinate
    0.0 maps to playfield-px 0; coordinate n_ticks.0 maps to the last
    pixel of the playfield."""
    if (not isinstance(bbox_ticks, (list, tuple))) or len(bbox_ticks) != 4:
        return (0, 0, 0, 0)
    try:
        r0, c0, r1, c1 = (float(v) for v in bbox_ticks)
    except (TypeError, ValueError):
        return (0, 0, 0, 0)
    r0 = max(0.0, min(float(n_ticks), r0))
    c0 = max(0.0, min(float(n_ticks), c0))
    r1 = max(r0, min(float(n_ticks), r1))
    c1 = max(c0, min(float(n_ticks), c1))
    py = playfield_h / float(n_ticks)
    px = playfield_w / float(n_ticks)
    y0 = max(0, min(playfield_h - 1, int(round(r0 * py))))
    x0 = max(0, min(playfield_w - 1, int(round(c0 * px))))
    y1 = max(0, min(playfield_h - 1, int(round(r1 * py)) - 1))
    x1 = max(0, min(playfield_w - 1, int(round(c1 * px)) - 1))
    # Ensure at least 1px width/height so it renders.
    if y1 < y0:
        y1 = y0
    if x1 < x0:
        x1 = x0
    return (y0, x0, y1, x1)


def _frame_with_label(frame_path: Path, turn: int, cell_px: int,
                       n_ticks: int,
                       label_stride: int = DEFAULT_LABEL_STRIDE,
                       ) -> Image.Image:
    """Load a frame, scale to cell_px square, wrap in a gridded overlay,
    then stamp a 'turn N' header strip on top.  The gridded overlay
    adds dark margins on top + left of the playfield; the header strip
    sits ABOVE the gridded overlay so it doesn't disturb the column-
    label alignment."""
    img = Image.open(frame_path).convert("RGB")
    if img.size != (cell_px, cell_px):
        img = img.resize((cell_px, cell_px), Image.Resampling.NEAREST)

    gridded, mtop, mleft = _add_grid_overlay(img, n_ticks=n_ticks,
                                              upscale=1,
                                              label_stride=label_stride)
    gw, gh = gridded.size

    header_h = max(28, int(round(cell_px * 0.07)))
    canvas = Image.new("RGB", (gw, gh + header_h), GRID_MARGIN_BG)
    canvas.paste(gridded, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    font = _font(max(18, int(round(header_h * 0.62))))
    label = f"turn {turn}"
    tb = draw.textbbox((0, 0), label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    draw.text(
        ((gw - tw) // 2, (header_h - th) // 2 - tb[1] // 2),
        label, fill=(240, 240, 240), font=font,
    )
    return canvas


def build_sequence_grid(turn_to_path: dict[int, Path],
                         n_ticks: int,
                         label_stride: int = DEFAULT_LABEL_STRIDE,
                         ) -> Image.Image:
    """Compose the 9 gridded+labelled frames into a 3x3 layout."""
    cells = [_frame_with_label(p, t, GRID_CELL_PX, n_ticks,
                                label_stride=label_stride)
             for t, p in turn_to_path.items()]
    cell_w, cell_h = cells[0].size
    cols = 3
    rows = 3
    gap = 12
    total_w = cols * cell_w + (cols - 1) * gap
    total_h = rows * cell_h + (rows - 1) * gap
    bg = Image.new("RGB", (total_w, total_h), (30, 30, 35))
    for i, cell in enumerate(cells):
        r, c = divmod(i, cols)
        x = c * (cell_w + gap)
        y = r * (cell_h + gap)
        bg.paste(cell, (x, y))
    return bg


# ---------------------------------------------------------------------------
# Stage artifacts
# ---------------------------------------------------------------------------


PENDING_DIR_NAME = "pending"


def _next_call_n(pending_dir: Path) -> int:
    """Highest existing call number + 1.  Counts both pending and
    consumed prompts so call IDs are monotonic."""
    existing = sorted(pending_dir.glob("call_*_prompt.md"))
    return len(existing) + 1


def _build_prompt_md(call_n: int, image_grid_name: str,
                     image_individual_names: list[str],
                     reply_name: str, n_ticks: int,
                     label_stride: int = DEFAULT_LABEL_STRIDE,
                     single_frame: bool = False) -> str:
    sys_text, user_text = _fmt_prompts(n_ticks,
                                        label_stride=label_stride,
                                        single_frame=single_frame)
    stride_suffix = (f", label_stride={label_stride}"
                     if label_stride != 1 else "")
    single_suffix = ", single-frame mode" if single_frame else ""
    if single_frame:
        image_desc = (
            f"Image attachment: `{image_grid_name}` — the source frame, "
            f"upscaled with a {n_ticks}-tick integer grid overlay"
            f"{stride_suffix}."
        )
    else:
        image_desc = (
            f"Image attachment: `{image_grid_name}` — 3x3 composite of "
            f"9 frames (turns {', '.join(str(t) for t in SAMPLE_TURNS)}), "
            f"each with a {n_ticks}x{n_ticks} integer grid overlay"
            f"{stride_suffix}.\n"
            f"Alternative individual frame PNGs (one per turn) if the "
            f"composite is too dense: "
            f"{', '.join(f'`{n}`' for n in image_individual_names)}."
        )
    # Merge system + user text into a single "Prompt" section.  The
    # operator pastes the whole block as one prompt to the VLM
    # (human:claude has no system/user role distinction — only the API
    # path needs it).
    return (
        f"# Call #{call_n} (sequence probe — perception, "
        f"{n_ticks}-tick grid{stride_suffix}{single_suffix})\n\n"
        f"Model handle: human:claude\n\n"
        f"## Prompt (paste this verbatim)\n\n"
        f"{image_desc}\n\n"
        f"```\n{sys_text}\n\n{user_text}\n```\n\n"
        f"---\n\n"
        f"## Reply instructions\n\n"
        f"Write the VLM's JSON response (top-level object, keys exactly "
        f"as described in the prompt) to:\n"
        f"  `{reply_name}`\n\n"
        f"Plain JSON, no markdown fences, no prose before or after.  "
        f"The probe script consumes the reply, renames it to "
        f"`*_reply.consumed.txt`, and renders an analysis page.\n"
    )


def stage_pending_call(pending_dir: Path,
                        new_call: bool,
                        n_ticks: int,
                        label_stride: int = DEFAULT_LABEL_STRIDE,
                        single_frame_turn: int | None = None,
                        single_frame_upscale: int = 2,
                        ) -> tuple[int, Path, Path,
                                    dict[int, Path], int, int]:
    """Compose the prompt + composite image + individual frame PNGs in
    pending_dir, write STATUS.txt, and return (n, prompt_path,
    reply_path, turn_to_src_path).

    If new_call=False and there's already a pending (un-consumed)
    call_NNN_prompt.md whose reply file does not yet exist, reuse it
    (idempotent across multiple script invocations while you compose
    your Claude session).  Otherwise mint a new call_NNN.
    """
    pending_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    turn_to_src: dict[int, Path] = {}
    for t in SAMPLE_TURNS:
        src = FIXTURE_DIR / f"turn_{t:03d}" / "frame.png"
        if not src.exists():
            raise FileNotFoundError(f"missing fixture frame: {src}")
        turn_to_src[t] = src

    # Reuse the most recent un-replied call unless --new-call.
    existing = sorted(pending_dir.glob("call_*_prompt.md"))
    if existing and not new_call:
        last = existing[-1]
        n = int(last.stem.split("_")[1])
        reply_path = pending_dir / f"call_{n:03d}_reply.txt"
        consumed_path = pending_dir / f"call_{n:03d}_reply.consumed.txt"
        if not consumed_path.exists():
            # Re-use: the prompt/image are already staged; just refresh
            # STATUS.txt so external watchers see current intent.
            _write_status(pending_dir, n, last, reply_path,
                          "WAITING for reply (resumed)")
            # Re-read n_ticks + label_stride from the existing prompt
            # header so the render uses the same grid.
            n_ticks_used = _read_n_ticks_from_prompt(last) or n_ticks
            stride_used = (
                _read_label_stride_from_prompt(last) or label_stride
            )
            return (n, last, reply_path, turn_to_src,
                    n_ticks_used, stride_used)

    n = _next_call_n(pending_dir)
    prompt_path = pending_dir / f"call_{n:03d}_prompt.md"
    grid_path = pending_dir / f"call_{n:03d}_image_grid.png"
    reply_path = pending_dir / f"call_{n:03d}_reply.txt"

    individual_names: list[str] = []

    if single_frame_turn is not None:
        # Single-frame mode: upscale just one turn's frame to high res,
        # with the gridded overlay, and skip the composite.
        src = turn_to_src[single_frame_turn]
        gridded, _, _ = _add_grid_overlay(
            Image.open(src).convert("RGB"),
            n_ticks=n_ticks, upscale=single_frame_upscale,
            label_stride=label_stride,
        )
        gridded.save(grid_path)
        turn_to_src = {single_frame_turn: src}
    else:
        grid = build_sequence_grid(turn_to_src, n_ticks=n_ticks,
                                    label_stride=label_stride)
        grid.save(grid_path)

        for t, src in turn_to_src.items():
            name = f"call_{n:03d}_image_turn_{t:03d}.png"
            out = pending_dir / name
            gridded, _, _ = _add_grid_overlay(
                Image.open(src).convert("RGB"),
                n_ticks=n_ticks, upscale=UPSCALE_PER_FRAME,
                label_stride=label_stride,
            )
            gridded.save(out)
            individual_names.append(name)

    prompt_path.write_text(
        _build_prompt_md(n, grid_path.name, individual_names,
                         reply_path.name, n_ticks=n_ticks,
                         label_stride=label_stride,
                         single_frame=(single_frame_turn is not None)),
        encoding="utf-8",
    )
    _write_status(pending_dir, n, prompt_path, reply_path,
                  "WAITING for reply")
    return n, prompt_path, reply_path, turn_to_src, n_ticks, label_stride


def stage_refinement_call(pending_dir: Path,
                           prev_call_n: int,
                           n_ticks: int,
                           label_stride: int,
                           single_frame_turn: int,
                           single_frame_upscale: int,
                           ) -> tuple[int, Path, Path,
                                       dict[int, Path], int, int]:
    """Stage a refinement round.  Reads the previous call's reply
    (consumed.txt), renders an overlay image showing those bboxes on
    the original frame, and stages a new call with both the original
    gridded image and the overlay image."""
    pending_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load previous reply
    prev_consumed = (pending_dir
                     / f"call_{prev_call_n:03d}_reply.consumed.txt")
    if not prev_consumed.exists():
        raise FileNotFoundError(
            f"No consumed reply at {prev_consumed} — "
            f"previous call must have been replied to.")
    raw = prev_consumed.read_text(encoding="utf-8")
    parsed, _ = _repair_truncated_json(_strip_fences(raw))
    if parsed is None or not parsed.get("entities"):
        raise ValueError(
            f"Could not parse entities from {prev_consumed}")
    prev_entities = parsed.get("entities", [])

    # Source frame for the overlay
    turn1_path = FIXTURE_DIR / f"turn_{single_frame_turn:03d}" / "frame.png"
    if not turn1_path.exists():
        raise FileNotFoundError(turn1_path)
    turn_to_src = {single_frame_turn: turn1_path}

    # Mint new call number
    n = _next_call_n(pending_dir)
    prompt_path = pending_dir / f"call_{n:03d}_prompt.md"
    grid_path = pending_dir / f"call_{n:03d}_image_grid.png"
    overlay_path = pending_dir / f"call_{n:03d}_image_overlay.png"
    reply_path = pending_dir / f"call_{n:03d}_reply.txt"

    # Image (1): original gridded frame at the single-frame upscale
    gridded, _, _ = _add_grid_overlay(
        Image.open(turn1_path).convert("RGB"),
        n_ticks=n_ticks, upscale=single_frame_upscale,
        label_stride=label_stride,
    )
    gridded.save(grid_path)

    # Image (2): gridded frame with prev VLM bboxes overlaid in cyan
    overlay_img = render_turn1_overlay(
        turn1_path, prev_entities, n_ticks=n_ticks,
    )
    overlay_img.save(overlay_path)

    # Prompt: the refinement template
    sys_text, _ = _fmt_prompts(n_ticks, label_stride=label_stride,
                                single_frame=True)
    ref_text = REFINEMENT_PROMPT_TEMPLATE.format(n_ticks=n_ticks)
    image_desc = (
        f"Image 1 (original): `{grid_path.name}` — the source frame, "
        f"gridded with per-pixel labels.\n"
        f"Image 2 (overlay):  `{overlay_path.name}` — your previous "
        f"bboxes drawn as cyan rectangles labeled #N."
    )
    prompt_text = (
        f"# Call #{n} (sequence probe — perception, "
        f"{n_ticks}-tick grid, label_stride={label_stride}, "
        f"REFINEMENT of call_{prev_call_n:03d})\n\n"
        f"Model handle: human:claude\n\n"
        f"## Prompt (paste this verbatim — and include BOTH images)\n\n"
        f"{image_desc}\n\n"
        f"```\n{sys_text}\n\n{ref_text}\n```\n\n"
        f"---\n\n"
        f"## Reply instructions\n\n"
        f"Write the refined JSON response to:\n"
        f"  `{reply_path.name}`\n\n"
        f"Same schema as the first pass.  Plain JSON, no fences.\n"
    )
    prompt_path.write_text(prompt_text, encoding="utf-8")
    _write_status(pending_dir, n, prompt_path, reply_path,
                  f"WAITING for refinement reply (round from call_{prev_call_n:03d})")
    return n, prompt_path, reply_path, turn_to_src, n_ticks, label_stride


def _read_n_ticks_from_prompt(prompt_path: Path) -> int | None:
    """Parse the n_ticks value out of the prompt header line.
    Returns None if it can't find one (older calls without the suffix)."""
    try:
        first = prompt_path.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return None
    # Header looks like: "# Call #1 (sequence probe — perception, 64-tick grid, label_stride=4)"
    import re
    m = re.search(r"(\d+)-tick grid", first)
    return int(m.group(1)) if m else None


def _read_label_stride_from_prompt(prompt_path: Path) -> int | None:
    """Parse the label_stride value out of the prompt header line."""
    try:
        first = prompt_path.read_text(encoding="utf-8").splitlines()[0]
    except Exception:
        return None
    import re
    m = re.search(r"label_stride=(\d+)", first)
    return int(m.group(1)) if m else None


def _write_status(pending_dir: Path, call_n: int,
                   prompt_path: Path, reply_path: Path,
                   state: str) -> None:
    status_path = pending_dir / "STATUS.txt"
    status_path.write_text(
        f"{state} on call_{call_n:03d}\n"
        f"Prompt: {prompt_path}\n"
        f"Reply to write: {reply_path}\n",
        encoding="utf-8",
    )


def poll_for_reply(reply_path: Path,
                    timeout_s: int = 1800,
                    poll_s: float = 2.0) -> str:
    """Block until reply_path contains non-empty content (or timeout).
    Returns the reply body.  Caller is responsible for renaming the
    file to call_NNN_reply.consumed.txt after parsing."""
    print(f"[human-vlm] waiting for reply at {reply_path} "
          f"(timeout {timeout_s}s, poll {poll_s}s)")
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if reply_path.exists():
            try:
                body = reply_path.read_text(encoding="utf-8").strip()
            except Exception:
                body = ""
            if body:
                return body
        time.sleep(poll_s)
    raise TimeoutError(
        f"human VLM did not write {reply_path} within {timeout_s}s"
    )


def _strip_fences(text: str) -> str:
    """Some VLM responses come wrapped in ```json fences.  Strip them
    before json.loads so the parse survives."""
    t = text.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        t = "\n".join(lines)
    return t.strip()


def _repair_truncated_json(text: str) -> tuple[dict | None, str]:
    """Attempt to parse JSON, falling back to closing-brace recovery if
    the response was truncated mid-write (common when a VLM hits its
    max-output-tokens limit while finishing a long overall_notes
    string).  Returns (parsed_dict_or_None, status_message)."""
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, "parsed cleanly"
        return None, f"JSON root is not an object: {type(parsed).__name__}"
    except json.JSONDecodeError as e:
        pass
    # Try a sequence of repair attempts: close one or more unclosed
    # quotes / braces.
    for suffix in (
        "}",            # only missing closing brace
        '"}',           # missing closing quote + brace
        '"\n}',         # missing closing quote + brace, with newline
        '"}}',          # missing inner brace too
        '"]\n}',        # mid-array truncation
    ):
        try:
            parsed = json.loads(text.rstrip() + suffix)
            if isinstance(parsed, dict):
                return parsed, f"repaired by appending {suffix!r}"
        except json.JSONDecodeError:
            continue
    return None, "JSON could not be repaired"


# ---------------------------------------------------------------------------
# Scoring — IoU between VLM bboxes and ground-truth bboxes
# ---------------------------------------------------------------------------


# Logical-pixel resolution of the bp35 frame (from raw_palette_size in
# truth.json).  bbox_pixels in truth.json are in this coord space.
TRUTH_LOGICAL_PX = 64


def _truth_bbox_to_ticks(bbox_px_logical: list[int],
                          n_ticks: int) -> list[float]:
    """Convert a [y0, x0, y1, x1] pixel bbox in the 64-logical-px frame
    (the convention used by truth.json) into a float tick bbox in the
    [0, n_ticks] space.  bbox_pixels are INCLUSIVE on both ends — a
    1-pixel entity has y0 == y1, x0 == x1."""
    y0, x0, y1, x1 = bbox_px_logical
    f = n_ticks / float(TRUTH_LOGICAL_PX)
    return [
        y0 * f,
        x0 * f,
        (y1 + 1) * f,
        (x1 + 1) * f,
    ]


def _iou(a: list[float], b: list[float]) -> float:
    """Intersection-over-union for two bboxes [y0, x0, y1, x1] (any
    coord system, as long as both use the same one)."""
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_truth_for_turn1(n_ticks: int) -> list[dict]:
    """Load truth.json from the fixture's turn_001 dir and convert each
    tile's bbox to tick coords.  Returns a list of
    {label, code, bbox_ticks} records."""
    truth_path = FIXTURE_DIR / "turn_001" / "truth.json"
    if not truth_path.exists():
        return []
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    out: list[dict] = []
    for tile in truth.get("tiles", []):
        bbox_px = tile.get("bbox_pixels")
        if not bbox_px or len(bbox_px) != 4:
            continue
        out.append({
            "label": tile.get("label", "?"),
            "code": tile.get("code", "?"),
            "bbox_ticks": _truth_bbox_to_ticks(bbox_px, n_ticks),
            "centroid_px_logical": tile.get("centroid_pixel"),
            "n_pixels": tile.get("n_pixels"),
        })
    return out


def score_response(vlm_entities: list[dict],
                    truth_entries: list[dict],
                    iou_match_threshold: float = 0.2,
                    ) -> dict:
    """Match each truth entry to its best-IoU VLM entity (greedy),
    classify each VLM entity as matched / hallucinated / unscored
    (no bbox).  Returns a scorecard dict with per-truth-entry IoU and
    aggregate metrics."""
    # Build candidate list of VLM entities that have a bbox.
    candidates: list[tuple[int, dict, list[float]]] = []
    for i, e in enumerate(vlm_entities):
        bbox = e.get("bbox_ticks_turn1")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        try:
            bbox_f = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        candidates.append((i, e, bbox_f))

    # Greedy assignment: for each truth entry, pick the unmatched
    # candidate with the highest IoU.
    matched_indices: set[int] = set()
    per_truth: list[dict] = []
    for t in truth_entries:
        best_iou = 0.0
        best_idx = -1
        for i, e, bbox in candidates:
            if i in matched_indices:
                continue
            iou = _iou(t["bbox_ticks"], bbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0 and best_iou >= iou_match_threshold:
            matched_indices.add(best_idx)
            vlm = vlm_entities[best_idx]
            per_truth.append({
                "truth_label": t["label"],
                "truth_code": t["code"],
                "truth_bbox_ticks": t["bbox_ticks"],
                "matched_vlm_name": vlm.get("name"),
                "matched_vlm_index_1based": best_idx + 1,
                "vlm_bbox_ticks": [float(v) for v in vlm["bbox_ticks_turn1"]],
                "iou": round(best_iou, 3),
            })
        else:
            per_truth.append({
                "truth_label": t["label"],
                "truth_code": t["code"],
                "truth_bbox_ticks": t["bbox_ticks"],
                "matched_vlm_name": None,
                "matched_vlm_index_1based": None,
                "vlm_bbox_ticks": None,
                "iou": round(best_iou, 3),
            })

    hallucinated: list[dict] = []
    for i, e, bbox in candidates:
        if i in matched_indices:
            continue
        hallucinated.append({
            "vlm_name": e.get("name"),
            "vlm_index_1based": i + 1,
            "vlm_bbox_ticks": bbox,
        })

    unscored_vlm = [
        {"vlm_name": e.get("name"), "vlm_index_1based": i + 1,
         "reason": "no bbox (null bbox_ticks_turn1)"}
        for i, e in enumerate(vlm_entities)
        if not (isinstance(e.get("bbox_ticks_turn1"), (list, tuple))
                and len(e.get("bbox_ticks_turn1") or []) == 4)
    ]

    matched_ious = [r["iou"] for r in per_truth
                    if r["matched_vlm_name"] is not None]
    n_truth = len(truth_entries)
    n_matched = len(matched_ious)
    mean_iou_matched = (sum(matched_ious) / len(matched_ious)
                        if matched_ious else 0.0)
    mean_iou_over_truth = (sum(r["iou"] for r in per_truth) / n_truth
                           if n_truth else 0.0)
    recall = n_matched / n_truth if n_truth else 0.0
    n_vlm_with_bbox = len(candidates)
    precision = (n_matched / n_vlm_with_bbox
                 if n_vlm_with_bbox else 0.0)

    return {
        "iou_match_threshold": iou_match_threshold,
        "n_truth_entities": n_truth,
        "n_vlm_entities": len(vlm_entities),
        "n_vlm_with_bbox": n_vlm_with_bbox,
        "n_matched": n_matched,
        "recall": round(recall, 3),
        "precision": round(precision, 3),
        "mean_iou_matched": round(mean_iou_matched, 3),
        "mean_iou_over_truth": round(mean_iou_over_truth, 3),
        "per_truth": per_truth,
        "hallucinated": hallucinated,
        "unscored_vlm": unscored_vlm,
    }


# ---------------------------------------------------------------------------
# Render — bbox overlay on turn 1 + structured analysis tables
# ---------------------------------------------------------------------------


_BBOX_COLOR = (0, 200, 255)


def render_turn1_overlay(turn1_path: Path, entities: list[dict],
                          n_ticks: int,
                          upscale: int = 2,
                          bbox_line_width: int = 4,
                          bbox_label_size: Optional[int] = None,
                          grid_line_width_major: int = GRID_LINE_WIDTH,
                          grid_line_width_minor: int = GRID_MINOR_WIDTH,
                          grid_major_alpha: Optional[int] = None,
                          grid_minor_alpha: Optional[int] = None,
                          grid_label_size: Optional[int] = None,
                          upscale_resample: int = Image.Resampling.NEAREST,
                          large_bbox_frac: float = 0.25,
                          index_labels: bool = False,
                          ) -> Image.Image:
    """Render the turn-1 frame with the SAME tick-grid overlay the VLM
    saw, then draw each entity's bbox_ticks_turn1 on the playfield with
    a numbered label.  Lets the operator visually verify whether the
    VLM placed each bbox in the cells it claimed.

    Defaults reproduce the VLM-input rendering (upscale=2, 4-px stripe
    pair grid lines, 4-px bbox borders, auto-computed label sizes).
    Trace-rendering callers can override these for human-readable
    output: higher upscale (so text is drawn at native target size,
    not interpolated), thinner lines, smaller labels.
    """
    img = Image.open(turn1_path).convert("RGB")
    gridded, margin_top, margin_left = _add_grid_overlay(
        img, n_ticks=n_ticks, upscale=upscale,
        line_width_major=grid_line_width_major,
        line_width_minor=grid_line_width_minor,
        major_alpha=grid_major_alpha,
        minor_alpha=grid_minor_alpha,
        label_size_override=grid_label_size,
        upscale_resample=upscale_resample,
    )
    pf_w = img.width * upscale
    pf_h = img.height * upscale
    big = gridded.convert("RGBA")
    overlay = Image.new("RGBA", big.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(big)
    odraw = ImageDraw.Draw(overlay)
    if bbox_label_size is not None:
        font = _font(max(6, int(bbox_label_size)))
    else:
        # Shrunk to ~70% of the previous size — the overlay is now
        # numbers-only ("#1", "#17") and the cyan digits carry a black
        # stroke, so they remain readable at the smaller scale while
        # taking far less room when labels overlap.
        font = _font(max(11, int(round(margin_top * 0.245))))

    # Filter out very-large "region" bboxes (background, large rooms,
    # HUD strips) so the visualization shows the entities that matter
    # — agent, collectables, threats — without giant outlines piling
    # on top of them.  Threshold: any bbox covering more than ~25%
    # of the playfield area is treated as scenery noise.  An entity
    # is still listed in the entity table; it just isn't drawn on
    # the overlay.
    PLAYFIELD_AREA = n_ticks * n_ticks
    LARGE_BBOX_FRAC = large_bbox_frac
    for i, e in enumerate(entities):
        bbox = e.get("bbox_ticks_turn1")
        if bbox is None:
            continue
        # Skip large region bboxes — they visually swamp smaller
        # entities sharing their edges. Threshold is now a kwarg so
        # callers (like the autonomous-trace renderer) can opt out
        # when the "large" entity is the agent itself.
        br, bc, br1, bc1 = bbox
        if LARGE_BBOX_FRAC < 1.0 and \
           (br1 - br) * (bc1 - bc) > PLAYFIELD_AREA * LARGE_BBOX_FRAC:
            continue
        # Skip explicit empty_cell placeholders.  These come from the
        # REPETITIVE-GRID COMPLETENESS prompt rule and are useful for
        # the symbolic-state downstream (they tell a planner that a
        # cell was checked and is intentionally empty), but they
        # visually clutter the overlay with bboxes around nothing.
        name = (e.get("name") or "").lower()
        if name.startswith("empty_cell"):
            continue
        py0, px0, py1, px1 = ticks_to_playfield_px(
            bbox, n_ticks, pf_w, pf_h,
        )
        if px1 <= px0 or py1 <= py0:
            continue
        # Shift into composite coords by adding margin offsets.
        x0 = margin_left + px0
        y0 = margin_top + py0
        x1 = margin_left + px1
        y1 = margin_top + py1
        draw.rectangle([x0, y0, x1, y1], outline=_BBOX_COLOR,
                        width=max(1, int(bbox_line_width)))
        # Label: with index_labels the on-image label is the positional
        # `#N` ONLY (matching the trace's entity-table `#` column) — long
        # entity NAMES overlap into an unreadable pile, so names/details
        # live in the table, not on the image.  Otherwise use the entity's
        # `name` field when present (so callers can pass persistent tids
        # like '#39'), falling back to positional `#N`.
        id_text = f"#{i+1}" if index_labels else (e.get("name") or f"#{i+1}")
        id_bbox = odraw.textbbox((0, 0), id_text, font=font)
        id_w = id_bbox[2] - id_bbox[0]
        id_h = id_bbox[3] - id_bbox[1]
        # Tighter padding — the text-stroke outline below carries
        # contrast against the underlying frame, so we don't need a
        # big fill box to make the label legible.
        pad = 3
        gw = id_w + 2 * pad
        gh = id_h + 2 * pad
        ax = max(0, min(x0, big.size[0] - gw))
        # Label sits ABOVE the bbox.  Subtract 1 from y0 so the label
        # background's bottom row doesn't paint over the bbox's top
        # edge row.
        ay = y0 - gh - 1
        if ay < 0:
            # No room above — drop the label BELOW the bbox instead,
            # leaving a 1-pixel gap so the bbox bottom edge stays
            # visible.
            ay = min(big.size[1] - gh, y1 + 2)
        ay = max(0, ay)
        # SEMI-TRANSPARENT label background (alpha 40 ≈ 16%) — three
        # overlapping labels still composite to ~40% opacity, so the
        # underlying frame stays readable.  The TEXT itself carries a
        # black stroke (outline) of stroke_width=2 around the cyan
        # numerals so the digits remain crisply legible even when
        # multiple label boxes overlap.  Replaces the original alpha-80
        # fill which made overlapping labels a near-opaque smear.
        odraw.rectangle([ax, ay, ax + gw, ay + gh], fill=(0, 0, 0, 40))
        odraw.text((ax + pad, ay + pad - id_bbox[1]),
                   id_text, fill=_BBOX_COLOR, font=font,
                   stroke_width=2, stroke_fill=(0, 0, 0, 220))
    return Image.alpha_composite(big, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

CSS = """\
body { font: 14px/1.5 -apple-system, sans-serif; margin: 20px;
       background: #f5f5f5; color: #222; max-width: 1400px; }
h1 { font-size: 24px; margin-bottom: 4px; }
h2 { font-size: 18px; margin-top: 28px; padding: 6px 10px;
     background: #2c3e50; color: white; border-radius: 4px; }
h3 { font-size: 15px; margin-top: 20px; color: #555; }
.subtitle { color: #666; margin-bottom: 18px; }
.gen-banner { background: #fff8e1; border: 1px solid #fbc02d;
              border-radius: 4px; padding: 8px 12px;
              font-size: 13px; margin: 10px 0; }
.gen-banner code { font-family: 'Courier New', monospace;
                   background: rgba(0,0,0,0.05); padding: 1px 4px;
                   border-radius: 2px; }
.gen-timeago { font-weight: bold; color: #e65100; }
.banner-stale { background: #ffebee; border-color: #c62828; color: #c62828; }
.panel { background: white; padding: 14px 18px; border-radius: 6px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin: 14px 0; }
.panel img { max-width: 100%; height: auto; display: block;
             image-rendering: pixelated; margin: 6px 0;
             border: 1px solid #ddd; border-radius: 3px; }
.paste-instructions { background: #e3f2fd; border: 1px solid #1976d2;
                       border-radius: 4px; padding: 10px 14px;
                       font-size: 13px; margin: 12px 0; }
.paste-instructions code { background: rgba(0,0,0,0.05); padding: 1px 4px;
                            border-radius: 2px;
                            font-family: 'Courier New', monospace; }
.paste-instructions ol { margin: 6px 0 6px 20px; padding: 0; }
.paste-instructions li { margin: 4px 0; }
details { margin: 8px 0; }
details > summary { cursor: pointer; font-weight: bold; color: #444; }
details pre { background: #fafafa; border: 1px solid #ddd;
              border-radius: 3px; padding: 8px; font-size: 11px;
              overflow-x: auto; max-height: 480px; }
table.analysis { width: 100%; border-collapse: collapse;
                  font-size: 13px; margin: 8px 0; }
table.analysis th, table.analysis td {
    border: 1px solid #ddd; padding: 4px 8px; text-align: left;
    vertical-align: top;
}
table.analysis th { background: #efefef; font-weight: bold;
                     font-size: 11px; text-transform: uppercase;
                     letter-spacing: 0.5px; }
table.analysis td.num { width: 28px; text-align: right;
                         font-family: monospace; color: #666; }
table.analysis td.bbox { font-family: monospace; font-size: 11px;
                          color: #555; }
table.analysis td.conf-high  { background: #d4edda; }
table.analysis td.conf-medium{ background: #fff3cd; }
table.analysis td.conf-low   { background: #f8d7da; }
.gap-row { background: #f0f0f0; height: 4px; }
.gameclaim { background: #f9f9f9; border-left: 4px solid #2c3e50;
              padding: 8px 12px; margin: 6px 0; }
.gameclaim .guess { font-size: 16px; font-weight: bold; color: #222; }
.gameclaim .evidence { color: #555; font-size: 13px; margin-top: 4px; }
.gameclaim .conf { display: inline-block; padding: 1px 6px;
                    border-radius: 3px; font-size: 11px;
                    text-transform: uppercase; margin-left: 8px; }
.gameclaim .conf.high  { background: #d4edda; color: #155724; }
.gameclaim .conf.medium{ background: #fff3cd; color: #856404; }
.gameclaim .conf.low   { background: #f8d7da; color: #721c24; }
.scorebanner { font-size: 14px; background: #e8f0fe;
                border: 1px solid #4a6cb3; border-radius: 4px;
                padding: 8px 12px; margin-bottom: 12px; }
"""

TIMEAGO_SCRIPT = """\
<script>
function _fmt_ago(s){if(s<5)return'just now';if(s<60)return s+' second'+(s===1?'':'s')+' ago';if(s<3600){const m=Math.floor(s/60);return m+' minute'+(m===1?'':'s')+' ago';}if(s<86400){const h=Math.floor(s/3600);return h+' hour'+(h===1?'':'s')+' ago';}const d=Math.floor(s/86400);return d+' day'+(d===1?'':'s')+' ago';}
function _tick(){document.querySelectorAll('.timeago').forEach(el=>{const gen=parseInt(el.dataset.gen);const now=Math.floor(Date.now()/1000);el.textContent=_fmt_ago(now-gen);});}
window.addEventListener('DOMContentLoaded',()=>{_tick();setInterval(_tick,1000);});
</script>
"""


def _html_escape(s) -> str:
    if s is None:
        return ""
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _conf_cls(c) -> str:
    return f"conf-{(c or 'low').lower()}"


def _render_entities_table(entities: list[dict]) -> str:
    if not entities:
        return "<p><em>No entities returned.</em></p>"
    parts = ['<table class="analysis"><thead><tr>',
             '<th>#</th><th>name</th><th>first seen</th><th>turn40</th>',
             '<th>ticks @ turn 1</th><th>appearance</th>',
             '<th>behavior</th><th>role hypothesis</th><th>conf</th>',
             '</tr></thead><tbody>']
    for i, e in enumerate(entities):
        bbox = e.get("bbox_ticks_turn1") or e.get("bbox_pct_turn1")
        bbox_str = str(bbox) if bbox else '<em>(not at turn 1)</em>'
        present = "✓" if e.get("still_present_turn40") else "✗"
        conf = (e.get("confidence") or "").lower()
        parts.append(
            f'<tr>'
            f'<td class="num">{i+1}</td>'
            f'<td><b>{_html_escape(e.get("name"))}</b></td>'
            f'<td>{e.get("first_seen_turn", "?")}</td>'
            f'<td>{present}</td>'
            f'<td class="bbox">{bbox_str}</td>'
            f'<td>{_html_escape(e.get("appearance"))}</td>'
            f'<td>{_html_escape(e.get("behavior_observed"))}</td>'
            f'<td>{_html_escape(e.get("role_hypothesis"))}</td>'
            f'<td class="{_conf_cls(conf)}">{_html_escape(conf)}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')
    return "".join(parts)


def _render_relationships_table(rels: list[dict]) -> str:
    if not rels:
        return "<p><em>No relationships returned.</em></p>"
    parts = ['<table class="analysis"><thead><tr>',
             '<th>from</th><th>relation</th><th>to</th>',
             '<th>evidence</th><th>conf</th>',
             '</tr></thead><tbody>']
    for r in rels:
        conf = (r.get("confidence") or "").lower()
        parts.append(
            f'<tr>'
            f'<td><b>{_html_escape(r.get("from"))}</b></td>'
            f'<td>{_html_escape(r.get("relation"))}</td>'
            f'<td><b>{_html_escape(r.get("to"))}</b></td>'
            f'<td>{_html_escape(r.get("evidence"))}</td>'
            f'<td class="{_conf_cls(conf)}">{_html_escape(conf)}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')
    return "".join(parts)


def _render_transition_table(transitions: list[dict]) -> str:
    if not transitions:
        return "<p><em>No frame-to-frame transitions returned.</em></p>"
    parts = ['<table class="analysis"><thead><tr>',
             '<th>from</th><th>→</th><th>to</th><th>what changed</th>',
             '</tr></thead><tbody>']
    for t in transitions:
        parts.append(
            f'<tr>'
            f'<td>turn {t.get("from_turn", "?")}</td>'
            f'<td>→</td>'
            f'<td>turn {t.get("to_turn", "?")}</td>'
            f'<td>{_html_escape(t.get("what_changed"))}</td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')
    return "".join(parts)


def _render_game_claim(label: str, claim) -> str:
    if not isinstance(claim, dict):
        return f"<p><em>No {label}.</em></p>"
    conf = (claim.get("confidence") or "low").lower()
    return (
        f'<div class="gameclaim">'
        f'<span class="guess">{_html_escape(claim.get("guess"))}</span>'
        f'<span class="conf {conf}">{_html_escape(conf)}</span>'
        f'<div class="evidence">{_html_escape(claim.get("evidence"))}</div>'
        f'</div>'
    )


def _render_scorecard(score: dict, n_ticks: int) -> str:
    if not score or not score.get("per_truth"):
        return ""
    iou_threshold = score["iou_match_threshold"]
    parts = [
        '<h2>Bbox accuracy vs fixture truth.json '
        f'(IoU threshold {iou_threshold}, {n_ticks}-tick grid)</h2>',
        '<div class="panel">',
        '<div class="scorebanner">',
        f'<b>Mean IoU (matched):</b> {score["mean_iou_matched"]:.3f} '
        f'&middot; <b>Mean IoU (over truth):</b> '
        f'{score["mean_iou_over_truth"]:.3f} &middot; '
        f'<b>Recall:</b> {score["recall"]:.2f} '
        f'({score["n_matched"]}/{score["n_truth_entities"]}) &middot; '
        f'<b>Precision:</b> {score["precision"]:.2f} '
        f'({score["n_matched"]}/{score["n_vlm_with_bbox"]})',
        '</div>',
        '<table class="analysis"><thead><tr>',
        '<th>truth label</th><th>code</th><th>truth bbox_ticks</th>',
        '<th>vlm match</th><th>vlm bbox_ticks</th><th>IoU</th>',
        '</tr></thead><tbody>',
    ]
    for r in score["per_truth"]:
        iou = r["iou"]
        iou_cls = ("conf-high" if iou >= 0.75 else
                   "conf-medium" if iou >= 0.4 else
                   "conf-low")
        truth_bbox = ", ".join(f"{v:.2f}" for v in r["truth_bbox_ticks"])
        vlm_bbox_str = (
            ", ".join(f"{v:.2f}" for v in r["vlm_bbox_ticks"])
            if r["vlm_bbox_ticks"] else
            '<em class="error">no match</em>'
        )
        match_str = (
            f'#{r["matched_vlm_index_1based"]} '
            f'{_html_escape(r["matched_vlm_name"])}'
            if r["matched_vlm_name"] else
            '<em class="error">unmatched</em>'
        )
        parts.append(
            f'<tr>'
            f'<td><b>{_html_escape(r["truth_label"])}</b></td>'
            f'<td>{_html_escape(r["truth_code"])}</td>'
            f'<td class="bbox">[{truth_bbox}]</td>'
            f'<td>{match_str}</td>'
            f'<td class="bbox">{vlm_bbox_str if isinstance(vlm_bbox_str, str) and vlm_bbox_str.startswith("<") else f"[{vlm_bbox_str}]"}</td>'
            f'<td class="{iou_cls}"><b>{iou:.3f}</b></td>'
            f'</tr>'
        )
    parts.append('</tbody></table>')
    if score.get("hallucinated"):
        parts.append('<h3>VLM entities with no truth match (hallucinated '
                     'or unscored real entity)</h3>')
        parts.append('<ul>')
        for h in score["hallucinated"]:
            bb = ", ".join(f"{v:.2f}" for v in h["vlm_bbox_ticks"])
            parts.append(
                f'<li>#{h["vlm_index_1based"]} '
                f'<b>{_html_escape(h["vlm_name"])}</b> '
                f'<code class="bbox">[{bb}]</code></li>'
            )
        parts.append('</ul>')
    if score.get("unscored_vlm"):
        parts.append('<details><summary>VLM entities not scored '
                     '(no bbox)</summary><ul>')
        for u in score["unscored_vlm"]:
            parts.append(
                f'<li>#{u["vlm_index_1based"]} '
                f'<b>{_html_escape(u["vlm_name"])}</b> — '
                f'{_html_escape(u["reason"])}</li>'
            )
        parts.append('</ul></details>')
    parts.append('</div>')
    return "".join(parts)


def render_html(turn_to_path: dict[int, Path],
                response: dict, out_path: Path,
                has_response: bool,
                call_n: int | None = None,
                prompt_path: Path | None = None,
                reply_path: Path | None = None,
                n_ticks: int = DEFAULT_N_TICKS,
                score: dict | None = None) -> None:
    gen_dt = datetime.now(timezone.utc)
    gen_iso = gen_dt.isoformat(timespec="seconds")
    gen_unix = int(gen_dt.timestamp())

    overlay_block = ""
    if has_response and response.get("entities"):
        overlay_img = render_turn1_overlay(
            turn_to_path[1], response["entities"], n_ticks=n_ticks,
        )
        overlay_path = OUTPUT_DIR / f"turn1_overlay_t{n_ticks}.png"
        overlay_img.save(overlay_path)
        overlay_block = (
            f'<h2>Entities — bboxes overlaid on turn 1 '
            f'({n_ticks}-tick grid)</h2>'
            f'<div class="panel">'
            f'<img src="{overlay_path.name}" alt="turn 1 with VLM bboxes">'
            f'</div>'
        )

    paste_block = ""
    if not has_response and call_n is not None and reply_path is not None:
        paste_block = (
            '<div class="paste-instructions">'
            f'<b>Probe is waiting (call_{call_n:03d}).</b>  '
            'The script is polling for the reply file.  To respond:'
            '<ol>'
            f'<li>Read <code>{prompt_path.as_posix() if prompt_path else "?"}</code> '
            '(system + user prompts inline).</li>'
            f'<li>Open <code>pending/call_{call_n:03d}_image_grid.png</code> '
            '(or the 9 individual <code>call_{call_n:03d}_image_turn_NNN.png</code> '
            'files in pending/).</li>'
            '<li>Generate the JSON response in your Claude session.</li>'
            f'<li>Write the JSON to <code>{reply_path.as_posix()}</code>.</li>'
            '<li>The polling probe consumes it and re-renders this page.</li>'
            '</ol>'
            '</div>'
        )

    banner_cls = "gen-banner banner-stale" if not has_response else "gen-banner"
    banner_msg = ("<b>Awaiting paste.</b> "
                  "Reload after you write response.json."
                  if not has_response
                  else "Rendered from response.json.")

    parts: list[str] = []
    parts.append(
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        '<title>bp35 lc=0 — VLM sequence probe</title>'
        f'<style>{CSS}</style></head><body>'
    )
    parts.append(TIMEAGO_SCRIPT)
    parts.append('<h1>bp35 lc=0 — VLM sequence probe (manual paste)</h1>')
    parts.append(
        f'<div class="subtitle">Single-game probe: '
        f'9 frames sampled across 40 turns of one bp35 lc=0 trial; '
        f'turns shown: {", ".join(str(t) for t in SAMPLE_TURNS)}.</div>'
    )
    parts.append(
        f'<div class="{banner_cls}">'
        f'Generated <span class="timeago" data-gen="{gen_unix}">just now</span>'
        f' &middot; <code>{gen_iso}Z</code> &middot; {banner_msg}'
        f'</div>'
    )
    parts.append(paste_block)

    parts.append('<h2>Source frames (the upload composite)</h2>')
    grid_rel = (f"pending/call_{call_n:03d}_image_grid.png"
                if call_n is not None else "")
    parts.append(
        '<div class="panel">'
        f'<img src="{grid_rel}" alt="probe source grid">'
        '</div>'
    )

    parts.append('<h2>Prompt sent to the operator</h2>')
    # Read the actual on-disk prompt file so the HTML always reflects
    # exactly what was staged (single-frame vs multi-frame, label_stride,
    # etc.) instead of re-deriving from templates with possibly-stale
    # default args.
    prompt_text = ""
    if call_n is not None:
        pp = OUTPUT_DIR / PENDING_DIR_NAME / f"call_{call_n:03d}_prompt.md"
        if pp.exists():
            try:
                prompt_text = pp.read_text(encoding="utf-8")
            except Exception as e:
                prompt_text = f"(failed to read {pp}: {e})"
    parts.append(
        '<div class="panel">'
        '<details open><summary>prompt (full markdown)</summary>'
        f'<pre>{_html_escape(prompt_text)}</pre></details>'
        '</div>'
    )

    if has_response:
        parts.append(overlay_block)

        if score:
            parts.append(_render_scorecard(score, n_ticks))

        parts.append('<h2>Game-level claims</h2>')
        parts.append('<div class="panel">')
        parts.append('<h3>Game type</h3>')
        parts.append(_render_game_claim("game_type", response.get("game_type")))
        parts.append('<h3>Game purpose</h3>')
        parts.append(_render_game_claim("game_purpose",
                                         response.get("game_purpose")))
        parts.append('</div>')

        parts.append('<h2>Entity inventory</h2>')
        parts.append('<div class="panel">')
        parts.append(_render_entities_table(response.get("entities", [])))
        parts.append('</div>')

        parts.append('<h2>Relationships</h2>')
        parts.append('<div class="panel">')
        parts.append(_render_relationships_table(
            response.get("relationships", [])
        ))
        parts.append('</div>')

        parts.append('<h2>Frame-to-frame summary</h2>')
        parts.append('<div class="panel">')
        parts.append(_render_transition_table(
            response.get("frame_to_frame_summary", [])
        ))
        parts.append('</div>')

        notes = response.get("overall_notes") or ""
        if notes:
            parts.append('<h2>Overall notes</h2>')
            parts.append(
                f'<div class="panel"><p>{_html_escape(notes)}</p></div>'
            )

        parts.append('<h2>Raw response JSON</h2>')
        parts.append(
            '<div class="panel">'
            '<details><summary>show JSON</summary>'
            f'<pre>{_html_escape(json.dumps(response, indent=2))}</pre>'
            '</details>'
            '</div>'
        )

    parts.append('</body></html>')
    out_path.write_text("".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _consume_reply(pending_dir: Path, call_n: int,
                    reply_path: Path, body: str) -> dict:
    """Parse the reply body as JSON, mirror it to OUTPUT_DIR/response.json,
    rename the reply file to *_reply.consumed.txt, and update STATUS.txt.
    Returns the parsed dict (or {'error': ..., 'raw': ...} if parse failed)."""
    repaired, status = _repair_truncated_json(_strip_fences(body))
    if repaired is None:
        parsed = {"error": status, "raw": body}
    else:
        parsed = repaired
        if status != "parsed cleanly":
            print(f"[json-repair] {status}")

    (OUTPUT_DIR / "response.json").write_text(
        json.dumps(parsed, indent=2), encoding="utf-8",
    )
    try:
        reply_path.rename(
            pending_dir / f"call_{call_n:03d}_reply.consumed.txt"
        )
    except Exception:
        pass
    (pending_dir / "STATUS.txt").write_text(
        f"RECEIVED call_{call_n:03d} reply ({len(body)} chars)\n",
        encoding="utf-8",
    )
    return parsed


def _is_real_response(d) -> bool:
    if not isinstance(d, dict):
        return False
    return bool(
        d.get("entities") or d.get("relationships")
        or d.get("game_type") or d.get("game_purpose")
        or d.get("overall_notes") or d.get("error")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-call", action="store_true",
                        help="Force a fresh call_NNN instead of "
                             "resuming the most recent un-consumed one")
    parser.add_argument("--render-only", action="store_true",
                        help="Don't stage or poll — just re-render the "
                             "page from OUTPUT_DIR/response.json")
    parser.add_argument("--timeout-s", type=int, default=1800,
                        help="poll timeout in seconds (default 1800)")
    parser.add_argument("--poll-s", type=float, default=2.0,
                        help="poll interval in seconds (default 2.0)")
    parser.add_argument("--ticks", type=int, default=DEFAULT_N_TICKS,
                        help=f"tick-grid divisions per axis "
                             f"(default {DEFAULT_N_TICKS}; substrate-"
                             f"agnostic viewing aid, not a game grid)")
    parser.add_argument("--label-stride", type=int,
                        default=DEFAULT_LABEL_STRIDE,
                        help=f"only every Nth tick gets a major label "
                             f"and prominent gridline; the other "
                             f"integer positions get faint minor "
                             f"gridlines.  Default {DEFAULT_LABEL_STRIDE} "
                             f"(labels at 0, 4, 8, ..., n_ticks).")
    parser.add_argument("--single-frame", type=int, default=None,
                        metavar="TURN",
                        help="if set, stage only the given TURN's "
                             "frame (not the 9-frame composite), "
                             "upscaled to a larger size.  Useful for "
                             "isolating bbox-accuracy tests at higher "
                             "per-frame resolution.")
    parser.add_argument("--refine-from-call", type=int, default=None,
                        metavar="N",
                        help="stage a REFINEMENT round using call_N's "
                             "previously-emitted bboxes.  The new call "
                             "shows the VLM (a) the original gridded "
                             "frame and (b) an overlay with the prior "
                             "bboxes drawn in cyan, and asks for "
                             "corrected bbox coords.  Refinement uses "
                             "the same protocol as initial calls.")
    parser.add_argument("--single-frame-upscale", type=int, default=2,
                        help="upscale factor for --single-frame mode "
                             "(default 2 = 512x512 raw -> 1024 playfield "
                             "+ margins ~= 1270x1270 total, comfortably "
                             "below the vision-API 1568 resample "
                             "threshold).")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pending_dir = OUTPUT_DIR / PENDING_DIR_NAME
    index_path = OUTPUT_DIR / "index.html"

    # --render-only: just re-render from the saved response.json.
    if args.render_only:
        resp_path = OUTPUT_DIR / "response.json"
        if not resp_path.exists():
            print(f"No {resp_path} to render from.  Run without "
                  f"--render-only to stage a call.")
            return
        response = json.loads(resp_path.read_text(encoding="utf-8"))
        # Build a turn->path map for the overlay render.
        turn_to_path = {
            t: FIXTURE_DIR / f"turn_{t:03d}" / "frame.png"
            for t in SAMPLE_TURNS
        }
        # Find the last consumed call so the page can link the right
        # grid image, and pull its n_ticks out of the prompt header.
        consumed = sorted(pending_dir.glob("call_*_reply.consumed.txt"))
        call_n = (int(consumed[-1].stem.split("_")[1])
                  if consumed else None)
        n_ticks_resolved = args.ticks
        if call_n is not None:
            pp = pending_dir / f"call_{call_n:03d}_prompt.md"
            n_ticks_resolved = _read_n_ticks_from_prompt(pp) or args.ticks
        truth = load_truth_for_turn1(n_ticks_resolved)
        score = (score_response(response.get("entities", []), truth)
                 if truth else None)
        render_html(turn_to_path, response, index_path,
                    has_response=_is_real_response(response),
                    call_n=call_n, n_ticks=n_ticks_resolved,
                    score=score)
        if score:
            print(f"Score: matched={score['n_matched']}/"
                  f"{score['n_truth_entities']}  "
                  f"mean_IoU={score['mean_iou_matched']:.3f} "
                  f"(over_truth={score['mean_iou_over_truth']:.3f})")
        print(f"Re-rendered: file:///{index_path.as_posix()} "
              f"(n_ticks={n_ticks_resolved})")
        return

    # Refinement mode: skip initial staging; instead stage a follow-up
    # call that shows the VLM the previous call's bboxes overlaid for
    # correction.
    if args.refine_from_call is not None:
        prev_n = args.refine_from_call
        (n, prompt_path, reply_path, turn_to_path,
         n_ticks_used, stride_used) = stage_refinement_call(
            pending_dir, prev_call_n=prev_n,
            n_ticks=args.ticks, label_stride=args.label_stride,
            single_frame_turn=args.single_frame or 1,
            single_frame_upscale=args.single_frame_upscale,
        )
    else:
        # Normal flow: stage a pending call, write a placeholder page,
        # then block on the reply.
        (n, prompt_path, reply_path, turn_to_path,
         n_ticks_used, stride_used) = stage_pending_call(
            pending_dir, new_call=args.new_call,
            n_ticks=args.ticks, label_stride=args.label_stride,
            single_frame_turn=args.single_frame,
            single_frame_upscale=args.single_frame_upscale,
        )

    # Render the "waiting" page immediately so you can open it while
    # the script polls.
    render_html(turn_to_path, {}, index_path, has_response=False,
                call_n=n, prompt_path=prompt_path, reply_path=reply_path,
                n_ticks=n_ticks_used)
    print(f"Sample turns: {SAMPLE_TURNS}")
    print(f"n_ticks:      {n_ticks_used} (per axis)")
    print(f"label_stride: {stride_used} (labels every Nth tick)")
    print(f"Pending dir:  {pending_dir.as_posix()}")
    print(f"Prompt:       {prompt_path.as_posix()}")
    print(f"Image grid:   pending/call_{n:03d}_image_grid.png")
    print(f"Reply target: {reply_path.as_posix()}")
    print(f"Page:         file:///{index_path.as_posix()}")
    print(f"Served at:    http://localhost:8780/bp35_sequence_probe/")
    print()

    body = poll_for_reply(reply_path, timeout_s=args.timeout_s,
                          poll_s=args.poll_s)
    response = _consume_reply(pending_dir, n, reply_path, body)
    if "error" in response and "entities" not in response:
        print(f"[parse] {response['error']}")
    else:
        print(f"[ok] reply parsed: {len(response.get('entities', []))} "
              f"entities, {len(response.get('relationships', []))} "
              f"relationships")

    truth = load_truth_for_turn1(n_ticks_used)
    score = (score_response(response.get("entities", []), truth)
             if truth else None)
    if score:
        print(f"[score] matched={score['n_matched']}/"
              f"{score['n_truth_entities']}  "
              f"mean_IoU={score['mean_iou_matched']:.3f} "
              f"(over_truth={score['mean_iou_over_truth']:.3f})  "
              f"hallucinated={len(score['hallucinated'])}")

    render_html(turn_to_path, response, index_path,
                has_response=_is_real_response(response),
                call_n=n, prompt_path=prompt_path, reply_path=reply_path,
                n_ticks=n_ticks_used, score=score)
    print(f"Rendered: file:///{index_path.as_posix()}")


if __name__ == "__main__":
    main()
