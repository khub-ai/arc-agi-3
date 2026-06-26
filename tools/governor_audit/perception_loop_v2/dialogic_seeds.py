"""Dialogic knowledge seeds — human-authored procedures, ingested
into the Subroutine KB through the honest dialogic path.

NORMALLY the system earns every subroutine from its own play.  This
module is the rare, deliberate exception: a human handed the system a
complex procedure in dialogue.  Such knowledge is kept here as the
reproducible *source of truth* (the live ``.tmp/subroutine_kb.json`` is
local scratch and not under version control), and is registered via
``subroutine_kb.ingest_dialogic_subroutine`` so that every entry is:

  * tagged ``source="dialogic"`` (auditable; never mistaken for earned);
  * given 0/0 attempt counters (the system has not run it);
  * given a human-input PRIOR credence, not a self-confirmed value;
  * stored GENERALISED in ``relational_steps`` — roles and relations
    only, so it transfers across piece colour, piece count, and (when
    the roles map) other games.

The originating concrete instance is kept ONLY in the notes / as the
example, never as the procedure itself.

Run ``python -m ...dialogic_seeds`` (or call ``apply_dialogic_seeds``)
to (re)apply the seeds to a KB file; it is idempotent on name.
"""
from __future__ import annotations

from pathlib import Path

from subroutine_kb import (
    DEFAULT_SUBROUTINE_KB_PATH,
    Subroutine,
    ingest_dialogic_subroutine,
)


# ---------------------------------------------------------------------------
# The seeds.  Each is a generalised procedure phrased in ROLES + RELATIONS.
#
# Role vocabulary used below (game-agnostic):
#   agent        — the entity the player directly controls (here: the arm).
#   piece        — a collectable/movable target the goal cares about.
#   target_order — the required order in which pieces must be collected
#                  (the win's ordered_completion sequence).
#   boundary     — an immovable wall/edge of the playfield.
#   lane         — a single row or column.
# ---------------------------------------------------------------------------

DIALOGIC_SUBROUTINES: list[dict] = [
    {
        "name": "free a wall-jammed cluster by relocating each piece to its own clear lane",
        "problem_solved": (
            "A set of pieces has been pushed flush against a boundary "
            "wall and packed together so that NO single normal action "
            "separates them or moves them off the wall usefully — a "
            "dead-end Layer C recognises as `undo_only` (only Undo "
            "recovers it).  The pieces must still be collected "
            "individually in a required target order, but in the jammed "
            "configuration that ordered collection is impossible: acting "
            "on one piece drags or blocks the others.  Goal: restore a "
            "configuration in which each piece can be addressed alone."
        ),
        "description": (
            "Recovery for a wall-jammed cluster, independent of piece "
            "colour and piece count.  Phases:\n"
            "  (0) DECIDE WHETHER TO RECOVER AT ALL (check FIRST — skipping "
            "this is the costliest mistake).  This relocation is warranted "
            "ONLY when the cluster is FLUSH against a boundary on the side a "
            "piece must be swept toward, i.e. `clearance` to that boundary "
            "is 0, so no separating sweep has room.  Read the clearance "
            "relations BEFORE moving anything: if the cluster ALREADY has "
            "clearance on BOTH sides of the separation axis (it is sitting "
            "in open mid-field), DO NOT relocate — go straight to phase (4) "
            "and separate the pieces WHERE THEY ARE.  Relocating an "
            "already-clear cluster is a frequent, expensive error: it tends "
            "to push the pieces toward a wall and DESTROY the clearance you "
            "already had, manufacturing the very jam this technique exists "
            "to escape.\n"
            "  (1) ATTACH — perform the agent's grab/skewer action so the "
            "jammed pieces couple to the agent and co-move with it "
            "(relation: co_displacement(agent, pieces)).  Attach as many "
            "of the jammed pieces as the agent can carry together.\n"
            "  (2) CARRY TO USABLE OPEN SPACE — \"open space\" is defined "
            "RELATIVE to the separation you are about to perform, NOT as a "
            "vague high-clearance-on-all-sides blob.  The destination must "
            "leave clearance on the side each piece will be swept TOWARD in "
            "phase (4), AND room on the offset side for the arm body.  "
            "LOAD-BEARING POST-CONDITION: never end the carry with the "
            "cluster FLUSH against a boundary — `clearance(cluster, "
            "sweep-target side)` must be > 0 afterward; ending flush "
            "re-creates the jam you are escaping (this is exactly the "
            "regression to avoid: 'relocated to open space' is WRONG if it "
            "left the pieces pinned against a wall).  Move the agent (pieces "
            "co-moving) away from the boundary and AROUND any non-target "
            "pieces; do not collide the carried cluster with pieces you are "
            "not carrying.\n"
            "  (3) RELEASE — move the agent to the extreme of its "
            "detach axis (e.g. all the way to one side) until the pieces "
            "uncouple (co_displacement(agent, pieces) ceases).\n"
            "  (4) SEPARATE ONTO CLEAR LANES — lateral push/swipe each "
            "piece into a lane (row or column) that contains no other "
            "piece, so AFTER the move that piece is alone in its lane "
            "(same_row/same_col count == 1).  LOAD-BEARING SWIPE SETUP "
            "(this is where a naive attempt fails): the swipe sweeps only "
            "pieces whose lane the arm's BODY already spans, so you must "
            "(a) put the agent on a CLEAR lane OFFSET from the pieces (one "
            "row/column to the side), (b) extend the arm THERE to reach "
            "across the target piece's lane — extending along the pieces' "
            "OWN lane arrests on contact with the first piece and never "
            "gets the arm body past/over it, so the sweep then moves "
            "nothing — then (c) move ACROSS the pieces' lane so the spanning "
            "arm body carries the piece off it.  Repeat until every piece "
            "occupies its own clear lane.\n"
            "  (4a) OFFSET-LANE SYMMETRY (do not skip this step): the "
            "offset can be on EITHER side of the pieces' lane (above OR "
            "below, left OR right) — the sweep then carries the piece to "
            "the OPPOSITE side.  BEFORE choosing a side, ENUMERATE BOTH "
            "and pick by the relations: prefer the side that puts the "
            "piece into HIGHER clearance after the sweep, and AVOID the "
            "side that would sweep a piece toward a nearby boundary "
            "(check `clearance` relations).  A wrong-side choice is what "
            "creates a wall-jam dead-end; the right-side choice is what "
            "prevents one.  The operation is RELATIVE (offset->across), "
            "not absolute (`below->up`); treating it as absolute reuses "
            "whichever side you used last and re-creates the jam.\n"
            "  (5) COLLECT IN ORDER — each piece is now individually "
            "addressable, so run the normal ordered-collection procedure: "
            "visit and act on each piece in the required target order "
            "(the win's ordered_completion sequence).\n"
            "RECOVERY-IF-ALREADY-JAMMED: if a cluster is already flush "
            "against a boundary with NO clear lane BEYOND it to sweep it "
            "back (e.g. shoved into the very top/edge row) AND the next "
            "ordered target sits BEHIND out-of-order pieces, the "
            "offset-sweep cannot extract it and pushing the chain toward "
            "the goal-wall does NOT help (the agent's tip cannot bypass the "
            "out-of-order pieces to reach the target).  The reliable escape "
            "is to UNDO (rewind) back to BEFORE the move that jammed it, "
            "then redo with the prevention/offset approach.  CLEARING THE "
            "ARM for a vertical reposition: retract ONLY until the tip is "
            "just past/clear of the target pieces — do NOT full-retract a "
            "carried pierced piece all the way to the base, which "
            "UN-completes it (a pierced piece survives partial drag but "
            "reverts if dragged fully back to the origin).\n"
            "WATCH OUT: prevention beats recovery — avoid sweeping pieces "
            "flush against a boundary in the first place.  Phase (2)'s "
            "open region must be large enough to separate ALL carried "
            "pieces in phase (4).  OPERATION CAUTION: if acting on a piece "
            "to grab/skewer it ALSO COMPLETES it (e.g. skewering lights its "
            "HUD swatch) AND completion order is constrained, do NOT use "
            "the skewer-attach phases (1-3) to merely reposition — that "
            "completes pieces out of order.  Separate with the NON-"
            "completing sweep (phase 4) instead, then complete in order in "
            "phase (5)."
        ),
        "expected_outcome": (
            "Every previously-jammed piece occupies its own clear lane "
            "(no two pieces share a row or column), the agent is "
            "uncoupled and free, and the pieces can be collected one at a "
            "time in the required target order.  The `undo_only` dead-end "
            "no longer holds."
        ),
        "relational_steps": [
            "GATE: recover ONLY if clearance(cluster, sweep-target side)==0 (flush to a boundary). If clearance>0 on both sides of the separation axis already, SKIP relocation and go straight to SEPARATE where the pieces are; relocating an already-clear cluster destroys usable clearance and re-jams it",
            "ATTACH: agent grab/skewer the jammed pieces -> co_displacement(agent, pieces)",
            "CARRY: move agent (pieces co-moving) off the boundary into open space defined RELATIVE to the separation — clearance on the side each piece is swept toward, plus room on the offset side for the arm body. POST-CONDITION: clearance(cluster, sweep-target side) > 0 afterward; NEVER end flush against a boundary. Avoid colliding with non-carried pieces",
            "RELEASE: move agent to the detach-axis extreme until co_displacement(agent, pieces) ends",
            "SEPARATE: relative operation (offset-lane -> across-targets-lane), NOT absolute (`below->up`). (a) agent on a CLEAR offset lane on EITHER side of the targets' lane — ENUMERATE BOTH sides and pick by `clearance` relations (the side that sweeps INTO open space, never toward a nearby boundary); (b) extend the arm at that offset lane to reach across the targets' lane (extending ALONG the targets' own lane arrests on the first piece and sweeps nothing); (c) move ACROSS the targets' lane so the spanning arm body carries the piece off it; result same_row/same_col count==1; repeat per piece",
            "COLLECT: run the ordered-completion procedure over the now-separated pieces in target order",
        ],
        "game_id": "sk48",
        "level": 1,
        "original_goal": (
            "escape a wall-jammed dead-end and reach a collectable "
            "configuration (originating example: sk48 lc=1, green+blue "
            "blocks shoved flush against the top wall at turn 31)"
        ),
        "notes": (
            "DIALOGIC-LEARNING SEED (2026-05-31). Human-authored "
            "generalised recovery technique. Originating concrete "
            "instance: sk48 lc=1, green and blue blocks jammed against "
            "the TOP wall — pierce both so they move with the arm, carry "
            "them around other blocks into open space, slide the arm "
            "fully left to unskewer, side-swipe each onto its own empty "
            "row, then pierce each in HUD order. Generalised here to any "
            "piece colour and any piece count. UNVERIFIED by the system's "
            "own play — credence is a human prior; attempts are 0/0."
        ),
        # No concrete_chain: no single literal action sequence generalises
        # over piece count.  The procedure lives in relational_steps.
        "concrete_chain": [],
        "turn_range": [31, 31],
    },
]


def apply_dialogic_seeds(
    path: Path = DEFAULT_SUBROUTINE_KB_PATH,
) -> list[Subroutine]:
    """Ingest every dialogic seed into the KB at ``path``.  Idempotent
    on name — re-running refreshes descriptive/step text without
    touching any counters the system has earned since."""
    out: list[Subroutine] = []
    for seed in DIALOGIC_SUBROUTINES:
        out.append(ingest_dialogic_subroutine(path=path, **seed))
    return out


if __name__ == "__main__":  # pragma: no cover
    subs = apply_dialogic_seeds()
    for s in subs:
        print(f"ingested dialogic subroutine: {s.subroutine_id}  "
              f"(credence={s.credence}, attempts={s.attempts.n_applied} "
              f"applied)")
