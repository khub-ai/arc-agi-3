"""Learn a mechanic from a DEMONSTRATION (game- and domain-agnostic).

Some elements DEMONSTRATE rather than act: activating them plays a PREVIEW
animation that shows a mover reaching a target, then reverts (the settled board
is unchanged).  The substrate already EXTRACTS such previews -- a mover that
traces a path and returns -- via ``animation_analysis.demonstration_narration``,
which yields per demonstration ``{identity, dir, ticks, d_row, d_col,
identity_credence}`` (the silhouette of the mover, the direction, and how far it
travelled).  Until now that was only narrated to the acting VLM; this module
SYNTHESISES it into reusable mechanic knowledge that drives the autonomous solve:

  (1) WIN from the demonstration.  A previewed mover motion SHOWS the desired
      outcome -- the mover should reach the demonstrated displacement.  ->
      a win-condition hypothesis, grounded in the OBSERVED preview.

  (2) PARAMETERISATION.  A controllable STRUCTURE whose unit COUNT equals the
      demonstrated MAGNITUDE parameterises that motion -- each unit is one step
      of travel -- so configuring the structure to the target produces the
      previewed result.  (The general form of '4 columns == 4 steps'.)

Both are grounded in a measured demonstration, so they carry MORE credence than
the merely-guessed structural priors -- which is what lets the win-pursuit and
structural-match machinery converge on the correct configuration instead of
speculative alternatives.

Borrowed cognitive mechanism: learning by demonstration / observation.  Nothing
here mentions any specific game -- only movers, magnitudes, directions, and
structures with a unit count -- so it transfers to any game (or domain) where an
element previews an outcome: a legend, a ghost piece, a tutorial animation, a
target overlay.
"""

from __future__ import annotations

from typing import Optional


def _magnitude_ticks(demo: dict) -> float:
    """How far the mover travelled in the preview, in frame ticks."""
    t = demo.get("ticks")
    if t:
        return abs(float(t))
    return float(max(abs(demo.get("d_row") or 0), abs(demo.get("d_col") or 0)))


def synthesize(demos, structure_cols, *, cell_ticks: int = 4) -> dict:
    """Synthesise mechanic knowledge from extracted demonstrations.

    Parameters
    ----------
    demos : iterable of demonstration dicts
        Each: ``{identity, dir, ticks, d_row, d_col, identity_credence}`` -- a
        mover that PREVIEWED a motion (and reverted).  Demonstrations without an
        identified mover, or with no travel, are skipped (nothing to learn).
    structure_cols : dict ``{structure_name: unit_count}``
        For each controllable STRUCTURE, how many units (columns / cells) it has
        -- computed by the caller (e.g. by decomposing a panel into its lattice).
        A structure whose count matches the demonstrated magnitude is the one
        that parameterises the motion.
    cell_ticks : int
        Ticks per grid cell, to convert the tick magnitude into STEPS (cells).

    Returns
    -------
    ``{"win": [win-hyp dicts], "claims": [claim dicts]}`` -- all grounded in the
    demonstration (credence above the guessed structural priors).  Empty when
    there is nothing to learn.
    """
    win: list = []
    claims: list = []
    ct = float(cell_ticks) if cell_ticks else 1.0
    for d in (demos or []):
        ident = d.get("identity")
        if not ident:
            continue                       # un-identified preview -> nothing to bind
        mag = _magnitude_ticks(d)
        if mag <= 0:
            continue
        steps = max(1, int(round(mag / ct)))
        dirn = (d.get("dir") or "").strip() or "?"
        idc = float(d.get("identity_credence") or 0.0)
        # WIN: the preview shows the desired outcome.  Credence scales with how
        # confidently the mover was identified, but stays moderate (it is a
        # preview, not a confirmed win).
        win.append({
            "description": (f"previewed goal: move '{ident}' ~{steps} step(s) "
                            f"{dirn} to the demonstrated target position"),
            "credence": round(min(0.85, 0.55 + 0.3 * idc), 2),
            "win_relation": {"mover": ident, "dir": dirn, "steps": steps,
                             "source": "demonstration"},
        })
        # PARAMETERISATION: a structure whose unit count == the demonstrated
        # magnitude controls that motion (each unit = one step).
        for sname, ncols in (structure_cols or {}).items():
            try:
                n = int(ncols)
            except Exception:
                continue
            if n >= 2 and abs(n - steps) <= 1:
                claims.append({
                    "id": f"demo_perstep__{sname}__{ident}",
                    "kind": "structural", "scope": "cross_game",
                    "statement": (
                        f"DEMONSTRATED: '{ident}' previewed moving {steps} step(s) "
                        f"{dirn}; structure '{sname}' has {n} units (== {steps} "
                        f"steps), so each unit sets ONE step -- configure '{sname}' "
                        f"to produce the demonstrated motion (this fulfils the win)."),
                    "target": [sname, ident],
                    "importance": 0.9,
                    "credence": 0.65,          # GROUNDED in the demo, > guessed (0.45)
                    "provenance": "observed",
                })
    return {"win": win, "claims": claims}
