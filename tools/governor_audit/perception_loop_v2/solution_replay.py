"""Deterministic replay of recorded solutions to FAST-FORWARD a live
game to the level under study, with **no VLM calls** (budget-free).

Why this exists
---------------
ARC-AGI-3 levels of one game are reached only by SOLVING the levels
before them (the live harness always resets to level 0; it cannot
seek).  Re-discovering an already-solved early level every time we want
to work on a later one wastes the (slow, budget-heavy) VLM loop, and a
long live run that dies mid-session (the idle-reap blocker) then has to
start over from level 0.

The substrate already records every win as a REPLAYABLE artifact: on
each ``lc++`` ``exploratory_driver._register_solution_on_solve`` saves
the exact winning act-sequence into the canonical ``solutions_kb``
(shortest path stays canonical; dedup is by act-signature).  This module
is the missing CONSUMER of that artifact: given the game adapter and a
``recall`` callback (``solutions_kb.recall_solution`` bound to the game),
it resets the game and steps each prior level's canonical win-path back
into the env, verifying the score advances after each level.

It is a pure CONSUMER -- it never writes to ``solutions_kb`` -- so a
replay can **never create a duplicate solution**.  The only solution
written in a fast-forwarded run is the genuinely-new one for the target
level, saved by the normal on-solve path once live play solves it.

Game-agnostic: it knows only the adapter contract (``turn_one_frame``,
``step``, ``_action_map``) and the ``win_path`` schema
(``[{action_id, name?, click_xy?}, ...]``).  It is unit-testable
against a fake adapter with no live env.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Act -> adapter action string
# ---------------------------------------------------------------------------

def build_id_to_name(action_map: dict) -> dict:
    """Reverse an adapter ``_action_map`` (name -> id) into id -> name,
    preferring the raw game-agnostic ``ACTION_N`` name over a cardinal
    alias (UP/DOWN/...) when several names share one id.  Used only as a
    fallback when a recorded act has no ``name`` field."""
    cardinals = {"UP", "DOWN", "LEFT", "RIGHT", "CLICK"}
    out: dict = {}
    # First pass: raw ACTION_N names (preferred).
    for nm, aid in (action_map or {}).items():
        if nm not in cardinals:
            out.setdefault(int(aid), nm)
    # Second pass: cardinal aliases fill any id still unnamed.
    for nm, aid in (action_map or {}).items():
        out.setdefault(int(aid), nm)
    return out


def act_to_action_string(
    rec: dict,
    click_action_id: Optional[int],
    id_to_name: dict,
) -> str:
    """Reconstruct the adapter action string for one recorded act.

    A recorded act is ``{action_id:int, name?:str, click_xy?:[x,y]}``.
    The adapter (``game_adapter.LiveHarnessAdapter.step``) consumes
    string actions: ``"ACTION4"`` / ``"UP"`` / ``"CLICK"`` /
    ``"CLICK:x,y"``.

    The ``name`` field is the ACTION STRING as executed, so it is
    usually already complete -- for a targeted click the driver records
    ``name="CLICK:32,15"`` (coords baked in) with ``click_xy`` empty.  So
    a present ``name`` is returned verbatim; the only special case is a
    BARE ``"CLICK"`` carrying separate ``click_xy`` coords, which are
    re-attached.  Only when ``name`` is absent do we reconstruct from the
    numeric ``action_id`` (+ coords for a click).  Either way the replay
    clicks exactly where the original solve did (same tick-space, x=col,
    y=row).
    """
    name = rec.get("name")
    aid = rec.get("action_id")
    xy = rec.get("click_xy")
    has_xy = isinstance(xy, (list, tuple)) and len(xy) == 2
    if name:
        # Already a complete action string ("CLICK:32,15", "ACTION4",
        # "UP", "CLICK") -- return verbatim, except a bare "CLICK" with
        # separately-recorded coords, which we re-attach.
        if name == "CLICK" and has_xy:
            return f"CLICK:{int(xy[0])},{int(xy[1])}"
        return str(name)
    # No name: reconstruct from the numeric action_id.
    is_click = (click_action_id is not None and aid is not None
                and int(aid) == int(click_action_id))
    if is_click and has_xy:
        return f"CLICK:{int(xy[0])},{int(xy[1])}"
    if is_click:
        return "CLICK"
    if aid is not None and int(aid) in id_to_name:
        return id_to_name[int(aid)]
    raise ValueError(f"cannot reconstruct an action string from act {rec!r}")


# ---------------------------------------------------------------------------
# Replay result types
# ---------------------------------------------------------------------------

@dataclass
class LevelReplay:
    """Outcome of replaying one prior level's recorded solution."""
    level: int
    n_acts_played: int = 0
    advanced: bool = False
    final_score: Optional[int] = None
    end_turn: Optional[int] = None   # adapter turn at which this level advanced
    note: str = ""


@dataclass
class ReplayResult:
    target_level: int
    reached: bool = False
    levels: list = field(default_factory=list)
    final_frame: Optional[object] = None   # Path to the last rendered frame
    final_score: Optional[int] = None
    final_turn: Optional[int] = None
    note: str = ""


# ---------------------------------------------------------------------------
# The replay driver
# ---------------------------------------------------------------------------

def replay_to_level(
    *,
    game,
    recall: Callable[[int], Optional[dict]],
    target_level: int,
    log: Callable[[str], None] = print,
) -> ReplayResult:
    """Reset ``game`` and replay the canonical recorded solution for each
    level ``0 .. target_level-1``, verifying the score advances after
    each.  No VLM is consulted.

    Parameters
    ----------
    game     : a GameAdapter (``turn_one_frame``, ``step``, ``_action_map``).
    recall   : ``level -> solution-dict | None`` -- typically
               ``functools.partial(solutions_kb.recall_solution, game_id,
               only_canonical=True)``.  The solution dict carries
               ``win_path``.
    target_level : the level to fast-forward TO (live play resumes here).

    Returns
    -------
    ReplayResult.  ``reached`` is True only if every prior level had a
    recorded canonical solution AND each advanced the score.  On the
    first missing/stale solution it returns early with ``reached=False``
    so the caller can fall back to full play from level 0.
    """
    res = ReplayResult(target_level=int(target_level))
    if target_level <= 0:
        res.reached = True
        res.note = "target_level<=0: nothing to fast-forward"
        return res

    action_map = dict(getattr(game, "_action_map", {}) or {})
    click_id = action_map.get("CLICK")
    id_to_name = build_id_to_name(action_map)

    # Reset to the very first level.
    res.final_frame = game.turn_one_frame()
    res.final_turn = int(getattr(game, "current_turn", 0) or 0)
    score = 0

    for lvl in range(int(target_level)):
        sol = None
        try:
            sol = recall(lvl)
        except Exception as e:           # a broken recall must not crash
            log(f"[replay] recall(level={lvl}) failed ({e})")
        win_path = list((sol or {}).get("win_path") or [])
        lr = LevelReplay(level=lvl)
        if not win_path:
            lr.note = "no recorded canonical solution"
            res.levels.append(lr)
            res.note = (f"level {lvl}: no recorded solution -- cannot "
                        f"fast-forward (play from level 0 instead)")
            log(f"[replay] {res.note}")
            return res

        for rec in win_path:
            try:
                action = act_to_action_string(rec, click_id, id_to_name)
            except ValueError as e:
                lr.note = f"unreconstructable act #{lr.n_acts_played}: {e}"
                break
            try:
                sr = game.step(action)
            except Exception as e:
                lr.note = (f"step failed at act #{lr.n_acts_played} "
                           f"({action!r}): {e}")
                break
            lr.n_acts_played += 1
            res.final_frame = getattr(sr, "frame_path", res.final_frame)
            res.final_turn = int(getattr(game, "current_turn",
                                          res.final_turn) or res.final_turn)
            sc = getattr(sr, "score", None)
            if sc is not None:
                score = int(sc)
                lr.final_score = score
            win_state = getattr(sr, "win_state", "playing")
            if score >= lvl + 1:
                lr.advanced = True
                lr.end_turn = int(getattr(game, "current_turn",
                                          res.final_turn) or res.final_turn)
                break
            if win_state not in ("playing", None):
                # Game ended (won/lost) without the expected advance.
                lr.note = f"win_state={win_state} before advancing"
                break

        res.levels.append(lr)
        if not lr.advanced:
            res.note = (f"level {lvl} did not advance after "
                        f"{lr.n_acts_played} act(s) "
                        f"({lr.note or 'score unchanged'}) -- recorded "
                        f"solution is stale; play from level 0 instead")
            log(f"[replay] {res.note}")
            return res
        log(f"[replay] level {lvl} replayed in {lr.n_acts_played} act(s) "
            f"-> score {score}")

    res.reached = True
    res.final_score = score
    res.note = f"fast-forwarded to level {target_level} via recorded solutions"
    log(f"[replay] {res.note} (score={score}, turn={res.final_turn})")
    return res
