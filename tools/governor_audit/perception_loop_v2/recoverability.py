"""Layer C — Undo-based recoverability.

Avoid one-way states by *trying* a candidate move and then taking the
game's Undo action (ACTION7 in the current ARC family) to check whether
the world's state restores.  Recoverability is MEASURED against the real
game, not inferred from a forward simulator.

See docs/SPEC_visual_reasoning_substrate.md (Layer C).

This module is deliberately split into:

  - PURE functions (fingerprints, classifier, table, veto surface) that
    are unit-testable offline against saved trials and synthetic inputs;
  - one DRIVER function (`probe_recoverability`) that executes real
    actions via injected step/perceive callables, so it is testable with
    a mock env and ready for the exploratory driver to wire real ones.

Four verdicts:
  reversible           — Undo restored the pre-move state (and any
                         progress the move made was preserved or the
                         move made none).  Safe to try.
  irreversible         — Undo did NOT restore the pre-move state.  The
                         move is one-way; veto unless deliberately
                         committed.
  progress_destructive — Undo restored geometry but wiped the win-
                         progress the move had made (score/level
                         regressed).  Recoverable for geometry, costly
                         for progress: surfaced as a warning, not a veto.
  undo_unavailable     — The undo action produced no observable change
                         (not implemented / inert in this game).  C
                         cannot judge recoverability; pass through.

Game-agnostic: every fingerprint is computed from entity positions and
scalar game state (score / level / win_state), never from game-specific
vocabulary.
"""
from __future__ import annotations

from typing import Callable, Optional

from world_knowledge import WorldKnowledge, EntityRecord   # noqa: E402


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def _entity_position(rec: EntityRecord) -> Optional[tuple[float, float, bool]]:
    """Best available position for an entity as (row, col, is_cell).

    `is_cell` is True when the position is a discrete grid cell (compare
    exactly — a 1-cell move matters) and False when it is a bbox
    centroid in tick-space (compare with jitter tolerance — sub-tick
    animation wobble is not a real move)."""
    if rec.current_cell is not None:
        return (float(rec.current_cell[0]), float(rec.current_cell[1]), True)
    bb = rec.current_bbox
    if bb is not None:
        return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0, False)
    return None


def exact_state_signature(world: WorldKnowledge) -> dict:
    """Precise enough to detect whether an Undo restored the world.

    Compares on entity POSITIONS (cell or bbox centroid) plus the
    scalar game state (score / level / win_state).  Deliberately NOT
    pixel-based — transient animations vary frame-to-frame but the
    symbolic state does not.  See spec: 'Fingerprints are compared on
    relations + key properties, not on raw pixels.'
    """
    positions: dict[str, tuple[float, float, bool]] = {}
    for name, rec in world.entities.items():
        p = _entity_position(rec)
        if p is not None:
            positions[name] = p
    return {
        "positions": positions,
        "score": world.score,
        "level": world.level,
        "win_state": world.win_state,
    }


# Cell-space positions are discrete; half a cell is the natural
# "different cell" threshold.  Tick-space (bbox centroid) positions need
# a looser tolerance to absorb sub-tick animation jitter.
_CELL_TOL = 0.5


def _positions_match(a: dict, b: dict, tick_tol: float = 2.0) -> bool:
    """Two position maps match if every shared entity is within the
    appropriate tolerance and neither side has an entity the other lacks
    (appearance / disappearance is a state difference).

    Cell-space entities (is_cell True) compare with _CELL_TOL so a
    one-cell move always registers; tick-space entities compare with
    `tick_tol` to absorb animation jitter."""
    if set(a.keys()) != set(b.keys()):
        return False
    for name, pa in a.items():
        pb = b[name]
        ar, ac = pa[0], pa[1]
        br, bc = pb[0], pb[1]
        is_cell = (len(pa) > 2 and pa[2]) or (len(pb) > 2 and pb[2])
        tol = _CELL_TOL if is_cell else tick_tol
        if abs(ar - br) > tol or abs(ac - bc) > tol:
            return False
    return True


def signatures_match(a: dict, b: dict, pos_tol: float = 2.0) -> bool:
    """Full exact-state match: positions within tolerance AND identical
    scalar game state.  `pos_tol` is the TICK-space tolerance; cell-space
    positions always use the discrete _CELL_TOL."""
    return (
        _positions_match(a.get("positions", {}), b.get("positions", {}),
                          pos_tol)
        and a.get("score") == b.get("score")
        and a.get("level") == b.get("level")
        and a.get("win_state") == b.get("win_state")
    )


def _progress_value(sig: dict) -> tuple:
    """Scalar progress extracted from a signature: (score, level).
    Used to detect whether a move made progress and whether Undo wiped
    it.  None scores are treated as 0 for ordering."""
    return (sig.get("score") or 0, sig.get("level") or 0)


# ---------------------------------------------------------------------------
# State-class (coarse key for caching verdicts)
# ---------------------------------------------------------------------------


def state_class(world: WorldKnowledge, action: str) -> str:
    """A COARSE, game-agnostic description of 'this kind of state',
    used to key cached recoverability verdicts so a verdict generalizes
    across encounters of the same situation.

    Derived from the support_relations active in the current state (the
    walls/solids bracketing entities — the recoverability-relevant
    feature) plus a coarse entity-role count.  NOT keyed on entity
    names, so the verdict transfers within the game and (when later
    promoted) across games.

    Approximation: 'current state' relations are read from the most
    recent delta's support_relation entries, which describe the present
    configuration (support_relation is a function of current bboxes, not
    of the transition).
    """
    support_feats: set[tuple] = set()
    deltas = getattr(world, "deltas_observed", None) or []
    if deltas:
        last = deltas[-1]
        rels = getattr(last, "relations", None) or []
        for r in rels:
            kind = r.get("kind") if isinstance(r, dict) else getattr(r, "kind", None)
            if kind != "support_relation":
                continue
            direction = (r.get("direction") if isinstance(r, dict)
                          else getattr(r, "direction", None))
            support_feats.add(("support", direction))
    # Coarse role count
    role_counts: dict[str, int] = {}
    for rec in world.entities.values():
        role = (rec.current_role or "unknown").lower()
        if role in ("scenery", "decoration", "hud", "background"):
            continue
        role_counts[role] = role_counts.get(role, 0) + 1
    role_str = ",".join(f"{k}={v}" for k, v in sorted(role_counts.items()))
    feat_str = ",".join(f"{k}:{d}" for (k, d) in sorted(
        support_feats, key=lambda x: (x[0], str(x[1]))))
    return f"action={action};support={feat_str};roles={role_str}"


def _table_key(action: str, sclass: str) -> str:
    return f"{action}@@{sclass}"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


def classify_recoverability(pre: dict,
                              post_move: dict,
                              post_undo: dict,
                              pos_tol: float = 2.0) -> tuple[str, dict]:
    """Classify a probe outcome into one of the four verdicts.

    Inputs are exact_state_signature dicts captured before the move,
    after the move, and after the undo.  Returns (verdict, evidence).
    """
    move_changed = not signatures_match(pre, post_move, pos_tol)
    undo_changed = not signatures_match(post_move, post_undo, pos_tol)
    geometry_restored = signatures_match(pre, post_undo, pos_tol)

    evidence = {
        "move_changed_state": move_changed,
        "undo_changed_state": undo_changed,
        "geometry_restored": geometry_restored,
        "pre_progress": _progress_value(pre),
        "post_move_progress": _progress_value(post_move),
        "post_undo_progress": _progress_value(post_undo),
    }

    # If the move itself changed nothing, recoverability is moot —
    # there's nothing to recover from.  Treat as reversible (safe).
    if not move_changed:
        return "reversible", {**evidence, "note": "move was a no-op"}

    # If undo produced no observable change while the move had changed
    # the state, undo is not functioning as an inverse here.
    if not undo_changed:
        return "undo_unavailable", {
            **evidence,
            "note": "undo produced no change after a state-changing move",
        }

    if geometry_restored:
        # Geometry is back to pre-move.  Did the move make progress that
        # undo then wiped?
        made_progress = _progress_value(post_move) > _progress_value(pre)
        progress_wiped = _progress_value(post_undo) < _progress_value(post_move)
        if made_progress and progress_wiped:
            return "progress_destructive", {
                **evidence,
                "note": "undo restored geometry but wiped progress",
            }
        return "reversible", evidence

    # Geometry NOT restored — one-way move.
    return "irreversible", {
        **evidence,
        "note": "undo did not restore the pre-move state",
    }


# ---------------------------------------------------------------------------
# Inverse-action table (within-trial cache on world.inverse_actions)
# ---------------------------------------------------------------------------


def lookup_verdict(world: WorldKnowledge,
                    action: str,
                    sclass: Optional[str] = None) -> Optional[dict]:
    """Return the cached verdict dict for (action, state-class), or
    None if not yet probed."""
    if sclass is None:
        sclass = state_class(world, action)
    return world.inverse_actions.get(_table_key(action, sclass))


def record_verdict(world: WorldKnowledge,
                    action: str,
                    sclass: str,
                    verdict: str,
                    evidence: dict,
                    undo_action: str) -> dict:
    """Store (or reinforce) a verdict in the within-trial table.
    Re-probing the same (action, state-class) with the same verdict
    bumps a confirmation counter; a conflicting verdict overwrites with
    recency and flags the conflict for B (coarse state-class)."""
    key = _table_key(action, sclass)
    existing = world.inverse_actions.get(key)
    if existing is not None and existing.get("verdict") == verdict:
        existing["confirmations"] = existing.get("confirmations", 1) + 1
        return existing
    entry = {
        "verdict": verdict,
        "action": action,
        "state_class": sclass,
        "undo_action": undo_action,
        "evidence": evidence,
        "discovered_at_turn": world.turn,
        "confirmations": 1,
    }
    if existing is not None and existing.get("verdict") != verdict:
        entry["conflict_with_prior"] = existing.get("verdict")
    world.inverse_actions[key] = entry
    return entry


# ---------------------------------------------------------------------------
# Probe driver — executes real actions via injected callables
# ---------------------------------------------------------------------------


def probe_recoverability(world: WorldKnowledge,
                          action: str,
                          step_fn: Callable[[str], None],
                          perceive_fn: Callable[[], None],
                          undo_action: str = "ACTION7",
                          pos_tol: float = 2.0) -> dict:
    """Execute a recoverability probe against the live game.

    Procedure:
      1. snapshot pre-move exact signature
      2. step_fn(action); perceive_fn()  -> world reflects post-move
      3. snapshot post-move signature
      4. step_fn(undo_action); perceive_fn() -> world reflects post-undo
      5. snapshot post-undo signature
      6. classify, store in the table, return the verdict entry

    `step_fn` executes one action against the real env.  `perceive_fn`
    updates `world` (entities, score, etc.) to reflect the new frame —
    the same perception path a normal turn uses.  Both are injected so
    this function is testable with a mock env and reusable by the
    exploratory driver with real ones.

    The state-class is computed from the PRE-move world so the verdict
    is keyed by the situation in which the move was attempted.
    """
    sclass = state_class(world, action)
    pre = exact_state_signature(world)

    step_fn(action)
    perceive_fn()
    post_move = exact_state_signature(world)

    step_fn(undo_action)
    perceive_fn()
    post_undo = exact_state_signature(world)

    verdict, evidence = classify_recoverability(
        pre, post_move, post_undo, pos_tol,
    )
    return record_verdict(world, action, sclass, verdict, evidence,
                           undo_action)


# ---------------------------------------------------------------------------
# Normal-action recoverability (the dead-end distinct from undo-reversibility)
#
# A move can be "reversible" by Undo yet still drop the game into a state
# that NO NORMAL action escapes — e.g. shoving a block against the top wall
# in a gravity-less game: Undo restores it, but no ordinary move brings it
# back, so it is a dead-end for normal play.  The undo-based verdict above
# CONFLATES these (it reports "reversible" because Undo works).  This pass
# distinguishes them with an explicit `undo_only` verdict: the move is
# recoverable ONLY by spending an Undo, so entering the state is a normal-
# action dead-end the actor should avoid.
# ---------------------------------------------------------------------------


def classify_normal_recoverability(pre: dict,
                                    post_move: dict,
                                    post_inverse: dict,
                                    post_undo: Optional[dict] = None,
                                    pos_tol: float = 2.0) -> tuple[str, dict]:
    """Classify a normal-recovery probe.  `post_inverse` is the state
    after attempting a NORMAL inverse action from post_move; `post_undo`
    (if given) is the state after undo-based cleanup, evaluated only when
    the normal inverse failed.  Verdicts:
      normal_recoverable — a normal action restored the pre-move state.
      undo_only          — normal inverse failed BUT Undo restored it:
                           a dead-end escapable only by Undo.
      irreversible       — neither normal nor Undo restored it.
      reversible         — the move was a no-op (nothing to recover).
    """
    if signatures_match(pre, post_move, pos_tol):
        return "reversible", {"note": "move was a no-op"}
    if signatures_match(pre, post_inverse, pos_tol):
        return "normal_recoverable", {
            "note": "a normal action restored the pre-move state"}
    if post_undo is not None and signatures_match(pre, post_undo, pos_tol):
        return "undo_only", {
            "note": "no normal action restored it; only Undo did — "
                    "normal-action dead-end"}
    return "irreversible", {
        "note": "neither the tried normal inverse nor Undo restored it"}


def probe_normal_recoverability(world: WorldKnowledge,
                                 action: str,
                                 normal_inverse: str,
                                 step_fn: Callable[[str], None],
                                 perceive_fn: Callable[[], None],
                                 undo_action: str = "ACTION7",
                                 pos_tol: float = 2.0) -> dict:
    """Probe whether `action` is recoverable by NORMAL play, using a
    single best-guess `normal_inverse` candidate (e.g. the opposite-
    direction action).  Single-path (no branching) so it works on a real
    env with no state-clone:

      1. snapshot pre; step(action) -> post_move
      2. step(normal_inverse) -> post_inverse
         - if it restored pre  -> normal_recoverable (world already clean)
      3. else undo twice (unwind the inverse, then the move) -> post_undo
         - if Undo restored pre -> undo_only (dead-end); else irreversible

    Heuristic: only ONE inverse candidate is tried, so a false
    `undo_only`/`irreversible` is possible if a *different* normal action
    (or a multi-step maneuver) would have recovered it.  Pass the most
    plausible inverse; the verdict is cached by state-class.
    """
    sclass = state_class(world, action)
    pre = exact_state_signature(world)

    step_fn(action)
    perceive_fn()
    post_move = exact_state_signature(world)

    step_fn(normal_inverse)
    perceive_fn()
    post_inverse = exact_state_signature(world)

    post_undo = None
    if not signatures_match(pre, post_inverse, pos_tol) \
            and not signatures_match(pre, post_move, pos_tol):
        # normal inverse failed and the move did change state — unwind
        # with Undo (the inverse, then the move) and check.
        step_fn(undo_action)
        perceive_fn()
        step_fn(undo_action)
        perceive_fn()
        post_undo = exact_state_signature(world)

    verdict, evidence = classify_normal_recoverability(
        pre, post_move, post_inverse, post_undo, pos_tol,
    )
    evidence["normal_inverse_tried"] = normal_inverse
    return record_verdict(world, action, sclass, verdict, evidence,
                           undo_action)


# ---------------------------------------------------------------------------
# Strategy-prompt veto surface
# ---------------------------------------------------------------------------


def recoverability_for_action(world: WorldKnowledge,
                               action: str) -> Optional[dict]:
    """Convenience: the cached verdict for `action` in the CURRENT
    state-class, or None if unprobed."""
    return lookup_verdict(world, action, state_class(world, action))


def format_recoverability_vetoes(world: WorldKnowledge,
                                   candidate_actions: list[str]) -> str:
    """Render per-action recoverability annotations for the strategy
    prompt.  Only emits a block when at least one candidate action has
    a non-trivial verdict (irreversible / progress_destructive /
    undo_unavailable).  reversible and unprobed actions are silent."""
    if not candidate_actions:
        return ""
    notes: list[str] = []
    for a in candidate_actions:
        entry = recoverability_for_action(world, a)
        if entry is None:
            continue
        verdict = entry.get("verdict")
        if verdict == "irreversible":
            notes.append(
                f"    - {a}: VETO — one-way in this state-class. "
                f"A previous probe showed {entry.get('undo_action')} did "
                f"NOT restore the pre-move state. Do not take {a} unless "
                f"you deliberately accept an irreversible change."
            )
        elif verdict == "undo_only":
            notes.append(
                f"    - {a}: VETO — DEAD-END. A previous probe showed NO "
                f"normal action escapes the resulting state; only "
                f"{entry.get('undo_action')} (Undo) recovers it. Avoid "
                f"entering it unless you will spend an Undo to get out. "
                f"NOTE: a recovery procedure for this kind of dead-end may "
                f"be listed under STORED SUBROUTINES below — check it "
                f"before abandoning this path; a stored procedure can turn "
                f"a dead-end into a detour."
            )
        elif verdict == "progress_destructive":
            notes.append(
                f"    - {a}: WARNING — recoverable geometry but "
                f"{entry.get('undo_action')} wipes the progress this move "
                f"makes. Undo is not a free scratch here."
            )
        elif verdict == "undo_unavailable":
            notes.append(
                f"    - {a}: NOTE — {entry.get('undo_action')} does not "
                f"reverse moves in this game; recoverability cannot be "
                f"verified. Proceed with care."
            )
    if not notes:
        return ""
    header = (
        "  RECOVERABILITY (Layer C — verdicts MEASURED by probing the "
        "real game with Undo; respect VETOs unless you state a reason):"
    )
    return header + "\n" + "\n".join(notes)
