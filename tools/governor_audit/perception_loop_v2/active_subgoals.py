"""Active-subgoal management — helpers for committing, updating,
and surfacing the actor's explicit subgoal commitments.

Substrate role: this module is the DURABLE CONTAINER side of the
subgoal flow.  It does NOT classify, plan, or interpret.  It just:

  * Creates ActiveSubgoal records when the actor commits.
  * Updates their status when the actor reports a change.
  * Renders the active set into the strategy prompt so the next
    turn's actor sees what it is currently committed to.

All judgement (when to commit, when to close, when to abandon) is
the actor's.  The substrate just forces the actor to be explicit.

See ``docs/SPEC_active_subgoals.md`` for the design discussion.
"""
from __future__ import annotations

import time
from typing import Optional

from world_knowledge import ActiveSubgoal, WorldKnowledge


# ---------------------------------------------------------------------------
# Id generation
# ---------------------------------------------------------------------------


def _new_id(name: str, turn: int) -> str:
    """Stable-ish id from name + turn.  Just unique per WK; no
    cryptographic intent."""
    safe = "".join(c if c.isalnum() or c == "_" else "_"
                   for c in name.lower())[:40]
    return f"sg_{safe}_t{turn}_{int(time.time()) % 100000}"


# ---------------------------------------------------------------------------
# Commit / update
# ---------------------------------------------------------------------------


def commit_subgoal(
    world: WorldKnowledge,
    *,
    name: str,
    problem_solved: str,
    expected_outcome: str,
    parent_id: Optional[str] = None,
    related_subroutine_id: Optional[str] = None,
    notes: str = "",
    forward_simulation: str = "",
    derived_from: str = "",
    win_condition_hypothesis_id: Optional[str] = None,
    depends_on: Optional[list] = None,
    acceptance_check: str = "",
    references_entities: Optional[list] = None,
    or_group: Optional[str] = None,
) -> ActiveSubgoal:
    """Author a new ActiveSubgoal at the current world turn.  All
    descriptive fields are free-form text the actor supplied.

    New (post-2026-05-28) discipline fields:
      - forward_simulation: the actor's mental-simulation trace
        for the planned action sequence.  See vlm_strategy
        STRATEGY_PROMPT for the discipline.
      - derived_from: the actor's free-form link to source —
        'from win_condition_hypothesis: <id>' or 'to advance
        parent <id>: <reasoning>'.  Forces non-orphan tactics.
      - win_condition_hypothesis_id: for top-level GOAL subgoals,
        points at the WinConditionHypothesis served.  When that
        hypothesis is contradicted, this subgoal is flagged.

    Sequential subgoals (added 2026-05-29):
      - depends_on: list of subgoal_ids that must reach status
        ACHIEVED before this subgoal becomes ACTIONABLE.  Bridged
        to a cognitive_os Goal so the engine's is_actionable check
        gates pursuit.  Empty / None means the subgoal is
        unconditionally actionable.  Cycles and references to
        non-existent subgoals are rejected at commit time (engine
        validation).
    """
    sg = ActiveSubgoal(
        subgoal_id=_new_id(name, world.turn),
        name=name,
        problem_solved=problem_solved,
        expected_outcome=expected_outcome,
        parent_id=parent_id,
        created_at_turn=world.turn,
        status="active",
        closed_at_turn=None,
        notes=notes,
        related_subroutine_id=related_subroutine_id,
        forward_simulation=forward_simulation,
        derived_from=derived_from,
        win_condition_hypothesis_id=win_condition_hypothesis_id,
        depends_on=list(depends_on or []),
        acceptance_check=acceptance_check or "",
        references_entities=list(references_entities or []),
        or_group=or_group,
    )
    world.active_subgoals.append(sg)

    # Bridge to cognitive_os Goal Forest so depends_on can be
    # enforced by the engine's is_actionable check.  Best-effort:
    # if validation rejects the dependency expression (missing
    # ref / cycle), we annotate the subgoal but don't crash the
    # turn — the actor can re-commit a corrected version next
    # turn.
    try:
        from subgoal_forest_bridge import register_subgoal  # noqa: E402
        register_subgoal(world, sg)
    except ValueError as e:
        sg.notes = ((sg.notes + "\n") if sg.notes else "") + (
            f"[bridge] dependency validation failed: {e}.  "
            f"depends_on={sg.depends_on}; treated as unconditional "
            f"(no gating).  Re-commit with corrected ids if needed."
        )
        sg.depends_on = []
    except Exception as e:
        # Bridge unavailable for any other reason: subgoal still
        # works as before (no gating).  Surface in notes for
        # debugging.
        sg.notes = ((sg.notes + "\n") if sg.notes else "") + (
            f"[bridge] unavailable ({type(e).__name__}: {e}); "
            f"depends_on ignored, no gating applied."
        )

    return sg


def update_subgoal_status(
    world: WorldKnowledge,
    *,
    subgoal_id: str,
    status: str,
        # "active" | "achieved" | "inferred_satisfied" |
        # "abandoned" | "blocked"
    notes_append: str = "",
    confirming_signal: str = "",
    impediment: str = "",
) -> Optional[ActiveSubgoal]:
    """Update the status of an existing subgoal.  Returns the
    updated record, or None if the id wasn't found.

    Substrate gate: status='achieved' REQUIRES a confirming_signal
    (free-form citation of an observed signal — delta index,
    visual event entity, score advance, win_state change).
    Without it, the status is DOWNGRADED to 'inferred_satisfied'
    and the subgoal stays in the open set.  This prevents the
    actor from marking subgoals achieved on inference alone.

    Component C gate: status='blocked' REQUIRES an ``impediment``
    description.  When supplied, the driver spawns a removal
    subgoal and links this one via depends_on (handled in the
    driver, not here — this function just records the impediment
    and refuses a bare 'blocked' with no impediment by keeping the
    subgoal active and noting the omission).
    """
    target: Optional[ActiveSubgoal] = None
    for sg in world.active_subgoals:
        if sg.subgoal_id == subgoal_id:
            target = sg
            break
    if target is None:
        return None

    requested_status = status
    if status == "blocked" and not (impediment or "").strip():
        # Component C gate: a bare 'blocked' with no named obstacle
        # is drift in disguise.  Keep the subgoal active and prod
        # the actor to name what blocks it.
        status = "active"
        notes_append = (
            (notes_append + "\n  ") if notes_append else ""
        ) + (
            "BLOCKED REJECTED: no impediment named.  To block a "
            "subgoal you must describe WHAT blocks progress so the "
            "substrate can spawn a removal subgoal.  Staying active."
        )
    if (status == "blocked") and (impediment or "").strip():
        target.impediment = impediment.strip()
    if status == "achieved" and not (confirming_signal or "").strip():
        # Downgrade: actor cited no observed signal.  Keep in the
        # open set as inferred_satisfied.
        status = "inferred_satisfied"
        downgrade_note = (
            "STATUS DOWNGRADED to inferred_satisfied: actor "
            "requested 'achieved' but cited no confirming "
            "signal.  Find a delta / visual event / score "
            "change that confirms expected_outcome, then "
            "re-submit with confirming_signal set."
        )
        notes_append = (
            (notes_append + "\n  ") if notes_append else ""
        ) + downgrade_note

    target.status = status
    if confirming_signal:
        sep = "\n" if target.confirming_signal else ""
        target.confirming_signal = (
            (target.confirming_signal or "") + sep
            + f"t{world.turn}: {confirming_signal}"
        )
    if notes_append:
        sep = "\n" if target.notes else ""
        target.notes = (target.notes or "") + sep + (
            f"t{world.turn}: {notes_append}"
        )
    if status in ("achieved", "abandoned", "invalidated"):
        target.closed_at_turn = world.turn

    # Bridge sync: when status flips to 'achieved' (the real one,
    # not the inferred_satisfied downgrade), mark the mirrored
    # Goal ACHIEVED so dependent subgoals become actionable.
    # 'abandoned' does NOT unblock dependents — by design, since
    # an abandoned precondition still means the dependent's
    # premise is missing.
    if status == "achieved":
        try:
            from subgoal_forest_bridge import mark_subgoal_achieved  # noqa: E402
            mark_subgoal_achieved(world, target.subgoal_id)
        except Exception:
            pass  # bridge unavailable; soft fail
        # OR-group: achieving any member supersedes the rest.
        _invalidate_or_siblings(world, target)

    return target


# ---------------------------------------------------------------------------
# Subgoal Completion Contract — substrate-side evaluation (A + invalidation)
# ---------------------------------------------------------------------------


def _invalidate(world: WorldKnowledge, sg: ActiveSubgoal,
                reason: str) -> None:
    """Substrate-automatic invalidation.  Distinct from actor
    'abandoned': the subgoal's premise evaporated.  Never gated by
    exhaustion."""
    if sg.status in ("achieved", "abandoned", "invalidated"):
        return
    sg.status = "invalidated"
    sg.closed_at_turn = world.turn
    sep = "\n" if sg.notes else ""
    sg.notes = (sg.notes or "") + sep + f"t{world.turn}: INVALIDATED — {reason}"


def _invalidate_or_siblings(world: WorldKnowledge,
                            achieved: ActiveSubgoal) -> None:
    """When an OR-group member is achieved, invalidate the other
    open members of the same group (achieve-one-of-N semantics)."""
    if not achieved.or_group:
        return
    for sg in world.active_subgoals:
        if sg.subgoal_id == achieved.subgoal_id:
            continue
        if sg.or_group != achieved.or_group:
            continue
        if sg.status in ("active", "blocked", "inferred_satisfied"):
            _invalidate(
                world, sg,
                f"superseded by OR-group sibling "
                f"{achieved.subgoal_id} (one-of-{achieved.or_group} "
                f"is sufficient)",
            )


def _delta_matches_token(delta, token: str, world=None) -> bool:
    """Evaluate one acceptance-predicate token against a
    DeltaRecord.  Closed vocabulary; entity names sourced from
    perception.

    Most tokens are delta-local (visual_event, entity_changed, ...).
    RELATIONAL tokens additionally need the WORLD (current geometry):
      - ``mediation_precondition_met``: a FREE intermediary is positioned
        under a target (the indirect-push precondition). This lets a
        relational subgoal like 'stage free blues under reds' achieve ONLY
        when the substrate actually detects the configuration — not on a
        generic block-moved visual event, which was firing far too loosely
        and falsely marking the whole subgoal chain achieved. Requires
        ``world``; returns False without it."""
    token = token.strip()
    if not token:
        return False
    # Ordered-objective step token: ``ordered_step_complete:<k>`` fires when
    # the first k+1 steps of the committed ordered objective are done IN ORDER.
    # This is what makes a compiled ordered-objective subgoal achieve only when
    # its step actually completes, and (via depends_on) keeps later steps gated.
    if token.startswith("ordered_step_complete:"):
        if world is None:
            return False
        try:
            from ordered_objective_goals import ordered_step_satisfied  # noqa
            return ordered_step_satisfied(world, int(token.split(":", 1)[1]))
        except Exception:
            return False
    # Terminal: the ordered objective is fully satisfied == the level is WON.
    if token == "ordered_objective_won":
        if world is None:
            return False
        try:
            from ordered_objective_goals import objective_won  # noqa: E402
            return objective_won(world)
        except Exception:
            return False
    # Relational (world-aware) tokens.
    if token == "mediation_precondition_met":
        if world is None:
            return False
        try:
            from plan_consistency import (                      # noqa: E402
                build_structural_context, detect_mediation_candidates,
                classify_attachment,
            )
            ctx = build_structural_context(world)
            att = classify_attachment(world)
            cands = detect_mediation_candidates(
                ctx["entities"], obstacles=ctx["obstacles"], attachment=att)
            return len(cands) > 0
        except Exception:
            return False
    changed = set(getattr(delta, "entities_changed", []) or [])
    appeared = set(getattr(delta, "entities_appeared", []) or [])
    disappeared = set(getattr(delta, "entities_disappeared", []) or [])
    viz = getattr(delta, "visual_events", []) or []
    viz_names = {
        (v.get("entity") if isinstance(v, dict) else str(v))
        for v in viz
    }
    if token == "score_increased":
        return bool(getattr(delta, "score_increased", False))
    if token == "win_state_changed":
        return bool(getattr(delta, "win_state_changed", False))
    if ":" in token:
        kind, _, arg = token.partition(":")
        if kind == "entity_changed":
            return arg in changed
        if kind == "entity_appeared":
            return arg in appeared
        if kind == "entity_disappeared":
            return arg in disappeared
        if kind == "visual_event":
            return arg in viz_names
        if kind == "agent_at_cell":
            cell = getattr(delta, "agent_new_cell", None)
            return cell is not None and str(cell).replace(" ", "") == arg
    return False


def evaluate_acceptance(sg: ActiveSubgoal, delta, world=None) -> Optional[str]:
    """Return the matching token (a confirming-signal string) if the
    subgoal's acceptance_check fires against ``delta``, else None.
    Grammar: tokens joined by ' OR ' (any) / ' AND ' (all).  No
    nesting — keep it shallow and machine-checkable.

    ``world`` enables RELATIONAL tokens (e.g. mediation_precondition_met)
    that need current geometry, not just the delta."""
    expr = (sg.acceptance_check or "").strip()
    if not expr:
        return None
    if " AND " in expr:
        toks = [t.strip() for t in expr.split(" AND ")]
        if all(_delta_matches_token(delta, t, world) for t in toks):
            return expr
        return None
    toks = [t.strip() for t in expr.split(" OR ")]
    for t in toks:
        if _delta_matches_token(delta, t, world):
            return t
    return None


def substrate_evaluate_subgoals(world: WorldKnowledge, delta) -> list:
    """Called by the driver AFTER each delta is folded in.  Runs the
    substrate-authority transitions on every OPEN subgoal:

      1. acceptance_check fired  -> auto-achieved (component A)
      2. premise fell            -> auto-invalidated
         (referenced entity gone multi-frame; served WC hypothesis
          refuted; parent closed by another route)

    OR-group supersession is handled inside the achieve path.
    Returns a list of (subgoal_id, new_status, reason) for logging.
    """
    transitions: list = []
    open_states = ("active", "blocked", "inferred_satisfied")

    # Pass 1 — acceptance tests (achievement is authoritative and
    # may cascade OR-group invalidation).
    for sg in list(world.active_subgoals):
        if sg.status not in open_states:
            continue
        sig = evaluate_acceptance(sg, delta, world)
        if sig:
            update_subgoal_status(
                world, subgoal_id=sg.subgoal_id, status="achieved",
                confirming_signal=(
                    f"acceptance_check fired: {sig} (substrate-"
                    f"evaluated on this turn's delta)"
                ),
            )
            transitions.append(
                (sg.subgoal_id, "achieved", f"acceptance:{sig}")
            )

    # Pass 2 — premise checks (invalidation).
    refuted_wc = _refuted_wc_ids(world)
    closed_ids = {
        sg.subgoal_id for sg in world.active_subgoals
        if sg.status in ("achieved", "abandoned", "invalidated")
    }
    for sg in list(world.active_subgoals):
        if sg.status not in open_states:
            continue
        # premise: served WC hypothesis refuted
        if (sg.win_condition_hypothesis_id
                and sg.win_condition_hypothesis_id in refuted_wc):
            _invalidate(
                world, sg,
                f"served win-condition hypothesis "
                f"{sg.win_condition_hypothesis_id} was refuted",
            )
            transitions.append(
                (sg.subgoal_id, "invalidated", "wc_refuted")
            )
            continue
        # premise: parent closed by another route
        if sg.parent_id and sg.parent_id in closed_ids:
            _invalidate(
                world, sg,
                f"parent {sg.parent_id} closed (achieved/abandoned/"
                f"invalidated) by another route",
            )
            transitions.append(
                (sg.subgoal_id, "invalidated", "parent_closed")
            )
            continue
        # premise: ALL referenced entities gone (multi-frame)
        if sg.references_entities and _all_entities_gone(
            world, sg.references_entities,
        ):
            _invalidate(
                world, sg,
                f"referenced entities {sg.references_entities} no "
                f"longer present (multi-frame)",
            )
            transitions.append(
                (sg.subgoal_id, "invalidated", "entities_gone")
            )
    return transitions


def _refuted_wc_ids(world: WorldKnowledge, floor: float = 0.15) -> set:
    """Win-condition hypothesis ids whose credence has dropped to/
    below ``floor`` — treated as refuted premises."""
    out = set()
    for h in getattr(world, "win_condition_hypotheses", []) or []:
        cred = getattr(h, "credence", None)
        hid = getattr(h, "hypothesis_id", None)
        if hid is not None and cred is not None and cred <= floor:
            out.add(hid)
    return out


def _all_entities_gone(world: WorldKnowledge, names: list,
                       absent_turns: int = 3) -> bool:
    """True iff every named entity has been absent for >= absent_turns
    (multi-frame persistence — occlusion != disappearance)."""
    cur = world.turn
    for name in names:
        rec = world.entities.get(name)
        if rec is None:
            # never seen at all -> treat as gone only if the subgoal
            # outlived the absence window
            continue
        last = getattr(rec, "last_seen_turn", cur)
        if (cur - last) < absent_turns:
            return False  # this one is still recently present
    # all names are either unknown or stale beyond the window
    return any(
        (world.entities.get(n) is None)
        or (cur - getattr(world.entities[n], "last_seen_turn", cur))
        >= absent_turns
        for n in names
    )


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def active_subgoals(world: WorldKnowledge) -> list[ActiveSubgoal]:
    """Subgoals currently in the OPEN SET: 'active', 'blocked',
    or 'inferred_satisfied' (the discipline downgrade for
    achievement claims without a confirming signal).  Commit
    order (oldest first)."""
    return [sg for sg in world.active_subgoals
            if sg.status in ("active", "blocked",
                              "inferred_satisfied")]


def closed_subgoals(world: WorldKnowledge) -> list[ActiveSubgoal]:
    """Subgoals that are 'achieved' or 'abandoned' — kept for
    provenance, not surfaced in the default prompt."""
    return [sg for sg in world.active_subgoals
            if sg.status in ("achieved", "abandoned")]


# ---------------------------------------------------------------------------
# Surface for the strategy prompt
# ---------------------------------------------------------------------------


def format_active_subgoals_surface(
    world: WorldKnowledge,
) -> str:
    """Render currently-active subgoals as a strategy-prompt block.

    The block forces the actor to either continue an active subgoal
    consciously, close it, abandon it, or commit a new one.  When
    the active list is empty, the prompt encourages the actor to
    commit if it has a meaningful next-step goal — but does not
    require it."""
    actives = active_subgoals(world)
    if not actives:
        return (
            "  (no active subgoals committed.  If your next action "
            "is in service of a goal worth remembering across "
            "turns, COMMIT it via the strategy reply's "
            "`commit_subgoal` field.)"
        )

    lines: list[str] = []
    lines.append(
        f"  {len(actives)} ACTIVE SUBGOAL(s).  These are your "
        "OWN PRIOR-TURN COMMITMENTS.  Each turn you must "
        "consciously either (a) continue them, (b) mark them "
        "achieved, (c) mark them abandoned with a reason, or "
        "(d) mark them blocked.  Do NOT silently drift off a "
        "subgoal you committed to without updating its status."
    )
    for sg in actives:
        age = world.turn - sg.created_at_turn
        parent_str = (
            f", parent={sg.parent_id}"
            if sg.parent_id else ""
        )
        related_str = (
            f", applying subroutine={sg.related_subroutine_id}"
            if sg.related_subroutine_id else ""
        )
        lines.append("")
        lines.append(
            f"  SUBGOAL id={sg.subgoal_id!r}  status={sg.status!r}"
            f"  age={age} turn(s){parent_str}{related_str}"
        )
        lines.append(f"    Name:             {sg.name}")
        lines.append(f"    Problem solved:   {sg.problem_solved}")
        lines.append(f"    Expected outcome: {sg.expected_outcome}")
        if sg.derived_from:
            lines.append(f"    Derived from:     {sg.derived_from}")
        if sg.win_condition_hypothesis_id:
            lines.append(
                f"    Serves WC hyp:    {sg.win_condition_hypothesis_id}"
            )
        if sg.forward_simulation:
            lines.append(f"    Forward simulation:")
            for sline in sg.forward_simulation.split("\n"):
                lines.append(f"      {sline}")
        if sg.notes:
            lines.append(f"    Notes:")
            for nline in sg.notes.split("\n"):
                lines.append(f"      {nline}")
    # OR-group sibling-coverage surface.  When a subgoal has an
    # or_group tag and ANOTHER member of the same group has been
    # closed (achieved or abandoned), the actor has likely been
    # focused on one branch.  Render the full group status so the
    # actor must consider untried-but-still-possible alternatives
    # before treating the OR class as exhausted.  Game-agnostic.
    groups: dict[str, dict] = {}
    for sg in (world.active_subgoals or []):
        og = (sg.or_group or "").strip()
        if not og:
            continue
        bucket = groups.setdefault(og, {
            "active": [], "achieved": [], "abandoned": [],
            "blocked": [], "invalidated": [],
        })
        bucket.setdefault(sg.status, []).append(sg)
    if groups:
        lines.append("")
        lines.append(
            "  OR-GROUP COVERAGE — each group below names an "
            "ACHIEVE-ONE-OF set.  Failure of one branch does NOT "
            "prove the class is exhausted: before abandoning the "
            "PARENT goal, you must consider every UNTRIED member of "
            "the group.  If candidates that would individually "
            "advance the goal are MISSING from a group, COMMIT them "
            "as siblings (share `or_group`) and try them."
        )
        for og, buckets in sorted(groups.items()):
            line_parts = []
            for st in ("achieved", "active", "blocked",
                        "abandoned", "invalidated"):
                if buckets.get(st):
                    names = ", ".join(s.name for s in buckets[st])
                    line_parts.append(f"{st}=[{names}]")
            lines.append(f"    or_group={og!r}: " + "  ".join(line_parts))

    lines.append("")
    lines.append(
        "  HOW TO UPDATE: in your strategy reply, use "
        "`subgoal_status_update` to set { subgoal_id, status, "
        "notes } for any subgoal that should change status this "
        "turn.  Use `commit_subgoal` to author a new one; if it "
        "is a child of an existing subgoal set its parent_id.  "
        "Leave both fields null to continue all active subgoals "
        "unchanged."
    )
    lines.append(
        "  PURSUING_SUBGOAL_ID (MANDATORY when any subgoal is "
        "active): set `pursuing_subgoal_id` in your reply to the "
        "id of the subgoal whose progress THIS TURN's action "
        "serves.  This anti-drift discipline lets the substrate "
        "tag the action with the subgoal it's advancing.  If you "
        "are committing a NEW subgoal this turn, leave "
        "`pursuing_subgoal_id` null — the substrate will use the "
        "newly-committed id automatically."
    )
    return "\n".join(lines)
