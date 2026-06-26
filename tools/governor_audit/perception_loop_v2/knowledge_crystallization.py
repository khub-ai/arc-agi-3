"""Knowledge crystallization — make transferable knowledge captured in a
structured, checkable, role-keyed form, and surfaced to the reader as a
computed gap rather than re-derived prose.

See:
  - docs/SPEC_cumulative_learning_loop.md   § Crystallization (writer)
  - docs/SPEC_goal_grounding_and_state_diff.md § Substrate-computed
    goal gap (reader)

This module implements the testable, mechanical cores of the three
pieces, reading/writing the EXISTING stores (no new store):

  1. Win-condition as a checkable relation + per-turn goal gap.
     `evaluate_win_relation` / `format_goal_gap`.
  2. Event-triggered credit assignment: derive the win relation by
     contrasting scoring vs non-scoring transitions.
     `derive_win_condition`.
  3. Consolidation: dedup / promote / prune a lesson set (the
     mechanical part; fragment->higher-claim COMPOSITION is an
     actor/LLM step left as a hook).  `consolidate_lessons`.

Game-agnostic throughout: relations are over ROLES and identity keys
(shared appearance), never specific entity names, so a derived win
relation transfers across level- and game-variations.

Deliberately deferred (need a live game or an LLM, flagged inline):
  - wiring credit-assignment to fire on live score events in the driver
  - fragment->higher-claim composition in consolidation
"""
from __future__ import annotations

import re
from typing import Optional

from world_knowledge import WorldKnowledge, EntityRecord   # noqa: E402


# Roles that never participate in objective relations.
_INERT_ROLES = {"scenery", "decoration", "background"}

# Roles that are FIXED references / targets (read-outs the player does
# not move), as opposed to ACTABLE roles the player rearranges.  Used to
# orient an ordered_match relation: the actable role is the one to
# rearrange; the reference role is the fixed target to match.  Open set,
# game-agnostic (role names come from perception's open vocabulary).
_REFERENCE_ROLES = {"hud", "target", "reference", "state_report",
                    "indicator", "goal", "marker", "readout"}


def _is_reference_role(role: str) -> bool:
    return (role or "").lower() in _REFERENCE_ROLES


def _orient_roles(roleA: str, roleB: str) -> tuple[str, str]:
    """Return (actable, reference) so the actable role is first.  If
    exactly one role is a reference, the other is actable and goes
    first.  Otherwise the input order is preserved."""
    a_ref, b_ref = _is_reference_role(roleA), _is_reference_role(roleB)
    if b_ref and not a_ref:
        return roleA, roleB
    if a_ref and not b_ref:
        return roleB, roleA
    return roleA, roleB


# ---------------------------------------------------------------------------
# Identity + position helpers (role-keyed, game-agnostic)
# ---------------------------------------------------------------------------


def _identity_key(rec: EntityRecord) -> str:
    """A cross-entity identity used to put a collectable in correspondence
    with its matching target (e.g. block_green <-> hud_target_green).
    Heuristic, game-agnostic: the trailing token of the entity name
    (its colour/type suffix), falling back to its appearance.  This is
    the 'which item maps to which' the goal-grounding spec calls IDENTITY
    MAPPING.  Falls back to the entity's appearance when there is no
    suffix token."""
    name = rec.name or ""
    if "_" in name:
        return name.rsplit("_", 1)[-1]
    return rec.appearance or name


def _bbox_at_turn(rec: EntityRecord, turn: int) -> Optional[list]:
    last = None
    for (t, bb) in (rec.bbox_history or []):
        if t <= turn:
            last = bb
        else:
            break
    return last


def _centroid(bb: list) -> tuple:
    return ((bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0)


def _role_of(rec: EntityRecord) -> str:
    return (rec.current_role or "unknown").lower()


def _ordered_identities(world: WorldKnowledge, role: str, axis: str,
                          turn: Optional[int] = None) -> list[str]:
    """Identities of entities of `role`, ordered along `axis`
    ('col' = left-to-right, 'row' = top-to-bottom), at `turn`
    (default current)."""
    idx = 1 if axis == "col" else 0
    items = []
    for rec in world.entities.values():
        if _role_of(rec) != role:
            continue
        bb = (_bbox_at_turn(rec, turn) if turn is not None
              else rec.current_bbox)
        if bb is None:
            continue
        items.append((_centroid(bb)[idx], _identity_key(rec), rec.name))
    items.sort(key=lambda t: t[0])
    return [ident for _, ident, _ in items]


def _roles_present(world: WorldKnowledge) -> list[str]:
    roles = {}
    for rec in world.entities.values():
        r = _role_of(rec)
        if r in _INERT_ROLES or r == "agent":
            continue
        roles[r] = roles.get(r, 0) + 1
    # roles with >=2 members can carry an ordering
    return [r for r, n in roles.items() if n >= 2]


# ---------------------------------------------------------------------------
# 1. Win-condition relation evaluation + goal gap
# ---------------------------------------------------------------------------


def evaluate_win_relation(world: WorldKnowledge,
                            rel: dict,
                            turn: Optional[int] = None) -> dict:
    """Evaluate a checkable win-relation against the world state at
    `turn` (default current).  Returns:
      {satisfied: bool, gap: str, detail: {...}}

    Seed type 'ordered_match(roleA, roleB)': the entities of roleA, in
    `axis` order, must have the same identity sequence as the entities
    of roleB in that order — compared on the SHARED identity set so a
    target strip with more slots than live pieces still matches on the
    pieces that exist.  This is the lc=0 win ('blocks laid out in the
    HUD's left-to-right order')."""
    rtype = rel.get("type")
    if rtype == "ordered_match":
        roleA, roleB = rel.get("roles", [None, None])
        axis = rel.get("axis", "col")
        seqA = _ordered_identities(world, roleA, axis, turn)
        seqB = _ordered_identities(world, roleB, axis, turn)
        shared = set(seqA) & set(seqB)
        if len(shared) < 2:
            return {
                "satisfied": False,
                "gap": (f"cannot compare: fewer than 2 shared identities "
                        f"between {roleA} and {roleB} "
                        f"({sorted(shared)})"),
                "detail": {"seqA": seqA, "seqB": seqB},
            }
        subA = [i for i in seqA if i in shared]
        subB = [i for i in seqB if i in shared]
        satisfied = subA == subB
        # Orient the gap toward the ACTABLE role: you rearrange the
        # blocks, not the fixed HUD reference.  Detect which side is the
        # reference and phrase accordingly, regardless of stored order.
        act_role, ref_role = _orient_roles(roleA, roleB)
        act_seq = subA if act_role == roleA else subB
        ref_seq = subB if act_role == roleA else subA
        if satisfied:
            gap = (f"SATISFIED: {act_role} are in the same {axis}-order as "
                   f"{ref_role}: {act_seq}")
        else:
            gap = (f"ORDER MISMATCH: {act_role} order is {act_seq} but "
                   f"{ref_role} (the fixed target) order is {ref_seq}. "
                   f"Rearrange the {act_role} to match the {ref_role} order"
                   + ("" if act_seq[::-1] != ref_seq
                      else " (currently REVERSED)"))
        return {"satisfied": satisfied, "gap": gap,
                "detail": {"actable": act_role, "reference": ref_role,
                            "actable_seq": act_seq, "reference_seq": ref_seq,
                            "trigger": rel.get("trigger")}}

    if rtype == "ordered_completion":
        # Temporal-order win: the members of `role` must be COMPLETED
        # (activated — observed via a visual_event on the entity) in the
        # reference `axis` order.  This is sk48's actual win ("pierce the
        # blocks in HUD left-to-right order; a swatch whitens per pierce;
        # win when all are white"), which a static spatial `ordered_match`
        # cannot express.  Order-of-events, not arrangement.
        role = (rel.get("roles") or [None])[0]
        axis = rel.get("axis", "col")
        ordered = _ordered_member_identities(world, role, axis, turn)
        if len(ordered) < 1:
            return {"satisfied": False,
                    "gap": f"(no {role} members to complete)",
                    "detail": {}}
        completed, seq = _completed_identities(world, role, turn)
        done = [i for i in ordered if i in completed]
        remaining = [i for i in ordered if i not in completed]
        next_target = remaining[0] if remaining else None
        # Was completion so far IN the reference order?  "Order is the
        # gate, no skips": the k-th completion must be the k-th member in
        # reference order, i.e. the completed prefix is contiguous from
        # the start.  Completing a later member before an earlier one
        # (even monotonically, skipping a gap) is out of order.
        positions = {ident: k for k, ident in enumerate(ordered)}
        seq_pos = [positions[i] for i in seq if i in positions]
        in_order = seq_pos == list(range(len(seq_pos)))
        satisfied = not remaining
        if satisfied:
            gap = (f"SATISFIED: all {role} completed in {axis}-order "
                   f"({ordered}).")
        else:
            gap = (f"COMPLETE IN ORDER: {role} completed so far {done}; "
                   f"NEXT (must complete next, in {axis}-order): "
                   f"{next_target}; remaining {remaining}."
                   + ("" if in_order else
                      "  WARNING: completions so far are OUT OF ORDER — "
                      "the order is the gate; out-of-order attempts do "
                      "not take."))
        return {"satisfied": satisfied, "gap": gap,
                "detail": {"role": role, "ordered": ordered, "done": done,
                            "remaining": remaining, "next": next_target,
                            "in_order": in_order,
                            "trigger": rel.get("trigger")}}

    # Unknown relation type — honest, not a crash.
    return {"satisfied": False,
            "gap": f"(unknown win-relation type {rtype!r}; cannot evaluate)",
            "detail": {}}


def _ordered_member_identities(world: WorldKnowledge, role: str,
                                 axis: str,
                                 turn: Optional[int] = None) -> list[str]:
    """Identities of `role` entities ordered along `axis` (col=L→R,
    row=T→B) at `turn` (default current).  Same as _ordered_identities
    but named for the completion case."""
    return _ordered_identities(world, role, axis, turn)


def _completed_identities(world: WorldKnowledge, role: str,
                            turn: Optional[int] = None
                            ) -> tuple[set, list]:
    """Which members of `role` have been COMPLETED (activated) and in
    what order.  Completion is observed game-agnostically as a
    visual_event naming the entity (perception emits these for watched
    entities, e.g. a HUD swatch's internal pixel change).  Scans
    deltas_observed up to `turn`; returns (completed_identity_set,
    ordered_completion_sequence)."""
    role_names = {rec.name: _identity_key(rec)
                  for rec in world.entities.values()
                  if _role_of(rec) == role}
    deltas = [d for d in (getattr(world, "deltas_observed", None) or [])
              if turn is None or getattr(d, "to_turn", 0) <= turn]
    # RE-GROUND AT EACH LEVEL BOUNDARY.  Completion is per-round: a score
    # advance (or win-state change) ENDS a round (that delta's completion
    # is the round's last), and the NEXT round starts with nothing
    # completed.  The "current round" is the deltas AFTER the last
    # boundary IF any exist (a new level is underway); otherwise it is the
    # round that the last boundary just completed (we are AT the win, so
    # that round still reads SATISFIED).  Without this, a prior level's
    # completions summed across the whole trial read as SATISFIED in the
    # next level (gap #2: the goal-gap reported lc=0's "all hud completed"
    # as still satisfied at lc=1).
    boundaries = [i for i, d in enumerate(deltas) if _score_at(world, d)]
    live = turn is None
    playing = getattr(world, "win_state", "playing") == "playing"
    if (live and playing and boundaries
            and boundaries[-1] == len(deltas) - 1):
        # LIVE view exactly at a level-advance (the score rose on this very
        # delta and play continues): the NEXT level has already begun, so
        # its round is EMPTY — surface next=<first member>, never the
        # just-cleared level's completions.  This is the turn-59 edge:
        # without it, lc=2's first prompt read lc=1's done set
        # (['red','orange','blue'], next 'green') instead of an empty round
        # (next 'red'), mis-targeting the actor's first move.
        start = len(deltas)
    elif boundaries and boundaries[-1] != len(deltas) - 1:
        # deltas exist after the last boundary -> a new round is underway
        start = boundaries[-1] + 1
    elif len(boundaries) >= 2:
        # EXPLICIT-turn evaluation (derive_win_condition's contrastive
        # check) AT a boundary -> the round that completed = deltas after
        # the PREVIOUS boundary.  derive must SEE the completion that
        # scored to lock the relation, so the boundary turn reads the
        # just-completed round here (the live view above does not).
        start = boundaries[-2] + 1
    else:
        start = 0
    completed: set = set()
    seq: list = []
    for d in deltas[start:]:
        for ve in (getattr(d, "visual_events", None) or []):
            ent = ve.get("entity") if isinstance(ve, dict) else None
            if ent not in role_names:
                continue
            ident = role_names[ent]
            # HONOR DIRECTION (bidirectional completion).  visual_events emits
            # 'activated' when a completion marker is SET and 'reverted' when
            # it is REMOVED (e.g. a pierced/skewered block dragged off / undone
            # -- top-down board, no gravity: the hazard is un-skewering).
            # Treating every event as a completion latched the done-set and
            # made it blind to un-completion, so the drop-guard and goal-gap
            # could not see a block leave the skewer.  A bare event with no
            # direction (legacy) is treated as 'activated'.
            direction = ve.get("direction") if isinstance(ve, dict) else None
            if direction == "reverted":
                completed.discard(ident)
                if ident in seq:
                    seq.remove(ident)
            elif ident not in completed:
                completed.add(ident)
                seq.append(ident)
    return completed, seq


def describe_win_relation(rel: dict) -> str:
    """A human-readable objective sentence for a checkable win relation —
    used to populate the actor's ``game_purpose_guess`` (and the trace)
    from the SUBSTRATE-DERIVED relation, so the objective is concrete from
    the moment it is crystallized rather than blank until the VLM guesses.
    Game-agnostic: phrased over roles + axis order, no game vocabulary."""
    rtype = rel.get("type")
    roles = rel.get("roles") or []
    axis = rel.get("axis", "col")
    order = "left-to-right" if axis == "col" else "top-to-bottom"
    if rtype == "ordered_completion":
        role = roles[0] if roles else "the targets"
        return (f"Complete each {role} member in {order} ({axis}) order; "
                f"satisfied when all are completed in that order "
                f"(substrate-derived from score events).")
    if rtype == "ordered_match":
        act = roles[0] if roles else "the movable items"
        ref = roles[1] if len(roles) > 1 else "the reference"
        return (f"Arrange the {act} to match the {order} ({axis}) order of "
                f"the {ref} (substrate-derived from score events).")
    return (f"Achieve relation {rtype}({', '.join(roles)}) on {axis} "
            f"(substrate-derived).")


def game_type_from_relation(rel: dict) -> str:
    """A coarse genre label derived from the win-relation type."""
    rtype = rel.get("type")
    if rtype == "ordered_completion":
        return ("sequential-activation puzzle — complete targets in a fixed "
                "order")
    if rtype == "ordered_match":
        return "arrangement/ordering puzzle — match a reference order"
    return "puzzle"


def format_goal_gap(world: WorldKnowledge,
                     rel: Optional[dict] = None) -> str:
    """Strategy-prompt surface: the live goal gap.  If no relation is
    given, pull the highest-credence WinConditionHypothesis that has a
    `win_relation`.  Empty string when there is no checkable win
    relation yet (so the prompt isn't cluttered before one is derived)."""
    if rel is None:
        cands = [h for h in world.win_condition_hypotheses
                 if getattr(h, "win_relation", None)]
        if not cands:
            return ""
        cands.sort(key=lambda h: h.credence, reverse=True)
        rel = cands[0].win_relation
    res = evaluate_win_relation(world, rel)
    head = ("  GOAL (substrate-computed gap — the win condition is a "
            "checked relation, not a remembered sentence; close this "
            "gap):")
    trig = rel.get("trigger")
    trig_line = (f"\n    once the gap is closed, the win trigger is: {trig}"
                 if trig else "")
    return f"{head}\n    {res['gap']}{trig_line}"


# ---------------------------------------------------------------------------
# 2. Event-triggered credit assignment — derive the win relation
# ---------------------------------------------------------------------------


def _score_at(world: WorldKnowledge, delta) -> bool:
    return bool(getattr(delta, "score_increased", False)
                or getattr(delta, "win_state_changed", False))


def derive_win_condition(world: WorldKnowledge) -> Optional[dict]:
    """Contrast the PRE-state of advancing transitions against
    non-advancing ones and return the win-relation that holds before a
    score/lc/win advance and not otherwise.

    v1 candidate space: ordered_match(roleA, roleB) over every ordered
    pair of roles with >=2 members.  A candidate is accepted if it is
    satisfied in the pre-state of EVERY advancing transition and NOT
    satisfied in the pre-state of at least one non-advancing transition
    (it discriminates).  Returns the discriminating relation, or None.

    The pre-state of a transition to_turn=T is the state at turn T-1
    (positions before the action that advanced the score).
    """
    deltas = list(getattr(world, "deltas_observed", []) or [])
    if not deltas:
        return None
    advancing = [d for d in deltas if _score_at(world, d)]
    non_adv = [d for d in deltas if not _score_at(world, d)]
    if not advancing:
        return None

    roles = _roles_present(world)

    # 1. ordered_completion (TEMPORAL): a role whose members complete (via
    #    visual_events) in axis order, the LAST completion coinciding with
    #    the advance.  Tried first — it is the order-of-events win that a
    #    static arrangement match cannot express.  Criterion: the relation
    #    goes false->true exactly on an advancing transition (the
    #    completion that scored) and never on a non-advancing one.
    for role in roles:
        for axis in ("col", "row"):
            cand = {"type": "ordered_completion", "roles": [role],
                    "axis": axis}
            ok_adv = True
            for d in advancing:
                det = evaluate_win_relation(world, cand, turn=d.to_turn)["detail"]
                done = det.get("done") or []
                ordered = det.get("ordered") or []
                # Accept on IN-ORDER PREFIX completion reached BY the advance
                # — rather than demanding that EVERY member's completion be
                # detected, or that the completion land exactly on the
                # scoring frame.  Two real-data facts break the strict
                # criterion: (1) a tail member's completion can be missed by
                # the pixel-diff (green's swatch never registered in the sk48
                # trial); (2) the score-advance frame often lands a few turns
                # AFTER the last detected completion (so it adds no new
                # completion of its own).  Require, per round (re-grounded at
                # boundaries): completions are in order, form a contiguous
                # prefix from the first member (no skipped earlier member),
                # and name >= 2 members (an ORDER needs two).
                contiguous_prefix = (done == ordered[:len(done)])
                if not (det.get("in_order", True) and contiguous_prefix
                        and len(done) >= 2):
                    ok_adv = False
                    break
            if not ok_adv:
                continue
            # not falsely completed on a non-advancing transition
            bad = any(
                evaluate_win_relation(world, cand,
                                      turn=d.to_turn)["satisfied"]
                and not evaluate_win_relation(
                    world, cand,
                    turn=(d.from_turn if d.from_turn is not None
                          else d.to_turn - 1))["satisfied"]
                for d in non_adv
            )
            if not bad:
                cand["trigger"] = "complete all in order"
                return cand

    # 2. ordered_match (SPATIAL): arrangement of one role matches another's
    #    order, in place BEFORE the scoring action.
    candidates: list[dict] = []
    for i, ra in enumerate(roles):
        for rb in roles:
            if ra == rb:
                continue
            for axis in ("col", "row"):
                candidates.append({"type": "ordered_match",
                                    "roles": [ra, rb], "axis": axis})

    best: Optional[dict] = None
    for cand in candidates:
        # held before every advance?
        held_all_adv = True
        for d in advancing:
            pre = (d.from_turn if d.from_turn is not None
                   else d.to_turn - 1)
            if not evaluate_win_relation(world, cand, turn=pre)["satisfied"]:
                held_all_adv = False
                break
        if not held_all_adv:
            continue
        # discriminates: not held before at least one non-advance?
        discriminates = False
        for d in non_adv:
            pre = (d.from_turn if d.from_turn is not None
                   else d.to_turn - 1)
            if not evaluate_win_relation(world, cand, turn=pre)["satisfied"]:
                discriminates = True
                break
        if discriminates or not non_adv:
            best = cand
            break
    if best is not None and best.get("type") == "ordered_match":
        # Store the relation oriented actable-first so the goal gap reads
        # "rearrange the <actable> to match the <reference>".
        rA, rB = best["roles"]
        best["roles"] = list(_orient_roles(rA, rB))
    return best


def commit_derived_win_relation(world: WorldKnowledge,
                                  rel: dict,
                                  credence: float = 0.7) -> None:
    """Attach a derived win relation to a WinConditionHypothesis (new or
    matching).  Idempotent on an identical relation."""
    for h in world.win_condition_hypotheses:
        if getattr(h, "win_relation", None) == rel:
            h.credence = min(1.0, max(h.credence, credence))
            return
    from world_knowledge import WinConditionHypothesis  # noqa: E402
    desc = (f"derived win relation: {rel.get('type')}"
            f"({','.join(rel.get('roles', []))}) "
            f"axis={rel.get('axis')}")
    world.win_condition_hypotheses.append(WinConditionHypothesis(
        hypothesis_id=f"wc_derived_{rel.get('type')}_"
                       f"{'_'.join(rel.get('roles', []))}_{rel.get('axis')}",
        description=desc, credence=credence,
        created_at_turn=getattr(world, "turn", 0),
        win_relation=rel,
    ))


# ---------------------------------------------------------------------------
# 3. Consolidation — dedup / promote / prune (mechanical)
# ---------------------------------------------------------------------------


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())[:160]


# ---------------------------------------------------------------------------
# Defect detection — the patterns the manual KB audit (2026-06-01) had to
# catch by hand.  These are MECHANICAL flags (not auto-rewrites): a lesson
# that "reads fine but mis-executes / contradicts" is surfaced so a
# consolidation/actor pass can fix it.  Catching them automatically keeps
# the KB from silently accreting the same defect classes again.
# ---------------------------------------------------------------------------

# Role-level lessons (mechanic/technique/win_condition) should be keyed on
# ROLES + RELATIONS, not pixel/tick coordinates.  A literal coordinate baked
# into one is over-fit to the level it was learned on and mis-applies
# elsewhere (e.g. "extend tip to col41" was lc=0-specific and mis-aligned at
# lc=1).
_COORD_RE = re.compile(
    r"\bcols?\s*\d{1,3}(\s*[-–]\s*\d{1,3})?\b"   # "col41", "cols 27-30"
    r"|\brows?\s*\d{1,3}\b"                              # "row 12"
    r"|\(\s*\d{1,3}\s*,\s*\d{1,3}\s*\)",                # "(57, 42)"
    re.IGNORECASE,
)
_ROLE_KINDS = {"mechanic", "technique", "win_condition"}
_MANIP_VERBS = ("sweep", "swipe", "displace", "side-swipe", "slide")
_PRECOND_HINTS = ("offset", "clear lane", "clear row", "clear col",
                   "span", "perpendicular", "one row", "one col",
                   "from a clear", "before it", "across")
_ACTION_RE = re.compile(r"\bACTION\d\b")

# Absolute-vs-relative: a maneuver should be defined RELATIVELY (toward /
# across / perpendicular / offset / same_* / clearance / either side), not
# anchored on absolute directions ("move up", "extend right", "below->up").
# An absolute-only operation fails to generalise — it reuses the same
# direction regardless of the entities' relative layout, which is the root
# cause of the sk48 second-swipe failure (the actor reused "below->up"
# instead of choosing the offset side by relation).  Scoped to technique /
# subroutine maneuvers; mechanic action-facts ("ACTION1 = agent moves up")
# are legitimately absolute and exempt.
_ABS_OP_RE = re.compile(
    r"\b(?:move|extend|retract|sweep|swipe|push|slide|shift|go|drag)\s+"
    r"(?:the\s+\w+\s+){0,2}(?:up|down|left|right|upward[s]?|downward[s]?|"
    r"above|below)\b",
    re.IGNORECASE,
)
_REL_ANCHOR_RE = re.compile(
    r"\b(toward|towards|across|perpendicular|offset|relative|"
    r"away\s+from|either\s+side|opposite\s+side|same_row|same_col|"
    r"co_displacement|adjacent|clearance|into\s+(?:the\s+)?open)\b",
    re.IGNORECASE,
)


def _inert_claimed_actions(desc: str) -> set:
    """Actions an inert_action lesson says to NOT propose — EXCLUDING any
    explicitly excused nearby as 'NOT inert' / the UNDO action."""
    du = (desc or "").upper()
    claimed = set()
    for m in _ACTION_RE.finditer(du):
        a = m.group(0)
        window = du[max(0, m.start() - 55):m.start() + 55]
        # excuse actions the lesson names as the exception / as effective —
        # an inert_action lesson often also lists the EFFECTIVE actions for
        # contrast ("the only effective actions are ACTION1/2/3/4"); those
        # are not inert claims.
        if ("NOT INERT" in window or "UNDO" in window
                or "EFFECTIVE" in window):
            continue
        claimed.add(a)
    return claimed


def _scan_record_defects(text: str, kind: str, label: str = "") -> list[dict]:
    """Text-level defect scan shared by lesson and subroutine auditing:
      - hardcoded_coords: a role-level / subroutine record carrying a
        literal pixel/tick coordinate (over-fit; should be generalised to
        adjacency/relation).  Self-disclaiming records ("NOT a fixed
        col41 ... per-level") are excused.
      - precondition_gap (soft): a manipulation record naming a
        sweep/swipe/displace verb with no precondition hint — the omission
        that let the side-swipe be mis-executed.
    """
    out: list[dict] = []
    low = (text or "").lower()
    if kind in _ROLE_KINDS or kind == "subroutine":
        m = _COORD_RE.search(text or "")
        if m and "not a fixed" not in low and "per-level" not in low:
            d = {"type": "hardcoded_coords", "kind": kind,
                 "match": m.group(0), "snippet": (text or "")[:90]}
            if label:
                d["label"] = label
            out.append(d)
    if kind in ("technique", "mechanic", "subroutine"):
        if (any(v in low for v in _MANIP_VERBS)
                and not any(h in low for h in _PRECOND_HINTS)):
            d = {"type": "precondition_gap", "kind": kind,
                 "snippet": (text or "")[:90]}
            if label:
                d["label"] = label
            out.append(d)
    # absolute_operation: a maneuver anchored on absolute directions with no
    # relative anchor (technique/subroutine only — claims should be relative
    # by default).
    if kind in ("technique", "subroutine"):
        if _ABS_OP_RE.search(text or "") and not _REL_ANCHOR_RE.search(text or ""):
            d = {"type": "absolute_operation", "kind": kind,
                 "snippet": (text or "")[:90]}
            if label:
                d["label"] = label
            out.append(d)
    return out


def detect_lesson_defects(lessons: list[dict]) -> list[dict]:
    """Flag the 'reads-fine-but-mis-executes / contradicts' defect classes
    the manual audit found:
      - hardcoded_coords / precondition_gap (per-record text scan), plus
      - action_contradiction: an action declared INERT in an inert_action
        lesson AND asserted to have an EFFECT in a mechanic lesson.
    Returns a list of defect records; does NOT modify the lessons."""
    defects: list[dict] = []
    for les in lessons:
        defects.extend(_scan_record_defects(
            les.get("description", "") or "", les.get("kind") or ""))

    inert: set = set()
    effective: set = set()
    for les in lessons:
        desc = les.get("description", "") or ""
        if les.get("kind") == "inert_action":
            inert |= _inert_claimed_actions(desc)
        elif les.get("kind") == "mechanic" and ("effect=" in desc or "->" in desc):
            effective |= set(_ACTION_RE.findall(desc.upper()))
    for a in sorted(inert & effective):
        defects.append({
            "type": "action_contradiction", "action": a,
            "detail": (f"{a} is listed as inert (do-not-propose) AND asserted "
                       f"to have an effect by a mechanic lesson"),
        })

    return defects


def _subroutine_text(s) -> str:
    """Combined auditable text of a subroutine (dict or object): the prose
    description + problem_solved + the generalised relational steps."""
    def g(k):
        v = s.get(k) if isinstance(s, dict) else getattr(s, k, None)
        return v or ("" if k != "relational_steps" else [])
    steps = g("relational_steps")
    steps_txt = " ".join(steps) if isinstance(steps, (list, tuple)) else str(steps)
    return " ".join([str(g("description")), str(g("problem_solved")), steps_txt])


def detect_subroutine_defects(subroutines) -> list[dict]:
    """Same text-level audit as lessons, applied to subroutine records
    (the side-swipe technique itself lives here, not in lessons).  Scans
    description + problem_solved + relational_steps for hardcoded coords
    and manipulation-without-precondition gaps.  Each defect carries the
    subroutine's name as `label`.  Does NOT modify the subroutines."""
    defects: list[dict] = []
    for s in (subroutines or []):
        name = s.get("name") if isinstance(s, dict) else getattr(s, "name", "")
        defects.extend(_scan_record_defects(
            _subroutine_text(s), "subroutine", label=name or ""))
    return defects


def consolidate_lessons(lessons: list[dict],
                         per_kind_cap: int = 12,
                         prune_below: float = 0.15) -> tuple[list[dict], dict]:
    """Mechanical consolidation of a per-game lesson set:
      - merge near-duplicate (kind, normalized-description) entries,
        keeping the max credence and summing confirmation counts;
      - prune low-credence non-refuted entries below `prune_below`
        (refuted/avoid-list entries are kept regardless — negative
        evidence is cheap to keep and dangerous to lose);
      - cap each kind to its `per_kind_cap` highest-credence entries.

    Returns (consolidated_lessons, stats).  This is the mechanical half;
    composing fragments INTO higher-level claims needs an actor/LLM pass
    and is left as a hook (see SPEC_cumulative_learning_loop.md).
    """
    by_key: dict[tuple, dict] = {}
    merged = 0
    for les in lessons:
        kind = les.get("kind") or "free_form"
        key = (kind, _norm(les.get("description", "")))
        if key in by_key:
            keep = by_key[key]
            keep["credence"] = max(keep.get("credence", 0),
                                    les.get("credence", 0))
            keep["confirmations"] = (keep.get("confirmations", 1)
                                      + les.get("confirmations", 1))
            merged += 1
        else:
            by_key[key] = dict(les)

    survivors = list(by_key.values())

    # prune low-credence non-refuted noise
    pruned = []
    kept = []
    for les in survivors:
        kind = les.get("kind")
        if kind == "refuted":
            kept.append(les)
            continue
        if les.get("credence", 0) < prune_below:
            pruned.append(les)
        else:
            kept.append(les)

    # cap per kind
    from collections import defaultdict
    by_kind: dict[str, list] = defaultdict(list)
    for les in kept:
        by_kind[les.get("kind") or "free_form"].append(les)
    capped: list[dict] = []
    dropped_to_cap = 0
    for kind, items in by_kind.items():
        items.sort(key=lambda l: (l.get("credence", 0),
                                   l.get("confirmations", 1)), reverse=True)
        capped.extend(items[:per_kind_cap])
        dropped_to_cap += max(0, len(items) - per_kind_cap)

    # Surface latent defects in the consolidated set so a follow-up
    # (actor/LLM) pass can fix them, instead of letting the KB silently
    # re-accrue over-fit coords / contradictions / precondition gaps.
    defects = detect_lesson_defects(capped)

    stats = {
        "input": len(lessons),
        "after_dedup": len(survivors),
        "merged": merged,
        "pruned_low_credence": len(pruned),
        "dropped_to_cap": dropped_to_cap,
        "output": len(capped),
        "defects": defects,
        "n_defects": len(defects),
    }
    return capped, stats
