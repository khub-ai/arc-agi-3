"""Plan-consistency gate — stop a freshly-formed plan from re-entering a
REFUTED approach or contradicting a CONFIRMED invariant.

THE FAILURE THIS PREVENTS
-------------------------
A plan formed later in a run reverts to greedy / first-principles
reasoning and walks straight back into an approach already debunked, or
ignores an invariant already confirmed. Insights are established, then
left passive: nothing checks the plan you just wrote against them.

WHY IT LIVES HERE (dumb substrate, intelligence in the VLM)
-----------------------------------------------------------
The substrate cannot judge whether a plan is *good*. But it CAN do two
mechanical things reliably:
  1. SURFACE, at plan-formation, the refuted-approaches avoid-list + the
     confirmed invariants, with an explicit "check your plan against
     these" directive (preventive).
  2. After the plan is written, LEXICALLY flag when the plan's text
     overlaps a refuted approach on distinctive terms (detective).
The actual judgement ("does my plan really repeat this?") stays with the
VLM; the substrate just makes the established insights impossible to
ignore and flags the obvious repeats.

This mirrors the reply-validator (which checks a reply against tracked
entities / the latest delta) and plan-non-regression (which checks plan
steps against world state) — now extended from world-state to the
ACCUMULATED INSIGHTS. It is only as good as the recorded insights, so it
depends on those being first-class checkable records (refuted lessons +
consolidated invariants). See
memory/feedback_plan_must_consult_established_insights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Tokens that carry no mechanism meaning — dropped before overlap scoring
# so a match is driven by distinctive content (click, void, cross, ...),
# not by filler ("the", "a", "block", "move").
_STOP = {
    "the", "a", "an", "to", "of", "is", "it", "and", "by", "with", "on",
    "in", "at", "or", "be", "as", "for", "this", "that", "into", "onto",
    "from", "each", "all", "its", "via", "so", "then", "when", "if",
    # domain-generic words too common to be distinctive on their own
    "block", "blocks", "move", "moves", "moved", "action", "arm", "row",
    "col", "cols", "right", "left", "up", "down", "side", "level",
    "score", "play", "area", "agent", "head", "tip", "edge", "one", "two",
}


def _terms(text: str) -> set:
    """Distinctive content tokens of a free-form string (lowercased,
    de-stopworded, length>2)."""
    if not text:
        return set()
    cleaned = "".join(
        c if (c.isalnum() or c.isspace()) else " " for c in text.lower()
    )
    return {t for t in cleaned.split() if t not in _STOP and len(t) > 2}


@dataclass
class RefutedMatch:
    refuted_id: Optional[str]
    refuted_desc: str
    shared_terms: list
    overlap: int


def check_plan_against_refuted(
    plan_text: str,
    refuted: list,
    *,
    min_shared: int = 3,
) -> list:
    """Flag where a proposed plan lexically resembles a REFUTED approach.

    ``plan_text``: the plan's free-form text (rationale + testing
    hypothesis + action names concatenated).
    ``refuted``: list of dicts {"lesson_id", "description"} for refuted
    approaches (e.g. per_game_lessons kind='refuted').

    A match is reported when the plan shares at least ``min_shared``
    distinctive content terms with a refuted approach — conservative on
    purpose (favour precision; a missed flag is cushioned by the
    always-on preventive surface). Reports the shared terms so the VLM
    can judge whether it is truly repeating the refuted approach.
    Returns matches sorted by overlap desc.
    """
    pt = _terms(plan_text)
    if not pt:
        return []
    out: list = []
    for r in refuted or []:
        desc = r.get("description", "") if isinstance(r, dict) else str(r)
        rid = r.get("lesson_id") if isinstance(r, dict) else None
        shared = pt & _terms(desc)
        if len(shared) >= min_shared:
            out.append(RefutedMatch(
                refuted_id=rid,
                refuted_desc=desc,
                shared_terms=sorted(shared),
                overlap=len(shared),
            ))
    out.sort(key=lambda m: -m.overlap)
    return out


def plan_text_of(reply: dict) -> str:
    """Concatenate the parts of a strategy reply that express the PLAN's
    intent, for lexical checking. Tolerant of missing keys."""
    if not isinstance(reply, dict):
        return str(reply or "")
    parts = [
        str(reply.get("rationale") or ""),
        str(reply.get("testing_hypothesis") or ""),
        str(reply.get("game_purpose") or ""),
    ]
    seq = reply.get("planned_action_sequence")
    if isinstance(seq, list):
        parts.append(" ".join(str(a) for a in seq))
    ea = reply.get("endorsed_action")
    if ea:
        parts.append(str(ea))
    return "  ".join(p for p in parts if p)


def format_plan_gate(refuted: list, invariants: list) -> str:
    """The preventive PLAN-GATE directive surfaced AT plan-formation.

    Short on purpose: the refuted list + invariants are already surfaced
    in the lessons block, so this does not re-list them — it makes the
    CHECK mandatory. The DEAD-END DISCIPLINE always shows (it guards against
    pruning on a guess, the costliest planning error).
    """
    lines = [
        "PLAN GATE (check BEFORE you commit a plan):",
        # Verify-before-gating: an impossibility / dead-end may PRUNE a plan
        # only when OBSERVED. A dead-end reasoned from a model of the mechanics
        # is a low-credence GUESS (see claim_credence): it never out-credences
        # an observation and must be PROBED first, not asserted as fact.
        "  DEAD-END DISCIPLINE: do NOT prune a plan or call something "
        "impossible from REASONING about the mechanics — that is a low-credence "
        "GUESS. Only an OBSERVED result may gate. Before abandoning a branch on "
        "an unverified impossibility, run a cheap PROBE and observe; if it "
        "can't be observed yet, the verdict is 'not found', NOT 'impossible'.",
    ]
    if not refuted and not invariants:
        return "\n".join(lines)
    lines += [
        "  Your plan must NOT repeat any REFUTED approach and must NOT "
        "contradict any CONFIRMED invariant (both listed in the prior-"
        "knowledge block above). Re-entering a cleared dead-end or "
        "ignoring an established insight is the single most common "
        "planning failure here.",
        "  Before committing: (1) name the established insights/invariants "
        "relevant to this situation; (2) confirm each plan step is "
        "consistent with them; (3) if a step repeats a refuted approach, "
        "discard it and replan.",
    ]
    if refuted:
        lines.append(f"  ({len(refuted)} refuted approach(es) on the "
                     f"avoid-list — do not re-propose them.)")
    if invariants:
        lines.append(f"  ({len(invariants)} confirmed invariant(s) — do "
                     f"not contradict or re-doubt them.)")
    return "\n".join(lines)


def format_refuted_match_warning(matches: list) -> str:
    """Detective warning, surfaced AFTER a plan is written when it
    lexically overlaps a refuted approach. Names the specific match so
    the VLM reconciles or revises. "" when nothing flagged."""
    if not matches:
        return ""
    lines = [
        "PLAN-CONSISTENCY WARNING: your plan resembles approach(es) "
        "already REFUTED here. Reconcile or revise before committing:",
    ]
    for m in matches[:4]:
        lines.append(f"  - shares {m.shared_terms} with refuted: "
                     f"\"{m.refuted_desc[:160]}\"")
    lines.append("  If your plan genuinely differs, state HOW it differs "
                 "from the refuted approach; otherwise replan.")
    return "\n".join(lines)


# ===========================================================================
# POSITIVE HALF — enforce CONFIRMED strategies (use-this list)
# ===========================================================================
# The negative half above stops a plan re-entering a REFUTED approach. This
# half stops the symmetric, equally-recurring failure: reverting to a naive
# DEFAULT action when a CONFIRMED strategy exists for that sub-goal (e.g.
# reaching for `push` to move a free block when the confirmed operator is
# `body-sweep` -- push re-impales). The naive action is usually the first
# thing that surfaces; the confirmed-but-non-obvious operator is not, so the
# substrate must surface it keyed to the sub-goal AND flag a plan that uses
# the naive action instead. See memory/feedback_plan_must_consult_established_insights
# (positive direction) + the replay-verification gate (operator_verification):
# only REPLAY-CONFIRMED strategies are enforced; hypotheses are merely offered.


@dataclass
class ConfirmedStrategy:
    """A replay-confirmed way to achieve a sub-goal effect, plus the naive
    action(s) known to FAIL for it."""
    effect: str            # sub-goal effect keywords, e.g. "move free block up"
    operator: str          # the confirmed operator, e.g. "body-sweep"
    avoid_actions: list    # naive actions that fail here, e.g. ["push"]
    note: str = ""         # why the naive one fails, e.g. "push re-impales"


@dataclass
class MissingStrategyWarning:
    effect: str
    confirmed_operator: str
    naive_used: list
    note: str


def check_plan_uses_confirmed(plan_text: str, confirmed: list) -> list:
    """Flag where a plan reaches for a naive action while a CONFIRMED
    operator exists for the sub-goal it is pursuing.

    For each confirmed strategy whose effect keywords appear in the plan
    text: if the plan mentions one of the strategy's avoid_actions AND does
    NOT mention the confirmed operator, flag it -- the plan is using the
    naive default where a confirmed strategy applies. Conservative: only
    flags when the effect is clearly in play and a naive action is named.
    Returns warnings sorted by effect.
    """
    pt = _terms(plan_text)
    low = (plan_text or "").lower()
    out: list = []
    for s in confirmed or []:
        eff_terms = _terms(s.effect)
        if not eff_terms or not (eff_terms & pt):
            continue                         # this sub-goal isn't in the plan
        op = (s.operator or "").lower()
        if op and op.replace("-", " ") in low.replace("-", " "):
            continue                         # plan already uses the confirmed op
        naive_used = [a for a in (s.avoid_actions or [])
                      if a.lower() in low]
        if naive_used:
            out.append(MissingStrategyWarning(
                effect=s.effect, confirmed_operator=s.operator,
                naive_used=naive_used, note=s.note))
    out.sort(key=lambda w: w.effect)
    return out


def format_confirmed_strategy_surface(confirmed: list) -> str:
    """The use-this surface at plan-formation: confirmed operators keyed to
    their sub-goal effect, so the actor applies them instead of defaulting
    to the naive action. Empty when there is nothing confirmed yet."""
    if not confirmed:
        return ""
    lines = [
        "CONFIRMED STRATEGIES (apply these for the matching sub-goal -- the "
        "naive first action is usually NOT the confirmed one):",
    ]
    for s in confirmed:
        avoid = (f" (NOT {', '.join(s.avoid_actions)}"
                 + (f": {s.note}" if s.note else "") + ")") if s.avoid_actions else ""
        lines.append(f"  - to {s.effect}: use {s.operator}{avoid}")
    return "\n".join(lines)


def format_missing_strategy_warning(warnings: list) -> str:
    """Detective warning when a plan used a naive action where a confirmed
    strategy exists. "" when nothing flagged."""
    if not warnings:
        return ""
    lines = [
        "PLAN-CONSISTENCY WARNING: your plan uses a naive action where a "
        "CONFIRMED strategy exists. Apply the confirmed operator or justify "
        "the deviation:",
    ]
    for w in warnings[:4]:
        note = f" ({w.note})" if w.note else ""
        lines.append(f"  - to {w.effect}: you used {w.naive_used}, but the "
                     f"confirmed strategy is {w.confirmed_operator}{note}")
    return "\n".join(lines)


# ===========================================================================
# STRUCTURAL HALF — geometric invariants + agent kinematics priors
# ===========================================================================
# The two halves above (negative refuted-text + positive confirmed-strategy)
# catch plans that resemble prior textual failures or skip a known operator.
# They DO NOT catch plans whose STRUCTURE is physically impossible — e.g.
# "extend the agent body across columns occupied by an impassable wall, then
# bend it down on the other side." That failure mode bit us live on
# sk48 lc=4 (turn 197-213): the central wall was not tracked as an entity at
# all, so neither the negative nor positive gate had anything to match.
#
# This half adds a STRUCTURAL gate keyed on geometric facts:
#   (1) ImpassableObstacle — bbox of any entity with role 'impassable_obstacle'
#       (or 'wall' + static). No moving entity may overlap it at any time.
#   (2) AgentKinematics — declarative prior for a spatial agent: which
#       actions move which state vars by how much, AND how to compute the
#       agent's occupied region (body bbox) from its state. The body-region
#       computation is INJECTABLE (`body_region_fn`) so different agent
#       topologies plug in without subclassing: a rail-mounted extendable
#       rod (sk48) uses the default head+tip span; a single-cell avatar
#       (ls20) uses an identity region around head; a multi-cell ship uses
#       a fixed footprint translated by head_row/head_col; etc.
# Both are surfaced into the prompt as STRUCTURAL INVARIANTS so the actor
# self-gates against them. The simulator that walks a planned_action_sequence
# through the prior and flags bbox-overlap violations lives in
# `simulate_plan_against_structure` below — a one-pass projector, no search.
# Scope: assumes a 2D grid-cell world with discrete actions (true across
# ARC-AGI-3). 3D / continuous-time generalization is out of scope here;
# see audit notes in the COS-gap analysis chapter.
# See SPEC_visual_reasoning_substrate.md + memory/feedback_plan_must_consult_established_insights.


@dataclass
class ImpassableObstacle:
    """A static bbox no moving entity may overlap at any planning step.

    `bbox` is (row_min, col_min, row_max, col_max) inclusive in CELL units.
    `name` is the entity name to cite in violations."""
    name: str
    bbox: tuple   # (row_min, col_min, row_max, col_max)
    note: str = ""


def _default_extendable_rod_body_region(state: dict) -> tuple:
    """Default body-region computation for the ARC-AGI-3 'rail-mounted
    extendable rod' agent topology (sk48, similar games).

    Body is a rigid horizontal segment at head_row spanning head_col..tip_col.
    Returns (row_min, col_min, row_max, col_max) cell-inclusive bbox.

    For other topologies, pass a different callable to `AgentKinematics.
    body_region_fn` — e.g. an identity-cell function for a 1x1 avatar, or
    a fixed-footprint function for a multi-cell ship.
    """
    hr = int(state.get("head_row", 0))
    hc = int(state.get("head_col", 0))
    tc = int(state.get("tip_col", hc))
    return (hr, min(hc, tc), hr, max(hc, tc))


@dataclass
class AgentKinematics:
    """Declarative prior for a spatial agent. Game-agnostic shape:

    `state_action_deltas`: nested dict {state_var_name: {action: int_delta}}.
        Example for sk48-style rail+rod agent:
            {"head_row": {"ACTION1": -1, "ACTION2": +1},
             "tip_col":  {"ACTION4": +1, "ACTION3": -1}}.
        For a 4-directional 1x1 avatar (ls20-style):
            {"head_row": {"UP": -1, "DOWN": +1},
             "head_col": {"LEFT": -1, "RIGHT": +1}}.
    `body_region_fn`: callable(state_dict) -> (r0, c0, r1, c1) bbox in
        cells. The simulator calls this to get the agent's occupied region
        after each step's projected state. Default is the extendable-rod
        topology used by sk48; replace for other agent shapes.
    `body_constraint_text`: free-form short text describing the topology
        constraint (e.g. 'rigid horizontal rod, body row == head row'),
        surfaced in the STRUCTURAL INVARIANTS prompt block. Domain-neutral
        rendering: do not include game-specific entity names here.
    `note`: optional extra surface-text line (e.g. 'head_col fixed at
        rail col 8').
    """
    state_action_deltas: dict
    body_region_fn: callable = _default_extendable_rod_body_region
    body_constraint_text: str = (
        "rigid horizontal segment, body row == head row at every step"
    )
    note: str = ""


# Backwards-compat alias — old code passes `RigidBodyKinematics(...)`. The
# previous shape (head_action_row_delta + tip_action_col_delta +
# body_row_eq_head_row) is now expressed via `state_action_deltas` +
# `body_region_fn`. New code should use `AgentKinematics` directly.
@dataclass
class RigidBodyKinematics:
    """DEPRECATED — use AgentKinematics. Retained as a thin adapter so old
    call sites still work; this shape only models the sk48 rail+rod
    topology."""
    head_action_row_delta: dict
    tip_action_col_delta: dict
    body_row_eq_head_row: bool = True
    note: str = ""

    def to_agent_kinematics(self) -> AgentKinematics:
        return AgentKinematics(
            state_action_deltas={
                "head_row": dict(self.head_action_row_delta or {}),
                "tip_col":  dict(self.tip_action_col_delta or {}),
            },
            body_region_fn=_default_extendable_rod_body_region,
            body_constraint_text=(
                "rigid horizontal segment, body row == head row at every step"
                if self.body_row_eq_head_row else
                "agent body topology (constraint not declared)"
            ),
            note=self.note,
        )


@dataclass
class StructuralViolation:
    step_index: int            # 0-based index into the projected sequence
    action: str
    violated: str              # what invariant was violated
    overlap_with: Optional[str]
    projected_state: dict      # head_row, head_col, tip_col, body_bbox
    detail: str = ""


def _bbox_overlap(a: tuple, b: tuple) -> bool:
    """True if two (row_min, col_min, row_max, col_max) bboxes overlap
    (inclusive, integer cells)."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def simulate_plan_against_structure(
    plan_actions: list,
    *,
    initial_state: dict,
    kinematics,                       # AgentKinematics | RigidBodyKinematics
    obstacles: list,
) -> list:
    """One-pass projector. Walks ``plan_actions`` through ``kinematics``
    starting from ``initial_state`` and emits a StructuralViolation for
    each step whose projected agent-body bbox overlaps any
    ImpassableObstacle.

    ``kinematics`` may be an ``AgentKinematics`` (new, game-agnostic) or
    a ``RigidBodyKinematics`` (legacy adapter, auto-converted). The
    simulator only uses the AgentKinematics API:
      - ``state_action_deltas``: per-action int deltas to state vars
      - ``body_region_fn(state) -> bbox``: occupied region after the step
    Unknown actions are treated as no-ops. The simulator catches the FIRST
    violating step per obstacle so a partial plan can still be salvaged.

    Returns an empty list when the plan is structurally feasible.
    """
    # Accept legacy RigidBodyKinematics by adapting at the call site.
    kin = (kinematics.to_agent_kinematics()
           if isinstance(kinematics, RigidBodyKinematics) else kinematics)
    state = dict(initial_state)
    violations: list = []
    for i, action in enumerate(plan_actions or []):
        a = str(action).upper()
        # Apply state deltas — for every (state_var, action_table) entry,
        # add the per-action delta (0 if action not in the table).
        for var, table in (kin.state_action_deltas or {}).items():
            d = int((table or {}).get(a, 0))
            if d:
                state[var] = int(state.get(var, 0)) + d
        body_bbox = kin.body_region_fn(state)
        for obs in obstacles or []:
            if _bbox_overlap(body_bbox, obs.bbox):
                violations.append(StructuralViolation(
                    step_index=i, action=a,
                    violated=("agent body would overlap impassable "
                             f"obstacle {obs.name}"),
                    overlap_with=obs.name,
                    projected_state={**state, "body_bbox": body_bbox},
                    detail=(f"step {i+1}/{len(plan_actions)} {a}: body "
                            f"projected to bbox={body_bbox} intersects "
                            f"{obs.name}.bbox={obs.bbox}"),
                ))
                break
    return violations


def format_structural_invariants(
    obstacles: list, kinematics=None,
) -> str:
    """The use-this STRUCTURAL surface at plan-formation: lists impassable
    obstacles by bbox + the agent kinematics prior, with an explicit
    "your plan must not cause the body to overlap any of these" directive.
    Returns "" when nothing structural is registered.

    ``kinematics`` accepts AgentKinematics (preferred) or legacy
    RigidBodyKinematics (auto-adapted). Surface text is domain-neutral —
    no game-specific entity names baked in; constraint description comes
    from the kinematics object's ``body_constraint_text``.
    """
    if not obstacles and not kinematics:
        return ""
    # Normalize to AgentKinematics
    kin = None
    if kinematics is not None:
        kin = (kinematics.to_agent_kinematics()
               if isinstance(kinematics, RigidBodyKinematics) else kinematics)
    lines = [
        "STRUCTURAL INVARIANTS (geometric facts your plan MUST respect — "
        "trajectory-checkable, not just textual):",
    ]
    if obstacles:
        lines.append("  IMPASSABLE OBSTACLES (no moving entity — including "
                     "the agent body — may overlap these bboxes at ANY "
                     "step of a plan):")
        for o in obstacles:
            r0, c0, r1, c1 = o.bbox
            note = f" — {o.note}" if o.note else ""
            lines.append(f"    - {o.name}: rows {r0}-{r1} cols {c0}-{c1}"
                         f"{note}")
    if kin:
        lines.append("  AGENT KINEMATICS PRIOR (how the agent's body moves "
                     "under each action — project your plan through this "
                     "before committing):")
        if kin.body_constraint_text:
            lines.append(f"    - body topology: {kin.body_constraint_text}")
        for var, table in (kin.state_action_deltas or {}).items():
            if not table:
                continue
            entries = ", ".join(
                f"{a}->{var}{('+%d'%d) if d>=0 else d}"
                for a, d in table.items() if d
            )
            if entries:
                lines.append(f"    - {var} deltas: {entries}.")
        if kin.note:
            lines.append(f"    - note: {kin.note}")
    lines.append("  Before committing a planned_action_sequence: PROJECT "
                 "the agent body bbox forward step-by-step using the "
                 "kinematics prior, and confirm NO step overlaps any "
                 "impassable obstacle. If the projection passes through "
                 "an obstacle bbox, the plan is REFUTED by geometry — "
                 "replan via a route that keeps the body out of the obstacle "
                 "bbox at every step.")
    return "\n".join(lines)


def format_structural_violations(violations: list) -> str:
    """Detective warning, surfaced AFTER a plan is written when the
    projector flagged a step that overlaps an impassable obstacle.
    "" when the plan is structurally feasible."""
    if not violations:
        return ""
    lines = [
        "PLAN-CONSISTENCY WARNING: your planned_action_sequence projects "
        "the agent body through an IMPASSABLE obstacle. The plan is "
        "structurally infeasible — revise the trajectory:",
    ]
    for v in violations[:4]:
        lines.append(f"  - {v.detail}")
    lines.append("  Use the STRUCTURAL INVARIANTS surface above; route the "
                 "body around the obstacle bbox in EVERY projected step.")
    lines.append("  HINT — when DIRECT reach to a target is gate-refuted, "
                 "the reflection-move 'act through an intermediary' is "
                 "the FIRST thing to try: see the MEDIATION CANDIDATES "
                 "surface (if populated below) — the substrate detects "
                 "(kind, intermediary, target, agent-setup) tuples for "
                 "each registered mediation kind (vertical_lift, "
                 "horizontal_push, ...) and names them concretely so you "
                 "do not have to re-derive the geometry.")
    return "\n".join(lines)


# ===========================================================================
# MEDIATION CANDIDATE DETECTORS (act-through-intermediary, instantiated)
# ===========================================================================
# The reflection-move library (reflection_moves.py) has "act through an
# intermediary" as an abstract reframe surfaced when the actor is stuck.
# That abstract surface failed live on sk48 lc=4: even with it visible, no
# VLM picked it because the move did not say WHICH intermediary, WHICH
# target, or WHERE the agent must go. The actor kept reverting to direct
# reach. The detectors below close that gap: each scans the world for
# concrete (intermediary, target, agent_setup) triples for a particular
# mediation KIND, and `detect_mediation_candidates` unions them into a
# single ranked list. The reasoning ("does this mechanic actually fire
# here") stays with the VLM; the substrate names the candidates so the
# VLM evaluates concrete plans instead of inventing the pattern.
#
# Game-agnostic across ARC-AGI-3:
#   - VERTICAL_LIFT: free intermediary in target's column BELOW target;
#     agent rises through intermediary's row, target co-displaces upward.
#     (sk48 lc=4 mechanic; also lift-stack puzzles generally.)
#   - HORIZONTAL_PUSH: free intermediary in target's row, between agent
#     side and target; agent pushes laterally, intermediary slides into
#     target. (sokoban-class chains; sk48 lc=0/1 with skewer absorbing
#     a free block on the path to the wall.)
# Pattern is generic — wherever a free block sits between an agent and a
# goal block on a shared axis with clear travel, the corresponding
# mediation kind applies. Domain assumptions: 2D grid-cell world,
# discrete actions, role taxonomy of {agent, collectable, obstacle, ...}.
# Domain defaults (agent/target/intermediary role names, axis convention)
# are module-level constants below — override at the call site for other
# domains.


# ARC-AGI-3 DOMAIN DEFAULTS — override at call site for other domains.
# These role names + axis convention are shared across ARC games; tracked
# here so callers don't have to repeat them.
ARC_DEFAULT_AGENT_NAME            = "manipulator_head"
ARC_DEFAULT_TARGET_ROLES          = ("collectable",)
ARC_DEFAULT_INTERMEDIARY_ROLES    = ("collectable",)
# In ARC-AGI-3, larger row index = visually lower (row-down convention).
ARC_ROW_DOWN_CONVENTION           = True


@dataclass
class MediationCandidate:
    """A concrete (intermediary, target, agent_setup) triple the substrate
    detected as a candidate for an act-through-intermediary mechanic.

    `mediation_kind`: which mediation pattern this is — one of
        'vertical_lift' (push up the column stack),
        'horizontal_push' (slide along a row into target),
        and (future) 'drop_from_above', 'chain_of_intermediaries', ...
        The kind tells the actor which mechanic to evaluate.
    `target`: entity name to displace (direct reach gate-refused).
    `intermediary`: entity name of the free block between agent and target.
    `target_bbox`, `intermediary_bbox`: cell-inclusive bboxes.
    `agent_setup`: dict describing where the agent body must be to engage
        — keys depend on `mediation_kind`. For vertical_lift:
            {'body_min_row': int, 'body_col_span': (int, int)}
        For horizontal_push:
            {'body_row': int, 'body_min_col': int} (or 'body_max_col' on
            right-of-target pushes).
    `engage_action_hint`: the action that typically propagates the
        mediation (domain-default; e.g. ACTION1 for vertical_lift in
        ARC-AGI-3). Hint, not authority — the kinematics may not bind
        actions this way.
    `note`: short rationale.
    """
    mediation_kind: str
    target: str
    target_bbox: tuple
    intermediary: str
    intermediary_bbox: tuple
    agent_setup: dict
    engage_action_hint: str
    note: str = ""


# Legacy alias for any external callers that imported the old name. New
# code uses MediationCandidate directly.
IndirectPushCandidate = MediationCandidate


def _bbox_same_col(a: tuple, b: tuple) -> bool:
    """True if two bboxes have overlapping COLUMN spans (axis-aligned)."""
    _, ac0, _, ac1 = a
    _, bc0, _, bc1 = b
    return not (ac1 < bc0 or bc1 < ac0)


def _bbox_below(a: tuple, b: tuple) -> bool:
    """True if bbox a is entirely BELOW bbox b (a.row_min > b.row_max).
    Cells with larger row index are visually lower on the playfield."""
    return a[0] > b[2]


# Small entity-dict helpers (shared across all detectors below). Kept
# inline so each detector reads top-to-bottom without indirection chasing.
def _bbox_of(e: dict) -> Optional[tuple]:
    """Normalize an entity dict's bbox to (r0,c0,r1,c1) cell units.
    Prefer ``bbox_cells``; fall back to ``bbox_ticks_turn1 // 4`` (sk48
    fallback while perception standardizes on cells)."""
    bc = e.get("bbox_cells")
    if isinstance(bc, (list, tuple)) and len(bc) == 4:
        return tuple(int(x) for x in bc)
    bt = e.get("bbox_ticks_turn1")
    if isinstance(bt, (list, tuple)) and len(bt) == 4:
        return tuple(int(x) // 4 for x in bt)
    return None


def _role_of(e: dict) -> str:
    return str(e.get("role_hypothesis") or e.get("role") or "").lower()


def _name_of(e: dict) -> str:
    return str(e.get("name") or "")


# ===========================================================================
# ATTACHMENT CLASSIFIER — is a block FREE or ATTACHED to the agent?
# ===========================================================================
# The single perceptual fact that gates the indirect-push: a block can only
# act as a lift intermediary if it is FREE (decoupled), not impaled/carried
# on the agent. Pixel contiguity is unreliable (a rendered rod cell can be
# missing for one frame — that artifact cost a whole debugging session: a
# block that LOOKED free by a one-cell gap was actually impaled). The robust
# signal is BEHAVIORAL: co_displacement. A block that moves with the agent
# when the agent moves is ATTACHED; one that stays put while the agent moves
# is FREE. This reads the substrate's already-computed co_displacement
# relations from the recent delta history — no new perception needed.

# Agent-side entity names whose motion defines "the agent moved" (and whose
# co_displacement with a block means that block is carried). ARC-AGI-3
# manipulator defaults; override per domain.
AGENT_BODY_NAMES = ("manipulator_head", "arm_body", "arm_shaft", "agent")


def _rel_get(r, k, default=None):
    return r.get(k, default) if isinstance(r, dict) else getattr(r, k, default)


def classify_attachment(world, *, block_roles=("collectable",),
                        agent_names=AGENT_BODY_NAMES, lookback=8) -> dict:
    """Classify each maneuverable block as 'attached' | 'free' | 'unknown'
    from the substrate's co_displacement history (behavioral, not pixel).

    For each block, scan the most recent ``lookback`` deltas newest-first;
    the first delta that carries decisive evidence wins:
      - co_displacement(agent_body, block) present  -> 'attached'
      - the agent moved that turn (it appears in some co_displacement or the
        delta records agent_moved) AND no co_displacement with the block
        -> 'free' (the block stayed while the agent moved)
    Falls back to 'unknown' when no turn in the window moved the agent.

    Returns {block_name: state}. Game-agnostic: roles + agent names are
    parameters. Tolerant of dict-or-object deltas/relations.
    """
    out: dict = {}
    ents = getattr(world, "entities", None) or {}
    block_names = []
    values = ents.values() if hasattr(ents, "values") else ents
    for rec in values:
        role = (getattr(rec, "current_role", None) or "").lower()
        if role in block_roles:
            block_names.append(getattr(rec, "name", None))
    block_names = [b for b in block_names if b]
    if not block_names:
        return out
    deltas = list(getattr(world, "deltas_observed", None) or [])[-lookback:]
    deltas.reverse()  # newest first
    for blk in block_names:
        state = "unknown"
        for dl in deltas:
            rels = _rel_get(dl, "relations", None) or []
            agent_moved = bool(_rel_get(dl, "agent_moved", False))
            codisp_with_block = False
            agent_in_codisp = False
            for r in rels:
                if _rel_get(r, "kind") != "co_displacement":
                    continue
                ents_in = list(_rel_get(r, "entities", []) or [])
                has_block = blk in ents_in
                has_agent = any(a in ents_in for a in agent_names)
                if has_agent:
                    agent_in_codisp = True
                if has_block and has_agent:
                    codisp_with_block = True
            if codisp_with_block:
                state = "attached"
                break
            if agent_moved or agent_in_codisp:
                # Agent moved this turn but the block didn't ride along.
                state = "free"
                break
        out[blk] = state
    return out


def format_attachment_surface(attachment: dict) -> str:
    """Render the attachment state so the actor (and the mediation detector)
    knows which blocks are FREE vs carried. Empty when nothing classified."""
    if not attachment:
        return ""
    known = {k: v for k, v in attachment.items() if v != "unknown"}
    if not known:
        return ""
    lines = ["BLOCK ATTACHMENT (behavioral, from co_displacement — which "
             "blocks are carried by the agent vs free):"]
    for name, st in sorted(known.items()):
        tag = ("ATTACHED (co-moves with the agent — cannot act as a free "
               "lift intermediary until decoupled)" if st == "attached"
               else "FREE (decoupled — can be lifted/swept independently)")
        lines.append(f"  - {name}: {tag}")
    return "\n".join(lines)


def _bbox_same_row(a: tuple, b: tuple) -> bool:
    """True if two bboxes have overlapping ROW spans."""
    ar0, _, ar1, _ = a
    br0, _, br1, _ = b
    return not (ar1 < br0 or br1 < ar0)


def detect_indirect_push_candidates(
    entities: list,
    *,
    target_roles: tuple = ARC_DEFAULT_TARGET_ROLES,
    intermediary_roles: tuple = ARC_DEFAULT_INTERMEDIARY_ROLES,
    agent_name: str = ARC_DEFAULT_AGENT_NAME,
    obstacles: Optional[list] = None,
    max_candidates: int = 4,
    engage_action_hint: str = "ACTION1",
) -> list:
    """VERTICAL_LIFT mediation detector. Scans ``entities`` for triples
    where an intermediary sits in a target's column BELOW the target with
    clear vertical air between them, so an agent rising through the
    intermediary's row would push the stack up.

    Returns a list of ``MediationCandidate`` with ``mediation_kind=
    'vertical_lift'``, sorted by smallest target-intermediary row gap
    (tightest stack first, most likely to push cleanly).

    Same-name aggregate tracks are skipped — if perception lumps multiple
    instances into one entity, the detector cannot disambiguate them.

    Defaults reflect ARC-AGI-3 conventions (collectable target +
    intermediary, manipulator_head agent, ACTION1 engages). Override at
    the call site for other domains.
    """
    out: list = []
    obs = list(obstacles or [])
    agent = next((e for e in entities if _name_of(e) == agent_name), None)
    if agent is None or _bbox_of(agent) is None:
        return []
    targets = [e for e in entities if _role_of(e) in target_roles]
    intermediaries = [e for e in entities
                      if _role_of(e) in intermediary_roles]
    for t in targets:
        tb = _bbox_of(t)
        if tb is None:
            continue
        for i in intermediaries:
            if i is t:
                continue
            ib = _bbox_of(i)
            if ib is None:
                continue
            if _name_of(i) == _name_of(t) and ib == tb:
                continue
            if not _bbox_same_col(ib, tb):
                continue
            if not _bbox_below(ib, tb):
                continue
            # No obstacle blocking the vertical channel between i and t
            # on shared columns.
            cmin = max(ib[1], tb[1])
            cmax = min(ib[3], tb[3])
            channel_r0 = tb[2] + 1
            channel_r1 = ib[0] - 1
            if channel_r0 <= channel_r1:
                channel_bbox = (channel_r0, cmin, channel_r1, cmax)
                if any(_bbox_overlap(channel_bbox, o.bbox) for o in obs):
                    continue
            agent_setup = {
                "body_min_row": ib[2] + 1,
                "body_col_span": (ib[1], ib[3]),
            }
            gap = ib[0] - tb[2] - 1
            note = (f"intermediary '{_name_of(i)}' sits in target "
                    f"'{_name_of(t)}'s column, {gap} cell(s) of clear "
                    f"vertical air between them — a rise action with body "
                    f"under '{_name_of(i)}' should propagate push up the "
                    f"stack")
            out.append(MediationCandidate(
                mediation_kind="vertical_lift",
                target=_name_of(t), target_bbox=tb,
                intermediary=_name_of(i), intermediary_bbox=ib,
                agent_setup=agent_setup,
                engage_action_hint=engage_action_hint,
                note=note,
            ))
    out.sort(key=lambda c: (c.intermediary_bbox[0] - c.target_bbox[2]))
    return out[:max_candidates]


def detect_horizontal_push_candidates(
    entities: list,
    *,
    target_roles: tuple = ARC_DEFAULT_TARGET_ROLES,
    intermediary_roles: tuple = ARC_DEFAULT_INTERMEDIARY_ROLES,
    agent_name: str = ARC_DEFAULT_AGENT_NAME,
    obstacles: Optional[list] = None,
    max_candidates: int = 4,
    engage_action_hint_right: str = "ACTION4",
    engage_action_hint_left: str = "ACTION3",
) -> list:
    """HORIZONTAL_PUSH mediation detector. Scans for triples where an
    intermediary sits in a target's ROW between the agent (or the agent's
    reach axis) and the target, so the agent pushing the intermediary
    along the row drives it into the target.

    Two sub-cases: intermediary LEFT of target (push right via
    ``engage_action_hint_right``) and intermediary RIGHT of target (push
    left via ``engage_action_hint_left``). The detector emits one
    candidate per sub-case that matches.

    Channel check: no impassable obstacle between intermediary and target
    on the shared rows (an obstacle in the gap would stop the slide).
    """
    out: list = []
    obs = list(obstacles or [])
    agent = next((e for e in entities if _name_of(e) == agent_name), None)
    if agent is None or _bbox_of(agent) is None:
        return []
    targets = [e for e in entities if _role_of(e) in target_roles]
    intermediaries = [e for e in entities
                      if _role_of(e) in intermediary_roles]
    for t in targets:
        tb = _bbox_of(t)
        if tb is None:
            continue
        for i in intermediaries:
            if i is t:
                continue
            ib = _bbox_of(i)
            if ib is None:
                continue
            if _name_of(i) == _name_of(t) and ib == tb:
                continue
            if not _bbox_same_row(ib, tb):
                continue
            # Intermediary LEFT of target (push right)
            if ib[3] < tb[1]:
                rmin = max(ib[0], tb[0])
                rmax = min(ib[2], tb[2])
                channel = (rmin, ib[3] + 1, rmax, tb[1] - 1)
                if channel[1] <= channel[3] and any(
                        _bbox_overlap(channel, o.bbox) for o in obs):
                    continue
                gap = tb[1] - ib[3] - 1
                note = (f"intermediary '{_name_of(i)}' sits in target "
                        f"'{_name_of(t)}'s row, {gap} cell(s) left of "
                        f"target — pushing intermediary right along the "
                        f"row should drive it into the target")
                out.append(MediationCandidate(
                    mediation_kind="horizontal_push",
                    target=_name_of(t), target_bbox=tb,
                    intermediary=_name_of(i), intermediary_bbox=ib,
                    agent_setup={
                        "body_row_span": (ib[0], ib[2]),
                        "body_max_col": ib[1] - 1,  # body must end just
                                                     # left of intermediary
                                                     # to push it
                    },
                    engage_action_hint=engage_action_hint_right,
                    note=note,
                ))
            # Intermediary RIGHT of target (push left)
            elif ib[1] > tb[3]:
                rmin = max(ib[0], tb[0])
                rmax = min(ib[2], tb[2])
                channel = (rmin, tb[3] + 1, rmax, ib[1] - 1)
                if channel[1] <= channel[3] and any(
                        _bbox_overlap(channel, o.bbox) for o in obs):
                    continue
                gap = ib[1] - tb[3] - 1
                note = (f"intermediary '{_name_of(i)}' sits in target "
                        f"'{_name_of(t)}'s row, {gap} cell(s) right of "
                        f"target — pushing intermediary left along the "
                        f"row should drive it into the target")
                out.append(MediationCandidate(
                    mediation_kind="horizontal_push",
                    target=_name_of(t), target_bbox=tb,
                    intermediary=_name_of(i), intermediary_bbox=ib,
                    agent_setup={
                        "body_row_span": (ib[0], ib[2]),
                        "body_min_col": ib[3] + 1,
                    },
                    engage_action_hint=engage_action_hint_left,
                    note=note,
                ))
    # Rank by smallest along-row gap
    def _row_gap(c: MediationCandidate) -> int:
        tb, ib = c.target_bbox, c.intermediary_bbox
        if ib[3] < tb[1]:
            return tb[1] - ib[3] - 1
        return ib[1] - tb[3] - 1
    out.sort(key=_row_gap)
    return out[:max_candidates]


def detect_mediation_candidates(
    entities: list,
    *,
    target_roles: tuple = ARC_DEFAULT_TARGET_ROLES,
    intermediary_roles: tuple = ARC_DEFAULT_INTERMEDIARY_ROLES,
    agent_name: str = ARC_DEFAULT_AGENT_NAME,
    obstacles: Optional[list] = None,
    max_per_kind: int = 4,
    attachment: Optional[dict] = None,
) -> list:
    """Unified entry point — runs every concrete mediation detector and
    returns the combined list of ``MediationCandidate`` (each tagged with
    its ``mediation_kind``). Detector list is intentionally short for v1
    (vertical_lift + horizontal_push); add new detectors here as they
    land. Each detector is responsible for its own ranking; the unified
    list is the concatenation, kind-grouped by call order.

    ``attachment`` (from classify_attachment): when provided, candidates
    whose intermediary is ATTACHED to the agent are filtered out — an
    impaled/carried block cannot serve as a FREE lift intermediary. This is
    the fix for the rod-rendering-gap trap: freeness is decided behaviorally
    (co_displacement), not by pixels."""
    kwargs = dict(
        target_roles=target_roles,
        intermediary_roles=intermediary_roles,
        agent_name=agent_name,
        obstacles=obstacles,
        max_candidates=max_per_kind,
    )
    cands = (
        detect_indirect_push_candidates(entities, **kwargs)
        + detect_horizontal_push_candidates(entities, **kwargs)
    )
    if attachment:
        cands = [c for c in cands
                 if attachment.get(c.intermediary, "free") != "attached"]
    return cands


def format_mediation_precondition(world, candidates: list, attachment: dict,
                                  *, target_roles=ARC_DEFAULT_TARGET_ROLES,
                                  intermediary_roles=ARC_DEFAULT_INTERMEDIARY_ROLES
                                  ) -> str:
    """Explicit TRIGGER line for the indirect-push precondition, so the
    actor is never left guessing whether the setup is ready.

    States one of:
      - MET: a free intermediary is under a target (N candidate(s)) — act.
      - NOT MET (attached): an intermediary is in position but still
        ATTACHED to the agent; it must be decoupled first.
      - NOT MET (no alignment): no intermediary is positioned under a target.
    Empty only when there are no maneuverable blocks at all."""
    if candidates:
        return ("INDIRECT-PUSH PRECONDITION: MET — "
                f"{len(candidates)} free intermediary/target pair(s) "
                "detected (see MEDIATION CANDIDATES below). Execute the "
                "lift; do not re-derive the geometry.")
    # No candidates — diagnose why, so the actor knows what to fix.
    attached = [k for k, v in (attachment or {}).items() if v == "attached"]
    if attached:
        return ("INDIRECT-PUSH PRECONDITION: NOT MET — intermediary "
                f"block(s) {attached} are ATTACHED to the agent "
                "(they co-move with it). An attached block cannot act as a "
                "free lift intermediary. NEXT GOAL: DECOUPLE one (pin its "
                "tip against a barrier on the retract side, then retract so "
                "it is left behind) BEFORE attempting the lift. The earlier "
                "'free' appearance from a one-cell pixel gap is NOT freeness "
                "— trust this co_displacement-based verdict.")
    # Blocks exist but none attached and no candidate => not aligned.
    return ("INDIRECT-PUSH PRECONDITION: NOT MET — no free intermediary is "
            "positioned directly under a target. NEXT GOAL: position a free "
            "block in a target's column, below it, with clear vertical air.")


def format_decomposition_directive(world, candidates: list, attachment: dict,
                                   *, pursued_goal: str = "") -> str:
    """AUTONOMOUS GOAL-CHAIN GENERATION directive.

    The substrate has just inferred (from the structural gate + mediation
    precondition + attachment) that the pursued goal is NOT directly
    achievable and WHAT enabling state is missing. This turns that
    inference into an explicit instruction to GROW THE GOAL FOREST: commit
    the missing enabling state as a subgoal, linked by ``depends_on`` to
    the goal it unblocks — and recurse if that subgoal is itself blocked.
    This is what lets COS *generate* the backward chain (pierce <- access
    <- lift <- stage) on its own instead of needing it hand-authored; the
    substrate supplies the trigger + grounded facts, the VLM supplies the
    means-ends decomposition (per SPEC_vlm_backward_reasoning.md). Empty
    when the precondition is already met (nothing to decompose).

    Game-agnostic: it names no game mechanics — only the generic missing
    enabling-relation derived from the substrate's own surfaces."""
    if candidates:
        return ""   # precondition met — direct action, no decomposition
    # Derive the missing enabling state in generic terms.
    attached = [k for k, v in (attachment or {}).items() if v == "attached"]
    if attached:
        missing = (f"a FREE (decoupled) intermediary — block(s) {attached} "
                   "are currently ATTACHED to the agent, so the enabling "
                   "subgoal is to DECOUPLE one")
    else:
        missing = ("a free intermediary POSITIONED under a target — the "
                   "enabling subgoal is to move a free block into a "
                   "target's column, below it")
    goal_txt = (f" '{pursued_goal}'" if pursued_goal else " the current goal")
    return (
        "GOAL DECOMPOSITION DIRECTIVE (grow the goal forest — do not just "
        "act greedily): the substrate has inferred that" + goal_txt + " is "
        "NOT directly achievable and that the missing enabling state is "
        + missing + ".\n"
        "  Therefore: COMMIT that enabling state as a NEW subgoal "
        "(commit_subgoal) and set its parent's depends_on to it, so the "
        "goal forest records the chain and the parent re-activates when the "
        "enabling subgoal is achieved. Give the new subgoal an "
        "acceptance_check that the SUBSTRATE can verify — prefer the "
        "relational token 'mediation_precondition_met' over a generic "
        "visual_event, so it achieves only when the enabling state truly "
        "holds.\n"
        "  PHRASE the subgoal's expected_outcome as a CLEAN, GENERALIZED, "
        "INSTANCE-FREE, RELATIVE functional effect — the operator-retrieval "
        "keys on it, and absolute/instance wording does not match the "
        "generalized operator keys. Normalize BOTH ways:\n"
        "    (1) instance -> role: 'orange block' -> 'a block'/'a "
        "collectable'; never name colors/ids.\n"
        "    (2) absolute -> RELATIVE: convert absolute coordinates and "
        "frame directions to displacements RELATIVE TO THE AGENT'S AXES. "
        "e.g. 'move the orange block to row 26' -> 'move a block one step "
        "PERPENDICULAR to the arm's reach axis'; 'release blue under red' "
        "-> 'release a carried object from the agent, left in place'. "
        "(Empirically, the relative/role phrasing retrieves the right "
        "method at ~0.64 vs ~0.34 for the absolute/instance phrasing.)\n"
        "  RECURSE: if the new enabling subgoal is ALSO not directly "
        "achievable (this directive re-fires for it next), decompose it the "
        "same way. Continue until a subgoal is directly actionable. This is "
        "how the full means-ends chain is built; do not collapse it into a "
        "single greedy action.")


# ===========================================================================
# PRODUCER — build the structural context from a live WorldKnowledge
# ===========================================================================
# The consumers above (format_structural_invariants, detect_mediation_
# candidates) need: a LIST of entity-dicts (in cell units), an obstacle
# list, and a kinematics prior. The live world stores entities as a DICT
# of EntityRecord with bboxes in TICK units. This producer bridges the two
# so vlm_strategy.py can populate the prompt surfaces with one call. Without
# it the consumers were dormant (the dict didn't iterate as the detector
# expected, and nothing built the obstacle/kinematics objects). See the
# turn-164 gap: lessons surfaced but the live geometric surfaces showed 0.


# Roles (lowercased) treated as static impassable obstacles. Game-agnostic
# set; extend per domain. 'agent_rail' is deliberately EXCLUDED (the arm's
# own mount rail is not an obstacle to itself).
IMPASSABLE_ROLES = ("impassable_obstacle", "obstacle", "wall", "barrier")


def _ticks_bbox_to_cells(bbox, cell_ticks: int = 4):
    """Convert a (r0,c0,r1,c1) tick bbox to cell units. Returns None on
    malformed input. Default 4 ticks/cell (ARC-AGI-3 64x64-tick / 16x16-cell)."""
    if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
        return None
    try:
        ct = cell_ticks or 4
        return tuple(int(x) // ct for x in bbox)
    except (TypeError, ValueError):
        return None


def world_entities_as_dicts(world, *, cell_ticks: int = 4) -> list:
    """Convert ``world.entities`` (dict of EntityRecord) into the list of
    entity-dicts the detectors/format functions expect, with bboxes in
    CELL units under the ``bbox_cells`` key. Tolerant of missing
    attributes / empty world. Skips entities without a current bbox/role."""
    out: list = []
    ents = getattr(world, "entities", None)
    if not ents:
        return out
    # Resolve cell_ticks from the world's grid inference when available.
    gi = getattr(world, "grid_inference", None)
    ct = getattr(gi, "cell_ticks", None) or cell_ticks or 4
    values = ents.values() if hasattr(ents, "values") else ents
    for rec in values:
        name = getattr(rec, "name", None)
        bbox_ticks = getattr(rec, "current_bbox", None)
        role = getattr(rec, "current_role", None)
        if name is None or bbox_ticks is None:
            continue
        bbox_cells = _ticks_bbox_to_cells(bbox_ticks, ct)
        if bbox_cells is None:
            continue
        out.append({
            "name": name,
            "role_hypothesis": (role or "").lower(),
            "bbox_cells": list(bbox_cells),
        })
    return out


def obstacles_from_entities(entity_dicts: list) -> list:
    """Build ImpassableObstacle list from entity-dicts whose role is in
    IMPASSABLE_ROLES. Cell units (entity_dicts already normalized)."""
    out: list = []
    for e in entity_dicts:
        if str(e.get("role_hypothesis", "")).lower() in IMPASSABLE_ROLES:
            bb = e.get("bbox_cells")
            if isinstance(bb, (list, tuple)) and len(bb) == 4:
                out.append(ImpassableObstacle(
                    name=str(e.get("name")),
                    bbox=tuple(int(x) for x in bb),
                    note="static; no moving entity may overlap",
                ))
    return out


def build_structural_context(world, *, agent_name: str = ARC_DEFAULT_AGENT_NAME,
                             default_kinematics=None) -> dict:
    """One-call PRODUCER for vlm_strategy.py. Returns a dict with:
      'entities'  : list of entity-dicts (cell units)
      'obstacles' : list[ImpassableObstacle] derived from impassable roles
      'kinematics': AgentKinematics (the passed default, or None)
      'arm_state' : {head_row, head_col, tip_col} estimate from the agent +
                    its body entity, or None if not derivable.
    Pure read of the world; sets nothing. The caller decides whether to
    stash the results on the world or use them directly. Game-agnostic:
    obstacle roles + agent name are parameters/constants, not literals.
    """
    ents = world_entities_as_dicts(world)
    obstacles = obstacles_from_entities(ents)
    # Estimate arm state from the agent head + (optional) arm_body entity.
    arm_state = None
    agent = next((e for e in ents if e.get("name") == agent_name), None)
    if agent is not None:
        ar0, ac0, ar1, ac1 = agent["bbox_cells"]
        head_row = (ar0 + ar1) // 2
        head_col = (ac0 + ac1) // 2
        tip_col = ac1
        body = next((e for e in ents if e.get("name") == "arm_body"), None)
        if body is not None:
            _, _, _, bc1 = body["bbox_cells"]
            tip_col = max(tip_col, bc1)
        arm_state = {"head_row": head_row, "head_col": head_col,
                     "tip_col": tip_col}
    return {
        "entities": ents,
        "obstacles": obstacles,
        "kinematics": default_kinematics,
        "arm_state": arm_state,
    }


def format_mediation_candidates(candidates: list) -> str:
    """Render the MEDIATION CANDIDATES surface — concrete instantiations
    of the 'act through an intermediary' reflection move across all
    detected mediation kinds. Each candidate is rendered with its
    ``mediation_kind`` so the actor picks the matching mechanic.

    Empty when no candidates — substrate stays silent rather than nag.

    Domain-neutral text: cites entity names from the candidates verbatim;
    no game-specific vocabulary baked in.
    """
    if not candidates:
        return ""
    lines = [
        "MEDIATION CANDIDATES (act-through-intermediary, substrate-"
        "detected from current entity geometry — concrete instantiations "
        "of the 'act through an intermediary' reflection move). Each "
        "candidate is a (kind, target, intermediary, agent setup) tuple "
        "where direct reach to the target is gate-refused but acting on "
        "an intermediary positioned between agent and target could "
        "transmit the effect:",
    ]
    for c in candidates:
        tr0, tc0, tr1, tc1 = c.target_bbox
        ir0, ic0, ir1, ic1 = c.intermediary_bbox
        if c.mediation_kind == "vertical_lift":
            verb = "LIFT"
            setup = c.agent_setup or {}
            min_row = setup.get("body_min_row")
            cs = setup.get("body_col_span", (ic0, ic1))
            setup_line = (
                f"      setup: position agent body at row >={min_row} "
                f"with col span covering cols {cs[0]}-{cs[1]}"
            )
            engage_line = (
                f"      engage: {c.engage_action_hint} — body rises "
                f"through the intermediary's row, pushing it up, and the "
                f"target (resting in the intermediary's column above) "
                f"co-displaces upward"
            )
        elif c.mediation_kind == "horizontal_push":
            verb = "PUSH-INTO"
            setup = c.agent_setup or {}
            rs = setup.get("body_row_span", (ir0, ir1))
            # rs is (row_min, row_max); pick whichever side is set
            if "body_max_col" in setup:
                setup_line = (
                    f"      setup: position agent body in rows "
                    f"{rs[0]}-{rs[1]} with body extending to col "
                    f"<={setup['body_max_col']} (just left of "
                    f"intermediary)"
                )
            else:
                setup_line = (
                    f"      setup: position agent body in rows "
                    f"{rs[0]}-{rs[1]} with body extending from col "
                    f">={setup.get('body_min_col', '?')} (just right of "
                    f"intermediary)"
                )
            engage_line = (
                f"      engage: {c.engage_action_hint} — body pushes "
                f"intermediary along the row, driving it into the target"
            )
        else:
            verb = c.mediation_kind.upper()
            setup_line = f"      setup: {c.agent_setup}"
            engage_line = (
                f"      engage: {c.engage_action_hint} (mediation kind "
                f"'{c.mediation_kind}')"
            )
        lines.append(
            f"  - {verb} '{c.target}' (rows {tr0}-{tr1} cols {tc0}-{tc1}) "
            f"via intermediary '{c.intermediary}' (rows {ir0}-{ir1} cols "
            f"{ic0}-{ic1}) [kind={c.mediation_kind}]:"
        )
        lines.append(setup_line)
        lines.append(engage_line)
        if c.note:
            lines.append(f"      why: {c.note}")
    lines.append("  Confirm the structural gate passes for each step of "
                 "the setup (body must NEVER overlap an impassable "
                 "obstacle), then commit the planned_action_sequence. "
                 "If the engage action produces a visual_event on the "
                 "target, the mechanic fired — repeat for the next "
                 "target if applicable.")
    return "\n".join(lines)


# Legacy alias — old callers wrote `format_indirect_push_candidates`.
# New code should use `format_mediation_candidates`.
format_indirect_push_candidates = format_mediation_candidates
