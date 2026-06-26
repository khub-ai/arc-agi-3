"""Playback mining — answer framing/world-model questions from the RECORDED
history (all prior steps/levels) instead of fresh live probing.

ORDERING DISCIPLINE -- DEDUP WASTED LOOPS BEFORE OPERATOR MINING.
Before invoking any operator/rule miner here (movement_rules, preconditions,
grasp_by_pinning, decouple, column_control, reposition_precondition, ...), run
a state-signature cycle-removal pass over the playback to drop revisit loops.
Wasted loops in the raw trace inflate dominant-action support counts and
pollute the contrastive splits these miners rely on (a U that cancelled a
prior D, then was itself undone, creates spurious "same action different
effect" evidence that looks like a state-class dependency but isn't). See
memory/feedback_playback_mining_dedup_first.md for the verification protocol;
the cleaning step is the same game-agnostic shape every operator miner uses
(perception-only state signature -> BFS over the observed graph -> REPLAY-
VERIFY each shortcut, since visible signatures miss hidden grip/load state).
Treat the cleaning pass as a precondition of trusting the mined numbers.

See SPEC_visual_reasoning_substrate.md § "Framing priors". A framing prior
states a checkable predicted consequence over observable relations; the
substrate confirms/refutes it by searching the recorded playback for the
natural experiment that already happened. This module provides that search.

The first concrete query: did a block ever LEAVE the skewer (un-skewer), and
when? That transition is the discriminating event a change-only channel misses
and that the latched HUD done-set cannot show. It is computed from the standing
`overlapping` relations A now records each turn (overlap-reachability from the
agent = the carried/skewered assembly), so the answer is bidirectional: a block
present in the skewered set on turn t-1 and absent on turn t un-skewered.

Game-agnostic: roles + overlap geometry over open-vocab entities; no sk48
vocabulary.
"""
from __future__ import annotations

from typing import Optional

from world_knowledge import WorldKnowledge          # noqa: E402
from relational_kinematics import overlap_reachable  # noqa: E402


def _roles(world: WorldKnowledge):
    agent = None
    collectables = set()
    for rec in world.entities.values():
        role = getattr(rec, "current_role", None)
        if role == "agent" and agent is None:
            agent = rec.name
        elif role == "collectable":
            collectables.add(rec.name)
    return agent, collectables


def skewered_history(world: WorldKnowledge) -> list[tuple[int, frozenset]]:
    """Per recorded delta: (to_turn, frozenset of collectables that were on
    the skewer/carry assembly that turn).  Derived from the standing
    `overlapping` relations; empty frozensets for turns predating that signal."""
    agent, collectables = _roles(world)
    out: list[tuple[int, frozenset]] = []
    for d in (getattr(world, "deltas_observed", None) or []):
        rels = (d.get("relations") if isinstance(d, dict)
                else getattr(d, "relations", None)) or []
        reached = overlap_reachable(rels, agent)
        out.append((
            (d.get("to_turn") if isinstance(d, dict) else getattr(d, "to_turn", 0)),
            frozenset(reached & collectables),
        ))
    return out


def find_membership_losses(series: list[tuple[int, frozenset]]):
    """General primitive: over a [(turn, set)] series, find members that were
    present and then absent — i.e. a condition LOST.  Returns
    [{turn, member, from_turn}] for each loss event."""
    losses = []
    for (pt, ps), (ct, cs) in zip(series, series[1:]):
        for m in (ps - cs):
            losses.append({"from_turn": pt, "turn": ct, "member": m})
    return losses


def find_membership_gains(series: list[tuple[int, frozenset]]):
    """Symmetric to find_membership_losses: members that became PRESENT — a
    condition was GAINED. The canonical case is a collectable JOINING the
    skewer (free -> impaled), the IMPORTANT state change whose enabling
    precondition (a backstop) we want to mine. Returns [{turn, member,
    from_turn}]."""
    gains = []
    for (pt, ps), (ct, cs) in zip(series, series[1:]):
        for m in (cs - ps):
            gains.append({"from_turn": pt, "turn": ct, "member": m})
    return gains


def unskewer_events(world: WorldKnowledge):
    """Turns at which a collectable LEFT the skewer (the natural experiments a
    framing prior about un-skewering is checked against)."""
    return find_membership_losses(skewered_history(world))


def skewer_gain_events(world: WorldKnowledge):
    """Turns at which a collectable JOINED the skewer (free -> impaled)."""
    return find_membership_gains(skewered_history(world))


def completion_history(world: WorldKnowledge):
    """Per recorded turn: (to_turn, frozenset of COMPLETED target identities)
    from the now-bidirectional done-set.  A member LOST between turns = its
    completion reverted (the block un-skewered / was undone).  This is the
    real un-skewer signal (the interior/HUD condition), as opposed to the
    geometric contiguity signal which a position-less un-skewer does not move.
    Empty if no win relation is committed."""
    try:
        from knowledge_crystallization import evaluate_win_relation  # noqa: E402
    except Exception:
        return []
    cands = [h for h in getattr(world, "win_condition_hypotheses", [])
             if getattr(h, "win_relation", None)]
    if not cands:
        return []
    rel = sorted(cands, key=lambda h: getattr(h, "credence", 0.0),
                 reverse=True)[0].win_relation
    out = []
    for d in (getattr(world, "deltas_observed", None) or []):
        t = (d.get("to_turn") if isinstance(d, dict)
             else getattr(d, "to_turn", 0))
        done = set((evaluate_win_relation(world, rel, turn=t).get("detail") or {})
                   .get("done") or [])
        out.append((t, frozenset(done)))
    return out


def completion_loss_events(world: WorldKnowledge):
    """Turns at which a target's COMPLETION was lost (un-skewer / undo),
    mined from the bidirectional done-set."""
    return find_membership_losses(completion_history(world))


def answer_unskewer_reverts_credit(world: WorldKnowledge) -> dict:
    """Framing query: does un-skewering a block REVERT its win credit, or
    does credit LATCH?  Answered from the bidirectional done-set: if a
    completed target's credit was ever LOST in the recorded playback, credit
    reverts; if blocks left the skewer but no credit was ever lost, it
    latches.  This supersedes the earlier geometric-contiguity attempt (which
    a position-less un-skewer does not register)."""
    losses = completion_loss_events(world)
    if losses:
        return {"verdict": "reverts", "events": losses,
                "note": "a target's completion credit was lost in the recorded "
                        "playback when it un-skewered -> credit does NOT latch; "
                        "un-skewering must be guarded against"}
    # no credit ever lost: did anything un-skewer at all?
    if completion_history(world):
        return {"verdict": "latches", "events": [],
                "note": "no completion credit was ever lost across the playback "
                        "despite play -> credit appears to latch once set"}
    return {"verdict": "no_evidence", "events": [],
            "note": "no committed win relation / completion history to mine"}


# ===========================================================================
# Per-level segmentation + auto-mine report
# ---------------------------------------------------------------------------
# The recording spans MULTIPLE levels (each score advance resets the board);
# mining ACROSS a boundary conflates levels.  `score_increased` on a delta is
# the game-agnostic boundary marker the driver already records.  These helpers
# segment the playback by that marker, scope the analysis to the CURRENT
# level, and assemble a compact report the strategy packet surfaces WHEN the
# system is stuck/regressing.  No game vocabulary — roles + the committed win
# relation only.
# ===========================================================================

def _d(o, k, default=None):
    return o.get(k, default) if isinstance(o, dict) else getattr(o, k, default)


def level_boundaries(world: WorldKnowledge) -> list[int]:
    """to_turn of every delta that advanced the score (a level boundary).
    The delta's completion belongs to the level that JUST ended; the next
    level begins on the following turn."""
    return [_d(d, "to_turn", 0)
            for d in (getattr(world, "deltas_observed", None) or [])
            if _d(d, "score_increased")]


def current_level_start_turn(world: WorldKnowledge) -> int:
    """The turn the current (latest) level began on — the turn AFTER the last
    score advance, or 0 if no advance has happened."""
    bnds = level_boundaries(world)
    return bnds[-1] if bnds else 0


def _action_by_turn(world: WorldKnowledge) -> dict:
    return {_d(d, "to_turn"): _d(d, "action")
            for d in (getattr(world, "deltas_observed", None) or [])}


def mine_current_level(world: WorldKnowledge) -> dict:
    """Mine the CURRENT level's recorded playback into a compact, game-agnostic
    report the actor can use to avoid repeating its own mistakes:

      {level_start, achieved, never_completed, best_count, collapses,
       catastrophe_actions, unskewer_verdict, peak_turn}

    - achieved          : every target completed at any point this level.
    - never_completed   : committed-win members never completed this level.
    - best_count        : the largest done-set reached this level (peak).
    - collapses         : [{turn, action, lost:[...]}] — turns where the
                          done-set REGRESSED (a target's credit was lost),
                          with the action that immediately preceded the loss.
    - catastrophe_actions: actions ranked by how many completions their step
                          dropped (the blame-assignment over negative events —
                          the action-class to guard against here).
    - unskewer_verdict  : 'reverts' / 'latches' / 'no_evidence' (credit model).

    Empty-ish ({achieved:[], ...}) when no win relation is committed yet."""
    start = current_level_start_turn(world)
    full = completion_history(world)            # bidirectional, per-level scoped
    cur = [(t, s) for (t, s) in full if t > start]
    achieved: set = set()
    best_count, peak_turn = 0, None
    for (t, s) in cur:
        achieved |= set(s)
        if len(s) > best_count:
            best_count, peak_turn = len(s), t
    # all committed members (for never_completed)
    members: list = []
    verdict = "no_evidence"
    try:
        from knowledge_crystallization import evaluate_win_relation  # noqa: E402
        cands = [h for h in getattr(world, "win_condition_hypotheses", [])
                 if getattr(h, "win_relation", None)]
        if cands:
            rel = sorted(cands, key=lambda h: getattr(h, "credence", 0.0),
                         reverse=True)[0].win_relation
            det = (evaluate_win_relation(world, rel).get("detail") or {})
            members = list(det.get("ordered") or [])
    except Exception:
        pass
    # collapses (regressions) within this level + the preceding action
    acts = _action_by_turn(world)
    collapses = []
    blame: dict = {}
    for ev in find_membership_losses(cur):
        act = acts.get(ev["turn"])
        collapses.append({"turn": ev["turn"], "action": act,
                          "lost": ev["member"]})
        if act:
            blame[act] = blame.get(act, 0) + 1
    catastrophe_actions = [a for a, _ in
                           sorted(blame.items(), key=lambda kv: -kv[1])]
    try:
        verdict = answer_unskewer_reverts_credit(world).get("verdict",
                                                            "no_evidence")
    except Exception:
        pass
    return {
        "level_start": start,
        "achieved": sorted(achieved),
        "never_completed": [m for m in members if m not in achieved],
        "best_count": best_count,
        "peak_turn": peak_turn,
        "n_members": len(members),
        "collapses": collapses,
        "catastrophe_actions": catastrophe_actions,
        "unskewer_verdict": verdict,
    }


def _mining_trigger(world: WorldKnowledge, n_recent: int = 6) -> bool:
    """Should the playback-mining block be surfaced this turn?  Fires WHEN
    NEEDED — not every turn — on any of:
      - a recent progress COLLAPSE (a completed target lost credit in the last
        few turns): the single most informative moment to learn from history;
      - a stall (the last few deltas produced no observable change);
      - a repeated-action rut (same action >= 3 times running).
    Cheap signals over deltas only; no game vocabulary."""
    deltas = getattr(world, "deltas_observed", None) or []
    if not deltas:
        return False
    cur_turn = getattr(world, "turn", _d(deltas[-1], "to_turn", 0)) or 0
    # recent collapse
    for c in mine_current_level(world).get("collapses", []):
        if cur_turn - c["turn"] <= n_recent:
            return True
    recent = deltas[-n_recent:]
    # stall: trailing run of no-observable-change deltas (>= 2)
    stall = 0
    for d in reversed(recent):
        changed = (_d(d, "agent_moved") or _d(d, "entities_appeared")
                   or _d(d, "entities_disappeared") or _d(d, "entities_changed"))
        if changed:
            break
        stall += 1
    if stall >= 2:
        return True
    # repeated-action rut (>= 3)
    last_a = _d(recent[-1], "action")
    rep = 0
    for d in reversed(recent):
        if _d(d, "action") == last_a:
            rep += 1
        else:
            break
    return rep >= 3


def format_mining_report(world: WorldKnowledge) -> str:
    """Render mine_current_level() as a short actor-facing block.  Returns ''
    when there is nothing worth saying (no win relation / no history)."""
    r = mine_current_level(world)
    if not r["n_members"] and not r["achieved"]:
        return ""
    lines = ["  PLAYBACK MINING (this level — learn from your own history):"]
    if r["achieved"]:
        lines.append(f"    - reached so far this level: {r['achieved']} "
                     f"(peak {r['best_count']}/{r['n_members']} at turn "
                     f"{r['peak_turn']}).")
    if r["never_completed"]:
        lines.append(f"    - NEVER completed this level: {r['never_completed']}"
                     " — this is the genuinely unsolved sub-goal; spend effort"
                     " here, not on re-doing what already completes.")
    if r["collapses"]:
        ca = r["catastrophe_actions"]
        lines.append(
            f"    - progress COLLAPSED {len(r['collapses'])}x this level "
            f"(a completed target lost its credit). The action right before "
            f"the loss was, most often: {ca}. Treat {ca[:1]} as a HAZARD in "
            f"this state — it un-does progress; prefer an alternative or stop "
            f"short of it.")
        # show the most recent collapse concretely
        last = r["collapses"][-1]
        lines.append(f"      most recent: turn {last['turn']} action "
                     f"{last['action']} lost {last['lost']!r}.")
    if r["unskewer_verdict"] == "reverts":
        lines.append("    - credit does NOT latch: a target that un-completes "
                     "LOSES its credit. Keep completed targets in their "
                     "completed state; never trade one away to chase another.")
    # IMPORTANT STATE-CHANGE rules (free<->impaled) + crystallize them into the
    # operator KB so the enabling precondition (e.g. a wall as impale backstop)
    # is RETRIEVABLE next time instead of re-derived by trial and error.
    try:
        sc = mine_state_change_rules(world)
        scs = format_state_change_rules(sc)
        if scs:
            lines.append("  " + scs.replace("\n", "\n  "))
        persist_state_change_operators(world, sc)
    except Exception:
        pass
    # EFFICIENCY HEURISTICS — discover from churn signatures + surface the
    # discovered rules-of-thumb so the actor avoids the same wasted work.
    try:
        from efficiency_heuristics import format_heuristics_surface
        heur = discover_and_persist_heuristics(world)
        hs = format_heuristics_surface(heur)
        if hs:
            lines.append("  " + hs.replace("\n", "\n  "))
    except Exception:
        pass
    # KNOWN NO-OPS — mined ineffective actions (e.g. push-against-flush-wall),
    # so the actor stops repeating an action that produces no state change.
    try:
        noop = mine_noop_rules(world)
        ns = format_noop_surface(noop)
        if ns:
            lines.append("  " + ns.replace("\n", "\n  "))
    except Exception:
        pass
    return "\n".join(lines)


# ===========================================================================
# Movement-rule mining (DATA-DRIVEN; "how does a movable object move?")
# ---------------------------------------------------------------------------
# Mines precondition->action->effect rules for moving a movable object from the
# recorded per-turn positions (bbox_history) + actions.  Rules are emitted in
# GEOMETRIC/RELATIONAL primitives only -- an object's relation to a structure's
# ANCHORED vs FREE end, and the motion direction along the structure's axis --
# so they (a) carry no game vocabulary, (b) transfer by signature, and (c)
# REFINE purely by accumulating support as more logs are mined (no per-rule
# code, no code change for later levels).  The "impale at the open end (tip)"
# rule is one OUTPUT of this single general pass, indistinguishable in form
# from any other movement rule it derives.  Anchored/free ends are COMPUTED
# from observed geometry (the endpoint nearest a cap/marker attachment is the
# anchored end); "toward tip" / "raise" are therefore never hard-coded.
# ===========================================================================

def _bbox_at(rec, turn: int):
    last = None
    for (t, b) in (getattr(rec, "bbox_history", None) or []):
        if t <= turn:
            last = b
        else:
            break
    return last


def _ctr(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)   # (y, x)


def _role_str(rec) -> str:
    return (getattr(rec, "current_role", None) or "").lower()


def _movables(world: WorldKnowledge):
    out = []
    for rec in world.entities.values():
        r, n = _role_str(rec), (rec.name or "").lower()
        if "collect" in r or "block" in n:
            out.append(rec)
    return out


def _agent(world: WorldKnowledge):
    """The manipulator/agent entity (the thing the actor directly drives)."""
    for rec in world.entities.values():
        if _role_str(rec) == "agent":
            return rec
    for rec in world.entities.values():
        n = (rec.name or "").lower()
        if "head" in n or "manipulator" in n or "arm" in n or "agent" in n:
            return rec
    return None


def _structures(world: WorldKnowledge):
    """Linear carrier structures, each with a derived ANCHORED end + FREE end.
    Geometry only: anchored end := the endpoint nearest a cap/marker entity (the
    attachment perception tags); free end := the opposite endpoint."""
    caps = []
    for r in world.entities.values():
        if "marker" in _role_str(r) or "cap" in (r.name or "").lower():
            b = _bbox_at(r, 10 ** 9)
            if b:
                caps.append(_ctr(b))
    structs = []
    for r in world.entities.values():
        role, nm = _role_str(r), (r.name or "").lower()
        if not ("structure" in role or "carrier" in role or "rod" in role
                or "rod" in nm or "shaft" in nm or "carrier" in nm):
            continue
        b = _bbox_at(r, 10 ** 9)
        if not b:
            continue
        h, w = b[2] - b[0], b[3] - b[1]
        if h >= w:                       # vertical structure, axis = y
            xc = (b[1] + b[3]) / 2.0
            ends = {"e0": (b[0], xc), "e1": (b[2], xc)}; axis = "y"; perp = xc
            ext = (b[0], b[2])
        else:                            # horizontal structure, axis = x
            yc = (b[0] + b[2]) / 2.0
            ends = {"e0": (yc, b[1]), "e1": (yc, b[3])}; axis = "x"; perp = yc
            ext = (b[1], b[3])
        if caps:
            def capd(pt):
                return min(abs(pt[0] - c[0]) + abs(pt[1] - c[1]) for c in caps)
            anch_k = min(ends, key=lambda k: capd(ends[k]))
        else:
            anch_k = "e0"
        free_k = "e1" if anch_k == "e0" else "e0"
        structs.append({"name": r.name, "axis": axis, "perp": perp, "ext": ext,
                        "anchored": ends[anch_k], "free": ends[free_k]})
    return structs


def _axis_pos(ctr, axis):
    return ctr[0] if axis == "y" else ctr[1]


def mine_movement_rules(world: WorldKnowledge, *, align_tol=2.0,
                        free_tol=4.0) -> dict:
    """Single general pass.  Returns {category: rule} where each rule is a
    precondition->action->effect record in geometric primitives with a support
    count.  Categories are DERIVED (impale = movable enters a structure's extent
    from outside; detach = leaves it; translate = moves not onto a structure)."""
    structs = _structures(world)
    movables = _movables(world)
    acts = _action_by_turn(world)
    deltas = getattr(world, "deltas_observed", None) or []
    rules: dict = {}

    def bump(cat, action, precond_facts, ex):
        key = cat
        r = rules.setdefault(key, {
            "category": cat, "actions": {}, "support": 0,
            "precond_facts": {}, "examples": []})
        r["support"] += 1
        r["actions"][action] = r["actions"].get(action, 0) + 1
        for k, v in precond_facts.items():
            agg = r["precond_facts"].setdefault(k, {"true": 0, "n": 0,
                                                    "vals": []})
            agg["n"] += 1
            if isinstance(v, bool):
                agg["true"] += int(v)
            else:
                agg["vals"].append(round(float(v), 2))
        if len(r["examples"]) < 8:
            r["examples"].append(ex)

    for d in deltas:
        t = _d(d, "to_turn")
        if t is None:
            continue
        action = acts.get(t) or _d(d, "action")
        for m in movables:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not b0 or not b1 or b0 == b1:
                continue
            c0, c1 = _ctr(b0), _ctr(b1)
            interacted = False
            for s in structs:
                ap = s["perp"]; lo, hi = s["ext"]
                al0 = abs((_axis_pos(c0, "x" if s["axis"] == "y" else "y")) - ap) <= align_tol
                al1 = abs((_axis_pos(c1, "x" if s["axis"] == "y" else "y")) - ap) <= align_tol
                pos0 = _axis_pos(c0, s["axis"]); pos1 = _axis_pos(c1, s["axis"])
                inext0 = al0 and (lo - 1 <= pos0 <= hi + 1)
                inext1 = al1 and (lo - 1 <= pos1 <= hi + 1)
                # distance (just before) to the FREE vs ANCHORED end
                d_free = abs(pos0 - _axis_pos(s["free"], s["axis"]))
                d_anch = abs(pos0 - _axis_pos(s["anchored"], s["axis"]))
                moved_toward_anchor = abs(pos1 - _axis_pos(s["anchored"], s["axis"])) \
                    < abs(pos0 - _axis_pos(s["anchored"], s["axis"]))
                if inext1 and not inext0:                       # ENTERED -> impale
                    bump("impale", action, {
                        "entered_at_free_end": d_free <= d_anch and d_free <= free_tol,
                        "aligned_to_axis_before": al0,
                        "moved_toward_anchored_end": moved_toward_anchor,
                        "gap_to_free_end_before": d_free,
                        "gap_to_anchored_end_before": d_anch,
                    }, {"turn": t, "obj": m.name, "struct": s["name"]})
                    interacted = True
                elif inext0 and not inext1:                     # LEFT -> detach
                    bump("detach", action, {
                        "aligned_to_axis_before": al0,
                        "moved_toward_free_end": not moved_toward_anchor,
                    }, {"turn": t, "obj": m.name, "struct": s["name"]})
                    interacted = True
            if not interacted:                                  # plain translate
                dy, dx = c1[0] - c0[0], c1[1] - c0[1]
                bump("translate", action, {
                    "horizontal": abs(dx) > abs(dy),
                    "vertical": abs(dy) > abs(dx),
                }, {"turn": t, "obj": m.name})
    return rules


def _movement_kb_path():
    # Resolve under the unified KB root (see kb_paths.py); falls back to cwd if
    # the KB module is somehow unavailable.
    try:
        from kb_paths import kb_path as _kb_path
        return _kb_path("movement_rules.json")
    except ImportError:
        try:
            from perception_loop_v2.kb_paths import kb_path as _kb_path
            return _kb_path("movement_rules.json")
        except Exception:
            from pathlib import Path as _P
            return _P("movement_rules.json")
    except Exception:
        from pathlib import Path as _P
        return _P("movement_rules.json")


def load_movement_rules(path=None) -> dict:
    import json
    from pathlib import Path as _P
    p = _P(path or _movement_kb_path())
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}


def save_movement_rules(rules: dict, path=None) -> None:
    import json
    from pathlib import Path as _P
    p = _P(path or _movement_kb_path())
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rules, indent=1))


# ===========================================================================
# STATE-CHANGE RULE MINING
# ---------------------------------------------------------------------------
# The movement miner above categorizes geometry (entered/left/translated). But
# the rules that matter most are the ones that flip an entity's BEHAVIORAL
# STATE — free<->impaled, completed<->reverted. Those transitions are the
# "important state changes": each one has an enabling PRECONDITION worth
# capturing once so the actor never re-derives it by trial and error.
#
# The decisive example is impale-by-backstop: a free block pushed with open
# space ahead just SLIDES (stays free), but pushed against a fixed backstop
# (a wall, the playfield edge, or another block) it cannot move, so the rod
# drives THROUGH it -> it becomes impaled. The DISCRIMINATOR is the backstop
# on the far side. The geometric miner sees "entered structure"; it never
# records WHY (blocked vs open), so the rule that a wall is required as a
# backstop was never captured -> re-derived every session. This pass mines
# that discriminator directly off the free->impaled transition.
# ===========================================================================

# Roles treated as fixed backstops (game-agnostic; the adapter/perception
# tags walls/obstacles with one of these).
IMPASSABLE_BACKSTOP_ROLES = {
    "impassable_obstacle", "wall", "obstacle", "barrier", "agent_rail",
}


def _play_bounds_ticks(world: WorldKnowledge):
    """Outer playfield bounds in tick coords. Uses the OBSERVED extent of all
    tracked entities (the de-facto playfield edges — e.g. the right wall a
    block impales against, which is narrower than the raw grid dimension),
    falling back to the grid inference when no geometry is available. The far
    edge being within `gap` of a bound is a BOUNDARY backstop."""
    minr = minc = 10 ** 9
    maxr = maxc = -10 ** 9
    for rec in getattr(world, "entities", {}).values():
        for (_t, b) in (getattr(rec, "bbox_history", None) or []):
            if not b:
                continue
            minr = min(minr, b[0]); minc = min(minc, b[1])
            maxr = max(maxr, b[2]); maxc = max(maxc, b[3])
    if maxc < 0:                                # no geometry -> grid dims
        gi = getattr(world, "grid_inference", None) or {}
        ct = _d(gi, "cell_ticks", 4) or 4
        rows = _d(gi, "rows", 16) or 16
        cols = _d(gi, "cols", 16) or 16
        return (0, 0, int(rows) * int(ct), int(cols) * int(ct))
    return (minr, minc, maxr, maxc)


def _static_backstop_bboxes(world: WorldKnowledge, turn: int):
    out = []
    for rec in world.entities.values():
        role = (getattr(rec, "current_role", None) or "").lower()
        if role in IMPASSABLE_BACKSTOP_ROLES:
            b = _bbox_at(rec, turn)
            if b:
                out.append((rec.name, b))
    return out


def _backstop_beyond(world: WorldKnowledge, b_before, b_after, turn,
                     *, gap: float = 6.0, self_name: str = ""):
    """Was a fixed backstop immediately beyond the block, in its motion
    direction, at the moment it stopped? Returns (has_backstop, kind) where
    kind in {'wall','boundary','block',None}. This is the IMPALE vs
    PUSH-STAYS-FREE discriminator. Pure geometry on tick bboxes."""
    c0, c1 = _ctr(b_before), _ctr(b_after)
    dr, dc = c1[0] - c0[0], c1[1] - c0[1]
    minr, minc, maxr, maxc = _play_bounds_ticks(world)
    if abs(dc) >= abs(dr):                      # horizontal motion
        d = 1 if dc > 0 else (-1 if dc < 0 else 0)
        if d == 0:
            return (False, None)
        far = b_after[3] if d > 0 else b_after[1]
        lo, hi = b_after[0], b_after[2]         # row span to require overlap
        bound = maxc if d > 0 else minc
        near_of = (lambda b: b[1]) if d > 0 else (lambda b: b[3])
        span_of = (lambda b: (b[0], b[2]))
    else:                                       # vertical motion
        d = 1 if dr > 0 else (-1 if dr < 0 else 0)
        if d == 0:
            return (False, None)
        far = b_after[2] if d > 0 else b_after[0]
        lo, hi = b_after[1], b_after[3]
        bound = maxr if d > 0 else minr
        near_of = (lambda b: b[0]) if d > 0 else (lambda b: b[2])
        span_of = (lambda b: (b[1], b[3]))

    if abs(far - bound) <= gap:
        return (True, "boundary")
    for _name, b in _static_backstop_bboxes(world, turn):
        slo, shi = span_of(b)
        if not (shi < lo or slo > hi) and 0 <= (near_of(b) - far) * d <= gap:
            return (True, "wall")
    for m in _movables(world):
        if getattr(m, "name", "") == self_name:
            continue
        b = _bbox_at(m, turn)
        if not b:
            continue
        slo, shi = span_of(b)
        if not (shi < lo or slo > hi) and 0 < (near_of(b) - far) * d <= gap:
            return (True, "block")
    return (False, None)


def mine_state_change_rules(world: WorldKnowledge, *, gap: float = 6.0) -> dict:
    """Mine rules off IMPORTANT STATE CHANGES (free<->impaled), capturing the
    discriminating precondition each transition needs. Returns {category:
    rule} with support + precond_facts, same shape as mine_movement_rules."""
    acts = _action_by_turn(world)
    rules: dict = {}

    def bump(cat, action, precond, ex):
        r = rules.setdefault(cat, {"category": cat, "actions": {},
                                   "support": 0, "precond_facts": {},
                                   "examples": []})
        r["support"] += 1
        if action:
            r["actions"][action] = r["actions"].get(action, 0) + 1
        for k, v in precond.items():
            agg = r["precond_facts"].setdefault(k, {"true": 0, "n": 0})
            agg["n"] += 1
            agg["true"] += int(bool(v))
        if len(r["examples"]) < 8:
            r["examples"].append(ex)

    # FREE -> IMPALED: mine the backstop discriminator.
    for ev in skewer_gain_events(world):
        t, m = ev["turn"], ev["member"]
        rec = world.entities.get(m)
        if not rec:
            continue
        b0, b1 = _bbox_at(rec, t - 1), _bbox_at(rec, t)
        if not b0 or not b1:
            continue
        has_bs, kind = _backstop_beyond(world, b0, b1, t, gap=gap, self_name=m)
        bump("impale_by_backstop", acts.get(t), {
            "backstop_on_far_side": has_bs,
            "backstop_is_wall_or_boundary": kind in ("wall", "boundary"),
        }, {"turn": t, "obj": m, "backstop_kind": kind})

    # IMPALED -> FREE: un-impale precondition (dragged toward the anchor wall).
    for ev in unskewer_events(world):
        t, m = ev["turn"], ev["member"]
        rec = world.entities.get(m)
        if not rec:
            continue
        b0, b1 = _bbox_at(rec, t - 1), _bbox_at(rec, t)
        toward_left = bool(b0 and b1 and _ctr(b1)[1] < _ctr(b0)[1])
        bump("release_by_drag_to_wall", acts.get(t), {
            "dragged_toward_anchor_wall": toward_left,
        }, {"turn": t, "obj": m})

    return rules


def _frac(agg: dict) -> float:
    n = agg.get("n", 0)
    return (agg.get("true", 0) / n) if n else 0.0


# ---------------------------------------------------------------------------
# CHURN-SIGNATURE MINING (the discovery half of efficiency heuristics)
# ---------------------------------------------------------------------------
# Wasted work has tell-tale signatures in the playback. Mining them is how COS
# discovers efficiency heuristics from its OWN history (game-agnostic):
#   * undo_redo            — a block was impaled then un-impaled (or vice
#                            versa): a committed/placed state was undone.
#   * repeated_reposition  — a block came to rest, then was moved AGAIN to a
#                            different rest position: placed in the wrong spot.
#   * shared_lane_interfere— two not-yet-finished blocks occupied the same
#                            row/lane while being manipulated: interference.
# Each signature (with support) maps to a typed efficiency heuristic
# (efficiency_heuristics.CHURN_TO_HEURISTICS).
# ---------------------------------------------------------------------------

def _rest_positions(rec, *, settle_tol: float = 1.5):
    """Distinct rest positions of a movable across its bbox history: a 'rest'
    is a center that holds (within tol) for >=2 consecutive samples, distinct
    from the previous rest. Returns the count of distinct rests."""
    hist = getattr(rec, "bbox_history", None) or []
    centers = [(_ctr(b)) for (_t, b) in hist if b]
    rests = []
    i = 0
    n = len(centers)
    while i < n:
        c = centers[i]
        j = i + 1
        while j < n and abs(centers[j][0] - c[0]) <= settle_tol \
                and abs(centers[j][1] - c[1]) <= settle_tol:
            j += 1
        if j - i >= 2:           # held -> a rest
            if not rests or (abs(rests[-1][0] - c[0]) > settle_tol
                             or abs(rests[-1][1] - c[1]) > settle_tol):
                rests.append(c)
        i = max(j, i + 1)
    return len(rests)


def mine_churn_signatures(world: WorldKnowledge) -> dict:
    """Mine wasted-work signatures from the playback. Returns
    {signature_kind: {"support": int, "examples": [...]}}. Game-agnostic:
    operates on skewer membership transitions + movable rest positions."""
    out: dict = {}

    def bump(kind, ex):
        r = out.setdefault(kind, {"support": 0, "examples": []})
        r["support"] += 1
        if len(r["examples"]) < 8:
            r["examples"].append(ex)

    # undo_redo: a block that gained skewer membership then lost it (impaled
    # then un-impaled) — a committed state undone.
    gains = skewer_gain_events(world)
    losses = unskewer_events(world)
    gain_turns: dict = {}
    for g in gains:
        gain_turns.setdefault(g["member"], []).append(g["turn"])
    for ls in losses:
        m, lt = ls["member"], ls["turn"]
        if any(gt < lt for gt in gain_turns.get(m, [])):
            bump("undo_redo", {"obj": m, "impaled_then_released_at": lt})

    # repeated_reposition: a movable with >=2 distinct rest positions.
    for m in _movables(world):
        if _rest_positions(m) >= 2:
            bump("repeated_reposition", {"obj": getattr(m, "name", "?")})

    # shared_lane_interference: two movables sharing a row band while both are
    # still off the skewer (being manipulated) at the same turn.
    series = skewered_history(world)
    skewered_by_turn = {t: s for (t, s) in series}
    movs = _movables(world)
    seen_pairs = set()
    for d in (getattr(world, "deltas_observed", None) or []):
        t = _d(d, "to_turn")
        if t is None:
            continue
        onsk = skewered_by_turn.get(t, frozenset())
        rows = []
        for m in movs:
            nm = getattr(m, "name", None)
            if nm in onsk:
                continue
            b = _bbox_at(m, t)
            if b:
                rows.append((nm, (b[0] + b[2]) / 2.0))
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                if abs(rows[i][1] - rows[j][1]) <= 2.0:
                    pair = tuple(sorted((rows[i][0], rows[j][0])))
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        bump("shared_lane_interference",
                             {"objs": list(pair), "turn": t})
    return out


# ---------------------------------------------------------------------------
# NO-OP RULE MINING (ineffective actions)
# ---------------------------------------------------------------------------
# "Pushing a block that's already flush against a wall does nothing" is basic
# real-world physics, but it was re-discovered the hard way by blindly
# repeating ACTION4. Mine it: an action that produced NO meaningful state
# change, in a recurring relational context, is a NO-OP there. Surfacing it
# lets the actor stop wasting actions (and lets a recalled operator that relies
# on that action be recognized as ineffective in that context).

def _any_movable_moved(world: WorldKnowledge, t: int) -> bool:
    for m in _movables(world):
        b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
        if b0 and b1 and b0 != b1:
            return True
    return False


def _movable_flush_boundary(world: WorldKnowledge, t: int, tol: int = 2) -> bool:
    """Is any movable flush against a playfield boundary at turn t (the
    'nothing to push it into' precondition for a push no-op)?"""
    minr, minc, maxr, maxc = _play_bounds_ticks(world)
    for m in _movables(world):
        b = _bbox_at(m, t)
        if not b:
            continue
        if (abs(b[3] - maxc) <= tol or abs(b[1] - minc) <= tol
                or abs(b[2] - maxr) <= tol or abs(b[0] - minr) <= tol):
            return True
    return False


def _entity_cols_at(rec, t):
    b = _bbox_at(rec, t)
    return (b[1], b[3]) if (b and len(b) >= 4) else None


def _agent_and_obstacle_cols(world: WorldKnowledge, t: int):
    """Column spans of the agent/manipulator and of obstacle entities at turn t.
    Game-agnostic: detected by role/name keywords from perception, never a
    game id."""
    agent = None
    obstacles = []
    for r in (getattr(world, "entities", {}) or {}).values():
        tag = (_role_str(r) + " " + (getattr(r, "name", "") or "")).lower()
        if agent is None and any(k in tag for k in (
                "agent", "arm", "manipulator", "rod", "gripper", "effector")):
            agent = r
        if any(k in tag for k in ("wall", "obstacle", "barrier")):
            obstacles.append(r)
    ac = _entity_cols_at(agent, t) if agent else None
    ocs = [c for c in (_entity_cols_at(o, t) for o in obstacles) if c]
    return ac, ocs


def _relational_noop_context(world: WorldKnowledge, t: int) -> str:
    """A RELATIONAL signature of why an action is a no-op, so the refuted
    approach TRANSFERS (e.g. raising a manipulator whose span crosses an
    obstacle is blocked -- the lc-4 failure). Game-agnostic predicates only."""
    preds = []
    ac, ocs = _agent_and_obstacle_cols(world, t)
    if ac and any(not (ac[1] < o[0] or ac[0] > o[1]) for o in ocs):
        preds.append("manipulator_spans_obstacle")
    if _movable_flush_boundary(world, t):
        preds.append("target_flush_against_boundary")
    return "+".join(preds) if preds else "no_state_change"


def last_action_was_noop(world: WorldKnowledge):
    """(is_noop, action, context) for the MOST RECENT action -- the un-missable
    post-action verdict. is_noop=True iff that action produced no movable
    movement and no score change."""
    deltas = getattr(world, "deltas_observed", None) or []
    if not deltas:
        return (False, None, None)
    d = deltas[-1]
    t = _d(d, "to_turn")
    if t is None:
        return (False, None, None)
    action = _action_by_turn(world).get(t) or _d(d, "action")
    if not action:
        return (False, None, None)
    if _any_movable_moved(world, t) or _d(d, "score_increased"):
        return (False, action, None)
    return (True, action, _relational_noop_context(world, t - 1))


def mine_refuted_approaches(world: WorldKnowledge, min_support: int = 2) -> list:
    """Promote high-support RELATIONAL no-ops into refuted-approach records: an
    action that repeatedly does nothing in a recurring relational context is an
    approach to AVOID. These feed the existing refuted-lesson gate so the
    SUBSTRATE -- not the VLM's memory -- carries 'don't retry X here'. Skips the
    generic 'no_state_change' context (too unspecific to transfer)."""
    out = []
    for r in mine_noop_rules(world, min_support=min_support).values():
        ctx = r["context"]
        if ctx == "no_state_change":
            continue
        pretty = ctx.replace("_", " ").replace("+", " and ")
        out.append({
            "action": r["action"], "context": ctx, "support": r["support"],
            # OBSERVED-grade: these are real no-ops seen in the trace, so they
            # may hard-gate (unlike an inferred/guessed impossibility, which
            # must be probed first -- see claim_credence).
            "provenance": "observed",
            # a directly-observed no-op may hard-gate at low support (it
            # literally did nothing); >=2 observations clears the gate threshold.
            "credence": min(0.8 + 0.05 * (r["support"] - 2), 0.97),
            "description": (f"REFUTED approach: action {r['action']} produces "
                            f"NO effect when {pretty} (no-op observed "
                            f"{r['support']}x) -- do not retry it in this "
                            f"situation; use a different mechanism."),
        })
    return out


def format_last_action_noop_surface(world: WorldKnowledge) -> str:
    """Un-missable post-action verdict: if the last action did nothing, say so
    LOUDLY so neither the actor nor a human driver claims a phantom effect."""
    is_noop, action, ctx = last_action_was_noop(world)
    if not is_noop:
        return ""
    extra = (f" (relational context: {ctx.replace('_', ' ')})"
             if ctx and ctx != "no_state_change" else "")
    return (f"LAST ACTION ({action}) PRODUCED NO STATE CHANGE -- it was a "
            f"NO-OP{extra}. Do NOT claim it had any effect and do NOT just "
            f"repeat it; the state is unchanged from before it.")


def mine_noop_rules(world: WorldKnowledge, min_support: int = 2) -> dict:
    """Mine (action, context) -> NO-OP rules: actions that produced no movable
    movement and no score change, grouped by a coarse relational context.
    Returns {key: {action, context, support, examples}} for support >=
    min_support. Game-agnostic (movables + boundary geometry only)."""
    acts = _action_by_turn(world)
    out: dict = {}

    def bump(action, ctx, ex):
        key = f"{action}|{ctx}"
        r = out.setdefault(key, {"action": action, "context": ctx,
                                 "support": 0, "examples": []})
        r["support"] += 1
        if len(r["examples"]) < 5:
            r["examples"].append(ex)

    for d in (getattr(world, "deltas_observed", None) or []):
        t = _d(d, "to_turn")
        if t is None:
            continue
        action = acts.get(t) or _d(d, "action")
        if not action:
            continue
        if _any_movable_moved(world, t) or _d(d, "score_increased"):
            continue                      # the action DID something
        # ineffective: characterize WHY relationally (a transferable
        # precondition -- manipulator spans an obstacle, target flush, etc.)
        ctx = _relational_noop_context(world, t - 1)
        bump(action, ctx, {"turn": t})
    return {k: v for k, v in out.items() if v["support"] >= min_support}


def format_noop_surface(rules: dict) -> str:
    """Render mined no-op rules so the actor doesn't waste the action."""
    if not rules:
        return ""
    lines = ["KNOWN NO-OPS (mined from your plays — do NOT repeat these; they "
             "produce no state change):"]
    for r in sorted(rules.values(), key=lambda x: -x["support"]):
        ctx = r["context"].replace("_", " ")
        lines.append(f"  - {r['action']} when {ctx} -> nothing happens "
                     f"[support={r['support']}]")
    return "\n".join(lines)


def discover_and_persist_heuristics(world: WorldKnowledge):
    """Mine churn -> discover typed efficiency heuristics -> persist. Returns
    the updated heuristic list (or [] if the module is unavailable)."""
    try:
        from efficiency_heuristics import (
            discover_heuristics, load_heuristics, save_heuristics)
    except Exception:
        return []
    churn = mine_churn_signatures(world)
    if not churn:
        return load_heuristics()
    sig = {"relation_signature": "ordered_collection_onto_carrier"}
    heur = discover_heuristics(churn, load_heuristics(), signature=sig)
    save_heuristics(heur)
    return heur


def persist_state_change_operators(world: WorldKnowledge,
                                   rules: dict | None = None) -> list:
    """Crystallize mined state-change rules into the operator KB as retrievable,
    function-keyed operators, so the actor gets them surfaced instead of
    re-deriving (the recurring failure). Returns the operator_ids written."""
    rules = rules if rules is not None else mine_state_change_rules(world)
    written = []
    try:
        from operator_kb import (load_kb, save_kb, commit_operator, neighbors)
    except Exception:
        return written
    records = load_kb()
    gid = getattr(world, "game_id", "?")
    sig = {"relation_signature": "ordered_collection_onto_carrier"}

    def _commit(effect_key, precondition, action_template, description,
                support):
        # author-against-the-neighborhood: merge into a near-duplicate key.
        reuse = None
        try:
            nb = neighbors(records, effect_key, k=1)   # -> [(record, sim)]
            if nb and nb[0][1] >= 0.9:
                reuse = nb[0][0].operator_id
        except Exception:
            reuse = None
        rec = commit_operator(
            records, effect_key=effect_key, precondition=precondition,
            action_template=action_template, description=description,
            credence=min(0.6 + 0.1 * support, 0.95),
            provenance={"derived_from": "playback", "support": support,
                        "game": gid},
            scope=sig, reuse_id=reuse)
        written.append(rec.operator_id)

    imp = rules.get("impale_by_backstop")
    if imp and imp.get("support", 0) >= 1:
        bs_frac = _frac(imp["precond_facts"].get("backstop_on_far_side",
                                                 {"true": 0, "n": 0}))
        if bs_frac >= 0.6:                      # the discriminator holds
            top_action = max(imp["actions"], key=imp["actions"].get) \
                if imp.get("actions") else None
            _commit(
                effect_key=("impale a free object onto the carrier by pushing "
                            "it against a fixed backstop until it cannot move"),
                precondition=("object is FREE AND a wall / playfield edge / "
                              "another blocked object lies immediately beyond "
                              "it in the push direction (open space ahead -> "
                              "it just slides and stays free)"),
                action_template=[f"push the object toward the backstop "
                                 f"(observed action: {top_action or 'extend'}) "
                                 f"until blocked; the carrier drives through "
                                 f"it -> impaled"],
                description=("Mined from the free->impaled transition: the "
                             "DISCRIMINATOR between impaling and a harmless "
                             "push is a backstop on the far side. Build an "
                             "ordered skewer by impaling against the wall "
                             "inward."),
                support=imp["support"])

    rel = rules.get("release_by_drag_to_wall")
    if rel and rel.get("support", 0) >= 1:
        _commit(
            effect_key=("release a carried object from the carrier without "
                        "relocating the rest, by dragging the assembly into "
                        "the anchor-side wall"),
            precondition=("object is impaled on the carrier AND the carrier "
                          "can be dragged so the object meets the anchor-side "
                          "wall, which strips it free"),
            action_template=["drag the carrier fully toward the anchor wall"],
            description=("Mined from the impaled->free transition: the "
                         "anchor-side wall strips a carried object off."),
            support=rel["support"])

    if written:
        try:
            save_kb(records)
        except Exception:
            pass
    return written


def format_state_change_rules(rules: dict) -> str:
    """Render mined state-change rules + their discriminating preconditions."""
    if not rules:
        return ""
    lines = ["STATE-CHANGE RULES (mined from free<->impaled transitions; "
             "the precondition is the DISCRIMINATOR, not decoration):"]
    imp = rules.get("impale_by_backstop")
    if imp:
        bs = _frac(imp["precond_facts"].get("backstop_on_far_side",
                                            {"true": 0, "n": 0}))
        acts = ", ".join(f"{a}x{c}" for a, c in
                         sorted(imp.get("actions", {}).items(),
                                key=lambda kv: -kv[1])) or "?"
        lines.append(
            f"  IMPALE a free object: push it against a BACKSTOP (wall / "
            f"playfield edge / blocked object) on its far side -> it can't "
            f"move, so it impales. Open space ahead -> it just slides (stays "
            f"free). [support={imp['support']}, backstop-present in "
            f"{bs*100:.0f}% of impales; action {acts}]")
    rel = rules.get("release_by_drag_to_wall")
    if rel:
        lines.append(
            f"  RELEASE an impaled object: drag the carrier into the "
            f"anchor-side wall to strip it free. [support={rel['support']}]")
    return "\n".join(lines)


def _merge_facts(into: dict, new: dict):
    for k, bv in new.items():
        av = into.setdefault(k, {"true": 0, "n": 0, "vals": []})
        av["true"] += bv.get("true", 0)
        av["n"] += bv.get("n", 0)
        av["vals"] = (av.get("vals", []) + bv.get("vals", []))[:200]


def merge_movement_rules(into: dict, new: dict) -> dict:
    """Accumulate `new` rules into `into` IN PLACE — support and precondition
    statistics add up, so re-mining new logs REFINES the ruleset by data alone
    (no code change for later levels)."""
    for cat, nr in new.items():
        r = into.setdefault(cat, {"category": cat, "actions": {}, "support": 0,
                                  "precond_facts": {}, "examples": []})
        r["support"] += nr.get("support", 0)
        for a, c in (nr.get("actions") or {}).items():
            r["actions"][a] = r["actions"].get(a, 0) + c
        _merge_facts(r["precond_facts"], nr.get("precond_facts") or {})
        r["examples"] = (r.get("examples", []) + (nr.get("examples") or []))[:12]
    return into


def accumulate_movement_rules(world_or_worlds, kb_path=None) -> dict:
    """Mine each world's movement rules and merge into the persistent KB, then
    save.  This is the refinement substrate: every trial that calls it makes
    the rules more supported / more accurate, with NO code change."""
    kb = load_movement_rules(kb_path)
    worlds = world_or_worlds if isinstance(world_or_worlds, list) else [world_or_worlds]
    for w in worlds:
        try:
            merge_movement_rules(kb, mine_movement_rules(w))
        except Exception:
            continue
    save_movement_rules(kb, kb_path)
    return kb


def _strong(rule: dict, key: str, frac: float = 0.8) -> bool:
    a = (rule.get("precond_facts") or {}).get(key)
    return bool(a and a.get("n") and (a["true"] / a["n"]) >= frac)


def movement_subgoal(rules: dict, shaft_struct: dict, *, category="impale",
                     min_support=1, frac=0.8) -> Optional[dict]:
    """GENERIC planner query: given the mined rules and a target structure,
    return the impale-ready sub-goal + action hint the rule prescribes —
    computed from the rule's strong preconditions + the structure's observed
    geometry.  Returns None if no supported rule.  This is what the backward
    reasoner calls instead of any hard-coded 'go to the tip' logic."""
    r = rules.get(category)
    if not r or r.get("support", 0) < min_support:
        return None
    cond = []
    if _strong(r, "aligned_to_axis_before", frac):
        cond.append({"rel": "align_to_axis", "perp": round(shaft_struct["perp"], 1)})
    if _strong(r, "entered_at_free_end", frac):
        cond.append({"rel": "at_free_end",
                     "pos": tuple(round(x) for x in shaft_struct["free"])})
    action_hint = None
    if _strong(r, "moved_toward_anchored_end", frac):
        action_hint = {"motion": "toward",
                       "target": tuple(round(x) for x in shaft_struct["anchored"])}
    dom = (max(r["actions"].items(), key=lambda kv: kv[1])[0]
           if r.get("actions") else None)
    return {"category": category, "subgoals": cond, "action_hint": action_hint,
            "dominant_action": dom, "support": r["support"]}


def describe_movement_rule(category: str, rule: dict) -> str:
    """One-line, game-agnostic description of a mined movement rule for a
    durable lesson."""
    parts = [f"MOVEMENT RULE (mined, support={rule.get('support',0)}): "
             f"to '{category}' a movable,"]
    strong = [k for k in (rule.get("precond_facts") or {})
              if _strong(rule, k)]
    if strong:
        parts.append("precondition = " + " AND ".join(strong) + ".")
    dom = (max(rule["actions"].items(), key=lambda kv: kv[1])[0]
           if rule.get("actions") else None)
    if dom:
        parts.append(f"dominant action = {dom}.")
    return " ".join(parts)


def _playfield_x_range(world: WorldKnowledge):
    """The x-extent a MOVABLE can reach (where it piles against a wall) — the
    reference for 'at_right_wall'/'at_left_wall'.  Computed from movables only,
    so a full-width structure (e.g. the arm rail) doesn't inflate it."""
    xs = []
    for rec in _movables(world):
        for (_t, b) in (getattr(rec, "bbox_history", None) or []):
            xs += [b[1], b[3]]
    return (min(xs), max(xs)) if xs else (0, 64)


def _movable_features(b0, structs, xrange):
    """Relational state of a movable BEFORE an action — geometric primitives
    only (no game vocabulary)."""
    xlo, xhi = xrange
    cx = (b0[1] + b0[3]) / 2.0
    return {
        "at_right_wall": (xhi - b0[3]) <= 2,
        "at_left_wall": (b0[1] - xlo) <= 2,
        "aligned_to_a_structure": any(
            abs(cx - s["perp"]) <= 2 for s in structs if s["axis"] == "y"),
    }


def mine_movement_preconditions(world: WorldKnowledge) -> dict:
    """CONTRASTIVE precondition mining.  For each ACTION, partition every
    movable event into POSITIVE (the action produced a leftward move = toward
    the head/anchor side) vs NEGATIVE (it did not, incl. no-move), snapshot the
    relational features BEFORE the action, and accumulate true/total per
    feature in each class.  The DISCRIMINANT (high in positives, low in
    negatives) is the rule's precondition -- derived, not authored.  This is the
    pass that distinguishes 'retract slides the block' (it was seated against
    the wall) from 'retract releases it' (it was not)."""
    structs = _structures(world)
    xr = _playfield_x_range(world)
    movables = _movables(world)
    acts = _action_by_turn(world)
    out: dict = {}
    for d in (getattr(world, "deltas_observed", None) or []):
        t = _d(d, "to_turn")
        if t is None:
            continue
        action = acts.get(t) or _d(d, "action")
        for m in movables:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not b0 or not b1:
                continue
            feats = _movable_features(b0, structs, xr)
            moved_left = _ctr(b1)[1] < _ctr(b0)[1] - 0.5
            slot = out.setdefault(action, {})
            for k, v in feats.items():
                agg = slot.setdefault(k, {"pos": [0, 0], "neg": [0, 0]})
                cell = agg["pos"] if moved_left else agg["neg"]
                cell[1] += 1
                cell[0] += int(bool(v))
    return out


def precondition_discriminants(pre: dict, *, min_gap=0.5) -> dict:
    """From contrastive stats, pick each action's discriminating precondition:
    feature with the largest (pos_rate - neg_rate) above min_gap."""
    res = {}
    for action, feats in pre.items():
        best = None
        for k, agg in feats.items():
            pr = agg["pos"][0] / agg["pos"][1] if agg["pos"][1] else 0.0
            nr = agg["neg"][0] / agg["neg"][1] if agg["neg"][1] else 0.0
            gap = pr - nr
            if agg["pos"][1] and gap >= min_gap and (best is None or gap > best[1]):
                best = (k, gap, pr, nr, agg["pos"][1], agg["neg"][1])
        if best:
            res[action] = {"precondition": best[0], "gap": round(best[1], 2),
                           "pos_rate": round(best[2], 2), "neg_rate": round(best[3], 2),
                           "pos_n": best[4], "neg_n": best[5]}
    return res


def format_movement_preconditions(pre: dict) -> str:
    disc = precondition_discriminants(pre)
    if not disc:
        return "(no discriminating preconditions mined — need both +/- cases)"
    lines = ["MINED PRECONDITIONS (contrastive; effect = move toward head/anchor):"]
    for action, d in disc.items():
        lines.append(
            f"  action {action}: PRECONDITION = {d['precondition']}  "
            f"(true in {d['pos_rate']:.0%} of slides vs {d['neg_rate']:.0%} of "
            f"non-slides; +{d['pos_n']}/-{d['neg_n']}, gap {d['gap']})")
    return "\n".join(lines)


def mine_engagement_by_body_sweep(world: WorldKnowledge) -> dict:
    """Mine the ENGAGEMENT-BY-BODY-SWEEP mechanic: a movable becomes ATTACHED
    to the agent on a step when (a) the agent's BBOX at the START of the step
    already extends past the movable on the agent's long axis (the body
    SPANS PAST the movable's coordinate), and (b) the agent translates along
    the agent's SHORT axis (perpendicular to its body).  On such a step, the
    movable's centroid co-displaces with the agent.

    This is a third independent engagement operator distinct from:
      - grasp_by_pinning  (push into a barrier ATTACHES at the barrier)
      - decouple          (retract+raise RELEASES the extreme-axis movable)
    Body-sweep ATTACHES via a perpendicular sweep when the body already
    SPANS the movable -- the elongated agent acts like a horizontal shelf
    moved vertically.  Game-agnostic: agent.bbox + movable.bbox + the
    derived long/short axis from the agent's aspect ratio; no game vocab.

    Contrastive evidence: among all PERPENDICULAR-axis agent translations
    where a movable is in the perpendicular plane within the agent's body
    extent, what fraction co-display the movable?  Compare against steps
    where the agent's body did NOT span the movable's coordinate.  A high
    purity gap is the rule.

    Returns {category:'engage_by_body_sweep', support, spanned_co_moves,
    spanned_no_moves, gap, perpendicular_axis, actions, effect, examples}
    or {} when the agent is untracked / no relevant steps.
    """
    agent = _agent(world)
    if agent is None:
        return {}
    movables = _movables(world)
    acts = _action_by_turn(world)
    deltas = getattr(world, "deltas_observed", None) or []
    spanned_co_moves = 0
    spanned_no_moves = 0
    no_span_co_moves = 0
    no_span_no_moves = 0
    actions: dict = {}
    examples: list = []
    for d in deltas:
        t = _d(d, "to_turn")
        if t is None:
            continue
        a0, a1 = _bbox_at(agent, t - 1), _bbox_at(agent, t)
        if not (a0 and a1):
            continue
        # determine the agent's LONG axis from its bbox aspect at t-1
        ah = a0[2] - a0[0]
        aw = a0[3] - a0[1]
        long_axis = "x" if aw >= ah else "y"
        # perpendicular axis is the SHORT one; we look at agent translations
        # along that axis
        ay0, ax0 = _ctr(a0)
        ay1, ax1 = _ctr(a1)
        a_dperp = (ay1 - ay0) if long_axis == "x" else (ax1 - ax0)
        if abs(a_dperp) <= 0.5:
            continue                              # no perpendicular agent move
        action = acts.get(t) or _d(d, "action")
        for m in movables:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not (b0 and b1):
                continue
            by0, bx0 = _ctr(b0)
            by1, bx1 = _ctr(b1)
            m_dperp = (by1 - by0) if long_axis == "x" else (bx1 - bx0)
            co_moved = abs(m_dperp - a_dperp) <= 1.5 and abs(m_dperp) > 0.5
            # does the agent's bbox at t-1 SPAN PAST the movable on the long axis?
            if long_axis == "x":
                # spans-past means agent's long-axis EXTENT includes the movable's
                # long-axis position (agent's right edge >= movable's center)
                spans = a0[3] >= bx0 - 0.5
            else:
                spans = a0[2] >= by0 - 0.5
            if spans:
                if co_moved:
                    spanned_co_moves += 1
                    actions[action] = actions.get(action, 0) + 1
                    if len(examples) < 6:
                        examples.append({
                            "turn": t, "action": action, "movable": m.name,
                            "agent_dperp": round(a_dperp, 1),
                            "movable_dperp": round(m_dperp, 1),
                            "long_axis": long_axis,
                            "agent_bbox_start": list(a0),
                            "movable_centroid_start": (round(by0, 1), round(bx0, 1))})
                else:
                    spanned_no_moves += 1
            else:
                if co_moved:
                    no_span_co_moves += 1
                else:
                    no_span_no_moves += 1
    support = spanned_co_moves + spanned_no_moves
    if support == 0:
        return {}
    span_rate = spanned_co_moves / support
    no_span_total = no_span_co_moves + no_span_no_moves
    no_span_rate = (no_span_co_moves / no_span_total) if no_span_total else 0.0
    # derive the agent's typical perpendicular axis (the SHORT axis of its
    # bbox) from a late-trace snapshot
    perp_axis = "y"
    late = _bbox_at(agent, getattr(world, "turn", 10 ** 9))
    if late is not None:
        perp_axis = "y" if (late[3] - late[1]) >= (late[2] - late[0]) else "x"
    return {
        "category": "engage_by_body_sweep",
        "support": spanned_co_moves,
        "spanned_co_moves": spanned_co_moves,
        "spanned_no_moves": spanned_no_moves,
        "no_span_co_moves": no_span_co_moves,
        "no_span_no_moves": no_span_no_moves,
        "perpendicular_axis": perp_axis,
        "span_co_move_rate": round(span_rate, 3),
        "no_span_co_move_rate": round(no_span_rate, 3),
        "gap": round(span_rate - no_span_rate, 3),
        "actions": actions,
        "effect": "movable co-displaces with the agent on a perpendicular-axis "
                  "translation when the agent's body bbox already SPANS PAST the "
                  "movable on the long axis (engage-by-body-sweep)",
        "examples": examples,
    }


def describe_engagement_by_body_sweep(rule: dict) -> str:
    if not rule:
        return "(no engage-by-body-sweep rule mined -- agent untracked or no qualifying perpendicular agent moves)"
    acts = ", ".join(f"{a}x{n}" for a, n in
                     sorted(rule["actions"].items(), key=lambda kv: -kv[1]))
    return (
        "MINED ENGAGE-BY-BODY-SWEEP RULE:\n"
        f"  to ENGAGE an off-axis movable, FIRST extend the agent's BODY so its bbox "
        f"SPANS PAST the movable on the long axis; THEN translate the agent along the "
        f"perpendicular axis (the SHORT axis) -- the movable co-displaces with the agent.\n"
        f"  evidence: span+co-move {rule['spanned_co_moves']}/{rule['spanned_co_moves']+rule['spanned_no_moves']} "
        f"({rule['span_co_move_rate']:.0%}) vs no-span+co-move "
        f"{rule['no_span_co_moves']}/{rule['no_span_co_moves']+rule['no_span_no_moves']} "
        f"({rule['no_span_co_move_rate']:.0%}); gap {rule['gap']:+.2f}; "
        f"perpendicular-action: {acts or '(n/a)'}.\n"
        f"  {rule['effect']}.")


def mine_engage_transition_by_body_sweep(world: WorldKnowledge) -> dict:
    """ENGAGEMENT-ONSET variant of body-sweep: isolate the free->attached
    TRANSITION and its true precondition geometry, distinct from CARRY (an
    already-attached movable co-moving).

    The discriminator, evaluated at the START of a perpendicular agent
    translation on which a movable co-displaces:
      - the movable's short-axis center is OUTSIDE the agent's short-axis
        extent (the body has not yet reached it) -> ENGAGE ONSET: the agent
        swept INTO the movable from one side; record that side + offset.
      - the movable's center is INSIDE the agent's body extent -> CARRY:
        already attached; NOT an engagement precondition.

    Why this exists: `mine_engagement_by_body_sweep` counts every spanned
    co-move, which mixes carries (block already within the arm) with true
    engagements. Reading the carry geometry as the engage precondition
    places the agent CO-LEVEL with a free block (a carry pose) instead of
    OFFSET (the engage pose), and the plan attaches nothing -- a real
    planning error. The mined ENGAGE precondition (for planning) is: span
    PAST the movable on the long axis AND start OFFSET on the short axis on
    the side the agent will sweep TOWARD, so the sweep moves the body into
    the still-free movable.

    Returns {category:'engage_transition_by_body_sweep', engage_onsets,
    carry_co_moves, start_side, typical_offset, perpendicular_axis,
    actions, effect, examples} or {} when none observed.
    """
    agent = _agent(world)
    if agent is None:
        return {}
    movables = _movables(world)
    acts = _action_by_turn(world)
    deltas = getattr(world, "deltas_observed", None) or []
    engage_onsets = 0
    carry_co_moves = 0
    side_counts: dict = {}
    offsets: list = []
    actions: dict = {}
    examples: list = []
    perp_axis = "y"
    for d in deltas:
        t = _d(d, "to_turn")
        if t is None:
            continue
        a0, a1 = _bbox_at(agent, t - 1), _bbox_at(agent, t)
        if not (a0 and a1):
            continue
        ah = a0[2] - a0[0]
        aw = a0[3] - a0[1]
        long_axis = "x" if aw >= ah else "y"
        perp_axis = "y" if long_axis == "x" else "x"
        ay0, ax0 = _ctr(a0)
        ay1, ax1 = _ctr(a1)
        a_dperp = (ay1 - ay0) if long_axis == "x" else (ax1 - ax0)
        if abs(a_dperp) <= 0.5:
            continue
        action = acts.get(t) or _d(d, "action")
        for m in movables:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not (b0 and b1):
                continue
            by0, bx0 = _ctr(b0)
            by1, bx1 = _ctr(b1)
            m_dperp = (by1 - by0) if long_axis == "x" else (bx1 - bx0)
            co_moved = abs(m_dperp - a_dperp) <= 1.5 and abs(m_dperp) > 0.5
            if not co_moved:
                continue
            if long_axis == "x":
                spans = a0[3] >= bx0 - 0.5
                inside = (a0[0] - 0.5) <= by0 <= (a0[2] + 0.5)
                offset = by0 - ay0            # block_center_row - agent_center_row
            else:
                spans = a0[2] >= by0 - 0.5
                inside = (a0[1] - 0.5) <= bx0 <= (a0[3] + 0.5)
                offset = bx0 - ax0
            if not spans:
                continue
            if inside:
                carry_co_moves += 1
                continue
            # ENGAGE ONSET: the still-free movable was OUTSIDE the agent body
            # and got swept in. Record which side the agent started on.
            engage_onsets += 1
            offsets.append(round(offset, 1))
            if long_axis == "x":
                side = "agent_below" if offset < 0 else "agent_above"
            else:
                side = "agent_right" if offset < 0 else "agent_left"
            side_counts[side] = side_counts.get(side, 0) + 1
            actions[action] = actions.get(action, 0) + 1
            if len(examples) < 6:
                examples.append({
                    "turn": t, "action": action, "movable": m.name,
                    "start_side": side, "short_axis_offset": round(offset, 1),
                    "agent_dperp": round(a_dperp, 1),
                    "long_axis": long_axis,
                    "agent_bbox_start": list(a0),
                    "movable_centroid_start": (round(by0, 1), round(bx0, 1))})
    if engage_onsets == 0:
        return {}
    start_side = max(side_counts.items(), key=lambda kv: kv[1])[0]
    typical_offset = round(sum(offsets) / len(offsets), 1) if offsets else None
    return {
        "category": "engage_transition_by_body_sweep",
        "engage_onsets": engage_onsets,
        "carry_co_moves": carry_co_moves,
        "start_side": start_side,
        "side_counts": side_counts,
        "typical_offset": typical_offset,
        "perpendicular_axis": perp_axis,
        "actions": actions,
        "effect": ("to ENGAGE a free movable by body-sweep, start the agent "
                   "OFFSET on the short axis (" + start_side +
                   f", ~{abs(typical_offset) if typical_offset else '?'} cells), "
                   "spanning past on the long axis, then sweep perpendicular "
                   "TOWARD it -- co-level placement is a CARRY pose and "
                   "attaches nothing"),
        "examples": examples,
    }


def describe_engage_transition_by_body_sweep(rule: dict) -> str:
    if not rule:
        return ("(no engage-TRANSITION-by-body-sweep mined -- no free->attached "
                "onset observed; only carries, or agent untracked)")
    acts = ", ".join(f"{a}x{n}" for a, n in
                     sorted(rule["actions"].items(), key=lambda kv: -kv[1]))
    return (
        "MINED ENGAGE-TRANSITION-BY-BODY-SWEEP RULE (precondition is the "
        "PRE-sweep geometry, not the carry pose):\n"
        f"  to ENGAGE a FREE movable, start the agent OFFSET on the short axis "
        f"({rule['start_side']}, ~{abs(rule['typical_offset']) if rule['typical_offset'] else '?'} "
        f"cells), body SPANNING PAST it on the long axis; THEN sweep "
        f"perpendicular TOWARD it -- it co-displaces in.\n"
        f"  evidence: {rule['engage_onsets']} engage-onset(s) (block OUTSIDE "
        f"the body, swept in) vs {rule['carry_co_moves']} carry(ies) (block "
        f"already INSIDE the body); perpendicular-action: {acts or '(n/a)'}.\n"
        f"  CAUTION: placing the agent CO-LEVEL with the free block is a CARRY "
        f"pose and attaches nothing -- the offset is the load-bearing "
        f"precondition.")


def mine_grasp_by_pinning(world: WorldKnowledge) -> dict:
    """Mine the GRASP-BY-PINNING rule from playback (data-driven, game-agnostic).

    Signature: a free object cannot be repositioned on its own, but after it is
    driven against an immovable BARRIER (a playfield extreme / fixed structure)
    and pushed further into it, it becomes ATTACHED to the manipulator -- and
    once attached it can be CARRIED in the direction OPPOSITE the push (back
    toward the manipulator's home), which a non-pinned object never does.

    The mine is STATEFUL + CONTRASTIVE: walk each movable's history, set
    `pinned` when it sits at the barrier across a push, and contrast
    P(moves against the push | pinned) vs P(... | not pinned).  A clean gap is
    the rule: pinning enables carry.  Returns a rule record
    {category:'grasp_by_pinning', support, pinned_carry, free_carry,
    precond_facts, effect, examples} or {} when there is no evidence.

    Robotics: pin a part against a fixture wall to secure it on the end-effector,
    then move it.  No game vocabulary -- barrier = at_extreme; carry = motion
    opposite the prior push."""
    xr = _playfield_x_range(world)
    movables = _movables(world)
    acts = _action_by_turn(world)
    deltas = getattr(world, "deltas_observed", None) or []
    pinned_carry = [0, 0]   # [carried-against-push, total moves] while pinned
    free_carry = [0, 0]     # same, while NOT pinned
    examples = []
    actions: dict = {}
    for m in movables:
        pinned = False
        last_push_dir = 0
        for d in deltas:
            t = _d(d, "to_turn")
            if t is None:
                continue
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not b0 or not b1:
                continue
            c0x = (b0[1] + b0[3]) / 2.0
            dx = (b1[1] + b1[3]) / 2.0 - c0x
            at_hi = (xr[1] - b0[3]) <= 2      # pinned against the max-side barrier
            at_lo = (b0[1] - xr[0]) <= 2      # pinned against the min-side barrier
            action = acts.get(t) or _d(d, "action")
            # set pinned when sitting at a barrier (object can't move further into it)
            if at_hi or at_lo:
                pinned = True
                last_push_dir = 1 if at_hi else -1
            if abs(dx) > 0.5:
                against_push = (dx * last_push_dir < 0) if last_push_dir else False
                slot = pinned_carry if pinned else free_carry
                slot[1] += 1
                slot[0] += int(against_push)
                if pinned and against_push:
                    actions[action] = actions.get(action, 0) + 1
                    if len(examples) < 6:
                        examples.append({"turn": t, "movable": m.name,
                                         "action": action, "dx": round(dx, 1),
                                         "carried_after_pin": True})
                # leaving the barrier region while carried keeps it attached;
                # only a structure-insertion (handled elsewhere) clears it.
    if pinned_carry[1] == 0 and free_carry[1] == 0:
        return {}
    pr = pinned_carry[0] / pinned_carry[1] if pinned_carry[1] else 0.0
    fr = free_carry[0] / free_carry[1] if free_carry[1] else 0.0
    return {
        "category": "grasp_by_pinning",
        "support": pinned_carry[0],
        "pinned_carry": pinned_carry, "free_carry": free_carry,
        "gap": round(pr - fr, 3),
        "actions": actions,
        "precond_facts": {"pinned_against_barrier": 1.0},
        "effect": "object becomes attached to the manipulator and can be CARRIED "
                  "opposite the push direction",
        "magnitude": "push further INTO the barrier to attach, then move the "
                     "manipulator to carry the attached object to the target",
        "examples": examples,
    }


def mine_decouple(world: WorldKnowledge) -> dict:
    """Mine the DECOUPLE / separation operator (the inverse of grasp-by-pinning).

    Signature: a manipulator RETRACT releases a movable from the manipulator,
    after which a vertical raise (the manipulator going UP) SEPARATES the
    released movable (it STAYS) from the still-coupled movables (they RISE with
    the manipulator).  This is the operator that lets one block of a pair be
    parked/raised onto a structure while its partner is left below -- the move
    that made red rise onto the yellow rod while orange stayed below it, and the
    move green->cyan needs re-applied to the blue/green pair.

    Evidence = an UP step on which the movables SPLIT (some rise, some stay)
    with a RETRACT in the recent lead-up (the decouple trigger).  Game-agnostic:
    retract = release; vertical move = separate coupled from released.

    TARGET-DIRECTED PRECONDITION (contrastive, data-mined; never authored):
    among the movables involved in a split, the STAYED one is the one at the
    agent's extension EXTREME -- the manipulator retracts toward its base, so
    the movable farthest along the agent's long axis falls outside the new
    reach.  This is what makes the operator TARGET-RESOLVABLE: to park a
    SPECIFIC named block X, position X at that extreme before retracting.  The
    rule records the agent's long axis (`extreme_axis`, x/y), which side is
    the stayed extreme (`extreme_side`, max/min), and the directional purity
    over split events with an observable agent.  Returns {category:'decouple',
    support, trigger, effect, extreme_axis, extreme_side, extreme_purity,
    extreme_support, precond_facts, examples} or {}."""
    acts = _action_by_turn(world)
    deltas = getattr(world, "deltas_observed", None) or []
    movs = _movables(world)
    agent = _agent(world)
    def _cyy(b):
        return (b[0] + b[2]) / 2.0
    def _cxx(b):
        return (b[1] + b[3]) / 2.0
    support = 0
    with_retract = 0
    examples = []
    recent_retract = -99
    # contrastive precondition: across split events with an observable agent,
    # count when stayed sits at the MAX vs MIN side of the agent's long axis.
    extreme_axis_votes = {"x": 0, "y": 0}
    stayed_at_max = 0
    stayed_at_min = 0
    splits_with_agent = 0
    for idx, d in enumerate(deltas):
        t = _d(d, "to_turn")
        if t is None:
            continue
        a = acts.get(t) or _d(d, "action")
        if a == "ACTION3":
            recent_retract = t
        if a != "ACTION1":          # separation is read on a vertical raise
            continue
        rose, stayed = [], []
        rose_bb, stayed_bb = [], []
        for m in movs:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not b0 or not b1:
                continue
            dy = _cyy(b1) - _cyy(b0)
            if dy < -0.5:
                rose.append(m.name); rose_bb.append(b0)
            else:
                stayed.append(m.name); stayed_bb.append(b0)
        if rose and stayed:          # a SPLIT == a movable decoupled from the rest
            support += 1
            had_retract = (t - recent_retract) <= 3
            if had_retract:
                with_retract += 1
            ex_axis = ex_side = None
            agent_bb = _bbox_at(agent, t - 1) if agent is not None else None
            if agent_bb is not None:
                ah = agent_bb[2] - agent_bb[0]
                aw = agent_bb[3] - agent_bb[1]
                ex_axis = "x" if aw >= ah else "y"
                def pos(b):
                    return _cxx(b) if ex_axis == "x" else _cyy(b)
                stayed_p = [pos(b) for b in stayed_bb]
                rose_p = [pos(b) for b in rose_bb]
                splits_with_agent += 1
                extreme_axis_votes[ex_axis] += 1
                if min(stayed_p) > max(rose_p):
                    stayed_at_max += 1; ex_side = "max"
                elif max(stayed_p) < min(rose_p):
                    stayed_at_min += 1; ex_side = "min"
            if len(examples) < 6:
                examples.append({"turn": t,
                                 "rose": [x.replace("block_", "") for x in rose],
                                 "stayed": [x.replace("block_", "") for x in stayed],
                                 "after_retract": had_retract,
                                 "agent_axis": ex_axis,
                                 "stayed_extreme_side": ex_side})
    if support == 0:
        return {}
    extreme_axis = None
    extreme_side = None
    extreme_purity = 0.0
    if splits_with_agent > 0:
        extreme_axis = max(extreme_axis_votes,
                           key=lambda k: extreme_axis_votes[k])
        if stayed_at_max >= stayed_at_min:
            extreme_side = "max"
            extreme_purity = stayed_at_max / splits_with_agent
        else:
            extreme_side = "min"
            extreme_purity = stayed_at_min / splits_with_agent
    return {
        "category": "decouple",
        "support": support,
        "with_retract": with_retract,
        "trigger": "ACTION3 (retract) releases a movable from the manipulator",
        "effect": "on a subsequent UP, the released movable STAYS while still-"
                  "coupled movables RISE -- vertically separating a pair",
        "extreme_axis": extreme_axis,
        "extreme_side": extreme_side,
        "extreme_purity": round(extreme_purity, 3),
        "extreme_support": splits_with_agent,
        "stayed_extreme_counts": {"max": stayed_at_max, "min": stayed_at_min},
        "precond_facts": {
            "stayed_block_at_agent_extension_extreme": round(extreme_purity, 3),
        },
        "examples": examples,
    }


def describe_decouple(rule: dict) -> str:
    if not rule:
        return "(no decouple/separation rule mined -- no split-on-raise evidence)"
    extra = ""
    if rule.get("extreme_side") and rule.get("extreme_axis"):
        extra = (
            f"\n  TARGET PRECONDITION (mined, contrastive): the STAYED movable is "
            f"the one at the {rule['extreme_side']}-{rule['extreme_axis']} extreme "
            f"of the agent's extension axis "
            f"({rule['extreme_purity']:.0%} of {rule['extreme_support']} splits). "
            f"To park a SPECIFIC named block X, first position X at that extreme; "
            f"then retract+raise releases THAT block.")
    return (
        "MINED DECOUPLE / SEPARATION RULE (inverse of grasp-by-pinning):\n"
        f"  {rule['trigger']}; then {rule['effect']}.\n"
        f"  evidence: {rule['support']} raise-steps split the movables "
        f"({rule['with_retract']} right after a retract). "
        f"e.g. {rule['examples'][0] if rule['examples'] else '(none)'}.{extra}\n"
        "  Use: park/raise one block of a pair onto its structure while leaving "
        "the partner below to be carried in next -- the move that completes a "
        "stacked pair (red over orange; blue over green).")


def at_agent_extension_extreme_check(world: WorldKnowledge, block_name: str,
                                     decouple_rule: dict, *,
                                     turn: Optional[int] = None) -> dict:
    """Evaluate the target-directed decouple precondition for `block_name`:
    among the movables candidate for release (every tracked movable visible
    at `turn`), is `block_name` the one at the agent's extension EXTREME
    (axis + side from the mined rule)?  Returns {holds, conjuncts, text} --
    a complete conjunction with per-constituent evidence (substrate-computed,
    SPEC_visual_reasoning_substrate § composed relations).

    Conjuncts: (1) the mined rule supplied an extreme axis + side at usable
    purity; (2) the agent is observable at `turn` (the axis is read from its
    bbox); (3) `block_name` is at that extreme relative to every other movable.
    This is what makes 'park[X]' resolvable: the harness can verify the
    precondition before issuing retract+raise, and a different block at the
    extreme makes it FALSE -- preventing the operator from parking the wrong
    block silently."""
    rec = next((r for r in world.entities.values()
                if (r.name or "") == block_name), None)
    t = turn if turn is not None else getattr(world, "turn", 10 ** 9)
    agent = _agent(world)
    agent_bb = _bbox_at(agent, t) if agent is not None else None
    axis = (decouple_rule or {}).get("extreme_axis")
    side = (decouple_rule or {}).get("extreme_side")
    purity = (decouple_rule or {}).get("extreme_purity", 0.0)
    conj = []
    conj.append({"label": "extreme_rule_mined",
                 "holds": bool(axis and side and purity >= 0.6),
                 "evidence": f"axis={axis} side={side} purity={purity:.0%}"})
    conj.append({"label": "agent_observable",
                 "holds": agent_bb is not None,
                 "evidence": (f"agent={agent.name if agent else None} bbox@t={agent_bb}"
                              if agent is not None else "agent untracked")})
    b = _bbox_at(rec, t) if rec else None
    others = []
    if rec is not None:
        for m in _movables(world):
            if (m.name or "") == block_name:
                continue
            bo = _bbox_at(m, t)
            if bo is not None:
                others.append((m.name, bo))
    if b is None or axis is None or side is None:
        conj.append({"label": "block_at_extreme", "holds": False,
                     "evidence": (f"block {block_name!r} bbox={b!r}; axis={axis};"
                                  f" side={side}")})
    else:
        def pos(bb):
            return ((bb[1] + bb[3]) / 2.0) if axis == "x" else ((bb[0] + bb[2]) / 2.0)
        bp = pos(b)
        if not others:
            holds = True
            ev = (f"{block_name}.pos({axis})={bp:.1f}; only candidate movable "
                  f"-> trivially {side}-extreme")
        else:
            other_ps = [(nm, pos(bb)) for (nm, bb) in others]
            if side == "max":
                holds = all(bp > p for (_, p) in other_ps)
                cmp = ">"
            else:
                holds = all(bp < p for (_, p) in other_ps)
                cmp = "<"
            ev = (f"{block_name}.pos({axis})={bp:.1f} {cmp} "
                  f"{ {nm: round(p,1) for (nm,p) in other_ps} } "
                  f"-> {side}-extreme={'T' if holds else 'F'}")
        conj.append({"label": "block_at_extreme", "holds": bool(holds),
                     "evidence": ev})
    holds = all(c["holds"] for c in conj)
    text = (f"at_agent_extension_extreme({block_name})="
            f"{'TRUE' if holds else 'FALSE'}: "
            + "; ".join(f"{c['label']}={'T' if c['holds'] else 'F'} "
                        f"({c['evidence']})" for c in conj))
    return {"holds": holds, "conjuncts": conj, "text": text}


def describe_grasp_by_pinning(rule: dict) -> str:
    if not rule:
        return "(no grasp-by-pinning rule mined -- no pin->carry evidence)"
    pc, fc = rule["pinned_carry"], rule["free_carry"]
    acts = ", ".join(f"{a}x{n}" for a, n in
                     sorted(rule["actions"].items(), key=lambda kv: -kv[1]))
    return (
        "MINED GRASP-BY-PINNING RULE:\n"
        "  a free object cannot be repositioned alone; drive it against a "
        "BARRIER and push further INTO it to ATTACH it to the manipulator, then "
        "move the manipulator to CARRY it.\n"
        f"  evidence: carried-opposite-to-push happens {pc[0]}/{pc[1]} of moves "
        f"while pinned vs {fc[0]}/{fc[1]} while free (gap {rule['gap']:+.2f}); "
        f"carry actions: {acts or '(n/a)'}.\n"
        f"  {rule['effect']}.")


def spawn_grasp_carry_plan(world: WorldKnowledge, block: str, shaft_struct: dict,
                           grasp_rule: dict):
    """Backward-reasoning chain for repositioning an otherwise-immovable isolated
    object onto a structure, by CONSUMING the mined grasp-by-pinning rule (it is
    NOT hardcoded -- this chain only forms when `grasp_rule` was generated by
    mine_grasp_by_pinning from logs/experimentation).  Chain:
      grasp[obj]  (push into a barrier until attached -- per the mined rule)
      -> carry[obj -> shaft column]  (move the manipulator; the attached obj co-moves)
      -> impale[obj -> shaft]  (insert at the open end).
    Returns [(label, subgoal)] or [] when no grasp rule has been mined."""
    if not grasp_rule:
        return []
    from active_subgoals import commit_subgoal      # noqa: E402
    sn = shaft_struct["name"]
    perp = shaft_struct["perp"]
    chain = []
    grasp = commit_subgoal(
        world, name=f"grasp[{block}]",
        problem_solved=f"{block} is free/isolated and cannot be repositioned alone",
        expected_outcome=f"{block} attached to the manipulator (carryable)",
        acceptance_check=f"attached({block}, manipulator) == true",
        derived_from=("CONSUMES mined grasp_by_pinning rule: push the object INTO "
                      "a barrier (at_extreme) to attach it -- "
                      + (grasp_rule.get('effect') or '')))
    chain.append(("grasp", grasp))
    carry = commit_subgoal(
        world, name=f"carry[{block}->{sn}]",
        problem_solved=f"{block} not yet under the open tip of {sn} (perp={perp:.0f})",
        expected_outcome=f"under_open_end({block},{sn}) holds (carried to column, below tip)",
        acceptance_check=f"under_open_end({block},{sn}).holds == true",
        depends_on=[grasp.subgoal_id],
        derived_from="CONSUMES mined grasp_by_pinning: an ATTACHED object is carried by moving the manipulator")
    chain.append(("carry", carry))
    imp = commit_subgoal(
        world, name=f"impale[{block}->{sn}]",
        problem_solved=f"{block} not impaled on {sn}",
        expected_outcome=f"{block} impaled (membership_gain; HUD+1)",
        acceptance_check="membership_gain(block, carrier_of(shaft))",
        depends_on=[carry.subgoal_id],
        derived_from="mined impale rule: insert at the open end once under_open_end holds")
    chain.append(("impale", imp))
    return chain


def spawn_reach_grasp_carry_plan(world: WorldKnowledge, block: str,
                                 shaft_struct: dict, grasp_rule: dict,
                                 decouple_rule: dict):
    """Backward reasoning that COMPOSES the mined DECOUPLE operator as the
    precondition for grasp-by-pinning -- the step the bare grasp->carry->impale
    chain was missing (and the reason hand-flailing never matched a real plan).

    grasp-by-pinning requires the target to be able to REACH a barrier; if other
    movables lie between it and the barrier, or the shaft's current occupant sits
    where the target must thread, those must be DECOUPLED (cleared / parked high)
    first.  This composes: decouple(blockers) + park(rod-occupant) -> grasp ->
    carry -> impale, with depends_on so the engine enforces clear-before-grasp.
    Only forms when BOTH grasp and decouple operators have been mined (else []).
    Game-agnostic: 'reach a barrier' + 'clear what blocks the reach'."""
    if not grasp_rule or not decouple_rule:
        return []
    from active_subgoals import commit_subgoal      # noqa: E402
    sn = shaft_struct["name"]
    perp = shaft_struct["perp"]
    t = getattr(world, "turn", 10 ** 9)

    def cx(m):
        b = _bbox_at(m, t)
        return None if not b else (b[1] + b[3]) / 2.0

    movs = _movables(world)
    tgt = next((m for m in movs if (m.name or "") == block), None)
    if tgt is None or cx(tgt) is None:
        return []
    tx = cx(tgt)
    right = [m for m in movs if m is not tgt and (cx(m) or -1) > tx]
    left = [m for m in movs if m is not tgt and (cx(m) or 1e9) < tx]
    # grasp toward whichever barrier has fewer blockers (the clearer side)
    if len(right) <= len(left):
        blockers, side = right, "max-side barrier"
    else:
        blockers, side = left, "min-side barrier"
    # the shaft's current occupant (the rod-FIRST already aligned to this shaft)
    occupant = next((m for m in movs if m is not tgt
                     and abs((cx(m) or -99) - perp) <= 2), None)

    chain = []
    pre = []
    for blk in blockers:
        sg = commit_subgoal(
            world, name=f"decouple[{blk.name}]",
            problem_solved=f"{blk.name} lies between {block} and the {side}, blocking the grasp reach",
            expected_outcome=f"{blk.name} cleared from {block}'s row (decoupled / parked on its structure)",
            acceptance_check=f"no movable between {block} and the {side}",
            derived_from="CONSUMES mined decouple rule: retract releases, then a raise separates -> clears the blocker")
        chain.append(("decouple_blocker", sg))
        pre.append(sg.subgoal_id)
    if occupant is not None:
        # TARGET-DIRECTED DECOUPLE PRECONDITION (data-mined, not hardcoded):
        # the mined decouple rule's precond_facts says which axis-extreme the
        # released movable sits at -- so to park THIS named occupant (not
        # whichever block happens to be at the extreme when retract fires),
        # position the occupant AT that extreme first.  Without this gate the
        # harness silently parks the wrong block, which is exactly what made
        # park[blue] unresolvable at 3/4.  Only emitted when the mined rule has
        # a usable extreme axis+side at sufficient purity; else we fall back to
        # the un-gated park and the actor sees the precondition as missing.
        pre_park_id = None
        ex_axis = decouple_rule.get("extreme_axis")
        ex_side = decouple_rule.get("extreme_side")
        ex_pur = decouple_rule.get("extreme_purity") or 0.0
        if ex_axis and ex_side and ex_pur >= 0.6:
            sgp = commit_subgoal(
                world, name=f"position_at_extreme[{occupant.name}]",
                problem_solved=(f"target-directed decouple: {occupant.name} must "
                                f"be at the {ex_side}-{ex_axis} extreme of all "
                                f"attached movables BEFORE retract, so the release "
                                f"hits THIS block (not whichever block is at the "
                                f"extreme by chance)"),
                expected_outcome=(f"at_agent_extension_extreme({occupant.name}) "
                                  f"holds (axis={ex_axis}, side={ex_side})"),
                acceptance_check=(f"at_agent_extension_extreme({occupant.name},"
                                  f"axis={ex_axis},side={ex_side}).holds == true"),
                derived_from=("CONSUMES mined decouple PRECONDITION: stayed block "
                              f"sits at the {ex_side}-{ex_axis} extreme of agent's "
                              f"extension ({ex_pur:.0%} purity); see "
                              "at_agent_extension_extreme_check in playback_mining"))
            chain.append(("position_for_park", sgp))
            pre_park_id = sgp.subgoal_id
        sg = commit_subgoal(
            world, name=f"park[{occupant.name} on {sn}]",
            problem_solved=f"{occupant.name} occupies {sn}; {block} must thread UNDER it",
            expected_outcome=f"{occupant.name} parked HIGH on {sn} (decoupled, persists) -> below-tip slot free",
            acceptance_check=f"within({occupant.name},{sn}) AND parked high",
            depends_on=[pre_park_id] if pre_park_id else None,
            derived_from=("CONSUMES mined decouple rule: park the rod-first high so "
                          "the second threads under it -- target identity enforced "
                          "by position_at_extreme precondition" if pre_park_id else
                          "CONSUMES mined decouple rule: park the rod-first high so the second threads under it"))
        chain.append(("park_occupant", sg))
        pre.append(sg.subgoal_id)
    grasp = commit_subgoal(
        world, name=f"grasp[{block}]",
        problem_solved=f"{block} cannot be repositioned alone; must be attached to the manipulator",
        expected_outcome=f"attached({block}, manipulator) (now reachable: blockers decoupled)",
        acceptance_check=f"attached({block}, manipulator) == true",
        depends_on=pre or None,
        derived_from="CONSUMES mined grasp_by_pinning; PRECONDITION: blockers decoupled so the object can reach a barrier")
    chain.append(("grasp", grasp))
    carry = commit_subgoal(
        world, name=f"carry[{block}->{sn}]",
        problem_solved=f"{block} not yet under the open tip of {sn}",
        expected_outcome=f"under_open_end({block},{sn}) holds (carried to column, below tip)",
        acceptance_check=f"under_open_end({block},{sn}).holds == true",
        depends_on=[grasp.subgoal_id],
        derived_from="CONSUMES mined grasp_by_pinning: an attached object is carried by moving the manipulator")
    chain.append(("carry", carry))
    imp = commit_subgoal(
        world, name=f"impale[{block}->{sn}]",
        problem_solved=f"{block} not impaled on {sn}",
        expected_outcome=f"{block} impaled (membership_gain; HUD+1)",
        acceptance_check="membership_gain(block, carrier_of(shaft))",
        depends_on=[carry.subgoal_id],
        derived_from="mined impale rule: insert at the open end once under_open_end holds")
    chain.append(("impale", imp))
    return chain


def mine_column_control(world: WorldKnowledge, *, row_tol=3.0, adj_tol=8.0) -> dict:
    """Mine HOW a single movable is brought to a chosen position along the free
    (horizontal) axis -- the 'bring it to a specific column' rule.

    Hypothesis tested against playback (NOT authored): a movable translates when
    the AGENT is on its row and adjacent, and it translates AWAY from the agent
    (a PUSH).  The contrast is directional PURITY: of the moves that happen while
    the agent is touching on the row, how many go away from the agent vs toward
    it.  Pure 'away' == a push operator; the agent's own motion direction picks
    the column, and a wall/structure is the natural stop (so the magnitude is
    'move toward it until the object reaches the target column or is blocked').

    Domain-agnostic: 'agent on object's row, adjacent, object moves away' is the
    pusher-behind-pushee relation -- the same primitive in any 2-D game and in
    robotics (end-effector pushing a part across a table to a fixture).  Returns
    a rule record {category:'push', support, away, toward, purity, actions,
    precond_facts, magnitude, examples} or {} when the agent is untracked."""
    ag = _agent(world)
    if ag is None:
        return {}
    structs = _structures(world)
    xr = _playfield_x_range(world)
    movables = _movables(world)
    acts = _action_by_turn(world)
    away = toward = 0
    actions: dict = {}
    examples: list = []
    for d in (getattr(world, "deltas_observed", None) or []):
        t = _d(d, "to_turn")
        if t is None:
            continue
        a0 = _bbox_at(ag, t - 1)
        if not a0:
            continue
        ay, ax = _ctr(a0)
        action = acts.get(t) or _d(d, "action")
        for m in movables:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not b0 or not b1:
                continue
            by, bx = _ctr(b0)
            bx1 = _ctr(b1)[1]
            dx = bx1 - bx
            if abs(dx) <= 0.5:                       # no horizontal move
                continue
            same_row = abs(ay - by) <= row_tol
            adjacent = abs(ax - bx) <= adj_tol
            if not (same_row and adjacent):
                continue
            if (bx - ax) * dx > 0:                   # object moved AWAY from agent
                away += 1
                actions[action] = actions.get(action, 0) + 1
                if len(examples) < 6:
                    examples.append({
                        "turn": t, "action": action, "movable": m.name,
                        "from_col": round(bx, 1), "to_col": round(bx1, 1),
                        "agent_side": "left" if ax < bx else "right"})
            else:                                    # object moved TOWARD agent
                toward += 1
    support = away + toward
    if support == 0:
        return {}
    purity = away / support
    return {
        "category": "push",
        "support": support,
        "away": away, "toward": toward, "purity": round(purity, 3),
        "actions": actions,
        "precond_facts": {
            "agent_on_object_row": 1.0,
            "agent_adjacent_on_axis": 1.0,
            "agent_on_far_side_of_target": 1.0},
        "effect": "object translates one step AWAY from the agent per action",
        "magnitude": "repeat the toward-object action until the object reaches "
                     "the target column or is stopped by a wall/structure",
        "examples": examples,
    }


def describe_column_control(rule: dict) -> str:
    if not rule:
        return "(no column-control rule mined — agent untracked or no pushes seen)"
    acts = ", ".join(f"{a}x{n}" for a, n in
                     sorted(rule["actions"].items(), key=lambda kv: -kv[1]))
    return (
        "MINED COLUMN-CONTROL RULE (push):\n"
        f"  to bring an object to a chosen position along the free axis, put the "
        f"AGENT on the object's row, adjacent on the side OPPOSITE the target, "
        f"then move the agent toward the object.\n"
        f"  effect: {rule['effect']}.\n"
        f"  magnitude: {rule['magnitude']}.\n"
        f"  support {rule['support']} pushes; directional purity "
        f"{rule['purity']:.0%} away-from-agent (toward {rule['toward']}); "
        f"driving actions: {acts or '(n/a)'}.")


def mine_reposition_precondition(world: WorldKnowledge, *, tol: float = 1.5) -> dict:
    """Mine the ATTACH/DETACH POLARITY of repositioning: when a movable changes
    position, is it ATTACHED to the agent (co-moving with the manipulator) or
    DETACHED (moving on its own, e.g. falling)?  Data-driven, contrastive, NOT an
    authored prior -- so a movement plan picks the correct precondition per game.

    Signature: for every step where a movable's centroid moved, contrast moves
    where the AGENT moved by the SAME displacement vector (co-displacement ->
    the movable rode the manipulator = ATTACHED) against moves where it did not
    (the movable moved independently = DETACHED).  The dominant class is the
    repositioning precondition.  Game-agnostic: 'do objects move WITH the agent
    or on their own?' -- attached for a manipulator/skewer game (carry), detached
    for a gravity/independent-physics game.

    Returns {category:'reposition_precondition', requires:'attached'|'detached',
    attached_moves, detached_moves, support, purity} or {} when the agent is
    untracked or nothing moved.  The polarity error this exists to prevent: the
    old spawn_movement_plan hard-coded 'detach to change row', which is correct
    for a gravity game but INVERTED for this substrate (a detached block is inert
    and un-engageable; blocks reposition only WHILE carried)."""
    ag = _agent(world)
    if ag is None:
        return {}
    movs = _movables(world)
    attached = 0
    detached = 0
    for d in (getattr(world, "deltas_observed", None) or []):
        t = _d(d, "to_turn")
        if t is None:
            continue
        a0, a1 = _bbox_at(ag, t - 1), _bbox_at(ag, t)
        if not a0 or not a1:
            continue
        ay0, ax0 = _ctr(a0)
        ay1, ax1 = _ctr(a1)
        da = (ay1 - ay0, ax1 - ax0)
        for m in movs:
            b0, b1 = _bbox_at(m, t - 1), _bbox_at(m, t)
            if not b0 or not b1:
                continue
            by0, bx0 = _ctr(b0)
            by1, bx1 = _ctr(b1)
            db = (by1 - by0, bx1 - bx0)
            if abs(db[0]) <= 0.5 and abs(db[1]) <= 0.5:
                continue                       # the movable did not move
            agent_moved = abs(da[0]) > 0.5 or abs(da[1]) > 0.5
            co_moved = (agent_moved
                        and abs(db[0] - da[0]) <= tol
                        and abs(db[1] - da[1]) <= tol)
            if co_moved:
                attached += 1
            else:
                detached += 1
    support = attached + detached
    if support == 0:
        return {}
    requires = "attached" if attached >= detached else "detached"
    purity = (attached if requires == "attached" else detached) / support
    return {
        "category": "reposition_precondition",
        "requires": requires,
        "attached_moves": attached,
        "detached_moves": detached,
        "support": support,
        "purity": round(purity, 3),
        "effect": ("a movable repositions ONLY while ATTACHED (co-moving with the "
                   "manipulator); a detached movable is inert"
                   if requires == "attached" else
                   "a movable repositions while DETACHED (moves independently of "
                   "the agent, e.g. under gravity)"),
    }


def describe_reposition_precondition(rule: dict) -> str:
    if not rule:
        return "(no reposition-precondition mined -- agent untracked or no movement)"
    return (
        "MINED REPOSITION PRECONDITION (attach/detach polarity, contrastive):\n"
        f"  to reposition a movable it must be {rule['requires'].upper()} "
        f"({rule['attached_moves']} attached-moves vs {rule['detached_moves']} "
        f"detached-moves; purity {rule['purity']:.0%}).\n"
        f"  {rule['effect']}.")


def spawn_movement_plan(world: WorldKnowledge, block: str, shaft_struct: dict,
                        rules: dict, preconds: dict, *, reposition_rule=None):
    """Translate mined movement rules + their contrastive preconditions into a
    GATED subgoal chain in the EXISTING cognitive_os Goal Forest, via
    active_subgoals.commit_subgoal + depends_on.  The engine's is_actionable
    then enforces 'achieve precondition before firing the rule' — no new
    planner.  Chain: <root precondition> -> to_tip_row -> align -> impale.

    ROOT PRECONDITION IS DATA-MINED, not authored.  Whether a block must be
    ENGAGED (attached/carried) or DETACHED (released) to be repositioned is read
    from `mine_reposition_precondition` (co-displacement polarity), NOT assumed.
    The earlier version hard-coded 'detach first' (a robotics gripper prior);
    that is INVERTED for a no-gravity manipulator substrate, where a detached
    block is inert and un-engageable and blocks reposition only WHILE carried.
    Returns [(label, sg)]."""
    from active_subgoals import commit_subgoal      # noqa: E402
    chain = []
    sn = shaft_struct["name"]
    free = tuple(round(x) for x in shaft_struct["free"])
    rp = reposition_rule if reposition_rule is not None else mine_reposition_precondition(world)
    attached_polarity = (rp or {}).get("requires") == "attached"
    # ROOT: establish the MINED repositioning precondition.  attached-polarity
    # (manipulator/no-gravity): ENGAGE the block (it repositions only while
    # carried).  detached-polarity (gravity/independent): DETACH it (it moves on
    # its own once released).  The polarity comes from data, never a hard prior.
    if attached_polarity:
        root = commit_subgoal(
            world, name=f"engage[{block}]",
            problem_solved=(f"{block} must be ENGAGED (carried by the manipulator) to be "
                            f"repositioned; a released block is inert and un-engageable here"),
            expected_outcome=f"attached({block}, manipulator) == true  (carryable to any row/col)",
            acceptance_check=f"attached({block},manipulator) == true",
            derived_from=("CONSUMES mined reposition precondition: movables reposition WHILE "
                          f"ATTACHED (co-move with the agent {rp.get('attached_moves')}/"
                          f"{rp.get('support')}, purity {rp.get('purity',0):.0%}). Engage via the "
                          "mined grasp operator. REPLACES the inverted detach-first prior."))
        chain.append(("engage", root))
        row_note = "mined movement rule: row-change is a CARRY while ATTACHED (block co-moves with the agent)"
    else:
        root = commit_subgoal(
            world, name=f"detach[{block}]",
            problem_solved=f"{block} moves independently once released; detach to let it reposition",
            expected_outcome=f"NOT attached({block}, arm)  (free to move on its own)",
            acceptance_check=f"attached({block},arm) == false",
            derived_from=("CONSUMES mined reposition precondition: movables reposition WHILE "
                          f"DETACHED ({rp.get('detached_moves') if rp else '?'}/"
                          f"{rp.get('support') if rp else '?'} detached-moves)"))
        chain.append(("detach", root))
        row_note = "mined movement rule: row-change occurs while detached (independent motion)"
    # to_tip_row: bring the block to the free-end ROW (below the tip).  DEPENDS
    # ON the root precondition.  Achieves ONLY the below-tip conjunct of
    # under_open_end; NOT impale-ready on its own (column conjunct still
    # unchecked).  Acceptance is that single conjunct, explicitly labelled, so
    # the planner cannot mistake row-reached for under-the-tip.
    to_row = commit_subgoal(
        world, name=f"to_tip_row[{block}]",
        problem_solved=f"{block} not yet at the free-end (below-tip) row of {sn}",
        expected_outcome=f"{block} beyond the open tip of {sn} (below-tip conjunct only)",
        acceptance_check=(f"under_open_end({block},{sn}).beyond_tip == true  "
                          f"(ROW-ONLY sub-conjunct; NOT impale-ready until align "
                          f"adds col_aligned)"),
        depends_on=[root.subgoal_id],
        derived_from=row_note)
    chain.append(("to_tip_row", to_row))
    # align completes under_open_end: BOTH the shaft column AND below the tip.
    # Acceptance is the FULL conjunction, not the column alone.
    align = commit_subgoal(
        world, name=f"align[{block}->{sn}]",
        problem_solved=f"{block} not under the open tip of {sn} (perp={shaft_struct['perp']:.0f})",
        expected_outcome=f"under_open_end({block},{sn}) holds := col_aligned AND beyond_tip",
        acceptance_check=(f"under_open_end({block},{sn}).holds == true  "
                          f"(BOTH col_aligned(x~={shaft_struct['perp']:.0f}) AND beyond_tip)"),
        depends_on=[to_row.subgoal_id],
        derived_from="mined impale precondition: under_open_end = aligned column AND below the tip")
    chain.append(("align", align))
    sg = movement_subgoal(rules, shaft_struct) or {}
    # impale's precondition IS the full under_open_end relation (checkable via
    # under_open_end_check), never a row-only proxy.
    imp = commit_subgoal(
        world, name=f"impale[{block}->{sn}]",
        problem_solved=f"{block} not impaled on {sn}",
        expected_outcome=f"{block} impaled (membership_gain; HUD+1)",
        acceptance_check="membership_gain(block, carrier_of(shaft))",
        notes=f"precondition: under_open_end({block},{sn}).holds == true (complete conjunction)",
        depends_on=[align.subgoal_id],
        derived_from=(f"mined impale rule: action={sg.get('action_hint')} "
                      f"({sg.get('dominant_action')}); precondition=under_open_end "
                      f"(complete conjunction, no single-axis proxy)"))
    chain.append(("impale", imp))
    return chain


def format_movement_rules(rules: dict) -> str:
    if not rules:
        return "(no movement rules mined — no positional history / movements)"
    lines = ["MINED MOVEMENT RULES (data-driven, geometric primitives):"]
    for cat, r in sorted(rules.items(), key=lambda kv: -kv[1]["support"]):
        dom_act = max(r["actions"].items(), key=lambda kv: kv[1])[0] if r["actions"] else "?"
        lines.append(f"  [{cat}] support={r['support']}  dominant_action={dom_act}")
        for k, agg in r["precond_facts"].items():
            if agg.get("n"):
                if agg["vals"]:
                    vs = agg["vals"]
                    lines.append(f"      {k}: mean={sum(vs)/len(vs):.2f} "
                                 f"min={min(vs)} max={max(vs)} (n={agg['n']})")
                else:
                    frac = agg["true"] / agg["n"]
                    lines.append(f"      {k}: {agg['true']}/{agg['n']} "
                                 f"({frac:.0%})")
    return "\n".join(lines)


# ===========================================================================
# Domain-agnostic state predicates + action-effect mining + threat resolution
# ---------------------------------------------------------------------------
# Predicates are generic RELATIONS over (object, structure / workspace) — no
# game vocabulary — so they apply to any game AND to robotics (object in/at/
# beyond a fixture; at a workspace boundary).  Delete-effect mining + threat
# detection give the planner the partial-order reasoning the plain depends_on
# forest lacks: achieving one subgoal can DELETE a condition another must keep
# (the Sussman anomaly).
# ===========================================================================

def state_predicates(world: WorldKnowledge, turn: int) -> set:
    """Generic relational predicates from the perception substrate (positions +
    roles).  (within/beyond_free/beyond_anchored object-vs-structure; at_extreme
    object-vs-workspace).  Robotics maps 1:1: within=object seated in/on a
    fixture, beyond_free=object past a fixture's open end, at_extreme=at a
    workspace bound."""
    structs = _structures(world)
    xlo, xhi = _playfield_x_range(world)
    facts = set()
    for m in _movables(world):
        b = _bbox_at(m, turn)
        if not b:
            continue
        c = _ctr(b)
        for s in structs:
            perp = c[1] if s["axis"] == "y" else c[0]
            if abs(perp - s["perp"]) <= 2:
                lo, hi = s["ext"]
                pos = _axis_pos(c, s["axis"])
                if lo - 1 <= pos <= hi + 1:
                    facts.add(("within", m.name, s["name"]))
                else:
                    fe = _axis_pos(s["free"], s["axis"])
                    an = _axis_pos(s["anchored"], s["axis"])
                    facts.add((("beyond_free" if abs(pos - fe) < abs(pos - an)
                                else "beyond_anchored"), m.name, s["name"]))
        if xhi - b[3] <= 2:
            facts.add(("at_extreme", m.name, "max"))
        if b[1] - xlo <= 2:
            facts.add(("at_extreme", m.name, "min"))
    # attached(m): m co-located on the manipulator's row (moves with the arm).
    # Its negation 'detached' is the precondition for changing m's ROW
    # independently of the arm.  Robotics: attached == grasped.
    agent = next((r for r in world.entities.values()
                  if _role_str(r) == "agent"), None)
    if agent is not None:
        ab = _bbox_at(agent, turn)
        if ab:
            ay = (ab[0] + ab[2]) / 2.0
            for m in _movables(world):
                b = _bbox_at(m, turn)
                if b and abs(((b[0] + b[2]) / 2.0) - ay) <= 2:
                    facts.add(("attached", m.name, "arm"))
    return facts


def mine_action_effects(world: WorldKnowledge) -> dict:
    """Per action: predicate ADD and DELETE effects, diffed from the generic
    predicate set before vs after each step (with support).  Delete-effects are
    what the threat-resolver needs.  Domain-agnostic."""
    acts = _action_by_turn(world)
    eff: dict = {}
    for d in (getattr(world, "deltas_observed", None) or []):
        t = _d(d, "to_turn")
        if t is None:
            continue
        action = acts.get(t) or _d(d, "action")
        before = state_predicates(world, t - 1)
        after = state_predicates(world, t)
        e = eff.setdefault(action, {"add": {}, "del": {}, "support": 0})
        e["support"] += 1
        for (p, *_rest) in (after - before):
            e["add"][p] = e["add"].get(p, 0) + 1
        for (p, *_rest) in (before - after):
            e["del"][p] = e["del"].get(p, 0) + 1
    return eff


def detect_threats(subgoal_actions: dict, protected: set, effects: dict) -> list:
    """Domain-agnostic threat detection (partial-order planning).  A subgoal
    whose action DELETES a protected predicate clobbers it.  `subgoal_actions`:
    {subgoal_name: action}; `protected`: predicate names that must persist;
    `effects`: mine_action_effects output.  Returns threats to resolve (reorder
    / re-parameterize) — the step the depends_on-only forest cannot make."""
    out = []
    for sg, a in subgoal_actions.items():
        dele = (effects.get(a, {}) or {}).get("del", {})
        clob = [p for p in protected if dele.get(p, 0) > 0]
        if clob:
            out.append({"subgoal": sg, "action": a, "clobbers": clob})
    return out


def spawn_protected_movement_plan(world, block, shaft_struct, rules, preconds, *,
                                  protected_facts, effects, clobber_action):
    """Backward-reasoning with THREAT RESOLUTION (the step plain depends_on
    cannot do).  Builds the means-ends chain (detach -> to_tip_row -> align ->
    impale) AND, when an operator on that chain DELETES a condition we must
    keep, PREPENDS a PROTECT subgoal so the protected condition persists across
    the clobbering operator -- then makes the clobbering subgoal depend_on the
    protect subgoal so the engine's is_actionable enforces protect-before-clobber.

    `protected_facts`: set of (pred, holder, structure) tuples to keep (e.g. the
    already-won pair {('within','block_red','carrier_rod_yellow'),
    ('within','block_blue','carrier_rod_cyan')}).
    `effects`: mine_action_effects output.  `clobber_action`: the operator the
    row-change uses (its descent), checked against effects for a `within` delete.

    The PROTECT operator is generic 'make holder's membership in S persist
    independent of the manipulator' = park-high / decouple-onto-structure.
    Robotics: clamp/fixture the already-placed part so moving the arm for the
    next part doesn't dislodge it.  Returns [(label, subgoal)]."""
    from active_subgoals import commit_subgoal      # noqa: E402
    chain = []
    deletes_within = (effects.get(clobber_action, {}) or {}).get("del", {}).get("within", 0) > 0
    protect_ids = []
    # THREAT RESOLUTION: for each protected membership the row-change would drop,
    # spawn a PROTECT subgoal that the row-change must wait on.
    if deletes_within:
        for fact in sorted(protected_facts):
            if not (isinstance(fact, tuple) and len(fact) == 3 and fact[0] == "within"):
                continue
            _, holder, struct = fact
            pr = commit_subgoal(
                world, name=f"protect[{holder} on {struct}]",
                problem_solved=(f"the row-change for {block} moves the manipulator "
                                f"down, which would dislodge {holder} from {struct} "
                                f"(threat: deletes within({holder},{struct}))"),
                expected_outcome=(f"within({holder},{struct}) PERSISTS independent of "
                                  f"the manipulator (parked high / decoupled)"),
                acceptance_check=f"within({holder},{struct}) stays true while arm descends",
                derived_from="threat resolution: protected membership clobbered by row-change operator")
            chain.append(("protect", pr))
            protect_ids.append(pr.subgoal_id)
    # the standard means-ends chain, with row-change gated behind the protects
    base = spawn_movement_plan(world, block, shaft_struct, rules, preconds)
    for label, sg in base:
        if label == "to_tip_row" and protect_ids:
            existing = list(getattr(sg, "depends_on", None) or [])
            try:
                sg.depends_on = existing + protect_ids   # row-change waits on protects
            except Exception:
                pass
        chain.append((label, sg))
    return chain


def under_open_end_check(world: WorldKnowledge, block_name: str,
                         shaft_struct: dict, *, col_tol: float = 2.0,
                         turn: Optional[int] = None) -> dict:
    """Evaluate the COMPLETE impale-ready relation for `block` vs a shaft:
    aligned to the shaft column AND beyond its open tip.  Returns
    {holds, conjuncts:[{label,holds,evidence}], text} -- a complete conjunction
    with per-constituent evidence, NEVER a single-axis proxy.

    Mirrors cognitive_os.world_model.relations.under_open_end; kept inline so
    the planner's acceptance check and the actor-reply guard can evaluate the
    full relation here without a cross-package import.  This is the substrate
    answer that makes 'orange is below the tip' checkable: it reports BOTH the
    column conjunct and the below-tip conjunct, so a row-only claim cannot read
    as satisfied (SPEC_visual_reasoning_substrate § composed relations;
    SPEC_goal_grounding_and_state_diff § acceptance IS the goal predicate)."""
    rec = next((r for r in world.entities.values()
                if (r.name or "") == block_name), None)
    t = turn if turn is not None else getattr(world, "turn", 10 ** 9)
    b = _bbox_at(rec, t) if rec else None
    if not b:
        return {"holds": False, "conjuncts": [],
                "text": f"under_open_end({block_name})=UNKNOWN: block not located"}
    cy, cx = _ctr(b)
    axis = shaft_struct["axis"]
    perp = shaft_struct["perp"]
    fe = _axis_pos(shaft_struct["free"], axis)        # open tip (free end)
    an = _axis_pos(shaft_struct["anchored"], axis)
    open_dir = 1 if fe > an else -1
    obj_axis = cy if axis == "y" else cx
    obj_perp = cx if axis == "y" else cy
    c_align = abs(obj_perp - perp) <= col_tol
    c_beyond = (obj_axis > fe) if open_dir > 0 else (obj_axis < fe)
    rel = ">" if open_dir > 0 else "<"
    conj = [
        {"label": "col_aligned", "holds": bool(c_align),
         "evidence": f"obj.perp={obj_perp:.1f} vs shaft={perp:.1f} tol={col_tol}"},
        {"label": "beyond_tip", "holds": bool(c_beyond),
         "evidence": f"obj.axis={obj_axis:.1f} {rel} tip={fe:.1f}"},
    ]
    holds = all(c["holds"] for c in conj)
    text = (f"under_open_end({block_name},{shaft_struct['name']})="
            f"{'TRUE' if holds else 'FALSE'}: "
            + "; ".join(f"{c['label']}={'T' if c['holds'] else 'F'} "
                        f"({c['evidence']})" for c in conj))
    return {"holds": holds, "conjuncts": conj, "text": text}


def _answer_unskewer_reverts_credit_contiguity(world: WorldKnowledge) -> dict:
    """Framing query: when a block un-skewers, does its win-credit revert, or
    does it latch?  Compares the skewered-state transition (bidirectional,
    from `overlapping`) against the win-relation's completion readout at the
    turns around each un-skewer event.

    Returns {verdict, events, note}. verdict in
    {'reverts','latches','no_evidence','needs_bidirectional_hud'}.

    NOTE: a fully decisive 'latches' requires the HUD/reference completion
    signal to itself be bidirectional (current swatch state per turn). The
    current completion readout is event-latched, so when the skewered state
    drops but the latched done-set does not, we cannot yet distinguish "the
    game latches credit" from "perception failed to see the revert" — we
    report 'needs_bidirectional_hud' rather than a false 'latches'. The
    skewer-side transition itself is reported regardless (it is the part this
    module can answer from existing recordings)."""
    events = unskewer_events(world)
    if not events:
        return {"verdict": "no_evidence", "events": [],
                "note": "no un-skewer transition found in the recorded playback"}
    try:
        from knowledge_crystallization import (        # noqa: E402
            evaluate_win_relation, _identity_key,
        )
    except Exception:
        return {"verdict": "no_evidence", "events": events,
                "note": "crystallization unavailable; reporting raw events only"}
    cands = [h for h in getattr(world, "win_condition_hypotheses", [])
             if getattr(h, "win_relation", None)]
    if not cands:
        return {"verdict": "no_evidence", "events": events,
                "note": "no committed win relation to read completion from"}
    rel = sorted(cands, key=lambda h: getattr(h, "credence", 0.0),
                 reverse=True)[0].win_relation
    ident_of = {rec.name: _identity_key(rec) for rec in world.entities.values()}
    decisive = []
    for ev in events:
        ident = ident_of.get(ev["member"])
        if ident is None:
            continue
        done_before = set((evaluate_win_relation(world, rel, turn=ev["from_turn"])
                           .get("detail") or {}).get("done") or [])
        done_after = set((evaluate_win_relation(world, rel, turn=ev["turn"])
                          .get("detail") or {}).get("done") or [])
        if ident in done_before:
            decisive.append({**ev, "identity": ident,
                             "credit_after": ident in done_after})
    if not decisive:
        return {"verdict": "no_evidence", "events": events,
                "note": "un-skewer events found, but none on a block whose "
                        "credit was set beforehand"}
    if any(not d["credit_after"] for d in decisive):
        return {"verdict": "reverts", "events": decisive,
                "note": "a block's completion credit dropped when it un-skewered"}
    return {"verdict": "needs_bidirectional_hud", "events": decisive,
            "note": "credit stayed set after un-skewer, but the completion "
                    "readout is event-latched; cannot distinguish true latch "
                    "from a missed revert without a bidirectional HUD reader"}


# ===========================================================================
# Playback dedup -- cycle-removal preprocessing
# ---------------------------------------------------------------------------
# Wasted-loop turns (state-revisit cycles) in the recorded playback pollute
# every operator miner here: they inflate dominant-action support counts and
# create spurious "same action different effect" contrastive splits.  Per
# memory/feedback_playback_mining_dedup_first.md, the discipline is:
# CYCLE-REMOVE THE TRACE FIRST, THEN MINE.  This pass does the cleaning.
# Game-agnostic -- the signature is built from perception only (cumulative
# score + entity centroids quantised to the inferred grid); no game vocab.
# ===========================================================================

def playback_signatures(world: WorldKnowledge, *, grid_cells: int = 4) \
        -> list[tuple]:
    """Per-turn state signatures over the recorded playback (perception only).

    Signature at turn T = (cumulative_score_at_T,
                            sorted tuple of (entity_name, cy_cell, cx_cell))
    where cell coords are centroids quantised to `grid_cells`-tick cells (the
    grid the substrate inferred for this game).  Turn list is the deltas'
    turn boundaries -- index 0 = state BEFORE the first recorded delta, index
    k = state AFTER the k-th delta.  No game vocabulary; the signature relies
    on what perception already recorded.
    """
    deltas = list(getattr(world, "deltas_observed", None) or [])
    if not deltas:
        return []
    turns = [_d(deltas[0], "from_turn", 0)]
    for d in deltas:
        turns.append(_d(d, "to_turn", 0))
    # cumulative score per turn: start from world.score backed out via the
    # `score_increased` flags on deltas (each True adds 1 to the running total)
    final_score = getattr(world, "score", 0) or 0
    increases = [int(bool(_d(d, "score_increased", False))) for d in deltas]
    start_score = final_score - sum(increases)
    scores = [start_score]
    for inc in increases:
        scores.append(scores[-1] + inc)

    def _sig_at(turn: int, score: int) -> tuple:
        parts: list = []
        for nm in sorted(world.entities.keys()):
            rec = world.entities[nm]
            b = _bbox_at(rec, turn)
            if b is None:
                continue
            cy = round((b[0] + b[2]) / 2.0 / grid_cells)
            cx = round((b[1] + b[3]) / 2.0 / grid_cells)
            parts.append((nm, cy, cx))
        return (score, tuple(parts))

    return [_sig_at(t, s) for t, s in zip(turns, scores)]


def find_revisit_cycles(sigs: list[tuple]) -> list[tuple]:
    """All revisit pairs (a, b) in the signature sequence with a<b and
    sigs[a]==sigs[b].  The actions taken between state index a and state
    index b form a CYCLE: they brought the observable state back to where
    it started.  Returned sorted by largest span first (biggest savings)."""
    seen: dict = {}
    pairs: list = []
    for i, s in enumerate(sigs):
        if s in seen:
            for prev in seen[s]:
                pairs.append((prev, i))
            seen[s].append(i)
        else:
            seen[s] = [i]
    pairs.sort(key=lambda p: -(p[1] - p[0]))
    return pairs


def _prune_turns(world: WorldKnowledge, drop_turns: set) -> "WorldKnowledge":
    """Return a NEW WorldKnowledge with all records for `drop_turns` removed.

    Deltas, ActionRecords, and per-entity bbox/role/cell history entries whose
    turn lies in `drop_turns` are excised.  Other turn numbers are LEFT
    UNCHANGED -- the miners' `_bbox_at(rec, t)` uses 'latest entry at-or-
    before t', so a gap in the recorded turn sequence implicitly splices the
    cycle: any later delta with from_turn=t_dropped will resolve to the most
    recent surviving bbox, which by signature equals the dropped state.
    """
    import copy
    out = copy.deepcopy(world)
    out.deltas_observed = [d for d in out.deltas_observed
                           if _d(d, "to_turn") not in drop_turns]
    out.actions_taken = [a for a in (getattr(out, "actions_taken", None) or [])
                         if getattr(a, "turn", None) not in drop_turns]
    for rec in out.entities.values():
        if rec.bbox_history:
            rec.bbox_history = [(t, b) for (t, b) in rec.bbox_history
                                if t not in drop_turns]
        if rec.role_history:
            rec.role_history = [(t, r, c) for (t, r, c) in rec.role_history
                                if t not in drop_turns]
        if getattr(rec, "cell_history", None):
            rec.cell_history = [(t, c) for (t, c) in rec.cell_history
                                if t not in drop_turns]
    return out


def dedup_playback(world: WorldKnowledge, *,
                   replay_fn=None,
                   max_iterations: int = 10,
                   grid_cells: int = 4) -> "WorldKnowledge":
    """Cycle-remove wasted loops from the recorded playback (preprocessing
    pass; CALL BEFORE any operator/rule miner -- see module docstring).

    Build perception-only state signatures (`playback_signatures`), find
    state-revisit pairs (`find_revisit_cycles`), and drop the actions taken
    between revisited states.  Iterates so further cycles surfaced by the
    first cut are also removed.  Returns a NEW WorldKnowledge -- input
    untouched.

    Modes (keyword-only):
      replay_fn=None        SIGNATURE-EQUIVALENT (cheap, probabilistic).
                            Trusts that sigs[a]==sigs[b] implies cycle-
                            equivalence.  Visible signatures miss HIDDEN
                            STATE (grip / load / pin / internal lock); a
                            cycle that looks observably trivial may have
                            changed unobservable state.  Use this when the
                            game has no significant hidden state, or as a
                            cheap pre-filter before a verifying pass.
      replay_fn=callable    REPLAY-VERIFIED (slow, accurate).  Called as
                            `replay_fn(list[action_str]) -> end_signature`;
                            a candidate drop is accepted only if the
                            replayed end-signature equals the original.

    Args:
      grid_cells: tick-per-cell quantisation (default 4 = the substrate's
                  cell_ticks for sk48).  Game-agnostic; choose based on
                  world.grid_inference.cell_ticks if available.

    Returns: a cleaned WorldKnowledge.  Idempotent on already-clean input.
    """
    cleaned = world
    # use the world's own grid if available
    gi = getattr(world, "grid_inference", None)
    if gi is not None and getattr(gi, "cell_ticks", None):
        grid_cells = gi.cell_ticks
    last_len = None
    for _it in range(max_iterations):
        sigs = playback_signatures(cleaned, grid_cells=grid_cells)
        if len(sigs) < 2:
            break
        pairs = find_revisit_cycles(sigs)
        if not pairs:
            break
        # turn boundaries (sigs index k corresponds to turn = turns[k])
        deltas = list(cleaned.deltas_observed)
        turns = [_d(deltas[0], "from_turn", 0)] + [_d(d, "to_turn", 0)
                                                    for d in deltas]
        applied = False
        actions_seq = [_d(d, "action") for d in deltas]
        for (a, b) in pairs:
            # candidate: drop actions between sigs[a] and sigs[b]
            kept_actions = actions_seq[:a] + actions_seq[b:]
            if replay_fn is not None:
                try:
                    end_sig = replay_fn(kept_actions)
                except Exception:
                    continue
                if end_sig != sigs[-1]:
                    continue
            drop_turns = set(turns[a + 1: b + 1])
            new_world = _prune_turns(cleaned, drop_turns)
            # safety: only accept if it actually shrunk the trace
            new_len = len(new_world.deltas_observed)
            if last_len is not None and new_len >= last_len:
                continue
            cleaned = new_world
            last_len = new_len
            applied = True
            break
        if not applied:
            break
    return cleaned


def describe_dedup_savings(original: WorldKnowledge,
                           cleaned: WorldKnowledge) -> str:
    """One-line summary of what dedup_playback removed (for trace headers /
    log lines).  Returns '' when nothing was dropped."""
    n0 = len(getattr(original, "deltas_observed", None) or [])
    n1 = len(getattr(cleaned, "deltas_observed", None) or [])
    dropped = n0 - n1
    if dropped <= 0:
        return ""
    pct = (100 * dropped / n0) if n0 else 0
    return (f"DEDUP: removed {dropped} wasted-loop turn(s) from a {n0}-turn "
            f"playback ({pct:.0f}% reduction) before mining")
