"""Render an HTML view of an ExploratoryDriver run.

Reads a run's WorldKnowledge + per-turn artifacts and produces a
single page showing:

  - Run summary (game_id, level, turns, win_state, mechanic counts)
  - Per-turn timeline: gridded frame, action taken, delta observed,
    actor's rationale, new mechanic hypotheses
  - Mechanic-hypothesis dashboard (credence, supports, contradictions,
    promoted status)
  - Entity inventory snapshot (current bbox / role / cell-history-tail)
  - Relationship inventory (with confidence + times_observed)

Game-agnostic: works on any ExploratoryDriver run directory
produced by exploratory_driver.py.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from trace_render_utils import timeago_assets, timeago_banner   # noqa: E402
from perception_substrate import (                              # noqa: E402
    render_turn1_overlay, _add_grid_overlay, DEFAULT_N_TICKS,
)
from PIL import Image                                            # noqa: E402

# Trace-render knobs (separate from the VLM-input renderer).
#
# Strategy: render the gridded composite at a HIGHER native upscale
# than the VLM gets, so the grid lines + labels are drawn ONCE at
# the target display resolution.  No browser-side CSS scaling = no
# pixelation jaggies; no PIL post-upscale = no interpolation
# softening text.  All the readability tuning (line widths, label
# sizes) is dialed in at this native resolution.
TRACE_UPSCALE = 10              # 64-px game → 640-px playfield, ~794 total
TRACE_GRID_LINE_WIDTH = 2       # 2 displayed pixels (was 4)
TRACE_GRID_MINOR_WIDTH = 1      # minor sub-tick lines, fainter
TRACE_TICK_LABEL_SIZE = 19      # ~60% of previous 32-displayed-px size
TRACE_BBOX_LINE_WIDTH = 1       # 1-px bbox border — sits right at the
                                # entity's edge so visible cyan matches
                                # geometric bbox; 2-px borders look like
                                # the bbox extends past small (4-tick)
                                # entities even when coords are correct.
TRACE_BBOX_LABEL_SIZE = 19      # ~30% of previous 64-displayed-px size

# Canonical heading for the per-level entity-analysis block.  This EXACT
# string is part of the canonical-trace contract asserted by
# tools/cos_driver/validate_trace.py (the PostToolUse regression guard) and is
# reused by the autonomous renderer (.tmp/render_autonomous_trace.py), so both
# trace formats share ONE source of truth for this section header.  Changing it
# means updating the guard's REQUIRED list too.
LEVEL_ANALYSIS_HEADING = "Entity analysis at each level start"


CSS = """\
body { font: 13px/1.5 -apple-system, sans-serif; margin: 16px;
       background: #f3f3f5; color: #222; }
h1 { font-size: 22px; margin-bottom: 4px; }
h2 { font-size: 16px; margin-top: 24px; padding: 6px 10px;
     background: #2c3e50; color: white; border-radius: 4px; }
h3 { font-size: 14px; margin: 14px 0 4px 0; color: #003366; }
.subtitle { color: #666; margin-bottom: 14px; }
.summary-table { border-collapse: collapse; font-size: 12px;
                 margin: 12px 0; }
.summary-table th, .summary-table td {
    border: 1px solid #ccc; padding: 4px 8px; text-align: left;
}
.summary-table th { background: #e8e8ec; }
.summary-table td.num { text-align: right;
                         font-variant-numeric: tabular-nums; }
.run-info { background: #fffdf3; border: 1px solid #d9c97a; }
.run-info th { background: #f3ecc4; white-space: nowrap; }
.run-info td { font-family: 'Courier New', monospace; }
.turns { display: grid;
         grid-template-columns:
             repeat(auto-fit, minmax(min(560px, 100%), 1fr));
         gap: 14px; }
.turn { background: white; border: 1px solid #ccc; border-radius: 6px;
        padding: 8px; display: flex; flex-direction: column; gap: 8px;
        min-width: 0; }
.turn h3 { margin: 0; color: #003366; font-size: 15px; }
.turn .meta { color: #666; font-size: 11px;
              font-family: 'Courier New', monospace; }
.turn img { width: 100%; max-width: 100%; height: auto;
            border: 1px solid #999; background: #000; }
.turn .label { font-weight: bold; color: #003366; }
.turn .action {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-family: 'Courier New', monospace; font-size: 12px;
    background: #1976d2; color: white;
}
.turn .action.NONE { background: #888; }
.turn .action.UP, .turn .action.DOWN { background: #6a1b9a; }
.turn .action.LEFT, .turn .action.RIGHT { background: #1976d2; }
.turn .action.CLICK { background: #c2185b; }
.turn .moved { background: #388e3c; color: white;
               padding: 1px 6px; border-radius: 3px; font-size: 11px; }
.turn .stuck { background: #d32f2f; color: white;
               padding: 1px 6px; border-radius: 3px; font-size: 11px; }
.turn .delta { font-size: 12px; }
.turn .delta-summary { font-style: italic; color: #555;
                       margin-top: 4px; }
.hyp-card { background: white; border: 1px solid #ccc; border-radius: 4px;
            padding: 6px 8px; margin: 4px 0; font-size: 12px;
            display: flex; gap: 12px; align-items: center; }
.hyp-card.promoted { background: #e8f5e9; border-color: #43a047; }
.hyp-card .credence { font-weight: bold; font-family: monospace;
                       min-width: 64px; text-align: right; }
.hyp-card .promoted-badge { background: #43a047; color: white;
                              padding: 2px 6px; border-radius: 3px;
                              font-size: 11px; }
.hyp-card .trigger-effect { flex: 1; font-family: monospace; }
.hyp-card .obs-counts { color: #666; font-size: 11px; }
.entity-table, .rel-table { width: 100%; border-collapse: collapse;
                              font-size: 12px; margin-top: 6px; }
.entity-table th, .rel-table th,
.entity-table td, .rel-table td {
    border: 1px solid #ddd; padding: 3px 6px; text-align: left;
}
.entity-table th, .rel-table th { background: #e8e8ec; }
.role-badge {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 11px; color: white;
}
.role-badge.agent { background: #1976d2; }
.role-badge.collectable { background: #388e3c; }
.role-badge.scenery { background: #7b8e8b; }
.role-badge.wall { background: #455a64; }
.role-badge.hud { background: #c2185b; }
.role-badge.trigger_target { background: #fbc02d; color: #222; }
.role-badge.unknown { background: #aaa; }
.subgoal-card { background: white; border: 1px solid #ccc;
                border-radius: 4px; padding: 8px 10px; margin: 6px 0;
                font-size: 12px; }
.subgoal-card.active   { border-left: 4px solid #1976d2; }
.subgoal-card.blocked  { border-left: 4px solid #d32f2f;
                          background: #fff5f5; }
.subgoal-card.achieved { border-left: 4px solid #388e3c;
                          background: #f1f8e9; opacity: 0.85; }
.subgoal-card.abandoned{ border-left: 4px solid #888;
                          background: #f5f5f5; opacity: 0.7; }
.subgoal-card .sg-head { font-weight: bold; color: #003366; }
.subgoal-card .sg-status {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 11px; color: white; margin-left: 6px;
}
.subgoal-card .sg-status.active   { background: #1976d2; }
.subgoal-card .sg-status.blocked  { background: #d32f2f; }
.subgoal-card .sg-status.achieved { background: #388e3c; }
.subgoal-card .sg-status.abandoned{ background: #888; }
.subgoal-card .sg-meta  { color: #666; font-size: 11px;
                           margin-top: 4px; }
.subgoal-card .sg-field { margin-top: 3px; }
.subgoal-card .sg-field b { color: #003366; }
.subgoal-card .sg-notes { white-space: pre-wrap;
                           font-family: 'Courier New', monospace;
                           font-size: 11px; background: #fafafa;
                           padding: 4px 6px; border-radius: 3px;
                           margin-top: 4px; }
.turn .sg-event {
    background: #e3f2fd; border-left: 3px solid #1976d2;
    padding: 4px 8px; margin: 4px 0; font-size: 12px;
}
.turn .sg-event.closed { background: #f1f8e9;
                          border-left-color: #388e3c; }
.turn .sg-event.closed.abandoned { background: #f5f5f5;
                                    border-left-color: #888; }
.turn .sg-event.closed.blocked   { background: #fff5f5;
                                    border-left-color: #d32f2f; }
.turn .goal-subgoal {
    background: #e8f0fe; color: #003366; padding: 1px 6px;
    border-radius: 3px; font-family: 'Courier New', monospace;
    font-size: 11px;
}
.wc-card { background: white; border: 1px solid #ccc;
           border-radius: 4px; padding: 8px 10px; margin: 6px 0;
           font-size: 12px; border-left: 4px solid #888; }
.wc-card.promoted { border-left-color: #388e3c;
                     background: #f1f8e9; }
.wc-card .wc-head { font-weight: bold; color: #003366; }
.wc-card .wc-badge {
    display: inline-block; padding: 1px 6px; border-radius: 3px;
    font-size: 11px; color: white; margin-left: 6px;
}
.wc-card .wc-badge.promoted { background: #388e3c; }
.wc-card .wc-badge.tentative { background: #888; }
.wc-card .wc-meta { color: #666; font-size: 11px;
                     margin-top: 4px; }
.wc-card .wc-notes { white-space: pre-wrap;
                     font-family: 'Courier New', monospace;
                     font-size: 11px; background: #fafafa;
                     padding: 4px 6px; border-radius: 3px;
                     margin-top: 4px; }
.turn .viz-event {
    background: #fff3e0; border-left: 3px solid #fb8c00;
    padding: 4px 8px; margin: 4px 0; font-size: 12px;
    font-family: 'Courier New', monospace;
}
.turn .anim-event {
    background: #e3f2fd; border-left: 3px solid #1e88e5;
    padding: 4px 8px; margin: 4px 0; font-size: 12px;
    font-family: 'Courier New', monospace;
}
.turn .anim-move { margin: 2px 0 2px 6px; }
.turn .anim-move .swatch {
    display: inline-block; width: 10px; height: 10px; margin-right: 4px;
    border: 1px solid #888; vertical-align: middle;
}
.turn .anim-filmstrip { margin: 4px 0; }
.turn .anim-filmstrip img { max-width: 100%; border: 1px solid #ccc; }
"""


def _action_html(action: str) -> str:
    return f'<span class="action {action}">{action}</span>'


def _format_win_condition_section(wk: dict) -> str:
    """Top-of-page dashboard for the actor's win-condition
    hypotheses.  These are the foundation every delivery subgoal
    stands on; surfacing them makes their credence + evidence
    legible at a glance."""
    hyps = wk.get("win_condition_hypotheses") or []
    if not hyps:
        return ('<h2>Win-condition hypotheses</h2>'
                '<p><i>(none recorded — actor has not yet '
                'committed any hypothesis about what triggers '
                'score / lc / win_state changes)</i></p>')
    # Sort by credence desc, then by created_at_turn desc
    ranked = sorted(
        hyps,
        key=lambda h: (-float(h.get("credence", 0)),
                        -int(h.get("created_at_turn", 0))),
    )
    parts = ['<h2>Win-condition hypotheses</h2>']
    parts.append(
        '<div class="subtitle">'
        f'{len(ranked)} recorded, '
        f'{sum(1 for h in ranked if h.get("promoted"))} promoted'
        '</div>'
    )
    for h in ranked:
        cred = float(h.get("credence", 0))
        prom = h.get("promoted", False)
        klass = "wc-card promoted" if prom else "wc-card"
        badge = ('<span class="wc-badge promoted">PROMOTED</span>'
                 if prom else
                 f'<span class="wc-badge tentative">tentative '
                 f'(c={cred:.2f})</span>')
        parts.append(f'<div class="{klass}">')
        parts.append(
            f'<div class="wc-head">{_clean_prose(h.get("description","(none)"))} '
            f'{badge}</div>'
        )
        parts.append(
            f'<div class="wc-meta">'
            f'id=<code>{_esc(h.get("hypothesis_id","?"))}</code> '
            f'&middot; committed t{h.get("created_at_turn","?")} '
            f'&middot; +{len(h.get("supporting_observations") or [])} '
            f'support / '
            f'-{len(h.get("contradicting_observations") or [])} '
            f'contradict'
            f'</div>'
        )
        if h.get("notes"):
            parts.append(f'<div class="wc-notes">{_clean_prose(h["notes"])}</div>')
        parts.append('</div>')
    return "".join(parts)


def _index_subgoals(wk: dict) -> dict:
    """Index the world's active_subgoals list by id, by
    created_at_turn, and by closed_at_turn so per-turn rendering
    can look up lifecycle events O(1)."""
    sgs = wk.get("active_subgoals") or []
    by_id: dict[str, dict] = {}
    by_created: dict[int, list[dict]] = {}
    by_closed:  dict[int, list[dict]] = {}
    for sg in sgs:
        by_id[sg["subgoal_id"]] = sg
        c = sg.get("created_at_turn")
        if c is not None:
            by_created.setdefault(int(c), []).append(sg)
        x = sg.get("closed_at_turn")
        if x is not None:
            by_closed.setdefault(int(x), []).append(sg)
    return {"by_id": by_id, "by_created": by_created,
            "by_closed": by_closed, "all": sgs}


def _format_subgoal_card(sg: dict) -> str:
    """Render one ActiveSubgoal as a card in the dashboard."""
    status = sg.get("status") or "active"
    parts = [
        f'<div class="subgoal-card {status}">',
        f'<div class="sg-head">{_esc(sg.get("name","(unnamed)"))} '
        f'<span class="sg-status {_esc(status)}">{_esc(status)}</span></div>',
        f'<div class="sg-meta">id=<code>{sg["subgoal_id"]}</code> '
        f'&middot; committed t{sg.get("created_at_turn","?")}',
    ]
    closed = sg.get("closed_at_turn")
    if closed is not None:
        parts.append(f' &middot; closed t{closed}')
    parent = sg.get("parent_id")
    if parent:
        parts.append(f' &middot; parent=<code>{parent}</code>')
    related = sg.get("related_subroutine_id")
    if related:
        parts.append(f' &middot; applying subroutine='
                     f'<code>{related}</code>')
    parts.append('</div>')
    ps = sg.get("problem_solved") or ""
    if ps:
        parts.append(
            f'<div class="sg-field"><b>Problem solved:</b> {_clean_prose(ps)}</div>'
        )
    eo = sg.get("expected_outcome") or ""
    if eo:
        parts.append(
            f'<div class="sg-field"><b>Expected outcome:</b> {_clean_prose(eo)}</div>'
        )
    notes = sg.get("notes") or ""
    if notes:
        parts.append(f'<div class="sg-notes">{_clean_prose(notes)}</div>')
    parts.append('</div>')
    return "".join(parts)


def _format_subgoals_section(wk: dict) -> str:
    """Top-of-page subgoals dashboard — active and blocked first,
    then closed (achieved/abandoned) for provenance."""
    sgs = wk.get("active_subgoals") or []
    if not sgs:
        return ('<h2>Active subgoals</h2>'
                '<p><i>(no subgoals committed this run)</i></p>')

    by_status: dict[str, list[dict]] = {}
    for sg in sgs:
        by_status.setdefault(sg.get("status") or "active", []
                              ).append(sg)

    parts = ['<h2>Active subgoals</h2>']
    counts = (
        f'{len(by_status.get("active",[]))} active, '
        f'{len(by_status.get("blocked",[]))} blocked, '
        f'{len(by_status.get("achieved",[]))} achieved, '
        f'{len(by_status.get("abandoned",[]))} abandoned'
    )
    parts.append(f'<div class="subtitle">{counts}</div>')

    # active + blocked first (in commit order)
    open_sgs = [sg for sg in sgs
                if sg.get("status") in ("active", "blocked")]
    if open_sgs:
        parts.append('<h3>Open</h3>')
        for sg in sorted(open_sgs,
                          key=lambda s: s.get("created_at_turn", 0)):
            parts.append(_format_subgoal_card(sg))

    closed_sgs = [sg for sg in sgs
                  if sg.get("status") in ("achieved", "abandoned")]
    if closed_sgs:
        parts.append(
            '<h3>Closed (kept for provenance)</h3>'
        )
        for sg in sorted(closed_sgs,
                          key=lambda s: s.get("closed_at_turn") or 0):
            parts.append(_format_subgoal_card(sg))

    return "".join(parts)


def _goal_id_html(goal_id: str, subgoals_by_id: dict) -> str:
    """Format an ActionRecord's goal_id.  When it's
    `subgoal:<id>`, look up the subgoal's name and link to its
    dashboard card.  Otherwise just show the raw id."""
    if not goal_id:
        return '(none)'
    if goal_id.startswith("subgoal:"):
        sgid = goal_id[len("subgoal:"):]
        sg = subgoals_by_id.get(sgid)
        if sg:
            name = sg.get("name", "(unnamed)")
            return (f'<span class="goal-subgoal" '
                    f'title="{sgid}">subgoal: {name}</span>')
    return goal_id


def _moved_html(moved: bool) -> str:
    return ('<span class="moved">moved</span>' if moved
            else '<span class="stuck">blocked</span>')


def _render_trace_grid(frame_path: Path, out: Path) -> bool:
    """Render a raw game frame with the trace-readable grid overlay
    (no entity bboxes — those need per-turn perception data).  Saves
    to ``out``.  Returns True on success, False on failure (caller
    should fall back to the raw frame)."""
    try:
        img = Image.open(frame_path).convert("RGB")
        gridded, _, _ = _add_grid_overlay(
            img, n_ticks=DEFAULT_N_TICKS,
            upscale=TRACE_UPSCALE,
            line_width_major=TRACE_GRID_LINE_WIDTH,
            line_width_minor=TRACE_GRID_MINOR_WIDTH,
            label_size_override=TRACE_TICK_LABEL_SIZE,
        )
        gridded.save(out)
        return True
    except (OSError, ValueError) as e:
        print(f"[render] grid overlay failed for {frame_path}: {e}",
              file=sys.stderr)
        return False


def _format_turn(turn_n: int, work_dir: Path, frame_dir: Path,
                  actions_by_turn: dict[int, dict],
                  deltas_by_to_turn: dict[int, dict],
                  notes_by_turn: dict[int, str],
                  subgoals_idx: Optional[dict] = None,
                  level_by_turn: Optional[dict] = None) -> str:
    _lvl = (level_by_turn or {}).get(turn_n)
    _hdr = (f'Turn {turn_n}, lc={_lvl}' if _lvl is not None
            else f'Turn {turn_n}')
    parts = [f'<div class="turn"><h3>{_hdr}</h3>']
    if subgoals_idx is None:
        subgoals_idx = {"by_id": {}, "by_created": {},
                         "by_closed": {}, "all": []}

    # frame image — render with the trace-readable grid overlay
    # (same grid the VLM saw, but drawn at TRACE_UPSCALE so labels
    # + lines are crisp at natural display size).  Fall back to a
    # raw copy if overlay rendering fails.
    frame_path = frame_dir / f"turn_{turn_n:03d}.png"
    if frame_path.exists():
        views_dir = work_dir / "views"
        views_dir.mkdir(parents=True, exist_ok=True)
        local = views_dir / f"frame_{turn_n:03d}.png"
        if local.exists():
            local.unlink()
        if not _render_trace_grid(frame_path, local):
            shutil.copy(frame_path, local)
        parts.append(f'<img src="views/{local.name}" alt="turn {turn_n} frame">')

    # action taken (the actor's decision BEFORE this turn's frame)
    act = actions_by_turn.get(turn_n)
    if act is not None:
        goal_disp = _goal_id_html(
            act.get("goal_id") or "", subgoals_idx["by_id"],
        )
        parts.append(
            f'<div class="meta">'
            f'<span class="label">actor chose</span> {_action_html(act["action"])} '
            f'<span class="label">plan</span> {act["actor_chose_from"]} '
            f'<span class="label">goal</span> {goal_disp}'
            f'</div>'
        )
        # Subgoals committed THIS turn (the actor opened a new
        # commitment as part of this decision).  Render the full
        # card so the trace shows the actor's framing.
        for sg in subgoals_idx["by_created"].get(turn_n, []):
            parts.append(
                f'<div class="sg-event">'
                f'<span class="label">COMMITTED subgoal:</span> '
                f'{_esc(sg.get("name","(unnamed)"))} '
                f'<code style="color:#666;font-size:11px;">'
                f'{sg["subgoal_id"]}</code>'
                f'</div>'
            )
        # Show the FULL rationale (not truncated) — the strategy
        # layer's reasoning often contains the explicit plan steps,
        # truncating loses that signal.
        parts.append(
            f'<div class="meta">rationale: {_clean_prose(act.get("rationale",""))}</div>'
        )
        # Multi-step plan, if the planner produced one.  First step
        # is the action above; subsequent steps are upcoming.  Shown
        # as a chain so the user sees the planner's INTENT, not just
        # the next-action decision.
        plan = act.get("full_plan_actions") or []
        if len(plan) > 1:
            chain = " &rarr; ".join(
                _action_html(a) for a in plan
            )
            parts.append(
                f'<div class="meta">'
                f'<span class="label">planner full plan ({len(plan)} steps):</span> '
                f'{chain}'
                f'</div>'
            )

    # delta observed (going from turn_n -> turn_n+1)
    delta = deltas_by_to_turn.get(turn_n + 1)
    if delta is not None:
        parts.append(
            f'<div class="delta">'
            f'<span class="label">delta to t{turn_n + 1}:</span> '
            f'action={_action_html(delta["action"])} '
            f'inferred={_action_html(delta.get("inferred_action") or "?")} '
            f'{_moved_html(delta["agent_moved"])} '
            f'agent_cell={delta.get("agent_new_cell")} '
            f'changes=+{len(delta.get("entities_appeared") or [])}'
            f'/-{len(delta.get("entities_disappeared") or [])}'
            f'/~{len(delta.get("entities_changed") or [])}'
            f'</div>'
        )
        # Full delta summary (was truncated to 300 chars) — captures
        # what the perception layer concluded about the action's
        # effect, including any newly-grounded mechanic.
        if delta.get("summary"):
            parts.append(
                f'<div class="delta-summary">{_clean_prose(delta["summary"])}</div>'
            )

    # Substrate-computed visual events (pixel-diff inside watched
    # bboxes, default role=hud).  Surfaced as evidence chips so the
    # actor and the user both see which indicators changed.
    if delta is not None:
        viz_events = delta.get("visual_events") or []
        for ev in viz_events:
            ent = ev.get("entity", "?")
            frac = ev.get("pixel_diff_fraction", 0)
            try:
                frac_pct = f"{float(frac) * 100:.1f}%"
            except Exception:
                frac_pct = str(frac)
            parts.append(
                f'<div class="viz-event">'
                f'<span class="label">VISUAL EVENT:</span> '
                f'<code>{ent}</code> internal pixel diff '
                f'<b>{frac_pct}</b>'
                f'</div>'
            )

    # Substrate ENTITY-LEVEL animation movements (Layer-3/4): which entity
    # moved / appeared / vanished ACROSS the action's animation sub-frames,
    # plus the filmstrip.  This is what the entity analysis of the animation
    # FOUND — surfaced here so the trace shows it, not just the prompt.
    if delta is not None:
        anim_events = delta.get("animation_events") or []
        if anim_events:
            rows = []
            for ev in anim_events[:8]:
                verbs = ", ".join(ev.get("verbs") or [])
                col = _esc(str(ev.get("colour_hex", "#888")))
                rows.append(
                    f'<div class="anim-move">'
                    f'<span class="swatch" style="background:{col}"></span>'
                    f'<code>{col} ~{_esc(str(ev.get("size", "?")))}</code> '
                    f'<b>{_esc(verbs)}</b> '
                    f'{_esc(str(ev.get("from")))} &rarr; {_esc(str(ev.get("to")))} '
                    f'net {_esc(str(ev.get("net")))} '
                    f'<span class="meta">f{ev.get("first_frame")}..{ev.get("last_frame")}; '
                    f'rows {_esc(str(ev.get("path_span_rows")))} '
                    f'cols {_esc(str(ev.get("path_span_cols")))}</span>'
                    f'</div>'
                )
            parts.append(
                '<div class="anim-event">'
                '<span class="label">ANIMATION — substrate entity movements '
                '(facts; the VLM interprets the correlation):</span>'
                + "".join(rows) + '</div>'
            )
        # PER-FRAME ENTITY ANALYSIS: each sub-frame with the substrate's detected
        # entity boxes drawn on it -- the verification view (is the per-frame
        # analysis correct?), shown ABOVE the raw filmstrip for direct comparison.
        # A filmstrip belongs to THIS run only if it is at least as fresh as that
        # turn's frame (frames are regenerated every run).  A stale prior-run
        # filmstrip this turn did NOT regenerate is older -> omit it, so a previous
        # run's (often a DIFFERENT game's) animation can't show through the reused
        # work dir.  See trace_sanity.check_referenced_images + test_trace_sanity.
        def _film_fresh(p):
            # Reference RUN START = earliest prompt.md across the turn dirs (this run
            # rewrites every turn's prompt).  A prior-run leftover filmstrip (this turn
            # produced no animation, so the reused dir kept the old one) is older than
            # run start by a whole run gap -> omit it.  This run's OWN filmstrips are
            # all written after run start -> kept.  (Run-level, not per-turn: a turn's
            # prompt and its filmstrip are written moments apart, so a per-turn compare
            # false-drops; and play-time filmstrips are older than render-time views/,
            # so don't compare to those either.)
            try:
                _starts = [q.stat().st_mtime for q in work_dir.glob("turn_*/prompt.md")]
                if not _starts:
                    return True                       # can't measure -> don't hide
                return Path(str(p)).stat().st_mtime >= min(_starts) - 120
            except Exception:
                return True
        ea = delta.get("animation_entities_filmstrip")
        if not ea:
            # Robustness: the field may not have persisted, but the PNG is the
            # source of truth -- fall back to it on disk (the delta is to turn_n+1,
            # so its animation lives in that turn's dir).
            _cand = work_dir / f"turn_{turn_n + 1:03d}" / "animation_entities_filmstrip.png"
            if _cand.exists():
                ea = str(_cand)
        if ea and _film_fresh(ea):
            _eas = str(ea).replace("\\", "/")
            if not _eas.startswith("file:"):
                _eas = "file:///" + _eas.lstrip("/")
            parts.append(
                f'<div class="anim-filmstrip">'
                f'<span class="label">PER-FRAME ENTITY ANALYSIS (substrate '
                f'detection per sub-frame; red boxes = detected entities; verify '
                f'these are correct):</span><br>'
                f'<img src="{_eas}" alt="per-frame entity analysis" loading="lazy"></div>'
            )
        fs = delta.get("animation_filmstrip")
        if fs and _film_fresh(fs):
            _src = str(fs).replace("\\", "/")
            if not _src.startswith("file:"):
                _src = "file:///" + _src.lstrip("/")
            parts.append(
                f'<div class="anim-filmstrip">'
                f'<span class="label">RAW animation sub-frames:</span><br>'
                f'<img src="{_src}" alt="animation filmstrip" loading="lazy"></div>'
            )

    # Subgoals CLOSED this turn — the actor reported
    # achieved/abandoned (or set status to blocked) at this turn.
    # Render after the delta because closures often reflect
    # post-action state.
    for sg in subgoals_idx["by_closed"].get(turn_n, []):
        status = sg.get("status") or "closed"
        parts.append(
            f'<div class="sg-event closed {_esc(status)}">'
            f'<span class="label">{_esc(status.upper())} subgoal:</span> '
            f'{_esc(sg.get("name","(unnamed)"))} '
            f'<code style="color:#666;font-size:11px;">'
            f'{sg["subgoal_id"]}</code>'
            f'</div>'
        )

    # Perception's `overall_notes` for this turn — the narrative
    # synthesis from the perception layer.  Often contains the
    # forward-looking plan (e.g. "next: extend rope to col 42").
    notes = notes_by_turn.get(turn_n)
    if notes:
        parts.append(
            f'<div class="meta" style="background:#fffde7;'
            f'border-left:3px solid #fbc02d;padding:6px 8px;'
            f'margin-top:6px;">'
            f'<span class="label">overall_notes:</span> {_clean_prose(notes)}'
            f'</div>'
        )

    parts.append('</div>')
    return "".join(parts)


def _format_hypothesis(h: dict) -> str:
    promoted = h.get("promoted")
    cls = "hyp-card promoted" if promoted else "hyp-card"
    badge = ('<span class="promoted-badge">PROMOTED</span>'
             if promoted else '')
    return (
        f'<div class="{cls}">'
        f'<span class="credence">{h["credence"]:.2f}</span>'
        f'{badge}'
        f'<span class="trigger-effect">'
        f'<b>trigger:</b> {_esc(h["trigger"])}<br>'
        f'<b>effect:</b> {_esc(h["effect"])}'
        f'</span>'
        f'<span class="obs-counts">'
        f'+{len(h.get("supporting_observations") or [])} '
        f'-{len(h.get("contradicting_observations") or [])}'
        f'</span>'
        f'</div>'
    )


def _format_entity_row(name: str, rec: dict, current_turn: int) -> str:
    role = ""
    if rec.get("role_history"):
        role = rec["role_history"][-1][1]
    bbox = ""
    if rec.get("bbox_history"):
        bbox = str(rec["bbox_history"][-1][1])
    cell = ""
    if rec.get("cell_history"):
        cell = str(rec["cell_history"][-1][1])
    still = "✓" if rec.get("last_seen_turn") == current_turn else "✗"
    return (
        f'<tr><td>{_esc(name)}</td>'
        f'<td><span class="role-badge {_esc(role or "unknown")}">{_esc(role)}</span></td>'
        f'<td>{_esc(bbox)}</td><td>{_esc(cell)}</td>'
        f'<td>t{rec.get("first_seen_turn")}-t{rec.get("last_seen_turn")} {still}</td></tr>'
    )


def _format_relationship_row(r: dict) -> str:
    return (
        f'<tr><td>{_esc(r["from_name"])}</td>'
        f'<td><b>{_esc(r["relation"])}</b></td>'
        f'<td>{_esc(r["to_name"])}</td>'
        f'<td>{_esc(str(r.get("evidence", "") or ""))}</td>'
        f'<td>{r["confidence"]:.2f}</td>'
        f'<td>×{r["times_observed"]}</td></tr>'
    )


def _load_latest_strategy(work_dir: Path) -> tuple[int, dict] | None:
    """Find the most recent turn_NNN with a consumed strategy
    reply and return (turn_n, parsed_reply).  None if no strategy
    calls have been made yet."""
    turn_dirs = sorted([d for d in work_dir.iterdir()
                         if d.is_dir() and d.name.startswith("turn_")],
                        reverse=True)
    for td in turn_dirs:
        for cand in (td / "strategy_reply.consumed.txt",
                      td / "strategy_reply.txt"):
            if cand.exists():
                body = cand.read_text(encoding="utf-8").strip()
                if body.startswith("```"):
                    body = body.split("\n", 1)[1] if "\n" in body else body
                    body = body.rsplit("```", 1)[0]
                for suffix in ("", "}", '"}'):
                    try:
                        d = json.loads(body + suffix)
                        try:
                            n = int(td.name.split("_")[1])
                        except (ValueError, IndexError):
                            n = -1
                        return n, d
                    except json.JSONDecodeError:
                        continue
    return None


def _format_plan_section(work_dir: Path, wk: dict) -> str:
    """Show the system's CURRENT PLAN: latest strategy decision +
    hypothesis under test + recent action outcomes + what the
    actor would do next based on accumulated knowledge.

    Game-agnostic: reads from the symbolic world model and the
    most recent strategy_reply, no game-specific assumptions.
    """
    parts = ['<h2>System plan (current intent + hypothesis under test)</h2>']

    # GAME PURPOSE — the perception layer's evolving best guess of
    # what the player is trying to accomplish.  This is the system's
    # high-level intent, distilled across all turns so far.  Shown
    # at the top of the plan section because it frames every
    # decision below.
    gp = wk.get("game_purpose_guess")
    gt = wk.get("game_type_guess")
    if gp or gt:
        parts.append(
            '<table class="summary-table">'
            + (f'<tr><th>game_type (latest)</th><td>{_clean_prose(gt)}</td></tr>' if gt else '')
            + (f'<tr><th>game_purpose (latest)</th><td><b>{_clean_prose(gp)}</b></td></tr>'
               if gp else '')
            + '</table>'
        )

    latest = _load_latest_strategy(work_dir)
    if latest is None:
        parts.append(
            '<p><i>No strategy calls yet '
            '(driver was launched with --no-strategy or has not '
            'reached the strategy phase).  '
            'The mechanical actor will choose the next action '
            'using BFS over goal_candidate_cells alone.</i></p>'
        )
    else:
        turn_n, reply = latest
        endorsed = reply.get("endorsed_action", "?")
        rationale = reply.get("rationale", "")
        hyp = reply.get("testing_hypothesis")
        conf = reply.get("confidence", "")
        parts.append(
            '<table class="summary-table">'
            f'<tr><th>latest strategy turn</th><td>{turn_n}</td></tr>'
            f'<tr><th>endorsed_action</th>'
            f'<td><span class="action {endorsed.split(":")[0]}">{endorsed}</span></td></tr>'
            f'<tr><th>rationale</th><td>{_clean_prose(rationale)}</td></tr>'
            f'<tr><th>testing_hypothesis</th><td><code>{_esc(hyp)}</code></td></tr>'
            f'<tr><th>strategy confidence</th><td>{_esc(conf)}</td></tr>'
            '</table>'
        )

    # Show the active high-credence mechanic rules that the
    # planner would use as priors
    promoted = [h for h in (wk.get("mechanic_hypotheses") or [])
                if h.get("credence", 0) >= 0.5]
    if promoted:
        parts.append('<h3>Mechanic rules the planner trusts (credence ≥ 0.5)</h3>')
        rows = []
        for h in sorted(promoted, key=lambda h: -h["credence"]):
            badge = ('<span class="promoted-badge">PROMOTED</span>'
                     if h.get("promoted") else '')
            rows.append(
                f'<tr><td>{h["credence"]:.2f}</td><td>{badge}</td>'
                f'<td><code>{_esc(h["trigger"])}</code></td>'
                f'<td><code>{_esc(h["effect"])}</code></td></tr>'
            )
        parts.append(
            '<table class="entity-table"><tr><th>credence</th>'
            '<th>status</th><th>trigger</th><th>effect</th></tr>'
            + "".join(rows) + '</table>'
        )

    # Show last 5 deltas as quick context for the plan
    deltas = wk.get("deltas_observed") or []
    if deltas:
        parts.append('<h3>Recent action outcomes (last 5)</h3>')
        rows = []
        for d in deltas[-5:]:
            moved = ('<span class="moved">moved</span>'
                     if d.get("agent_moved")
                     else '<span class="stuck">stuck</span>')
            rows.append(
                f'<tr><td>t{d["from_turn"]}→t{d["to_turn"]}</td>'
                f'<td>{_esc(d["action"])}</td>'
                f'<td>{_esc(d.get("inferred_action","?"))}</td>'
                f'<td>{moved}</td>'
                f'<td>{_esc(d.get("agent_new_cell",""))}</td>'
                f'<td>{_clean_prose(d.get("summary","") or "", maxlen=120)}</td></tr>'
            )
        parts.append(
            '<table class="entity-table">'
            '<tr><th>turn</th><th>requested</th>'
            '<th>inferred</th><th>outcome</th>'
            '<th>new agent_cell</th><th>summary</th></tr>'
            + "".join(rows) + '</table>'
        )

    return "".join(parts)


def _load_turn1_perception(work_dir: Path) -> dict | None:
    """Read turn_001's perception reply (the initial-frame
    extraction).  Returns the parsed JSON dict (entities, groups,
    relationships, grid_inference, symbolic_state, game_type,
    game_purpose, overall_notes), or None if unavailable.

    Prefers perception_grounded.json -- the GROUNDED perception whose entity
    bboxes the substrate snapped to its measured components -- so the
    entity-analysis view shows accurate geometry, not the raw hand-estimated
    reply.  Falls back to the consumed/raw reply for older runs."""
    for cand in (work_dir / "turn_001" / "perception_grounded.json",
                  work_dir / "turn_001" / "reply.consumed.txt",
                  work_dir / "turn_001" / "reply.txt"):
        if cand.exists():
            body = cand.read_text(encoding="utf-8").strip()
            if body.startswith("```"):
                body = body.split("\n", 1)[1] if "\n" in body else body
                body = body.rsplit("```", 1)[0]
            for suffix in ("", "}", '"}'):
                try:
                    return json.loads(body + suffix)
                except json.JSONDecodeError:
                    continue
    return None


def _format_initial_extraction(perception: dict | None,
                                 frame_path: Path | None,
                                 gridded_path: Path | None,
                                 views_dir: Path,
                                 heading: str = "Initial frame extraction "
                                                "(perception of turn 1)",
                                 view_name: str = "initial_frame.png") -> str:
    """Render the 'Initial frame extraction' section showing what
    the perception layer extracted from turn 1 — before any
    actions were taken.  This is the system's foundational
    understanding of the game.

    Game-agnostic: shows whatever the perception VLM returned
    using the standard schema (entities, groups, relationships,
    grid_inference, symbolic_state, game_type, game_purpose).

    Image strategy:
      - If the RAW frame + perception entities are both available,
        re-render the SAME gridded view the VLM saw via
        render_turn1_overlay(), with each entity's bbox drawn as
        a numbered cyan rectangle.  Then NEAREST-upscale the
        whole composite by DISPLAY_UPSCALE so grid lines + tick
        labels stay crisp in a browser without CSS interpolation.
      - Otherwise fall back to copying the gridded VLM input
        (image_grid.png) if present, or the raw frame.
    """
    if perception is None:
        return (f'<h2>{_esc(heading)}</h2>'
                '<p><i>(perception reply not found)</i></p>')

    parts = [f'<h2>{_esc(heading)}</h2>']

    # GROUNDING-QA caption (entity-analysis quality, surfaced where it's inspected):
    # how tightly the substrate could ground the VLM's entity bboxes to measured
    # components.  A high score = tight, trustworthy boxes; a low score / many
    # unverified or dropped boxes is the visible signal of an entity-image
    # regression -- so it can't degrade silently.
    qa = perception.get("grounding_qa") or {}
    if qa:
        score = qa.get("score")
        cap = (f"grounding QA: {qa.get('snapped', 0)}/{qa.get('total', 0)} "
               f"entities tight (score {score})")
        extras = []
        if qa.get("dropped"):
            extras.append(f"{qa['dropped']} phantom dropped")
        if qa.get("unmatched"):
            extras.append(f"{qa['unmatched']} UNVERIFIED")
        if qa.get("missed"):
            extras.append(f"{qa['missed']} missed")
        if qa.get("overlaps"):
            extras.append(f"{qa['overlaps']} OVERLAP-CONFLICT")
        if extras:
            cap += " — " + ", ".join(extras)
        color = "#2a2" if (score or 0) >= 0.9 and not extras else "#c80"
        parts.append(f'<div class="subtitle" style="color:{color}">{_esc(cap)}</div>')

    img_html = ""
    views_dir.mkdir(parents=True, exist_ok=True)
    local = views_dir / view_name
    if local.exists():
        local.unlink()

    entities = perception.get("entities") or []
    rendered = False
    if (frame_path is not None and frame_path.exists()
            and entities):
        try:
            annotated = render_turn1_overlay(
                frame_path, entities, n_ticks=DEFAULT_N_TICKS,
                upscale=TRACE_UPSCALE,
                bbox_line_width=TRACE_BBOX_LINE_WIDTH,
                bbox_label_size=TRACE_BBOX_LABEL_SIZE,
                grid_line_width_major=TRACE_GRID_LINE_WIDTH,
                grid_line_width_minor=TRACE_GRID_MINOR_WIDTH,
                grid_label_size=TRACE_TICK_LABEL_SIZE,
                index_labels=True,   # image shows #N only; names go in the table
            )
            annotated.save(local)
            rendered = True
        except (OSError, ValueError) as e:
            # render_turn1_overlay failed (corrupt frame, bad bbox,
            # PIL error).  Fall through to fallback path below.
            print(f"[render] turn1 overlay failed: {e}", file=sys.stderr)

    if not rendered:
        pick = None
        if gridded_path is not None and gridded_path.exists():
            pick = gridded_path
        elif frame_path is not None and frame_path.exists():
            pick = frame_path
        if pick is not None:
            shutil.copy(pick, local)
            rendered = True

    if rendered:
        # Natural-size display: image is rendered ONCE at the target
        # display resolution (TRACE_UPSCALE), so no further CSS
        # scaling is needed.  image-rendering: auto lets the browser
        # smooth gracefully if the viewport is narrower than the
        # image — keeps text legible at any zoom level.
        img_html = (
            f'<img src="views/{local.name}" alt="initial frame with entity overlays" '
            f'style="max-width: 100%; height: auto; '
            f'border: 1px solid #999; background: #000;">'
        )

    # Top-level guesses
    gt = perception.get("game_type") or {}
    gp = perception.get("game_purpose") or {}
    if isinstance(gt, str):
        gt = {"guess": gt}
    if isinstance(gp, str):
        gp = {"guess": gp}
    gi = perception.get("grid_inference") or {}
    ss = perception.get("symbolic_state") or {}

    overview = (
        '<table class="summary-table">'
        f'<tr><th>game_type guess</th><td>{_clean_prose(gt.get("guess",""))}</td></tr>'
        f'<tr><th>game_type confidence</th><td>{_esc(gt.get("confidence",""))}</td></tr>'
        f'<tr><th>game_type evidence</th><td>{_clean_prose(gt.get("evidence",""))}</td></tr>'
        f'<tr><th>game_purpose guess</th><td>{_clean_prose(gp.get("guess",""))}</td></tr>'
        f'<tr><th>game_purpose confidence</th><td>{_esc(gp.get("confidence",""))}</td></tr>'
        f'<tr><th>game_purpose evidence</th><td>{_clean_prose(gp.get("evidence",""))}</td></tr>'
        f'<tr><th>grid_inference</th><td>'
        f'is_grid_based={gi.get("is_grid_based")}, '
        f'cell_ticks={gi.get("cell_ticks")}, '
        f'rows×cols={gi.get("rows")}×{gi.get("cols")}, '
        f'origin={gi.get("origin_ticks")}'
        '</td></tr>'
        f'<tr><th>symbolic_state.agent_cell</th><td>{ss.get("agent_cell")}</td></tr>'
        f'<tr><th>goal_candidate_cells</th><td>{ss.get("goal_candidate_cells")}</td></tr>'
        '</table>'
    )

    # Entities table — index column matches the `#N` label drawn on
    # the overlay image (render_turn1_overlay uses 1-based i+1).
    # render_turn1_overlay filters out very-large region bboxes
    # (>25% of playfield area) and empty_cell placeholders, so those
    # rows show "—" in the # column to signal "listed but not drawn".
    ents = perception.get("entities") or []
    ent_rows = []
    PLAYFIELD_AREA = DEFAULT_N_TICKS * DEFAULT_N_TICKS
    LARGE_BBOX_FRAC = 0.25
    for i, e in enumerate(ents):
        bbox = e.get("bbox_ticks_turn1") or e.get("bbox_ticks") or ""
        role = e.get("role_hypothesis") or ""
        conf = e.get("confidence") or ""
        appearance = (e.get("appearance") or "")[:80]
        drawn = True
        name = (e.get("name") or "").lower()
        if name.startswith("empty_cell"):
            drawn = False
        elif isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            br, bc, br1, bc1 = bbox
            if (br1 - br) * (bc1 - bc) > PLAYFIELD_AREA * LARGE_BBOX_FRAC:
                drawn = False
        idx_cell = f"#{i+1}" if drawn else "—"
        ent_rows.append(
            f'<tr><td>{idx_cell}</td>'
            f'<td>{_esc(e.get("name",""))}</td>'
            f'<td><span class="role-badge {_esc(role or "unknown")}">{_esc(role)}</span></td>'
            f'<td>{_esc(bbox)}</td><td>{_esc(appearance)}</td><td>{_esc(conf)}</td></tr>'
        )
    ent_table = (
        '<h3>Entities perceived '
        '<span style="font-weight:normal;color:#666;font-size:12px;">'
        '(# column matches the cyan label on the overlay image; '
        '— = not drawn: region bbox &gt;25% playfield or empty_cell '
        'placeholder)</span></h3>'
        '<table class="entity-table">'
        '<tr><th>#</th><th>name</th><th>role</th><th>bbox_ticks</th>'
        '<th>appearance</th><th>confidence</th></tr>'
        + "".join(ent_rows) + '</table>'
    )

    # Groups
    groups = perception.get("groups") or []
    group_rows = []
    for g in groups:
        members = ", ".join(g.get("members", []))
        group_rows.append(
            f'<tr><td>{_esc(g.get("name",""))}</td>'
            f'<td>{_esc(g.get("criterion",""))}</td>'
            f'<td>{_esc(members)}</td>'
            f'<td>{_esc(g.get("evidence","")[:120])}</td></tr>'
        )
    group_table = (
        '<h3>Groups</h3>'
        '<table class="entity-table">'
        '<tr><th>name</th><th>criterion</th>'
        '<th>members</th><th>evidence</th></tr>'
        + "".join(group_rows) + '</table>'
    )

    # Relationships
    rels = perception.get("relationships") or []
    rel_rows = []
    for r in rels:
        rel_rows.append(
            f'<tr><td>{_esc(r.get("from",""))}</td>'
            f'<td><b>{_esc(r.get("relation",""))}</b></td>'
            f'<td>{_esc(r.get("to",""))}</td>'
            f'<td>{_esc(r.get("confidence",""))}</td>'
            f'<td>{_esc((r.get("evidence","") or "")[:120])}</td></tr>'
        )
    rel_table = (
        '<h3>Relationships</h3>'
        '<table class="rel-table">'
        '<tr><th>from</th><th>relation</th><th>to</th>'
        '<th>confidence</th><th>evidence</th></tr>'
        + "".join(rel_rows) + '</table>'
    )

    notes = perception.get("overall_notes") or ""
    notes_block = (
        f'<h3>Overall notes</h3><p>{_clean_prose(notes)}</p>'
        if notes else ''
    )

    # Full-width image on top, overview table below it.
    parts.append(f'<div>{img_html}</div>')
    parts.append(overview)
    parts.append(ent_table)
    parts.append(group_table)
    parts.append(rel_table)
    parts.append(notes_block)
    return "".join(parts)


def _load_level_start_analysis(work_dir: Path, turn: int) -> dict | None:
    """Read the fresh entity-analysis a level-advance turn produced into its
    ``level_start/`` subdir (see ExploratoryDriver.run_level_start_analysis)."""
    base = work_dir / f"turn_{turn:03d}" / "level_start"
    # Prefer the GROUNDED perception (bboxes snapped to measured components), then
    # the fully-distilled analysis; fall back to the consumed replies and finally
    # the raw reply.txt (a level-start that ran on the mechanical path, or was cut
    # short by a directed sequence, may leave only an unconsumed reply.txt — still
    # the right perception to show).
    for cand in (base / "perception_grounded.json",
                  base / "level_start_analysis.json",
                  base / "refinement_reply.consumed.txt",
                  base / "reply.consumed.txt",
                  base / "reply.txt"):
        if not cand.exists():
            continue
        body = cand.read_text(encoding="utf-8").strip()
        if body.startswith("```"):
            body = body.split("\n", 1)[1] if "\n" in body else body
            body = body.rsplit("```", 1)[0]
        # raw_decode parses the FIRST JSON object and ignores any trailing data
        # (e.g. a reply accidentally written twice -> "Extra data").
        try:
            return json.JSONDecoder().raw_decode(body)[0]
        except json.JSONDecodeError:
            for suffix in ("}", '"}'):
                try:
                    return json.loads(body + suffix)
                except json.JSONDecodeError:
                    continue
    return None


def _format_per_level_entity_analysis(work_dir: Path, frame_dir: Path,
                                        wk: dict) -> str:
    """Render an entity-analysis section PER LEVEL — the foundational
    perception each level starts from.  Level 0 is turn one; each subsequent
    level is the fresh discovery run at its score-advance turn.  This is what
    lets the reader see (and the system reason about) the new structures a
    level introduces — not just level 0.

    The block opens with the canonical umbrella heading
    (``LEVEL_ANALYSIS_HEADING``) shared with the autonomous renderer, so every
    canonical trace carries the same per-level entity-analysis section marker
    regardless of which renderer wrote it."""
    parts: list[str] = [
        f'<h2>{LEVEL_ANALYSIS_HEADING}</h2>',
        '<div class="subtitle">Foundational perception at the first frame of '
        'each level — the entities and structure the level introduces, which '
        'every downstream decision builds on.</div>',
    ]
    views = work_dir / "views"

    # base_level = the lc the run STARTED at (its turn-1 state).  CRITICAL: the
    # live adapter ALWAYS resets the game to level 0 (LiveHarnessAdapter.reset()
    # takes no level arg) — the --level flag is METADATA ONLY, it does NOT seek
    # the game, so wk['level'] is NOT the level actually being played.  Derive
    # the start level from score = score - (advances during this run): for a
    # normal run that is 0 (score == #advances → labels lc=0, lc=1, ...); for a
    # DD replayed-prefix trial the prefix replay raises score WITHOUT recorded
    # score_increased deltas, so score-len(advs) recovers the entry level (e.g.
    # score 4 with 1 in-run advance → base 3).
    advs = sorted(d["to_turn"] for d in (wk.get("deltas_observed") or [])
                  if d.get("score_increased"))
    base_level = max(0, (wk.get("score") or 0) - len(advs))

    # The start level — turn one.
    p0 = _load_turn1_perception(work_dir)
    g0 = work_dir / "turn_001" / "image_grid.png"
    parts.append(_format_initial_extraction(
        p0, frame_dir / "turn_001.png", g0 if g0.exists() else None, views,
        heading=f"Entity analysis — level {base_level} start (turn 1)",
        view_name=f"entity_analysis_lc{base_level}.png"))

    # Subsequent levels: scan for the level_start/ dirs (one per advance);
    # lc = base_level + the score advances up to that turn.
    ls_turns = sorted(
        int(p.parent.name.split("_")[-1])
        for p in work_dir.glob("turn_*/level_start")
        if p.is_dir() and _load_level_start_analysis(
            work_dir, int(p.parent.name.split("_")[-1])) is not None
    )
    for t in ls_turns:
        lc = base_level + sum(1 for a in advs if a <= t)
        analysis = _load_level_start_analysis(work_dir, t)
        grid = work_dir / f"turn_{t:03d}" / "level_start" / "image_grid.png"
        # Overlay on THIS level's RAW start frame (saved by run_level_start_
        # analysis), NOT frame_dir/turn_{t}.png -- the latter is the transition
        # turn's frame (the just-finished level), which made the lc=1 image show
        # the lc=0 board.  Fall back to the old path only if the raw frame is
        # missing (older runs).
        ls_raw = work_dir / f"turn_{t:03d}" / "level_start" / "frame.png"
        raw_frame = ls_raw if ls_raw.exists() else (frame_dir / f"turn_{t:03d}.png")
        parts.append(_format_initial_extraction(
            analysis, raw_frame,
            grid if grid.exists() else None, views,
            heading=f"Entity analysis — level {lc} start (turn {t})",
            view_name=f"entity_analysis_lc{lc}.png"))
    return "".join(parts)


def _format_run_info_section(work_dir: Path, wk: dict) -> str:
    """Restore the general run-information block that older traces
    showed at the top: which model/server actually drove the run, the
    perception pipeline, endpoints, prompts, etc.

    Reads ``run_info.json`` from the run directory (written/merged by
    the driver and by whatever process calls the model).  Always
    backfills game/level/harness from the world model so the section
    can never silently vanish again, even when no writer populated it.
    """
    info: dict = {}
    info_path = work_dir / "run_info.json"
    if info_path.exists():
        try:
            info = json.loads(info_path.read_text(encoding="utf-8")) or {}
        except Exception:
            info = {}

    # Backfill from the world model (authoritative for game/level).
    info.setdefault("game_id", wk.get("game_id"))
    info.setdefault("level", wk.get("level"))
    info.setdefault("harness", "exploratory_driver.py")

    # Display order: known keys first (friendly labels), then extras.
    labels = [
        ("game_id",             "Game"),
        ("level",               "Level (lc)"),
        ("acting_model",        "Acting model"),
        ("acting_provider",     "Acting server / provider"),
        ("acting_endpoint",     "Acting endpoint"),
        ("strategy_channel",    "Strategy channel"),
        ("perception_model",    "Perception model"),
        ("perception_pipeline", "Perception pipeline"),
        ("harness",             "Harness"),
        ("started_at",          "Run started"),
    ]
    # `segments` (how each part of the trial is being run) and `cost`
    # (model usage) get bespoke multi-line rendering, not the flat
    # key/value treatment.
    special = {"segments", "cost"}
    rows, seen = [], set()
    for key, label in labels:
        val = info.get(key)
        if val not in (None, ""):
            rows.append((label, _esc(str(val))))
        seen.add(key)

    # Run mode — per-segment description of replay vs. autonomous, etc.
    seg_html = _format_run_segments(info.get("segments"))
    if seg_html:
        rows.append(("Run mode", seg_html))
    seen.add("segments")

    # Cost — model token usage / latency / dollars when available.
    cost_html = _format_run_cost(info.get("cost"))
    if cost_html:
        rows.append(("Cost", cost_html))
    seen.add("cost")

    for key, val in info.items():
        if key in seen or key in special or val in (None, ""):
            continue
        rows.append((key.replace("_", " ").capitalize(), _esc(str(val))))

    body = "".join(
        f"<tr><th>{_esc(str(l))}</th><td>{v}</td></tr>"
        for l, v in rows
    )
    return (
        '<h2>Run information</h2>'
        f'<table class="summary-table run-info">{body}</table>'
    )


# A delta summary is the perception layer's free-text sentence about an
# action's effect.  A well-behaved VLM writes one clean paragraph, but a
# misbehaving responder can echo a prior reply's JSON back into it; that nests
# recursively turn-over-turn into an unreadable multi-KB blob.  Legitimate
# prose never contains a JSON key like ``"summary":`` — treat the first such
# marker as the boundary where leaked structure begins and cut there.  Purely
# structural (no length threshold), so any field rendered through it stays
# readable no matter what the upstream layer emits.
_LEAKED_STRUCTURE_MARKERS = (
    '"summary":', '\\"summary\\":',
    '"perception":', '\\"perception\\":',
    '"delta":', '\\"delta\\":',
)


def _clean_prose(text, maxlen: Optional[int] = None) -> str:
    """HTML-escaped, render-safe prose for a free-text field.

    Cuts at the first leaked-JSON-structure marker so a recursively
    self-embedded field can never blow up the page.  ``maxlen`` (when given)
    additionally bounds the visible length for compact overview cells; the
    main per-turn summary passes no maxlen so full reasoning is preserved."""
    s = str(text or "")
    cut = len(s)
    for m in _LEAKED_STRUCTURE_MARKERS:
        i = s.find(m)
        if 0 <= i < cut:
            cut = i
    trimmed = cut < len(s)
    s = s[:cut].rstrip(' \t\r\n"\\,:{')
    if maxlen is not None and len(s) > maxlen:
        s = s[:maxlen].rstrip()
        trimmed = True
    if trimmed:
        s += ' …'
    return html.escape(s)


def _esc(s) -> str:
    return html.escape(str(s))


def _format_run_segments(segments) -> str:
    """Render the per-segment run-mode breakdown (replay vs. autonomous,
    which levels, by which component).  Each segment is a dict; missing
    keys are skipped gracefully."""
    if not segments or not isinstance(segments, list):
        return ""
    lines = []
    for seg in segments:
        if not isinstance(seg, dict):
            lines.append(_esc(str(seg)))
            continue
        mode = seg.get("mode", "?")
        scope = seg.get("scope")
        turns = seg.get("turns")
        detail = seg.get("detail")
        src = seg.get("source")
        head = f"<b>{_esc(mode)}</b>"
        if scope:
            head += f" — {_esc(scope)}"
        bits = []
        if turns:
            bits.append(f"turns {_esc(turns)}")
        if detail:
            bits.append(_esc(detail))
        if src:
            bits.append(f"via {_esc(src)}")
        tail = f" <span style='color:#666'>({'; '.join(bits)})</span>" if bits else ""
        lines.append(head + tail)
    return "<br>".join(lines)


def _format_run_cost(cost) -> str:
    """Render model-usage cost: token counts (authoritative) plus dollars
    and latency when the provider reports them."""
    if not cost or not isinstance(cost, dict):
        return ""
    calls = cost.get("calls", 0)
    itok = cost.get("input_tokens", 0)
    otok = cost.get("output_tokens", 0)
    tot = cost.get("total_tokens", itok + otok)
    usd = cost.get("usd", 0) or 0
    lat_ms = cost.get("latency_ms", 0) or 0
    parts = [
        f"{calls} model call(s)",
        f"{itok:,} in + {otok:,} out = {tot:,} tokens",
    ]
    if lat_ms:
        parts.append(f"{lat_ms/1000:.0f}s total model latency")
    if usd and float(usd) > 0:
        parts.append(f"~${float(usd):.4f}")
    else:
        parts.append("USD not reported by provider (token counts authoritative)")
    return "; ".join(_esc(p) for p in parts)


def render(work_dir: Path, frame_dir: Path) -> Path:
    wk_path = work_dir / "world_knowledge.json"
    if not wk_path.exists():
        raise FileNotFoundError(f"no world_knowledge.json at {wk_path}")
    wk = json.loads(wk_path.read_text(encoding="utf-8"))

    actions_by_turn = {a["turn"]: a for a in wk.get("actions_taken") or []}
    deltas_by_to_turn = {d["to_turn"]: d for d in wk.get("deltas_observed") or []}
    notes_by_turn = {
        int(entry[0]): entry[1]
        for entry in (wk.get("perception_notes_by_turn") or [])
    }
    subgoals_idx = _index_subgoals(wk)

    n_turns = wk.get("turn", 0)

    # per-turn level (lc): base level + cumulative score advances. Score
    # advances mark level boundaries; the base accounts for levels reached
    # before the first recorded delta (e.g. a replayed prefix).
    _advs = sorted(d["to_turn"] for d in (wk.get("deltas_observed") or [])
                   if d.get("score_increased"))
    # base level = the lc the run STARTED at = score - (advances so far).  The
    # adapter ALWAYS starts the game at level 0 (--level is metadata, it does NOT
    # seek), so for a normal run base = 0 (score == #advances) and per-turn lc =
    # advances so far (lc=0, lc=1, ...).  Mirrors the per-level base_level.  NOT
    # wk['level'] (the --level arg) — that is not the level being played.
    _base_level = max(0, (wk.get("score") or 0) - len(_advs))
    level_by_turn = {
        n: _base_level + sum(1 for t in _advs if t <= n)
        for n in range(1, n_turns + 1)
    }

    summary_table = (
        '<table class="summary-table">'
        f'<tr><th>game_id</th><td>{wk.get("game_id")}</td></tr>'
        f'<tr><th>level</th><td>{wk.get("level")}</td></tr>'
        f'<tr><th>turn</th><td>{wk.get("turn")}</td></tr>'
        f'<tr><th>win_state</th><td>{wk.get("win_state")}</td></tr>'
        f'<tr><th>score (lc)</th><td>{wk.get("score")}</td></tr>'
        f'<tr><th>game_type_guess</th><td>{wk.get("game_type_guess")}</td></tr>'
        f'<tr><th>game_purpose_guess</th><td>{wk.get("game_purpose_guess")}</td></tr>'
        f'<tr><th>entities</th><td>{len(wk.get("entities") or {})}</td></tr>'
        f'<tr><th>groups</th><td>{len(wk.get("groups") or {})}</td></tr>'
        f'<tr><th>relationships</th><td>{len(wk.get("relationships") or [])}</td></tr>'
        f'<tr><th>mechanic_hypotheses</th><td>{len(wk.get("mechanic_hypotheses") or [])}</td></tr>'
        f'<tr><th>actions_taken</th><td>{len(wk.get("actions_taken") or [])}</td></tr>'
        '</table>'
    )

    turn_blocks = []
    for n in range(1, n_turns + 1):
        turn_blocks.append(
            _format_turn(n, work_dir, frame_dir,
                          actions_by_turn, deltas_by_to_turn,
                          notes_by_turn,
                          subgoals_idx=subgoals_idx,
                          level_by_turn=level_by_turn)
        )

    # sort hypotheses by credence
    hypotheses = sorted(wk.get("mechanic_hypotheses") or [],
                         key=lambda h: -h["credence"])
    hyp_html = "".join(_format_hypothesis(h) for h in hypotheses)

    entities = wk.get("entities") or {}
    entity_rows = "".join(
        _format_entity_row(n, r, wk.get("turn", 0))
        for n, r in entities.items()
    )
    rel_rows = "".join(
        _format_relationship_row(r) for r in (wk.get("relationships") or [])
    )

    # Initial-frame perception extract (turn 1) — what the system
    # knew BEFORE any actions were taken.  Prefer the GRIDDED
    # frame the VLM actually saw (with tick labels) over the raw
    # game frame so the labels are legible.
    # Entity analysis PER LEVEL (turn 1 + each level's fresh discovery pass),
    # not just level 0.
    initial_section = _format_per_level_entity_analysis(
        work_dir, frame_dir, wk,
    )

    html = (
        '<!DOCTYPE html><html><head><meta charset="utf-8">'
        f'<title>Exploratory run — {wk.get("game_id")} lc={wk.get("level")}</title>'
        f'<style>{CSS}\n{timeago_assets.css}</style></head><body>'
        f'{timeago_assets.script}'
        f'<h1>Exploratory run — {wk.get("game_id")} lc={wk.get("level")}</h1>'
        f'<div class="subtitle">{n_turns} turns,'
        f' {len(wk.get("actions_taken") or [])} actions executed,'
        f' {len(wk.get("mechanic_hypotheses") or [])} mechanic hypotheses</div>'
        f'{timeago_banner("Generated")}'
        f'{_format_run_info_section(work_dir, wk)}'
        '<h2>Run summary</h2>'
        f'{summary_table}'
        f'{_format_plan_section(work_dir, wk)}'
        f'{_format_win_condition_section(wk)}'
        f'{_format_subgoals_section(wk)}'
        f'{initial_section}'
        '<h2>Mechanic hypotheses (sorted by credence)</h2>'
        f'<div>{hyp_html or "<i>(none discovered yet)</i>"}</div>'
        '<h2>Turn-by-turn timeline</h2>'
        '<div class="turns">' + "".join(turn_blocks) + '</div>'
        '<h2>Entity inventory (current state)</h2>'
        '<table class="entity-table">'
        '<tr><th>name</th><th>role</th><th>bbox</th>'
        '<th>cell</th><th>lifespan</th></tr>'
        f'{entity_rows}'
        '</table>'
        '<h2>Relationship inventory — compatible/complementary pairs (current state)</h2>'
        '<table class="rel-table">'
        '<tr><th>from</th><th>relation</th><th>to</th>'
        '<th>detail</th><th>confidence</th><th>times</th></tr>'
        f'{rel_rows}'
        '</table>'
        '</body></html>'
    )
    out = work_dir / "index.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, required=True,
                        help="ExploratoryDriver work directory "
                             "(contains world_knowledge.json + "
                             "turn_NNN subdirs)")
    parser.add_argument("--frame-dir", type=Path, required=True,
                        help="Directory containing the raw game "
                             "frames as turn_NNN.png (the adapter's "
                             "work_frame_dir).")
    args = parser.parse_args()
    out = render(args.work_dir, args.frame_dir)
    print(f"Rendered: file:///{out.as_posix()}")


if __name__ == "__main__":
    main()
