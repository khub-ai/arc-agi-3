"""Render a perception output (parsed.json + frame.png) as a markup HTML
page suitable for inspection in the Claude Preview Panel.

Usage::

    python -m cognitive_os.perception.viz <perception_run_dir>

Where <perception_run_dir> contains at minimum frame.png and parsed.json
(the layout written by test_runner / perception_hook).  Emits
``markup.html`` in the same directory; the page embeds the frame as a
base64 data URI so it can be served standalone.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional


_ROLE_COLOURS = {
    # entity_role -> stroke colour for the bbox rectangle
    "agent_avatar":          "#FF7043",  # orange
    "reference_glyph":       "#42A5F5",  # blue
    "working_glyph":         "#66BB6A",  # green
    "shape_changer":         "#FFFFFF",  # white
    "color_changer":         "#E91E63",  # pink
    "launcher":              "#BDBDBD",  # grey
    "life_refuel":           "#FFEB3B",  # yellow
    "life_indicator":        "#F44336",  # red
    "budget_meter":          "#FBC02D",  # dim yellow
    "wall":                  "#555555",  # mid grey
    "play_area":             "#212121",  # near-black
    "void":                  "#212121",
    "movable_block":         "#AB47BC",  # purple
    "movable_pin":           "#FF8A65",
    "anchor_endpoint":       "#5C6BC0",
    "linked_midpoint":       "#26A69A",
    "rigid_link":            "#9CCC65",
    "piercer_head":          "#EF5350",
    "piercer_tail":          "#26C6DA",
    "reference_arrangement": "#FFCA28",
    "reference_pair_member": "#42A5F5",
    "working_sequence":      "#66BB6A",
    "selection_bracket":     "#FF7043",
    "divider":               "#9E9E9E",
    "target_slot":           "#D81B60",
    "unknown":               "#9E9E9E",
    "noise":                 "#616161",
}

_DEFAULT_COLOUR = "#00E5FF"  # cyan — for any role not in the table above


def _colour_for(role: str) -> str:
    return _ROLE_COLOURS.get(role, _DEFAULT_COLOUR)


def _frame_png_data_uri(frame_png: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(
        frame_png.read_bytes()
    ).decode("ascii")


def _render_entities_table(entities: List[Mapping[str, Any]]) -> str:
    rows: List[str] = []
    for i, e in enumerate(entities, 1):
        bbox = e.get("bbox_pixels") or [0, 0, 0, 0]
        bbox_str = ",".join(str(v) for v in bbox)
        pals = e.get("palettes") or []
        pals_str = ",".join(str(p) for p in pals)
        role = html.escape(str(e.get("role") or ""))
        credence = str(e.get("role_credence") or "")
        if credence == "tentative" and role and role != "unknown":
            role = (f"{role}<span style='color:#FFB347' "
                    f"title='Tentative role — no pixel-level KB confirmation. "
                    f"Candidate for curiosity exploration.'>?</span>")
        role_hint = e.get("role_hint")
        if role_hint:
            role = f"{role} <em>(→ {html.escape(str(role_hint))})</em>"
        eid  = html.escape(str(e.get("id") or ""))
        notes_text = str(e.get("notes") or "").strip()
        # Structured properties from interaction observation come first
        # (the formal entity-property registry); free-text corrections
        # come after.
        props = e.get("properties") or {}
        props_html = ""
        if props:
            props_lines: List[str] = []
            if props.get("is_agent"):
                props_lines.append("✓ agent (entity moves with actions)")
            elif props.get("moves_with_actions"):
                props_lines.append("✓ moves with actions "
                                   f"({props.get('total_displacement_px',0):.0f}px total)")
            else:
                props_lines.append("• static position (never moved)")
            if props.get("is_autonomous_animation"):
                props_lines.append(
                    f"✓ autonomous animation "
                    f"({props.get('n_pattern_changes',0)} pattern + "
                    f"{props.get('n_palette_changes',0)} palette changes / "
                    f"{props.get('n_frames_seen',0)} frames)"
                )
            elif props.get("appearance_changes"):
                props_lines.append(
                    f"✓ appearance changes when triggered "
                    f"({props.get('n_palette_changes',0)} palette + "
                    f"{props.get('n_pattern_changes',0)} pattern changes)"
                )
            else:
                props_lines.append("• stable appearance (never changed)")
            bb = props.get("initial_bbox") or []
            if bb and len(bb) == 4:
                cr, cc = props.get("initial_center_px") or [0,0]
                zone = props.get("spatial_zone") or ""
                zone_str = f" — {zone}" if zone else ""
                props_lines.append(
                    f"location: bbox r{bb[0]}-{bb[2]}, c{bb[1]}-{bb[3]} "
                    f"(center ≈ pixel {cr:.0f},{cc:.0f}); "
                    f"{props.get('initial_size_px',0)}px{zone_str}"
                )
            rot = props.get("rotation_observed")
            if rot:
                props_lines.append(f"✓ permanent transform: {rot}")
            elif props.get("is_permanent"):
                props_lines.append(
                    "✓ permanent state change (cells rearranged; not a clean rotation/reflection)"
                )
            flash = props.get("transient_flash_palettes") or []
            if flash:
                props_lines.append(
                    f"⚡ transient flash in bbox: palettes {flash} appeared "
                    "then reverted (feedback signal — often indicates "
                    "this entity is RELATED to others that flashed at the same time)"
                )
            all_pals = props.get("all_palettes_seen") or []
            init_pals = props.get("initial_palettes") or []
            if all_pals and tuple(all_pals) != tuple(init_pals):
                props_lines.append(
                    f"palettes observed across time: {all_pals} "
                    f"(initial: {init_pals})"
                )
            trig = props.get("triggered_changes") or []
            if trig:
                props_lines.append(
                    f"observed to TRIGGER {len(trig)} change events "
                    f"(visiting this entity caused another entity to change)"
                )
            hyp = props.get("trigger_hypothesis")
            if hyp:
                props_lines.append(
                    f"⚠ HYPOTHESIS only: correlated with "
                    f"{hyp.get('hypothesised_role')} events "
                    f"{hyp.get('observations')} times "
                    f"(need {hyp.get('needed_for_confirmation')} to confirm)"
                )
            props_html = ("<div class='props'>" +
                          "<br>".join(html.escape(l) for l in props_lines) +
                          "</div>")
        # Free-text corrections from the learning loop come after the
        # structured property block.
        learned: List[str] = []
        for c in (e.get("_corrections") or []):
            learned.append(html.escape(str(c)))
        learned_html = ""
        if learned:
            learned_html = ("<div class='learned'>" +
                            "<br>".join(f"• {l}" for l in learned) +
                            "</div>")
        notes_html = html.escape(notes_text) + props_html + learned_html
        colour = _colour_for(e.get("role") or "")
        rows.append(
            f'<tr data-entity-idx="{i-1}" '
            f'onmouseover="highlightEntity({i-1})" '
            f'onmouseout="unhighlight()">'
            f'<td><span class="swatch" style="background:{colour}"></span>{i}</td>'
            f'<td>{eid}</td>'
            f'<td><strong>{role}</strong></td>'
            f'<td class="mono">{bbox_str}</td>'
            f'<td class="mono">[{pals_str}]</td>'
            f'<td class="notes">{notes_html}</td>'
            f'</tr>'
        )
    return "\n".join(rows)


def _render_relationships(rels: List[Mapping[str, Any]]) -> str:
    if not rels:
        return "<p class='empty'>(no relationships emitted)</p>"
    items: List[str] = []
    for r in rels:
        kind = html.escape(str(r.get("kind") or ""))
        members = r.get("members") or []
        members_str = ", ".join(html.escape(str(m)) for m in members)
        rationale = html.escape(str(r.get("rationale") or ""))
        items.append(
            f'<li><strong>{kind}</strong>: [{members_str}]'
            f' <span class="rationale">{rationale}</span></li>'
        )
    return "<ul>" + "\n".join(items) + "</ul>"


def _render_validation(msgs: List[str]) -> str:
    if not msgs:
        return "<p class='ok'>(no validation issues)</p>"
    items = "\n".join(f"<li>{html.escape(str(m))}</li>" for m in msgs)
    return f"<ul class='warn'>{items}</ul>"


def _render_bounce_plates(plates: List[dict]) -> str:
    if not plates:
        return ("<p class='empty'>No bounce plates discovered in this "
                "frame.</p>")
    rows = []
    for i, p in enumerate(plates, 1):
        bar_bb  = p.get("marker_bbox") or [0, 0, 0, 0]
        plate_bb = p.get("portal_bbox") or [0, 0, 0, 0]
        rest_bb = p.get("predicted_exit_bbox") or [0, 0, 0, 0]
        push    = p.get("exit_direction_name", "?")
        dist    = p.get("predicted_path_len", 0)
        status  = p.get("status", "predicted_only")
        pal     = ",".join(str(x) for x in (p.get("marker_palettes") or []))
        size    = p.get("marker_size_px")
        verified = ""
        if status == "observed":
            err = p.get("prediction_error_px")
            obs = p.get("observed_exit_bbox")
            action = p.get("observed_action")
            verified = (f"<br/><span class='ok'>✓ VERIFIED</span> "
                        f"observed_exit_bbox={obs} via {action} "
                        f"(prediction_error {err}px)")
        rows.append(
            f"<tr>"
            f"<td><strong>#{i}</strong></td>"
            f"<td>bar bbox={bar_bb}<br/>pal={pal} size={size}px</td>"
            f"<td>plate bbox={plate_bb}<br/>activate by being here</td>"
            f"<td><strong>{push}</strong></td>"
            f"<td>resting bbox={rest_bb}<br/>slide_distance={dist} cells"
            f"{verified}</td>"
            f"</tr>"
        )
    return (
        "<table>"
        "<thead><tr><th>#</th><th>Bar (pusher)</th><th>Plate cell</th>"
        "<th>Push direction</th><th>Predicted resting cell</th></tr></thead>"
        "<tbody>" + "\n".join(rows) + "</tbody></table>"
    )


def _render_alignment(history: List[dict], achieved: bool) -> str:
    if not history:
        return ("<p class='empty'>working_glyph and reference_glyph never "
                "matched at any tier (no alignment detected during this run).</p>")
    items = []
    for ev in history:
        items.append(
            f"<li>step {ev.get('step')}: matched at <strong>{ev.get('tier')}</strong> tier "
            f"({ev.get('working_id')} ↔ {ev.get('reference_id')})</li>"
        )
    label = "✓ ALIGNMENT ACHIEVED" if achieved else ""
    return (f"<p class='ok'>{label}</p>" if label else "") + \
           "<ul>" + "\n".join(items) + "</ul>"


def _render_palette_role_map(pmap: Mapping[Any, Any]) -> str:
    if not pmap:
        return "<p class='empty'>(no palette role map)</p>"
    rows = []
    for k, v in sorted(pmap.items(), key=lambda kv: int(kv[0])):
        rows.append(f"<li>palette <strong>{int(k)}</strong> → {html.escape(str(v))}</li>")
    return "<ul>" + "\n".join(rows) + "</ul>"


def build_markup_html(
    *,
    parsed:    Mapping[str, Any],
    frame_png: Path,
    title:     str,
    geometry:  Optional[Mapping[str, Any]] = None,
) -> str:
    img_uri = _frame_png_data_uri(frame_png)
    entities = list(parsed.get("entities") or [])
    rels     = list(parsed.get("relationships") or [])
    msgs     = list(parsed.get("validation_messages") or [])
    pmap     = parsed.get("palette_role_map") or {}
    win      = parsed.get("win_condition") or {}
    bg_pals  = list(parsed.get("background_palettes") or [])
    interaction_log = parsed.get("interaction_log") or {}
    alignment_history = interaction_log.get("alignment_history") or []
    alignment_achieved = bool(interaction_log.get("alignment_achieved"))

    # bboxes packaged for JS overlay rendering.  All bboxes are in
    # 64x64 pixel space; the SVG is overlaid on a 64x64 viewBox so the
    # numbers translate directly regardless of display size.
    overlay_entities = []
    for i, e in enumerate(entities):
        bbox = e.get("bbox_pixels") or [0, 0, 0, 0]
        role = str(e.get("role") or "")
        credence = str(e.get("role_credence") or "")
        # Tentative-credence roles render with a "?" suffix so an
        # operator (and the planner) can see at a glance that the
        # label is a hypothesis, not a confirmed fact.
        display_role = (f"{role}?" if credence == "tentative" and role
                                       and role != "unknown" else role)
        overlay_entities.append({
            "idx":      i,
            "id":       str(e.get("id") or ""),
            "role":     display_role,
            "bbox":     [int(v) for v in bbox],
            "palettes": [int(p) for p in (e.get("palettes") or [])],
            "size_px":  int((e.get("properties") or {}).get("initial_size_px") or 0),
            "colour":   _colour_for(role),
            "credence": credence,
        })
    overlay_json = json.dumps(overlay_entities)

    # Layer A candidates (optional) — show as dashed outlines.
    cand_overlay = []
    if geometry:
        for c in geometry.get("candidates") or []:
            bbox = c.get("bbox_pixels") or [0, 0, 0, 0]
            cand_overlay.append({
                "idx":  int(c.get("id", 0)),
                "bbox": [int(v) for v in bbox],
            })
    cand_overlay_json = json.dumps(cand_overlay)

    # Bounce-plate overlays: bar, plate cell, and predicted-resting-cell
    # bboxes with an arrow showing push direction.
    bounce_plates = (interaction_log.get("bounce_plates")
                     or interaction_log.get("portal_endpoints") or [])
    plate_overlay = []
    for i, p in enumerate(bounce_plates, 1):
        plate_overlay.append({
            "idx":           i,
            "bar_bbox":      [int(v) for v in (p.get("marker_bbox") or [0,0,0,0])],
            "plate_bbox":    [int(v) for v in (p.get("portal_bbox") or [0,0,0,0])],
            "rest_bbox":     [int(v) for v in (p.get("predicted_exit_bbox") or [0,0,0,0])],
            "push":          str(p.get("exit_direction_name") or ""),
            "observed":      (p.get("status") == "observed"),
            "obs_bbox":      [int(v) for v in (p.get("observed_exit_bbox") or [0,0,0,0])]
                              if p.get("observed_exit_bbox") else None,
        })
    plate_overlay_json = json.dumps(plate_overlay)

    # Cell-grid lines: pulled from parsed["cell_system"] when set by
    # the runner (perception derived origin + size).  Falls back to
    # "no grid" when absent.
    cell_system = parsed.get("cell_system") or {}
    cell_origin = cell_system.get("origin")
    cell_size   = int(cell_system.get("cell_size") or 0)
    grid_json = json.dumps({
        "origin":    cell_origin if cell_origin else None,
        "cell_size": cell_size,
    })

    win_kind = html.escape(str(win.get("kind") or "(none)"))
    win_desc = html.escape(str(win.get("description") or ""))
    win_inv  = ", ".join(html.escape(str(x)) for x in (win.get("involves") or []))

    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg:       #121212;
  --panel:    #1e1e1e;
  --muted:    #888;
  --accent:   #00E5FF;
  --text:     #e0e0e0;
  --warn:     #FFAB00;
  --ok:       #66BB6A;
}}
* {{ box-sizing: border-box; }}
body {{
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0; padding: 16px;
}}
h1 {{ font-size: 16px; font-weight: 600; margin: 0 0 12px 0; }}
h2 {{ font-size: 13px; font-weight: 600; margin: 12px 0 6px 0; color: var(--accent); }}
.frame-wrap {{
  display: inline-block; vertical-align: top;
  border: 1px solid #333; background: #000;
  margin-right: 20px;
  width: 512px; height: 512px;
  position: relative;
}}
.frame-img {{
  width: 100%; height: 100%;
  image-rendering: pixelated;
  -ms-interpolation-mode: nearest-neighbor;
  display: block;
}}
.overlay {{
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  pointer-events: none;
}}
.bbox-rect {{
  fill: transparent;
  stroke-width: 0.4;
  vector-effect: non-scaling-stroke;
}}
.bbox-rect.cand {{
  stroke: #444;
  stroke-dasharray: 0.5,0.5;
  stroke-width: 0.2;
}}
.bbox-rect.highlighted {{
  stroke-width: 1.0;
  fill: rgba(255,255,255,0.15);
}}
.bbox-label {{
  fill: white;
  font-size: 2px;
  font-family: monospace;
  paint-order: stroke;
  stroke: black;
  stroke-width: 0.4;
}}
.right-pane {{
  display: inline-block; vertical-align: top;
  width: calc(100% - 540px);
  min-width: 400px;
}}
.summary {{ background: var(--panel); padding: 10px 14px; border-radius: 6px; margin-bottom: 12px; }}
.summary .row {{ display: flex; gap: 8px; margin: 4px 0; font-size: 12px; }}
.summary .label {{ color: var(--muted); min-width: 110px; }}
.summary .value {{ color: var(--text); }}
.summary .kind {{ color: var(--accent); font-weight: 600; }}
table {{
  width: 100%; border-collapse: collapse; font-size: 11px;
  background: var(--panel); border-radius: 4px;
}}
th, td {{ padding: 5px 8px; text-align: left; border-bottom: 1px solid #2a2a2a; }}
th {{ color: var(--muted); font-weight: 500; }}
tr:hover {{ background: #2a2a2a; cursor: pointer; }}
td.mono {{ font-family: monospace; color: #bbb; }}
td.notes {{ color: var(--muted); font-size: 10px; max-width: 380px; }}
.props {{ color: #4FC3F7; font-size: 10px; margin-top: 4px;
         border-left: 2px solid #4FC3F7; padding-left: 6px; }}
.learned {{ color: #FFCA28; font-size: 10px; margin-top: 4px;
           border-left: 2px solid #FFCA28; padding-left: 6px; }}
.swatch {{
  display: inline-block; width: 10px; height: 10px;
  border-radius: 2px; margin-right: 6px; vertical-align: middle;
}}
ul {{ font-size: 12px; padding-left: 18px; margin: 4px 0; }}
ul.warn li {{ color: var(--warn); }}
.rationale {{ color: var(--muted); }}
.empty {{ color: var(--muted); font-style: italic; font-size: 12px; }}
.ok {{ color: var(--ok); font-size: 12px; }}
</style></head>
<body>
<h1>Perception markup — {html.escape(title)}</h1>

<div class="frame-wrap">
  <img class="frame-img" src="{img_uri}" alt="frame">
  <svg class="overlay" viewBox="0 0 64 64" preserveAspectRatio="none">
    <g id="cand-layer"></g>
    <g id="ent-layer"></g>
  </svg>
</div>

<div class="right-pane">
  <div class="summary">
    <div class="row"><span class="label">Win condition</span>
      <span class="value kind">{win_kind}</span></div>
    <div class="row"><span class="label">Description</span>
      <span class="value">{win_desc}</span></div>
    <div class="row"><span class="label">Involves</span>
      <span class="value">{win_inv}</span></div>
    <div class="row"><span class="label">Background pals</span>
      <span class="value">{', '.join(str(p) for p in bg_pals) or '(none)'}</span></div>
    <div class="row"><span class="label">Entities</span>
      <span class="value">{len(entities)}</span></div>
    <div class="row"><span class="label">Relationships</span>
      <span class="value">{len(rels)}</span></div>
  </div>

  <h2>Bounce plates (push → slide → stop at obstacle)</h2>
  {_render_bounce_plates(bounce_plates)}

  <h2>Alignment</h2>
  {_render_alignment(alignment_history, alignment_achieved)}

  <h2>Relationships</h2>
  {_render_relationships(rels)}

  <h2>Validation</h2>
  {_render_validation(msgs)}
</div>

<h2>Entities</h2>
<table>
<thead><tr>
  <th>#</th><th>id</th><th>role</th><th>bbox (r0,c0,r1,c1)</th>
  <th>palettes</th><th>notes + learned-through-interaction</th>
</tr></thead>
<tbody>
{_render_entities_table(entities)}
</tbody></table>

<script>
const entities  = {overlay_json};
const candidates = {cand_overlay_json};
const SVGNS = "http://www.w3.org/2000/svg";

function rectFor(bbox, attrs) {{
  const [r0, c0, r1, c1] = bbox;
  const r = document.createElementNS(SVGNS, "rect");
  r.setAttribute("x", c0);
  r.setAttribute("y", r0);
  r.setAttribute("width", (c1 - c0 + 1));
  r.setAttribute("height", (r1 - r0 + 1));
  for (const k in attrs) r.setAttribute(k, attrs[k]);
  return r;
}}

function labelFor(bbox, text) {{
  const [r0, c0] = bbox;
  const t = document.createElementNS(SVGNS, "text");
  t.setAttribute("x", c0);
  t.setAttribute("y", Math.max(0, r0 - 0.5));
  t.setAttribute("class", "bbox-label");
  t.textContent = text;
  return t;
}}

// Cell-grid overlay -- thin dashed lines at cell boundaries, anchored
// to the perception-derived cell_system.  Drawn FIRST so entity
// outlines render on top.
const grid = {grid_json};
const gridLayer = document.getElementById("cand-layer");
if (grid.origin && grid.cell_size > 0) {{
  const [or_r, or_c] = grid.origin;
  const cs = grid.cell_size;
  // Vertical lines.
  for (let x = or_c % cs; x < 64; x += cs) {{
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("x1", x); ln.setAttribute("y1", 0);
    ln.setAttribute("x2", x); ln.setAttribute("y2", 64);
    ln.setAttribute("stroke", "#FFFFFF");
    ln.setAttribute("stroke-width", "0.08");
    ln.setAttribute("stroke-opacity", "0.35");
    ln.setAttribute("stroke-dasharray", "0.6,0.4");
    gridLayer.appendChild(ln);
  }}
  // Horizontal lines.
  for (let y = or_r % cs; y < 64; y += cs) {{
    const ln = document.createElementNS(SVGNS, "line");
    ln.setAttribute("x1", 0); ln.setAttribute("y1", y);
    ln.setAttribute("x2", 64); ln.setAttribute("y2", y);
    ln.setAttribute("stroke", "#FFFFFF");
    ln.setAttribute("stroke-width", "0.08");
    ln.setAttribute("stroke-opacity", "0.35");
    ln.setAttribute("stroke-dasharray", "0.6,0.4");
    gridLayer.appendChild(ln);
  }}
  // Tiny cell-coord labels at the top edge and left edge, every 1 cell.
  // Helps the user read off cell numbers without counting lines.
  for (let cc = 0; ; cc++) {{
    const x = or_c + cc * cs - cs;
    if (x >= 64) break;
    if (x < 0) continue;
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", x + 1);
    t.setAttribute("y", 1.5);
    t.setAttribute("class", "bbox-label");
    t.setAttribute("fill", "#FFFFFF");
    t.setAttribute("fill-opacity", "0.5");
    t.setAttribute("font-size", "1.0");
    t.textContent = String(cc - 1);
    gridLayer.appendChild(t);
  }}
  for (let cr = 0; ; cr++) {{
    const y = or_r + cr * cs - cs;
    if (y >= 64) break;
    if (y < 0) continue;
    const t = document.createElementNS(SVGNS, "text");
    t.setAttribute("x", 0.3);
    t.setAttribute("y", y + 2);
    t.setAttribute("class", "bbox-label");
    t.setAttribute("fill", "#FFFFFF");
    t.setAttribute("fill-opacity", "0.5");
    t.setAttribute("font-size", "1.0");
    t.textContent = String(cr - 1);
    gridLayer.appendChild(t);
  }}
}}

// Candidate dashed outlines (Layer A).
const candLayer = document.getElementById("cand-layer");
for (const c of candidates) {{
  candLayer.appendChild(rectFor(c.bbox, {{class: "bbox-rect cand"}}));
}}

// Entity solid outlines (Layer B parsed).
// Skip region-role entities (wall, play_area, void) — their bboxes are
// palette extents covering most of the frame and would obscure
// everything underneath.  They're still listed in the table below.
const REGION_ROLES = new Set(["wall", "play_area", "void", "floor", "hud_background"]);
const FRAME_AREA = 64 * 64;
const entLayer = document.getElementById("ent-layer");
const entRects = [];
// Track placed labels so we can stagger if two would land on top of
// each other (e.g. perception's agent_avatar + the interaction-
// discovered sub-bitmap at the same cell).
const placedLabels = [];   // [{{r, c, span}}]
entities.forEach((e, i) => {{
  const [r0, c0, r1, c1] = e.bbox;
  const w = c1 - c0 + 1;
  const h = r1 - r0 + 1;
  const isRegion = REGION_ROLES.has(e.role);
  const tooLarge = (w * h) / FRAME_AREA > 0.25;
  const useCornerBrackets = isRegion || tooLarge;
  // Always create the rect for highlight-on-hover; for region/huge
  // entities we hide the full outline (it would obscure the frame)
  // and add small corner-bracket marks instead so the bbox extent
  // is still visible.
  const r = rectFor(e.bbox, {{class: "bbox-rect", stroke: e.colour, "data-idx": i}});
  if (useCornerBrackets) {{
    r.setAttribute("opacity", "0");
  }}
  entLayer.appendChild(r);
  entRects.push(r);
  if (useCornerBrackets) {{
    // Four L-shaped corner brackets, each ~2 cells long.
    const brk = 2;
    const segs = [
      // Top-left corner: horizontal segment + vertical segment
      [[c0, r0], [c0 + brk, r0]], [[c0, r0], [c0, r0 + brk]],
      // Top-right
      [[c1 + 1 - brk, r0], [c1 + 1, r0]], [[c1 + 1, r0], [c1 + 1, r0 + brk]],
      // Bottom-left
      [[c0, r1 + 1 - brk], [c0, r1 + 1]], [[c0, r1 + 1], [c0 + brk, r1 + 1]],
      // Bottom-right
      [[c1 + 1 - brk, r1 + 1], [c1 + 1, r1 + 1]],
      [[c1 + 1, r1 + 1 - brk], [c1 + 1, r1 + 1]],
    ];
    for (const [[x1, y1], [x2, y2]] of segs) {{
      const ln = document.createElementNS(SVGNS, "line");
      ln.setAttribute("x1", x1); ln.setAttribute("y1", y1);
      ln.setAttribute("x2", x2); ln.setAttribute("y2", y2);
      ln.setAttribute("stroke", e.colour);
      ln.setAttribute("stroke-width", "0.4");
      ln.setAttribute("vector-effect", "non-scaling-stroke");
      entLayer.appendChild(ln);
    }}
  }}
  // ALWAYS render the role label.  Place it at the top-left of the
  // bbox; stagger vertically if another label landed nearby.
  let labelRow = Math.max(0, r0 - 0.5);
  while (placedLabels.some(p =>
      Math.abs(p.r - labelRow) < 2.5 && Math.abs(p.c - c0) < 10)) {{
    labelRow += 2.5;
  }}
  placedLabels.push({{r: labelRow, c: c0, span: 6}});
  const t = document.createElementNS(SVGNS, "text");
  t.setAttribute("x", c0);
  t.setAttribute("y", labelRow);
  t.setAttribute("class", "bbox-label");
  t.setAttribute("fill", e.colour);
  let label = `${{i+1}}:${{e.role}}`;
  if (e.role === "unknown") {{
    const pals = (e.palettes && e.palettes.length)
      ? `p${{e.palettes.join("+")}}`
      : "p?";
    const sz = e.size_px || (w * h);
    label = `${{i+1}}:${{e.role}}(${{pals}}/${{sz}}px)`;
  }}
  t.textContent = label;
  entLayer.appendChild(t);
}});

function highlightEntity(idx) {{
  entRects.forEach((r, i) => {{
    if (i === idx) r.classList.add("highlighted");
    else r.classList.remove("highlighted");
  }});
}}
function unhighlight() {{
  entRects.forEach(r => r.classList.remove("highlighted"));
}}

// Bounce-plate overlay: yellow bar outline = pusher; cyan dashed =
// plate cell; magenta dashed = predicted resting cell; arrow shows
// push direction.
const plates = {plate_overlay_json};
for (const p of plates) {{
  // Bar (pusher) -- thick yellow outline.
  const bar = rectFor(p.bar_bbox, {{
    class: "bbox-rect", stroke: "#FFD54F", "stroke-width": "0.5"
  }});
  bar.setAttribute("opacity", "1");
  entLayer.appendChild(bar);
  // Resting cell dot -- magenta unfilled (verified -> green filled).
  // Replaces the full-cell rectangle so 8 plates don't tile the
  // frame with overlapping outlines.
  const restColour = p.observed ? "#66BB6A" : "#E040FB";
  const [pr0, pc0, pr1, pc1] = p.plate_bbox;
  const [rr0, rc0, rr1, rc1] = p.rest_bbox;
  const pc = (pc0 + pc1) / 2, pr = (pr0 + pr1) / 2;
  const rc = (rc0 + rc1) / 2, rr = (rr0 + rr1) / 2;
  const restDot = document.createElementNS(SVGNS, "circle");
  restDot.setAttribute("cx", rc);
  restDot.setAttribute("cy", rr);
  restDot.setAttribute("r", 1.2);
  restDot.setAttribute("fill", p.observed ? restColour : "none");
  restDot.setAttribute("stroke", restColour);
  restDot.setAttribute("stroke-width", "0.3");
  entLayer.appendChild(restDot);
  // Short arrow from bar centre extending one cell in push direction,
  // not full line to resting cell -- the resting dot already marks
  // where the slide ends.
  const barCR = (p.bar_bbox[0] + p.bar_bbox[2]) / 2;
  const barCC = (p.bar_bbox[1] + p.bar_bbox[3]) / 2;
  const arrowEnd = {{
    "up":    [barCR - 4, barCC],
    "down":  [barCR + 4, barCC],
    "left":  [barCR, barCC - 4],
    "right": [barCR, barCC + 4],
  }}[p.push] || [barCR, barCC];
  const line = document.createElementNS(SVGNS, "line");
  line.setAttribute("x1", barCR == arrowEnd[0] ? barCC : barCC);
  line.setAttribute("y1", barCR);
  line.setAttribute("x2", arrowEnd[1]);
  line.setAttribute("y2", arrowEnd[0]);
  line.setAttribute("stroke", "#FFD54F");
  line.setAttribute("stroke-width", "0.35");
  entLayer.appendChild(line);
  // Tiny numeric label at the BAR, offset away from the push direction
  // so it doesn't sit on the slide path.  Reduces clutter at the
  // centre of the frame when many plates land near each other.
  const labelOff = {{
    "up":    [3.5, 0],
    "down":  [-3.5, 0],
    "left":  [0, 3.5],
    "right": [0, -3.5],
  }}[p.push] || [0, 0];
  const t = document.createElementNS(SVGNS, "text");
  t.setAttribute("x", barCC + labelOff[1]);
  t.setAttribute("y", barCR + labelOff[0]);
  t.setAttribute("class", "bbox-label");
  t.setAttribute("fill", "#FFD54F");
  t.setAttribute("font-size", "1.6");
  t.setAttribute("text-anchor", "middle");
  t.textContent = `${{p.idx}}`;
  entLayer.appendChild(t);
}}
</script>
</body></html>
"""


def render_run_dir(
    run_dir:    Path,
    *,
    title:      Optional[str] = None,
    frame_png:  Optional[Path] = None,
    out_path:   Optional[Path] = None,
) -> Path:
    """Render markup.html from ``run_dir``'s parsed.json + an associated
    frame.png.

    ``frame_png`` overrides the default lookup.  If omitted, we look in
    ``run_dir/frame.png`` first, then walk up the directory tree to
    find one (test_runner outputs don't carry the frame; perception_hook
    outputs do).

    ``out_path`` selects where to write markup.html (defaults to
    ``run_dir/markup.html``).
    """
    parsed_json = run_dir / "parsed.json"
    if not parsed_json.exists():
        raise FileNotFoundError(f"missing parsed.json in {run_dir}")
    if frame_png is None:
        candidate = run_dir / "frame.png"
        if candidate.exists():
            frame_png = candidate
        else:
            # Walk up looking for frame.png.
            for parent in run_dir.parents:
                f = parent / "frame.png"
                if f.exists():
                    frame_png = f
                    break
    if frame_png is None or not frame_png.exists():
        raise FileNotFoundError(
            f"could not locate frame.png for {run_dir}; pass --frame"
        )
    parsed = json.loads(parsed_json.read_text(encoding="utf-8"))
    geom = None
    geom_json = run_dir / "geometry.json"
    if geom_json.exists():
        try:
            geom = json.loads(geom_json.read_text(encoding="utf-8"))
        except Exception:
            geom = None
    title = title or run_dir.name
    html_str = build_markup_html(
        parsed    = parsed,
        frame_png = frame_png,
        title     = title,
        geometry  = geom,
    )
    out = out_path if out_path is not None else (run_dir / "markup.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_str, encoding="utf-8")
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir",
                    help="Directory containing parsed.json (and optionally "
                         "frame.png + geometry.json)")
    ap.add_argument("--title", default=None)
    ap.add_argument("--frame", default=None,
                    help="Path to frame.png override")
    ap.add_argument("--out", default=None,
                    help="Path to write the markup.html (default: <run_dir>/markup.html)")
    a = ap.parse_args(argv)
    p = render_run_dir(
        Path(a.run_dir),
        title     = a.title,
        frame_png = Path(a.frame) if a.frame else None,
        out_path  = Path(a.out) if a.out else None,
    )
    print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
