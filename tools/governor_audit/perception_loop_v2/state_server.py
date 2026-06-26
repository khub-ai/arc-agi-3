"""state_server.py -- live, SSE-backed server for the Scene State (state-as-medium).

Serves the canonical state + frame and pushes Server-Sent-Events on change, so the
views are real-time + robust (no regenerate-a-PNG-and-hope, no stale cache).  TWO
views over ONE backend:
  * /vlm  -- the VLM-facing entity view: frame + measured entities (id, role,
             resolution, needs_inspection) + the refinement event log, clickable.
  * /user -- the user-facing view: an iframe wrapper that live-reloads the EXISTING
             trace.html (served BYTE-IDENTICAL at /trace -- the wrapper does the SSE
             reload, so the trace's own format/content is never modified).
Backend routes: /state.json (current snapshot), /frame.png (current frame),
/events (SSE, fires when scene_state.json changes).  ThreadingHTTPServer so an SSE
long-poll never blocks other requests.  Reuses the http.server/launch.json/preview
pattern.  The driver writes scene_state.json + frame.png into `root`.
"""
from __future__ import annotations

import base64
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


_USER_VIEW = """<!doctype html><meta charset=utf-8><title>COS trace (live)</title>
<style>html,body{margin:0;height:100%}iframe{border:0;width:100%;height:100vh}</style>
<iframe id=f src="/trace"></iframe>
<script>new EventSource('/events').onmessage=function(ev){if(ev.data!='update')return;
 var f=document.getElementById('f');var s=f.contentWindow?f.contentWindow.scrollY:0;
 f.onload=function(){try{f.contentWindow.scrollTo(0,s)}catch(e){}};
 f.src='/trace?_='+Date.now();};</script>"""


_ROLE_COL = {"switch_bar_H": "#39d0ff", "switch_bar_stem": "#39d0ff", "selector": "#7c9cff",
             "trigger": "#27e0d0", "mover": "#ffd23d", "goal": "#5ce65c",
             "barrier": "#ff5bb0", "reference": "#cfa6ff", "scene": "#8a8a8a"}


def _render_vlm(root: Path) -> str:
    """SERVER-SIDE render the VLM view: the true frame embedded as a base64 data URI
    with measured-entity markers drawn ON it, returned as a self-contained page.  No
    client-side fetch / external <image> (which failed to render) -- it shows anywhere.
    A tiny SSE hook reloads on state change so it stays live."""
    root = Path(root)
    fp, sp = root / "frame.png", root / "scene_state.json"
    b64 = base64.b64encode(fp.read_bytes()).decode() if fp.exists() else ""
    try:
        snap = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    except Exception:
        snap = {}
    ents = snap.get("entities", {}) or {}
    svg = ['<svg viewBox="0 0 1024 1024" style="width:100%;height:auto;display:block">',
           f'<image href="data:image/png;base64,{b64}" x="0" y="0" width="1024" height="1024"/>']
    # large outlined labels are legible against any background
    _TS = ('stroke="#000" stroke-width="4" paint-order="stroke" font-weight="bold" '
           'font-family="monospace"')

    def _glabel(x, y, s, anchor="start"):
        # dark backing plate + outlined text -> readable even over bars / their white outlines
        bw = 14 * len(s) + 10
        bx = x - (bw // 2 if anchor == "middle" else 4)
        return (f'<rect x="{bx}" y="{y-21}" width="{bw}" height="27" fill="#000" '
                f'fill-opacity="0.65" rx="3"/>'
                f'<text x="{x}" y="{y}" fill="#39d0ff" font-size="22" '
                f'text-anchor="{anchor}" {_TS}>{s}</text>')
    # right-panel grid guides (cols 34/39/44/49/54/59, bar-rows 33/36/39/42/45/48)
    for i, c in enumerate([34, 39, 44, 49, 54, 59]):
        svg.append(_glabel(c * 16, 505, f"c{i}", "middle"))
    for i, r in enumerate([33, 36, 39, 42, 45, 48]):          # in the gutter LEFT of the panel
        svg.append(_glabel(492, r * 16 + 18, f"r{i}"))
    for eid, e in ents.items():
        bb = e.get("bbox")
        if not bb:
            continue
        r0, c0, r1, c1 = bb
        col = "#ff9d2e" if e.get("needs_inspection") else _ROLE_COL.get(e.get("role"), "#ffffff")
        on = bool(e.get("on") or e.get("clicked"))
        x, y, w, h = c0 * 16, r0 * 16, (c1 - c0 + 1) * 16, (r1 - r0 + 1) * 16
        if on:        # a SET / target bar -> BOLD solid green, unmistakable on the white panel
            svg.append(f'<rect x="{x-2}" y="{y-2}" width="{max(w,6)+4}" height="{max(h,6)+4}" '
                       f'fill="#00e676" fill-opacity="0.93" stroke="#064" stroke-width="2"/>')
        else:         # an unset bar -> quiet outline (the full array stays visible)
            svg.append(f'<rect x="{x}" y="{y}" width="{max(w,4)}" height="{max(h,4)}" '
                       f'fill="none" stroke="{col}" stroke-width="3"/>')
        if not str(eid).startswith("R/"):             # label scene entities only (bars stay unlabelled)
            svg.append(f'<text x="{x+2}" y="{max(y-5,24)}" fill="{col}" font-size="26" {_TS}>{eid}</text>')
    svg.append("</svg>")
    side = "".join(
        f'<div style="color:{("#ff9d2e" if e.get("needs_inspection") else _ROLE_COL.get(e.get("role"),"#ddd"))}">'
        f'{eid} [{e.get("resolution","?")}] {e.get("role","?")}'
        f'{" ON" if (e.get("on") or e.get("clicked")) else ""}</div>'
        for eid, e in ents.items() if not str(eid).startswith("R/"))
    skill = snap.get("active_skill") or ""
    events = "".join(f'<div style="color:#ffd23d">&#8226; {e.get("op","")}</div>'
                     for e in (snap.get("events") or []))
    try:                                              # timeago tracks the STATE mtime, NOT render time
        import os as _os
        from datetime import datetime as _dt, timezone as _tz
        from trace_render_utils import timeago_assets as _ta
        ta_css, ta_script = _ta.css, _ta.script
        _m = int(_os.path.getmtime(sp)) if sp.exists() else 0
        _iso = _dt.fromtimestamp(_m, _tz.utc).isoformat(timespec="seconds") if _m else "?"
        banner = (f'<div class="gen-banner">State updated '
                  f'<span class="timeago" data-gen="{_m}">just now</span> &middot; <code>{_iso}Z</code> '
                  f'(persists across refresh)</div>')
    except Exception:
        banner = ta_css = ta_script = ""
    _game, _level = snap.get("game", "?"), snap.get("level", "?")
    title = (f'<div style="padding:6px 8px;font-size:16px">'
             f'<b><a href="https://arcprize.org/tasks/{_game}" target="_blank" style="color:#6cf">'
             f'{_game}</a></b> &middot; level {_level} &middot; scene state (VLM)</div>')
    return (f'<!doctype html><meta charset=utf-8><title>{_game} L{_level} scene</title>'
            f'<style>{ta_css}body{{margin:0;background:#111;color:#ddd;font:13px monospace}}'
            f'#w{{display:flex}}#v{{flex:2;padding:6px}}#s{{flex:1;padding:8px;max-height:100vh;overflow:auto}}'
            f'.gen-banner{{margin:6px}}</style>'
            f'{title}{banner}{ta_script}'
            f'<div id=w><div id=v>{"".join(svg)}</div>'
            f'<div id=s><b>WHAT I JUST DID</b>{events}'
            f'<hr><b>true frame + markers (live)</b><br>'
            f'<i>solid GREEN = a bar in my TARGET program (set ON); cyan outline = an unset bar in '
            f'the right 6x6 array; coloured = scene/controls.</i>'
            f'<hr><div style="color:#9ad">{skill}</div><hr>{side}</div></div>'
            f'<script>new EventSource("/events").onmessage=function(e){{if(e.data=="update")location.reload()}}</script>')


def make_handler(root: Path):
    root = Path(root)

    class H(BaseHTTPRequestHandler):
        def _send(self, code, ctype, body):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if body is not None:
                self.wfile.write(body if isinstance(body, bytes) else body.encode("utf-8"))

        def do_GET(self):  # noqa: N802
            p = self.path.split("?")[0]
            try:
                if p in ("/", "/user"):
                    return self._send(200, "text/html", _USER_VIEW)
                if p == "/vlm":
                    return self._send(200, "text/html", _render_vlm(root))
                if p == "/trace":
                    t = root / "trace.html"
                    return self._send(200, "text/html",
                                      t.read_bytes() if t.exists() else b"<i>no trace yet</i>")
                if p == "/state.json":
                    f = root / "scene_state.json"
                    return self._send(200, "application/json",
                                      f.read_bytes() if f.exists() else b"{}")
                if p == "/frame.png":
                    f = root / "frame.png"
                    return (self._send(200, "image/png", f.read_bytes()) if f.exists()
                            else self._send(404, "text/plain", "no frame"))
                if p == "/events":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    f = root / "scene_state.json"
                    # seed `last` with the CURRENT mtime so connect does NOT look like a
                    # change (otherwise every reconnect fires "update" -> reload -> reconnect
                    # = an infinite reload loop that pins the CPU + makes the page blink).
                    last = f.stat().st_mtime if f.exists() else 0
                    self.wfile.write(b"data: init\n\n"); self.wfile.flush()
                    while True:
                        m = f.stat().st_mtime if f.exists() else 0
                        if m != last:
                            last = m
                            self.wfile.write(b"data: update\n\n"); self.wfile.flush()
                        time.sleep(1.0)
                return self._send(404, "text/plain", "not found")
            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                try:
                    return self._send(500, "text/plain", "error")
                except Exception:
                    return

        def log_message(self, *a):
            pass

    return H


def serve(root, port: int = 8090):
    """Start the state server on ``port`` serving ``root`` (where the driver writes
    scene_state.json + frame.png + trace.html).  Blocks; run in a thread/process."""
    httpd = ThreadingHTTPServer(("127.0.0.1", int(port)), make_handler(Path(root)))
    httpd.serve_forever()
    return httpd


if __name__ == "__main__":
    import sys
    _root = sys.argv[1] if len(sys.argv) > 1 else "."
    _port = int(sys.argv[2]) if len(sys.argv) > 2 else 8090
    serve(_root, _port)
