"""Ground-truth trial validator — the SINGLE source of truth for game state.

WHY
---
I keep asserting state/progress from assumptions (e.g. "sw2 pierced" when the
blue swatch was white from turn 1) instead of from the actual frames. This tool
reads the frames directly and reports ONLY what is grounded:
  - HUD swatches vs the LEVEL-START BASELINE (a pierce = white-count INCREASE
    from baseline, per swatch -- never an absolute threshold, since the blue
    swatch renders some white even unpierced);
  - block positions at the current turn (connected components);
  - the DELTA vs the previous turn (what moved / NO-OP);
  - score + win flag.
No claim about state should be made without running this.

Usage: python trial_validate.py --work-dir <wd> --frame-dir <fd> [--baseline N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

from PIL import Image

RED = lambda r, g, b: r > 180 and g < 90 and b < 90          # noqa: E731
BLUE = lambda r, g, b: b > 170 and r < 100                   # noqa: E731
WHITE = lambda r, g, b: r > 185 and g > 185 and b > 185      # noqa: E731


def _frame(fd: str, t: int):
    p = os.path.join(fd, f"turn_{t}.png")
    return Image.open(p).convert("RGB") if os.path.exists(p) else None


def _turns(work_dir: str):
    ts = []
    for d in glob.glob(os.path.join(work_dir, "turn_*")):
        s = os.path.basename(d).split("_")[1]
        if s.isdigit():
            ts.append(int(s))
    return sorted(ts)


def hud_swatches(im):
    """Detect swatches in the HUD strip and their white fraction. Returns a list
    of dicts {color, lo, hi, white, cells} left-to-right."""
    px = im.load()
    W, H = im.size
    rows = range(max(0, H - 8), H - 2)        # bottom strip
    # find the dominant HUD row (most colored pixels)
    best_r, best_n = None, -1
    for r in range(max(0, H - 8), H - 1):
        n = sum(1 for c in range(W) if RED(*px[c, r]) or BLUE(*px[c, r]))
        if n > best_n:
            best_n, best_r = n, r
    if best_r is None:
        return []
    # group contiguous colored runs on the HUD band
    runs = []
    c = 0
    while c < W:
        col = None
        rr, gg, bb = px[c, best_r]
        if RED(rr, gg, bb):
            col = "R"
        elif BLUE(rr, gg, bb):
            col = "B"
        if col:
            start = c
            while c < W:
                r2, g2, b2 = px[c, best_r]
                if (col == "R" and RED(r2, g2, b2)) or (col == "B" and BLUE(r2, g2, b2)):
                    c += 1
                else:
                    break
            if c - start >= 2:
                runs.append((col, start, c - 1))
        else:
            c += 1
    out = []
    for col, lo, hi in runs:
        white = sum(1 for r in range(H - 8, H)
                    for c2 in range(lo, hi + 1) if WHITE(*px[c2, r]))
        out.append({"color": "RED" if col == "R" else "BLUE",
                    "lo": lo, "hi": hi, "white": white})
    return out


def _cca(im, pred, play_rows=56, min_area=4):
    """Connected components matching pred (excluding the HUD strip). Returns
    bboxes (top,left,bottom,right)."""
    px = im.load()
    W, H = im.size
    H = min(H, play_rows)
    seen = [[False] * W for _ in range(H)]
    boxes = []
    for r in range(H):
        for c in range(W):
            if seen[r][c] or not pred(*px[c, r]):
                continue
            stack = [(r, c)]
            seen[r][c] = True
            t = b = r
            l = rr = c
            while stack:
                y, x = stack.pop()
                t = min(t, y); b = max(b, y); l = min(l, x); rr = max(rr, x)
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and pred(*px[nx, ny]):
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if (b - t + 1) * (rr - l + 1) >= min_area:
                boxes.append((t, l, b, rr))
    return sorted(boxes, key=lambda x: (x[0], x[1]))


def validate(work_dir: str, frame_dir: str, baseline: int = None,
             turn: int = None) -> dict:
    ts = _turns(work_dir)
    if not ts:
        return {"error": "no turns"}
    cur = turn or ts[-1]
    base = baseline or ts[0]
    fcur, fbase = _frame(frame_dir, cur), _frame(frame_dir, base)
    prev = max([t for t in ts if t < cur], default=base)
    fprev = _frame(frame_dir, prev)
    rep = {"baseline_turn": base, "current_turn": cur, "prev_turn": prev}
    try:
        wk = json.load(open(os.path.join(work_dir, "world_knowledge.json")))
        rep["score"] = wk.get("score")
        rep["win_state"] = wk.get("win_state")
    except Exception:
        pass
    if fcur and fbase:
        sb, sc = hud_swatches(fbase), hud_swatches(fcur)
        sws = []
        for i, s in enumerate(sc):
            bw = sb[i]["white"] if i < len(sb) else None
            pierced = (bw is not None and s["white"] > bw)
            sws.append({"i": i + 1, "color": s["color"], "baseline_white": bw,
                        "now_white": s["white"], "pierced_vs_baseline": pierced})
        rep["hud"] = sws
        rep["pierced_count"] = sum(1 for s in sws if s["pierced_vs_baseline"])
    if fcur:
        rep["reds_now"] = _cca(fcur, RED)
        rep["blues_now"] = _cca(fcur, BLUE)
    if fcur and fprev:
        rp, bp = _cca(fprev, RED), _cca(fprev, BLUE)
        moved = (rp != rep.get("reds_now") or bp != rep.get("blues_now"))
        rep["delta_vs_prev"] = "MOVED" if moved else "NO-OP (nothing moved)"
    return rep


def format_report(rep: dict) -> str:
    if "error" in rep:
        return f"VALIDATE: {rep['error']}"
    L = [f"=== VALIDATED STATE (ground truth) — turn {rep['current_turn']} "
         f"(baseline {rep['baseline_turn']}), score={rep.get('score')} "
         f"win={rep.get('win_state')} ==="]
    for s in rep.get("hud", []):
        v = ("PIERCED (white rose from baseline)" if s["pierced_vs_baseline"]
             else "not pierced")
        L.append(f"  swatch {s['i']} {s['color']}: baseline_white="
                 f"{s['baseline_white']} now={s['now_white']} -> {v}")
    if "pierced_count" in rep:
        L.append(f"  => {rep['pierced_count']}/{len(rep.get('hud', []))} "
                 f"pierced vs the level-start baseline.")
    L.append(f"  reds now: {rep.get('reds_now')}")
    L.append(f"  blues now: {rep.get('blues_now')}")
    L.append(f"  delta vs turn {rep.get('prev_turn')}: {rep.get('delta_vs_prev')}")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--frame-dir", required=True)
    ap.add_argument("--baseline", type=int, default=None)
    ap.add_argument("--turn", type=int, default=None)
    a = ap.parse_args()
    print(format_report(validate(a.work_dir, a.frame_dir, a.baseline, a.turn)))
