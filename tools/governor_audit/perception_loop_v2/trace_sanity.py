"""Trace-page sanity scanner -- automatically flag things that are OBVIOUSLY wrong
on a rendered exploratory trace, so senseless regressions are caught by a test
instead of by eye, run after run.

Game-agnostic + deterministic.  Every check degrades to SKIP (not FAIL) when its
input is unavailable, so a thin trace never produces false alarms.

Checks (each returns Problem records):
  - single_game        : no OTHER game's id leaks into the page (cross-game contamination)
  - referenced_images  : every <img>/file: ref EXISTS and is FRESH (not a stale prior-run PNG)
  - clicks_in_grid     : every CLICK:x,y is inside the 0..63 tick grid (no pixel-space leak)
  - clicks_in_playfield: every click lands inside the measured playfield bbox (not the HUD/void)
  - entity_bboxes      : perception entity bboxes are in-range, non-degenerate, not whole-frame
  - entity_count       : the registry hasn't exploded (name churn -> 71 aliases for ~8 objects)
  - game_responds      : at least one turn shows a real (>3px native) game-content change
  - model_recorded     : the page records which model produced the run

Usage:
  python trace_sanity.py [TRACE_DIR]      # default: <repo>/.tmp/training_data/latest
  -> prints a report; exit code 1 if any FAIL-severity problem is found.
"""
from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

N_TICKS = 64
GAME_ID_RE = re.compile(r"\b[a-z0-9]{4}-[0-9a-f]{8}\b")
CLICK_RE = re.compile(r"CLICK:(\d+),(\d+)")
IMG_REF_RE = re.compile(r'(?:src|href)\s*=\s*["\']([^"\']+)["\']')
# files a trace dir should NOT carry from a different run/game
SIBLING_HINTS = ("exploration_log", "turn_start_", "level_inspect")


@dataclass
class Problem:
    severity: str          # "FAIL" | "WARN" | "SKIP"
    check: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.severity}] {self.check}: {self.detail}"


def _native(path: Path):
    """Load a (possibly upscaled/gridded) frame PNG as a clean 64x64 RGB array by
    sampling the CENTRE of each upscale block -- avoids both the grid lines (drawn at
    block boundaries) and the tick labels (drawn at block corners), so the recovered
    game content is clean."""
    try:
        import numpy as np
        from PIL import Image
        a = np.array(Image.open(path).convert("RGB"))
        s = max(1, a.shape[0] // N_TICKS)
        off = s // 2
        return a[off::s, off::s][:N_TICKS, :N_TICKS]
    except Exception:
        return None


def _canonical_game(html: str) -> str | None:
    # traces use the SHORT game id (e.g. "r11l"), not the full "r11l-aa269680"
    m = re.search(r"Exploratory run\s*[—-]\s*(\S+?)\s+lc=", html)
    if m:
        return m.group(1)
    m = re.search(r"game_id</th><td>([^<]+)</td>", html)
    return m.group(1).strip() if m else None


_TURN_PNG_RE = re.compile(r"(.*[\\/]turn_\d+)[\\/][^\\/]+\.png$")


def _run_start_mtime(session_dir: Path | None) -> float | None:
    """RUN START = earliest prompt.md across the session's turn dirs (this run rewrites
    every turn's prompt).  A per-turn filmstrip OLDER than run start is a prior-run
    leftover (this turn produced no animation, so the reused dir kept the old one) --
    older by a whole run gap.  This run's own filmstrips are all newer than run start.
    (Run-level: a turn's prompt and its filmstrip are written moments apart, so a
    per-turn compare false-drops; and play-time filmstrips are older than render-time
    views/, so don't compare to those either.)"""
    if not session_dir or not session_dir.exists():
        return None
    m = [q.stat().st_mtime for q in session_dir.glob("turn_*/prompt.md")]
    return min(m) if m else None


# ---- individual checks -------------------------------------------------------

def check_single_game(html: str, canonical: str | None) -> list[Problem]:
    ids = set(GAME_ID_RE.findall(html))
    if not ids:
        return [Problem("SKIP", "single_game", "no game id found in page")]
    if canonical is None:
        return [Problem("WARN", "single_game", f"no canonical game in title; ids seen: {sorted(ids)}")]
    others = sorted(i for i in ids if i != canonical)
    if others:
        return [Problem("FAIL", "single_game",
                        f"page for {canonical} also references OTHER game(s): {others} "
                        f"(cross-game contamination / stale artifacts)")]
    return []


def check_referenced_images(html: str, trace_dir: Path,
                            run_start: float | None = None) -> list[Problem]:
    # Every referenced image must EXIST.  For per-turn session filmstrips, also check
    # it is not a PRIOR-run leftover: a filmstrip OLDER than RUN START (earliest
    # prompt.md across the turn dirs) was not regenerated this run -> it shows stale
    # (often a DIFFERENT game's) content -- the "PER-FRAME ENTITY ANALYSIS for an
    # unrelated game" leak.  Compared to run start, NOT render-time views/ (filmstrips
    # are written at play time, so a views compare false-flags every one).
    probs, seen = [], set()
    for ref in IMG_REF_RE.findall(html):
        if ref.startswith(("http:", "https:", "data:")):
            continue
        p = ref[len("file:///"):] if ref.startswith("file:") else ref
        path = Path(p) if Path(p).is_absolute() else (trace_dir / p)
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not path.exists():
            probs.append(Problem("FAIL", "referenced_images", f"missing image: {ref}"))
            continue
        if run_start is not None and _TURN_PNG_RE.match(str(path).replace("\\", "/")):
            if path.stat().st_mtime < run_start - 120:
                probs.append(Problem("FAIL", "referenced_images",
                                     f"STALE filmstrip (older than run start -> "
                                     f"prior-run leftover, likely a different game): "
                                     f"{ref}"))
    if not seen:
        return [Problem("SKIP", "referenced_images", "no local image refs")]
    return probs


def check_stale_siblings(trace_dir: Path) -> list[Problem]:
    th = trace_dir / "trace.html"
    if not th.exists():
        return []
    tm, probs = th.stat().st_mtime, []
    for child in trace_dir.iterdir():
        if child.name == "trace.html" or child.name == "views":
            continue
        if any(h in child.name for h in SIBLING_HINTS) and child.stat().st_mtime < tm - 3600:
            probs.append(Problem("WARN", "stale_siblings",
                                 f"stale artifact from a prior run/game: {child.name}"))
    return probs


def check_clicks_in_grid(html: str) -> list[Problem]:
    probs = []
    for x, y in CLICK_RE.findall(html):
        x, y = int(x), int(y)
        if not (0 <= x < N_TICKS and 0 <= y < N_TICKS):
            probs.append(Problem("FAIL", "clicks_in_grid",
                                 f"CLICK:{x},{y} is outside the 0..{N_TICKS-1} grid "
                                 f"(pixel-space leak / off the board)"))
    return probs


def _playfield_bbox(frames):
    """Union bbox of non-background pixels across frames (background = most common
    colour over all frames).  Returns (r0,c0,r1,c1) or None."""
    import numpy as np
    if not frames:
        return None
    stack = np.stack(frames)
    flat = stack.reshape(-1, 3)
    cols, counts = np.unique(flat, axis=0, return_counts=True)
    bg = cols[counts.argmax()]
    fg = np.any(np.any(stack != bg, axis=3), axis=0)   # any frame non-bg
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return None
    return int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())


def check_clicks_in_playfield(html: str, frames) -> list[Problem]:
    if not frames:
        return [Problem("SKIP", "clicks_in_playfield", "no frames to measure playfield")]
    bb = _playfield_bbox(frames)
    if bb is None:
        return [Problem("SKIP", "clicks_in_playfield", "could not measure playfield")]
    r0, c0, r1, c1 = bb
    if (r1 - r0) >= N_TICKS - 3 and (c1 - c0) >= N_TICKS - 3:
        return [Problem("SKIP", "clicks_in_playfield",
                        "playfield spans the whole frame -- cannot isolate a HUD/void region")]
    probs = []
    for sx, sy in set(CLICK_RE.findall(html)):
        x, y = int(sx), int(sy)              # x=col, y=row
        if not (0 <= x < N_TICKS and 0 <= y < N_TICKS):
            continue                          # already caught by clicks_in_grid
        if not (c0 <= x <= c1 and r0 <= y <= r1):
            probs.append(Problem("WARN", "clicks_in_playfield",
                                 f"CLICK:{x},{y} is outside the playfield bbox "
                                 f"rows[{r0}-{r1}] cols[{c0}-{c1}] (HUD/void click)"))
    return probs


def check_entity_bboxes(entities: list[dict]) -> list[Problem]:
    if not entities:
        return [Problem("SKIP", "entity_bboxes", "no perception entities found")]
    probs = []
    for e in entities:
        bb = e.get("bbox_ticks_turn1") or e.get("bbox_ticks") or e.get("bbox")
        nm = e.get("name", "?")
        if not (isinstance(bb, (list, tuple)) and len(bb) >= 4):
            continue
        r0, c0, r1, c1 = bb[:4]
        if not all(0 <= v <= N_TICKS for v in (r0, c0, r1, c1)):
            probs.append(Problem("FAIL", "entity_bboxes", f"{nm}: bbox out of range {bb}"))
        elif r1 < r0 or c1 < c0:
            probs.append(Problem("FAIL", "entity_bboxes", f"{nm}: inverted bbox {bb}"))
        elif (r1 - r0) >= N_TICKS - 2 and (c1 - c0) >= N_TICKS - 2:
            probs.append(Problem("WARN", "entity_bboxes",
                                 f"{nm}: bbox spans the whole frame {bb} (likely a bad detection)"))
    return probs


def check_entity_count(entities: list[dict], limit: int = 30) -> list[Problem]:
    n = len(entities or [])
    if n > limit:
        return [Problem("WARN", "entity_count",
                        f"{n} entities (>{limit}) -- likely name churn / alias explosion")]
    return []


def check_game_responds(frames) -> list[Problem]:
    if not frames or len(frames) < 2:
        return [Problem("SKIP", "game_responds", "fewer than 2 frames")]
    import numpy as np
    changed = sum(1 for i in range(1, len(frames))
                  if int(np.any(frames[i] != frames[i - 1], axis=2).sum()) > 3)
    if changed == 0:
        return [Problem("FAIL", "game_responds",
                        "NO turn produced a real (>3px) game-content change across the whole "
                        "trial -- actions are not reaching the game (frozen)")]
    return []


def check_model_recorded(html: str) -> list[Problem]:
    if re.search(r"(?:acting_model|perception_model|\bmodel\b)\s*[<:]", html, re.I):
        return []
    return [Problem("WARN", "model_recorded", "page does not record the model used")]


# ---- orchestration ----------------------------------------------------------

def _locate_session(html: str) -> Path | None:
    for ref in IMG_REF_RE.findall(html):
        m = re.search(r"(.*[\\/][a-z]+_session)[\\/]turn_\d+", ref.replace("file:///", ""))
        if m:
            p = Path(m.group(1))
            if p.exists():
                return p
    return None


def _load_frames(cos_session: Path | None, trace_dir: Path):
    frames = []
    src = None
    if cos_session and cos_session.exists():
        src = sorted(cos_session.glob("turn_*/curr_frame.png"))
    if not src:
        src = sorted((trace_dir / "views").glob("frame_*.png"))
    for f in src:
        fr = _native(f)
        if fr is not None:
            frames.append(fr)
    return frames


def _load_entities(cos_session: Path | None) -> list[dict]:
    """Latest turn's perception entities (the actor's current view)."""
    if not cos_session or not cos_session.exists():
        return []
    for turn in sorted(cos_session.glob("turn_*"), reverse=True):
        for fn in ("reply.consumed.txt", "reply.txt"):
            f = turn / fn
            if not f.exists():
                continue
            try:
                s = f.read_text(encoding="utf-8", errors="replace")
                d = json.loads(s[s.find("{"):])
                ents = (d.get("perception") or {}).get("entities")
                if ents:
                    return ents
            except Exception:
                continue
    return []


def scan_trace_dir(trace_dir: str | Path) -> list[Problem]:
    trace_dir = Path(trace_dir)
    th = trace_dir / "trace.html"
    if not th.exists():
        return [Problem("FAIL", "trace_exists", f"no trace.html in {trace_dir}")]
    html = th.read_text(encoding="utf-8", errors="replace")
    canonical = _canonical_game(html)
    cos = _locate_session(html)
    frames = _load_frames(cos, trace_dir)
    entities = _load_entities(cos)

    run_start = _run_start_mtime(cos)
    probs: list[Problem] = []
    probs += check_single_game(html, canonical)
    probs += check_referenced_images(html, trace_dir, run_start)
    probs += check_stale_siblings(trace_dir)
    probs += check_clicks_in_grid(html)
    probs += check_clicks_in_playfield(html, frames)
    probs += check_entity_bboxes(entities)
    probs += check_entity_count(entities)
    probs += check_game_responds(frames)
    probs += check_model_recorded(html)
    return probs


def report(probs: list[Problem]) -> int:
    fails = [p for p in probs if p.severity == "FAIL"]
    warns = [p for p in probs if p.severity == "WARN"]
    for p in probs:
        if p.severity != "SKIP":
            print(p)
    print(f"\ntrace sanity: {len(fails)} FAIL, {len(warns)} WARN, "
          f"{sum(1 for p in probs if p.severity == 'SKIP')} skipped")
    return 1 if fails else 0


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if argv:
        trace_dir = argv[0]
    else:
        repo = Path(__file__).resolve().parents[3]
        trace_dir = repo / ".tmp" / "training_data" / "latest"
    print(f"scanning {trace_dir}")
    return report(scan_trace_dir(trace_dir))


if __name__ == "__main__":
    raise SystemExit(main())
