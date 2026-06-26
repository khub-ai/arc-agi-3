"""Durable, ORGANIZED per-game/per-level archive of solved play.

Every time COS solves a game level, we persist a long-term, perusable record --
for another AI, a human, or a knowledge-mining program -- under
``<repoRoot>/.archive/trace_pages/<game_id>/lc<level>/`` (kept locally; gitignored;
``.archive/`` itself is reserved for other future categories).  Unlike
the ephemeral ``.tmp`` work-dirs and the single ``latest`` trace (which each run
overwrites), this archive is keyed by GAME + LEVEL, indexed, never silently
clobbered (a prior record for the same level is moved to ``history/<n>/``), and
written in both machine-readable (JSON) and human-readable (the HTML trace)
forms.

Per-level entry:
  record.json              -- machine-readable summary (win-path, score,
                              solution id, digest, grounding-QA, timestamps)
  trace.html + views/      -- the human-readable turn-by-turn trace
  world_knowledge.json     -- the full detailed log (actions, deltas, entities,
                              hypotheses) for mining
  level_start_analysis.json-- the entity analysis + grounding-QA at level start
  history/<n>/...          -- prior versions, never overwritten

Indexes (rebuilt each write): ``.archive/trace_pages/index.json`` (all records) and
``.archive/trace_pages/INDEX.md`` (a human-readable table).

Game-agnostic; pure stdlib; fully guarded so archiving NEVER breaks a run.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def repo_root() -> Path:
    """Walk up from this file to the repository root (the dir holding .git)."""
    cur = Path(__file__).resolve()
    for p in [cur] + list(cur.parents):
        if (p / ".git").exists():
            return p
    # fallback: tools/governor_audit/perception_loop_v2 -> repo root
    return cur.parents[3] if len(cur.parents) >= 4 else cur.parent


def archive_root() -> Path:
    import os
    env = os.environ.get("COS_ARCHIVE_ROOT")
    # Trace pages live under .archive/trace_pages/ so .archive/ itself can hold other categories.
    return Path(env) if env else repo_root() / ".archive" / "trace_pages"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _level_dir(game_id: str, level: int) -> Path:
    return archive_root() / str(game_id) / f"lc{int(level)}"


def _rotate_to_history(ld: Path) -> Optional[str]:
    """Move the current canonical files (everything except history/) into
    history/<n>/ so a prior record is preserved, not overwritten.  Returns the
    history slot name, or None if there was nothing to rotate."""
    if not (ld / "record.json").exists():
        return None
    hist = ld / "history"
    hist.mkdir(parents=True, exist_ok=True)
    n = 1 + sum(1 for d in hist.iterdir() if d.is_dir())
    slot = hist / f"v{n}"
    slot.mkdir(parents=True, exist_ok=True)
    for item in ld.iterdir():
        if item.name == "history":
            continue
        shutil.move(str(item), str(slot / item.name))
    return slot.name


def archive_solve(*, game_id: str, level: int, work_dir, win_path=None,
                  solution_id: Optional[str] = None, score=None,
                  digest=None, grounding_qa=None, source_session: str = "",
                  frame_dir=None, render: bool = True,
                  extra: Optional[dict] = None) -> Optional[Path]:
    """Write a durable archive entry for a solved (game, level).  Copies the
    detailed logs + (re)renders the trace from ``work_dir``; preserves any prior
    record under history/; rebuilds the index.  Returns the entry dir, or None on
    failure (guarded -- never raises)."""
    try:
        work = Path(work_dir)
        ld = _level_dir(game_id, level)
        rotated = _rotate_to_history(ld) if ld.exists() else None
        ld.mkdir(parents=True, exist_ok=True)

        # 1) detailed log + level-start analysis (+ grounding QA pulled from it)
        wk = work / "world_knowledge.json"
        if wk.exists():
            shutil.copy2(wk, ld / "world_knowledge.json")
        lsa = None
        cands = sorted(work.glob("turn_*/level_start/level_start_analysis.json"))
        if cands:
            lsa = cands[-1]
            shutil.copy2(lsa, ld / "level_start_analysis.json")
            if grounding_qa is None:
                try:
                    grounding_qa = json.loads(lsa.read_text(encoding="utf-8")).get("grounding_qa")
                except Exception:
                    pass

        # 2) human-readable trace (render fresh if possible, else copy existing)
        trace_ok = False
        if render and frame_dir is not None:
            try:
                import render_exploratory_run as _R
                out = _R.render(work, Path(frame_dir))
                shutil.copy2(out, ld / "trace.html")
                vsrc = work / "views"
                if vsrc.exists():
                    shutil.copytree(vsrc, ld / "views", dirs_exist_ok=True)
                trace_ok = True
            except Exception:
                trace_ok = False
        if not trace_ok and (work / "index.html").exists():
            shutil.copy2(work / "index.html", ld / "trace.html")
            if (work / "views").exists():
                shutil.copytree(work / "views", ld / "views", dirs_exist_ok=True)

        # 2b) STATE-SPACE SNAPSHOTS (state-as-medium): the canonical scene state +
        # its per-turn history, kept next to the trace for AI/human/miner perusal.
        ss = work / "scene_state.json"
        if ss.exists():
            shutil.copy2(ss, ld / "scene_state.json")
        sh = work / "scene_history"
        if sh.exists():
            shutil.copytree(sh, ld / "scene_history", dirs_exist_ok=True)

        # 3) machine-readable record
        record = {
            "game_id": str(game_id), "level": int(level),
            "score": score, "n_acts": len(win_path or []),
            "solved_at": _now(), "source_session": source_session,
            "solution_id": solution_id, "win_path": win_path or [],
            "digest": digest, "grounding_qa": grounding_qa,
            "artifacts": {
                "trace": "trace.html" if (ld / "trace.html").exists() else None,
                "world_knowledge": "world_knowledge.json" if (ld / "world_knowledge.json").exists() else None,
                "level_start_analysis": "level_start_analysis.json" if (ld / "level_start_analysis.json").exists() else None,
                "scene_state": "scene_state.json" if (ld / "scene_state.json").exists() else None,
                "scene_history": "scene_history" if (ld / "scene_history").exists() else None,
            },
            "rotated_prior_to": rotated,
        }
        if extra:
            record.update(extra)
        (ld / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

        rebuild_index()
        return ld
    except Exception as e:
        print(f"[archive] archive_solve skipped ({e})")
        return None


def update_digest(game_id: str, level: int, digest) -> None:
    """Patch the crystallised mechanic digest into an existing record (the digest
    is crystallised at the NEXT level start, after archive_solve).  Guarded."""
    try:
        rec = _level_dir(game_id, level) / "record.json"
        if not rec.exists() or digest is None:
            return
        d = json.loads(rec.read_text(encoding="utf-8"))
        d["digest"] = digest
        rec.write_text(json.dumps(d, indent=2), encoding="utf-8")
        rebuild_index()
    except Exception as e:
        print(f"[archive] update_digest skipped ({e})")


def rebuild_index() -> None:
    """Rebuild .archive/trace_pages/index.json + INDEX.md from every per-level record.json.
    Guarded -- never raises."""
    try:
        root = archive_root()
        if not root.exists():
            return
        rows = []
        for rec in sorted(root.glob("*/lc*/record.json")):
            try:
                d = json.loads(rec.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append({
                "game_id": d.get("game_id"), "level": d.get("level"),
                "score": d.get("score"), "n_acts": d.get("n_acts"),
                "solved_at": d.get("solved_at"), "solution_id": d.get("solution_id"),
                "path": str(rec.parent.relative_to(root)).replace("\\", "/"),
                "has_trace": bool((rec.parent / "trace.html").exists()),
            })
        rows.sort(key=lambda r: (str(r.get("game_id")), int(r.get("level") or 0)))
        (root / "index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        md = ["# Solved game/level archive", "",
              "Durable per-game/per-level play records (gitignored). One row per "
              "solved level; see each `path/record.json` + `trace.html`.", "",
              "| game | level | score | acts | solved_at (UTC) | solution_id | trace | path |",
              "|------|------:|------:|-----:|-----------------|-------------|:----:|------|"]
        for r in rows:
            md.append(f"| {r['game_id']} | {r['level']} | {r['score']} | {r['n_acts']} | "
                      f"{r['solved_at']} | {r['solution_id'] or ''} | "
                      f"{'yes' if r['has_trace'] else 'no'} | {r['path']} |")
        (root / "INDEX.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    except Exception as e:
        print(f"[archive] rebuild_index skipped ({e})")
