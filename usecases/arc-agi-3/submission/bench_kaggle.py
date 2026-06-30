"""On-Kaggle VLM bake-off: serve each attached model + play games through COS.

Runs INSIDE a Kaggle RTX PRO 6000 notebook (96 GB). For each attached model it
launches serve_vlm.py (transformers, bf16), plays each game via
run_framework_offline.py against that local endpoint, and reports lc-advance +
mean grounding (perception quality). Models are served ONE AT A TIME, so a 32B in
bf16 (~64 GB) fits; each server is killed before the next loads, freeing the GPU.

Attach: the `cos-code` dataset, the competition data, and the candidate Models.
Then, in a cell:

    !python /kaggle/input/<cos-code>/usecases/arc-agi-3/submission/bench_kaggle.py \
        --games ka59,ls20 --max-actions 60

With no --model-dirs it benchmarks every attached safetensors model (skipping the
competition's framework). Pass --model-dirs a,b,c to pick/order them.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SUB = Path(__file__).resolve().parent
SERVE = SUB / "kaggle" / "serve_vlm.py"
RUNNER = SUB / "run_framework_offline.py"
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-3"
REPO = SUB.parents[2]                      # .../usecases/arc-agi-3/submission -> repo root
_DONE = re.compile(r"max_lc=(\d+)\s+actions=(\d+)")
_GROUND = re.compile(r"grounding QA score=([0-9.]+)")


def default_envs_dir():
    """Where the games (environment_files) live. Prefer the attached competition
    data; fall back to the repo's own environment_files -- so the dry-run works
    WITHOUT the competition data attached (internet can stay ON to clone the repo,
    avoiding the competition-forces-internet-off trap)."""
    for d in (f"{COMP}/environment_files", str(REPO / "environment_files")):
        if os.path.isdir(d):
            return d
    return None


def discover_models():
    out = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(cfg)
        if COMP in d:                       # skip the competition's own files
            continue
        try:
            if any(f.endswith(".safetensors") for f in os.listdir(d)):
                out.append(d)
        except Exception:
            pass
    return sorted(out)


def serve(model_dir, port=8000):
    log = open("/kaggle/working/serve_vlm.log", "w")
    p = subprocess.Popen([sys.executable, str(SERVE), "--model", model_dir,
                          "--port", str(port), "--name", "bench"],
                         stdout=log, stderr=subprocess.STDOUT)
    for _ in range(360):                    # up to ~30 min (a 32B load is slow)
        if p.poll() is not None:
            raise SystemExit("serve_vlm exited early; see /kaggle/working/serve_vlm.log")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5)
            return p
        except Exception:
            time.sleep(5)
    p.terminate()
    raise SystemExit("serve_vlm did not come up in time")


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=30)                  # ensure the GPU memory is released
    except Exception:
        p.kill()
    time.sleep(8)


_TRACE = re.compile(r"grounding QA|\[fw\]|\[perception\]|remap|cands|levels_completed|SCORED|Traceback|"
                    r"Error|Exception|ACTION|CLICK|RESET|turn[_ ]?\d|claim|triage",
                    re.IGNORECASE)


def play(game, maxact, envs_dir):
    env = dict(os.environ, QWEN_MODEL_SLUG="vllm/127.0.0.1:8000/bench")
    if envs_dir:
        env["COS_ENVIRONMENTS_DIR"] = envs_dir
    t0 = time.time()
    # Stream the KEY per-turn lines LIVE (perception/grounding/actions/errors) so a
    # dry run is observable, while still capturing the full output for the lc parse.
    p = subprocess.Popen([sys.executable, str(RUNNER), game, str(maxact)], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    lines = []
    for line in p.stdout:                  # type: ignore[union-attr]
        lines.append(line)
        if _TRACE.search(line):
            print(f"    | {line}", end="", flush=True)
    p.wait()
    el = time.time() - t0                  # wall-clock for this game (serving warm)
    out = "".join(lines)
    g = [float(x) for x in _GROUND.findall(out)]
    mg = round(sum(g) / len(g), 2) if g else None
    m = _DONE.search(out)
    lc = int(m.group(1)) if m else None
    acts = int(m.group(2)) if m else 0
    spt = el / max(1, acts)                 # the number that calibrates the budget
    res = {"lc": lc, "actions": acts, "grounding": mg,
           "elapsed_s": round(el), "s_per_turn": round(spt, 1), "err": None}
    if m is None:                           # no DONE line -> surface WHY (don't swallow)
        res["err"] = "\n".join(out.strip().splitlines()[-15:]) or "<no output>"
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", default="")
    ap.add_argument("--games", default="ka59,ls20")
    ap.add_argument("--max-actions", type=int, default=60)
    ap.add_argument("--envs-dir", default=None,
                    help="environment_files dir (default: competition data if attached, "
                         "else the repo's environment_files)")
    args = ap.parse_args()

    models = [m for m in args.model_dirs.split(",") if m] or discover_models()
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    envs_dir = args.envs_dir or default_envs_dir()
    if not models:
        raise SystemExit("no attached safetensors models found under /kaggle/input")
    print(f"[bench] games dir: {envs_dir}", flush=True)

    def label(md):
        return "/".join(Path(md).parts[-3:])

    res = {}
    for md in models:
        print(f"\n##### serving {label(md)} #####", flush=True)
        t_load = time.time()
        p = serve(md)
        print(f"  [serve up in {time.time() - t_load:.0f}s]", flush=True)
        try:
            for game in games:
                r = play(game, args.max_actions, envs_dir)
                print(f"  {game:8} lc={r['lc']} actions={r['actions']} "
                      f"grounding={r['grounding']} {r['elapsed_s']}s "
                      f"({r['s_per_turn']}s/turn)", flush=True)
                if r.get("err"):
                    print(f"    !! {game} produced no result -- last lines:\n"
                          + "\n".join("      " + ln for ln in r["err"].splitlines()),
                          flush=True)
                res[(md, game)] = r
        finally:
            stop(p)

    cols = [("lc-advance", "lc"), ("grounding (perception)", "grounding"),
            ("PACE s/turn  <-- budget driver", "s_per_turn")]
    for title, key in cols:
        print(f"\n=== {title} ===")
        print(f"{'model':40} " + " ".join(f"{g:>10}" for g in games))
        for md in models:
            cells = " ".join(f"{(res.get((md, g)) or {}).get(key)!s:>10}" for g in games)
            print(f"{label(md):40} {cells}")


if __name__ == "__main__":
    main()
