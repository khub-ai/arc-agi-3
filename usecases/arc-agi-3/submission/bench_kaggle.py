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
_DONE = re.compile(r"max_lc=(\d+)\s+actions=(\d+)")
_GROUND = re.compile(r"grounding QA score=([0-9.]+)")


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


def play(game, maxact):
    env = dict(os.environ, QWEN_MODEL_SLUG="vllm/127.0.0.1:8000/bench",
               COS_ENVIRONMENTS_DIR=f"{COMP}/environment_files")
    r = subprocess.run([sys.executable, str(RUNNER), game, str(maxact)],
                       env=env, capture_output=True, text=True)
    out = r.stdout + r.stderr
    g = [float(x) for x in _GROUND.findall(out)]
    mg = round(sum(g) / len(g), 2) if g else None
    m = _DONE.search(out)
    return (int(m.group(1)) if m else None), mg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", default="")
    ap.add_argument("--games", default="ka59,ls20")
    ap.add_argument("--max-actions", type=int, default=60)
    args = ap.parse_args()

    models = [m for m in args.model_dirs.split(",") if m] or discover_models()
    games = [g.strip() for g in args.games.split(",") if g.strip()]
    if not models:
        raise SystemExit("no attached safetensors models found under /kaggle/input")

    def label(md):
        return "/".join(Path(md).parts[-3:])

    res = {}
    for md in models:
        print(f"\n##### serving {label(md)} #####", flush=True)
        p = serve(md)
        try:
            for game in games:
                lc, mg = play(game, args.max_actions)
                print(f"  {game:8} lc={lc} grounding={mg}", flush=True)
                res[(md, game)] = (lc, mg)
        finally:
            stop(p)

    for title, idx in (("lc-advance", 0), ("grounding / perception quality", 1)):
        print(f"\n=== {title} ===")
        print(f"{'model':40} " + " ".join(f"{g:>8}" for g in games))
        for md in models:
            cells = " ".join(f"{res.get((md, g), (None, None))[idx]!s:>8}" for g in games)
            print(f"{label(md):40} {cells}")


if __name__ == "__main__":
    main()
