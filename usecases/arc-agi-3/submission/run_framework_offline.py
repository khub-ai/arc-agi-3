"""Run CosAgent through the framework's REAL Agent.main() loop, OFFLINE.

This is the faithful offline submission path (the framework's main.py is online -
it fetches the game list from /api/games). We do what Swarm does internally:
build the SDK's LocalEnvironmentWrapper via Arcade(OFFLINE).make(game), instantiate
CosAgent (a real `agents.agent.Agent` subclass), and run `agent.main()` bounded by
MAX_ACTIONS. Reports the max level reached.

Usage: python run_framework_offline.py [game-id] [max-actions]
Env:   QWEN_MODEL_SLUG (model slug), MANIFEST/seed handled here.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SUB = REPO / "usecases" / "arc-agi-3" / "submission"
FRAMEWORK = SUB / "ARC-AGI-3-Agents"

os.environ["COS_STRICT"] = "1"
os.environ["ARC_AGI_3_REPO"] = str(REPO)
os.environ["OPERATION_MODE"] = "offline"
os.environ.setdefault("ARC_API_KEY", "")          # offline; no API calls
os.environ["COS_WORKDIR"] = str(SUB / ".run")
# KB mode (see COLD_SOLVE_STATUS.md). COLD (default) = competition-faithful: a
# fresh KB seeded with general Tier-0 only, no per-game prior. WARM = the
# accumulated .tmp/kb (per-game KB present) for a cold-vs-warm A/B.
_WARM = "--warm-kb" in sys.argv
os.environ["COS_KB_ROOT"] = (str(REPO / ".tmp" / "kb") if _WARM
                             else str(SUB / ".run" / "fw_kb"))
os.environ["COS_SESSION_DIR"] = str(SUB / ".run" / "fw_session")
os.environ.setdefault("QWEN_MODEL_SLUG",
                      "ollama/127.0.0.1:11434/qwen3-vl:8b-instruct")
for _p in (str(SUB), str(FRAMEWORK),
           str(REPO / "tools/governor_audit/perception_loop_v2"),
           str(REPO / "usecases/arc-agi-3/python")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    import shutil
    shutil.rmtree(Path(os.environ["COS_SESSION_DIR"]), ignore_errors=True)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    game = args[0] if args else "tn36-ef4dde99"
    maxact = int(args[1]) if len(args) > 1 else 40

    if _WARM:
        print(f"[fw] *** WARM KB *** {os.environ['COS_KB_ROOT']} (per-game KB "
              "present -- NOT competition-faithful)", flush=True)
    else:
        import kaggle_entry
        kaggle_entry.seed_general_kb()
        print(f"[fw] COLD KB {os.environ['COS_KB_ROOT']}: general Tier-0 seed only, "
              "no per-game prior (competition-faithful)", flush=True)

    from arc_agi import Arcade, OperationMode
    env = Arcade(operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files")).make(game)
    asp = list(getattr(env, "action_space", []) or [])
    base = list(getattr(getattr(env, "info", None), "baseline_actions", []) or [])
    print(f"[fw] game={game}  action_space={asp}  baseline={base}", flush=True)

    from cos_agent import CosAgent
    agent = CosAgent(card_id="local", game_id=game, agent_name="cosagent",
                     ROOT_URL="", record=False, arc_env=env, tags=[])
    agent.MAX_ACTIONS = maxact
    print(f"[fw] model={os.environ['QWEN_MODEL_SLUG']}  running Agent.main() "
          f"bounded to {maxact} actions ...", flush=True)

    t0 = time.time()
    try:
        agent.main()
    except Exception:
        import traceback
        traceback.print_exc()

    max_lc = 0
    for f in getattr(agent, "frames", []):
        try:
            max_lc = max(max_lc, int(getattr(f, "levels_completed", 0)))
        except Exception:
            pass
    print(f"\n[fw] DONE  max_lc={max_lc}  actions={getattr(agent,'action_counter',0)}"
          f"  frames={len(getattr(agent,'frames',[]))}  elapsed={time.time()-t0:.0f}s",
          flush=True)


if __name__ == "__main__":
    main()
