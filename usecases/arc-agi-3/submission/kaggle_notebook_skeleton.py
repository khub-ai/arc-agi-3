"""Kaggle (offline) entry-notebook SKELETON for the ARC-AGI-3 submission.

Paste these cells (split on `# %%`) into a Kaggle Notebook. It runs COS's real
loop offline against the eval games with a bundled local model. Datasets assumed
attached as Kaggle inputs:
  - cos-code  : output of build_package.py (this repo's runtime closure + framework)
  - cos-wheels: offline wheelhouse (arc-agi/arcengine + libs)
  - qwen-model: the model blob + Ollama binary (or swap for vLLM/SGLang)

!!! VERIFY FIRST: the EXACT Kaggle offline-eval invocation. The framework's
    main.py is ONLINE (fetches games from /api/games). This skeleton uses the
    faithful offline pattern (build LocalEnvironmentWrapper + run Agent.main()),
    but confirm against the competition's Code tab / starter notebook how the
    HIDDEN eval games are provided and how the agent is scored.
"""

# %% [cell 1] offline + strict env
import os
os.environ["COS_STRICT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OPERATION_MODE"] = "offline"
os.environ["ARC_API_KEY"] = ""                       # no online API
os.environ["COS_KB_ROOT"] = "/kaggle/working/kb"      # fresh, writable, seeded below
os.environ["COS_WORKDIR"] = "/kaggle/working/cos"
# Model slug. Recommended: gemma-4-31b via LOCAL vLLM (cell 4 sets this too).
os.environ["QWEN_MODEL_SLUG"] = "vllm/127.0.0.1:8000/gemma-4-31b-it"
# (8B-on-Ollama fallback: "ollama/127.0.0.1:11434/qwen3-vl:8b-instruct")

# %% [cell 2] install the SDK from the offline wheelhouse (no internet)
# !! arc-agi/arcengine REQUIRE Python >=3.12 -- the Kaggle image MUST be 3.12,
#    else this fails (and the SDK cannot install on 3.11 at all). The competition
#    offline image may already ship arc-agi preinstalled -- check first.
# Use --no-deps so the wheelhouse numpy/pillow do NOT clobber the versions
# Kaggle's torch/transformers were built against (the big ML libs come from the
# preinstalled image):
# !pip install --no-index --find-links=/kaggle/input/cos-wheels --no-deps arc-agi arcengine
# (If a dep is reported missing on the image, drop --no-deps -- the full closure
#  is bundled in cos-wheels.)

# %% [cell 3] put the bundled code on sys.path
import sys
BUNDLE = "/kaggle/input/cos-code"
for p in (BUNDLE,
          f"{BUNDLE}/tools/governor_audit/perception_loop_v2",
          f"{BUNDLE}/usecases/arc-agi-3/python",
          f"{BUNDLE}/usecases/arc-agi-3/submission",
          f"{BUNDLE}/ARC-AGI-3-Agents"):
    if p not in sys.path:
        sys.path.insert(0, p)

# %% [cell 4] serve the model
# OPTION A (recommended, gemma-4-31b): vLLM serving the ATTACHED Kaggle Model.
#   Attach google/gemma-4 (31b-it) via "Add Input -> Models" -> mounts under
#   /kaggle/input. NO upload. vLLM must be present + recent enough for gemma-4
#   (Kaggle GPU images usually ship vLLM; verify, else add it to the wheelhouse).
import subprocess, time, glob, urllib.request
GEMMA_DIR = next(iter(sorted(glob.glob("/kaggle/input/gemma-4/**/31b-it/**/",
                                       recursive=True))), "/kaggle/input/gemma-4")
subprocess.Popen(
    ["vllm", "serve", GEMMA_DIR, "--port", "8000",
     "--served-model-name", "gemma-4-31b-it",
     "--max-model-len", "8192", "--gpu-memory-utilization", "0.92"],
    env={**os.environ})
for _ in range(120):                                   # 31B load is slow
    try:
        urllib.request.urlopen("http://127.0.0.1:8000/v1/models", timeout=5); break
    except Exception:
        time.sleep(10)
os.environ["QWEN_MODEL_SLUG"] = "vllm/127.0.0.1:8000/gemma-4-31b-it"
import vllm_backend; vllm_backend.install()             # route vllm/ slugs locally
print("vLLM up; slug =", os.environ["QWEN_MODEL_SLUG"])

# OPTION B (fallback, 8B): Ollama + a bundled qwen-model dataset (PACKAGE_MODEL.md):
#   MODEL_DS="/kaggle/input/qwen-model"; os.environ["OLLAMA_MODELS"]=f"{MODEL_DS}/models"
#   subprocess.Popen([f"{MODEL_DS}/bin/ollama","serve"]); time.sleep(10)
#   os.environ["QWEN_MODEL_SLUG"]="ollama/127.0.0.1:11434/qwen3-vl:8b-instruct"

# %% [cell 5] seed the competition KB (general Tier-0 knowledge only)
import kaggle_entry
kaggle_entry.seed_general_kb()

# %% [cell 6] run COS over the eval games  (offline, faithful pattern)
#   --- ADAPT to the verified Kaggle eval flow (see header warning) ---
from arc_agi import Arcade, OperationMode
from cos_agent import CosAgent

ENV_DIR = "/kaggle/input/cos-code/environment_files"   # or wherever Kaggle mounts the eval games
arc = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=ENV_DIR)

GAMES = [...]   # the eval game_ids (from the Kaggle eval harness)
for game_id in GAMES:
    env = arc.make(game_id)
    agent = CosAgent(card_id="kaggle", game_id=game_id, agent_name="cosagent",
                     ROOT_URL="", record=False, arc_env=env, tags=[])
    agent.main()          # framework loop: choose_action (COS) + take_action (env)
    lc = max((int(getattr(f, "levels_completed", 0)) for f in agent.frames),
             default=0)
    print(f"{game_id}: levels_completed={lc}")
