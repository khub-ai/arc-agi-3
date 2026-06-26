"""Import shim for the ARC-AGI-3 agents framework types.

CONFIRMED against the downloaded framework (arcprize/ARC-AGI-3-Agents):
  - `Agent` lives in `agents.agent`
  - `GameAction` / `GameState` / `FrameData` come from **`arcengine`**
    (the `arc-agi` SDK pulls it; `agents/agent.py` itself imports
    `from arcengine import FrameData, FrameDataRaw, GameAction, GameState`).

We self-discover the extracted framework dir (submission/ARC-AGI-3-Agents) so
`import agents` resolves, then import the real types. If the framework isn't
present (bare dev box), we fall back to minimal stubs so this folder is still
importable for authoring — those stubs are NEVER shipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

# --- make `agents` importable: add the extracted framework to sys.path --------
_SUB = Path(__file__).resolve().parent
for _cand in (_SUB / "ARC-AGI-3-Agents", _SUB.parent / "ARC-AGI-3-Agents"):
    if (_cand / "agents" / "agent.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
        break

Agent = GameAction = GameState = FrameData = None
_errors = []

# Structs come from `arcengine` — INDEPENDENT of the `agents` package (and of its
# optional template deps). Import them directly so a template-import failure
# never blocks the types.
try:
    import arcengine as _ae
    GameAction, GameState, FrameData = _ae.GameAction, _ae.GameState, _ae.FrameData
except Exception as exc:  # noqa: BLE001
    _errors.append(f"  arcengine structs: {exc!r}")

# Agent base from `agents.agent`. The package __init__ imports the example
# templates LAST (after defining Agent); if those optional deps are absent the
# __init__ raises, but `agents.agent.Agent` is already loaded — recover it from
# sys.modules so we still get the real base class.
try:
    from agents.agent import Agent as _Agent
    Agent = _Agent
except Exception as exc:  # noqa: BLE001
    _m = sys.modules.get("agents.agent")
    if _m is not None and hasattr(_m, "Agent"):
        Agent = _m.Agent
    else:
        _errors.append(f"  agents.agent.Agent: {exc!r}")

FRAMEWORK_AVAILABLE = Agent is not None and GameAction is not None

if not FRAMEWORK_AVAILABLE:
    _MSG = ("ARC-AGI-3 agents framework not importable. Extract the bundle to "
            "submission/ARC-AGI-3-Agents and `pip install arc-agi>=0.9.1` "
            "(provides arcengine). Tried:\n" + "\n".join(_errors))

    # Minimal dev stubs so `import cos_agent` works WITHOUT the framework. NEVER
    # shipped; the real run uses the framework types above.
    import enum

    class GameState(enum.Enum):           # type: ignore[no-redef]
        NOT_PLAYED = "NOT_PLAYED"
        NOT_FINISHED = "NOT_FINISHED"
        WIN = "WIN"
        GAME_OVER = "GAME_OVER"

    class GameAction(enum.Enum):          # type: ignore[no-redef]
        RESET = "RESET"
        ACTION1 = "ACTION1"
        ACTION2 = "ACTION2"
        ACTION3 = "ACTION3"
        ACTION4 = "ACTION4"
        ACTION5 = "ACTION5"
        ACTION6 = "ACTION6"
        ACTION7 = "ACTION7"

        def is_simple(self):
            return self not in (GameAction.RESET, GameAction.ACTION6)

        def is_complex(self):
            return self is GameAction.ACTION6

        def set_data(self, data):
            self._data = data
            return self

    class FrameData:                      # type: ignore[no-redef]
        state = GameState.NOT_PLAYED
        frame = None
        available_actions = ()
        levels_completed = 0
        win_levels = 0

    class Agent:                          # type: ignore[no-redef]
        MAX_ACTIONS = 10_000

        def __init__(self, *a, **kw):
            raise RuntimeError(_MSG)
