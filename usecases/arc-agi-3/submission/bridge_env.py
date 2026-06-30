"""Control-inversion bridge so the Kaggle harness runs COS's REAL play loop.

COS owns its loop (`env.reset()/step()`); the harness owns its loop (calls
`choose_action` per turn). We bridge them so COS's FULL stack runs unchanged:
COS plays a `BridgeEnv` (an arc_agi-env stand-in) in a background thread; its
`reset()/step()` hand actions out to `choose_action` and block for the next
frame.

The incoming `FrameData` already carries `.frame/.state/.levels_completed/
.win_levels/.available_actions`, so it IS a valid COS obs (identity pass).

Protocol (two single-slot queues):
  COS side (BridgeEnv): emit action -> wait for the resulting obs.
  Harness side: the FIRST call returns COS's initial action (its reset's RESET)
    WITHOUT feeding a frame; thereafter feed the incoming frame (the result of
    the previously-returned action) then return COS's next action.
"""
from __future__ import annotations

import queue
import re
from types import SimpleNamespace
from typing import Optional

from _compat import GameAction

DONE = object()  # COS thread finished -> sentinel placed on the action queue

_BY_NAME = {a.name: a for a in GameAction}


def to_game_action(action, data: Optional[dict] = None) -> GameAction:
    """COS action (int 1-7, 'ACTIONn', or a GameAction) -> framework GameAction,
    with ACTION6's x/y taken from `data` and clamped to 0..63."""
    if isinstance(action, GameAction):
        ga = action
    else:
        s = str(getattr(action, "value", action)).upper()
        if "RESET" in s:
            return GameAction.RESET
        # A "CLICK:x,y" string must map to ACTION6 with its coords -- NOT be fed to
        # the digit search below, where re.search([1-7]) would grab the first digit
        # of the coordinate (e.g. CLICK:30,46 -> "3" -> ACTION3) and drop the click.
        if s.startswith("CLICK"):
            ga = GameAction.ACTION6
            mxy = re.search(r"(\d+)\s*,\s*(\d+)", s)
            if mxy and not (isinstance(data, dict) and "x" in data):
                data = {"x": int(mxy.group(1)), "y": int(mxy.group(2))}
        else:
            m = re.search(r"([1-7])", s)
            ga = _BY_NAME.get(f"ACTION{m.group(1)}" if m else "ACTION1",
                              GameAction.ACTION1)
    if ga.is_complex():
        x, y = 32, 32
        if isinstance(data, dict):
            try:
                x, y = int(data.get("x", 32)), int(data.get("y", 32))
            except Exception:
                pass
        ga.set_data({"x": max(0, min(63, x)), "y": max(0, min(63, y))})
        ga.reasoning = {"desired_action": ga.value, "my_reason": "cos"}
    else:
        ga.reasoning = "cos"
    return ga


class Channel:
    """Two single-slot queues bridging the COS thread and the harness thread."""

    def __init__(self):
        self.obs_q: queue.Queue = queue.Queue(maxsize=1)   # harness -> COS
        self.act_q: queue.Queue = queue.Queue(maxsize=1)   # COS -> harness

    # COS thread (BridgeEnv) ------------------------------------------------
    def cos_emit(self, ga) -> None:
        self.act_q.put(ga)

    def cos_wait_obs(self):
        return self.obs_q.get()

    # harness thread (the agent) -------------------------------------------
    def harness_feed(self, frame) -> None:
        self.obs_q.put(frame)

    def harness_read(self, timeout: Optional[float] = None):
        return self.act_q.get(timeout=timeout)

    def finish(self) -> None:
        try:
            self.act_q.put(DONE, timeout=5)
        except Exception:
            pass


class BridgeEnv:
    """arc_agi-env stand-in injected into COS (via monkeypatched Arcade.make).
    COS drives it exactly like the real offline env."""

    def __init__(self, channel: Channel, baseline_actions=None,
                 action_space=None):
        self._ch = channel
        # COS reads env.info.baseline_actions (per-sub-level step budgets) for
        # its budget-pressure scheduler. An empty list mis-fires that logic, so
        # plumb the REAL values from the harness/SDK env.
        self.info = SimpleNamespace(baseline_actions=list(baseline_actions or []))
        # LiveHarnessAdapter reads env.action_space to filter the strategy VLM's
        # action vocabulary to the game's REAL actions (e.g. tn36 = [ACTION6]).
        # Only expose it when we have the real (non-empty) list: an empty
        # action_space advertises NOTHING, whereas a MISSING attr makes the
        # adapter fall back to advertising everything. So set it only if real.
        if action_space:
            self.action_space = list(action_space)

    def reset(self, *a, **kw):
        self._ch.cos_emit(GameAction.RESET)
        return self._ch.cos_wait_obs()

    def step(self, action, data: Optional[dict] = None, *a, **kw):
        self._ch.cos_emit(self._supported(to_game_action(action, data)))
        return self._ch.cos_wait_obs()

    def _supported(self, ga):
        """Never emit an action the game's action_space lacks -- the SDK rejects it
        with a hard ValueError that crashes the whole game (e.g. CLICK/ACTION6 on a
        directional game like ls20). Substitute a supported, non-terminal action and
        log loudly (the decision logic SHOULD filter to the action_space upstream)."""
        import sys
        sp = getattr(self, "action_space", None)
        if not sp:
            return ga                                  # unknown space -> can't guard
        names = {getattr(a, "name", str(a)) for a in sp}
        if ga.name in names or ga.name == "RESET":     # RESET is always valid
            return ga
        sub = next((a for a in sp if getattr(a, "name", "") not in ("NONE", "RESET")),
                   None) or sp[0]
        print(f"[bridge] GUARD: {ga.name} not in action_space {sorted(names)} -> "
              f"substituting {getattr(sub, 'name', sub)} (decision logic should filter)",
              file=sys.stderr, flush=True)
        return sub
