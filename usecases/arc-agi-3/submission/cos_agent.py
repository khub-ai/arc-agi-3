"""Runs the REAL COS loop against a per-turn harness via the control-inversion
bridge.

`CosPlayer` is the framework-independent core: it runs COS's `run_session` (full
perception/strategy/world-model/MEA, local Qwen oracle) in a thread behind a
`BridgeEnv`, and answers `choose_action` per turn. The lean single-shot chooser
is ONLY a timeout fallback. `CosPlayer` needs NO agents-framework - just COS +
the arc-agi SDK + Ollama - so it runs locally (local_run drives it).

`CosAgent(Agent)` is the thin Kaggle wrapper: it subclasses the framework Agent
and delegates to a `CosPlayer`. Use it where the agents framework is installed.
"""
from __future__ import annotations

import os
import random
import sys
from pathlib import Path

from _compat import Agent, GameAction, GameState
from qwen_oracle import QwenOracle, default_slug

# COS trees on sys.path (read-only).
_REPO = Path(__file__).resolve().parents[3]
for _p in (_REPO / "tools" / "governor_audit" / "perception_loop_v2",
           _REPO / "usecases" / "arc-agi-3" / "python"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


class CosPlayer:
    """Framework-independent core: real COS via the bridge + a fallback."""

    def __init__(self, game_id: str, model_slug: str | None = None,
                 baseline_actions=None, action_space=None):
        os.environ.setdefault("COS_STRICT", "1")        # competition-clean
        self.model_slug = (model_slug or os.environ.get("QWEN_MODEL_SLUG")
                           or default_slug())
        self._session = None
        self._fallback_chooser = None
        self._fallback_oracle = None
        try:
            from bridge_session import BridgeSession
            self._session = BridgeSession(game_id or "", self.model_slug,
                                          baseline_actions=baseline_actions,
                                          action_space=action_space)
        except Exception as exc:
            print(f"[cos] bridge unavailable -> fallback only: {exc!r}")

    def is_done(self, frames, latest_frame) -> bool:
        if self._session is not None and self._session.done:
            return True
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        if self._session is not None:
            a = self._session.next_action(latest_frame)
            if a is not None:
                return a                        # COS's real decision
            if self._session.done:
                return GameAction.RESET
            # COS stalled this turn -> fall back for one action
        return self._fallback(frames, latest_frame)

    def _fallback(self, frames, latest_frame) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            return GameAction.RESET
        try:
            from bridge_env import to_game_action
            if self._fallback_chooser is None:
                from cos_chooser import LeanCosChooser
                self._fallback_chooser = LeanCosChooser()
                self._fallback_oracle = QwenOracle(
                    dry_run=os.environ.get("ORACLE_DRY_RUN") == "1")
            res = self._fallback_chooser.choose(frames, latest_frame,
                                                self._fallback_oracle)
            if isinstance(res, tuple):
                return to_game_action(res[0], res[1])
            return to_game_action(res)
        except Exception as exc:
            print(f"[cos] fallback failed: {exc!r}")
            return random.choice([a for a in GameAction
                                  if a is not GameAction.RESET])


class CosAgent(Agent):
    """Kaggle wrapper: framework Agent -> CosPlayer. Needs the agents framework
    (only available where it's installed); for local runs use CosPlayer."""

    MAX_ACTIONS = 8000

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pass the per-sub-level step budgets + the real action_space through to
        # COS from the framework env (budget-pressure scheduler + the adapter's
        # action-vocabulary filter).
        ba, asp = [], None
        try:
            ba = list(getattr(getattr(self.arc_env, "info", None),
                              "baseline_actions", []) or [])
        except Exception:
            pass
        try:
            asp = list(getattr(self.arc_env, "action_space", None) or []) or None
        except Exception:
            pass
        self._player = CosPlayer(getattr(self, "game_id", "") or "",
                                 baseline_actions=ba, action_space=asp)

    def is_done(self, frames, latest_frame) -> bool:
        return self._player.is_done(frames, latest_frame)

    def choose_action(self, frames, latest_frame) -> GameAction:
        return self._player.choose_action(frames, latest_frame)
