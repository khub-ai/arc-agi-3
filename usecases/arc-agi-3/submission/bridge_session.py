"""Runs COS's REAL `run_session` in a background thread against a BridgeEnv, and
exposes the per-turn handshake the agent's `choose_action` uses.

The env is injected by monkeypatching `discovery_play.Arcade` (submission-side
runtime patch, NOT a COS source edit) so COS's `arc.make()` returns our
BridgeEnv and no real environment_files are needed. The model is COS's own
offline path (`model="ollama/<host>/<tag>"`), so the FULL COS stack runs exactly
as locally, just against the competition frames.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Optional

# COS python tree on sys.path (discovery_play).
_COS_PY = Path(__file__).resolve().parents[1] / "python"
if _COS_PY.is_dir() and str(_COS_PY) not in sys.path:
    sys.path.insert(0, str(_COS_PY))

from bridge_env import DONE, BridgeEnv, Channel


class BridgeSession:
    def __init__(self, game_id: str, model_slug: str, *,
                 session_dir: Optional[str] = None,
                 run_kwargs: Optional[dict] = None,
                 baseline_actions=None,
                 action_space=None,
                 turn_timeout_s: Optional[float] = None,
                 start_timeout_s: Optional[float] = None):
        self.game_id = game_id
        self.model_slug = model_slug
        self.baseline_actions = list(baseline_actions or [])
        self.action_space = list(action_space) if action_space else None
        self.session_dir = session_dir or os.environ.get(
            "COS_SESSION_DIR",
            str(Path(os.environ.get("COS_WORKDIR", ".")) / "cos_session"))
        # run_session is a discovery session (default max_turns=6); for real
        # play we let COS run long and keep it offline-clean (no replay).
        # cost_cap_usd must be POSITIVE: the check is `spent >= cap`, so 0.0
        # stops instantly. The local Qwen is free (spend stays $0), so a large
        # cap effectively means "no cost limit".
        self.run_kwargs = {
            "max_turns": int(os.environ.get("COS_MAX_TURNS_PER_GAME", 400)),
            "max_rounds": 1, "record_solution": False, "replay_solved": False,
            "cost_cap_usd": 1_000_000.0}
        if run_kwargs:
            self.run_kwargs.update(run_kwargs)
        # Harness-side waits: default preserved (600 / 1800 s) but env-overridable
        # so the session profile can tighten stalls without touching the engine.
        self.turn_timeout_s = (turn_timeout_s if turn_timeout_s is not None
                               else float(os.environ.get("COS_TURN_TIMEOUT_S", 600.0)))
        self.start_timeout_s = (start_timeout_s if start_timeout_s is not None
                                else float(os.environ.get("COS_START_TIMEOUT_S", 1800.0)))
        self.ch = Channel()
        self._thread: Optional[threading.Thread] = None
        self._first = True
        self._done = False

    def _run_cos(self) -> None:
        try:
            import sys
            import threading
            repo = Path(__file__).resolve().parents[3]
            sys.path.insert(0, str(repo / "tools/governor_audit/perception_loop_v2"))
            from game_adapter import LiveHarnessAdapter
            from world_knowledge import WorldKnowledge
            from exploratory_driver import ExploratoryDriver
            import cos_responder

            # Route vllm/<host:port>/<name> slugs (the local serve_vlm) to the local
            # OpenAI endpoint. The submission entry (my_agent) installs this; the
            # bench/local path doesn't, so do it here too (idempotent) -- otherwise
            # backends has no local route, every VLM call falls back, and COS plays
            # BLIND on substrate-only (0 real perception).
            try:
                _sub = str(Path(__file__).resolve().parent)
                if _sub not in sys.path:
                    sys.path.insert(0, _sub)
                import vllm_backend
                vllm_backend.install()
            except Exception as _e:                       # noqa: BLE001
                print(f"[bridge] vllm_backend.install skipped: {_e!r}")

            bridge = BridgeEnv(self.ch, baseline_actions=self.baseline_actions,
                               action_space=self.action_space)
            work = Path(self.session_dir)
            work.mkdir(parents=True, exist_ok=True)

            # In-process VLM: a daemon thread answers the driver's file-handoff prompts
            # via backends.call_oracle (the offline model slug).
            self._stop = threading.Event()
            threading.Thread(
                target=cos_responder.serve,
                args=(work, self.model_slug, self.game_id),
                kwargs={"stop_evt": self._stop}, daemon=True).start()

            # The v2 ExploratoryDriver (perception_loop_v2) -- merge mechanism, kb_recall,
            # instincts, the seeded general knowledge -- IS COS; drive the bridge's frame-
            # channel as its game.  (The old path ran discovery_play.run_session, a separate
            # engine that has none of this -- the reason the submission couldn't solve.)
            game = LiveHarnessAdapter(self.game_id, level=0, env=bridge)
            world = WorldKnowledge(game_id=self.game_id, level=0)
            world.win_state = "playing"
            # Per-VLM-call timeout: default 600 s, env-overridable so the session
            # profile can tighten a stalled call without an engine edit.
            _vlm_to = float(os.environ.get("COS_VLM_TIMEOUT_S", 600))
            driver = ExploratoryDriver(game, world, work, timeout_s=_vlm_to,
                                       vlm_timeout_s=_vlm_to, use_strategy=True, use_planner=True)
            driver.run(max_turns=int(self.run_kwargs.get("max_turns", 400)), start_level=0)
            self._stop.set()
        except Exception as exc:              # never crash the harness
            print(f"[bridge] COS thread ended: {exc!r}")
        finally:
            self.ch.finish()                  # unblock the harness with DONE

    def _start(self) -> None:
        self._thread = threading.Thread(target=self._run_cos, daemon=True)
        self._thread.start()

    def next_action(self, latest_frame):
        """Return COS's GameAction for this turn, or None (caller falls back /
        ends). The feed/read handshake: the first call returns COS's initial
        RESET without feeding; thereafter feed the resulting frame then read."""
        if self._done:
            return None
        if self._thread is None:
            self._start()
        if self._first:
            self._first = False
            timeout = self.start_timeout_s
        else:
            self.ch.harness_feed(latest_frame)
            timeout = self.turn_timeout_s
        try:
            a = self.ch.harness_read(timeout=timeout)
        except Exception:                     # queue.Empty -> COS stalled
            return None
        if a is DONE:
            self._done = True
            return None
        return a

    @property
    def done(self) -> bool:
        return self._done
