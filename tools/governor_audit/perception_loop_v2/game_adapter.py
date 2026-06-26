"""Pluggable GameAdapter interface — game-agnostic.

The ExploratoryDriver talks to the world through this interface.
Two implementations:

  - FixtureReplayAdapter: replays an already-recorded trial from a
    fixture directory.  Used for offline validation and regression.
    Actions are looked up by comparing consecutive truth states; if
    the actor wants an action the recorded trial did not take, the
    adapter raises so the driver can fall back to "use the closest
    recorded action and warn".

  - LiveHarnessAdapter (stub for now): sends actions to the actual
    game server and receives next frames.  Implementation deferred
    to Phase 2.

The adapter exposes no game-specific information — only frames,
actions, and a generic ``available_actions()`` set.  No level
counts, no scoring, no win conditions.  Those are observed via
perception.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from PIL import Image


@dataclass
class StepResult:
    """Result of one game step (action -> next frame).

    ``frame_path`` is the path to the new frame PNG.
    ``win_state`` is "playing" / "won" / "lost" / "unknown".
    ``lives`` / ``score`` may be None if the adapter can't observe
    them externally; in that case perception has to discover them.
    """
    frame_path: Path
    win_state: str
    lives: Optional[int] = None
    score: Optional[int] = None
    note: str = ""        # adapter-emitted free-text note for the driver
    # The engine hands back a FRAMESTACK per action (a list of frames): a plain
    # move is one frame, but an animated effect (a pour, a bounce, a cascade) is
    # many.  Keep the WHOLE stack -- the in-between frames carry vital mechanic
    # clues -- instead of discarding all but the settled last frame.
    frame_stack: Optional[list] = None   # list of 2D int grids (the animation)
    anim_dir: Optional[Path] = None      # dir of saved sub-frame PNGs (>1 frame)


class GameAdapter(Protocol):
    """All adapters must implement these methods."""

    game_id: str
    level: int

    def turn_one_frame(self) -> Path:
        """Return path to the initial (turn 1) frame PNG."""
        ...

    def step(self, action: str) -> StepResult:
        """Take one action, return the next frame + observable game state.
        Raises if action is not in available_actions()."""
        ...

    def available_actions(self) -> list[str]:
        """List of action strings the actor may attempt.  Standard
        ARC-AGI-3 actions: UP, DOWN, LEFT, RIGHT, CLICK, NONE.
        A specific adapter may expose a subset."""
        ...

    @property
    def current_turn(self) -> int:
        """Current turn number (1-indexed; equals last frame's turn)."""
        ...


# ---------------------------------------------------------------------------
# FixtureReplayAdapter
# ---------------------------------------------------------------------------


class FixtureReplayAdapter:
    """Replays an existing fixture trial.

    Fixture layout (matches the bp35 fixture convention):
        <fixture_dir>/
            sequence.json           -- describes the recorded turns
            turn_001/frame.png      -- per-turn frame
            turn_001/truth.json     -- per-turn ground truth (optional)
            ...

    The adapter derives the "action taken" between consecutive turns
    by comparing truth.json's agent_position when available; otherwise
    it just sequences turns.

    Game-agnostic: knows nothing about the game's mechanics; only the
    fixture file layout.
    """

    def __init__(self, fixture_dir: Path, game_id: str, level: int = 0):
        self.fixture_dir = Path(fixture_dir)
        self.game_id = game_id
        self.level = level
        # Discover available turns
        turn_dirs = sorted([
            p for p in self.fixture_dir.iterdir()
            if p.is_dir() and p.name.startswith("turn_")
        ])
        self._turn_paths: dict[int, Path] = {}
        for td in turn_dirs:
            try:
                n = int(td.name.split("_")[1])
                frame = td / "frame.png"
                if frame.exists():
                    self._turn_paths[n] = frame
            except (ValueError, IndexError):
                continue
        self._current_turn = 1
        if not self._turn_paths:
            raise FileNotFoundError(
                f"no turn frames found under {self.fixture_dir}"
            )

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def turn_one_frame(self) -> Path:
        if 1 not in self._turn_paths:
            raise FileNotFoundError(f"{self.fixture_dir} has no turn_001")
        self._current_turn = 1
        return self._turn_paths[1]

    def available_actions(self) -> list[str]:
        # Standard ARC-AGI-3 cardinal moves, since fixtures encode
        # whatever was tried originally.
        return ["UP", "DOWN", "LEFT", "RIGHT", "CLICK", "NONE"]

    def step(self, action: str) -> StepResult:
        """Replay the next recorded turn.  The `action` argument is
        IGNORED by the fixture adapter (it just advances to the next
        recorded frame) — but is logged in the StepResult.note so the
        caller knows the action they REQUESTED may not match what
        actually happened in the recorded trial."""
        next_turn = self._current_turn + 1
        if next_turn not in self._turn_paths:
            raise StopIteration(f"no turn_{next_turn:03d} in fixture")
        actual_action = self._derive_recorded_action(
            self._current_turn, next_turn,
        )
        note = (
            f"fixture-replay: requested {action!r}, "
            f"recorded action between turn {self._current_turn} and "
            f"{next_turn} was {actual_action!r}"
        )
        self._current_turn = next_turn
        # Try to read truth.json for win_state / lives if present
        win_state = "playing"
        lives = None
        score = None
        truth_path = self._turn_paths[next_turn].parent / "truth.json"
        if truth_path.exists():
            try:
                t = json.loads(truth_path.read_text(encoding="utf-8"))
                win_state = t.get("win_state", win_state)
                lives = t.get("lives", lives)
                score = t.get("score", score)
            except (json.JSONDecodeError, OSError):
                pass
        return StepResult(
            frame_path=self._turn_paths[next_turn],
            win_state=win_state, lives=lives, score=score, note=note,
        )

    def _derive_recorded_action(self, prev: int, curr: int) -> str:
        """Compare consecutive turns' truth.json agent_position to
        infer the action recorded between them.  Returns 'NONE' if
        no movement is detected and 'UNKNOWN' if truth files are
        missing."""
        try:
            t_prev = json.loads(
                (self._turn_paths[prev].parent
                 / "truth.json").read_text(encoding="utf-8")
            )
            t_curr = json.loads(
                (self._turn_paths[curr].parent
                 / "truth.json").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return "UNKNOWN"
        ap = t_prev.get("agent_position")
        ac = t_curr.get("agent_position")
        if ap is None or ac is None:
            return "UNKNOWN"
        dr, dc = ac[0] - ap[0], ac[1] - ap[1]
        if (dr, dc) == (0, 0):
            return "NONE"
        if (dr, dc) == (-1, 0):
            return "UP"
        if (dr, dc) == (1, 0):
            return "DOWN"
        if (dr, dc) == (0, -1):
            return "LEFT"
        if (dr, dc) == (0, 1):
            return "RIGHT"
        return f"OTHER({dr},{dc})"


# ---------------------------------------------------------------------------
# LiveHarnessAdapter — wraps the project's arc_agi Arcade runner
# ---------------------------------------------------------------------------


# Game-agnostic action name -> integer action_id mapping.
#
# ARC-AGI-3 games expose a per-game action_space subset of
# {ACTION1..ACTION7}.  Each game uses these action_ids with its
# OWN SEMANTICS -- ACTION1 might be "move up" in one game and
# "retract rope" in another.  Naming the actions with cardinal
# directions (UP/DOWN/LEFT/RIGHT) would inject game-specific
# semantic assumptions into the substrate, so the substrate
# vocabulary uses the raw ACTION_N names.
#
# Cardinal-direction ALIASES are provided as a convenience for
# the cell_actor's BFS (which emits UP/DOWN/LEFT/RIGHT).  The
# adapter accepts both raw names and aliases on input; the
# strategy VLM gets the raw names in available_actions() so it
# can discover each action's actual semantics through play
# rather than assuming.
_DEFAULT_ACTION_MAP: dict[str, int] = {
    # Raw, game-agnostic names (preferred for the strategy VLM)
    "ACTION1": 1,
    "ACTION2": 2,
    "ACTION3": 3,
    "ACTION4": 4,
    "ACTION5": 5,
    "ACTION6": 6,
    "ACTION7": 7,
    # Cardinal aliases (used by cell_actor; assume the standard
    # ARC-AGI-3 convention where ACTION1=up, ACTION2=down, etc.).
    # The adapter accepts these on input but does NOT advertise
    # them via available_actions() unless the underlying
    # action_id is actually in the game's action_space.
    "UP":    1,
    "DOWN":  2,
    "LEFT":  3,
    "RIGHT": 4,
    "CLICK": 6,
}

# Action names whose presence in available_actions() is GATED
# by the game's action_space (i.e. we don't advertise UP unless
# the game actually supports ACTION1).
_CARDINAL_ALIASES = {"UP", "DOWN", "LEFT", "RIGHT", "CLICK"}


# Canonical ARC-AGI-3 16-color palette.  Mirrors
# `trial_driver._default_palette()`; duplicated here to avoid a
# cross-module import on the trial_driver during adapter-only runs.
_DEFAULT_PALETTE = [
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
]


class LiveHarnessAdapter:
    """Live-game adapter wrapping the project's `arc_agi.Arcade`
    runner.  Sends action_ids to the actual game environment and
    returns the rendered PNG frame plus observable state
    (win/loss/level/score).

    Game-agnostic: knows nothing about specific games' mechanics;
    only about action_id <-> direction-string mapping and the
    palette used to render frames.  For games whose action
    semantics differ from the cardinal-direction default (e.g.
    sk48 where action_id=4 is "extend rope" rather than "move
    right"), pass a custom ``action_map`` to the constructor.
    """

    def __init__(self, game_id: str, level: int = 0,
                  environments_dir: Optional[Path] = None,
                  work_frame_dir: Optional[Path] = None,
                  action_map: Optional[dict[str, int]] = None,
                  env: Optional[object] = None):
        # Ensure the project's `arc_agi` module is importable.  The
        # canonical location is under usecases/arc-agi-3/python.
        import sys
        repo_root = Path(__file__).resolve().parents[3]
        for cand in (
            repo_root / "usecases" / "arc-agi-3" / "python",
            repo_root,
        ):
            if cand.exists() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))

        try:
            from arc_agi import Arcade, OperationMode  # type: ignore[import-not-found]
            from dsl_executor import _normalise_frame  # type: ignore[import-not-found]
        except ImportError as e:
            raise RuntimeError(
                f"could not import arc_agi runner; ensure "
                f"usecases/arc-agi-3/python is on sys.path: {e}"
            )

        self.game_id = game_id
        self.level = level
        self._normalise_frame = _normalise_frame
        self._action_map = dict(action_map or _DEFAULT_ACTION_MAP)

        # INJECTED env (e.g. the submission's BridgeEnv frame-channel): use it
        # directly instead of opening our own offline Arcade.  This lets the v2
        # ExploratoryDriver drive the competition bridge -- it only needs the env's
        # reset()/step(action_id, data=...) interface, which BridgeEnv provides.
        if env is not None:
            self._arc = None
            self._env = env
        else:
            env_dir = (environments_dir
                        or repo_root / "environment_files")
            if not env_dir.exists():
                raise FileNotFoundError(
                    f"environment_files dir not found at {env_dir}"
                )
            self._arc = Arcade(
                operation_mode=OperationMode.OFFLINE,
                environments_dir=str(env_dir),
            )
            self._env = self._arc.make(game_id)
        self._current_turn = 0
        self._obs = None
        self._work_frame_dir = (
            work_frame_dir
            or Path(".tmp/"
                     "exploratory_play_frames") / game_id
        )
        # The auto dump dir is keyed by GAME (not run) and turn numbers restart at 0 every run, so a
        # prior -- possibly longer -- run's leftover frames/anim-dirs would shadow this run's and
        # contaminate analysis.  Start each run from a clean slate.  A caller that passes an explicit
        # work_frame_dir owns its lifecycle, so leave that untouched.
        if work_frame_dir is None and self._work_frame_dir.exists():
            import shutil as _sh
            _sh.rmtree(self._work_frame_dir, ignore_errors=True)
        self._work_frame_dir.mkdir(parents=True, exist_ok=True)

        # Query the game's actual action_space and filter the
        # action_map / available-actions list to only what's
        # supported.  Without this, the adapter would advertise
        # cardinal aliases (UP/DOWN/...) that the game doesn't
        # actually accept, leading the strategy VLM to propose
        # actions that silently no-op.
        try:
            supported_ids = set()
            for a in self._env.action_space:
                # GameAction enum values: a.value is the int
                supported_ids.add(
                    int(a.value) if hasattr(a, "value") else int(a)
                )
        except (AttributeError, TypeError):
            # Env doesn't expose action_space -- conservatively
            # advertise everything in the action_map.
            supported_ids = set(self._action_map.values())
        self._supported_action_ids: set[int] = supported_ids

    @property
    def current_turn(self) -> int:
        return self._current_turn

    def available_actions(self) -> list[str]:
        """Action names the strategy VLM can legitimately propose.
        Filtered by the game's actual ``env.action_space``:
        cardinal-direction aliases (UP/DOWN/LEFT/RIGHT/CLICK) are
        only advertised when the underlying action_id is supported.

        Always includes:
          - raw ACTION_N names for every supported action_id
            (these are the game-agnostic vocabulary; the strategy
            VLM should reason from these and discover semantics
            through play, not assume ACTION1 means "up")
          - cardinal aliases for supported action_ids
            (convenience for the cell_actor's BFS)
          - "NONE" (a substrate no-op; not a game action)
        """
        out: list[str] = []
        seen_action_ids: set[int] = set()
        # Add raw ACTION_N names for every supported id (sorted by id
        # so the output is stable)
        for name, aid in sorted(self._action_map.items(),
                                  key=lambda kv: kv[1]):
            if name in _CARDINAL_ALIASES:
                continue
            if aid in self._supported_action_ids:
                out.append(name)
                seen_action_ids.add(aid)
        # Add cardinal aliases only for supported action_ids
        for name, aid in self._action_map.items():
            if name not in _CARDINAL_ALIASES:
                continue
            if aid in self._supported_action_ids:
                out.append(name)
        out.append("NONE")
        return out

    def turn_one_frame(self) -> Path:
        """Reset the game and render the initial frame as a PNG."""
        self._obs = self._env.reset()
        self._current_turn = 1
        return self._render_frame(self._obs, turn=1)

    def step(self, action: str) -> StepResult:
        """Send one action to the game; render and return the new
        frame.

        ``action`` accepts these forms:

          - "UP" / "DOWN" / "LEFT" / "RIGHT" — cardinal moves
          - "CLICK"            — click at playfield center (default
                                 fallback; old behaviour)
          - "CLICK:px,py"      — click at specific source-pixel
                                 coords (px and py both integers
                                 in the game's pixel resolution)
          - "NONE"             — no-op; re-render current frame
                                 without env.step

        The "CLICK:px,py" form is what the VLM strategy layer emits
        when it wants to click a specific entity; the driver
        translates the entity's bbox centroid to pixel coords
        before calling step.  Game-agnostic.
        """
        if action == "NONE":
            assert self._obs is not None
            self._current_turn += 1
            return self._build_step_result(
                self._obs, note="NONE (no env.step)",
            )

        # Parse "CLICK:px,py" into action="CLICK" + click coords
        click_coords: Optional[tuple[int, int]] = None
        if action.startswith("CLICK:"):
            payload = action[len("CLICK:"):]
            try:
                px_str, py_str = payload.split(",")
                click_coords = (int(px_str.strip()), int(py_str.strip()))
                action = "CLICK"
            except (ValueError, IndexError):
                raise ValueError(
                    f"CLICK action must be 'CLICK' or "
                    f"'CLICK:px,py' (got {action!r})"
                )

        action_id = self._action_map.get(action)
        if action_id is None:
            raise ValueError(
                f"unknown action {action!r} (known: "
                f"{list(self._action_map.keys())})"
            )
        # Reject actions whose action_id is NOT in the game's
        # actual action_space.  Without this guard, sending an
        # unsupported action silently no-ops, which the upstream
        # strategy VLM then has to diagnose by elimination.
        if action_id not in self._supported_action_ids:
            raise ValueError(
                f"action {action!r} (action_id={action_id}) is not "
                f"supported by this game; available: "
                f"{self.available_actions()}"
            )

        # CLICK actions need a {x, y} data payload.  If the caller
        # passed coords via "CLICK:px,py", use them; otherwise default
        # to the centre of the 64x64 grid (tick space) — the env
        # consumes click coords in the SAME frame as the perception
        # grid, NOT a 512x512 pixel space (a 256,256 default lands out
        # of bounds and silently no-ops).
        data = None
        if action_id == self._action_map.get("CLICK"):
            if click_coords is not None:
                data = {"x": click_coords[0], "y": click_coords[1]}
            else:
                data = {"x": 32, "y": 32}

        self._obs = self._env.step(action_id, data=data)
        self._current_turn += 1
        return self._build_step_result(
            self._obs,
            note=(f"sent action_id={action_id} "
                  f"data={data} (string {action!r})"),
        )

    # ------------------------------------------------------------------
    # Frame rendering
    # ------------------------------------------------------------------

    def _render_frame(self, obs, turn: int) -> Path:
        """Convert the raw observation frame (palette ndarray) into a
        PNG and save under work_frame_dir/turn_NNN.png."""
        import numpy as np
        grid = self._normalise_frame(obs.frame)
        palette = np.array(_DEFAULT_PALETTE, dtype="uint8")
        rgb = palette[grid.astype(int) % 16]
        img = Image.fromarray(rgb, mode="RGB")
        out = self._work_frame_dir / f"turn_{turn:03d}.png"
        img.save(out)
        return out

    def _frame_stack(self, obs) -> list:
        """The FULL animation framestack as a list of 2D int grids.  The engine
        returns a list of frames per action (a plain move = 1 frame; an animated
        effect = many).  We keep them all -- the in-between frames are the
        mechanic clue."""
        import numpy as np
        raw = getattr(obs, "frame", None)
        if isinstance(raw, list):
            out = []
            for f in raw:
                g = np.array(f, dtype=int)
                out.append(g[-1] if g.ndim == 3 else g)
            return out or [np.zeros((64, 64), dtype=int)]
        g = np.array(raw, dtype=int)
        return [g[t] for t in range(g.shape[0])] if g.ndim == 3 else [g]

    def _save_animation(self, grids: list, turn: int):
        """Save each sub-frame of a multi-frame animation as a PNG so the
        animation is inspectable (trace / debugging).  Returns the dir, or None
        when the action produced just one settled frame."""
        import numpy as np
        if len(grids) <= 1:
            return None
        palette = np.array(_DEFAULT_PALETTE, dtype="uint8")
        d = self._work_frame_dir / f"turn_{turn:03d}_anim"
        # This dump is keyed by turn# in a per-game dir SHARED across runs.  mkdir(exist_ok) alone
        # would let a prior run's frames at the same turn survive a partial overwrite (e.g. 8 old
        # vs 6 new -> 2 stale frames), silently contaminating the filmstrip / any frame analysis
        # with images from other runs or levels.  Clear prior frames so the dir holds ONLY this turn.
        if d.exists():
            for _old in d.glob("frame_*.png"):
                _old.unlink()
        d.mkdir(parents=True, exist_ok=True)
        for i, g in enumerate(grids):
            rgb = palette[g.astype(int) % 16]
            Image.fromarray(rgb, mode="RGB").save(d / f"frame_{i:03d}.png")
        return d

    def _build_step_result(self, obs, note: str) -> StepResult:
        frame_path = self._render_frame(obs, turn=self._current_turn)
        try:
            grids = self._frame_stack(obs)
            anim_dir = self._save_animation(grids, self._current_turn)
        except Exception:
            grids, anim_dir = None, None
        state_name = getattr(obs.state, "name", str(obs.state)).lower()
        lc = int(getattr(obs, "levels_completed", 0) or 0)
        # Map arc_agi state names to our adapter convention
        win_state = {
            "playing": "playing",
            "not_finished": "playing",
            "in_progress": "playing",
            "won": "won",
            "win": "won",
            "lost": "lost",
            "loss": "lost",
            "game_over": "lost",
        }.get(state_name, state_name)
        return StepResult(
            frame_path=frame_path,
            win_state=win_state,
            lives=None,  # arc_agi obs doesn't expose lives directly
            score=lc,    # use lc as a coarse progress indicator
            note=note,
            frame_stack=grids,
            anim_dir=anim_dir,
        )
