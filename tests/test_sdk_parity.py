"""Parity tests: exercise the adapter against the real arc_agi SDK types.

The synthetic fixtures in ``test_adapter.py`` use plain Python lists
and ``str``-typed state names; the *live* SDK returns:

* ``state`` as a :class:`arcengine.enums.GameState` StrEnum instance.
* ``frame`` as ``list[numpy.ndarray]`` — an outer 1-element viewport
  list wrapping a 2-D ndarray grid (not the ``list[list[list[int]]]``
  its pydantic annotation claims).
* ``available_actions`` as ``list[int]`` per-frame, while
  ``env.action_space`` is ``list[GameAction]`` (enum instances with
  ``.value`` and ``.name``).

Each test below builds a real SDK-typed object and drives it through
the adapter to verify the translation layer handles the live shapes
identically to the synthetic ones.  These tests require
``arc_agi`` / ``arcengine`` to be importable but never touch the
network.  They are the regression net that guards against a future
SDK change silently breaking the live path.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest

# Import real SDK types — if these fail we skip rather than red-X,
# because the CI image on some hosts may not ship the SDK.
pytestmark = pytest.mark.skipif(
    pytest.importorskip("arcengine.enums", reason="arc_agi SDK not installed")
    is None,
    reason="arc_agi SDK not installed",
)

from arcengine.enums import GameAction, GameState   # noqa: E402
from cognitive_os import WorldState   # noqa: E402

from arc_agi_3 import ArcAdapter   # noqa: E402
from arc_agi_3.adapter import _normalise_state_name, _to_list_2d   # noqa: E402
from arc_agi_3.action_mapping import (   # noqa: E402
    engine_action_for,
    engine_action_space,
    native_action_for,
)


# ---------------------------------------------------------------------------
# _to_list_2d — shape normalisation
# ---------------------------------------------------------------------------


def test_to_list_2d_plain_list_passes_through() -> None:
    g = [[0, 1, 2], [3, 4, 5]]
    assert _to_list_2d(g) == g


def test_to_list_2d_bare_numpy_2d() -> None:
    # Some SDK versions return a bare ndarray; .tolist() handles it.
    arr = np.array([[1, 2], [3, 4]], dtype=np.int32)
    out = _to_list_2d(arr)
    assert out == [[1, 2], [3, 4]]
    assert isinstance(out[0][0], int)


def test_to_list_2d_viewport_wrapped_numpy() -> None:
    # The live ls20 shape: outer Python list of length 1 wrapping a
    # 2-D ndarray.  This is the shape that fails the naive
    # ``isinstance(frame[0], list)`` check; must still unwrap cleanly.
    inner = np.array([[7, 8], [9, 10]], dtype=np.int32)
    frame = [inner]
    out = _to_list_2d(frame)
    assert out == [[7, 8], [9, 10]]


def test_to_list_2d_viewport_wrapped_list_of_list() -> None:
    # Pure-Python 3-D shape (the pydantic annotation's nominal form).
    frame = [[[1, 2], [3, 4]]]
    out = _to_list_2d(frame)
    assert out == [[1, 2], [3, 4]]


def test_to_list_2d_empty_inputs() -> None:
    assert _to_list_2d(None) == []
    assert _to_list_2d([])   == []


# ---------------------------------------------------------------------------
# _normalise_state_name — StrEnum vs plain str
# ---------------------------------------------------------------------------


def test_normalise_state_name_from_gamestate_enum() -> None:
    # The live SDK hands GameState.<X> instances; the adapter must
    # surface the bare name ("WIN") — not "GameState.WIN", which is
    # what ``str(enum)`` would produce.
    assert _normalise_state_name(GameState.WIN)           == "WIN"
    assert _normalise_state_name(GameState.GAME_OVER)     == "GAME_OVER"
    assert _normalise_state_name(GameState.NOT_FINISHED)  == "NOT_FINISHED"
    assert _normalise_state_name(GameState.NOT_PLAYED)    == "NOT_PLAYED"


def test_normalise_state_name_accepts_plain_strings() -> None:
    # Replay fixtures use plain strings; parity must preserve those.
    assert _normalise_state_name("PLAYING")   == "PLAYING"
    assert _normalise_state_name("WIN")       == "WIN"


def test_normalise_state_name_none_is_unknown() -> None:
    assert _normalise_state_name(None) == "UNKNOWN"


# ---------------------------------------------------------------------------
# Action translation — real GameAction enum + raw ints
# ---------------------------------------------------------------------------


def test_engine_action_for_accepts_gameaction_enum() -> None:
    a = engine_action_for(GameAction.ACTION2)
    assert a.id   == "ACTION2"
    assert a.name == "ACTION2"


def test_engine_action_for_accepts_raw_int() -> None:
    # Per-frame ``available_actions`` is ``list[int]``: the adapter
    # reads this directly and must round-trip ints through
    # engine_action_for without an .value attribute.
    a = engine_action_for(3)
    assert a.name == "ACTION3"


def test_engine_action_space_mixes_enum_and_int() -> None:
    # Belt-and-braces: the adapter falls back to env.action_space
    # (enums) when the per-frame list is empty, so both code paths
    # must produce the same engine Actions.
    space = engine_action_space([GameAction.ACTION1, 2, GameAction.ACTION3])
    assert [a.name for a in space] == ["ACTION1", "ACTION2", "ACTION3"]


def test_native_action_for_finds_gameaction_by_name() -> None:
    raw_space = [GameAction.ACTION1, GameAction.ACTION4, GameAction.ACTION7]
    engine_a  = engine_action_for(GameAction.ACTION4)
    found     = native_action_for(engine_a, raw_space)
    assert found is GameAction.ACTION4


def test_native_action_for_finds_int_by_name() -> None:
    raw_space = [1, 2, 3]
    engine_a  = engine_action_for(2)
    found     = native_action_for(engine_a, raw_space)
    assert found == 2


def test_native_action_for_missing_raises_key_error() -> None:
    with pytest.raises(KeyError):
        native_action_for(engine_action_for(99), [GameAction.ACTION1])


# ---------------------------------------------------------------------------
# End-to-end: FrameDataRaw-like object → adapter → Observation
# ---------------------------------------------------------------------------


class _RawFrame:
    """Minimal duck-type of the live SDK's FrameDataRaw.

    We deliberately do *not* import ``FrameDataRaw`` itself because
    pydantic validates types at construction and the real ``frame``
    field is not part of the pydantic schema (the SDK stamps it on
    dynamically).  A bare attribute container matches what the
    adapter actually reads via ``getattr``.
    """

    def __init__(self, frame: Any, state: Any, levels: int, actions: List[Any]) -> None:
        self.frame             = frame
        self.state             = state
        self.levels_completed  = levels
        self.available_actions = actions


class _FakeLiveEnv:
    """Mimics arc_agi.LocalEnvironmentWrapper shape: reset+step both
    return an object whose ``frame`` is ``list[ndarray]`` and whose
    ``state`` is a ``GameState`` enum."""

    def __init__(self) -> None:
        self._step = 0
        self.action_space = [GameAction.ACTION1, GameAction.ACTION2, GameAction.ACTION3]

    def _make_frame(self, state: GameState) -> _RawFrame:
        # 3×3 numpy grid wrapped in a single-element viewport list —
        # exactly the live SDK shape.
        arr = np.array([[self._step, 1, 0],
                        [0, 2, 0],
                        [0, 0, 0]], dtype=np.int32)
        return _RawFrame(
            frame   = [arr],
            state   = state,
            levels  = 0,
            actions = [1, 2, 3],         # raw ints, per the live shape
        )

    def reset(self) -> _RawFrame:
        self._step = 0
        return self._make_frame(GameState.NOT_FINISHED)

    def step(self, _action: Any) -> _RawFrame:
        self._step += 1
        if self._step >= 3:
            return self._make_frame(GameState.WIN)
        return self._make_frame(GameState.NOT_FINISHED)


def test_adapter_ingests_live_shape_frames() -> None:
    env     = _FakeLiveEnv()
    adapter = ArcAdapter(raw_env=env, env_id="parity_probe")
    ws      = WorldState()
    adapter.initialize(ws)

    obs = adapter.reset()
    # Frame must arrive as a plain list-of-list-of-int, not an ndarray
    # or a wrapper list — otherwise every downstream tool breaks on
    # isinstance checks.
    assert isinstance(obs.raw_frame, list)
    assert isinstance(obs.raw_frame[0], list)
    assert isinstance(obs.raw_frame[0][0], int)

    # State name round-tripped to a plain string.
    state_name = obs.agent_state.get("state_name")
    assert state_name == "NOT_FINISHED"
    assert isinstance(state_name, str)


def test_adapter_action_cycle_with_live_shape() -> None:
    env     = _FakeLiveEnv()
    adapter = ArcAdapter(raw_env=env, env_id="parity_probe")
    ws      = WorldState()
    adapter.initialize(ws)
    adapter.reset()

    # action_space must expose engine Actions named after the raw ints.
    space = adapter.action_space()
    assert [a.name for a in space] == ["ACTION1", "ACTION2", "ACTION3"]

    # Execute one of them; adapter must resolve it back to the raw
    # int for env.step().  No exception == success (env.step() sees
    # the int, increments counter).
    adapter.execute(space[1])

    obs = adapter.observe()
    assert obs.agent_state["state_name"] in {"NOT_FINISHED", "WIN"}


def test_adapter_terminal_state_detected_via_strenum() -> None:
    # is_done() lives in a set-membership check against {"WIN",
    # "GAME_OVER"}; StrEnum hashes equal to its value so this works
    # *today*, but if a future SDK switches to a non-StrEnum Enum
    # this test tells us immediately.
    env     = _FakeLiveEnv()
    adapter = ArcAdapter(raw_env=env, env_id="parity_probe")
    ws      = WorldState()
    adapter.initialize(ws)
    adapter.reset()

    # Drive through to terminal.
    for _ in range(5):
        if adapter.is_done():
            break
        space = adapter.action_space()
        adapter.execute(space[0])
        adapter.observe()

    assert adapter.is_done()
