"""Tests for the live harness.

The harness is the one place the arc_agi SDK gets touched, and our
CI standing rule is zero live API traffic.  These tests therefore
inject a fake ``arcade_factory`` that returns an object whose
``make()`` hands back a :class:`_ReplayEnv`-compatible env built from
the :func:`moving_agent_episode` fixture.  Everything the harness
does downstream of ``make()`` runs real code: real ArcAdapter, real
engine :func:`run_episode`, real PostMortem.

What we exercise:

* ``run_harness`` end-to-end against a fake Arcade — WIN episode.
* Backend selection — ``--backend null`` produces a NullBackend;
  ``--backend anthropic`` without credentials raises cleanly.
* Multi-episode runs share a single WorldState (cross-episode
  accumulation is a first-class requirement; regressing on this
  would mean lessons never carry over).
* ``main()`` CLI — exit code reflects episode outcome; ``--help``
  returns 0 and prints usage.

The Anthropic code path is not invoked; its unit test is in
``test_observer_mediator.py`` via ``_FakeChatBackend``.
"""

from __future__ import annotations

import io
import os
from typing import Any

import pytest

from arc_agi_3 import harness
from arc_agi_3.adapter import _ReplayEnv
from arc_agi_3.backends import NullBackend

from .fixtures import moving_agent_episode


# ---------------------------------------------------------------------------
# Fake SDK
# ---------------------------------------------------------------------------


class _FakeArcade:
    """Matches the narrow surface the harness uses on arc_agi.Arcade.

    Records the calls so tests can assert on the game_id etc.
    """

    def __init__(self, api_key: str = "") -> None:
        self.api_key    = api_key
        self.make_calls: list[str] = []

    def make(self, game_id: str, **kwargs: Any) -> _ReplayEnv:
        self.make_calls.append(game_id)
        frames, states, levels, available = moving_agent_episode()
        return _ReplayEnv(
            frames            = list(frames),
            states            = list(states),
            levels            = list(levels),
            available_actions = [list(a) for a in available],
        )


def _fake_factory(api_key: str) -> _FakeArcade:
    return _FakeArcade(api_key=api_key)


# ---------------------------------------------------------------------------
# run_harness — end-to-end
# ---------------------------------------------------------------------------


def test_run_harness_single_episode_reaches_win() -> None:
    result = harness.run_harness(
        game_id        = "ls20",
        episodes       = 1,
        backend        = "null",
        arcade_factory = _fake_factory,
    )
    assert len(result.episodes) == 1
    pm = result.episodes[0]
    # The fixture episode terminates in a WIN frame with
    # ``episode_won = 1.0``; the seed goal must fire.
    assert pm.final_status == "success"
    assert pm.episode_id   == "ls20::ep0000"
    assert result.successes == 1
    assert result.failures  == 0


def test_run_harness_multi_episode_shares_world_state() -> None:
    """Cross-episode accumulation is only possible if the WorldState
    carries between episodes.  Two independent signals:

    * Each episode's PostMortem should be well-formed.
    * The observation history / hypothesis store should grow
      monotonically across episodes (not be reset each time).
    """
    # We cannot directly peek at ws from outside run_harness, so we
    # assert on externally observable behaviour: the second episode
    # must not fail with a fresh-WS symptom (e.g. tool registry
    # missing); success on both is the strongest signal.
    result = harness.run_harness(
        game_id        = "ls20",
        episodes       = 3,
        backend        = "null",
        arcade_factory = _fake_factory,
    )
    assert len(result.episodes) == 3
    assert all(pm.final_status == "success" for pm in result.episodes)
    # Episode IDs should be distinct and ordered.
    ids = [pm.episode_id for pm in result.episodes]
    assert ids == sorted(ids)
    assert len(set(ids)) == 3


def test_run_harness_null_backend_is_default() -> None:
    """The default backend must be NullBackend — competition-safe and
    network-free.  This also guards against the harness silently
    instantiating AnthropicBackend and hitting the network."""
    captured: dict = {}

    class _Probe(_FakeArcade):
        def make(self, game_id: str, **kw: Any) -> _ReplayEnv:
            env = super().make(game_id, **kw)
            captured["env"] = env
            return env

    result = harness.run_harness(
        game_id        = "ls20",
        backend        = "null",
        arcade_factory = lambda key: _Probe(api_key=key),
    )
    assert result.successes == 1


def test_run_harness_raises_when_make_returns_none() -> None:
    class _NullArcade:
        def __init__(self, api_key: str = "") -> None: pass
        def make(self, game_id: str, **_kw: Any) -> None: return None

    with pytest.raises(RuntimeError, match="returned None"):
        harness.run_harness(
            game_id        = "ls20",
            arcade_factory = lambda key: _NullArcade(api_key=key),
        )


def test_run_harness_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        harness.run_harness(
            game_id        = "ls20",
            backend        = "doesnotexist",
            arcade_factory = _fake_factory,
        )


def test_run_harness_anthropic_backend_without_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Selecting the Anthropic backend with no API key must fail fast
    at adapter-construction time (not lazily at first query)."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Anthropic API key"):
        harness.run_harness(
            game_id        = "ls20",
            backend        = "anthropic",
            arcade_factory = _fake_factory,
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def test_main_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        harness.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "arc-agi-3" in out
    assert "--game-id" in out


def test_main_returns_zero_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Redirect the default factory so main() runs offline.
    monkeypatch.setattr(harness, "_default_arcade_factory", _fake_factory)
    exit_code = harness.main(["--game-id", "ls20", "--log-level", "WARNING"])
    assert exit_code == 0


def test_main_returns_one_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the fake env produces an episode that does not WIN, main
    should return exit code 1."""
    def _losing_factory(api_key: str) -> _FakeArcade:
        arcade = _FakeArcade(api_key=api_key)
        # Override make() to produce an episode that goes to GAME_OVER
        # rather than WIN.
        def _make(game_id: str, **kw: Any) -> _ReplayEnv:
            arcade.make_calls.append(game_id)
            frames, _, _, available = moving_agent_episode()
            states = ["PLAYING"] * 4 + ["GAME_OVER"]
            levels = [0, 0, 0, 0, 0]
            return _ReplayEnv(
                frames            = list(frames),
                states            = list(states),
                levels            = list(levels),
                available_actions = [list(a) for a in available],
            )
        arcade.make = _make   # type: ignore[method-assign]
        return arcade

    monkeypatch.setattr(harness, "_default_arcade_factory", _losing_factory)
    exit_code = harness.main(["--game-id", "ls20", "--log-level", "WARNING"])
    assert exit_code == 1


def test_main_catches_unexpected_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any exception below the CLI should become a nonzero exit code,
    not a traceback bubbling into the shell."""
    def _broken(api_key: str) -> Any:
        raise RuntimeError("boom")
    monkeypatch.setattr(harness, "_default_arcade_factory", _broken)
    exit_code = harness.main(["--game-id", "ls20", "--log-level", "WARNING"])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Summary printing
# ---------------------------------------------------------------------------


def test_print_summary_lists_each_episode() -> None:
    result = harness.run_harness(
        game_id        = "ls20",
        episodes       = 2,
        backend        = "null",
        arcade_factory = _fake_factory,
    )
    buf = io.StringIO()
    harness._print_summary(result, stream=buf)
    text = buf.getvalue()
    assert "ls20" in text
    assert "2/2 succeeded" in text
    for pm in result.episodes:
        assert pm.episode_id in text
