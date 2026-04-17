"""Translation between engine :class:`cognitive_os.Action` objects and
the ``arc_agi`` SDK's native action enum.

The ARC-AGI-3 environment exposes an ``action_space`` list of enum-
like objects with ``.value`` integer codes and human-readable names
(``ACTION1``, ``ACTION2``, …).  The engine wants symbolic
:class:`Action` objects with stable names so that hypothesis
evidence, cached solutions, and learned Options reference actions
consistently across episodes.

This module deliberately keeps the mapping *mechanical*: it does NOT
assign meaning ("ACTION1 is up") to any code — that meaning must be
learned by the engine's miners from observation, not hard-coded
here.  Hard-coding semantics would reintroduce game-specific
knowledge into the submission and defeat the whole purpose of the
engine separation.

What this module DOES provide:

* :func:`engine_action_for` — wrap a raw arc_agi Action in a
  :class:`cognitive_os.Action` with a stable name derived from its
  ``.value``.  The name is ``"ACTION<int>"`` — meaningless on
  purpose.
* :func:`native_action_for` — given an engine :class:`Action` that
  came from ``engine_action_for`` (or shares its naming convention),
  return the matching raw object from a current ``action_space``.
* :func:`engine_action_space` — convenience: map the raw
  ``action_space`` list to engine :class:`Action`\\s in one call.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from cognitive_os import Action


# Stable prefix used by both directions of the translation.  Changing
# it is a breaking change for persisted Options / CachedSolutions
# (they reference actions by this name).
_NAME_PREFIX = "ACTION"


def _raw_value(raw: Any) -> Optional[int]:
    """Extract the integer code from an arc_agi Action-like object.

    The SDK's Action type is an IntEnum-style object with ``.value``,
    but we fall back to ``int(raw)`` and then to string parsing so
    future SDK changes don't silently break the mapping.
    """
    val = getattr(raw, "value", None)
    if isinstance(val, int):
        return val
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    s = str(raw)
    if s.startswith(_NAME_PREFIX):
        try:
            return int(s[len(_NAME_PREFIX):])
        except ValueError:
            return None
    return None


def engine_action_for(raw: Any) -> Action:
    """Wrap a raw arc_agi Action as an engine :class:`Action`.

    The engine identifier is ``ACTION<value>``; the ``id`` field
    matches the ``name`` so the Action is self-identifying in logs.
    Raises :class:`ValueError` if the raw object has no extractable
    integer code.
    """
    val = _raw_value(raw)
    if val is None:
        raise ValueError(f"cannot extract integer code from raw action: {raw!r}")
    name = f"{_NAME_PREFIX}{val}"
    return Action(id=name, name=name, parameters=())


def engine_action_space(raw_space: Sequence[Any]) -> List[Action]:
    """Map an entire ``env.action_space`` to engine actions."""
    return [engine_action_for(r) for r in raw_space]


def native_action_for(
    engine_action: Action,
    raw_space:     Sequence[Any],
) -> Any:
    """Return the raw arc_agi Action that matches ``engine_action`` in
    the currently-available ``raw_space``.

    ``raw_space`` is refetched every step because the ARC-AGI-3 API
    allows the action list to shrink / grow between frames.  If the
    requested action is no longer available, :class:`KeyError` is
    raised — the runner interprets that as "the engine proposed a
    currently-invalid action" and falls through to the explorer.
    """
    target = _parse_name(engine_action.name)
    if target is None:
        raise KeyError(
            f"engine action {engine_action.name!r} is not in the "
            f"{_NAME_PREFIX}<int> namespace"
        )
    for raw in raw_space:
        if _raw_value(raw) == target:
            return raw
    raise KeyError(
        f"action {engine_action.name!r} not available in current action space"
    )


def _parse_name(name: str) -> Optional[int]:
    if not name.startswith(_NAME_PREFIX):
        return None
    try:
        return int(name[len(_NAME_PREFIX):])
    except ValueError:
        return None
