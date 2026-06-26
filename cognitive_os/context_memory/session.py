"""Per-game perception session helper.

Bundles the canonical context_memory usage pattern:

  1. Load any prior committed claims for this game from disk.
  2. Route this session's parsed-perception view through the bridge
     into a WorldState (proposing / merging claims).
  3. Save the WorldState's committed claims back to disk.

Each game gets its own KB directory under ``kb_root/<game_id>/`` so
cross-game leakage is impossible by construction -- the helper never
reads or writes outside the game's directory.

Probes, the discovery_play harness, and any future use-case adapter
can replace 20+ lines of boilerplate with one
``run_perception_session(parsed, kb_root=..., game_id=..., step=...)``
call.

Substrate-general: the helper takes a generic ``ScopeKind`` (default
``GAME``) so robotics or other-domain adapters using different scope
semantics plug in unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from ..config import EngineConfig
from ..types import Scope, ScopeKind, WorldState
from .markdown_io import (
    load_committed_hierarchy,
    save_committed_hierarchy,
)
from .perception_bridge import propose_from_parsed


@dataclass
class SessionResult:
    """What happened during one perception session.

    Attributes
    ----------
    ws
        The :class:`WorldState` the session built and saved to disk.
        Owned by the helper; caller may inspect it (read-only) but
        should not mutate after the helper returns.
    kb_dir
        The per-game directory the helper read from and wrote back
        to.  Always ``kb_root / game_id``.
    n_loaded
        Number of prior committed claims loaded from disk at the
        start of the session.  ``0`` on a fresh KB.
    n_proposed
        Number of claims the bridge proposed (including ones merged
        with priors via the hypothesis store's dedup; the same
        canonical_key counts as one proposal regardless of whether
        it's literally new).
    n_saved
        Number of committed claims (point >= commit_threshold)
        written back to disk at the end.
    by_type_loaded
        ``{ClaimClassName: count}`` for the loaded set.
    by_type_proposed
        ``{ClaimClassName: count}`` for what the bridge produced.
    """
    ws:                WorldState
    kb_dir:            Path
    n_loaded:          int
    n_proposed:        int
    n_saved:           int
    by_type_loaded:    Dict[str, int] = field(default_factory=dict)
    by_type_proposed:  Dict[str, int] = field(default_factory=dict)


def run_perception_session(
    parsed:        Mapping,
    *,
    kb_root:       Path,
    game_id:       str,
    step:          int,
    level_id:      Optional[str] = None,
    scope_kind:    ScopeKind = ScopeKind.GAME,
    source_prefix: str       = "perception",
    ws:            Optional[WorldState] = None,
) -> SessionResult:
    """Load -> bridge -> save for one perception session, with full
    scope-hierarchy support.

    On LOAD, the helper walks every applicable scope directory under
    ``kb_root`` (broadest first):

      ``kb_root/universal/``           -- GLOBAL scope priors
      ``kb_root/<game_id>/``           -- GAME scope claims
      ``kb_root/<game_id>/levels/<level_id>/``  -- LEVEL scope claims

    On SAVE, each committed hypothesis is partitioned by its own
    ``Scope.kind`` and persisted to the matching directory.  A
    GAME-scoped claim never lands in the universal directory; a
    LEVEL-scoped claim never leaks to the per-game one.

    Parameters
    ----------
    parsed
        The substrate's parsed view.  Not mutated.
    kb_root
        Filesystem root for the entire knowledge base.
    game_id
        Stable per-game identifier.  Required.
    step
        Current step / frame count.
    level_id
        Optional level identifier.  When provided, LEVEL-scoped
        claims load from / save to
        ``kb_root/<game_id>/levels/<level_id>/``.
    scope_kind
        ``ScopeKind`` for hypotheses the bridge proposes this
        session.  Defaults to GAME.  Pass LEVEL for facts that
        should not transfer across levels.
    source_prefix
        Provenance tag prefix; passed through to the bridge.
    ws
        Optional caller-supplied :class:`WorldState`.

    Returns
    -------
    SessionResult
        Counts include claims loaded from EVERY applicable scope and
        every claim saved to EVERY applicable scope.  ``kb_dir`` is
        the *primary* directory for this session's proposed claims
        (game or level depending on ``scope_kind``) -- useful for
        diagnostics; cross-scope reads/writes are not visible
        through ``kb_dir`` alone.
    """
    kb_root = Path(kb_root)
    kb_root.mkdir(parents=True, exist_ok=True)

    if ws is None:
        ws = WorldState()
        ws.config = EngineConfig()

    loaded_by_scope = load_committed_hierarchy(
        ws, kb_root,
        game_id=game_id, level_id=level_id, step=0,
    )
    loaded_ids: List[str] = [hid for ids in loaded_by_scope.values()
                             for hid in ids]
    by_type_loaded: Dict[str, int] = {}
    for hid in loaded_ids:
        h = ws.hypotheses[hid]
        name = type(h.claim).__name__
        by_type_loaded[name] = by_type_loaded.get(name, 0) + 1

    bridge_ids = propose_from_parsed(
        parsed, ws,
        scope=Scope(kind=scope_kind),
        step=step,
        source_prefix=source_prefix,
    )
    by_type_proposed: Dict[str, int] = {}
    for hid in bridge_ids:
        h = ws.hypotheses[hid]
        name = type(h.claim).__name__
        by_type_proposed[name] = by_type_proposed.get(name, 0) + 1

    saved_by_scope = save_committed_hierarchy(
        ws, kb_root,
        game_id=game_id, level_id=level_id,
    )
    n_saved = sum(len(v) for v in saved_by_scope.values())

    # The "primary" kb_dir is the scope this session's proposals
    # belong to (game vs level), useful for callers that want to
    # display a single path in their logs.
    if scope_kind == ScopeKind.LEVEL and level_id:
        kb_dir = kb_root / game_id / "levels" / str(level_id)
    else:
        kb_dir = kb_root / game_id

    return SessionResult(
        ws               = ws,
        kb_dir           = kb_dir,
        n_loaded         = len(loaded_ids),
        n_proposed       = len(bridge_ids),
        n_saved          = n_saved,
        by_type_loaded   = by_type_loaded,
        by_type_proposed = by_type_proposed,
    )
