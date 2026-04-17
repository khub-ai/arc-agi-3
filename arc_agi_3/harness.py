"""Live harness — CLI entry point for running the engine against the
ARC-AGI-3 SDK.

This module wires the three pieces that have been built in isolation:

* The domain adapter (:class:`ArcAdapter`).
* A pluggable LLM backend (``NullBackend`` by default;
  ``AnthropicBackend`` when ``--backend anthropic`` is selected).
* The engine's :func:`cognitive_os.run_episode` driver.

The harness is intentionally small.  It owns:

1. argparse — CLI surface for the competition script.
2. Arcade lifecycle — construct, ``make(game_id)``, cleanup.
3. Backend construction — pick the right :class:`LLMBackend` based on
   flags.
4. WorldState / EngineConfig wiring — one fresh WorldState per
   episode; ``EngineConfig.arc_agi3_default()`` unless overridden.
5. Per-episode loop — run one or more episodes, aggregate
   PostMortems, persist lessons between episodes so cross-episode
   accumulation works even inside a single CLI invocation.

It deliberately does NOT know anything about the SDK beyond the
``Arcade`` / ``make`` / ``step`` / ``reset`` contract — that is the
adapter's job.  Swapping the adapter (for a simulator, say) would
leave this file untouched.

Tests drive ``main`` against a fake ``Arcade`` so no live API traffic
occurs in CI; see :mod:`tests.test_harness`.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence

from cognitive_os import (
    EngineConfig,
    PostMortem,
    WorldState,
    run_episode,
)

from .adapter import ArcAdapter
from .backends import AnthropicBackend, LLMBackend, NullBackend
from .persistence import load_knowledge, save_knowledge


_LOG = logging.getLogger("arc_agi_3.harness")


# ---------------------------------------------------------------------------
# Result aggregation
# ---------------------------------------------------------------------------


@dataclass
class HarnessResult:
    """Summary of one ``main()`` invocation.

    Exposed as a return value (and as program-exit metadata) so that
    callers embedding the harness in a larger pipeline (scorecard
    sweeps, grid search) can inspect the run without parsing stdout.
    """
    game_id:       str
    episodes:      List[PostMortem]
    wall_time_s:   float

    @property
    def successes(self) -> int:
        return sum(1 for pm in self.episodes if pm.final_status == "success")

    @property
    def failures(self) -> int:
        return sum(1 for pm in self.episodes if pm.final_status != "success")


# ---------------------------------------------------------------------------
# Arcade factory (injection point for tests)
# ---------------------------------------------------------------------------


ArcadeFactory = Callable[[str], Any]
"""Signature ``(api_key) -> Arcade-like``.  The object returned must
implement ``make(game_id, ...)`` returning an env with ``reset()`` /
``step(action)`` / ``action_space``.  Tests pass a fake factory; the
default imports :class:`arc_agi.Arcade` lazily so importing this module
does not require the SDK."""


def _default_arcade_factory(api_key: str) -> Any:
    # Lazy import — keeps harness import cheap and test setups (which
    # inject their own factory) free of the SDK dependency.
    from arc_agi import Arcade  # type: ignore
    return Arcade(arc_api_key=api_key)


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def _build_backend(name: str, *, api_key: Optional[str], model: Optional[str]) -> LLMBackend:
    """Construct the backend named on the CLI.

    Any new backend (open-source LLM for competition submission) gets
    a new branch here and nothing else needs to change.  The adapter
    reads ``self.backend`` through the :class:`LLMBackend` protocol.
    """
    if name == "null":
        return NullBackend()
    if name == "anthropic":
        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if model:
            kwargs["model"] = model
        return AnthropicBackend(**kwargs)
    raise ValueError(f"unknown backend {name!r}")


# ---------------------------------------------------------------------------
# Episode orchestration
# ---------------------------------------------------------------------------


def run_harness(
    *,
    game_id:         str,
    episodes:        int           = 1,
    max_steps:       int           = 10_000,
    backend:         str           = "null",
    api_key:         Optional[str] = None,
    backend_api_key: Optional[str] = None,
    model:           Optional[str] = None,
    knowledge_dir:   Optional[str] = None,
    load_knowledge_: bool          = True,
    save_knowledge_: bool          = True,
    arcade_factory:  Optional[ArcadeFactory] = None,
    cfg:             Optional[EngineConfig]  = None,
) -> HarnessResult:
    """Run ``episodes`` episodes of ``game_id`` and return a summary.

    Separated from :func:`main` so test code and embedding callers can
    invoke it without argparse.  Cross-episode knowledge accumulation
    works naturally here: the :class:`WorldState` is preserved across
    episodes within a single ``run_harness`` call, so lessons and
    synthesised Options committed by the post-mortem are available to
    the next episode.
    """
    factory = arcade_factory or _default_arcade_factory
    arcade  = factory(api_key or "")

    env = arcade.make(game_id)
    if env is None:
        raise RuntimeError(
            f"Arcade.make({game_id!r}) returned None.  Check the game id "
            "and operation mode."
        )

    llm_backend = _build_backend(backend, api_key=backend_api_key, model=model)

    adapter = ArcAdapter(raw_env=env, env_id=game_id, backend=llm_backend)

    # A single WorldState across episodes is the vehicle for
    # cross-episode accumulation: the engine commits lessons, options,
    # and credence updates into the hypothesis store at episode end;
    # the next episode sees them.  Callers that want fresh runs each
    # time should invoke ``run_harness`` per episode.
    ws = WorldState()
    cfg = cfg or EngineConfig.arc_agi3_default()

    # Cross-invocation knowledge: load before the first episode runs
    # so CachedSolutions are available to the planner from step 0.
    # ``load_knowledge_`` being False is competition-safe — no prior
    # knowledge leaks in.
    if knowledge_dir and load_knowledge_:
        report = load_knowledge(ws, knowledge_dir)
        _LOG.info(
            "loaded %d cached_solutions from %s",
            report.cached_solutions, report.path,
        )

    pms: List[PostMortem] = []
    started = time.time()
    for ep_idx in range(episodes):
        episode_id = f"{game_id}::ep{ep_idx:04d}"
        _LOG.info("starting episode %s", episode_id)
        pm = run_episode(
            adapter,
            ws,
            cfg,
            episode_id = episode_id,
            max_steps  = max_steps,
        )
        pms.append(pm)
        _LOG.info(
            "episode %s finished: status=%s steps=%d wall=%.2fs",
            episode_id, pm.final_status, pm.total_steps, pm.wall_time_seconds,
        )
    wall = time.time() - started

    # Persist after all episodes so that lessons and CachedSolutions
    # committed by each PostMortem are on disk for the next CLI run.
    # Saving is opt-out to keep the default "training mode" loud about
    # accumulation; competition runs pass ``save_knowledge_=False``.
    if knowledge_dir and save_knowledge_:
        save_knowledge(ws, knowledge_dir)

    return HarnessResult(game_id=game_id, episodes=pms, wall_time_s=wall)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog        = "arc-agi-3",
        description = "Run the Cognitive OS engine against ARC-AGI-3.",
    )
    p.add_argument(
        "--game-id", required=True,
        help="ARC-AGI-3 game identifier, e.g. 'ls20' or 'ls20-<version>'.",
    )
    p.add_argument(
        "--episodes", type=int, default=1,
        help="Number of episodes to run (default: 1).",
    )
    p.add_argument(
        "--max-steps", type=int, default=10_000,
        help="Per-episode step cap (default: 10000).",
    )
    p.add_argument(
        "--backend", choices=("null", "anthropic"), default="null",
        help=(
            "LLM backend for Observer / Mediator queries.  'null' means "
            "zero-confidence answers (engine relies on symbolic miners "
            "only).  'anthropic' calls Claude via the Anthropic SDK."
        ),
    )
    p.add_argument(
        "--model", default=None,
        help="Override the backend's model id (backend-specific).",
    )
    p.add_argument(
        "--api-key", default=None,
        help=(
            "ARC API key.  Falls back to the ARC_API_KEY environment "
            "variable.  Omit in offline replay tests."
        ),
    )
    p.add_argument(
        "--backend-api-key", default=None,
        help=(
            "LLM backend API key (e.g. Anthropic).  Falls back to the "
            "backend-specific env var (e.g. ANTHROPIC_API_KEY)."
        ),
    )
    p.add_argument(
        "--knowledge-dir", default=None,
        help=(
            "Directory used for cross-invocation knowledge persistence "
            "(CachedSolutions etc.).  Loaded before the first episode "
            "and saved after the last.  Omit for an ephemeral run."
        ),
    )
    p.add_argument(
        "--no-load-knowledge", action="store_true",
        help=(
            "Skip loading prior knowledge from --knowledge-dir.  Useful "
            "for competition mode where prior solutions must not leak "
            "into the run."
        ),
    )
    p.add_argument(
        "--no-save-knowledge", action="store_true",
        help=(
            "Skip saving knowledge to --knowledge-dir at the end of the "
            "run.  Useful for read-only sweeps over a shared store."
        ),
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level (default: INFO).",
    )
    return p


def _print_summary(result: HarnessResult, stream: Any = sys.stdout) -> None:
    print(
        f"\n=== {result.game_id}: {result.successes}/{len(result.episodes)} "
        f"succeeded ({result.wall_time_s:.2f}s wall) ===",
        file=stream,
    )
    for pm in result.episodes:
        print(
            f"  {pm.episode_id}: status={pm.final_status} "
            f"steps={pm.total_steps} lessons={len(pm.lessons)} "
            f"options={len(pm.options_synthesised)}",
            file=stream,
        )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.  Returns a POSIX-style exit status."""
    parser = _build_parser()
    args   = parser.parse_args(argv)

    logging.basicConfig(
        level  = getattr(logging, args.log_level),
        format = "%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Env-var fallbacks are applied here (not in ``run_harness``) so
    # programmatic callers can bypass them by passing explicit kwargs.
    api_key         = args.api_key         or os.environ.get("ARC_API_KEY")
    backend_api_key = args.backend_api_key or os.environ.get("ANTHROPIC_API_KEY")

    try:
        result = run_harness(
            game_id         = args.game_id,
            episodes        = args.episodes,
            max_steps       = args.max_steps,
            backend         = args.backend,
            api_key         = api_key,
            backend_api_key = backend_api_key,
            model           = args.model,
            knowledge_dir   = args.knowledge_dir,
            load_knowledge_ = not args.no_load_knowledge,
            save_knowledge_ = not args.no_save_knowledge,
        )
    except Exception as exc:
        _LOG.exception("harness failed: %s", exc)
        return 2

    _print_summary(result)
    # Exit code: 0 if every episode succeeded, 1 otherwise.  Useful
    # for shell scripting over a scorecard sweep.
    return 0 if result.failures == 0 else 1


if __name__ == "__main__":   # pragma: no cover — manual invocation only
    sys.exit(main())
