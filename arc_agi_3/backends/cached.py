"""Prompt-keyed caching wrapper for any :class:`ChatBackend`.

Motivation
----------

The COS-solves-L1 loop re-runs the same game level many times while
we iterate on engine capabilities.  Each attempt's initial frame is
byte-identical to every prior attempt's initial frame, so the
:class:`~cognitive_os.oracle.InitialFrameScanTrigger` builds an
identical ``ObserverQuery`` — which in turn produces an identical
messages list into :meth:`ChatBackend.chat`.

Caching at the :meth:`chat` seam means attempt N+1 replays attempt
N's Sonnet call at zero API cost, and it does so *without* coupling
the cache to any specific trigger: as soon as a future trigger
evolves its prompt, the canonicalised messages change → the cache
key changes → a fresh call is made.  The cache needs no surgery
when prompts grow.

Key shape
---------

SHA-1 of ``json.dumps({"messages": [{"role": ..., "content": ...}],
"max_tokens": ..., "temperature": ..., "model": ...}, sort_keys=True,
ensure_ascii=False)``.

``model`` is included so swapping Sonnet 4.5 ↔ 4.6 does not serve
stale answers.  ``temperature`` is included because a non-zero
temperature by definition means "do not cache" — but we still want
the round-trip logged, so we store it and let multiple entries
coexist at the same messages hash.

Cache entry schema (JSON)::

    {
        "key":         "<sha1 hex>",
        "model":       "claude-sonnet-4-5",
        "max_tokens":  600,
        "temperature": 0.0,
        "messages":    [{"role": "...", "content": "..."}, ...],
        "reply":       "<assistant text>",
        "created_at":  1731808800.0,
        "latency_ms":  1234.5
    }

Files are named ``<sha1>.json`` under the cache directory.  The
directory is created lazily and is safe to delete at any time
(the next run will repopulate it).

Budget posture
--------------

Cache hits still count against the engine's ``LLMBudget``
(``observer_per_episode`` / ``mediator_per_episode``) because the
budget lives one layer up in
:class:`~arc_agi_3.backends.base.ChatBackend`.  That is the right
default for now: cache hits are free money-wise but not attention-
wise — the engine still spent a query slot thinking about that
question.  If budget pressure starts to bite we can teach the
wrapper to refund.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .base import ChatBackend, ChatMessage


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Counters the loop runner can read to log cache behaviour."""

    hits:        int = 0
    misses:      int = 0
    writes:      int = 0
    read_errors: int = 0
    # Per-key latency on miss (ms) — useful to estimate Sonnet cost.
    miss_latencies_ms: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, float]:
        total = self.hits + self.misses
        hit_rate = (self.hits / total) if total else 0.0
        return {
            "hits":        float(self.hits),
            "misses":      float(self.misses),
            "writes":      float(self.writes),
            "read_errors": float(self.read_errors),
            "hit_rate":    hit_rate,
            "mean_miss_latency_ms": (
                sum(self.miss_latencies_ms) / len(self.miss_latencies_ms)
                if self.miss_latencies_ms else 0.0
            ),
        }


# ---------------------------------------------------------------------------
# Keying
# ---------------------------------------------------------------------------


def canonicalise_call(
    messages:    List[ChatMessage],
    *,
    max_tokens:  int,
    temperature: float,
    model:       Optional[str],
) -> Dict[str, object]:
    """Reduce a chat() call to a dict that is stable across runs.

    Keeping this pure + public means tests can assert identical
    prompts hash to the same key without going through the filesystem.
    """
    return {
        "messages": [
            {"role": m.role, "content": m.content} for m in messages
        ],
        "max_tokens":  int(max_tokens),
        "temperature": float(temperature),
        "model":       model or "",
    }


def key_for(
    messages:    List[ChatMessage],
    *,
    max_tokens:  int,
    temperature: float,
    model:       Optional[str],
) -> str:
    payload = canonicalise_call(
        messages,
        max_tokens  = max_tokens,
        temperature = temperature,
        model       = model,
    )
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# CachedChatBackend
# ---------------------------------------------------------------------------


class CachedChatBackend(ChatBackend):
    """Wrap any :class:`ChatBackend` with a prompt-keyed on-disk cache.

    The inner backend is consulted only on cache miss.  On miss the
    reply is persisted under ``cache_dir/<sha1>.json`` so the next
    identical call — within this process, or on a later run — serves
    from disk.

    Intentionally **not** a full drop-in for every ``ChatBackend``
    feature: we wrap the ``chat()`` seam only.  Budget accounting
    and usage tracking remain on the inner backend (we forward
    ``usage`` / ``budget`` through read-only proxies below), so that
    the engine's per-episode budget enforcement keeps working as if
    the wrapper were not here.
    """

    def __init__(
        self,
        inner:     ChatBackend,
        *,
        cache_dir: str | os.PathLike,
        model_id:  Optional[str] = None,
    ) -> None:
        # Skip ChatBackend.__init__'s budget setup — we proxy to
        # inner.  LLMBackend.__init__ does the same thing we'd do,
        # but we intentionally do not call either.
        self._inner     = inner
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        # Model id lets us key on the concrete backend model.  If the
        # caller doesn't pass one, we probe the inner for a ``_model``
        # attribute (AnthropicBackend exposes that) and fall back to
        # the class name as a last-ditch differentiator.
        self._model_id = (
            model_id
            or getattr(inner, "_model", None)
            or inner.__class__.__name__
        )
        self.stats = CacheStats()

    # ------------------------------------------------------------------
    # Proxy the usage / budget surface to the inner backend so the
    # engine's budget checks and reporting all still funnel through
    # the one source of truth.
    # ------------------------------------------------------------------

    @property
    def budget(self):  # type: ignore[override]
        return self._inner.budget

    @budget.setter
    def budget(self, value) -> None:
        self._inner.budget = value

    @property
    def usage(self):  # type: ignore[override]
        return self._inner.usage

    @usage.setter
    def usage(self, value) -> None:
        self._inner.usage = value

    def observer_budget_available(self) -> bool:  # type: ignore[override]
        return self._inner.observer_budget_available()

    def mediator_budget_available(self) -> bool:  # type: ignore[override]
        return self._inner.mediator_budget_available()

    def reset_usage(self) -> None:  # type: ignore[override]
        self._inner.reset_usage()

    def usage_snapshot(self) -> Dict[str, float]:  # type: ignore[override]
        return self._inner.usage_snapshot()

    # ------------------------------------------------------------------
    # The actual cache
    # ------------------------------------------------------------------

    def chat(
        self,
        messages:    List[ChatMessage],
        *,
        max_tokens:  int   = 1024,
        temperature: float = 0.0,
    ) -> str:
        key  = key_for(
            messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            model       = self._model_id,
        )
        path = self._cache_dir / f"{key}.json"

        cached_reply = self._try_read(path)
        if cached_reply is not None:
            self.stats.hits += 1
            return cached_reply

        self.stats.misses += 1
        t0 = time.perf_counter()
        reply = self._inner.chat(
            messages,
            max_tokens  = max_tokens,
            temperature = temperature,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0
        self.stats.miss_latencies_ms.append(latency_ms)

        self._try_write(
            path,
            key         = key,
            messages    = messages,
            max_tokens  = max_tokens,
            temperature = temperature,
            reply       = reply,
            latency_ms  = latency_ms,
        )
        return reply

    # ------------------------------------------------------------------
    # Delegate the typed oracle surface straight to the inner backend.
    # ChatBackend's default implementations call ``self.chat`` which
    # resolves to *our* cached chat() via normal method dispatch, so
    # we don't need to duplicate them.  We do, however, need to keep
    # the method resolution path working: ChatBackend.answer_*_query
    # uses the *instance's* chat(), and inheritance chain is
    # CachedChatBackend → ChatBackend → LLMBackend.
    # ------------------------------------------------------------------

    # (inherits ChatBackend.answer_observer_query / answer_mediator_query)

    # ------------------------------------------------------------------
    # IO helpers — tolerant of corruption.  A corrupt cache entry must
    # never break a live run; we treat it as a miss and let the next
    # write overwrite it.
    # ------------------------------------------------------------------

    def _try_read(self, path: Path) -> Optional[str]:
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                entry = json.load(fh)
            reply = entry.get("reply")
            return reply if isinstance(reply, str) else None
        except (OSError, json.JSONDecodeError):
            self.stats.read_errors += 1
            return None

    def _try_write(
        self,
        path: Path,
        *,
        key:         str,
        messages:    List[ChatMessage],
        max_tokens:  int,
        temperature: float,
        reply:       str,
        latency_ms:  float,
    ) -> None:
        entry = {
            "key":         key,
            "model":       self._model_id,
            "max_tokens":  int(max_tokens),
            "temperature": float(temperature),
            "messages":    [
                {"role": m.role, "content": m.content} for m in messages
            ],
            "reply":       reply,
            "created_at":  time.time(),
            "latency_ms":  latency_ms,
        }
        try:
            # Atomic-ish write: write to temp then rename.  Avoids
            # half-written files if the process is killed mid-write.
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(entry, fh, ensure_ascii=False, indent=2)
            tmp.replace(path)
            self.stats.writes += 1
        except OSError:
            # Best-effort — a failed write must not break the run.
            pass
