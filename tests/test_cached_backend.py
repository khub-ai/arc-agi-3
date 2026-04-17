"""Tests for :class:`arc_agi_3.backends.CachedChatBackend`.

We exercise:

* cache miss → forwards to inner + writes to disk
* cache hit  → serves from disk without calling inner
* identical prompts across two separate wrapper instances share the
  cache (persistence across runs)
* changing ``model_id`` changes the key → fresh call
* corrupt cache file does not break the run
* typed ``ObserverQuery`` flow still lands in the cache
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pytest

from cognitive_os import ObserverQuery, QuestionType

from arc_agi_3.backends import (
    CachedChatBackend,
    ChatBackend,
)
from arc_agi_3.backends.base import ChatMessage


class _CountingChat(ChatBackend):
    """ChatBackend that returns a counter string and records call count."""

    def __init__(self, *, reply: str = "ok") -> None:
        super().__init__()
        self.reply      = reply
        self.call_count = 0

    def chat(
        self,
        messages:    List[ChatMessage],
        *,
        max_tokens:  int   = 1024,
        temperature: float = 0.0,
    ) -> str:
        self.call_count += 1
        return self.reply


def _msg() -> List[ChatMessage]:
    return [
        ChatMessage(role="system", content="sys"),
        ChatMessage(role="user",   content="hello"),
    ]


def test_cache_miss_forwards_and_writes(tmp_path: Path) -> None:
    inner   = _CountingChat(reply=json.dumps({"r": 1}))
    wrapper = CachedChatBackend(inner, cache_dir=tmp_path, model_id="m")

    r = wrapper.chat(_msg(), max_tokens=42, temperature=0.0)
    assert r == json.dumps({"r": 1})
    assert inner.call_count == 1
    assert wrapper.stats.misses == 1
    assert wrapper.stats.writes == 1
    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1


def test_cache_hit_skips_inner(tmp_path: Path) -> None:
    inner   = _CountingChat(reply="first")
    wrapper = CachedChatBackend(inner, cache_dir=tmp_path, model_id="m")

    r1 = wrapper.chat(_msg(), max_tokens=42, temperature=0.0)
    # Change the inner reply — if the second call ever hit it we'd see
    # "second" instead of "first".
    inner.reply = "second"
    r2 = wrapper.chat(_msg(), max_tokens=42, temperature=0.0)

    assert r1 == "first"
    assert r2 == "first"
    assert inner.call_count == 1
    assert wrapper.stats.hits == 1
    assert wrapper.stats.misses == 1


def test_cache_persists_across_wrapper_instances(tmp_path: Path) -> None:
    # First wrapper populates the cache.
    inner1   = _CountingChat(reply="cached")
    wrapper1 = CachedChatBackend(inner1, cache_dir=tmp_path, model_id="m")
    wrapper1.chat(_msg(), max_tokens=10, temperature=0.0)

    # Second wrapper, brand new, same cache dir — should hit on disk.
    inner2   = _CountingChat(reply="WOULD_BE_WRONG")
    wrapper2 = CachedChatBackend(inner2, cache_dir=tmp_path, model_id="m")
    r = wrapper2.chat(_msg(), max_tokens=10, temperature=0.0)

    assert r == "cached"
    assert inner2.call_count == 0
    assert wrapper2.stats.hits == 1


def test_model_id_is_part_of_key(tmp_path: Path) -> None:
    inner = _CountingChat(reply="x")
    w1    = CachedChatBackend(inner, cache_dir=tmp_path, model_id="model-A")
    w2    = CachedChatBackend(inner, cache_dir=tmp_path, model_id="model-B")

    w1.chat(_msg(), max_tokens=10, temperature=0.0)
    w2.chat(_msg(), max_tokens=10, temperature=0.0)

    # Both should miss — different models → different keys.
    assert inner.call_count == 2


def test_corrupt_cache_is_treated_as_miss(tmp_path: Path) -> None:
    from arc_agi_3.backends.cached import key_for

    inner   = _CountingChat(reply="fresh")
    wrapper = CachedChatBackend(inner, cache_dir=tmp_path, model_id="m")

    # Pre-seed a corrupt cache file at the expected key.
    k = key_for(_msg(), max_tokens=10, temperature=0.0, model="m")
    (tmp_path / f"{k}.json").write_text("not valid json {", encoding="utf-8")

    r = wrapper.chat(_msg(), max_tokens=10, temperature=0.0)
    assert r == "fresh"
    assert wrapper.stats.read_errors == 1
    assert wrapper.stats.misses == 1


def test_observer_query_flow_through_cache(tmp_path: Path) -> None:
    """The typed oracle surface (answer_observer_query) should cache."""
    canned = json.dumps({
        "result":      [{"position": [5, 5], "role": "agent",
                         "description": "red square"}],
        "confidence":  0.7,
        "explanation": "test",
    })
    inner   = _CountingChat(reply=canned)
    wrapper = CachedChatBackend(inner, cache_dir=tmp_path, model_id="m")

    q = ObserverQuery(
        query_id = "q1",
        question = QuestionType.ENUMERATE_OBJECTS,
        targets  = [],
        frames   = [[[0, 1], [2, 3]]],
    )
    a1 = wrapper.answer_observer_query(q)
    assert a1.confidence > 0.0
    assert inner.call_count == 1

    # Same prompt → cache hit on the chat() seam, inner not called.
    a2 = wrapper.answer_observer_query(q)
    assert inner.call_count == 1
    assert a2.confidence == pytest.approx(a1.confidence)
    # Two oracle calls handled → both count against usage (budget
    # is enforced one layer up regardless of cache).
    assert wrapper.usage.observer_calls == 2
    assert wrapper.stats.hits == 1
