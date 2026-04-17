"""LLM backends — the pluggable seam behind Observer / Mediator.

The :class:`LLMBackend` abstraction decouples the ARC-AGI-3 adapter
from any specific LLM provider.  Two concrete backends are shipped
with Phase 5b:

* :class:`NullBackend` — returns a zero-confidence answer for every
  query, matching the engine's "no oracle available" convention.
  Default in tests and CI (no API key required, fully deterministic).

* :class:`AnthropicBackend` — chats with Claude via the Anthropic
  SDK.  Used for local development and, eventually, the first
  iteration of the ARC Prize submission.

A third, :class:`MockBackend`, lets tests inject canned answers
without touching any network path.

For competition submission we will need an open-source LLM backend.
That is deliberately straightforward in this architecture: subclass
:class:`ChatBackend` and implement the single ``chat()`` method; the
prompt / parse machinery in :mod:`arc_agi_3.observer` and
:mod:`arc_agi_3.mediator` is shared.
"""

from .base import ChatBackend, LLMBackend
from .cached import CachedChatBackend, CacheStats, key_for, canonicalise_call
from .mock import MockBackend
from .null import NullBackend

__all__ = [
    "LLMBackend",
    "ChatBackend",
    "NullBackend",
    "MockBackend",
    "CachedChatBackend",
    "CacheStats",
    "key_for",
    "canonicalise_call",
]

# AnthropicBackend is imported lazily so that environments without the
# anthropic SDK installed can still import arc_agi_3.backends (e.g.
# for tests that use NullBackend / MockBackend).
def _anthropic_backend() -> type:
    from .anthropic_backend import AnthropicBackend
    return AnthropicBackend


def __getattr__(name: str):
    if name == "AnthropicBackend":
        return _anthropic_backend()
    raise AttributeError(f"module 'arc_agi_3.backends' has no attribute {name!r}")
