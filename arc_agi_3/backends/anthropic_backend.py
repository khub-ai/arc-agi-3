"""Anthropic Claude backend.

Requires the ``anthropic`` SDK (install via ``pip install
arc-agi-3[llm]``).  The backend is intentionally minimal: it takes a
list of :class:`ChatMessage`\\s, builds an Anthropic ``messages``
call, and returns the assistant's text.  Prompt and parse logic
live in :mod:`arc_agi_3.observer` and :mod:`arc_agi_3.mediator` so
that swapping to an open-source LLM — required before competition
submission per the user's standing direction — is a single-file
change.
"""

from __future__ import annotations

import os
from typing import List, Optional

from .base import ChatBackend, ChatMessage


# Default model identifier.  We pin to a known-stable Sonnet 4.5
# string rather than a moving "-latest" alias so a silent Anthropic
# model rotation can never change our behaviour between CI and live
# runs.  Override per-invocation with ``--model`` when you want to
# benchmark a newer tier (e.g. ``--model claude-sonnet-4-6``).
_DEFAULT_MODEL = "claude-sonnet-4-5"
_API_KEY_ENV   = "ANTHROPIC_API_KEY"


class AnthropicBackend(ChatBackend):
    """ChatBackend that forwards to Anthropic Claude.

    Parameters
    ----------
    api_key
        Anthropic API key.  Falls back to the ``ANTHROPIC_API_KEY``
        environment variable when omitted.  Raising here (rather
        than lazily at first call) gives the caller a fast failure
        signal at adapter construction.
    model
        Model identifier; defaults to ``claude-3-5-sonnet-latest``.
        Override when benchmarking different Claude tiers.
    budget
        Optional :class:`cognitive_os.LLMBudget`; omit to accept the
        engine's default caps.
    """

    def __init__(
        self,
        *,
        api_key:  Optional[str]             = None,
        model:    str                       = _DEFAULT_MODEL,
        budget:   Optional[object]          = None,
    ) -> None:
        super().__init__(budget=budget)
        key = api_key or os.environ.get(_API_KEY_ENV)
        if not key:
            raise RuntimeError(
                f"Anthropic API key not provided and {_API_KEY_ENV} is unset."
            )
        # Lazy import so tests using other backends need not have the
        # SDK installed.
        try:
            import anthropic  # type: ignore
        except ImportError as exc:   # pragma: no cover — import error branch
            raise RuntimeError(
                "anthropic SDK not installed.  Install with "
                "`pip install arc-agi-3[llm]` to use AnthropicBackend."
            ) from exc
        self._client = anthropic.Anthropic(api_key=key)
        self._model  = model

    def chat(
        self,
        messages:    List[ChatMessage],
        *,
        max_tokens:  int   = 1024,
        temperature: float = 0.0,
    ) -> str:
        system, user_messages = _split_system(messages)
        response = self._client.messages.create(
            model       = self._model,
            max_tokens  = max_tokens,
            temperature = temperature,
            system      = system,
            messages    = [
                {"role": m.role, "content": m.content}
                for m in user_messages
            ],
        )
        # Anthropic replies are content blocks; for our text-only
        # prompts we concatenate all text blocks into a single string.
        return "".join(
            getattr(block, "text", "") for block in getattr(response, "content", [])
        )


def _split_system(messages: List[ChatMessage]) -> tuple[str, List[ChatMessage]]:
    """Anthropic's ``messages`` API takes the system prompt separately
    from the user / assistant turns.  Extract all ``system`` messages,
    concatenate them, and return the remaining turns."""
    system_parts: List[str] = []
    rest:         List[ChatMessage] = []
    for msg in messages:
        if msg.role == "system":
            system_parts.append(msg.content)
        else:
            rest.append(msg)
    return ("\n\n".join(system_parts), rest)
