"""VLM backend abstraction for the perception loop.

Defines a small interface every backend implementation satisfies.
Lets the orchestration loop (vlm_perception.py) stay vendor-agnostic
and lets the test harness swap in stub backends.

Implementations included here:

  ManualPasteBackend  - file-based handoff: writes the prompt + image
                        to a pending directory and polls for a reply
                        JSON file.  Mirrors the harness's
                        call_human_vlm pattern but is self-contained
                        (no harness imports).  Runs today without an
                        API key.

  StubBackend         - returns pre-canned responses keyed by call
                        index.  For tests and dry runs.

A future LiveAnthropicBackend would call the Anthropic API directly
using the anthropic SDK with claude-opus-4-7; same .call() signature.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


class VLMBackend(Protocol):
    """A backend is anything with a .call(system, user, image_b64) -> str
    method.  Stateless across calls (or carries its own state — the
    loop doesn't care).
    """

    def call(
        self,
        *,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        ...


@dataclass
class ManualPasteBackend:
    """File-based VLM handoff: writes prompt + image to a pending
    directory, polls for a reply file the operator writes.

    Directory layout under `pending_dir`:
      call_NNN_prompt.md       - system + user prompt (operator copies
                                 into Claude, ChatGPT, etc.)
      call_NNN_image.png       - the frame image
      call_NNN_reply.json      - {"raw_text": "..."} written by the
                                 operator after pasting the VLM
                                 response.  Polled until present.
      call_NNN_reply.consumed.json - the file is renamed after read
                                 so the next call gets a fresh slot.
      STATUS.txt               - one-line summary the operator's CLI
                                 / browser can poll.
    """

    pending_dir: Path
    timeout_s: int = 1800
    poll_interval_s: float = 2.0
    # Per-instance call counter.  Files are call_001_..., call_002_...
    _next_n: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        self.pending_dir = Path(self.pending_dir)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        # Resume the counter if the directory already has call files.
        existing = sorted(self.pending_dir.glob("call_*_prompt.md"))
        if existing:
            self._next_n = len(existing) + 1

    def call(
        self,
        *,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        n = self._next_n
        self._next_n += 1
        prompt_path = self.pending_dir / f"call_{n:03d}_prompt.md"
        image_path = self.pending_dir / f"call_{n:03d}_image.png"
        reply_path = self.pending_dir / f"call_{n:03d}_reply.json"
        status_path = self.pending_dir / "STATUS.txt"

        prompt_text = (
            f"# Call #{n}\n\n"
            f"## SYSTEM PROMPT\n\n```\n{system}\n```\n\n---\n\n"
            f"## USER PROMPT\n\n```\n{user}\n```\n\n---\n\n"
            f"## Reply instructions\n\n"
            f"Save your reply as JSON to:\n  `{reply_path.name}`\n\n"
            f'Body shape: `{{"raw_text": "<your full VLM response>"}}`\n'
        )
        prompt_path.write_text(prompt_text, encoding="utf-8")
        if image_b64:
            try:
                image_path.write_bytes(base64.b64decode(image_b64))
            except Exception:
                pass
        has_image = " + image" if image_b64 else ""
        status_path.write_text(
            f"WAITING FOR REPLY on call_{n:03d}{has_image}\n"
            f"Wrote prompt to: {prompt_path}\n"
            f"Reply expected at: {reply_path}\n",
            encoding="utf-8",
        )

        start = time.time()
        body = ""
        while time.time() - start < self.timeout_s:
            if reply_path.exists():
                try:
                    body = reply_path.read_text(encoding="utf-8").strip()
                except Exception:
                    body = ""
                if body:
                    break
            time.sleep(self.poll_interval_s)
        if not body:
            raise TimeoutError(
                f"manual-paste VLM did not write {reply_path} within "
                f"{self.timeout_s}s"
            )
        # Archive consumed reply so it doesn't get re-served.
        try:
            consumed = self.pending_dir / f"call_{n:03d}_reply.consumed.json"
            reply_path.rename(consumed)
        except Exception:
            pass
        status_path.write_text(
            f"RECEIVED reply for call_{n:03d}\n",
            encoding="utf-8",
        )
        # Extract raw_text if the reply is wrapped, else return as-is.
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict) and "raw_text" in parsed:
                return str(parsed["raw_text"])
        except json.JSONDecodeError:
            pass
        return body


@dataclass
class StubBackend:
    """Returns pre-canned responses keyed by call index.  For tests
    and dry runs.  Useful to:
      - Validate the loop end-to-end without a real VLM
      - Replay captured responses from a real prior run
    """

    responses: list[str] = field(default_factory=list)
    _next_n: int = field(default=0, init=False)

    def call(
        self,
        *,
        system: str,
        user: str,
        image_b64: Optional[str] = None,
    ) -> str:
        if self._next_n >= len(self.responses):
            raise RuntimeError(
                f"StubBackend exhausted at call {self._next_n + 1}; "
                f"only {len(self.responses)} canned responses provided"
            )
        out = self.responses[self._next_n]
        self._next_n += 1
        return out


def load_canned_responses(directory: Path) -> list[str]:
    """Load canned VLM responses from a directory of call_NNN_reply
    JSONs (raw_text field), in numeric order.  Used to seed a
    StubBackend from a captured manual-paste session.
    """
    out: list[str] = []
    for p in sorted(directory.glob("call_*_reply*.json")):
        try:
            body = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(body, dict) and "raw_text" in body:
                out.append(str(body["raw_text"]))
            else:
                out.append(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            out.append(p.read_text(encoding="utf-8"))
    return out
