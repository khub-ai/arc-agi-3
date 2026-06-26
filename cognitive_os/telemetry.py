"""Telemetry sinks — the engine side of the wire protocol.

Three sinks are provided:

* :class:`NullSink`  — default; drops every event at near-zero cost.  Kept
  as the default on :class:`EngineConfig` so pure engine callers pay
  nothing for the instrumentation they do not consume.
* :class:`NDJSONSink` — appends one JSON line per event to a file on
  disk.  Enables offline replay and post-hoc analysis without any GUI.
* :class:`WebSocketSink` — fire-and-forget send of each event over a
  websocket connection.  Implemented in a later PR when the sidecar
  server lands; a stub lives in this module so the import surface is
  stable.

The :class:`TelemetrySink` protocol keeps the engine decoupled from the
transport: a subsystem calls ``sink.emit(payload, ...)`` and never
learns whether the bytes went to ``/dev/null``, a file, or a socket.

Thread-safety
-------------

:class:`NDJSONSink` serialises writes under an internal lock, so
multiple threads can emit concurrently.  Sinks are expected to be
*non-blocking from the engine's point of view* — if a future sink
needs to do network I/O it must do so on its own thread with a
bounded queue.  Monitoring must never slow training.

Standing directive: the engine may call :meth:`TelemetrySink.emit` at
any hook point; the sink must tolerate high call rates and large NDJSON
files.  Backpressure is the sink's responsibility, not the engine's.
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional, Protocol, runtime_checkable

from .telemetry_schema import (
    SCHEMA_VERSION,
    TelemetryEnvelope,
    envelope_from_dict,
    envelope_to_dict,
    mint_session_id,
    payload_from_dict,
    payload_to_dict,
)


# ===========================================================================
# Sink protocol
# ===========================================================================


@runtime_checkable
class TelemetrySink(Protocol):
    """Protocol for telemetry sinks.

    Implementations wrap whatever transport is appropriate (NDJSON file,
    websocket, in-memory ring buffer).  Call sites in the engine invoke
    :meth:`emit` with a payload dataclass and optional routing metadata;
    the sink is responsible for constructing the envelope and moving
    the bytes.

    The protocol is deliberately minimal — one method — so that every
    hook point in the engine looks the same and the :class:`NullSink`
    default costs exactly one attribute lookup and one no-op call.
    """

    def emit(
        self,
        payload: Any,
        *,
        episode: Optional[str] = None,
        step:    Optional[int] = None,
        subject: Optional[str] = None,
        prev:    Optional[str] = None,
    ) -> Optional[str]:
        """Serialise ``payload`` into an envelope and dispatch it.

        Returns the minted event id (so the caller can record it as the
        ``prev`` of the next event with the same subject), or ``None``
        if the sink is a no-op.
        """
        ...

    def close(self) -> None:
        """Flush and release any resources held by the sink.

        Safe to call multiple times.  A sink must remain usable after
        ``close`` only if its docstring explicitly says so — the
        default contract is "closed sinks silently drop further
        emits" so the engine need not branch on sink state during
        shutdown.
        """
        ...


# ===========================================================================
# NullSink — the default
# ===========================================================================


class NullSink:
    """Zero-cost sink that drops every event.

    Used as the default on :class:`EngineConfig.telemetry` so that code
    paths with no interest in telemetry pay nothing for the emit call
    beyond a method dispatch.  ``emit`` returns ``None`` because no id
    was minted.
    """

    __slots__ = ()

    def emit(
        self,
        payload: Any,
        *,
        episode: Optional[str] = None,
        step:    Optional[int] = None,
        subject: Optional[str] = None,
        prev:    Optional[str] = None,
    ) -> Optional[str]:
        return None

    def close(self) -> None:
        return None


# ===========================================================================
# NDJSONSink — disk recorder
# ===========================================================================


class NDJSONSink:
    """Append events as NDJSON lines to a single file.

    One sink instance owns one open file handle and one session id.
    Event ids within the sink are of the form ``"<session>-<seq>"``
    where ``<seq>`` is a monotonic counter starting at 0.

    Parameters
    ----------
    path
        Filesystem path for the NDJSON file.  Parent directories are
        created on construction.  Existing files are appended to.
    session_id
        Override for the auto-minted session id.  Pass an explicit id
        when resuming a logical session across process restarts so
        downstream tools can stitch the runs together.
    flush_every
        Call ``file.flush()`` this often (events).  Default ``1``
        (flush on every write) — cheap on local disk and guarantees
        crash-dump visibility during development.  Raise to 100+ for
        high-throughput training runs on slow storage.

    Notes
    -----
    The sink uses a clock baseline captured at construction time so
    ``ts`` values are monotonic and comparable across all events from
    this sink.  The baseline is not correlated with the wall clock —
    see ``wall`` in the envelope if wall-time alignment is needed.
    """

    def __init__(
        self,
        path:          os.PathLike[str] | str,
        *,
        session_id:    Optional[str] = None,
        flush_every:   int = 1,
    ) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: Optional[io.TextIOBase] = open(
            self._path, "a", encoding="utf-8"
        )
        self._lock = threading.Lock()
        self._session = session_id if session_id is not None else mint_session_id()
        self._seq = 0
        self._flush_every = max(1, int(flush_every))
        self._since_flush = 0
        self._t0_ns = time.monotonic_ns()

    # -----------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------

    @property
    def path(self) -> Path:
        """The NDJSON file this sink is writing to."""
        return self._path

    @property
    def session_id(self) -> str:
        """The 6-character session id shared by every event this sink emits."""
        return self._session

    # -----------------------------------------------------------------
    # Sink protocol
    # -----------------------------------------------------------------

    def emit(
        self,
        payload: Any,
        *,
        episode: Optional[str] = None,
        step:    Optional[int] = None,
        subject: Optional[str] = None,
        prev:    Optional[str] = None,
    ) -> Optional[str]:
        tag = getattr(type(payload), "TYPE", None)
        if not isinstance(tag, str) or not tag:
            raise TypeError(
                f"{type(payload).__name__} is not a registered telemetry "
                f"event (missing TYPE class variable)"
            )
        with self._lock:
            if self._fh is None:
                return None
            self._seq += 1
            event_id = f"{self._session}-{self._seq:x}"
            ts_ms = (time.monotonic_ns() - self._t0_ns) / 1_000_000.0
            env = TelemetryEnvelope(
                v       = SCHEMA_VERSION,
                id      = event_id,
                ts      = ts_ms,
                wall    = datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                type    = tag,
                payload = payload_to_dict(payload),
                episode = episode,
                step    = step,
                subject = subject,
                prev    = prev,
            )
            line = json.dumps(envelope_to_dict(env), separators=(",", ":"))
            self._fh.write(line)
            self._fh.write("\n")
            self._since_flush += 1
            if self._since_flush >= self._flush_every:
                self._fh.flush()
                self._since_flush = 0
            return event_id

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.flush()
                finally:
                    self._fh.close()
                    self._fh = None

    # -----------------------------------------------------------------
    # Context-manager ergonomics
    # -----------------------------------------------------------------

    def __enter__(self) -> "NDJSONSink":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:           # best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


# ===========================================================================
# Reader — replay and offline analysis
# ===========================================================================


def read_ndjson(
    path: os.PathLike[str] | str,
) -> Iterator[TelemetryEnvelope]:
    """Yield envelopes from an NDJSON file, one per line.

    Lines that fail to parse as JSON or fail envelope deserialisation
    are skipped with no error — log tailing must be tolerant of
    partial lines at the tail.  Strict validation is a client concern.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            try:
                yield envelope_from_dict(obj)
            except (KeyError, TypeError):
                continue


def emit_from_ws(
    ws:      Any,
    payload: Any,
    *,
    subject: Optional[str] = None,
    step:    Optional[int] = None,
) -> Optional[str]:
    """Emit a telemetry event using the sink attached to ``ws.config``.

    Convenience wrapper for engine subsystems (hypothesis_store,
    goal_forest) whose functions operate on a :class:`WorldState` but
    are not passed :class:`EngineConfig` directly.  Returns ``None`` —
    i.e. is a no-op — when no config is attached or no telemetry sink
    is configured; this keeps all paths usable from tests that build a
    bare ``WorldState``.

    Episode id and step are read from ``ws.agent['_episode_id']`` and
    ``ws.step`` respectively when not supplied explicitly.  The runner
    seeds ``_episode_id`` at episode start.
    """
    cfg = getattr(ws, "config", None)
    sink = getattr(cfg, "telemetry", None) if cfg is not None else None
    if sink is None:
        return None
    agent = getattr(ws, "agent", None) or {}
    episode = agent.get("_episode_id")
    if step is None:
        step = getattr(ws, "step", None)
    return sink.emit(
        payload,
        episode = episode,
        step    = step,
        subject = subject,
    )


def decode_payload(env: TelemetryEnvelope) -> Any:
    """Reconstruct the payload dataclass for an envelope.

    Returns the frozen dataclass instance, or raises :class:`KeyError`
    if the event type is not registered in this runtime.  Unknown
    event types are a forward-compatibility concern for clients — use
    :func:`envelope_from_dict` directly and inspect ``env.type`` if the
    caller needs to tolerate unknown types.
    """
    return payload_from_dict(env.type, env.payload)


# ===========================================================================
# WebSocketSink — stub, filled in when the sidecar lands
# ===========================================================================


class WebSocketSink:
    """Placeholder for the live-streaming sink.

    Will wrap a websockets client with a bounded outbound queue,
    dropping low-priority events (``Heartbeat``, ``LogMessage``) under
    backpressure.  Implemented in the sidecar PR; the class is defined
    here only so the import surface is stable.
    """

    def __init__(self, url: str) -> None:
        raise NotImplementedError(
            "WebSocketSink is not yet implemented — wait for the sidecar PR"
        )

    def emit(self, payload: Any, **kwargs: Any) -> Optional[str]:  # pragma: no cover
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover
        raise NotImplementedError


__all__ = [
    "TelemetrySink",
    "NullSink",
    "NDJSONSink",
    "WebSocketSink",
    "read_ndjson",
    "decode_payload",
    "emit_from_ws",
]
