"""cos-tail — pretty-print a telemetry NDJSON stream.

Installed as the ``cos-tail`` console script (see ``pyproject.toml``).
Two modes:

* ``cos-tail path/to/run.ndjson``      — read the file to end, then exit.
* ``cos-tail -f path/to/run.ndjson``   — follow (like ``tail -f``);
                                          print each new line as it
                                          appears, exit on ^C.

The pretty-printer is deliberately minimal: one line per event, colour
cues for category, compact rendering of common payload fields.  It is
a debugging and development aid — not a replacement for the GUI.

Colour output uses ANSI escapes only; pass ``--no-color`` or set
``NO_COLOR`` to disable, per the informal ``NO_COLOR`` convention.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterator, Optional, TextIO


# ---------------------------------------------------------------------------
# ANSI palette — intentionally small; category-only, no per-type colouring
# ---------------------------------------------------------------------------

_RESET = "\033[0m"
_DIM   = "\033[2m"
_BOLD  = "\033[1m"

_CATEGORY_COLOUR: Dict[str, str] = {
    "lifecycle":  "\033[38;5;250m",   # light grey
    "perception": "\033[38;5;110m",   # soft blue
    "belief":     "\033[38;5;151m",   # soft green
    "intention":  "\033[38;5;180m",   # warm tan
    "action":     "\033[38;5;222m",   # amber
    "anomaly":    "\033[38;5;203m",   # red-orange
    "oracle":     "\033[38;5;141m",   # soft violet
    "meta":       "\033[38;5;245m",   # neutral grey
}

_CATEGORY_OF: Dict[str, str] = {
    # lifecycle
    "EpisodeBegin":  "lifecycle", "EpisodeEnd": "lifecycle",
    "LevelChanged":  "lifecycle", "Reset":      "lifecycle",
    "StepBegin":     "lifecycle", "StepEnd":    "lifecycle",
    # perception
    "ObservationIngested": "perception", "EventEmitted": "perception",
    # belief
    "HypothesisAdded":           "belief", "HypothesisCredenceUpdated": "belief",
    "HypothesisSpecialised":     "belief", "HypothesisRetired":         "belief",
    # intention
    "GoalAdded":         "intention", "GoalDerived":       "intention",
    "GoalStatusChanged": "intention", "ActiveGoalChanged": "intention",
    "ConflictDetected":  "intention",
    # action
    "PlanComputed":    "action", "PlanInvalidated": "action",
    "PlanExhausted":   "action", "ActionSelected":  "action",
    "ActionExecuted":  "action", "ExploreFallback": "action",
    # anomaly
    "SurpriseEventRaised":   "anomaly",
    "FutilePatternDetected": "anomaly",
    "MinerFinding":          "anomaly",
    # oracle
    "ObserverQueryFired":     "oracle", "ObserverAnswerReceived": "oracle",
    "MediatorQueryFired":     "oracle", "MediatorAnswerReceived": "oracle",
    # meta
    "OptionSynthesised":   "meta", "OptionUsed":         "meta",
    "PostMortemProduced":  "meta", "RuleLearned":        "meta",
    "RuleRetired":         "meta", "Heartbeat":          "meta",
    "LogMessage":          "meta", "GapMarker":          "meta",
}


def _colour_for(event_type: str, enabled: bool) -> str:
    if not enabled:
        return ""
    cat = _CATEGORY_OF.get(event_type, "meta")
    return _CATEGORY_COLOUR.get(cat, "")


# ---------------------------------------------------------------------------
# One-line summary per event type
# ---------------------------------------------------------------------------


def _summarise_payload(event_type: str, payload: Dict[str, object]) -> str:
    """Return a short one-line summary of the payload contents."""
    g = payload.get
    if event_type == "EpisodeBegin":
        return f"{g('adapter_kind')} mode={g('operating_mode')}"
    if event_type == "EpisodeEnd":
        return f"{g('final_status')} in {g('total_steps')} steps"
    if event_type == "StepEnd":
        return f"action={g('action_kind') or '-'} plan_t={g('planner_latency_ms'):.1f}ms"
    if event_type == "HypothesisAdded":
        return f"{g('claim_type')} scope={g('scope_kind')} c={g('initial_credence'):.2f}"
    if event_type == "HypothesisCredenceUpdated":
        old, new = g("old_credence"), g("new_credence")
        return f"{old:.2f} -> {new:.2f} ({g('reason')})"
    if event_type == "HypothesisRetired":
        return f"{g('reason')}"
    if event_type == "GoalAdded":
        return f"{g('root_node_type')} p={g('priority'):.2f} {g('condition_summary')}"
    if event_type == "GoalStatusChanged":
        return f"{g('old_status')} -> {g('new_status')}"
    if event_type == "ActiveGoalChanged":
        return f"{g('old_goal_id') or '-'} -> {g('new_goal_id') or '-'}"
    if event_type == "PlanComputed":
        return f"goal={g('goal_id')} steps={g('step_count')} head={g('head_action') or '-'}"
    if event_type == "ActionSelected":
        return f"{g('action_kind')} ({g('source')})"
    if event_type == "ActionExecuted":
        ok = "ok" if g("success") else "fail"
        return f"{g('action_kind')} {ok} {g('duration_ms'):.1f}ms"
    if event_type == "SurpriseEventRaised":
        return f"{g('surprise_kind')} entity={g('entity_id') or '-'}"
    if event_type == "ObserverQueryFired" or event_type == "MediatorQueryFired":
        return f"{g('question_type')} qid={g('query_id')}"
    if event_type == "ObserverAnswerReceived" or event_type == "MediatorAnswerReceived":
        cache = "cache" if g("cache_hit") else "live"
        return f"{g('parsed_kind')} {cache} {g('latency_ms'):.0f}ms"
    if event_type == "LogMessage":
        return f"[{g('level')}] {g('logger')}: {g('message')}"
    if event_type == "Heartbeat":
        return f"{g('step_rate_hz'):.1f}Hz up={g('uptime_s'):.0f}s"
    # Fallback — compact JSON of the payload.  Truncate long values.
    s = json.dumps(payload, separators=(",", ":"))
    return s if len(s) <= 80 else s[:77] + "..."


# ---------------------------------------------------------------------------
# Line formatter
# ---------------------------------------------------------------------------


def format_line(env: Dict[str, object], colour: bool) -> str:
    event_type = str(env.get("type", "?"))
    c = _colour_for(event_type, colour)
    r = _RESET if colour else ""
    dim = _DIM if colour else ""
    bold = _BOLD if colour else ""

    ts_ms = float(env.get("ts", 0.0))
    step = env.get("step")
    ep = env.get("episode")
    subject = env.get("subject")

    prefix_parts = []
    prefix_parts.append(f"{dim}{ts_ms:10.1f}ms{r}")
    if ep is not None:
        prefix_parts.append(f"{dim}ep={ep}{r}")
    prefix_parts.append(f"{dim}s={step if step is not None else '-'}{r}")
    prefix = " ".join(prefix_parts)

    subj = f" {dim}[{subject}]{r}" if subject else ""
    payload = env.get("payload") or {}
    if isinstance(payload, dict):
        summary = _summarise_payload(event_type, payload)
    else:
        summary = str(payload)

    return f"{prefix}  {c}{bold}{event_type:<28}{r}{subj} {summary}"


# ---------------------------------------------------------------------------
# Follow loop
# ---------------------------------------------------------------------------


def _iter_file(fh: TextIO, follow: bool, poll_interval: float) -> Iterator[str]:
    """Yield full lines from ``fh``; if ``follow`` is set, block for more."""
    buf = ""
    while True:
        chunk = fh.readline()
        if chunk:
            buf += chunk
            if buf.endswith("\n"):
                yield buf.rstrip("\n")
                buf = ""
            continue
        if not follow:
            if buf:
                yield buf
            return
        time.sleep(poll_interval)


def _run(path: Path, follow: bool, colour: bool, poll: float, out: TextIO) -> int:
    if not path.exists() and not follow:
        print(f"cos-tail: file not found: {path}", file=sys.stderr)
        return 1
    # In follow mode we keep trying until the file appears
    while follow and not path.exists():
        time.sleep(poll)
    with path.open("r", encoding="utf-8") as fh:
        for raw in _iter_file(fh, follow=follow, poll_interval=poll):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            out.write(format_line(obj, colour))
            out.write("\n")
            out.flush()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cos-tail",
        description="Pretty-print a arc-agi-3 telemetry NDJSON file.",
    )
    parser.add_argument("path", type=Path, help="Path to the NDJSON file.")
    parser.add_argument(
        "-f", "--follow", action="store_true",
        help="Follow the file like `tail -f` — exit on ^C.",
    )
    parser.add_argument(
        "--no-color", dest="colour", action="store_false",
        help="Disable ANSI colour output (also honours NO_COLOR env var).",
    )
    parser.add_argument(
        "--poll", type=float, default=0.25,
        help="Poll interval in seconds when following (default 0.25).",
    )
    args = parser.parse_args(argv)

    colour = args.colour and "NO_COLOR" not in os.environ and sys.stdout.isatty()

    try:
        return _run(args.path, args.follow, colour, args.poll, sys.stdout)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
