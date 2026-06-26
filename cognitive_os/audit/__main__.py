"""CLI entry point for the pre-run capability audit.

Usage:
    python -m cognitive_os.audit --kb <path-to-kb.json> [--level <N>]
                                  [--report-format text|json]
                                  [--adapter-hooks-module <module>]

The CLI loads a persisted-KB JSON, constructs a minimal WorldState
fixture (no env, no replay, no Oracle), runs the default check
inventory plus any adapter-extra checks specified via
``--adapter-hooks-module``, and prints the report.

See ``docs/SPEC_pre_run_audit.md`` §"CLI" for the design.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from .runner import run_audit
from .checks import DEFAULT_CHECKS


def _load_kb(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"KB not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _build_minimal_ws(kb: dict, level: "int | None"):
    """Construct a minimal WorldState for audit purposes.

    The audit checks read goal_forest and ws.agent — we populate
    those from the KB in the most defensible way possible without
    replaying the env.  Adapters that need a richer fixture should
    pass an ``adapter_hooks_module`` whose ``build_ws(kb, level)``
    returns the full session-state WorldState.
    """
    from cognitive_os.types import WorldState
    ws = WorldState()
    if level is not None:
        ws.agent.setdefault("_arc_state", {})["levels_completed"] = int(level)
    return ws


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m cognitive_os.audit")
    ap.add_argument("--kb", required=True,
                    help="Path to persisted-KB JSON (e.g. <game>_runtime.json)")
    ap.add_argument("--level", type=int, default=None,
                    help="Current level index (used by trusted_win_conditions check)")
    ap.add_argument("--report-format", choices=("text", "json"), default="text",
                    help="Output format (default: text)")
    ap.add_argument("--adapter-hooks-module", default=None,
                    help=("Optional dotted module name exposing extra "
                          "AuditCheck instances under ADAPTER_CHECKS, plus "
                          "an optional build_ws(kb, level) for richer "
                          "fixtures."))
    args = ap.parse_args(argv)

    kb = _load_kb(Path(args.kb))

    checks = list(DEFAULT_CHECKS)
    ws_builder = _build_minimal_ws
    if args.adapter_hooks_module:
        mod = importlib.import_module(args.adapter_hooks_module)
        extra = getattr(mod, "ADAPTER_CHECKS", None)
        if extra:
            checks.extend(extra)
        custom_builder = getattr(mod, "build_ws", None)
        if callable(custom_builder):
            ws_builder = custom_builder

    ws = ws_builder(kb, args.level)
    report = run_audit(ws, kb=kb, checks=checks)

    if args.report_format == "json":
        print(json.dumps(report.to_dict(), indent=2, default=str))
    else:
        # Use the dialogic verbaliser for text mode.
        from cognitive_os.dialogic import verbalize
        print(verbalize(report))

    sys.exit(2 if report.has_failures() else 0)


if __name__ == "__main__":
    main()
