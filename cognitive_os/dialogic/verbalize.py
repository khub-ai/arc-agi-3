"""Verbalize — turn structured engine objects into prose.

Single-dispatch on payload type.  Default templates ship for
:class:`AuditReport` and :class:`AuditResult`.  Adapters extend the
registry via :func:`register_template`.

See ``docs/SPEC_dialogic_component.md`` §"verbalize".

Mode parameter:
  ``"brief"``    — single-line outputs.
  ``"default"``  — headline + bullet details.
  ``"verbose"``  — adds the full ``metrics`` dict per check.
"""
from __future__ import annotations

from functools import singledispatch
from typing import Any, Callable, Dict, Type

from ..audit.check import AuditResult, Severity
from ..audit.report import AuditReport


_DEFAULT_MODES = ("brief", "default", "verbose")


# Template registry: payload type -> render function.  We use a
# manual registry rather than functools.singledispatch's because
# templates also accept a ``mode`` keyword and we want adapters to
# overwrite default templates without monkey-patching.
_TEMPLATES: Dict[Type, Callable[..., str]] = {}


def register_template(payload_type: Type, template: Callable[..., str]) -> None:
    """Register a prose template for ``payload_type``.  Overwrites
    any existing registration so adapters can specialise the output
    for their domain (e.g. ARC-specific audit results)."""
    _TEMPLATES[payload_type] = template


def verbalize(payload: Any, *, mode: str = "default") -> str:
    """Render any structured engine object as prose.

    Dispatches on the payload's type, walking the MRO so subclasses
    inherit their parent's template by default.  Falls back to
    ``repr(payload)`` when no template is registered — a defensive
    surface that lets the caller see what type lacked a template
    rather than silently producing empty output.
    """
    if mode not in _DEFAULT_MODES:
        raise ValueError(
            f"unknown verbalize mode {mode!r}; "
            f"expected one of {_DEFAULT_MODES}"
        )
    for cls in type(payload).__mro__:
        if cls in _TEMPLATES:
            return _TEMPLATES[cls](payload, mode=mode)
    return f"<no template for {type(payload).__name__}: {payload!r}>"


# ---------------------------------------------------------------------------
# Default templates
# ---------------------------------------------------------------------------


def _audit_report_template(report: AuditReport, *, mode: str = "default") -> str:
    """Render an entire audit report — header + per-result lines."""
    head = (
        f"AUDIT REPORT @ {report.timestamp.strftime('%Y-%m-%d %H:%M:%SZ')}\n"
        f"Aggregate severity: {report.aggregate_severity.name}"
    )
    if not report.results:
        return head + "\n(no checks ran)"
    bar = "=" * 67
    lines = [bar, head, bar]
    for r in report.results:
        lines.append(verbalize(r, mode=mode))
    return "\n".join(lines)


def _audit_result_template(result: AuditResult, *, mode: str = "default") -> str:
    """Render a single audit-result line.

    All formats: ``[SEV] check_name — headline``.
    Default adds indented details + fix hint.
    Verbose adds metrics.
    """
    sev_tag = f"[{result.severity.name}]"
    head = f"{sev_tag:<7} {result.check_name} — {result.headline}"
    if mode == "brief":
        return head
    parts = [head]
    for d in result.details:
        parts.append(f"        {d}")
    if result.fix_hint and result.severity != Severity.OK:
        parts.append(f"        Fix: {result.fix_hint}")
    if mode == "verbose" and result.metrics:
        parts.append("        Metrics:")
        for k, v in result.metrics.items():
            parts.append(f"          {k}: {v}")
    return "\n".join(parts)


# Register the foundational templates at module import.  Adapter
# modules can call register_template(...) to add or override.
register_template(AuditReport, _audit_report_template)
register_template(AuditResult, _audit_result_template)
