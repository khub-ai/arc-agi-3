"""Pre-run capability audit — see ``docs/SPEC_pre_run_audit.md``.

The audit checks the system's structural readiness to form a winning
plan for a game *without* running the env or calling Oracle.  It's
deterministic, side-effect-free, and serves as a cheap gating tool
before expensive end-to-end runs.

Public API:

    from cognitive_os.audit import (
        AuditCheck, AuditResult, AuditReport, Severity, run_audit,
    )

Example:

    from cognitive_os.audit import run_audit
    from cognitive_os.audit.checks import DEFAULT_CHECKS

    report = run_audit(ws, kb=kb, checks=DEFAULT_CHECKS)
    if report.has_failures():
        for r in report.by_severity(Severity.FAIL):
            print(r.headline)
"""
from __future__ import annotations

from .check import AuditCheck, AuditResult, Severity
from .report import AuditReport
from .runner import run_audit

__all__ = [
    "AuditCheck",
    "AuditResult",
    "AuditReport",
    "Severity",
    "run_audit",
]
