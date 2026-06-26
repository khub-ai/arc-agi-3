"""Audit runner — execute a list of AuditChecks against a WorldState
and return an AuditReport.

See ``docs/SPEC_pre_run_audit.md`` §"Runner".

Exception isolation: a check that raises is caught and converted to
a FAIL result with the exception text in ``details``.  The audit
continues running the remaining checks.  This guarantees a single
buggy check can never abort the whole audit.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from .check import AuditCheck, AuditResult, Severity
from .report import AuditReport


def run_audit(
    ws,                                                # WorldState
    kb:             Optional[Mapping[str, Any]]      = None,
    checks:         Optional[Sequence[AuditCheck]]   = None,
    adapter_hooks:  Optional[Mapping[str, Any]]      = None,
) -> AuditReport:
    """Run all checks and return a structured report.

    Parameters
    ----------
    ws
        The :class:`WorldState` snapshot to audit.  No mutation —
        each check is forbidden from changing state.
    kb
        Optional persisted-KB dict.  Many checks read fields like
        ``learned_facts.trusted_*`` from here, so adapters that
        carry KB state out-of-band should pass it.
    checks
        Sequence of :class:`AuditCheck` instances to run.  When
        ``None``, falls back to the default check inventory exposed
        by :mod:`cognitive_os.audit.checks`.
    adapter_hooks
        Optional mapping of adapter-specific helpers a check may
        consult (e.g. ``{"cs": cell_system, "agent_fp": (...)}``).
        Entirely opt-in; checks that don't need them ignore the
        argument.

    Returns
    -------
    AuditReport
        Collected results plus aggregate severity.
    """
    if checks is None:
        # Local import to avoid loading the check inventory before
        # callers that pass an explicit list (e.g. tests with synthetic
        # checks) need it.
        from .checks import DEFAULT_CHECKS
        checks = DEFAULT_CHECKS
    results: list[AuditResult] = []
    for check in checks:
        try:
            r = check.run(ws=ws, kb=kb, adapter_hooks=adapter_hooks)
            if not isinstance(r, AuditResult):
                # Defensive: a check that returned the wrong type is
                # itself a FAIL — surface the contract violation so
                # the bug is visible rather than silent.
                results.append(AuditResult(
                    check_name = getattr(check, "name", repr(check)),
                    severity   = Severity.FAIL,
                    headline   = f"check returned {type(r).__name__}, not AuditResult",
                    fix_hint   = "fix the check implementation to return AuditResult",
                ))
                continue
            results.append(r)
        except Exception as exc:
            results.append(AuditResult(
                check_name = getattr(check, "name", repr(check)),
                severity   = Severity.FAIL,
                headline   = f"check raised {type(exc).__name__}",
                details    = [str(exc)[:300]],
                fix_hint   = "investigate the exception; checks must not raise",
            ))
    return AuditReport(results=results)
