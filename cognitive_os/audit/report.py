"""AuditReport — collection of AuditResults with aggregate semantics.

See ``docs/SPEC_pre_run_audit.md`` §"AuditReport".
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List

from .check import AuditResult, Severity


@dataclass
class AuditReport:
    """Outcome of running a set of audit checks.

    The aggregate severity is the maximum across all child results;
    a single FAIL therefore makes the whole audit FAIL.  Helpers
    select results by severity for quick lookups.
    """
    results:    List[AuditResult]
    timestamp:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def aggregate_severity(self) -> Severity:
        if not self.results:
            return Severity.OK
        return max(r.severity for r in self.results)

    def by_severity(self, sev: Severity) -> List[AuditResult]:
        """Return results matching the given severity exactly."""
        return [r for r in self.results if r.severity == sev]

    def at_or_above(self, sev: Severity) -> List[AuditResult]:
        """Return results whose severity is sev or stricter."""
        return [r for r in self.results if r.severity.value >= sev.value]

    def has_failures(self) -> bool:
        return any(r.severity == Severity.FAIL for r in self.results)

    def to_dict(self) -> dict:
        """Serialise the report to a JSON-friendly dict.  Used by
        the CLI when --report-format=json."""
        return {
            "timestamp":          self.timestamp.isoformat(),
            "aggregate_severity": self.aggregate_severity.name,
            "results": [
                {
                    "check_name": r.check_name,
                    "severity":   r.severity.name,
                    "headline":   r.headline,
                    "details":    list(r.details),
                    "fix_hint":   r.fix_hint,
                    "metrics":    dict(r.metrics),
                }
                for r in self.results
            ],
        }
