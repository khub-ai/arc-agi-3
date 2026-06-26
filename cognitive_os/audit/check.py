"""Audit-check protocol + result types.

See ``docs/SPEC_pre_run_audit.md`` §"AuditCheck protocol".

Implementations are stateless pure functions over a frozen
``WorldState`` snapshot plus an optional ``kb`` mapping.  Side
effects are forbidden — running the same check twice on the same
inputs must produce the same ``AuditResult``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


class Severity(Enum):
    """Audit-result severity tier.  Aggregate severity of a report
    is the maximum across its results, with FAIL > WARN > INFO > OK.

    Ordering operators are defined explicitly so callers can use
    ``max()``, ``sorted()``, and direct ``>=`` comparisons against
    other Severities.  Comparisons with non-Severity objects fall
    through to ``NotImplemented`` (Python's default).
    """
    OK   = 0
    INFO = 1
    WARN = 2
    FAIL = 3

    def __lt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value <  other.value

    def __le__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value <= other.value

    def __gt__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value >  other.value

    def __ge__(self, other: "Severity") -> bool:
        if not isinstance(other, Severity):
            return NotImplemented
        return self.value >= other.value


@dataclass
class AuditResult:
    """Outcome of a single audit check.

    Attributes
    ----------
    check_name
        Stable id of the check that produced this result
        (e.g. ``"goal_forest_coverage"``).  Used by the dialogic
        verbaliser to look up the check's prose template.
    severity
        OK / INFO / WARN / FAIL.  The runner aggregates the most
        severe across all results.
    headline
        One-line human-readable summary.  This is the primary
        prose surface — templates layer typography around it.
    details
        Optional list of structured findings, each one bullet-line.
        Empty list when the headline is sufficient.
    fix_hint
        Optional one-line suggestion for what to change to clear
        this result.  Only populated when severity > OK.
    metrics
        Free-form dict of numeric or structured facts the check
        produced.  Verbaliser may reference these for verbose
        rendering.  Keys are check-scoped; no cross-check schema.
    """
    check_name:  str
    severity:    Severity
    headline:    str
    details:     list[str]               = field(default_factory=list)
    fix_hint:    Optional[str]           = None
    metrics:     dict[str, Any]          = field(default_factory=dict)


@runtime_checkable
class AuditCheck(Protocol):
    """Protocol every audit check implements.

    Implementations expose a stable ``name`` (used for verbaliser
    template lookup), a one-line ``description``, and a ``run``
    method that returns an :class:`AuditResult`.

    Stateless: instances should hold no run-specific state — the
    runner may reuse a single instance across multiple audits.
    """
    name:        str
    description: str

    def run(
        self,
        ws,                                          # WorldState
        kb:            Optional[Mapping[str, Any]]   = None,
        adapter_hooks: Optional[Mapping[str, Any]]   = None,
    ) -> AuditResult:
        ...
