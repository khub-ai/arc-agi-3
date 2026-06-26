"""Validity-scope primitives for persisted observations.

Implements the engine surface specified in
``docs/SPEC_validity_scope.md``: every persisted observation carries
a :class:`ValidityScope` stamp; at scope boundaries (trial, level,
world-version bump, post-respawn) records whose stamp no longer
matches the current scope are demoted from ``ENFORCING`` to
``HINT``.  Downstream consumers (BFS, selector, plan-search,
plan-gate) read authoritative records only; cold-start priors and
observation pipelines may read hints.

This module is engine-side and domain-agnostic.  Adapters populate
:class:`ValidityScope` (they know each observation's context); the
engine consumes scopes via the two-tier read API and never inspects
scope fields directly.

Phase 1 lands the primitives only.  Adapter-side write sites are
migrated one observation class at a time (Phase 2 in the spec);
no call sites use this module yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

from .types import WorldState


_STORE_KEY  = "_validity_scope_store"
_SCOPE_KEY  = "_current_validity_scope"


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidityScope:
    """Context in which a persisted observation was committed.

    Four required axes (trial, level, world-version, committed-at);
    one optional axis (life_id) reserved for the per-life scope future
    axis described in :doc:`SPEC_validity_scope.md` "Future scope axes".

    The engine uses these fields opaquely — equality on the relevant
    axis is the only operation performed.  Adapters choose how to
    populate them (e.g. ``trial_id`` may be the ARC GUID; ``level_key``
    may be ``str(lc)``; ``world_version`` is a monotone counter the
    adapter bumps on environment changes per the spec's "What counts
    as a world_version bump" enumeration).
    """
    trial_id:        str
    level_key:       str
    world_version:   int
    committed_at:    int
    life_id:         Optional[int] = None


class RecordStatus(Enum):
    """Two-tier authority for a persisted record.

    ``ENFORCING`` records appear in :func:`read_authoritative` and
    drive BFS / selector / plan-search / plan-gate.  ``HINT`` records
    appear only in :func:`read_hint` and are advisory; consumers may
    use them for cold-start priors but must not enforce them.
    """
    ENFORCING = "enforcing"
    HINT      = "hint"


class BoundaryKind(Enum):
    """Which axis of the current scope governs demotion at this boundary.

    The engine maps each kind to a per-axis match check:

    * ``TRIAL``         — demote records whose ``trial_id`` differs.
    * ``LEVEL``         — demote records whose ``level_key`` differs.
    * ``WORLD_VERSION`` — demote records whose ``world_version`` is
                          strictly less than the current scope's.
    * ``RESPAWN``       — demote records whose ``life_id`` differs
                          AND is not None (life-agnostic records are
                          unaffected by respawn).
    """
    TRIAL          = "trial"
    LEVEL          = "level"
    WORLD_VERSION  = "world_version"
    RESPAWN        = "respawn"


@dataclass(frozen=True)
class ScopedRecord:
    """One persisted observation: ``value`` plus its commit-time scope
    stamp and current authority status.

    Records start as ``ENFORCING`` when written via :func:`persist`;
    :func:`validate_at_scope_boundary` may flip status to ``HINT``;
    re-confirmation in the new scope re-promotes (a fresh
    :func:`persist` overwrites with current scope and ENFORCING
    status when the caller supplies a ``replace_key``).
    """
    value:   Any
    scope:   ValidityScope
    status:  RecordStatus = RecordStatus.ENFORCING


# ---------------------------------------------------------------------------
# Store accessors
# ---------------------------------------------------------------------------


def _store(ws: WorldState) -> Dict[str, List[ScopedRecord]]:
    s = ws.agent.get(_STORE_KEY)
    if s is None:
        s = {}
        ws.agent[_STORE_KEY] = s
    return s


def current_scope(ws: WorldState) -> Optional[ValidityScope]:
    """Return the ambient :class:`ValidityScope` the adapter set, or
    ``None`` if no scope has been declared yet.

    The current scope is what every fresh :func:`persist` stamps on
    its record by default, and what :func:`validate_at_scope_boundary`
    matches against.
    """
    return ws.agent.get(_SCOPE_KEY)


def set_current_scope(ws: WorldState, scope: ValidityScope) -> None:
    """Adapter declares the current ambient scope.

    Typically called once at trial start, again on every level entry,
    and bumped (with a new ``world_version``) on environment changes
    that satisfy the spec's bump rules.  Setting a new scope does NOT
    automatically run boundary validation — the adapter calls
    :func:`validate_at_scope_boundary` explicitly so the caller chooses
    which axis is the relevant one.
    """
    ws.agent[_SCOPE_KEY] = scope


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def persist(
    ws:           WorldState,
    store_key:    str,
    value:        Any,
    *,
    scope:        Optional[ValidityScope] = None,
    replace_key:  Optional[Callable[[Any], Any]] = None,
) -> ScopedRecord:
    """Append a scoped record at ``store_key``.

    ``scope`` defaults to :func:`current_scope`; supplying it
    explicitly is an integration test escape hatch (or for replaying
    historical observations whose scope predates the current ambient).

    ``replace_key``, if supplied, is a function the engine calls on
    each existing record's value AND on ``value``; existing records
    whose key matches the new value's key are removed before append.
    Use this for upserts (re-observing a refuel pad at the same cell
    refreshes its scope to current rather than appending duplicates).
    Without ``replace_key`` the call is a pure append.

    The new record's status is always ``ENFORCING``.  Re-persisting
    a previously-demoted value via ``replace_key`` therefore
    re-promotes it in one call.

    Raises :class:`ValueError` if no scope is supplied and none is
    set as ambient — the engine refuses to stamp a record with an
    unknown scope rather than silently making one up.
    """
    if scope is None:
        scope = current_scope(ws)
        if scope is None:
            raise ValueError(
                "persist() called with no scope and no ambient scope set; "
                "call set_current_scope(ws, ...) first"
            )
    s = _store(ws)
    bucket = s.setdefault(store_key, [])
    if replace_key is not None:
        new_key = replace_key(value)
        bucket[:] = [r for r in bucket if replace_key(r.value) != new_key]
    rec = ScopedRecord(value=value, scope=scope, status=RecordStatus.ENFORCING)
    bucket.append(rec)
    return rec


# ---------------------------------------------------------------------------
# Read (two-tier API)
# ---------------------------------------------------------------------------


def read_authoritative(ws: WorldState, store_key: str) -> List[Any]:
    """Return the values of every ``ENFORCING`` record at ``store_key``,
    in insertion order.  Empty list if the key is absent.

    Downstream consumers that drive behaviour (BFS reachability, the
    selector's affordability check, plan-search's primary set,
    plan-gate's regression check) read through this surface.
    """
    bucket = _store(ws).get(store_key, ())
    return [r.value for r in bucket if r.status is RecordStatus.ENFORCING]


def read_hint(
    ws:        WorldState,
    store_key: str,
) -> List[Tuple[Any, RecordStatus]]:
    """Return ``(value, status)`` for every record at ``store_key``,
    enforcing and hint alike.  Empty list if the key is absent.

    Cold-start priors and observation-pipeline bootstraps consume
    through this surface so hint-tier observations remain visible
    without driving enforcement.
    """
    bucket = _store(ws).get(store_key, ())
    return [(r.value, r.status) for r in bucket]


def list_demoted(ws: WorldState, store_key: str) -> List[Any]:
    """Return the values of every ``HINT`` record at ``store_key``.

    Convenience for telemetry and migration verification ("show me
    what got demoted at the last boundary so I can verify the rule
    fired correctly").
    """
    bucket = _store(ws).get(store_key, ())
    return [r.value for r in bucket if r.status is RecordStatus.HINT]


# ---------------------------------------------------------------------------
# Boundary validation
# ---------------------------------------------------------------------------


def validate_at_scope_boundary(
    ws:             WorldState,
    current:        ValidityScope,
    *,
    boundary_kind:  BoundaryKind,
) -> List[Tuple[str, ScopedRecord]]:
    """Walk every persisted record; demote those whose stamp no longer
    matches ``current`` along the axis the boundary kind selects.

    Returns ``(store_key, demoted_record)`` tuples for each record
    that flipped from ``ENFORCING`` to ``HINT``.  Records already
    demoted are left alone (no double-demote).  The return value is
    intended for telemetry, integration with plan-stack invalidation
    (per :doc:`SPEC_goal_plan_search.md` "Plan-stack invalidation on
    demotion"), and migration verification.

    The current scope is also stored as the new ambient via
    :func:`set_current_scope`, so callers don't need a separate call
    for the common case where the boundary IS the scope transition.

    The relevant-axis match rule per :class:`BoundaryKind`:

    * ``TRIAL``         — match on ``trial_id``.
    * ``LEVEL``         — match on ``level_key``.
    * ``WORLD_VERSION`` — match on ``world_version >= current``
                          (older versions demote; same or newer pass).
    * ``RESPAWN``       — match on ``life_id`` (only when the record's
                          life_id is not None — life-agnostic records
                          carry through respawn unaffected).

    The match rules are intentionally conservative (only the relevant
    axis demotes); broader scope changes typically arrive as multiple
    sequential boundary calls (a level entry triggers WORLD_VERSION
    AND LEVEL boundaries, both run by the adapter).
    """
    demoted: List[Tuple[str, ScopedRecord]] = []
    s = _store(ws)
    for store_key, bucket in s.items():
        new_bucket: List[ScopedRecord] = []
        for r in bucket:
            if r.status is RecordStatus.ENFORCING and _should_demote(r.scope, current, boundary_kind):
                d = ScopedRecord(value=r.value, scope=r.scope, status=RecordStatus.HINT)
                new_bucket.append(d)
                demoted.append((store_key, d))
            else:
                new_bucket.append(r)
        bucket[:] = new_bucket
    set_current_scope(ws, current)
    return demoted


def _should_demote(
    record_scope:  ValidityScope,
    current:       ValidityScope,
    boundary:      BoundaryKind,
) -> bool:
    if boundary is BoundaryKind.TRIAL:
        return record_scope.trial_id != current.trial_id
    if boundary is BoundaryKind.LEVEL:
        return record_scope.level_key != current.level_key
    if boundary is BoundaryKind.WORLD_VERSION:
        return record_scope.world_version < current.world_version
    if boundary is BoundaryKind.RESPAWN:
        if record_scope.life_id is None:
            return False
        return record_scope.life_id != current.life_id
    return False


# ---------------------------------------------------------------------------
# Convenience: bulk operations for tests and migration tooling
# ---------------------------------------------------------------------------


def all_records(ws: WorldState) -> Dict[str, Tuple[ScopedRecord, ...]]:
    """Snapshot every (store_key → records) pair.

    Returns tuples (immutable) so callers can iterate without
    accidentally mutating the live store.  Test and migration
    tooling hook here; production consumers should use the two-tier
    read API above.
    """
    return {k: tuple(v) for k, v in _store(ws).items()}


def clear(ws: WorldState, store_key: Optional[str] = None) -> None:
    """Remove all records at ``store_key`` (or every record if
    ``store_key`` is ``None``).

    Intended for test fixtures and end-of-trial cleanup; production
    code should use boundary validation rather than bulk-clear, so
    that demotion-as-hint semantics are preserved.
    """
    s = _store(ws)
    if store_key is None:
        s.clear()
    else:
        s.pop(store_key, None)
