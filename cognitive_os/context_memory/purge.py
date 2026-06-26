"""Programmatic, audited, scoped purge for context_memory claims.

Step 8 of the context_memory migration.  The markdown layout from
step 6 already gives operators a working "delete one file"
purge -- this module adds the programmatic variant that the spec
calls out:

* **Scoped.**  Restrict to one scope (universal / game / level),
  or to one game / level identifier.
* **Filtered.**  Restrict by ``claim_type``, or by an arbitrary
  callable that inspects the parsed frontmatter dict and returns
  ``True`` to delete that claim.
* **Audited.**  Every purge call appends one entry to
  ``kb_root/purge_log.jsonl`` recording the filter, the operator-
  supplied reason, the timestamp, and (for every deleted file) the
  scope label, relative path, and full frontmatter snapshot.  An
  operator who needs to undo a purge has every byte of the claim's
  prior state in that log.
* **Dry-run.**  ``dry_run=True`` returns what WOULD have been
  deleted without touching the filesystem.

Cross-scope reads/writes do not happen here -- this module only
walks the directories under ``kb_root`` that the
:func:`save_committed_hierarchy` / :func:`load_committed_hierarchy`
pair already owns.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from ..types import ScopeKind
from .markdown_io import (
    _scope_dir,
    claim_filename,
    parse_frontmatter,
    write_memory_index,
)


@dataclass
class PurgeRecord:
    """One claim deleted by a purge call.

    Attributes
    ----------
    path
        Path to the deleted file, relative to ``kb_root``.  POSIX-
        style separators so the log is portable across hosts.
    scope_label
        ``"universal"``, ``"game"``, or ``"level"`` -- which scope
        directory the claim lived in.
    frontmatter
        Full parsed frontmatter of the claim immediately before
        deletion.  Enough to reconstruct the file by hand if an
        operator needs to undo the purge.
    """
    path:        str
    scope_label: str
    frontmatter: Dict[str, Any]


@dataclass
class PurgeResult:
    """Outcome of one :func:`purge` call.

    Attributes
    ----------
    purged
        One :class:`PurgeRecord` per file deleted (or, in
        ``dry_run`` mode, per file that WOULD have been deleted).
    log_path
        Where the audit entry was appended.  ``None`` when
        ``dry_run`` is True (no log entry written).
    dry_run
        Echo of the caller's flag.  Inspecting ``purged`` is
        identical between real and dry-run modes; only
        ``log_path`` and the on-disk side effects differ.
    filter
        Snapshot of the filter the caller passed.  Mirrors the
        ``filter`` field of the audit-log entry so callers don't
        need to re-derive it.
    """
    purged:    List[PurgeRecord]
    log_path:  Optional[Path]
    dry_run:   bool
    filter:    Dict[str, Any] = field(default_factory=dict)


def _scope_dirs_for_purge(
    kb_root:  Path,
    *,
    game_id:  Optional[str],
    level_id: Optional[str],
    scope:    Optional[ScopeKind],
) -> List[Tuple[str, ScopeKind, Path]]:
    """Resolve which scope directories the purge call should walk.

    When ``scope`` is None, every applicable directory is included.
    When ``scope`` is set, only that one is included (and only if
    it has the identifiers it needs -- e.g. ``LEVEL`` without
    ``level_id`` yields an empty list, since there's no concrete
    directory to walk).
    """
    candidates: List[Tuple[str, ScopeKind]] = [
        ("universal", ScopeKind.GLOBAL),
        ("game",      ScopeKind.GAME),
        ("level",     ScopeKind.LEVEL),
    ]
    out: List[Tuple[str, ScopeKind, Path]] = []
    for label, kind in candidates:
        if scope is not None and kind != scope:
            continue
        scope_dir = _scope_dir(kb_root, kind,
                               game_id=game_id, level_id=level_id)
        if scope_dir is None or not scope_dir.is_dir():
            continue
        out.append((label, kind, scope_dir))
    return out


def _matches(record:     Mapping[str, Any],
             *,
             claim_type: Optional[str],
             match:      Optional[Callable[[Dict[str, Any]], bool]]) -> bool:
    """AND-compose the filter predicates over one record."""
    if claim_type is not None and record.get("claim_type") != claim_type:
        return False
    if match is not None and not match(dict(record)):
        return False
    return True


def _surviving_records(claims_dir: Path) -> Tuple[List[Dict[str, Any]],
                                                  List[str]]:
    """Read every remaining claim file in ``claims_dir`` and return
    ``(records, filenames)`` suitable for :func:`write_memory_index`.
    """
    records:   List[Dict[str, Any]] = []
    filenames: List[str]            = []
    for md_path in sorted(claims_dir.glob("*.md")):
        try:
            text = md_path.read_text(encoding="utf-8")
        except OSError:
            continue
        rec = parse_frontmatter(text)
        if not rec:
            continue
        records.append(rec)
        # Use the file's actual name -- hand-edited files might not
        # round-trip through claim_filename, and we want the index
        # to point at the file the operator actually has on disk.
        filenames.append(md_path.name)
    return records, filenames


def purge(
    kb_root:    Path,
    *,
    game_id:    Optional[str]                          = None,
    level_id:   Optional[str]                          = None,
    scope:      Optional[ScopeKind]                    = None,
    claim_type: Optional[str]                          = None,
    match:      Optional[Callable[[Dict[str, Any]], bool]] = None,
    reason:     str                                    = "",
    operator:   str                                    = "operator",
    dry_run:    bool                                   = False,
) -> PurgeResult:
    """Delete the claims in ``kb_root`` that match every supplied
    filter, with an audit log.

    Parameters
    ----------
    kb_root
        Filesystem root for the knowledge base.
    game_id, level_id
        Scope identifiers.  Required for the corresponding scope
        layers; missing identifiers cause that layer to be skipped
        rather than to error.
    scope
        When set, restrict the walk to one ``ScopeKind`` layer
        (``GLOBAL`` / ``GAME`` / ``LEVEL``).  When ``None``, walk
        every applicable layer.
    claim_type
        When set, only delete claims whose frontmatter
        ``claim_type`` field matches this string (e.g.
        ``"BitmapRoleClaim"``).
    match
        Optional predicate over the parsed frontmatter dict.  AND-
        composed with ``claim_type``.  Receives a fresh ``dict``
        copy so the caller can't accidentally mutate file state.
    reason
        Operator-supplied free-text reason; stored verbatim in the
        audit log.  Keep it concise (one line).
    operator
        Short identifier for the operator / tool that issued the
        purge.  Defaults to ``"operator"``; CLI wrappers will pass
        their own.
    dry_run
        When True, find matching files and populate ``purged``
        but DON'T delete and DON'T append to the log.  Useful for
        previewing a purge that would have wide blast radius.

    Returns
    -------
    PurgeResult
        ``purged`` is one record per file matched (deleted or, in
        dry-run mode, would-have-been-deleted).  ``log_path`` is
        the audit-log file when a real purge ran, ``None`` when
        ``dry_run`` was True or no claims matched.

    Notes
    -----
    * When no filters are provided the call is treated as a request
      to purge *every* claim in the resolved scope directories.
      This is intentional: an operator passing ``scope=ScopeKind.
      LEVEL`` + ``game_id="bp35"`` + ``level_id="3"`` and no other
      filters wants to wipe that level cleanly.  The audit log
      still records what was deleted.
    * ``MEMORY.md`` is regenerated after deletes from the files
      that remain on disk (not from any in-memory state), so a
      hand-edited claim that survived a purge keeps its hand-
      edited frontmatter in the regenerated index.
    """
    kb_root = Path(kb_root)
    if not kb_root.is_dir():
        return PurgeResult(purged=[], log_path=None, dry_run=dry_run,
                           filter=_filter_snapshot(
                               game_id, level_id, scope,
                               claim_type, match, reason, operator))

    purged:        List[PurgeRecord] = []
    touched_dirs:  List[Path]        = []

    for label, _kind, scope_dir in _scope_dirs_for_purge(
        kb_root, game_id=game_id, level_id=level_id, scope=scope,
    ):
        claims_dir = scope_dir / "claims"
        if not claims_dir.is_dir():
            continue
        matched_in_dir: List[Path] = []
        for md_path in sorted(claims_dir.glob("*.md")):
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            rec = parse_frontmatter(text)
            if not rec:
                continue
            if not _matches(rec, claim_type=claim_type, match=match):
                continue
            matched_in_dir.append(md_path)
            rel = md_path.relative_to(kb_root).as_posix()
            purged.append(PurgeRecord(
                path        = rel,
                scope_label = label,
                frontmatter = rec,
            ))
        if matched_in_dir and not dry_run:
            for p in matched_in_dir:
                try:
                    p.unlink()
                except OSError:
                    continue
            touched_dirs.append(scope_dir)

    if not dry_run and touched_dirs:
        for scope_dir in touched_dirs:
            records, filenames = _surviving_records(scope_dir / "claims")
            write_memory_index(scope_dir, records, filenames)

    log_path: Optional[Path] = None
    if not dry_run and purged:
        log_path = _append_audit_log(
            kb_root,
            reason=reason, operator=operator,
            game_id=game_id, level_id=level_id, scope=scope,
            claim_type=claim_type, match=match,
            purged=purged,
        )

    return PurgeResult(
        purged    = purged,
        log_path  = log_path,
        dry_run   = dry_run,
        filter    = _filter_snapshot(game_id, level_id, scope,
                                     claim_type, match, reason, operator),
    )


def _filter_snapshot(
    game_id:    Optional[str],
    level_id:   Optional[str],
    scope:      Optional[ScopeKind],
    claim_type: Optional[str],
    match:      Optional[Callable[..., bool]],
    reason:     str,
    operator:   str,
) -> Dict[str, Any]:
    """JSON-safe snapshot of the filter the caller passed in.

    Used both inside :class:`PurgeResult` and as the ``filter``
    field of the audit-log entry.  Callable predicates have no
    portable serialization, so we record their ``__name__`` /
    ``__qualname__`` rather than the bytecode -- enough for an
    operator reading the log later to find which predicate ran.
    """
    out: Dict[str, Any] = {
        "game_id":    game_id,
        "level_id":   level_id,
        "scope":      scope.name if scope is not None else None,
        "claim_type": claim_type,
        "match":      None,
        "reason":     reason,
        "operator":   operator,
    }
    if match is not None:
        out["match"] = (getattr(match, "__qualname__", None)
                        or getattr(match, "__name__", None)
                        or repr(match))
    return out


def _append_audit_log(
    kb_root:    Path,
    *,
    reason:     str,
    operator:   str,
    game_id:    Optional[str],
    level_id:   Optional[str],
    scope:      Optional[ScopeKind],
    claim_type: Optional[str],
    match:      Optional[Callable[..., bool]],
    purged:     List[PurgeRecord],
) -> Path:
    """Append one purge entry to ``kb_root/purge_log.jsonl``.

    The audit log is append-only.  Older entries are never
    rewritten, so a corrupted or hand-edited line never blocks
    future purges from logging.
    """
    log_path = kb_root / "purge_log.jsonl"
    entry: Dict[str, Any] = {
        "ts":      datetime.now(timezone.utc).isoformat(),
        "filter":  _filter_snapshot(game_id, level_id, scope,
                                    claim_type, match, reason, operator),
        "purged":  [
            {
                "path":        r.path,
                "scope_label": r.scope_label,
                "frontmatter": r.frontmatter,
            }
            for r in purged
        ],
    }
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, sort_keys=True))
        f.write("\n")
    return log_path
