"""Markdown-with-frontmatter persistence for context_memory claims.

Step 6 of the context_memory migration: replace the per-game
``claims.jsonl`` blob with a per-claim markdown file layout plus a
per-scope ``MEMORY.md`` index.  Goals:

* Hand-editable.  Operators can grep / open / fix / delete an
  individual claim file without touching others.
* Git-friendly.  Diffs are line-by-line legible.
* Convergent with Claude Code's memory pattern -- if another agent
  later inspects this directory, it sees the same shape it already
  understands.
* Backward-compatible during transition.  ``load_committed_markdown``
  falls back to ``claims.jsonl`` when no ``claims/`` directory
  exists, so a partial roll-out doesn't strand existing KBs.

Frontmatter format is a deliberately small YAML subset:

  * ``key: scalar``  -- string / int / float / bool
  * ``key: [a, b, c]``  -- flat list of scalars
  * no nested mappings, no multiline strings, no anchors

This is enough for everything the existing
``persistence._encode_claim`` whitelist produces, and stays trivial
to parse without a YAML dependency.

The serializer reuses ``persistence._encode_claim`` and
``persistence._reconstruct_and_propose`` so the type dispatch and
field schema stay in one place; this module is purely the on-disk
format.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from .. import hypothesis_store as _store
from ..persistence import _encode_claim, _reconstruct_and_propose
from ..types import ScopeKind, WorldState


# ---------------------------------------------------------------------------
# Frontmatter format
# ---------------------------------------------------------------------------


_FENCE = "---"
_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_-]+")


def _format_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_format_value(x) for x in v)
        return f"[{inner}]"
    s = str(v)
    # Quote when the value would be ambiguous as a bare scalar.
    if any(c in s for c in ":[]{}#,'\"") or s != s.strip():
        return '"' + s.replace('"', r'\"') + '"'
    return s


def _parse_value(s: str) -> Any:
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        # Naive split -- our values never contain commas; lists are
        # palettes / ints / strings only.
        return [_parse_value(part) for part in inner.split(",")]
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if s in ("null", "None", "~"):
        return None
    if (len(s) >= 2 and s[0] == s[-1] and s[0] in '"\''):
        return s[1:-1].replace(r'\"', '"')
    # Try int then float; otherwise bare string.
    try:
        if s.lstrip("-").isdigit():
            return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def format_frontmatter(d: Mapping[str, Any]) -> str:
    """Render a flat dict as a YAML-subset frontmatter block."""
    lines: List[str] = [_FENCE]
    for k, v in d.items():
        lines.append(f"{k}: {_format_value(v)}")
    lines.append(_FENCE)
    return "\n".join(lines)


def parse_frontmatter(text: str) -> Dict[str, Any]:
    """Parse a YAML-subset frontmatter block (the lines between two
    ``---`` fences).  Returns ``{}`` when no frontmatter is present."""
    lines = text.split("\n")
    # Locate opening and closing fences.
    start: Optional[int] = None
    end:   Optional[int] = None
    for i, line in enumerate(lines):
        stripped = line.rstrip()
        if stripped == _FENCE:
            if start is None:
                start = i
            else:
                end = i
                break
    if start is None or end is None:
        return {}
    out: Dict[str, Any] = {}
    for line in lines[start + 1:end]:
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" not in s:
            continue
        k, _, v = s.partition(":")
        out[k.strip()] = _parse_value(v)
    return out


# ---------------------------------------------------------------------------
# Claim filename + narrative body
# ---------------------------------------------------------------------------


def _safe_token(x: Any) -> str:
    s = str(x)
    s = _SAFE_TOKEN_RE.sub("_", s).strip("_")
    return s[:80] or "x"


def claim_filename(record: Mapping[str, Any]) -> str:
    """Deterministic filename for a claim record.

    Includes enough of the claim's identifying fields that two
    structurally-distinct claims of the same type never collide,
    while staying short enough to be human-readable in a git diff.
    """
    ct = record.get("claim_type") or "unknown"
    if ct == "BitmapRoleClaim":
        return f"bitmap_role__{_safe_token(record.get('bitmap_id'))}.md"
    if ct == "RegionPaletteClaim":
        pals = record.get("palettes") or []
        pal_tok = "_".join(str(p) for p in pals) or "empty"
        role = _safe_token(record.get("role") or "x")
        return f"region_palette__pal{pal_tok}__{role}.md"
    if ct == "ControlledActorClaim":
        return (f"controlled_actor__c{record.get('colour')}"
                f"_b{record.get('background')}.md")
    if ct == "CausalClaim:ActionJustTaken:RegionMotion":
        return (f"causal_action_region_motion__"
                f"{_safe_token(record.get('action_id'))}__"
                f"c{record.get('colour')}_b{record.get('background')}_"
                f"d{record.get('dr_sign')}{record.get('dc_sign')}.md")
    # Fallback: hash the record so the filename is still stable.
    import hashlib, json
    h = hashlib.sha256(
        json.dumps(record, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return f"{_safe_token(ct).lower()}__{h}.md"


def _narrative_body(record: Mapping[str, Any]) -> str:
    """Generate a short prose body describing the claim.  Operators
    can edit this freely; the parser ignores it."""
    ct = record.get("claim_type") or "unknown"
    supports = int(record.get("supports") or 0)
    cred     = record.get("credence") or 0.0
    src      = record.get("source") or "unknown"
    if ct == "BitmapRoleClaim":
        head = (f"# {ct} — bitmap `{record.get('bitmap_id')}` plays "
                f"role `{record.get('role')}`")
    elif ct == "RegionPaletteClaim":
        head = (f"# {ct} — palette signature `{record.get('palettes')}` "
                f"plays role `{record.get('role')}`")
    elif ct == "ControlledActorClaim":
        head = (f"# {ct} — sprite colour `{record.get('colour')}` "
                f"on background `{record.get('background')}` is "
                f"the controllable agent")
    elif ct == "CausalClaim:ActionJustTaken:RegionMotion":
        head = (f"# {ct} — `{record.get('action_id')}` moves the "
                f"sprite (colour=`{record.get('colour')}`, bg="
                f"`{record.get('background')}`) by direction "
                f"`(dr={record.get('dr_sign')}, dc={record.get('dc_sign')})`")
    else:
        head = f"# {ct}"
    body = (
        f"{head}\n\n"
        f"Auto-generated by the context_memory bridge.  Supporting "
        f"evidence has accumulated across {supports} observation"
        f"{'s' if supports != 1 else ''}; credence is {cred:.3f}.  "
        f"Source: `{src}`.\n\n"
        f"If this claim is wrong, edit the relevant field in the "
        f"frontmatter above, or delete this file entirely.  The "
        f"next perception session will not silently re-propose an "
        f"identical claim unless its underlying signature is "
        f"genuinely observed again.\n"
    )
    return body


# ---------------------------------------------------------------------------
# MEMORY.md index
# ---------------------------------------------------------------------------


def _index_line(record: Mapping[str, Any], filename: str) -> str:
    ct = record.get("claim_type") or "unknown"
    cred = record.get("credence")
    cred_s = f"{cred:.2f}" if isinstance(cred, (int, float)) else "?"
    if ct == "BitmapRoleClaim":
        gist = f"bitmap `{record.get('bitmap_id')}` → `{record.get('role')}`"
    elif ct == "RegionPaletteClaim":
        gist = (f"palette `{record.get('palettes')}` → "
                f"`{record.get('role')}`")
    elif ct == "ControlledActorClaim":
        gist = (f"agent colour={record.get('colour')} "
                f"bg={record.get('background')}")
    elif ct == "CausalClaim:ActionJustTaken:RegionMotion":
        gist = (f"{record.get('action_id')} → motion "
                f"({record.get('dr_sign')}, {record.get('dc_sign')}) "
                f"on c{record.get('colour')}_b{record.get('background')}")
    else:
        gist = ct
    return f"- [{gist}](claims/{filename}) — cred {cred_s}"


def write_memory_index(kb_dir: Path,
                       records: List[Mapping[str, Any]],
                       filenames: List[str]) -> Path:
    """Write a MEMORY.md index file pointing at every claim file.

    One line per claim, ~150 chars max, short title + 1-line gist.
    Mirrors Claude Code's memory-index convention.
    """
    kb_dir = Path(kb_dir)
    path = kb_dir / "MEMORY.md"
    lines: List[str] = [
        f"# Memory index — {kb_dir.name}",
        "",
        f"{len(records)} claim{'s' if len(records) != 1 else ''}.  "
        f"Each line below points at one claim file under `claims/`.",
        "",
    ]
    for rec, fn in zip(records, filenames):
        lines.append(_index_line(rec, fn))
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def save_committed_markdown(
    ws:        WorldState,
    kb_dir:    Path,
    *,
    filter_fn: Optional[Any] = None,
) -> List[dict]:
    """Save every committed hypothesis as a per-claim markdown file
    under ``kb_dir/claims/``, then refresh ``kb_dir/MEMORY.md``.

    When ``filter_fn`` is given (callable taking a Hypothesis and
    returning bool), only hypotheses where ``filter_fn(h)`` is True
    are saved.  Used by ``save_committed_hierarchy`` to partition by
    scope; left None for the common single-scope case.

    Returns the list of encoded records that were written (same shape
    as ``persistence.save_committed_knowledge``).  Wrappers /
    callers can treat this as a drop-in replacement.
    """
    kb_dir = Path(kb_dir)
    claims_dir = kb_dir / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)

    records:   List[Dict[str, Any]] = []
    filenames: List[str]            = []
    for h in _store.committed(ws):
        if filter_fn is not None and not filter_fn(h):
            continue
        rec = _encode_claim(h)
        if rec is None:
            continue
        fn = claim_filename(rec)
        content = format_frontmatter(rec) + "\n\n" + _narrative_body(rec)
        (claims_dir / fn).write_text(content, encoding="utf-8")
        records.append(rec)
        filenames.append(fn)

    write_memory_index(kb_dir, records, filenames)
    return records


def load_committed_markdown(ws: WorldState,
                            kb_dir: Path,
                            *,
                            step: int = 0) -> List[str]:
    """Load every per-claim markdown file under ``kb_dir/claims/``.

    When ``kb_dir/claims/`` doesn't exist, falls back to reading
    ``claims.jsonl`` via the engine's existing loader so a partial
    roll-out doesn't strand existing KBs.

    Returns the list of hypothesis IDs registered (or merged via
    dedup) in ``ws``.
    """
    kb_dir = Path(kb_dir)
    claims_dir = kb_dir / "claims"
    if claims_dir.is_dir() and any(claims_dir.glob("*.md")):
        out_ids: List[str] = []
        for md_path in sorted(claims_dir.glob("*.md")):
            try:
                text = md_path.read_text(encoding="utf-8")
            except OSError:
                continue
            rec = parse_frontmatter(text)
            if not rec:
                continue
            hid = _reconstruct_and_propose(ws, rec, step=step)
            if hid is not None:
                out_ids.append(hid)
        return out_ids
    # Fallback to JSONL.  Imported lazily so the test fixture for
    # the markdown path can run without the legacy data present.
    from ..persistence import load_committed_knowledge
    return load_committed_knowledge(ws, kb_dir, step=step)


# ---------------------------------------------------------------------------
# Scope-hierarchy layout: universal / game / level
# ---------------------------------------------------------------------------
#
# Step 7 of the migration extends the per-game directory into a
# scope-hierarchy.  Claims live in the directory matching their
# Scope.kind:
#
#   ScopeKind.GLOBAL   -> kb_root/universal/
#   ScopeKind.GAME     -> kb_root/<game_id>/
#   ScopeKind.LEVEL    -> kb_root/<game_id>/levels/<level_id>/
#
# On load we walk ALL applicable layers (broadest first) so a game
# session gets universal priors + per-game knowledge + per-level
# specifics in one call.  Cross-scope leakage is impossible: each
# claim sits in exactly one directory determined by its own
# scope.kind.
#
# Family scope (kb_root/game_families/<family>/) is intentionally
# deferred -- ScopeKind has no FAMILY today; adding it is its own
# migration step.


def _scope_dir(kb_root:  Path,
               scope_kind: ScopeKind,
               *,
               game_id:  Optional[str] = None,
               level_id: Optional[str] = None) -> Optional[Path]:
    """Filesystem directory for the given scope.  Returns ``None``
    when the scope kind isn't supported in the hierarchy (e.g.
    EPISODE / LIFE / STEP -- those are ephemeral and don't persist).
    """
    kb_root = Path(kb_root)
    if scope_kind == ScopeKind.GLOBAL:
        return kb_root / "universal"
    if scope_kind == ScopeKind.GAME:
        if not game_id:
            return None
        return kb_root / game_id
    if scope_kind == ScopeKind.LEVEL:
        if not (game_id and level_id):
            return None
        return kb_root / game_id / "levels" / str(level_id)
    return None


def _applicable_load_dirs(kb_root:  Path,
                          *,
                          game_id:  Optional[str] = None,
                          level_id: Optional[str] = None) -> List[Path]:
    """Directories to walk on load, broadest scope first.

    The order matters when the SAME claim somehow exists in two
    scopes -- the broader one is registered first and the narrower
    one's proposal merges as supporting evidence rather than
    duplicating.  Hypothesis-store dedup makes this safe.
    """
    out: List[Path] = []
    universal = _scope_dir(kb_root, ScopeKind.GLOBAL)
    if universal is not None:
        out.append(universal)
    if game_id:
        game = _scope_dir(kb_root, ScopeKind.GAME, game_id=game_id)
        if game is not None:
            out.append(game)
        if level_id:
            level = _scope_dir(kb_root, ScopeKind.LEVEL,
                               game_id=game_id, level_id=level_id)
            if level is not None:
                out.append(level)
    return out


def save_committed_hierarchy(
    ws:       WorldState,
    kb_root:  Path,
    *,
    game_id:  Optional[str] = None,
    level_id: Optional[str] = None,
) -> Dict[str, List[dict]]:
    """Save committed hypotheses, partitioned by each claim's
    ``Scope.kind``, into the matching scope directories under
    ``kb_root``.

    A GLOBAL-scoped claim lands in ``kb_root/universal/``; a
    GAME-scoped claim lands in ``kb_root/<game_id>/``; a
    LEVEL-scoped claim lands in
    ``kb_root/<game_id>/levels/<level_id>/``.  Claims with scopes
    that don't map to a hierarchy layer (EPISODE / LIFE / STEP) are
    not persisted -- those are ephemeral by intent.

    Returns ``{scope_label: [records]}`` for the operator's
    inspection.  ``scope_label`` is one of ``"universal"``,
    ``"game"``, ``"level"``.
    """
    kb_root = Path(kb_root)
    out: Dict[str, List[dict]] = {}
    by_kind = [
        ("universal", ScopeKind.GLOBAL),
        ("game",      ScopeKind.GAME),
        ("level",     ScopeKind.LEVEL),
    ]
    for label, kind in by_kind:
        scope_dir = _scope_dir(kb_root, kind,
                               game_id=game_id, level_id=level_id)
        if scope_dir is None:
            continue
        def _filter(h, _kind=kind):
            return h.scope is not None and h.scope.kind == _kind
        records = save_committed_markdown(
            ws, scope_dir, filter_fn=_filter)
        out[label] = records
    return out


def load_committed_hierarchy(
    ws:       WorldState,
    kb_root:  Path,
    *,
    game_id:  Optional[str] = None,
    level_id: Optional[str] = None,
    step:     int = 0,
) -> Dict[str, List[str]]:
    """Walk every applicable scope directory under ``kb_root`` and
    load each one's claims into ``ws``.  Broadest scope first
    (universal -> game -> level) so duplicate claims at narrower
    scopes merge as supporting evidence via hypothesis-store dedup.

    Returns ``{scope_label: [hypothesis_ids]}`` so callers can tell
    which scope each loaded claim came from.
    """
    kb_root = Path(kb_root)
    out: Dict[str, List[str]] = {}
    pairs = [
        ("universal", _scope_dir(kb_root, ScopeKind.GLOBAL)),
        ("game",      _scope_dir(kb_root, ScopeKind.GAME,  game_id=game_id)),
        ("level",     _scope_dir(kb_root, ScopeKind.LEVEL,
                                 game_id=game_id, level_id=level_id)),
    ]
    for label, scope_dir in pairs:
        if scope_dir is None or not scope_dir.is_dir():
            continue
        ids = load_committed_markdown(ws, scope_dir, step=step)
        if ids:
            out[label] = ids
    return out


# ---------------------------------------------------------------------------
# One-shot migration
# ---------------------------------------------------------------------------


def migrate_jsonl_to_markdown(kb_dir: Path,
                              *,
                              archive_legacy: bool = True) -> int:
    """Convert an existing ``kb_dir/claims.jsonl`` to per-claim
    markdown files + MEMORY.md.  Returns the number of claims
    migrated.

    When ``archive_legacy`` is True (default) the original JSONL
    file is renamed to ``legacy/claims.jsonl.migrated`` under
    ``kb_dir`` rather than deleted, so a botched migration is
    recoverable.
    """
    import json
    kb_dir = Path(kb_dir)
    jsonl = kb_dir / "claims.jsonl"
    if not jsonl.is_file():
        return 0
    records: List[Dict[str, Any]] = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        records.append(rec)
    claims_dir = kb_dir / "claims"
    claims_dir.mkdir(parents=True, exist_ok=True)
    filenames: List[str] = []
    for rec in records:
        fn = claim_filename(rec)
        content = format_frontmatter(rec) + "\n\n" + _narrative_body(rec)
        (claims_dir / fn).write_text(content, encoding="utf-8")
        filenames.append(fn)
    write_memory_index(kb_dir, records, filenames)
    if archive_legacy:
        legacy_dir = kb_dir / "legacy"
        legacy_dir.mkdir(exist_ok=True)
        jsonl.rename(legacy_dir / "claims.jsonl.migrated")
    return len(records)
