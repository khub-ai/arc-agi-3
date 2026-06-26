"""Unified KB root + path resolution for COS's knowledge base.

The KB is all persisted, REUSABLE knowledge COS accumulates: per-game lessons,
cross-game knowledge, induced operators, replayable subroutines, mined movement
rules, etc.  Historically each store hard-coded its own absolute path and the
files ended up scattered loose in ``.tmp/`` (and one even duplicated).  This
module gives every store ONE folder to live under so the KB is a single
inspectable / portable / snapshottable unit.

Resolution rules (see docs/SPEC_knowledge_base.md):
  - The KB root is ``<repo>/.tmp/kb`` by default, overridable with the
    ``COS_KB_ROOT`` environment variable (so a run can point the whole KB
    elsewhere — a clean profile, a shared cache — with one setting).
  - ``kb_path(name)`` returns ``<kb_root>/name``.  On first access, if the file
    isn't in the root yet but a legacy copy exists at the old flat ``.tmp/name``
    location, it is MIGRATED into the root (moved) so existing knowledge carries
    over with no loss.
  - If the root can't be created/used for any reason, ``kb_path`` falls back to
    the legacy flat ``.tmp/name`` path, so nothing breaks.

Pure stdlib, no project imports — safe to import from any store module.
"""
from __future__ import annotations
import os
import shutil
from pathlib import Path

# <repo>/tools/governor_audit/perception_loop_v2/kb_paths.py -> parents[3] = repo
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LEGACY_FLAT_DIR = _REPO_ROOT / ".tmp"
_DEFAULT_KB_ROOT = _LEGACY_FLAT_DIR / "kb"

# Every store file the KB owns (used by migrate_all / inventory).
KNOWN_STORES = (
    "per_game_lessons.json",
    "cross_game_knowledge.json",
    "operator_kb.json",
    "subroutine_kb.json",
    "tactic_kb.json",
    "movement_rules.json",
    "error_ledger.json",          # the system's own significant errors + the solutions that fixed them
    "solutions.jsonl",            # replayable win-paths (was .tmp/vlm_cross_session/_solutions.jsonl)
    "visual_catalog.json",        # cross-game visual memory of salient entities (icon -> meaning/function)
)


def kb_root() -> Path:
    """The single folder all KB stores live under (``COS_KB_ROOT`` or
    ``<repo>/.tmp/kb``)."""
    return Path(os.environ.get("COS_KB_ROOT", str(_DEFAULT_KB_ROOT)))


def kb_path(filename: str) -> Path:
    """Resolve a KB store file under the unified root, migrating a legacy
    flat-``.tmp`` copy into the root on first access.  Falls back to the legacy
    flat path if the root is unavailable (so nothing ever breaks)."""
    try:
        root = kb_root()
        root.mkdir(parents=True, exist_ok=True)
        target = root / filename
        if not target.exists():
            legacy = _LEGACY_FLAT_DIR / filename
            if legacy.exists() and legacy.resolve() != target.resolve():
                try:
                    # COPY, never move: migration must be NON-DESTRUCTIVE.  A run with an alternate /
                    # temp COS_KB_ROOT (e.g. a test) must not relocate the REAL legacy store into a
                    # throwaway root and lose it.  The legacy file lingers harmlessly as a backup.
                    shutil.copy2(str(legacy), str(target))
                except Exception:
                    return legacy              # last-resort fallback
        return target
    except Exception:
        return _LEGACY_FLAT_DIR / filename     # KB root unusable -> legacy path


def migrate_all() -> dict:
    """Idempotently pull every known legacy flat-``.tmp`` store into the unified
    KB root.  For a store that exists in BOTH places (e.g. the historical
    duplicate ``movement_rules.json``), keep the NEWER file and drop the stale
    one.  Returns ``{filename: resolved_path}``."""
    report = {}
    root = kb_root()
    root.mkdir(parents=True, exist_ok=True)
    for name in KNOWN_STORES:
        target = root / name
        legacy = _LEGACY_FLAT_DIR / name
        try:
            if target.exists() and legacy.exists() and \
                    legacy.resolve() != target.resolve():
                # duplicate: keep newer, remove older
                if legacy.stat().st_mtime > target.stat().st_mtime:
                    shutil.copy2(str(legacy), str(target))
                legacy.unlink()
            else:
                kb_path(name)                  # triggers move-migration if needed
        except Exception:
            pass
        report[name] = str(root / name)
    return report
