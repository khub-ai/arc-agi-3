"""Load the operational primitives catalog into memory.

Walks the ``catalog/`` subdirectory tree.  Each ``.json`` file is one
primitive entry; the file's subdirectory determines its
``primitive_kind``.  Returns a typed structure the prompt builder
and validators can consume.

Validation at load time:

* The JSON parses.
* ``primitive_id`` matches the filename (without ``.json``).
* ``primitive_kind`` matches the parent directory name.
* Required fields are present (``description``, ``vlm_recognition_hints``,
  ``anchored_in_samples``).

Malformed entries are reported with their file path and skipped — the
rest of the catalog still loads, so a single bad entry doesn't break
the perception pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


_CATALOG_ROOT = Path(__file__).parent / "catalog"


_REQUIRED_FIELDS = ("primitive_id", "primitive_kind", "description",
                    "vlm_recognition_hints", "anchored_in_samples")


_KIND_DIRS = (
    "interaction",
    "entity_role",
    "match_condition",
    "relationship_kind",
    "state_change_effect",
)


@dataclass(frozen=True)
class CatalogEntry:
    """One primitive entry, normalised.

    ``motion_class`` and ``appearance_class`` declare an entity_role's
    intrinsic behaviour under play.  The substrate code reads these
    instead of hardcoding role names so that adding a new role to the
    catalog automatically informs the heuristic gates.  See the
    schema notes inside each entity_role JSON for the controlled
    vocabulary.  Both fields are ``None`` for non-entity_role
    primitives (interaction, match_condition, ...).
    """
    primitive_id:           str
    primitive_kind:         str
    description:            str
    vlm_recognition_hints:  Sequence[str]
    anchored_in_samples:    Sequence[str]
    interaction_signature:  Optional[Mapping[str, Any]] = None
    planner_consumption:    Optional[Mapping[str, Any]] = None
    extension_notes:        Optional[str] = None
    motion_class:           Optional[str] = None
    appearance_class:       Optional[str] = None
    source_path:            Optional[Path] = None


@dataclass
class Catalog:
    """The full catalog, indexed by primitive_id and primitive_kind."""
    entries:        List[CatalogEntry] = field(default_factory=list)
    by_id:          Dict[str, CatalogEntry] = field(default_factory=dict)
    by_kind:        Dict[str, List[CatalogEntry]] = field(default_factory=dict)
    load_errors:    List[tuple] = field(default_factory=list)


def _validate_entry(data: Mapping[str, Any], path: Path) -> List[str]:
    """Return list of validation error messages.  Empty when valid."""
    errors: list = []
    for fld in _REQUIRED_FIELDS:
        if fld not in data:
            errors.append(f"missing required field {fld!r}")
    if "primitive_id" in data:
        expected_id = path.stem
        if data["primitive_id"] != expected_id:
            errors.append(
                f"primitive_id {data['primitive_id']!r} does not match "
                f"filename {expected_id!r}"
            )
    if "primitive_kind" in data:
        expected_kind = path.parent.name
        if data["primitive_kind"] != expected_kind:
            errors.append(
                f"primitive_kind {data['primitive_kind']!r} does not "
                f"match directory {expected_kind!r}"
            )
    if "vlm_recognition_hints" in data:
        if not isinstance(data["vlm_recognition_hints"], list):
            errors.append("vlm_recognition_hints must be a list")
        elif not all(isinstance(s, str) for s in data["vlm_recognition_hints"]):
            errors.append("vlm_recognition_hints must be a list of strings")
    return errors


def load_catalog(root: Optional[Path] = None) -> Catalog:
    """Walk the catalog directory and return a populated Catalog.

    ``root`` defaults to the package's ``catalog/`` subdirectory.
    Pass a different path for testing or alternative catalogs.
    """
    root = root or _CATALOG_ROOT
    cat = Catalog()
    for kind in _KIND_DIRS:
        kind_dir = root / kind
        if not kind_dir.is_dir():
            continue
        for json_path in sorted(kind_dir.glob("*.json")):
            try:
                with open(json_path, encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                cat.load_errors.append((str(json_path), f"parse failed: {e}"))
                continue
            errs = _validate_entry(data, json_path)
            if errs:
                cat.load_errors.append((str(json_path), "; ".join(errs)))
                continue
            entry = CatalogEntry(
                primitive_id           = str(data["primitive_id"]),
                primitive_kind         = str(data["primitive_kind"]),
                description            = str(data["description"]),
                vlm_recognition_hints  = tuple(data["vlm_recognition_hints"]),
                anchored_in_samples    = tuple(data["anchored_in_samples"]),
                interaction_signature  = data.get("interaction_signature"),
                planner_consumption    = data.get("planner_consumption"),
                extension_notes        = data.get("extension_notes"),
                motion_class           = data.get("motion_class"),
                appearance_class       = data.get("appearance_class"),
                source_path            = json_path,
            )
            cat.entries.append(entry)
            cat.by_id[entry.primitive_id] = entry
            cat.by_kind.setdefault(entry.primitive_kind, []).append(entry)
    return cat


def summarise(catalog: Catalog) -> str:
    """Operator-facing one-page summary of catalog contents."""
    lines: List[str] = []
    lines.append(f"Catalog: {len(catalog.entries)} entries across "
                 f"{len(catalog.by_kind)} kinds")
    for kind, entries in sorted(catalog.by_kind.items()):
        lines.append(f"\n[{kind}] {len(entries)} entries")
        for e in sorted(entries, key=lambda x: x.primitive_id):
            anchors = ", ".join(e.anchored_in_samples) or "—"
            lines.append(f"  - {e.primitive_id}: {anchors}")
    if catalog.load_errors:
        lines.append(f"\n{len(catalog.load_errors)} load errors:")
        for path, msg in catalog.load_errors:
            lines.append(f"  ! {path}: {msg}")
    return "\n".join(lines)


if __name__ == "__main__":
    cat = load_catalog()
    print(summarise(cat))
