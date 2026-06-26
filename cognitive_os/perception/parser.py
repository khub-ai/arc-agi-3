"""Validate and normalise the VLM's perception output.

The Layer B prompt asks the VLM for a specific JSON shape (see
``prompt.py`` ``_OUTPUT_SCHEMA``).  This module:

* Checks required top-level keys.
* Normalises types (e.g., ``bbox_pixels`` as 4-int tuple).
* Validates every claimed ``role`` / ``relationship.kind`` /
  ``win_condition_hypothesis.kind`` against the loaded catalog.
* Returns a typed structure plus a list of validation messages.

Malformed entries are reported, not silently dropped — the operator
needs visibility into where the VLM's output drifts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .catalog_loader import Catalog, load_catalog
from .geometry import GeometryCandidate, GeometryResult


@dataclass
class ParsedEntity:
    """One named entity in the perceived output.  Built either from
    a single Layer-A candidate or from a group of candidates."""
    id:               str
    bbox_pixels:      tuple
    palettes:         tuple
    role:             str
    candidate_ids:    Sequence[int]    = ()   # which Layer-A candidates this came from
    related_to:       Optional[str]    = None
    member_of_group:  Optional[str]    = None
    notes:            str              = ""


@dataclass
class ParsedRelationship:
    kind:        str
    members:     Sequence[str]
    rationale:   str = ""


@dataclass
class ParsedWinCondition:
    kind:        str
    description: str             = ""
    involves:    Sequence[str]   = ()


@dataclass
class ParsedPerception:
    background_palettes:      Sequence[int]
    palette_role_map:         Mapping[int, str]
    entities:                 List[ParsedEntity]
    relationships:            List[ParsedRelationship]
    win_condition_hypothesis: Optional[ParsedWinCondition]
    uncertainty_notes:        str
    validation_messages:      List[str] = field(default_factory=list)


_TOP_LEVEL_REQUIRED = (
    "candidate_assignments",
    "win_condition_hypothesis",
)


def _merge_candidates(cands: Sequence[GeometryCandidate]) -> tuple:
    """Return (bbox, palettes) for the union of given candidates."""
    if not cands:
        return ((0, 0, 0, 0), ())
    r0 = min(c.bbox_pixels[0] for c in cands)
    c0 = min(c.bbox_pixels[1] for c in cands)
    r1 = max(c.bbox_pixels[2] for c in cands)
    c1 = max(c.bbox_pixels[3] for c in cands)
    pals = tuple(sorted({p for c in cands for p in c.palettes}))
    return ((r0, c0, r1, c1), pals)


def _coerce_bbox(b: Any) -> Optional[tuple]:
    if not isinstance(b, (list, tuple)) or len(b) != 4:
        return None
    try:
        return tuple(int(x) for x in b)
    except (TypeError, ValueError):
        return None


def _coerce_palette_list(p: Any) -> tuple:
    if not isinstance(p, (list, tuple)):
        return ()
    out = []
    for v in p:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return tuple(out)


def parse_vlm_output(
    payload: Mapping[str, Any],
    *,
    geometry: Optional[GeometryResult] = None,
    catalog: Optional[Catalog] = None,
) -> ParsedPerception:
    """Validate a VLM JSON payload and return a typed ParsedPerception.

    When ``geometry`` is supplied, the parser maps candidate_id
    references in the VLM output back to Layer A bboxes and palettes.
    Each VLM-named group becomes one ParsedEntity with bbox =
    union(member candidates).
    """
    cat = catalog if catalog is not None else load_catalog()
    valid_roles      = {e.primitive_id for e in cat.by_kind.get("entity_role", [])}
    valid_roles.add("unknown")
    valid_roles.add("noise")
    valid_rels       = {e.primitive_id for e in cat.by_kind.get("relationship_kind", [])}
    valid_matches    = {e.primitive_id for e in cat.by_kind.get("match_condition", [])}
    msgs: List[str] = []
    cand_by_id: Mapping[int, GeometryCandidate] = (
        {c.candidate_id: c for c in geometry.candidates}
        if geometry else {}
    )

    for key in _TOP_LEVEL_REQUIRED:
        if key not in payload:
            msgs.append(f"missing top-level key {key!r}")

    # background_palettes (still optional in the new schema; VLM can
    # report via palette_role_map instead).
    bg = payload.get("background_palettes", [])
    if not isinstance(bg, list):
        bg = []
    try:
        bg = [int(x) for x in bg]
    except (TypeError, ValueError):
        bg = []

    # palette_role_map
    palette_role_map: Dict[int, str] = {}
    raw_map = payload.get("palette_role_map") or {}
    if not isinstance(raw_map, dict):
        msgs.append("palette_role_map must be an object")
    else:
        for k, v in raw_map.items():
            try:
                ki = int(k)
            except (TypeError, ValueError):
                msgs.append(f"palette_role_map key {k!r} is not an int")
                continue
            if not isinstance(v, str):
                msgs.append(f"palette_role_map value for {ki} is not a string")
                continue
            if v not in valid_roles:
                msgs.append(f"palette_role_map[{ki}] role {v!r} not in catalog")
            palette_role_map[ki] = v

    # candidate_assignments — the VLM's per-candidate role tag.
    raw_assigns = payload.get("candidate_assignments") or {}
    if not isinstance(raw_assigns, dict):
        msgs.append("candidate_assignments must be an object")
        raw_assigns = {}
    cand_role: dict = {}    # candidate_id -> role
    cand_group: dict = {}   # candidate_id -> group label (or None)
    cand_notes: dict = {}
    for k, v in raw_assigns.items():
        try:
            cid = int(k)
        except (TypeError, ValueError):
            msgs.append(f"candidate_assignments key {k!r} is not an int")
            continue
        if cand_by_id and cid not in cand_by_id:
            msgs.append(f"candidate_assignments references unknown id {cid}")
            continue
        if not isinstance(v, dict):
            msgs.append(f"candidate_assignments[{cid}] is not an object")
            continue
        role = str(v.get("role", "unknown") or "unknown")
        if role not in valid_roles:
            msgs.append(f"candidate {cid} has unknown role {role!r}")
        group = v.get("group")
        cand_role[cid] = role
        cand_group[cid] = (str(group) if group else None)
        cand_notes[cid] = str(v.get("notes", "") or "")

    # Coverage check: every Layer A candidate should be assigned.
    if cand_by_id:
        missing = sorted(set(cand_by_id) - set(cand_role))
        if missing:
            msgs.append(f"{len(missing)} candidate(s) lack assignment: "
                        f"{missing[:10]}{'...' if len(missing) > 10 else ''}")

    # groups — VLM-named groups of candidates forming one logical entity.
    raw_groups = payload.get("groups") or []
    if not isinstance(raw_groups, list):
        raw_groups = []
    group_info: dict = {}  # group_label -> {role, members, rationale}
    for i, g in enumerate(raw_groups):
        if not isinstance(g, dict):
            msgs.append(f"group #{i} is not an object")
            continue
        gid = str(g.get("group_id", "") or f"group_{i}")
        members = g.get("members") or []
        if not isinstance(members, list):
            msgs.append(f"group {gid} members is not a list")
            members = []
        try:
            members = [int(m) for m in members]
        except (TypeError, ValueError):
            msgs.append(f"group {gid} has non-int member ids")
            continue
        if cand_by_id:
            unknown_members = [m for m in members if m not in cand_by_id]
            if unknown_members:
                msgs.append(f"group {gid} references unknown candidates "
                            f"{unknown_members}")
        role = str(g.get("role", "unknown") or "unknown")
        if role not in valid_roles:
            msgs.append(f"group {gid} role {role!r} not in catalog")
        group_info[gid] = {
            "role":      role,
            "members":   members,
            "rationale": str(g.get("rationale", "") or ""),
        }

    # Build ParsedEntity records: one per group; plus one per
    # ungrouped candidate that isn't tagged as noise.
    #
    # Special case: when the VLM tags a group with a role that's a
    # *relationship_kind* primitive (e.g. ``reference_pair``), the
    # members are NOT one merged entity -- they're two entities WITH
    # a pairing relationship.  We keep them as ungrouped candidates
    # (so their original cand-level roles survive) and emit a
    # ParsedRelationship instead.  This matches the catalog's
    # entity_role vs relationship_kind distinction: pairing is
    # between members, not a merger of them.
    relationship_groups: List[Tuple[str, List[int], str, str]] = []
    entity_groups: Dict[str, dict] = {}
    for gid, ginfo in group_info.items():
        role = ginfo["role"]
        members = ginfo["members"]
        if role in valid_rels and role not in valid_roles:
            # Relationship_kind group: members stay as individual
            # entities; emit a relationship between them.
            relationship_groups.append((gid, members, role, ginfo["rationale"]))
            continue
        # Entity-role merger is only valid when the group's role is
        # consistent with the cand-level roles of its members.  If the
        # VLM assigned cand-level roles X, Y, Z to members and group
        # role is W (none of X/Y/Z), it's making a SEMANTIC inference
        # ("these together form a W") rather than a visual merger.
        # Keep members as individual entities; record the group as a
        # relationship with kind=role when role is also a known
        # relationship_kind (rare), or just skip the merger.
        member_cand_roles = {cand_role.get(m) for m in members
                             if m in cand_role}
        member_cand_roles.discard(None)
        member_cand_roles.discard("noise")
        if role in member_cand_roles or not member_cand_roles:
            entity_groups[gid] = ginfo
        else:
            # Suspect merger.  Keep candidates separate; record the
            # VLM's semantic claim as a note.  Don't emit a relationship
            # because the role isn't a relationship_kind.
            msgs.append(
                f"group {gid!r} role {role!r} doesn't match any member "
                f"cand-role {sorted(member_cand_roles)}; members kept "
                f"as individual entities (no merger)"
            )

    entities: List[ParsedEntity] = []
    seen_ids: set = set()
    in_group: set = set()
    for gid, ginfo in entity_groups.items():
        cands = [cand_by_id[m] for m in ginfo["members"] if m in cand_by_id]
        in_group.update(c.candidate_id for c in cands)
        bbox, pals = _merge_candidates(cands)
        entities.append(ParsedEntity(
            id              = gid,
            bbox_pixels     = bbox,
            palettes        = pals,
            role            = ginfo["role"],
            candidate_ids   = tuple(c.candidate_id for c in cands),
            related_to      = None,
            member_of_group = None,
            notes           = ginfo["rationale"],
        ))
        seen_ids.add(gid)
    for cid, role in cand_role.items():
        if cid in in_group:
            continue
        if role == "noise":
            continue
        cand = cand_by_id.get(cid) if cand_by_id else None
        if cand is None:
            continue
        eid = f"cand_{cid}_{role}"
        suffix = 0
        while eid in seen_ids:
            suffix += 1
            eid = f"cand_{cid}_{role}_{suffix}"
        seen_ids.add(eid)
        entities.append(ParsedEntity(
            id              = eid,
            bbox_pixels     = cand.bbox_pixels,
            palettes        = cand.palettes,
            role            = role,
            candidate_ids   = (cid,),
            related_to      = None,
            member_of_group = cand_group.get(cid),
            notes           = cand_notes.get(cid, ""),
        ))

    # Region-role consolidation: palette-defined regions (wall, void,
    # floor) often appear as many disconnected fragments after Layer A
    # splits over-merged objects.  Even with the prompt hint, the VLM
    # sometimes still tags each fragment as its own role-X entity.
    # Collapse them: one entity per region role, with the union bbox
    # and union palettes.  This matches the operator-stated principle
    # that palette regions are ONE conceptual entity.
    REGION_ROLES = {"wall", "play_area", "void", "floor", "hud_background"}
    region_buckets: Dict[str, List[ParsedEntity]] = {}
    other_entities: List[ParsedEntity] = []
    for ent in entities:
        if ent.role in REGION_ROLES:
            region_buckets.setdefault(ent.role, []).append(ent)
        else:
            other_entities.append(ent)
    merged_regions: List[ParsedEntity] = []
    for role, ents in region_buckets.items():
        if len(ents) == 1:
            merged_regions.append(ents[0])
            continue
        r0 = min(e.bbox_pixels[0] for e in ents)
        c0 = min(e.bbox_pixels[1] for e in ents)
        r1 = max(e.bbox_pixels[2] for e in ents)
        c1 = max(e.bbox_pixels[3] for e in ents)
        pals = tuple(sorted({p for e in ents for p in e.palettes}))
        cids = tuple(sorted({cid for e in ents for cid in e.candidate_ids}))
        merged_regions.append(ParsedEntity(
            id              = role + "_region",
            bbox_pixels     = (r0, c0, r1, c1),
            palettes        = pals,
            role            = role,
            candidate_ids   = cids,
            related_to      = None,
            member_of_group = None,
            notes           = f"merged from {len(ents)} fragments",
        ))
    # Replace the entity list, preserving order: regions first, then others.
    entities = merged_regions + other_entities
    seen_ids = {e.id for e in entities}

    # Build alias map: VLMs reference relationship/win-condition members
    # via a few different conventions (bare candidate id, group label, or
    # the canonical parser id).  Normalize all of them to the entity.id
    # the scorer expects.
    alias_map: Dict[str, str] = {}
    for ent in entities:
        alias_map[ent.id] = ent.id
        for cid in ent.candidate_ids:
            alias_map.setdefault(str(cid), ent.id)
        if ent.member_of_group:
            alias_map.setdefault(str(ent.member_of_group), ent.id)

    def _resolve(m: str) -> str:
        return alias_map.get(m, m)

    # relationships
    rels: List[ParsedRelationship] = []
    raw_rels = payload.get("relationships", []) or []
    if not isinstance(raw_rels, list):
        msgs.append("relationships must be a list")
        raw_rels = []
    for i, r in enumerate(raw_rels):
        if not isinstance(r, dict):
            msgs.append(f"relationship #{i} is not an object")
            continue
        kind = str(r.get("kind", "") or "")
        if kind not in valid_rels:
            msgs.append(f"relationship #{i} kind {kind!r} not in catalog")
        members = r.get("members") or []
        if not isinstance(members, list):
            msgs.append(f"relationship #{i} members is not a list")
            members = []
        members = [_resolve(str(m)) for m in members]
        for m in members:
            if m not in seen_ids:
                msgs.append(f"relationship #{i} references unknown entity {m!r}")
        rationale = str(r.get("rationale", "") or "")
        rels.append(ParsedRelationship(
            kind=kind, members=tuple(members), rationale=rationale,
        ))

    # Synthesize relationships from groups whose role was a
    # relationship_kind primitive (handled above as "relationship
    # groups").  Each such group contributes one relationship whose
    # members are the entities that came from the group's candidate
    # IDs.
    for gid, member_cand_ids, rkind, rationale in relationship_groups:
        member_entity_ids = []
        for m in member_cand_ids:
            eid = alias_map.get(str(m))
            if eid:
                member_entity_ids.append(eid)
        if len(member_entity_ids) < 2:
            continue
        rels.append(ParsedRelationship(
            kind     = rkind,
            members  = tuple(member_entity_ids),
            rationale = (f"VLM proposed as group {gid!r}: {rationale}"
                          if rationale else f"from VLM group {gid!r}"),
        ))

    # win_condition_hypothesis
    win: Optional[ParsedWinCondition] = None
    raw_win = payload.get("win_condition_hypothesis")
    if isinstance(raw_win, dict):
        kind = str(raw_win.get("kind", "") or "")
        if kind and kind not in valid_matches:
            msgs.append(f"win_condition_hypothesis kind {kind!r} not in catalog")
        involves = raw_win.get("involves") or []
        if not isinstance(involves, list):
            involves = []
        involves = [_resolve(str(x)) for x in involves]
        win = ParsedWinCondition(
            kind        = kind,
            description = str(raw_win.get("description", "") or ""),
            involves    = tuple(involves),
        )
    elif raw_win is not None:
        msgs.append("win_condition_hypothesis must be an object")

    uncertainty = str(payload.get("uncertainty_notes", "") or "")

    return ParsedPerception(
        background_palettes      = tuple(bg),
        palette_role_map         = palette_role_map,
        entities                 = entities,
        relationships            = rels,
        win_condition_hypothesis = win,
        uncertainty_notes        = uncertainty,
        validation_messages      = msgs,
    )
