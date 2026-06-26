"""Extract structured level-memory invariants from one or more played
levels, and render them as a prior_notes block for the next level's
Layer B prompt.

The 2026-05-12 experiment on ls20 4/7 showed that the *form* of the
prior matters enormously: a loose paragraph took gemma to 0.871, while
a structured prior (palette-count + size range + spatial anchor +
disambiguation rules) took it to a perfect 1.000.  This module captures
that structured form so the live agent can build it automatically from
playthroughs of earlier levels of the same game.

What we preserve, per entity role observed across levels:
* palette_count: range over observed instances (1 -> monochrome,
                  2 -> two-tone, 4+ -> multi-palette cluster)
* size_pixels:   (min, max) across observed instances, with margin
* spatial_cells: cell coordinates each instance was seen at
* spawn_only:    true when the role is always at the agent's spawn cell

What we DON'T preserve:
* per-level entity bboxes — those drift level to level and create false
  anchors when transferred forward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


@dataclass
class RoleObservation:
    """One sighting of a role on a specific played level."""
    level:        str
    palette_count: int
    palettes:     Tuple[int, ...]
    pixel_size:   int
    cell_anchor:  Optional[Tuple[int, int]] = None
    bbox_pixels:  Optional[Tuple[int, int, int, int]] = None


@dataclass
class RoleInvariants:
    """Aggregated invariants for one role across observed levels."""
    role:                 str
    n_observations:       int = 0
    palette_count_min:    int = 0
    palette_count_max:    int = 0
    pixel_size_min:       int = 0
    pixel_size_max:       int = 0
    cell_anchors:         List[Tuple[int, int]] = field(default_factory=list)
    instances_per_level:  Dict[str, int] = field(default_factory=dict)


@dataclass
class ConfirmedRole:
    """A role label that has been confirmed by gameplay (causal observation),
    not just inferred from a single frame.  Keyed by bitmap_id so it
    transfers across levels where the same sprite recurs."""
    bitmap_id:        str
    description:      str
    palette_hint:     Optional[int]  = None
    size_hint:        Optional[int]  = None


@dataclass
class GameMemory:
    """All-levels-so-far memory for one game."""
    game:                 str
    levels_played:        List[str]
    role_invariants:      Dict[str, RoleInvariants]
    hud_row_range:        Optional[Tuple[int, int]] = None
    win_condition_kinds:  List[str] = field(default_factory=list)
    agent_spawn_cell:     Optional[Tuple[int, int]] = None
    # Causally-confirmed roles from harness learned_facts.  These OVERRIDE
    # single-frame perception observations because they were validated by
    # the agent successfully interacting with the entity.
    confirmed_roles:      List[ConfirmedRole] = field(default_factory=list)
    action_semantics:     Dict[str, str]      = field(default_factory=dict)
    # Palette assignments observed in earlier play of THIS game.  These
    # are game-specific facts (different games use different palettes for
    # floor/wall) -- never treat them as universal rules.
    floor_palettes:               List[int] = field(default_factory=list)
    wall_palettes_current_level:  List[int] = field(default_factory=list)
    current_level_label:          Optional[str] = None


_REGION_ROLES = {"wall", "play_area", "void", "floor", "hud_background"}

# Roles that we should NEVER emit into a prior_notes block because they
# either don't help disambiguation (region roles span the frame and
# vary level-to-level) or actively poison it (`unknown` means the agent
# didn't recognise the entity, so its size/palette signature is
# meaningless going forward).
_ROLES_EXCLUDED_FROM_PRIOR = _REGION_ROLES | {"unknown"}

# Roles whose primary bitmap is worth persisting to the KB as a
# transferable bitmap->role mapping.  HUD-class roles (budget_meter,
# life_indicator) are excluded because their bitmap is the whole HUD
# bar that drifts level-to-level.  Region roles are excluded for the
# same reason.  `unknown` and `static_observed_not_agent` are excluded
# because the system has not actually identified the entity.
_TRANSFERABLE_BITMAP_ROLES = {
    "agent", "agent_avatar",
    "rotator", "shape_changer", "color_changer",
    "working_glyph", "reference_glyph",
    "target_slot", "life_refuel", "launcher",
    "movable_block", "movable_pin",
    "anchor_endpoint", "linked_midpoint", "rigid_link",
    "piercer_head", "piercer_tail",
    "divider", "selection_bracket",
    "reference_arrangement", "working_sequence",
    # HUD-class roles: their bitmap is the entire bar and changes
    # during play (the meter shrinks), but the INITIAL bar bitmap is
    # consistent across levels in many games (the bar starts at the
    # same fill level on every level entry).  We include them so the
    # initial frame can match; the inevitable mid-play size change
    # falls naturally to the shape/topo tiers.
    "budget_meter", "life_indicator",
}


def extract_observations(gt: Mapping[str, Any], level_label: str
                         ) -> Dict[str, List[RoleObservation]]:
    """Pull per-role observations from one ground_truth-style record.

    Skips palette-region entities (wall, play_area, void) since their
    bboxes are palette-extents, not real shapes.  Skips entries flagged
    `_note_only`.
    """
    out: Dict[str, List[RoleObservation]] = {}
    for e in gt.get("entities") or []:
        if not isinstance(e, dict) or e.get("_note_only"):
            continue
        role = str(e.get("role") or "").strip()
        if not role or role in _REGION_ROLES:
            continue
        if e.get("_bbox_is_palette_extent_not_actual_shape"):
            continue
        pals = tuple(int(p) for p in (e.get("palettes") or []))
        bbox = e.get("bbox_pixels")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        r0, c0, r1, c1 = (int(x) for x in bbox)
        # bbox area is the proxy for pixel size; cells_occupied is the
        # cell-grid count (a different unit) and is only used here to
        # pick a spatial anchor.
        size = (r1 - r0 + 1) * (c1 - c0 + 1)
        cells = e.get("cells_occupied")
        cell_anchor: Optional[Tuple[int, int]] = None
        if isinstance(cells, list) and cells and len(cells[0]) == 2:
            cell_anchor = tuple(int(x) for x in cells[0])
        obs = RoleObservation(
            level         = level_label,
            palette_count = len(pals),
            palettes      = pals,
            pixel_size    = size,
            cell_anchor   = cell_anchor,
            bbox_pixels   = (r0, c0, r1, c1),
        )
        out.setdefault(role, []).append(obs)
    return out


def _aggregate_role(role: str, obs_list: Sequence[RoleObservation]) -> RoleInvariants:
    inv = RoleInvariants(role=role, n_observations=len(obs_list))
    pcounts = [o.palette_count for o in obs_list]
    sizes   = [o.pixel_size for o in obs_list]
    inv.palette_count_min = min(pcounts)
    inv.palette_count_max = max(pcounts)
    inv.pixel_size_min    = min(sizes)
    inv.pixel_size_max    = max(sizes)
    for o in obs_list:
        if o.cell_anchor is not None:
            inv.cell_anchors.append(o.cell_anchor)
        inv.instances_per_level[o.level] = (
            inv.instances_per_level.get(o.level, 0) + 1
        )
    return inv


def build_game_memory(
    played_levels: Sequence[Mapping[str, Any]],
    *,
    game: str,
) -> GameMemory:
    """Merge observations from multiple played levels into one GameMemory.

    ``played_levels`` is a list of dicts.  Each dict has at least:
        "level":        str         (e.g. "1/7")
        "ground_truth": Mapping     (the operator-confirmed ParsedPerception
                                     in GT format)
    """
    by_role: Dict[str, List[RoleObservation]] = {}
    win_kinds: List[str] = []
    hud_row_start: Optional[int] = None
    hud_row_end:   Optional[int] = None
    agent_spawn:   Optional[Tuple[int, int]] = None
    levels_played: List[str] = []

    for record in played_levels:
        level = record.get("level", "?")
        gt    = record.get("ground_truth") or {}
        levels_played.append(level)
        observations = extract_observations(gt, level)
        for role, obs_list in observations.items():
            by_role.setdefault(role, []).extend(obs_list)
        # Mechanic / win condition.
        mech = gt.get("_game_mechanic") or {}
        wc = (mech.get("win_condition") or "").strip()
        if wc and wc not in win_kinds:
            # Best-effort categorisation if it's a structured kind.
            win_kinds.append(wc)
        # HUD row range from any entity with a HUD-ish role.
        for ent in gt.get("entities") or []:
            if not isinstance(ent, dict):
                continue
            role = str(ent.get("role") or "")
            if role in ("budget_meter", "life_indicator", "hud_background"):
                bb = ent.get("bbox_pixels")
                if isinstance(bb, (list, tuple)) and len(bb) == 4:
                    r0, _, r1, _ = (int(x) for x in bb)
                    hud_row_start = r0 if hud_row_start is None else min(hud_row_start, r0)
                    hud_row_end   = r1 if hud_row_end   is None else max(hud_row_end,   r1)
            if role == "agent_avatar" and agent_spawn is None:
                cells = ent.get("cells_occupied")
                if isinstance(cells, list) and cells and len(cells[0]) == 2:
                    agent_spawn = tuple(int(x) for x in cells[0])

    role_invariants = {
        role: _aggregate_role(role, obs_list)
        for role, obs_list in by_role.items()
    }

    return GameMemory(
        game                = game,
        levels_played       = levels_played,
        role_invariants     = role_invariants,
        hud_row_range       = ((hud_row_start, hud_row_end)
                               if hud_row_start is not None else None),
        win_condition_kinds = win_kinds,
        agent_spawn_cell    = agent_spawn,
    )


def attach_kb_learned_facts(
    mem:                GameMemory,
    kb_path:            Path,
    *,
    current_level_idx:  Optional[int] = None,
) -> GameMemory:
    """Augment a GameMemory with the harness's KB `learned_facts`.

    The harness validates roles by gameplay outcome (agent interacted
    with this bitmap; the level advanced as a result; this role
    description is now in `kb["learned_facts"]["component_roles"]`).
    These confirmed roles are HIGHER-CONFIDENCE than single-frame
    perception output and should drive the next level's prior.
    """
    import re
    if not kb_path.exists():
        return mem
    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception:
        return mem
    lf = kb.get("learned_facts") or {}
    component_roles = lf.get("component_roles") or {}
    confirmed: List[ConfirmedRole] = []
    pal_re = re.compile(r"pal=?(\d+)")
    sz_re = re.compile(r"sz=?(\d+)")
    for bm_id, desc_raw in component_roles.items():
        if isinstance(desc_raw, list):
            desc = " | ".join(str(d) for d in desc_raw)
        else:
            desc = str(desc_raw)
        if not desc:
            continue
        pal_match = pal_re.search(desc)
        sz_match  = sz_re.search(desc)
        confirmed.append(ConfirmedRole(
            bitmap_id    = str(bm_id),
            description  = desc,
            palette_hint = int(pal_match.group(1)) if pal_match else None,
            size_hint    = int(sz_match.group(1))  if sz_match  else None,
        ))
    mem.confirmed_roles = confirmed
    aseman = lf.get("action_semantics") or {}
    mem.action_semantics = {
        str(k): str(v) for k, v in aseman.items()
        if isinstance(v, str) and v.strip()
    }
    floors = lf.get("floor_palettes") or []
    if isinstance(floors, list):
        mem.floor_palettes = [int(p) for p in floors if isinstance(p, (int, float))]
    if current_level_idx is not None:
        wbl = kb.get("wall_palette_by_level") or {}
        wlv = wbl.get(str(current_level_idx)) or []
        if isinstance(wlv, list):
            # If we have vote counts, prefer the top-voted palette(s).
            votes = (kb.get("wall_palette_votes") or {}).get(str(current_level_idx)) or {}
            if isinstance(votes, dict) and votes:
                ranked = sorted(((int(p), int(n)) for p, n in votes.items()),
                                key=lambda kv: -kv[1])
                mem.wall_palettes_current_level = [p for p, _ in ranked]
            else:
                mem.wall_palettes_current_level = [int(p) for p in wlv
                                                   if isinstance(p, (int, float))]
        mem.current_level_label = str(current_level_idx)
    return mem


# ---------------------------------------------------------------------------
# Render to prior_notes string for the next level's prompt.
# ---------------------------------------------------------------------------


def _palette_count_phrase(lo: int, hi: int) -> str:
    if lo == hi:
        if lo == 1:
            return "MONOCHROME — exactly ONE palette in the candidate"
        if lo == 2:
            return "TWO-TONE — exactly TWO palettes meeting along a seam"
        return f"exactly {lo} distinct palettes"
    if lo == 1 and hi == 2:
        return "MONOCHROME or two-tone (1-2 palettes)"
    if hi >= 4:
        return f"MULTI-PALETTE cluster ({lo}-{hi}+ distinct palettes packed together)"
    return f"{lo}-{hi} distinct palettes"


def _size_range_with_margin(lo: int, hi: int) -> Tuple[int, int]:
    """Apply a small margin so the next level isn't excluded by an
    unusual size variant.  +/- 25% with min 2px floor."""
    margin_lo = max(2, int(lo * 0.75))
    margin_hi = int(hi * 1.25) + 1
    return margin_lo, margin_hi


def render_prior_notes(mem: GameMemory) -> str:
    """Render the GameMemory as a prompt-ready prior_notes block.

    The block mirrors the structure of the v2 prior that took gemma to
    a perfect score on ls20 4/7: frame layout, per-role recognition
    rules, disambiguation rules.
    """
    lines: List[str] = []
    levels = ", ".join(mem.levels_played) or "(none yet)"
    lines.append(
        f"PRIOR-LEVEL MEMORY (the agent has already played {mem.game} "
        f"levels {levels}):"
    )
    lines.append("")
    lines.append("# Frame layout learned from prior levels")
    lines.append("")
    if mem.hud_row_range:
        r0, r1 = mem.hud_row_range
        lines.append(
            f"- HUD strictly occupies rows {r0}-{r1} of the 64x64 frame.  "
            f"ANY entity whose bbox top row is row {r0-1} or ABOVE is a "
            "PLAY-AREA entity, NOT a HUD element.  Do NOT dismiss "
            "play-area sprites as 'HUD-adjacent'."
        )
        lines.append("")
    if mem.agent_spawn_cell is not None:
        cr, cc = mem.agent_spawn_cell
        lines.append(
            f"- The agent_avatar spawns at cell ({cr}, {cc}) at level start."
        )
        lines.append("")
    lines.append("# Entity recognition rules from prior levels")
    lines.append("")

    # Stable, opinionated role order: agent first, then glyphs, then
    # triggers, then HUD/regions, then everything else.
    role_order = [
        "agent_avatar", "reference_glyph", "working_glyph",
        "shape_changer", "color_changer",
        "movable_block", "movable_pin", "anchor_endpoint",
        "linked_midpoint", "rigid_link",
        "piercer_head", "piercer_tail",
        "reference_arrangement", "selection_bracket",
        "launcher", "life_refuel", "life_indicator", "budget_meter",
        "divider", "target_slot", "working_sequence",
    ]
    seen = set(role_order)
    extras = [r for r in mem.role_invariants if r not in seen]
    n_levels_played = max(1, len(mem.levels_played))
    for role in role_order + sorted(extras):
        if role in _ROLES_EXCLUDED_FROM_PRIOR:
            continue
        inv = mem.role_invariants.get(role)
        if inv is None:
            continue
        # Robustness against bad memory: a role seen in only ONE level
        # is a single observation -- its size/palette signature may be
        # an error (perception miscalled the candidate).  Emit a weaker
        # entry without strong size/palette assertions.
        n_levels_with_role = len(inv.instances_per_level)
        unreliable = (n_levels_played >= 2 and n_levels_with_role <= 1)
        size_lo, size_hi = _size_range_with_margin(inv.pixel_size_min,
                                                  inv.pixel_size_max)
        pcount_phrase = _palette_count_phrase(inv.palette_count_min,
                                              inv.palette_count_max)
        lines.append(f"{role}:")
        if unreliable:
            lines.append(
                f"  - Observed only in 1 of {n_levels_played} prior levels.  "
                "The signature below comes from a single observation and "
                "may reflect a misclassification.  Use the catalog's own "
                "recognition hints (above) as the primary guide; treat the "
                "numbers here as weak hints only."
            )
        lines.append(f"  - {pcount_phrase}.")
        lines.append(f"  - Pixel size {size_lo}-{size_hi}px "
                     f"(observed range {inv.pixel_size_min}-{inv.pixel_size_max}).")
        if inv.cell_anchors and not unreliable:
            anchor_text = ", ".join(f"({cr},{cc})" for cr, cc in inv.cell_anchors[:5])
            lines.append(f"  - Observed at cells: {anchor_text}.")
        n_per_level = inv.instances_per_level
        if n_per_level and max(n_per_level.values()) > 1:
            avg = sum(n_per_level.values()) / max(1, len(n_per_level))
            lines.append(f"  - Multiple instances per level (~{avg:.0f} typical).")
        lines.append("")

    # Win condition history.  Phrase as a HYPOTHESIS to test against
    # the current frame -- never as a constraint.  Prior-level perception
    # may have miscalled the win condition, and if we assert it strongly
    # the VLM will lock onto that error and miss what the current frame
    # actually shows.  See trial_perception_priors: perception said
    # 'reach_cell' on ls20 levels 1-3 (wrong; actual is alignment_match)
    # and the strongly-worded prior carried that error into level 4.
    if mem.win_condition_kinds:
        n_levels = len(mem.levels_played) or 1
        # Count how many distinct kinds and how often each was seen.
        # We only have one win_condition per level so frequency == 1
        # per kind in the current schema; treat any single observation
        # as a weak signal.
        lines.append("# Win condition observed in prior levels (hypothesis only)")
        lines.append("")
        kinds = ", ".join(repr(k) for k in mem.win_condition_kinds)
        lines.append(
            f"In the {n_levels} prior level(s), perception hypothesised "
            f"win-condition kind(s): {kinds}.  Treat this as a HYPOTHESIS, "
            "not a constraint -- prior-level perception may have been "
            "wrong.  Examine THIS frame independently and assert whichever "
            "match_condition the visible entities suggest, even if it "
            "differs from the prior hypothesis."
        )
        lines.append("")

    # Confirmed roles from gameplay outcomes (harness learned_facts).
    # These are HIGHEST-CONFIDENCE because the agent interacted with these
    # entities and the level state changed as expected — different from
    # the single-frame perception observations above, which may be wrong.
    if mem.confirmed_roles:
        lines.append("# Causally-confirmed entity roles (from prior-level gameplay)")
        lines.append("")
        lines.append(
            "The following roles were CONFIRMED by gameplay outcomes in "
            "earlier levels — the agent interacted with these entities "
            "and observed the documented behaviour.  Treat these as "
            "AUTHORITATIVE: if you see a candidate matching one of these "
            "bitmap signatures in THIS frame, assign the corresponding "
            "role even if other heuristics would suggest something else."
        )
        lines.append("")
        for r in mem.confirmed_roles:
            sig = []
            if r.palette_hint is not None:
                sig.append(f"palette={r.palette_hint}")
            if r.size_hint is not None:
                sig.append(f"size={r.size_hint}px")
            sig_str = (" (" + ", ".join(sig) + ")") if sig else ""
            lines.append(
                f"- bitmap_id={r.bitmap_id}{sig_str}: {r.description}"
            )
        lines.append("")
    if mem.action_semantics:
        lines.append("# Action semantics confirmed in prior play")
        lines.append("")
        for aid, desc in sorted(mem.action_semantics.items()):
            lines.append(f"- {aid}: {desc}")
        lines.append("")

    if mem.floor_palettes or mem.wall_palettes_current_level:
        lines.append(
            "# Wall and floor palettes for THIS game "
            "(observed from prior play -- game-specific, not universal)"
        )
        lines.append("")
        lines.append(
            "Different games assign different palettes to walls and "
            "floors; do NOT assume a fixed colour-to-role mapping.  The "
            "values below are what perception observed for "
            f"{mem.game} specifically:"
        )
        lines.append("")
        if mem.floor_palettes:
            pal_list = ", ".join(str(p) for p in mem.floor_palettes)
            lines.append(
                f"- FLOOR palette(s) in this game: {{{pal_list}}}.  The large "
                "interior region of the play area is filled with this "
                "palette.  Assign role=play_area (or role=floor) to its "
                "bounding region, NEVER role=wall."
            )
        if mem.wall_palettes_current_level:
            pal_list = ", ".join(str(p) for p in mem.wall_palettes_current_level)
            lvl = (f" (level index {mem.current_level_label})"
                   if mem.current_level_label is not None else "")
            lines.append(
                f"- WALL palette(s){lvl}: {{{pal_list}}} (top-voted first).  "
                "Thin border strips and chamber outlines in this palette "
                "are role=wall, NEVER role=play_area."
            )
        lines.append("")

    # Auto-derived disambiguation rules.
    rules = _derive_disambiguation_rules(mem)
    if rules:
        lines.append("# Disambiguation rules (derived from prior-level invariants)")
        lines.append("")
        for r in rules:
            lines.append(f"- {r}")
        lines.append("")

    # Catalog-term lookalike warnings.  Same name root appears in both
    # entity_role and relationship_kind catalogs (e.g. reference_pair vs
    # reference_pair_member).  Models can confuse them; remind explicitly.
    lookalikes = _lookalike_glossary(mem)
    if lookalikes:
        lines.append("# Catalog terms that look alike (use the entity_role form for roles)")
        lines.append("")
        for line in lookalikes:
            lines.append(f"- {line}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _lookalike_glossary(mem: GameMemory) -> List[str]:
    """Emit confusable-primitive warnings derived from the catalog.

    Two sources:
    1. Entity_role whose ``primitive_id`` root collides with a
       relationship_kind primitive (e.g., ``reference_pair_member``
       vs ``reference_pair``).  The VLM often picks the shorter form
       for an entity role; the warning steers it back.
    2. Per-primitive ``confusable_with`` annotations: each catalog
       entry can declare a list of primitive_ids it's been observed
       to be confused with, plus a short disambiguating note.

    Generated catalog-wide, no longer gated on
    ``mem.role_invariants`` -- the disambiguation helps every game
    from the first frame.
    """
    try:
        from .catalog_loader import load_catalog
        catalog = load_catalog()
        rel_ids  = {e.primitive_id for e in catalog.by_kind.get("relationship_kind", [])}
        role_ids = [e.primitive_id for e in catalog.by_kind.get("entity_role", [])]
    except Exception:
        return []
    lines: List[str] = []
    seen_pairs: set = set()
    # Pattern 1: name-root collision with relationship_kind.
    for role in role_ids:
        if "_" in role:
            root = role.rsplit("_", 1)[0]
            if root in rel_ids and (root, role) not in seen_pairs:
                lines.append(
                    f"{role!r} is the ENTITY ROLE; {root!r} is the "
                    f"RELATIONSHIP_KIND.  When labelling an individual "
                    f"sprite, use {role!r} not {root!r}."
                )
                seen_pairs.add((root, role))
    # Pattern 2: catalog-declared confusable_with annotations.
    for entry in catalog.by_kind.get("entity_role", []):
        ext = getattr(entry, "confusable_with", None) or []
        # ``confusable_with`` may live in extension_notes_meta dicts
        # depending on catalog schema; fall back to raw_data parsing
        # when the typed field isn't present.
        if not ext and entry.source_path:
            try:
                data = json.loads(entry.source_path.read_text(encoding="utf-8"))
                ext = data.get("confusable_with") or []
            except Exception:
                ext = []
        for partner in ext:
            if isinstance(partner, dict):
                pid = partner.get("primitive_id")
                note = partner.get("note") or ""
            else:
                pid = str(partner)
                note = ""
            if not pid:
                continue
            pair = tuple(sorted([entry.primitive_id, pid]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            extra = f"  ({note})" if note else ""
            lines.append(
                f"{entry.primitive_id!r} vs {pid!r} -- these are often "
                f"confused.{extra}"
            )
    return lines


def _derive_disambiguation_rules(mem: GameMemory) -> List[str]:
    """Produce per-pair disambiguation lines from the role invariants.

    Looks for confusable pairs (similar size or overlapping palette
    counts) and emits a rule citing the *distinguishing* axis.
    """
    rules: List[str] = []
    rv = mem.role_invariants
    # Agent vs glyph confusion is the cardinal one — but only emit the
    # "multi-palette => agent" rule when observations actually support it.
    # If prior perception saw glyphs as multi-palette (a known failure
    # mode), the rule would contradict the per-role recognition signatures
    # above and confuse the model.
    if "agent_avatar" in rv:
        agent_pmin = rv["agent_avatar"].palette_count_min
        glyph_pmax = max(
            (rv[k].palette_count_max
             for k in ("reference_glyph", "working_glyph")
             if k in rv),
            default=0,
        )
        if agent_pmin > glyph_pmax > 0:
            rules.append(
                f"agent_avatar has {agent_pmin}+ palettes; reference_glyph "
                f"and working_glyph have AT MOST {glyph_pmax} palette(s).  "
                "A multi-palette sprite is ALWAYS the agent, never a glyph."
            )
    # Working glyph vs color_changer: both can have multiple palettes in some
    # games — distinguish by size.
    if "working_glyph" in rv and "color_changer" in rv:
        wg = rv["working_glyph"]
        cc = rv["color_changer"]
        if wg.pixel_size_min > cc.pixel_size_max:
            rules.append(
                f"working_glyph is LARGER than color_changer "
                f"(working {wg.pixel_size_min}-{wg.pixel_size_max}px, "
                f"color_changer {cc.pixel_size_min}-{cc.pixel_size_max}px).  "
                "Pixel size separates them when palette signatures look similar."
            )
    # Reference vs working: both monochrome, distinguished by size.
    if "reference_glyph" in rv and "working_glyph" in rv:
        ref = rv["reference_glyph"]
        wg  = rv["working_glyph"]
        if wg.pixel_size_min > ref.pixel_size_max:
            rules.append(
                f"reference_glyph is SMALLER ({ref.pixel_size_min}-"
                f"{ref.pixel_size_max}px); working_glyph is LARGER "
                f"({wg.pixel_size_min}-{wg.pixel_size_max}px).  Both are "
                "monochrome; size distinguishes them."
            )
    # Shape changer is typically the smallest monochrome cluster.
    if "shape_changer" in rv and "reference_glyph" in rv:
        sc = rv["shape_changer"]
        ref = rv["reference_glyph"]
        if sc.pixel_size_max < ref.pixel_size_min:
            rules.append(
                f"shape_changer ({sc.pixel_size_min}-{sc.pixel_size_max}px) is "
                f"SMALLER than reference_glyph ({ref.pixel_size_min}-"
                f"{ref.pixel_size_max}px).  Both monochrome; size separates them."
            )
    # Three-monochrome reminder if all three present.
    if all(r in rv for r in ("shape_changer", "reference_glyph", "working_glyph")):
        rules.append(
            "There are typically THREE prominent monochrome shapes: "
            "shape_changer (smallest), reference_glyph (medium), "
            "working_glyph (largest).  Order them by pixel size when in doubt."
        )
    return rules


# ---------------------------------------------------------------------------
# Convenience: build memory from a list of GT file paths.
# ---------------------------------------------------------------------------


# System-generated "negative" labels: not role hypotheses but
# behavioural assertions (the system observed the entity didn't move,
# didn't change, etc.).  These should not get a credence rating because
# they're not claims about what the entity IS, only about what was NOT
# observed during this short window.
_NON_ROLE_LABELS = {
    "static_observed_not_agent",
    "static_observed_no_effect",
}

# Tier names for role candidates, with confidence values used by the
# central resolver.  Higher = stronger evidence; "causal" is
# direct-interaction evidence (motion, trigger, observed slide) and
# wins over any structural / pixel-pattern signal.
_TIER_CONFIDENCE = {
    "causal":  100,
    "bitmap":   90,
    "region":   85,
    "shape":    60,
    "topo":     50,
    "soft":     30,
    "vlm":      10,
}


def add_role_candidate(
    ent:        dict,
    role:       str,
    tier:       str,
    evidence:   str,
    *,
    confidence: Optional[int] = None,
) -> None:
    """Record a role proposal on an entity for the central resolver.

    Matchers no longer mutate ``ent["role"]`` directly; instead they
    call this with their proposal.  The resolver picks the
    highest-confidence candidate at the end of the pipeline.

    Deduplicates by (role, tier) so the same matcher firing twice on
    the same entity doesn't accumulate phantom candidates.
    """
    if not isinstance(ent, dict) or not role:
        return
    if confidence is None:
        confidence = _TIER_CONFIDENCE.get(tier, 10)
    cands = ent.setdefault("_role_candidates", [])
    for c in cands:
        if c.get("role") == role and c.get("tier") == tier:
            return
    cands.append({
        "role":       role,
        "tier":       tier,
        "confidence": int(confidence),
        "evidence":   str(evidence),
    })


def resolve_roles(parsed: Mapping[str, Any]) -> Dict[str, int]:
    """Pick the winning role per entity from its candidate list.

    Order of evaluation: highest confidence wins.  Ties are broken by
    tier specificity (causal > bitmap > region > shape > topo > soft >
    vlm), then by appearance order (first-emitted wins so the matcher
    that ran earliest gets the tie-break).

    Mutates ``ent["role"]`` to the winning role and records the
    winning tier on ``ent["_role_resolved_tier"]`` so the credence
    pass can read it directly.  Entities with no candidates keep their
    existing role (e.g. region roles like ``play_area`` set by the
    VLM, which are not in the candidate system).

    Returns a count of how many entities were re-assigned vs.
    confirmed-as-is.
    """
    counts = {"reassigned": 0, "kept": 0, "no_candidates": 0}
    tier_order = list(_TIER_CONFIDENCE.keys())
    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        cands = ent.get("_role_candidates") or []
        if not cands:
            counts["no_candidates"] += 1
            continue
        ranked = sorted(
            enumerate(cands),
            key=lambda ic: (
                -ic[1].get("confidence", 0),
                tier_order.index(ic[1].get("tier", "vlm"))
                if ic[1].get("tier", "vlm") in tier_order else 999,
                ic[0],
            ),
        )
        winner = ranked[0][1]
        ent["_role_resolved_tier"] = winner["tier"]
        ent["_role_resolved_confidence"] = winner["confidence"]
        if ent.get("role") != winner["role"]:
            ent.setdefault("_corrections", []).append(
                f"resolver: role={winner['role']} via {winner['tier']} "
                f"tier (conf {winner['confidence']}); overrode "
                f"{ent.get('role')!r}"
            )
            ent["role"] = winner["role"]
            counts["reassigned"] += 1
        else:
            counts["kept"] += 1
    return counts

# Roles whose identity is anchored more reliably by REGION features
# (bbox row range, palette set, spatial zone) than by an exact bitmap.
# A budget_meter shrinks during play and its primary bitmap changes,
# but the strip it occupies and its palette set are stable.
_HUD_CLASS_ROLES = {"budget_meter", "life_indicator"}


def demote_mismatched_tentative_roles(
    parsed:  Mapping[str, Any],
    kb_path: Path,
) -> int:
    """For every entity carrying a transferable role that the VLM
    proposed but no causal/structural matcher confirmed, check whether
    the candidate's palette and shape are consistent with any of the
    KB's records for that role.  Demote to ``unknown`` when there is
    no plausible signature overlap.

    The VLM tends to propose roles from prior_notes guidance ("small
    pal=1 sprite -> rotator") even when the candidate's palette
    differs.  Without this gate, a pal=0 sprite of similar size to the
    KB rotator gets a misleading ``rotator?`` tag.  After this pass
    the same sprite becomes ``unknown`` and surfaces in
    ``curiosity_targets`` for explicit probing.

    Mutates ``parsed`` in place; returns the number of roles demoted.
    """
    if not kb_path.exists():
        return 0
    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    crm = (kb.get("learned_facts") or {}).get("component_role_meta") or {}
    role_pals: Dict[str, set] = {}
    role_shapes: Dict[str, set] = {}
    for meta in crm.values():
        if not isinstance(meta, dict):
            continue
        role = meta.get("role")
        if not role:
            continue
        for p in meta.get("palettes") or []:
            role_pals.setdefault(role, set()).add(int(p))
        sh = meta.get("shape_id")
        if sh:
            role_shapes.setdefault(role, set()).add(str(sh))

    n_demoted = 0
    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        role = str(ent.get("role") or "")
        if role not in _TRANSFERABLE_BITMAP_ROLES:
            continue
        # Confirmed roles already pass; only re-check tentative.
        corrections = [str(c) for c in (ent.get("_corrections") or [])]
        if any(
            ("KB bitmap match" in c)
            or ("KB shape match" in c)
            or ("KB region match" in c)
            or ("Bounce-plate match" in c)
            or ("confirmed agent role" in c)
            or ("confirmed " in c and "via observed trigger function" in c)
            or ("observed PERMANENT appearance change" in c)
            or ("visiting this entity caused another entity to change" in c)
            or ("CONFIRMED by interaction" in c)
            for c in corrections
        ):
            continue
        # Compare candidate palettes / shape_ids against KB role.
        kb_pals = role_pals.get(role) or set()
        kb_shapes = role_shapes.get(role) or set()
        if not kb_pals and not kb_shapes:
            continue   # no KB record for this role; nothing to compare
        ent_pals = {int(p) for p in (ent.get("palettes") or [])}
        ent_shapes = {str(s.get("shape_id")) for s in (ent.get("_sub_bitmaps") or [])
                      if s.get("shape_id")}
        palette_ok = bool(kb_pals & ent_pals) if kb_pals else False
        shape_ok   = bool(kb_shapes & ent_shapes) if kb_shapes else False
        if palette_ok or shape_ok:
            continue
        # Neither palette nor shape supports this role -- demote.
        # Emit an "unknown" candidate at soft tier so the resolver
        # picks it over the VLM's original rotator/launcher/etc guess
        # (vlm tier).  Stronger evidence (causal/bitmap/region/shape)
        # would still outrank this demotion, which is correct.
        add_role_candidate(
            ent, "unknown", "soft",
            f"demoted from {role!r}: palettes {sorted(ent_pals)} don't "
            f"intersect KB palettes {sorted(kb_pals)}; no shape match"
        )
        ent["role"] = "unknown"
        ent.setdefault("_corrections", []).append(
            f"role {role!r} demoted to 'unknown': palettes {sorted(ent_pals)} "
            f"don't intersect KB palettes for that role {sorted(kb_pals)} and "
            f"no shape match"
        )
        n_demoted += 1
    return n_demoted


def assign_role_credence(parsed: Mapping[str, Any]) -> Dict[str, int]:
    """Assign a ``role_credence`` field to every non-region entity.

    Credence levels reflect how the role was reached:

    * ``confirmed`` -- KB exact-bitmap OR shape-id match (pixel-level
      evidence; same sprite, possibly recoloured).
    * ``tentative`` -- KB topo match, KB soft match, or VLM-only role
      with no pixel-level confirmation.  The label should be displayed
      with a ``?`` suffix and the entity is a candidate for curiosity-
      directed exploration.

    Also populates ``parsed["curiosity_targets"]`` with the indices of
    entities marked tentative, so a planner can iterate them.
    Returns a count dict {"confirmed": N, "tentative": M}.
    """
    counts = {"confirmed": 0, "tentative": 0}
    curiosity_targets: List[int] = []
    for i, ent in enumerate(parsed.get("entities") or []):
        if not isinstance(ent, dict):
            continue
        role = str(ent.get("role") or "")
        if (not role
                or role in _REGION_ROLES
                or role == "unknown"
                or role in _NON_ROLE_LABELS):
            ent["role_credence"] = "n/a"
            continue
        corrections = [str(c) for c in (ent.get("_corrections") or [])]
        # Prefer the resolver's tier when it ran; fall back to
        # correction-string scanning for snapshots that pre-date the
        # candidate refactor.
        tier = ent.get("_role_resolved_tier")
        if tier in ("causal", "bitmap", "region", "shape"):
            confirmed = True
        elif tier in ("topo", "soft", "vlm"):
            confirmed = False
        else:
            confirmed = any(
                ("KB bitmap match" in c)
                or ("KB shape match" in c)
                or ("KB region match" in c)
                or ("Bounce-plate match" in c)
                or ("confirmed agent role" in c)
                or ("confirmed " in c and "via observed trigger function" in c)
                or ("observed PERMANENT appearance change" in c)
                or ("visiting this entity caused another entity to change" in c)
                or ("CONFIRMED by interaction" in c)
                for c in corrections
            )
        if confirmed:
            ent["role_credence"] = "confirmed"
            counts["confirmed"] += 1
        else:
            ent["role_credence"] = "tentative"
            counts["tentative"] += 1
            curiosity_targets.append(i)
    if isinstance(parsed, dict):
        parsed["curiosity_targets"] = curiosity_targets
    return counts


def apply_kb_region_roles(
    parsed:  Mapping[str, Any],
    ws:      "WorldState",
) -> int:
    """Match HUD-class entities to ``RegionPaletteClaim`` priors in
    ``ws`` -- a coarse role-recognition tier that survives even when
    the entity's primary bitmap drifts (e.g. a budget_meter bar
    shrinking during play).

    Region-tier priors live in the engine's hypothesis store as
    :class:`RegionPaletteClaim` instances with optional
    ``row_range`` and ``spatial_zone`` metadata.  The caller is
    expected to have loaded those priors via
    ``context_memory.load_committed_hierarchy(ws, kb_root, game_id,
    level_id)`` before invoking this matcher -- no filesystem reads
    happen here.

    Match criteria (all three must hold):

    * bbox row range overlaps the claim's ``row_range`` by at least
      75% of the claim's row-range height,
    * the entity's palette set is a superset of, or shares >=2 palettes
      with, the claim's palette set,
    * the entity's spatial_zone equals the claim's zone (when both
      are known).
    """
    from ..claims import RegionPaletteClaim
    region_priors: List[Tuple[RegionPaletteClaim, float]] = []
    for h in ws.hypotheses.values():
        if not isinstance(h.claim, RegionPaletteClaim):
            continue
        if h.claim.role in _REGION_ROLES:
            # Skip background / wall / floor priors -- those are
            # region-LABELING claims, not HUD-role claims, and the
            # bitmap-role tier handles them separately.
            continue
        if h.claim.row_range is None:
            continue
        region_priors.append((h.claim, float(h.credence.point)))
    if not region_priors:
        return 0

    n_matches = 0
    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        role = str(ent.get("role") or "")
        if role in _REGION_ROLES:
            continue
        bbox = ent.get("bbox_pixels")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        ent_r0, _, ent_r1, _ = (int(x) for x in bbox)
        ent_pals = {int(p) for p in (ent.get("palettes") or [])}
        ent_zone = (ent.get("properties") or {}).get("spatial_zone")
        # Don't downgrade a role that was already confirmed via the
        # pixel-level matchers -- region match is a fallback, not an
        # override of stronger evidence.
        already_confirmed = any(
            ("KB bitmap match" in c) or ("KB shape match" in c)
            for c in (ent.get("_corrections") or [])
        )
        if already_confirmed:
            continue

        best: Optional[Tuple[float, str, List[str]]] = None
        for claim, _cred in region_priors:
            r0, r1 = claim.row_range  # type: ignore[misc]
            d_height = max(1, int(r1) - int(r0) + 1)
            overlap = max(0, min(ent_r1, int(r1)) - max(ent_r0, int(r0)) + 1)
            if overlap < 0.75 * d_height:
                continue
            d_pals = {int(p) for p in claim.palettes}
            shared = ent_pals & d_pals
            if len(shared) < min(2, len(d_pals)):
                if not (len(d_pals) <= 1 and len(shared) >= 1):
                    continue
            d_zone = claim.spatial_zone
            zone_ok = (not d_zone or not ent_zone or ent_zone == d_zone)
            if not zone_ok:
                continue
            score = (overlap / d_height) * 2.0 + len(shared) * 1.0
            reasons = [f"rows {ent_r0}-{ent_r1}~{r0}-{r1}",
                       f"shared pals {sorted(shared)}"]
            if d_zone and ent_zone:
                reasons.append(f"zone={ent_zone}")
            if best is None or score > best[0]:
                best = (score, claim.role, reasons)

        if best is None:
            continue
        _score, new_role, reasons = best
        add_role_candidate(
            ent, new_role, "region",
            f"KB region match: {'; '.join(reasons)}",
        )
        old_role = ent.get("role")
        if old_role == new_role:
            ent.setdefault("_corrections", []).append(
                f"KB region match: confirms role={new_role} "
                f"({'; '.join(reasons)})"
            )
        else:
            ent["role"] = new_role
            ent.setdefault("_corrections", []).append(
                f"KB region match: role={new_role} "
                f"({'; '.join(reasons)}; overrides {old_role!r})"
            )
        n_matches += 1
    return n_matches


def propose_hypotheses_for_unknowns(
    parsed:  Mapping[str, Any],
    kb_path: Path,
) -> int:
    """For entities still labelled ``unknown`` after exact KB bitmap /
    shape / topo matching, propose a role hypothesis from SOFT signals
    that mirror how human vision combines weak cues:

    * **size proximity** -- sub-bitmap pixel size matches a KB role's
      recorded size (the strongest soft cue).
    * **palette compatibility** -- either the sub-bitmap palette is the
      same as a KB role's palette, OR the KB role's palette appears in
      the parent entity's frame palettes.
    * **spatial zone consistency** -- the entity sits in the same
      9-zone bin (corner/side/centre) where this role was previously
      observed.  Zones generalise across levels much better than exact
      bboxes.

    A candidate role must reach ``score >= 2.0`` (about two cues lining
    up) to be proposed.  The contributing cues are recorded in the
    correction line so the hypothesis is auditable.
    """
    if not kb_path.exists():
        return 0
    try:
        kb = json.loads(kb_path.read_text(encoding="utf-8"))
    except Exception:
        return 0
    lf  = kb.get("learned_facts") or {}
    crm = lf.get("component_role_meta") or {}
    # Wall palettes across all known levels -- candidates whose palette
    # is in this set are floor / wall artefacts, not role-carrying
    # sprites, and soft match should never propose a role for them.
    wall_set: set = set()
    wbl = kb.get("wall_palette_by_level") or {}
    for lvl_pals in wbl.values():
        if isinstance(lvl_pals, list):
            wall_set.update(int(p) for p in lvl_pals)
    # Build {role -> list of meta dicts}.
    role_signatures: Dict[str, List[Dict[str, Any]]] = {}
    for meta in crm.values():
        if not isinstance(meta, dict):
            continue
        role = meta.get("role")
        if not role or role not in _TRANSFERABLE_BITMAP_ROLES:
            continue
        role_signatures.setdefault(role, []).append(meta)

    if not role_signatures:
        return 0

    n_proposed = 0
    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        if str(ent.get("role") or "") != "unknown":
            continue
        subs = ent.get("_sub_bitmaps") or []
        if not subs:
            continue
        ent_palettes = set(int(p) for p in (ent.get("palettes") or []))
        ent_zone     = (ent.get("properties") or {}).get("spatial_zone")

        # Wall-palette guard: entities whose palettes intersect any
        # known wall palette are level structure (chamber outlines,
        # divider strips), not role-bearing sprites.  Don't propose.
        if wall_set and (ent_palettes & wall_set):
            continue

        best_role: Optional[str] = None
        best_score = 0.0
        best_reasons: List[str] = []

        for role, metas in role_signatures.items():
            for s in subs:
                sub_size = int(s.get("size_px") or 0)
                sub_pals = [int(p) for p in (s.get("palettes") or [])]
                for meta in metas:
                    kb_sz   = meta.get("size_px")
                    kb_pals = meta.get("palettes") or []
                    kb_pal  = kb_pals[0] if kb_pals else None
                    kb_zone = meta.get("spatial_zone")
                    score = 0.0
                    reasons: List[str] = []
                    # Size proximity.
                    if kb_sz is not None and sub_size > 0:
                        if abs(sub_size - kb_sz) <= 1:
                            score += 2.0
                            reasons.append(f"size {sub_size}=={kb_sz}")
                        elif kb_sz and abs(sub_size - kb_sz) / kb_sz <= 0.25:
                            score += 1.0
                            reasons.append(f"size {sub_size}~{kb_sz}")
                    # Palette compatibility.
                    if kb_pal is not None:
                        if kb_pal in sub_pals:
                            score += 1.0
                            reasons.append(f"pal {kb_pal} matches")
                        elif kb_pal in ent_palettes:
                            score += 1.0
                            reasons.append(f"pal {kb_pal} in frame")
                        elif sub_pals:
                            # Different palette in inner shape AND we
                            # already have a strong size match -- treat
                            # as a recolour candidate (no score, just a
                            # note).
                            if score >= 2.0:
                                reasons.append(
                                    f"pal {sub_pals[0]}!={kb_pal} (recoloured?)")
                    # Spatial zone consistency.
                    if ent_zone and kb_zone and ent_zone == kb_zone:
                        score += 1.0
                        reasons.append(f"zone {ent_zone}")
                    if score > best_score:
                        best_score   = score
                        best_role    = role
                        best_reasons = reasons[:]
        if best_role is None or best_score < 2.0:
            continue
        add_role_candidate(
            ent, best_role, "soft",
            f"KB soft match: score={best_score:.1f} ({', '.join(best_reasons)})",
        )
        old_role = ent.get("role")
        ent["role"] = best_role
        ent.setdefault("_corrections", []).append(
            f"KB soft match: hypothesised role={best_role} "
            f"(score={best_score:.1f}: {', '.join(best_reasons)}; "
            f"overrides {old_role!r})"
        )
        n_proposed += 1
    return n_proposed


def _bitmap_role_indices_from_ws(
    ws: "WorldState",
    *,
    cross_game: bool = False,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    """Walk ``ws.hypotheses`` for committed :class:`BitmapRoleClaim`
    instances and build three lookup tables:

    * ``bm_to_role``     : {bitmap_id -> role}  -- exact-bitmap tier
    * ``shape_to_role``  : {shape_id  -> role}  -- recolour tier
    * ``topo_to_role``   : {topo_id   -> role}  -- topology tier

    Replaces the legacy ``_load_kb_role_index`` reader of
    ``<game>_runtime.json``.  All four indices come from claims in
    ``ws``, which the caller is expected to have populated via
    ``context_memory.load_committed_hierarchy(ws, kb_root, game_id,
    level_id)``.

    When the same shape_id appears across BitmapRoleClaims with
    different roles, FIRST one wins (matches the legacy `setdefault`
    semantics).  In practice this rarely happens -- shape_id is
    discriminating enough within a game -- and when it does, the
    higher-credence claim should be ordered first by the caller.

    ``cross_game=True`` drops the shape and topo indices entirely:
    silhouette / topology don't transfer reliably across games, so
    only exact bitmap match (pixel-identical sprites) is reliable.
    """
    from ..claims import BitmapRoleClaim
    bm_to_role:    Dict[str, str] = {}
    shape_to_role: Dict[str, str] = {}
    topo_to_role:  Dict[str, str] = {}
    # Sort by credence descending so the highest-credence claim wins
    # the shape/topo `setdefault` race when several BitmapRoleClaims
    # share a shape_id.
    bitmap_claims = sorted(
        (h for h in ws.hypotheses.values()
         if isinstance(h.claim, BitmapRoleClaim)),
        key=lambda h: -float(h.credence.point),
    )
    for h in bitmap_claims:
        claim = h.claim
        if not claim.role:
            continue
        bm_to_role.setdefault(str(claim.bitmap_id), claim.role)
        if cross_game:
            continue
        if claim.shape_id:
            shape_to_role.setdefault(str(claim.shape_id), claim.role)
        if claim.topo_id:
            topo_to_role.setdefault(str(claim.topo_id),  claim.role)
    return bm_to_role, shape_to_role, topo_to_role


def apply_kb_bitmap_roles(
    parsed: Mapping[str, Any],
    ws:     "WorldState",
    *,
    cross_game: bool = False,
) -> int:
    """Override entity roles whose sub-bitmaps match a KB-confirmed
    bitmap-role mapping in ``ws``, with three matching tiers:

    1. **Exact bitmap_id** -- identical pixel pattern.
    2. **shape_id** -- same canonicalised silhouette in a different
       palette (recolour of a known role).  This is the "human can tell
       it's the same thing despite the colour change" tier.
    3. **topo_id** -- same connectivity topology, possibly at a
       different scale.  Loosest deterministic tier.

    Highest-priority match wins.  The matching tier is recorded in the
    correction string so downstream consumers know how confident to be.

    ``cross_game=True`` restricts to the exact-bitmap tier only --
    shape and topo signatures are reused freely across games (similar
    silhouettes carry different meanings in different games) so the
    fallback tiers would produce visual-coincidence false positives.
    """
    bm_to_role, shape_to_role, topo_to_role = _bitmap_role_indices_from_ws(
        ws, cross_game=cross_game,
    )
    if not (bm_to_role or shape_to_role or topo_to_role):
        return 0

    import re
    bm_re   = re.compile(r"bm=(bm_[0-9a-f]+)")
    size_re = re.compile(r"size=(\d+)px")
    n_overrides = 0

    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        if str(ent.get("role") or "") in _REGION_ROLES:
            continue
        # Scan the structured _sub_bitmaps first; fall back to the legacy
        # string corrections so older snapshots still work.
        subs: List[Dict[str, Any]] = list(ent.get("_sub_bitmaps") or [])
        if not subs:
            for line in ent.get("_corrections") or []:
                bm_m = bm_re.search(str(line))
                sz_m = size_re.search(str(line))
                if bm_m:
                    subs.append({
                        "bitmap_id": bm_m.group(1),
                        "size_px":   int(sz_m.group(1)) if sz_m else 0,
                        "shape_id":  None, "topo_id": None,
                    })
        if not subs:
            continue
        # Skip if region tier already matched -- region descriptors
        # (HUD strips, etc.) are more specific than a coincidental
        # shape/topo collision and should not be overridden by them.
        already_region_matched = any(
            ("KB region match" in c)
            for c in (ent.get("_corrections") or [])
        )
        if already_region_matched:
            continue
        # Match each sub, prefer the LARGEST sub at the strongest tier.
        # Tier priority: bitmap > shape > topo.  Within a tier, pick the
        # sub with the largest size_px (entity's defining shape).
        # Shape and topo tiers REQUIRE a minimum sub-bitmap size to
        # avoid trivial collisions: tiny bitmaps (1-4 px) canonicalize
        # to very few unique shape_ids and will spuriously match
        # unrelated roles across the KB.
        MIN_SHAPE_SIZE = 5
        best: Optional[Tuple[int, str, str, str]] = None  # (tier_rank, tier_name, role, bm_id)
        TIERS = [("bitmap", bm_to_role), ("shape", shape_to_role), ("topo", topo_to_role)]
        for tier_rank, (tier_name, table) in enumerate(TIERS):
            for s in subs:
                if tier_name == "bitmap":
                    key = s.get("bitmap_id")
                elif tier_name == "shape":
                    key = s.get("shape_id")
                else:
                    key = s.get("topo_id")
                if key and key in table:
                    size_px = int(s.get("size_px") or 0)
                    # Reject trivial-shape collisions: a 1-4 px sprite
                    # has too few possible canonical shapes for a
                    # shape/topo match to be discriminating.  Bitmap
                    # tier is exempt because exact-pixel match implies
                    # the same sprite regardless of size.
                    if tier_name in ("shape", "topo") and size_px < MIN_SHAPE_SIZE:
                        continue
                    new_role = table[key]
                    if (best is None
                            or tier_rank < best[0]
                            or (tier_rank == best[0] and size_px > _sub_size(best, subs))):
                        best = (tier_rank, tier_name, new_role,
                                str(s.get("bitmap_id") or key))
                        break  # take first per tier; size break-tie below
            if best and best[0] == tier_rank:
                break  # already found at this tier; don't degrade to weaker tier
        if best is None:
            continue
        _, tier_name, new_role, key = best
        add_role_candidate(
            ent, new_role, tier_name,
            f"KB {tier_name} match on {key}",
        )
        old_role = ent.get("role")
        ent.setdefault("_corrections", []).append(
            f"KB {tier_name} match: {key} -> role={new_role}"
            + (f" (overrides VLM's {old_role!r})" if old_role != new_role else "")
        )
        if old_role != new_role:
            n_overrides += 1
    return n_overrides


def _sub_size(best_tuple: Tuple[int, str, str, str],
              subs:       Sequence[Mapping[str, Any]]) -> int:
    """Helper: look up size_px of the sub-bitmap that produced `best`."""
    _, _, _, key = best_tuple
    for s in subs:
        if s.get("bitmap_id") == key:
            return int(s.get("size_px") or 0)
    return 0


def writeback_learned_roles_from_parsed(
    parsed:  Mapping[str, Any],
    ws:      "WorldState",
    *,
    scope:   Optional["Scope"] = None,
    step:    int               = 0,
    level_label: str           = "",
) -> Dict[str, str]:
    """Propose confirmed bitmap->role and HUD-region role mappings
    from a finalised parsed snapshot into the hypothesis store on
    ``ws``.

    Replaces the legacy writer that mutated ``<game>_runtime.json``
    directly.  Each confirmed entity becomes a
    :class:`BitmapRoleClaim` (with shape_id / topo_id / size_px /
    spatial_zone metadata) and each HUD-class entity becomes a
    :class:`RegionPaletteClaim` (with row_range / spatial_zone
    metadata).  Persisting these claims to disk is the caller's job
    via ``context_memory.save_committed_hierarchy``.

    Selection criteria (unchanged from the legacy writer):

    * Role must be in ``_TRANSFERABLE_BITMAP_ROLES`` (bitmap tier) or
      ``_HUD_CLASS_ROLES`` (region tier).
    * For the bitmap tier the entity's ``role_credence`` must be
      ``"confirmed"`` -- tentative VLM / soft-match guesses pollute
      the KB and cause spurious shape/palette matches in future
      levels.
    * The primary signature is the LARGEST non-primary sub-bitmap
      (the inner glyph that defines the role).  Fall back to the
      primary track when no distinct inner sub-bitmap exists.

    Returns a ``{bitmap_id: description}`` dict matching the legacy
    return shape, for callers that log this.  Returns ``{}`` when no
    eligible entities were found.
    """
    from ..claims import BitmapRoleClaim, RegionPaletteClaim
    from ..hypothesis_store import propose as _propose
    from ..types import Scope as _Scope, ScopeKind as _ScopeKind

    if scope is None:
        scope = _Scope(kind=_ScopeKind.GAME)

    import re
    sub_re = re.compile(
        r"bm=(bm_[0-9a-f]+),\s*bbox=\[[^\]]+\],\s*size=(\d+)px,\s*pal=\[(\d+)\]"
    )
    out: Dict[str, str] = {}

    # ----- bitmap-tier claims -----
    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        role = str(ent.get("role") or "").strip()
        if role not in _TRANSFERABLE_BITMAP_ROLES:
            continue
        if ent.get("role_credence") and ent["role_credence"] != "confirmed":
            continue
        properties = ent.get("properties") or {}
        zone = properties.get("spatial_zone")

        subs = ent.get("_sub_bitmaps") or []
        best_struct: Optional[Dict[str, Any]] = None
        if subs:
            non_primary = [s for s in subs if not s.get("is_primary")]
            pool = non_primary if non_primary else subs
            for s in pool:
                if (best_struct is None
                        or s.get("size_px", 0) > best_struct.get("size_px", 0)):
                    best_struct = s
        if best_struct is None:
            best: Optional[Tuple[int, str, int]] = None
            for line in ent.get("_corrections") or []:
                m = sub_re.search(str(line))
                if not m:
                    continue
                bm_id, size_s, pal_s = m.group(1), int(m.group(2)), int(m.group(3))
                if best is None or size_s > best[0]:
                    best = (size_s, bm_id, pal_s)
            if best is None:
                continue
            size_px_v, bm_id, pal_v = best
            best_struct = {
                "bitmap_id": bm_id, "size_px": size_px_v,
                "palettes":  [pal_v],
                "shape_id":  None, "topo_id": None, "scaled_id": None,
            }

        bm_id   = str(best_struct["bitmap_id"])
        size_px = int(best_struct.get("size_px") or 0)
        pals    = best_struct.get("palettes") or []
        primary_pal = pals[0] if pals else None
        lvl = f"L{level_label} " if level_label else ""
        parts = [f"pal={primary_pal}" if primary_pal is not None else None,
                 f"sz={size_px}"]
        if zone:
            parts.append(f"zone={zone}")
        desc = f"{lvl}role={role} ({' '.join(p for p in parts if p)})"
        out[bm_id] = desc

        claim = BitmapRoleClaim(
            bitmap_id    = bm_id,
            role         = role,
            shape_id     = (str(best_struct["shape_id"])
                            if best_struct.get("shape_id") is not None else None),
            topo_id      = (str(best_struct["topo_id"])
                            if best_struct.get("topo_id")  is not None else None),
            size_px      = size_px or None,
            spatial_zone = str(zone) if zone else None,
        )
        _propose(
            ws,
            claim            = claim,
            source           = f"perception:writeback:L{level_label}"
                               if level_label else "perception:writeback",
            scope            = scope,
            step             = step,
            initial_credence = 0.92,
        )

    # ----- region-tier claims (HUD-class roles) -----
    for ent in parsed.get("entities") or []:
        if not isinstance(ent, dict):
            continue
        role = str(ent.get("role") or "").strip()
        if role not in _HUD_CLASS_ROLES:
            continue
        bbox = ent.get("bbox_pixels")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            continue
        r0, _, r1, _ = (int(x) for x in bbox)
        pals = sorted({int(p) for p in (ent.get("palettes") or [])})
        zone = (ent.get("properties") or {}).get("spatial_zone")
        region_claim = RegionPaletteClaim.make(
            palettes     = pals,
            role         = role,
            row_range    = (r0, r1),
            spatial_zone = str(zone) if zone else None,
        )
        _propose(
            ws,
            claim            = region_claim,
            source           = f"perception:writeback:L{level_label}"
                               if level_label else "perception:writeback",
            scope            = scope,
            step             = step,
            initial_credence = 0.86,
        )

    return out


def memory_from_gt_files(
    paths: Sequence[Path],
    *,
    game: str,
    level_labels: Optional[Sequence[str]] = None,
) -> GameMemory:
    """Load GTs from disk and build a GameMemory.

    `level_labels` parallels `paths` (e.g. ["1/7", "2/7"]).  When omitted,
    we use the parent dir name of each path.
    """
    if level_labels is None:
        level_labels = [Path(p).parent.name for p in paths]
    played: List[Dict[str, Any]] = []
    for p, lab in zip(paths, level_labels):
        gt = json.loads(Path(p).read_text(encoding="utf-8"))
        played.append({"level": lab, "ground_truth": gt})
    return build_game_memory(played, game=game)
