"""Build the Layer B perception prompt.

The prompt has four parts:

1. **Preamble** — sets the VLM's task: identify entities, infer roles,
   propose relationships and a likely win condition.

2. **Operational primitives catalog** — the recognition hints from
   every catalog entry, organised by primitive_kind.  This is the
   cross-game vocabulary the VLM uses to classify what it sees.

3. **Frame context** — palette histogram, frame size, agent-position
   hint when available.  Stays minimal; the image is the source of
   truth.

4. **Output schema** — explicit JSON structure the VLM must produce.

The image is supplied separately by the VLM client; this module
returns the text portion plus a list of recognition hints to embed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence

from .catalog_loader import Catalog, CatalogEntry, load_catalog
from .geometry import GeometryResult, format_candidates_for_prompt


_PREAMBLE = """\
You are a vision-language model analysing an ARC-AGI-3 game frame.

A deterministic geometry extractor (Layer A) has already identified
every visually-distinct entity candidate on the frame and produced
a NUMBERED LIST with exact pixel positions.  Your job is to:

1. Assign a role (from the catalog) to each numbered candidate.
2. Propose GROUPS of candidates that together form one logical
   entity (e.g. three life-indicator dots, a bracket made of two
   non-touching pieces).
3. Identify candidates that should be discarded as noise.
4. Identify relationships between entities (visual similarity,
   complementary shape, paired members, etc.).
5. Hypothesise the most likely win condition.

You do NOT invent positions.  All bboxes come from Layer A; you
reference candidates by their numeric id.

You have the OPERATIONAL PRIMITIVES CATALOG below — recognition
hints for the typical primitives the ARC-AGI-3 family uses.  Use
catalog primitive_id strings as your role / relationship / match
vocabulary.  When something doesn't fit any catalog primitive,
mark its role as "unknown" rather than inventing.

Important:

* Palette-defined regions (wall, play_area, void, floor) often
  appear as large irregular shapes — sometimes one connected blob,
  sometimes many disconnected fragments.  These are ONE conceptual
  entity, not many.  Recognise them by: multiple candidates share
  the SAME palette and together they cover a large/spread area.
  Handle them as follows — and DO NOT try harder than this:
    1. Tag each fragment candidate with the same role (wall, etc.)
       via candidate_assignments.
    2. Do NOT create groups, relationships, or per-fragment notes
       for these region candidates.  Move on.
  In particular: do not search for "the right pair" of wall fragments
  or assign them distinct sub-roles.  Walls and floors are infrastructure,
  not gameplay entities.
* When several candidates together form one logical entity (e.g.
  a bracket = two non-touching hooks, a life_indicator = three
  separate dot pairs, a piercer's head + striped tail extending
  from it), put them in a SINGLE group and assign the role at the
  GROUP level.  Layer A may split a multi-part sprite into its
  per-palette pieces — your job is to re-assemble them by visual
  cohesion (touching or near-touching bboxes that together form a
  recognisable sprite).
* "noise" means a PNG-conversion artefact, not "small".  A 5-pixel
  candidate may be the most important entity on the frame (e.g. a
  static reference glyph the mutable working_glyph must match).
  Before marking anything "noise" ask: could this be the static
  reference for an alignment puzzle?  If yes, tag it appropriately.
* The agent_avatar is the entity the player CONTROLS — often
  multi-colour or otherwise visually distinct from the puzzle's
  paired entities.  Do not confuse it with reference_glyph: the
  reference is small, monochrome, and visually-matched to the
  working_glyph elsewhere on the frame.
* For relationships, scan every PAIR of non-noise entities and ask
  "do these look related?".  Strong signals:
    - Same shape / silhouette at different sizes  -> working_reference_pair
    - Identical-looking duplicates                -> reference_pair
    - Three or more collinear like-items          -> collinear_triple
    - One large + one small with matching outline -> complementary_shape_pair
    - One device whose head/tail are distinct     -> device_head_tail
  When in doubt, emit the relationship.  A spurious relationship is
  cheaper than a missed one; both are penalised, but missing the
  game's central pairing kills the win-condition hypothesis.
* Each relationship's "members" list MUST use the SAME ids you used
  in candidate_assignments / groups — bare candidate-id integers (as
  strings) for ungrouped candidates, or the group_id label for
  grouped entities.  Do NOT invent new names.
"""


_OUTPUT_SCHEMA = """\
Output a single JSON object with this shape:

{
  "candidate_assignments": {
    "<candidate_id_int>": {
      "role":       "<primitive_id from catalog entity_role, or 'unknown', or 'noise'>",
      "group":      "<group label or null>",
      "related_to": "<another candidate_id or group label, or null>",
      "notes":      "<short rationale>"
    },
    ...
  },
  "groups": [
    {
      "group_id":   "<short label>",
      "members":    [<candidate_id_int>, ...],
      "role":       "<primitive_id from catalog entity_role>",
      "rationale":  "<why these candidates are one logical entity>"
    },
    ...
  ],
  "relationships": [
    {
      "kind":      "<primitive_id from catalog relationship_kind>",
      "members":   ["<candidate_id or group_id>", ...],
      "rationale": "<short>"
    },
    ...
  ],
  "win_condition_hypothesis": {
    "kind":        "<primitive_id from catalog match_condition>",
    "description": "<short prose>",
    "involves":    ["<candidate_id or group_id>", ...]
  },
  "uncertainty_notes": "<anything you are not sure about>"
}

Return only this JSON, no surrounding prose.

Coverage rule: every candidate in the Layer A list MUST have an
entry in candidate_assignments.  Use role="noise" if you think the
candidate is a PNG-conversion artefact.
"""


@dataclass
class FrameContext:
    """Operator-supplied frame metadata that complements the image."""
    sample_name:       str
    palette_histogram: Mapping[int, int]     # palette -> pixel count
    frame_size:        tuple = (64, 64)
    cell_system:       Optional[Mapping[str, Any]] = None
    operational_notes: str = ""              # optional sample-specific operational hints


def _format_catalog_hints(catalog: Catalog) -> str:
    """Render the catalog's recognition hints as compact prompt text."""
    sections: List[str] = []
    kind_order = ("entity_role", "interaction", "match_condition",
                  "relationship_kind", "state_change_effect")
    for kind in kind_order:
        entries = catalog.by_kind.get(kind, [])
        if not entries:
            continue
        section_lines: List[str] = [f"\n## {kind}"]
        for e in sorted(entries, key=lambda x: x.primitive_id):
            section_lines.append(f"\n### {e.primitive_id}")
            section_lines.append(e.description.strip())
            for hint in e.vlm_recognition_hints:
                section_lines.append(f"- {hint.strip()}")
        sections.append("\n".join(section_lines))
    return "# Operational primitives catalog\n" + "\n".join(sections)


def _format_frame_context(ctx: FrameContext) -> str:
    lines: List[str] = ["# Frame context"]
    lines.append(f"Sample: {ctx.sample_name}")
    lines.append(f"Frame size: {ctx.frame_size[0]} x {ctx.frame_size[1]} pixels")
    if ctx.cell_system:
        lines.append(
            f"Cell system: origin={ctx.cell_system.get('origin')}, "
            f"cell_size={ctx.cell_system.get('cell_size')}"
        )
    else:
        lines.append("Cell system: none (pixel-coordinate game)")
    hist = sorted(ctx.palette_histogram.items(), key=lambda kv: -kv[1])
    hist_str = ", ".join(f"p{p}={n}" for p, n in hist)
    lines.append(f"Palette pixel counts: {hist_str}")
    if ctx.operational_notes:
        lines.append("\nOperator notes:\n" + ctx.operational_notes.strip())
    return "\n".join(lines)


def build_prompt(
    ctx: FrameContext,
    *,
    catalog:           Optional[Catalog] = None,
    geometry:          Optional[GeometryResult] = None,
    similarity_pairs:  Optional[Any] = None,
) -> str:
    """Assemble the complete Layer B text prompt for the VLM.

    The frame image is passed by the VLM client alongside this text.
    When ``geometry`` is supplied, the Layer A candidate list is
    embedded in the prompt; otherwise the VLM is asked to invent
    positions (less accurate — only useful for sanity checks).
    When ``similarity_pairs`` is supplied (a list of
    ``cognitive_os.perception.similarity.SimilarityPair``), a section
    listing structurally-similar pairs is appended — strong hints for
    game-mechanical pairing the VLM should propose as relationships.
    """
    cat = catalog if catalog is not None else load_catalog()
    parts = [
        _PREAMBLE,
        _format_catalog_hints(cat),
        _format_frame_context(ctx),
    ]
    if geometry is not None:
        parts.append(format_candidates_for_prompt(geometry))
    if similarity_pairs:
        from .similarity import format_pairs_for_prompt
        sim_text = format_pairs_for_prompt(similarity_pairs)
        if sim_text:
            parts.append(sim_text)
    parts.append(_OUTPUT_SCHEMA)
    return "\n\n".join(p.strip() for p in parts) + "\n"


if __name__ == "__main__":
    # Sanity check: render the prompt for ls20_4of7 with a stub context.
    ctx = FrameContext(
        sample_name = "ls20_4of7",
        palette_histogram = {3: 1500, 5: 800, 1: 40, 11: 24, 0: 16},
        cell_system = {"origin": [5, 56], "cell_size": 5},
        operational_notes = "Cardinal navigation; some entities teleport the agent on contact.",
    )
    print(build_prompt(ctx))
