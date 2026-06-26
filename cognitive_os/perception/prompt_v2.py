"""ID-grounded VLM prompt builder.

The substrate has already done its best to identify candidates --
Layer A geometry, tracker observations from exploration, structural
unique-palette promotion, sub-bitmap promotion.  Each candidate has
a stable integer id that's ALSO rendered onto the annotated frame
image.  This prompt asks the VLM to:

  * Assign a catalog role to every candidate id.
  * Identify the background palette.
  * Group related candidates.
  * Propose a win_condition that references candidate ids
    (not free-form prose).

The key contract with the VLM: every reference in the output JSON
MUST be a candidate id from the input list.  No invented labels,
no prose-only references.  This makes the response directly
actionable by the substrate.
"""
from __future__ import annotations

from typing import List, Mapping, Sequence

from .catalog_loader import Catalog


_INTRO = """\
You are looking at an ARC-AGI-3 game frame that the substrate has
already annotated for you.  Each entity it identified is outlined
in a distinct color and labeled with an INTEGER ID (1, 2, 3, ...).
A behavioral observation digest follows below.

Your job has two parts:

1. For each candidate id, pick the best matching role from the
   catalog.  Describe the game's win condition, referencing
   candidates BY ID.

2. If the digest reports parallel structures (multiple cells that
   look similar with one OUTLIER), do cross-instance reasoning.

   STEP-BY-STEP procedure (do this faithfully -- guesses without
   per-cell inspection are usually wrong):

   a) For EACH inlier cell INDIVIDUALLY, look at the annotated
      frame and describe its two layers in concrete spatial terms:
        - central sub-region: what pixel pattern is INSIDE it?
          (e.g. "small 3x3 icon with dark cells at top-edge and
          left-edge and bottom-edge, red at centre")
        - peripheral arrangement: which positions are which
          colour? (e.g. "blue at corners (0,0), (0,2), (2,0),
          (2,2); red at edges (0,1), (1,0), (1,2), (2,1)")

   b) Look for a TRANSFORMATION that maps the central sub-region's
      pattern to the peripheral arrangement.  Plausible rules:
      "the central icon at scale 1/N IS the peripheral pattern",
      "the central icon is the negation/mirror/rotation of the
      peripheral", "the central icon labels the peripheral
      colours".  Test the rule on EACH inlier individually: do
      ALL of them satisfy the same transformation?  If yes, you
      have the rule.

   c) IMPORTANT: the inlier cells will typically have DIFFERENT
      peripheral arrangements from each other.  The rule is what
      MAKES each inlier internally consistent, not what makes
      them identical.  Do not propose "make outlier match an
      inlier" -- that's almost always wrong.  Propose "make
      outlier internally consistent under the same rule".

   d) Apply the rule to the outlier's central sub-region to
      DERIVE what its peripheral arrangement should look like.
      Write the target out in concrete spatial language --
      which positions must be which colour.

Hard rule: every id you mention in the output JSON must appear in
the candidate list provided.  Do not invent ids or labels.  When
unsure about the parallel-structure rule, set confidence='low'
rather than guessing -- the harness can still act on a confident
'no rule found' result.
"""


_OUTPUT_SCHEMA = """\
Output a single JSON object with this shape:

{
  "candidate_assignments": {
    "<id>": {
      "role":      "<primitive_id from catalog entity_role, or 'unknown', or 'noise'>",
      "rationale": "<one-sentence reason>"
    },
    ...
  },
  "background_palettes": [<int>, ...],
  "groups": [
    {
      "group_id":  "<short label>",
      "members":   ["<id>", ...],
      "role":      "<primitive_id from catalog entity_role>",
      "rationale": "<short>"
    }
  ],
  "relationships": [
    {
      "kind":      "<primitive_id from catalog relationship_kind>",
      "members":   ["<id or group_id>", ...],
      "rationale": "<short>"
    }
  ],
  "win_condition": {
    "kind":             "<primitive_id from catalog match_condition>",
    "agent_candidate":  "<id or null>",
    "target_candidate": "<id or null>",
    "other_involved":   ["<id>", ...],
    "description":      "<short prose>"
  },
  "parallel_structure_analysis": {
    "applies":            <true|false -- only true when the digest reported parallel structures>,
    "inlier_cell_ids":    ["<id>", ...],
    "outlier_cell_id":    "<id or null>",
    "consistency_rule":   "<short description of what is the SAME across inlier cells' internal structure, AND what relationship holds between an inlier cell's parts (e.g. 'the central sub-CC and the peripheral sub-CC arrangement are visually consistent', 'the marker icon's shape predicts the surrounding tile pattern', etc.)  Look closely at the annotated frame.  If no rule is visible, say so.>",
    "outlier_target_description": "<if a rule was inferred above, describe what the outlier cell's internal structure SHOULD look like to satisfy the same rule.  Use concrete spatial language: 'the (0,0), (1,0), (1,2), (2,0) outer tiles should be red, the rest blue'.>",
    "confidence":         "<high|medium|low>"
  },
  "uncertainty_notes": "<anything you are unsure about>"
}

Coverage rule: every candidate in the digest's "# Candidates"
section MUST appear in candidate_assignments.  Use role="noise"
only for PNG-conversion artefacts that aren't game entities at all.

Grounding rule: ANY id you write must be one of the candidate ids
in the input.  Win-condition agent_candidate / target_candidate
must each be a candidate id (or null if the game doesn't have that
role).  Relationship "members" list contains only candidate ids
or group_ids you defined in this output.

Return only the JSON, no surrounding prose.
"""


def _render_catalog(catalog: Catalog) -> str:
    """Compact catalog dump for the prompt.  Only the kinds the new
    schema references: entity_role, relationship_kind, match_condition."""
    parts: List[str] = ["# Catalog primitives"]
    for kind in ("entity_role", "relationship_kind", "match_condition"):
        entries = catalog.by_kind.get(kind, [])
        if not entries:
            continue
        parts.append(f"\n## {kind}")
        for e in sorted(entries, key=lambda x: x.primitive_id):
            mc = getattr(e, "motion_class", None)
            ac = getattr(e, "appearance_class", None)
            tags = []
            if mc:
                tags.append(f"motion={mc}")
            if ac:
                tags.append(f"appearance={ac}")
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            parts.append(f"- **{e.primitive_id}**{tag_str}: {e.description}")
    return "\n".join(parts)


def build_prompt(
    catalog:           Catalog,
    digest_text:       str,
    candidates:        Sequence[Mapping],
) -> str:
    """Assemble the full text prompt that accompanies the annotated frame.

    Parameters
    ----------
    catalog
        Loaded operational primitives catalog.
    digest_text
        Output of ``observation_digest.build_digest`` -- the textual
        observation digest.
    candidates
        List of candidate dicts (same dicts used to render the annotated
        frame).  Used to enumerate valid ids in the schema preamble.
    """
    valid_ids = ", ".join(str(c.get("id")) for c in candidates)
    id_block = (
        "# Valid candidate IDs\n"
        f"You may reference any of these IDs in your output: [{valid_ids}].\n"
        "No others.\n"
    )
    parts = [
        _INTRO,
        _render_catalog(catalog),
        "",
        digest_text,
        "",
        id_block,
        "",
        _OUTPUT_SCHEMA,
    ]
    return "\n".join(parts)
