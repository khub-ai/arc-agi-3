"""SYMBOLIC structure mapping (stage 2) — type sections by BEHAVIOUR and orient
the correspondence by RESPONSE, so COS discovers which of two similar structures
is the active control and which is the fixed reference instead of being told.

Two-stage structure mapping ([[pointer_two_stage_structure_mapping]]):
  stage 1 (bitmap/geometry): structure_correspondences matches same-shape twin
           sections by arrangement + size -- re-skin invariant but intra-game.
  stage 2 (symbolic, HERE): the geometric twins are ORIENTED by the measured
           response facts -- the section whose members are SETTABLE (change in
           place on their own click) is the PROGRAM / active control; its inert
           twin is the REFERENCE.  Roles come from BEHAVIOUR, never from a label
           the VLM typed, and survive a re-skin / shuffle.

The emitted claim carries a ``MATCH:active~reference+TRIGGER`` plan, so the
EXISTING match-execute-verify machinery (_structural_detect / _structural_match_
probe) runs it -- but now with active/reference grounded in measurement.  Every
claim is GUESSED and confirmed only by the SCORE.

Pure functions over {entities:{name:{role,bbox}}, groups:{name:{members}},
response_facts:{name:{settable,is_trigger}}}; no pixels, no World dependency.
"""
from __future__ import annotations

from structural_claims import structure_correspondences


def settable_fraction(response_facts, members) -> float:
    """Fraction of members measured SETTABLE (own click changed them in place)."""
    members = list(members or [])
    if not members:
        return 0.0
    rf = response_facts or {}
    return sum(1 for m in members if (rf.get(m) or {}).get("settable")) / len(members)


def scene_elements(response_facts) -> list:
    """Entities measured as DRIVEN outputs (a mover departed from them during a
    trigger's animation) -- the SCENE the program controls.  Symbolic-only, from
    the measured responds_to_fire fact; no judgement here."""
    return sorted(n for n, f in (response_facts or {}).items()
                  if (f or {}).get("responds_to_fire"))


def program_role(response_facts, members) -> str:
    """'program' when the section's members are settable inputs; 'inert' when it
    has members but none are settable (a candidate reference OR simply un-probed);
    'unknown' for an empty section.  Measured-only -- no semantic judgement."""
    members = list(members or [])
    if not members:
        return "unknown"
    return "program" if settable_fraction(response_facts, members) > 0.5 else "inert"


def orient_correspondence(corr, response_facts):
    """Orient a geometric correspondence {a, b, pairs, score} by RESPONSE.  The
    side with the higher settable-fraction is the ACTIVE program; the other is the
    REFERENCE.  Returns {active, reference, pairs(active~reference), score} or None
    when the two sides are equally settable (can't tell which to change yet --
    needs more probing; do NOT guess)."""
    a_members = [p[0] for p in corr["pairs"]]
    b_members = [p[1] for p in corr["pairs"]]
    fa = settable_fraction(response_facts, a_members)
    fb = settable_fraction(response_facts, b_members)
    if fa == fb:
        return None
    if fa > fb:
        return {"active": corr["a"], "reference": corr["b"],
                "pairs": list(corr["pairs"]), "score": corr["score"]}
    return {"active": corr["b"], "reference": corr["a"],
            "pairs": [(p[1], p[0]) for p in corr["pairs"]], "score": corr["score"]}


def propose_program_map_claims(entities, groups, response_facts, *, sim_floor=0.5):
    """For each similar-structure pair the response facts can ORIENT, emit a
    program-mapping claim: 'set the active program like its reference, then fire
    the trigger'.  active/reference are DISCOVERED from behaviour; the plan reuses
    the match executor (MATCH:active~reference+TRIGGER).  GUESSED credence scaled
    by the geometric similarity; verified only by the SCORE.  Pure; guarded by the
    caller."""
    out = []
    scene = scene_elements(response_facts)
    scene_txt = (f"  Firing drives the SCENE {scene} (measured driven outputs)."
                 if scene else "")
    for corr in structure_correspondences(entities, groups, sim_floor=sim_floor):
        o = orient_correspondence(corr, response_facts)
        if o is None:
            continue
        pair_txt = "; ".join(f"{x}~{y}" for x, y in o["pairs"])
        out.append({
            "id": f"program_map__{o['active']}__like__{o['reference']}",
            "kind": "structural", "scope": "level",
            "statement": (f"'{o['active']}' is the ACTIVE program (its members are "
                          f"measured SETTABLE) and '{o['reference']}' is its matching "
                          f"REFERENCE (same structure, members not settable); "
                          f"correspondence {pair_txt}.  Hypothesis: set "
                          f"'{o['active']}' to mirror '{o['reference']}', then fire "
                          f"the trigger.{scene_txt}  Roles discovered by response; "
                          f"TEST by score."),
            "target": ([o["active"], o["reference"]]
                       + [m for pr in o["pairs"] for m in pr] + scene),
            "plan": f"MATCH:{o['active']}~{o['reference']}+TRIGGER",
            "importance": 0.85, "credence": round(0.40 * o["score"], 2),
            "provenance": "guessed"})
    return out
