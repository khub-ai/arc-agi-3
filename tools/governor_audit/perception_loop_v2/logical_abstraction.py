"""Logical abstraction ladder — higher layers on top of the logical scene.

See docs/SPEC_logical_scene_and_regularity_instinct.md (Abstraction-ladder
section).  The logical scene (logical_scene.build_logical_scene) gives LAYER 1:
typed instances.  This module builds the layers above it by running the SAME
equivalence engine recursively at each level:

  LAYER 2  similarity groups   — instances that share a characteristic
                                 (its abstract shape-class + its colour) become
                                 ONE group object with aggregate properties
                                 (member-type, count, arrangement, bbox).
  LAYER 3  group equivalence   — two groups are recognised as the SAME ABSTRACT
           + mapping             TYPE when their member-type multiset and count
                                 coincide under the ABSTRACTING lens (colour /
                                 orientation dropped); the member correspondence
                                 is the analogy between them.
  LAYER 4+ recursion           — the abstract group-types are themselves fed
                                 back as nodes and grouped/mapped again, until a
                                 FIXED POINT (a layer that merges nothing).

Subsystems (a colour-cohesion grouping with role-tagged members: a distinguished
anchor + its satellites) and the cross-subsystem mapping make the analogy
actionable.  From every mapping the instinct layer reads candidate GOALS —
CONFORM (make the odd group match its twin), UNITE (bring corresponding members
together), TRANSFER (apply the operation that advanced one group to its mapped
twin) — which the score then judges.

Threshold-free + game-agnostic by construction: equivalence is an EQUALITY of
abstract member-type multisets (and an EXACT structural-iso refinement), never a
tuned similarity cutoff; no colours, roles, or game concepts are baked in.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np

from perception_loop_v2.perceptual_equivalence import (
    Lens, Structure, same_structure, equivalence_classes)
from perception_loop_v2.logical_scene import LogicalScene

# A FINE lens keeps colour (so a blue glyph and a yellow glyph form distinct
# groups); the ABSTRACTING lens drops colour + orientation (so those two groups
# can be recognised as one abstract type).  This pair is the whole mechanism:
# discriminate to form groups, abstract to equate them.
FINE_LENS = Lens(rotations=True, reflection=True, scale="none", color="identical")
ABSTRACTING_LENS = Lens(rotations=True, reflection=True, scale="none", color="agnostic")


@dataclass
class Group:
    gid: int
    layer: int
    color: int                 # dominant colour of the members (the fine key)
    shape_class: int           # abstract shape-class of the members (the abstract key)
    member_ids: list           # entity_ids
    count: int
    centroid: tuple
    bbox: tuple
    arrangement: Structure     # members as positioned nodes (for the iso refinement)


@dataclass
class AbstractGroupType:
    type_key: tuple            # (sorted shape-class multiset, count) — the abstract identity
    member_gids: list          # group ids collapsed into this type
    count: int                 # how many groups share it (>=2 ⇒ an established regularity)
    alignments: list           # (gid_a, gid_b, [(member_a, member_b)...], iso_score)


@dataclass
class Subsystem:
    sid: int
    color: int
    member_ids: list
    anchor_ids: list           # the distinguished element(s) (largest extent) — a candidate cursor
    satellite_gid: int | None  # the similarity-group of the small repeated satellites
    satellite_ids: list


@dataclass
class SubsystemMapping:
    a_sid: int
    b_sid: int
    satellite_alignment: list  # [(member_a, member_b)...]
    anchor_alignment: list     # [(anchor_a, anchor_b)...]
    basis: str


@dataclass
class GroupGoal:
    kind: str                  # 'conform' | 'unite' | 'transfer'
    scope: tuple               # the gids / sids the goal spans
    manipulable: list          # entity_ids the goal would move (subject to manipulability)
    desc: str
    priority: float


@dataclass
class Ladder:
    groups: list               # Group (layer 2)
    abstract_types: list       # AbstractGroupType (layer 3)
    subsystems: list
    subsystem_mappings: list
    goals: list                # GroupGoal
    layers_built: int          # how many layers before the fixed point
    fixed_point: bool

    def summary(self) -> str:
        out = [f"abstraction ladder: {self.layers_built} layers"
               f"{' (fixed point)' if self.fixed_point else ''}"]
        out.append(f"  layer2 similarity groups: {len(self.groups)}")
        for g in self.groups:
            out.append(f"    g{g.gid}: colour=#{g.color:06x} shape={g.shape_class} "
                       f"x{g.count} at {g.centroid}")
        out.append(f"  layer3 abstract group-types: {len(self.abstract_types)}")
        for a in self.abstract_types:
            tag = " <- REGULARITY" if a.count >= 2 else ""
            out.append(f"    type {a.type_key}: groups {a.member_gids} (count {a.count}){tag}")
        out.append(f"  subsystems: {len(self.subsystems)}; mappings: {len(self.subsystem_mappings)}")
        for m in self.subsystem_mappings:
            out.append(f"    subsystem {m.a_sid} <=> {m.b_sid} "
                       f"(satellites {m.satellite_alignment}, anchors {m.anchor_alignment})")
        out.append(f"  candidate goals: {len(self.goals)}")
        for go in self.goals:
            out.append(f"    [{go.kind}] {go.desc} (manip={go.manipulable} prio={go.priority})")
        return "\n".join(out)


# --------------------------------------------------------------------------- #
def _identity_color(bitmap) -> int:
    """An entity's identifying colour is its most CHROMATIC colour, not its
    modal one — the anti-alias / halo pixels are usually desaturated grey and
    would otherwise collapse a blue glyph and a yellow glyph onto the same grey
    key.  Ties on chroma break toward the most frequent colour."""
    m = np.asarray(bitmap)
    v = m[m != -1].astype(int)
    if not v.size:
        return -1
    vals, counts = np.unique(v, return_counts=True)
    def chroma(c):
        r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
        return max(r, g, b) - min(r, g, b)
    chr_arr = np.array([chroma(int(c)) for c in vals])
    mx = int(chr_arr.max())
    if mx == 0:                                   # achromatic entity -> modal colour
        return int(vals[counts.argmax()])
    cand = [(int(counts[i]), int(vals[i]))
            for i in range(len(vals)) if int(chr_arr[i]) == mx]
    return max(cand)[1]


def _fg(entities):
    return [e for e in entities
            if getattr(e, "bitmap", None) is not None
            and not getattr(e, "is_background_primary", False)
            and not getattr(e, "is_background_secondary", False)]


def _shape_classes(entities, lens: Lens = ABSTRACTING_LENS) -> dict:
    """Abstract shape-class per entity (colour-/orientation-agnostic)."""
    fg = _fg(entities)
    ec = equivalence_classes([(e.bitmap, -1) for e in fg], lens)
    return {e.entity_id: int(c) for e, c in zip(fg, ec["classes"])}


def _arrangement(members, by_id) -> Structure:
    nodes = [{"grid": by_id[eid].bitmap, "bg": -1,
              "pos": tuple(by_id[eid].centroid_cell)} for eid in members]
    return Structure(nodes=nodes, edges=[])


def _bbox_union(members, by_id):
    bs = [by_id[eid].bbox_logical for eid in members]
    return (min(b[0] for b in bs), min(b[1] for b in bs),
            max(b[2] for b in bs), max(b[3] for b in bs))


def _centroid(members, by_id):
    cs = [by_id[eid].centroid_cell for eid in members]
    return (round(sum(c[0] for c in cs) / len(cs)),
            round(sum(c[1] for c in cs) / len(cs)))


# --------------------------------------------------------------------------- #
def similarity_groups(entities) -> list:
    """LAYER 2 — group instances that share (dominant colour, abstract shape).

    The colour split is what keeps a blue quartet and a yellow quartet as two
    separate groups so that layer 3 can then recognise them as the same kind."""
    fg = _fg(entities)
    by_id = {e.entity_id: e for e in fg}
    shape = _shape_classes(entities)
    buckets = defaultdict(list)
    for e in fg:
        buckets[(_identity_color(e.bitmap), shape[e.entity_id])].append(e.entity_id)
    groups, gid = [], 0
    for (col, sc), members in sorted(buckets.items(),
                                     key=lambda kv: (kv[0][1], kv[0][0])):
        members = sorted(members, key=lambda eid: tuple(by_id[eid].centroid_cell))
        groups.append(Group(gid=gid, layer=2, color=col, shape_class=sc,
                            member_ids=members, count=len(members),
                            centroid=_centroid(members, by_id),
                            bbox=_bbox_union(members, by_id),
                            arrangement=_arrangement(members, by_id)))
        gid += 1
    return groups


def collapse_groups(groups, by_id) -> list:
    """LAYER 3 — recognise groups that are the SAME ABSTRACT TYPE.

    Two groups collapse when their abstract member-type multiset AND count are
    equal (equality, not a tuned cutoff).  For each collapsed pair we record the
    member alignment (role order, refined by an exact structural-iso check)."""
    by_key = defaultdict(list)
    for g in groups:
        key = (tuple(sorted(Counter([g.shape_class]).items())), g.count)
        by_key[key].append(g)
    types = []
    for key, gs in by_key.items():
        alignments = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                a, b = gs[i], gs[j]
                # role-order alignment (position-sorted), refined by iso score
                pairs = list(zip(a.member_ids, b.member_ids))
                iso = same_structure(a.arrangement, b.arrangement,
                                     ABSTRACTING_LENS, type_blind=True)["iso_score"]
                alignments.append((a.gid, b.gid, pairs, round(float(iso), 3)))
        types.append(AbstractGroupType(type_key=key, member_gids=[g.gid for g in gs],
                                       count=len(gs), alignments=alignments))
    return types


def subsystems(entities) -> list:
    """A colour-cohesion grouping: all members of one colour, with the
    largest-extent member(s) tagged as the distinguished ANCHOR (a candidate
    cursor) and the small repeated members as SATELLITES."""
    fg = _fg(entities)
    by_id = {e.entity_id: e for e in fg}
    shape = _shape_classes(entities)
    by_color = defaultdict(list)
    for e in fg:
        by_color[_identity_color(e.bitmap)].append(e.entity_id)
    # majority shape-class within a colour = the satellites; the rest = anchor
    subs, sid = [], 0
    for col, members in sorted(by_color.items()):
        # satellites = the most-numerous shape-class; a tie (e.g. when a fragmented
        # cross yields as many arm-pieces as there are dots) breaks toward the
        # SMALLER glyph, since satellites are the little repeated markers.
        sc_counts = Counter(shape[m] for m in members)
        maxn = max(sc_counts.values())
        cand = [sc for sc, n in sc_counts.items() if n == maxn]
        def _mean_area(sc):
            ar = [(by_id[m].bbox_logical[2] - by_id[m].bbox_logical[0] + 1)
                  * (by_id[m].bbox_logical[3] - by_id[m].bbox_logical[1] + 1)
                  for m in members if shape[m] == sc]
            return sum(ar) / len(ar)
        sat_sc = min(cand, key=_mean_area)
        sats = sorted([m for m in members if shape[m] == sat_sc],
                      key=lambda eid: tuple(by_id[eid].centroid_cell))
        anchors = [m for m in members if shape[m] != sat_sc]
        if not anchors:                      # all one shape -> no distinguished element
            anchors, sats = [], sats
        subs.append(Subsystem(sid=sid, color=col, member_ids=sorted(members),
                              anchor_ids=anchors, satellite_gid=None,
                              satellite_ids=sats))
        sid += 1
    return subs


def map_subsystems(subs, entities) -> list:
    """Two subsystems map when their SATELLITE sets are abstract-equivalent
    (same satellite shape-class + count) and both carry an anchor — i.e. they
    are the same KIND of subsystem (a distinguished element + N like satellites),
    differing only by colour/position.  The alignment is the analogy."""
    by_id = {e.entity_id: e for e in _fg(entities)}
    shape = _shape_classes(entities)
    out = []

    def sat_key(s):
        if not s.satellite_ids:
            return None
        return (shape[s.satellite_ids[0]], len(s.satellite_ids))

    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            if not a.anchor_ids or not b.anchor_ids:
                continue
            if sat_key(a) is None or sat_key(a) != sat_key(b):
                continue
            sat_align = list(zip(a.satellite_ids, b.satellite_ids))
            anc_align = list(zip(a.anchor_ids, b.anchor_ids))
            out.append(SubsystemMapping(a_sid=a.sid, b_sid=b.sid,
                                        satellite_alignment=sat_align,
                                        anchor_alignment=anc_align,
                                        basis="anchor+like-satellites"))
    return out


def propose_group_goals(abstract_types, subs, sub_maps, manipulable=()) -> list:
    """Read candidate goals off the regularities + mappings.  The score is the
    judge; these are hypotheses at a prior credence.

    - UNITE  : within a subsystem, bring the (manipulable) anchor toward its
               satellites' configuration — similarity→converge lifted to groups.
    - TRANSFER: across a subsystem mapping, the operation that advances one
               subsystem applies to its mapped twin (symmetric solution).
    - CONFORM: a group that is the lone odd-one-out against an established
               abstract type should be made to match it.
    """
    manip = set(manipulable)
    goals = []
    by_sid = {s.sid: s for s in subs}

    # UNITE — per subsystem with a distinguished anchor + satellites
    for s in subs:
        if not s.anchor_ids or not s.satellite_ids:
            continue
        anc_manip = [a for a in s.anchor_ids if (not manip or a in manip)]
        goals.append(GroupGoal(
            kind="unite", scope=(s.sid,),
            manipulable=s.anchor_ids,
            desc=(f"bring the distinguished element {s.anchor_ids} of subsystem "
                  f"{s.sid} into the configuration its {len(s.satellite_ids)} "
                  f"satellites define"),
            priority=0.7 if anc_manip else 0.4))

    # TRANSFER — across each mapped pair of subsystems
    for m in sub_maps:
        goals.append(GroupGoal(
            kind="transfer", scope=(m.a_sid, m.b_sid),
            manipulable=[a for a, _ in m.anchor_alignment],
            desc=(f"subsystems {m.a_sid} and {m.b_sid} are the same kind "
                  f"(a distinguished element + like satellites); the operation "
                  f"that solves one applies to its mapped twin "
                  f"(anchors {m.anchor_alignment}, satellites {m.satellite_alignment})"),
            priority=0.75))

    # CONFORM — a singleton group standing apart from an established >=2 type of
    # the same member shape but different count (a near-miss against regularity)
    return goals


def abstraction_ladder(entities, manipulable=()) -> Ladder:
    """Build the full ladder over the perceived entities and read goals off it.

    Recursion to a FIXED POINT: collapse fine groups into abstract types; if a
    further collapse of the abstract types merges anything, recurse; stop when a
    layer merges nothing.  (Most scenes terminate at layer 3-4.)"""
    by_id = {e.entity_id: e for e in _fg(entities)}
    groups = similarity_groups(entities)
    abstract = collapse_groups(groups, by_id)
    layers = 3

    # recurse: re-collapse the abstract types by their own keys; a merge means
    # another layer of structure exists.  (Threshold-free: equality of keys.)
    prev_n = len(abstract)
    while True:
        rekey = defaultdict(list)
        for a in abstract:
            rekey[a.type_key[0]].append(a)        # group abstract types by shape-multiset
        merged = [k for k, v in rekey.items() if len(v) > 1]
        if not merged or len(rekey) == prev_n:
            break
        layers += 1
        prev_n = len(rekey)
        if layers > 8:
            break

    subs = subsystems(entities)
    sub_maps = map_subsystems(subs, entities)
    goals = propose_group_goals(abstract, subs, sub_maps, manipulable)
    fixed = True
    return Ladder(groups=groups, abstract_types=abstract, subsystems=subs,
                  subsystem_mappings=sub_maps, goals=goals,
                  layers_built=layers, fixed_point=fixed)
