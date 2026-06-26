"""Regularity instinct (slice 3) — see docs/SPEC_logical_scene_and_regularity_instinct.md.

Once repetition establishes a prototype relation with a consistent binding (e.g. the
legend's cyan-type -> pink-type dictionary), a NEAR-MISS instance — a candidate that
fits the prototype's skeleton but whose binding is violated — becomes a tension.
Closing it, by changing the one MANIPULABLE element to the prototype-required type,
becomes a goal (entropy reduction).  Goals are HYPOTHESES; the score is the judge.

Generalizes goal_priors.  No magic thresholds: regularity strength = repetition
count; adjacency = immediate-neighbour (nothing between), not a distance cutoff.
"""
from __future__ import annotations
from dataclasses import dataclass

from perception_loop_v2.logical_scene import LogicalScene, DEFAULT_LENS
from perception_loop_v2.perceptual_equivalence import Lens


# Registry BehaviourEvent kinds that mean "this entity's state responded to play".
CHANGE_KINDS = {"moved", "recolor", "grew", "shrank", "reappeared", "drifted"}
# A subset: the entity's APPEARANCE/shape changed in place (not merely moved).  A
# conform-goal must change a thing's shape, so it needs SHAPE-manipulability — a pure
# mover (a cursor) can't be conformed to a target shape by the keys that move it.
SHAPE_CHANGE_KINDS = {"recolor", "grew", "shrank"}


def manipulable_from_events(tracks, kinds=CHANGE_KINDS) -> set:
    """Track ids observed to change under play — the learned manipulability set.  Pass
    `kinds=SHAPE_CHANGE_KINDS` for conform-goals (the element's shape must be editable);
    the default (any change) is the general 'responds to play' set."""
    out = set()
    for t in tracks:
        if any(getattr(ev, "kind", None) in kinds
               for ev in getattr(t, "behaviour_events", [])):
            out.add(t.track_id)
    return out


@dataclass
class ConformGoal:
    entity_id: int          # the manipulable element to change
    current_type: int
    target_type: int        # the prototype-required type
    priority: float
    desc: str


def _ov(a, b, ax):          # bbox range overlap on the non-bridging axis
    if ax == "row":
        return a.bbox[0] <= b.bbox[2] and b.bbox[0] <= a.bbox[2]
    return a.bbox[1] <= b.bbox[3] and b.bbox[1] <= a.bbox[3]


def _adjacency_pairs(scene: LogicalScene):
    """Immediate-neighbour typed pairs (role0 = upper/left), parameter-free: B is A's
    nearest neighbour to the right/below with overlap and NOTHING between.  Surfaces
    candidate instances (e.g. bottom word columns) that carry no explicit relation."""
    nodes = scene.nodes
    pairs = []
    for a in nodes:
        right = [b for b in nodes if b.entity_id != a.entity_id
                 and _ov(a, b, "row") and b.bbox[1] > a.bbox[3]]
        if right:
            b = min(right, key=lambda b: b.bbox[1])
            if not any(c.entity_id not in (a.entity_id, b.entity_id)
                       and a.bbox[3] < c.bbox[1] and c.bbox[3] < b.bbox[1]
                       and _ov(a, c, "row") for c in nodes):
                pairs.append((a.entity_id, a.type_id, b.entity_id, b.type_id))
        below = [b for b in nodes if b.entity_id != a.entity_id
                 and _ov(a, b, "col") and b.bbox[0] > a.bbox[2]]
        if below:
            b = min(below, key=lambda b: b.bbox[0])
            if not any(c.entity_id not in (a.entity_id, b.entity_id)
                       and a.bbox[2] < c.bbox[0] and c.bbox[2] < b.bbox[0]
                       and _ov(a, c, "col") for c in nodes):
                pairs.append((a.entity_id, a.type_id, b.entity_id, b.type_id))
    return pairs


def _rng(n, ax):
    return (n.bbox[1], n.bbox[3]) if ax == "x" else (n.bbox[0], n.bbox[2])


def _ovl(a, b, ax):
    al, ah = _rng(a, ax); bl, bh = _rng(b, ax)
    return al <= bh and bl <= ah


def _lanes(nodes, ax):
    """Group compact nodes into lanes that share their range on `ax` (a lane is a
    connected component of the ax-overlap graph) — a row of glyphs (ax='y') or a
    column of glyphs (ax='x')."""
    adj = {n.entity_id: set() for n in nodes}
    by = {n.entity_id: n for n in nodes}
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if _ovl(a, b, ax):
                adj[a.entity_id].add(b.entity_id); adj[b.entity_id].add(a.entity_id)
    seen, lanes = set(), []
    for n in nodes:
        if n.entity_id in seen:
            continue
        comp, stack = [], [n.entity_id]; seen.add(n.entity_id)
        while stack:
            k = stack.pop(); comp.append(by[k])
            for m in adj[k]:
                if m not in seen:
                    seen.add(m); stack.append(m)
        if len(comp) >= 2:
            lanes.append(comp)
    return lanes


def _stack(lanes, extent, cross):
    """Pair two parallel lanes that are STACKED along `cross` (no cross overlap),
    equal length, and aligned element-wise on `extent`.  role 0 = the lane nearer
    the origin on `cross`.  This is how two aligned sequences (a cyan word over a
    pink word) become column-wise candidate pairs that no pairwise adjacency finds."""
    out = []
    norm = [sorted(L, key=lambda n: _rng(n, extent)[0]) for L in lanes]
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            A, B = norm[i], norm[j]
            if len(A) != len(B):
                continue
            amin = min(_rng(n, cross)[0] for n in A)
            bmin = min(_rng(n, cross)[0] for n in B)
            U, Lo = (A, B) if amin <= bmin else (B, A)
            if max(_rng(n, cross)[1] for n in U) >= min(_rng(n, cross)[0] for n in Lo):
                continue                        # must be stacked, not overlapping
            if all(_ovl(U[k], Lo[k], extent) for k in range(len(U))):
                for k in range(len(U)):
                    out.append((U[k].entity_id, U[k].type_id,
                                Lo[k].entity_id, Lo[k].type_id))
    return out


def _contains(a, b):
    return (a.bbox[0] <= b.bbox[0] and a.bbox[1] <= b.bbox[1]
            and a.bbox[2] >= b.bbox[2] and a.bbox[3] >= b.bbox[3] and a.bbox != b.bbox)


def _seq_nodes(scene):
    alln = scene.nodes
    return [n for n in alln if not n.is_region
            and not any(_contains(n, m) for m in alln if m is not n)]


def _sequence_pairs(scene: LogicalScene):
    # Sequence elements are compact glyphs, not the bars/panels they sit on: drop
    # regions and any node that spatially CONTAINS another (a word-bar contains its
    # glyphs) — a backdrop is not a sequence member.
    nodes = _seq_nodes(scene)
    return (_stack(_lanes(nodes, "y"), extent="x", cross="y")    # rows stacked vertically
            + _stack(_lanes(nodes, "x"), extent="y", cross="x"))  # cols stacked horizontally


def _expand_manipulable(scene: LogicalScene, manipulable: set) -> set:
    """A homogeneous sequence shares manipulability: if any member of an aligned,
    paired sequence-lane is editable, the whole lane is — so confirming ONE editable
    slot by probing generalizes to its whole word, without probing every slot.  Only
    PAIRED sequence-lanes count (a word paired against a parallel word), which is what
    keeps the generalization from leaking across a column that happens to span the
    legend and the word."""
    out = set(manipulable)
    for lane in _paired_lanes(scene):
        if any(n.entity_id in manipulable for n in lane):
            out.update(n.entity_id for n in lane)
    return out


def _aligned_lane_pairs(lanes, extent, cross):
    """Lane pairs that are STACKED along `cross`, equal length, and aligned
    element-wise on `extent` — i.e. two parallel sequences (a cyan word over a pink
    word)."""
    norm = [sorted(L, key=lambda n: _rng(n, extent)[0]) for L in lanes]
    out = []
    for i in range(len(norm)):
        for j in range(i + 1, len(norm)):
            A, B = norm[i], norm[j]
            if len(A) != len(B):
                continue
            amin = min(_rng(n, cross)[0] for n in A)
            bmin = min(_rng(n, cross)[0] for n in B)
            U, Lo = (A, B) if amin <= bmin else (B, A)
            if max(_rng(n, cross)[1] for n in U) >= min(_rng(n, cross)[0] for n in Lo):
                continue
            if all(_ovl(U[k], Lo[k], extent) for k in range(len(U))):
                out.append((U, Lo))
    return out


def _paired_lanes(scene: LogicalScene):
    nodes = _seq_nodes(scene)
    lanes = []
    for (U, Lo) in (_aligned_lane_pairs(_lanes(nodes, "y"), "x", "y")
                    + _aligned_lane_pairs(_lanes(nodes, "x"), "y", "x")):
        lanes.append(U); lanes.append(Lo)
    return lanes


def _lane_uniform(lane):
    """A lane is uniform when all its elements share one type — a row of identical
    BLANKS (unfilled slots), which carries no information of its own and so reads as
    the editable TARGET to be filled, not the reference."""
    return len({n.type_id for n in lane}) == 1


def _template_pairs(scene: LogicalScene):
    """Aligned sequence-lane pairs in which exactly ONE lane is uniform (a row of
    identical blanks) and the other is DIVERSE (the reference): the uniform lane is
    the unfilled TARGET, the diverse lane the REFERENCE.  Yields (ref_id, ref_type,
    tgt_id, tgt_type) element-wise.  No dictionary is needed here — the implied
    binding is IDENTITY (make each blank match the reference above/below it).  This
    is distinct from the legend case, where BOTH lanes are diverse and a learned
    cross-type dictionary supplies the binding; there, neither lane is uniform, so
    this yields nothing and the dictionary path is used instead."""
    nodes = _seq_nodes(scene)
    out = []
    for (U, Lo) in (_aligned_lane_pairs(_lanes(nodes, "y"), "x", "y")
                    + _aligned_lane_pairs(_lanes(nodes, "x"), "y", "x")):
        u_uni, l_uni = _lane_uniform(U), _lane_uniform(Lo)
        if u_uni == l_uni:
            continue                       # both blank or both diverse -> not a fill
        ref, tgt = (Lo, U) if u_uni else (U, Lo)   # diverse lane = reference
        for r, t in zip(ref, tgt):
            out.append((r.entity_id, r.type_id, t.entity_id, t.type_id))
    return out


def find_conform_goals(scene: LogicalScene, manipulable: set,
                       lens: Lens = DEFAULT_LENS, *, min_count: int = 2):
    """Goals that make near-miss pairs conform to an established prototype relation.

    A 2-node relation that RECURS (>= min_count) with a consistent role0->role1
    binding is a prototype (the dictionary).  Any candidate pair (an existing pair
    composite, or an immediate-neighbour pair) whose role0 type is a known key but
    whose role1 type violates the binding is a near-miss; if its role1 entity is
    MANIPULABLE, emit a goal to set it to the required type.  Priority grows with the
    prototype's repetition strength."""
    # Prototype pairs = connector relations, LIFTED THROUGH CONTAINMENT: a connector
    # between two containers is really a relation between their CONTENTS (the glyphs),
    # which carry the meaning; the tiles are just slots.  Lift each connector end to
    # its contained figure when it has one, then order roles positionally.
    child_of = {a: b for (a, b, k) in scene.relations if k == "contains"}

    def _lift(eid):
        return child_of.get(eid, eid)

    eff = []
    for (a, b, k) in scene.relations:
        if k != "connector":
            continue
        n0, n1 = _node(scene, _lift(a)), _node(scene, _lift(b))
        if n0 is None or n1 is None:
            continue
        # role 0 = the consistent CONTEXT side (e.g. the cyan/prompt context), so the
        # binding stays functional regardless of small centroid jitter between a
        # pair's two glyphs (rows differing by 1 must NOT flip the pairing).  Tie-break
        # by position only when the two share a context.
        if _role_key(n0) > _role_key(n1):
            n0, n1 = n1, n0
        eff.append((n0.entity_id, n0.type_id, n1.entity_id, n1.type_id))
    # Build the cross-type dictionary from recurring connector relations, if any.  A
    # missing or ambiguous dictionary is NOT fatal: the template-conform path below
    # (identity binding) still applies, so a game with no legend can still yield goals.
    fwd, valid = {}, set()
    if len(eff) >= min_count:
        for (e0, t0, e1, t1) in eff:
            valid.add((t0, t1))
            if t0 in fwd and fwd[t0] != t1:
                fwd, valid = {}, set()          # ambiguous -> no dictionary, not abort
                break
            fwd[t0] = t1
    strength = min(1.0, len(eff) / (len(eff) + 1.0))
    manipulable = _expand_manipulable(scene, manipulable)

    # Sequence pairs first: the column-aligned word pairing is the trustworthy role
    # assignment, so it wins the one-goal-per-element dedup over looser adjacency.
    cands = _sequence_pairs(scene) + list(eff) + _adjacency_pairs(scene)

    goals, seen = [], set()
    for (e0, t0, e1, t1) in cands:
        if (t0, t1) in valid:
            continue                            # already a valid instance
        # Near-miss: role0 is a known dictionary key and its (manipulable) partner
        # does NOT match the binding.  The noise filter is LEARNED MANIPULABILITY, not
        # a type whitelist: fixed legend tiles never change under play, so they are
        # not manipulable and their internal pairings can't yield goals.  This lets a
        # near-miss whose role1 currently holds ANY wrong symbol still be corrected.
        if t0 in fwd and fwd[t0] != t1 and e1 in manipulable:
            if e1 in seen:                      # one goal per editable element
                continue
            seen.add(e1)
            goals.append(ConformGoal(
                e1, t1, fwd[t0], strength,
                f"set #{e1} (type {t1}) -> type {fwd[t0]} to match the prototype "
                f"partner of type {t0}"))

    # Template-conform (identity binding, no dictionary): a row of identical blanks
    # aligned with a diverse reference row -> fill each blank to MATCH the reference
    # above/below it.  The entropy-reduction BASE CASE of conform.  Fires only where a
    # lane is uniform (the unfilled side), so legend/word pairs (both diverse) never
    # reach here and keep using the dictionary path.  Regularity strength = how many
    # element-pairs repeat the fill pattern (count-based, like the dictionary's).
    tpairs = _template_pairs(scene)
    tstrength = min(1.0, len(tpairs) / (len(tpairs) + 1.0))
    for (er, tr, et, tt) in tpairs:
        if tr == tt or et not in manipulable or et in seen:
            continue                            # already matches, or not editable
        seen.add(et)
        goals.append(ConformGoal(
            et, tt, tr, tstrength,
            f"fill #{et} (type {tt}) -> type {tr} to match its reference "
            f"partner #{er} (type {tr})"))
    return goals


def _type(scene, eid):
    n = _node(scene, eid)
    return n.type_id if n else None


def _node(scene, eid):
    for n in scene.nodes:
        if n.entity_id == eid:
            return n
    return None


def _role_key(n):
    """Sort key that puts a legend pair's roles in a CONTEXT-consistent order (the
    background colour a glyph sits on is its role), with position as a tie-break only
    within the same context.  Robust to small centroid jitter that a position-only key
    would let flip the pairing."""
    return (n.context_bg if n.context_bg is not None else -1, n.centroid)
