"""Perceptual Equivalence — object & structure constancy.

See docs/SPEC_perceptual_equivalence.md.  This file is the ATOM engine (the first
slice): decide whether two single objects are the same/related under a chosen
transform group (a `Lens`), returning pose-factored evidence — never a bare boolean.

Design commitments realized here (the 2D-grid restriction of the 3D problem):
- identity is pose-factored: an atom is (canonical descriptor, pose, stabilizer),
  never a raw bitmap; appearance is explained, not stored.
- equivalence is always relative to an explicit `Lens` (the transform group G).
- canonicalization folds scale -> color -> orientation so a glyph and its rotated /
  recolored / up-scaled copies share one canonical descriptor.
- symmetry yields a pose SET (the stabilizer), never a single unjustified pose.
- matching is graded (residual) and partial (visible_fraction).

Structures, rule-induction, and the registry filter layer on top of this engine.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# --------------------------------------------------------------------------- #
# Lens (the transform group G) and Pose                                         #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Lens:
    """A concrete equivalence group, composed of independent, physically-grounded
    invariances.  `rotations`: include the 4 rotations (C4) vs none (C1).
    `reflection`: include mirror (D-group) — a SEPARATE toggle because a mirror image
    is not reachable by rigid motion (chirality).  `scale`: 'none' | 'integer'
    (strip integer upscaling) | 'resample' (resize to a canonical box).  `color`:
    'identical' | 'permutation' (recolor with a consistent bijection) | 'agnostic'
    (any recoloring)."""
    rotations: bool = True
    reflection: bool = False
    scale: str = "none"           # 'none' | 'integer' | 'resample'
    color: str = "identical"      # 'identical' | 'permutation' | 'agnostic'
    resample_box: int = 8

    def variants(self):
        """The (k, flip) dihedral elements this lens admits."""
        ks = (0, 1, 2, 3) if self.rotations else (0,)
        flips = (False, True) if self.reflection else (False,)
        return [(k, fl) for fl in flips for k in ks]


@dataclass(frozen=True)
class Pose:
    """Transform from the canonical frame to the observed atom (2D restriction of a
    6-DoF pose).  `k`/`reflected` are the orientation; `scale` the size factor;
    `translation` the (row, col) of the observed tight bbox top-left."""
    k: int = 0
    reflected: bool = False
    scale: float = 1.0
    translation: tuple = (0, 0)


@dataclass
class Match:
    identity: Optional[int]        # model id, or None if no model given
    pose: Pose
    residual: float                # in [0,1]; 0 == exact under the lens
    visible_fraction: float        # in [0,1]
    equivalent: bool               # residual <= threshold

    @property
    def score(self) -> float:
        return 1.0 - self.residual


@dataclass
class AtomModel:
    model_id: int
    descriptor: tuple
    bitmap: np.ndarray
    count: int = 1


# --------------------------------------------------------------------------- #
# Canonicalization primitives                                                   #
# --------------------------------------------------------------------------- #
def _crop(grid: np.ndarray, bg) -> np.ndarray:
    fg = np.argwhere(grid != bg)
    if len(fg) == 0:
        return np.empty((0, 0), grid.dtype)
    r0, c0 = fg.min(0)
    r1, c1 = fg.max(0)
    return grid[r0:r1 + 1, c0:c1 + 1]


def _reduce_integer_scale(a: np.ndarray):
    """Strip integer upscaling: largest f s.t. `a` is an f x f block tiling."""
    if a.size == 0:
        return a, 1
    h, w = a.shape
    for f in range(min(h, w), 1, -1):
        if h % f or w % f:
            continue
        ok = True
        for i in range(0, h, f):
            for j in range(0, w, f):
                blk = a[i:i + f, j:j + f]
                if (blk != blk[0, 0]).any():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return a[::f, ::f], f
    return a, 1


def _resample(a: np.ndarray, box: int, bg) -> np.ndarray:
    if a.size == 0:
        return a
    h, w = a.shape
    out = np.full((box, box), bg, a.dtype)
    for i in range(box):
        for j in range(box):
            out[i, j] = a[min(h - 1, i * h // box), min(w - 1, j * w // box)]
    return out


def _relabel(a: np.ndarray, bg, mode: str) -> np.ndarray:
    """Color-normalize an (already-oriented) array.  bg -> -1 sentinel.
    'identical': keep colors.  'permutation': relabel by first-occurrence order in
    THIS orientation's row-major scan (so a recolor with a consistent bijection
    collapses to the same labels).  'agnostic': all foreground -> 0."""
    out = np.full(a.shape, -1, dtype=np.int64)
    if a.size == 0:
        return out
    if mode == "agnostic":
        out[a != bg] = 0
        return out
    if mode == "identical":
        m = a != bg
        out[m] = a[m].astype(np.int64)
        return out
    # permutation
    seen = {}
    for idx, v in enumerate(a.flatten()):
        if v == bg:
            continue
        if v not in seen:
            seen[v] = len(seen)
    flat = a.flatten()
    of = out.flatten()
    for i, v in enumerate(flat):
        if v != bg:
            of[i] = seen[v]
    return of.reshape(a.shape)


def _apply(a: np.ndarray, k: int, flip: bool) -> np.ndarray:
    b = np.fliplr(a) if flip else a
    return np.rot90(b, k)


def _key(labeled: np.ndarray) -> tuple:
    return (labeled.shape[0], labeled.shape[1], tuple(int(x) for x in labeled.flatten()))


# --------------------------------------------------------------------------- #
# Public atom API                                                               #
# --------------------------------------------------------------------------- #
def canonical(grid: np.ndarray, bg=0, lens: Lens = Lens()):
    """Canonical descriptor of one atom under `lens`.

    Returns {descriptor, pose, stabilizer}:
      descriptor  -- hashable canonical form (lexicographically-minimal labeled array
                     over the lens's orientation variants, after scale+color norm).
      pose        -- Pose mapping canonical -> observed (the winning variant + scale +
                     bbox translation).
      stabilizer  -- list of (k,flip) variants that also achieve the minimum (symmetry
                     coset); len>1 means orientation is genuinely ambiguous.
    """
    grid = np.asarray(grid)
    fg = np.argwhere(grid != bg)
    translation = tuple(int(x) for x in fg.min(0)) if len(fg) else (0, 0)
    a = _crop(grid, bg)
    scale = 1.0
    if lens.scale == "integer":
        a, f = _reduce_integer_scale(a)
        scale = float(f)
    elif lens.scale == "resample":
        a = _resample(a, lens.resample_box, bg)
    best_key = None
    best_var = (0, False)
    stab = []
    for (k, flip) in lens.variants():
        labeled = _relabel(_apply(a, k, flip), bg, lens.color)
        kk = _key(labeled)
        if best_key is None or kk < best_key:
            best_key = kk
            best_var = (k, flip)
            stab = [(k, flip)]
        elif kk == best_key:
            stab.append((k, flip))
    return {
        "descriptor": best_key,
        "pose": Pose(k=best_var[0], reflected=best_var[1], scale=scale,
                     translation=translation),
        "stabilizer": stab,
    }


def same_object(A: np.ndarray, B: np.ndarray, lens: Lens = Lens(),
                bg_a=0, bg_b=0, threshold: float = 0.0) -> Match:
    """Atom equivalence under `lens`.  residual 0 == equal canonical descriptors;
    otherwise a graded mismatch fraction after best alignment."""
    ca = canonical(A, bg_a, lens)
    cb = canonical(B, bg_b, lens)
    if ca["descriptor"] == cb["descriptor"]:
        return Match(identity=None, pose=cb["pose"], residual=0.0,
                     visible_fraction=1.0, equivalent=True)
    res = _graded_residual(ca["descriptor"], cb["descriptor"])
    return Match(identity=None, pose=cb["pose"], residual=res,
                 visible_fraction=1.0, equivalent=res <= threshold)


def _graded_residual(da: tuple, db: tuple) -> float:
    """Mismatch fraction between two canonical descriptors (already orientation/
    color/scale-normalized).  Shapes must match to align; else residual 1."""
    ha, wa = da[0], da[1]
    hb, wb = db[0], db[1]
    if (ha, wa) != (hb, wb):
        return 1.0
    la = np.array(da[2]).reshape(ha, wa)
    lb = np.array(db[2]).reshape(hb, wb)
    union = (la >= 0) | (lb >= 0)
    if union.sum() == 0:
        return 0.0
    mism = (la != lb) & union
    return float(mism.sum()) / float(union.sum())


def equivalence_classes(atoms, lens: Lens = Lens(), bg=0):
    """Type a population of atoms under `lens`.

    `atoms`: list of 2D grids (or (grid, bg) tuples).  Returns {classes, store,
    n_types, L}:
      classes -- class id per atom (index into store).
      store   -- list of AtomModel (one per distinct canonical descriptor).
      n_types -- number of distinct types (== len(store)).
      L       -- TYPE-SET compression cost: prototype storage + cost to name each
                 atom's type.  L = sum over models of (canonical foreground cells)
                 + n_atoms * log2(max(1, n_types)).  Smaller == a more compressed
                 type-set, and L is monotone NON-INCREASING as the lens grows.

    NOTE: pose is deliberately NOT charged here — it is the explained-away nuisance,
    and charging it would cancel the collapse we want to measure.  Because L is
    monotone in lens richness, this module does NOT select the lens: minimizing L
    alone would always pick the richest lens and over-merge.  Lens SELECTION —
    balancing collapse against keeping game-needed distinctions, and tying the
    choice to downstream structural consistency (e.g. rule-induction coverage) — is
    owned by the reasoning layer.  This module only reports per-lens type structure.
    """
    store = []
    by_desc = {}
    classes = []
    for item in atoms:
        grid, b = item if isinstance(item, tuple) else (item, bg)
        c = canonical(grid, b, lens)
        d = c["descriptor"]
        if d not in by_desc:
            mid = len(store)
            by_desc[d] = mid
            store.append(AtomModel(model_id=mid, descriptor=d,
                                   bitmap=np.array(d[2]).reshape(d[0], d[1])))
        else:
            store[by_desc[d]].count += 1
        classes.append(by_desc[d])
    model_cells = sum(int((m.bitmap >= 0).sum()) for m in store)
    n_types = len(store)
    L = float(model_cells) + len(atoms) * float(np.log2(max(1, n_types)))
    return {"classes": classes, "store": store, "n_types": n_types, "L": L}


# --------------------------------------------------------------------------- #
# Structure layer: clusters as typed attributed relational graphs              #
# --------------------------------------------------------------------------- #
@dataclass
class Structure:
    """A cluster: nodes are atoms (each {grid, bg, pos}); edges are explicit
    allocentric relations (i, j, relation).  Implicit spatial relations are handled
    by the matcher through node positions, so only explicit relations (e.g. a drawn
    connector) need to be listed."""
    nodes: list                       # list of {'grid': ndarray, 'bg': int, 'pos': (r,c)}
    edges: list = field(default_factory=list)   # list of (i, j, relation:str)


# the eight D4 actions on integer (row, col) coordinates
def _group_elements(lens: Lens):
    R = {0: np.array([[1, 0], [0, 1]]), 1: np.array([[0, -1], [1, 0]]),
         2: np.array([[-1, 0], [0, -1]]), 3: np.array([[0, 1], [-1, 0]])}
    Fl = np.array([[1, 0], [0, -1]])
    ks = (0, 1, 2, 3) if lens.rotations else (0,)
    flips = (False, True) if lens.reflection else (False,)
    return [((k, fl), (R[k] @ Fl if fl else R[k])) for fl in flips for k in ks]


def _node_types(nodes, lens: Lens):
    return [canonical(nd["grid"], nd.get("bg", 0), lens)["descriptor"] for nd in nodes]


def _apply_g(pos, M, s):
    w = (M @ np.array(pos)) * s
    return (int(round(w[0])), int(round(w[1])))


def _align_under(A_nodes, A_types, B_nodes, B_types, M, s, type_blind=False):
    """Best node bijection A->B after transforming A by (M, s) and normalizing
    translation to the min-corner.  Returns (mapping dict A_idx->B_idx, matched count).
    Greedy unique match on (position, type); positions are integer-exact on the grid."""
    At = [_apply_g(nd["pos"], M, s) for nd in A_nodes]
    if At:
        ar0 = min(p[0] for p in At); ac0 = min(p[1] for p in At)
        At = [(p[0] - ar0, p[1] - ac0) for p in At]
    Bt = [nd["pos"] for nd in B_nodes]
    if Bt:
        br0 = min(p[0] for p in Bt); bc0 = min(p[1] for p in Bt)
        Bt = [(p[0] - br0, p[1] - bc0) for p in Bt]
    used = set(); mapping = {}
    bidx = {}
    for j, p in enumerate(Bt):
        bidx.setdefault(p, []).append(j)
    for i, p in enumerate(At):
        for j in bidx.get(p, []):
            if j in used:
                continue
            if type_blind or A_types[i] == B_types[j]:
                mapping[i] = j; used.add(j); break
    return mapping, len(mapping)


def _edge_overlap(A_edges, B_edges, mapping):
    bset = set((i, j, r) for (i, j, r) in B_edges)
    m = 0
    for (i, j, r) in A_edges:
        if i in mapping and j in mapping and (mapping[i], mapping[j], r) in bset:
            m += 1
    return m


def _scales(A_nodes, B_nodes, lens: Lens):
    if lens.scale == "none" or len(A_nodes) < 2 or len(B_nodes) < 2:
        return [1.0]
    def span(ns):
        rs = [n["pos"][0] for n in ns]; cs = [n["pos"][1] for n in ns]
        return max(1, (max(rs) - min(rs)) + (max(cs) - min(cs)))
    ratio = span(B_nodes) / span(A_nodes)
    cand = {1.0, round(ratio) if ratio >= 1 else 1.0}
    return [s for s in cand if s >= 1]


def same_structure(A: Structure, B: Structure, lens: Lens = Lens(), type_blind=False):
    """Graded structural equivalence under ONE global transform from the lens group.
    Node types are matched via the atom engine (rotation/scale/recolor-invariant per
    the lens); positions must align under a single global g; explicit edges must
    correspond.  Returns {iso_score, node_mapping, edit_distance, transform}."""
    A_types = [None] * len(A.nodes) if type_blind else _node_types(A.nodes, lens)
    B_types = [None] * len(B.nodes) if type_blind else _node_types(B.nodes, lens)
    n = max(len(A.nodes), len(B.nodes)); e = max(len(A.edges), len(B.edges))
    best = None
    for (_gid, M) in _group_elements(lens):
        for s in _scales(A.nodes, B.nodes, lens):
            mapping, nmatch = _align_under(A.nodes, A_types, B.nodes, B_types, M, s, type_blind)
            ematch = _edge_overlap(A.edges, B.edges, mapping)
            edit = (n - nmatch) + (e - ematch)
            score = 1.0 - edit / max(1, n + e)
            if best is None or score > best["iso_score"]:
                best = {"iso_score": score, "node_mapping": mapping,
                        "edit_distance": edit, "transform": _gid}
    return best


def structural_classes(structures, lens: Lens = Lens(), threshold: float = 1.0):
    """Group structures into types: each joins the first existing class it matches at
    iso_score >= threshold, else starts a new class."""
    reps = []; classes = []
    for st in structures:
        placed = None
        for ci, rep in enumerate(reps):
            if same_structure(st, rep, lens)["iso_score"] >= threshold:
                placed = ci; break
        if placed is None:
            placed = len(reps); reps.append(st)
        classes.append(placed)
    return {"classes": classes, "n_types": len(reps)}


def induce_relation(instances, lens: Lens = Lens()):
    """Analogical rule-induction by anti-unification (least general generalization).

    Align every instance to the first by TOPOLOGY (type-blind: same arrangement +
    edges under one global transform), then per node-role collect the atom-types seen
    across instances: a role with one type is CONSTANT, a role with several is a
    VARIABLE.  Returns {template, bindings, coverage}: the invariant relational
    skeleton, the per-instance variable->atom-type bindings, and the fraction of
    instances that fit the skeleton.  (A consistent binding across instances IS the
    induced rule; given a new atom on one side, it predicts the other.)"""
    if not instances:
        return {"template": None, "bindings": [], "coverage": 0.0}
    base = instances[0]
    roles = len(base.nodes)
    bindings = []
    for inst in instances:
        m = same_structure(base, inst, lens, type_blind=True)
        if m["iso_score"] >= 1.0 and len(m["node_mapping"]) == roles:
            it = _node_types(inst.nodes, lens)
            bindings.append({r: it[m["node_mapping"][r]] for r in range(roles)})
    template_roles = {}
    for r in range(roles):
        seen = {b[r] for b in bindings if r in b}
        template_roles[r] = ("const", next(iter(seen))) if len(seen) == 1 else ("var", r)
    return {"template": {"roles": template_roles, "edges": base.edges},
            "bindings": bindings, "coverage": len(bindings) / len(instances)}


# --------------------------------------------------------------------------- #
# Tracking: single-association scoring (the registry's filter primitive)        #
# --------------------------------------------------------------------------- #
def pose_change_plausibility(predicted: Pose, observed: Pose) -> float:
    """How plausible is the predicted -> observed pose change, in [0,1]?  A tracked
    rigid object may translate, rotate a little, and change apparent scale; it may NOT
    teleport or flip chirality.  Factors multiply so any single implausible component
    (a long jump, a mirror flip) collapses the score."""
    dt = (abs(predicted.translation[0] - observed.translation[0])
          + abs(predicted.translation[1] - observed.translation[1]))
    p_t = 1.0 / (1.0 + dt)                                   # translation closeness
    dk = (observed.k - predicted.k) % 4
    dk = min(dk, 4 - dk)
    p_r = 0.6 ** dk                                          # rotation discounted, never impossible
    p_f = 1.0 if predicted.reflected == observed.reflected else 0.0   # no chirality flip
    if predicted.scale and observed.scale:
        ratio = max(predicted.scale, observed.scale) / min(predicted.scale, observed.scale)
        p_s = 1.0 / ratio
    else:
        p_s = 1.0
    return p_t * p_r * p_f * p_s


def score_association(model, predicted_pose: Pose, obs_grid, lens: Lens = Lens(),
                      obs_bg=0, threshold: float = 0.0):
    """Single-association score a tracker consumes (spec §5).

    `model`: the tracked object's canonical identity (an AtomModel or a raw
    descriptor).  `predicted_pose`: the Pose the tracker expects this turn (its
    motion prediction).  `obs_grid`: a candidate observation.  Returns
    {match, pose_change_plausibility}: the identity Match of obs vs model (identity
    preserved iff residual <= threshold) and how plausible the implied pose change is.

    One estimator subsumes the registry's scattered passes: a translated object
    matches with identity preserved and high plausibility (the move pass); a recolored
    object matches under a color-permissive lens (the recolor pass); a rotated object
    matches under a rotation lens with a moderate-but-nonzero plausibility (which the
    old passes could not represent at all); a teleport or a different shape is rejected
    by plausibility or residual respectively (the long-jump gate, now principled)."""
    model_desc = model.descriptor if isinstance(model, AtomModel) else model
    c = canonical(obs_grid, obs_bg, lens)
    obs_desc, obs_pose = c["descriptor"], c["pose"]
    residual = 0.0 if obs_desc == model_desc else _graded_residual(model_desc, obs_desc)
    match = Match(identity=None, pose=obs_pose, residual=residual,
                  visible_fraction=1.0, equivalent=residual <= threshold)
    return {"match": match,
            "pose_change_plausibility": pose_change_plausibility(predicted_pose, obs_pose)}
