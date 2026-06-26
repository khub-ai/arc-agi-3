"""Logical scene graph — see docs/SPEC_logical_scene_and_regularity_instinct.md.

Lifts geometric entities into a LOGICAL representation: typed instances (each entity
= a type + the variation/pose that distinguishes it from the type prototype),
relations (connectors, containment), and composites (relation-connected subgraphs,
e.g. a tile-connector-tile pair).  Prototype grouping then folds recurring composites
into one prototype with a repetition count — the regularity the instinct layer
(slice 3) turns near-misses against into goals.

Pure assembly of shipped primitives (perceptual_equivalence + line_relations); no
role names, no magic thresholds, no game-specific code.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque, defaultdict
import numpy as np

from perception_loop_v2.perceptual_equivalence import (
    Lens, Pose, canonical, equivalence_classes, Structure, same_structure)
from perception_loop_v2.line_relations import detect_connectors


# Typing lens: orientation- and recolor-tolerant by default, so an entity and its
# rotated/recolored sibling collapse to one type and their difference becomes the
# node's `variation`.  The caller may override (lens selection is its job).
DEFAULT_LENS = Lens(rotations=True, reflection=True, scale="none", color="agnostic")


@dataclass
class LogicalNode:
    entity_id: int
    type_id: int            # equivalence class under the lens
    variation: Pose         # pose distinguishing this instance from the prototype
    centroid: tuple
    bbox: tuple
    is_region: bool = False  # a large low-fill region (a bar/panel), not a compact glyph
    context_bg: object = None  # palette of the background it sits on (its role/context)


@dataclass
class Composite:
    ids: list               # entity_ids in this relation-connected unit
    structure: Structure    # its typed relational graph (for prototype matching)


@dataclass
class LogicalScene:
    nodes: list             # LogicalNode
    relations: list         # (a_id, b_id, kind)
    composites: list        # Composite


def build_logical_scene(entities, frame, lens: Lens = DEFAULT_LENS) -> LogicalScene:
    fg = [e for e in entities
          if not getattr(e, "is_background_primary", False)
          and not getattr(e, "is_background_secondary", False)
          and getattr(e, "bitmap", None) is not None]
    by_id = {e.entity_id: e for e in fg}

    # --- typed instances ------------------------------------------------------
    # Type glyphs WITHIN their context (the background colour they sit on), so two
    # same-shaped glyphs on different-coloured backgrounds — a cyan-role glyph and a
    # pink-role glyph — get distinct symbol-ids instead of collapsing together.  Type
    # ids are globally unique across contexts via a (context, local-class) registry.
    ctx_groups = defaultdict(list)
    for e in fg:
        ctx_groups[getattr(e, "context_bg", None)].append(e)
    type_of, gmap = {}, {}
    for ctx, group in ctx_groups.items():
        ec = equivalence_classes([(e.bitmap, -1) for e in group], lens)
        for e, cls in zip(group, ec["classes"]):
            key = (ctx, cls)
            gmap.setdefault(key, len(gmap))
            type_of[e.entity_id] = gmap[key]
    nodes = []
    for e in fg:
        pose = canonical(e.bitmap, -1, lens)["pose"]
        nodes.append(LogicalNode(e.entity_id, type_of[e.entity_id], pose,
                                 tuple(e.centroid_cell), tuple(e.bbox_logical),
                                 is_region=bool(getattr(e, "is_region", False)),
                                 context_bg=getattr(e, "context_bg", None)))

    # --- relations ------------------------------------------------------------
    relations = [(c.a, c.b, "connector") for c in detect_connectors(frame, entities)]
    for e in fg:
        ci = getattr(e, "contained_in", None)
        if ci is not None and ci in by_id:
            relations.append((ci, e.entity_id, "contains"))

    # --- composites = connected components over the relation graph -------------
    adj = {e.entity_id: set() for e in fg}
    for a, b, _k in relations:
        if a in adj and b in adj:
            adj[a].add(b); adj[b].add(a)
    seen = set(); composites = []
    for start in adj:
        if start in seen or not adj[start]:
            continue
        comp = []; q = deque([start]); seen.add(start)
        while q:
            n = q.popleft(); comp.append(n)
            for m in adj[n]:
                if m not in seen:
                    seen.add(m); q.append(m)
        composites.append(_make_composite(sorted(comp), by_id, relations))
    return LogicalScene(nodes, relations, composites)


def _make_composite(ids, by_id, relations) -> Composite:
    # order nodes by position (row, then col) so a composite's ROLES are consistent
    # regardless of entity-id order or global orientation (role 0 = upper-left).
    ids = sorted(ids, key=lambda eid: tuple(by_id[eid].centroid_cell))
    local = {eid: i for i, eid in enumerate(ids)}
    snodes = [{"grid": by_id[eid].bitmap, "bg": -1,
               "pos": tuple(by_id[eid].centroid_cell)} for eid in ids]
    sedges = [(local[a], local[b], k) for (a, b, k) in relations
              if a in local and b in local]
    return Composite(ids=ids, structure=Structure(nodes=snodes, edges=sedges))


def group_prototypes(scene: LogicalScene, lens: Lens = DEFAULT_LENS):
    """Fold structurally-equivalent composites into prototypes.

    Returns a list of {members: [composite indices], count, rep: Composite}.  A
    prototype with count > 1 is an established regularity; its strength is `count`."""
    protos = []
    for ci, comp in enumerate(scene.composites):
        placed = None
        for p in protos:
            if same_structure(comp.structure, p["rep"].structure, lens)["iso_score"] >= 1.0:
                placed = p; break
        if placed is None:
            protos.append({"members": [ci], "count": 1, "rep": comp})
        else:
            placed["members"].append(ci); placed["count"] += 1
    return protos
