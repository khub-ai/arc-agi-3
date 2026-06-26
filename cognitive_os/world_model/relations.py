"""Generic spatial relation extraction for tracked objects."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Iterable, Tuple

from .schema import BBox, Relation, TrackedObject


# ---------------------------------------------------------------------------
# Composed relations: complete conjunctions with per-constituent evidence.
#
# A named relation that combines more than one primitive (e.g. "under the open
# end of a vertical fixture" = aligned-to-its-column AND below-its-tip) is TRUE
# only when EVERY constituent primitive is true, and it always carries the
# measurement that decided each constituent.  This is the substrate side of the
# rule in SPEC_visual_reasoning_substrate.md ("Composed relations are complete
# conjunctions, evaluated by the substrate, never asserted"): a composed
# relation may not be reported by its truth alone, nor satisfied by a single-
# axis proxy.  Domain-agnostic (games + robotics): the operands are bboxes /
# scalar reference geometry, no game vocabulary.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Conjunct:
    """One constituent of a composed relation: its truth + the measurement."""

    label: str
    holds: bool
    evidence: str


@dataclass(frozen=True)
class RelationCheck:
    """Result of evaluating a composed relation.  `holds` is True only when
    there is at least one conjunct and ALL conjuncts hold — so a proxy that
    drops a conjunct cannot read as satisfied."""

    name: str
    conjuncts: Tuple[Conjunct, ...]

    @property
    def holds(self) -> bool:
        return len(self.conjuncts) > 0 and all(c.holds for c in self.conjuncts)

    @property
    def unmet(self) -> Tuple[str, ...]:
        return tuple(c.label for c in self.conjuncts if not c.holds)

    def describe(self) -> str:
        verdict = "TRUE" if self.holds else "FALSE"
        parts = "; ".join(
            f"{c.label}={'T' if c.holds else 'F'} ({c.evidence})"
            for c in self.conjuncts
        )
        return f"{self.name}={verdict}: {parts}"


def _cx(b: BBox) -> float:
    return (b[1] + b[3]) / 2.0


def _cy(b: BBox) -> float:
    return (b[0] + b[2]) / 2.0


def aligned_to_column(obj: BBox, ref_x: float, tol: float = 2.0) -> Conjunct:
    cx = _cx(obj)
    return Conjunct("col_aligned", abs(cx - ref_x) <= tol,
                    f"obj.x={cx:.1f} vs ref.x={ref_x:.1f} tol={tol}")


def aligned_to_row(obj: BBox, ref_y: float, tol: float = 2.0) -> Conjunct:
    cy = _cy(obj)
    return Conjunct("row_aligned", abs(cy - ref_y) <= tol,
                    f"obj.y={cy:.1f} vs ref.y={ref_y:.1f} tol={tol}")


def beyond_edge(obj: BBox, edge: float, axis: str, direction: int) -> Conjunct:
    """direction +1: object center is beyond `edge` in the increasing
    direction of `axis` ('y' or 'x'); -1: in the decreasing direction."""
    pos = _cy(obj) if axis == "y" else _cx(obj)
    holds = (pos > edge) if direction > 0 else (pos < edge)
    rel = ">" if direction > 0 else "<"
    return Conjunct(f"beyond_{axis}_edge", holds,
                    f"obj.{axis}={pos:.1f} {rel} edge={edge:.1f}")


def composed(name: str, *conjuncts: Conjunct) -> RelationCheck:
    return RelationCheck(name, tuple(conjuncts))


def under_open_end(obj: BBox, struct_col: float, tip_pos: float, *,
                   axis: str = "y", open_dir: int = 1,
                   col_tol: float = 2.0) -> RelationCheck:
    """Object is positioned to enter a fixture through its OPEN end: aligned to
    the fixture's column AND beyond its open tip (so a move along the fixture
    inserts it).  For a top-anchored vertical shaft whose open end is the
    bottom tip: axis='y', open_dir=+1 (below the tip).  Impale-ready iff this
    holds.  Robotics: part aligned under an insertion bore, past its mouth."""
    perp_conj = (aligned_to_column(obj, struct_col, col_tol) if axis == "y"
                 else aligned_to_row(obj, struct_col, col_tol))
    return composed("under_open_end", perp_conj,
                    beyond_edge(obj, tip_pos, axis, open_dir))


def at_pose(obj: BBox, ref_y: float, ref_x: float, tol: float = 2.0) -> RelationCheck:
    """Object at a target 2-D pose: row AND column both matched."""
    return composed("at_pose", aligned_to_row(obj, ref_y, tol),
                    aligned_to_column(obj, ref_x, tol))


def bbox_area(bbox: BBox) -> float:
    y0, x0, y1, x1 = bbox
    return max(0.0, y1 - y0 + 1.0) * max(0.0, x1 - x0 + 1.0)


def bbox_iou(a: BBox, b: BBox) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    iy0 = max(ay0, by0)
    ix0 = max(ax0, bx0)
    iy1 = min(ay1, by1)
    ix1 = min(ax1, bx1)
    if iy0 > iy1 or ix0 > ix1:
        return 0.0
    inter = bbox_area((iy0, ix0, iy1, ix1))
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def center_distance(a: BBox, b: BBox) -> float:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    acy, acx = (ay0 + ay1) / 2.0, (ax0 + ax1) / 2.0
    bcy, bcx = (by0 + by1) / 2.0, (bx0 + bx1) / 2.0
    return ((acy - bcy) ** 2 + (acx - bcx) ** 2) ** 0.5


def extract_relations(
    objects: Iterable[TrackedObject],
    *,
    alignment_tolerance: float = 2.0,
    near_distance: float = 8.0,
) -> Tuple[Relation, ...]:
    """Return pairwise spatial relations for the current object set."""
    objs = list(objects)
    relations: list[Relation] = []
    for a, b in permutations(objs, 2):
        ay0, ax0, ay1, ax1 = a.bbox
        by0, bx0, by1, bx1 = b.bbox
        acy, acx = a.center
        bcy, bcx = b.center

        if bbox_iou(a.bbox, b.bbox) > 0:
            relations.append(Relation(a.object_id, "overlaps", b.object_id))
        if ax1 < bx0:
            relations.append(Relation(a.object_id, "left_of", b.object_id))
        if ax0 > bx1:
            relations.append(Relation(a.object_id, "right_of", b.object_id))
        if ay1 < by0:
            relations.append(Relation(a.object_id, "above", b.object_id))
        if ay0 > by1:
            relations.append(Relation(a.object_id, "below", b.object_id))
        if abs(acy - bcy) <= alignment_tolerance:
            relations.append(Relation(a.object_id, "row_aligned", b.object_id))
        if abs(acx - bcx) <= alignment_tolerance:
            relations.append(Relation(a.object_id, "col_aligned", b.object_id))
        dist = center_distance(a.bbox, b.bbox)
        if dist <= near_distance:
            strength = max(0.0, 1.0 - dist / max(near_distance, 1e-6))
            relations.append(Relation(a.object_id, "near", b.object_id, strength=strength))
    return tuple(relations)


__all__ = [
    "bbox_area", "bbox_iou", "center_distance", "extract_relations",
    "Conjunct", "RelationCheck", "composed",
    "aligned_to_column", "aligned_to_row", "beyond_edge",
    "under_open_end", "at_pose",
]
