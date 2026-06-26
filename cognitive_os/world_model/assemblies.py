"""Domain-neutral helpers for object-candidate assemblies.

Trackers are only as good as the detections they receive.  Many visual
domains expose both atomic detections and larger composite objects: ARC
frames have small connected components that form tools or widgets, while
robotics scenes have parts attached to manipulable objects.  This module
keeps that assembly step generic by using only geometry, adjacency, and
surface-like shape cues.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .schema import Detection


@dataclass(frozen=True)
class AssemblyConfig:
    """Geometric thresholds for assembling detections into candidates."""

    max_gap: float = 2.0
    min_parts: int = 1
    min_mask_area: float = 1.0
    large_area_fraction: float = 0.08
    panel_fill_ratio: float = 0.75
    panel_min_extent_fraction: float = 0.15
    border_span_fraction: float = 0.70


def bbox_extent(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    y0, x0, y1, x1 = bbox
    return (max(0.0, y1 - y0 + 1.0), max(0.0, x1 - x0 + 1.0))


def bbox_gap(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """Return Chebyshev gap between two inclusive boxes.

    Overlapping or touching boxes have gap 0.
    """

    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    dy = max(0.0, by0 - ay1 - 1.0, ay0 - by1 - 1.0)
    dx = max(0.0, bx0 - ax1 - 1.0, ax0 - bx1 - 1.0)
    return max(dy, dx)


def is_surface_like(
    detection: Detection,
    *,
    frame_size: tuple[float, float] | None = None,
    config: AssemblyConfig | None = None,
) -> bool:
    """Return True for large filled support/background regions.

    The rule intentionally uses only geometric cues.  A detector may pass
    ``attributes["fill_ratio"]``; otherwise the function falls back to
    mask-area divided by bbox area when available.
    """

    cfg = config or AssemblyConfig()
    h, w = bbox_extent(detection.bbox)
    bbox_area = max(1.0, h * w)
    mask_area = float(detection.mask_area if detection.mask_area is not None else bbox_area)
    fill_ratio = detection.attributes.get("fill_ratio")
    if fill_ratio is None:
        fill_ratio = mask_area / bbox_area
    fill_ratio = float(fill_ratio)

    if frame_size is None:
        return False

    fh, fw = max(1.0, float(frame_size[0])), max(1.0, float(frame_size[1]))
    frame_area = fh * fw
    if mask_area >= cfg.large_area_fraction * frame_area:
        return True
    if (
        fill_ratio >= cfg.panel_fill_ratio
        and h >= cfg.panel_min_extent_fraction * fh
        and w >= cfg.panel_min_extent_fraction * fw
    ):
        return True
    if fill_ratio >= 0.70 and (w >= cfg.border_span_fraction * fw or h >= cfg.border_span_fraction * fh):
        return True
    return False


def assemble_detections(
    detections: Iterable[Detection],
    *,
    frame_size: tuple[float, float] | None = None,
    config: AssemblyConfig | None = None,
    include_surface_like: bool = False,
) -> tuple[Detection, ...]:
    """Group nearby non-surface detections into composite candidates."""

    cfg = config or AssemblyConfig()
    atoms = list(detections)
    if not include_surface_like:
        atoms = [d for d in atoms if not is_surface_like(d, frame_size=frame_size, config=cfg)]

    parent = list(range(len(atoms)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, left in enumerate(atoms):
        for j in range(i + 1, len(atoms)):
            if bbox_gap(left.bbox, atoms[j].bbox) <= cfg.max_gap:
                union(i, j)

    buckets: dict[int, list[Detection]] = {}
    for i, det in enumerate(atoms):
        buckets.setdefault(find(i), []).append(det)

    assemblies: list[Detection] = []
    for idx, parts in enumerate(buckets.values(), start=1):
        if len(parts) < cfg.min_parts:
            continue
        area = sum(float(p.mask_area if p.mask_area is not None else bbox_extent(p.bbox)[0] * bbox_extent(p.bbox)[1]) for p in parts)
        if area < cfg.min_mask_area:
            continue
        y0 = min(p.bbox[0] for p in parts)
        x0 = min(p.bbox[1] for p in parts)
        y1 = max(p.bbox[2] for p in parts)
        x1 = max(p.bbox[3] for p in parts)
        colors = tuple(sorted({p.color for p in parts}, key=repr))
        labels = tuple(sorted({p.label for p in parts if p.label}, key=repr))
        assemblies.append(
            Detection(
                bbox=(y0, x0, y1, x1),
                label="assembly",
                color=colors,
                mask_area=area,
                attributes={
                    "assembly_index": idx,
                    "part_count": len(parts),
                    "part_colors": colors,
                    "part_labels": labels,
                    "part_bboxes": tuple(p.bbox for p in parts),
                    "part_areas": tuple(p.mask_area for p in parts),
                },
            )
        )

    return tuple(sorted(assemblies, key=lambda d: (-(d.mask_area or 0.0), d.bbox)))


def detections_intersecting_bbox(
    detections: Iterable[Detection],
    bbox: tuple[float, float, float, float],
) -> tuple[Detection, ...]:
    """Return detections whose bbox overlaps ``bbox``."""

    y0, x0, y1, x1 = bbox
    out: list[Detection] = []
    for det in detections:
        dy0, dx0, dy1, dx1 = det.bbox
        if max(y0, dy0) <= min(y1, dy1) and max(x0, dx0) <= min(x1, dx1):
            out.append(det)
    return tuple(out)


def describe_assemblies(detections: Sequence[Detection]) -> tuple[dict, ...]:
    """Compact serializable summary for diagnostics or LLM context."""

    rows: list[dict] = []
    for i, det in enumerate(detections, start=1):
        h, w = bbox_extent(det.bbox)
        rows.append(
            {
                "index": i,
                "label": det.label,
                "bbox": tuple(round(float(v), 3) for v in det.bbox),
                "extent": (round(h, 3), round(w, 3)),
                "mask_area": round(float(det.mask_area or 0.0), 3),
                "colors": det.color,
                "part_count": det.attributes.get("part_count"),
            }
        )
    return tuple(rows)


__all__ = [
    "AssemblyConfig",
    "assemble_detections",
    "bbox_extent",
    "bbox_gap",
    "describe_assemblies",
    "detections_intersecting_bbox",
    "is_surface_like",
]
