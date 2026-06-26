"""Reusable object-world model for COS.

Domain adapters convert raw perception (ARC frames, camera frames, RGB-D
detectors, simulation state) into ``ObservationFrame`` detections.  The
core tracker then provides persistent object identities, object-level
changes, and generic spatial relations without knowing domain semantics.
"""

from .schema import (
    BBox,
    Detection,
    ObservationFrame,
    ObjectChange,
    Relation,
    TrackedObject,
    WorldDelta,
    WorldState,
)
from .tracker import ObjectTracker, TrackerConfig
from .relations import bbox_area, bbox_iou, center_distance, extract_relations
from .assemblies import (
    AssemblyConfig,
    assemble_detections,
    bbox_extent,
    bbox_gap,
    describe_assemblies,
    detections_intersecting_bbox,
    is_surface_like,
)

__all__ = [
    "AssemblyConfig",
    "BBox",
    "Detection",
    "ObservationFrame",
    "ObjectChange",
    "ObjectTracker",
    "Relation",
    "TrackedObject",
    "TrackerConfig",
    "WorldDelta",
    "WorldState",
    "assemble_detections",
    "bbox_area",
    "bbox_extent",
    "bbox_gap",
    "bbox_iou",
    "center_distance",
    "describe_assemblies",
    "detections_intersecting_bbox",
    "extract_relations",
    "is_surface_like",
]
