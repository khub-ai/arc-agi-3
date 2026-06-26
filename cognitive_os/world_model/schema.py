"""Domain-neutral object-world schema.

This module defines the normalized perception boundary for COS.  ARC
frames, robot cameras, simulators, and detector pipelines should all
adapt their raw observations into ``ObservationFrame`` + ``Detection``
before the reusable object tracker sees them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


BBox = Tuple[float, float, float, float]  # (y0, x0, y1, x1), inclusive


@dataclass(frozen=True)
class Detection:
    """One source-specific detection normalized into COS coordinates.

    ``source_id`` is optional.  Camera detectors often provide track ids;
    ARC connected-components usually do not.  The tracker treats it as a
    strong but not required identity hint.
    """

    bbox: BBox
    label: Optional[str] = None
    source_id: Optional[str] = None
    confidence: float = 1.0
    color: Optional[Any] = None
    mask_area: Optional[float] = None
    embedding: Tuple[float, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        y0, x0, y1, x1 = self.bbox
        return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)

    @property
    def size(self) -> Tuple[float, float]:
        y0, x0, y1, x1 = self.bbox
        return (max(0.0, y1 - y0 + 1.0), max(0.0, x1 - x0 + 1.0))

    @property
    def appearance_key(self) -> Tuple[Any, ...]:
        return (
            self.label,
            self.color,
            round(float(self.confidence), 1),
        )


@dataclass(frozen=True)
class ObservationFrame:
    """Normalized perception frame from any vision source."""

    frame_id: str
    timestamp: float
    source_id: str
    detections: Tuple[Detection, ...]
    raw_ref: Optional[Any] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackedObject:
    """Persistent object identity estimated by the world model."""

    object_id: str
    bbox: BBox
    label: Optional[str]
    color: Optional[Any]
    first_seen: float
    last_seen: float
    observations: int
    missing_frames: int = 0
    source_ids: Tuple[str, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Tuple[float, float]:
        y0, x0, y1, x1 = self.bbox
        return ((y0 + y1) / 2.0, (x0 + x1) / 2.0)

    @property
    def size(self) -> Tuple[float, float]:
        y0, x0, y1, x1 = self.bbox
        return (max(0.0, y1 - y0 + 1.0), max(0.0, x1 - x0 + 1.0))


@dataclass(frozen=True)
class ObjectChange:
    """Object-level effect observed between two frames."""

    object_id: str
    kind: str
    before_bbox: Optional[BBox] = None
    after_bbox: Optional[BBox] = None
    before_label: Optional[str] = None
    after_label: Optional[str] = None
    before_color: Optional[Any] = None
    after_color: Optional[Any] = None
    magnitude: float = 0.0


@dataclass(frozen=True)
class Relation:
    """Generic spatial relation between two tracked objects."""

    subject_id: str
    relation: str
    object_id: str
    strength: float = 1.0


@dataclass(frozen=True)
class WorldDelta:
    """Object-level delta for the latest tracker update."""

    changes: Tuple[ObjectChange, ...] = ()

    def by_kind(self, kind: str) -> Tuple[ObjectChange, ...]:
        return tuple(c for c in self.changes if c.kind == kind)


@dataclass(frozen=True)
class WorldState:
    """Current object-world estimate."""

    frame_id: str
    timestamp: float
    source_id: str
    objects: Mapping[str, TrackedObject]
    relations: Tuple[Relation, ...] = ()
    delta: WorldDelta = field(default_factory=WorldDelta)


__all__ = [
    "BBox",
    "Detection",
    "ObservationFrame",
    "TrackedObject",
    "ObjectChange",
    "Relation",
    "WorldDelta",
    "WorldState",
]
