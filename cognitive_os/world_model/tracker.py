"""Persistent object tracker for normalized detection streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

from .relations import bbox_iou, center_distance, extract_relations
from .schema import Detection, ObjectChange, ObservationFrame, TrackedObject, WorldDelta, WorldState


@dataclass(frozen=True)
class TrackerConfig:
    """Association thresholds for the generic object tracker."""

    iou_threshold: float = 0.35
    max_center_distance: float = 8.0
    max_missing_frames: int = 3
    move_epsilon: float = 1.0
    resize_epsilon: float = 1.0


class ObjectTracker:
    """Associate detections across frames into persistent objects.

    The tracker is deliberately semantic-free.  It estimates identity and
    object-level changes; TUTOR, miners, or domain adapters decide what an
    object means.
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        self.config = config or TrackerConfig()
        self._objects: Dict[str, TrackedObject] = {}
        self._next_id = 1

    @property
    def objects(self) -> Dict[str, TrackedObject]:
        return dict(self._objects)

    def reset(self) -> None:
        self._objects.clear()
        self._next_id = 1

    def update(self, frame: ObservationFrame) -> WorldState:
        matches, unmatched_detections, unmatched_objects = self._associate(frame.detections)
        changes: list[ObjectChange] = []
        new_objects: Dict[str, TrackedObject] = {}

        for obj_id, detection in matches:
            before = self._objects[obj_id]
            after = self._updated_object(before, detection, frame.timestamp)
            new_objects[obj_id] = after
            changes.extend(self._diff_object(before, after))

        for detection in unmatched_detections:
            obj = self._new_object(detection, frame.timestamp)
            new_objects[obj.object_id] = obj
            changes.append(
                ObjectChange(
                    object_id=obj.object_id,
                    kind="appeared",
                    after_bbox=obj.bbox,
                    after_label=obj.label,
                    after_color=obj.color,
                )
            )

        for obj_id in unmatched_objects:
            before = self._objects[obj_id]
            missing = TrackedObject(
                object_id=before.object_id,
                bbox=before.bbox,
                label=before.label,
                color=before.color,
                first_seen=before.first_seen,
                last_seen=before.last_seen,
                observations=before.observations,
                missing_frames=before.missing_frames + 1,
                source_ids=before.source_ids,
                attributes=before.attributes,
            )
            if missing.missing_frames <= self.config.max_missing_frames:
                new_objects[obj_id] = missing
            else:
                changes.append(
                    ObjectChange(
                        object_id=obj_id,
                        kind="disappeared",
                        before_bbox=before.bbox,
                        before_label=before.label,
                        before_color=before.color,
                    )
                )

        self._objects = new_objects
        relations = extract_relations(self._objects.values())
        return WorldState(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            source_id=frame.source_id,
            objects=dict(self._objects),
            relations=relations,
            delta=WorldDelta(tuple(changes)),
        )

    def _associate(
        self,
        detections: Iterable[Detection],
    ) -> Tuple[list[tuple[str, Detection]], list[Detection], list[str]]:
        detections = list(detections)
        available_objects = set(self._objects.keys())
        matches: list[tuple[str, Detection]] = []
        unmatched_detections: list[Detection] = []

        for det in detections:
            obj_id = self._best_match(det, available_objects)
            if obj_id is None:
                unmatched_detections.append(det)
                continue
            available_objects.remove(obj_id)
            matches.append((obj_id, det))

        return matches, unmatched_detections, sorted(available_objects)

    def _best_match(self, detection: Detection, candidates: set[str]) -> Optional[str]:
        if detection.source_id:
            for obj_id in candidates:
                if detection.source_id in self._objects[obj_id].source_ids:
                    return obj_id

        best_id: Optional[str] = None
        best_score = 0.0
        for obj_id in candidates:
            obj = self._objects[obj_id]
            appearance_match = (
                obj.label == detection.label
                and obj.color == detection.color
            )
            iou = bbox_iou(obj.bbox, detection.bbox)
            dist = center_distance(obj.bbox, detection.bbox)
            distance_score = max(0.0, 1.0 - dist / max(self.config.max_center_distance, 1e-6))
            score = max(iou, distance_score * 0.75)
            if not appearance_match:
                score *= 0.5
            if score > best_score:
                best_score = score
                best_id = obj_id

        if best_id is None:
            return None
        if best_score >= self.config.iou_threshold:
            return best_id
        return None

    def _new_object(self, detection: Detection, timestamp: float) -> TrackedObject:
        obj_id = f"obj_{self._next_id}"
        self._next_id += 1
        source_ids = (detection.source_id,) if detection.source_id else ()
        return TrackedObject(
            object_id=obj_id,
            bbox=detection.bbox,
            label=detection.label,
            color=detection.color,
            first_seen=timestamp,
            last_seen=timestamp,
            observations=1,
            source_ids=source_ids,
            attributes=dict(detection.attributes),
        )

    @staticmethod
    def _updated_object(before: TrackedObject, detection: Detection, timestamp: float) -> TrackedObject:
        source_ids = before.source_ids
        if detection.source_id and detection.source_id not in source_ids:
            source_ids = tuple([*source_ids, detection.source_id])
        return TrackedObject(
            object_id=before.object_id,
            bbox=detection.bbox,
            label=detection.label,
            color=detection.color,
            first_seen=before.first_seen,
            last_seen=timestamp,
            observations=before.observations + 1,
            missing_frames=0,
            source_ids=source_ids,
            attributes=dict(detection.attributes),
        )

    def _diff_object(self, before: TrackedObject, after: TrackedObject) -> list[ObjectChange]:
        changes: list[ObjectChange] = []
        move = center_distance(before.bbox, after.bbox)
        if move > self.config.move_epsilon:
            changes.append(
                ObjectChange(
                    object_id=after.object_id,
                    kind="moved",
                    before_bbox=before.bbox,
                    after_bbox=after.bbox,
                    magnitude=move,
                )
            )

        bh, bw = before.size
        ah, aw = after.size
        resize = abs(bh - ah) + abs(bw - aw)
        if resize > self.config.resize_epsilon:
            changes.append(
                ObjectChange(
                    object_id=after.object_id,
                    kind="resized",
                    before_bbox=before.bbox,
                    after_bbox=after.bbox,
                    magnitude=resize,
                )
            )

        if before.color != after.color:
            changes.append(
                ObjectChange(
                    object_id=after.object_id,
                    kind="recolored",
                    before_color=before.color,
                    after_color=after.color,
                )
            )

        if before.label != after.label:
            changes.append(
                ObjectChange(
                    object_id=after.object_id,
                    kind="relabeled",
                    before_label=before.label,
                    after_label=after.label,
                )
            )

        if not changes:
            changes.append(ObjectChange(object_id=after.object_id, kind="unchanged"))
        return changes


__all__ = ["ObjectTracker", "TrackerConfig"]
