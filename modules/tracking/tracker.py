"""
ByteTrack wrapper for the Spatial Scene Monitor.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from ultralytics.trackers.byte_tracker import BYTETracker

from modules.fusion.data_types import COCO_CLASS_NAMES, Detection, ROAD_CLASS_IDS, TrackedObject


class _DetectionBatch:
    """
    Minimal Results-like adapter so BYTETracker.update() can consume our own
    Detection list. BYTETracker only ever touches .xywh, .conf, .cls, len(),
    and boolean-mask slicing (results[mask]) — this is that entire surface.
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> "_DetectionBatch":
        return _DetectionBatch(self.xywh[mask], self.conf[mask], self.cls[mask])


def _to_batch(detections: list[Detection]) -> _DetectionBatch:
    if not detections:
        return _DetectionBatch(
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    xywh = np.array(
        [[(d.x1 + d.x2) / 2.0, (d.y1 + d.y2) / 2.0, d.width, d.height] for d in detections],
        dtype=np.float32,
    )
    conf = np.array([d.confidence for d in detections], dtype=np.float32)
    cls = np.array([d.class_id for d in detections], dtype=np.float32)
    return _DetectionBatch(xywh, conf, cls)


class Tracker:
    """
    Thin wrapper around BYTETracker (ultralytics' bundled implementation —
    already a project dependency via Detector, so no second tracking
    codebase needs vendoring).

    Class-aware: one independent BYTETracker per class ID, so a car's lost
    track can never be re-matched against an incoming pedestrian detection
    just because their boxes happen to overlap (CLAUDE.md Design Decision #4).

    Note on track_id == -1: CLAUDE.md's gotcha about skipping unconfirmed
    tracks assumes a backend that emits a -1 sentinel. Ultralytics' BYTETracker
    instead filters unconfirmed tracks out internally (STrack.is_activated)
    before update() returns anything — so every TrackedObject this class
    produces is already confirmed. is_confirmed=True reflects that guarantee,
    not a computation we perform ourselves.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        class_filter: list[int] | None = None,
    ) -> None:
        self.class_filter = class_filter if class_filter is not None else list(ROAD_CLASS_IDS)

        args = SimpleNamespace(
            track_high_thresh=track_thresh,
            track_low_thresh=0.1,     # ByteTrack's own default; configs/default.yaml has no separate knob
            new_track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
            fuse_score=True,
        )
        # One tracker per class — independent ID space and IoU-matching pool.
        self._trackers: dict[int, BYTETracker] = {
            class_id: BYTETracker(args) for class_id in self.class_filter
        }

    def update(self, detections: list[Detection]) -> list[TrackedObject]:
        """
        Advance all per-class trackers by one frame and return active tracks.

        Detections are grouped by class_id and routed to that class's own
        BYTETracker instance, then results are merged back into one list.
        ByteTrack returns tracks in arbitrary order — callers must not rely
        on output order being stable across frames (per CLAUDE.md gotcha).
        """
        tracked_objects: list[TrackedObject] = []

        for class_id, tracker in self._trackers.items():
            class_detections = [d for d in detections if d.class_id == class_id]
            batch = _to_batch(class_detections)
            results = tracker.update(batch)  # [N, 8]: x1,y1,x2,y2,track_id,score,cls,idx

            for x1, y1, x2, y2, track_id, score, cls, _idx in results:
                tracked_objects.append(
                    TrackedObject(
                        track_id=int(track_id),
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                        confidence=float(score),
                        class_id=int(cls),
                        class_name=COCO_CLASS_NAMES.get(int(cls), "unknown"),
                        is_confirmed=True,
                    )
                )

        return tracked_objects
