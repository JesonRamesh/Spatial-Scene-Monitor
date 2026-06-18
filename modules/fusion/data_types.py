"""
Shared data structures for the Spatial Scene Monitor pipeline.

All modules import from here. Nothing here imports from anywhere else in the
project — zero internal dependencies by design.

Reading order:
  Detection       — raw output from the detector (one box, one frame)
  TrackedObject   — Detection + persistent track_id from ByteTrack
  RiskLevel       — enum for the three approach states
  TrackState      — full per-ID spatial state owned by FusionEngine
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

# Avoid a circular import: track_kalman imports data_types, so we reference
# DepthKalmanFilter only as a type annotation and guard it behind TYPE_CHECKING.
if TYPE_CHECKING:
    from modules.fusion.track_kalman import DepthKalmanFilter


# ---------------------------------------------------------------------------
# Road-scene class filter
# ---------------------------------------------------------------------------

ROAD_CLASS_IDS: list[int] = [0, 1, 2, 3, 5, 7]

COCO_CLASS_NAMES: dict[int, str] = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """
    Single bounding box from the detector, before tracking.

    Coordinates are pixel-space at the original frame resolution, not the
    detector's internal resized resolution.
    """
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def to_bytetrack_row(self) -> list[float]:
        """[x1, y1, x2, y2, score] — format expected by ByteTracker.update()."""
        return [self.x1, self.y1, self.x2, self.y2, self.confidence]


# ---------------------------------------------------------------------------
# TrackedObject
# ---------------------------------------------------------------------------

@dataclass
class TrackedObject:
    """
    A Detection that has been assigned a persistent track_id by ByteTrack.

    Coordinates here reflect ByteTrack's Kalman-smoothed box, not the raw
    detection. FusionEngine skips unconfirmed tracks (is_confirmed=False).
    """
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    is_confirmed: bool = True

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


# ---------------------------------------------------------------------------
# RiskLevel
# ---------------------------------------------------------------------------

class RiskLevel(Enum):
    """
    Directional risk classification for a tracked object.

    Sign convention: Depth Anything v2 outputs disparity (higher = closer).
    So an approaching object has INCREASING disparity, i.e. depth_velocity > 0.
    FusionEngine stores disparity as "depth" and uses this convention throughout.

    Thresholds come from configs/default.yaml (risk.approach_threshold).
    """
    APPROACHING = "APPROACHING"   # depth_velocity > +threshold
    RECEDING    = "RECEDING"      # depth_velocity < -threshold
    STATIC      = "STATIC"        # |depth_velocity| <= threshold

    @property
    def colour_bgr(self) -> tuple[int, int, int]:
        """OpenCV BGR colour for bounding-box and badge rendering."""
        return {
            RiskLevel.APPROACHING: (0,   0,   220),   # red
            RiskLevel.RECEDING:    (0,   180, 0  ),   # green
            RiskLevel.STATIC:      (200, 200, 200),   # grey
        }[self]


# ---------------------------------------------------------------------------
# TrackState
# ---------------------------------------------------------------------------

@dataclass
class TrackState:
    """
    Full spatial state for one tracked object, maintained across frames.

    Owned and updated by FusionEngine. Read by Visualiser and JSONLogger.
    All depth values are normalised relative disparity units — not metres.
    Higher disparity = closer to camera.
    """
    track_id: int
    class_id: int
    class_name: str

    # Most recent known bounding box (x1, y1, x2, y2). Not part of the original
    # CLAUDE.md spec — added when building the Visualiser (module 9), which
    # needs box corners to draw a rectangle and a label anchor point; the
    # trajectory_2d centroid alone isn't enough geometry for that. Updated
    # every frame FusionEngine sees a real detection for this track; left
    # unchanged (last known position) on frames where the track is missing.
    box: tuple[float, float, float, float] | None = None

    # Depth (normalised disparity units)
    depth_raw: float = 0.0        # raw reading from depth map this frame
    depth_smoothed: float = 0.0   # Kalman-filtered output
    depth_velocity: float = 0.0   # d(depth_smoothed)/dt; positive = approaching.
                                   # NOT ego-motion-compensated — see relative_velocity.

    # depth_velocity minus FusionEngine's estimated ego-motion baseline for
    # this frame. A car parked at the roadside shows positive depth_velocity
    # throughout an approach (the camera is moving toward it), but should
    # show relative_velocity near zero, since it isn't moving relative to
    # the rest of the (also static) scene. This is what _compute_risk()
    # actually classifies on — see CLAUDE.md "Critical Design Decisions"
    # for why depth_velocity alone can't distinguish ego-motion from
    # genuine object motion.
    relative_velocity: float = 0.0

    # Trajectory ring buffers; maxlen is set at track creation via TrackState.create()
    trajectory_2d:    deque = field(default_factory=lambda: deque(maxlen=30))
    trajectory_depth: deque = field(default_factory=lambda: deque(maxlen=30))

    # Risk
    risk_level: RiskLevel = RiskLevel.STATIC

    # Bookkeeping
    age: int = 0                    # frames since track was first created
    frames_since_update: int = 0    # frames since last matched detection
    kalman: "DepthKalmanFilter | None" = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        track_id: int,
        class_id: int,
        class_name: str,
        initial_box: tuple[float, float, float, float],
        initial_depth: float,
        trajectory_maxlen: int,
        kalman: "DepthKalmanFilter",
    ) -> "TrackState":
        """
        Factory for a brand-new track.

        Uses a classmethod rather than __post_init__ because dataclass
        __post_init__ can't easily accept extra constructor args (like
        trajectory_maxlen) that aren't stored as fields.
        """
        return cls(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            box=initial_box,
            depth_raw=initial_depth,
            depth_smoothed=initial_depth,
            depth_velocity=0.0,
            trajectory_2d=deque(maxlen=trajectory_maxlen),
            trajectory_depth=deque(maxlen=trajectory_maxlen),
            kalman=kalman,
        )

    def to_dict(self) -> dict:
        """JSON-serialisable snapshot. Excludes the non-serialisable Kalman object."""
        return {
            "track_id":           self.track_id,
            "class_id":           self.class_id,
            "class_name":         self.class_name,
            "box":                list(self.box) if self.box is not None else None,
            "depth_raw":          round(self.depth_raw, 4),
            "depth_smoothed":     round(self.depth_smoothed, 4),
            "depth_velocity":     round(self.depth_velocity, 4),
            "relative_velocity":  round(self.relative_velocity, 4),
            "risk_level":         self.risk_level.value,
            "trajectory_2d":      [list(p) for p in self.trajectory_2d],
            "trajectory_depth":   list(self.trajectory_depth),
            "age":                self.age,
            "frames_since_update": self.frames_since_update,
        }
