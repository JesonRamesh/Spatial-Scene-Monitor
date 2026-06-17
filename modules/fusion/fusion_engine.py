"""
Fusion engine for the Spatial Scene Monitor — the system's core.
"""

from __future__ import annotations

import numpy as np

from modules.depth.depth_utils import extract_box_depth, normalise_depth_map
from modules.fusion.data_types import RiskLevel, TrackedObject, TrackState
from modules.fusion.track_kalman import DepthKalmanFilter

# Neutral starting depth for a brand-new track whose very first depth reading
# failed (box at the frame edge). 1.0 is the "average scene depth" anchor
# point normalise_depth_map() establishes for every frame.
_NEUTRAL_INITIAL_DEPTH = 1.0


class FusionEngine:
    """
    Owns the TrackStateStore and updates it once per frame.

    Per frame: normalises the raw depth map (DepthEstimator deliberately
    doesn't), then for every TrackedObject either births a new TrackState or
    advances an existing one's Kalman filter, ages every track, computes risk,
    and drops tracks that have been missing too long.

    Note on track_id reuse: CLAUDE.md's gotcha about reinitialising the Kalman
    filter when ByteTrack reassigns an old ID to a new object assumes a backend
    that can reuse IDs mid-run. Our Tracker (modules/tracking/tracker.py) uses
    ultralytics' BYTETracker, whose ID counter (BaseTrack._count) increments
    monotonically for the lifetime of the process and is never reset unless
    something explicitly calls tracker.reset() — which nothing in this codebase
    does. So within one continuous run, a track_id is never reassigned to a
    different physical object, and there is no reliable signal here to detect
    such a case if a future tracker swap reintroduced it. Flagging this rather
    than adding speculative detection logic with nothing real to act on.
    """

    def __init__(
        self,
        trajectory_length: int = 30,
        kalman_process_noise: float = 0.01,
        kalman_measurement_noise: float = 0.1,
        max_frames_missing: int = 10,
        approach_threshold: float = 0.02,
    ) -> None:
        self.trajectory_length = trajectory_length
        self.kalman_process_noise = kalman_process_noise
        self.kalman_measurement_noise = kalman_measurement_noise
        self.max_frames_missing = max_frames_missing
        self.approach_threshold = approach_threshold

        self.track_states: dict[int, TrackState] = {}

    def update(self, tracked_objects: list[TrackedObject], depth_map: np.ndarray) -> dict[int, TrackState]:
        """
        Advance the TrackStateStore by one frame.

        depth_map is the RAW map from DepthEstimator.estimate() — normalisation
        happens here, once per frame, not in the depth module (separation of
        concerns per CLAUDE.md Critical Design Decision #1).
        """
        normalised_map = normalise_depth_map(depth_map)
        seen_ids: set[int] = set()

        for obj in tracked_objects:
            seen_ids.add(obj.track_id)
            depth_reading = extract_box_depth(normalised_map, obj.xyxy)

            if obj.track_id in self.track_states:
                self._update_seen_track(self.track_states[obj.track_id], obj, depth_reading)
            else:
                self._birth_track(obj, depth_reading)

        for track_id, state in self.track_states.items():
            if track_id not in seen_ids:
                self._update_missing_track(state)

        for state in self.track_states.values():
            state.age += 1
            state.risk_level = self._compute_risk(state.depth_velocity)

        self._cull_dead_tracks()
        return self.track_states

    def _birth_track(self, obj: TrackedObject, depth_reading: float | None) -> None:
        """A track_id we haven't seen before — create its TrackState and Kalman filter."""
        initial_depth = depth_reading if depth_reading is not None else _NEUTRAL_INITIAL_DEPTH

        kalman = DepthKalmanFilter(
            initial_depth=initial_depth,
            process_noise=self.kalman_process_noise,
            measurement_noise=self.kalman_measurement_noise,
        )
        state = TrackState.create(
            track_id=obj.track_id,
            class_id=obj.class_id,
            class_name=obj.class_name,
            initial_box=obj.xyxy,
            initial_depth=initial_depth,
            trajectory_maxlen=self.trajectory_length,
            kalman=kalman,
        )
        state.trajectory_2d.append(obj.center)
        state.trajectory_depth.append(initial_depth)
        self.track_states[obj.track_id] = state

    def _update_seen_track(self, state: TrackState, obj: TrackedObject, depth_reading: float | None) -> None:
        """
        Track matched a detection this frame (frames_since_update resets),
        independent of whether the depth reading itself succeeded.
        """
        state.frames_since_update = 0
        state.box = obj.xyxy

        if depth_reading is not None:
            state.kalman.update(depth_reading)
            state.depth_raw = depth_reading
        else:
            state.kalman.predict()

        state.depth_smoothed = state.kalman.depth
        state.depth_velocity = state.kalman.velocity
        state.trajectory_2d.append(obj.center)
        state.trajectory_depth.append(state.depth_smoothed)

    def _update_missing_track(self, state: TrackState) -> None:
        """
        Track exists but had no matching detection this frame (ByteTrack lost
        it). No box to extract depth from at all, so predict-only — and don't
        append to the trajectory, which should only record real observations.
        """
        state.kalman.predict()
        state.depth_smoothed = state.kalman.depth
        state.depth_velocity = state.kalman.velocity
        state.frames_since_update += 1

    def _compute_risk(self, depth_velocity: float) -> RiskLevel:
        """
        Disparity convention: higher = closer, so an approaching object has
        positive depth_velocity. See RiskLevel docstring in data_types.py.
        """
        if depth_velocity > self.approach_threshold:
            return RiskLevel.APPROACHING
        if depth_velocity < -self.approach_threshold:
            return RiskLevel.RECEDING
        return RiskLevel.STATIC

    def _cull_dead_tracks(self) -> None:
        dead_ids = [
            track_id
            for track_id, state in self.track_states.items()
            if state.frames_since_update > self.max_frames_missing
        ]
        for track_id in dead_ids:
            del self.track_states[track_id]
