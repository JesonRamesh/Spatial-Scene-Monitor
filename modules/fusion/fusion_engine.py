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

# Floor for depth_smoothed when fitting the ego-motion scale factor, guarding
# division/squaring blowup if a track's depth ever drifts near zero. Real
# depth values cluster around 1.0 (the per-frame median), so this floor is
# never expected to bind in practice.
_MIN_DEPTH_FOR_EGO_MOTION_FIT = 0.05


class FusionEngine:
    """
    Owns the TrackStateStore and updates it once per frame.

    Per frame: normalises the raw depth map (DepthEstimator deliberately
    doesn't), then for every TrackedObject either births a new TrackState or
    advances an existing one's Kalman filter, estimates and compensates for
    ego-motion, ages every track, computes risk, and drops tracks that have
    been missing too long.

    Ego-motion compensation: depth_velocity alone can't distinguish "the
    camera is driving toward a stationary object" from "the object is moving
    toward the camera" — both produce identical increasing-disparity readings
    from a single moving camera with no other motion sensor. Confirmed
    directly against real KITTI footage: every parked car along a street the
    dashcam drove past showed APPROACHING for the entire approach, despite
    never moving.

    A flat per-frame velocity offset (first version of this fix) isn't
    enough: motion parallax means a stationary object's own depth_velocity
    from ego-motion alone scales with the SQUARE of how close it already is
    (nearby things sweep past faster than far things at the same vehicle
    speed) — so two equally-stationary cars at different distances need
    different baselines, not one shared number. _fit_ego_motion_scale()
    fits expected_velocity = k * depth_smoothed**2 from the population of
    currently tracked objects (k acts as a stand-in for the vehicle's own
    speed, inferred from the scene rather than a sensor), and _compute_risk()
    classifies on relative_velocity = depth_velocity - k * depth_smoothed**2.
    See CLAUDE.md Critical Design Decision on ego-motion for the derivation,
    the documented limitation with fewer than 2 tracks, and validation
    against KITTI's real GPS/IMU (oxts) vehicle-speed data.

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
        self.last_ego_motion_scale: float = 0.0

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

        ego_motion_scale = self._fit_ego_motion_scale()
        self.last_ego_motion_scale = ego_motion_scale  # exposed for offline validation against real ego-speed

        for state in self.track_states.values():
            state.age += 1
            depth_sq = max(state.depth_smoothed, _MIN_DEPTH_FOR_EGO_MOTION_FIT) ** 2
            expected_velocity = ego_motion_scale * depth_sq
            state.relative_velocity = state.depth_velocity - expected_velocity
            state.risk_level = self._compute_risk(state.relative_velocity)

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

    def _fit_ego_motion_scale(self) -> float:
        """
        Fits k in expected_velocity = k * depth_smoothed**2 — the ego-motion
        contribution to a STATIONARY object's depth_velocity, as a function
        of its own current depth.

        Derivation: disparity ~= C / Z for true distance Z and some camera-
        and-normalisation-dependent constant C. For a stationary object with
        the camera closing at speed V_ego, dZ/dt = -V_ego, so:
            d(disparity)/dt = -C/Z^2 * dZ/dt = V_ego * (disparity^2 / C)
        i.e. depth_velocity = k * depth_smoothed**2, where k = V_ego / C
        bundles the (unknown) true ego-speed and calibration constant into
        one fittable number per frame. k acts as a stand-in for the
        vehicle's own speed, inferred from the scene rather than a sensor —
        validated against KITTI's real recorded vehicle speed (oxts 'vf')
        in scripts/validate_ego_motion.py.

        k is fit as the median of (depth_velocity_i / depth_smoothed_i**2)
        across all currently tracked objects — median, not mean, for the
        same reason extract_box_depth uses median over mean: robust to the
        minority of tracks that ARE genuinely moving on their own, which
        shouldn't drag the fit away from the majority stationary-background
        signal.

        Honest limitation: with fewer than 2 tracks, the median collapses to
        that single track's own ratio, making its expected_velocity exactly
        equal to its own depth_velocity and relative_velocity always exactly
        0 (STATIC) regardless of its real motion — there's no other track to
        compare against. Degrades gracefully rather than failing loudly.
        """
        ratios = [
            state.depth_velocity / (max(state.depth_smoothed, _MIN_DEPTH_FOR_EGO_MOTION_FIT) ** 2)
            for state in self.track_states.values()
        ]
        if not ratios:
            return 0.0
        return float(np.median(ratios))

    def _compute_risk(self, relative_velocity: float) -> RiskLevel:
        """
        Classifies on relative_velocity (ego-motion-compensated), not raw
        depth_velocity — see _estimate_ego_motion(). Disparity convention:
        higher = closer, so an object closing distance faster than the
        background has positive relative_velocity. See RiskLevel docstring
        in data_types.py.
        """
        if relative_velocity > self.approach_threshold:
            return RiskLevel.APPROACHING
        if relative_velocity < -self.approach_threshold:
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
