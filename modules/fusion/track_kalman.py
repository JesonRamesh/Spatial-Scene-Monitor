"""
Per-track 1D Kalman filter on depth for the Spatial Scene Monitor.
"""

from __future__ import annotations

import numpy as np


class DepthKalmanFilter:
    """
    Kalman filter modeling one track's depth as a constant-velocity process.

    State vector x = [depth, depth_velocity]^T. Measurement is a single
    scalar depth reading (from depth_utils.extract_box_depth). One frame
    is treated as one discrete time step (dt = 1), so velocity is already
    in "depth units per frame" — matching RiskLevel's threshold, which is
    also defined in per-frame units.

    Lives in modules/fusion, not modules/tracking, because it operates on
    depth, not 2D box position — ByteTrack has its own separate Kalman
    filter for box motion that this class knows nothing about.
    """

    def __init__(
        self,
        initial_depth: float,
        process_noise: float = 0.01,
        measurement_noise: float = 0.1,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

        # Constant-velocity model: depth_next = depth + velocity; velocity_next = velocity.
        self._F = np.array([[1.0, 1.0], [0.0, 1.0]])
        # We only ever observe depth directly, never velocity.
        self._H = np.array([[1.0, 0.0]])
        # Single scalar per config (configs/default.yaml), not a full noise model —
        # CLAUDE.md exposes one tunable number per noise source, not a matrix.
        self._Q = np.eye(2) * process_noise
        self._R = np.array([[measurement_noise]])

        self.x = np.array([[initial_depth], [0.0]])
        # Initial uncertainty: depth starts as confident as a single measurement
        # (measurement_noise); velocity starts completely unknown (wide prior).
        self.P = np.diag([measurement_noise, 1.0])

    @property
    def depth(self) -> float:
        """Current smoothed depth estimate."""
        return float(self.x[0, 0])

    @property
    def velocity(self) -> float:
        """Current depth-velocity estimate, in depth units per frame."""
        return float(self.x[1, 0])

    def predict(self) -> None:
        """
        Advance state by one frame with no new measurement.

        Used when extract_box_depth() returns None (box off-frame, etc.) —
        the filter coasts forward on its last velocity estimate instead of
        being corrected, per CLAUDE.md's Kalman-over-EMA rationale.
        """
        self.x = self._F @ self.x
        self.P = self._F @ self.P @ self._F.T + self._Q

    def update(self, measurement: float) -> None:
        """
        Advance one frame AND incorporate a real depth reading.

        Always predicts first, then corrects with the measurement — the
        standard predict-then-update Kalman cycle. Callers should call
        either this or predict() each frame, never neither.
        """
        self.predict()

        z = np.array([[measurement]])
        innovation = z - self._H @ self.x
        innovation_cov = self._H @ self.P @ self._H.T + self._R
        kalman_gain = self.P @ self._H.T @ np.linalg.inv(innovation_cov)

        self.x = self.x + kalman_gain @ innovation
        self.P = (np.eye(2) - kalman_gain @ self._H) @ self.P
