"""
Tests for modules/fusion/track_kalman.py.
"""

import numpy as np

from modules.fusion.track_kalman import DepthKalmanFilter


def test_converges_to_constant_depth_with_near_zero_velocity():
    np.random.seed(0)
    kf = DepthKalmanFilter(initial_depth=1.0, process_noise=0.01, measurement_noise=0.05)
    for reading in 1.0 + np.random.normal(0, 0.05, 50):
        kf.update(float(reading))
    assert abs(kf.depth - 1.0) < 0.05
    assert abs(kf.velocity) < 0.02


def test_smooths_noisy_measurements():
    np.random.seed(0)
    readings = 1.0 + np.random.normal(0, 0.05, 50)
    kf = DepthKalmanFilter(initial_depth=1.0, process_noise=0.01, measurement_noise=0.05)
    outputs = []
    for reading in readings:
        kf.update(float(reading))
        outputs.append(kf.depth)
    assert np.var(outputs) < np.var(readings)


def test_velocity_converges_to_true_slope_for_linear_ramp():
    np.random.seed(1)
    kf = DepthKalmanFilter(initial_depth=1.0, process_noise=0.02, measurement_noise=0.01)
    true_velocity = 0.03
    depth = 1.0
    for _ in range(60):
        depth += true_velocity
        kf.update(depth + np.random.normal(0, 0.01))
    assert abs(kf.velocity - true_velocity) < 0.01


def test_predict_only_coasts_forward_at_last_velocity():
    kf = DepthKalmanFilter(initial_depth=1.0, process_noise=0.01, measurement_noise=0.05)
    v, d = 0.02, 1.0
    for _ in range(10):
        d += v
        kf.update(d)
    depth_before, velocity_before = kf.depth, kf.velocity

    for _ in range(3):
        kf.predict()

    assert abs(kf.depth - (depth_before + 3 * velocity_before)) < 1e-6


def test_higher_process_noise_tracks_step_change_faster():
    true_value = 2.0
    kf_low_q = DepthKalmanFilter(initial_depth=1.0, process_noise=0.001, measurement_noise=0.1)
    kf_high_q = DepthKalmanFilter(initial_depth=1.0, process_noise=0.5, measurement_noise=0.1)
    for _ in range(5):
        kf_low_q.update(true_value)
        kf_high_q.update(true_value)

    assert abs(kf_high_q.depth - true_value) < abs(kf_low_q.depth - true_value)


def test_lower_measurement_noise_tracks_step_change_faster():
    true_value = 2.0
    kf_low_r = DepthKalmanFilter(initial_depth=1.0, process_noise=0.05, measurement_noise=0.01)
    kf_high_r = DepthKalmanFilter(initial_depth=1.0, process_noise=0.05, measurement_noise=5.0)
    for _ in range(5):
        kf_low_r.update(true_value)
        kf_high_r.update(true_value)

    assert abs(kf_low_r.depth - true_value) < abs(kf_high_r.depth - true_value)
