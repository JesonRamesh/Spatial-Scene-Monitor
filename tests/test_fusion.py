"""
Tests for modules/fusion/fusion_engine.py.
"""

import numpy as np

from modules.fusion.data_types import RiskLevel, TrackedObject
from modules.fusion.fusion_engine import FusionEngine

_BOX = (70.0, 30.0, 130.0, 70.0)


def _make_depth_map(object_value: float, h: int = 100, w: int = 200) -> np.ndarray:
    m = np.full((h, w), 1.0, dtype=np.float32)
    m[40:60, 80:120] = object_value
    return m


def _car(track_id: int, box=_BOX) -> TrackedObject:
    return TrackedObject(
        track_id=track_id, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
        confidence=0.9, class_id=2, class_name="car",
    )


def _box_for_slot(i: int) -> tuple[float, float, float, float]:
    """Non-overlapping box for the i-th of several simultaneously tracked objects."""
    x1 = 50.0 + i * 150.0
    return (x1, 100.0, x1 + 100.0, 200.0)


def _make_multi_depth_map(values: list[float], h: int = 300, w: int = 800) -> np.ndarray:
    """One flat depth value per slot's box. A box's center-crop is always a
    subset of the box itself, so filling the whole box guarantees the median
    extraction reads exactly `values[i]` for slot i."""
    m = np.full((h, w), 1.0, dtype=np.float32)
    for i, val in enumerate(values):
        x1, y1, x2, y2 = _box_for_slot(i)
        m[int(y1):int(y2), int(x1):int(x2)] = val
    return m


def test_track_birth_creates_track_state():
    engine = FusionEngine()
    store = engine.update([_car(1)], _make_depth_map(1.0))
    assert 1 in store
    assert store[1].box == _BOX


def _ego_motion_trajectory(initial_depth: float, k: float, n_frames: int) -> list[float]:
    """
    Exact solution of dD/dt = k * D**2 — the depth trajectory a stationary
    object follows under the model's own assumption (constant-speed camera
    closing on it). Used to build test data that's actually self-consistent
    with the model, rather than a linear ramp (which is NOT a solution to
    that equation, and silently drifts the test's "background" tracks away
    from a true ego-motion-only signature the longer the test runs).
    """
    return [1.0 / (1.0 / initial_depth - k * t) for t in range(n_frames)]


def _multi_objects(prefix: int, depths: list[float]) -> list[TrackedObject]:
    return [
        TrackedObject(track_id=prefix + i, x1=_box_for_slot(i)[0], y1=_box_for_slot(i)[1],
                      x2=_box_for_slot(i)[2], y2=_box_for_slot(i)[3],
                      confidence=0.9, class_id=2, class_name="car")
        for i in range(len(depths))
    ]


def test_ego_motion_compensation_distinguishes_approaching_object_from_static_background():
    """
    3 'parked' background tracks follow the camera's own approach rate
    (k_bg); one foreground track has a genuinely higher closing rate
    (k_fg > k_bg) — e.g. another car actually merging toward the lane, not
    just sitting at the roadside. Only the foreground track should read
    APPROACHING.

    Uses _ego_motion_trajectory, not a linear ramp: an earlier version of
    this test used "background velocity + a flat extra amount" for the
    foreground track, which isn't a solution to the model's own dD/dt = k*D^2
    relationship. It worked at first but silently flipped to RECEDING after
    enough frames, once the foreground track's depth diverged far enough from
    the background's that the model's depth-squared extrapolation overshot.
    Real independently-moving objects don't follow a flat-offset trajectory
    either — a genuinely faster closing rate is itself proportional to
    depth-squared, just with a larger k, exactly like this test now models.
    """
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    n = 15
    k_bg, k_fg = 0.01, 0.03
    bg_trajectories = [_ego_motion_trajectory(1.0, k_bg, n) for _ in range(3)]
    fg_trajectory = _ego_motion_trajectory(1.0, k_fg, n)

    for t in range(n):
        depths = [traj[t] for traj in bg_trajectories] + [fg_trajectory[t]]
        store = engine.update(_multi_objects(100, depths), _make_multi_depth_map(depths))

    for i in range(3):
        assert store[100 + i].risk_level == RiskLevel.STATIC
    assert store[103].risk_level == RiskLevel.APPROACHING


def test_ego_motion_compensation_distinguishes_receding_object_from_static_background():
    """Mirror of the approaching test: a negative k means the object's own
    motion away from the camera outpaces the camera's closing speed, so its
    disparity genuinely decreases over time rather than just rising slower."""
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    n = 15
    k_bg, k_fg = 0.01, -0.04
    bg_trajectories = [_ego_motion_trajectory(1.0, k_bg, n) for _ in range(3)]
    fg_trajectory = _ego_motion_trajectory(1.0, k_fg, n)

    for t in range(n):
        depths = [traj[t] for traj in bg_trajectories] + [fg_trajectory[t]]
        store = engine.update(_multi_objects(200, depths), _make_multi_depth_map(depths))

    for i in range(3):
        assert store[200 + i].risk_level == RiskLevel.STATIC
    assert store[203].risk_level == RiskLevel.RECEDING


def test_single_track_always_classifies_static_regardless_of_absolute_motion():
    """
    Documented limitation (CLAUDE.md Critical Design Decision #6): with fewer
    than 2 tracks, the fitted ego-motion scale k collapses to exactly that
    single track's own depth_velocity / depth_smoothed**2 ratio, making its
    expected_velocity equal to its own depth_velocity and relative_velocity
    always exactly 0 — a lone track can never be classified APPROACHING or
    RECEDING, however fast its raw depth_velocity actually is.
    """
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    for i in range(15):
        store = engine.update([_car(1)], _make_depth_map(1.0 + 0.05 * i))
    assert store[1].depth_velocity > 0.02  # genuinely moving in absolute terms
    assert store[1].relative_velocity == 0.0
    assert store[1].risk_level == RiskLevel.STATIC


def test_constant_disparity_classified_static():
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    for _ in range(10):
        store = engine.update([_car(3)], _make_depth_map(1.0))
    assert store[3].risk_level == RiskLevel.STATIC


def test_track_survives_short_occlusion_and_recovers():
    engine = FusionEngine(max_frames_missing=3, kalman_process_noise=0.05, kalman_measurement_noise=0.05)
    engine.update([_car(10)], _make_depth_map(1.0))
    engine.update([_car(10)], _make_depth_map(1.0))

    engine.update([], _make_depth_map(1.0))
    store = engine.update([], _make_depth_map(1.0))
    assert 10 in store

    store = engine.update([_car(10)], _make_depth_map(1.0))
    assert store[10].frames_since_update == 0


def test_track_culled_after_exceeding_max_frames_missing():
    engine = FusionEngine(max_frames_missing=3, kalman_process_noise=0.05, kalman_measurement_noise=0.05)
    engine.update([_car(10)], _make_depth_map(1.0))
    for _ in range(5):
        store = engine.update([], _make_depth_map(1.0))
    assert 10 not in store


def test_matched_track_with_failed_depth_extraction_skips_kalman_update_but_resets_missing_counter():
    engine = FusionEngine(max_frames_missing=3, kalman_process_noise=0.05, kalman_measurement_noise=0.05)
    engine.update([_car(20)], _make_depth_map(1.0))

    off_frame_box = (500.0, 500.0, 600.0, 600.0)
    store = engine.update([_car(20, box=off_frame_box)], _make_depth_map(1.0))

    assert store[20].frames_since_update == 0  # tracker matched it, even though depth read failed
    assert store[20].depth_raw == 1.0  # unchanged — kalman.update() was never called
