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


def test_ego_motion_compensation_distinguishes_approaching_object_from_static_background():
    """
    3 'parked' background tracks share the camera's own approach rate; one
    foreground track closes distance faster than that. Only the foreground
    track should read APPROACHING — the background tracks' depth_velocity is
    entirely explained by ego-motion, so their relative_velocity is ~0.
    """
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    ego_rate, extra_rate = 0.05, 0.10
    depths = [1.0, 1.0, 1.0, 1.0]

    for _ in range(20):
        depths = [d + ego_rate for d in depths[:3]] + [depths[3] + ego_rate + extra_rate]
        objs = [TrackedObject(track_id=100 + i, x1=_box_for_slot(i)[0], y1=_box_for_slot(i)[1],
                               x2=_box_for_slot(i)[2], y2=_box_for_slot(i)[3],
                               confidence=0.9, class_id=2, class_name="car") for i in range(4)]
        store = engine.update(objs, _make_multi_depth_map(depths))

    for i in range(3):
        assert store[100 + i].risk_level == RiskLevel.STATIC
    assert store[103].risk_level == RiskLevel.APPROACHING


def test_ego_motion_compensation_distinguishes_receding_object_from_static_background():
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    ego_rate, extra_rate = 0.05, -0.10
    depths = [1.0, 1.0, 1.0, 1.0]

    for _ in range(20):
        depths = [d + ego_rate for d in depths[:3]] + [max(0.05, depths[3] + ego_rate + extra_rate)]
        objs = [TrackedObject(track_id=200 + i, x1=_box_for_slot(i)[0], y1=_box_for_slot(i)[1],
                               x2=_box_for_slot(i)[2], y2=_box_for_slot(i)[3],
                               confidence=0.9, class_id=2, class_name="car") for i in range(4)]
        store = engine.update(objs, _make_multi_depth_map(depths))

    for i in range(3):
        assert store[200 + i].risk_level == RiskLevel.STATIC
    assert store[203].risk_level == RiskLevel.RECEDING


def test_single_track_always_classifies_static_regardless_of_absolute_motion():
    """
    Documented limitation (CLAUDE.md Critical Design Decision #6): with fewer
    than 2 tracks, the median ego-motion baseline collapses to that single
    track's own velocity, so relative_velocity is always exactly 0 — a lone
    track can never be classified APPROACHING or RECEDING, however fast its
    raw depth_velocity actually is.
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
