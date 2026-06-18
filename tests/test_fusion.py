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


def test_track_birth_creates_track_state():
    engine = FusionEngine()
    store = engine.update([_car(1)], _make_depth_map(1.0))
    assert 1 in store
    assert store[1].box == _BOX


def test_increasing_disparity_classified_approaching():
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    for i in range(15):
        store = engine.update([_car(1)], _make_depth_map(1.0 + 0.05 * i))
    assert store[1].risk_level == RiskLevel.APPROACHING


def test_decreasing_disparity_classified_receding():
    engine = FusionEngine(kalman_process_noise=0.05, kalman_measurement_noise=0.05, approach_threshold=0.02)
    for i in range(15):
        store = engine.update([_car(2)], _make_depth_map(max(0.1, 2.0 - 0.05 * i)))
    assert store[2].risk_level == RiskLevel.RECEDING


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
