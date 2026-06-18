"""
Tests for modules/depth/depth_utils.py.
"""

import numpy as np

from modules.depth.depth_utils import extract_box_depth, normalise_depth_map


def _make_depth_map(object_value: float, h: int = 100, w: int = 200) -> np.ndarray:
    m = np.full((h, w), 1.0, dtype=np.float32)
    m[40:60, 80:120] = object_value  # 'object' patch, fully containing any box's 50% crop below
    return m


def test_extract_box_depth_ignores_background_via_center_crop():
    depth_map = _make_depth_map(object_value=5.0)
    # Box with background margin around the object patch — naive mean/full-box
    # median would be dragged toward the background value of 1.0.
    value = extract_box_depth(depth_map, (70, 30, 130, 70))
    assert value == 5.0


def test_extract_box_depth_returns_none_when_box_fully_outside_frame():
    depth_map = _make_depth_map(object_value=5.0)
    assert extract_box_depth(depth_map, (500, 500, 600, 600)) is None


def test_extract_box_depth_handles_box_straddling_edge():
    depth_map = _make_depth_map(object_value=5.0)
    value = extract_box_depth(depth_map, (180, 30, 250, 70))
    assert value is not None


def test_extract_box_depth_returns_none_for_zero_width_box():
    depth_map = _make_depth_map(object_value=5.0)
    assert extract_box_depth(depth_map, (50, 50, 50, 80)) is None


def test_extract_box_depth_handles_tiny_box():
    depth_map = _make_depth_map(object_value=5.0)
    value = extract_box_depth(depth_map, (10, 10, 11, 11))
    assert value is not None


def test_normalise_depth_map_anchors_median_to_one():
    depth_map = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    normalised = normalise_depth_map(depth_map)
    assert abs(float(np.median(normalised)) - 1.0) < 1e-6


def test_normalise_depth_map_handles_degenerate_all_zero_map():
    zeros = np.zeros((10, 10), dtype=np.float32)
    result = normalise_depth_map(zeros)
    assert np.array_equal(result, zeros)
