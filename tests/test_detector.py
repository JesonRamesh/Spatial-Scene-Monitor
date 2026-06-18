"""
Tests for modules/detection/detector.py.

_parse() is tested against mock YOLO results, not a real loaded model —
keeps this suite fast and independent of model weights being downloaded.
"""

import pytest
import torch

from modules.detection.detector import Detector, _select_device


def test_select_device_returns_valid_choice():
    assert _select_device() in {"mps", "cuda", "cpu"}


def test_parse_converts_yolo_boxes_to_detections():
    class MockBoxes:
        xyxy = torch.tensor([[10.0, 20.0, 110.0, 120.0], [50.0, 60.0, 200.0, 300.0]])
        conf = torch.tensor([0.91, 0.75])
        cls = torch.tensor([2.0, 0.0])

        def __len__(self):
            return 2

    class MockResult:
        boxes = MockBoxes()

    detector = object.__new__(Detector)  # skip __init__, no model download needed
    detections = detector._parse(MockResult())

    assert len(detections) == 2
    assert detections[0].class_name == "car"
    assert detections[0].x1 == 10.0 and detections[0].x2 == 110.0
    assert detections[0].confidence == pytest.approx(0.91)  # source tensor is float32, not exact
    assert detections[1].class_name == "person"


def test_parse_empty_boxes_returns_empty_list():
    class MockBoxes:
        def __len__(self):
            return 0

    class MockResult:
        boxes = MockBoxes()

    detector = object.__new__(Detector)
    assert detector._parse(MockResult()) == []
