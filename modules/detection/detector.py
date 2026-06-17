"""
YOLOv8 detector for the Spatial Scene Monitor.
"""

from __future__ import annotations

import numpy as np
import torch
from ultralytics import YOLO

from modules.fusion.data_types import COCO_CLASS_NAMES, Detection, ROAD_CLASS_IDS


def _select_device() -> str:
    """MPS → CUDA → CPU, in order of preference for this project."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Detector:
    """
    Thin wrapper around YOLOv8 that returns List[Detection] per frame.

    Caller passes BGR frames as loaded by OpenCV. The class filter is applied
    inside YOLO's NMS pass, so non-road classes never reach the tracker.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.3,
        nms_threshold: float = 0.45,
        class_filter: list[int] | None = None,
        input_size: int = 640,
        device: str | None = None,
    ) -> None:
        self.conf = confidence_threshold
        self.iou = nms_threshold
        self.class_filter = class_filter if class_filter is not None else list(ROAD_CLASS_IDS)
        self.input_size = input_size
        self.device = device or _select_device()

        self.model = YOLO(model_path)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8 on one BGR frame. Returns detections at original resolution.

        Returns [] when no road-scene objects are found above threshold.
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.input_size,
            conf=self.conf,
            iou=self.iou,
            classes=self.class_filter,
            device=self.device,
            verbose=False,
        )
        return self._parse(results[0])

    def _parse(self, result) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy    = boxes.xyxy.cpu().numpy()
        confs   = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        return [
            Detection(
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                confidence=float(conf),
                class_id=int(cid),
                class_name=COCO_CLASS_NAMES.get(int(cid), "unknown"),
            )
            for (x1, y1, x2, y2), conf, cid in zip(xyxy, confs, cls_ids)
        ]
