"""
KITTI sequence loader for the Spatial Scene Monitor.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class KITTILoader:
    """
    Iterates a KITTI image sequence with a cv2.VideoCapture-compatible interface.

    Expects `sequence_dir` to contain `image_02/data/*.png` (KITTI's raw-data
    layout, left camera only). Optionally loads KITTI tracking labels for
    future evaluation — not consumed by the live pipeline today.
    """

    def __init__(self, sequence_dir: str, label_file: str | None = None) -> None:
        self.sequence_dir = Path(sequence_dir)
        self.image_dir = self.sequence_dir / "image_02" / "data"

        if not self.image_dir.is_dir():
            raise FileNotFoundError(
                f"Expected KITTI images at {self.image_dir}, found nothing."
            )

        # KITTI filenames are zero-padded (000000.png, 000001.png, ...), so
        # lexicographic sort is already frame order.
        self.frame_paths: list[Path] = sorted(self.image_dir.glob("*.png"))
        self.frame_count = len(self.frame_paths)
        self._next_index = 0

        self.annotations: dict[int, list[dict]] | None = (
            self._load_labels(label_file) if label_file else None
        )

    def isOpened(self) -> bool:
        return self.frame_count > 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        """
        Mirrors cv2.VideoCapture.read(): returns (ret, frame).

        frame is BGR uint8, matching what cv2.VideoCapture would hand back,
        so downstream modules (Detector, DepthEstimator) need no special-casing.
        """
        if self._next_index >= self.frame_count:
            return False, None

        path = self.frame_paths[self._next_index]
        frame = cv2.imread(str(path))
        self._next_index += 1

        if frame is None:
            return False, None
        return True, frame

    @property
    def current_frame_index(self) -> int:
        """Index of the frame most recently returned by read()."""
        return self._next_index - 1

    def get_annotations(self, frame_idx: int) -> list[dict]:
        """Ground-truth boxes for frame_idx, or [] if no label file was loaded."""
        if self.annotations is None:
            return []
        return self.annotations.get(frame_idx, [])

    def release(self) -> None:
        """Mirrors cv2.VideoCapture.release(). No real resource to free here."""
        self._next_index = self.frame_count

    def _load_labels(self, label_file: str) -> dict[int, list[dict]]:
        """
        Parses KITTI tracking label format:
        frame track_id type truncated occluded alpha x1 y1 x2 y2 ...(3D fields, unused)
        """
        label_path = Path(label_file)
        if not label_path.is_file():
            return {}

        annotations: dict[int, list[dict]] = {}
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[2] == "DontCare":
                    continue
                frame_idx = int(parts[0])
                annotations.setdefault(frame_idx, []).append({
                    "track_id": int(parts[1]),
                    "type": parts[2],
                    "x1": float(parts[6]),
                    "y1": float(parts[7]),
                    "x2": float(parts[8]),
                    "y2": float(parts[9]),
                })
        return annotations
