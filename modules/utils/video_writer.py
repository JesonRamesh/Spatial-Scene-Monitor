"""
Video writer wrapper for the Spatial Scene Monitor.
"""

from __future__ import annotations

import platform
from pathlib import Path

import cv2
import numpy as np

_CODEC_CANDIDATES: dict[str, list[str]] = {
    "Darwin": ["mp4v"],
    "Linux": ["avc1", "XVID"],
}
_DEFAULT_CANDIDATES = ["mp4v"]


def _codec_candidates() -> list[str]:
    return _CODEC_CANDIDATES.get(platform.system(), _DEFAULT_CANDIDATES)


def _round_up_to_even(value: int) -> int:
    return value + (value % 2)


class VideoWriter:
    """
    Thin wrapper around cv2.VideoWriter with platform-aware codec selection.

    OpenCV's codec ("fourcc") support depends on how it was built and which
    system codec libraries are installed — the wrong choice doesn't raise an
    error, it silently produces a zero-byte or unplayable file. We try
    CLAUDE.md's platform-specific candidates in order (mp4v on macOS;
    avc1 then XVID on Linux) and only fail loudly if every candidate's
    VideoWriter genuinely refuses to open.

    Also pads frame_size up to even width/height. KITTI's native resolution
    is 1242x375 — 375 is odd — and codecs using YUV420 chroma subsampling
    (mp4v included) silently truncate odd dimensions by one row/column rather
    than erroring. Confirmed directly: writing a 375-tall frame and reading
    it back came back as 374 tall. Padding with a black border (removed on
    nothing — the original content is never cropped) avoids that silent loss.
    """

    def __init__(
        self,
        output_path: str | Path,
        frame_size: tuple[int, int],
        fps: float = 10.0,
        fourcc: str | None = None,
    ) -> None:
        """
        frame_size is (width, height) — the order cv2.VideoWriter expects,
        the OPPOSITE of a numpy frame's own .shape (height, width, channels).
        Prefer VideoWriter.from_frame() to avoid mixing this up.
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        width, height = frame_size
        padded_width = _round_up_to_even(width)
        padded_height = _round_up_to_even(height)
        self._pad_right = padded_width - width
        self._pad_bottom = padded_height - height

        candidates = [fourcc] if fourcc else _codec_candidates()
        self._writer = None
        self.fourcc: str | None = None

        for candidate in candidates:
            writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*candidate),
                fps,
                (padded_width, padded_height),
            )
            if writer.isOpened():
                self._writer = writer
                self.fourcc = candidate
                break
            writer.release()

        if self._writer is None:
            raise RuntimeError(
                f"Could not open a VideoWriter for {self.output_path} with any of "
                f"{candidates}. Check that OpenCV's build supports one of these codecs."
            )

    @classmethod
    def from_frame(
        cls,
        output_path: str | Path,
        sample_frame: np.ndarray,
        fps: float = 10.0,
        fourcc: str | None = None,
    ) -> "VideoWriter":
        """Derives frame_size correctly from a real frame's (H, W, C) shape."""
        h, w = sample_frame.shape[:2]
        return cls(output_path, frame_size=(w, h), fps=fps, fourcc=fourcc)

    def write(self, frame: np.ndarray) -> None:
        if self._pad_bottom or self._pad_right:
            frame = cv2.copyMakeBorder(
                frame, 0, self._pad_bottom, 0, self._pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0),
            )
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.release()
