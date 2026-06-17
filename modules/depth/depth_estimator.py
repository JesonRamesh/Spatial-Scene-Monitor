"""
Depth Anything v2 wrapper for the Spatial Scene Monitor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# depth_anything_v2 is vendored via scripts/setup_depth_anything.sh, not pip-installed
# (see CLAUDE.md gotchas). Make it importable before reaching for it below.
_THIRD_PARTY_DIR = Path(__file__).resolve().parents[2] / "third_party" / "Depth-Anything-V2"
if str(_THIRD_PARTY_DIR) not in sys.path:
    sys.path.insert(0, str(_THIRD_PARTY_DIR))

from depth_anything_v2.dpt import DepthAnythingV2  # noqa: E402


_MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64,  "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def _select_device() -> str:
    """MPS → CUDA → CPU. Mirrors modules/detection/detector.py's selection order."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DepthEstimator:
    """
    Thin wrapper around Depth Anything v2.

    Returns raw relative-disparity maps — higher value = closer to camera.
    Deliberately does NOT normalise. Normalisation against the scene median
    happens in modules/depth/depth_utils.py at fusion time, not here, keeping
    "estimate depth" and "make depth comparable across frames" separate.
    """

    def __init__(
        self,
        model_size: str = "vits",
        input_size: int = 518,
        checkpoint_dir: str | Path = "checkpoints",
        device: str | None = None,
    ) -> None:
        if model_size not in _MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_size {model_size!r}, expected one of {list(_MODEL_CONFIGS)}"
            )

        self.model_size = model_size
        self.input_size = input_size
        self.device = device or _select_device()

        checkpoint_path = Path(checkpoint_dir) / f"depth_anything_v2_{model_size}.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Run scripts/setup_depth_anything.sh first."
            )

        self.model = DepthAnythingV2(**_MODEL_CONFIGS[model_size])
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model = self.model.to(self.device).eval()

    def estimate(self, frame: np.ndarray) -> np.ndarray:
        """
        Run depth estimation on one BGR frame (as loaded by OpenCV).

        Returns a float32 array [H, W] at the input frame's original
        resolution, in relative disparity units. Not normalised — callers
        must run modules/depth/depth_utils.normalise_depth_map() before
        comparing values across frames.
        """
        return self.model.infer_image(frame, self.input_size)
