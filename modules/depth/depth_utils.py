"""
Depth map utility functions for the Spatial Scene Monitor.

Pure numpy — no model loading, no project-internal imports. Operates on
raw depth maps and plain box tuples, not on Detection/TrackedObject, so
it stays usable regardless of which detector or tracker produced the box.
"""

from __future__ import annotations

import numpy as np

# Guards normalise_depth_map against division by zero on a degenerate
# (e.g. all-black) frame. Never triggered on real depth output, which is
# strictly positive almost everywhere.
_EPSILON = 1e-6

# Center-crop fraction: keep the central CROP_FRACTION of the box's width
# and height, i.e. CROP_FRACTION**2 of its area. See CLAUDE.md "Critical
# Design Decisions #3" — background pixels concentrate near box edges, so
# shrinking toward the center reduces background contamination.
CROP_FRACTION = 0.5


def extract_box_depth(
    depth_map: np.ndarray,
    box: tuple[float, float, float, float],
) -> float | None:
    """
    Median depth of the central 50%-width/height region of `box`.

    `box` is (x1, y1, x2, y2) in the same pixel coordinates as depth_map.

    Returns None if the box doesn't overlap the frame at all (e.g. a stale
    track prediction that's drifted off-screen). Callers (FusionEngine)
    should treat None as "skip this frame's measurement" and let the
    Kalman filter predict-only, per CLAUDE.md's Kalman-over-EMA rationale.
    """
    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = box

    # Clamp to frame bounds first — boxes can extend past frame edges.
    x1c, x2c = max(0.0, x1), min(float(w), x2)
    y1c, y2c = max(0.0, y1), min(float(h), y2)
    if x2c <= x1c or y2c <= y1c:
        return None

    # Shrink to the central CROP_FRACTION of width/height, same center.
    cx, cy = (x1c + x2c) / 2.0, (y1c + y2c) / 2.0
    half_w = (x2c - x1c) * CROP_FRACTION / 2.0
    half_h = (y2c - y1c) * CROP_FRACTION / 2.0

    ix1 = int(round(cx - half_w))
    iy1 = int(round(cy - half_h))
    ix2 = int(round(cx + half_w))
    iy2 = int(round(cy + half_h))

    # Clamp into valid index range, guaranteeing at least a 1x1 region —
    # rounding can otherwise collapse a thin box to zero width or height.
    ix1 = max(0, min(ix1, w - 1))
    iy1 = max(0, min(iy1, h - 1))
    ix2 = max(ix1 + 1, min(ix2, w))
    iy2 = max(iy1 + 1, min(iy2, h))

    region = depth_map[iy1:iy2, ix1:ix2]
    return float(np.median(region))


def normalise_depth_map(depth_map: np.ndarray) -> np.ndarray:
    """
    Anchor a frame's depth values to its own median: median -> 1.0.

    Depth Anything v2's output scale drifts frame to frame (CLAUDE.md
    "Critical Design Decisions #1") — this anchoring is what makes
    depth_velocity comparable across frames instead of comparing
    two arbitrary, unrelated scales.
    """
    median = float(np.median(depth_map))
    if median < _EPSILON:
        return depth_map.copy()
    return depth_map / median
