#!/usr/bin/env python3
"""
Validates FusionEngine's fitted ego-motion scale (k) against KITTI's real
recorded vehicle speed (oxts field 'vf', forward velocity in m/s).

Not part of the production pipeline — a one-time sanity check that k
(modules/fusion/fusion_engine.py's depth-squared scale fit) actually tracks
genuine ego-motion rather than being a plausible-looking but unfounded
heuristic. Requires a KITTI sequence directory containing both
image_02/data/ and oxts/data/ (the raw-data download includes both; our
KITTILoader only reads the former, so this script reads oxts separately).

k is only expected to CORRELATE with real vf, not match it numerically —
k = V_ego / C, where C bundles the camera's focal length and this frame's
depth normalisation constant, neither of which is calibrated against metric
units anywhere in this project. A strong correlation is the right thing to
check; an exact numeric match is not a meaningful target.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from modules.depth.depth_estimator import DepthEstimator  # noqa: E402
from modules.detection.detector import Detector  # noqa: E402
from modules.fusion.fusion_engine import FusionEngine  # noqa: E402
from modules.tracking.tracker import Tracker  # noqa: E402
from modules.utils.kitti_loader import KITTILoader  # noqa: E402

# Kalman velocities start at 0 and only become meaningful after a couple of
# real measurement updates — the first few frames' k is trivially near 0
# regardless of real vehicle speed, and would only dilute the correlation.
_WARMUP_FRAMES = 5


def load_forward_velocities(oxts_dir: Path, n_frames: int) -> list[float]:
    """KITTI oxts field index 8 is 'vf' — forward velocity in m/s."""
    vf = []
    for i in range(n_frames):
        path = oxts_dir / f"{i:010d}.txt"
        with open(path) as f:
            fields = f.read().split()
        vf.append(float(fields[8]))
    return vf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="KITTI sequence directory (needs oxts/data/)")
    parser.add_argument("--plot", default=None, help="Optional path to save a comparison PNG")
    args = parser.parse_args()

    sequence_dir = Path(args.source)
    oxts_dir = sequence_dir / "oxts" / "data"
    if not oxts_dir.is_dir():
        raise FileNotFoundError(f"No oxts/data/ found under {sequence_dir}")

    loader = KITTILoader(args.source)
    detector = Detector()
    tracker = Tracker()
    depth_estimator = DepthEstimator(model_size="vits", checkpoint_dir=str(_REPO_ROOT / "checkpoints"))
    fusion = FusionEngine()

    fitted_k: list[float] = []
    while True:
        ret, frame = loader.read()
        if not ret:
            break
        detections = detector.detect(frame)
        tracked_objects = tracker.update(detections)
        depth_map = depth_estimator.estimate(frame)
        fusion.update(tracked_objects, depth_map)
        fitted_k.append(fusion.last_ego_motion_scale)

    vf = load_forward_velocities(oxts_dir, len(fitted_k))

    k_arr = np.array(fitted_k[_WARMUP_FRAMES:])
    vf_arr = np.array(vf[_WARMUP_FRAMES:])
    correlation = float(np.corrcoef(k_arr, vf_arr)[0, 1])

    print(f"Frames: {len(fitted_k)} (first {_WARMUP_FRAMES} excluded as Kalman warmup)")
    print(f"Real vf (m/s):  min={vf_arr.min():.2f}  max={vf_arr.max():.2f}  mean={vf_arr.mean():.2f}")
    print(f"Fitted k:       min={k_arr.min():.5f}  max={k_arr.max():.5f}  mean={k_arr.mean():.5f}")
    print(f"Pearson correlation (fitted k vs real vehicle speed): {correlation:.3f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        frames = range(_WARMUP_FRAMES, len(fitted_k))
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(frames, vf_arr, color="tab:blue", label="Real vehicle speed (vf, m/s)")
        ax1.set_xlabel("Frame")
        ax1.set_ylabel("vf (m/s)", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(frames, k_arr, color="tab:red", label="Fitted ego-motion scale (k)")
        ax2.set_ylabel("k", color="tab:red")
        fig.suptitle(f"Fitted k vs real vehicle speed (Pearson r = {correlation:.3f})")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"Saved comparison plot to {args.plot}")


if __name__ == "__main__":
    main()
