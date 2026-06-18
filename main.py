"""
Entry point for the Spatial Scene Monitor.

Loads configs/default.yaml, instantiates every module, and runs the frame
loop. No business logic lives here — each frame is just: detect, track,
estimate depth, fuse, render, and send the results to whichever outputs
the config has enabled. The actual perception logic lives in the modules.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import yaml

from modules.depth.depth_estimator import DepthEstimator
from modules.detection.detector import Detector
from modules.fusion.fusion_engine import FusionEngine
from modules.tracking.tracker import Tracker
from modules.utils.kitti_loader import KITTILoader
from modules.utils.logger import JSONLogger
from modules.utils.video_writer import VideoWriter
from modules.visualisation.visualiser import Visualiser


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_source(source_arg: str):
    """
    Resolves --source into anything offering the cv2.VideoCapture-style
    interface (isOpened / read / release) — a webcam index, a KITTI sequence
    directory, or a plain video file. This is the one place main.py decides
    *which* frame source to use; the frame loop below never knows or cares
    which kind it got, which is the whole point of KITTILoader mirroring
    cv2.VideoCapture's interface (CLAUDE.md's "extend to webcam without code
    changes" target).
    """
    if source_arg.isdigit():
        return cv2.VideoCapture(int(source_arg))

    source_path = Path(source_arg)
    if (source_path / "image_02" / "data").is_dir():
        return KITTILoader(source_arg)

    return cv2.VideoCapture(source_arg)


def infer_sequence_name(source_arg: str) -> str:
    if source_arg.isdigit():
        return f"webcam{source_arg}"
    return Path(source_arg).stem or Path(source_arg).name


def main() -> None:
    parser = argparse.ArgumentParser(description="Spatial Scene Monitor")
    parser.add_argument(
        "--source", required=True,
        help="KITTI sequence directory, video file path, or webcam index (e.g. 0)",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--sequence-name", default=None,
        help="Used for the JSON log filename; inferred from --source if omitted",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    sequence_name = args.sequence_name or infer_sequence_name(args.source)

    source = build_source(args.source)
    if not source.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    detector = Detector(
        model_path=config["detection"]["model"],
        confidence_threshold=config["detection"]["confidence_threshold"],
        nms_threshold=config["detection"]["nms_threshold"],
        class_filter=config["detection"]["class_filter"],
        input_size=config["detection"]["input_size"],
    )
    tracker = Tracker(
        track_thresh=config["tracking"]["track_thresh"],
        track_buffer=config["tracking"]["track_buffer"],
        match_thresh=config["tracking"]["match_thresh"],
        class_filter=config["detection"]["class_filter"],  # one shared class filter, not duplicated in config
    )
    depth_estimator = DepthEstimator(
        model_size=config["depth"]["model_size"],
        input_size=config["depth"]["input_size"],
        checkpoint_dir=config["depth"]["checkpoint_dir"],
    )
    fusion = FusionEngine(
        trajectory_length=config["fusion"]["trajectory_length"],
        kalman_process_noise=config["fusion"]["kalman_process_noise"],
        kalman_measurement_noise=config["fusion"]["kalman_measurement_noise"],
        max_frames_missing=config["fusion"]["max_frames_missing"],
        approach_threshold=config["risk"]["approach_threshold"],
    )
    visualiser = Visualiser()

    json_logger = (
        JSONLogger(log_dir=config["output"]["log_dir"], sequence_name=sequence_name)
        if config["output"]["log_json"] else None
    )
    video_writer = None  # lazily built from the first annotated frame, once we know its size

    display = config["output"]["display"]
    frame_index = 0

    try:
        while True:
            ret, frame = source.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracked_objects = tracker.update(detections)
            raw_depth_map = depth_estimator.estimate(frame)
            track_states = fusion.update(tracked_objects, raw_depth_map)
            annotated = visualiser.render(frame, track_states)

            if json_logger is not None:
                json_logger.log_frame(frame_index, track_states)

            if config["output"]["save_video"]:
                if video_writer is None:
                    video_writer = VideoWriter.from_frame(
                        config["output"]["video_path"],
                        annotated,
                        fps=config["tracking"]["frame_rate"],
                    )
                video_writer.write(annotated)

            if display:
                cv2.imshow("Spatial Scene Monitor", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        source.release()
        if json_logger is not None:
            json_logger.close()
        if video_writer is not None:
            video_writer.release()
        if display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
