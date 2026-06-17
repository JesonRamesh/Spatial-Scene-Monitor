"""
Per-sequence JSON log writer for the Spatial Scene Monitor.
"""

from __future__ import annotations

import json
from pathlib import Path

from modules.fusion.data_types import TrackState


class JSONLogger:
    """
    Writes one JSON Lines (.jsonl) file per sequence — one line per frame,
    each line a self-contained JSON object listing every active track's
    state that frame.

    JSON Lines, not a single growing JSON array, per CLAUDE.md's gotcha that
    long KITTI sequences (1500+ frames) make JSON logs large: appending to a
    JSON array file means rewriting the whole file (or careful bracket/comma
    bookkeeping) every frame, and risks losing the whole file if the process
    dies before the closing bracket is written. JSON Lines lets each frame be
    written and flushed independently — no in-memory accumulation across
    frames, and a crash mid-sequence only loses the frame in progress.
    """

    def __init__(self, log_dir: str | Path, sequence_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / f"{sequence_name}.jsonl"
        self._file = open(self.log_path, "w")

    def log_frame(self, frame_index: int, track_states: dict[int, TrackState]) -> None:
        """Write one frame's record as a single JSON line, flushed immediately."""
        record = {
            "frame_index": frame_index,
            "tracks": [state.to_dict() for state in track_states.values()],
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "JSONLogger":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
