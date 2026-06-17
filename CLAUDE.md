# CLAUDE.md — Spatial Scene Monitor

This file is the authoritative reference for the architecture, design decisions,
data structures, and constraints of this project. Update it whenever a decision
changes. Every module should be understandable by reading this file first.

---

## Project Overview

**Spatial Scene Monitor** is a real-time monocular perception system for road
scenes. Given a video sequence (KITTI or webcam), it produces a per-frame spatial
state for every tracked object: estimated relative distance from camera, pseudo-3D
trajectory over the last N frames, and a risk score (APPROACHING / RECEDING /
STATIC).

**Target use case:** offline analysis of pre-recorded driving sequences (KITTI),
with the architecture designed to extend to live webcam input without code changes.

**Primary output:** annotated video frames + per-frame JSON logs of track states.

---

## Stack

| Component         | Choice                    | Notes                                      |
|-------------------|---------------------------|--------------------------------------------|
| Detection         | YOLOv8n / YOLOv8s         | COCO weights; ONNX INT8 available if needed |
| Tracking          | ByteTrack                 | Existing modular codebase reused           |
| Depth estimation  | Depth Anything v2 (ViT-S) | Relative disparity, NOT metric depth       |
| Visualisation     | OpenCV                    | Frame annotation + video write             |
| Input             | KITTI tracking sequences  | KITTILoader mirrors VideoCapture interface |
| Language          | Python 3.10+              |                                            |
| Deep learning     | PyTorch 2.x               | MPS on MacBook M4 Pro, CUDA on UCL cluster |

---

## Architecture

### Pipeline (Sequential — v1)

```
[Video Frame]
      │
      ├────────────────────────────────┐
      │                                │
      ▼                                ▼
[Detector]                    [DepthEstimator]
 YOLOv8n/s                    Depth Anything v2
 → raw_detections              → depth_map [H×W]
      │                                │
      ▼                                │
[Tracker]                             │
 ByteTrack                            │
 → tracked_boxes                      │
   (with track_id)                    │
      │                                │
      └──────────────┬─────────────────┘
                     ▼
              [FusionEngine]
         For each (track_id, box):
           1. Extract depth from box region
           2. Normalise against scene reference
           3. Update per-track Kalman filter
           4. Compute depth velocity
           5. Compute risk score
                     │
                     ▼
            [TrackStateStore]
              dict[int → TrackState]
                     │
                     ▼
              [Visualiser]            [JSONLogger]
          Annotated frame out      Per-frame JSON log


```

**Why sequential (not parallel) in v1?**
Parallel detection + depth would reduce latency but requires threading or async
coordination, making the first version harder to debug. The depth map must exist
before fusion, and tracked boxes must exist before depth extraction. Sequential
keeps the data flow obvious. Async is a documented upgrade path (see below).

---

## Module Responsibilities

### `modules/detection/detector.py`
- Wraps YOLOv8 inference via `ultralytics`
- Filters to road-scene COCO classes only (see class filter below)
- Returns `List[Detection]` per frame
- Handles device routing (MPS / CUDA / CPU)

### `modules/tracking/tracker.py`
- Thin wrapper around ByteTrack
- Consumes `List[Detection]`, returns `List[TrackedObject]`
- Maintains no state of its own beyond what ByteTrack holds internally
- Class-aware: separate track pools per class (prevents cross-class ID collisions)

### `modules/depth/depth_estimator.py`
- Wraps Depth Anything v2 ViT-S inference
- Returns raw depth map `np.ndarray [H×W]` in relative disparity units
- Does NOT normalise — normalisation happens in fusion (separation of concerns)
- Handles device routing

### `modules/depth/depth_utils.py`
- `extract_box_depth(depth_map, box)` — center-crop median extraction
- `normalise_depth_map(depth_map)` — per-frame normalisation against scene median
- Stateless utility functions, no class needed

### `modules/fusion/fusion_engine.py`
- Core of the system
- Owns the `TrackStateStore` (dict of `track_id → TrackState`)
- Per frame: receives tracked boxes + depth map, updates all active track states
- Manages track birth (new ID seen) and track death (ID not seen for N frames)
- Returns updated `TrackStateStore`

### `modules/fusion/track_kalman.py`
- 1D Kalman filter for depth, per track
- State vector: `[depth, depth_velocity]` (2D)
- Measurement: scalar depth reading from `extract_box_depth`
- Provides: smoothed depth + depth velocity at each update
- Lives here, not in the tracking module, because it operates on depth not 2D position

### `modules/visualisation/visualiser.py`
- Renders annotated frame given a raw frame + `TrackStateStore`
- Draws: bounding box, track ID, class label, smoothed depth, risk score badge
- Optionally draws pseudo-3D trajectory trail (last N box centroids)
- Returns annotated `np.ndarray` — does NOT display or write, caller handles that

### `modules/utils/kitti_loader.py`
- Loads KITTI tracking sequences as a frame iterator
- Interface matches `cv2.VideoCapture`: supports `read()` returning `(ret, frame)`
- Handles KITTI's directory structure (image_02/data/*.png)
- Also returns frame index and optional ground-truth annotations (for future eval)

### `modules/utils/video_writer.py`
- Thin wrapper around `cv2.VideoWriter`
- Handles codec selection for macOS (mp4v) vs Linux (avc1/XVID)

### `configs/default.yaml`
- Single source of truth for all tunable parameters
- See Parameters section below

### `main.py`
- Entry point: loads config, instantiates all modules, runs the frame loop
- No business logic here — just wires modules together

---

## Key Data Structures

### `Detection`
```python
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
```
Output of the detector, input to ByteTrack.

### `TrackedObject`
```python
@dataclass
class TrackedObject:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
```
Output of ByteTrack. Same as `Detection` plus `track_id`. The track_id is the
persistent integer assigned by ByteTrack across frames.

### `TrackState`
```python
@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str

    # Depth (relative disparity units, normalised per-frame)
    depth_raw: float               # this frame's raw reading
    depth_smoothed: float          # Kalman-filtered output
    depth_velocity: float          # rate of change (negative = approaching)

    # Trajectory (last N frames of 2D box centroid + depth)
    trajectory_2d: deque           # deque of (cx, cy) tuples
    trajectory_depth: deque        # deque of smoothed depth values

    # Risk
    risk_score: RiskScore          # enum: APPROACHING / RECEDING / STATIC

    # Bookkeeping
    age: int                       # frames since first seen
    frames_since_update: int       # frames since last matched detection
    kalman: DepthKalmanFilter      # per-track Kalman instance
```
This is the central object in the system. `FusionEngine` owns a dict of these,
keyed by `track_id`. All downstream consumers (visualiser, logger) read from this.

### `RiskScore` (enum)
```python
class RiskScore(Enum):
    APPROACHING = "APPROACHING"   # depth_velocity < -threshold
    RECEDING    = "RECEDING"      # depth_velocity > +threshold
    STATIC      = "STATIC"        # |depth_velocity| <= threshold
```

---

## Critical Design Decisions

### 1. Depth Anything v2 outputs relative disparity, not metric depth

**What this means:** The values in the depth map are proportional to 1/distance
but with an arbitrary scale that varies between frames. A value of 0.8 in frame N
and 0.7 in frame N+1 for the same static object does not mean the object moved.

**Consequence:** You cannot compare raw depth values across frames directly.
You must normalise each frame before using depth for cross-frame comparisons.

**Our normalisation approach:** Divide all values in the depth map by the median
depth value of the frame. This anchors each frame's depth distribution to a
consistent reference (the "average scene depth" = 1.0) and dramatically reduces
low-frequency drift while preserving relative depth ordering within a frame.

### 2. Per-track Kalman filter on depth, not EMA

**Why Kalman over EMA:**
- EMA requires choosing alpha blindly; Kalman adapts based on measurement noise
- Kalman's state vector includes depth velocity as a first-class quantity — we need
  this for the risk score anyway, and EMA would require a finite difference on the
  smoothed output (introducing lag and noise)
- Kalman handles frames where a track exists but depth extraction fails (e.g. the
  box is at the frame edge) — we simply don't call update, and the filter propagates
  its prediction forward

**Kalman state:** `[depth, depth_velocity]`
**Measurement:** scalar depth reading (1D observation)
**Process noise Q:** tunable — high Q = trust measurements more, lower smoothing
**Measurement noise R:** tunable — high R = distrust measurements, more smoothing

### 3. Depth extraction via center-crop median

**Why center-crop:** Bounding boxes always include background pixels, especially
at the edges of the box. The central 50% of the box (area = 25% of full box) is
far more likely to contain only the object of interest.

**Why median over mean:** Depth maps have outlier spikes (reflective surfaces,
thin occluders). Median is robust to these; mean is not.

**Upgrade path:** Switch to YOLOv8-seg instance masks for precise per-pixel
extraction. Not done in v1 to keep the detection module simple.

### 4. Class-aware ByteTrack

**Why:** Without class-awareness, a car and a pedestrian with overlapping boxes
in consecutive frames could get the same track ID if one disappears and the other
appears. For a road scene risk monitor this would be dangerous.

**Implementation:** We run ByteTrack with detections tagged by class. Internally
this means separate track pools per class ID. The ByteTrack codebase supports this
via class filtering before matching.

### 5. Sequential pipeline in v1

**Why not parallel:** Detection and depth can theoretically run in parallel (both
take the raw frame as input). But threading introduces synchronisation complexity,
and debugging a concurrent bug in a new codebase is painful. We build sequential
first, profile it, and add async only if throughput is genuinely insufficient.

**Upgrade path:** Wrap `Detector` and `DepthEstimator` in `concurrent.futures`
`ThreadPoolExecutor` with a shared frame queue. Both produce results that the
fusion engine consumes once both are ready.

---

## Class Filter (Road Scene)

Only these COCO classes are passed to ByteTrack. Everything else is discarded
after detection.

| Class name  | COCO ID |
|-------------|---------|
| person      | 0       |
| bicycle     | 1       |
| car         | 2       |
| motorcycle  | 3       |
| bus         | 5       |
| truck       | 7       |

---

## Risk Score Rules

Applied in `FusionEngine._compute_risk()` after Kalman update:

```
depth_velocity < -RISK_APPROACH_THRESH  →  APPROACHING
depth_velocity >  RISK_APPROACH_THRESH  →  RECEDING
else                                    →  STATIC
```

`RISK_APPROACH_THRESH` is set in `configs/default.yaml`. Default: 0.02
(normalised disparity units per frame).

Secondary signal (not yet implemented, v2 upgrade): box area growth rate from
`trajectory_2d`. An object whose 2D box is growing AND has negative depth velocity
gets a higher confidence APPROACHING classification.

---

## Parameters (`configs/default.yaml`)

```yaml
detection:
  model: yolov8n.pt
  input_size: 640
  confidence_threshold: 0.3
  nms_threshold: 0.45
  class_filter: [0, 1, 2, 3, 5, 7]

tracking:
  # ByteTrack params — see ByteTrack paper for meaning
  track_thresh: 0.5
  track_buffer: 30       # frames to keep lost track alive
  match_thresh: 0.8
  frame_rate: 10         # KITTI sequences are ~10 FPS

depth:
  model_size: vits        # vits | vitb | vitl
  input_size: 518         # Depth Anything v2 native input size
  normalise: true

fusion:
  trajectory_length: 30  # frames to keep in trajectory deques
  kalman_process_noise: 0.01
  kalman_measurement_noise: 0.1
  max_frames_missing: 10  # drop TrackState after this many missed frames

risk:
  approach_threshold: 0.02

output:
  log_json: true
  log_dir: outputs/json
  frame_dir: outputs/frames
  save_video: true
  video_path: outputs/annotated.mp4
  display: false          # set true for live window (slower)
```

---

## Compute Split: MacBook M4 Pro vs UCL GPU Cluster

### Run on MacBook (MPS)
- All development and iteration
- Single-frame debugging and visualisation
- Short KITTI sequences (< 200 frames) for integration testing
- Kalman filter tuning (CPU-bound anyway)
- JSON log analysis and plotting

### Run on UCL Cluster (CUDA)
- Full KITTI sequence processing (sequences 0-20, some > 1000 frames)
- Depth Anything v2 ViT-B or ViT-L if we want to test larger models
- Any throughput benchmarking
- Generating final output videos for the portfolio/demo

### Cluster session planning
UCL cluster bookings are limited to 3 working days. Plan each booking around a
specific deliverable (not open-ended experimentation):

- **Session 1:** Run full pipeline on KITTI sequences 0, 1, 5. Generate JSON logs
  and annotated video. Validate that depth smoothing is working.
- **Session 2:** Parameter sweep on `kalman_process_noise`, `kalman_measurement_noise`,
  and `approach_threshold`. Log outputs for offline analysis.
- **Session 3:** (if needed) Run ViT-B variant, compare depth quality vs ViT-S.

---

## Gotchas and Constraints

### Depth Anything v2
- Input must be RGB, not BGR. OpenCV reads BGR — always convert before passing to
  the model: `rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`
- The model outputs a single-channel float32 map. Higher value = closer to camera
  (disparity convention, not depth convention). Some people get this backwards.
- At 518×518 input on MPS (M4 Pro), expect ~80-120ms per frame. This makes
  real-time (>10 FPS) tight in sequential mode on laptop. On KITTI (10 FPS source),
  this is borderline acceptable for development.
- Depth Anything v2 ViT-S weights must be downloaded separately from HuggingFace.
  The `depth_anything_v2` package from the official repo is the cleanest way to
  load them. Do not use the `transformers` pipeline for this — it adds overhead
  and doesn't give raw disparity maps cleanly.
- Run `scripts/setup_depth_anything.sh` once per machine (MacBook, UCL cluster)
  before using `depth_estimator.py`. It clones the official repo's
  `depth_anything_v2` package into `third_party/Depth-Anything-V2/` (stripped to
  just the importable model code — no demo app, no gradio UI, no metric-depth
  subproject) and downloads `depth_anything_v2_vits.pth` (~95MB) from
  `depth-anything/Depth-Anything-V2-Small` into `checkpoints/`. Both directories
  are gitignored — they're machine-local artifacts, not source. The script is
  idempotent: re-running it skips steps that already succeeded.
- `model.infer_image(bgr_frame, input_size=518)` handles the BGR→RGB conversion,
  resize to square input, and resize back to original resolution internally —
  `depth_estimator.py` does not need to do this manually.

### ByteTrack
- ByteTrack's `track_buffer` param is in frames, not seconds. KITTI is ~10 FPS,
  so `track_buffer=30` = 3 seconds of keeping a lost track alive. Adjust for
  webcam if FPS differs.
- ByteTrack returns tracks in an arbitrary order each frame. Do not assume the
  order is consistent across frames.
- Tracks with `track_id == -1` are tentative (not yet confirmed). We skip these
  in fusion until they are confirmed.

### KITTI
- KITTI image sequences are stored as PNG files in `image_02/data/`. Left camera
  only. Resolution is 1242×375.
- KITTI is not square — Depth Anything v2 expects square input. We resize to
  518×518 for depth, but track and detect on the native resolution (or 640px
  short-side resize). The depth map must be resized back to original resolution
  before fusion.
- KITTI sequences have variable length (some < 50 frames, some > 1500). Test on
  short sequences first.

### MPS (Apple Silicon)
- Some PyTorch ops fall back to CPU on MPS silently. If you see unexpected slowness,
  add `PYTORCH_ENABLE_MPS_FALLBACK=1` to your env and check the logs for fallback
  warnings.
- `torch.compile()` does not work reliably on MPS as of PyTorch 2.x. Do not use it
  for laptop development.
- Mixed precision (`torch.autocast`) on MPS uses bfloat16, not float16. This is
  fine for inference but worth knowing.

### General
- All depth values in `TrackState` are in normalised relative disparity units
  (not metres, not pixels). Never label them as "distance in metres" in the
  visualiser or logs without a caveat.
- The Kalman filter must be reset if a track ID is reassigned by ByteTrack. This
  can happen if a track is lost and a new object gets the same ID. Track age
  (`TrackState.age`) helps detect this: if age resets to 0 for an existing ID,
  reinitialise the Kalman filter.
- JSON logs can get large on long sequences. Implement a rolling log or per-sequence
  file, not one global append.

---

## File Structure

```
spatial-scene-monitor/
├── CLAUDE.md                        ← this file
├── README.md
├── main.py                          ← entry point
├── requirements.txt
├── configs/
│   └── default.yaml                 ← all tunable parameters
├── modules/
│   ├── detection/
│   │   ├── __init__.py
│   │   └── detector.py              ← YOLOv8 wrapper + class filter
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── tracker.py               ← ByteTrack wrapper
│   ├── depth/
│   │   ├── __init__.py
│   │   ├── depth_estimator.py       ← Depth Anything v2 wrapper
│   │   └── depth_utils.py           ← extract_box_depth, normalise_depth_map
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── fusion_engine.py         ← TrackStateStore + per-frame update
│   │   ├── track_kalman.py          ← 1D Kalman on depth
│   │   └── data_types.py            ← Detection, TrackedObject, TrackState, RiskScore
│   ├── visualisation/
│   │   ├── __init__.py
│   │   └── visualiser.py            ← frame annotation
│   └── utils/
│       ├── __init__.py
│       ├── kitti_loader.py          ← KITTI sequence iterator
│       ├── video_writer.py          ← cv2.VideoWriter wrapper
│       └── logger.py                ← JSON log writer
├── data/
│   ├── kitti/                       ← KITTI sequences go here (not committed)
│   └── samples/                     ← short test clips
├── outputs/
│   ├── logs/
│   ├── frames/
│   └── json/
├── scripts/
│   └── download_kitti.sh            ← helper to fetch a sequence
└── tests/
    ├── test_detector.py
    ├── test_depth_utils.py
    ├── test_kalman.py
    └── test_fusion.py
```

---

## Build Order (Module by Module)

We build in dependency order — nothing depends on a module that isn't built yet.

1. `modules/fusion/data_types.py` — shared data structures, no dependencies
2. `modules/detection/detector.py` — only depends on ultralytics + data_types
3. `modules/utils/kitti_loader.py` — only depends on OpenCV
4. `modules/depth/depth_estimator.py` — only depends on Depth Anything v2
5. `modules/depth/depth_utils.py` — only depends on numpy
6. `modules/tracking/tracker.py` — depends on ByteTrack + data_types
7. `modules/fusion/track_kalman.py` — pure numpy, no other module dependencies
8. `modules/fusion/fusion_engine.py` — depends on all of the above
9. `modules/visualisation/visualiser.py` — depends on data_types + OpenCV
10. `modules/utils/logger.py` — depends on data_types
11. `modules/utils/video_writer.py` — only depends on OpenCV
12. `main.py` — wires everything together
