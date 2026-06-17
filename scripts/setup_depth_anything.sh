#!/usr/bin/env bash
# Fetches the Depth Anything V2 model code and ViT-S checkpoint.
#
# Not committed to git (see .gitignore) — run this once per machine
# (MacBook, UCL cluster, etc.) before using modules/depth/depth_estimator.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="$REPO_ROOT/third_party/Depth-Anything-V2"
CHECKPOINT_DIR="$REPO_ROOT/checkpoints"

if [ -d "$THIRD_PARTY_DIR" ]; then
    echo "third_party/Depth-Anything-V2 already exists, skipping clone."
else
    echo "Cloning Depth Anything V2 model code..."
    git clone --depth 1 https://github.com/DepthAnything/Depth-Anything-V2 "$THIRD_PARTY_DIR"
    # We only need the importable depth_anything_v2 package, not the demo
    # app, gradio UI, or metric-depth subproject.
    rm -rf "$THIRD_PARTY_DIR/app.py" "$THIRD_PARTY_DIR/assets" \
           "$THIRD_PARTY_DIR/metric_depth" "$THIRD_PARTY_DIR/run.py" \
           "$THIRD_PARTY_DIR/run_video.py" "$THIRD_PARTY_DIR/DA-2K.md" \
           "$THIRD_PARTY_DIR/.git"
fi

mkdir -p "$CHECKPOINT_DIR"
CHECKPOINT_FILE="$CHECKPOINT_DIR/depth_anything_v2_vits.pth"

if [ -f "$CHECKPOINT_FILE" ]; then
    echo "ViT-S checkpoint already exists, skipping download."
else
    echo "Downloading ViT-S checkpoint from HuggingFace..."
    python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='depth-anything/Depth-Anything-V2-Small',
    filename='depth_anything_v2_vits.pth',
    local_dir='$CHECKPOINT_DIR',
)
"
fi

echo "Done. Model code: $THIRD_PARTY_DIR"
echo "      Checkpoint: $CHECKPOINT_FILE"
