#!/usr/bin/env bash
# Record a dataset via teleoperation.
# Controls: → save & next episode  ← redo episode  ESC finish
#
# Usage: ./record.sh

set -euo pipefail

# Shared camera configuration (edit camera settings in cameras.sh).
source "$(dirname "$0")/cameras.sh"

# ── Configuration ─────────────────────────────────────────────────────────────
# Set these before recording.

# DATASET_REPO_ID="local/my_task"
DATASET_REPO_ID="local/my_task_20260620_223300"  # stamped repo id from previous session
TASK_DESCRIPTION="pick up the cube and stack tower"
NUM_EPISODES=50
EPISODE_TIME_S=30
RESET_TIME_S=3
PUSH_TO_HUB=false                          # set true to upload to HuggingFace after recording

# Resume only if this dataset already exists locally; otherwise start a new one.
# A dataset is "existing" once meta/info.json has been written. Hardcoding
# --resume=true makes lerobot-record fall back to the Hub and 404 on a new repo.
DATASET_ROOT="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${DATASET_REPO_ID}"
if [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
  RESUME=true
  echo "Resuming existing dataset at ${DATASET_ROOT}"
else
  RESUME=false
  echo "Creating new dataset at ${DATASET_ROOT}"
fi

echo "Starting recording..."

# ── Record ────────────────────────────────────────────────────────────────────
# v4l2_controls are applied by lerobot after OpenCV opens each camera, so they
# survive the driver reset that happens at stream start.

uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=bw_follower \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=bw_leader \
  --resume="${RESUME}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.single_task="${TASK_DESCRIPTION}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data=true
