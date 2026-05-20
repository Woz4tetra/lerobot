#!/usr/bin/env bash
# Record a dataset via teleoperation.
# Controls: → save & next episode  ← redo episode  ESC finish
#
# Usage: ./record.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Set these before recording.

# DATASET_REPO_ID="local/my_task"
DATASET_REPO_ID="local/my_task_20260519_221316"  # stamped repo id from previous session
TASK_DESCRIPTION="pick up the object and place it in the box"
NUM_EPISODES=50
EPISODE_TIME_S=30
RESET_TIME_S=10
PUSH_TO_HUB=false                          # set true to upload to HuggingFace after recording

echo "Starting recording..."

# ── Record ────────────────────────────────────────────────────────────────────
# v4l2_controls are applied by lerobot after OpenCV opens each camera, so they
# survive the driver reset that happens at stream start.

uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=bw_follower \
  --robot.cameras="{ \
    gripper: { \
      type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 31, \
      v4l2_controls: { \
        auto_exposure: 1, exposure_time_absolute: 157, exposure_dynamic_framerate: 0, \
        white_balance_automatic: 0, white_balance_temperature: 3830, hue: 15 \
      } \
    }, \
    overhead: { \
      type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG, \
      v4l2_controls: { \
        auto_exposure: 1, exposure_time_absolute: 333, exposure_dynamic_framerate: 0, \
        white_balance_automatic: 0, white_balance_temperature: 3669, focus_automatic_continuous: 0 \
      } \
    } \
  }" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=bw_leader \
  --resume=true \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${DATASET_REPO_ID}" \
  --dataset.single_task="${TASK_DESCRIPTION}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub="${PUSH_TO_HUB}" \
  --display_data=true
