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

# ── Camera settings ──────────────────────────────────────────────────────────
# Lock exposure, white balance, and focus so the image is consistent.
# These reset on camera reconnect, so run this script each session.

echo "Configuring cameras..."

# Gripper camera (/dev/video4) — YUYV 640x480 @ 31fps
v4l2-ctl --device=/dev/video4 --set-ctrl=auto_exposure=1
v4l2-ctl --device=/dev/video4 --set-ctrl=exposure_time_absolute=157
v4l2-ctl --device=/dev/video4 --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl --device=/dev/video4 --set-ctrl=white_balance_automatic=0
v4l2-ctl --device=/dev/video4 --set-ctrl=white_balance_temperature=3830
v4l2-ctl --device=/dev/video4 --set-ctrl=hue=15

# Overhead camera (/dev/video2) — MJPG 640x480 @ 30fps
v4l2-ctl --device=/dev/video2 --set-ctrl=auto_exposure=1
v4l2-ctl --device=/dev/video2 --set-ctrl=exposure_time_absolute=333
v4l2-ctl --device=/dev/video2 --set-ctrl=exposure_dynamic_framerate=0
v4l2-ctl --device=/dev/video2 --set-ctrl=white_balance_automatic=0
v4l2-ctl --device=/dev/video2 --set-ctrl=white_balance_temperature=3669
v4l2-ctl --device=/dev/video2 --set-ctrl=focus_automatic_continuous=0

echo "Cameras configured. Starting recording..."

# ── Record ────────────────────────────────────────────────────────────────────

uv run lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=bw_follower \
  --robot.cameras="{ \
    gripper:  {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 31}, \
    overhead: {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 30, fourcc: MJPG} \
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
