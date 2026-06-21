#!/usr/bin/env bash
# Run the trained ACT policy on the robot.
# Controls: ESC to stop
#
# Usage: ./rollout.sh

set -euo pipefail

# Shared camera configuration (edit camera settings in cameras.sh).
source "$(dirname "$0")/cameras.sh"

# ── Configuration ─────────────────────────────────────────────────────────────

POLICY_PATH="outputs/train/act_my_task/checkpoints/050000/pretrained_model"
TASK_DESCRIPTION="pick up the cube and stack tower"
EPISODE_TIME_S=30

echo "Starting rollout..."

# ── Rollout ───────────────────────────────────────────────────────────────────
# v4l2_controls are applied by lerobot after OpenCV opens each camera, so they
# survive the driver reset that happens at stream start.

uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path="${POLICY_PATH}" \
  --robot.type=so101_follower \
  --robot.port=/dev/lerobot_follower \
  --robot.id=bw_follower \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --task="${TASK_DESCRIPTION}" \
  --duration="${EPISODE_TIME_S}" \
  --display_data=true \
  --repeat=true
