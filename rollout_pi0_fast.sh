#!/usr/bin/env bash
# Run the trained Pi0-Fast policy on the robot.
# Controls: ESC to stop
#
# Usage: ./rollout_pi0_fast.sh

set -euo pipefail

# Shared camera configuration (edit camera settings in cameras.sh).
source "$(dirname "$0")/cameras.sh"

# ── Configuration ─────────────────────────────────────────────────────────────

POLICY_PATH="outputs/train/pi0_fast_my_task/checkpoints/006000/pretrained_model"
TASK_DESCRIPTION="pick up the object and place it in the box"
EPISODE_TIME_S=60

echo "Starting rollout..."

# ── Rollout ───────────────────────────────────────────────────────────────────

uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path="${POLICY_PATH}" \
  --robot.type=so101_follower \
  --robot.port=/dev/lerobot_follower \
  --robot.id=bw_follower \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --task="${TASK_DESCRIPTION}" \
  --duration="${EPISODE_TIME_S}" \
  --display_data=true
