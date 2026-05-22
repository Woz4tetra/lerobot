#!/usr/bin/env bash
# Run the trained Pi0-Fast policy on the robot.
# Controls: ESC to stop
#
# Usage: ./rollout_pi0_fast.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

POLICY_PATH="outputs/train/pi0_fast_my_task/checkpoints/002000/pretrained_model"
TASK_DESCRIPTION="pick up the object and place it in the box"
EPISODE_TIME_S=60

echo "Starting rollout..."

# ── Rollout ───────────────────────────────────────────────────────────────────

uv run lerobot-rollout \
  --strategy.type=base \
  --policy.path="${POLICY_PATH}" \
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
  --task="${TASK_DESCRIPTION}" \
  --duration="${EPISODE_TIME_S}" \
  --display_data=true
