#!/usr/bin/env bash
# Teleoperate for practice/sanity check. Does not record a dataset.
# Usage: ./teleop.sh

set -euo pipefail

# Shared camera configuration (edit camera settings in cameras.sh).
source "$(dirname "$0")/cameras.sh"

echo "Starting teleoperation..."

# ── Teleoperate ───────────────────────────────────────────────────────────────
# v4l2_controls are applied by lerobot after OpenCV opens each camera, so they
# survive the driver reset that happens at stream start.

uv run lerobot-teleoperate \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=bw_follower \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=bw_leader \
  --display_data=true
