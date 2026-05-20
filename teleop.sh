#!/usr/bin/env bash
# Teleoperate for practice/sanity check. Does not record a dataset.
# Usage: ./teleop.sh

set -euo pipefail

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

echo "Cameras configured. Starting teleoperation..."

# ── Teleoperate ───────────────────────────────────────────────────────────────

uv run lerobot-teleoperate \
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
  --display_data=true
