#!/usr/bin/env bash
# Run pi05 rollout with inference on the cluster, robot control on this machine.
# Requires policy_server.sh to be running on megamind first.
#
# Run on megamind:  ./policy_server.sh
# Run on laptop:    ./rollout_pi05_remote.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

CLUSTER="ben@megamind"
LOCAL_PORT=8080

# Absolute path inside the cluster container (/lerobot/outputs is the mounted volume).
CHECKPOINT_STEP="010000"
SERVER_CHECKPOINT="/lerobot/outputs/train/pi05_my_task/checkpoints/${CHECKPOINT_STEP}/pretrained_model"

TASK_DESCRIPTION="pick up the object and place it in the box"
EPISODE_TIME_S=60

# ── SSH tunnel ─────────────────────────────────────────────────────────────────

echo "Opening SSH tunnel to ${CLUSTER}:${LOCAL_PORT}..."
ssh -L "${LOCAL_PORT}:localhost:${LOCAL_PORT}" -N "${CLUSTER}" &
TUNNEL_PID=$!
trap "kill ${TUNNEL_PID} 2>/dev/null; exit" EXIT INT TERM
sleep 1  # give ssh a moment to establish the tunnel
echo "Tunnel open (PID ${TUNNEL_PID})"

# ── Rollout ───────────────────────────────────────────────────────────────────

echo "Starting remote rollout..."

uv run python -m lerobot.async_inference.robot_client \
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
  --server_address="localhost:${LOCAL_PORT}" \
  --policy_type=pi05 \
  --pretrained_name_or_path="${SERVER_CHECKPOINT}" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.3 \
  --fps=30
