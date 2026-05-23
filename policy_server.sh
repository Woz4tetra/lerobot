#!/usr/bin/env bash
# Run the policy inference server on the cluster GPU.
# The robot_client on the laptop connects via SSH tunnel.
# Policy type and checkpoint are set by the client (rollout script).
#
# Run on megamind:  ./policy_server.sh
# Run on laptop:    ./rollout_pi05_remote.sh

set -euo pipefail

PORT=8080
FPS=30

export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
export HF_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface}"
export WANDB_API_KEY=""  # not needed for inference

# Pick the GPU with the most free memory so we don't collide with training.
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
  | sort -t',' -k2 -rn | head -1 | cut -d',' -f1 | tr -d ' ')
echo "Selected GPU ${FREE_GPU} for policy server"
export NVIDIA_VISIBLE_DEVICES="${FREE_GPU}"

echo "Starting policy server on port ${PORT}..."

docker compose run --rm \
  -p "${PORT}:${PORT}" \
  -e NVIDIA_VISIBLE_DEVICES \
  -w /lerobot/outputs \
  train \
  python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port="${PORT}" \
    --fps="${FPS}" \
    --inference_latency=0.0
