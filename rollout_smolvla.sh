#!/usr/bin/env bash
# Run a fine-tuned SmolVLA policy on the SO-101 follower.
#
# SmolVLA is ~2GB at inference and runs locally, so there is no _remote variant
# (unlike pi0.5). If latency is too high on CPU, set --policy.device=cuda below.
#
# SmolVLA is language-conditioned: TASK_DESCRIPTION must match the task strings
# your dataset was recorded/trained with, or behavior will be off.
#
# Usage: ./rollout_smolvla.sh

set -euo pipefail

# Shared camera configuration (edit camera settings in cameras.sh).
source "$(dirname "$0")/cameras.sh"

POLICY_PATH="outputs/train/smolvla_my_task/checkpoints/006000/pretrained_model"
TASK_DESCRIPTION="pick up the cube and stack tower"
EPISODE_TIME_S=30

echo "Starting rollout..."

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
