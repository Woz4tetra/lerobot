#!/usr/bin/env bash
# Fine-tune a Pi0-Fast policy on a recorded dataset, running inside the official
# LeRobot GPU Docker image to avoid native dependency issues on the cluster.
#
# Prerequisites on the host:
#   docker, nvidia-container-toolkit, ~/wandb_key with W&B API key
#
# Usage: ./train_pi0.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_REPO_ID="local/my_task_20260519_221316"  # stamped repo id from record session
OUTPUT_NAME="pi0_fast_my_task"
STEPS=10000                               # start here and check loss; may need 50k-100k
BATCH_SIZE=4
PUSH_TO_HUB=false
WANDB_PROJECT="lerobot"

HF_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface}"
export WANDB_API_KEY="$(tr -d '[:space:]' < ~/wandb_key)"

# ── Train ─────────────────────────────────────────────────────────────────────

docker run --rm --gpus all --shm-size 16gb --user "$(id -u):$(id -g)" \
  -v "${HF_CACHE}:/home/user_lerobot/.cache/huggingface" \
  -v "${PWD}/outputs:/lerobot/outputs" \
  -e WANDB_API_KEY="${WANDB_API_KEY}" \
  huggingface/lerobot-gpu:latest \
  lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="/home/user_lerobot/.cache/huggingface/lerobot/${DATASET_REPO_ID}" \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0_fast_base \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
    --output_dir="outputs/train/${OUTPUT_NAME}" \
    --job_name="${OUTPUT_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=2000 \
    --policy.push_to_hub="${PUSH_TO_HUB}" \
    --policy.repo_id="local/${OUTPUT_NAME}" \
    --wandb.enable=true \
    --wandb.project="${WANDB_PROJECT}"
