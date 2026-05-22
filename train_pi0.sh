#!/usr/bin/env bash
# Fine-tune a Pi0-Fast policy on a recorded dataset.
# Runs inside a custom Docker image built via docker-compose.
#
# First run: docker compose build train  (takes a few minutes)
# Subsequent runs: ./train_pi0.sh
#
# Usage: ./train_pi0.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_REPO_ID="local/my_task_20260519_221316"  # stamped repo id from record session
OUTPUT_NAME="pi0_fast_my_task"
STEPS=50000                               # start here and check loss; may need 50k-100k
BATCH_SIZE=4
PUSH_TO_HUB=false
WANDB_PROJECT="lerobot"

# ── Environment for docker-compose ────────────────────────────────────────────

export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
export HF_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface}"
export WANDB_API_KEY="$(tr -d '[:space:]' < ~/wandb_key)"

# ── Train ─────────────────────────────────────────────────────────────────────

docker compose build train
docker compose run --rm train \
  lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="/hf_cache/lerobot/${DATASET_REPO_ID}" \
    --policy.type=pi0_fast \
    --policy.pretrained_path=lerobot/pi0fast-base \
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
