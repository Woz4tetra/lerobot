#!/usr/bin/env bash
# Fine-tune a Pi0.5 policy on a recorded dataset.
# Runs inside a custom Docker image built via docker-compose.
#
# First run: docker compose build train  (takes a few minutes)
# Subsequent runs: ./train_pi05.sh
#
# NOTE: Pi0.5 expects quantile-normalized datasets by default. If your dataset
# was recorded without quantile stats, either run:
#   uv run python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py --repo-id=<DATASET_REPO_ID>
# or add: --policy.normalization_mapping='{"ACTION": "MEAN_STD", "STATE": "MEAN_STD", "VISUAL": "IDENTITY"}'
#
# Usage: ./train_pi05.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_REPO_ID="local/my_task_20260519_221316"  # stamped repo id from record session
OUTPUT_NAME="pi05_my_task"
NUM_GPUS=3
BATCH_SIZE=8                              # per GPU; effective batch = BATCH_SIZE * NUM_GPUS
STEPS=100000                               # start here and check loss; may need 50k-100k
PUSH_TO_HUB=false
WANDB_PROJECT="lerobot"

# ── Environment for docker-compose ────────────────────────────────────────────

export USER_ID="$(id -u)"
export GROUP_ID="$(id -g)"
export HF_CACHE="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface}"
export WANDB_API_KEY="$(tr -d '[:space:]' < ~/wandb_key)"
export NVIDIA_VISIBLE_DEVICES=all  # override any single-GPU selection left by policy_server.sh

# ── Detect resume vs fresh start ──────────────────────────────────────────────

OUTPUT_DIR="outputs/train/${OUTPUT_NAME}"
LAST_LINK="${OUTPUT_DIR}/checkpoints/last"

if [ -L "${LAST_LINK}" ]; then
  LAST_STEP="$(readlink "${LAST_LINK}")"
  TRAIN_CONFIG="${OUTPUT_DIR}/checkpoints/${LAST_STEP}/pretrained_model/train_config.json"
  echo "Resuming from checkpoint: ${LAST_STEP}"
  RESUME_ARGS="--resume=true --config_path=${TRAIN_CONFIG}"
  PRETRAINED_ARGS=""
elif [ -d "${OUTPUT_DIR}" ]; then
  echo "No checkpoint found — removing stale output dir and starting fresh."
  rm -rf "${OUTPUT_DIR}"
  RESUME_ARGS=""
  PRETRAINED_ARGS="--policy.pretrained_path=lerobot/pi05_base"
else
  echo "Starting fresh run (output dir: ${OUTPUT_DIR})"
  RESUME_ARGS=""
  PRETRAINED_ARGS="--policy.pretrained_path=lerobot/pi05_base"
fi

# ── Train ─────────────────────────────────────────────────────────────────────

docker compose build train
docker compose run --rm train \
  accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=bf16 \
  -m lerobot.scripts.lerobot_train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="/hf_cache/lerobot/${DATASET_REPO_ID}" \
    --policy.type=pi05 \
    ${PRETRAINED_ARGS} \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${OUTPUT_NAME}" \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq=2000 \
    --policy.push_to_hub="${PUSH_TO_HUB}" \
    --policy.repo_id="local/${OUTPUT_NAME}" \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project="${WANDB_PROJECT}" \
    ${RESUME_ARGS}
