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

# ── Detect resume vs fresh start ──────────────────────────────────────────────

OUTPUT_DIR="outputs/train/${OUTPUT_NAME}"
LAST_LINK="${OUTPUT_DIR}/checkpoints/last"

if [ -L "${LAST_LINK}" ]; then
  # Read the symlink target (just the step name, e.g. "010000") and build a
  # relative path so it resolves correctly inside the container's working dir.
  LAST_STEP="$(readlink "${LAST_LINK}")"
  TRAIN_CONFIG="${OUTPUT_DIR}/checkpoints/${LAST_STEP}/pretrained_model/train_config.json"
  echo "Resuming from checkpoint: ${LAST_STEP}"
  RESUME_ARGS="--resume=true --config_path=${TRAIN_CONFIG}"
  PRETRAINED_ARGS=""
elif [ -d "${OUTPUT_DIR}" ]; then
  # Output dir exists but no checkpoint saved yet (crashed before first save).
  # Safe to wipe since there's nothing to resume from.
  echo "No checkpoint found — removing stale output dir and starting fresh."
  rm -rf "${OUTPUT_DIR}"
  RESUME_ARGS=""
  PRETRAINED_ARGS="--policy.pretrained_path=lerobot/pi0fast-base"
else
  echo "Starting fresh run (output dir: ${OUTPUT_DIR})"
  RESUME_ARGS=""
  PRETRAINED_ARGS="--policy.pretrained_path=lerobot/pi0fast-base"
fi

# ── Train ─────────────────────────────────────────────────────────────────────

docker compose build train
docker compose run --rm train \
  lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="/hf_cache/lerobot/${DATASET_REPO_ID}" \
    --policy.type=pi0_fast \
    ${PRETRAINED_ARGS} \
    --policy.device=cuda \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing=true \
    --policy.chunk_size=10 \
    --policy.n_action_steps=10 \
    --policy.max_action_tokens=256 \
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
