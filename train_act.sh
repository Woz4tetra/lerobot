#!/usr/bin/env bash
# Train an ACT policy on a recorded dataset.
#
# Step calculation:
#   total_frames     = NUM_EPISODES x EPISODE_TIME_S x FPS  (e.g. 50 x 30 x 30 = 45,000)
#   steps_per_epoch  = ceil(total_frames / BATCH_SIZE)       (e.g. ceil(45000/16) = 2813)
#   STEPS            = epochs x steps_per_epoch              (e.g. 10 x 2813 = 28,130)
#
# Usage: ./train.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_REPO_ID="local/my_task_20260620_223300"  # stamped repo id from record session
OUTPUT_NAME="act_my_task"                 # name for the output dir and HF policy repo
STEPS=50000                               # resumed from checkpoint
BATCH_SIZE=16                             # 8 is safe for RTX 4080 12GB with ACT
PUSH_TO_HUB=false                         # set true to upload trained policy to HuggingFace
WANDB_PROJECT="lerobot"

# ── Detect resume vs fresh start ──────────────────────────────────────────────
# Resume if a checkpoint exists (checkpoints/last symlinks to the newest step);
# otherwise train ACT from scratch. ACT has no pretrained base to start from.

OUTPUT_DIR="outputs/train/${OUTPUT_NAME}"
LAST_LINK="${OUTPUT_DIR}/checkpoints/last"

if [ -L "${LAST_LINK}" ]; then
  LAST_STEP="$(readlink "${LAST_LINK}")"
  TRAIN_CONFIG="${OUTPUT_DIR}/checkpoints/${LAST_STEP}/pretrained_model/train_config.json"
  echo "Resuming from checkpoint: ${LAST_STEP}"
  RESUME_ARGS="--resume=true --config_path=${TRAIN_CONFIG}"
elif [ -d "${OUTPUT_DIR}" ]; then
  echo "No checkpoint found — removing stale output dir and starting fresh."
  rm -rf "${OUTPUT_DIR}"
  RESUME_ARGS=""
else
  echo "Starting fresh run (output dir: ${OUTPUT_DIR})"
  RESUME_ARGS=""
fi

# ── Train ─────────────────────────────────────────────────────────────────────

uv run lerobot-train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${DATASET_REPO_ID}" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${OUTPUT_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq=5000 \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --policy.repo_id="local/${OUTPUT_NAME}" \
  --wandb.enable=true \
  --wandb.project="${WANDB_PROJECT}" \
  ${RESUME_ARGS}
