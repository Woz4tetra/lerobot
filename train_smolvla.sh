#!/usr/bin/env bash
# Fine-tune a SmolVLA policy on a recorded dataset.
# Runs inside the same custom Docker image used by the other train_*.sh scripts.
#
# First run: docker compose build train  (takes a few minutes)
# Subsequent runs: ./train_smolvla.sh
#
# SmolVLA is a ~450M VLA (~2GB at inference). Unlike pi0.5 it fits on a single
# consumer GPU, so you can also run it locally with `uv run lerobot-train ...`
# using the same flags below if you'd rather skip docker.
#
# Notes vs train_pi05.sh:
#   - No quantile stats needed: SmolVLA uses MEAN_STD/IDENTITY normalization by default.
#   - No --policy.gradient_checkpointing flag (SmolVLAConfig doesn't define it).
#   - SmolVLA is language-conditioned: the --task string used at rollout time should
#     match the task strings in your dataset.
#
# Usage: ./train_smolvla.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────

DATASET_REPO_ID="local/my_task_20260620_223300"  # stamped repo id from record session
OUTPUT_NAME="smolvla_my_task"
BATCH_SIZE=64                              # SmolVLA is small; 64 is the documented default. Raise if VRAM allows.
STEPS=20000                                # ~4 hrs on a single A100; tune based on loss/performance
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
  LAST_STEP="$(readlink "${LAST_LINK}")"
  TRAIN_CONFIG="${OUTPUT_DIR}/checkpoints/${LAST_STEP}/pretrained_model/train_config.json"
  echo "Resuming from checkpoint: ${LAST_STEP}"
  RESUME_ARGS="--resume=true --config_path=${TRAIN_CONFIG}"
  PRETRAINED_ARGS=""
elif [ -d "${OUTPUT_DIR}" ]; then
  echo "No checkpoint found — removing stale output dir and starting fresh."
  rm -rf "${OUTPUT_DIR}"
  RESUME_ARGS=""
  PRETRAINED_ARGS="--policy.pretrained_path=lerobot/smolvla_base"
else
  echo "Starting fresh run (output dir: ${OUTPUT_DIR})"
  RESUME_ARGS=""
  PRETRAINED_ARGS="--policy.pretrained_path=lerobot/smolvla_base"
fi

# ── Train ─────────────────────────────────────────────────────────────────────

docker compose build train
docker compose run --rm train \
  lerobot-train \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="/hf_cache/lerobot/${DATASET_REPO_ID}" \
    --policy.type=smolvla \
    ${PRETRAINED_ARGS} \
    --policy.device=cuda \
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
