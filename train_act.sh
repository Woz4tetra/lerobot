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

DATASET_REPO_ID="local/my_task_20260519_221316"  # stamped repo id from record session
OUTPUT_NAME="act_my_task"                 # name for the output dir and HF policy repo
STEPS=80000                               # resumed from checkpoint
BATCH_SIZE=8                              # 8 is safe for RTX 4080 12GB with ACT
PUSH_TO_HUB=false                         # set true to upload trained policy to HuggingFace
WANDB_PROJECT="lerobot"

# ── Train ─────────────────────────────────────────────────────────────────────

uv run lerobot-train \
  --resume=true \
  --config_path="outputs/train/${OUTPUT_NAME}/checkpoints/030000/pretrained_model/train_config.json" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${DATASET_REPO_ID}" \
  --policy.type=act \
  --policy.device=cuda \
  --output_dir="outputs/train/${OUTPUT_NAME}" \
  --job_name="${OUTPUT_NAME}" \
  --batch_size="${BATCH_SIZE}" \
  --steps="${STEPS}" \
  --save_freq=5000 \
  --policy.push_to_hub="${PUSH_TO_HUB}" \
  --policy.repo_id="local/${OUTPUT_NAME}" \
  --wandb.enable=true \
  --wandb.project="${WANDB_PROJECT}"
