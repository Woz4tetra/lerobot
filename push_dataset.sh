#!/usr/bin/env bash
# Sync the recorded dataset to the GPU cluster for training.
#
# Usage: ./push_dataset.sh

set -euo pipefail

DATASET_REPO_ID="local/my_task_20260620_223300"
LOCAL_ROOT="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}"
REMOTE="ben@megamind"
REMOTE_ROOT="/home/ben/.cache/huggingface/lerobot"

SRC="${LOCAL_ROOT}/${DATASET_REPO_ID}"
DST="${REMOTE}:${REMOTE_ROOT}/${DATASET_REPO_ID}"

echo "Pushing ${DATASET_REPO_ID} to ${REMOTE}..."
ssh "${REMOTE}" "mkdir -p ${REMOTE_ROOT}/${DATASET_REPO_ID}"
rsync -avz --progress "${SRC}/" "${DST}/"
echo "Done."
