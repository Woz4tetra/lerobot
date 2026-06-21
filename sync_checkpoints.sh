#!/usr/bin/env bash
# Sync the latest pi0_fast checkpoint from the training cluster to this machine.
# Only downloads pretrained_model/ (weights + processor state), not the
# 15 GB optimizer/rng training state.
#
# Usage: ./sync_checkpoints.sh

set -euo pipefail

CLUSTER="ben@megamind"
REMOTE_BASE="~/lerobot/outputs/train/smolvla_my_task/checkpoints"
LOCAL_BASE="outputs/train/smolvla_my_task/checkpoints"

STEP=$(ssh "${CLUSTER}" "ls ${REMOTE_BASE}/ | grep -v last | sort -V | tail -1")

if [ -z "${STEP}" ]; then
    echo "No checkpoints found on cluster." >&2
    exit 1
fi

echo "Latest checkpoint: ${STEP}"
LOCAL_DIR="${LOCAL_BASE}/${STEP}/pretrained_model"
mkdir -p "${LOCAL_DIR}"
rsync -avz --progress \
    "${CLUSTER}:${REMOTE_BASE}/${STEP}/pretrained_model/" \
    "${LOCAL_DIR}/"

echo ""
echo "Updating rollout script to use step ${STEP}..."
sed -i "s|checkpoints/.*/pretrained_model|checkpoints/${STEP}/pretrained_model|g" rollout_pi0_fast.sh
echo "Done. Run ./rollout_pi0_fast.sh to test."
