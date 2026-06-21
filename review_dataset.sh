#!/usr/bin/env bash
# Review a recorded LeRobotDataset episode by episode, keeping or rejecting each.
#
# For every episode it opens the rerun viewer, then asks you to keep or reject.
# Rejected episodes are collected and deleted in a single batch at the end, so
# the original indices stay valid throughout the review.
#
# Usage:
#   ./review_dataset.sh                 # review all episodes from 0
#   ./review_dataset.sh 10              # start reviewing from episode 10
#   ./review_dataset.sh --summary-only  # print summary, no review
#
# Override the dataset with the DATASET_REPO_ID env var:
#   DATASET_REPO_ID=local/other_task ./review_dataset.sh

set -euo pipefail

# ── Configuration ─────────────────────────────────────────────────────────────
# Defaults to the same dataset record.sh writes to. Override via env var.

DATASET_REPO_ID="${DATASET_REPO_ID:-local/my_task_20260620_223300}"
DATASET_ROOT="${HF_LEROBOT_HOME:-${HOME}/.cache/huggingface/lerobot}/${DATASET_REPO_ID}"

# ── Argument parsing ──────────────────────────────────────────────────────────

START_EPISODE=0
SUMMARY_ONLY=false
for arg in "$@"; do
  case "${arg}" in
    --summary-only) SUMMARY_ONLY=true ;;
    *[!0-9]*) echo "Unknown argument: ${arg}" >&2; exit 1 ;;
    *) START_EPISODE="${arg}" ;;
  esac
done

# ── Existence check ───────────────────────────────────────────────────────────

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
  echo "No dataset found at ${DATASET_ROOT}" >&2
  echo "Record one first (./record.sh) or set DATASET_REPO_ID." >&2
  exit 1
fi

# ── Summary ───────────────────────────────────────────────────────────────────

echo "Dataset: ${DATASET_REPO_ID}"
echo "Root:    ${DATASET_ROOT}"
echo

uv run python - "${DATASET_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
info = json.loads((root / "meta" / "info.json").read_text())

total_eps = info.get("total_episodes", 0)
fps = info.get("fps", 0)
total_frames = info.get("total_frames", 0)
secs = total_frames / fps if fps else 0

print("Summary")
print(f"  codebase_version : {info.get('codebase_version')}")
print(f"  robot_type       : {info.get('robot_type')}")
print(f"  fps              : {fps}")
print(f"  episodes         : {total_eps}")
print(f"  frames           : {total_frames}  (~{secs / 60:.1f} min total)")
if total_eps:
    avg = total_frames / total_eps
    print(f"  avg episode      : {avg:.0f} frames (~{avg / fps:.1f}s)" if fps else f"  avg episode      : {avg:.0f} frames")
print(f"  tasks            : {info.get('total_tasks')}")
print(f"  data size        : {info.get('data_files_size_in_mb')} MB")
print(f"  video size       : {info.get('video_files_size_in_mb')} MB")

feats = info.get("features", {})
cams = [k for k in feats if k.startswith("observation.images.")]
print(f"  cameras          : {', '.join(c.split('.')[-1] for c in cams) or 'none'}")
for c in cams:
    shape = feats[c].get("shape")
    print(f"      {c.split('.')[-1]:<12} shape={shape}")
print(f"  features         : {', '.join(feats.keys())}")

tasks_file = root / "meta" / "tasks.parquet"
if tasks_file.exists():
    import pandas as pd
    tasks = pd.read_parquet(tasks_file).reset_index()
    col = "task" if "task" in tasks.columns else tasks.columns[0]
    print("\nTasks")
    for t in tasks[col].tolist():
        print(f"  - {t}")
PY

echo

if [[ "${SUMMARY_ONLY}" == "true" ]]; then
  exit 0
fi

# ── Sequential review ─────────────────────────────────────────────────────────

TOTAL_EPISODES=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" \
  "${DATASET_ROOT}/meta/info.json")

if (( START_EPISODE >= TOTAL_EPISODES )); then
  echo "Start episode ${START_EPISODE} is past the last episode (${TOTAL_EPISODES} total)." >&2
  exit 1
fi

REJECTED=()

echo "Reviewing episodes ${START_EPISODE}..$((TOTAL_EPISODES - 1)) of ${DATASET_REPO_ID}"
echo "For each episode: [k]eep (default) · [r]eject · [q]uit (stop and apply)"
echo

for (( ep = START_EPISODE; ep < TOTAL_EPISODES; ep++ )); do
  echo "──────────────────────────────────────────────────────────────────────"
  echo "Episode ${ep} / $((TOTAL_EPISODES - 1))   (rejected so far: ${#REJECTED[@]})"
  echo "Opening in rerun..."

  # Local mode spawns/reuses the viewer and returns after streaming the frames.
  uv run lerobot-dataset-viz \
    --repo-id "${DATASET_REPO_ID}" \
    --root "${DATASET_ROOT}" \
    --episode-index "${ep}" \
    >/dev/null 2>&1 || echo "  (viewer failed for episode ${ep}, review anyway)"

  while true; do
    read -r -p "Episode ${ep}: [k]eep / [r]eject / [q]uit > " ans </dev/tty || ans="q"
    case "${ans:-k}" in
      k|K|keep|"") echo "  kept"; break ;;
      r|R|reject) REJECTED+=("${ep}"); echo "  rejected"; break ;;
      q|Q|quit) echo "  stopping review"; ep=${TOTAL_EPISODES}; break ;;
      *) echo "  please answer k, r, or q" ;;
    esac
  done
done

echo
echo "──────────────────────────────────────────────────────────────────────"

# ── Apply rejections ──────────────────────────────────────────────────────────

if (( ${#REJECTED[@]} == 0 )); then
  echo "No episodes rejected. Dataset unchanged."
  exit 0
fi

# Build a JSON list like [0, 3, 7] for --operation.episode_indices.
JOINED=$(IFS=,; echo "${REJECTED[*]}")
echo "Rejected ${#REJECTED[@]} episode(s): ${JOINED}"
read -r -p "Move these to trash? (the full dataset is backed up first) [y/N] > " confirm </dev/tty || confirm="n"
if [[ ! "${confirm}" =~ ^[yY]$ ]]; then
  echo "Aborted. Dataset unchanged. Rejected indices were: ${JOINED}"
  exit 0
fi

# Nothing is permanently deleted. The in-place edit moves the original dataset to
# "<name>_old" and writes the filtered copy in its place. We then move that backup
# into a timestamped trash folder so it survives future edits (the built-in
# "<name>_old" backup is overwritten on the next in-place edit).
TRASH_DIR="$(dirname "${DATASET_ROOT}")/.trash"
TRASH_PATH="${TRASH_DIR}/$(basename "${DATASET_ROOT}")_$(date +%Y%m%d_%H%M%S)"
BACKUP_PATH="${DATASET_ROOT}_old"

echo "Removing episodes [${JOINED}] from the working dataset..."
uv run lerobot-edit-dataset \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATASET_ROOT}" \
  --operation.type delete_episodes \
  --operation.episode_indices "[${JOINED}]"

if [[ -d "${BACKUP_PATH}" ]]; then
  mkdir -p "${TRASH_DIR}"
  mv "${BACKUP_PATH}" "${TRASH_PATH}"
  echo "Done. Original dataset (with the rejected episodes) moved to trash:"
  echo "  ${TRASH_PATH}"
  echo "Restore it with:  rm -rf '${DATASET_ROOT}' && mv '${TRASH_PATH}' '${DATASET_ROOT}'"
else
  echo "Warning: expected backup at ${BACKUP_PATH} was not found." >&2
  echo "The edit may not have run in-place. Check ${DATASET_ROOT}." >&2
fi

echo "Re-run with --summary-only to see the updated counts."
