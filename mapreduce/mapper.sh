#!/bin/bash
set -euo pipefail

# CM4IR Hadoop Streaming mapper
# - Reads image paths from stdin (one per line)
# - Selects GPU based on $HADOOP_TASK_ID (or 0)
# - Copies images from SOURCE_DIR to EXP_DATASET_DIR
# - Runs CM4IR main.py once per task_id (batch on copied dir)
# - Emits one status line per processed image: "OK\t<task_id>\t<fname>\t<out_png>" or "FAIL\t<task_id>\t<fname>\t<reason>"

# Configurable via environment variables (with safe defaults)
GPU_COUNT=${GPU_COUNT:-2}

# Derive TASK_ID from several env vars used by Hadoop Streaming
TASK_ID="${HADOOP_TASK_ID-}"
if [[ -z "${TASK_ID}" ]]; then
  TASK_ID="${mapreduce_task_partition-}"
fi
if [[ -z "${TASK_ID}" ]]; then
  TASK_ID="${MAPREDUCE_TASK_PARTITION-}"
fi
if [[ -z "${TASK_ID}" && -n "${mapred_task_id-}" ]]; then
  # e.g., task_1695135791600_0001_m_000007 -> 7
  if [[ "${mapred_task_id}" =~ _([0-9]{6})$ ]]; then
    TASK_ID=$((10#${BASH_REMATCH[1]}))
  fi
fi
if [[ -z "${TASK_ID}" && -n "${MAPRED_TASK_ID-}" ]]; then
  if [[ "${MAPRED_TASK_ID}" =~ _([0-9]{6})$ ]]; then
    TASK_ID=$((10#${BASH_REMATCH[1]}))
  fi
fi
TASK_ID=${TASK_ID:-0}
export CUDA_VISIBLE_DEVICES=$(( TASK_ID % GPU_COUNT ))

# Paths (tweak for Hadoop nodes)
CM4IR_ROOT=${CM4IR_ROOT:-/workspace}
SOURCE_DIR=${SOURCE_DIR:-/kaggle/working/data}
# Dataset name must match datasets/__init__.py expectations
DATASET_NAME=${DATASET_NAME:-lsun_cat}
TARGET_DIR=${TARGET_DIR:-"${CM4IR_ROOT}/exp/datasets/${DATASET_NAME}/cat"}

# Inference config
CONFIG_FILE=${CONFIG_FILE:-configs/lsun_cat_256.yml}
DEG=${DEG:-sr_bicubic}
DEG_SCALE=${DEG_SCALE:-4}
SIGMA_Y=${SIGMA_Y:-0.05}
INPAINT_MASK=${INPAINT_MASK:-random_80_mask.npy}
MODEL_CKPT=${MODEL_CKPT:-lsun_cat/cd_cat256_lpips.pt}
IMAGE_FOLDER="${TASK_ID}"

# Dry-run: copy only, skip python run
DRY_RUN=${DRY_RUN:-0}

# Prepare working dirs
mkdir -p "${TARGET_DIR}"
rm -rf "${TARGET_DIR:?}"/*

# Read stdin list and stage files
STAGED_FILES=()
while IFS= read -r img || [[ -n "$img" ]]; do
  [[ -z "$img" ]] && continue
  fname=$(basename -- "$img")
  src="${SOURCE_DIR}/${fname}"
  if [[ ${DRY_RUN} -eq 1 ]]; then
    # Skip copying in dry-run; just record staged items
    echo -e "STAGED\t${TASK_ID}\t${fname}" 1>&2
    STAGED_FILES+=("$fname")
    continue
  fi
  if cp -f -- "$src" "${TARGET_DIR}/${fname}" 2>/dev/null; then
    echo -e "STAGED\t${TASK_ID}\t${fname}" 1>&2
  else
    echo -e "FAIL\t${TASK_ID}\t${fname}\tCOPY"; continue
  fi
  STAGED_FILES+=("$fname")
done

if [[ ${DRY_RUN} -eq 1 ]]; then
  for f in "${STAGED_FILES[@]}"; do
    echo -e "OK\t${TASK_ID}\t${f}\tDRY_RUN"
  done
  exit 0
fi

# Run CM4IR once per task
cd "${CM4IR_ROOT}"
python3 "${CM4IR_ROOT}/main.py" \
  --config "$(basename "${CONFIG_FILE}")" \
  --path_y "${DATASET_NAME}" \
  --deg "${DEG}" \
  --deg_scale "${DEG_SCALE}" \
  --sigma_y "${SIGMA_Y}" \
  -i "${IMAGE_FOLDER}" \
  --iN 250 \
  --gamma 0.2 \
  --model_ckpt "${MODEL_CKPT}" || {
    for f in "${STAGED_FILES[@]}"; do
      echo -e "FAIL\t${TASK_ID}\t${f}\tRUN";
    done
    exit 0
  }

OUT_DIR="${CM4IR_ROOT}/exp/image_samples/${IMAGE_FOLDER}"
shopt -s nullglob
for png in "${OUT_DIR}"/*.png; do
  base=$(basename -- "$png")
  echo -e "OK\t${TASK_ID}\t${base%.*}.png\t${png}"
done

# Optional: upload results to HDFS if enabled
if [[ "${ENABLE_HDFS_PUT:-0}" -eq 1 ]]; then
  HDFS_OUTDIR=${HDFS_OUTDIR:-/result}
  hdfs dfs -mkdir -p "${HDFS_OUTDIR}" || true
  hdfs dfs -put -f "${OUT_DIR}"/*.png "${HDFS_OUTDIR}/" || true
fi
