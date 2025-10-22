#!/usr/bin/env bash
set -euo pipefail

# =====================
# CM4IR Hadoop Mapper
# - Assigns GPU per task
# - Copies shard images to expected dataset folder
# - Runs CM4IR once per shard
# - Uploads results back to HDFS
# =====================

# --- GPU assignment ---
N_GPUS_DEFAULT=1
if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPUS_DEFAULT=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print ($1==""?0:$1)}')
  if [[ "$N_GPUS_DEFAULT" -eq 0 ]]; then N_GPUS_DEFAULT=1; fi
fi
GPU_COUNT=${N_GPUS:-$N_GPUS_DEFAULT}

# Extract a numeric task id from common Hadoop vars
TASK_ID_RAW=${HADOOP_TASK_ID:-${mapreduce_task_id:-${mapred_task_id:-0}}}
# Fallbacks if not numeric
if [[ "$TASK_ID_RAW" =~ ^[0-9]+$ ]]; then
  TASK_ID="$TASK_ID_RAW"
else
  # typical format: task_1714396889410_0001_m_000003
  TASK_ID=$(echo "$TASK_ID_RAW" | grep -oE '[0-9]{1,6}$' || true)
  TASK_ID=${TASK_ID:-0}
fi

# Map task id to GPU index (if GPUs available)
GPU_INDEX=$(( TASK_ID % GPU_COUNT ))
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

# --- Base paths ---
BASE_DIR=${BASE_DIR:-/workspace}
if [[ ! -d "$BASE_DIR" ]]; then BASE_DIR=$(pwd); fi

# Config can be file name or path; main.py expects just the file name
CM4IR_CONFIG=${CM4IR_CONFIG:-lsun_cat_256.yml}
CONFIG_NAME=$(basename "$CM4IR_CONFIG")

# Degradation and hyperparameters (tune via -cmdenv on job submit)
DEG=${DEG:-sr_bicubic}
DEG_SCALE=${DEG_SCALE:-4}
SIGMA_Y=${SIGMA_Y:-0.05}
I_N=${I_N:-250}
GAMMA=${GAMMA:-0.2}
ETA=${ETA:-0.1}
ZETA=${ZETA:-0}
DELTAS=${DELTAS:-}
DELTAS_INJECTION_TYPE=${DELTAS_INJECTION_TYPE:-0}

# Dataset routing
PATH_Y=${PATH_Y:-lsun_cat}  # lsun_cat | lsun_bedroom | imagenet | celeba_hq ...
# Class subdir used by torchvision.datasets.ImageFolder for LSUN variants
if [[ -z "${CLASS_SUBDIR:-}" ]]; then
  if [[ "$PATH_Y" == *cat* ]]; then CLASS_SUBDIR=cat; 
  elif [[ "$PATH_Y" == *bedroom* ]]; then CLASS_SUBDIR=bedroom; 
  else CLASS_SUBDIR=""; fi
fi

# Where to place shard images for this mapper so that main.py finds them
# NOTE: For LSUN Cat/Bedroom configs, datasets/__init__.py reads from:
#   $BASE_DIR/exp/datasets/<config.data.dataset>/{cat|bedroom}
# which matches PATH_Y when PATH_Y is lsun_cat or lsun_bedroom
if [[ -n "$CLASS_SUBDIR" ]]; then
  TARGET_DIR="$BASE_DIR/exp/datasets/$PATH_Y/$CLASS_SUBDIR"
else
  # Special-case ImageNet subset: code reads from exp/datasets/imagenet/imagenet
  if [[ "$PATH_Y" == "imagenet" ]]; then
    TARGET_DIR="$BASE_DIR/exp/datasets/imagenet/imagenet"
  else
    # Fallback for custom ImageFolder-style datasets (set DATASET_DIR explicitly if needed)
    TARGET_DIR=${DATASET_DIR:-"$BASE_DIR/exp/datasets/$PATH_Y"}
  fi
fi
mkdir -p "$TARGET_DIR"

# Source: either local directory containing the raw files OR download from HDFS
SOURCE_DIR=${SOURCE_DIR:-}
DOWNLOAD_FROM_HDFS=${DOWNLOAD_FROM_HDFS:-}

# Ensure a clean shard directory (mapper-local container filesystem)
rm -rf "${TARGET_DIR}"/*

# HDFS output directory (per-task subfolder recommended)
HDFS_OUTDIR=${HDFS_OUTDIR:-/result}

# optional extra args passed straight to main.py
EXTRA_ARGS=${EXTRA_ARGS:-}

# --- Read each input line (one image path per line) ---
if [[ "${IGNORE_STDIN:-}" != "1" ]]; then
  while IFS= read -r img || [[ -n "$img" ]]; do
    [[ -z "$img" ]] && continue
    fname=$(basename -- "$img")

    if [[ -n "$SOURCE_DIR" && -f "$SOURCE_DIR/$fname" ]]; then
      cp "$SOURCE_DIR/$fname" "$TARGET_DIR/$fname" || echo "FAIL COPY $img" >&2
    else
      if command -v hdfs >/dev/null 2>&1 && [[ -n "$DOWNLOAD_FROM_HDFS" ]]; then
        # Attempt to fetch from HDFS path in input line
        hdfs dfs -get -f "$img" "$TARGET_DIR/$fname" 2>/dev/null || echo "FAIL GET $img" >&2
      else
        # If neither local nor HDFS is available, skip
        echo "SKIP $img" >&2
      fi
    fi

  done
fi

# --- Run CM4IR once for this shard ---
set +e
PY_ARGS=(
  --config "$CONFIG_NAME"
  --path_y "$PATH_Y"
  --deg "$DEG"
  --deg_scale "$DEG_SCALE"
  --sigma_y "$SIGMA_Y"
  -i "$TASK_ID"
  --iN "$I_N"
  --gamma "$GAMMA"
  --eta "$ETA"
  --zeta "$ZETA"
  --ni
)

if [[ -n "$DELTAS" ]]; then
  PY_ARGS+=(--deltas "$DELTAS" --deltas_injection_type "$DELTAS_INJECTION_TYPE")
fi

# Per-dataset checkpoints (caller should override for non-default experiments)
MODEL_CKPT=${MODEL_CKPT:-lsun_cat/cd_cat256_lpips.pt}
PY_ARGS+=(--model_ckpt "$MODEL_CKPT")

# Optional inpainting mask
if [[ -n "${INPAINTING_MASK_PATH:-}" ]]; then
  PY_ARGS+=(--inpainting_mask_path "$INPAINTING_MASK_PATH")
fi

# Append any free-form extras
if [[ -n "$EXTRA_ARGS" ]]; then
  # shellcheck disable=SC2206
  PY_ARGS+=( $EXTRA_ARGS )
fi

python3 "$BASE_DIR/main.py" "${PY_ARGS[@]}"
STATUS=$?
set -e

OUTDIR="$BASE_DIR/exp/image_samples/$TASK_ID"

if [[ $STATUS -ne 0 ]]; then
  echo "FAIL RUN task=$TASK_ID" >&2
  # Still attempt to upload logs if present
else
  echo "DONE $TASK_ID -> $OUTDIR"
fi

# Upload results to HDFS if available
if command -v hdfs >/dev/null 2>&1; then
  hdfs dfs -mkdir -p "$HDFS_OUTDIR/$TASK_ID" || true
  if ls "$OUTDIR"/*.png >/dev/null 2>&1; then
    hdfs dfs -put -f "$OUTDIR"/*.png "$HDFS_OUTDIR/$TASK_ID/" || true
  fi
  # also push log if exists
  if [[ -f "$OUTDIR/0_logs.log" ]]; then
    hdfs dfs -put -f "$OUTDIR/0_logs.log" "$HDFS_OUTDIR/$TASK_ID/" || true
  fi
fi

exit $STATUS
