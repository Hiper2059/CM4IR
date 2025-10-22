#!/usr/bin/env bash
set -euo pipefail

# =====================
# CM4IR Kaggle Mapper
# - No HDFS dependency; works inside Kaggle notebooks/TPUs/GPUs
# - Stages a shard of images into exp/datasets/<PATH_Y>/<class_subdir>
# - Invokes main.py once to process the shard
# - Writes PNG results and logs into exp/image_samples/<TASK_ID>
# =====================

# --- GPU assignment (optional) ---
N_GPUS_DEFAULT=1
if command -v nvidia-smi >/dev/null 2>&1; then
  N_GPUS_DEFAULT=$(nvidia-smi -L 2>/dev/null | wc -l | awk '{print ($1==""?0:$1)}')
  if [[ "$N_GPUS_DEFAULT" -eq 0 ]]; then N_GPUS_DEFAULT=1; fi
fi
GPU_COUNT=${N_GPUS:-$N_GPUS_DEFAULT}
TASK_ID=${KAGGLE_TASK_ID:-${TASK_ID:-0}}
GPU_INDEX=$(( TASK_ID % GPU_COUNT ))
export CUDA_VISIBLE_DEVICES=$GPU_INDEX

# --- Base paths ---
# Prefer Kaggle working dir, fall back to /workspace or current dir
BASE_DIR=${BASE_DIR:-/kaggle/working}
if [[ ! -d "$BASE_DIR" ]]; then BASE_DIR=${BASE_DIR:-/workspace}; fi
if [[ ! -d "$BASE_DIR" ]]; then BASE_DIR=$(pwd); fi

# Config/model and degradation knobs (override via env)
CM4IR_CONFIG=${CM4IR_CONFIG:-lsun_cat_256.yml}
CONFIG_NAME=$(basename "$CM4IR_CONFIG")
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
# Determine class subdir used by torchvision.datasets.ImageFolder for LSUN variants
if [[ -z "${CLASS_SUBDIR:-}" ]]; then
  if [[ "$PATH_Y" == *cat* ]]; then CLASS_SUBDIR=cat;
  elif [[ "$PATH_Y" == *bedroom* ]]; then CLASS_SUBDIR=bedroom;
  else CLASS_SUBDIR=""; fi
fi

# Target dir where main.py expects the staged images
if [[ -n "$CLASS_SUBDIR" ]]; then
  TARGET_DIR="$BASE_DIR/exp/datasets/$PATH_Y/$CLASS_SUBDIR"
else
  if [[ "$PATH_Y" == "imagenet" ]]; then
    TARGET_DIR="$BASE_DIR/exp/datasets/imagenet/imagenet"
  else
    TARGET_DIR=${DATASET_DIR:-"$BASE_DIR/exp/datasets/$PATH_Y"}
  fi
fi
mkdir -p "$TARGET_DIR"

# Source configuration
# - SOURCE_DIR: directory with input images (e.g., /kaggle/input/my-images)
# - INPUT_LIST: optional text file with 1 path per line
#   When set, paths can be absolute, or basenames looked up under SOURCE_DIR
SOURCE_DIR=${SOURCE_DIR:-}
INPUT_LIST=${INPUT_LIST:-}
MAX_IMAGES=${MAX_IMAGES:-}

# Clean previous shard contents
rm -rf "${TARGET_DIR}"/* || true

copy_one() {
  local src="$1"
  local base
  base=$(basename -- "$src")
  if [[ -f "$src" ]]; then
    cp -f "$src" "$TARGET_DIR/$base"
  elif [[ -n "$SOURCE_DIR" && -f "$SOURCE_DIR/$base" ]]; then
    cp -f "$SOURCE_DIR/$base" "$TARGET_DIR/$base"
  else
    echo "WARN: missing $src" >&2
    return 1
  fi
}

# Populate shard
count=0
shopt -s nullglob nocaseglob
if [[ -n "$INPUT_LIST" && -f "$INPUT_LIST" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    [[ -z "$line" ]] && continue
    copy_one "$line" || true
    ((count++))
    if [[ -n "$MAX_IMAGES" && "$count" -ge "$MAX_IMAGES" ]]; then break; fi
  done < "$INPUT_LIST"
else
  # Fall back to taking all images from SOURCE_DIR
  if [[ -z "$SOURCE_DIR" ]]; then
    echo "ERROR: Either INPUT_LIST or SOURCE_DIR must be provided" >&2
    exit 2
  fi
  for ext in jpg jpeg png JPG JPEG PNG; do
    for f in "$SOURCE_DIR"/*.$ext; do
      [[ ! -e "$f" ]] && continue
      cp -f "$f" "$TARGET_DIR/" || true
      ((count++))
      if [[ -n "$MAX_IMAGES" && "$count" -ge "$MAX_IMAGES" ]]; then break 2; fi
    done
  done
fi
shopt -u nullglob nocaseglob

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

# Model checkpoint path; users typically mount a Kaggle dataset with checkpoints
MODEL_CKPT=${MODEL_CKPT:-lsun_cat/cd_cat256_lpips.pt}
PY_ARGS+=(--model_ckpt "$MODEL_CKPT")

# Optional inpainting mask
if [[ -n "${INPAINTING_MASK_PATH:-}" ]]; then
  PY_ARGS+=(--inpainting_mask_path "$INPAINTING_MASK_PATH")
fi

# Free-form extra args
if [[ -n "${EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  PY_ARGS+=( $EXTRA_ARGS )
fi

python3 "$BASE_DIR/main.py" "${PY_ARGS[@]}"
STATUS=$?
set -e

OUTDIR="$BASE_DIR/exp/image_samples/$TASK_ID"
if [[ $STATUS -ne 0 ]]; then
  echo "FAIL RUN task=$TASK_ID" >&2
else
  echo "DONE $TASK_ID -> $OUTDIR"
fi

# Optionally zip results for easy Kaggle download
if [[ "${ZIP_RESULTS:-0}" == "1" && -d "$OUTDIR" ]]; then
  (cd "$OUTDIR" && zip -q -r "$BASE_DIR/cm4ir_outputs_task_${TASK_ID}.zip" .) || true
fi

exit $STATUS
