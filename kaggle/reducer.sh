#!/usr/bin/env bash
set -euo pipefail

# =====================
# CM4IR Kaggle Reducer
# - Designed to run after one or more mapper runs finished
# - Modes:
#   * identity: passthrough stdin -> stdout
#   * summary:  summarise exp/image_samples/* (default)
#   * run:      execute a single CM4IR run (per user's recipe)
# =====================

MODE=${MODE:-summary}
BASE_DIR=${BASE_DIR:-/kaggle/working}
if [[ ! -d "$BASE_DIR" ]]; then BASE_DIR=${BASE_DIR:-/workspace}; fi
if [[ ! -d "$BASE_DIR" ]]; then BASE_DIR=$(pwd); fi

if [[ "$MODE" == "identity" ]]; then
  cat
  exit 0
fi

# --- Run mode: execute the provided Kaggle-style command ---
if [[ "$MODE" == "run" ]]; then
  # Discover repo root that contains main.py
  REPO_DIR=${REPO_DIR:-$BASE_DIR}
  if [[ ! -f "$REPO_DIR/main.py" ]]; then
    if [[ -f "$BASE_DIR/CM4IR/main.py" ]]; then REPO_DIR="$BASE_DIR/CM4IR";
    elif [[ -f "$BASE_DIR/cm4ir/main.py" ]]; then REPO_DIR="$BASE_DIR/cm4ir";
    else REPO_DIR=$(pwd); fi
  fi

  # Map user's exact recipe to configurable knobs with sane defaults
  CM4IR_CONFIG=${CM4IR_CONFIG:-lsun_bedroom_256.yml}
  # Resolve absolute config path or fall back to $REPO_DIR/configs/<name>
  if [[ "$CM4IR_CONFIG" = /* ]]; then
    CONFIG_PATH="$CM4IR_CONFIG"
  else
    CONFIG_PATH="$REPO_DIR/configs/$CM4IR_CONFIG"
  fi
  EXP_DIR=${EXP_DIR:-"$REPO_DIR/exp"}
  PATH_Y=${PATH_Y:-lsun_bedroom}
  DEG=${DEG:-inpainting}
  SIGMA_Y=${SIGMA_Y:-0.025}
  IMAGE_FOLDER=${IMAGE_FOLDER:-CM4IR_lsun_bedroom_inpainting_random_80_sigma_y_0.025}
  I_N=${I_N:-150}
  GAMMA=${GAMMA:-0.2}
  INPAINTING_MASK_PATH=${INPAINTING_MASK_PATH:-random_80_mask.npy}
  MODEL_CKPT=${MODEL_CKPT:-lsun_bedroom/cd_bedroom256_lpips.pt}
  DELTAS=${DELTAS:-}
  DELTAS_INJECTION_TYPE=${DELTAS_INJECTION_TYPE:-1}

  set +e
  PY_ARGS=(
    --config "$CONFIG_PATH"
    --exp "$EXP_DIR"
    --path_y "$PATH_Y"
    --deg "$DEG"
    --sigma_y "$SIGMA_Y"
    -i "$IMAGE_FOLDER"
    --iN "$I_N"
    --gamma "$GAMMA"
    --inpainting_mask_path "$INPAINTING_MASK_PATH"
    --model_ckpt "$MODEL_CKPT"
    --ni
  )
  if [[ -n "$DELTAS" ]]; then
    PY_ARGS+=(--deltas "$DELTAS" --deltas_injection_type "$DELTAS_INJECTION_TYPE")
  fi
  python3 "$REPO_DIR/main.py" "${PY_ARGS[@]}"
  STATUS=$?
  set -e
  exit $STATUS
fi

# --- Summary mode (default) ---
SAMPLES_ROOT="$BASE_DIR/exp/image_samples"
if [[ ! -d "$SAMPLES_ROOT" ]]; then
  echo "SUMMARY\tDONE=0\tIMAGES=0" 1>&2
  exit 0
fi

DONE_COUNT=0
IMAGE_TOTAL=0
FAIL_COUNT=0

for dir in "$SAMPLES_ROOT"/*; do
  [[ ! -d "$dir" ]] && continue
  # Count PNGs
  cnt=$(ls "$dir"/*.png 2>/dev/null | wc -l | awk '{print ($1==""?0:$1)}')
  if [[ "$cnt" -gt 0 ]]; then
    ((DONE_COUNT++))
    ((IMAGE_TOTAL+=cnt))
  fi
  # Look for error markers in logs if present
  if [[ -f "$dir/0_logs.log" ]]; then
    if grep -qiE "(error|exception|traceback)" "$dir/0_logs.log"; then
      ((FAIL_COUNT++))
    fi
  fi
  echo "DIR $dir\tIMAGES=$cnt"

done

echo "SUMMARY\tDONE=$DONE_COUNT\tIMAGES=$IMAGE_TOTAL\tFAIL=$FAIL_COUNT" 1>&2
