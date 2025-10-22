#!/usr/bin/env bash
set -euo pipefail

# =====================
# CM4IR Hadoop Reducer (identity/merge)
# - Option A (default): identity reducer, simply forward stdin -> stdout
# - Option B: summarise per-task results (counts) and emit a summary
# Choose via REDUCER_MODE: identity | summary
# =====================

MODE=${REDUCER_MODE:-identity}

if [[ "$MODE" == "identity" ]]; then
  # passthrough
  cat
  exit 0
fi

# summary mode: count lines per task id from mapper outputs
# Expect each mapper to optionally emit lines like:
#   DONE <task_id> -> <outdir>
#   FAIL RUN task=<task_id>

DONE_COUNT=0
FAIL_COUNT=0

while IFS= read -r line || [[ -n "$line" ]]; do
  if [[ "$line" == DONE* ]]; then
    ((DONE_COUNT++))
  elif [[ "$line" == FAIL* ]]; then
    ((FAIL_COUNT++))
  fi
  # forward the line as well
  echo "$line"

done

echo "SUMMARY\tDONE=$DONE_COUNT\tFAIL=$FAIL_COUNT" 1>&2
