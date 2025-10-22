#!/bin/bash
set -euo pipefail

# Identity reducer for Hadoop Streaming; also can aggregate status.
# Accepts lines from mapper: PREFIX \t TASK_ID \t FILENAME \t INFO
# Sort is guaranteed by Hadoop; we simply re-emit or tally minimal stats.

EMIT_ONLY=${EMIT_ONLY:-1}

if [[ "$EMIT_ONLY" -eq 1 ]]; then
  cat
  exit 0
fi

ok=0
fail=0
while IFS=$'\t' read -r status task_id fname info; do
  case "$status" in
    OK) ((ok++));;
    FAIL) ((fail++));;
  esac
  echo -e "$status\t$task_id\t$fname\t$info"
done

echo -e "SUMMARY\tok=${ok}\tfail=${fail}" 1>&2
