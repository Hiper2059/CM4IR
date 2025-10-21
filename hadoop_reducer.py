#!/usr/bin/env python3
import sys

# Reducer aggregates averages for each metric key
# Input format: key\tvalue (sorted by Hadoop Streaming before reducer)

current_key = None
count = 0
sum_val = 0.0

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    try:
        key, value = line.split('\t', 1)
        value = float(value)
    except Exception:
        continue

    if current_key is None:
        current_key = key
        sum_val = value
        count = 1
    elif key == current_key:
        sum_val += value
        count += 1
    else:
        avg = sum_val / count if count else 0.0
        print(f"{current_key}\t{avg}")
        current_key = key
        sum_val = value
        count = 1

# Flush last key
if current_key is not None:
    avg = sum_val / count if count else 0.0
    print(f"{current_key}\t{avg}")
