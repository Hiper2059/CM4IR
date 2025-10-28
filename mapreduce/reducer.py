#!/usr/bin/env python3
import sys

# Word Count reducer: sums counts per word.
# Input must be sorted by key (word). Hadoop Streaming guarantees this between mapper and reducer.

current_word = None
current_count = 0

for line in sys.stdin:
    line = line.rstrip("\n")
    if not line:
        continue
    try:
        word, count_str = line.split("\t", 1)
        count = int(count_str)
    except ValueError:
        # Skip malformed lines
        continue

    if current_word is None:
        current_word = word
        current_count = count
        continue

    if word == current_word:
        current_count += count
    else:
        sys.stdout.write(f"{current_word}\t{current_count}\n")
        current_word = word
        current_count = count

# Flush last word
if current_word is not None:
    sys.stdout.write(f"{current_word}\t{current_count}\n")
