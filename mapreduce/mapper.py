#!/usr/bin/env python3
import sys
import re

# Word Count mapper: reads lines from stdin, emits "word\t1" per token
# - Normalizes to lowercase
# - Strips punctuation using a regex
# - Skips empty tokens

def tokenize(text: str):
    # Replace non-word characters with space, keep unicode letters/digits/underscore
    # Then split on whitespace
    text = text.lower()
    # Use unicode-aware tokenization: split on any char that isn't a letter/number/underscore
    tokens = re.split(r"\W+", text, flags=re.UNICODE)
    return [t for t in tokens if t]

for line in sys.stdin:
    for token in tokenize(line):
        # Hadoop Streaming expects key\tvalue per line
        sys.stdout.write(f"{token}\t1\n")
