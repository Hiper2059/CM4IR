#!/usr/bin/env python3
"""
Hadoop Streaming mapper: emits "<word>\t1" per token from stdin.
- Tokenization: case-insensitive, word characters (\w), Unicode-friendly
- Output: one line per token, tab-separated key/value suitable for reducers
- Usage in Hadoop Streaming: -mapper /path/to/my_mapper.py
"""
from __future__ import annotations

import re
import sys
from typing import Iterable, TextIO

# Treat consecutive word characters as tokens; ignore punctuation and symbols
WORD_REGEX = re.compile(r"\w+", flags=re.UNICODE)


def tokenize_to_words(text: str) -> Iterable[str]:
    for match in WORD_REGEX.finditer(text.lower()):
        yield match.group(0)


def process_stream(input_stream: TextIO, output_stream: TextIO) -> None:
    for raw_line in input_stream:
        # raw_line is already str under Python3's text stdin; keep simple and streaming-safe
        for token in tokenize_to_words(raw_line):
            output_stream.write(f"{token}\t1\n")


if __name__ == "__main__":
    # Stream from stdin to stdout; avoid buffering at scale by writing per token
    process_stream(sys.stdin, sys.stdout)
