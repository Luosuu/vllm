#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache configuration and tokenizer files needed by dummy-weight benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

OVERHEAD_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OVERHEAD_DIR))

from run_profiler_matrix import DEFAULT_MODELS, MODELS  # noqa: E402

ALLOW_PATTERNS = (
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
    "vocab.txt",
)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=DEFAULT_MODELS)
    args, _ = parser.parse_known_args()
    for name in args.models:
        model_id, _ = MODELS[name]
        print(f"Caching metadata for {model_id}", flush=True)
        snapshot_download(repo_id=model_id, allow_patterns=ALLOW_PATTERNS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
