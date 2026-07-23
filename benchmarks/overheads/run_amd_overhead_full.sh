#!/bin/bash
# Paper-grade AMD profiler-overhead run: 2048 prompts, 3 paired repeats,
# high concurrency. Thin wrapper over run_amd_overhead_smoke.sh — same
# 8-row table printed to stdout and saved to $OUTPUT_ROOT/table.md.
#
# Run from the vLLM repo root inside the ROCm container. Expected runtime
# is several hours on one MI300X: 2 modes x (1 baseline + 3 profilers)
# x 3 repeats = 24 fresh server launches, each serving 512 warmup + 2048
# benchmark requests. Ensure enough disk for torch traces (GB-scale per
# repeat at this workload).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export NUM_PROMPTS="${NUM_PROMPTS:-2048}"
export NUM_WARMUPS="${NUM_WARMUPS:-512}"
export MAX_CONCURRENCY="${MAX_CONCURRENCY:-256}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export REPEATS="${REPEATS:-3}"
export LENGTH_PAIR="${LENGTH_PAIR:-2000:500}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-amd-overhead-full}"
export TABLE_FILE="${TABLE_FILE:-table_full.md}"

exec bash "$SCRIPT_DIR/run_amd_overhead_smoke.sh"
