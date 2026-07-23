#!/bin/bash
# Run the profiler-overhead smoke tests on AMD (proton, torch, rocprofv3)
# in both eager and cudagraph modes, then print the combined 8-row
# markdown table (baseline/proton/torch/rocprof x eager/cudagraph).
#
# Run from the vLLM repo root (profiler-overhead-benchmarks branch) inside
# a ROCm container that has the branch's python changes installed.
set -euo pipefail

# ---- Configuration (override via environment) -------------------------------
MODEL="${MODEL:-Qwen/Qwen3-8B}"          # ungated; no HF token needed
LENGTH_PAIR="${LENGTH_PAIR:-500:100}"    # input:output lengths
NUM_PROMPTS="${NUM_PROMPTS:-16}"         # paper-grade: 2048
NUM_WARMUPS="${NUM_WARMUPS:-4}"          # paper-grade: 512
MAX_CONCURRENCY="${MAX_CONCURRENCY:-8}"  # paper-grade: 256
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"        # paper-grade: 256
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"  # paper-grade: 8192
REPEATS="${REPEATS:-1}"                  # paper-grade: 3
TP_SIZE="${TP_SIZE:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-amd-overhead-results}"

# ---- AMD environment requirements -------------------------------------------
# Proton on AMD requires ROCR_VISIBLE_DEVICES and rejects HIP/CUDA_VISIBLE_DEVICES.
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
unset HIP_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES

command -v rocprofv3 >/dev/null || {
  echo "rocprofv3 is required on PATH (part of ROCm)" >&2
  exit 1
}
# The harness resolves the vllm CLI next to the python executable.
PYBIN="$(dirname "$(command -v python3)")"
[ -x "$PYBIN/vllm" ] || ln -sf "$(command -v vllm)" "$PYBIN/vllm"

# ---- Run: one invocation per graph mode; the harness pairs each profiler
# ---- with a same-seed baseline run automatically. ----------------------------
for mode in eager cudagraph; do
  [ "$mode" = "cudagraph" ] && graph_flag="--cudagraph" || graph_flag="--no-cudagraph"
  python3 benchmarks/overheads/benchmark_serving_profiler.py \
    --model "$MODEL" \
    --length-pairs "$LENGTH_PAIR" \
    --profilers proton torch rocprof \
    --num-prompts "$NUM_PROMPTS" \
    --num-warmups "$NUM_WARMUPS" \
    --max-concurrency "$MAX_CONCURRENCY" \
    --tensor-parallel-size "$TP_SIZE" \
    --data-parallel-size 1 \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --repeats "$REPEATS" \
    "$graph_flag" \
    --server-shutdown-timeout 300 \
    --output-dir "$OUTPUT_ROOT/$mode"
done

# ---- Generate the combined markdown table ------------------------------------
python3 - "$OUTPUT_ROOT" <<'EOF' | tee "$OUTPUT_ROOT/table.md"
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
order = ("proton", "torch", "rocprof")

print(
    "| Mode | Profiler | Output tput (tok/s) | Tput overhead "
    "| Mean TTFT (ms) | TTFT overhead | Mean TPOT (ms) | TPOT overhead "
    "| Trace size |"
)
print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for mode in ("eager", "cudagraph"):
    data = json.loads((root / mode / "results.json").read_text())
    summaries = [s for s in data["summary"] if s["profiler"] != "none"]
    first = summaries[0]
    print(
        f"| {mode} | baseline "
        f"| {first['baseline_output_throughput']:.1f} | — "
        f"| {first['baseline_mean_ttft_ms']:.1f} | — "
        f"| {first['baseline_mean_tpot_ms']:.2f} | — | — |"
    )
    ranked = sorted(
        summaries,
        key=lambda s: next(
            (i for i, name in enumerate(order) if s["profiler"].startswith(name)),
            len(order),
        ),
    )
    for s in ranked:
        print(
            f"| {mode} | {s['profiler']} "
            f"| {s['profiled_output_throughput']:.1f} "
            f"| {s['output_throughput_overhead_pct']:+.1f}% "
            f"| {s['profiled_mean_ttft_ms']:.1f} "
            f"| {s['mean_ttft_ms_overhead_pct']:+.1f}% "
            f"| {s['profiled_mean_tpot_ms']:.2f} "
            f"| {s['mean_tpot_ms_overhead_pct']:+.1f}% "
            f"| {s['profile_bytes'] / 1e6:.2f} MB |"
        )
EOF
