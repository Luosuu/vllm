#!/bin/bash
# Fair online serving profiling for SGLang with Qwen3-1.7B using Proton (v2).
#
# Key fairness measures vs v1:
#   - CUDA graphs ENABLED (real-world config, no --disable-cuda-graph)
#   - SGLANG_PROTON_CUDA_GRAPH=1 creates persistent Proton session before graph
#     capture so Proton owns CUPTI and graphs capture fine
#   - Proton-only profiling (--no-cpu --no-gpu --proton: no torch.profiler)
#   - Per-rate sessions: server restarted per rate to guarantee separate .hatchet files
#   - 10-prompt warmup at inf rate before each profiled rate (not profiled)
#   - Identical workload: 100 prompts, input_len=1024, output_len=256, seed=42
#   - Higher load range: 4, 16, 32, inf req/s
#   - --num-steps 30000 to avoid early cutoff before benchmark finishes
#
# Usage: bash scripts/profiling/comparison/sglang_online_v2.sh [OPTIONS]
#   --model MODEL           Model name or path (default: Qwen/Qwen3-1.7B)
#   --num-prompts N         Number of prompts per rate (default: 100)
#   --input-len N           Input sequence length (default: 1024)
#   --output-len N          Max tokens to generate per prompt (default: 256)
#   --seed N                Random seed for reproducibility (default: 42)
#   --port N                Port for SGLang server (default: 30000)
#   --num-steps N           Forward steps to profile per rate (default: 30000)
#   --request-rates RATES   Space-separated rates in quotes (default: '4 16 32 inf')

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-1.7B"
NUM_PROMPTS=100
INPUT_LEN=1024
OUTPUT_LEN=256
SEED=42
PORT=30000
NUM_STEPS=30000
REQUEST_RATES_STR="4 16 32 inf"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        --request-rates) REQUEST_RATES_STR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Convert space-separated string to array
read -ra REQUEST_RATES <<< "$REQUEST_RATES_STR"

# Output directories
OUTPUT_BASE="$REPO_ROOT/profiling_output/comparison_v2/sglang/online"
BASE_URL="http://localhost:$PORT"

# Number of warmup prompts (sent at inf rate, not profiled)
WARMUP_PROMPTS=10

echo "============================================"
echo "SGLang Online Serving Profiling v2 (Fair)"
echo "============================================"
echo "Model:          $MODEL"
echo "Num prompts:    $NUM_PROMPTS"
echo "Input len:      $INPUT_LEN"
echo "Output len:     $OUTPUT_LEN"
echo "Seed:           $SEED"
echo "Port:           $PORT"
echo "Num steps:      $NUM_STEPS"
echo "Request rates:  ${REQUEST_RATES[*]} req/s"
echo "Warmup prompts: $WARMUP_PROMPTS (at inf rate, not profiled)"
echo "Output base:    $OUTPUT_BASE"
echo ""

# Activate SGLang virtual environment
echo "Activating ~/sglang/.venv..."
. ~/sglang/.venv/bin/activate

# Track server PID for cleanup (set per rate iteration)
SERVER_PID=""

# Cleanup function to ensure server is killed on exit
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Shutting down SGLang server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped"
    fi
}
trap cleanup EXIT

# start_server: launches SGLang server with CUDA graphs and persistent Proton session.
# SGLANG_PROTON_CUDA_GRAPH=1 ensures Proton owns CUPTI before graph capture.
start_server() {
    echo "Starting SGLang server on port $PORT (CUDA graphs + Proton)..."
    SGLANG_PROTON_CUDA_GRAPH=1 python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$PORT" &
    SERVER_PID=$!
}

# wait_for_server: polls /health until server is ready or times out.
wait_for_server() {
    echo "Waiting for server to become healthy..."
    for i in $(seq 1 300); do
        if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
            echo "Server ready after ${i}s"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: Server process died"
            exit 1
        fi
        sleep 1
    done
    echo "ERROR: Server did not become healthy within 300s"
    exit 1
}

# stop_server: gracefully kills the current server process.
stop_server() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Stopping SGLang server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        echo "Server stopped"
        # Brief pause to release port
        sleep 2
    fi
}

# --- Profile each request rate in a separate server session ---
for RATE in "${REQUEST_RATES[@]}"; do
    echo ""
    echo "============================================"
    echo "Rate: $RATE req/s"
    echo "============================================"

    # Per-rate output directory
    RATE_DIR="$OUTPUT_BASE/rate_${RATE}"
    mkdir -p "$RATE_DIR"

    # Start fresh server for this rate
    stop_server
    start_server
    wait_for_server

    # Warmup: send prompts at inf rate (not profiled)
    echo "Sending $WARMUP_PROMPTS warmup prompts at inf rate..."
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --dataset-name random \
        --random-input "$INPUT_LEN" \
        --random-output "$OUTPUT_LEN" \
        --num-prompts "$WARMUP_PROMPTS" \
        --seed "$SEED" \
        --request-rate inf

    # Start profiler in background (blocking call that returns after num_steps).
    # --no-cpu --no-gpu ensures ONLY Proton subscribes to CUPTI (no torch.profiler).
    echo "Starting Proton profiler ($NUM_STEPS steps)..."
    python -m sglang.profiler \
        --url "$BASE_URL" \
        --output-dir "$RATE_DIR" \
        --num-steps "$NUM_STEPS" \
        --no-cpu --no-gpu \
        --proton \
        --proton-context shadow \
        --proton-data tree \
        --proton-hook triton \
        --profile-prefix "rate${RATE}_" &
    PROFILER_PID=$!

    # Brief pause to let profiler attach before sending requests
    sleep 2

    # Run benchmark at this request rate
    echo "Benchmarking with $NUM_PROMPTS prompts at rate=$RATE..."
    python -m sglang.bench_serving \
        --backend sglang \
        --port "$PORT" \
        --dataset-name random \
        --random-input "$INPUT_LEN" \
        --random-output "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --seed "$SEED" \
        --request-rate "$RATE"

    # Wait for profiler to finish collecting data
    echo "Waiting for profiler to complete..."
    wait "$PROFILER_PID"

    echo "Completed rate=$RATE req/s"
done

# Stop server after last rate
stop_server

# --- Validate .hatchet files with proton-viewer ---
echo ""
echo "============================================"
echo "Validating .hatchet output files"
echo "============================================"

hatchet_count=0
hatchet_valid=0
for RATE in "${REQUEST_RATES[@]}"; do
    RATE_DIR="$OUTPUT_BASE/rate_${RATE}"
    for f in $(find "$RATE_DIR" -name "*.hatchet" 2>/dev/null); do
        hatchet_count=$((hatchet_count + 1))
        if proton-viewer -m time/ns "$f" > /dev/null 2>&1; then
            echo "  [OK] rate_${RATE}/$(basename "$f")"
            hatchet_valid=$((hatchet_valid + 1))
        else
            echo "  [FAIL] rate_${RATE}/$(basename "$f")"
        fi
    done
done

if [ "$hatchet_count" -eq 0 ]; then
    echo "  ERROR: No .hatchet files found in $OUTPUT_BASE"
    exit 1
fi

# --- Summary ---
echo ""
echo "============================================"
echo "OUTPUT FILE SUMMARY"
echo "============================================"
echo "  Base directory: $OUTPUT_BASE"
for RATE in "${REQUEST_RATES[@]}"; do
    RATE_DIR="$OUTPUT_BASE/rate_${RATE}"
    file_count=$(find "$RATE_DIR" -type f 2>/dev/null | wc -l)
    echo "  rate_${RATE}/: $file_count files"
    find "$RATE_DIR" -type f 2>/dev/null | sort | while read -r f; do
        size=$(stat --printf="%s" "$f" 2>/dev/null || echo "?")
        echo "    - $(basename "$f") ($size bytes)"
    done
done
echo ""
echo "  .hatchet files: $hatchet_count found, $hatchet_valid valid"
echo "============================================"

if [ "$hatchet_valid" -eq "$hatchet_count" ]; then
    echo "All .hatchet files validated successfully"
    exit 0
else
    echo "Some .hatchet files failed validation"
    exit 1
fi
