#!/bin/bash
# Fair online serving profiling for vLLM with Qwen3-1.7B using Proton (v2).
#
# Key fairness measures vs v1:
#   - CUDA graphs ENABLED (real-world config, no --enforce-eager)
#   - Proton-only profiling (no torch.profiler / PyTorch GPU profiler)
#   - Per-rate sessions: server restarted per rate to guarantee separate .hatchet files
#   - 10-prompt warmup at inf rate before each profiled rate (not profiled)
#   - Identical workload: 100 prompts, input_len=1024, output_len=256, seed=42
#   - Higher load range: 4, 16, 32, inf req/s
#
# Usage: bash scripts/profiling/comparison/vllm_online_v2.sh [OPTIONS]
#   --model MODEL           Model name or path (default: Qwen/Qwen3-1.7B)
#   --num-prompts N         Number of prompts per rate (default: 100)
#   --input-len N           Input sequence length (default: 1024)
#   --output-len N          Max tokens to generate per prompt (default: 256)
#   --seed N                Random seed for reproducibility (default: 42)
#   --port N                Port for vLLM server (default: 8000)
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
PORT=8000
REQUEST_RATES_STR="4 16 32 inf"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --request-rates) REQUEST_RATES_STR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Convert space-separated string to array
read -ra REQUEST_RATES <<< "$REQUEST_RATES_STR"

# Output directories
OUTPUT_BASE="$REPO_ROOT/profiling_output/comparison_v2/vllm/online"
BASE_URL="http://localhost:$PORT"

# Number of warmup prompts (sent at inf rate, not profiled)
WARMUP_PROMPTS=10

echo "============================================"
echo "vLLM Online Serving Profiling v2 (Fair)"
echo "============================================"
echo "Model:          $MODEL"
echo "Num prompts:    $NUM_PROMPTS"
echo "Input len:      $INPUT_LEN"
echo "Output len:     $OUTPUT_LEN"
echo "Seed:           $SEED"
echo "Port:           $PORT"
echo "Request rates:  ${REQUEST_RATES[*]} req/s"
echo "Warmup prompts: $WARMUP_PROMPTS (at inf rate, not profiled)"
echo "Output base:    $OUTPUT_BASE"
echo ""

# Activate vLLM virtual environment
echo "Activating ~/vllm/.venv..."
. ~/vllm/.venv/bin/activate

# Track server PID for cleanup (set per rate iteration)
SERVER_PID=""

# Cleanup function to ensure server is killed on exit
cleanup() {
    if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Shutting down vLLM server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        echo "Server stopped"
    fi
}
trap cleanup EXIT

# start_server: launches vLLM server with Proton profiling, output to given dir.
# Args: $1 = proton output directory
start_server() {
    local proton_dir="$1"
    mkdir -p "$proton_dir"

    echo "Starting vLLM server on port $PORT (proton dir: $proton_dir)..."
    vllm serve "$MODEL" \
        --port "$PORT" \
        --profiler-config "{
            \"profiler\": \"proton\",
            \"proton_profiler_dir\": \"$proton_dir\",
            \"proton_context\": \"shadow\",
            \"proton_data\": \"tree\",
            \"proton_hook\": \"triton\"
        }" &
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
        echo "Stopping vLLM server (PID $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        echo "Server stopped"
        # Brief pause to release port
        sleep 2
    fi
}

# Common bench args (rate-independent)
BENCH_ARGS=(
    --backend vllm
    --model "$MODEL"
    --base-url "$BASE_URL"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --seed "$SEED"
)

# --- Profile each request rate in a separate server session ---
for RATE in "${REQUEST_RATES[@]}"; do
    echo ""
    echo "============================================"
    echo "Rate: $RATE req/s"
    echo "============================================"

    # Per-rate output directory
    RATE_DIR="$OUTPUT_BASE/rate_${RATE}"

    # Start fresh server for this rate
    stop_server
    start_server "$RATE_DIR"
    wait_for_server

    # Warmup: send prompts at inf rate (not profiled)
    echo "Sending $WARMUP_PROMPTS warmup prompts at inf rate..."
    vllm bench serve "${BENCH_ARGS[@]}" \
        --num-prompts "$WARMUP_PROMPTS" \
        --request-rate inf

    # Start profiling
    curl -s -X POST "$BASE_URL/start_profile"
    echo " [start_profile]"

    # Run benchmark at this request rate
    echo "Benchmarking with $NUM_PROMPTS prompts at rate=$RATE..."
    vllm bench serve "${BENCH_ARGS[@]}" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$RATE"

    # Stop profiling
    curl -s -X POST "$BASE_URL/stop_profile"
    echo " [stop_profile]"

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
