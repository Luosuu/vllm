#!/bin/bash
# Online serving profiling for vLLM with Qwen3-1.7B using Proton.
#
# Starts a vLLM server with Proton profiling enabled, then benchmarks at
# multiple request rates to capture scheduling and kernel behavior under load.
# Uses REST API (/start_profile, /stop_profile) to control profiling window.
#
# Usage: bash scripts/profiling/comparison/vllm_online.sh [OPTIONS]
#   --model MODEL         Model name or path (default: Qwen/Qwen3-1.7B)
#   --num-prompts N       Number of prompts per rate (default: 50)
#   --input-len N         Input sequence length (default: 1024)
#   --output-len N        Max tokens to generate per prompt (default: 256)
#   --seed N              Random seed for reproducibility (default: 42)
#   --port N              Port for vLLM server (default: 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-1.7B"
NUM_PROMPTS=50
INPUT_LEN=1024
OUTPUT_LEN=256
SEED=42
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Request rates to test (req/s)
REQUEST_RATES=(1 4)

# Output directory
OUTPUT_DIR="$REPO_ROOT/profiling_output/comparison/vllm/online"
mkdir -p "$OUTPUT_DIR"

BASE_URL="http://localhost:$PORT"

echo "============================================"
echo "vLLM Online Serving Profiling (Comparison)"
echo "============================================"
echo "Model:         $MODEL"
echo "Num prompts:   $NUM_PROMPTS"
echo "Input len:     $INPUT_LEN"
echo "Output len:    $OUTPUT_LEN"
echo "Seed:          $SEED"
echo "Port:          $PORT"
echo "Request rates: ${REQUEST_RATES[*]} req/s"
echo "Output:        $OUTPUT_DIR"
echo ""

# Activate vLLM virtual environment
echo "Activating ~/vllm/.venv..."
. ~/vllm/.venv/bin/activate

# --- Start vLLM server in background ---
echo "Starting vLLM server on port $PORT..."
vllm serve "$MODEL" \
    --port "$PORT" \
    --profiler-config "{
        \"profiler\": \"proton\",
        \"proton_profiler_dir\": \"$OUTPUT_DIR\",
        \"proton_context\": \"shadow\",
        \"proton_data\": \"tree\",
        \"proton_hook\": \"triton\"
    }" &
SERVER_PID=$!

# Cleanup function to ensure server is killed on exit
cleanup() {
    echo "Shutting down vLLM server (PID $SERVER_PID)..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    echo "Server stopped"
}
trap cleanup EXIT

# --- Wait for server health ---
echo "Waiting for server to become healthy..."
for i in $(seq 1 300); do
    if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "ERROR: Server did not become healthy within 300s"
    exit 1
fi

# Common bench args for workload generation
BENCH_ARGS=(
    --backend vllm
    --model "$MODEL"
    --base-url "$BASE_URL"
    --dataset-name random
    --random-input-len "$INPUT_LEN"
    --random-output-len "$OUTPUT_LEN"
    --num-prompts "$NUM_PROMPTS"
    --seed "$SEED"
)

# --- Profile each request rate ---
for RATE in "${REQUEST_RATES[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Profiling at request rate: $RATE req/s"
    echo "--------------------------------------------"

    # Start profiling
    curl -s -X POST "$BASE_URL/start_profile"
    echo " [start_profile]"

    # Run benchmark at this request rate
    vllm bench serve "${BENCH_ARGS[@]}" \
        --request-rate "$RATE"

    # Stop profiling
    curl -s -X POST "$BASE_URL/stop_profile"
    echo " [stop_profile]"

    echo "Completed rate=$RATE req/s"
done

# --- Validate .hatchet files with proton-viewer ---
echo ""
echo "--------------------------------------------"
echo "Validating .hatchet output files"
echo "--------------------------------------------"

hatchet_count=0
hatchet_valid=0
for f in $(find "$OUTPUT_DIR" -name "*.hatchet" 2>/dev/null); do
    hatchet_count=$((hatchet_count + 1))
    if proton-viewer -m time/ns "$f" > /dev/null 2>&1; then
        echo "  [OK] $(basename "$f")"
        hatchet_valid=$((hatchet_valid + 1))
    else
        echo "  [FAIL] $(basename "$f")"
    fi
done

if [ "$hatchet_count" -eq 0 ]; then
    echo "  ERROR: No .hatchet files found in $OUTPUT_DIR"
    exit 1
fi

# --- Summary ---
echo ""
echo "============================================"
echo "OUTPUT FILE SUMMARY"
echo "============================================"
echo "  Directory: $OUTPUT_DIR"
file_count=$(find "$OUTPUT_DIR" -type f 2>/dev/null | wc -l)
echo "  Total files: $file_count"
find "$OUTPUT_DIR" -type f 2>/dev/null | sort | while read -r f; do
    size=$(stat --printf="%s" "$f" 2>/dev/null || echo "?")
    echo "    - $(basename "$f") ($size bytes)"
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
