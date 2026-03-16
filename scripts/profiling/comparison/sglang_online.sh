#!/bin/bash
# Online serving profiling for SGLang with Qwen3-1.7B using Proton.
#
# Starts an SGLang server with --disable-cuda-graph (required so Proton can
# subscribe to CUPTI without conflict). For each request rate: activates Proton
# profiling via sglang.profiler (--no-cpu --no-gpu --proton), runs bench_serving
# concurrently, and collects .hatchet output files.
#
# Usage: bash scripts/profiling/comparison/sglang_online.sh [OPTIONS]
#   --model MODEL         Model name or path (default: Qwen/Qwen3-1.7B)
#   --num-prompts N       Number of prompts per rate (default: 50)
#   --input-len N         Input sequence length (default: 1024)
#   --output-len N        Max tokens to generate per prompt (default: 256)
#   --seed N              Random seed for reproducibility (default: 42)
#   --port N              Port for SGLang server (default: 30000)
#   --num-steps N         Number of forward steps to profile per rate (default: 100)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-1.7B"
NUM_PROMPTS=50
INPUT_LEN=1024
OUTPUT_LEN=256
SEED=42
PORT=30000
NUM_STEPS=100

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --num-steps) NUM_STEPS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Request rates to test (req/s)
REQUEST_RATES=(1 4)

# Output directory
OUTPUT_DIR="$REPO_ROOT/profiling_output/comparison/sglang/online"
mkdir -p "$OUTPUT_DIR"

BASE_URL="http://localhost:$PORT"

echo "============================================"
echo "SGLang Online Serving Profiling (Comparison)"
echo "============================================"
echo "Model:         $MODEL"
echo "Num prompts:   $NUM_PROMPTS"
echo "Input len:     $INPUT_LEN"
echo "Output len:    $OUTPUT_LEN"
echo "Seed:          $SEED"
echo "Port:          $PORT"
echo "Num steps:     $NUM_STEPS"
echo "Request rates: ${REQUEST_RATES[*]} req/s"
echo "Output:        $OUTPUT_DIR"
echo ""

# Activate SGLang virtual environment
echo "Activating ~/sglang/.venv..."
. ~/sglang/.venv/bin/activate

# --- Start SGLang server in background ---
echo "Starting SGLang server on port $PORT..."
# --disable-cuda-graph prevents CUDA graph capture which holds a CUPTI
# subscription and blocks Proton from subscribing (CUPTI error 39).
python -m sglang.launch_server \
    --model-path "$MODEL" \
    --port "$PORT" \
    --disable-cuda-graph &
SERVER_PID=$!

# Cleanup function to ensure server is killed on exit
cleanup() {
    echo "Shutting down SGLang server (PID $SERVER_PID)..."
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

# --- Profile each request rate ---
for RATE in "${REQUEST_RATES[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo "Profiling at request rate: $RATE req/s"
    echo "--------------------------------------------"

    RATE_OUTPUT_DIR="$OUTPUT_DIR/rate_${RATE}"
    mkdir -p "$RATE_OUTPUT_DIR"

    # Start profiling in background (blocking call that returns after num_steps).
    # --no-cpu --no-gpu disables PyTorch profiler to avoid CUPTI conflict with Proton.
    python -m sglang.profiler \
        --url "$BASE_URL" \
        --output-dir "$RATE_OUTPUT_DIR" \
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
    echo "    - ${f#$OUTPUT_DIR/} ($size bytes)"
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
