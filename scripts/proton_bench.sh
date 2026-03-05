#!/bin/bash
# Proton contiguous profiling: launch vllm serve then run vllm bench serve
# with periodic flushing enabled. Each bench run produces a .part_N output file.
#
# Usage:
#   ./scripts/proton_bench.sh [NUM_CYCLES] [NUM_PROMPTS]
#
# Examples:
#   ./scripts/proton_bench.sh          # 3 cycles, 5 prompts each
#   ./scripts/proton_bench.sh 5 10     # 5 cycles, 10 prompts each

set -euo pipefail

MODEL="openai/gpt-oss-20b"
TP_SIZE=2
PORT=8000
PROTON_OUTPUT_DIR="/tmp/proton_bench_$(date +%Y%m%d_%H%M%S)"
NUM_CYCLES="${1:-3}"
NUM_PROMPTS="${2:-5}"

mkdir -p "$PROTON_OUTPUT_DIR"

echo "=== Proton Contiguous Profiling Benchmark ==="
echo "Model:       $MODEL"
echo "TP size:     $TP_SIZE"
echo "Output dir:  $PROTON_OUTPUT_DIR"
echo "Cycles:      $NUM_CYCLES"
echo "Prompts/cycle: $NUM_PROMPTS"
echo ""

# Start vllm serve in background with periodic flushing
echo "--- Starting vllm serve ---"
vllm serve "$MODEL" \
    --tensor-parallel-size "$TP_SIZE" \
    --port "$PORT" \
    --profiler-config "{
        \"profiler\": \"proton\",
        \"proton_profiler_dir\": \"$PROTON_OUTPUT_DIR\",
        \"proton_mode\": \"periodic_flushing\",
        \"proton_hook\": \"triton\",
        \"proton_context\": \"shadow\",
        \"proton_data\": \"tree\"
    }" &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: Server process died"
        exit 1
    fi
    sleep 1
done

if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
    echo "ERROR: Server did not become ready in 120s"
    kill "$SERVER_PID" 2>/dev/null || true
    exit 1
fi

# Run profiling cycles
for cycle in $(seq 1 "$NUM_CYCLES"); do
    echo ""
    echo "--- Cycle $cycle/$NUM_CYCLES ---"

    # Start profiling
    curl -s -X POST "http://localhost:$PORT/start_profile"
    echo " [start_profile]"

    # Run bench
    vllm bench serve \
        --backend vllm \
        --model "$MODEL" \
        --base-url "http://localhost:$PORT" \
        --dataset-name sharegpt \
        --num-prompts "$NUM_PROMPTS" \
        2>&1 | tail -5

    # Stop profiling
    curl -s -X POST "http://localhost:$PORT/stop_profile"
    echo " [stop_profile]"

    # Check status
    echo "Status:"
    curl -s "http://localhost:$PORT/profile_status" | python3 -m json.tool
done

# Shut down server (triggers finalize for remaining phases)
echo ""
echo "--- Shutting down server ---"
kill "$SERVER_PID" 2>/dev/null || true
wait "$SERVER_PID" 2>/dev/null || true

# List output files
echo ""
echo "--- Output files ---"
ls -lh "$PROTON_OUTPUT_DIR"/ 2>/dev/null || echo "No output files found"
echo ""
echo "Done. Output directory: $PROTON_OUTPUT_DIR"
