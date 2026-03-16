#!/bin/bash
# Offline batch profiling for vLLM with Qwen3-1.7B using Proton (shadow+tree).
#
# Profiles vLLM offline batch inference as a baseline for comparison with SGLang.
# Uses shadow context + tree data for lowest profiling overhead.
#
# Usage: bash scripts/profiling/comparison/vllm_offline.sh [OPTIONS]
#   --model MODEL         Model name or path (default: Qwen/Qwen3-1.7B)
#   --batch-size N        Number of prompts per batch (default: 8)
#   --input-len N         Input sequence length (default: 1024)
#   --output-len N        Max tokens to generate per prompt (default: 256)
#   --seed N              Random seed for reproducibility (default: 42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-1.7B"
BATCH_SIZE=8
INPUT_LEN=1024
OUTPUT_LEN=256
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Output directory
OUTPUT_DIR="$REPO_ROOT/profiling_output/comparison/vllm/offline"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "vLLM Offline Batch Profiling (Comparison)"
echo "============================================"
echo "Model:       $MODEL"
echo "Batch size:  $BATCH_SIZE"
echo "Input len:   $INPUT_LEN"
echo "Output len:  $OUTPUT_LEN"
echo "Seed:        $SEED"
echo "Output:      $OUTPUT_DIR"
echo ""

# Activate vLLM virtual environment
echo "Activating ~/vllm/.venv..."
. ~/vllm/.venv/bin/activate

# Run profiled inference
echo "Running vLLM offline batch inference with Proton profiling..."
vllm bench latency \
    --model "$MODEL" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --num-iters 1 \
    --num-iters-warmup 1 \
    --seed "$SEED" \
    --profile \
    --profiler-config "{
        \"profiler\": \"proton\",
        \"proton_profiler_dir\": \"$OUTPUT_DIR\",
        \"proton_context\": \"shadow\",
        \"proton_data\": \"tree\",
        \"proton_hook\": \"triton\"
    }"

echo ""
echo "Inference completed."

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
