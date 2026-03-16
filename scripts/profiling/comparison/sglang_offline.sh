#!/bin/bash
# Offline batch profiling for SGLang with Qwen3-1.7B using Proton (shadow+tree).
#
# Profiles SGLang offline batch inference for comparison with vLLM.
# Uses sglang.bench_one_batch (single-process) instead of bench_offline_throughput
# (multi-process) because Proton profiling requires the GPU work to happen in
# the same process that owns the Proton session.
#
# Usage: bash scripts/profiling/comparison/sglang_offline.sh [OPTIONS]
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
OUTPUT_DIR="$REPO_ROOT/profiling_output/comparison/sglang/offline"
mkdir -p "$OUTPUT_DIR"

echo "============================================"
echo "SGLang Offline Batch Profiling (Comparison)"
echo "============================================"
echo "Model:       $MODEL"
echo "Batch size:  $BATCH_SIZE"
echo "Input len:   $INPUT_LEN"
echo "Output len:  $OUTPUT_LEN"
echo "Seed:        $SEED"
echo "Output:      $OUTPUT_DIR"
echo ""

# Activate SGLang virtual environment
echo "Activating ~/sglang/.venv..."
. ~/sglang/.venv/bin/activate

# Run profiled inference using bench_one_batch with Proton
echo "Running SGLang offline batch inference with Proton profiling..."
python -m sglang.bench_one_batch \
    --model-path "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --input-len "$INPUT_LEN" \
    --output-len "$OUTPUT_LEN" \
    --random-seed "$SEED" \
    --profile \
    --profile-activities PROTON \
    --profile-output-dir "$OUTPUT_DIR" \
    --proton-context shadow \
    --proton-data tree \
    --proton-hook triton

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
