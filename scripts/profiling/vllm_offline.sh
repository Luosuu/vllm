#!/bin/bash
# Offline batch profiling for vLLM with Qwen3-32B using all 4 Proton configurations.
#
# Runs vLLM's offline inference with Proton profiling in 4 configurations:
#   1. shadow+tree   2. shadow+trace   3. python+tree   4. python+trace
# Each config outputs to a separate subdirectory under profiling_output/vllm/offline/.
#
# Usage: bash scripts/profiling/vllm_offline.sh [OPTIONS]
#   --model MODEL         Model name or path (default: Qwen/Qwen3-32B)
#   --batch-size N        Number of prompts per batch (default: 4)
#   --input-len N         Max input sequence length / max_model_len (default: 2048)
#   --output-len N        Max tokens to generate per prompt (default: 128)
#   --seed N              Random seed for reproducibility (default: 42)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-32B"
BATCH_SIZE=4
INPUT_LEN=2048
OUTPUT_LEN=128
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

# Output base directory
OUTPUT_BASE="$REPO_ROOT/profiling_output/vllm/offline"

# All 4 profiling configurations: context x data
CONFIGS=(
    "shadow:tree"
    "shadow:trace"
    "python:tree"
    "python:trace"
)

echo "============================================"
echo "vLLM Offline Batch Profiling"
echo "============================================"
echo "Model:       $MODEL"
echo "Batch size:  $BATCH_SIZE"
echo "Input len:   $INPUT_LEN"
echo "Output len:  $OUTPUT_LEN"
echo "Seed:        $SEED"
echo "Output base: $OUTPUT_BASE"
echo "Configs:     ${#CONFIGS[@]} (shadow+tree, shadow+trace, python+tree, python+trace)"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
. "$REPO_ROOT/.venv/bin/activate"

# Track results
total=0
passed=0
failed=0

for config in "${CONFIGS[@]}"; do
    # Parse context:data
    CONTEXT="${config%%:*}"
    DATA="${config##*:}"

    CONFIG_DIR="$OUTPUT_BASE/${CONTEXT}_${DATA}"
    mkdir -p "$CONFIG_DIR"

    total=$((total + 1))

    echo "--------------------------------------------"
    echo "[$total/${#CONFIGS[@]}] Config: context=$CONTEXT, data=$DATA"
    echo "  Output: $CONFIG_DIR"
    echo "--------------------------------------------"

    if python "$SCRIPT_DIR/helpers/vllm_offline_run.py" \
        --model "$MODEL" \
        --output-dir "$CONFIG_DIR" \
        --context "$CONTEXT" \
        --data "$DATA" \
        --hook "triton" \
        --num-prompts "$BATCH_SIZE" \
        --max-tokens "$OUTPUT_LEN" \
        --max-model-len "$INPUT_LEN"; then
        echo "  [OK] Inference completed for $CONTEXT+$DATA"
        passed=$((passed + 1))
    else
        echo "  [FAIL] Inference failed for $CONTEXT+$DATA"
        failed=$((failed + 1))
    fi
    echo ""
done

# --- Validate .hatchet files with proton-viewer ---
echo "--------------------------------------------"
echo "Validating .hatchet output files"
echo "--------------------------------------------"

hatchet_count=0
hatchet_valid=0
for config in "${CONFIGS[@]}"; do
    CONTEXT="${config%%:*}"
    DATA="${config##*:}"
    CONFIG_DIR="$OUTPUT_BASE/${CONTEXT}_${DATA}"

    hatchet_files=$(find "$CONFIG_DIR" -name "*.hatchet" 2>/dev/null || true)
    if [ -n "$hatchet_files" ]; then
        for f in $hatchet_files; do
            hatchet_count=$((hatchet_count + 1))
            if proton-viewer -m "$f" > /dev/null 2>&1; then
                echo "  [OK] proton-viewer: $(basename "$f") ($CONTEXT+$DATA)"
                hatchet_valid=$((hatchet_valid + 1))
            else
                echo "  [FAIL] proton-viewer: $(basename "$f") ($CONTEXT+$DATA)"
            fi
        done
    fi
done

# --- Summary of output files ---
echo ""
echo "============================================"
echo "OUTPUT FILE SUMMARY"
echo "============================================"
for config in "${CONFIGS[@]}"; do
    CONTEXT="${config%%:*}"
    DATA="${config##*:}"
    CONFIG_DIR="$OUTPUT_BASE/${CONTEXT}_${DATA}"

    echo ""
    echo "  [$CONTEXT+$DATA] $CONFIG_DIR"
    if [ -d "$CONFIG_DIR" ]; then
        file_count=$(find "$CONFIG_DIR" -type f 2>/dev/null | wc -l)
        echo "    Files: $file_count"
        find "$CONFIG_DIR" -type f 2>/dev/null | sort | while read -r f; do
            size=$(stat --printf="%s" "$f" 2>/dev/null || echo "?")
            echo "    - $(basename "$f") ($size bytes)"
        done
    else
        echo "    (directory not found)"
    fi
done

echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "  Configs run:    $total"
echo "  Passed:         $passed"
echo "  Failed:         $failed"
echo "  .hatchet files: $hatchet_count found, $hatchet_valid valid"
echo "============================================"

if [ "$failed" -eq 0 ] && [ "$passed" -gt 0 ]; then
    echo "All profiling runs completed successfully"
    exit 0
else
    echo "Some profiling runs FAILED"
    exit 1
fi
