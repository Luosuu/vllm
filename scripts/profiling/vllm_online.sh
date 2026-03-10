#!/bin/bash
# Online serving profiling for vLLM with Proton periodic flushing to capture
# warmup and steady-state phases separately.
#
# Starts a vLLM server, uses REST API (/start_profile, /stop_profile) to
# control profiling phases, and sends chat completion requests matching the
# nano-vllm online workload for fair comparison.
#
# Usage: bash scripts/profiling/vllm_online.sh [OPTIONS]
#   --model MODEL           Model name or path (default: Qwen/Qwen3-32B)
#   --num-prompts N         Number of prompts per steady-state phase (default: 4)
#   --warmup-prompts N      Number of prompts for warmup phase (default: 2)
#   --input-len N           Max input sequence length / max_model_len (default: 2048)
#   --output-len N          Max tokens to generate per prompt (default: 128)
#   --seed N                Random seed for reproducibility (default: 42)
#   --port N                Port for vLLM server (default: 8000)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-32B"
NUM_PROMPTS=4
WARMUP_PROMPTS=2
INPUT_LEN=2048
OUTPUT_LEN=128
SEED=42
PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --warmup-prompts) WARMUP_PROMPTS="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Output base directory
OUTPUT_BASE="$REPO_ROOT/profiling_output/vllm/online"

# Hook for all configs
HOOK="triton"

# Profiling context: use shadow for low-overhead phase comparison
CONTEXT="shadow"
DATA="tree"

echo "============================================"
echo "vLLM Online Profiling (Periodic Flushing)"
echo "============================================"
echo "Model:           $MODEL"
echo "Num prompts:     $NUM_PROMPTS"
echo "Warmup prompts:  $WARMUP_PROMPTS"
echo "Input len:       $INPUT_LEN"
echo "Output len:      $OUTPUT_LEN"
echo "Seed:            $SEED"
echo "Port:            $PORT"
echo "Hook:            $HOOK"
echo "Context:         $CONTEXT"
echo "Data:            $DATA"
echo "Output base:     $OUTPUT_BASE"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
. "$REPO_ROOT/.venv/bin/activate"

# Create output directory
CONFIG_DIR="$OUTPUT_BASE/${CONTEXT}_${DATA}"
mkdir -p "$CONFIG_DIR"

# Track results
run_ok=false

echo "--------------------------------------------"
echo "Running online serving profiling with periodic flushing"
echo "  Context: $CONTEXT, Data: $DATA"
echo "  Output: $CONFIG_DIR"
echo "--------------------------------------------"

if python "$SCRIPT_DIR/helpers/vllm_online_run.py" \
    --model "$MODEL" \
    --output-dir "$CONFIG_DIR" \
    --context "$CONTEXT" \
    --data "$DATA" \
    --hook "$HOOK" \
    --num-prompts "$NUM_PROMPTS" \
    --warmup-prompts "$WARMUP_PROMPTS" \
    --max-tokens "$OUTPUT_LEN" \
    --max-model-len "$INPUT_LEN" \
    --seed "$SEED" \
    --port "$PORT"; then
    echo "[OK] Online profiling completed"
    run_ok=true
else
    echo "[FAIL] Online profiling failed"
fi

# --- Validate output files ---
echo ""
echo "--------------------------------------------"
echo "Validating output files"
echo "--------------------------------------------"

hatchet_count=0
hatchet_valid=0
part_count=0

hatchet_files=$(find "$CONFIG_DIR" -name "*.hatchet" 2>/dev/null || true)
if [ -n "$hatchet_files" ]; then
    for f in $hatchet_files; do
        hatchet_count=$((hatchet_count + 1))

        # Check for per-phase .part_N files
        if echo "$f" | grep -q "\.part_"; then
            part_count=$((part_count + 1))
        fi

        if proton-viewer -m "$f" > /dev/null 2>&1; then
            echo "  [OK] proton-viewer: $(basename "$f")"
            hatchet_valid=$((hatchet_valid + 1))
        else
            echo "  [FAIL] proton-viewer: $(basename "$f")"
        fi
    done
else
    echo "  No .hatchet files found in $CONFIG_DIR"
fi

# --- Output file summary ---
echo ""
echo "============================================"
echo "OUTPUT FILE SUMMARY"
echo "============================================"
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

echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
echo "  Run status:       $(if $run_ok; then echo "PASS"; else echo "FAIL"; fi)"
echo "  .hatchet files:   $hatchet_count found, $hatchet_valid valid"
echo "  Per-phase files:  $part_count (.part_N.hatchet)"
echo "============================================"

if $run_ok && [ "$hatchet_count" -gt 0 ] && [ "$hatchet_valid" -eq "$hatchet_count" ]; then
    echo "Online profiling completed successfully"
    exit 0
else
    echo "Online profiling FAILED"
    exit 1
fi
