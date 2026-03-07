#!/bin/bash
# Sanity check: profile both nano-vllm and vLLM with a small model (Qwen3-0.6B)
# using shadow+tree configuration to verify the profiling pipeline works.
#
# Usage: bash scripts/profiling/sanity_check.sh [MODEL]
#   MODEL: HuggingFace model ID or local path (default: Qwen/Qwen3-0.6B)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configurable model (small model for fast sanity check)
MODEL="${1:-Qwen/Qwen3-0.6B}"
MAX_MODEL_LEN=512
NUM_PROMPTS=4
MAX_TOKENS=32

# Output directories
OUTPUT_BASE="$REPO_ROOT/profiling_output/sanity_check"
NANO_OUTPUT="$OUTPUT_BASE/nano_vllm"
VLLM_OUTPUT="$OUTPUT_BASE/vllm"

# Profiling config: shadow context + tree format
CONTEXT="shadow"
DATA="tree"
HOOK="triton"

# Track pass/fail for each framework
nano_pass=false
vllm_pass=false

echo "============================================"
echo "Proton Profiling Sanity Check"
echo "============================================"
echo "Model:   $MODEL"
echo "Config:  context=$CONTEXT, data=$DATA, hook=$HOOK"
echo "Output:  $OUTPUT_BASE"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
. "$REPO_ROOT/.venv/bin/activate"

# Create output directories
mkdir -p "$NANO_OUTPUT" "$VLLM_OUTPUT"

# --- nano-vllm profiling ---
echo ""
echo "--------------------------------------------"
echo "[1/2] Profiling nano-vllm with $MODEL"
echo "--------------------------------------------"
if python "$SCRIPT_DIR/helpers/nano_vllm_offline_run.py" \
    --model "$MODEL" \
    --output-dir "$NANO_OUTPUT" \
    --context "$CONTEXT" \
    --data "$DATA" \
    --hook "$HOOK" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN"; then
    echo "[nano-vllm] Inference completed successfully"
else
    echo "[nano-vllm] FAIL: Inference failed"
fi

# --- vLLM profiling ---
echo ""
echo "--------------------------------------------"
echo "[2/2] Profiling vLLM with $MODEL"
echo "--------------------------------------------"
if python "$SCRIPT_DIR/helpers/vllm_offline_run.py" \
    --model "$MODEL" \
    --output-dir "$VLLM_OUTPUT" \
    --context "$CONTEXT" \
    --data "$DATA" \
    --hook "$HOOK" \
    --num-prompts "$NUM_PROMPTS" \
    --max-tokens "$MAX_TOKENS" \
    --max-model-len "$MAX_MODEL_LEN"; then
    echo "[vLLM] Inference completed successfully"
else
    echo "[vLLM] FAIL: Inference failed"
fi

# --- Validate output files ---
echo ""
echo "--------------------------------------------"
echo "Validating output files"
echo "--------------------------------------------"

# Check nano-vllm output
nano_hatchet_files=$(find "$NANO_OUTPUT" -name "*.hatchet" 2>/dev/null)
if [ -n "$nano_hatchet_files" ]; then
    echo "[nano-vllm] Found .hatchet files:"
    echo "$nano_hatchet_files" | while read -r f; do echo "  $f"; done

    # Validate with proton-viewer
    nano_viewer_ok=true
    for f in $nano_hatchet_files; do
        if proton-viewer -m "$f" > /dev/null 2>&1; then
            echo "  [OK] proton-viewer loaded: $(basename "$f")"
        else
            echo "  [FAIL] proton-viewer failed: $(basename "$f")"
            nano_viewer_ok=false
        fi
    done
    if $nano_viewer_ok; then
        nano_pass=true
    fi
else
    echo "[nano-vllm] FAIL: No .hatchet files found in $NANO_OUTPUT"
fi

# Check vLLM output
vllm_hatchet_files=$(find "$VLLM_OUTPUT" -name "*.hatchet" 2>/dev/null)
if [ -n "$vllm_hatchet_files" ]; then
    echo "[vLLM] Found .hatchet files:"
    echo "$vllm_hatchet_files" | while read -r f; do echo "  $f"; done

    # Validate with proton-viewer
    vllm_viewer_ok=true
    for f in $vllm_hatchet_files; do
        if proton-viewer -m "$f" > /dev/null 2>&1; then
            echo "  [OK] proton-viewer loaded: $(basename "$f")"
        else
            echo "  [FAIL] proton-viewer failed: $(basename "$f")"
            vllm_viewer_ok=false
        fi
    done
    if $vllm_viewer_ok; then
        vllm_pass=true
    fi
else
    echo "[vLLM] FAIL: No .hatchet files found in $VLLM_OUTPUT"
fi

# --- Summary ---
echo ""
echo "============================================"
echo "SUMMARY"
echo "============================================"
if $nano_pass; then
    echo "  nano-vllm:  PASS"
else
    echo "  nano-vllm:  FAIL"
fi
if $vllm_pass; then
    echo "  vLLM:       PASS"
else
    echo "  vLLM:       FAIL"
fi
echo "============================================"

if $nano_pass && $vllm_pass; then
    echo "All sanity checks PASSED"
    exit 0
else
    echo "Some sanity checks FAILED"
    exit 1
fi
