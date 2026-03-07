#!/bin/bash
# Master script: run the entire Proton profiling study end-to-end.
#
# Runs all profiling scripts in sequence:
#   1. Sanity check with small model (Qwen3-0.6B)
#   2. Offline batch profiling (nano-vllm + vLLM) with Qwen3-32B
#   3. Online serving profiling (nano-vllm + vLLM) with periodic flushing
#   4. Profile analysis with proton-viewer
#
# The sanity check must pass before proceeding to expensive 32B runs.
# All workload parameters are forwarded to sub-scripts.
#
# Usage: bash scripts/profiling/run_all.sh [OPTIONS]
#   --model MODEL           Model for 32B profiling runs (default: Qwen/Qwen3-32B)
#   --sanity-model MODEL    Model for sanity check (default: Qwen/Qwen3-0.6B)
#   --batch-size N          Number of prompts per batch (default: 4)
#   --input-len N           Max input sequence length (default: 2048)
#   --output-len N          Max tokens to generate (default: 128)
#   --seed N                Random seed for reproducibility (default: 42)
#   --num-prompts N         Number of prompts for online steady-state (default: 4)
#   --warmup-prompts N      Number of prompts for online warmup (default: 2)
#   --port N                Port for vLLM server (default: 8000)
#   --skip-sanity           Skip the sanity check step
#   --skip-offline          Skip offline profiling steps
#   --skip-online           Skip online profiling steps
#   --skip-analysis         Skip profile analysis step

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
MODEL="Qwen/Qwen3-32B"
SANITY_MODEL="Qwen/Qwen3-0.6B"
BATCH_SIZE=4
INPUT_LEN=2048
OUTPUT_LEN=128
SEED=42
NUM_PROMPTS=4
WARMUP_PROMPTS=2
PORT=8000
SKIP_SANITY=false
SKIP_OFFLINE=false
SKIP_ONLINE=false
SKIP_ANALYSIS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --sanity-model) SANITY_MODEL="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --input-len) INPUT_LEN="$2"; shift 2 ;;
        --output-len) OUTPUT_LEN="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --num-prompts) NUM_PROMPTS="$2"; shift 2 ;;
        --warmup-prompts) WARMUP_PROMPTS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --skip-sanity) SKIP_SANITY=true; shift ;;
        --skip-offline) SKIP_OFFLINE=true; shift ;;
        --skip-online) SKIP_ONLINE=true; shift ;;
        --skip-analysis) SKIP_ANALYSIS=true; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Track step results
steps_total=0
steps_passed=0
steps_failed=0
failed_steps=""

echo "============================================"
echo "Proton Profiling Study - Master Script"
echo "============================================"
echo "Model (32B):       $MODEL"
echo "Sanity model:      $SANITY_MODEL"
echo "Batch size:        $BATCH_SIZE"
echo "Input len:         $INPUT_LEN"
echo "Output len:        $OUTPUT_LEN"
echo "Seed:              $SEED"
echo "Online prompts:    $NUM_PROMPTS (warmup: $WARMUP_PROMPTS)"
echo "Port:              $PORT"
echo "Skip sanity:       $SKIP_SANITY"
echo "Skip offline:      $SKIP_OFFLINE"
echo "Skip online:       $SKIP_ONLINE"
echo "Skip analysis:     $SKIP_ANALYSIS"
echo "============================================"
echo ""

# --- Step 1: Sanity check ---
if ! $SKIP_SANITY; then
    steps_total=$((steps_total + 1))
    echo ""
    echo "############################################"
    echo "# STEP 1: Sanity Check ($SANITY_MODEL)"
    echo "############################################"
    echo ""

    if bash "$SCRIPT_DIR/sanity_check.sh" "$SANITY_MODEL"; then
        echo ""
        echo "[STEP 1] PASSED: Sanity check"
        steps_passed=$((steps_passed + 1))
    else
        echo ""
        echo "[STEP 1] FAILED: Sanity check"
        echo "Aborting: sanity check must pass before proceeding to 32B runs."
        steps_failed=$((steps_failed + 1))
        failed_steps="$failed_steps sanity_check"

        echo ""
        echo "============================================"
        echo "FINAL SUMMARY"
        echo "============================================"
        echo "  Steps run:    $steps_total"
        echo "  Passed:       $steps_passed"
        echo "  Failed:       $steps_failed"
        echo "  Failed steps: $failed_steps"
        echo "============================================"
        exit 1
    fi
else
    echo "[SKIP] Step 1: Sanity check"
fi

# --- Step 2: Offline profiling (nano-vllm) ---
if ! $SKIP_OFFLINE; then
    steps_total=$((steps_total + 1))
    echo ""
    echo "############################################"
    echo "# STEP 2: Offline Profiling - nano-vllm"
    echo "############################################"
    echo ""

    if bash "$SCRIPT_DIR/nano_vllm_offline.sh" \
        --model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --seed "$SEED"; then
        echo ""
        echo "[STEP 2] PASSED: nano-vllm offline profiling"
        steps_passed=$((steps_passed + 1))
    else
        echo ""
        echo "[STEP 2] FAILED: nano-vllm offline profiling"
        steps_failed=$((steps_failed + 1))
        failed_steps="$failed_steps nano_vllm_offline"
    fi

    # --- Step 3: Offline profiling (vLLM) ---
    steps_total=$((steps_total + 1))
    echo ""
    echo "############################################"
    echo "# STEP 3: Offline Profiling - vLLM"
    echo "############################################"
    echo ""

    if bash "$SCRIPT_DIR/vllm_offline.sh" \
        --model "$MODEL" \
        --batch-size "$BATCH_SIZE" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --seed "$SEED"; then
        echo ""
        echo "[STEP 3] PASSED: vLLM offline profiling"
        steps_passed=$((steps_passed + 1))
    else
        echo ""
        echo "[STEP 3] FAILED: vLLM offline profiling"
        steps_failed=$((steps_failed + 1))
        failed_steps="$failed_steps vllm_offline"
    fi
else
    echo "[SKIP] Steps 2-3: Offline profiling"
fi

# --- Step 4: Online profiling (nano-vllm) ---
if ! $SKIP_ONLINE; then
    steps_total=$((steps_total + 1))
    echo ""
    echo "############################################"
    echo "# STEP 4: Online Profiling - nano-vllm"
    echo "############################################"
    echo ""

    if bash "$SCRIPT_DIR/nano_vllm_online.sh" \
        --model "$MODEL" \
        --num-prompts "$NUM_PROMPTS" \
        --warmup-prompts "$WARMUP_PROMPTS" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --seed "$SEED"; then
        echo ""
        echo "[STEP 4] PASSED: nano-vllm online profiling"
        steps_passed=$((steps_passed + 1))
    else
        echo ""
        echo "[STEP 4] FAILED: nano-vllm online profiling"
        steps_failed=$((steps_failed + 1))
        failed_steps="$failed_steps nano_vllm_online"
    fi

    # --- Step 5: Online profiling (vLLM) ---
    steps_total=$((steps_total + 1))
    echo ""
    echo "############################################"
    echo "# STEP 5: Online Profiling - vLLM"
    echo "############################################"
    echo ""

    if bash "$SCRIPT_DIR/vllm_online.sh" \
        --model "$MODEL" \
        --num-prompts "$NUM_PROMPTS" \
        --warmup-prompts "$WARMUP_PROMPTS" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --seed "$SEED" \
        --port "$PORT"; then
        echo ""
        echo "[STEP 5] PASSED: vLLM online profiling"
        steps_passed=$((steps_passed + 1))
    else
        echo ""
        echo "[STEP 5] FAILED: vLLM online profiling"
        steps_failed=$((steps_failed + 1))
        failed_steps="$failed_steps vllm_online"
    fi
else
    echo "[SKIP] Steps 4-5: Online profiling"
fi

# --- Step 6: Profile analysis ---
if ! $SKIP_ANALYSIS; then
    steps_total=$((steps_total + 1))
    echo ""
    echo "############################################"
    echo "# STEP 6: Profile Analysis"
    echo "############################################"
    echo ""

    if bash "$SCRIPT_DIR/analyze_profiles.sh"; then
        echo ""
        echo "[STEP 6] PASSED: Profile analysis"
        steps_passed=$((steps_passed + 1))
    else
        echo ""
        echo "[STEP 6] FAILED: Profile analysis"
        steps_failed=$((steps_failed + 1))
        failed_steps="$failed_steps analyze_profiles"
    fi
else
    echo "[SKIP] Step 6: Profile analysis"
fi

# --- Final summary ---
echo ""
echo "============================================"
echo "FINAL SUMMARY"
echo "============================================"
echo "  Steps run:    $steps_total"
echo "  Passed:       $steps_passed"
echo "  Failed:       $steps_failed"
if [ -n "$failed_steps" ]; then
    echo "  Failed steps: $failed_steps"
fi
echo ""
echo "  Output directory: $REPO_ROOT/profiling_output/"
echo "  Analysis output:  $REPO_ROOT/profiling_output/analysis/"
echo "============================================"

if [ "$steps_failed" -eq 0 ] && [ "$steps_total" -gt 0 ]; then
    echo "All profiling steps completed successfully"
    exit 0
else
    echo "Some profiling steps FAILED"
    exit 1
fi
