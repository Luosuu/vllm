#!/bin/bash
# Analyze Proton profiles from vLLM and SGLang, comparing kernel-level performance.
#
# Runs proton-viewer on all .hatchet files from both frameworks,
# categorizes kernels by type, and prints a summary comparison table.
#
# Usage: bash scripts/profiling/comparison/analyze.sh
#
# Expects profile files in profiling_output/comparison/{vllm,sglang}/{offline,online}/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

PROFILE_ROOT="$REPO_ROOT/profiling_output/comparison"
ANALYSIS_DIR="$PROFILE_ROOT/analysis"
mkdir -p "$ANALYSIS_DIR"

# Activate vLLM environment (has proton-viewer installed)
echo "Activating ~/vllm/.venv..."
. ~/vllm/.venv/bin/activate

echo "============================================"
echo "Proton Profile Analysis"
echo "============================================"
echo "Profile root: $PROFILE_ROOT"
echo "Analysis dir: $ANALYSIS_DIR"
echo ""

# --- Define profile file locations ---
# Each entry: <label> <hatchet_file_path>
# We search for .hatchet files in expected directories.

declare -a PROFILE_LABELS=()
declare -a PROFILE_FILES=()

# find_hatchet: locate .hatchet files in a directory
# Args: $1=directory
# Prints the first .hatchet file found (sorted by name).
find_hatchet() {
    local dir="$1"
    if [ ! -d "$dir" ]; then
        return 1
    fi
    # Find .hatchet files, return first by name
    find "$dir" -maxdepth 2 -name "*.hatchet*" -type f 2>/dev/null | sort | head -1
}

# Discover profile files for each framework and mode
discover_profiles() {
    local label="$1"
    local dir="$2"

    local hatchet_file
    hatchet_file=$(find_hatchet "$dir") || true

    if [ -z "$hatchet_file" ]; then
        echo "  WARNING: No .hatchet file found in $dir — skipping $label"
        return
    fi

    PROFILE_LABELS+=("$label")
    PROFILE_FILES+=("$hatchet_file")
    echo "  Found: $label -> $(basename "$hatchet_file")"
}

echo "Discovering profile files..."
discover_profiles "vllm-offline" "$PROFILE_ROOT/vllm/offline"
discover_profiles "sglang-offline" "$PROFILE_ROOT/sglang/offline"
discover_profiles "vllm-online" "$PROFILE_ROOT/vllm/online"
# SGLang online has per-rate subdirectories
for rate_dir in "$PROFILE_ROOT/sglang/online/rate_"*; do
    if [ -d "$rate_dir" ]; then
        rate_name=$(basename "$rate_dir")
        discover_profiles "sglang-online-${rate_name}" "$rate_dir"
    fi
done
echo ""

if [ ${#PROFILE_LABELS[@]} -eq 0 ]; then
    echo "ERROR: No profile files found. Run profiling scripts first."
    exit 1
fi

# --- Run proton-viewer and save raw output ---
echo "--------------------------------------------"
echo "Extracting kernel data with proton-viewer"
echo "--------------------------------------------"

declare -a VIEWER_LABELS=()
declare -a VIEWER_FILES=()

for i in "${!PROFILE_LABELS[@]}"; do
    label="${PROFILE_LABELS[$i]}"
    hatchet_file="${PROFILE_FILES[$i]}"
    output_file="$ANALYSIS_DIR/${label}.txt"

    echo "  Processing: $label"
    if proton-viewer -m time/ns "$hatchet_file" > "$output_file" 2>&1; then
        echo "    -> Saved to $(basename "$output_file")"
        VIEWER_LABELS+=("$label")
        VIEWER_FILES+=("$output_file")
    else
        echo "    WARNING: proton-viewer failed for $hatchet_file"
    fi
done
echo ""

if [ ${#VIEWER_LABELS[@]} -eq 0 ]; then
    echo "ERROR: proton-viewer produced no output."
    exit 1
fi

# --- Run Python analysis ---
echo "--------------------------------------------"
echo "Analyzing kernel categories"
echo "--------------------------------------------"

# Build argument list: label1 file1 label2 file2 ...
PYTHON_ARGS=()
for i in "${!VIEWER_LABELS[@]}"; do
    PYTHON_ARGS+=("${VIEWER_LABELS[$i]}" "${VIEWER_FILES[$i]}")
done

python "$SCRIPT_DIR/analyze_kernels.py" "${PYTHON_ARGS[@]}" | tee "$ANALYSIS_DIR/summary.txt"

echo ""
echo "============================================"
echo "Analysis complete"
echo "============================================"
echo "Raw proton-viewer output:  $ANALYSIS_DIR/*.txt"
echo "Summary:                   $ANALYSIS_DIR/summary.txt"
echo "============================================"
