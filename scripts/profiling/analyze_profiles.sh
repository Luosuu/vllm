#!/bin/bash
# Analyze all captured Proton profiles using proton-viewer.
#
# Runs proton-viewer on all .hatchet files from offline and online profiling,
# extracts top-10 kernels by time, total GPU time, and kernel count.
# Compares vLLM vs nano-vllm side by side for each profiling mode.
# Saves analysis output to profiling_output/analysis/ as text files.
# Lists chrome trace files with instructions for Perfetto.
# Compares warmup vs steady-state phases for periodic flushing profiles.
#
# Usage: bash scripts/profiling/analyze_profiles.sh [OPTIONS]
#   --profile-dir DIR     Base profiling output directory (default: profiling_output)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# --- Parse arguments ---
PROFILE_DIR="$REPO_ROOT/profiling_output"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile-dir) PROFILE_DIR="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# Analysis output directory
ANALYSIS_DIR="$PROFILE_DIR/analysis"
mkdir -p "$ANALYSIS_DIR"

echo "============================================"
echo "Proton Profile Analysis"
echo "============================================"
echo "Profile dir:  $PROFILE_DIR"
echo "Analysis dir: $ANALYSIS_DIR"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
. "$REPO_ROOT/.venv/bin/activate"

# Track totals
total_files=0
analyzed_files=0
failed_files=0

# --- Helper: run proton-viewer and save output ---
# Runs proton-viewer on a .hatchet file, saves output to analysis dir,
# and prints a summary to stdout.
# Args: $1 = hatchet file path, $2 = output label (e.g., "nano_vllm_offline_shadow_tree")
analyze_hatchet() {
    local hatchet_file="$1"
    local label="$2"
    local output_file="$ANALYSIS_DIR/${label}.txt"

    total_files=$((total_files + 1))

    if ! [ -f "$hatchet_file" ]; then
        echo "  [SKIP] File not found: $hatchet_file"
        failed_files=$((failed_files + 1))
        return 1
    fi

    echo "  Analyzing: $(basename "$hatchet_file")"

    # Run proton-viewer with time/ns metric to get kernel timing
    if proton-viewer -m time/ns "$hatchet_file" > "$output_file" 2>&1; then
        analyzed_files=$((analyzed_files + 1))
        echo "  [OK] Saved to: $(basename "$output_file")"

        # Extract summary stats from proton-viewer output
        local kernel_count
        kernel_count=$(grep -c "^" "$output_file" 2>/dev/null || echo "0")
        # Subtract header lines (proton-viewer prints a header row)
        if [ "$kernel_count" -gt 1 ]; then
            kernel_count=$((kernel_count - 1))
        fi
        echo "    Kernel entries: $kernel_count"

        # Print top-10 kernels (first 11 lines: 1 header + 10 data)
        echo "    Top-10 kernels by time:"
        head -n 11 "$output_file" | sed 's/^/      /'
    else
        echo "  [FAIL] proton-viewer failed for: $(basename "$hatchet_file")"
        failed_files=$((failed_files + 1))
        return 1
    fi
}

# --- Analyze offline profiles ---
echo "============================================"
echo "OFFLINE PROFILE ANALYSIS"
echo "============================================"

OFFLINE_CONFIGS=(
    "shadow:tree"
    "shadow:trace"
    "python:tree"
    "python:trace"
)

for framework in nano_vllm vllm; do
    echo ""
    echo "--- $framework (offline) ---"

    for config in "${OFFLINE_CONFIGS[@]}"; do
        CONTEXT="${config%%:*}"
        DATA="${config##*:}"
        CONFIG_DIR="$PROFILE_DIR/$framework/offline/${CONTEXT}_${DATA}"

        # Only analyze .hatchet files (tree format)
        if [ "$DATA" = "tree" ]; then
            hatchet_files=$(find "$CONFIG_DIR" -name "*.hatchet" 2>/dev/null | sort || true)
            if [ -n "$hatchet_files" ]; then
                for f in $hatchet_files; do
                    label="${framework}_offline_${CONTEXT}_${DATA}_$(basename "$f" .hatchet)"
                    analyze_hatchet "$f" "$label"
                done
            else
                echo "  [SKIP] No .hatchet files in $CONFIG_DIR"
            fi
        fi
    done
done

# --- Analyze online profiles (periodic flushing) ---
echo ""
echo "============================================"
echo "ONLINE PROFILE ANALYSIS (Periodic Flushing)"
echo "============================================"

for framework in nano_vllm vllm; do
    echo ""
    echo "--- $framework (online) ---"

    CONFIG_DIR="$PROFILE_DIR/$framework/online/shadow_tree"

    hatchet_files=$(find "$CONFIG_DIR" -name "*.hatchet" 2>/dev/null | sort || true)
    if [ -n "$hatchet_files" ]; then
        for f in $hatchet_files; do
            label="${framework}_online_$(basename "$f" .hatchet)"
            analyze_hatchet "$f" "$label"
        done
    else
        echo "  [SKIP] No .hatchet files in $CONFIG_DIR"
    fi
done

# --- Warmup vs steady-state phase comparison ---
echo ""
echo "============================================"
echo "WARMUP vs STEADY-STATE PHASE COMPARISON"
echo "============================================"

for framework in nano_vllm vllm; do
    echo ""
    echo "--- $framework ---"

    CONFIG_DIR="$PROFILE_DIR/$framework/online/shadow_tree"

    # Look for .part_0 (warmup) and .part_1 (steady-state) files
    warmup_file=$(find "$CONFIG_DIR" -name "*.part_0.hatchet" 2>/dev/null | head -n 1 || true)
    steady_file=$(find "$CONFIG_DIR" -name "*.part_1.hatchet" 2>/dev/null | head -n 1 || true)

    if [ -n "$warmup_file" ] && [ -n "$steady_file" ]; then
        echo "  Warmup file:       $(basename "$warmup_file")"
        echo "  Steady-state file: $(basename "$steady_file")"

        warmup_out="$ANALYSIS_DIR/${framework}_online_warmup.txt"
        steady_out="$ANALYSIS_DIR/${framework}_online_steady.txt"

        proton-viewer -m time/ns "$warmup_file" > "$warmup_out" 2>&1 || true
        proton-viewer -m time/ns "$steady_file" > "$steady_out" 2>&1 || true

        echo ""
        echo "  WARMUP phase (top-10):"
        head -n 11 "$warmup_out" 2>/dev/null | sed 's/^/    /' || echo "    (no data)"

        echo ""
        echo "  STEADY-STATE phase (top-10):"
        head -n 11 "$steady_out" 2>/dev/null | sed 's/^/    /' || echo "    (no data)"
    else
        echo "  [SKIP] Phase files not found (.part_0.hatchet / .part_1.hatchet)"
    fi
done

# --- Side-by-side comparison ---
echo ""
echo "============================================"
echo "SIDE-BY-SIDE COMPARISON: vLLM vs nano-vllm"
echo "============================================"

# Compare offline shadow+tree profiles
echo ""
echo "--- Offline (shadow+tree) ---"
nano_offline="$ANALYSIS_DIR/nano_vllm_offline_shadow_tree_proton_rank0.txt"
vllm_offline="$ANALYSIS_DIR/vllm_offline_shadow_tree_proton_rank0.txt"

if [ -f "$nano_offline" ] && [ -f "$vllm_offline" ]; then
    echo ""
    echo "  nano-vllm top-10:"
    head -n 11 "$nano_offline" | sed 's/^/    /'
    echo ""
    echo "  vLLM top-10:"
    head -n 11 "$vllm_offline" | sed 's/^/    /'
else
    echo "  [SKIP] Offline shadow+tree analysis files not found"
fi

# Compare offline python+tree profiles
echo ""
echo "--- Offline (python+tree) ---"
nano_python="$ANALYSIS_DIR/nano_vllm_offline_python_tree_proton_rank0.txt"
vllm_python="$ANALYSIS_DIR/vllm_offline_python_tree_proton_rank0.txt"

if [ -f "$nano_python" ] && [ -f "$vllm_python" ]; then
    echo ""
    echo "  nano-vllm top-10:"
    head -n 11 "$nano_python" | sed 's/^/    /'
    echo ""
    echo "  vLLM top-10:"
    head -n 11 "$vllm_python" | sed 's/^/    /'
else
    echo "  [SKIP] Offline python+tree analysis files not found"
fi

# --- Chrome trace files listing ---
echo ""
echo "============================================"
echo "CHROME TRACE FILES (.chrome_trace)"
echo "============================================"
echo ""
echo "Chrome trace files can be opened in Perfetto for visual timeline inspection:"
echo "  1. Open https://ui.perfetto.dev/"
echo "  2. Click 'Open trace file'"
echo "  3. Select the .chrome_trace file"
echo ""

chrome_count=0
for framework in nano_vllm vllm; do
    trace_files=$(find "$PROFILE_DIR/$framework" -name "*.chrome_trace" 2>/dev/null | sort || true)
    if [ -n "$trace_files" ]; then
        echo "  $framework:"
        for f in $trace_files; do
            size=$(stat --printf="%s" "$f" 2>/dev/null || echo "?")
            echo "    - $f ($size bytes)"
            chrome_count=$((chrome_count + 1))
        done
        echo ""
    fi
done

if [ "$chrome_count" -eq 0 ]; then
    echo "  No .chrome_trace files found"
fi

# Save chrome trace listing to analysis dir
{
    echo "Chrome Trace Files"
    echo "=================="
    echo ""
    echo "Open in Perfetto: https://ui.perfetto.dev/"
    echo ""
    for framework in nano_vllm vllm; do
        trace_files=$(find "$PROFILE_DIR/$framework" -name "*.chrome_trace" 2>/dev/null | sort || true)
        if [ -n "$trace_files" ]; then
            echo "$framework:"
            for f in $trace_files; do
                size=$(stat --printf="%s" "$f" 2>/dev/null || echo "?")
                echo "  - $f ($size bytes)"
            done
            echo ""
        fi
    done
} > "$ANALYSIS_DIR/chrome_traces.txt"

# --- Final summary ---
echo ""
echo "============================================"
echo "ANALYSIS SUMMARY"
echo "============================================"
echo "  .hatchet files analyzed: $analyzed_files / $total_files"
echo "  Failed:                  $failed_files"
echo "  .chrome_trace files:     $chrome_count"
echo "  Analysis output dir:     $ANALYSIS_DIR"
echo ""
echo "  Analysis files:"
find "$ANALYSIS_DIR" -type f 2>/dev/null | sort | while read -r f; do
    size=$(stat --printf="%s" "$f" 2>/dev/null || echo "?")
    echo "    - $(basename "$f") ($size bytes)"
done
echo "============================================"

if [ "$failed_files" -eq 0 ] && [ "$analyzed_files" -gt 0 ]; then
    echo "Profile analysis completed successfully"
    exit 0
elif [ "$total_files" -eq 0 ]; then
    echo "WARNING: No .hatchet files found to analyze"
    echo "Run profiling scripts first (nano_vllm_offline.sh, vllm_offline.sh, etc.)"
    exit 1
else
    echo "Some profile analyses FAILED"
    exit 1
fi
