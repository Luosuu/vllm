#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OVERHEAD_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
PYTHON=${VLLM_PYTHON:-python3}
RESULTS_MOUNT=${VLLM_RESULTS_MOUNT:-/mnt/vllm-profile-results}
JOB_NAME=${VLLM_JOB_NAME:-vllm-profile-job}
STORAGE_MODE=${VLLM_STORAGE_MODE:-object}
GRAPH_MODES=${VLLM_GRAPH_MODES:-cudagraph}
SYNC_INTERVAL=${VLLM_SYNC_INTERVAL:-300}
EXPECTED_GPUS=${VLLM_EXPECTED_GPUS:-8}
PERSIST_DIR=$RESULTS_MOUNT/$JOB_NAME
SYNC_PID=

if [[ $STORAGE_MODE == filesystem ]]; then
  WORK_DIR=$PERSIST_DIR
elif [[ $STORAGE_MODE == object ]]; then
  WORK_DIR=${VLLM_LOCAL_WORK_DIR:-/workspace/vllm-profile-results/$JOB_NAME}
else
  echo "VLLM_STORAGE_MODE must be filesystem or object" >&2
  exit 2
fi

mkdir -p "$PERSIST_DIR" "$WORK_DIR"

sync_results() {
  local mode=${1:-checkpoint}
  [[ $STORAGE_MODE == object ]] || return 0
  local options=(-a --delete --exclude='*.tmp')
  if [[ $mode != final ]]; then
    options+=(--exclude='profiles/')
  fi
  rsync "${options[@]}" "$WORK_DIR/" "$PERSIST_DIR/"
  if [[ $mode != final ]]; then
    local case_file case_dir relative
    while IFS= read -r -d '' case_file; do
      case_dir=$(dirname "$case_file")
      relative=${case_dir#"$WORK_DIR"/}
      mkdir -p "$PERSIST_DIR/$relative"
      rsync -a --delete "$case_dir/" "$PERSIST_DIR/$relative/"
    done < <(find "$WORK_DIR" -name case.json -type f -print0)
  fi
}

write_status() {
  local status=$1
  local exit_code=$2
  "$PYTHON" -c '
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "job_name": os.environ.get("VLLM_JOB_NAME"),
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "storage_mode": os.environ.get("VLLM_STORAGE_MODE"),
    "platform": os.environ.get("NEBIUS_PLATFORM"),
    "preset": os.environ.get("NEBIUS_PRESET"),
}, indent=2) + "\n")
' "$WORK_DIR/job_status.json" "$status" "$exit_code"
}

stop_sync_loop() {
  if [[ -n $SYNC_PID ]]; then
    kill "$SYNC_PID" 2>/dev/null || true
    wait "$SYNC_PID" 2>/dev/null || true
    SYNC_PID=
  fi
}

on_exit() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_sync_loop
  if [[ $exit_code == 0 ]]; then
    write_status completed "$exit_code" || true
  else
    write_status failed "$exit_code" || true
  fi
  sync_results final || true
  exit "$exit_code"
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ $STORAGE_MODE == object && -n $(find "$PERSIST_DIR" -mindepth 1 -print -quit) ]]; then
  rsync -a "$PERSIST_DIR/" "$WORK_DIR/"
fi

args_file=$(mktemp)
printf '%s' "${VLLM_BENCHMARK_ARGS_B64:-}" | base64 -d >"$args_file"
mapfile -d '' -t BENCHMARK_ARGS <"$args_file"
rm -f "$args_file"

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ $gpu_count != "$EXPECTED_GPUS" ]]; then
  echo "expected $EXPECTED_GPUS GPUs, found $gpu_count" >&2
  exit 1
fi
nvidia-smi -L
df -h "$WORK_DIR" "$RESULTS_MOUNT"
nsys --version
"$PYTHON" -c '
import triton.profiler as proton
import vllm

required = ("start", "activate", "deactivate", "finalize")
missing = [name for name in required if not hasattr(proton, name)]
if missing:
    raise RuntimeError(f"Triton Proton API is missing: {missing}")
print(f"vLLM {vllm.__version__}; Proton API ready")
'

if [[ ${VLLM_JOB_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'Preflight passed. Matrix arguments:'
  printf ' %q' "${BENCHMARK_ARGS[@]}"
  printf '\n'
  exit 0
fi

env -u HF_HUB_OFFLINE "$PYTHON" "$SCRIPT_DIR/prepare_model_metadata.py" \
  "${BENCHMARK_ARGS[@]}"

write_status running 0
sync_results checkpoint

if [[ $STORAGE_MODE == object && $SYNC_INTERVAL -gt 0 ]]; then
  (
    while true; do
      sleep "$SYNC_INTERVAL"
      sync_results checkpoint || true
    done
  ) &
  SYNC_PID=$!
fi

read -r -a graph_modes <<<"$GRAPH_MODES"
for graph_mode in "${graph_modes[@]}"; do
  "$PYTHON" "$OVERHEAD_DIR/run_profiler_matrix.py" \
    "${BENCHMARK_ARGS[@]}" \
    --graph-mode "$graph_mode" \
    --output-dir "$WORK_DIR/$graph_mode"
  sync_results checkpoint
done
