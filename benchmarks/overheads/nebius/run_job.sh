#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -Eeuo pipefail

PYTHON=${VLLM_PYTHON:-python3}
NSIGHT_SYSTEMS_PACKAGE=${NSIGHT_SYSTEMS_PACKAGE:-cuda-nsight-systems-13-0}
REPO_URL=${VLLM_REPO_URL:-https://github.com/Luosuu/vllm.git}
UPSTREAM_REPO_URL=${VLLM_UPSTREAM_REPO_URL:-https://github.com/vllm-project/vllm.git}
SOURCE_REVISION=${VLLM_SOURCE_REVISION:-proton-profiler-clean}
BENCHMARK_SOURCE_REVISION=${VLLM_BENCHMARK_SOURCE_REVISION:-profiler-overhead-benchmarks}
SOURCE_DIR=${VLLM_SOURCE_DIR:-/workspace/vllm-source}
BENCHMARK_DIR=${VLLM_BENCHMARK_DIR:-/workspace/vllm-benchmark}
OVERHEAD_DIR=$BENCHMARK_DIR/benchmarks/overheads
RESULTS_MOUNT=${VLLM_RESULTS_MOUNT:-/mnt/vllm-profile-results}
JOB_NAME=${VLLM_JOB_NAME:-vllm-profile-job}
STORAGE_MODE=${VLLM_STORAGE_MODE:-object}
GRAPH_MODES=${VLLM_GRAPH_MODES:-cudagraph}
SYNC_INTERVAL=${VLLM_SYNC_INTERVAL:-300}
EXPECTED_GPUS=${VLLM_EXPECTED_GPUS:-8}
JOB_ID_MARKER=${VLLM_JOB_ID_MARKER:-}
SYNC_PID=

install_bootstrap_dependencies() {
  local packages=()
  command -v git >/dev/null || packages+=(git)
  command -v nsys >/dev/null || packages+=("$NSIGHT_SYSTEMS_PACKAGE")
  if ((${#packages[@]})); then
    apt-get update
    apt-get install -y --no-install-recommends "${packages[@]}"
    rm -rf /var/lib/apt/lists/*
  fi
  uv pip install --system \
    huggingface_hub llnl-hatchet matplotlib ninja pandas regex
}

install_bootstrap_dependencies

RESULT_ID=$JOB_NAME
if [[ $STORAGE_MODE == object && -n $JOB_ID_MARKER ]]; then
  marker_path=$RESULTS_MOUNT/$JOB_ID_MARKER
  for _ in {1..60}; do
    if [[ -s $marker_path ]]; then
      RESULT_ID=$(tr -d '\r\n' <"$marker_path")
      break
    fi
    sleep 5
  done
  [[ $RESULT_ID =~ ^aijob-[a-z0-9]+$ ]] || {
    echo "timed out waiting for a valid Nebius Job ID marker" >&2
    exit 1
  }
fi
export VLLM_NEBIUS_JOB_ID=$RESULT_ID
PERSIST_DIR=$RESULTS_MOUNT/$RESULT_ID

if [[ $STORAGE_MODE == filesystem ]]; then
  WORK_DIR=$PERSIST_DIR
elif [[ $STORAGE_MODE == object ]]; then
  WORK_DIR=${VLLM_LOCAL_WORK_DIR:-/workspace/vllm-profile-results/$RESULT_ID}
else
  echo "VLLM_STORAGE_MODE must be filesystem or object" >&2
  exit 2
fi

mkdir -p "$PERSIST_DIR" "$WORK_DIR"

sync_results() {
  local mode=${1:-checkpoint}
  [[ $STORAGE_MODE == object ]] || return 0
  if [[ $mode == final ]]; then
    rm -rf "$PERSIST_DIR"
    mkdir -p "$PERSIST_DIR"
    cp -R "$WORK_DIR/." "$PERSIST_DIR/"
    return
  fi
  if [[ -f $WORK_DIR/job_status.json ]]; then
    rm -f "$PERSIST_DIR/job_status.json"
    cp "$WORK_DIR/job_status.json" "$PERSIST_DIR/job_status.json"
  fi
  local case_file case_dir relative destination
  while IFS= read -r -d '' case_file; do
    case_dir=$(dirname "$case_file")
    relative=${case_dir#"$WORK_DIR"/}
    destination=$PERSIST_DIR/$relative
    rm -rf "$destination"
    mkdir -p "$(dirname "$destination")"
    cp -R "$case_dir" "$destination"
  done < <(find "$WORK_DIR" -name case.json -type f -print0)
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
    "nebius_job_id": os.environ.get("VLLM_NEBIUS_JOB_ID"),
    "status": sys.argv[2],
    "exit_code": int(sys.argv[3]),
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "storage_mode": os.environ.get("VLLM_STORAGE_MODE"),
    "platform": os.environ.get("NEBIUS_PLATFORM"),
    "preset": os.environ.get("NEBIUS_PRESET"),
    "repository": os.environ.get("VLLM_REPO_URL"),
    "requested_vllm_revision": os.environ.get("VLLM_SOURCE_REVISION"),
    "vllm_commit": os.environ.get("VLLM_BUILD_COMMIT"),
    "requested_benchmark_revision": os.environ.get(
        "VLLM_BENCHMARK_SOURCE_REVISION"
    ),
    "benchmark_commit": os.environ.get("VLLM_BENCHMARK_REVISION"),
}, indent=2) + "\n")
' "$WORK_DIR/job_status.json" "$status" "$exit_code"
}

summarize_retained_proton_profiles() {
  local summary_script=$OVERHEAD_DIR/skills/analyze-proton-profile/scripts/summarize_profile.py
  local profile_file profile_dir summary_path
  local -A seen=()

  while IFS= read -r -d '' profile_file; do
    profile_dir=$(dirname "$profile_file")
    [[ -z ${seen[$profile_dir]:-} ]] || continue
    seen[$profile_dir]=1
    summary_path=$(dirname "$profile_dir")/profile-summary.md
    printf 'Proton profile summary: %s\n' "$summary_path"
    "$PYTHON" "$summary_script" --viewer proton-viewer "$profile_dir" |
      tee "$summary_path"
  done < <(find "$WORK_DIR" -type f -name '*.hatchet' -print0)
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
  cp -R "$PERSIST_DIR/." "$WORK_DIR/"
fi

args_file=$(mktemp)
printf '%s' "${VLLM_BENCHMARK_ARGS_B64:-}" | base64 -d >"$args_file"
mapfile -d '' -t BENCHMARK_ARGS <"$args_file"
rm -f "$args_file"

checkout_revision() {
  local destination=$1
  local revision=$2
  local sparse_path=${3:-}

  git clone --filter=blob:none --no-checkout "$REPO_URL" "$destination"
  if [[ -n $sparse_path ]]; then
    git -C "$destination" sparse-checkout set "$sparse_path"
  fi
  git -C "$destination" fetch origin "$revision"
  git -C "$destination" checkout --detach FETCH_HEAD
  git -C "$destination" rev-parse HEAD
}

gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [[ $gpu_count != "$EXPECTED_GPUS" ]]; then
  echo "expected $EXPECTED_GPUS GPUs, found $gpu_count" >&2
  exit 1
fi
nvidia-smi -L
df -h "$WORK_DIR" "$RESULTS_MOUNT"
nsys --version

vllm_commit=$(checkout_revision "$SOURCE_DIR" "$SOURCE_REVISION")
benchmark_commit=$(checkout_revision \
  "$BENCHMARK_DIR" "$BENCHMARK_SOURCE_REVISION" benchmarks/overheads)
export VLLM_BUILD_COMMIT=$vllm_commit
export VLLM_BENCHMARK_REVISION=$benchmark_commit

git -C "$SOURCE_DIR" remote add upstream "$UPSTREAM_REPO_URL"
git -C "$SOURCE_DIR" fetch --filter=blob:none upstream main
merge_base=$(git -C "$SOURCE_DIR" merge-base HEAD upstream/main)
if [[ -n $(git -C "$SOURCE_DIR" diff --name-only "$merge_base" HEAD -- \
  CMakeLists.txt cmake csrc setup.py pyproject.toml rust vllm/vllm-rs) ]]; then
  echo "the selected vLLM revision changes compiled/build inputs" >&2
  echo "build an exact vLLM image instead of using precompiled extensions" >&2
  exit 1
fi

VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT=$merge_base \
  uv pip install --system --editable "$SOURCE_DIR" --torch-backend=auto

# The base image can contain newer FlashInfer binary packages than the revision
# under test. uv resolves flashinfer-python from the checkout but may leave the
# preinstalled cubin and JIT cache in place. FlashInfer requires every package
# to use the same release version.
flashinfer_cubin_version=$(sed -n 's/^flashinfer-cubin==//p' \
  "$SOURCE_DIR/requirements/cuda.txt")
[[ -n $flashinfer_cubin_version ]] || {
  echo "flashinfer-cubin is not pinned in requirements/cuda.txt" >&2
  exit 1
}
uv pip install --system \
  --extra-index-url https://flashinfer.ai/whl/ \
  "flashinfer-cubin==$flashinfer_cubin_version"
flashinfer_cuda_suffix=$(
  "$PYTHON" -c 'import torch; print("cu" + torch.version.cuda.replace(".", ""))'
)
uv pip install --system \
  --extra-index-url "https://flashinfer.ai/whl/$flashinfer_cuda_suffix/" \
  "flashinfer-jit-cache==$flashinfer_cubin_version+$flashinfer_cuda_suffix"

"$PYTHON" -c '
import triton.profiler as proton
import vllm

required = ("start", "activate", "deactivate", "finalize")
missing = [name for name in required if not hasattr(proton, name)]
if missing:
    raise RuntimeError(f"Triton Proton API is missing: {missing}")
print(f"vLLM {vllm.__version__}; Proton API ready")
'
printf 'vLLM source: %s at %s\n' "$REPO_URL" "$vllm_commit"
printf 'Benchmark source: %s at %s\n' "$REPO_URL" "$benchmark_commit"

if [[ ${VLLM_JOB_PREFLIGHT_ONLY:-0} == 1 ]]; then
  printf 'Preflight passed. Matrix arguments:'
  printf ' %q' "${BENCHMARK_ARGS[@]}"
  printf '\n'
  exit 0
fi

env -u HF_HUB_OFFLINE "$PYTHON" "$OVERHEAD_DIR/nebius/prepare_model_metadata.py" \
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

summarize_retained_proton_profiles
sync_results checkpoint
