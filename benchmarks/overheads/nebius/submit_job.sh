#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

usage() {
  cat <<'EOF'
Submit the vLLM profiler-overhead matrix as a Nebius Serverless AI Job.

Usage:
  submit_job.sh [submission options] -- [run_profiler_matrix.py options]

Required:
  --image IMAGE                 Official/prewarmed vLLM image (prefer a digest).
  --volume-source SOURCE        Filesystem/bucket ID, name, or s3://bucket.

Submission options:
  --storage-mode MODE           filesystem or object (default: object).
  --volume-profile PROFILE      S3 credential profile/secret selector.
  --profile PROFILE             Nebius CLI profile.
  --parent-id ID                Nebius project ID; profile default if omitted.
  --subnet-id ID                Subnet ID; otherwise resolve --subnet-name.
  --subnet-name NAME            Subnet name (default: default-subnet).
  --platform PLATFORM           Default: gpu-h100-sxm.
  --preset PRESET               Default: 8gpu-128vcpu-1600gb.
  --disk-size SIZE              Default: 1Ti.
  --shm-size SIZE               Default: 64Gi.
  --timeout DURATION            Default: 24h.
  --job-name NAME               Default: vllm-profile-<UTC timestamp>.
  --graph-modes "MODES"         Default: cudagraph.
  --sync-interval SECONDS       Object-store checkpoint interval (default: 300).
  --repo-url URL                vLLM fork to clone in the Job.
  --upstream-repo-url URL       Upstream used to find the PR merge-base.
  --vllm-revision REV           vLLM PR branch/commit under test.
  --benchmark-revision REV      Benchmark branch/commit to run.
  --hf-secret SELECTOR          MysteryBox secret for HF_TOKEN.
  --registry-secret SELECTOR    MysteryBox registry credential secret.
  --preemptible                 Request a preemptible VM.
  --preflight-only              Validate the image and GPUs without benchmarking.
  --detach                      Do not follow logs after submission.
  --dry-run                     Print the shell-escaped create command only.
  -h, --help                    Show this help.

The corresponding NEBIUS_* environment variables can provide every option.
EOF
}

image=${NEBIUS_IMAGE:-}
volume_source=${NEBIUS_VOLUME_SOURCE:-}
storage_mode=${NEBIUS_STORAGE_MODE:-object}
volume_profile=${NEBIUS_VOLUME_PROFILE:-}
profile=${NEBIUS_PROFILE:-}
parent_id=${NEBIUS_PARENT_ID:-}
subnet_id=${NEBIUS_SUBNET_ID:-}
subnet_name=${NEBIUS_SUBNET_NAME:-default-subnet}
platform=${NEBIUS_PLATFORM:-gpu-h100-sxm}
preset=${NEBIUS_PRESET:-8gpu-128vcpu-1600gb}
disk_size=${NEBIUS_DISK_SIZE:-1Ti}
shm_size=${NEBIUS_SHM_SIZE:-64Gi}
job_timeout=${NEBIUS_JOB_TIMEOUT:-24h}
job_name=${NEBIUS_JOB_NAME:-vllm-profile-$(date -u +%Y%m%d-%H%M%S)-$(printf '%04x%04x' "$RANDOM" "$RANDOM")}
graph_modes=${NEBIUS_GRAPH_MODES:-cudagraph}
sync_interval=${NEBIUS_SYNC_INTERVAL:-300}
repo_url=${NEBIUS_VLLM_REPO_URL:-https://github.com/Luosuu/vllm.git}
upstream_repo_url=${NEBIUS_VLLM_UPSTREAM_REPO_URL:-https://github.com/vllm-project/vllm.git}
vllm_revision=${NEBIUS_VLLM_REVISION:-proton-profiler-clean}
benchmark_revision=${NEBIUS_BENCHMARK_REVISION:-profiler-overhead-benchmarks}
hf_secret=${NEBIUS_HF_SECRET:-}
registry_secret=${NEBIUS_REGISTRY_SECRET:-}
preemptible=${NEBIUS_PREEMPTIBLE:-0}
preflight_only=${NEBIUS_PREFLIGHT_ONLY:-0}
detach=${NEBIUS_DETACH:-0}
dry_run=0

while (($#)); do
  case "$1" in
    --image) image=$2; shift 2 ;;
    --volume-source) volume_source=$2; shift 2 ;;
    --storage-mode) storage_mode=$2; shift 2 ;;
    --volume-profile) volume_profile=$2; shift 2 ;;
    --profile) profile=$2; shift 2 ;;
    --parent-id) parent_id=$2; shift 2 ;;
    --subnet-id) subnet_id=$2; shift 2 ;;
    --subnet-name) subnet_name=$2; shift 2 ;;
    --platform) platform=$2; shift 2 ;;
    --preset) preset=$2; shift 2 ;;
    --disk-size) disk_size=$2; shift 2 ;;
    --shm-size) shm_size=$2; shift 2 ;;
    --timeout) job_timeout=$2; shift 2 ;;
    --job-name) job_name=$2; shift 2 ;;
    --graph-modes) graph_modes=$2; shift 2 ;;
    --sync-interval) sync_interval=$2; shift 2 ;;
    --repo-url) repo_url=$2; shift 2 ;;
    --upstream-repo-url) upstream_repo_url=$2; shift 2 ;;
    --vllm-revision) vllm_revision=$2; shift 2 ;;
    --benchmark-revision) benchmark_revision=$2; shift 2 ;;
    --hf-secret) hf_secret=$2; shift 2 ;;
    --registry-secret) registry_secret=$2; shift 2 ;;
    --preemptible) preemptible=1; shift ;;
    --preflight-only) preflight_only=1; shift ;;
    --detach) detach=1; shift ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) echo "unknown submission option: $1" >&2; usage >&2; exit 2 ;;
  esac
done
benchmark_args=("$@")

command -v nebius >/dev/null || {
  echo "nebius CLI is required" >&2
  exit 1
}
command -v jq >/dev/null || {
  echo "jq is required" >&2
  exit 1
}
[[ -n $image ]] || { echo "--image is required" >&2; exit 2; }
[[ -n $repo_url ]] || { echo "--repo-url is required" >&2; exit 2; }
[[ -n $upstream_repo_url ]] || {
  echo "--upstream-repo-url is required" >&2
  exit 2
}
[[ -n $vllm_revision ]] || { echo "--vllm-revision is required" >&2; exit 2; }
[[ -n $benchmark_revision ]] || {
  echo "--benchmark-revision is required" >&2
  exit 2
}
[[ -n $volume_source ]] || {
  echo "--volume-source is required so results survive the Job" >&2
  exit 2
}
[[ $storage_mode == filesystem || $storage_mode == object ]] || {
  echo "--storage-mode must be filesystem or object" >&2
  exit 2
}
if [[ $storage_mode == object && -n ${NEBIUS_BUCKET_NAME:-} ]]; then
  command -v aws >/dev/null || {
    echo "aws CLI is required to initialize the per-Job result prefix" >&2
    exit 1
  }
fi
[[ $sync_interval =~ ^[0-9]+$ ]] || {
  echo "--sync-interval must be a non-negative integer" >&2
  exit 2
}

nebius_cmd=(nebius)
[[ -z $profile ]] || nebius_cmd+=(--profile "$profile")

create_help=$("${nebius_cmd[@]}" ai job create --help 2>&1)
if [[ $create_help != *--inject-file* || $create_help != *--args* ]]; then
  echo "this nebius CLI lacks --inject-file/--args; update it first" >&2
  exit 1
fi
if [[ $create_help != *--async* ]]; then
  echo "this nebius CLI lacks --async; update it first" >&2
  exit 1
fi
if [[ -n $hf_secret && $create_help != *--env-secret* ]]; then
  echo "this nebius CLI lacks --env-secret; update it before passing HF_TOKEN" >&2
  exit 1
fi
if [[ -n $registry_secret && $create_help != *--registry-secret* ]]; then
  echo "this nebius CLI lacks --registry-secret; update it first" >&2
  exit 1
fi
if [[ $preemptible == 1 && $create_help != *--preemptible* ]]; then
  echo "this nebius CLI lacks --preemptible; update it first" >&2
  exit 1
fi

if [[ -z $subnet_id ]]; then
  if [[ $dry_run == 1 ]]; then
    subnet_id='<resolved-subnet-id>'
  else
    subnet_id=$("${nebius_cmd[@]}" vpc subnet get-by-name \
      --name "$subnet_name" --format jsonpath='{.metadata.id}' 2>/dev/null || true)
    if [[ -z $subnet_id ]]; then
      resolved_parent=$parent_id
      if [[ -z $resolved_parent ]]; then
        resolved_parent=$("${nebius_cmd[@]}" config get parent-id)
      fi
      subnets=$("${nebius_cmd[@]}" vpc subnet list \
        --parent-id "$resolved_parent" --all --format json)
      subnet_count=$(jq '.items | length' <<<"$subnets")
      if [[ $subnet_count == 1 ]]; then
        subnet_id=$(jq -er '.items[0].metadata.id' <<<"$subnets")
      else
        echo "subnet '$subnet_name' was not found and the project has $subnet_count subnets:" >&2
        jq -r '.items[] | "  \(.metadata.name) (\(.metadata.id))"' \
          <<<"$subnets" >&2
        echo "pass --subnet-id or --subnet-name explicitly" >&2
        exit 1
      fi
    fi
  fi
fi

benchmark_args_b64=
if ((${#benchmark_args[@]})); then
  benchmark_args_b64=$(printf '%s\0' "${benchmark_args[@]}" | base64 | tr -d '\n')
fi
results_mount=/mnt/vllm-profile-results
volume_spec="${volume_source}:${results_mount}:rw"
job_id_marker=
if [[ $storage_mode == object && -n ${NEBIUS_BUCKET_NAME:-} ]]; then
  job_id_marker=".llmprof-submissions/${job_name}/job-id"
fi
if [[ -n $volume_profile ]]; then
  volume_spec+=":${volume_profile}"
fi

create=(
  "${nebius_cmd[@]}" ai job create
  --name "$job_name"
  --image "$image"
  --platform "$platform"
  --preset "$preset"
  --disk-size "$disk_size"
  --shm-size "$shm_size"
  --timeout "$job_timeout"
  --subnet-id "$subnet_id"
  --restart-policy never
  --volume "$volume_spec"
  --inject-file "$SCRIPT_DIR/run_job.sh:/opt/vllm-job/run_job.sh"
  --container-command /bin/bash
  --args /opt/vllm-job/run_job.sh
  --env "VLLM_BENCHMARK_ARGS_B64=${benchmark_args_b64}"
  --env "VLLM_REPO_URL=${repo_url}"
  --env "VLLM_UPSTREAM_REPO_URL=${upstream_repo_url}"
  --env "VLLM_SOURCE_REVISION=${vllm_revision}"
  --env "VLLM_BENCHMARK_SOURCE_REVISION=${benchmark_revision}"
  --env "VLLM_GRAPH_MODES=${graph_modes}"
  --env "VLLM_STORAGE_MODE=${storage_mode}"
  --env "VLLM_RESULTS_MOUNT=${results_mount}"
  --env "VLLM_JOB_NAME=${job_name}"
  --env "VLLM_JOB_ID_MARKER=${job_id_marker}"
  --env "VLLM_SYNC_INTERVAL=${sync_interval}"
  --env "VLLM_EXPECTED_GPUS=8"
  --env "VLLM_JOB_PREFLIGHT_ONLY=${preflight_only}"
  --env "VLLM_WORKER_MULTIPROC_METHOD=spawn"
  --env "NEBIUS_PLATFORM=${platform}"
  --env "NEBIUS_PRESET=${preset}"
)
[[ -z $parent_id ]] || create+=(--parent-id "$parent_id")
[[ -z $hf_secret ]] || create+=(--env-secret "HF_TOKEN=${hf_secret}")
[[ -z $registry_secret ]] || create+=(--registry-secret "$registry_secret")
[[ $preemptible == 0 ]] || create+=(--preemptible)
create+=(--async --format json)

if [[ $dry_run == 1 ]]; then
  printf '%q ' "${create[@]}"
  printf '\n'
  exit 0
fi

response=$("${create[@]}")
job_id=
list_parent=$parent_id
[[ -n $list_parent ]] || list_parent=$("${nebius_cmd[@]}" config get parent-id)
deadline=$((SECONDS + 120))
while ((SECONDS < deadline)); do
  jobs=$("${nebius_cmd[@]}" ai job list \
    --parent-id "$list_parent" --format json)
  job_id=$(jq -r --arg name "$job_name" '
    [.items[] | select(.metadata.name == $name)]
    | sort_by(.metadata.created_at) | last | .metadata.id // empty
  ' <<<"$jobs")
  [[ -z $job_id ]] || break
  sleep 2
done
if [[ -z $job_id ]]; then
  echo "could not find the asynchronously created Job '$job_name':" >&2
  printf '%s\n' "$response" >&2
  exit 1
fi
printf 'Submitted Nebius Job %s (%s)\n' "$job_name" "$job_id"
printf 'Status:  nebius ai job get %q\n' "$job_id"
printf 'Logs:    nebius ai job logs %q --follow\n' "$job_id"

result_id=$job_name
if [[ -n $job_id_marker ]]; then
  s3_region=${NEBIUS_REGION:-eu-north1}
  s3_endpoint=${NEBIUS_S3_ENDPOINT:-https://storage.${s3_region}.nebius.cloud}
  printf '%s\n' "$job_id" |
    aws --endpoint-url "$s3_endpoint" --region "$s3_region" s3 cp - \
      "s3://${NEBIUS_BUCKET_NAME}/${job_id_marker}" --only-show-errors
  result_id=$job_id
fi

printf 'Results: %s/%s\n' "$volume_source" "$result_id"
if [[ $storage_mode == object && -n ${NEBIUS_BUCKET_NAME:-} ]]; then
  printf 'Monitor: watch -n 30 %q --list %q\n' \
    "$SCRIPT_DIR/download_results.sh" "$result_id"
  printf 'Download: %q %q\n' \
    "$SCRIPT_DIR/download_results.sh" "$result_id"
fi

if [[ $detach == 0 ]]; then
  "${nebius_cmd[@]}" ai job logs "$job_id" --follow --timestamps
  "${nebius_cmd[@]}" ai job get "$job_id"
fi
