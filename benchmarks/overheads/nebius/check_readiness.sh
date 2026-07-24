#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ENV_FILE=${1:-"$HOME/.config/llmprof-ae.env"}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -r $ENV_FILE ]] || fail "cannot read environment file: $ENV_FILE"
# shellcheck disable=SC1090
source "$ENV_FILE"

required_commands=(git jq nebius)
for command_name in "${required_commands[@]}"; do
  command -v "$command_name" >/dev/null ||
    fail "$command_name is required on PATH"
done

required_variables=(
  ARTIFACT_REPOSITORY
  ARTIFACT_COMMIT
  VLLM_REPO_URL
  NEBIUS_IMAGE
  NEBIUS_VOLUME_SOURCE
  NEBIUS_PARENT_ID
  NEBIUS_SUBNET_ID
)
for variable_name in "${required_variables[@]}"; do
  [[ -n ${!variable_name:-} ]] ||
    fail "$variable_name is missing from $ENV_FILE"
done

[[ $ARTIFACT_COMMIT =~ ^[0-9a-f]{40}$ ]] ||
  fail "ARTIFACT_COMMIT must be a full 40-character commit"
[[ $NEBIUS_IMAGE =~ @sha256:[0-9a-f]{64}$ ]] ||
  fail "NEBIUS_IMAGE must use an immutable sha256 digest"
[[ -d $ARTIFACT_REPOSITORY/.git ]] ||
  fail "ARTIFACT_REPOSITORY is not a Git checkout"
git -C "$ARTIFACT_REPOSITORY" cat-file -e "$ARTIFACT_COMMIT^{commit}" ||
  fail "ARTIFACT_COMMIT is absent from the local checkout"
[[ -x $ARTIFACT_REPOSITORY/benchmarks/overheads/nebius/submit_job.sh ]] ||
  fail "submit_job.sh is absent or not executable at ARTIFACT_COMMIT"

nebius_cmd=(nebius)
[[ -z ${NEBIUS_PROFILE:-} ]] ||
  nebius_cmd+=(--profile "$NEBIUS_PROFILE")

create_help=$("${nebius_cmd[@]}" ai job create --help 2>&1)
for option_name in --inject-file --args --env-secret --registry-secret; do
  [[ $create_help == *"$option_name"* ]] ||
    fail "Nebius CLI does not support $option_name"
done

if [[ -n ${NEBIUS_PROFILE:-} ]]; then
  profile_parent=$(
    "${nebius_cmd[@]}" config get parent-id 2>/dev/null || true
  )
  [[ $profile_parent == "$NEBIUS_PARENT_ID" ]] ||
    fail "profile parent-id does not match NEBIUS_PARENT_ID"
fi

"${nebius_cmd[@]}" vpc subnet get "$NEBIUS_SUBNET_ID" \
  --format json |
  jq -e '.status.state == "READY"' >/dev/null ||
  fail "subnet is not readable and READY"

case "$NEBIUS_VOLUME_SOURCE" in
  storagebucket-*)
    [[ ${NEBIUS_STORAGE_MODE:-object} == object ]] ||
      fail "a storage bucket requires NEBIUS_STORAGE_MODE=object"
    command -v aws >/dev/null || fail "aws is required on PATH"
    [[ -n ${NEBIUS_BUCKET_NAME:-} ]] ||
      fail "NEBIUS_BUCKET_NAME is missing from $ENV_FILE"
    [[ -n ${NEBIUS_REGION:-} ]] ||
      fail "NEBIUS_REGION is missing from $ENV_FILE"
    bucket_json=$("${nebius_cmd[@]}" storage bucket \
      get "$NEBIUS_VOLUME_SOURCE" --format json)
    jq -e '.status.state == "ACTIVE"' <<<"$bucket_json" >/dev/null ||
      fail "storage bucket is not readable and ACTIVE"
    [[ $(jq -r '.metadata.name' <<<"$bucket_json") == \
      "$NEBIUS_BUCKET_NAME" ]] ||
      fail "NEBIUS_BUCKET_NAME does not match NEBIUS_VOLUME_SOURCE"
    [[ $(jq -r '.status.region' <<<"$bucket_json") == "$NEBIUS_REGION" ]] ||
      fail "NEBIUS_REGION does not match the storage bucket"
    s3_endpoint=${NEBIUS_S3_ENDPOINT:-https://storage.${NEBIUS_REGION}.nebius.cloud}
    aws --endpoint-url "$s3_endpoint" --region "$NEBIUS_REGION" \
      s3api list-objects-v2 --bucket "$NEBIUS_BUCKET_NAME" \
      --max-keys 1 >/dev/null ||
      fail "AWS CLI cannot read the Object Storage bucket"
    ;;
  computefilesystem-*)
    [[ ${NEBIUS_STORAGE_MODE:-filesystem} == filesystem ]] ||
      fail "a compute filesystem requires NEBIUS_STORAGE_MODE=filesystem"
    "${nebius_cmd[@]}" compute filesystem \
      get "$NEBIUS_VOLUME_SOURCE" --format json |
      jq -e '.status.state == "READY"' >/dev/null ||
      fail "compute filesystem is not readable and READY"
    ;;
  *)
    fail "unsupported NEBIUS_VOLUME_SOURCE identifier"
    ;;
esac

dry_run_output=$(mktemp)
trap 'rm -f "$dry_run_output"' EXIT
submit_args=()
[[ -z ${NEBIUS_PROFILE:-} ]] ||
  submit_args+=(--profile "$NEBIUS_PROFILE")
"$ARTIFACT_REPOSITORY/benchmarks/overheads/nebius/submit_job.sh" \
  "${submit_args[@]}" \
  --parent-id "$NEBIUS_PARENT_ID" \
  --subnet-id "$NEBIUS_SUBNET_ID" \
  --image "$NEBIUS_IMAGE" \
  --volume-source "$NEBIUS_VOLUME_SOURCE" \
  --storage-mode "${NEBIUS_STORAGE_MODE:-object}" \
  --platform "${NEBIUS_PLATFORM:-gpu-h100-sxm}" \
  --preset "${NEBIUS_PRESET:-8gpu-128vcpu-1600gb}" \
  --disk-size "${NEBIUS_DISK_SIZE:-1Ti}" \
  --shm-size "${NEBIUS_SHM_SIZE:-64Gi}" \
  --timeout "${NEBIUS_JOB_TIMEOUT:-24h}" \
  --sync-interval "${NEBIUS_SYNC_INTERVAL:-300}" \
  --repo-url "$VLLM_REPO_URL" \
  --vllm-revision "$ARTIFACT_COMMIT" \
  --benchmark-revision "$ARTIFACT_COMMIT" \
  --graph-modes cudagraph \
  --preflight-only \
  --dry-run \
  -- \
  --models gpt-oss-20b \
  --workloads in2000_out500 \
  --tp-sizes 2 --total-gpus 8 \
  --skip-ep-cases --profilers proton \
  --num-prompts 16 --num-warmups 4 \
  --max-concurrency 8 --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --repeats 1 --max-cases 1 >"$dry_run_output"

grep -q 'ai job create' "$dry_run_output" ||
  fail "submitter dry run did not produce a Job create command"
echo "Readiness checks passed; no cloud resources were created."
