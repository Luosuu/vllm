#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

usage() {
  cat <<'EOF'
Download one profiler-overhead Job's artifacts from Nebius Object Storage.

Usage:
  download_results.sh JOB_ID [DESTINATION]
  download_results.sh --list JOB_ID

Required environment:
  NEBIUS_BUCKET_NAME            Object Storage bucket name.

Optional environment:
  NEBIUS_REGION                 Nebius region (default: eu-north1).
  NEBIUS_S3_ENDPOINT            Override the Object Storage endpoint.

The access host must already have an AWS CLI profile configured by the
artifact maintainer. No Nebius or S3 credential setup is required by reviewers.
EOF
}

list_only=0
if [[ ${1:-} == --list ]]; then
  list_only=1
  shift
fi

if (($# < 1 || $# > 2 || (list_only == 1 && $# != 1))); then
  usage >&2
  exit 2
fi

job_id=$1
destination=${2:-results/$job_id}
bucket_name=${NEBIUS_BUCKET_NAME:-}
region=${NEBIUS_REGION:-eu-north1}
endpoint=${NEBIUS_S3_ENDPOINT:-https://storage.${region}.nebius.cloud}

command -v aws >/dev/null || {
  echo "aws CLI is required on the prepared access host" >&2
  exit 1
}
[[ -n $bucket_name ]] || {
  echo "NEBIUS_BUCKET_NAME is required" >&2
  exit 2
}
[[ $job_id =~ ^aijob-[a-z0-9]+$ ]] || {
  echo "JOB_ID must be a Nebius aijob-* resource ID" >&2
  exit 2
}

if [[ $list_only == 1 ]]; then
  aws --endpoint-url "$endpoint" --region "$region" s3 ls \
    "s3://${bucket_name}/${job_id}/" --recursive
  exit
fi

mkdir -p "$destination"
aws --endpoint-url "$endpoint" --region "$region" s3 sync \
  "s3://${bucket_name}/${job_id}/" "$destination"

printf 'Downloaded s3://%s/%s/ to %s\n' \
  "$bucket_name" "$job_id" "$destination"
if [[ -f $destination/job_status.json ]]; then
  printf 'Job status:\n'
  cat "$destination/job_status.json"
fi
