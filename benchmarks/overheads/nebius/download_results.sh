#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

usage() {
  cat <<'EOF'
Download one profiler-overhead Job's artifacts from Nebius Object Storage.

Usage:
  download_results.sh JOB_NAME [DESTINATION]

Required environment:
  NEBIUS_BUCKET_NAME            Object Storage bucket name.

Optional environment:
  NEBIUS_REGION                 Nebius region (default: eu-north1).
  NEBIUS_S3_ENDPOINT            Override the Object Storage endpoint.

The access host must already have an AWS CLI profile configured by the
artifact maintainer. No Nebius or S3 credential setup is required by reviewers.
EOF
}

if (($# < 1 || $# > 2)); then
  usage >&2
  exit 2
fi

job_name=$1
destination=${2:-results/$job_name}
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

mkdir -p "$destination"
aws --endpoint-url "$endpoint" --region "$region" s3 sync \
  "s3://${bucket_name}/${job_name}/" "$destination"

printf 'Downloaded s3://%s/%s/ to %s\n' \
  "$bucket_name" "$job_name" "$destination"
if [[ -f $destination/job_status.json ]]; then
  printf 'Job status:\n'
  cat "$destination/job_status.json"
fi
