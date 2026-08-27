# Nebius Serverless AI Jobs

This wrapper submits the profiler-overhead benchmark as a one-shot Nebius Job.
It requests one eight-GPU node, runs the existing resumable matrix, persists
results, and releases the compute resources when the container exits.

## Image choice

No custom image is required. Use an official vLLM image; the submitter injects
the small Job entrypoint with Nebius `--inject-file`. At startup, it installs
missing profiling tools and clones the selected source revisions:

```bash
image="docker.io/vllm/vllm-openai:nightly"
```

For reproducible runs, resolve the tag once and submit its immutable digest.
The runtime defaults to `cuda-nsight-systems-13-0`; set
`NSIGHT_SYSTEMS_PACKAGE` when using a base image with another CUDA release.

For repeated Jobs, the optional `nebius/Dockerfile` preinstalls nsys, git, and
Python utilities to shorten startup. It still contains no PR or
benchmark source and therefore does not need rebuilding for each revision:

```bash
docker build -f benchmarks/overheads/nebius/Dockerfile \
  --build-arg "VLLM_IMAGE=$image" \
  -t "cr.<region>.nebius.cloud/<registry>/vllm-profiler:bootstrap-cu130" .
docker push \
  "cr.<region>.nebius.cloud/<registry>/vllm-profiler:bootstrap-cu130"
```

At Job startup, the runner clones the requested vLLM revision and installs it
in editable mode with `VLLM_USE_PRECOMPILED=1`. vLLM selects the PR merge-base
wheel and reuses its compiled extensions, so Python-only PRs do not trigger a
CUDA build. The runner refuses revisions that change common compiled/build
inputs; those revisions require an exact source-built image.

The merge-base is computed against `https://github.com/vllm-project/vllm.git`,
not the fork's `main` branch. Override it with `--upstream-repo-url` only when
the PR targets another upstream repository.

Use an immutable image digest for submitted runs. A registry in the same
Nebius project does not require credentials in the Job configuration.

## Prepare the access host

Artifact maintainers complete this section before giving the reviewer SSH
access.  The reviewer does not configure Nebius credentials or cloud
resources.  Attach a dedicated least-privilege service account to the VM so
the preinstalled Nebius CLI uses the VM instance identity.  Do not store a
service-account private key in the artifact or environment file.

VM instance identity authenticates the Nebius CLI, but AWS-compatible Object
Storage clients require an S3 access key.  Before handoff, the maintainer
installs AWS CLI and configures a time-limited access key for the service
account on the access host.  Set `NEBIUS_BUCKET_NAME` and `NEBIUS_REGION` in
`~/.config/llmprof-ae.env`.  The key must expire after the evaluation and must
not be committed or copied into the artifact.

The Job needs a shared filesystem, bucket ID/name, or `s3://bucket` mounted
read-write.  Copy `reviewer.env.example` to
`~/.config/llmprof-ae.env`, replace every placeholder with an immutable value,
and run the readiness check before handoff:

```bash
benchmarks/overheads/nebius/check_readiness.sh
```

The check verifies the pinned commit and image, CLI features, profile/project,
subnet, persistent storage, and exact submission command.  It ends with
`--dry-run` and creates no Job.  This is an author-side handoff check; the
reviewer directly runs the designated evaluation Job.

Use `--storage-mode filesystem` for a POSIX shared filesystem. The matrix runs
directly in the mounted path and is resumable across Jobs. Use
`--storage-mode object` for a bucket: active profiler files stay on the Job's
local disk, while every completed case (including retained profiles) is copied
periodically and all remaining artifacts are copied on exit.

Prefer a shared filesystem when finalizing and retaining every raw Torch/nsys
trace. Object Storage is a good default with `--profile-retention proton`:
the runtime metrics for Torch/nsys are preserved while only the Proton
profiles are retained.

For gated Hugging Face repositories, use a MysteryBox secret with a recent CLI
that supports `--env-secret`; do not put an HF token in `--env` or in a saved
Job specification.

## Reviewer submission

After the maintainer installs the supplied SSH public key, the reviewer logs
in to the prepared host, loads `~/.config/llmprof-ae.env`, and directly uses
the wrapper below.  No Nebius login or credential setup is required.

```bash
benchmarks/overheads/nebius/submit_job.sh \
  --image docker.io/vllm/vllm-openai@sha256:<digest> \
  --volume-source storagebucket-<id> \
  --storage-mode object \
  --hf-secret hf-token \
  --repo-url https://github.com/Luosuu/vllm.git \
  --vllm-revision <vllm-pr-commit-sha> \
  --benchmark-revision <benchmark-commit-sha> \
  --graph-modes cudagraph \
  -- \
  --models gpt-oss-20b \
  --workloads in2000_out500 \
  --tp-sizes 2 \
  --total-gpus 8 \
  --skip-ep-cases \
  --profilers proton torch nsys \
  --num-prompts 2048 --num-warmups 512 \
  --max-concurrency 256 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --repeats 3 \
  --profile-retention proton \
  --profile-save-timeout 600 \
  --min-free-gb 128
```

Branch names are accepted when intentionally testing their latest state, but
full commit hashes make separate Jobs reproducible. The runner records both
the requested refs and resolved commit hashes in `job_status.json` and matrix
environment metadata.

The default resource request is `gpu-h100-sxm` with the
`8gpu-128vcpu-1600gb` preset, a 1 TiB container disk, 64 GiB `/dev/shm`, and a
24-hour timeout. Override these with submission flags when the project exposes
another platform or preset. When changing the preset's GPU count, pass the
matching `--expected-gpus` value so the Job validates the intended allocation.

Add `--dry-run` to print the exact `nebius ai job create` command without
creating cloud resources. Add `--detach` to return after submission; otherwise
the wrapper follows logs and then prints the final Job state. Immediately after
creation, before following logs, the wrapper prints the Job ID, result prefix,
artifact-monitor command, and download command. The designated reviewer
workflow uses one Job; `--preflight-only` remains available for maintainer
troubleshooting.

For Object Storage, results are stored below
`<mounted-volume>/<Nebius-Job-ID>/`.  The submitter records the newly allocated
`aijob-*` ID in a unique handoff marker; the Job reads it before creating its
output directory. Concurrent submissions therefore never share a result
prefix. The directory contains one subdirectory per graph mode plus
`job_status.json`. Matrix outputs retain their existing layout, including
`results.csv`, `failures.csv`, plots, per-case JSON, and any profiles selected
by `--profile-retention`. Each retained Proton run also contains
`profile-summary.md`; the Job prints these summaries to its log before
synchronizing the final output.

The printed monitor command refreshes the recursive object listing every
30 seconds, so a reviewer can see completed cases and retained profiles arrive
while the Job is still running. On the prepared access host, download the
complete output using the Job ID printed by `submit_job.sh`:

```bash
source ~/.config/llmprof-ae.env
benchmarks/overheads/nebius/download_results.sh <aijob-id>
```

This synchronizes `s3://$NEBIUS_BUCKET_NAME/<aijob-id>/` to
`results/<aijob-id>/`.  The command prints `job_status.json` when present.
