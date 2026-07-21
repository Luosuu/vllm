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

For repeated Jobs, the optional `nebius/Dockerfile` preinstalls nsys, git,
rsync, and Python utilities to shorten startup. It still contains no PR or
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

## Configure the CLI and storage

Create a Nebius CLI profile and select a project before submission. The Job
needs a shared filesystem, bucket ID/name, or `s3://bucket` mounted read-write.

Use `--storage-mode filesystem` for a POSIX shared filesystem. The matrix runs
directly in the mounted path and is resumable across Jobs. Use
`--storage-mode object` for a bucket: active profiler files stay on the Job's
local disk, while every completed case (including retained profiles) is copied
periodically and all remaining artifacts are copied on exit.

Prefer a shared filesystem when finalizing and retaining every raw Torch/nsys
trace. Object Storage is a good default with `--no-finalize-non-proton` and
`--profile-retention all`: the runtime metrics for Torch/nsys are preserved,
while only the finalized Proton traces need the final bulk copy.

For gated Hugging Face repositories, use a MysteryBox secret with a recent CLI
that supports `--env-secret`; do not put an HF token in `--env`.

## Submit

```bash
benchmarks/overheads/nebius/submit_job.sh \
  --image docker.io/vllm/vllm-openai@sha256:<digest> \
  --volume-source storagebucket-<id> \
  --storage-mode object \
  --hf-secret hf-token \
  --repo-url https://github.com/Luosuu/vllm.git \
  --vllm-revision <vllm-pr-commit-sha> \
  --benchmark-revision <benchmark-commit-sha> \
  --graph-modes "cudagraph eager" \
  -- \
  --models gpt-oss-20b gpt-oss-120b mixtral-8x7b qwen3-32b \
  --workloads in2000_out500 in1000_out1000 in500_out2000 \
  --tp-sizes 2 \
  --total-gpus 8 \
  --skip-ep-cases \
  --profilers proton torch nsys \
  --repeats 1 \
  --no-finalize-non-proton \
  --profile-retention all
```

Branch names are accepted when intentionally testing their latest state, but
full commit hashes make separate Jobs reproducible. The runner records both
the requested refs and resolved commit hashes in `job_status.json` and matrix
environment metadata.

The default resource request is `gpu-h100-sxm` with the
`8gpu-128vcpu-1600gb` preset, a 1 TiB container disk, 64 GiB `/dev/shm`, and a
24-hour timeout. Override these with submission flags when the project exposes
another platform or preset.

Add `--dry-run` to print the exact `nebius ai job create` command without
creating cloud resources. Add `--detach` to return after submission; otherwise
the wrapper follows logs and then prints the final Job state. Use
`--preflight-only` for the first cloud submission to validate the image,
eight-GPU allocation, Proton API, Nsight Systems, and mounted storage without
starting a benchmark.

Results are stored below `<mounted-volume>/<job-name>/`. The directory contains
one subdirectory per graph mode plus `job_status.json`. Matrix outputs retain
their existing layout, including `results.csv`, `failures.csv`, plots, per-case
JSON, and any profiles selected by `--profile-retention`.
