# Nebius Serverless AI Jobs

This wrapper submits the profiler-overhead benchmark as a one-shot Nebius Job.
It requests one eight-GPU node, runs the existing resumable matrix, persists
results, and releases the compute resources when the container exits.

## Build and push the image

Build the vLLM image from the exact source revision under test, then add the
benchmark Job layer:

```bash
revision=$(git rev-parse --short=12 HEAD)
docker build --target vllm-openai -f docker/Dockerfile \
  --build-arg VLLM_BUILD_COMMIT="$revision" \
  -t "vllm-profiler-base:$revision" .
docker build -f benchmarks/overheads/nebius/Dockerfile \
  --build-arg "VLLM_IMAGE=vllm-profiler-base:$revision" \
  -t "cr.<region>.nebius.cloud/<registry>/vllm-profiler:$revision" .
docker push "cr.<region>.nebius.cloud/<registry>/vllm-profiler:$revision"
```

The Job layer defaults to the `cuda-nsight-systems-13-0` package matching this
vLLM branch's CUDA 13 image. Override `NSIGHT_SYSTEMS_PACKAGE` at build time if
the selected base image uses another CUDA release.

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
  --image cr.<region>.nebius.cloud/<registry>/vllm-profiler@sha256:<digest> \
  --volume-source storagebucket-<id> \
  --storage-mode object \
  --hf-secret hf-token \
  --graph-modes "cudagraph eager" \
  -- \
  --models gpt-oss-20b gpt-oss-120b llama-3.1-8b \
  --workloads in2000_out500 in1000_out1000 in500_out2000 \
  --tp-sizes 2 \
  --total-gpus 8 \
  --skip-ep-cases \
  --profilers proton torch nsys \
  --repeats 1 \
  --no-finalize-non-proton \
  --profile-retention all
```

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
