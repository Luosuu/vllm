---
name: run-nebius-vllm-profiler-overhead
description: Submit, monitor, resume, and validate vLLM profiler-overhead benchmark matrices on Nebius Serverless AI Jobs. Use when running Proton, PyTorch profiler, or Nsight Systems overhead experiments on Nebius GPUs; selecting Object Storage or a shared filesystem; pinning vLLM and benchmark revisions; checking job progress and failures; or retrieving reproducible benchmark artifacts.
---

# Run Nebius vLLM Profiler Overhead

Operate the repository's Nebius wrapper rather than assembling `nebius ai job
create` manually. Keep cloud mutations explicit: a dry run and read-only status
checks are safe defaults, but submit or cancel a paid Job only when the user has
asked for it.

## Read the source of truth

Work from the repository root. Read these files before changing or submitting a
Job because their flags and output contract may evolve with the benchmark
branch:

- `benchmarks/overheads/nebius/README.md`
- `benchmarks/overheads/README.md`
- `benchmarks/overheads/nebius/submit_job.sh`

Use `submit_job.sh --help` to confirm the checked-out interface. Do not copy the
examples blindly when the user's requested matrix differs.

## Establish an immutable experiment

1. Record the model set, workloads, profilers, graph modes, GPU topology,
   concurrency, scheduler limits, repetitions, trace retention, and save policy.
2. Resolve both the vLLM-under-test revision and benchmark revision to full
   commit hashes. Prefer `git ls-remote <fork> <ref>` for remote state and
   `git rev-parse <ref>` for a local checkout.
3. Confirm the selected vLLM revision is compatible with the wrapper's
   precompiled-wheel path. The Job rejects revisions that modify compiled or
   build inputs; use an exact source-built image for those revisions.
4. Use an immutable image digest for a reproducible run. A nightly tag is
   acceptable for an exploratory preflight, but record that it is mutable.
5. Never put Hugging Face or registry credentials in plain `--env` values. Use
   `--hf-secret` and `--registry-secret`.

For eight GPUs with `TP=2`, set `--total-gpus 8 --tp-sizes 2`; the matrix derives
`DP=4`. Treat `--max-concurrency` and `--max-num-seqs` as global inputs. Verify
that the recorded per-engine `max_num_seqs` is the local capacity (for example,
64 when global concurrency is 256 and DP is 4). Always record
`max_num_batched_tokens` and chunked-prefill state.

## Validate Nebius access and persistence

Before reviewer handoff, the artifact maintainer runs these checks on the
prepared access host:

```bash
command -v nebius
nebius version
nebius ai job create --help
benchmarks/overheads/nebius/check_readiness.sh
```

Confirm that the CLI exposes `--inject-file`, `--args`, and any requested secret
or preemptible flags. Confirm the project, subnet, GPU platform/preset, and
writable storage source before submission. Do not print a full CLI profile:
it can contain private-key material or credential paths. The readiness script
queries only non-secret fields and ends with a submission dry run.

The reviewer should not configure Nebius authentication.  The access-host VM
uses an attached least-privilege service account as its instance identity.
After SSH access is installed, reviewers load the prepared environment and
directly run the documented AE submissions.

Choose persistence deliberately:

- Use `--storage-mode object` for periodic completed-case checkpoints and a
  final copy from local Job disk. This is the normal choice when Torch/nsys
  finalization is skipped.
- Use `--storage-mode filesystem` when the matrix must run directly on a POSIX
  shared filesystem or retain very large finalized traces.

Results live below `<volume-source>/<job-name>/`. Never claim completion from
the cloud Job state alone; validate `job_status.json`, case counts, and failure
records in persistent storage.

## Submit in stages

First print the exact command without allocating GPUs:

```bash
benchmarks/overheads/nebius/submit_job.sh \
  --image '<official-vllm-image@sha256:digest>' \
  --volume-source '<bucket-or-filesystem>' \
  --storage-mode object \
  --repo-url '<vllm-fork-url>' \
  --vllm-revision '<full-vllm-commit>' \
  --benchmark-revision '<full-benchmark-commit>' \
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
  --repeats 1 --max-cases 1
```

Keep the readiness invocation as a dry run during access-host handoff. The
reviewer directly submits the designated evaluation Job; use a paid preflight
only while troubleshooting a failed environment.

For the established eight-GPU TP2 runtime-overhead matrix, use this argument
shape unless the user requests another experiment:

```text
--graph-modes cudagraph
--models gpt-oss-20b
--workloads in2000_out500
--tp-sizes 2 --total-gpus 8 --skip-ep-cases
--profilers proton torch nsys
--num-prompts 2048 --num-warmups 512 --max-concurrency 256
--max-num-seqs 256 --max-num-batched-tokens 8192
--repeats 3
--profile-save-timeout 600 --profile-retention proton
```

Keep submission flags before `--` and matrix-runner flags after it. Use
`--detach` for asynchronous Jobs and preserve the printed Job ID and job name.

## Monitor without disturbing the run

Use compact read-only snapshots:

```bash
nebius ai job get "$JOB_ID" --format json \
  | jq '{id:.metadata.id,name:.metadata.name,state:.status.state,instances:.status.instances}'
nebius ai job logs "$JOB_ID" --since 15m --timestamps | tail -n 100
```

Extract the latest `[N/TOTAL] RUN`, PASS/FAIL counts, current case, last log
timestamp, and cloud state. Distinguish these conditions:

- `RUNNING` with recent progress or active GPU work: healthy;
- no new completed case while a long generation is active: possibly healthy;
- repeated server initialization, CUDA, or request-completeness errors: case
  failure requiring diagnosis;
- terminal cloud state without `job_status.json` and final checkpoint: artifact
  persistence failure or interrupted final sync.

Do not cancel solely because a case is slower than the average. Compare against
the configured per-case timeout and inspect the current server/benchmark logs.
Cancel only when the user requested it or continued execution is clearly
unproductive and cancellation is within the granted scope.

## Validate and report results

After completion or interruption, inspect persistent artifacts:

```bash
jq . '<results>/<job-name>/job_status.json'
find '<results>/<job-name>' -name case.json -type f | wc -l
find '<results>/<job-name>' -name case.json -type f -print0 \
  | xargs -0 jq -r '.succeeded' | sort | uniq -c
find '<results>/<job-name>' -name failures.csv -type f -size +0c -print
```

Check each graph-mode directory for `environment.json`, `results.csv`, plots,
and expected cases. Confirm that requested and resolved vLLM/benchmark commits
in metadata match the submitted hashes. Classify failures as benchmark runtime,
request completeness, server startup, profiler collection, profiler save, or
artifact sync failures; do not collapse them into one count.

Report the Job ID/name, immutable revisions and image, resource topology,
storage destination, completed/PASS/FAIL case counts, current or final state,
and any reproducibility caveats. Compute profiler overhead only from metrics
emitted by `vllm bench serve`, not wrapper wall-clock or trace-save time.
