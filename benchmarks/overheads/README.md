# Profiler overhead benchmarks

These scripts measure the serving overhead of the Proton, PyTorch, Nsight
Systems, and rocprofv3 profilers with metrics reported by `vllm bench serve`.
They support tensor parallelism (TP), data parallelism (DP), expert
parallelism (EP), eager execution, and CUDA graphs.

The benchmark matrix uses dummy model weights by default. It downloads only
model configuration and tokenizer files, so it can exercise the real model
architecture and kernels without downloading full checkpoints.

## Files

- `benchmark_serving_profiler.py` runs one model and parallel configuration.
  It pairs unprofiled and profiled runs and produces per-run and summarized
  results.
- `run_profiler_matrix.py` expands models, workloads, parallel configurations,
  graph modes, and profiler backends into a resumable matrix. It also combines
  results into CSV files and plots.
- `setup_profiler_matrix.sh` creates a virtual environment, installs vLLM and
  plotting dependencies, validates Proton and Nsight Systems, and caches the
  required Hugging Face metadata.
- `nebius/check_readiness.sh` validates a pinned reviewer configuration and
  renders the exact Nebius submission without creating cloud resources.
- `skills/run-nebius-vllm-profiler-overhead/` gives an agent a reproducible
  workflow for submitting, monitoring, resuming, and validating the matrix on
  Nebius Serverless AI Jobs.

Run all commands from the repository root.

## Requirements

- Linux with NVIDIA GPUs, or AMD GPUs with ROCm (see the AMD notes below).
- `uv` available on `PATH`.
- Nsight Systems CLI (`nsys`) on `PATH` (NVIDIA), or `rocprofv3` on `PATH`
  (AMD, part of ROCm).
- A Triton build that provides `triton.profiler` (Proton).
- Hugging Face access to the selected model repositories. Llama models may
  require accepting the license and authenticating with `hf auth login`.
- Enough local storage for profiler output. PyTorch and nsys traces can be
  large even though model weights are not downloaded.

Prepare the default environment:

```bash
./benchmarks/overheads/setup_profiler_matrix.sh
```

The default environment is `.venv-profiler-overhead`. Override it or select a
precompiled vLLM build with environment variables:

```bash
VENV=/path/to/venv \
PYTHON_VERSION=3.12 \
VLLM_BUILD_MODE=precompiled \
./benchmarks/overheads/setup_profiler_matrix.sh
```

`VLLM_BUILD_MODE` accepts `source` (the default) or `precompiled`.

## Quick smoke test

Start with one small case before launching the full matrix:

```bash
.venv-profiler-overhead/bin/python \
  benchmarks/overheads/run_profiler_matrix.py \
  --models llama-3.1-8b \
  --workloads in2000_out500 \
  --tp-sizes 2 \
  --total-gpus 8 \
  --profilers proton \
  --num-prompts 16 \
  --num-warmups 4 \
  --max-concurrency 8 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --repeats 1 \
  --graph-mode eager \
  --max-cases 1 \
  --profile-retention all \
  --output-dir profiler-smoke
```

Use `--dry-run` to inspect generated commands without starting servers:

```bash
.venv-profiler-overhead/bin/python \
  benchmarks/overheads/run_profiler_matrix.py \
  --dry-run --max-cases 10
```

## Default matrix

The default matrix includes:

- Models: `gpt-oss-20b`, `gpt-oss-120b`, `llama-3.1-8b`, and
  `mixtral-8x7b`. `qwen3-32b` and `qwen3.5-27b` remain available through an
  explicit `--models` selection but are excluded from the default matrix.
- Workloads: input/output lengths `2000/500`, `1000/1000`, and
  `500/2000`.
- Profilers: Proton, PyTorch profiler, and Nsight Systems.
- Ordinary TP configurations that fill `--total-gpus` with DP replicas.
- Additional EP configurations for MoE models.
- Three paired repeats per case.

For eight GPUs, ordinary TP cases map to:

| TP | DP | GPUs used | Per-engine `max_num_seqs` with global concurrency 256 |
| ---: | ---: | --------: | ----------------------------------------------------: |
| 2 | 4 | 8 | 64 |
| 4 | 2 | 8 | 128 |
| 8 | 1 | 8 | 256 |

MoE EP cases use `TP=1`, `DP=EP`, and `--enable-expert-parallel`, producing
EP sizes 2, 4, and 8 by default. Ordinary TP cases are also run for MoE models.

Run both CUDA graph and eager modes:

```bash
GRAPH_MODES="cudagraph eager" \
./benchmarks/overheads/setup_profiler_matrix.sh --run \
  --total-gpus 8 \
  --num-prompts 2048 \
  --num-warmups 512 \
  --max-concurrency 256 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --repeats 3 \
  --torch-profiler-with-stack \
  --torch-profiler-dump-cuda-time-total \
  --nsys-cuda-graph-trace node \
  --profile-retention none \
  --profile-save-timeout 600 \
  --min-free-gb 128 \
  --output-dir profiler-matrix
```

The setup script accepts all matrix-runner arguments after `--run`.

To run only the `TP=2, DP=4` cases once and omit the additional MoE expert-
parallel cases:

```bash
GRAPH_MODES="cudagraph eager" \
./benchmarks/overheads/setup_profiler_matrix.sh --run \
  --total-gpus 8 \
  --tp-sizes 2 \
  --skip-ep-cases \
  --repeats 1 \
  --num-prompts 2048 \
  --num-warmups 512 \
  --max-concurrency 256 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --no-finalize-non-proton \
  --profile-save-timeout 600 \
  --output-dir profiler-matrix-tp2
```

## Measurement methodology

Each profiler configuration uses the following sequence:

1. Start a fresh `vllm serve` process with one API server and the requested
   TP/DP/EP configuration.
2. Wait for the health endpoint.
3. Run a separate, unprofiled warmup workload with a different random seed.
4. Start the profiler.
5. Run `vllm bench serve` with random prompts, `--ignore-eos`, fixed input and
   output lengths, and continuous batching.
6. Save benchmark metrics before stopping/finalizing the profiler.
7. Stop the profiler, enforce the save timeout, and terminate the server and
   its process group.

The paired baseline and profiled runs use the same repeat seed. Their execution
order is shuffled to reduce systematic ordering bias.

The harness records both global and local scheduling controls:

- `offered_global_concurrency` is the smaller of `--max-concurrency` and
  `--num-prompts`.
- `nominal_local_concurrency` is the global concurrency divided across DP
  engines, rounded up.
- `per_engine_max_num_seqs` is the smaller of configured `--max-num-seqs` and
  nominal local concurrency.
- `global_scheduler_capacity` is per-engine `max_num_seqs` multiplied by DP.

`--max-num-batched-tokens` and chunked-prefill status are recorded explicitly.

Overhead is computed only from metrics emitted by `vllm bench serve`:

- Throughput overhead is `1 - profiled / baseline`.
- Latency overhead is `profiled / baseline - 1`.

The summary includes request, output-token, and total-token throughput, plus
mean, median, p50, p90, and p99 TTFT, TPOT, ITL, and end-to-end latency.
Profiler initialization and trace-save time are not included in these request
metrics.

## Profiler behavior

### Proton

The matrix defaults to Proton's `auto` backend with `shadow` context and `tree`
data. The single-case runner additionally supports:

```text
--proton-backends
--proton-context {shadow,python}
--proton-data {tree,trace}
--proton-mode
--proton-hook triton
--proton-output-format {hatchet,hatchet_msgpack,chrome_trace}
```

Worker output names include parallel ranks so DP workers do not overwrite one
another.

### PyTorch profiler

The matrix exposes shape, memory, stack, FLOP, gzip, and CUDA-time-total
options. Shape and memory collection can substantially increase trace size and
save time, so they are disabled by default.

### Nsight Systems

The harness follows an explicit session lifecycle:

1. `nsys launch` starts vLLM without collecting initialization or warmup.
2. `nsys start` begins collection immediately before the benchmark.
3. `nsys stop` ends collection and saves the report.
4. `nsys shutdown` cleans up the session and target process.

The default trace domains are `cuda,nvtx,osrt`, with CUDA graph granularity
`node`, matching the detailed vLLM profiling setup. Use
`--nsys-cuda-graph-trace graph` to reduce trace volume.

### rocprofv3 (AMD)

`--profilers rocprof` wraps the server with `rocprofv3 -- vllm serve ...`,
the AMD counterpart of the Nsight Systems baseline. Unlike `nsys`, rocprofv3
has no external start/stop session control, so collection also covers server
startup and warmup. The benchmark metrics window is unaffected because
collection is active throughout it; trace-save time is measured during
server shutdown, when rocprofv3 finalizes its output.

The harness passes `--disable-signal-handlers true` because rocprofv3's
signal handler chains back into vLLM's SIGINT handling and recurses, hanging
shutdown. As a consequence, traces are finalized only by processes that exit
gracefully: vLLM force-kills the engine-core process during shutdown, so its
trace is usually lost even though its collection overhead is fully measured.
Overhead results are unaffected; use Proton or the PyTorch profiler when the
retained trace content matters on AMD.

Options:

```text
--rocprof-path             rocprofv3 binary (default rocprofv3)
--rocprof-trace            runtime (default), sys, or kernel
--rocprof-output-format    rocpd (default), csv, json, pftrace, otf2
```

`runtime` collects HIP runtime API, marker (ROCTx), kernel dispatch, and
memory operations, the closest match to the default Nsight domains. `sys`
adds HSA API tracing. `kernel` collects kernel dispatches only. Each traced
process writes rank-distinguishing `profile_<pid>` output files.

## AMD notes

- Select `--profilers proton torch rocprof`; `nsys` is NVIDIA-only.
- Export a non-empty `ROCR_VISIBLE_DEVICES` before running: Proton on AMD
  requires it and rejects `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES`.
- The Proton `rocprofiler` backend and `periodic_flushing` mode require
  Triton >= 3.8; older Triton builds can still use the default `auto`
  backend.

## Save timeout and trace retention

`--profile-save-timeout` applies only after benchmark metrics are saved. If
finalization exceeds the limit, the metrics remain valid and the result is
marked with:

```text
profile_save_failed = true
profile_save_error = "profile save exceeded ... seconds"
```

When only runtime overhead is needed, pass `--no-finalize-non-proton`. Torch
and nsys collection is terminated immediately after `vllm bench serve` saves
its metrics, while Proton is still finalized normally. Results record
`profile_finalize_skipped=true` for those Torch and nsys runs.

Save failures also appear in `failures.csv` with
`failure_type=profile_save`. They do not discard throughput or latency rows.

Raw traces are handled according to `--profile-retention`:

- `none`: delete traces after recording their size (recommended for matrices).
- `proton`: retain Proton profiles and delete successful Torch/nsys traces.
- `failed`: retain traces only for failed cases.
- `all`: retain every trace.

`--min-free-gb` stops the matrix before starting a new case when disk space is
below the configured floor.

## Output layout

The matrix output directory contains:

```text
environment.json       Environment recorded at the first invocation
last_environment.json  Environment recorded at the latest invocation
results.csv             Combined successful benchmark summaries
failures.csv            Runtime and profile-save failures
plots/                  Throughput, TTFT, and TPOT overhead charts
<model>/<workload>/<parallelism>/<graph-mode>/<profiler>/
  case.json             Command, signature, status, and disk usage
  results.json          Arguments, raw metrics, summaries, and failures
  summary.csv           Per-case summarized overhead
  run.log               Case-runner output
  runs/                 Per-repeat server, warmup, benchmark, and nsys logs
```

The matrix is resumable. Re-running the same command skips cases whose command,
Git revision, package versions, and script hashes match. Use `--force` to rerun
completed cases.

## Nebius Serverless AI Jobs

The [Nebius Job wrapper](nebius/README.md) builds an immutable benchmark image
and submits this matrix to an eight-GPU Serverless AI Job with one command. It
supports both shared filesystems and Object Storage for persistent results,
secure Hugging Face tokens through MysteryBox, log following, and dry-run
submission.

## Running one custom configuration

Use the lower-level runner for a model or configuration not represented by the
matrix:

```bash
.venv-profiler-overhead/bin/python \
  benchmarks/overheads/benchmark_serving_profiler.py \
  --model Qwen/Qwen3-32B \
  --length-pairs 2000:500 1000:1000 500:2000 \
  --profilers proton torch nsys \
  --num-prompts 2048 \
  --num-warmups 512 \
  --max-concurrency 256 \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192 \
  --cudagraph \
  --output-dir serving-profiler-qwen3
```

This runner accepts `--load-format`; its default is `dummy`. Use a normal vLLM
load format only when real model weights are intentionally available.
