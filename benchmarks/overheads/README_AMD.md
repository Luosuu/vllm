# AMD profiler-overhead benchmark (one-click)

`run_amd_overhead_smoke.sh` runs the full profiler-overhead comparison on an
AMD GPU — baseline, Proton, PyTorch profiler, and rocprofv3, in both eager
and CUDA-graph modes (8 cases total) — and generates the combined result
table. **The table is printed to stdout at the end of the run and also saved
to `$OUTPUT_ROOT/table.md`.**

Tested on MI300X (gfx942) with `rocm/vllm-dev:nightly`.

## 1. Environment setup

### 1.1 Start a ROCm vLLM container

The host needs an AMD GPU, the `amdgpu` driver, and Docker. Pull AMD's
prebuilt vLLM dev image and start a container with GPU access:

```bash
docker pull rocm/vllm-dev:nightly

docker run -d --name vllm-overhead \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --ipc=host --shm-size=32g --security-opt seccomp=unconfined \
  -v "$PWD":/work \
  rocm/vllm-dev:nightly sleep infinity

docker exec -it vllm-overhead bash
```

All remaining steps run inside the container.

### 1.2 Install the benchmark branch

The benchmarks live on the `profiler-overhead-benchmarks` branch of
`Luosuu/vllm`. Its engine-side changes are pure Python (no compiled
kernels), so they can be overlaid onto the image's prebuilt ROCm vLLM
instead of building from source:

```bash
cd /work
git clone --depth 1 -b profiler-overhead-benchmarks \
  https://github.com/Luosuu/vllm.git vllm-bench
cd vllm-bench

SP=$(python3 -c 'import vllm, os; print(os.path.dirname(os.path.dirname(vllm.__file__)))')
for f in benchmarks/latency.py config/profiler.py \
         entrypoints/serve/profile/api_router.py profiler/wrapper.py \
         v1/executor/abstract.py v1/worker/cpu_worker.py \
         v1/worker/gpu_worker.py v1/worker/xpu_worker.py; do
  cp "vllm/$f" "$SP/vllm/$f"
done

# sanity check
python3 -c 'from vllm.profiler.wrapper import ProtonProfilerWrapper; print("OK")'
```

Notes:

- `rocprofv3` ships with ROCm and is already on `PATH` in the image.
- The default model `Qwen/Qwen3-8B` is ungated and only its config/tokenizer
  are downloaded (`--load-format dummy`), so no HF token is required.
- The Proton `rocprofiler` backend and `periodic_flushing` mode need
  Triton >= 3.8; with older Triton (e.g. 3.6 in current images) Proton runs
  on its default `auto` backend.

## 2. Run

From the repository root:

```bash
bash benchmarks/overheads/run_amd_overhead_smoke.sh
```

That is the entire workflow: the script exports `ROCR_VISIBLE_DEVICES`
(required by Proton on AMD), fixes up the `vllm` CLI path, runs the eager
and cudagraph experiment sets (each pairs every profiler with a same-seed
unprofiled baseline), then generates the table. Smoke scale takes roughly
15–20 minutes on one MI300X.

Scale-up example (paper-grade settings) via environment variables:

```bash
NUM_PROMPTS=2048 NUM_WARMUPS=512 MAX_CONCURRENCY=256 \
MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 REPEATS=3 \
LENGTH_PAIR=2000:500 \
bash benchmarks/overheads/run_amd_overhead_smoke.sh
```

Other knobs: `MODEL`, `TP_SIZE`, `OUTPUT_ROOT` (default
`amd-overhead-results`), `ROCR_VISIBLE_DEVICES`.

## 3. Output

When the run finishes, the combined 8-row markdown table is:

- **printed to stdout**, and
- **saved to `$OUTPUT_ROOT/table.md`**.

Example shape:

| Mode | Profiler | Output tput (tok/s) | Tput overhead | Mean TTFT (ms) | TTFT overhead | Mean TPOT (ms) | TPOT overhead | Trace size |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| eager | baseline | ... | — | ... | — | ... | — | — |
| eager | proton:auto:shadow:tree | ... | +…% | ... | +…% | ... | +…% | … MB |
| eager | torch:minimal | ... | +…% | ... | +…% | ... | +…% | … MB |
| eager | rocprof:runtime | ... | +…% | ... | +…% | ... | +…% | … MB |
| cudagraph | baseline | ... | — | ... | — | ... | — | — |
| cudagraph | proton:auto:shadow:tree | ... | +…% | ... | +…% | ... | +…% | … MB |
| cudagraph | torch:minimal | ... | +…% | ... | +…% | ... | +…% | … MB |
| cudagraph | rocprof:runtime | ... | +…% | ... | +…% | ... | +…% | … MB |

Raw data lands in `$OUTPUT_ROOT/{eager,cudagraph}/`: `results.json`
(arguments, per-run metrics, paired-overhead summary), `summary.csv`, and
`runs/<label>/<workload>/<repeat>/` with server/warmup/bench logs and
retained traces. Overheads are computed against the same-seed paired
baseline: throughput overhead is `1 - profiled/baseline`; latency overhead
is `profiled/baseline - 1`.

## 4. Caveats

- `nsys` is NVIDIA-only and intentionally excluded on AMD.
- rocprofv3 has no external start/stop session control: collection also
  covers server startup/warmup (benchmark-window metrics are unaffected),
  and its reported trace size counts only files finalized by
  gracefully-exiting processes — vLLM force-kills the engine core at
  shutdown, so that trace is usually lost while its collection overhead is
  still fully measured. Use Proton or the PyTorch profiler when retained
  trace content matters on AMD.
- If a case fails, the harness marks it `FAILED` and continues; the table
  simply omits that row.
- Smoke-scale percentages are amplified by short-run tail effects; use the
  paper-grade settings above for publishable numbers.
