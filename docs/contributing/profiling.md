# Profiling vLLM

!!! warning
    Profiling is only intended for vLLM developers and maintainers to understand the proportion of time spent in different parts of the codebase. **vLLM end-users should never turn on profiling** as it will significantly slow down the inference.

## Profile with PyTorch Profiler

We support tracing vLLM workers using different profilers. You can enable profiling by setting the `--profiler-config` flag when launching the server.

!!! note
    The `--profiler-config` flag is available in vLLM v0.13.0 and later. If you are using an earlier version, please upgrade to use this feature.

To use the `torch.profiler` module, set the `profiler` entry to `'torch'` and `torch_profiler_dir` to the directory where you want to save the traces. Additionally, you can control the profiling content by specifying the following additional arguments in the config:

- `torch_profiler_record_shapes` to enable recording Tensor Shapes, off by default
- `torch_profiler_with_memory` to record memory, off by default
- `torch_profiler_with_stack` to enable recording stack information, on by default
- `torch_profiler_with_flops` to enable recording FLOPs, off by default
- `torch_profiler_use_gzip` to control gzip-compressing profiling files, on by default
- `torch_profiler_dump_cuda_time_total` to control dumping and printing the aggregated CUDA self time table, on by default

When using `vllm bench serve`, you can enable profiling by passing the `--profile` flag.

Traces can be visualized using <https://ui.perfetto.dev/>.

!!! tip
    You can directly call bench module without installing vLLM using `python -m vllm.entrypoints.cli.main bench`.

!!! tip
    Only send a few requests through vLLM when profiling, as the traces can get quite large. Also, no need to untar the traces, they can be viewed directly.

!!! tip
    To stop the profiler - it flushes out all the profile trace files to the directory. This takes time, for example for about 100 requests worth of data for a llama 70b, it takes about 10 minutes to flush out on a H100.
    Set the env variable VLLM_RPC_TIMEOUT to a big number before you start the server. Say something like 30 minutes.
    `export VLLM_RPC_TIMEOUT=1800000`

### Example commands and usage

#### Offline Inference

Refer to [examples/offline_inference/simple_profiling.py](../../examples/offline_inference/simple_profiling.py) for an example.

#### OpenAI Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config '{"profiler": "torch", "torch_profiler_dir": "./vllm_profile"}'
```

vllm bench command:

```bash
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

Or use http request:

```shell
# We need first call /start_profile api to start profile.
$ curl -X POST http://localhost:8000/start_profile

# Call model generate.
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [
                        {
                                "role": "user",
                                "content": "San Francisco is a"
                        }
                ]
    }'

# After need call /stop_profile api to stop profile.
$ curl -X POST http://localhost:8000/stop_profile
```

## Profile with Triton Proton Profiler

[Proton](https://github.com/triton-lang/triton/tree/main/third_party/proton) is the built-in profiler in Triton. It provides Triton-kernel-level profiling with GPU activity tracing, and is available wherever Triton is installed (which includes standard PyTorch installations).

### Architecture

Proton is integrated into vLLM as a profiler backend alongside `torch` and `cuda`. The implementation lives in two files:

- `vllm/config/profiler.py` — `ProfilerConfig` with `proton_*` configuration fields
- `vllm/profiler/wrapper.py` — `ProtonProfilerWrapper` extending `WorkerProfiler`

The wrapper supports two lifecycle modes:

- **Non-periodic (default):** If the Proton session was early-started (for CUDA graph capture), `/start_profile` reactivates it via `proton.activate()` and `/stop_profile` pauses it via `proton.deactivate(flushing=True)`. The session persists and is reused across cycles; `proton.finalize()` is only called on server shutdown. If no early-started session exists, each cycle creates a fresh session via `proton.start()` and finalizes it via `proton.finalize()`.
- **Periodic flushing:** Enabled by setting `proton_mode` to `"periodic_flushing"`. A single session is created on the first start and persists across cycles. Subsequent starts call `proton.activate()`, stops call `proton.deactivate(flushing=True)`, and `proton.finalize()` is only called on server shutdown. Per-phase output files (`.part_N`) are written automatically, enabling memory-bounded long-running profiling.

### Configuration Options

Set Proton options via `--profiler-config` JSON. All fields use the `proton_` prefix:

| Field | Values | Default | Description |
|-------|--------|---------|-------------|
| `proton_profiler_dir` | directory path | (required) | Output directory for profiling files |
| `proton_context` | `"shadow"`, `"python"` | `"shadow"` | `shadow` uses user-annotated scopes (faster); `python` captures full Python call stacks (richer but larger output) |
| `proton_data` | `"tree"`, `"trace"` | `"tree"` | `tree` produces `.hatchet` files (aggregated); `trace` produces `.chrome_trace` files (timeline) |
| `proton_backend` | `"cupti"`, `"roctracer"`, `"instrumentation"`, `null` | `null` | GPU profiling backend. `null` auto-detects (`cupti` for NVIDIA, `roctracer` for AMD) |
| `proton_mode` | string | `null` | Backend-specific mode. Options: `"pcsampling"` (CUPTI), `"periodic_flushing"`, `"periodic_flushing:format=hatchet"` |
| `proton_hook` | `"triton"`, `null` | `null` | Set to `"triton"` to capture Triton kernel launch metadata |

General scheduling fields shared with the torch profiler also apply:

| Field | Default | Description |
|-------|---------|-------------|
| `delay_iterations` | `0` | Skip this many worker steps before starting profiling |
| `max_iterations` | `0` | Stop profiling after this many steps (`0` = unlimited) |

### Example Commands and Usage

#### Basic Profiling (Non-Periodic)

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --profiler-config '{"profiler": "proton", "proton_profiler_dir": "./proton_output", "proton_hook": "triton"}'
```

Then trigger profiling:

```bash
# Start profiling
curl -X POST http://localhost:8000/start_profile

# Send requests
curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}]
    }'

# Stop profiling — writes output files
curl -X POST http://localhost:8000/stop_profile
```

Output files appear in `./proton_output/`:

```
proton_rank0.hatchet    # Rank 0 profiling data
proton_rank1.hatchet    # Rank 1 (if using tensor parallelism)
```

#### Long-Running Profiling (Periodic Flushing)

For production servers where you want to profile multiple intervals without restarting:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --profiler-config '{"profiler": "proton", "proton_profiler_dir": "./proton_output", "proton_mode": "periodic_flushing", "proton_hook": "triton"}'
```

Each start/stop cycle writes a separate `.part_N` file:

```bash
# Cycle 1
curl -X POST http://localhost:8000/start_profile
# ... send requests ...
curl -X POST http://localhost:8000/stop_profile

# Cycle 2
curl -X POST http://localhost:8000/start_profile
# ... send requests ...
curl -X POST http://localhost:8000/stop_profile
```

Output files accumulate across cycles:

```
proton_rank0.part_0.hatchet    # Data from cycle 1
proton_rank0.part_1.hatchet    # Data from cycle 2
```

!!! note
    Due to Proton's phase design, output for phase N appears after phase N+1's deactivate. This means the first output file appears after the **second** start/stop cycle.

#### Checking Profiling Status

```bash
curl http://localhost:8000/profile_status
```

Returns:

```json
{
    "active": false,
    "profiler": "proton",
    "current_phase": 2,
    "output_dir": "./proton_output",
    "output_files": ["proton_rank0.part_0.hatchet", "proton_rank1.part_0.hatchet"]
}
```

#### Using Python Context for Rich Call Stacks

Set `proton_context` to `"python"` for full Python-level profiling:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --profiler-config '{"profiler": "proton", "proton_profiler_dir": "./proton_output", "proton_context": "python", "proton_hook": "triton"}'
```

This produces much richer output with Python call stacks but generates larger files.

#### Timeline Trace Output

Set `proton_data` to `"trace"` for Chrome trace format, viewable in <https://ui.perfetto.dev/>:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --profiler-config '{"profiler": "proton", "proton_profiler_dir": "./proton_output", "proton_data": "trace"}'
```

Output: `proton_rank0.chrome_trace` — open in Perfetto or `chrome://tracing`.

### Viewing Results

- **`.hatchet` files** (tree format): View with `proton-viewer` CLI tool:
    ```bash
    pip install llnl-hatchet
    proton-viewer -m time/ns proton_rank0.hatchet
    ```
- **`.chrome_trace` files** (trace format): Open in <https://ui.perfetto.dev/> or `chrome://tracing`.

## Profile with NVIDIA Nsight Systems

Nsight systems is an advanced tool that exposes more profiling details, such as register and shared memory usage, annotated code regions and low-level CUDA APIs and events.

[Install nsight-systems](https://docs.nvidia.com/nsight-systems/InstallationGuide/index.html) using your package manager.
The following block is an example for Ubuntu.

```bash
apt update
apt install -y --no-install-recommends gnupg
echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" | tee /etc/apt/sources.list.d/nvidia-devtools.list
apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
apt update
apt install nsight-systems-cli
```

!!! tip
    When profiling with `nsys`, it is advisable to set the environment variable `VLLM_WORKER_MULTIPROC_METHOD=spawn`. The default is to use the `fork` method instead of `spawn`. More information on the topic can be found in the [Nsight Systems release notes](https://docs.nvidia.com/nsight-systems/ReleaseNotes/index.html#general-issues).

The Nsight Systems profiler can be launched with `nsys profile ...`, with a few recommended flags for vLLM: `--trace-fork-before-exec=true --cuda-graph-trace=node`.

### Example commands and usage

#### Offline Inference

For basic usage, you can just append the profiling command before any existing script you would run for offline inference.

The following is an example using the `vllm bench latency` script:

```bash
nsys profile  \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --num-iters-warmup 5 \
    --num-iters 1 \
    --batch-size 16 \
    --input-len 512 \
    --output-len 8
```

#### OpenAI Server

To profile the server, you will want to prepend your `vllm serve` command with `nsys profile` just like for offline inference, but you will need to specify a few other arguments to enable dynamic capture similarly to the Torch Profiler:

```bash
# server
nsys profile \
    --trace-fork-before-exec=true \
    --cuda-graph-trace=node \
    --capture-range=cudaProfilerApi \
    --capture-range-end repeat \
    vllm serve meta-llama/Llama-3.1-8B-Instruct --profiler-config.profiler cuda

# client
vllm bench serve \
    --backend vllm \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path sharegpt.json \
    --profile \
    --num-prompts 2
```

With `--profile`, vLLM will capture a profile for each run of `vllm bench serve`. Once the server is killed, the profiles will all be saved.

#### Analysis

You can view these profiles either as summaries in the CLI, using `nsys stats [profile-file]`, or in the GUI by installing Nsight [locally following the directions here](https://developer.nvidia.com/nsight-systems/get-started).

??? console "CLI example"

    ```bash
    nsys stats report1.nsys-rep
    ...
    ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

    Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)  Max (ns)   StdDev (ns)                                                  Name
    --------  ---------------  ---------  -----------  -----------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
        46.3   10,327,352,338     17,505    589,965.9    144,383.0    27,040  3,126,460    944,263.8  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64_warpgroupsize1x1x1_execute_segment_k_of…
        14.8    3,305,114,764      5,152    641,520.7    293,408.0   287,296  2,822,716    867,124.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize256x128x64_warpgroupsize2x1x1_execute_segment_k_of…
        12.1    2,692,284,876     14,280    188,535.4     83,904.0    19,328  2,862,237    497,999.9  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x128x64_warpgroupsize1x1x1_execute_segment_k_off…
        9.5    2,116,600,578     33,920     62,399.8     21,504.0    15,326  2,532,285    290,954.1  sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize64x64x64_warpgroupsize1x1x1_execute_segment_k_off_…
        5.0    1,119,749,165     18,912     59,208.4      9,056.0     6,784  2,578,366    271,581.7  void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel<c10::BFloat16>, (bool)1>(T1 *, cons…
        4.1      916,662,515     21,312     43,011.6     19,776.0     8,928  2,586,205    199,790.1  void cutlass::device_kernel<flash::enable_sm90_or_later<flash::FlashAttnFwdSm90<flash::CollectiveMa…
        2.6      587,283,113     37,824     15,526.7      3,008.0     2,719  2,517,756    139,091.1  std::enable_if<T2>(int)0&&vllm::_typeConvert<T1>::exists, void>::type vllm::fused_add_rms_norm_kern…
        1.9      418,362,605     18,912     22,121.5      3,871.0     3,328  2,523,870    175,248.2  void vllm::rotary_embedding_kernel<c10::BFloat16, (bool)1>(const long *, T1 *, T1 *, const T1 *, in…
        0.7      167,083,069     18,880      8,849.7      2,240.0     1,471  2,499,996    101,436.1  void vllm::reshape_and_cache_flash_kernel<__nv_bfloat16, __nv_bfloat16, (vllm::Fp8KVCacheDataType)0…
    ...
    ```

GUI example:

<img width="1799" alt="Screenshot 2025-03-05 at 11 48 42 AM" src="https://github.com/user-attachments/assets/c7cff1ae-6d6f-477d-a342-bd13c4fc424c" />

## Continuous Profiling

There is a [GitHub CI workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-profiling.yml) in the PyTorch infrastructure repository that provides continuous profiling for different models on vLLM. This automated profiling helps track performance characteristics over time and across different model configurations.

### How It Works

The workflow currently runs weekly profiling sessions for selected models, generating detailed performance traces that can be analyzed using different tools to identify performance regressions or optimization opportunities. But, it can be triggered manually as well, using the Github Action tool.

### Adding New Models

To extend the continuous profiling to additional models, you can modify the [profiling-tests.json](https://github.com/pytorch/pytorch-integration-testing/blob/main/vllm-profiling/cuda/profiling-tests.json) configuration file in the PyTorch integration testing repository. Simply add your model specifications to this file to include them in the automated profiling runs.

### Viewing Profiling Results

The profiling traces generated by the continuous profiling workflow are publicly available on the [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm). Look for the **Profiling traces** table to access and download the traces for different models and runs.

## Profiling vLLM Python Code

The Python standard library includes
[cProfile](https://docs.python.org/3/library/profile.html) for profiling Python
code. vLLM includes a couple of helpers that make it easy to apply it to a section of vLLM.
Both the `vllm.utils.profiling.cprofile` and `vllm.utils.profiling.cprofile_context` functions can be
used to profile a section of code.

!!! note
    The legacy import paths `vllm.utils.cprofile` and `vllm.utils.cprofile_context` are deprecated.
    Please use `vllm.utils.profiling.cprofile` and `vllm.utils.profiling.cprofile_context` instead.

### Example usage - decorator

The first helper is a Python decorator that can be used to profile a function.
If a filename is specified, the profile will be saved to that file. If no filename is
specified, profile data will be printed to stdout.

```python
from vllm.utils.profiling import cprofile

@cprofile("expensive_function.prof")
def expensive_function():
    # some expensive code
    pass
```

### Example Usage - context manager

The second helper is a context manager that can be used to profile a block of
code. Similar to the decorator, the filename is optional.

```python
from vllm.utils.profiling import cprofile_context

def another_function():
    # more expensive code
    pass

with cprofile_context("another_function.prof"):
    another_function()
```

### Analyzing Profile Results

There are multiple tools available that can help analyze the profile results.
One example is [snakeviz](https://jiffyclub.github.io/snakeviz/).

```bash
pip install snakeviz
snakeviz expensive_function.prof
```

### Analyzing Garbage Collection Costs

Leverage VLLM_GC_DEBUG environment variable to debug GC costs.

- VLLM_GC_DEBUG=1: enable GC debugger with gc.collect elapsed times
- VLLM_GC_DEBUG='{"top_objects":5}': enable GC debugger to log top 5
  collected objects for each gc.collect
