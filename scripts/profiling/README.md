# Proton Profiling Study: vLLM vs nano-vllm

Reproducible performance profiling study comparing vLLM and nano-vllm using Triton's Proton profiler on 1x H100 80GB GPU.

## Hardware Setup

- **GPU:** 1x NVIDIA H100 80GB SXM
- **Profiler:** Triton Proton (shadow + python contexts, tree + trace formats)
- **Precision:** bf16 (no quantization)
- **Parallelism:** Single GPU (TP=1)

## Model

- **Model:** Qwen/Qwen3-32B (62GB bf16 weights, 64 decoder layers, GQA with 8 KV heads)
- **Sanity check model:** Qwen/Qwen3-0.6B (fast validation before expensive runs)
- **Memory budget:** ~18GB remaining for KV cache after model loading on 80GB H100

## Workload Parameters

All profiling scripts use identical workload parameters for fair comparison:

| Parameter | Offline | Online |
|-----------|---------|--------|
| Batch size / num_prompts | 4 | 4 (steady-state) |
| Warmup prompts | N/A | 2 |
| Input length (max_model_len) | 2048 | 2048 |
| Output length (max_tokens) | 128 | 128 |
| Random seed | 42 | 42 |
| Hook | triton | triton |

## How to Run

### Prerequisites

```bash
# Activate the virtual environment (managed by uv)
. .venv/bin/activate

# Verify Proton is available
python -c "import triton.profiler as proton; print('Proton available')"

# Verify proton-viewer is available
proton-viewer --help
```

### Run the Full Study

The master script runs all steps in sequence: sanity check, offline profiling, online profiling, and analysis.

```bash
# Run everything with defaults (Qwen3-32B)
bash scripts/profiling/run_all.sh

# Run with a different model
bash scripts/profiling/run_all.sh --model Qwen/Qwen3-14B

# Skip the sanity check (if already validated)
bash scripts/profiling/run_all.sh --skip-sanity

# Run only offline profiling + analysis
bash scripts/profiling/run_all.sh --skip-sanity --skip-online

# Run only online profiling + analysis
bash scripts/profiling/run_all.sh --skip-sanity --skip-offline

# Custom workload parameters
bash scripts/profiling/run_all.sh --batch-size 8 --input-len 1024 --output-len 64
```

### Run Individual Scripts

```bash
# 1. Sanity check with small model
bash scripts/profiling/sanity_check.sh [MODEL]

# 2. Offline profiling - nano-vllm
bash scripts/profiling/nano_vllm_offline.sh --model Qwen/Qwen3-32B

# 3. Offline profiling - vLLM
bash scripts/profiling/vllm_offline.sh --model Qwen/Qwen3-32B

# 4. Online profiling - nano-vllm (periodic flushing)
bash scripts/profiling/nano_vllm_online.sh --model Qwen/Qwen3-32B

# 5. Online profiling - vLLM (periodic flushing via REST API)
bash scripts/profiling/vllm_online.sh --model Qwen/Qwen3-32B

# 6. Analyze all profiles
bash scripts/profiling/analyze_profiles.sh
```

## Output Directories

All profiling output is saved under `profiling_output/` at the repository root:

```
profiling_output/
  sanity_check/
    nano_vllm/                    # Qwen3-0.6B sanity check output
      proton_rank0.hatchet
    vllm/
      proton_rank0.hatchet
  nano_vllm/
    offline/
      shadow_tree/                # shadow context + tree format
        proton_rank0.hatchet
      shadow_trace/               # shadow context + trace format
        proton_rank0.chrome_trace
      python_tree/                # python context + tree format
        proton_rank0.hatchet
      python_trace/               # python context + trace format
        proton_rank0.chrome_trace
    online/
      shadow_tree/                # periodic flushing output
        proton_rank0.part_0.hatchet    # warmup phase
        proton_rank0.part_1.hatchet    # steady-state phase
  vllm/
    offline/
      shadow_tree/
      shadow_trace/
      python_tree/
      python_trace/
    online/
      shadow_tree/
        proton_rank0.part_0.hatchet
        proton_rank0.part_1.hatchet
  analysis/
    nano_vllm_offline_shadow_tree_proton_rank0.txt
    vllm_offline_shadow_tree_proton_rank0.txt
    nano_vllm_offline_python_tree_proton_rank0.txt
    vllm_offline_python_tree_proton_rank0.txt
    nano_vllm_online_warmup.txt
    nano_vllm_online_steady.txt
    vllm_online_warmup.txt
    vllm_online_steady.txt
    chrome_traces.txt
```

### File Naming Conventions

- `.hatchet` - Tree-format profiling data (JSON), analyzable by `proton-viewer`
- `.chrome_trace` - Chrome trace format, viewable in [Perfetto](https://ui.perfetto.dev/)
- `proton_rank{N}` - Rank-specific output (rank0 for single GPU)
- `.part_{N}` - Per-phase output from periodic flushing (part_0 = warmup, part_1 = steady-state)

## Key Findings

### Quantitative Comparison

The table below summarizes expected metrics from `proton-viewer` analysis. Actual values are populated after running the study.

| Metric | nano-vllm | vLLM | Notes |
|--------|-----------|------|-------|
| Total GPU time (ns) | TBD | TBD | From shadow+tree profile |
| Top kernel time (ns) | TBD | TBD | Single most expensive kernel |
| Kernel count | TBD | TBD | Total unique kernel entries |
| Scheduling overhead | TBD | TBD | Time in scheduler vs GPU kernels |

After running the study, populate this table with values from:
- `profiling_output/analysis/nano_vllm_offline_shadow_tree_proton_rank0.txt`
- `profiling_output/analysis/vllm_offline_shadow_tree_proton_rank0.txt`

### Shadow vs Python Context Differences

| Aspect | Shadow Context | Python Context |
|--------|----------------|----------------|
| **Overhead** | Low (~1-5%) | Higher (~10-20%) |
| **Output size** | Small (KB range) | Large (MB range) |
| **Data captured** | GPU kernel timing only | Full Python call stacks + GPU timing |
| **Use case** | Accurate kernel timing | Call stack attribution, identifying Python bottlenecks |
| **When to use** | Production benchmarking, throughput measurement | Debugging, understanding framework overhead |

- Shadow context captures only GPU kernel-level data (CUPTI callbacks). Output is minimal — often just a ROOT node with children for each kernel launch. Best for measuring raw GPU performance.
- Python context instruments Python call stacks in addition to GPU kernels. Output includes the full function call hierarchy, making it possible to attribute GPU time to specific Python functions (e.g., `Attention.forward` vs `Sampler.forward`).
- For comparing frameworks, shadow context provides the fairest kernel-level comparison since it doesn't include Python instrumentation overhead.

### Tree vs Trace Format Trade-offs

| Aspect | Tree (.hatchet) | Trace (.chrome_trace) |
|--------|------------------|-----------------------|
| **Format** | JSON tree structure | Chrome trace JSON |
| **File size** | Small (KB-MB) | Large (MB-GB) |
| **Analysis** | `proton-viewer -m time/ns` | Visual timeline in Perfetto |
| **Data structure** | Aggregated call tree with metrics | Per-event timeline with timestamps |
| **Use case** | Quantitative kernel comparison | Visual timeline inspection |
| **Programmatic access** | Yes (JSON parsing, proton-viewer) | Limited (Perfetto UI only) |

- Tree format aggregates all kernel invocations into a hierarchical call tree with summed metrics. Ideal for automated analysis and comparison scripts.
- Trace format preserves the temporal ordering of every kernel launch. Ideal for understanding scheduling patterns, overlap, and pipeline behavior.
- For this study, tree format is primary for quantitative comparison; trace format supplements with visual timeline analysis.

### Periodic Flushing: Warmup vs Steady-State Comparison

Periodic flushing captures profiling data in phases, enabling warmup vs steady-state comparison:

| Aspect | Warmup (part_0) | Steady-State (part_1) |
|--------|------------------|-----------------------|
| **Prompts** | 2 (small batch) | 4 (full batch) |
| **Expected behavior** | JIT compilation, CUDA graph capture, cache warmup | Optimized execution, cached graphs |
| **Kernel profile** | More diverse kernels (compilation + inference) | Fewer, more focused kernels (inference only) |
| **GPU time** | Higher per-token (compilation overhead) | Lower per-token (optimized paths) |

- In Proton's periodic flushing mode, the current phase is never flushed — only previous phases are flushed during `deactivate()`. So `part_0.hatchet` appears after the second `start()/stop()` cycle.
- nano-vllm uses direct `ProtonProfiler.start()/stop()` calls; vLLM uses REST API endpoints (`/start_profile`, `/stop_profile`).
- Compare warmup vs steady-state files in `profiling_output/analysis/` to quantify JIT/warmup overhead.

## Script Reference

| Script | Description |
|--------|-------------|
| `run_all.sh` | Master script — runs all steps in sequence |
| `sanity_check.sh` | Quick validation with Qwen3-0.6B |
| `nano_vllm_offline.sh` | nano-vllm offline batch profiling (4 configs) |
| `vllm_offline.sh` | vLLM offline batch profiling (4 configs) |
| `nano_vllm_online.sh` | nano-vllm online profiling with periodic flushing |
| `vllm_online.sh` | vLLM online profiling with periodic flushing |
| `analyze_profiles.sh` | Run proton-viewer on all profiles, compare results |

### Helper Python Scripts

Located in `scripts/profiling/helpers/`:

| Script | Description |
|--------|-------------|
| `nano_vllm_offline_run.py` | nano-vllm offline inference with Proton |
| `vllm_offline_run.py` | vLLM offline inference with Proton |
| `nano_vllm_online_run.py` | nano-vllm online simulation with periodic flushing |
| `vllm_online_run.py` | vLLM server + REST API profiling control |

## Profiling Configurations

Each framework is profiled in 4 configurations (context x data format):

| # | Context | Data | Output Format | Analyzable |
|---|---------|------|---------------|------------|
| 1 | shadow | tree | `.hatchet` | proton-viewer |
| 2 | shadow | trace | `.chrome_trace` | Perfetto |
| 3 | python | tree | `.hatchet` | proton-viewer |
| 4 | python | trace | `.chrome_trace` | Perfetto |

Additionally, periodic flushing profiles are captured in shadow+tree configuration for warmup vs steady-state comparison.

## Viewing Chrome Traces

Chrome trace files (`.chrome_trace`) can be viewed in Perfetto:

1. Open [Perfetto UI](https://ui.perfetto.dev/)
2. Click "Open trace file"
3. Select the `.chrome_trace` file from `profiling_output/*/offline/shadow_trace/` or `python_trace/`

Chrome trace files show a timeline view of all kernel launches, enabling visual inspection of scheduling patterns, kernel overlap, and pipeline behavior.
