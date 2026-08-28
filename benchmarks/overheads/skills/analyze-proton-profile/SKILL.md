---
name: analyze-proton-profile
description: Analyze vLLM and Triton Proton .hatchet tree profiles with an LLM inference performance-diagnosis taxonomy. Use when inspecting proton-viewer output, explaining throughput, TTFT, or TPOT regressions, separating prefill and decode behavior, classifying CPU, scheduling, kernel-launch, communication, compute, or memory bottlenecks, comparing ranks or configurations, and proposing evidence-backed follow-up profiling.
---

# Analyze Proton Profile

Diagnose Proton profiles from symptoms down to supported root-cause hypotheses.
Keep measured facts, inferences, and unverified hypotheses visibly separate.

## Required preparation

Read [references/taxonomy.md](references/taxonomy.md) completely before making
a diagnosis. It defines the taxonomy, evidence gates, caveats, and report
contract.

Collect these artifacts before interpreting a trace:

- all `.hatchet` files for the selected run and ranks;
- the paired unprofiled/profiled `results.json` or `summary.csv`;
- model, workload, prompt count, fixed input/output lengths, and `ignore_eos`;
- TP, PP, DP, EP, global/local concurrency, `max_num_seqs`, and
  `max_num_batched_tokens`;
- CUDA Graph/eager mode and the full Proton configuration.

Do not compare profiles unless these controls match or the difference is the
explicit experimental variable. Use unprofiled vLLM metrics for the product
performance baseline; a Proton trace is a perturbed measurement.

## Use proton-viewer

Resolve the viewer from the same environment that produced the trace:

```bash
command -v proton-viewer
proton-viewer --help
proton-viewer -l profile.hatchet
```

Inspect progressively instead of printing the full tree immediately:

```bash
# Root and high-level execution scopes.
proton-viewer -m time/ms,time/% -t 500 -d 5 profile.hatchet

# Kernel launch volume and short-kernel behavior.
proton-viewer -m count -d 1 profile.hatchet
proton-viewer -m avg_time/us,count --print-sorted -t 10 profile.hatchet

# Scheduler shape and token attribution.
proton-viewer -m time/ms,num_generation_tokens -t 500 -d 3 profile.hatchet

# Compare GPU time with host-side execute time.
proton-viewer -m time/us,cpu_time/us,count -i '.*execute_.*' -d 2 profile.hatchet

# Verify CUDA Graph replay-to-capture context linking.
proton-viewer -m time/us,count \
  -i '.*(cudagraph_replay|captured_at|cudagraph_capture).*' \
  -d 8 profile.hatchet

# Attribute linked kernels to model layers or compiled layer regions.
proton-viewer -m time/us,count -i '.*model\.layers\..*' -d 8 profile.hatchet

# Inspect startup capture independently from measured runtime.
proton-viewer -m time/us,count -i '.*cudagraph_capture.*' -d 8 \
  proton_rank0_cuda_graph_capture.hatchet

# Isolate scopes or kernels by regular expression.
proton-viewer -m time/ms -i '.*execute_[0-9]+_context_[1-9].*' -d 2 profile.hatchet
proton-viewer -m time/ms -i '.*(AllGather|ReduceScatter|allreduce).*' \
  -d 6 profile.hatchet

# Compare matching ranks or configurations. This computes second - first.
proton-viewer -m time/ms --diff-profile candidate.hatchet baseline.hatchet
```

Use `-t` to suppress low-time nodes, `-d` to bound tree depth, `-i` and `-e`
to include or exclude matching paths, and `--print-sorted` only when a flat
ranking is more useful than call-path context. Never treat an inclusive parent
and its children as independent time contributions.

Run the bundled summarizer for repeatable rank, phase, kernel-category, and
short-kernel evidence:

```bash
.venv/bin/python \
  benchmarks/overheads/skills/analyze-proton-profile/scripts/summarize_profile.py \
  --viewer .venv/bin/proton-viewer \
  path/to/profile-directory
```

Pass `--format json` when another tool will consume the output. Review the
classification rules in the script before applying them to unfamiliar kernel
naming schemes; unmatched kernels remain `other`.

The summarizer excludes `*_cuda_graph_capture.hatchet` sidecars by default and
restricts context-linked runtime profiles to `execute_*` subtrees. Thus startup
capture does not distort runtime GPU time, CPU execute time, kernel categories,
or rank comparisons. Pass `--include-hidden` when the capture artifact itself
is the object of analysis.

## Follow the diagnosis workflow

### 1. Establish the high-level symptom

Report total-token throughput, output-token throughput, request throughput,
TTFT, TPOT, ITL, and end-to-end latency from the unprofiled run. Then report
the paired profiler overhead. Do not infer a root cause from one aggregate
metric or one repeat.

### 2. Split prefill and decode

Use input/output lengths and Proton execution-scope metrics. Treat scopes with
nonzero context work as prefill or mixed batches and scopes with zero context
work as decode. Continuous batching can mix both phases, so report the mixed
share rather than forcing every scope into a pure phase.

### 3. Separate outside-GPU and inside-GPU evidence

Compare benchmark duration with per-rank Proton GPU time cautiously. Large CPU
gaps, request preprocessing, scheduling, and launch delays require a timeline
profiler for confirmation. Kernel-duration sums can overlap across streams and
are not automatically the wall-clock critical path.

### 4. Classify the dominant category

Quantify communication, attention, GEMM/MoE, routing, normalization, and other
leaf-kernel time. Report per-rank values and the range across ranks. Describe a
run as compute/communication co-dominant when their shares are similar; do not
force a single label.

### 5. Identify the bottleneck type

Check, in order:

1. preprocessing and TTFT symptoms;
2. scheduler fill, tail batches, and DP balance;
3. kernel count and average-duration distribution;
4. collective type, frequency, and rank skew;
5. GEMM, quantized kernel, attention, and routing hotspots;
6. hardware-counter evidence for compute or memory saturation.

### 6. Gate root-cause claims

Call something a root cause only when a controlled comparison or direct
counter supports it. Otherwise label it a hypothesis and name the cheapest
experiment that would confirm or reject it. Explicitly record which candidates
are contradicted, such as missing CUDA Graph when capture is visibly enabled,
or workload imbalance when rank times are within a small range.

### 7. Compare configurations

Change one variable at a time. Useful comparisons include CUDA Graph versus
eager, TP1 versus TP2, local batch 64 versus 128, communication backends, or a
single kernel/backend change. Compare both unprofiled metrics and normalized
Proton category shares; absolute profiled time alone is insufficient.

### 8. Report the result

Produce:

- workload and topology identity;
- high-level baseline and profiler overhead;
- taxonomy classification with confidence per layer;
- phase, rank-balance, category, and short-kernel evidence;
- confirmed exclusions;
- ranked root-cause hypotheses;
- minimal next experiments and the metric that would decide each one;
- caveats about profiler perturbation, overlap, missing counters, and repeats.

Prefer exact values and compact tables. Avoid claiming bandwidth-bound,
compute-bound, poor L2 locality, low occupancy, or inefficient quantization
from kernel names and duration alone.
