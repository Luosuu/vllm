# LLM inference performance diagnosis taxonomy

Use the taxonomy as an evidence funnel. Start with the user-visible symptom and
descend only as far as the available measurements support.

## Taxonomy layers

### 1. High-level metrics

Establish what regressed before explaining why:

- total-token, output-token, and request throughput;
- time to first token (TTFT);
- time per output token (TPOT) and inter-token latency (ITL);
- end-to-end latency and relevant percentiles.

Throughput and latency can move differently. TTFT emphasizes queueing and
prefill; TPOT emphasizes decode; end-to-end latency combines both.

### 2. Phases

Separate prefill, decode, and mixed continuous-batching execution:

- **Prefill** processes input tokens and usually exposes larger GEMMs and
  attention over prompt tokens.
- **Decode** generates tokens iteratively and often exposes smaller matrices,
  high collective frequency, and launch latency.
- **Mixed** batches contain context and generation work together. Preserve this
  label instead of assigning the full scope to either phase.

Proton scope names and numeric metrics encode context/generation requests and
tokens. Aggregate nonzero-context scopes separately from context-zero scopes.

### 3. Position

Determine whether the lost time is predominantly outside or inside GPU work:

- **Outside GPU:** preprocessing, HTTP/tokenization, scheduling, Python or CPU
  gaps, launch delay, synchronization, and host/network coordination.
- **Inside GPU:** GEMM, attention, quantization, routing, collectives, memory
  operations, and device-side synchronization.

Communication crosses this boundary. NCCL and fused collective kernels consume
GPU time, while rendezvous and network progress can also create outside-GPU
gaps. Treat communication as a cross-cutting category even if a diagram places
it exclusively outside GPU.

Proton tree time is a sum of recorded kernel durations. With concurrent
streams, it can exceed critical-path wall time. Use nsys or another timeline
when outside-GPU gaps and overlap decide the diagnosis.

### 4. Category

Classify measured time without yet claiming a root cause:

| Category | Evidence | Unsupported leap |
| --- | --- | --- |
| CPU-bound | Large host gaps or CPU samples on the critical path | Benchmark wall time minus summed GPU time alone |
| Communication-bound | Collective share, frequency, and critical-path placement | Calling every NCCL duration bandwidth saturation |
| Compute-bound | High SM/Tensor Core utilization and arithmetic throughput | GEMM kernel names or large GEMM share alone |
| Bandwidth-bound | DRAM/L2 counters near the applicable roofline | Copy/attention kernel names alone |

A run can be co-dominant. Report shares and confidence rather than choosing the
largest category by a narrow margin.

### 5. Bottleneck type

Map categories to actionable mechanisms:

- **Preprocess:** request creation, tokenization, serialization, or input copy.
- **Schedule:** insufficient fill, tail batches, queueing, DP skew, or chunking.
- **Kernel launch/fragmentation:** many short kernels and gaps that fixed launch
  overhead cannot amortize.
- **Network/collective:** frequent or slow AllGather, ReduceScatter, AllReduce,
  or all-to-all operations.
- **Inefficient quantized kernel:** low achieved throughput for a supported
  shape/dtype, confirmed with hardware counters or a controlled backend swap.
- **Memory access pattern:** low L2 reuse, high DRAM pressure, or uncoalesced
  accesses confirmed with counters.

Short-kernel distributions are strong evidence of fragmentation, but only a
timeline distinguishes device kernel latency from CPU launch gaps. CUDA Graph
reduces the latter, not the former.

### 6. Root cause

Require direct evidence or a controlled experiment:

| Candidate | Confirmation | Rejection evidence |
| --- | --- | --- |
| Asynchronous pipeline problem | Timeline bubbles or blocked stages | Sustained overlap and no critical gaps |
| Batching strategy | Better metrics and fewer short kernels after one batch-control change | Consistently full batches with no meaningful tail cost |
| Missing CUDA Graph | Eager launches and improvement after enabling capture | Captured scopes and matching graph configuration |
| Improper parallelism | TP/DP/EP A/B isolates communication or imbalance cost | Similar normalized metrics and balanced ranks |
| Low L2 hit rate | NCU L2 hit/traffic counters | Healthy counters for the target kernel |
| Workload imbalance | Rank, stage, or expert time/token skew | Small rank-time and token-distribution range |
| Bad tuning | Backend/tile/config A/B improves the same kernel and workload | Comparable achieved throughput across candidates |

## Evidence available from Proton

Typical tree profiles provide:

- kernel names, counts, and durations;
- hierarchical execution scopes;
- per-rank device attribution;
- vLLM context/generation request and token metrics;
- derived time units, average time, and time percentage in `proton-viewer`.

Do not assume every profile contains FLOPs, bytes, peak-bandwidth time, or peak
FLOP time. Run `proton-viewer -l` first. If those source metrics are absent,
`util`, FLOP/s, byte/s, compute-bound, bandwidth-bound, and L2 conclusions need
another profiler.

## Confidence scale

- **High:** direct metric repeated across matching ranks/runs or a controlled
  A/B comparison.
- **Medium:** consistent indirect evidence with a plausible alternative.
- **Low:** kernel-name heuristic or a single noisy observation.
- **Unknown:** required counter or timeline is absent.

Always state confidence and the missing measurement that would raise it.

## Common interpretation traps

1. Do not mix profiled throughput with the unprofiled product baseline.
2. Do not add inclusive parent and child times.
3. Do not equate summed GPU duration with wall-clock critical-path time.
4. Do not interpret a high collective share as bandwidth saturation without
   message-size, topology, or timeline evidence.
5. Do not call a quantized kernel inefficient without achieved-throughput or
   backend-comparison evidence.
6. Do not infer a scheduling problem from a small tail alone; quantify its time
   and token share.
7. Do not compare ranks that executed different workloads or roles.
8. Do not rely on one repeat for tail latency or small percentage differences.

## Minimal follow-up tools

- Use `vllm bench serve` for controlled high-level metrics.
- Use Proton tree profiles for hierarchy, categories, counts, scopes, and rank
  comparison.
- Use Proton trace output or nsys for CPU/GPU gaps, overlap, and critical path.
- Use NCU on selected kernels for SM throughput, Tensor Core utilization, DRAM
  bandwidth, L2 hit rate, occupancy, registers, and shared memory.

Choose the cheapest tool that can decide the current hypothesis. Avoid broad,
high-volume traces when a targeted kernel or short workload is sufficient.

## Report template

Use a compact table for the evidence funnel:

| Layer | Finding | Evidence | Confidence |
| --- | --- | --- | --- |
| High-level metric | | | |
| Phase | | | |
| Position | | | |
| Category | | | |
| Bottleneck type | | | |
| Root cause | | | |

Then list confirmed exclusions, ranked hypotheses, next experiments, and
measurement caveats.
