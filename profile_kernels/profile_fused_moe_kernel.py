import argparse
import functools
from collections.abc import Callable

import torch
import triton
import triton.language as tl
import triton.profiler as proton
import triton.testing

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk,
    invoke_fused_moe_kernel,
    moe_align_block_size,
    try_get_optimal_moe_config,
)


def build_fused_moe_runner(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    dtype_name: str,
) -> tuple[Callable[[], None], float, int]:
    """Prepare tensors/configs once and return a callable for benchmarking."""
    a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 10
    w2 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 10
    intermediate_cache = torch.empty((m, topk, n), device="cuda", dtype=dtype)
    gating_output = torch.randn((m, e), device="cuda", dtype=dtype)
    compute_type = tl.bfloat16 if dtype == torch.bfloat16 else tl.float16

    topk_weights, topk_ids, _ = fused_topk(a, gating_output, topk, renormalize=True)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        topk,
        # dtype_name,
        None
    )
    config = get_config_func(m)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, config["BLOCK_SIZE_M"], e
    )
    padded_tokens = int(num_tokens_post_padded.item())

    total_flops = 4.0 * m * topk * n * k

    def runner():
        with proton.scope(
            f"fused_moe_kernel_M{m}_n{n}_k{k}_e{e}_top{topk}",
            {"flops": total_flops},
        ):
            invoke_fused_moe_kernel(
                A=a,
                B=w1,
                C=intermediate_cache,
                A_scale=None,
                B_scale=None,
                B_zp=None,
                topk_weights=topk_weights,
                sorted_token_ids=sorted_token_ids,
                expert_ids=expert_ids,
                num_tokens_post_padded=num_tokens_post_padded,
                mul_routed_weight=False,
                top_k=topk,
                config=config,
                compute_type=compute_type,
                use_fp8_w8a8=False,
                use_int8_w8a8=False,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                per_channel_quant=False,
                block_shape=None,
                B_bias=None,
            )

    return runner, total_flops, padded_tokens


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m-values",
                        type=int,
                        nargs="+",
                        default=[64, 128, 256, 512, 1024, 2048, 3072, 3200, 3300, 3584, 4096, 4097, 5120, 6144, 8192, 9000],
                        help="Token counts (M) to benchmark.")
    parser.add_argument("--intermediate-size", "-n", type=int, default=256)
    parser.add_argument("--hidden-size", "-k", type=int, default=256)
    parser.add_argument("--num-experts", "-e", type=int, default=512)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--dtype",
                        type=str,
                        choices=["float16", "bfloat16"],
                        default="bfloat16")
    parser.add_argument("--name", type=str, default="tutorial")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--pcsampling", action="store_true", default=False)
    parser.add_argument("--context",
                        type=str,
                        default="shadow",
                        choices=["shadow", "python"])
    parser.add_argument("--backend",
                        type=str,
                        default=None,
                        choices=[None, "cupti", "roctracer", "instrumentation"])
    parser.add_argument("--mode",
                        type=str,
                        default=None,
                        choices=[None, "pcsampling"])
    return parser.parse_args()


args = parse_args()
m_values = args.m_values
dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
dtype_name = args.dtype


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["M"],
        x_vals=m_values,
        line_arg="provider",
        line_vals=["fused-moe"],
        line_names=["Fused MoE"],
        styles=[("purple", "-")],
        ylabel="TFLOPS",
        plot_name="fused-moe-kernel-performance",
        args={},
    ))
def benchmark(M, provider):
    assert provider == "fused-moe"
    runner, total_flops, padded_tokens = build_fused_moe_runner(
        m=M,
        n=args.intermediate_size,
        k=args.hidden_size,
        e=args.num_experts,
        topk=args.topk,
        dtype=dtype,
        dtype_name=dtype_name,
    )
    runner()
    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(runner, quantiles=quantiles)
    tflops = total_flops * 1e-12 / (ms * 1e-3)
    tflops_p80 = total_flops * 1e-12 / (max_ms * 1e-3)
    tflops_p20 = total_flops * 1e-12 / (min_ms * 1e-3)
    total_flops_tf = total_flops * 1e-12
    print(
        f"M={M} (aligned tokens={padded_tokens}): "
        f"total={total_flops_tf:.6f} TFLOPs | "
        f"time={ms:.3f} ms (p20/p80={min_ms:.3f}/{max_ms:.3f}) | "
        f"throughput={tflops:.3f} TFLOP/s "
        f"(p20/p80={tflops_p20:.3f}/{tflops_p80:.3f})"
    )
    return tflops, tflops_p80, tflops_p20


def main():
    if args.profile:
        proton_name = (
            f"{args.name}_fused_moe_kernel_mSweep_n{args.intermediate_size}"
            f"_k{args.hidden_size}_e{args.num_experts}_top{args.topk}"
        )
        proton.start(
            name=proton_name,
            hook="triton",
            context=args.context,
            backend=args.backend,
            mode="pcsampling" if args.pcsampling else args.mode,
        )
        benchmark.run(show_plots=True, print_data=True)
        proton.finalize()
    else:
        benchmark.run(show_plots=True, print_data=True)


if __name__ == "__main__":
    main()
