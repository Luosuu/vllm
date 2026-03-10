"""nano-vllm offline inference with Proton profiling.

Runs a small batch of prompts through nano-vllm's LLM API with Proton
profiling enabled, then outputs .hatchet files to the specified directory.
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download
from nanovllm import LLM, SamplingParams
from nanovllm.profiler import ProtonProfiler, set_profiler


def resolve_model_path(model: str) -> str:
    """Resolve a HuggingFace model ID to a local directory path.

    nano-vllm requires a local directory path, not a HuggingFace model ID.
    If the model is already a local path, return it as-is.
    """
    if os.path.isdir(model):
        return model
    # Download or use cached snapshot
    return snapshot_download(model)


def main():
    """Run nano-vllm offline inference with Proton profiling."""
    parser = argparse.ArgumentParser(
        description="nano-vllm offline inference with Proton profiling"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Model name or path"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for Proton output files",
    )
    parser.add_argument(
        "--context",
        type=str,
        default="shadow",
        choices=["shadow", "python"],
        help="Proton context mode (default: shadow)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="tree",
        choices=["tree", "trace"],
        help="Proton data format (default: tree)",
    )
    parser.add_argument(
        "--hook",
        type=str,
        default="triton",
        choices=["triton"],
        help="Proton hook (default: triton)",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=4,
        help="Number of prompts to run (default: 4)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens per completion (default: 32)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=512,
        help="Max model context length (default: 512)",
    )
    args = parser.parse_args()

    # Create and register profiler
    profiler = ProtonProfiler(
        output_dir=args.output_dir,
        context=args.context,
        data=args.data,
        hook=args.hook,
    )
    set_profiler(profiler)

    model_path = resolve_model_path(args.model)
    print(f"Loading model: {args.model} (path: {model_path})")
    llm = LLM(model_path, enforce_eager=True, max_model_len=args.max_model_len)

    # Simple prompts for sanity check
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is 2 + 2?",
        "List three colors:",
    ][: args.num_prompts]

    # nano-vllm does not support greedy (temperature=0.0)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=args.max_tokens)

    # Warmup run (no profiling)
    print("Running warmup...")
    llm.generate(prompts[:1], sampling_params)

    # Profiled run
    print("Starting profiled run...")
    profiler.start()
    llm.generate(prompts, sampling_params)
    profiler.stop()
    profiler.shutdown()

    # Print status
    status = profiler.get_status()
    print(f"Profiler status: {status}")
    print(f"Proton output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
