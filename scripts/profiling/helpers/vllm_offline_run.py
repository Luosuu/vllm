"""vLLM offline batch inference with Proton profiling.

Runs a small batch of prompts through vLLM's LLM API with Proton profiling
enabled, then outputs .hatchet files to the specified directory.
"""

import argparse
import sys

from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig


def main():
    """Run vLLM offline inference with Proton profiling."""
    parser = argparse.ArgumentParser(
        description="vLLM offline inference with Proton profiling"
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

    # Build profiler config
    profiler_config = ProfilerConfig(
        profiler="proton",
        proton_profiler_dir=args.output_dir,
        proton_context=args.context,
        proton_data=args.data,
        proton_hook=args.hook,
    )

    print(f"Loading model: {args.model}")
    llm = LLM(
        model=args.model,
        enforce_eager=False,
        max_model_len=args.max_model_len,
        profiler_config=profiler_config,
    )

    # Simple prompts for sanity check
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "What is 2 + 2?",
        "List three colors:",
    ][: args.num_prompts]

    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=args.max_tokens
    )

    # Warmup run (no profiling)
    print("Running warmup...")
    llm.generate(prompts[:1], sampling_params)

    # Profiled run
    print("Starting profiled run...")
    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()

    print(f"Generated {len(outputs)} completions")
    for output in outputs:
        text = output.outputs[0].text
        print(f"  prompt={output.prompt!r:.40s}... -> {text!r:.40s}...")

    print(f"Proton output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
