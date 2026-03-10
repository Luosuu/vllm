"""nano-vllm online-style inference with Proton periodic flushing profiling.

Simulates online serving by sending sequential requests with varying prompt
lengths through nano-vllm's add_request/step API. Uses Proton's periodic
flushing mode to capture warmup and steady-state phases separately, producing
per-phase .part_N.hatchet output files.
"""

import argparse
import os
import random

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
    return snapshot_download(model)


# Prompt templates with varying lengths to simulate online traffic
PROMPT_TEMPLATES = [
    "Hello, my name is",
    "The capital of France is",
    "Explain the theory of relativity in simple terms.",
    "Write a short poem about the ocean and its waves crashing on the shore.",
    "What are the key differences between Python and JavaScript? List at least five.",
    "Describe the process of photosynthesis step by step, including the light and dark reactions.",
    "Tell me a story about a brave knight who goes on a quest to find a magical artifact hidden in a mountain.",
    "Summarize the history of artificial intelligence from its origins in the 1950s to the present day, covering key milestones.",
]


def generate_prompts(num_prompts: int, seed: int) -> list[str]:
    """Generate a list of prompts with varying lengths using seeded random selection.

    Args:
        num_prompts: Number of prompts to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of prompt strings.
    """
    rng = random.Random(seed)
    return [rng.choice(PROMPT_TEMPLATES) for _ in range(num_prompts)]


def run_phase(
    llm: LLM,
    prompts: list[str],
    sampling_params: SamplingParams,
    phase_name: str,
) -> None:
    """Run a batch of prompts through add_request/step to simulate online serving.

    Adds all prompts as requests, then steps until all are finished.

    Args:
        llm: The nano-vllm LLM engine instance.
        prompts: List of prompt strings to process.
        sampling_params: Sampling parameters for generation.
        phase_name: Name of the phase for logging.
    """
    print(f"  [{phase_name}] Adding {len(prompts)} requests...")
    for prompt in prompts:
        llm.add_request(prompt, sampling_params)

    step_count = 0
    while not llm.is_finished():
        llm.step()
        step_count += 1

    print(f"  [{phase_name}] Completed in {step_count} steps")


def main():
    """Run nano-vllm online-style inference with periodic flushing profiling."""
    parser = argparse.ArgumentParser(
        description="nano-vllm online-style inference with Proton periodic flushing"
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
        help="Number of prompts per phase (default: 4)",
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
        default=2048,
        help="Max model context length (default: 2048)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt generation (default: 42)",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=2,
        help="Number of prompts for warmup phase (default: 2)",
    )
    args = parser.parse_args()

    # Create and register profiler with periodic flushing mode
    profiler = ProtonProfiler(
        output_dir=args.output_dir,
        context=args.context,
        data=args.data,
        hook=args.hook,
        mode="periodic_flushing",
    )
    set_profiler(profiler)

    model_path = resolve_model_path(args.model)
    print(f"Loading model: {args.model} (path: {model_path})")
    llm = LLM(model_path, enforce_eager=True, max_model_len=args.max_model_len)

    # nano-vllm does not support greedy (temperature=0.0)
    sampling_params = SamplingParams(temperature=0.8, max_tokens=args.max_tokens)

    # Generate prompts with varying lengths
    warmup_prompts = generate_prompts(args.warmup_prompts, seed=args.seed)
    steady_prompts = generate_prompts(args.num_prompts, seed=args.seed + 1)

    # Unprofiled warmup run (no profiling, just warm up the model)
    print("Running unprofiled warmup...")
    run_phase(llm, warmup_prompts[:1], sampling_params, "unprofiled-warmup")

    # Phase 1: Profiled warmup — first requests after profiler starts
    print("\n--- Phase 1: Profiled warmup ---")
    profiler.start()
    run_phase(llm, warmup_prompts, sampling_params, "warmup")
    profiler.stop()

    # Phase 2: Steady-state — normal serving traffic
    print("\n--- Phase 2: Steady-state ---")
    profiler.start()
    run_phase(llm, steady_prompts, sampling_params, "steady-state")
    profiler.stop()

    # Finalize profiler
    profiler.shutdown()

    # Print status
    status = profiler.get_status()
    print(f"\nProfiler status: {status}")
    print(f"Proton output directory: {args.output_dir}")
    print(f"Phases captured: {status['current_phase']}")
    print(f"Output files: {status['output_files']}")


if __name__ == "__main__":
    main()
