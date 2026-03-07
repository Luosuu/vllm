"""vLLM online serving with Proton periodic flushing profiling.

Starts a vLLM server, sends sequential requests with varying prompt lengths,
and uses the REST API (/start_profile, /stop_profile) to control profiling
phases. Captures warmup and steady-state phases separately using Proton's
periodic flushing mode, producing per-phase .part_N.hatchet output files.
"""

import argparse
import random
import subprocess
import sys
import time

import requests


# Prompt templates matching nano-vllm online script for fair comparison
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


def wait_for_server(base_url: str, timeout: int = 300) -> bool:
    """Wait for the vLLM server to become healthy.

    Polls the /health endpoint until it returns 200 or timeout is reached.

    Args:
        base_url: Base URL of the vLLM server (e.g. http://localhost:8000).
        timeout: Maximum seconds to wait.

    Returns:
        True if server is healthy, False if timeout reached.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"{base_url}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(2)
    return False


def start_profile(base_url: str) -> None:
    """Start profiling via the REST API.

    Args:
        base_url: Base URL of the vLLM server.
    """
    resp = requests.post(f"{base_url}/start_profile", timeout=30)
    resp.raise_for_status()
    print(f"  start_profile: {resp.status_code}")


def stop_profile(base_url: str) -> None:
    """Stop profiling via the REST API.

    Args:
        base_url: Base URL of the vLLM server.
    """
    resp = requests.post(f"{base_url}/stop_profile", timeout=30)
    resp.raise_for_status()
    print(f"  stop_profile: {resp.status_code}")


def get_profile_status(base_url: str) -> dict:
    """Get profiling status via the REST API.

    Args:
        base_url: Base URL of the vLLM server.

    Returns:
        Profile status dict from the server.
    """
    resp = requests.get(f"{base_url}/profile_status", timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_chat_completions(
    base_url: str,
    prompts: list[str],
    model: str,
    max_tokens: int,
    phase_name: str,
) -> None:
    """Send chat completion requests sequentially to simulate online traffic.

    Args:
        base_url: Base URL of the vLLM server.
        prompts: List of prompt strings to send.
        model: Model name for the chat completion API.
        max_tokens: Max tokens per completion.
        phase_name: Name of the phase for logging.
    """
    print(f"  [{phase_name}] Sending {len(prompts)} requests...")
    for i, prompt in enumerate(prompts):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
        }
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        content = result["choices"][0]["message"]["content"]
        print(f"    [{i+1}/{len(prompts)}] {prompt[:40]!r}... -> {content[:40]!r}...")
    print(f"  [{phase_name}] All {len(prompts)} requests completed")


def main():
    """Run vLLM online serving with periodic flushing profiling."""
    parser = argparse.ArgumentParser(
        description="vLLM online serving with Proton periodic flushing"
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
        help="Number of prompts per steady-state phase (default: 4)",
    )
    parser.add_argument(
        "--warmup-prompts",
        type=int,
        default=2,
        help="Number of prompts for warmup phase (default: 2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens per completion (default: 128)",
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
        "--port",
        type=int,
        default=8000,
        help="Port for vLLM server (default: 8000)",
    )
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # Build profiler config JSON for vllm serve
    import json
    profiler_config = json.dumps({
        "profiler": "proton",
        "proton_profiler_dir": args.output_dir,
        "proton_context": args.context,
        "proton_data": args.data,
        "proton_hook": args.hook,
        "proton_mode": "periodic_flushing",
    })

    # Start vLLM server
    print(f"Starting vLLM server on port {args.port}...")
    server_cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--port", str(args.port),
        "--enforce-eager",
        "--max-model-len", str(args.max_model_len),
        "--profiler-config", profiler_config,
    ]
    print(f"  Command: {' '.join(server_cmd)}")
    server_proc = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    try:
        # Wait for server to be ready
        print("Waiting for server to become healthy...")
        if not wait_for_server(base_url, timeout=300):
            print("ERROR: Server did not become healthy within timeout")
            sys.exit(1)
        print("Server is healthy")

        # Generate prompts with varying lengths (matching nano-vllm seeds)
        warmup_prompts = generate_prompts(args.warmup_prompts, seed=args.seed)
        steady_prompts = generate_prompts(args.num_prompts, seed=args.seed + 1)

        # Unprofiled warmup run
        print("\nRunning unprofiled warmup...")
        send_chat_completions(
            base_url, warmup_prompts[:1], args.model, args.max_tokens,
            "unprofiled-warmup",
        )

        # Phase 1: Profiled warmup — first requests after profiler starts
        print("\n--- Phase 1: Profiled warmup ---")
        start_profile(base_url)
        send_chat_completions(
            base_url, warmup_prompts, args.model, args.max_tokens, "warmup",
        )
        stop_profile(base_url)

        # Phase 2: Steady-state — normal serving traffic
        print("\n--- Phase 2: Steady-state ---")
        start_profile(base_url)
        send_chat_completions(
            base_url, steady_prompts, args.model, args.max_tokens,
            "steady-state",
        )
        stop_profile(base_url)

        # Get final status
        status = get_profile_status(base_url)
        print(f"\nProfiler status: {status}")
        print(f"Proton output directory: {args.output_dir}")
        if "current_phase" in status:
            print(f"Phases captured: {status['current_phase']}")
        if "output_files" in status:
            print(f"Output files: {status['output_files']}")

    finally:
        # Shut down server gracefully
        print("\nShutting down vLLM server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait()
        print("Server stopped")


if __name__ == "__main__":
    main()
