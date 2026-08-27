#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure profiler overhead using metrics emitted by ``vllm bench serve``."""

import argparse
import csv
import json
import os
import random
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any

THROUGHPUT_METRICS = (
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
)
LATENCY_METRICS = tuple(
    f"{stat}_{metric}_ms"
    for metric in ("ttft", "tpot", "itl", "e2el")
    for stat in ("mean", "median", "p50", "p90", "p99")
)


class ProfileSaveTimeout(TimeoutError):
    pass


def parse_length_pair(value: str) -> tuple[int, int]:
    try:
        input_len, output_len = (int(item) for item in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("expected INPUT_LEN:OUTPUT_LEN") from exc
    if input_len <= 0 or output_len <= 0:
        raise argparse.ArgumentTypeError("lengths must be positive")
    return input_len, output_len


def parse_json_object(value: str) -> dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("expected a JSON object")
    return parsed


def add_bool_argument(
    parser: argparse.ArgumentParser, name: str, default: bool
) -> None:
    parser.add_argument(name, action=argparse.BooleanOptionalAction, default=default)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--length-pairs", nargs="+", type=parse_length_pair, required=True
    )
    parser.add_argument(
        "--profilers",
        nargs="+",
        choices=("proton", "torch", "nsys", "rocprof"),
        default=("proton", "torch", "nsys"),
    )
    parser.add_argument("--num-prompts", "--batch-size", type=int, default=2048)
    parser.add_argument("--num-warmups", "--num-iters-warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int)

    parser.add_argument("--proton-backends", nargs="+", default=("auto",))
    parser.add_argument(
        "--proton-context", choices=("shadow", "python"), default="shadow"
    )
    parser.add_argument("--proton-data", choices=("tree", "trace"), default="tree")
    parser.add_argument("--proton-mode")
    parser.add_argument("--proton-hook", choices=("triton",))
    parser.add_argument(
        "--proton-output-format",
        choices=("hatchet", "hatchet_msgpack", "chrome_trace"),
    )
    add_bool_argument(parser, "--detailed-trace-annotation", False)
    add_bool_argument(parser, "--torch-profiler-record-shapes", False)
    add_bool_argument(parser, "--torch-profiler-with-memory", False)
    add_bool_argument(parser, "--torch-profiler-with-stack", False)
    add_bool_argument(parser, "--torch-profiler-with-flops", False)
    add_bool_argument(parser, "--torch-profiler-use-gzip", True)
    add_bool_argument(parser, "--torch-profiler-dump-cuda-time-total", False)
    parser.add_argument("--nsys-path", default="nsys")
    parser.add_argument("--nsys-trace", default="cuda,nvtx,osrt")
    parser.add_argument(
        "--nsys-cuda-graph-trace", choices=("graph", "node"), default="node"
    )
    parser.add_argument("--rocprof-path", default="rocprofv3")
    parser.add_argument(
        "--rocprof-trace", choices=("runtime", "sys", "kernel"), default="runtime"
    )
    parser.add_argument(
        "--rocprof-output-format",
        choices=("rocpd", "csv", "json", "pftrace", "otf2"),
        default="rocpd",
    )

    parser.add_argument("--load-format", default="dummy")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--tensor-parallel-size", "-tp", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", "-pp", type=int, default=1)
    parser.add_argument("--data-parallel-size", "-dp", type=int, default=1)
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--all2all-backend", default="allgather_reducescatter")
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--max-num-batched-tokens", type=int, required=True)
    parser.add_argument(
        "--max-model-len-margin",
        type=int,
        default=32,
        help=(
            "Extra tokens on top of input+output for --max-model-len. The "
            "random dataset's decode/re-tokenize roundtrip can lengthen a "
            "prompt by a token or two; with zero margin such prompts are "
            "rejected with HTTP 400 for every seed that produces one."
        ),
    )
    add_bool_argument(parser, "--enable-chunked-prefill", True)
    add_bool_argument(parser, "--cudagraph", True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--engine-args-json", type=parse_json_object, default={})
    parser.add_argument(
        "--use-v2-model-runner",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Explicitly set VLLM_USE_V2_MODEL_RUNNER for the server process.",
    )

    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--server-ready-timeout", type=float, default=900)
    parser.add_argument("--server-shutdown-timeout", type=float, default=3600)
    parser.add_argument("--profile-save-timeout", type=float, default=600)
    add_bool_argument(parser, "--finalize-non-proton", True)
    parser.add_argument("--output-dir", type=Path, default=Path("serving_profiler"))
    parser.add_argument("--fail-fast", action="store_true")
    return parser


def torch_label(args: argparse.Namespace) -> str:
    settings = (
        args.torch_profiler_record_shapes,
        args.torch_profiler_with_memory,
        args.torch_profiler_with_stack,
        args.torch_profiler_with_flops,
        args.torch_profiler_use_gzip,
        args.torch_profiler_dump_cuda_time_total,
    )
    return (
        "torch:minimal" if settings == (False,) * 4 + (True, False) else "torch:custom"
    )


def profiler_cases(args: argparse.Namespace) -> list[dict[str, str]]:
    cases = [{"profiler": "none", "label": "none"}]
    for profiler in args.profilers:
        if profiler == "proton":
            cases.extend(
                {
                    "profiler": "proton",
                    "backend": backend,
                    "label": (
                        f"proton:{backend}:{args.proton_context}:{args.proton_data}"
                    ),
                }
                for backend in args.proton_backends
            )
        elif profiler == "torch":
            cases.append({"profiler": "torch", "label": torch_label(args)})
        elif profiler == "rocprof":
            cases.append(
                {
                    "profiler": "rocprof",
                    "label": f"rocprof:{args.rocprof_trace}",
                }
            )
        else:
            cases.append(
                {
                    "profiler": "nsys",
                    "label": f"nsys:{args.nsys_cuda_graph_trace}",
                }
            )
    return cases


def profiler_config(
    args: argparse.Namespace, case: dict[str, str], profile_dir: Path
) -> dict[str, Any] | None:
    profiler = case["profiler"]
    if profiler in ("none", "nsys", "rocprof"):
        return None
    if profiler == "torch":
        return {
            "profiler": "torch",
            "torch_profiler_dir": str(profile_dir),
            "detailed_trace_annotation": args.detailed_trace_annotation,
            "torch_profiler_record_shapes": args.torch_profiler_record_shapes,
            "torch_profiler_with_memory": args.torch_profiler_with_memory,
            "torch_profiler_with_stack": args.torch_profiler_with_stack,
            "torch_profiler_with_flops": args.torch_profiler_with_flops,
            "torch_profiler_use_gzip": args.torch_profiler_use_gzip,
            "torch_profiler_dump_cuda_time_total": (
                args.torch_profiler_dump_cuda_time_total
            ),
        }
    config: dict[str, Any] = {
        "profiler": "proton",
        "proton_profiler_dir": str(profile_dir),
        "proton_context": args.proton_context,
        "proton_data": args.proton_data,
        "detailed_trace_annotation": args.detailed_trace_annotation,
    }
    if case["backend"] != "auto":
        config["proton_backend"] = case["backend"]
    for name in ("proton_mode", "proton_hook", "proton_output_format"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    return config


def free_port(host: str) -> int:
    with socket.socket() as sock:
        sock.bind((host, 0))
        return sock.getsockname()[1]


def scheduler_limits(args: argparse.Namespace) -> tuple[int, int, int]:
    offered_global_concurrency = min(
        args.max_concurrency or args.num_prompts, args.num_prompts
    )
    nominal_local_concurrency = (
        offered_global_concurrency + args.data_parallel_size - 1
    ) // args.data_parallel_size
    return (
        offered_global_concurrency,
        nominal_local_concurrency,
        min(args.max_num_seqs, nominal_local_concurrency),
    )


def server_command(
    args: argparse.Namespace,
    case: dict[str, str],
    profile_dir: Path,
    port: int,
    max_model_len: int,
    nsys_session: str | None,
) -> list[str]:
    _, _, per_engine_max_num_seqs = scheduler_limits(args)
    command = [
        shutil.which("vllm") or str(Path(sys.executable).with_name("vllm")),
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(port),
        "--api-server-count",
        "1",
        "--load-format",
        args.load_format,
        "--dtype",
        args.dtype,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--pipeline-parallel-size",
        str(args.pipeline_parallel_size),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--max-model-len",
        str(max_model_len),
        "--max-num-seqs",
        str(per_engine_max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--generation-config",
        "vllm",
        "--uvicorn-log-level",
        "warning",
        (
            "--enable-chunked-prefill"
            if args.enable_chunked_prefill
            else "--no-enable-chunked-prefill"
        ),
        "--no-enforce-eager" if args.cudagraph else "--enforce-eager",
    ]
    if args.enable_expert_parallel:
        command.append("--enable-expert-parallel")
    config = profiler_config(args, case, profile_dir)
    if config is not None:
        command.extend(["--profiler-config", json.dumps(config)])
    for key, value in args.engine_args_json.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            command.append(option if value else "--no-" + option.removeprefix("--"))
        else:
            command.extend([option, str(value)])
    if case["profiler"] == "nsys":
        assert nsys_session is not None
        command = [
            args.nsys_path,
            "launch",
            f"--session-new={nsys_session}",
            "--trace-fork-before-exec=true",
            f"--trace={args.nsys_trace}",
            f"--cuda-graph-trace={args.nsys_cuda_graph_trace}",
            "--wait=all",
            *command,
        ]
    if case["profiler"] == "rocprof":
        # rocprofv3 has no external start/stop session control, so collection
        # covers server startup and warmup in addition to the benchmark. The
        # benchmark metrics window is unaffected because collection is active
        # throughout it. Traces finalize when the server process tree exits.
        trace_flags = {
            "runtime": "--runtime-trace",
            "sys": "--sys-trace",
            "kernel": "--kernel-trace",
        }
        command = [
            args.rocprof_path,
            trace_flags[args.rocprof_trace],
            "--output-format",
            args.rocprof_output_format,
            # vLLM installs its own SIGINT handling for graceful shutdown;
            # rocprofv3's signal handler chains back into it and recurses,
            # hanging the server. Let the application handle signals and
            # finalize traces on normal process exit instead.
            "--disable-signal-handlers",
            "true",
            "-d",
            str(profile_dir),
            "-o",
            "profile_%pid%",
            "--",
            *command,
        ]
    return command


def bench_command(
    args: argparse.Namespace,
    result_path: Path,
    port: int,
    input_len: int,
    output_len: int,
    num_prompts: int,
    seed: int,
    profile: bool,
) -> list[str]:
    command = [
        shutil.which("vllm") or str(Path(sys.executable).with_name("vllm")),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://{args.host}:{port}",
        "--model",
        args.model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(input_len),
        "--random-output-len",
        str(output_len),
        "--ignore-eos",
        "--num-prompts",
        str(num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--num-warmups",
        "0",
        "--seed",
        str(seed),
        "--disable-tqdm",
        "--temperature",
        "0",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--result-dir",
        str(result_path.parent),
        "--result-filename",
        result_path.name,
    ]
    if args.max_concurrency is not None:
        command.extend(["--max-concurrency", str(args.max_concurrency)])
    if profile:
        command.append("--profile")
    return command


def wait_until_ready(
    process: subprocess.Popen[Any], host: str, port: int, timeout: float
) -> None:
    deadline = time.monotonic() + timeout
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"server did not become ready within {timeout}s")


def profile_request(host: str, port: int, action: str, timeout: float) -> None:
    request = urllib.request.Request(
        f"http://{host}:{port}/{action}_profile", method="POST"
    )
    with urllib.request.urlopen(request, timeout=timeout or None) as response:
        if response.status != 200:
            raise RuntimeError(f"{action}_profile returned HTTP {response.status}")


def stop_process_group(
    process: subprocess.Popen[Any], timeout: float
) -> tuple[float, bool]:
    start = time.monotonic()
    timed_out = False
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGINT)
    deadline = start + timeout
    while True:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            timed_out = True
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            break
        time.sleep(0.1)
    try:
        process.wait(timeout=max(0, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timed_out = True
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
    return time.monotonic() - start, timed_out


def kill_process_group(process: subprocess.Popen[Any]) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def wait_for_nsys_collection(
    args: argparse.Namespace,
    session: str,
    start_process: subprocess.Popen[Any],
    env: dict[str, str],
    timeout: float = 30,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sessions = subprocess.run(
            [args.nsys_path, "sessions", "list"],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        session_line = next(
            (line for line in sessions.stdout.splitlines() if session in line), ""
        )
        if "Collection" in session_line or "Configured" in session_line:
            return
        if start_process.poll() not in (None, 0):
            raise RuntimeError(f"nsys start exited {start_process.returncode}")
        time.sleep(0.5)
    raise TimeoutError(f"nsys session {session} did not start collecting")


def shutdown_nsys_session(
    args: argparse.Namespace, session: str, kill: str, env: dict[str, str]
) -> None:
    with suppress(subprocess.TimeoutExpired):
        subprocess.run(
            [
                args.nsys_path,
                "shutdown",
                f"--session={session}",
                f"--kill={kill}",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            env=env,
            timeout=30,
            check=False,
        )


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def run_once(
    args: argparse.Namespace,
    case: dict[str, str],
    input_len: int,
    output_len: int,
    repeat: int,
    run_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = (run_dir / "profiles").resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "bench_result.json"
    port = args.port or free_port(args.host)
    nsys_session = (
        f"vllm_{os.getpid()}_{port}_{repeat}" if case["profiler"] == "nsys" else None
    )
    command = server_command(
        args,
        case,
        profile_dir,
        port,
        input_len + output_len + args.max_model_len_margin,
        nsys_session,
    )
    python_bin = str(Path(sys.executable).parent)
    env = os.environ | {
        "PATH": python_bin + os.pathsep + os.environ.get("PATH", ""),
        "VLLM_ALL2ALL_BACKEND": args.all2all_backend,
    }
    if args.use_v2_model_runner is not None:
        env["VLLM_USE_V2_MODEL_RUNNER"] = "1" if args.use_v2_model_runner else "0"
    if case["profiler"] in ("nsys", "rocprof"):
        env |= {"VLLM_WORKER_MULTIPROC_METHOD": "spawn"}
    temp_context = (
        tempfile.TemporaryDirectory(prefix="vllm-profile-")
        if case["profiler"] != "none"
        else nullcontext()
    )
    server_log_path = run_dir / "server.log"
    with temp_context as profile_tmp, server_log_path.open("w") as server_log:
        if profile_tmp:
            env |= {"TMPDIR": profile_tmp}
        server = subprocess.Popen(
            command,
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        nsys_start: subprocess.Popen[Any] | None = None
        nsys_shutdown_complete = False
        profile_save_error: str | None = None
        profile_save_s = 0.0
        profile_finalize_skipped = False
        try:
            wait_until_ready(server, args.host, port, args.server_ready_timeout)
            if args.num_warmups:
                warmup_result_path = run_dir / "warmup_result.json"
                warmup = bench_command(
                    args,
                    warmup_result_path,
                    port,
                    input_len,
                    output_len,
                    args.num_warmups,
                    args.seed + repeat + 1_000_000,
                    False,
                )
                with (run_dir / "warmup.log").open("w") as warmup_log:
                    warmed = subprocess.run(
                        warmup,
                        cwd=Path(__file__).resolve().parents[2],
                        env=env,
                        stdout=warmup_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                if warmed.returncode != 0:
                    raise RuntimeError(f"warmup exited {warmed.returncode}")
                warmup_metrics = json.loads(warmup_result_path.read_text())
                if (
                    warmup_metrics["completed"] != args.num_warmups
                    or warmup_metrics["failed"]
                ):
                    raise RuntimeError(
                        "warmup completed="
                        f"{warmup_metrics['completed']} "
                        f"failed={warmup_metrics['failed']}"
                    )
            if case["profiler"] in ("proton", "torch"):
                profile_request(
                    args.host,
                    port,
                    "start",
                    args.server_ready_timeout,
                )
            if nsys_session is not None:
                with (run_dir / "nsys_start.log").open("w") as nsys_start_log:
                    nsys_start = subprocess.Popen(
                        [
                            args.nsys_path,
                            "start",
                            f"--session={nsys_session}",
                            "--force-overwrite=true",
                            f"--output={profile_dir / 'profile'}",
                        ],
                        stdout=nsys_start_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        start_new_session=True,
                    )
                wait_for_nsys_collection(args, nsys_session, nsys_start, env)
            benchmark = bench_command(
                args,
                result_path,
                port,
                input_len,
                output_len,
                args.num_prompts,
                args.seed + repeat,
                False,
            )
            bench_start = time.monotonic()
            with (run_dir / "bench.log").open("w") as bench_log:
                completed = subprocess.Popen(
                    benchmark,
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=bench_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                )
                while completed.poll() is None:
                    time.sleep(1)
            bench_wall_s = time.monotonic() - bench_start
            if completed.returncode:
                raise RuntimeError(f"vllm bench serve exited {completed.returncode}")
            metrics = json.loads(result_path.read_text())
            if metrics["completed"] != args.num_prompts or metrics["failed"]:
                raise RuntimeError(
                    f"completed={metrics['completed']} failed={metrics['failed']}"
                )
            expected_output = args.num_prompts * output_len
            if metrics["total_output_tokens"] != expected_output:
                raise RuntimeError(
                    f"expected {expected_output} output tokens, got "
                    f"{metrics['total_output_tokens']}"
                )
            should_finalize = case["profiler"] == "proton" or args.finalize_non_proton
            if case["profiler"] in ("proton", "torch") and should_finalize:
                profile_save_start = time.monotonic()
                try:
                    profile_request(
                        args.host,
                        port,
                        "stop",
                        args.profile_save_timeout,
                    )
                except TimeoutError:
                    profile_save_error = (
                        f"profile save exceeded {args.profile_save_timeout:g} seconds"
                    )
                    kill_process_group(server)
                except Exception as exc:
                    profile_save_error = f"stop_profile failed: {exc}"
                    kill_process_group(server)
                profile_save_s = time.monotonic() - profile_save_start
            if nsys_session is not None and should_finalize:
                profile_save_start = time.monotonic()
                with (run_dir / "nsys_stop.log").open("w") as nsys_stop_log:
                    nsys_stop = subprocess.Popen(
                        [args.nsys_path, "stop", f"--session={nsys_session}"],
                        stdout=nsys_stop_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                        start_new_session=True,
                    )
                    try:
                        nsys_stop.wait(timeout=args.profile_save_timeout or None)
                    except subprocess.TimeoutExpired:
                        kill_process_group(nsys_stop)
                        shutdown_nsys_session(args, nsys_session, "sigkill", env)
                        nsys_shutdown_complete = True
                        profile_save_error = (
                            "profile save exceeded "
                            f"{args.profile_save_timeout:g} seconds"
                        )
                profile_save_s = time.monotonic() - profile_save_start
                if profile_save_error is None:
                    if nsys_stop.returncode:
                        raise RuntimeError(f"nsys stop exited {nsys_stop.returncode}")
                    assert nsys_start is not None
                    try:
                        nsys_start.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        kill_process_group(nsys_start)
                    shutdown_nsys_session(args, nsys_session, "sigterm", env)
                    nsys_shutdown_complete = True
            if case["profiler"] in ("torch", "nsys") and not should_finalize:
                profile_finalize_skipped = True
                if nsys_session is not None:
                    shutdown_nsys_session(args, nsys_session, "sigkill", env)
                    nsys_shutdown_complete = True
                kill_process_group(server)
        finally:
            if nsys_session is not None and not nsys_shutdown_complete:
                shutdown_nsys_session(args, nsys_session, "sigkill", env)
            if nsys_start is not None and nsys_start.poll() is None:
                kill_process_group(nsys_start)
            shutdown_timeout = args.server_shutdown_timeout
            shutdown_s, shutdown_timed_out = stop_process_group(
                server, shutdown_timeout
            )
        if shutdown_timed_out:
            raise ProfileSaveTimeout(
                f"profile save exceeded {shutdown_timeout:g} seconds during shutdown"
            )
        if case["profiler"] == "rocprof" and profile_save_error is None:
            # rocprofv3 saves traces while the wrapped server exits, so the
            # shutdown wait is the trace-save wait.
            profile_save_s = shutdown_s
    profile_bytes = directory_size(profile_dir)
    if case["profiler"] != "none" and not profile_finalize_skipped:
        server_output = server_log_path.read_text()
        if (
            profile_save_error is None
            and case["profiler"] not in ("nsys", "rocprof")
            and not all(
                message in server_output
                for message in ("Profiler started.", "Profiler stopped.")
            )
        ):
            raise RuntimeError("server failed to start or stop profiler")
        if profile_bytes == 0 and profile_save_error is None:
            raise RuntimeError("profiler completed without producing an output file")
    (
        offered_global_concurrency,
        nominal_local_concurrency,
        per_engine_max_num_seqs,
    ) = scheduler_limits(args)
    return {
        "input_len": input_len,
        "output_len": output_len,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.max_concurrency,
        "offered_global_concurrency": offered_global_concurrency,
        "nominal_local_concurrency": nominal_local_concurrency,
        "request_rate": args.request_rate,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "enable_expert_parallel": args.enable_expert_parallel,
        "configured_max_num_seqs": args.max_num_seqs,
        "per_engine_max_num_seqs": per_engine_max_num_seqs,
        "global_scheduler_capacity": (
            per_engine_max_num_seqs * args.data_parallel_size
        ),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_chunked_prefill": args.enable_chunked_prefill,
        "cudagraph_requested": args.cudagraph,
        "profiler": case["label"],
        "profiler_config": profiler_config(args, case, profile_dir),
        "repeat": repeat,
        "bench_wall_s": bench_wall_s,
        "server_shutdown_s": shutdown_s,
        "profile_bytes": profile_bytes,
        "profile_save_s": profile_save_s,
        "profile_finalize_skipped": profile_finalize_skipped,
        "profile_save_failed": profile_save_error is not None,
        "profile_save_error": profile_save_error,
        "metrics": metrics,
    }


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def summarize(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = {}
    for row in raw:
        key = row["input_len"], row["output_len"], row["profiler"]
        grouped.setdefault(key, []).append(row)
    summary = []
    for (input_len, output_len, profiler), rows in sorted(grouped.items()):
        baseline = grouped.get((input_len, output_len, "none"))
        if not baseline:
            continue
        baseline_by_repeat = {row["repeat"]: row for row in baseline}
        pairs = [
            (baseline_by_repeat[row["repeat"]], row)
            for row in rows
            if row["repeat"] in baseline_by_repeat
        ]
        if not pairs:
            continue
        baseline_rows = [pair[0] for pair in pairs]
        profiled_rows = [pair[1] for pair in pairs]
        result = {
            key: value
            for key, value in profiled_rows[0].items()
            if key not in ("metrics", "repeat", "bench_wall_s", "server_shutdown_s")
        }
        result["profiler_config"] = json.dumps(
            result["profiler_config"], sort_keys=True
        )
        result["repeats"] = len(pairs)
        result["profile_bytes"] = sum(row["profile_bytes"] for row in profiled_rows)
        result["profile_save_s"] = mean(
            [row["profile_save_s"] for row in profiled_rows]
        )
        save_errors = {
            row["profile_save_error"]
            for row in profiled_rows
            if row["profile_save_error"] is not None
        }
        result["profile_save_failed"] = bool(save_errors)
        result["profile_save_failures"] = sum(
            row["profile_save_failed"] for row in profiled_rows
        )
        result["profile_save_error"] = "; ".join(sorted(save_errors)) or None
        for metric in THROUGHPUT_METRICS + LATENCY_METRICS:
            if metric not in profiled_rows[0]["metrics"]:
                continue
            baseline_value = mean([row["metrics"][metric] for row in baseline_rows])
            profiled_value = mean([row["metrics"][metric] for row in profiled_rows])
            result[f"baseline_{metric}"] = baseline_value
            result[f"profiled_{metric}"] = profiled_value
            if metric in THROUGHPUT_METRICS:
                overhead = (1 - profiled_value / baseline_value) * 100
            else:
                overhead = (profiled_value / baseline_value - 1) * 100
            result[f"{metric}_overhead_pct"] = overhead
        result["baseline_bench_wall_s"] = mean(
            [row["bench_wall_s"] for row in baseline_rows]
        )
        result["profiled_bench_wall_s"] = mean(
            [row["bench_wall_s"] for row in profiled_rows]
        )
        result["profiled_server_shutdown_s"] = mean(
            [row["server_shutdown_s"] for row in profiled_rows]
        )
        summary.append(result)
    return summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(args: argparse.Namespace) -> int:
    if args.num_prompts <= 0 or args.repeats <= 0:
        raise ValueError("num prompts and repeats must be positive")
    if args.data_parallel_size <= 0:
        raise ValueError("data parallel size must be positive")
    if args.max_concurrency is not None and args.max_concurrency <= 0:
        raise ValueError("max concurrency must be positive")
    if args.profile_save_timeout < 0:
        raise ValueError("profile save timeout must be non-negative")
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (input_len, output_len, case, repeat)
        for input_len, output_len in args.length_pairs
        for case in profiler_cases(args)
        for repeat in range(args.repeats)
    ]
    random.Random(args.seed).shuffle(tasks)
    raw = []
    errors = []
    save_failures = []
    for index, (input_len, output_len, case, repeat) in enumerate(tasks, 1):
        label = case["label"].replace(":", "_")
        run_dir = root / "runs" / label / f"in{input_len}_out{output_len}" / str(repeat)
        description = f"in={input_len} out={output_len} {case['label']} repeat={repeat}"
        print(f"[{index}/{len(tasks)}] {description}", flush=True)
        try:
            row = run_once(args, case, input_len, output_len, repeat, run_dir)
            raw.append(row)
            if row["profile_save_failed"]:
                error = row["profile_save_error"]
                save_failures.append({"case": description, "error": error})
                print(f"  SAVE FAILED: {error}", file=sys.stderr)
                if args.fail_fast:
                    break
        except Exception as exc:
            errors.append({"case": description, "error": str(exc)})
            print(f"  FAILED: {exc}", file=sys.stderr)
            if args.fail_fast or isinstance(exc, ProfileSaveTimeout):
                break
    summary = summarize(raw)
    (root / "results.json").write_text(
        json.dumps(
            {
                "arguments": vars(args) | {"output_dir": str(root)},
                "raw": raw,
                "summary": summary,
                "errors": errors,
                "save_failures": save_failures,
            },
            indent=2,
        )
        + "\n"
    )
    write_csv(root / "summary.csv", summary)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main(build_parser().parse_args()))
