#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run and visualize the profiler overhead model/workload matrix."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MODELS = {
    "gpt-oss-20b": ("openai/gpt-oss-20b", True),
    "gpt-oss-120b": ("openai/gpt-oss-120b", True),
    "llama-3.1-8b": ("meta-llama/Llama-3.1-8B", False),
    "mixtral-8x7b": ("mistralai/Mixtral-8x7B-v0.1", True),
    "qwen3-32b": ("Qwen/Qwen3-32B", False),
    "qwen3.5-27b": ("Qwen/Qwen3.5-27B", False),
}
DEFAULT_MODELS = tuple(
    name for name in MODELS if name not in {"qwen3-32b", "qwen3.5-27b"}
)
WORKLOADS = {
    "in2000_out500": (2000, 500),
    "in1000_out1000": (1000, 1000),
    "in500_out2000": (500, 2000),
}


@dataclass(frozen=True)
class MatrixCase:
    model_name: str
    model_id: str
    workload: str
    input_len: int
    output_len: int
    parallelism: str
    tensor_parallel_size: int
    data_parallel_size: int
    expert_parallel: bool
    profiler: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=DEFAULT_MODELS)
    parser.add_argument(
        "--workloads", nargs="+", choices=WORKLOADS, default=list(WORKLOADS)
    )
    parser.add_argument("--tp-sizes", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--ep-sizes", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument(
        "--skip-ep-cases",
        action="store_true",
        help="Skip the additional expert-parallel cases for MoE models.",
    )
    parser.add_argument(
        "--total-gpus",
        type=int,
        default=8,
        help="Fill this many GPUs with DP replicas for ordinary TP cases.",
    )
    parser.add_argument(
        "--profilers",
        nargs="+",
        choices=("proton", "torch", "nsys", "rocprof"),
        default=("proton", "torch", "nsys"),
    )
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
    parser.add_argument(
        "--detailed-trace-annotation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--num-prompts", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument(
        "--num-warmups", "--num-iters-warmup", dest="num_warmups", type=int, default=1
    )
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--all2all-backend",
        default="allgather_reducescatter",
        help="VLLM_ALL2ALL_BACKEND used by MoE expert-parallel cases.",
    )
    parser.add_argument(
        "--torch-profiler-record-shapes",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--torch-profiler-with-memory",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--torch-profiler-with-stack",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--torch-profiler-with-flops",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--torch-profiler-use-gzip",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torch-profiler-dump-cuda-time-total",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--graph-mode", choices=("cudagraph", "eager"), default="cudagraph"
    )
    parser.add_argument("--output-dir", type=Path, default=Path("profiler_matrix"))
    parser.add_argument(
        "--profile-retention",
        choices=("none", "proton", "failed", "all"),
        default="none",
        help=(
            "Raw traces to retain after their sizes are recorded. 'proton' "
            "keeps only Proton profiles."
        ),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=32,
        help="Free-space floor checked before each profiler case.",
    )
    parser.add_argument(
        "--case-timeout",
        type=float,
        default=21600,
        help="Maximum seconds for one baseline/profile case; zero disables it.",
    )
    parser.add_argument(
        "--profile-save-timeout",
        type=float,
        default=600,
        help="Maximum seconds to finalize profiler output; zero disables it.",
    )
    parser.add_argument(
        "--finalize-non-proton",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Finalize Torch and nsys traces after metrics are saved. Disable this "
            "when only their runtime overhead is needed."
        ),
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--nsys-path", default="nsys")
    parser.add_argument(
        "--nsys-cuda-graph-trace",
        choices=("graph", "node"),
        default="node",
        help=(
            "Nsight CUDA graph granularity. 'node' matches the vLLM profiling "
            "guide; 'graph' reduces trace volume."
        ),
    )
    parser.add_argument(
        "--nsys-trace",
        default="cuda,nvtx,osrt",
        help="Comma-separated Nsight Systems trace domains.",
    )
    parser.add_argument("--rocprof-path", default="rocprofv3")
    parser.add_argument(
        "--rocprof-trace",
        choices=("runtime", "sys", "kernel"),
        default="runtime",
        help=(
            "rocprofv3 trace aggregate. 'runtime' collects HIP runtime, "
            "marker, kernel, and memory operations; 'sys' adds HSA API "
            "tracing; 'kernel' collects kernel dispatches only."
        ),
    )
    parser.add_argument(
        "--rocprof-output-format",
        choices=("rocpd", "csv", "json", "pftrace", "otf2"),
        default="rocpd",
    )
    return parser


def make_cases(args: argparse.Namespace) -> list[MatrixCase]:
    cases = []
    invalid_tp_sizes = [size for size in args.tp_sizes if args.total_gpus % size]
    if invalid_tp_sizes:
        raise ValueError(
            f"TP sizes {invalid_tp_sizes} do not divide total GPUs {args.total_gpus}"
        )
    for model_name in args.models:
        model_id, is_moe = MODELS[model_name]
        for workload in args.workloads:
            input_len, output_len = WORKLOADS[workload]
            for size in args.tp_sizes:
                dp_size = args.total_gpus // size
                for profiler in args.profilers:
                    cases.append(
                        MatrixCase(
                            model_name,
                            model_id,
                            workload,
                            input_len,
                            output_len,
                            f"tp{size}_dp{dp_size}",
                            size,
                            dp_size,
                            False,
                            profiler,
                        )
                    )
            if is_moe and not args.skip_ep_cases:
                for size in args.ep_sizes:
                    for profiler in args.profilers:
                        cases.append(
                            MatrixCase(
                                model_name,
                                model_id,
                                workload,
                                input_len,
                                output_len,
                                f"tp1_dp{size}_ep{size}",
                                1,
                                size,
                                True,
                                profiler,
                            )
                        )
    return cases[: args.max_cases] if args.max_cases else cases


def case_dir(root: Path, case: MatrixCase, graph_mode: str) -> Path:
    return (
        root
        / case.model_name
        / case.workload
        / case.parallelism
        / graph_mode
        / case.profiler
    )


def case_complete(path: Path, profiler_label: str, signature: str) -> bool:
    result_path = path / "results.json"
    case_path = path / "case.json"
    if not result_path.exists() or not case_path.exists():
        return False
    try:
        result = json.loads(result_path.read_text())
        metadata = json.loads(case_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not metadata.get("succeeded") or metadata.get("signature") != signature:
        return False
    return result_complete(result, profiler_label)


def result_complete(result: dict[str, Any], profiler_label: str) -> bool:
    completed = {row["profiler"] for row in result.get("summary", [])}
    return not result.get("errors") and {"none", profiler_label} <= completed


def expected_profiler_label(args: argparse.Namespace, profiler: str) -> str:
    torch_settings = (
        args.torch_profiler_record_shapes,
        args.torch_profiler_with_memory,
        args.torch_profiler_with_stack,
        args.torch_profiler_with_flops,
        args.torch_profiler_use_gzip,
        args.torch_profiler_dump_cuda_time_total,
    )
    torch_label = (
        "torch:minimal"
        if torch_settings == (False,) * 4 + (True, False)
        else "torch:custom"
    )
    return {
        "proton": f"proton:auto:{args.proton_context}:{args.proton_data}",
        "torch": torch_label,
        "nsys": f"nsys:{args.nsys_cuda_graph_trace}",
        "rocprof": f"rocprof:{args.rocprof_trace}",
    }[profiler]


def retain_profiles(retention: str, profiler: str, succeeded: bool) -> bool:
    return (
        retention == "all"
        or (retention == "proton" and profiler == "proton")
        or (retention == "failed" and not succeeded)
    )


def case_signature(command: list[str], metadata: dict[str, Any]) -> str:
    scripts = (Path(__file__).resolve(), Path(command[1]).resolve())
    payload = {
        "command": command,
        "vllm_revision": metadata["vllm_revision"],
        "benchmark_revision": metadata["benchmark_revision"],
        "python_packages": metadata["python_packages"],
        "scripts": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def build_command(
    args: argparse.Namespace, case: MatrixCase, output_dir: Path
) -> list[str]:
    benchmark = Path(__file__).with_name("benchmark_serving_profiler.py")
    command = [
        args.python,
        str(benchmark),
        "--model",
        case.model_id,
        "--length-pairs",
        f"{case.input_len}:{case.output_len}",
        "--profilers",
        case.profiler,
        "--batch-size",
        str(args.num_prompts),
        "--num-warmups",
        str(args.num_warmups),
        "--request-rate",
        str(args.request_rate),
        "--repeats",
        str(args.repeats),
        "--tensor-parallel-size",
        str(case.tensor_parallel_size),
        "--data-parallel-size",
        str(case.data_parallel_size),
        "--max-num-seqs",
        str(min(args.max_num_seqs, args.num_prompts)),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--enable-chunked-prefill",
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--load-format",
        "dummy",
        "--output-dir",
        str(output_dir),
        "--nsys-path",
        args.nsys_path,
        "--nsys-cuda-graph-trace",
        args.nsys_cuda_graph_trace,
        "--nsys-trace",
        args.nsys_trace,
        "--rocprof-path",
        args.rocprof_path,
        "--rocprof-trace",
        args.rocprof_trace,
        "--rocprof-output-format",
        args.rocprof_output_format,
        "--all2all-backend",
        args.all2all_backend,
        "--profile-save-timeout",
        str(args.profile_save_timeout),
        "--proton-context",
        args.proton_context,
        "--proton-data",
        args.proton_data,
        (
            "--finalize-non-proton"
            if args.finalize_non_proton
            else "--no-finalize-non-proton"
        ),
    ]
    if args.max_concurrency is not None:
        command.extend(["--max-concurrency", str(args.max_concurrency)])
    for option in ("proton-mode", "proton-hook", "proton-output-format"):
        value = getattr(args, option.replace("-", "_"))
        if value is not None:
            command.extend([f"--{option}", value])
    command.append(
        "--detailed-trace-annotation"
        if args.detailed_trace_annotation
        else "--no-detailed-trace-annotation"
    )
    torch_options = {
        "torch-profiler-record-shapes": args.torch_profiler_record_shapes,
        "torch-profiler-with-memory": args.torch_profiler_with_memory,
        "torch-profiler-with-stack": args.torch_profiler_with_stack,
        "torch-profiler-with-flops": args.torch_profiler_with_flops,
        "torch-profiler-use-gzip": args.torch_profiler_use_gzip,
        "torch-profiler-dump-cuda-time-total": (
            args.torch_profiler_dump_cuda_time_total
        ),
    }
    for option, enabled in torch_options.items():
        command.append(f"--{option}" if enabled else f"--no-{option}")
    command.append(
        "--cudagraph" if args.graph_mode == "cudagraph" else "--no-cudagraph"
    )
    if case.expert_parallel:
        command.append("--enable-expert-parallel")
    return command


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def terminate_process_group(
    process: subprocess.Popen[Any], timeout: float = 60
) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=timeout)


def print_log_tail(path: Path, lines: int = 80) -> None:
    try:
        content = path.read_text(errors="replace").splitlines()
    except OSError as exc:
        print(f"  could not read {path}: {exc}", file=sys.stderr)
        return
    print(f"  --- tail of {path} ---", file=sys.stderr)
    for line in content[-lines:]:
        print(f"  {line}", file=sys.stderr)


def collect_results(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for case_path in root.glob("*/*/*/*/*/case.json"):
        relative = case_path.relative_to(root).parts
        model, workload, parallelism, graph_mode, profile_run = relative[:5]
        try:
            case_metadata = json.loads(case_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(
                {
                    "model": model,
                    "workload": workload,
                    "parallelism": parallelism,
                    "graph_mode": graph_mode,
                    "profile_run": profile_run,
                    "error": f"invalid case metadata: {exc}",
                }
            )
            continue
        result_path = case_path.with_name("results.json")
        try:
            result = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            result = {"summary": [], "errors": []}
            if case_metadata.get("succeeded"):
                failures.append(
                    {
                        "model": model,
                        "workload": workload,
                        "parallelism": parallelism,
                        "graph_mode": graph_mode,
                        "profile_run": profile_run,
                        "error": f"invalid benchmark result: {exc}",
                    }
                )
        profile_bytes = case_metadata.get("profile_bytes", 0)
        peak_disk_delta_bytes = case_metadata.get("peak_disk_delta_bytes", 0)
        if not case_metadata.get("succeeded"):
            failures.append(
                {
                    "model": model,
                    "workload": workload,
                    "parallelism": parallelism,
                    "graph_mode": graph_mode,
                    "profile_run": profile_run,
                    "error": case_metadata.get("error")
                    or f"benchmark exited {case_metadata.get('returncode')}",
                }
            )
        for row in result.get("summary", []):
            if not case_metadata.get("succeeded"):
                break
            rows.append(
                {
                    "model": model,
                    "workload": workload,
                    "parallelism": parallelism,
                    "graph_mode": graph_mode,
                    "profile_run": profile_run,
                    "profile_bytes": profile_bytes,
                    "peak_disk_delta_bytes": peak_disk_delta_bytes,
                    **row,
                }
            )
        for error in result.get("errors", []):
            failures.append(
                {
                    "model": model,
                    "workload": workload,
                    "parallelism": parallelism,
                    "graph_mode": graph_mode,
                    "profile_run": profile_run,
                    **error,
                }
            )
        for error in result.get("save_failures", []):
            failures.append(
                {
                    "model": model,
                    "workload": workload,
                    "parallelism": parallelism,
                    "graph_mode": graph_mode,
                    "profile_run": profile_run,
                    "failure_type": "profile_save",
                    **error,
                }
            )
    return rows, failures


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.unlink(missing_ok=True)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_results(root: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping plots", file=sys.stderr)
        return
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    profilers = [
        "proton:auto:shadow:tree",
        "torch:minimal",
        "torch:custom",
        "nsys:graph",
        "nsys:node",
        "rocprof:runtime",
        "rocprof:sys",
        "rocprof:kernel",
    ]
    for workload in WORKLOADS:
        for graph_mode in ("cudagraph", "eager"):
            selected = [
                row
                for row in rows
                if row["workload"] == workload
                and row["graph_mode"] == graph_mode
                and row["profiler"] in profilers
            ]
            labels = sorted({f"{r['model']}\n{r['parallelism']}" for r in selected})
            if not labels:
                continue
            metrics = (
                ("request_throughput_overhead_pct", "Request throughput loss"),
                ("mean_ttft_ms_overhead_pct", "Mean TTFT increase"),
                ("mean_tpot_ms_overhead_pct", "Mean TPOT increase"),
            )
            figure, axes = plt.subplots(
                len(metrics), 1, figsize=(max(12, len(labels) * 0.65), 12)
            )
            width = 0.8 / len(profilers)
            positions = list(range(len(labels)))
            for profiler_index, profiler in enumerate(profilers):
                by_label = {
                    f"{r['model']}\n{r['parallelism']}": r
                    for r in selected
                    if r["profiler"] == profiler
                }
                offset = (profiler_index - (len(profilers) - 1) / 2) * width
                x_values = [position + offset for position in positions]
                for axis, (metric, _) in zip(axes, metrics):
                    axis.bar(
                        x_values,
                        [
                            by_label.get(label, {}).get(metric, float("nan"))
                            for label in labels
                        ],
                        width,
                        label=profiler,
                    )
            for axis, (_, title) in zip(axes, metrics):
                axis.axhline(0, color="black", linewidth=0.8)
                axis.set_ylabel("Overhead (%)")
                axis.set_title(f"{title}: {workload}, {graph_mode}")
                axis.set_xticks(positions, labels, rotation=45, ha="right")
                axis.legend()
                axis.grid(axis="y", alpha=0.25)
            figure.tight_layout()
            figure.savefig(plots / f"{workload}_{graph_mode}.png", dpi=160)
            plt.close(figure)


def environment_metadata(args: argparse.Namespace) -> dict[str, Any]:
    def output(command: list[str]) -> str:
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False
            )
        except FileNotFoundError:
            return ""
        return (completed.stdout or completed.stderr).strip()

    return {
        "arguments": vars(args) | {"output_dir": str(args.output_dir)},
        "vllm_revision": os.environ.get("VLLM_BUILD_COMMIT") or "unknown",
        "benchmark_revision": os.environ.get("VLLM_BENCHMARK_REVISION")
        or output(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": output(["nvidia-smi", "-L"]),
        "amd_smi": output(["amd-smi", "list", "--csv"]),
        "nsys_version": output([args.nsys_path, "--version"]),
        "rocprof_version": output([args.rocprof_path, "--version"]),
        "python_packages": output(
            [
                args.python,
                "-c",
                (
                    "import torch, triton, vllm; "
                    "print('vllm=%s torch=%s triton=%s' % "
                    "(vllm.__version__, torch.__version__, triton.__version__))"
                ),
            ]
        ),
    }


def main(args: argparse.Namespace) -> int:
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    run_metadata = environment_metadata(args)
    write_json(root / "last_environment.json", run_metadata)
    if not (root / "environment.json").exists():
        write_json(root / "environment.json", run_metadata)
    cases = make_cases(args)
    print(f"Matrix contains {len(cases)} profiler/workload/topology cases")
    failures_seen = False
    for index, case in enumerate(cases, 1):
        output_dir = case_dir(root, case, args.graph_mode)
        label = f"{case.model_name}/{case.workload}/{case.parallelism}/{case.profiler}"
        command = build_command(args, case, output_dir)
        signature = case_signature(command, run_metadata)
        profiler_label = expected_profiler_label(args, case.profiler)
        if not args.force and case_complete(output_dir, profiler_label, signature):
            print(f"[{index}/{len(cases)}] SKIP {label}")
            continue
        print(f"[{index}/{len(cases)}] RUN  {label}", flush=True)
        if args.dry_run:
            print(" ".join(command))
            continue
        free_gb = shutil.disk_usage(root).free / 1024**3
        if free_gb < args.min_free_gb:
            raise RuntimeError(
                f"only {free_gb:.1f} GiB free; require --min-free-gb={args.min_free_gb}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        succeeded = False
        profile_bytes = 0
        peak_disk_delta_bytes = 0
        returncode = -1
        error = None
        metadata = {
            **asdict(case),
            "command": command,
            "signature": signature,
        }
        try:
            with (output_dir / "run.log").open("w") as log:
                free_before = shutil.disk_usage(root).free
                process = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[2],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=os.environ | {"HF_HUB_OFFLINE": "1"},
                    start_new_session=True,
                )
                try:
                    case_start = time.monotonic()
                    while process.poll() is None:
                        if (
                            args.case_timeout
                            and time.monotonic() - case_start > args.case_timeout
                        ):
                            terminate_process_group(process)
                            raise subprocess.TimeoutExpired(command, args.case_timeout)
                        free_now = shutil.disk_usage(root).free
                        peak_disk_delta_bytes = max(
                            peak_disk_delta_bytes, free_before - free_now
                        )
                        time.sleep(2)
                except BaseException:
                    terminate_process_group(process)
                    raise
            returncode = process.returncode
            result = json.loads((output_dir / "results.json").read_text())
            succeeded = returncode == 0 and result_complete(result, profiler_label)
            if not succeeded:
                error = "benchmark result is incomplete"
        except (OSError, json.JSONDecodeError, subprocess.SubprocessError) as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            profile_dirs = list(output_dir.rglob("profiles"))
            profile_bytes = sum(directory_size(path) for path in profile_dirs)
            metadata |= {
                "returncode": returncode,
                "succeeded": succeeded,
                "error": error,
                "profile_bytes": profile_bytes,
                "peak_disk_delta_bytes": peak_disk_delta_bytes,
            }
            write_json(output_dir / "case.json", metadata)
            keep = retain_profiles(args.profile_retention, case.profiler, succeeded)
            if not keep:
                for profile_dir in output_dir.rglob("profiles"):
                    shutil.rmtree(profile_dir)
        failures_seen |= not succeeded
        rows, failures = collect_results(root)
        write_csv(root / "results.csv", rows)
        write_csv(root / "failures.csv", failures)
        plot_results(root, rows)
        print(
            f"  {'PASS' if succeeded else 'FAIL'}; "
            f"raw profiles={profile_bytes / 1024**2:.1f} MiB"
        )
        if not succeeded:
            print(f"  error: {error}", file=sys.stderr)
            print_log_tail(output_dir / "run.log")
            for server_log in sorted(output_dir.rglob("server.log")):
                print_log_tail(server_log)
    if not args.dry_run:
        rows, failures = collect_results(root)
        write_csv(root / "results.csv", rows)
        write_csv(root / "failures.csv", failures)
        plot_results(root, rows)
    return 1 if failures_seen else 0


if __name__ == "__main__":
    sys.exit(main(build_parser().parse_args()))
