#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize Proton Hatchet profiles for taxonomy-based diagnosis."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CATEGORY_PATTERNS = (
    (
        "communication",
        re.compile(
            r"nccl|allreduce|allgather|reducescatter|all_to_all|alltoall",
            re.IGNORECASE,
        ),
    ),
    (
        "attention",
        re.compile(
            r"flashattn|flash.*attn|reshape_and_cache|prepare_varlen",
            re.IGNORECASE,
        ),
    ),
    ("gemm_moe", re.compile(r"matmul|nvjet|cublas|gemm", re.IGNORECASE)),
    (
        "routing",
        re.compile(
            r"routing|topk|bitmatrix|reduce_grouped|pack_bitmatrix",
            re.IGNORECASE,
        ),
    ),
    (
        "normalization",
        re.compile(r"rms_norm|rmsnorm|layer_norm", re.IGNORECASE),
    ),
)
SCOPE_PATTERN = re.compile(r"execute_(?:\d+_)?context_(\d+)\(.*\)_generation_\d+\(.*\)")


@dataclass
class ProfileSummary:
    """Taxonomy evidence derived from one Proton profile."""

    profile: str
    total_gpu_ms: float
    total_cpu_ms: float
    kernel_count: int
    categories_ms: dict[str, float]
    categories_pct: dict[str, float]
    short_kernel_count_pct: dict[str, float]
    short_kernel_time_pct: dict[str, float]
    mixed_context_ms: float
    mixed_context_pct: float
    pure_decode_ms: float
    pure_decode_pct: float
    context_requests: float
    context_tokens: float
    generation_requests: float
    generation_tokens: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Hatchet files or directories searched recursively for Hatchet files.",
    )
    parser.add_argument("--viewer", default="proton-viewer")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden and CUDA Graph capture sidecar profiles.",
    )
    return parser


def discover_profiles(paths: Iterable[Path], include_hidden: bool) -> list[Path]:
    profiles: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = (path,)
        elif path.is_dir():
            candidates = path.rglob("*.hatchet")
        else:
            raise FileNotFoundError(path)
        for candidate in candidates:
            is_capture_sidecar = "_cuda_graph_capture" in candidate.stem
            if include_hidden or (
                not candidate.name.startswith(".") and not is_capture_sidecar
            ):
                profiles.add(candidate.resolve())
    if not profiles:
        raise ValueError("no Hatchet profiles found")
    return sorted(profiles)


def viewer_root_time_ms(viewer: str, profile: Path) -> float:
    completed = subprocess.run(
        [viewer, "-m", "time/ms", "-d", "1", str(profile)],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in completed.stdout.splitlines():
        match = re.match(r"\s*([0-9.]+)\s+ROOT\s*$", line)
        if match:
            return float(match.group(1))
    raise ValueError(f"proton-viewer did not report ROOT time for {profile}")


def walk_nodes(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    yield node
    for child in node.get("children", ()):
        yield from walk_nodes(child)


def leaf_nodes(node: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for item in walk_nodes(node):
        if not item.get("children") and "frame" in item:
            yield item


def subtree_time_ns(node: dict[str, Any]) -> int:
    return sum(
        int(item.get("metrics", {}).get("time (ns)", 0)) for item in leaf_nodes(node)
    )


def category(name: str) -> str:
    for label, pattern in CATEGORY_PATTERNS:
        if pattern.search(name):
            return label
    return "other"


def scope_metric(scopes: Iterable[dict[str, Any]], name: str) -> float:
    return sum(float(scope.get("metrics", {}).get(name, 0)) for scope in scopes)


def summarize_profile(viewer: str, profile: Path) -> ProfileSummary:
    document = json.loads(profile.read_text())
    root = document[0] if isinstance(document, list) else document
    execution_scopes = []
    for child in root.get("children", ()):
        name = str(child.get("frame", {}).get("name", ""))
        if SCOPE_PATTERN.fullmatch(name):
            execution_scopes.append(child)

    # A context-linked runtime profile also retains startup capture nodes at
    # ROOT. Restrict runtime summaries to execute scopes so capture does not
    # inflate GPU time, kernel counts, or category shares. Capture-only
    # sidecars intentionally fall back to their complete tree.
    analysis_roots = execution_scopes or [root]
    if execution_scopes:
        total_ns = sum(subtree_time_ns(scope) for scope in execution_scopes)
    else:
        total_ns = int(viewer_root_time_ms(viewer, profile) * 1_000_000)
    total_gpu_ms = total_ns / 1_000_000
    total_cpu_ns = scope_metric(execution_scopes, "cpu_time (ns)")
    categories_ns = {label: 0 for label, _ in CATEGORY_PATTERNS}
    categories_ns["other"] = 0
    kernel_count = 0
    short_count = {10: 0, 20: 0, 50: 0}
    short_time = {10: 0, 20: 0, 50: 0}

    for analysis_root in analysis_roots:
        for node in leaf_nodes(analysis_root):
            metrics = node.get("metrics", {})
            time_ns = int(metrics.get("time (ns)", 0))
            count = int(metrics.get("count", 0))
            name = str(node.get("frame", {}).get("name", ""))
            categories_ns[category(name)] += time_ns
            kernel_count += count
            if count:
                average_us = time_ns / count / 1_000
                for threshold in short_count:
                    if average_us < threshold:
                        short_count[threshold] += count
                        short_time[threshold] += time_ns

    mixed_context_ns = 0
    pure_decode_ns = 0
    for child in execution_scopes:
        name = str(child.get("frame", {}).get("name", ""))
        match = SCOPE_PATTERN.fullmatch(name)
        assert match is not None
        scope_time_ns = subtree_time_ns(child)
        if int(match.group(1)):
            mixed_context_ns += scope_time_ns
        else:
            pure_decode_ns += scope_time_ns

    categories_ms = {label: value / 1_000_000 for label, value in categories_ns.items()}
    categories_pct = {
        label: 100 * value / total_ns if total_ns else 0
        for label, value in categories_ns.items()
    }
    short_kernel_count_pct = {
        f"lt_{threshold}us": 100 * value / kernel_count if kernel_count else 0
        for threshold, value in short_count.items()
    }
    short_kernel_time_pct = {
        f"lt_{threshold}us": 100 * value / total_ns if total_ns else 0
        for threshold, value in short_time.items()
    }
    return ProfileSummary(
        profile=str(profile),
        total_gpu_ms=total_gpu_ms,
        total_cpu_ms=total_cpu_ns / 1_000_000,
        kernel_count=kernel_count,
        categories_ms=categories_ms,
        categories_pct=categories_pct,
        short_kernel_count_pct=short_kernel_count_pct,
        short_kernel_time_pct=short_kernel_time_pct,
        mixed_context_ms=mixed_context_ns / 1_000_000,
        mixed_context_pct=100 * mixed_context_ns / total_ns if total_ns else 0,
        pure_decode_ms=pure_decode_ns / 1_000_000,
        pure_decode_pct=100 * pure_decode_ns / total_ns if total_ns else 0,
        context_requests=scope_metric(execution_scopes, "num_context_requests"),
        context_tokens=scope_metric(execution_scopes, "num_context_tokens"),
        generation_requests=scope_metric(execution_scopes, "num_generation_requests"),
        generation_tokens=scope_metric(execution_scopes, "num_generation_tokens"),
    )


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def render_markdown(summaries: list[ProfileSummary]) -> str:
    lines = [
        "# Proton profile summary",
        "",
        (
            "| Profile | GPU ms | CPU execute ms | Kernels | "
            "Mixed context % | Pure decode % |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {Path(item.profile).name} | {item.total_gpu_ms:.1f} | "
            f"{item.total_cpu_ms:.1f} | {item.kernel_count} | "
            f"{item.mixed_context_pct:.2f} | "
            f"{item.pure_decode_pct:.2f} |"
        )
    times = [item.total_gpu_ms for item in summaries]
    imbalance = 100 * (max(times) / min(times) - 1) if min(times) else 0
    lines.extend(
        [
            "",
            f"Rank GPU-time imbalance (max/min - 1): **{imbalance:.2f}%**",
            "",
            "## Mean leaf-kernel category share",
            "",
            "| Category | Time % |",
            "| --- | ---: |",
        ]
    )
    for label in (*[item[0] for item in CATEGORY_PATTERNS], "other"):
        value = mean(item.categories_pct[label] for item in summaries)
        lines.append(f"| {label} | {value:.2f} |")
    lines.extend(
        [
            "",
            "## Mean short-kernel distribution",
            "",
            "| Threshold | Kernel-count % | GPU-time % |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in ("lt_10us", "lt_20us", "lt_50us"):
        count_pct = mean(item.short_kernel_count_pct[key] for item in summaries)
        time_pct = mean(item.short_kernel_time_pct[key] for item in summaries)
        lines.append(f"| {key.replace('_', ' ')} | {count_pct:.2f} | {time_pct:.2f} |")
    lines.extend(
        [
            "",
            (
                "Heuristic categories classify leaf kernels by name. Validate "
                "`other` and unfamiliar names before drawing conclusions."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = build_parser().parse_args()
    profiles = discover_profiles(args.paths, args.include_hidden)
    summaries = [summarize_profile(args.viewer, profile) for profile in profiles]
    if args.format == "json":
        print(json.dumps([asdict(item) for item in summaries], indent=2))
    else:
        print(render_markdown(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
