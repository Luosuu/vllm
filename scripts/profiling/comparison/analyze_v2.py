"""Analyze and compare per-rate Proton profiles from vLLM vs SGLang (v2).

Runs proton-viewer on all per-rate .hatchet files from both frameworks,
categorizes kernels, produces per-rate comparison tables, scaling analysis,
and a self-contained Markdown report.

Usage:
    python scripts/profiling/comparison/analyze_v2.py

Expects .hatchet files in:
    profiling_output/comparison_v2/{vllm,sglang}/online/rate_{4,16,32,inf}/
"""

import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Reuse kernel parsing and categorization from analyze_kernels.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from analyze_kernels import (
    aggregate_by_category,
    categorize_kernel,
    format_time,
    parse_proton_viewer_output,
)

REPO_ROOT = SCRIPT_DIR.parent.parent.parent
PROFILE_ROOT = REPO_ROOT / "profiling_output" / "comparison_v2"
ANALYSIS_DIR = PROFILE_ROOT / "analysis"
REPORT_PATH = PROFILE_ROOT / "report.md"

# Default request rates (must match v2 profiling scripts)
RATES = ["4", "16", "32", "inf"]
FRAMEWORKS = ["vllm", "sglang"]

# Ordered categories for consistent table rendering
CATEGORY_ORDER = [
    "mlp_ffn",
    "attention",
    "normalization",
    "sampling",
    "embedding",
    "other",
]

# V2 methodology documentation
METHODOLOGY = {
    "model": "Qwen/Qwen3-1.7B (bf16)",
    "hardware": "1x NVIDIA H100 80GB",
    "profiler": "Triton Proton (context=shadow, data=tree, hook=triton)",
    "cuda_graphs": "ENABLED for both frameworks (real-world config)",
    "workload": "100 prompts, input_len=1024, output_len=256, seed=42",
    "warmup": "10 prompts at inf rate (not profiled) before each rate",
    "request_rates": "4, 16, 32, inf req/s",
    "session_isolation": "Server restarted per rate for separate .hatchet files",
    "cupti": "Proton-only — no torch.profiler/PyTorch GPU profiler",
}

# What changed from v1 to v2
V1_TO_V2_CHANGES = [
    "CUDA graphs ON for both (v1: ON for vLLM, OFF for SGLang)",
    "Per-rate sessions via server restart (v1: accumulated single session)",
    "Proton-only profiling (v1: torch.profiler also active for SGLang)",
    "Higher load range: 4, 16, 32, inf req/s (v1: 1, 4 req/s)",
    "100 prompts per rate (v1: 50 prompts total)",
    "SGLang --num-steps=30000 (v1: 100, caused early cutoff)",
    "Identical warmup for both frameworks (10 prompts at inf rate)",
]


def find_hatchet_file(directory):
    """Find the first .hatchet file in a directory.

    Args:
        directory: Path to search for .hatchet files.

    Returns:
        Path to the first .hatchet file found, or None.
    """
    if not directory.is_dir():
        return None
    hatchet_files = sorted(directory.glob("*.hatchet*"))
    return hatchet_files[0] if hatchet_files else None


def run_proton_viewer(hatchet_path):
    """Run proton-viewer on a .hatchet file and return its text output.

    Args:
        hatchet_path: Path to the .hatchet file.

    Returns:
        Raw proton-viewer output string.
    """
    result = subprocess.run(
        ["proton-viewer", "-m", "time/ns", str(hatchet_path)],
        capture_output=True,
        text=True,
    )
    result.check_returncode()
    return result.stdout


def discover_and_analyze():
    """Discover .hatchet files and run proton-viewer on each.

    Returns:
        Dict mapping (framework, rate) -> {
            "kernels": [(time_ns, name), ...],
            "categories": {category: total_ns, ...},
            "hatchet_path": Path,
            "raw_output": str,
        }
    """
    profiles = {}

    for fw in FRAMEWORKS:
        for rate in RATES:
            rate_dir = PROFILE_ROOT / fw / "online" / f"rate_{rate}"
            hatchet_path = find_hatchet_file(rate_dir)

            if hatchet_path is None:
                print(f"  WARNING: No .hatchet in {rate_dir} — skipping {fw}/rate_{rate}")
                continue

            print(f"  Processing: {fw}/rate_{rate} -> {hatchet_path.name}")
            raw_output = run_proton_viewer(hatchet_path)
            kernels = parse_proton_viewer_output(raw_output)

            if not kernels:
                print(f"    WARNING: No kernels parsed from {hatchet_path}")
                continue

            profiles[(fw, rate)] = {
                "kernels": kernels,
                "categories": aggregate_by_category(kernels),
                "hatchet_path": hatchet_path,
                "raw_output": raw_output,
            }

    return profiles


def save_raw_outputs(profiles):
    """Save raw proton-viewer output to analysis directory.

    Args:
        profiles: Dict from discover_and_analyze().
    """
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    for (fw, rate), data in profiles.items():
        output_path = ANALYSIS_DIR / f"{fw}-rate_{rate}.txt"
        output_path.write_text(data["raw_output"])
        print(f"    Saved: {output_path.name}")


def md_table(headers, rows, alignments=None):
    """Build a Markdown table string.

    Args:
        headers: List of column header strings.
        rows: List of row tuples/lists.
        alignments: Optional list of "l", "r", or "c" per column.

    Returns:
        Markdown table as a string.
    """
    if alignments is None:
        alignments = ["l"] * len(headers)

    # Compute column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(cells):
        """Format a single row with proper alignment."""
        parts = []
        for i, cell in enumerate(cells):
            s = str(cell)
            if alignments[i] == "r":
                parts.append(s.rjust(widths[i]))
            else:
                parts.append(s.ljust(widths[i]))
        return "| " + " | ".join(parts) + " |"

    # Separator line
    sep_parts = []
    for i, w in enumerate(widths):
        if alignments[i] == "r":
            sep_parts.append("-" * (w - 1) + ":")
        elif alignments[i] == "c":
            sep_parts.append(":" + "-" * (w - 2) + ":")
        else:
            sep_parts.append("-" * w)

    lines = [
        fmt_row(headers),
        "| " + " | ".join(sep_parts) + " |",
    ]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def category_table_for_rate(profiles, rate):
    """Build a side-by-side category comparison table for one rate.

    Args:
        profiles: Dict from discover_and_analyze().
        rate: Rate string (e.g. "4", "inf").

    Returns:
        Markdown table string, or None if no data for this rate.
    """
    available = []
    for fw in FRAMEWORKS:
        if (fw, rate) in profiles:
            available.append(fw)
    if not available:
        return None

    # Collect all categories
    all_cats = set()
    grand = {}
    for fw in available:
        cats = profiles[(fw, rate)]["categories"]
        all_cats.update(cats.keys())
        grand[fw] = sum(cats.values())

    ordered_cats = [c for c in CATEGORY_ORDER if c in all_cats]
    ordered_cats += sorted(all_cats - set(CATEGORY_ORDER))

    headers = ["Category"]
    alignments = ["l"]
    for fw in available:
        headers += [f"{fw} Time", f"{fw} %"]
        alignments += ["r", "r"]

    rows = []
    for cat in ordered_cats:
        row = [cat]
        for fw in available:
            ns = profiles[(fw, rate)]["categories"].get(cat, 0)
            pct = (ns / grand[fw] * 100) if grand[fw] > 0 else 0
            row += [format_time(ns), f"{pct:.1f}%"]
        rows.append(row)

    # Total row
    total_row = ["**TOTAL**"]
    for fw in available:
        total_row += [f"**{format_time(grand[fw])}**", "**100.0%**"]
    rows.append(total_row)

    return md_table(headers, rows, alignments)


def scaling_table(profiles, framework):
    """Build a table showing how category times change across rates.

    Args:
        profiles: Dict from discover_and_analyze().
        framework: Framework name ("vllm" or "sglang").

    Returns:
        Markdown table string, or None if insufficient data.
    """
    available_rates = [r for r in RATES if (framework, r) in profiles]
    if len(available_rates) < 2:
        return None

    # Collect all categories across rates
    all_cats = set()
    grand = {}
    for rate in available_rates:
        cats = profiles[(framework, rate)]["categories"]
        all_cats.update(cats.keys())
        grand[rate] = sum(cats.values())

    ordered_cats = [c for c in CATEGORY_ORDER if c in all_cats]
    ordered_cats += sorted(all_cats - set(CATEGORY_ORDER))

    headers = ["Category"]
    alignments = ["l"]
    for rate in available_rates:
        headers += [f"{rate} req/s", f"%"]
        alignments += ["r", "r"]

    rows = []
    for cat in ordered_cats:
        row = [cat]
        for rate in available_rates:
            ns = profiles[(framework, rate)]["categories"].get(cat, 0)
            pct = (ns / grand[rate] * 100) if grand[rate] > 0 else 0
            row += [format_time(ns), f"{pct:.1f}%"]
        rows.append(row)

    # Total row
    total_row = ["**TOTAL**"]
    for rate in available_rates:
        total_row += [f"**{format_time(grand[rate])}**", "**100.0%**"]
    rows.append(total_row)

    return md_table(headers, rows, alignments)


def top_kernels_table(kernels, top_n=15):
    """Build a Markdown table of top N kernels.

    Args:
        kernels: List of (time_ns, name) tuples, sorted descending.
        top_n: Number of kernels to include.

    Returns:
        Markdown table string.
    """
    headers = ["Rank", "Time", "Category", "Kernel"]
    rows = []
    for i, (time_ns, name) in enumerate(kernels[:top_n], 1):
        cat = categorize_kernel(name)
        short = name if len(name) <= 70 else name[:67] + "..."
        rows.append((str(i), format_time(time_ns), cat, f"`{short}`"))
    return md_table(headers, rows, ["r", "r", "l", "l"])


def find_highest_load_rate(profiles):
    """Find the highest load rate that has data for at least one framework.

    Args:
        profiles: Dict from discover_and_analyze().

    Returns:
        Rate string (e.g. "inf", "32"), or None.
    """
    # Check rates in reverse order (inf is highest load)
    for rate in reversed(RATES):
        for fw in FRAMEWORKS:
            if (fw, rate) in profiles:
                return rate
    return None


def build_report(profiles):
    """Build the full v2 Markdown report.

    Args:
        profiles: Dict from discover_and_analyze().

    Returns:
        Report content as a string.
    """
    sections = []

    # Title
    sections.append(
        "# vLLM vs SGLang: Fair Online Profiling Comparison (v2)\n"
    )
    sections.append(f"*Generated: {date.today().isoformat()}*\n")

    # --- Section 1: Executive Summary ---
    sections.append("## 1. Executive Summary\n")

    # Compute summary stats
    vllm_totals = {}
    sglang_totals = {}
    for rate in RATES:
        if ("vllm", rate) in profiles:
            vllm_totals[rate] = sum(profiles[("vllm", rate)]["categories"].values())
        if ("sglang", rate) in profiles:
            sglang_totals[rate] = sum(profiles[("sglang", rate)]["categories"].values())

    bullets = []
    bullets.append(
        "This report compares vLLM and SGLang kernel-level GPU performance "
        "under fair conditions: CUDA graphs enabled for both, Proton-only "
        "profiling (no torch.profiler CUPTI contention), per-rate isolated "
        "sessions, and identical workloads."
    )

    # Per-rate total comparison
    for rate in RATES:
        v_total = vllm_totals.get(rate)
        s_total = sglang_totals.get(rate)
        if v_total is not None and s_total is not None:
            ratio = v_total / s_total if s_total > 0 else float("inf")
            bullets.append(
                f"- **{rate} req/s**: vLLM total kernel time "
                f"{format_time(v_total)} vs SGLang {format_time(s_total)} "
                f"({ratio:.2f}x)"
            )

    sections.append("\n".join(bullets) + "\n")

    # --- Section 2: Methodology ---
    sections.append("## 2. Methodology\n")
    sections.append("### Configuration\n")
    rows = [(k.replace("_", " ").title(), v) for k, v in METHODOLOGY.items()]
    sections.append(md_table(["Parameter", "Value"], rows))
    sections.append("")

    sections.append("### Changes from v1\n")
    for change in V1_TO_V2_CHANGES:
        sections.append(f"- {change}")
    sections.append("")

    # --- Section 3: Per-Rate Comparison ---
    sections.append("## 3. Per-Rate Comparison\n")
    sections.append(
        "Kernel category breakdown at each request rate, comparing "
        "vLLM and SGLang side-by-side.\n"
    )

    for rate in RATES:
        table = category_table_for_rate(profiles, rate)
        if table:
            sections.append(f"### Rate: {rate} req/s\n")
            sections.append(table)
            sections.append("")

    # --- Section 4: Scaling Analysis ---
    sections.append("## 4. Scaling Analysis\n")
    sections.append(
        "How kernel time changes as request rate increases from "
        f"{RATES[0]} to {RATES[-1]} req/s.\n"
    )

    for fw in FRAMEWORKS:
        table = scaling_table(profiles, fw)
        if table:
            sections.append(f"### {fw} — Scaling Across Rates\n")
            sections.append(table)
            sections.append("")

    # --- Section 5: Top Kernel Comparison ---
    highest_rate = find_highest_load_rate(profiles)
    sections.append("## 5. Top Kernel Comparison\n")
    if highest_rate:
        sections.append(
            f"Top 15 kernels per framework at the highest-load rate "
            f"({highest_rate} req/s).\n"
        )
        for fw in FRAMEWORKS:
            key = (fw, highest_rate)
            if key in profiles:
                sections.append(f"### {fw} — Top 15 Kernels at {highest_rate} req/s\n")
                sections.append(top_kernels_table(profiles[key]["kernels"]))
                sections.append("")
    else:
        sections.append("No profile data available for top kernel comparison.\n")

    # --- Section 6: Observations & Recommendations ---
    sections.append("## 6. Observations & Recommendations\n")
    sections.append(
        "Observations below are generated from the kernel category data. "
        "Rerun this script after profiling to get updated analysis.\n"
    )

    # Generate data-driven observations
    observations = []

    # Compare MLP proportions at highest rate
    if highest_rate:
        for fw in FRAMEWORKS:
            key = (fw, highest_rate)
            if key in profiles:
                cats = profiles[key]["categories"]
                total = sum(cats.values())
                if total > 0:
                    mlp_pct = cats.get("mlp_ffn", 0) / total * 100
                    attn_pct = cats.get("attention", 0) / total * 100
                    observations.append(
                        f"- **{fw}** at {highest_rate} req/s: "
                        f"MLP/FFN = {mlp_pct:.1f}%, Attention = {attn_pct:.1f}%"
                    )

    # Scaling trend observation
    for fw in FRAMEWORKS:
        available_rates = [r for r in RATES if (fw, r) in profiles]
        if len(available_rates) >= 2:
            first_rate = available_rates[0]
            last_rate = available_rates[-1]
            first_total = sum(profiles[(fw, first_rate)]["categories"].values())
            last_total = sum(profiles[(fw, last_rate)]["categories"].values())
            if first_total > 0:
                growth = last_total / first_total
                observations.append(
                    f"- **{fw}** total kernel time scales "
                    f"{growth:.2f}x from {first_rate} to {last_rate} req/s"
                )

    if observations:
        sections.append("### Key Observations\n")
        sections.append("\n".join(observations))
        sections.append("")

    sections.append("### Recommendations\n")
    sections.append(
        "1. **Profile with `data=full`** for memory bandwidth and occupancy metrics.\n"
        "2. **Investigate dominant kernels** in each framework's MLP/FFN category "
        "to identify optimization opportunities.\n"
        "3. **Compare attention implementations** — both use Flash Attention variants "
        "but may differ in KV cache management overhead.\n"
        "4. **Test with larger models** to see if scaling trends hold beyond 1.7B parameters.\n"
    )

    return "\n".join(sections)


def main():
    """Discover profiles, run proton-viewer, generate analysis and report."""
    print("============================================")
    print("v2 Fair Comparison Analysis")
    print("============================================")
    print(f"Profile root: {PROFILE_ROOT}")
    print(f"Analysis dir: {ANALYSIS_DIR}")
    print("")

    # Discover and analyze all per-rate .hatchet files
    print("Discovering and analyzing .hatchet files...")
    profiles = discover_and_analyze()

    if not profiles:
        print("ERROR: No profile data found. Run v2 profiling scripts first.")
        sys.exit(1)

    print(f"\nFound {len(profiles)} profiles:")
    for (fw, rate) in sorted(profiles.keys()):
        total = sum(profiles[(fw, rate)]["categories"].values())
        print(f"  {fw}/rate_{rate}: {format_time(total)} total kernel time")

    # Save raw proton-viewer output
    print("\nSaving raw proton-viewer output...")
    save_raw_outputs(profiles)

    # Print console summary
    print("\n--------------------------------------------")
    print("Per-Rate Category Comparison")
    print("--------------------------------------------")
    for rate in RATES:
        table = category_table_for_rate(profiles, rate)
        if table:
            print(f"\n--- Rate: {rate} req/s ---")
            print(table)

    # Generate and write report
    print("\n--------------------------------------------")
    print("Generating report...")
    print("--------------------------------------------")
    report = build_report(profiles)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report)
    print(f"Report written to {REPORT_PATH}")

    print("\n============================================")
    print("Analysis complete")
    print("============================================")
    print(f"Raw output:  {ANALYSIS_DIR}/*.txt")
    print(f"Report:      {REPORT_PATH}")


if __name__ == "__main__":
    main()
