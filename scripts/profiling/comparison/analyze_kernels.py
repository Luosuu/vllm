"""Parse proton-viewer output and categorize GPU kernels by type.

Reads proton-viewer text output (time/ns metric), extracts top-level kernel
entries, categorizes them, and prints a summary table with per-category
GPU time breakdown.

Usage:
    python analyze_kernels.py <label> <proton_viewer_file> [<label2> <file2> ...]

Each label+file pair produces a column in the summary table.
"""

import re
import sys
from collections import defaultdict
from pathlib import Path

# Kernel category classification rules.
# Order matters: first match wins.
CATEGORY_RULES = [
    # Attention-related kernels
    ("attention", [
        r"FlashAttn",
        r"flash\d*::",
        r"prepare_varlen",
        r"reshape_and_cache",
        r"fused_rope",
        r"fused_qknorm",
        r"store_kvcache",
        r"SoftMax",
        r"page_attention",
        r"paged_attention",
    ]),
    # MLP / FFN (GEMM, activations)
    ("mlp_ffn", [
        r"gemm",
        r"gemv",
        r"splitKreduce",
        r"nvjet_tst",           # NVIDIA JIT GEMM
        r"act_and_mul",         # flashinfer SiLU
        r"silu",
        r"gelu",
        r"triton_poi_fused_mul_silu",
        r"triton_poi_fused_4",  # vLLM fused MLP kernel
    ]),
    # Normalization
    ("normalization", [
        r"RMSNorm",
        r"[Ll]ayer[Nn]orm",
        r"fused.*mean.*mul.*pow.*rsqrt",  # fused RMSNorm pattern
    ]),
    # Embedding / indexing
    ("embedding", [
        r"embedding",
        r"indexSelect",
        r"vectorized_gather",
    ]),
    # Sampling
    ("sampling", [
        r"ArgMax",
        r"topk_topp",
        r"TopK",
        r"TopP",
        r"distribution.*exponential",
        r"multinomial",
        r"tensor_kernel_scan_innermost_dim",  # cumsum for top-p
        r"RadixSort",
        r"sort_postprocess",
        r"fill_reverse_indices",
        r"fill_index_and_segment",
    ]),
]

# Proton scope markers to filter out (not actual GPU kernels).
# These contain nested kernels whose times are already aggregated
# at the top level, so including them would double-count.
SCOPE_PATTERNS = [
    re.compile(r"^execute_context_"),
    re.compile(r"^<captured_at>$"),
]


def categorize_kernel(name):
    """Classify a kernel name into a category string.

    Args:
        name: Mangled or human-readable kernel name.

    Returns:
        Category string (e.g. "attention", "mlp_ffn", "other").
    """
    for category, patterns in CATEGORY_RULES:
        for pat in patterns:
            if re.search(pat, name, re.IGNORECASE):
                return category
    return "other"


def parse_proton_viewer_output(text):
    """Extract kernel names and times from proton-viewer output.

    Parses all tree levels. Top-level entries are aggregated totals;
    nested entries are per-context breakdowns. To avoid double-counting:
    - If a kernel appears at the top level, use that value (it's the aggregate).
    - If a kernel only appears nested (inside execute_context_* scopes),
      sum its occurrences across all scopes.

    Args:
        text: Raw proton-viewer -m time/ns output string.

    Returns:
        List of (time_ns, kernel_name) tuples sorted by time descending.
    """
    # Parse all tree entries at any depth
    # Depth is determined by leading "│  " segments before "├─" or "└─"
    entry_pattern = re.compile(r'^((?:│\s+)*)[├└]─\s+([\d.]+)\s+(.+)$')

    top_level_kernels = {}  # name -> time_ns (depth 0)
    nested_kernels = defaultdict(float)  # name -> sum of time_ns (depth > 0)

    for line in text.splitlines():
        match = entry_pattern.match(line)
        if not match:
            continue

        prefix = match.group(1)
        time_ns = float(match.group(2))
        name = match.group(3).strip()

        # Skip Proton scope markers (not actual kernels)
        if any(sp.search(name) for sp in SCOPE_PATTERNS):
            continue

        # Determine depth by counting "│" prefix segments
        depth = prefix.count("│")

        if depth == 0:
            # Top-level: aggregated total
            top_level_kernels[name] = (
                top_level_kernels.get(name, 0) + time_ns
            )
        else:
            # Nested: per-context entry
            nested_kernels[name] += time_ns

    # Merge: sum top-level and nested times.
    # Top-level entries are kernels that ran outside CUDA graph replay.
    # Nested entries (under <captured_at>) are kernels from CUDA graph replay.
    # Both represent real GPU time and must be summed.
    merged = dict(top_level_kernels)
    for name, time_ns in nested_kernels.items():
        merged[name] = merged.get(name, 0) + time_ns

    # Convert to sorted list
    kernels = [(time_ns, name) for name, time_ns in merged.items()]
    kernels.sort(key=lambda x: x[0], reverse=True)
    return kernels


def aggregate_by_category(kernels):
    """Sum kernel times by category.

    Args:
        kernels: List of (time_ns, name) tuples.

    Returns:
        Dict mapping category -> total_time_ns, sorted by time descending.
    """
    totals = defaultdict(float)
    for time_ns, name in kernels:
        cat = categorize_kernel(name)
        totals[cat] += time_ns
    # Return sorted by time descending
    return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


def format_time(ns):
    """Format nanoseconds as a human-readable string.

    Args:
        ns: Time in nanoseconds.

    Returns:
        Formatted string (e.g. "12.34 ms", "567.89 us").
    """
    if ns >= 1e9:
        return f"{ns / 1e9:.2f} s"
    elif ns >= 1e6:
        return f"{ns / 1e6:.2f} ms"
    elif ns >= 1e3:
        return f"{ns / 1e3:.2f} us"
    return f"{ns:.0f} ns"


def print_top_kernels(label, kernels, top_n=20):
    """Print the top N kernels by GPU time for a given profile.

    Args:
        label: Display label for the profile.
        kernels: List of (time_ns, name) tuples.
        top_n: Number of top kernels to display.
    """
    print(f"\n{'=' * 80}")
    print(f"Top {top_n} Kernels by GPU Time: {label}")
    print(f"{'=' * 80}")
    print(f"{'Rank':<6} {'Time':>12} {'Category':<16} {'Kernel'}")
    print(f"{'-' * 6} {'-' * 12} {'-' * 16} {'-' * 44}")

    for i, (time_ns, name) in enumerate(kernels[:top_n], 1):
        cat = categorize_kernel(name)
        # Truncate long kernel names for display
        display_name = name if len(name) <= 80 else name[:77] + "..."
        print(f"{i:<6} {format_time(time_ns):>12} {cat:<16} {display_name}")


def print_category_comparison(profiles):
    """Print a side-by-side category breakdown table.

    Args:
        profiles: Dict mapping label -> category_totals dict.
    """
    # Collect all categories across all profiles
    all_categories = set()
    for totals in profiles.values():
        all_categories.update(totals.keys())

    # Compute grand totals per profile
    grand_totals = {}
    for label, totals in profiles.items():
        grand_totals[label] = sum(totals.values())

    labels = list(profiles.keys())

    print(f"\n{'=' * 80}")
    print("Kernel Category Breakdown (GPU Time)")
    print(f"{'=' * 80}")

    # Header with label names
    label_header = f"{'':16}"
    for label in labels:
        label_header += f" {label:>20}  "
    print(label_header)
    header = f"{'Category':<16}"
    for _ in labels:
        header += f" {'Time':>12} {'%':>6}  "
    print(header)
    print("-" * len(header))

    # Sort categories by max time across profiles
    sorted_cats = sorted(
        all_categories,
        key=lambda c: max(profiles[l].get(c, 0) for l in labels),
        reverse=True,
    )

    for cat in sorted_cats:
        row = f"{cat:<16}"
        for label in labels:
            time_ns = profiles[label].get(cat, 0)
            grand = grand_totals[label]
            pct = (time_ns / grand * 100) if grand > 0 else 0
            row += f" {format_time(time_ns):>12} {pct:>5.1f}%  "
        print(row)

    # Totals row
    row = f"{'TOTAL':<16}"
    for label in labels:
        row += f" {format_time(grand_totals[label]):>12} {'100.0':>5}%  "
    print("-" * len(header))
    print(row)


def main():
    """Entry point: parse arguments, process files, print analysis."""
    if len(sys.argv) < 3 or len(sys.argv) % 2 != 1:
        print(f"Usage: {sys.argv[0]} <label> <file> [<label2> <file2> ...]")
        sys.exit(1)

    # Parse label+file pairs
    pairs = []
    for i in range(1, len(sys.argv), 2):
        label = sys.argv[i]
        filepath = sys.argv[i + 1]
        pairs.append((label, filepath))

    profiles = {}  # label -> category_totals
    all_kernels = {}  # label -> kernel list

    for label, filepath in pairs:
        path = Path(filepath)
        if not path.exists():
            print(f"WARNING: File not found, skipping: {filepath}")
            continue

        text = path.read_text()
        kernels = parse_proton_viewer_output(text)

        if not kernels:
            print(f"WARNING: No kernels found in {filepath}")
            continue

        all_kernels[label] = kernels
        profiles[label] = aggregate_by_category(kernels)

    if not profiles:
        print("ERROR: No valid profile data found")
        sys.exit(1)

    # Print top kernels for each profile
    for label, kernels in all_kernels.items():
        print_top_kernels(label, kernels)

    # Print category comparison
    print_category_comparison(profiles)


if __name__ == "__main__":
    main()
