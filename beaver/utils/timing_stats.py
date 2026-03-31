"""Summarize timing profiles from BEAVER run logs.

Supports both individual run log folders and batch result directories.

Usage:
    # Single run
    python beaver/utils/timing_stats.py logging/logs_20260316185322

    # Batch results (walks model/experiment dirs)
    python beaver/utils/timing_stats.py logging/batch_results/my_batch --batch

    # CSV output
    python beaver/utils/timing_stats.py logging/logs_20260316185322 -o timing.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from beaver.logging import get_profile_data


def compute_timing_stats(all_profile_data: dict) -> dict:
    """Compute timing statistics from profile data.

    Returns a dict mapping each timing key to {avg, min, max, std, median}
    computed over ALL transitions across all instances.
    Also includes per-instance totals (avg/min/max of the sum across transitions).
    """
    if not all_profile_data:
        return {}

    # Collect all values per key across all instances and transitions
    all_values = {}
    # Per-instance total time per key
    instance_totals = {}

    for file, entries in all_profile_data.items():
        file_sums = {}
        for entry in entries:
            for key, value in entry.items():
                all_values.setdefault(key, []).append(value)
                file_sums.setdefault(key, 0.0)
                file_sums[key] += value
        for key, total in file_sums.items():
            instance_totals.setdefault(key, []).append(total)

    stats = {}
    for key in all_values:
        vals = np.array(all_values[key])
        totals = np.array(instance_totals.get(key, []))
        stats[key] = {
            # Per-transition stats
            "avg": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "std": float(np.std(vals)),
            "median": float(np.median(vals)),
            # Per-instance total stats
            "instance_avg_total": float(np.mean(totals)) if len(totals) > 0 else 0.0,
            "instance_min_total": float(np.min(totals)) if len(totals) > 0 else 0.0,
            "instance_max_total": float(np.max(totals)) if len(totals) > 0 else 0.0,
        }

    return stats


def print_timing_report(label: str, stats: dict):
    """Print a formatted timing report."""
    if not stats:
        print(f"No profile data found for {label}")
        return

    print(f"\n{'=' * 85}")
    print(f"  TIMING PROFILE: {label}")
    print(f"{'=' * 85}")

    # Header
    print(f"  {'task':<25s} {'avg':>10s} {'median':>10s} {'min':>10s} {'max':>10s} {'std':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10} {'-' * 10}")

    # Per-transition stats
    print("  Per-transition (seconds):")
    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"    {key:<23s} {s['avg']:>10.4f} {s['median']:>10.4f} {s['min']:>10.4f} {s['max']:>10.4f} {s['std']:>10.4f}")

    # Per-instance totals
    print()
    print(f"  {'task':<25s} {'avg_total':>10s} {'min_total':>10s} {'max_total':>10s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 10} {'-' * 10}")
    print("  Per-instance total (seconds):")
    for key in sorted(stats.keys()):
        s = stats[key]
        print(f"    {key:<23s} {s['instance_avg_total']:>10.2f} {s['instance_min_total']:>10.2f} {s['instance_max_total']:>10.2f}")

    print(f"{'=' * 85}\n")


def find_batch_log_dirs(batch_dir: Path) -> list[tuple[str, str, Path]]:
    """Walk a batch results dir and find (model, experiment, logs_dir) tuples."""
    results = []
    if not batch_dir.is_dir():
        return results
    for model_dir in sorted(batch_dir.iterdir()):
        if not model_dir.is_dir() or model_dir.name.startswith("."):
            continue
        model_name = model_dir.name.replace("--", "/")
        for exp_dir in sorted(model_dir.iterdir()):
            if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                continue
            log_dirs = sorted(
                [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")],
                reverse=True,
            )
            if log_dirs:
                results.append((model_name, exp_dir.name, log_dirs[0]))
    return results


def save_csv(all_rows: list[dict], output_path: Path):
    """Save timing stats to CSV."""
    if not all_rows:
        return
    fieldnames = list(all_rows[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"Timing stats saved to: {output_path}")


def stats_to_csv_rows(stats: dict, model: str = "", experiment: str = "") -> list[dict]:
    """Flatten timing stats into CSV rows (one row per timing key)."""
    rows = []
    for key in sorted(stats.keys()):
        s = stats[key]
        row = {"task": key, "avg": s["avg"], "median": s["median"],
               "min": s["min"], "max": s["max"], "std": s["std"],
               "instance_avg_total": s["instance_avg_total"],
               "instance_min_total": s["instance_min_total"],
               "instance_max_total": s["instance_max_total"]}
        if model:
            row = {"model": model, "experiment": experiment, **row}
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize timing profiles from BEAVER run logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python beaver/utils/timing_stats.py logging/logs_20260316185322\n"
            "  python beaver/utils/timing_stats.py logging/batch_results/my_batch --batch\n"
            "  python beaver/utils/timing_stats.py logging/logs_20260316185322 -o timing.csv\n"
        ),
    )
    parser.add_argument("path", type=str, help="Run log folder or batch results directory")
    parser.add_argument("--batch", action="store_true", help="Treat path as a batch results directory")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")

    args = parser.parse_args()
    path = Path(args.path)

    if not path.is_dir():
        print(f"Error: {path} does not exist or is not a directory")
        return

    if args.batch:
        entries = find_batch_log_dirs(path)
        if not entries:
            print(f"No log directories found under {path}")
            return

        all_csv_rows = []
        for model, exp, log_dir in entries:
            profile_data = get_profile_data(log_dir)
            stats = compute_timing_stats(profile_data)
            print_timing_report(f"{model} / {exp}", stats)
            all_csv_rows.extend(stats_to_csv_rows(stats, model, exp))

        output_path = Path(args.output) if args.output else path / "timing_stats.csv"
        save_csv(all_csv_rows, output_path)
    else:
        profile_data = get_profile_data(path)
        stats = compute_timing_stats(profile_data)
        print_timing_report(path.name, stats)

        csv_rows = stats_to_csv_rows(stats)
        output_path = Path(args.output) if args.output else path / "timing_stats.csv"
        save_csv(csv_rows, output_path)


if __name__ == "__main__":
    main()
