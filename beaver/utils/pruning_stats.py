"""Compute pruning statistics from BEAVER run logs.

Supports both individual run log folders and batch result directories.

Usage:
    # Single run
    python -m beaver.utils.pruning_stats logging/logs_20260316185322

    # Batch results (walks model/experiment dirs)
    python -m beaver.utils.pruning_stats logging/batch_results/my_batch --batch

    # CSV output
    python -m beaver.utils.pruning_stats logging/logs_20260316185322 -o stats.csv
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from beaver.logging import get_log_data


def compute_instance_pruning_stats(entry_list: list[dict]) -> dict:
    """Compute pruning statistics for a single instance from its transition log entries."""
    transition_entries = [e for e in entry_list if "transition" in e]
    if not transition_entries:
        return {}

    n_transitions = len(transition_entries)
    final = transition_entries[-1]

    # Final total pruned mass (cumulative)
    final_pruned = final.get("pruned prob sum", 0.0)

    # Per-step pruned mass (diff of cumulative pruned prob sum)
    pruned_cumulative = [e.get("pruned prob sum", 0.0) for e in transition_entries]
    per_step_pruned = [pruned_cumulative[0]] + [
        pruned_cumulative[i] - pruned_cumulative[i - 1]
        for i in range(1, len(pruned_cumulative))
    ]

    return {
        "num_transitions": n_transitions,
        "total_pruned_mass": final_pruned,
        "avg_per_step_pruned_mass": float(np.mean(per_step_pruned)),
        "max_per_step_pruned_mass": float(max(per_step_pruned)),
    }


def compute_aggregate_stats(per_instance: dict[str, dict]) -> dict:
    """Aggregate per-instance pruning stats into dataset-level summary."""
    if not per_instance:
        return {}

    stats_list = [s for s in per_instance.values() if s]
    n = len(stats_list)
    if n == 0:
        return {}

    def avg(key):
        return float(np.mean([s[key] for s in stats_list]))

    def med(key):
        return float(np.median([s[key] for s in stats_list]))

    def mx(key):
        return float(max(s[key] for s in stats_list))

    def mn(key):
        return float(min(s[key] for s in stats_list))

    def std(key):
        return float(np.std([s[key] for s in stats_list]))

    return {
        "num_instances": n,
        # Total pruned mass
        "avg_total_pruned_mass": avg("total_pruned_mass"),
        "std_total_pruned_mass": std("total_pruned_mass"),
        "median_total_pruned_mass": med("total_pruned_mass"),
        "min_total_pruned_mass": mn("total_pruned_mass"),
        "max_total_pruned_mass": mx("total_pruned_mass"),
        # Per-step pruned mass
        "avg_per_step_pruned_mass": avg("avg_per_step_pruned_mass"),
        "max_per_step_pruned_mass": mx("max_per_step_pruned_mass"),
    }


def analyze_run(log_folder: Path) -> tuple[dict[str, dict], dict]:
    """Analyze a single run log folder. Returns (per_instance_stats, aggregate_stats)."""
    all_data = get_log_data(log_folder)
    if not all_data:
        return {}, {}

    per_instance = {}
    for instance_id, entries in all_data.items():
        per_instance[instance_id] = compute_instance_pruning_stats(entries)

    aggregate = compute_aggregate_stats(per_instance)
    return per_instance, aggregate


def print_run_report(log_folder: Path, per_instance: dict, aggregate: dict):
    """Print a formatted pruning statistics report for a single run."""
    if not aggregate:
        print(f"No data found in {log_folder}")
        return

    print(f"\n{'=' * 70}")
    print(f"  PRUNING STATISTICS: {log_folder.name}")
    print(f"{'=' * 70}")
    print(f"  Instances: {aggregate['num_instances']}")
    print()

    print("  Total Pruned Mass:")
    print(f"    avg:    {aggregate['avg_total_pruned_mass']:.6f}  (std: {aggregate['std_total_pruned_mass']:.6f})")
    print(f"    median: {aggregate['median_total_pruned_mass']:.6f}")
    print(f"    range:  [{aggregate['min_total_pruned_mass']:.6f}, {aggregate['max_total_pruned_mass']:.6f}]")
    print()

    print("  Per-Step Pruned Mass:")
    print(f"    avg:   {aggregate['avg_per_step_pruned_mass']:.6f}")
    print(f"    max:   {aggregate['max_per_step_pruned_mass']:.6f}")
    print(f"{'=' * 70}\n")


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
            # Find latest logs_* dir
            log_dirs = sorted(
                [d for d in exp_dir.iterdir() if d.is_dir() and d.name.startswith("logs_")],
                reverse=True,
            )
            if log_dirs:
                results.append((model_name, exp_dir.name, log_dirs[0]))
    return results


def save_csv(per_instance: dict[str, dict], output_path: Path):
    """Save per-instance pruning stats to CSV."""
    if not per_instance:
        return
    first = next(v for v in per_instance.values() if v)
    fieldnames = ["instance_id"] + list(first.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for inst_id, stats in sorted(per_instance.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
            if stats:
                row = {"instance_id": inst_id, **stats}
                writer.writerow(row)
    print(f"Per-instance stats saved to: {output_path}")


def save_batch_csv(all_aggregates: list[tuple[str, str, dict]], output_path: Path):
    """Save batch aggregate stats to CSV."""
    if not all_aggregates:
        return
    first_agg = next(a for _, _, a in all_aggregates if a)
    fieldnames = ["model", "experiment"] + list(first_agg.keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model, exp, agg in all_aggregates:
            if agg:
                writer.writerow({"model": model, "experiment": exp, **agg})
    print(f"Batch pruning stats saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute pruning statistics from BEAVER run logs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m beaver.utils.pruning_stats logging/logs_20260316185322\n"
            "  python -m beaver.utils.pruning_stats logging/batch_results/my_batch --batch\n"
            "  python -m beaver.utils.pruning_stats logging/logs_20260316185322 -o stats.csv\n"
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

        all_aggregates = []
        for model, exp, log_dir in entries:
            print(f"\nAnalyzing: {model} / {exp}")
            per_instance, aggregate = analyze_run(log_dir)
            print_run_report(log_dir, per_instance, aggregate)
            all_aggregates.append((model, exp, aggregate))

        output_path = Path(args.output) if args.output else path / "pruning_stats.csv"
        save_batch_csv(all_aggregates, output_path)
    else:
        per_instance, aggregate = analyze_run(path)
        print_run_report(path, per_instance, aggregate)

        if args.output:
            save_csv(per_instance, Path(args.output))
        else:
            save_csv(per_instance, path / "pruning_stats.csv")


if __name__ == "__main__":
    main()
