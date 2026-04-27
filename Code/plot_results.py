"""
Visualization script for ride-matching experiments.

This script:
- Loads experiment results from CSV
- Aggregates metrics by (problem size, algorithm)
- Generates comparison plots for:
    - Runtime
    - Total score
    - Match count
    - Solution quality vs Hungarian
    - Stability violations

Purpose:
To visually compare algorithm performance and highlight tradeoffs
as problem size scales.
"""

import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT_CSV = os.path.join(os.getcwd(), "..", "Results", "rideshare_results.csv")
GRAPH_DIR = os.path.join(os.getcwd(), "..", "Graphs")
os.makedirs(GRAPH_DIR, exist_ok=True)


def try_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_rows(filename):
    rows = []
    with open(filename, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def aggregate_by_size_and_algorithm(rows):
    grouped = defaultdict(list)
    for row in rows:
        size = int(float(row["size"]))
        algorithm = row["algorithm"]
        grouped[(size, algorithm)].append(row)

    summary = {}
    for (size, algorithm), group in grouped.items():
        summary[(size, algorithm)] = {
            "avg_runtime_seconds": sum(try_float(r["runtime_seconds"]) for r in group) / len(group),
            "avg_total_score": sum(try_float(r["total_score"]) for r in group) / len(group),
            "avg_num_matches": sum(try_float(r["num_matches"]) for r in group) / len(group),
            "avg_solution_quality_vs_hungarian": sum(
                try_float(r["solution_quality_vs_hungarian"]) for r in group
            ) / len(group),
            "avg_stability_violations": sum(
                try_float(r.get("stability_violations", ""), 0.0) for r in group
            ) / len(group),
        }
    return summary


def algorithms_and_sizes(summary):
    algorithms = sorted({algorithm for (_, algorithm) in summary.keys()})
    sizes = sorted({size for (size, _) in summary.keys()})
    return algorithms, sizes


def print_summary_table(summary):
    algorithms, sizes = algorithms_and_sizes(summary)
    print("\\n=== Aggregated Data ===")
    print(
        "size,algorithm,avg_runtime_seconds,avg_total_score,"
        "avg_num_matches,avg_solution_quality_vs_hungarian,avg_stability_violations"
    )
    for size in sizes:
        for algorithm in algorithms:
            if (size, algorithm) in summary:
                row = summary[(size, algorithm)]
                print(
                    f"{size},{algorithm},"
                    f"{row['avg_runtime_seconds']:.6f},"
                    f"{row['avg_total_score']:.2f},"
                    f"{row['avg_num_matches']:.2f},"
                    f"{row['avg_solution_quality_vs_hungarian']:.4f},"
                    f"{row['avg_stability_violations']:.2f}"
                )


def annotate_overlaps(ax, points):
    overlap_groups = defaultdict(list)
    for algorithm, x, y in points:
        overlap_groups[(x, round(y, 6))].append(algorithm)

    for (x, y), algorithms in overlap_groups.items():
        if len(algorithms) > 1:
            label = "overlap: " + ", ".join(sorted(algorithms))
        else:
            label = algorithms[0]
        ax.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
        )


def make_line_plot(summary, metric_key, ylabel, title, output_name):
    algorithms, sizes = algorithms_and_sizes(summary)
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    all_points = []

    for algorithm in algorithms:
        x = []
        y = []
        for size in sizes:
            if (size, algorithm) in summary:
                x.append(size)
                y.append(summary[(size, algorithm)][metric_key])
                all_points.append((algorithm, size, summary[(size, algorithm)][metric_key]))
        if x:
            ax.plot(x, y, marker="o", linewidth=2, alpha=0.85, label=algorithm)
            ax.scatter(x, y, s=70)

    annotate_overlaps(ax, all_points)

    ax.set_xlabel("Problem Size (Drivers = Riders)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(GRAPH_DIR, output_name), dpi=200)
    plt.close()


def main():
    rows = load_rows(INPUT_CSV)
    if not rows:
        print("No rows found in rideshare_results.csv")
        return

    print(f"Loaded {len(rows)} rows")
    print(f"Graphs saved to: {GRAPH_DIR}")
    summary = aggregate_by_size_and_algorithm(rows)
    print_summary_table(summary)

    make_line_plot(
        summary,
        "avg_runtime_seconds",
        "Average Runtime (seconds)",
        "Average Runtime vs Problem Size",
        "runtime_vs_size.png",
    )

    make_line_plot(
        summary,
        "avg_total_score",
        "Average Total Score",
        "Average Total Score vs Problem Size",
        "score_vs_size.png",
    )

    make_line_plot(
        summary,
        "avg_num_matches",
        "Average Number of Matches",
        "Average Number of Matches vs Problem Size",
        "matches_vs_size.png",
    )

    make_line_plot(
        summary,
        "avg_solution_quality_vs_hungarian",
        "Average Solution Quality / Hungarian",
        "Solution Quality vs Problem Size",
        "quality_vs_size.png",
    )

    make_line_plot(
        summary,
        "avg_stability_violations",
        "Average Stability Violations",
        "Stability Violations vs Problem Size",
        "stability_vs_size.png",
    )

    print("\\nCreated:")
    print("  runtime_vs_size.png")
    print("  score_vs_size.png")
    print("  matches_vs_size.png")
    print("  quality_vs_size.png")
    print("  stability_vs_size.png")


if __name__ == "__main__":
    main()