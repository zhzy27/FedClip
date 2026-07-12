#!/usr/bin/env python3

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


LOCAL_METRIC = "total_forward_gflops_per_epoch"
SERVER_METRIC = "server_total_seconds"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot local forward FLOPs and server time for Ours plus the two "
            "lowest-cost and two highest-cost baselines."
        )
    )
    parser.add_argument(
        "--input-csv",
        nargs="+",
        default=["compute_results/cifar10_pat2_compute.csv"],
        help="One or more compute-result CSV files.",
    )
    parser.add_argument(
        "--our-algorithm",
        default="FedCLIP",
        help="Algorithm name used to identify Ours in the CSV.",
    )
    parser.add_argument(
        "--our-label",
        default="Ours",
        help="Display label for the proposed method.",
    )
    parser.add_argument(
        "--aggregation",
        choices=["mean", "median"],
        default="mean",
        help="How repeated round records are summarized for each algorithm.",
    )
    parser.add_argument(
        "--output-dir",
        default="figures/compute_cost",
        help="Directory for PNG, PDF, and selected-value CSV outputs.",
    )
    parser.add_argument(
        "--output-prefix",
        default="cifar10_pat2",
        help="Prefix used for output filenames.",
    )
    return parser.parse_args()


def read_records(csv_paths):
    records = []
    required = {"algorithm", "status", LOCAL_METRIC, SERVER_METRIC}
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.is_file():
            raise FileNotFoundError(f"Compute CSV does not exist: {path}")
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            reader = csv.DictReader(file)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"{path} is missing required columns: {sorted(missing)}"
                )
            for row in reader:
                if row.get("status", "").strip().lower() != "ok":
                    continue
                row["_source_csv"] = str(path)
                records.append(row)
    if not records:
        raise RuntimeError("No successful compute records were found.")
    return records


def finite_float(value, field, algorithm):
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Algorithm {algorithm} has an invalid {field} value: {value!r}"
        ) from error
    if not math.isfinite(result):
        raise ValueError(
            f"Algorithm {algorithm} has a non-finite {field} value: {value!r}"
        )
    return result


def summarize(records, aggregation):
    grouped = defaultdict(lambda: {LOCAL_METRIC: [], SERVER_METRIC: []})
    for row in records:
        algorithm = row["algorithm"].strip()
        if not algorithm:
            continue
        grouped[algorithm][LOCAL_METRIC].append(
            finite_float(row[LOCAL_METRIC], LOCAL_METRIC, algorithm)
        )
        grouped[algorithm][SERVER_METRIC].append(
            finite_float(row[SERVER_METRIC], SERVER_METRIC, algorithm)
        )

    reducer = statistics.mean if aggregation == "mean" else statistics.median
    summary = {}
    for algorithm, metrics in grouped.items():
        summary[algorithm] = {
            LOCAL_METRIC: reducer(metrics[LOCAL_METRIC]),
            SERVER_METRIC: reducer(metrics[SERVER_METRIC]),
            "records": len(metrics[LOCAL_METRIC]),
        }
    return summary


def select_methods(summary, metric, our_algorithm):
    if our_algorithm not in summary:
        raise KeyError(
            f"Ours ({our_algorithm}) was not found. Available algorithms: "
            f"{sorted(summary)}"
        )

    baselines = [
        (algorithm, values[metric])
        for algorithm, values in summary.items()
        if algorithm != our_algorithm
    ]
    if len(baselines) < 4:
        raise RuntimeError(
            f"At least four baselines are required, but only {len(baselines)} were found."
        )

    ordered = sorted(baselines, key=lambda item: (item[1], item[0]))
    lowest = ordered[:2]
    highest = ordered[-2:]
    return [
        (lowest[0][0], lowest[0][1], "Lowest baseline"),
        (lowest[1][0], lowest[1][1], "Lowest baseline"),
        (our_algorithm, summary[our_algorithm][metric], "Ours"),
        (highest[0][0], highest[0][1], "Highest baseline"),
        (highest[1][0], highest[1][1], "Highest baseline"),
    ]


def value_label(value, metric):
    if metric == LOCAL_METRIC:
        return f"{value:,.1f}"
    if value < 1:
        return f"{value:.3f}"
    return f"{value:.2f}"


def draw_bar_chart(selection, metric, our_algorithm, our_label, title, ylabel, output_stem):
    role_colors = {
        "Lowest baseline": "#4C78A8",
        "Ours": "#C44E52",
        "Highest baseline": "#E39C37",
    }
    labels = [our_label if algorithm == our_algorithm else algorithm for algorithm, _, _ in selection]
    values = [value for _, value, _ in selection]
    colors = [role_colors[role] for _, _, role in selection]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bars = ax.bar(
        labels,
        values,
        color=colors,
        width=0.68,
        edgecolor="#333333",
        linewidth=0.7,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    maximum = max(values)
    ax.set_ylim(0, maximum * 1.18 if maximum > 0 else 1)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + maximum * 0.025,
            value_label(value, metric),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    legend_handles = [
        Patch(facecolor=role_colors[role], edgecolor="#333333", label=role)
        for role in ("Lowest baseline", "Ours", "Highest baseline")
    ]
    ax.legend(handles=legend_handles, frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def write_selection_csv(path, selections, summary, aggregation):
    fieldnames = [
        "chart",
        "role",
        "algorithm",
        "metric",
        "value",
        "unit",
        "aggregation",
        "record_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for chart, metric, selection in selections:
            unit = "GFLOPs" if metric == LOCAL_METRIC else "seconds"
            for algorithm, value, role in selection:
                writer.writerow({
                    "chart": chart,
                    "role": role,
                    "algorithm": algorithm,
                    "metric": metric,
                    "value": f"{value:.12g}",
                    "unit": unit,
                    "aggregation": aggregation,
                    "record_count": summary[algorithm]["records"],
                })


def print_selection(title, selection):
    print(title)
    for algorithm, value, role in selection:
        print(f"  {role:16s} {algorithm:12s} {value:.6f}")


def main():
    args = parse_args()
    records = read_records(args.input_csv)
    summary = summarize(records, args.aggregation)
    local_selection = select_methods(summary, LOCAL_METRIC, args.our_algorithm)
    server_selection = select_methods(summary, SERVER_METRIC, args.our_algorithm)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_stem = output_dir / f"{args.output_prefix}_local_forward_flops"
    server_stem = output_dir / f"{args.output_prefix}_server_time"

    draw_bar_chart(
        local_selection,
        LOCAL_METRIC,
        args.our_algorithm,
        args.our_label,
        "Local Computation Cost",
        "Forward FLOPs per Local Epoch (GFLOPs)",
        local_stem,
    )
    draw_bar_chart(
        server_selection,
        SERVER_METRIC,
        args.our_algorithm,
        args.our_label,
        "Server Computation Cost",
        "Server Time per Round (s)",
        server_stem,
    )

    selection_csv = output_dir / f"{args.output_prefix}_selected_methods.csv"
    write_selection_csv(
        selection_csv,
        [
            ("local_forward_flops", LOCAL_METRIC, local_selection),
            ("server_time", SERVER_METRIC, server_selection),
        ],
        summary,
        args.aggregation,
    )

    print_selection("Local forward FLOPs selection:", local_selection)
    print_selection("Server time selection:", server_selection)
    print(f"Saved: {local_stem.with_suffix('.png')}")
    print(f"Saved: {local_stem.with_suffix('.pdf')}")
    print(f"Saved: {server_stem.with_suffix('.png')}")
    print(f"Saved: {server_stem.with_suffix('.pdf')}")
    print(f"Saved: {selection_csv}")


if __name__ == "__main__":
    main()
