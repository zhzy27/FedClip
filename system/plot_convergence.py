import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def safe_path_component(value):
    text = str(value)
    text = re.sub(r"[^0-9A-Za-z._-]+", "_", text)
    return text.strip("_") or "default"


def float_path_component(value):
    return str(value).replace(".", "p")


def h5_datetime_from_name(path):
    match = re.search(r"(\d{8}_\d{6})", path.name)
    if not match:
        return datetime.min
    try:
        return datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
    except ValueError:
        return datetime.min


def partition_tag(args):
    if args.heterogeneity_tag:
        return args.heterogeneity_tag
    if args.partition == "dir":
        if args.dir_alpha is None:
            raise ValueError("--partition dir 需要指定 --dir-alpha，例如 --dir-alpha 0.3")
        return f"dir_alpha{float_path_component(args.dir_alpha)}"
    if args.partition == "pat":
        if args.cpc is None:
            raise ValueError("--partition pat 需要指定 --cpc，例如 --cpc 4")
        return f"pat_cpc{args.cpc}"
    if args.partition == "exdir":
        if args.dir_alpha is None:
            raise ValueError("--partition exdir 需要指定 --dir-alpha")
        return f"exdir_alpha{float_path_component(args.dir_alpha)}"
    return safe_path_component(args.partition)


def read_args_from_h5(path):
    try:
        with h5py.File(path, "r") as hf:
            args_json = hf.attrs.get("args_json", None)
        if args_json:
            return json.loads(args_json)
    except Exception:
        pass

    sidecar = path.with_suffix(path.suffix + ".json")
    if sidecar.exists():
        try:
            with sidecar.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data.get("args", {})
        except Exception:
            return {}
    return {}


def read_metric(path, metric):
    with h5py.File(path, "r") as hf:
        if metric not in hf:
            raise KeyError(f"{path} 中没有指标 {metric}")
        return np.asarray(hf[metric], dtype=float)


def moving_average(values, window):
    if window <= 1:
        return values
    out = values.copy()
    for col in range(values.shape[1]):
        series = values[:, col]
        valid = ~np.isnan(series)
        if valid.sum() == 0:
            continue
        dense = series[valid]
        kernel = np.ones(window, dtype=float) / window
        smoothed = np.convolve(dense, kernel, mode="same")
        out[valid, col] = smoothed
    return out


def stack_runs(curves):
    max_len = max(len(curve) for curve in curves)
    values = np.full((max_len, len(curves)), np.nan, dtype=float)
    for idx, curve in enumerate(curves):
        values[:len(curve), idx] = curve
    return values


def discover_model_dirs(algorithm_dir, model_family):
    if model_family:
        target = algorithm_dir / safe_path_component(model_family)
        return [target] if target.exists() else []
    return sorted(path for path in algorithm_dir.iterdir() if path.is_dir()) if algorithm_dir.exists() else []


def resolve_model_family_for_algorithm(args, alg_idx):
    if not args.model_families:
        return None
    if len(args.model_families) == 1:
        return args.model_families[0]
    if len(args.model_families) != len(args.algorithms):
        raise ValueError("--model-families 的数量必须是 1 个，或与 --algorithms 数量一致")
    return args.model_families[alg_idx]


def collect_series(args):
    root = Path(args.root)
    dataset_dir = root / safe_path_component(args.dataset)
    het_tag = partition_tag(args)
    checked_dirs = []
    series = []

    for alg_idx, algorithm in enumerate(args.algorithms):
        algorithm_dir = dataset_dir / safe_path_component(algorithm)
        model_family = resolve_model_family_for_algorithm(args, alg_idx)
        model_dirs = discover_model_dirs(algorithm_dir, model_family)
        if model_family and not model_dirs:
            checked_dirs.append(str(algorithm_dir / safe_path_component(model_family)))
            continue

        for model_dir in model_dirs:
            partition_dir = model_dir / het_tag
            data_dirs = []
            if args.num_classes is not None or args.niid is not None:
                ncl = args.num_classes if args.num_classes is not None else "*"
                niid = args.niid if args.niid is not None else "*"
                data_dirs = sorted(partition_dir.glob(f"ncl{ncl}_niid{niid}"))
            elif partition_dir.exists():
                data_dirs = sorted(path for path in partition_dir.iterdir() if path.is_dir())
            else:
                checked_dirs.append(str(partition_dir))
                continue

            for data_dir in data_dirs:
                if args.num_clients is not None or args.join_ratio is not None:
                    clients = args.num_clients if args.num_clients is not None else "*"
                    jr = float_path_component(args.join_ratio) if args.join_ratio is not None else "*"
                    join_dirs = sorted(data_dir.glob(f"clients{clients}_jr{jr}"))
                else:
                    join_dirs = sorted(path for path in data_dir.iterdir() if path.is_dir())

                for join_dir in join_dirs:
                    checked_dirs.append(str(join_dir))
                    h5_files = sorted(join_dir.glob("*.h5"), key=h5_datetime_from_name)
                    if not h5_files:
                        continue
                    curves = []
                    used_files = []
                    for h5_file in h5_files:
                        try:
                            curves.append(read_metric(h5_file, args.metric))
                            used_files.append(h5_file)
                        except KeyError:
                            continue
                    if not curves:
                        continue

                    label = algorithm
                    model_name = model_dir.name
                    if len(model_dirs) > 1 or model_family is None:
                        label = f"{algorithm}-{model_name}"

                    if args.labels:
                        if len(args.labels) != len(args.algorithms):
                            raise ValueError("--labels 的数量必须与 --algorithms 一致")
                        label = args.labels[alg_idx]
                        if len(model_dirs) > 1 or model_family is None:
                            label = f"{label}-{model_name}"

                    series.append({
                        "algorithm": algorithm,
                        "model": model_name,
                        "partition": het_tag,
                        "data_tag": data_dir.name,
                        "join_tag": join_dir.name,
                        "label": label,
                        "files": used_files,
                        "curves": curves,
                    })
    return series, checked_dirs


def format_metric_name(metric):
    return {
        "rs_test_acc": "Test Accuracy",
        "rs_test_auc": "Test AUC",
        "rs_train_loss": "Train Loss",
    }.get(metric, metric)


def plot_series(args, series):
    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, ax = plt.subplots(figsize=(args.width, args.height))
    metric_name = format_metric_name(args.metric)

    for item in series:
        curves = item["curves"]
        files = item["files"]
        if args.runs == "latest":
            curves = [curves[-1]]
            files = [files[-1]]

        values = stack_runs(curves)
        values = moving_average(values, args.smooth)
        if args.max_round is not None:
            values = values[:args.max_round + 1]

        if args.unit == "percent" and args.metric != "rs_train_loss":
            values = values * 100.0

        x = np.arange(values.shape[0]) * args.round_step
        if args.runs == "all":
            for run_idx in range(values.shape[1]):
                run_label = item["label"] if run_idx == 0 else None
                ax.plot(x, values[:, run_idx], linewidth=args.linewidth, alpha=0.45, label=run_label)
        else:
            mean = np.nanmean(values, axis=1)
            std = np.nanstd(values, axis=1)
            ax.plot(x, mean, linewidth=args.linewidth, label=item["label"])
            if args.show_std and values.shape[1] > 1:
                ax.fill_between(x, mean - std, mean + std, alpha=0.15)

        print(f"{item['label']}: {len(files)} 个 h5")
        for path in files:
            print(f"  - {path}")

    ax.set_xlabel("Communication Round")
    ylabel = metric_name
    if args.unit == "percent" and args.metric != "rs_train_loss":
        ylabel += " (%)"
    ax.set_ylabel(ylabel)

    title = args.title
    if not title:
        title = f"{args.dataset} {partition_tag(args)} {metric_name}"
    ax.set_title(title)
    ax.legend(loc=args.legend_loc)
    fig.tight_layout()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.output_name
    if not name:
        alg_part = "_".join(safe_path_component(alg) for alg in args.algorithms)
        name = f"convergence_{safe_path_component(args.dataset)}_{partition_tag(args)}_{safe_path_component(alg_part)}_{args.metric}"

    png_path = output_dir / f"{name}.png"
    pdf_path = output_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"已保存 PNG: {png_path}")
    print(f"已保存 PDF: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot convergence curves from structured h5_results.")
    parser.add_argument("--root", type=str, default="./h5_results", help="Structured h5 result root")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. Cifar10")
    parser.add_argument("--algorithms", type=str, nargs="+", required=True, help="One or more algorithms to plot")
    parser.add_argument("--labels", type=str, nargs="*", help="Display labels, same count as --algorithms")
    parser.add_argument("--model-families", type=str, nargs="*", help="Optional model family filter. One value for all algorithms, or one per algorithm")
    parser.add_argument("--partition", type=str, default="dir", choices=["dir", "pat", "exdir"], help="Partition type")
    parser.add_argument("--dir-alpha", type=str, help="Dirichlet alpha, e.g. 0.3")
    parser.add_argument("--cpc", type=str, help="Classes per client for pathological partition")
    parser.add_argument("--heterogeneity-tag", type=str, help="Direct partition folder tag, e.g. dir_alpha0p3 or pat_cpc4")
    parser.add_argument("--num-classes", type=str, help="Filter ncl folder")
    parser.add_argument("--niid", type=str, help="Filter niid folder")
    parser.add_argument("--num-clients", type=str, help="Filter clients folder")
    parser.add_argument("--join-ratio", type=str, help="Filter join ratio folder, e.g. 1.0 or 0.4")
    parser.add_argument("--metric", type=str, default="rs_test_acc", choices=["rs_test_acc", "rs_test_auc", "rs_train_loss"])
    parser.add_argument("--runs", type=str, default="mean", choices=["mean", "all", "latest"], help="How to draw repeated runs")
    parser.add_argument("--show-std", action="store_true", default=True, help="Show std band when --runs mean")
    parser.add_argument("--no-std", dest="show_std", action="store_false", help="Hide std band")
    parser.add_argument("--smooth", type=int, default=1, help="Moving average window")
    parser.add_argument("--max-round", type=int, help="Only plot up to this round index")
    parser.add_argument("--round-step", type=int, default=1, help="Round interval between saved points")
    parser.add_argument("--unit", type=str, default="percent", choices=["percent", "ratio"], help="Y-axis unit for accuracy/AUC")
    parser.add_argument("--output-dir", type=str, default="./figures/convergence")
    parser.add_argument("--output-name", type=str)
    parser.add_argument("--title", type=str)
    parser.add_argument("--legend-loc", type=str, default="best")
    parser.add_argument("--width", type=float, default=7.2)
    parser.add_argument("--height", type=float, default=5.0)
    parser.add_argument("--font-size", type=int, default=11)
    parser.add_argument("--linewidth", type=float, default=2.0)
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    series, checked_dirs = collect_series(args)
    if not series:
        checked_preview = "\n".join(f"  - {path}" for path in checked_dirs[:80])
        extra = "" if len(checked_dirs) <= 80 else f"\n  ... 还有 {len(checked_dirs) - 80} 个目录"
        raise FileNotFoundError(
            "没有找到符合条件的 h5 数据。\n"
            f"root={args.root}\n"
            f"dataset={args.dataset}\n"
            f"algorithms={args.algorithms}\n"
            f"heterogeneity={partition_tag(args)}\n"
            "已检查目录:\n"
            f"{checked_preview}{extra}"
        )

    plot_series(args, series)


if __name__ == "__main__":
    main()
