from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator


SOURCE_DIR = Path(r"C:\Users\Administrator\Desktop\fsdownload")
OUTPUT_DIR = Path(__file__).resolve().parent

RUNS_100 = [
    (
        "Sym lr=0.0035",
        "Cifar100_FedCLIP_test_0_20260812_035301_1786457215020947365_2614258_0.h5",
        "#2563A6",
        "-",
        2.1,
    ),
    (
        "Sym lr=0.005",
        "Cifar100_FedCLIP_test_0_20260812_034607_1786457214953620974_2614235_0.h5",
        "#2E8B57",
        "-",
        2.1,
    ),
    (
        "Slow-U (U=0.0015, V=0.005)",
        "Cifar100_FedCLIP_test_0_20260812_035526_1786457215101646771_2614242_0.h5",
        "#D62728",
        "-",
        3.0,
    ),
    (
        "Slow-V (U=0.005, V=0.0015)",
        "Cifar100_FedCLIP_test_0_20260812_035510_1786457214920046101_2614249_0.h5",
        "#7B4AB5",
        "--",
        2.1,
    ),
]

RUNS_200 = [
    (
        "Sym lr=0.002",
        "Cifar100_FedCLIP_test_0_20260813_201131_1786602748156771525_627414_0.h5",
        "#168AAD",
        "-",
        2.1,
    ),
    (
        "Sym lr=0.0035",
        "Cifar100_FedCLIP_test_0_20260812_200728_1786509804908799693_232529_0.h5",
        "#2563A6",
        "-",
        2.1,
    ),
    (
        "Sym lr=0.005",
        "Cifar100_FedCLIP_test_0_20260812_200540_1786509805021035849_232508_0.h5",
        "#2E8B57",
        "-",
        2.1,
    ),
    (
        "Slow-U (U=0.0015, V=0.005)",
        "Cifar100_FedCLIP_test_0_20260812_200823_1786509805021762886_232515_0.h5",
        "#D62728",
        "-",
        3.0,
    ),
    (
        "Slow-V (U=0.005, V=0.0015)",
        "Cifar100_FedCLIP_test_0_20260812_201050_1786509804961469885_232522_0.h5",
        "#7B4AB5",
        "--",
        2.1,
    ),
]


def read_test_accuracy(filename, expected_rounds):
    path = SOURCE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as h5_file:
        if "rs_test_acc" not in h5_file:
            raise KeyError(f"{path}: rs_test_acc not found")
        accuracy = np.asarray(h5_file["rs_test_acc"][:], dtype=float).reshape(-1)

    if len(accuracy) != expected_rounds:
        raise ValueError(
            f"{path}: expected {expected_rounds} rounds, got {len(accuracy)}"
        )
    return accuracy


def draw_comparison(runs, rounds, output_stem):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

    for label, filename, color, linestyle, linewidth in runs:
        accuracy = read_test_accuracy(filename, rounds)
        x_values = np.arange(1, rounds + 1)
        best_index = int(np.argmax(accuracy))
        legend_label = f"{label} (best={accuracy[best_index]:.4f})"
        is_slow_u = label.startswith("Slow-U")

        ax.plot(
            x_values,
            accuracy,
            label=legend_label,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=0.96,
            zorder=5 if is_slow_u else 2,
        )
        if is_slow_u:
            ax.scatter(
                x_values[best_index],
                accuracy[best_index],
                s=48,
                color=color,
                edgecolor="white",
                linewidth=0.9,
                zorder=6,
            )

    ax.set_title(
        f"CIFAR-100 ({rounds} Communication Rounds)",
        fontweight="semibold",
        pad=12,
    )
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy")
    ax.set_xlim(1, rounds)
    ax.set_ylim(0.0, 0.62)
    ax.xaxis.set_major_locator(MultipleLocator(10 if rounds == 100 else 20))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.grid(
        True,
        which="major",
        color="#D8D8D8",
        linewidth=0.75,
        linestyle="--",
        alpha=0.75,
    )
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(
        loc="lower right",
        frameon=True,
        framealpha=0.95,
        edgecolor="#C8C8C8",
    )

    png_path = OUTPUT_DIR / f"{output_stem}.png"
    pdf_path = OUTPUT_DIR / f"{output_stem}.pdf"
    fig.savefig(png_path, dpi=320, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def main():
    draw_comparison(RUNS_100, 100, "cifar100_100_rounds_lr_comparison")
    draw_comparison(RUNS_200, 200, "cifar100_200_rounds_lr_comparison")


if __name__ == "__main__":
    main()
