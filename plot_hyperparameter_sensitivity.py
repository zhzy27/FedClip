"""Plot the supplied lambda_a x rho accuracy table as a 3D bar chart.

Requires numpy and matplotlib. Works headlessly on a server.
Example: python plot_hyperparameter_sensitivity.py --output-dir figures
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator


# Columns: rho. Rows: lambda_a. Values are accuracy in percent, not fractions.
RHO = [0, 0.1, 0.3, 0.7, 1.0, 2.0]
LAMBDA_A = [0, 0.1, 0.3, 0.7, 1.0, 3.0]
ACCURACY = np.array([
    [47.74, 48.58, 48.58, 48.41, 48.77, 48.77],
    [53.82, 54.34, 54.16, 53.62, 53.66, 54.36],
    [55.00, 56.01, 55.85, 54.39, 54.74, 54.86],
    [55.57, 55.93, 56.39, 55.17, 55.67, 55.35],
    [54.17, 56.59, 57.23, 55.64, 55.39, 55.57],
    [51.47, 52.99, 53.95, 53.02, 52.74, 52.93],
], dtype=float)
COLORS = ["#e76055", "#43a3c2", "#73b65b", "#a276b5", "#e8b13e", "#619d93"]


def plot(args):
    if ACCURACY.shape != (len(LAMBDA_A), len(RHO)):
        raise ValueError("Accuracy rows/columns must match LAMBDA_A/RHO.")
    if not np.isfinite(ACCURACY).all():
        raise ValueError("Accuracy contains NaN or Inf.")
    if args.z_min < 0 or args.z_min >= ACCURACY.min():
        raise ValueError("--z-min must be nonnegative and below the smallest accuracy.")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(11, 8.7), facecolor="white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    x, y = np.meshgrid(np.arange(len(RHO)), np.arange(len(LAMBDA_A)))
    width = 0.56
    ax.bar3d(
        x.ravel() - width / 2, y.ravel() - width / 2,
        np.full(ACCURACY.size, args.z_min), width, width,
        ACCURACY.ravel() - args.z_min,
        color=[COLORS[row % len(COLORS)] for row in y.ravel()],
        shade=True, edgecolor=(0, 0, 0, 0.18), linewidth=0.4, zsort="average",
    )
    if args.show_values:
        for row, col in np.ndindex(ACCURACY.shape):
            ax.text(col, row, ACCURACY[row, col] + 0.16,
                    f"{ACCURACY[row, col]:.2f}", ha="center", fontsize=7)

    best_row, best_col = np.unravel_index(np.argmax(ACCURACY), ACCURACY.shape)
    best = ACCURACY[best_row, best_col]
    ax.scatter([best_col], [best_row], [best + 0.16], marker="*", s=150,
               color="#b71c1c", edgecolors="white", linewidths=0.7,
               depthshade=False, zorder=10)
    fig.suptitle(args.title, fontsize=18, fontweight="bold", y=0.975)
    fig.text(0.5, 0.928,
             rf"Best: {best:.2f}%  ($\lambda_a={LAMBDA_A[best_row]}$, $\rho={RHO[best_col]}$)",
             ha="center", fontsize=12, color="#b71c1c")

    ax.set_xticks(range(len(RHO)), [str(value) for value in RHO])
    ax.set_yticks(range(len(LAMBDA_A)), [str(value) for value in LAMBDA_A])
    ax.set_xlabel(r"$\rho$", fontsize=19, labelpad=12)
    ax.set_ylabel(r"$\lambda_a$", fontsize=19, labelpad=12)
    ax.set_zlabel("Accuracy (%)", fontsize=13, labelpad=10)
    ax.set_xlim(-0.6, len(RHO) - 0.4)
    ax.set_ylim(-0.6, len(LAMBDA_A) - 0.4)
    ax.set_zlim(args.z_min, np.ceil(best) + 1)
    ax.zaxis.set_major_locator(MultipleLocator(2 if args.z_min >= 40 else 10))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_box_aspect((1.25, 1.15, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0.98, 0.98, 0.98, 1.0))
    ax.tick_params(labelsize=10)
    fig.legend(handles=[Patch(facecolor=COLORS[i % len(COLORS)],
                              label=rf"$\lambda_a={value}$")
                        for i, value in enumerate(LAMBDA_A)],
               loc="lower center", bbox_to_anchor=(0.5, 0.05),
               ncol=6, frameon=False, fontsize=11)
    fig.text(0.5, 0.025,
             f"Equally spaced parameter levels; bar baseline = {args.z_min:g}%.",
             ha="center", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.12, top=0.89)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"hyperparameter_sensitivity_3d.{suffix}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {path.resolve()}")
    plt.close(fig)
    print(f"Best accuracy: {best:.2f}%; lambda_a={LAMBDA_A[best_row]}, rho={RHO[best_col]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--title", default="Joint Hyperparameter Sensitivity")
    parser.add_argument("--z-min", type=float, default=45.0,
                        help="Visible bar baseline in percent; use 0 for a zero baseline.")
    parser.add_argument("--elev", type=float, default=26, help="View elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-55, help="View azimuth in degrees.")
    parser.add_argument("--show-values", action="store_true",
                        help="Label all 36 bars (may overlap in a 3D view).")
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
