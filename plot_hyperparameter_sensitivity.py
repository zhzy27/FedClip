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
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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
COLORS = ["#fa3825", "#3b9cbe", "#54be45", "#b46aaf", "#f4a113", "#39aa9c"]


def gradient_bars(ax, values, baseline):
    # All faces share one collection so depth sorting works across all bars.
    faces, colors = [], []
    bottom = np.array(to_rgb("#ffff99"))
    width, depth = 0.43, 0.62
    for row, col in np.ndindex(values.shape):
        top = np.array(to_rgb(COLORS[row % len(COLORS)]))
        x0, x1 = col - width / 2, col + width / 2
        y0, y1 = row - depth / 2, row + depth / 2
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        levels = np.linspace(baseline, values[row, col], 81)
        for index, (z0, z1) in enumerate(zip(levels[:-1], levels[1:])):
            amount = (index + 0.5) / (len(levels) - 1)
            color = bottom * (1 - amount) + top * amount
            for side, shade in enumerate((1.0, 0.87, 0.82, 0.94)):
                a, b = corners[side], corners[(side + 1) % 4]
                faces.append([(a[0], a[1], z0), (b[0], b[1], z0),
                              (b[0], b[1], z1), (a[0], a[1], z1)])
                colors.append(color * shade)
        faces.append([(x, y, values[row, col]) for x, y in corners])
        colors.append(np.minimum(top * 0.92 + 0.04, 1))
    ax.add_collection3d(Poly3DCollection(
        faces, facecolors=colors, edgecolors="none", linewidths=0,
        antialiased=False, zsort="average", rasterized=True,
    ))


def plot(args):
    if ACCURACY.shape != (len(LAMBDA_A), len(RHO)):
        raise ValueError("Accuracy rows/columns must match LAMBDA_A/RHO.")
    if not np.isfinite(ACCURACY).all():
        raise ValueError("Accuracy contains NaN or Inf.")
    if args.z_min < 0 or args.z_min >= ACCURACY.min():
        raise ValueError("--z-min must be nonnegative and below the smallest accuracy.")

    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })
    fig = plt.figure(figsize=(8.2, 6.0), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    # The reference shows descending values on its front parameter axis.
    display_rho = RHO[::-1]
    display_accuracy = ACCURACY[:, ::-1]
    gradient_bars(ax, display_accuracy, args.z_min)
    if args.show_values:
        for row, col in np.ndindex(display_accuracy.shape):
            ax.text(col, row, display_accuracy[row, col] + 0.16,
                    f"{display_accuracy[row, col]:.2f}", ha="center", fontsize=7)

    best_row, best_col = np.unravel_index(np.argmax(ACCURACY), ACCURACY.shape)
    best = ACCURACY[best_row, best_col]
    if args.title:
        ax.set_title(args.title, fontsize=16, pad=8)

    ax.set_xticks(range(len(RHO)), [f"{value:g}" for value in display_rho])
    ax.set_yticks(range(len(LAMBDA_A)), [f"{value:g}" for value in LAMBDA_A])
    ax.set_xlabel(r"$\rho$", fontsize=24, labelpad=10)
    ax.yaxis.set_rotate_label(False)
    ax.set_ylabel(r"$\lambda_a$", fontsize=24, labelpad=10, rotation=0)
    ax.set_zlabel("")
    ax.text2D(-0.065, 0.5, "Accuracy (%)", transform=ax.transAxes,
              rotation=90, va="center", ha="center", fontsize=17)
    ax.set_xlim(-0.6, len(RHO) - 0.4)
    ax.set_ylim(-0.6, len(LAMBDA_A) - 0.4)
    ax.set_zlim(args.z_min, 2 * np.ceil(best / 2))
    ax.zaxis.set_major_locator(MultipleLocator(2 if args.z_min >= 40 else 10))
    ax.view_init(elev=args.elev, azim=args.azim)
    ax.set_proj_type("ortho")
    ax.set_box_aspect((1.5, 1.0, 0.92), zoom=0.99)
    # Matplotlib has no public control for putting only the vertical axis left.
    ax.zaxis._axinfo["juggled"] = (1, 2, 0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 1))
        axis.pane.set_edgecolor("#333333")
        axis.pane.set_linewidth(0.8)
        axis._axinfo["grid"].update(color="#d5d5d5", linewidth=0.7)
    ax.tick_params(labelsize=13, pad=2)
    fig.subplots_adjust(left=0.10, right=0.91, bottom=0.07, top=0.98)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = args.output_dir / f"hyperparameter_sensitivity_3d.{suffix}"
        fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.12, facecolor="white")
        print(f"Saved: {path.resolve()}")
    plt.close(fig)
    print(f"Best accuracy: {best:.2f}%; lambda_a={LAMBDA_A[best_row]}, rho={RHO[best_col]}")
    print(f"Bar baseline: {args.z_min:g}%; horizontal axes use equally spaced levels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--title", default="")
    parser.add_argument("--z-min", type=float, default=46.0,
                        help="Visible bar baseline in percent; use 0 for a zero baseline.")
    parser.add_argument("--elev", type=float, default=18, help="View elevation in degrees.")
    parser.add_argument("--azim", type=float, default=-76, help="View azimuth in degrees.")
    parser.add_argument("--show-values", action="store_true",
                        help="Label all 36 bars (may overlap in a 3D view).")
    plot(parser.parse_args())


if __name__ == "__main__":
    main()
