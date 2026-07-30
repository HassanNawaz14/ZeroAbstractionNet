"""Animate the XOR-quadrant dataset being generated, point by point.

Usage:
    python data/animate_data_generation.py --n-per-quadrant 25 --seed 0
    python data/animate_data_generation.py --save animations/dataset_generation.gif
"""

import argparse
import math
import sys
import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate_data import generate_dataset


def main():
    parser = argparse.ArgumentParser(description="Animate XOR-quadrant dataset generation")
    parser.add_argument("--n-per-quadrant", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--interval", type=int, default=50, help="ms per frame")
    parser.add_argument("--save", type=str, default=None, help="path to save .gif")
    args = parser.parse_args()

    X, y = generate_dataset(args.n_per_quadrant, args.seed, args.noise_std)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("XOR-quadrant dataset generation")

    ax.fill_between([0, 1.1], 0, 1.1, color="blue", alpha=0.04)
    ax.fill_between([-1.1, 0], 0, 1.1, color="red", alpha=0.04)
    ax.fill_between([-1.1, 0], -1.1, 0, color="blue", alpha=0.04)
    ax.fill_between([0, 1.1], -1.1, 0, color="red", alpha=0.04)

    ax.text(0.55, 0.55, "Q1  → 1", fontsize=9, color="blue", alpha=0.5)
    ax.text(-1.05, 0.55, "Q2  → 0", fontsize=9, color="red", alpha=0.5)
    ax.text(-1.05, -1.05, "Q3  → 1", fontsize=9, color="blue", alpha=0.5)
    ax.text(0.55, -1.05, "Q4  → 0", fontsize=9, color="red", alpha=0.5)

    scatter = ax.scatter(
        [p[0] for p in X], [p[1] for p in X],
        c=y, s=30, cmap="bwr", vmin=-0.5, vmax=1.5, alpha=0.0,
    )

    golden = [(0.5, 0.5, 1), (-0.5, -0.5, 1), (0.5, -0.5, 0), (-0.5, 0.5, 0)]
    ax.scatter(
        [p[0] for p in golden], [p[1] for p in golden],
        marker="*", s=200, c=[p[2] for p in golden],
        cmap="bwr", vmin=-0.5, vmax=1.5, edgecolors="black", linewidths=0.5,
        zorder=5, alpha=0.6,
    )

    legend_patches = [
        mpatches.Patch(color="blue", alpha=0.3, label="Label 1 (Q1, Q3)"),
        mpatches.Patch(color="red", alpha=0.3, label="Label 0 (Q2, Q4)"),
    ]
    legend = ax.legend(handles=legend_patches, loc="lower right", fontsize=8)

    info_text = ax.text(
        -1.05, 1.02, "", fontsize=9, family="monospace",
        verticalalignment="top",
    )

    n = len(X)
    alphas = [0.0] * n

    def update(frame):
        if frame > 0:
            alphas[frame - 1] = 1.0
        scatter.set_alpha(alphas)

        quadrant_idx = min((frame - 1) // args.n_per_quadrant + 1, 4) if frame > 0 else 0
        qcount = ((frame - 1) % args.n_per_quadrant + 1) if frame > 0 else 0
        quadrant_names = {0: "", 1: "Q1 (label 1)", 2: "Q2 (label 0)",
                          3: "Q3 (label 1)", 4: "Q4 (label 0)"}
        qname = quadrant_names.get(quadrant_idx, "Done")
        info_text.set_text(f"Point {frame}/{n}  |  {qname}: {qcount}/{args.n_per_quadrant}")
        return scatter, info_text

    ani = FuncAnimation(
        fig, update, frames=n + 1, interval=args.interval,
        blit=True, repeat=True,
    )

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        ani.save(args.save, writer="pillow", dpi=120)
        print(f"Saved animation to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
