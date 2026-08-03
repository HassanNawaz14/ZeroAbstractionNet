"""Render the README banner (plan.md Build 2 item 3).

1920x640 dark banner: title + tagline on the left, a mini budget-reel motif
(python/c/asm lanes with n-chips) on the right. Reuses the palette from
animate_budget so the whole presentation phase shares one brand.

Usage:
    python present/banner.py [--out present/banner.png]
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from animate_budget import PALETTE, _fit_fontsize

CANVAS = (1920, 640)
RIGHT_MOTIF = (1100, 180, 1830, 560)   # lane motif box (verify target)
LANES = {
    "python": (PALETTE["LANE_PY"], 420, 470, 0.002, "n=256"),
    "c": (PALETTE["LANE_C"], 330, 380, 0.125, "n=1024"),
    "asm": (PALETTE["LANE_ASM"], 240, 290, 1.0, "n=2048"),
}


def draw_banner(ax):
    ax.clear()
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), CANVAS[0], CANVAS[1],
                               facecolor=PALETTE["BG"], zorder=0))
    ax.add_patch(plt.Rectangle((0, CANVAS[1] - 10), CANVAS[0], 10,
                               facecolor=PALETTE["ACCENT"], zorder=1))
    ax.add_patch(plt.Rectangle((0, 0), CANVAS[0], 8,
                               facecolor=PALETTE["DIM"], zorder=1))

    title = "ZEROABSTRACTIONNET"
    ax.text(90, 500, title,
            fontsize=_fit_fontsize(ax, title, 96, 1000, "bold"),
            fontweight="bold", color=PALETTE["TEXT"], ha="left", va="center", zorder=3)
    tagline = "a from-scratch neural network whose matmul backends go from "
    tagline2 = "pure Python to hand-written x86-64 AVX2/FMA assembly — measured, not claimed."
    ax.text(90, 415, tagline + tagline2,
            fontsize=_fit_fontsize(ax, tagline + tagline2, 28, 1000),
            color=PALETTE["DIM"], ha="left", va="center", zorder=3)
    ax.text(90, 365, "zero third-party compute libraries · C and asm built from the source in this repo",
            fontsize=22, color=PALETTE["ACCENT"], ha="left", va="center", zorder=3)

    chips = ["94 tests", "best-of-3 timings", "single core · WSL2 · Skylake-class"]
    x = 90
    for chip in chips:
        w = 18 + _fit_fontsize(ax, chip, 20, 400) * 0.62 * len(chip)
        ax.add_patch(plt.Rectangle((x, 275), w, 46, facecolor=PALETTE["PANEL"],
                                   edgecolor=PALETTE["DIM"], linewidth=1.5, zorder=2))
        ax.text(x + w / 2, 298, chip, fontsize=20, color=PALETTE["TEXT"],
                ha="center", va="center", zorder=3)
        x += w + 16

    ax.text(1465, 560, "same 5-second budget · one n×n matmul each",
            fontsize=22, color=PALETTE["DIM"], ha="center", va="center", zorder=3)
    x0, x1 = 1100, 1830
    for name, (colour, y0, y1, frac, chip) in LANES.items():
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                   facecolor=PALETTE["PANEL"],
                                   edgecolor=PALETTE["DIM"], linewidth=1.5, zorder=2))
        w = (x1 - x0) * frac
        ax.add_patch(plt.Rectangle((x0, y0), max(w, 3), y1 - y0,
                                   facecolor=colour, zorder=3))
        ax.text(x0 + w + 14, (y0 + y1) / 2, chip, fontsize=24, fontweight="bold",
                color=colour, ha="left", va="center", zorder=4)
    ax.plot([x1, x1], [225, 480], color=PALETTE["ACCENT"], lw=3, zorder=5)
    ax.text(x1 - 8, 480, "5 s", fontsize=18, fontweight="bold",
            color=PALETTE["ACCENT"], ha="right", va="bottom", zorder=5)
    ax.text(1465, 205, "8× the matrix size · 512× the math — same wall clock",
            fontsize=24, fontweight="bold", color=PALETTE["ACCENT"],
            ha="center", va="center", zorder=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("present", "banner.png"))
    args = parser.parse_args()
    fig, ax = plt.subplots(figsize=(19.2, 6.4), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    draw_banner(ax)
    fig.savefig(args.out, dpi=100)
    plt.close(fig)
    print(f"banner -> {args.out}")


if __name__ == "__main__":
    main()
