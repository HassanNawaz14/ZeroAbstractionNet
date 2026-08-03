"""Render the "same 5-second budget" progress reel (plan.md Build 1).

Three lanes (python-naive / c-blocked / asm-blocked) fill with completed
math (flops) over a fixed 5 s wall-clock budget. A marker chip lands on each
lane when that backend completes its largest n x n matmul that fits in the
budget: python n=256 (2.64 s), c-blocked n=1024 (3.11 s), asm-blocked
n=2048 (4.96 s) — 8x the matrix size, 512x the math, same wall clock.

This module also owns the colour palette + layout geometry constants that
`verify_frames.py` imports for pixel verification, so the verification spec
can never drift from what the renderer actually draws.

Usage:
    python present/animate_budget.py [--out present/budget_reel.mp4]
    python present/animate_budget.py --still 5.0   # dump a single PNG frame
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation

# ---------------------------------------------------------------------------
# Palette (single source of truth — verify_frames.py imports this)
# ---------------------------------------------------------------------------
PALETTE = {
    "BG": "#0E1116",
    "PANEL": "#1B222C",
    "LANE_PY": "#F0563B",
    "LANE_C": "#3FA7FF",
    "LANE_ASM": "#3DDC84",
    "TEXT": "#F2F4F8",
    "DIM": "#8A94A6",
    "ACCENT": "#F5C451",
    "CAPTION_BG": "#151A22",
}

LANES = {
    "python-naive": ("python", "naive", PALETTE["LANE_PY"], "PYTHON", "naive"),
    "c-blocked": ("c", "blocked", PALETTE["LANE_C"], "C", "blocked"),
    "asm-blocked": ("asm", "blocked", PALETTE["LANE_ASM"], "ASM", "blocked"),
}

BUDGET_SEC = 5.0

# ---------------------------------------------------------------------------
# Layout (pixel coords on a 1080x1080 canvas, y up from the bottom) — shared
# with verify_frames.py.
# ---------------------------------------------------------------------------
CANVAS = (1080, 1080)
LANE_X0, LANE_X1 = 150, 1010          # full lane strip span
TRACK_X0, TRACK_W = 260, 750          # bar track inside the strip
BUDGET_X = 1010                       # vertical "5 s budget" line
LANE_STRIPS = {                       # y-bands per lane (verify targets)
    "python-naive": (150, 700, 1010, 800),
    "c-blocked": (150, 570, 1010, 670),
    "asm-blocked": (150, 440, 1010, 540),
}
CAPTION_BAND = (0, 0, 1080, 300)      # bottom caption strip (verify target)
BUDGET_LINE_BOX = (990, 360, 1080, 1050)
CHIP_W, CHIP_H = 140, 46

TRACKS = {                            # bar fill box inside each strip
    "python-naive": (260, 720, 1010, 780),
    "c-blocked": (260, 590, 1010, 650),
    "asm-blocked": (260, 460, 1010, 520),
}

# Caption text — every line is fitted to the canvas width (see _fit_fontsize),
# so nothing can bleed off the left/right edges.
CHART_TITLE = "WHAT FITS IN 5 SECONDS?"
CHART_SUB = "one n×n matmul per lane · best-of-3 · single core"
CHART_LEGEND = "bar = 2·n³ flops completed · chip lands when that matmul finishes"
TOP_RIGHT = "python · C-blocked · asm-blocked (AVX2/FMA)"

KICKER = "SAME WALL-CLOCK BUDGET — ONE MATMUL, THREE BACKENDS"
HEADLINE = "PYTHON n=256  →  ASM-BLOCKED n=2048"
TAGLINE = "8× THE MATRIX SIZE · 512× THE MATH"
SUB = "the clock ticks once for everyone — python crawls to n=256, AVX2/FMA asm lands n=2048."
PROVENANCE = "best-of-3 · single core · WSL2 Ubuntu · Skylake-class · no BLAS/numpy"

# ---------------------------------------------------------------------------
# text fitting: shrink fontsize until the rendered width fits the target
# ---------------------------------------------------------------------------
_FIT_CACHE = {}


def _fit_fontsize(ax, text, start, target_w, weight="normal"):
    """Return the largest fontsize (<= start) whose width fits target_w px."""
    key = (text, target_w, weight)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    renderer = ax.figure.canvas.get_renderer()
    fs = start
    while fs >= 8:
        probe = ax.text(0, 0, text, fontsize=fs, fontweight=weight)
        width = probe.get_window_extent(renderer=renderer).width
        probe.remove()
        if width <= target_w:
            _FIT_CACHE[key] = fs
            return fs
        fs -= 1
    _FIT_CACHE[key] = 8
    return 8


def load_lanes(csv_path):
    """Read benchmark_results.csv; return {lane: [(size, seconds), ...]} sorted."""
    with open(csv_path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for name, (backend, variant, *_) in LANES.items():
        sel = [
            (int(r["size"]), float(r["seconds"]))
            for r in rows
            if r["backend"] == backend and r["variant"] == variant
        ]
        sel.sort()
        out[name] = sel
    return out


def targets(lane_rows):
    """Largest n fitting the 5 s budget per lane: {lane: (n, seconds)}."""
    out = {}
    for name, rows in lane_rows.items():
        fit = [r for r in rows if r[1] <= BUDGET_SEC]
        if fit:
            out[name] = fit[-1]
    return out


def draw_frame(ax, t, lane_rows, tgt, scale_flops):
    """Draw the full frame for story-time t (0..BUDGET_SEC) on axes `ax`."""
    ax.clear()
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.axis("off")

    ax.add_patch(plt.Rectangle((0, 0), 1080, 1080, color=PALETTE["BG"], zorder=0))
    ax.add_patch(plt.Rectangle((0, 0), 1080, 300, color=PALETTE["CAPTION_BG"], zorder=1))
    ax.add_patch(plt.Rectangle((0, 300), 1080, 4, color=PALETTE["ACCENT"], zorder=2))

    # --- chart header -------------------------------------------------------
    ax.text(40, 1048, "ZEROABSTRACTIONNET", fontsize=20, fontweight="bold",
            color=PALETTE["TEXT"], va="bottom", zorder=3)
    ax.text(1040, 1048, TOP_RIGHT,
            fontsize=_fit_fontsize(ax, TOP_RIGHT, 16, 620),
            color=PALETTE["DIM"], ha="right", va="bottom", zorder=3)
    ax.text(540, 1002, CHART_TITLE,
            fontsize=_fit_fontsize(ax, CHART_TITLE, 32, 1000, "bold"),
            fontweight="bold", color=PALETTE["TEXT"], ha="center", va="center", zorder=3)
    ax.text(540, 966, CHART_SUB,
            fontsize=_fit_fontsize(ax, CHART_SUB, 18, 1000),
            color=PALETTE["DIM"], ha="center", va="center", zorder=3)
    ax.text(540, 928, CHART_LEGEND,
            fontsize=_fit_fontsize(ax, CHART_LEGEND, 17, 1000),
            color=PALETTE["DIM"], ha="center", va="center", zorder=3)

    # --- lanes --------------------------------------------------------------
    for name, (strip_x0, strip_y0, strip_x1, strip_y1) in LANE_STRIPS.items():
        _, _, colour, label, variant = LANES[name]
        ax.add_patch(plt.Rectangle((strip_x0, strip_y0), strip_x1 - strip_x0,
                                   strip_y1 - strip_y0, facecolor=PALETTE["PANEL"],
                                   edgecolor=PALETTE["DIM"], linewidth=1.5, zorder=2))
        ax.text(strip_x0 + 16, (strip_y0 + strip_y1) / 2 + 8, label,
                fontsize=26, fontweight="bold", color=PALETTE["TEXT"],
                ha="left", va="center", zorder=4)
        ax.text(strip_x0 + 16, (strip_y0 + strip_y1) / 2 - 14, variant,
                fontsize=17, color=PALETTE["DIM"], ha="left", va="center", zorder=4)

    for name, (tx0, ty0, tx1, ty1) in TRACKS.items():
        ax.add_patch(plt.Rectangle((tx0, ty0), tx1 - tx0, ty1 - ty0,
                                   color=PALETTE["BG"], zorder=3))

    # --- budget line + label ------------------------------------------------
    ax.plot([BUDGET_X, BUDGET_X], [378, 1040], color=PALETTE["ACCENT"], lw=3, zorder=5)
    ax.text(BUDGET_X - 6, 990, "5 s budget", fontsize=22, fontweight="bold",
            color=PALETTE["ACCENT"], ha="right", va="top", zorder=5)

    # --- time axis ----------------------------------------------------------
    ax.plot([LANE_X0, LANE_X1], [378, 378], color=PALETTE["DIM"], lw=2, zorder=3)
    for sec in range(6):
        x = LANE_X0 + (LANE_X1 - LANE_X0) * sec / BUDGET_SEC
        ax.plot([x, x], [378, 391], color=PALETTE["DIM"], lw=2, zorder=3)
        ax.text(x, 350, f"{sec} s", fontsize=18, color=PALETTE["DIM"],
                ha="center", va="top", zorder=3)

    # --- size chips on the shared timeline: where each backend lands --------
    for name, (n, secs) in tgt.items():
        x = LANE_X0 + (LANE_X1 - LANE_X0) * secs / BUDGET_SEC
        ax.text(min(x, LANE_X1 - 30), 318, f"n={n}", fontsize=18,
                color=PALETTE["DIM"], ha="center", va="top", zorder=3)

    # --- lane fills (flops done vs the asm-blocked target) + landing chips --
    for name, (tx0, ty0, tx1, ty1) in TRACKS.items():
        n, secs = tgt[name]
        gflops = 2.0 * n ** 3 / 1e9
        rate = gflops / secs
        done = min(rate * t, scale_flops)
        frac = done / scale_flops
        _, _, colour, _, _ = LANES[name]
        w = max((tx1 - tx0) * frac, 2.0)
        ax.add_patch(plt.Rectangle((tx0, ty0), w, ty1 - ty0, color=colour, zorder=4))
        if t >= secs:
            tip = tx0 + (tx1 - tx0) * frac
            chip_x = min(max(tip - CHIP_W, tx0), tx1 - CHIP_W)
            chip_y = ty1 + 6
            ax.add_patch(plt.Rectangle((chip_x, chip_y), CHIP_W, CHIP_H,
                                       color=colour, zorder=6))
            ax.text(chip_x + CHIP_W / 2, chip_y + CHIP_H - 12, f"n={n}",
                    fontsize=20, fontweight="bold", color="#FFFFFF",
                    ha="center", va="center", zorder=7)
            ax.text(chip_x + CHIP_W / 2, chip_y + 10, f"{secs:.2f} s",
                    fontsize=13, color="#FFFFFF", alpha=0.92,
                    ha="center", va="center", zorder=7)

    # --- caption band -------------------------------------------------------
    ax.text(540, 265, KICKER,
            fontsize=_fit_fontsize(ax, KICKER, 19, 950, "bold"),
            fontweight="bold", color=PALETTE["ACCENT"],
            ha="center", va="center", zorder=4)
    ax.text(540, 212, HEADLINE,
            fontsize=_fit_fontsize(ax, HEADLINE, 44, 950, "bold"),
            fontweight="bold", color=PALETTE["TEXT"],
            ha="center", va="center", zorder=4)
    ax.text(540, 158, TAGLINE,
            fontsize=_fit_fontsize(ax, TAGLINE, 32, 950, "bold"),
            fontweight="bold", color=PALETTE["ACCENT"],
            ha="center", va="center", zorder=4)
    ax.text(540, 100, SUB,
            fontsize=_fit_fontsize(ax, SUB, 18, 950),
            color=PALETTE["TEXT"], ha="center", va="center", zorder=4)
    ax.text(540, 46, PROVENANCE,
            fontsize=_fit_fontsize(ax, PROVENANCE, 15, 950),
            color=PALETTE["DIM"], ha="center", va="center", zorder=4)


def make_figure():
    fig, ax = plt.subplots(figsize=(10.8, 10.8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def render_movie(lane_rows, tgt, out_path, frames, fps, scale_flops):
    fig, ax = make_figure()
    story_per_frame = BUDGET_SEC / frames
    hold = max(0, frames // 8)

    def update(i):
        t = min(story_per_frame * i, BUDGET_SEC) if i < frames - hold else BUDGET_SEC
        draw_frame(ax, t, lane_rows, tgt, scale_flops)
        return []

    anim = FuncAnimation(fig, update, frames=frames, blit=False)
    writer = FFMpegWriter(fps=fps)
    print(f"rendering {frames} frames @ {fps} fps -> {out_path}")
    anim.save(out_path, writer=writer, dpi=100)
    plt.close(fig)


def render_still(lane_rows, tgt, out_path, t, scale_flops):
    fig, ax = make_figure()
    draw_frame(ax, min(t, BUDGET_SEC), lane_rows, tgt, scale_flops)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"still at story-time {min(t, BUDGET_SEC)}s -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("present", "budget_reel.mp4"))
    parser.add_argument("--csv", default="benchmark_results.csv")
    parser.add_argument("--frames", type=int, default=136, help="total frames (16 hold at the end)")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--still", type=float, default=None,
                        help="instead of the movie, dump a PNG at this story-time (s)")
    args = parser.parse_args()

    lane_rows = load_lanes(args.csv)
    tgt = targets(lane_rows)
    asm_n, asm_t = tgt["asm-blocked"]
    scale_flops = 2.0 * asm_n ** 3 / 1e9
    print("lanes (largest n within 5 s):",
          {k: (v[0], round(v[1], 2)) for k, v in tgt.items()})

    if args.still is not None:
        out = os.path.join("present", f"budget_still_t{args.still}.png")
        render_still(lane_rows, tgt, out, args.still, scale_flops)
    else:
        render_movie(lane_rows, tgt, args.out, args.frames, args.fps, scale_flops)


if __name__ == "__main__":
    main()
