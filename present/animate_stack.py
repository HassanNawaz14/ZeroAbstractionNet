"""Render the "one 8x8 matmul, four layers" stack drill-down (plan.md Build 2).

Same 8x8 product computed down the stack: pure Python (triple loop) ->
ctypes bridge -> C (blocked kernel) -> x86-64 AVX2/FMA asm (8-lane kernel).
The code panel walks the REAL committed sources (ops/backend_python.py,
ops/backend_c.py, native/c/matmul.c, native/asm/matmul.asm), highlighting
the executing line with its real line number — the animation literally walks
the repo.

This module owns the palette + layout constants that `verify_frames.py`
imports for the 'stack' pixel spec.

Usage:
    python present/animate_stack.py [--out present/stack_drilldown.mp4]
    python present/animate_stack.py --still 16.0   # dump a single PNG frame
"""

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from animate_budget import PALETTE, _fit_fontsize

# ---------------------------------------------------------------------------
# Palette + layout (source of truth — verify_frames.py imports this)
# ---------------------------------------------------------------------------
CANVAS = (1920, 1080)
LEVEL_COLORS = {
    "python": PALETTE["LANE_PY"],    # #F0563B
    "ctypes": "#B58CFF",
    "c": PALETTE["LANE_C"],          # #3FA7FF
    "asm": PALETTE["LANE_ASM"],      # #3DDC84
}
CODE_DIM = "#9AA5B5"

FPS = 24
FRAMES = 720                       # 30.0 s
PHASES = [
    ("intro", 0, 72),              # 0.0 - 3.0 s
    ("python", 72, 240),           # 3.0 - 10.0 s
    ("ctypes", 240, 324),          # 10.0 - 13.5 s
    ("c", 324, 468),               # 13.5 - 19.5 s
    ("asm", 468, 636),             # 19.5 - 26.5 s
    ("outro", 636, 720),           # 26.5 - 30.0 s
]

GRIDS = {"A": (60, 590, 337, 867), "B": (375, 590, 652, 867), "C": (690, 590, 967, 867)}
CELL = 32                          # cell size px
STEP = 35                          # cell size + gap
CODE_PANEL = (1030, 340, 1880, 800)
CHIP_BOX = (1030, 838, 1790, 898)  # breadcrumb chips row
OUTRO_STACK_BOX = (180, 280, 760, 840)
TITLE_BOX = (380, 540, 1540, 860)

SOURCES = {
    "python": "ops/backend_python.py",
    "ctypes": "ops/backend_c.py",
    "c": "native/c/matmul.c",
    "asm": "native/asm/matmul.asm",
}
LEVEL_META = {
    "python": ("LEVEL 1 — PURE PYTHON", "ops/backend_python.py",
               "interpreted · one scalar dot product at a time"),
    "ctypes": ("LEVEL 2 — CTYPES BRIDGE", "ops/backend_c.py",
               "marshalling only — no math computed here"),
    "c": ("LEVEL 3 — COMPILED C", "native/c/matmul.c",
          "blocked loops · contiguous memory walk"),
    "asm": ("LEVEL 4 — HAND-WRITTEN ASSEMBLY", "native/asm/matmul.asm",
            "AVX2 · FMA · 8 floats per instruction"),
}

# (relative frame, real source line, annotation) per level
WALKS = {
    "python": [
        (0, 16, "row i of the output"),
        (24, 18, "column j"),
        (36, 19, "fresh accumulator"),
        (44, 20, "k sweep: 8 terms"),
        (56, 21, "the multiply-add"),
        (120, 22, "store the row"),
        (140, 23, "append into C"),
    ],
    "ctypes": [
        (0, 21, "dlopen the shared object"),
        (18, 24, "declare the ABI"),
        (30, 26, "six arguments, no return"),
        (40, 79, "flatten A -> flat double[]"),
        (44, 80, "flatten B"),
        (52, 84, "the call crosses into C"),
        (70, 92, "unflatten back to lists"),
    ],
    "c": [
        (0, 41, "outer loop: row blocks"),
        (10, 43, "outer loop: column blocks"),
        (20, 44, "row i inside the tile"),
        (30, 45, "k step p"),
        (42, 46, "hoisted: a_ip loaded once"),
        (60, 47, "innermost j loop"),
        (90, 48, "contiguous walk: C += a_ip * B"),
        (120, 44, "next row (one block covers 8x8)"),
    ],
    "asm": [
        (0, 292, "8-wide output-column groups"),
        (15, 301, "acc = 8 lanes (ymm0)"),
        (45, 309, "A[i][p] broadcast to all 8 lanes"),
        (75, 313, "B band: 8 contiguous floats"),
        (100, 314, "vfmadd231ps: fused multiply-add"),
        (135, 321, "re-add previous k-block"),
        (150, 323, "store 8 lanes back to C"),
    ],
}
CODE_WINDOW = 13                   # visible lines above/below the highlight
MAX_LINE = 76                      # chars shown per source line


# ---------------------------------------------------------------------------
# data: one deterministic 8x8 product (same everywhere, exact in float32)
# ---------------------------------------------------------------------------
def build_matrices():
    A = [[(i * 7 + j * 13) % 10 - 4 for j in range(8)] for i in range(8)]
    B = [[((i * 11 + j * 5) % 8) / 2.0 + 0.5 for j in range(8)] for i in range(8)]
    C = [[0.0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(8):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(8))
    return A, B, C


A, B, C = build_matrices()


def load_source(path):
    with open(path, encoding="utf-8") as fh:
        return [ln.rstrip() for ln in fh]


LINES = {name: load_source(p) for name, p in SOURCES.items()}


# ---------------------------------------------------------------------------
# drawing helpers
# ---------------------------------------------------------------------------
def cell_box(grid, i, j):
    """Pixel box (x0, y0, x1, y1) of cell (row i, col j) in a grid origin."""
    ox, oy = {"A": (60, 590), "B": (375, 590), "C": (690, 590)}[grid]
    x = ox + j * STEP
    y = oy + (7 - i) * STEP
    return x, y, x + CELL, y + CELL


def grid_value(grid, i, j, partial=None):
    if grid == "A":
        return f"{A[i][j]:g}"
    if grid == "B":
        return f"{B[i][j]:g}"
    v = partial if partial is not None else C[i][j]
    return f"{v:.1f}"


def draw_grid(ax, grid, fill_cells, pulse_cells, colour, value_map=None):
    """Draw one 8x8 grid; fill_cells/pulse_cells = sets of (i, j) highlighted."""
    ox, oy = {"A": (60, 590), "B": (375, 590), "C": (690, 590)}[grid]
    x1, y1 = ox + 8 * STEP - 3, oy + 8 * STEP - 3
    ax.add_patch(plt.Rectangle((ox - 4, oy - 4), 8 * STEP - 3 + 8, 8 * STEP - 3 + 8,
                               facecolor="none", edgecolor=PALETTE["DIM"],
                               linewidth=2, zorder=2))
    for i in range(8):
        for j in range(8):
            x, y, ex, ey = cell_box(grid, i, j)
            if (i, j) in fill_cells:
                face = colour
                txt = PALETTE["TEXT"]
            elif (i, j) in pulse_cells:
                face = colour
                txt = PALETTE["TEXT"]
            else:
                face = PALETTE["PANEL"]
                txt = PALETTE["DIM"]
            ax.add_patch(plt.Rectangle((x, y), CELL, CELL, facecolor=face,
                                       edgecolor=PALETTE["DIM"], linewidth=1, zorder=3))
            value = value_map.get((i, j)) if value_map else grid_value(grid, i, j)
            ax.text(x + CELL / 2, y + CELL / 2, value, fontsize=13,
                    fontweight="bold" if face == colour else "normal",
                    color=txt, ha="center", va="center", zorder=4)


def draw_code(ax, level, fr_rel):
    """Code panel: real source window with the executing line highlighted."""
    colour = LEVEL_COLORS[level]
    title, path, tag = LEVEL_META[level]
    lines = LINES[level]
    walk = WALKS[level]

    ax.add_patch(plt.Rectangle((CODE_PANEL[0], CODE_PANEL[1]),
                               CODE_PANEL[2] - CODE_PANEL[0],
                               CODE_PANEL[3] - CODE_PANEL[1],
                               facecolor=PALETTE["BG"], edgecolor=colour,
                               linewidth=3, zorder=1))
    ax.add_patch(plt.Rectangle((CODE_PANEL[0], CODE_PANEL[3] - 6),
                               CODE_PANEL[2] - CODE_PANEL[0], 6,
                               facecolor=colour, zorder=2))

    hl_line = None
    note = ""
    for start, line_no, ann in walk:
        if fr_rel >= start:
            hl_line, note = line_no, ann
    if hl_line is None:
        hl_line, note = walk[0][1], walk[0][2]

    ax.text(CODE_PANEL[0] + 24, CODE_PANEL[3] - 30, title,
            fontsize=28, fontweight="bold", color=colour, ha="left", va="center", zorder=4)
    ax.text(CODE_PANEL[2] - 24, CODE_PANEL[3] - 30, path,
            fontsize=16, color=PALETTE["DIM"], ha="right", va="center", zorder=4)
    ax.text(CODE_PANEL[0] + 24, CODE_PANEL[3] - 62, tag,
            fontsize=17, color=PALETTE["ACCENT"], ha="left", va="center", zorder=4)

    lo = max(0, hl_line - 6)
    hi = min(len(lines), lo + CODE_WINDOW)
    lo = max(0, hi - CODE_WINDOW)
    y_top = CODE_PANEL[3] - 92
    line_h = 24
    for idx, line_no in enumerate(range(lo + 1, hi + 1)):
        y = y_top - idx * line_h - line_h / 2
        src = lines[line_no - 1]
        if len(src) > MAX_LINE:
            src = src[:MAX_LINE - 1] + "…"
        if line_no == hl_line:
            ax.add_patch(plt.Rectangle((CODE_PANEL[0] + 4, y - line_h / 2 + 1),
                                       CODE_PANEL[2] - CODE_PANEL[0] - 8, line_h - 2,
                                       facecolor=colour, zorder=3))
            ax.text(CODE_PANEL[0] + 16, y, f"{line_no:>3}", fontsize=13,
                    color=PALETTE["TEXT"], fontfamily="monospace",
                    ha="left", va="center", zorder=4)
            ax.text(CODE_PANEL[0] + 56, y, src, fontsize=13, fontweight="bold",
                    color=PALETTE["TEXT"], fontfamily="monospace",
                    ha="left", va="center", zorder=4)
        else:
            ax.text(CODE_PANEL[0] + 16, y, f"{line_no:>3}", fontsize=13,
                    color=PALETTE["DIM"], fontfamily="monospace",
                    ha="left", va="center", zorder=4)
            ax.text(CODE_PANEL[0] + 56, y, src, fontsize=13,
                    color=CODE_DIM, fontfamily="monospace",
                    ha="left", va="center", zorder=4)
    ax.text(CODE_PANEL[0] + 24, y_top - len(range(lo, hi)) * line_h - 26, f"▸ {note}",
            fontsize=18, fontweight="bold", color=PALETTE["ACCENT"],
            ha="left", va="center", zorder=4)


def draw_breadcrumb(ax, current):
    names = ["python", "ctypes", "c", "asm"]
    x = CHIP_BOX[0]
    for i, name in enumerate(names):
        active = name == current
        ax.add_patch(plt.Rectangle((x, CHIP_BOX[1]), 160, CHIP_BOX[3] - CHIP_BOX[1],
                                   facecolor=LEVEL_COLORS[name] if active else PALETTE["PANEL"],
                                   edgecolor=PALETTE["DIM"], linewidth=1.5, zorder=3))
        ax.text(x + 80, (CHIP_BOX[1] + CHIP_BOX[3]) / 2, name.upper(),
                fontsize=20, fontweight="bold",
                color=PALETTE["TEXT"] if active else PALETTE["DIM"],
                ha="center", va="center", zorder=4)
        x += 160
        if i < 3:
            ax.text(x + 15, (CHIP_BOX[1] + CHIP_BOX[3]) / 2, "→", fontsize=20,
                    color=PALETTE["DIM"], ha="center", va="center", zorder=3)
            x += 30


def draw_progress(ax, fr):
    seg_x0 = 40
    seg_w = (CANVAS[0] - 80) / len(PHASES)
    for i, (name, f0, f1) in enumerate(PHASES):
        x = seg_x0 + i * seg_w
        active = f0 <= fr < f1
        colour = LEVEL_COLORS.get(name, PALETTE["PANEL"]) if active else PALETTE["PANEL"]
        ax.add_patch(plt.Rectangle((x, 24), seg_w - 6, 14,
                                   facecolor=colour,
                                   edgecolor=PALETTE["DIM"], linewidth=1, zorder=3))
    head = seg_x0 + (CANVAS[0] - 80) * (fr / FRAMES)
    ax.plot([head, head], [18, 44], color=PALETTE["ACCENT"], lw=3, zorder=4)


def draw_flat_bar(ax, x0, y, label, colour, fill):
    """64-cell flat array visual for the ctypes marshalling story."""
    ax.text(x0, y + 26, label, fontsize=16, color=PALETTE["DIM"],
            ha="left", va="center", zorder=3)
    n = int(fill * 64)
    for k in range(64):
        x = x0 + k * 5
        filled = k < n
        ax.add_patch(plt.Rectangle((x, y), 4, 14,
                                   facecolor=colour if filled else PALETTE["PANEL"],
                                   edgecolor=PALETTE["DIM"], linewidth=0.5, zorder=3))


def draw_outro(ax):
    names = ["python", "ctypes", "c", "asm"]
    x0, y0 = OUTRO_STACK_BOX[0], OUTRO_STACK_BOX[3]
    box_w = OUTRO_STACK_BOX[2] - OUTRO_STACK_BOX[0]
    box_h = 100
    for i, name in enumerate(names):
        y = y0 - i * (box_h + 26)
        title, path, tag = LEVEL_META[name]
        ax.add_patch(plt.Rectangle((x0, y - box_h), box_w, box_h,
                                   facecolor=PALETTE["PANEL"],
                                   edgecolor=LEVEL_COLORS[name], linewidth=3, zorder=3))
        ax.add_patch(plt.Rectangle((x0, y - box_h), 10, box_h,
                                   facecolor=LEVEL_COLORS[name], zorder=4))
        ax.text(x0 + 30, y - 26, title, fontsize=22, fontweight="bold",
                color=PALETTE["TEXT"], ha="left", va="center", zorder=4)
        ax.text(x0 + 30, y - 70, path, fontsize=15, color=LEVEL_COLORS[name],
                ha="left", va="center", zorder=4)
        if i < 3:
            ax.text(x0 + box_w / 2, y - box_h - 13, "↓", fontsize=20,
                    color=LEVEL_COLORS[name], ha="center", va="center", zorder=4)

    sx = 980
    ax.text(sx, 800, "SAME RESULT", fontsize=30, fontweight="bold",
            color=PALETTE["ACCENT"], ha="center", va="center", zorder=4)
    for k, name in enumerate(["python", "c", "asm"]):
        y = 700 - k * 120
        ax.add_patch(plt.Rectangle((sx - 210, y - 34), 420, 68,
                                   facecolor=PALETTE["PANEL"],
                                   edgecolor=LEVEL_COLORS[name], linewidth=2, zorder=3))
        ax.text(sx - 190, y, name.upper(), fontsize=17, fontweight="bold",
                color=LEVEL_COLORS[name], ha="left", va="center", zorder=4)
        ax.text(sx + 190, y, f"C[0][0] = {C[0][0]:.1f}", fontsize=20,
                fontweight="bold", color=PALETTE["TEXT"],
                ha="right", va="center", zorder=4)
    ax.text(sx, 300, "then watch the same budget: n=256 → n=2048",
            fontsize=24, fontweight="bold", color=PALETTE["TEXT"],
            ha="center", va="center", zorder=4)


def render_frame(ax, fr):
    ax.clear()
    ax.set_xlim(0, CANVAS[0])
    ax.set_ylim(0, CANVAS[1])
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), CANVAS[0], CANVAS[1],
                               facecolor=PALETTE["BG"], zorder=0))

    phase = PHASES[-1][0]
    for name, f0, f1 in PHASES:
        if f0 <= fr < f1:
            phase = name
            fr_rel = fr - f0
            break

    draw_progress(ax, fr)

    if phase == "intro":
        ax.text(960, 800, "ONE 8×8 MATMUL. FOUR LAYERS. SAME RESULT.",
                fontsize=_fit_fontsize(ax, "ONE 8×8 MATMUL. FOUR LAYERS. SAME RESULT.",
                                       54, 1500, "bold"),
                fontweight="bold", color=PALETTE["TEXT"],
                ha="center", va="center", zorder=4)
        ax.text(960, 730, "python → ctypes → C → hand-written AVX2/FMA asm — every line below is real code from this repo.",
                fontsize=_fit_fontsize(ax, "python → ctypes → C → hand-written AVX2/FMA asm — every line below is real code from this repo.",
                                       22, 1500),
                color=PALETTE["DIM"], ha="center", va="center", zorder=4)
        ax.text(960, 680, "same 8×8 product · python/C float64 · asm float32 — same bytes out",
                fontsize=19, color=PALETTE["ACCENT"],
                ha="center", va="center", zorder=4)
        draw_breadcrumb(ax, None)
        draw_grid(ax, "A", set(), set(), LEVEL_COLORS["python"])
        draw_grid(ax, "B", set(), set(), LEVEL_COLORS["python"])
        draw_grid(ax, "C", set(), set(), LEVEL_COLORS["python"])

    elif phase == "python":
        colour = LEVEL_COLORS["python"]
        a_done = a_pulse = b_done = b_pulse = set()
        c_done, c_pulse, values = set(), set(), {}
        if fr_rel < 60:
            k = min(7, fr_rel * 8 // 60)
            a_done = {(0, kk) for kk in range(k)}
            a_pulse = {(0, k)}
            b_done = {(kk, 0) for kk in range(k)}
            b_pulse = {(k, 0)}
            c_pulse = {(0, 0)}
            s = sum(A[0][kk] * B[kk][0] for kk in range(k + 1))
            values[(0, 0)] = f"{s:.1f}"
        else:
            ci = min(63, (fr_rel - 60) * 64 // 108)
            i, j = divmod(ci, 8)
            c_done = {(ii, jj) for ii in range(8) for jj in range(8)
                      if ii * 8 + jj < ci}
            c_pulse = {(i, j)}
        draw_grid(ax, "A", a_done, a_pulse, colour)
        draw_grid(ax, "B", b_done, b_pulse, colour)
        draw_grid(ax, "C", c_done, c_pulse, colour, value_map=values)
        draw_code(ax, "python", fr_rel)
        draw_breadcrumb(ax, "python")
        ax.text(828, 540, "C[0][0] = sum over k", fontsize=17, color=PALETTE["DIM"],
                ha="center", va="center", zorder=4)

    elif phase == "ctypes":
        colour = LEVEL_COLORS["ctypes"]
        n_a = min(1.0, fr_rel * 64 // 30 / 64) if fr_rel < 30 else 1.0
        n_b = min(1.0, max(0, fr_rel - 10) * 64 // 25 / 64)
        n_cf = min(1.0, max(0, fr_rel - 30) * 64 // 15 / 64)
        done = set()
        if fr_rel >= 45:
            ci = min(63, (fr_rel - 45) * 64 // 39)
            done = {(ii, jj) for ii in range(8) for jj in range(8)
                    if ii * 8 + jj < ci}
        draw_flat_bar(ax, 60, 470, "A → flat double[64] (ctypes)", colour, n_a)
        draw_flat_bar(ax, 375, 470, "B → flat double[64]", colour, n_b)
        draw_flat_bar(ax, 690, 470, "C flat double[64] ← result", colour, n_cf)
        draw_grid(ax, "A", set(), set(), colour)
        draw_grid(ax, "B", set(), set(), colour)
        draw_grid(ax, "C", done, set(), colour)
        draw_code(ax, "ctypes", fr_rel)
        draw_breadcrumb(ax, "ctypes")
        ax.text(960, 540, "matrices cross as flat row-major arrays — the ABI boundary",
                fontsize=17, color=PALETTE["DIM"], ha="center", va="center", zorder=4)
        if fr_rel >= 50:
            ax.text(960, 460, "✓ same result as Python", fontsize=22,
                    fontweight="bold", color=PALETTE["ACCENT"],
                    ha="center", va="center", zorder=4)

    elif phase == "c":
        colour = LEVEL_COLORS["c"]
        row = min(7, fr_rel // 18)
        sub = fr_rel % 18
        p = min(7, sub // 2)
        done = {(ii, jj) for ii in range(row) for jj in range(8)}
        done |= {(row, jj) for jj in range(p)}
        pulse = {(row, p)}
        a_pulse = {(row, p)}
        b_band = {(p, jj) for jj in range(8)}
        draw_grid(ax, "A", a_pulse, set(), colour)
        draw_grid(ax, "B", b_band, set(), colour)
        draw_grid(ax, "C", done, pulse, colour)
        draw_code(ax, "c", fr_rel)
        draw_breadcrumb(ax, "c")
        ax.text(828, 540, "a_ip hoisted · j walks B's row = contiguous memory",
                fontsize=17, color=PALETTE["DIM"], ha="center", va="center", zorder=4)

    elif phase == "asm":
        colour = LEVEL_COLORS["asm"]
        row = min(7, fr_rel // 21)
        sub = fr_rel % 21
        p = min(7, sub // 2)
        done = {(ii, jj) for ii in range(row) for jj in range(8)}
        done |= {(row, jj) for jj in range(p)}
        lanes = {(row, jj) for jj in range(8)}
        values = {(row, jj): f"{sum(A[row][pp] * B[pp][jj] for pp in range(p + 1)):.1f}"
                  for jj in range(8)}
        draw_grid(ax, "A", {(row, p)}, set(), colour)
        draw_grid(ax, "B", {(p, jj) for jj in range(8)}, set(), colour)
        draw_grid(ax, "C", done, lanes, colour, value_map=values)
        draw_code(ax, "asm", fr_rel)
        draw_breadcrumb(ax, "asm")
        ax.text(828, 540, "8 lanes = 8 output columns in one YMM register",
                fontsize=17, color=PALETTE["DIM"], ha="center", va="center", zorder=4)
        for jj in range(8):
            x, y, ex, ey = cell_box("C", row, jj)
            ax.text(x + CELL / 2, y + CELL + 10, f"L{jj}", fontsize=10,
                    color=colour, ha="center", va="center", zorder=4)

    elif phase == "outro":
        draw_outro(ax)
        draw_breadcrumb(ax, None)


def make_figure():
    fig, ax = plt.subplots(figsize=(19.2, 10.8), dpi=100)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig, ax


def _render_one(args):
    """Render frame `fr` to a PNG in `outdir`. Top-level for ProcessPool."""
    fr, outdir = args
    fig, ax = make_figure()
    render_frame(ax, fr)
    fig.savefig(os.path.join(outdir, f"f_{fr:04d}.png"), dpi=100)
    plt.close(fig)
    return fr


def render_movie(out_path, frames, fps, workers):
    """Render frames in parallel across cores, then encode with ffmpeg.

    Per-frame work is ~200-500 artists, so serial rendering at 720 frames
    takes far too long; this bounds wall-clock time by using all cores and
    reports progress so it never looks hung.
    """
    with tempfile.TemporaryDirectory(prefix="stack_frames_") as tmpdir:
        print(f"rendering {frames} frames with {workers} workers -> PNGs in {tmpdir}")
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = [pool.submit(_render_one, (fr, tmpdir)) for fr in range(frames)]
            for fut in as_completed(futs):
                fut.result()
                done += 1
                if done % 60 == 0 or done == frames:
                    print(f"  {done}/{frames} frames ({done / frames * 100:.0f}%)")

        print(f"encoding {frames} frames @ {fps} fps -> {out_path}")
        res = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", os.path.join(tmpdir, "f_%04d.png"),
             "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
             "-movflags", "+faststart", out_path],
            capture_output=True, text=True,
        )
        if res.returncode != 0:
            sys.exit(f"ffmpeg encode failed: {res.stderr}")
    print(f"done -> {out_path}")


def render_still(out_path, t):
    fig, ax = make_figure()
    render_frame(ax, min(int(t * FPS), FRAMES - 1))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"still at {t}s (frame {min(int(t * FPS), FRAMES - 1)}) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join("present", "stack_drilldown.mp4"))
    parser.add_argument("--frames", type=int, default=FRAMES,
                        help="total frames (default 720 = 30 s @ 24 fps)")
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--still", type=float, default=None,
                        help="instead of the movie, dump a PNG at this video-time (s)")
    args = parser.parse_args()

    if args.still is not None:
        out = os.path.join("present", f"stack_still_t{args.still}.png")
        render_still(out, args.still)
    else:
        render_movie(args.out, args.frames, args.fps, args.workers)


if __name__ == "__main__":
    main()
