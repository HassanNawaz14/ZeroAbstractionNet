"""Cross-backend showcase comparison: per-backend showcase-tier runs,
a comparison table (benchmark_report.md), and an animated log-scale plot.

Runs the showcase tier (--layers 2,32,32,1 --n-per-quadrant 50, 250 epochs,
lr 2.5, log-every 5) for every backend the ops selector can load. Backends
that are not yet implemented (c/asm before their phases land) are skipped
with a note, so this works today with python-only and grows as phases 2-3
arrive.

Usage:
    python compare_backends.py                     # all available backends
    python compare_backends.py --backends python   # explicit selection
    python compare_backends.py --regenerate        # re-render report+plot from
                                                   # saved logs/CSVs, no training
"""

import argparse
import json
import os
import random
import sys
import time

from analyze_run import _matmul_shapes, _random_shaped, GOLDEN_POINTS
from benchmark_matmul import timed_best_of
from config import LR, SEED
from data.generate_data import generate_dataset, generate_probe_grid
from network import Network
from ops import get_backend
from train import parse_layers

SHOWCASE_LAYERS = [2, 32, 32, 1]
SHOWCASE_N = 50
SHOWCASE_EPOCHS = 250
SHOWCASE_LR = 2.5
SHOWCASE_LOG_EVERY = 5
SHOWCASE_PROBE_RES = 40


def run_showcase(backend_name, epochs=SHOWCASE_EPOCHS, log_dir=None):
    """Train the showcase net with `backend_name`; returns a summary dict.
    Logs per the train.py schema into `logs/showcase_<backend>`."""
    backend = get_backend(backend_name)
    if log_dir is None:
        log_dir = f"logs/showcase_{backend_name}"
    os.makedirs(log_dir, exist_ok=True)

    X, y = generate_dataset(SHOWCASE_N, SEED)
    probe = generate_probe_grid(SHOWCASE_PROBE_RES)
    n_points = len(X)

    meta = {
        "backend": backend_name,
        "layer_sizes": SHOWCASE_LAYERS,
        "lr": SHOWCASE_LR,
        "seed": SEED,
        "epochs": epochs,
        "n_per_quadrant": SHOWCASE_N,
        "dataset_points": X,
        "dataset_labels": y,
        "probe_grid": probe,
        "probe_grid_resolution": SHOWCASE_PROBE_RES,
        "compare": True,
    }
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    net = Network(SHOWCASE_LAYERS, backend, SEED)

    log_path = os.path.join(log_dir, "epochs.jsonl")
    with open(log_path, "w"):
        pass

    def should_log(epoch):
        return (epoch == 0 or epoch == epochs - 1
                or epoch % SHOWCASE_LOG_EVERY == 0)

    t_forward = t_backward = t_update = 0.0
    losses = []
    for epoch in range(epochs):
        t0 = time.perf_counter()
        pred, cache = net.forward(X)
        t1 = time.perf_counter()
        loss = Network.binary_cross_entropy(y, pred)
        grads = net.backward(X, y, cache)
        t2 = time.perf_counter()
        net.update(grads, SHOWCASE_LR)
        t3 = time.perf_counter()

        t_forward += t1 - t0
        t_backward += t2 - t1
        t_update += t3 - t2
        losses.append(loss)

        if should_log(epoch):
            probe_pred, _ = net.forward(probe)
            state = net.get_state()
            record = {
                "epoch": epoch,
                "loss": round(loss, 10),
                "weights": state["weights"],
                "biases": state["biases"],
                "dataset_predictions": [row[0] for row in pred],
                "probe_predictions": [row[0] for row in probe_pred],
                "wall_time_sec": round(t1 - t0, 6),
                "t_forward": round(t1 - t0, 9),
                "t_backward": round(t2 - t1, 9),
                "t_update": round(t3 - t2, 9),
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"  [{backend_name}] epoch {epoch:4d}  loss {loss:.6f}  "
                  f"fwd {t1-t0:.4f}s  bwd {t2-t1:.4f}s")

    total = t_forward + t_backward + t_update

    golden_preds = [row[0] for row in net.forward([pt for pt, _ in GOLDEN_POINTS])[0]]
    golden_ok = all((p > 0.5) == (label == 1.0)
                    for p, (_, label) in zip(golden_preds, GOLDEN_POINTS))

    rng = random.Random(0)
    shaped = {}
    for (m, k, n), count in _matmul_shapes(SHOWCASE_LAYERS, n_points):
        A, B = _random_shaped(m, k, n, rng)
        shaped[(m, k, n)] = {
            "count": count,
            "seconds": timed_best_of(backend, A, B, 3),
        }

    import csv
    shaped_csv = "benchmark_shaped.csv"
    shaped_rows = [
        {"backend": backend_name, "M": m, "K": k, "N": n,
         "count": info["count"], "seconds": info["seconds"]}
        for (m, k, n), info in shaped.items()
    ]
    file_exists = os.path.isfile(shaped_csv)
    with open(shaped_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "M", "K", "N", "count", "seconds"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(shaped_rows)
    print(f"Shaped results appended to {shaped_csv}")

    with open(os.path.join(log_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    meta.update({
        "epochs": epochs,
        "final_loss": losses[-1],
        "golden_ok": golden_ok,
        "golden_preds": golden_preds,
        "total_sec": total,
        "epoch_ms": total / epochs * 1000.0,
        "fwd_pct": t_forward / total * 100.0,
        "bwd_pct": t_backward / total * 100.0,
        "upd_pct": t_update / total * 100.0,
    })
    with open(os.path.join(log_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    return {
        "backend": backend_name,
        "epochs": epochs,
        "final_loss": losses[-1],
        "golden_ok": golden_ok,
        "golden_preds": golden_preds,
        "total_sec": total,
        "epoch_ms": total / epochs * 1000.0,
        "fwd_pct": t_forward / total * 100.0,
        "bwd_pct": t_backward / total * 100.0,
        "upd_pct": t_update / total * 100.0,
        "shaped": shaped,
        "log_dir": log_dir,
    }


def _load_sweep(csv_path):
    """Load benchmark_results.csv rows: {label: [(size, seconds)...]}."""
    if not os.path.isfile(csv_path):
        return {}
    import csv
    series = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            label = f"{row['backend']}-{row.get('variant', 'naive')}"
            series.setdefault(label, []).append((int(row["size"]), float(row["seconds"])))
    for label in series:
        series[label].sort()
    return series


def _load_shaped(csv_path):
    """Load benchmark_shaped.csv rows: {backend: [(shape, seconds)...]}."""
    if not os.path.isfile(csv_path):
        return {}
    import csv
    series = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            shape = (int(row["M"]), int(row["K"]), int(row["N"]))
            series.setdefault(row["backend"], []).append((shape, float(row["seconds"])))
    return series


def _results_from_logs(log_dir_fmt="logs/showcase_{name}"):
    """Reconstruct per-run summary dicts from saved showcase logs."""
    results = []
    for name in ("python", "c", "asm"):
        log_dir = log_dir_fmt.format(name=name)
        if not os.path.isfile(os.path.join(log_dir, "epochs.jsonl")):
            continue
        meta = json.load(open(os.path.join(log_dir, "meta.json")))
        n_epochs = meta.get("epochs", SHOWCASE_EPOCHS)
        if "total_sec" in meta:
            results.append({
                "backend": name,
                "epochs": n_epochs,
                "final_loss": meta["final_loss"],
                "golden_ok": meta["golden_ok"],
                "total_sec": meta["total_sec"],
                "epoch_ms": meta["epoch_ms"],
                "fwd_pct": meta["fwd_pct"],
                "bwd_pct": meta["bwd_pct"],
                "upd_pct": meta["upd_pct"],
            })
            continue

        n_records = 0
        t_forward = t_backward = t_update = 0.0
        last_loss = 0.0
        for line in open(os.path.join(log_dir, "epochs.jsonl")):
            rec = json.loads(line)
            t_forward += rec["t_forward"]
            t_backward += rec["t_backward"]
            t_update += rec["t_update"]
            last_loss = rec["loss"]
            n_records += 1
        scale = n_epochs / max(n_records, 1)
        t_forward *= scale
        t_backward *= scale
        t_update *= scale
        total = t_forward + t_backward + t_update
        results.append({
            "backend": name,
            "epochs": n_epochs,
            "final_loss": last_loss,
            "golden_ok": _golden_ok(log_dir, meta),
            "total_sec": total,
            "epoch_ms": total / max(n_epochs, 1) * 1000.0,
            "fwd_pct": t_forward / total * 100.0,
            "bwd_pct": t_backward / total * 100.0,
            "upd_pct": t_update / total * 100.0,
        })
    return results


def _golden_ok(log_dir, meta):
    """Re-check the golden points from the last recorded weights."""
    try:
        from ops import get_backend
        net = Network(meta["layer_sizes"], get_backend(meta["backend"]), meta["seed"])
        lines = open(os.path.join(log_dir, "epochs.jsonl")).readlines()
        rec = json.loads(lines[-1])
        net.weights = rec["weights"]
        net.biases = rec["biases"]
    except Exception:
        return False
    preds = [row[0] for row in net.forward([pt for pt, _ in GOLDEN_POINTS])[0]]
    return all((p > 0.5) == (label == 1.0)
               for p, (_, label) in zip(preds, GOLDEN_POINTS))


def write_report(results=None, sweep_path="benchmark_results.csv",
                 shaped_path="benchmark_shaped.csv", out="benchmark_report.md"):
    """Write the comparison table to benchmark_report.md.
    When results is None, summaries are reconstructed from logs/showcase_*."""
    if results is None:
        results = _results_from_logs()
    sweep = _load_sweep(sweep_path)
    shaped = _load_shaped(shaped_path)

    lines = ["# Benchmark report - cross-backend comparison", ""]
    lines.append("Showcase tier: "
                 + ",".join(str(x) for x in SHOWCASE_LAYERS)
                 + f", n={SHOWCASE_N * 4}, {SHOWCASE_EPOCHS} epochs, lr {SHOWCASE_LR}")
    lines.append("")

    baseline = next((r for r in results if r["backend"] == "python"), None)
    lines.append("## Showcase training runs (per epoch)")
    lines.append("")
    lines.append("| backend | epoch (ms) | fwd % | bwd % | upd % | speedup vs python | final loss | golden ok |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        speedup = f"{baseline['epoch_ms'] / r['epoch_ms']:.1f}x" if baseline else "-"
        lines.append(
            f"| {r['backend']} | {r['epoch_ms']:.1f} | {r['fwd_pct']:.1f} | "
            f"{r['bwd_pct']:.1f} | {r['upd_pct']:.1f} | {speedup} | "
            f"{r['final_loss']:.6f} | {'yes' if r['golden_ok'] else 'NO'} |"
        )
    lines.append("")

    if sweep:
        lines.append("## Square matmul sweep (seconds, best-of-N)")
        lines.append("")
        lines.append("| size | " + " | ".join(sweep.keys()) + " |")
        lines.append("|---|" + "---|" * len(sweep))
        sizes = sorted({s for v in sweep.values() for s, _ in v})
        for n in sizes:
            cells = []
            for label in sweep:
                times = dict(sweep[label])
                cells.append(f"{times.get(n, '-'):.4g}" if n in times else "-")
            lines.append(f"| {n} | " + " | ".join(cells) + " |")
        lines.append("")

    if shaped:
        lines.append("## Shaped matmul (showcase shapes, seconds)")
        lines.append("")
        lines.append("| shape | " + " | ".join(shaped.keys()) + " |")
        lines.append("|---|" + "---|" * len(shaped))
        shapes = sorted({shape for v in shaped.values() for shape, _ in v})
        for shape in shapes:
            cells = []
            for backend in shaped:
                times = dict(shaped[backend])
                cells.append(f"{times[shape]:.5g}" if shape in times else "-")
            lines.append(f"| {'x'.join(str(x) for x in shape)} | " + " | ".join(cells) + " |")
        lines.append("")

    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return out


# Shared line styling for the comparison plots (animated mp4 + static png):
# one colour ramp per backend family, darkening as the implementation gets
# faster (python stays the grey reference baseline). Two lines in the same
# family share the shade so the ramp is obvious; the marker additionally
# differs per stage so lines stay distinguishable even in monochrome.
_FAMILY = {"python": "Python", "c": "C", "asm": "asm"}
_VARIANT = {
    "naive": "naive",
    "scalar": "scalar (stage A)",
    "vectorized": "vectorized (stage B)",
    "blocked": "blocked (final)",
}
_SWEEP_STYLE = {
    "python-naive": ("#4f4f4f", "o"),
    "c-naive": ("#9dc3e8", "^"),
    "c-blocked": ("#1f4e9c", "s"),
    "asm-scalar": ("#ffd884", "^"),
    "asm-vectorized": ("#f5a01d", "s"),
    "asm-blocked": ("#b3340d", "D"),
}
_FAMILY_COLOR = {"python": "#4f4f4f", "c": "#1f4e9c", "asm": "#f5a01d"}


def _display_label(key: str) -> str:
    family, _, variant = key.partition("-")
    return f"{_FAMILY.get(family, family)} {_VARIANT.get(variant, variant)}"


def _flops(shape) -> float:
    m, k, n = shape
    return 2.0 * m * k * n


def _format_axes(ax1, ax2, sweep: dict, shaped: dict):
    """Shared axis setup (scales, labels, titles, grids, limits, legends)
    for both panels. Used by the animated figure and the static companion."""
    max_y = 0.0
    for pts in sweep.values():
        max_y = max(max_y, max(t for _, t in pts))
    for pts in shaped.values():
        max_y = max(max_y, max(t for _, t in pts))

    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Matrix size n (n x n)")
    ax1.set_ylabel("Seconds (best-of-N)")
    ax1.set_title("Matmul sweep (log-log)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.95,
               edgecolor="0.6", title="Implementation", title_fontsize=9)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("FLOPs (2*M*K*N)")
    ax2.set_ylabel("Seconds")
    ax2.set_title("Shaped matmul at network shapes (log-log)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8, framealpha=0.95,
               edgecolor="0.6", title="Backend", title_fontsize=9)

    sweep_sizes = sorted({s for v in sweep.values() for s, _ in v})
    if sweep_sizes:
        ax1.set_xlim(sweep_sizes[0] / 2, sweep_sizes[-1] * 2)
    if shaped:
        flops = [_flops(shape) for v in shaped.values() for shape, _ in v]
        ax2.set_xlim(min(flops) / 2, max(flops) * 2)
    if max_y > 0:
        ax1.set_ylim(1e-5, max_y * 5)
        ax2.set_ylim(1e-5, max_y * 5)


def _render_static(sweep: dict, shaped: dict, out: str, dpi: int = 140) -> str:
    """Static companion to the mp4: a fresh figure with every sweep/shaped
    line drawn at its FULL extent. Dedicated figure so the png never depends
    on whatever frame the animation happens to be sitting on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for label in sorted(sweep):
        color, marker = _SWEEP_STYLE.get(label, ("#888888", "o"))
        ax1.plot([s for s, _ in sweep[label]], [t for _, t in sweep[label]],
                 marker=marker, label=_display_label(label),
                 color=color, lw=1.8, ms=6, mfc="white")

    for backend in sorted(shaped):
        ax2.plot([_flops(shape) for shape, _ in shaped[backend]],
                 [t for _, t in shaped[backend]],
                 marker="s", label=_display_label(backend),
                 color=_FAMILY_COLOR.get(backend, "#888888"),
                 lw=1.8, ms=6, mfc="white")

    fig.suptitle("ZeroAbstractionNet — native matmul backends", fontsize=13)
    _format_axes(ax1, ax2, sweep, shaped)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def render_plot(sweep_path="benchmark_results.csv",
                shaped_path="benchmark_shaped.csv", out="animations/backend_comparison.mp4",
                dpi=100):
    """Animated log-scale comparison: panel A grows the square-sweep lines
    size by size; panel B shows shaped-matmul times per backend (log-y)."""
    sweep = _load_sweep(sweep_path)
    shaped = _load_shaped(shaped_path)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    sweep_sizes = sorted({s for v in sweep.values() for s, _ in v})

    lines1 = []
    for label in sorted(sweep):
        color, marker = _SWEEP_STYLE.get(label, ("#888888", "o"))
        line, = ax1.plot([], [], marker=marker, label=_display_label(label),
                         color=color, lw=1.8, ms=6, mfc="white")
        lines1.append(line)

    lines2 = []
    for backend in sorted(shaped):
        line, = ax2.plot([], [], marker="s", label=_display_label(backend),
                         color=_FAMILY_COLOR.get(backend, "#888888"),
                         lw=1.8, ms=6, mfc="white")
        lines2.append(line)

    fig.suptitle("ZeroAbstractionNet — native matmul backends", fontsize=13)
    _format_axes(ax1, ax2, sweep, shaped)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    def update(frame):
        n_shown = frame + 1
        for line, (label, pts) in zip(lines1, sorted(sweep.items())):
            shown = [(s, t) for s, t in pts if s <= sweep_sizes[min(n_shown - 1, len(sweep_sizes) - 1)]]
            line.set_data([s for s, _ in shown], [t for _, t in shown])
        for line, (backend, pts) in zip(lines2, sorted(shaped.items())):
            line.set_data([_flops(shape) for shape, _ in pts], [t for _, t in pts])
        return lines1 + lines2

    if sweep_sizes:
        frames = len(sweep_sizes)
        anim = FuncAnimation(fig, update, frames=frames, interval=600, repeat=False)
    else:
        anim = None

    if anim is not None:
        print(f"Rendering comparison animation ({frames} frames)...")
        anim.save(out, writer="ffmpeg", dpi=dpi)
        print(f"Animation saved to {out}")
        png_out = out.replace(".mp4", ".png")
        _render_static(sweep, shaped, png_out, dpi=140)
        print(f"Static plot saved to {png_out}")
    else:
        print("No sweep data - saving static plot only")
        fig.savefig(out.replace(".mp4", ".png"), dpi=dpi)


def main():
    parser = argparse.ArgumentParser(description="Cross-backend showcase comparison")
    parser.add_argument("--backends", type=str, default=None,
                        help="comma-separated backends to compare (default: all loadable)")
    parser.add_argument("--epochs", type=int, default=SHOWCASE_EPOCHS)
    parser.add_argument("--no-animation", action="store_true",
                        help="skip the animation, only write the report")
    parser.add_argument("--regenerate", action="store_true",
                        help="re-write the report and plot from saved logs/CSVs, "
                             "no retraining")
    args = parser.parse_args()

    if args.regenerate:
        report = write_report()
        print(f"Comparison table written to {report}")
        if not args.no_animation:
            render_plot()
        return 0

    candidates = ["python", "c", "asm"]
    if args.backends:
        candidates = [b.strip() for b in args.backends.split(",")]

    results = []
    for name in candidates:
        try:
            get_backend(name)
        except NotImplementedError as e:
            print(f"skipping backend '{name}' (not implemented yet): {e}")
            continue
        print(f"\n=== Showcase run with backend '{name}' ===")
        results.append(run_showcase(name, epochs=args.epochs))

    if not results:
        print("No usable backends. Nothing to compare.")
        return 1

    report = write_report(results)
    print(f"\nComparison table written to {report}")
    print("Showcase summary:")
    baseline = next((r for r in results if r["backend"] == "python"), None)
    for r in results:
        speedup = f"{baseline['epoch_ms'] / r['epoch_ms']:.1f}x" if baseline else "-"
        print(f"  {r['backend']:8s}  epoch {r['epoch_ms']:8.1f} ms  "
              f"loss {r['final_loss']:.6f}  golden ok={r['golden_ok']}  speedup {speedup}")

    if not args.no_animation:
        render_plot()
    return 0


if __name__ == "__main__":
    sys.exit(main())
