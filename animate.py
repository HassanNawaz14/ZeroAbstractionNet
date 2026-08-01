"""Backend-agnostic training animation renderer.

Reads logs/ to produce mp4 (or gif fallback) with three synchronized panels:
network diagram, decision-boundary heatmap, and loss curve.

Usage:
    python animate.py --log-dir logs/run_001 --out animations/run_001.gif
"""

import argparse
import json
import math
import os
import subprocess
import sys

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation


def _check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _load_log(log_dir: str):
    with open(os.path.join(log_dir, "meta.json")) as f:
        meta = json.load(f)

    records = []
    with open(os.path.join(log_dir, "epochs.jsonl")) as f:
        for line in f:
            records.append(json.loads(line))

    # epochs.jsonl may contain duplicate epochs if train.py ran multiple times
    # into the same log dir (append mode). Keep the LAST record per epoch so a
    # polluted log still animates as one coherent run.
    by_epoch = {}
    for rec in records:
        by_epoch[rec["epoch"]] = rec
    epochs = [by_epoch[ep] for ep in sorted(by_epoch)]

    return meta, epochs


def _select_frames(epochs, max_frames):
    """Sample records so the animation spans the whole run (first and last
    epoch always included) without exceeding `max_frames` frames."""
    n = len(epochs)
    if n <= max_frames:
        return epochs
    stride = math.ceil(n / max_frames)
    idxs = list(range(0, n, stride))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)
    return [epochs[i] for i in idxs]


def _draw_network(ax, weights, biases, layer_sizes):
    ax.clear()
    num_layers = len(layer_sizes)
    max_edges_per_layer = 300

    x_positions = list(range(num_layers))
    node_positions = []
    for layer_idx, size in enumerate(layer_sizes):
        ys = [i - (size - 1) / 2 for i in range(size)]
        xs = [x_positions[layer_idx]] * size
        node_positions.append(list(zip(xs, ys)))

    for layer_idx in range(num_layers):
        for i, (x, y) in enumerate(node_positions[layer_idx]):
            color = "lightgray"
            if layer_idx == 0:
                color = "lightblue"
            elif layer_idx == num_layers - 1:
                color = "lightgreen"
            else:
                color = "white"
            circle = plt.Circle((x, y), 0.3, color=color, ec="black", linewidth=1.0, zorder=3)
            ax.add_patch(circle)
            bias = biases[layer_idx - 1][i] if layer_idx > 0 else 0
            ax.text(x, y - 0.55, f"{bias:.2f}", ha="center", va="top", fontsize=6, color="gray")

    thinned = False
    for layer_idx in range(num_layers - 1):
        w = weights[layer_idx]
        src_nodes = node_positions[layer_idx]
        dst_nodes = node_positions[layer_idx + 1]

        # At showcase scale a layer can have thousands of connections (e.g.
        # 32x32 = 1024) — drawing them all is slow AND unreadable. Keep only
        # the largest-magnitude edges per layer, like a saliency view.
        if len(src_nodes) * len(dst_nodes) > max_edges_per_layer:
            edges = sorted(
                ((i, j, w[i][j]) for i in range(len(src_nodes)) for j in range(len(dst_nodes))),
                key=lambda t: abs(t[2]), reverse=True,
            )[:max_edges_per_layer]
            thinned = True
        else:
            edges = list(
                (i, j, w[i][j]) for i in range(len(src_nodes)) for j in range(len(dst_nodes))
            )

        max_w = max(abs(val) for _, _, val in edges) or 1.0
        for i, j, val in edges:
            x1, y1 = src_nodes[i]
            x2, y2 = dst_nodes[j]
            width = 0.5 + 2.5 * abs(val) / max_w
            color = "blue" if val >= 0 else "red"
            ax.plot(
                [x1, x2], [y1, y2],
                color=color, linewidth=width, alpha=min(1.0, abs(val) / max_w + 0.2),
                zorder=1,
            )

    ax.set_xlim(-0.8, num_layers - 1 + 0.8)
    y_all = [y for nodes in node_positions for _, y in nodes]
    y_margin = max(1.0, (max(y_all) - min(y_all)) / 2 + 1.0)
    ax.set_ylim(-y_margin, y_margin)
    ax.set_aspect("equal")
    ax.axis("off")
    title = "Network (top 300 edges/layer)" if thinned else "Network"
    ax.set_title(title, fontsize=10)


def _create_boundary_artists(ax, meta):
    """Create the static parts of the decision-boundary panel ONCE (dataset
    scatter, labels, limits) and return the heatmap image to update per frame.
    Re-creating artists every frame was the main rendering cost."""
    res = meta["probe_grid_resolution"]
    cmap = plt.cm.RdYlBu
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    im = ax.imshow(
        [[0.0] * res for _ in range(res)],
        extent=[-1, 1, -1, 1], origin="lower", cmap=cmap, norm=norm, alpha=0.8,
    )

    X = meta["dataset_points"]
    y = meta["dataset_labels"]
    colors = ["blue" if label == 1.0 else "red" for label in y]
    ax.scatter(
        [pt[0] for pt in X], [pt[1] for pt in X], c=colors, s=10,
        edgecolors="white", linewidths=0.3, zorder=2,
    )

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Decision boundary", fontsize=10)
    return im


def _update_boundary(im, probe_pred, meta):
    res = meta["probe_grid_resolution"]
    im.set_data([[probe_pred[i * res + j] for j in range(res)] for i in range(res)])


def _create_loss_artists(ax, total_epochs, all_losses):
    """Static loss-panel setup; returns (line, scatter) to update per frame."""
    line, = ax.plot([], [], color="black", linewidth=1.5)
    pts = ax.scatter([], [], s=10, color="black", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curve", fontsize=10)
    ax.set_xlim(-5, max(total_epochs + [1]) + 10)
    ax.set_ylim(0, max(all_losses + [0.7]) * 1.15)
    ax.grid(True, alpha=0.3)
    return line, pts


def _update_loss(line, pts, epochs_so_far, losses_so_far):
    line.set_data(epochs_so_far, losses_so_far)
    pts.set_offsets(list(zip(epochs_so_far, losses_so_far)))


def _create_phase_time_artists(ax, records):
    """Optional 4th panel: forward/backward/update wall time per epoch (ms),
    shown only when the log records carry per-phase timings (analyze_run.py
    writes them; train.py does not)."""
    line_f, = ax.plot([], [], color="tab:blue", linewidth=1.2, label="forward")
    line_b, = ax.plot([], [], color="tab:red", linewidth=1.2, label="backward")
    line_u, = ax.plot([], [], color="tab:green", linewidth=1.2, label="update")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ms")
    ax.set_title("Phase time / epoch", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    last_epoch = records[-1]["epoch"]
    all_ms = [
        rec["t_forward"] * 1000.0 for rec in records
    ] + [
        rec["t_backward"] * 1000.0 for rec in records
    ] + [
        rec["t_update"] * 1000.0 for rec in records
    ]
    ax.set_xlim(-5, last_epoch + 10)
    ax.set_ylim(0, max(all_ms + [0.5]) * 1.15)
    return line_f, line_b, line_u


def _update_phase_time(lines, records_so_far):
    f, b, u = lines
    epochs = [rec["epoch"] for rec in records_so_far]
    f.set_data(epochs, [rec["t_forward"] * 1000.0 for rec in records_so_far])
    b.set_data(epochs, [rec["t_backward"] * 1000.0 for rec in records_so_far])
    u.set_data(epochs, [rec["t_update"] * 1000.0 for rec in records_so_far])


def render(log_dir: str, out: str, interval: int = 150, max_frames: int = 400, dpi: int = 100) -> str:
    """Render the training animation for a log dir. Returns the output path
    actually written (may switch to .gif if ffmpeg is missing)."""
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    meta, epochs = _load_log(log_dir)
    layer_sizes = meta["layer_sizes"]
    epochs = _select_frames(epochs, max_frames)
    total_epochs = [e["epoch"] for e in epochs]
    all_losses = [e["loss"] for e in epochs]
    has_phase_times = "t_forward" in epochs[0]

    has_ffmpeg = _check_ffmpeg()
    writer = "ffmpeg" if has_ffmpeg else "pillow"
    ext = os.path.splitext(out)[1]
    if ext == ".mp4" and not has_ffmpeg:
        out = os.path.splitext(out)[0] + ".gif"
        print(f"ffmpeg not found, falling back to {out}")
    elif not has_ffmpeg:
        print("ffmpeg not found, using pillow (gif output)")

    ncols = 4 if has_phase_times else 3
    fig = plt.figure(figsize=(17, 5) if ncols == 4 else (14, 5))
    # Create the panels ONCE and reuse them — calling add_subplot inside
    # update() would create a fresh copy of every axes each frame, making
    # rendering quadratic in the number of frames (a frame-250 figure was
    # rendering 750 stacked axes per draw).
    ax1 = fig.add_subplot(1, ncols, 1)
    ax2 = fig.add_subplot(1, ncols, 2)
    ax3 = fig.add_subplot(1, ncols, 3)
    fig.tight_layout()

    im_boundary = _create_boundary_artists(ax2, meta)
    line_loss, pts_loss = _create_loss_artists(ax3, total_epochs, all_losses)

    phase_lines = None
    if has_phase_times:
        ax4 = fig.add_subplot(1, ncols, 4)
        phase_lines = _create_phase_time_artists(ax4, epochs)

    def update(frame_idx):
        record = epochs[frame_idx]
        epoch = record["epoch"]
        loss = record["loss"]
        weights = record["weights"]
        biases = record["biases"]

        fig.suptitle(f"Epoch {epoch}  |  Loss: {loss:.6f}", fontsize=12)

        _draw_network(ax1, weights, biases, layer_sizes)
        _update_boundary(im_boundary, record["probe_predictions"], meta)
        _update_loss(line_loss, pts_loss, total_epochs[: frame_idx + 1], all_losses[: frame_idx + 1])
        if phase_lines is not None:
            _update_phase_time(phase_lines, epochs[: frame_idx + 1])

    print(f"Rendering {len(epochs)} frames ({epochs[0]['epoch']} -> {epochs[-1]['epoch']}) "
          f"with writer '{writer}', dpi={dpi}, interval={interval}ms ...")
    sys.stdout.flush()

    def progress(frame_idx, *args):
        if frame_idx % 50 == 0 or frame_idx == len(epochs) - 1:
            print(f"  frame {frame_idx + 1}/{len(epochs)}", flush=True)

    ani = FuncAnimation(
        fig, update, frames=len(epochs), interval=interval, repeat=False,
    )
    ani.save(out, writer=writer, dpi=dpi, progress_callback=progress)
    print(f"Animation saved to {out}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Render training animation from logs")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="animations/training.gif")
    parser.add_argument("--interval", type=int, default=150, help="ms per frame")
    parser.add_argument("--max-frames", type=int, default=400,
                        help="frame budget; whole run is always covered (first/last epoch included)")
    parser.add_argument("--dpi", type=int, default=100, help="render resolution (lower = faster)")
    args = parser.parse_args()
    render(args.log_dir, args.out, args.interval, args.max_frames, args.dpi)


if __name__ == "__main__":
    main()
