"""Backend-agnostic training animation renderer.

Reads logs/ to produce mp4 (or gif fallback) with three synchronized panels:
network diagram, decision-boundary heatmap, and loss curve.

Usage:
    python animate.py --log-dir logs/run_001 --out animations/run_001.gif
"""

import argparse
import json
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

    epochs = []
    with open(os.path.join(log_dir, "epochs.jsonl")) as f:
        for line in f:
            epochs.append(json.loads(line))

    return meta, epochs


def _draw_network(ax, weights, biases, layer_sizes):
    ax.clear()
    num_layers = len(layer_sizes)

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

    for layer_idx in range(num_layers - 1):
        w = weights[layer_idx]
        src_nodes = node_positions[layer_idx]
        dst_nodes = node_positions[layer_idx + 1]
        max_w = max(abs(val) for row in w for val in row) or 1.0
        for i, (x1, y1) in enumerate(src_nodes):
            for j, (x2, y2) in enumerate(dst_nodes):
                val = w[i][j]
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
    ax.set_title("Network", fontsize=10)


def _draw_decision_boundary(ax, probe_pred, meta, frame_epochs, current_epoch):
    ax.clear()
    res = meta["probe_grid_resolution"]
    grid = [[probe_pred[i * res + j] for j in range(res)] for i in range(res)]

    cmap = plt.cm.RdYlBu
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    ax.imshow(grid, extent=[-1, 1, -1, 1], origin="lower", cmap=cmap, norm=norm, alpha=0.8)

    X = meta["dataset_points"]
    y = meta["dataset_labels"]
    for i, pt in enumerate(X):
        color = "blue" if y[i] == 1.0 else "red"
        ax.scatter(pt[0], pt[1], c=color, s=10, edgecolors="white", linewidths=0.3, zorder=2)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Decision boundary (epoch {current_epoch})", fontsize=10)


def _draw_loss_curve(ax, epochs, losses, current_epoch):
    ax.clear()
    if len(epochs) > 0:
        ax.plot(epochs, losses, color="black", linewidth=1.5)
        ax.scatter(epochs, losses, s=10, color="black", zorder=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss curve", fontsize=10)
    ax.set_xlim(-5, (current_epoch or 500) + 10)
    ax.set_ylim(0, max(losses + [0.7]) * 1.15)
    ax.grid(True, alpha=0.3)


def main():
    parser = argparse.ArgumentParser(description="Render training animation from logs")
    parser.add_argument("--log-dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="animations/training.gif")
    parser.add_argument("--interval", type=int, default=200, help="ms per frame")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    meta, epochs = _load_log(args.log_dir)
    layer_sizes = meta["layer_sizes"]
    total_epochs = [e["epoch"] for e in epochs]
    all_losses = [e["loss"] for e in epochs]

    has_ffmpeg = _check_ffmpeg()
    writer = "ffmpeg" if has_ffmpeg else "pillow"
    ext = os.path.splitext(args.out)[1]
    if ext == ".mp4" and not has_ffmpeg:
        args.out = os.path.splitext(args.out)[0] + ".gif"
        print(f"ffmpeg not found, falling back to {args.out}")
    elif not has_ffmpeg:
        print("ffmpeg not found, using pillow (gif output)")

    fig = plt.figure(figsize=(14, 5))

    def update(frame_idx):
        record = epochs[frame_idx]
        epoch = record["epoch"]
        loss = record["loss"]
        weights = record["weights"]
        biases = record["biases"]

        fig.suptitle(f"Epoch {epoch}  |  Loss: {loss:.6f}", fontsize=12)

        ax1 = fig.add_subplot(1, 3, 1)
        _draw_network(ax1, weights, biases, layer_sizes)

        ax2 = fig.add_subplot(1, 3, 2)
        _draw_decision_boundary(ax2, record["probe_predictions"], meta, total_epochs, epoch)

        ax3 = fig.add_subplot(1, 3, 3)
        _draw_loss_curve(ax3, total_epochs[: frame_idx + 1], all_losses[: frame_idx + 1], epoch)

        plt.tight_layout()

    ani = FuncAnimation(
        fig, update, frames=len(epochs), interval=args.interval, repeat=True,
    )

    ani.save(args.out, writer=writer, dpi=120)
    print(f"Animation saved to {args.out}")


if __name__ == "__main__":
    main()
