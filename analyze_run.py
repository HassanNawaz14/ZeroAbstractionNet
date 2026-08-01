"""Benchmark + profile the ACTUAL training MLP, end to end, with animation.

Composes the two standalone measurement tools but points them at the network
train.py actually trains:

- benchmark_matmul.py  -> timed_best_of() best-of-N timing, applied to the
                         real matmul shapes the network uses (batched
                         forward/backward on the actual dataset), not random
                         square matrices.
- profile_run.py       -> forward / backward / update wall-clock split,
                         accumulated over the whole training run.

The run itself is identical to train.py (same config, dataset, log schema),
so animate.py renders it unchanged — with one addition: each logged record
carries t_forward/t_backward/t_update, which makes animate.py show a 4th
"Phase time / epoch" panel. train.py is deliberately untouched: running this
tool never slows down normal training runs.

Usage:
    python analyze_run.py --epochs 250 --lr 2.5 --log-dir logs/analyze_250 \
        --out animations/analyze_250.mp4
"""

import argparse
import csv
import json
import os
import random
import time

from benchmark_matmul import timed_best_of
from config import LAYER_SIZES, LR, SEED, N_PER_QUADRANT, PROBE_RESOLUTION
from data.generate_data import generate_dataset, generate_probe_grid
from network import Network
from ops import get_backend
from train import parse_layers

GOLDEN_POINTS = [
    ([0.5, 0.5], 1.0),
    ([-0.5, -0.5], 1.0),
    ([0.5, -0.5], 0.0),
    ([-0.5, 0.5], 0.0),
]


def _matmul_shapes(layer_sizes, n_points):
    """The (M, K, N) shapes backend.matmul is called with during a training
    epoch on this network and dataset: forward is X*W per layer, backward is
    grad*W^T per layer (same K*N product, transposed). Identical shapes
    (e.g. two 32->32 hidden layers) are deduplicated with a count."""
    shape_counts = {}
    for fan_in, fan_out in zip(layer_sizes, layer_sizes[1:]):
        for shape in ((n_points, fan_in, fan_out), (n_points, fan_out, fan_in)):
            shape_counts[shape] = shape_counts.get(shape, 0) + 1
    return sorted(shape_counts.items())


def _random_shaped(m, k, n, rng):
    A = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(m)]
    B = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(k)]
    return A, B


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark + profile the actual training MLP, with animation output"
    )
    parser.add_argument("--backend", type=str, default="python", choices=["python", "c", "asm"])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-quadrant", type=int, default=N_PER_QUADRANT)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--log-dir", type=str, default="logs/analyze_run")
    parser.add_argument("--layers", type=str, default=None,
                        help="comma-separated layer sizes, e.g. '2,32,32,1' (default: config.LAYER_SIZES)")
    parser.add_argument("--probe-resolution", type=int, default=PROBE_RESOLUTION)
    parser.add_argument("--repeats", type=int, default=3, help="best-of-N for shaped matmul timing")
    parser.add_argument("--out", type=str, default=None,
                        help="animation output path (renders the run after training)")
    parser.add_argument("--interval", type=int, default=150)
    parser.add_argument("--max-frames", type=int, default=400)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    backend = get_backend(args.backend)
    os.makedirs(args.log_dir, exist_ok=True)

    layer_sizes = parse_layers(args.layers) if args.layers else LAYER_SIZES

    X, y = generate_dataset(args.n_per_quadrant, args.seed)
    probe = generate_probe_grid(args.probe_resolution)
    n_points = len(X)

    meta = {
        "backend": args.backend,
        "layer_sizes": layer_sizes,
        "lr": args.lr,
        "seed": args.seed,
        "n_per_quadrant": args.n_per_quadrant,
        "dataset_points": X,
        "dataset_labels": y,
        "probe_grid": probe,
        "probe_grid_resolution": args.probe_resolution,
        "analyze": True,
    }
    with open(os.path.join(args.log_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    net = Network(layer_sizes, backend, args.seed)

    log_path = os.path.join(args.log_dir, "epochs.jsonl")
    with open(log_path, "w"):
        pass

    def should_log(epoch):
        return epoch == 0 or epoch == args.epochs - 1 or epoch % args.log_every == 0

    t_forward = t_backward = t_update = 0.0
    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        pred, cache = net.forward(X)
        t1 = time.perf_counter()
        loss = Network.binary_cross_entropy(y, pred)
        grads = net.backward(X, y, cache)
        t2 = time.perf_counter()
        net.update(grads, args.lr)
        t3 = time.perf_counter()

        t_forward += t1 - t0
        t_backward += t2 - t1
        t_update += t3 - t2

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
            print(f"epoch {epoch:4d}  loss {loss:.6f}  fwd {t1-t0:.4f}s  bwd {t2-t1:.4f}s  upd {t3-t2:.4f}s")

    total = t_forward + t_backward + t_update

    golden = net.forward([pt for pt, _ in GOLDEN_POINTS])[0]
    golden_preds = [row[0] for row in golden]
    golden_ok = all((p > 0.5) == (label == 1.0) for p, (_, label) in zip(golden_preds, GOLDEN_POINTS))

    rng = random.Random(0)
    shape_counts = _matmul_shapes(layer_sizes, n_points)
    print("\nShaped matmul benchmark (actual network shapes, best of "
          f"{args.repeats}, backend '{args.backend}'):")
    shaped_rows = []
    for (m, k, n), count in shape_counts:
        A, B = _random_shaped(m, k, n, rng)
        best = timed_best_of(backend, A, B, args.repeats)
        note = f" x{count}" if count > 1 else ""
        print(f"  ({m:3d}x{k:2d})*({k:2d}x{n:2d}){note}  best={best:.6f}s")
        shaped_rows.append({
            "backend": args.backend, "M": m, "K": k, "N": n,
            "count": count, "seconds": round(best, 9),
        })

    shaped_csv = "benchmark_shaped.csv"
    file_exists = os.path.isfile(shaped_csv)
    with open(shaped_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "M", "K", "N", "count", "seconds"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(shaped_rows)

    summary = (
        f"analyze_run: backend={args.backend}, epochs={args.epochs}, lr={args.lr}, "
        f"seed={args.seed}, n_per_quadrant={args.n_per_quadrant}\n"
        f"layers={layer_sizes}, dataset_points={n_points}\n"
        f"final loss={loss:.8f}\n"
        f"golden points: {[f'{p:.4f}' for p in golden_preds]} -> all correct: {golden_ok}\n"
        f"\nphase split (sum over {args.epochs} epochs):\n"
        f"  forward:  {t_forward:.4f}s  ({t_forward/total*100:.1f}%)\n"
        f"  backward: {t_backward:.4f}s  ({t_backward/total*100:.1f}%)\n"
        f"  update:   {t_update:.4f}s  ({t_update/total*100:.1f}%)\n"
        f"  total:    {total:.4f}s\n"
        f"per-epoch avg: fwd {t_forward/args.epochs*1000:.3f}ms, "
        f"bwd {t_backward/args.epochs*1000:.3f}ms, upd {t_update/args.epochs*1000:.3f}ms\n"
        f"\nshaped matmul best-of-{args.repeats} (seconds):\n"
        + "".join(
            f"  ({r['M']}x{r['K']})*({r['K']}x{r['N']})"
            f"{' x' + str(r['count']) if r['count'] > 1 else ''}  {r['seconds']:.6f}\n"
            for r in shaped_rows
        )
    )
    report_path = os.path.join(args.log_dir, "analysis.txt")
    with open(report_path, "w") as f:
        f.write(summary)
    print(f"\nReport saved to {report_path}")
    print(f"Shaped results appended to {shaped_csv}")

    if args.out:
        from animate import render
        render(args.log_dir, args.out, args.interval, args.max_frames, args.dpi)


if __name__ == "__main__":
    main()
