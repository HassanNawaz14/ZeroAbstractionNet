"""Training loop CLI.

Usage:
    python train.py --backend python --epochs 500 --lr 0.5 --n-per-quadrant 25 \
                     --seed 0 --log-every 5 --log-dir logs/run_001
    python train.py --layers 2,32,32,1 --n-per-quadrant 50 --epochs 250 \
                     --lr 2.5 --log-dir logs/showcase_python   # showcase tier
"""

import argparse
import json
import os
import time

from config import LAYER_SIZES, LR, SEED, N_PER_QUADRANT, LOG_EVERY, EPOCHS, PROBE_RESOLUTION
from data.generate_data import generate_dataset, generate_probe_grid
from network import Network
from ops import get_backend


def parse_layers(s: str) -> list[int]:
    """Parse '2,32,32,1' into [2, 32, 32, 1]."""
    layers = [int(x.strip()) for x in s.split(",")]
    if len(layers) < 2:
        raise ValueError("--layers must be a comma-separated list of >= 2 sizes")
    return layers


def main():
    parser = argparse.ArgumentParser(description="Train a feedforward network on XOR-quadrant data")
    parser.add_argument("--backend", type=str, default="python", choices=["python", "c", "asm"])
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-per-quadrant", type=int, default=N_PER_QUADRANT)
    parser.add_argument("--log-every", type=int, default=LOG_EVERY)
    parser.add_argument("--log-dir", type=str, default="logs/run_001")
    parser.add_argument("--layers", type=str, default=None,
                        help="comma-separated layer sizes, e.g. '2,32,32,1' (default: config.LAYER_SIZES)")
    parser.add_argument("--probe-resolution", type=int, default=PROBE_RESOLUTION,
                        help="resolution of the animation probe grid (default: config.PROBE_RESOLUTION)")
    args = parser.parse_args()

    backend = get_backend(args.backend)
    os.makedirs(args.log_dir, exist_ok=True)

    layer_sizes = parse_layers(args.layers) if args.layers else LAYER_SIZES

    X, y = generate_dataset(args.n_per_quadrant, args.seed)
    probe = generate_probe_grid(args.probe_resolution)

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
    }
    with open(os.path.join(args.log_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    net = Network(layer_sizes, backend, args.seed)

    log_path = os.path.join(args.log_dir, "epochs.jsonl")
    # Truncate any stale log from a previous run into this dir — each run is
    # self-contained; appending would interleave duplicates for animate.py.
    with open(log_path, "w"):
        pass

    def should_log(epoch):
        return epoch == 0 or epoch == args.epochs - 1 or epoch % args.log_every == 0

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        pred, cache = net.forward(X)
        loss = Network.binary_cross_entropy(y, pred)
        grads = net.backward(X, y, cache)
        net.update(grads, args.lr)
        t1 = time.perf_counter()

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
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(record) + "\n")
            print(f"epoch {epoch:4d}  loss {loss:.6f}  {t1-t0:.4f}s")


if __name__ == "__main__":
    main()
