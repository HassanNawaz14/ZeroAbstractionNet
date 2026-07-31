"""cProfile harness for the training loop.

Usage:
    python profile_run.py --epochs 50 --backend python
"""

import argparse
import cProfile
import io
import pstats
import time

from config import LAYER_SIZES, LR, SEED, N_PER_QUADRANT
from data.generate_data import generate_dataset
from network import Network
from ops import get_backend


def main():
    parser = argparse.ArgumentParser(description="Profile the training loop")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--backend", type=str, default="python", choices=["python", "c", "asm"])
    args = parser.parse_args()

    backend = get_backend(args.backend)
    X, y = generate_dataset(N_PER_QUADRANT, SEED)
    net = Network(LAYER_SIZES, backend, SEED)

    t_forward = 0.0
    t_backward = 0.0
    t_update = 0.0

    pr = cProfile.Profile()
    pr.enable()

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        pred, cache = net.forward(X)
        t1 = time.perf_counter()
        loss = Network.binary_cross_entropy(y, pred)
        grads = net.backward(X, y, cache)
        t2 = time.perf_counter()
        net.update(grads, LR)
        t3 = time.perf_counter()

        t_forward += t1 - t0
        t_backward += t2 - t1
        t_update += t3 - t2

        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  loss {loss:.6f}")

    pr.disable()

    out = io.StringIO()
    ps = pstats.Stats(pr, stream=out)
    ps.sort_stats("cumulative")
    ps.print_stats(15)
    cprofile_output = out.getvalue()

    print("\n" + "=" * 60)
    print("cProfile top 15 by cumulative time:")
    print("=" * 60)
    print(cprofile_output)

    total = t_forward + t_backward + t_update
    print("=" * 60)
    print("Manual timing breakdown (wall-clock, sum over all epochs):")
    print("=" * 60)
    print(f"  forward:  {t_forward:.4f}s  ({t_forward/total*100:.1f}%)")
    print(f"  backward: {t_backward:.4f}s  ({t_backward/total*100:.1f}%)")
    print(f"  update:   {t_update:.4f}s  ({t_update/total*100:.1f}%)")
    print(f"  total:    {total:.4f}s")

    report = (
        f"Profile run: backend={args.backend}, epochs={args.epochs}\n"
        f"{'='*60}\n"
        f"{cprofile_output}\n"
        f"Manual timing:\n"
        f"  forward:  {t_forward:.4f}s  ({t_forward/total*100:.1f}%)\n"
        f"  backward: {t_backward:.4f}s  ({t_backward/total*100:.1f}%)\n"
        f"  update:   {t_update:.4f}s  ({t_update/total*100:.1f}%)\n"
        f"  total:    {total:.4f}s\n"
    )
    with open("profile_baseline.txt", "w") as f:
        f.write(report)
    print(f"\nFull report saved to profile_baseline.txt")


if __name__ == "__main__":
    main()
