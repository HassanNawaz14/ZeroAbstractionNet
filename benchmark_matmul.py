"""Standalone matmul-only scaling benchmark for all backends.

Usage:
    python benchmark_matmul.py --backend python --sizes 16,32,64,128,256,512 --repeats 3
"""

import argparse
import csv
import os
import random
import time

from ops import get_backend


def _generate_matrix(n: int, rng: random.Random) -> list[list[float]]:
    return [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]


def _estimate_time(n: int, ref_size: int, ref_time: float) -> float:
    """Estimate wall time for n x n matmul given measured time at ref_size.
    Assumes O(n^3) scaling.
    """
    if ref_time <= 0:
        return 0.0
    ratio = (n / ref_size) ** 3
    return ref_time * ratio


def main():
    parser = argparse.ArgumentParser(description="Matmul-only scaling benchmark")
    parser.add_argument("--backend", type=str, default="python", choices=["python", "c", "asm"])
    parser.add_argument("--sizes", type=str, default="16,32,64,128,256,512")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--force", action="store_true", help="skip time estimation safety check")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    backend = get_backend(args.backend)
    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    rng = random.Random(0)

    calib_size = sizes[0]
    calib_time = None

    rows = []
    for n in sizes:
        if calib_time is not None and n != calib_size:
            estimated = _estimate_time(n, calib_size, calib_time)
            if estimated > 60.0 and not args.force:
                print(
                    f"  n={n}: estimated {estimated:.1f}s (>{60}s), "
                    f"skip with --force to override"
                )
                continue

        A, B = _generate_matrix(n, rng), _generate_matrix(n, rng)

        best = float("inf")
        for rep in range(args.repeats):
            t0 = time.perf_counter()
            _ = backend.matmul(A, B)
            t1 = time.perf_counter()
            elapsed = t1 - t0
            if elapsed < best:
                best = elapsed

        if calib_time is None:
            calib_time = best
            calib_size = n

        print(f"  n={n:4d}  best={best:.4f}s")
        rows.append({"backend": args.backend, "size": n, "seconds": round(best, 6)})

    file_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "size", "seconds"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Appended {len(rows)} results to {args.output}")


if __name__ == "__main__":
    main()
