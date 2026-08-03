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


def timed_best_of(backend, A: list[list[float]], B: list[list[float]],
                  repeats: int = 3, variant: str | None = None) -> float:
    """Best-of-N wall-clock timing of backend.matmul(A, B), in seconds.

    Shared by the CLI sweep below and by analyze_run.py, which benchmarks the
    actual matmul shapes the training network uses. `variant` threads the
    implementation variant (e.g. asm 'scalar'/'vectorized'/'blocked', c
    'naive'/'blocked'); None means "use the backend's default variant".
    """
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        if variant is None:
            _ = backend.matmul(A, B)
        else:
            _ = backend.matmul(A, B, variant=variant)
        t1 = time.perf_counter()
        elapsed = t1 - t0
        if elapsed < best:
            best = elapsed
    return best


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
    parser.add_argument("--variant", type=str, default="naive",
                        choices=["naive", "blocked", "scalar", "vectorized"],
                        help="implementation variant; python only has 'naive'")
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

        # The pure-Python backend has no `variant` parameter; thread it for
        # the native backends (which do).
        variant_kw = args.variant if args.backend != "python" else None
        best = timed_best_of(backend, A, B, args.repeats, variant=variant_kw)

        if calib_time is None:
            calib_time = best
            calib_size = n

        print(f"  n={n:4d}  best={best:.4f}s")
        rows.append({"backend": args.backend, "variant": args.variant,
                     "size": n, "seconds": round(best, 6)})

    file_exists = os.path.isfile(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "variant", "size", "seconds"])
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Appended {len(rows)} results to {args.output}")


if __name__ == "__main__":
    main()
