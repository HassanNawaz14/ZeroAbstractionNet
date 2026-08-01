# Phase 2 — C Matmul Backend

## Prerequisite
Phase 1 (`01_python_phase.md`) is complete: `train.py --backend python`
works, `benchmark_matmul.py` and `profile_run.py` have produced baseline
numbers, `profile_baseline.txt` confirms matmul dominates runtime. This
phase does **not** touch `network.py`, `train.py`, `animate.py`, the
dataset, or the logging schema — it only adds a new `ops` backend and wires
it into the existing selector. If you find yourself editing
`network.py`, stop — the interface from phase 1 should already support
this.

## Two-tier verification — where phase 2 must show its speedup

The phase-1 two-tier strategy applies to everything in this phase.

**Correctness is verified at the demo tier** (tiny `[2,4,4,1]` net, 100
points): C output must match the pure-Python backend within `1e-9`, and
`train.py --backend c` must reach the same converged loss as
`--backend python` within `1e-6` with all golden points correct. Do NOT
expect C to be faster at demo tier — an entire epoch is only ~2-10 ms and
the ~1-5 µs ctypes marshalling overhead per call swamps the compute. "C
looks equal or slower at the tiny scale" is expected, not a bug, and must
never be "fixed" by restructuring the demo.

**Efficiency is demonstrated at the showcase tier** (`--layers 2,32,32,1
--n-per-quadrant 50 --epochs 250 --lr 2.5 --log-every 5`, pure-Python
epoch ~160 ms). There the C backend's compute advantage dwarfs
marshalling and should measure ~30-80× faster per epoch. Always report
showcase-tier numbers — via `compare_backends.py` and
`benchmark_report.md` — alongside the demo-tier parity results.

## Scope
Replace **only** `matmul` with a C implementation. `add_bias`,
`transpose`, and `elementwise` (activations) stay pure Python in every
phase — they're O(n), not the bottleneck, and rewriting them adds
ctypes marshalling overhead that would swamp their tiny cost.

## Precision decision
Use `double` (not `float`) in C for this phase, matching Python's native
`float` type exactly. This lets correctness tests diff phase 2 output
against phase 1 output with a very tight tolerance (effectively exact,
modulo floating-point summation order differences from loop reordering —
see below). Phase 3 (assembly) will introduce a `float32` variant
separately for SIMD-width reasons; don't pre-optimize for that here.

## Data representation
Python lists-of-lists don't cross the ctypes boundary cheaply or safely.
Define one conversion layer, used only inside `ops/backend_c.py`:
- Matrices are flattened to **row-major 1D arrays** of `double` before
  crossing into C, and the C function writes into a caller-allocated flat
  output buffer.
- Shapes (`n`, `k`, `m`) are passed explicitly as `int` arguments — C code
  never infers shape from the data.

## Directory additions
```
native/c/
├── matmul.c
├── matmul.h
└── Makefile
```

## `native/c/matmul.h`
```c
#ifndef MATMUL_H
#define MATMUL_H

// A is n x k (row-major), B is k x m (row-major), C is n x m (row-major,
// pre-allocated by the caller, this function only writes into it).
void matmul_naive(const double *A, const double *B, double *C, int n, int k, int m);

// Same contract, but with loop order and access pattern optimized for
// cache locality (see matmul.c for details). Both functions must be
// exported — the naive one stays as a benchmarking baseline forever, it's
// not dead code to delete once the optimized one exists.
void matmul_blocked(const double *A, const double *B, double *C, int n, int k, int m);

#endif
```

## `native/c/matmul.c` — `matmul_naive`
Direct, unoptimized triple loop. This exists specifically to isolate "the
speedup from compiled C vs interpreted Python" from "the speedup from
smarter memory access patterns" — don't let it accidentally become
optimized.
```c
void matmul_naive(const double *A, const double *B, double *C, int n, int k, int m) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++) {
            double sum = 0.0;
            for (int p = 0; p < k; p++) {
                sum += A[i * k + p] * B[p * m + j];
            }
            C[i * m + j] = sum;
        }
    }
}
```
Note this has the classic bad access pattern: `B[p * m + j]` strides
through memory by `m` doubles on every inner-loop iteration — cache-hostile
by design. That's the point of this version.

## `native/c/matmul.c` — `matmul_blocked`
Two independent improvements, both required:
1. **Loop reorder (i-k-j)**: swap loop nesting so the innermost loop walks
   contiguous memory in both `B` and `C`:
   ```c
   for (int i = 0; i < n; i++) {
       for (int p = 0; p < k; p++) {
           double a_ip = A[i * k + p];
           for (int j = 0; j < m; j++) {
               C[i * m + j] += a_ip * B[p * m + j];
           }
       }
   }
   ```
   (Remember to zero-initialize `C` first since this version accumulates
   with `+=` across the `p` loop.)
2. **Cache blocking/tiling** on top of the reordered loops: process the
   matrices in fixed-size blocks (start with a `BLOCK_SIZE` of 64, expose
   it as a `#define` so it's easy to tune) so working sets stay resident
   in L1/L2 cache for large `n`. Structure: three nested "outer" loops over
   block indices, with the i-k-j loops from above running inside each
   block. Reference "loop tiling for matrix multiplication" if unfamiliar
   with the exact block-loop nesting pattern — the key invariant is that
   at every point in the innermost loop, the memory being touched should
   fit within L1 (~32KB typically) or L2.
3. Compile with `-O2` (not `-O3 -march=native` for this baseline — see
   note below on why we hold back auto-vectorization here).

**Important**: do not pass `-march=native`/`-mavx2`/`-ffast-math` for this
phase's build. We want phase 2's numbers to reflect "what a careful but
portable C implementation gets you," so that phase 3's hand-written SIMD
assembly has a fair, distinct story to tell ("here's what happens when we
go further than the compiler will by default"). It's fine — expected,
even — if `gcc -O3 -march=native` would auto-vectorize this loop into
something close to what phase 3 hand-writes; don't use those flags here,
note this reasoning in the Makefile as a comment so it doesn't look like an
oversight later.

## `native/c/Makefile`
Build a shared library, not a static binary:
```makefile
CC = gcc
CFLAGS = -O2 -fPIC -Wall -Wextra
TARGET = libmatmul.so

$(TARGET): matmul.c matmul.h
	$(CC) $(CFLAGS) -shared -o $(TARGET) matmul.c

clean:
	rm -f $(TARGET)

.PHONY: clean
```
Output `libmatmul.so` should land in `native/c/` — `backend_c.py` loads it
by relative path from there (compute the path from `__file__`, don't
hardcode absolute paths).

## `ops/backend_c.py`
```python
import ctypes
import os

_lib_path = os.path.join(os.path.dirname(__file__), "..", "native", "c", "libmatmul.so")
_lib = ctypes.CDLL(_lib_path)

_lib.matmul_naive.argtypes = [
    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
    ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int, ctypes.c_int,
]
_lib.matmul_naive.restype = None
# same argtypes/restype registration for matmul_blocked

def _flatten(A: list[list[float]]) -> tuple[ctypes.Array, int, int]:
    """Row-major flatten to a ctypes double array, returns (array, rows, cols)."""

def _unflatten(flat: ctypes.Array, n: int, m: int) -> list[list[float]]:
    """Back to list-of-lists for the rest of the codebase."""

def matmul(A: list[list[float]], B: list[list[float]], variant: str = "blocked") -> list[list[float]]:
    """variant in {'naive', 'blocked'}. Default 'blocked' is what train.py
    uses; benchmark_matmul.py exercises both explicitly."""
```
This module must expose the **same public function signature** as
`ops/backend_python.py`'s `matmul(A, B)` (list-of-lists in, list-of-lists
out) — all the ctypes flattening is an internal implementation detail
hidden inside this file. `add_bias`/`transpose`/`elementwise` are simply
re-exported from `backend_python` (import and alias them) rather than
reimplemented — every non-C backend module does this.

Update `ops/__init__.py`'s `get_backend("c")` to import and return this
module instead of raising `NotImplementedError`.

## Correctness testing — `tests/test_ops_c.py`
1. Reuse the exact same golden 2x2 case from phase 1
   (`matmul([[1,2],[3,4]], [[5,6],[7,8]]) == [[19,22],[43,50]]`) against
   both `variant='naive'` and `variant='blocked'`.
2. Property test: generate several random matrices (seeded, various
   non-square shapes including edge cases like `k=1` and non-multiple-of-
   `BLOCK_SIZE` dimensions — blocking bugs love to hide at boundary sizes),
   compute with `backend_python.matmul` and both C variants, assert all
   three agree within `1e-9` absolute tolerance (not exact equality —
   summation order differs between naive/blocked/python so tiny float
   rounding differences are expected and fine; anything larger than
   `1e-9` on doubles indicates an actual bug, not rounding).
3. Full pipeline test: run `train.py --backend c --epochs 500` on the same
   seed as the phase-1 golden run, assert the final loss matches the
   phase-1 pure-Python run's final loss within `1e-6` — this validates the
   whole plumbing, not just the matmul function in isolation.

## Benchmarking
Extend `benchmark_matmul.py` (already built in phase 1) with
`--backend c` and a `--variant {naive,blocked}` flag, appending rows to the
same `benchmark_results.csv`. Because C is dramatically faster, the size
sweep can now go much larger (e.g. up to 1024 or 2048) before hitting the
same wall-clock budget that pure Python hit at 256-512 — extend `--sizes`
accordingly and let the "estimate first, ask before proceeding" logic from
phase 1 handle it automatically.

The showcase-tier training runs themselves are compared with
`compare_backends.py`: it re-runs the showcase config per available
backend, writes `benchmark_report.md` with a table of epoch time / phase
split / speedup-vs-python / final loss / golden-check, and renders the
animated log-scale plot from `benchmark_results.csv` + `benchmark_shaped.csv`
(one line per backend/variant). This is the per-network companion to the
square-sweep plot — both are required.

Produce the final `benchmark_report.md` (via `compare_backends.py` — it
also re-renders the report and plot from saved `logs/showcase_*` runs
without retraining when called without arguments) with a table: `size |
python (s) | c-naive (s) | c-blocked (s) |
speedup (python/c-blocked)`. This is the deliverable that shows the story
so far — save it, phase 3 appends to it rather than replacing it.

## Definition of done for phase 2
- [ ] `native/c/libmatmul.so` builds cleanly via `make -C native/c` with no
      warnings under `-Wall -Wextra`.
- [ ] `train.py --backend c` produces the same converged loss (within
      `1e-6`) as `train.py --backend python` on the same seed, and the
      same 4 golden points still classify correctly.
- [ ] `tests/test_ops_c.py` passes, including the boundary-size blocking
      cases.
- [ ] `benchmark_results.csv` / `benchmark_report.md` show naive-C beating
      naive-Python by roughly 1-2 orders of magnitude, and blocked-C
      beating naive-C further at larger sizes (the gap should widen, not
      stay constant, as size grows past cache size — if it doesn't widen,
      the blocking implementation likely has a bug or `BLOCK_SIZE` is
      poorly tuned for this machine's cache).
- [ ] `animate.py --log-dir logs/<c-run>` works unchanged, no edits needed.
- [ ] Showcase-tier comparison: `compare_backends.py` shows c-blocked at
      an order of magnitude (or better) speedup over python per epoch at
      `[2,32,32,1]`/n=200, while demo-tier loss parity (1e-6) and the
      golden points still hold.
