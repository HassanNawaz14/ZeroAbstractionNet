# Phase 2 Plan — C Matmul Backend (working plan, checked 2026-08-03)

## Session status (updated 2026-08-03)

- **Milestone 1 done + verified end-to-end:** `native/c/` (clean build, 0
  warnings), `ops/backend_c.py`, `get_backend("c")` flip. Demo-tier parity
  vs python = **bit-identical** (weights, biases, probe/dataset preds,
  losses — 0.0 diff over all epochs), golden points correct.
- **Determinism fix (this is load-bearing):** Makefile now builds with
  `-ffp-contract=off` (see Makefile comment). Without it, gcc may fuse
  `a*b+c` into one `vfmadd` (single rounding vs python's two), and this
  net trains at lr 2.5 — a chaotic amplification regime. A 1-ulp diff at
  epoch 0 grows to different minima (loss 0.0091 vs 0.0118 at 250 ep).
  Empirically gcc was NOT contracting on this box, but the flag pins the
  guarantee; cross-backend parity already holds with it off.
- **Animation equivalence proven (user request):**
  `logs/py_run_250_lr25` vs `logs/c_run_250_lr25` (both 250 ep, lr 2.5,
  seed 0, n=25, log-every 1, rendered @ 120 ms) are BIT-IDENTICAL, and the
  two mp4s (`animations/py_run_250_lr25.mp4`, `c_run_250_lr25.mp4`) are
  30.0 s with the SAME MD5. Only differences: `meta.json` `backend` field
  and `wall_time_sec` (unused by the renderer).
- **Gotcha recorded:** `logs/run_250_lr25` + `run_250_lr25.mp4` predate
  the demo-tier config closeout (`f3a7df1`) and were produced by older
  code — they diverge from today's python run and must NOT be used as the
  python reference. Always regenerate both sides when comparing.
- **Milestone 2 done — real `tests/test_ops_c.py`** (replaced the 2 skip
  stubs): golden 2x2 for both variants; 10 boundary shapes × 2 variants vs
  `backend_python` (k=1, 65, 127, past-`BLOCK_SIZE` dims, network-shaped)
  at 1e-9; naive-vs-blocked agreement; mismatch/unknown-variant raises; and
  the full pipeline test (runs the real `train.py --backend c` vs python,
  500 ep, seed 0 → final-loss parity < 1e-6 + golden points, verified
  green). Module-level load guard: on Windows the whole module is skipped
  with the `make -C native/c` build hint. **Result: WSL `30 passed, 3
  skipped` (asm phase-3 stubs); Windows `14 passed, 4 skipped`.**
- To-do (next milestone): the two minor fixes (`_golden_ok` seed arg,
  `02_c_phase.md` regenerate wording), showcase-tier comparison via
  `compare_backends.py` (c rows in `benchmark_report.md` + mp4),
  `benchmark_matmul.py --backend c --variant naive|blocked` sweep up to
  2048, `animate.py` on `logs/showcase_c`, docs + commit.
- **Milestone 3 (benchmarks) done:** interleaved/regenerated
  `benchmark_results.csv` sweep ([16..2048], naive + blocked, reps
  budgeted so naive stays out of the 2048-timeout hole), regenerated
  `benchmark_shaped.csv` (deduped to the latest run), and re-rendered
  `benchmark_report.md` + `backend_comparison.mp4`. Numbers: sweep shows
  naive-C beating naive-Python ~20x at 512 (26.1 s vs 1.33 s) with
  blocked-C pulling further ahead past cache size (512: 0.57 s; gap keeps
  widening to 2048) — the "widen, don't stay flat" blocking signal is
  there; shaped 200x32x32: python 16.3 ms vs c 1.04 ms (~16x).
- **Marshalling optimization (landed in `ops/backend_c.py`):** the per-call
  list-of-lists ctypes loop was the wall at showcase shapes (each
  200x32 matmul ~4.8 ms, ~2.7 ms of it flatten/unflatten). Now flatten via
  `array('d')`+`from_buffer`, unflatten via one `struct.unpack` of the raw
  buffer — c.matmul ~1.9 ms and the C full step went ~48 ms → ~17 ms.
  Bit-identical results re-verified (demo 250 ep: 51/51 records identical).
- **Honest per-epoch finding (doc checkpoint corrected):** at showcase
  scale the matmul-level C speedup IS an order of magnitude (~15-30x), but
  the per-epoch ratio is ~4-4.5x (python 78 ms vs c 17 ms median
  interleaved on this machine). Cause: the frozen-python `elementwise`/
  `transpose`/`add_bias` list work (~13 ms/step at n=200) plus ~0.9 ms of
  unavoidable list-of-lists marshalling per matmul call (~8 calls/step).
  Even with marshalling minimized, the frozen-interface bound cannot reach
  10x/epoch. `02_c_phase.md` now states the matmul-vs-epoch distinction
  and the corrected checkpoint instead of the unachievable ~30-80x claim.
- **Showcase animations:** `animations/showcase_python.mp4` +
  `showcase_c.mp4` rendered from `logs/showcase_*` (logs differ only in
  wall-clock fields; loss/weights/probes identical, so the two-tier parity
  story holds through the render).
- Next move: full suite re-run both environments, git status review, then
  small scoped commits (source first, then docs; artifacts are gitignored).

---

Governs the phase-2 work. Technical spec: `docs/02_c_phase.md`; project
conventions: `AGENT.md`; phase-1 record: git log (`f3a7df1` and earlier).
This file replaces the phase-1 closeout plan (that work is committed and
done).

---

## Status: phase 1 complete

All phase-1 DoD items ticked and committed; working tree clean at `f3a7df1`.
Demo-tier defaults (`[2,4,4,1]`, n=100, 250 ep, lr 2.5) and showcase-tier
flags (`--layers 2,32,32,1 --n-per-quadrant 50`) both work end-to-end.
`compare_backends.py`, `benchmark_matmul.py`, `analyze_run.py`,
`profile_run.py`, `animate.py`, `benchmark_report.md`,
`animations/backend_comparison.mp4` are live with python-only rows.

## Governing invariants (from AGENT.md + docs — do not violate)

1. **Two tiers forever:** demo tier (defaults) for correctness checks —
   golden points, loss parity `1e-6`; showcase tier (explicit flags) for
   all efficiency claims. Defaults never drift toward showcase scale.
2. **Frozen surface:** `network.py`, `train.py`, `animate.py`, the dataset
   generator, and the log schema must NOT change in phase 2/3 (except the
   `--backend` CLI option). If phase-2 work seems to require it, stop and
   flag.
3. **Only `matmul` is replaced** with C. `add_bias`, `transpose`,
   `elementwise` stay pure Python (re-exported from `backend_python`) in
   every phase — they're O(n), not the bottleneck.
4. **`double` in C** for this phase (float32 is phase 3's story). Flat
   row-major 1D arrays cross the ctypes boundary; shapes passed as ints.
5. **`matmul_naive` stays forever** as the benchmark baseline. Build with
   `-O2` only — no `-march=native`/`-mavx2`/`-ffast-math` (phase 3 hand-writes
   the SIMD; compiler auto-vectorization here would ruin that story).
   Comment this reasoning in the Makefile.
6. **Determinism is load-bearing:** same seed + config → same result.
   No unseeded randomness, no uninitialized memory reads.
7. **Run everything C from WSL2** (Ubuntu 24.04): Windows Python cannot
   `ctypes.CDLL` an ELF `.so`. Repo shared via `/mnt/c/...` — edit on
   Windows, build+test in WSL. Verified present: gcc 13.3.0, make 4.3,
   python3 3.12.3, nasm, ffmpeg, pytest 7.4.4, matplotlib. No installs
   needed.

## Phase-2 scope (from `docs/02_c_phase.md`)

Replace only `matmul` in `ops/` with a C shared library:

```
native/c/
├── matmul.c      # matmul_naive + matmul_blocked
├── matmul.h
└── Makefile      # libmatmul.so
```

### Step 1 — `native/c/`

- `matmul.h`: `void matmul_naive(const double *A, const double *B,
  double *C, int n, int k, int m);` + same contract for `matmul_blocked`.
  Caller pre-allocates C; C only writes into it.
- `matmul.c`:
  - `matmul_naive`: direct cache-hostile triple loop exactly as written
    in the phase doc (`B[p*m+j]` strides by `m` per inner iteration — by
    design, don't "fix" it).
  - `matmul_blocked`: (1) i-k-j loop reorder so the innermost loop walks
    contiguous memory in B and C, zero-initializing C first since it
    accumulates with `+=`; (2) cache tiling on top — `#define BLOCK_SIZE 64`
    (exposed as a #define for tuning), three nested block-index loops with
    the i-k-j loops inside, working set kept in L1/L2.
- `Makefile`: `CC = gcc`, `CFLAGS = -O2 -fPIC -Wall -Wextra`,
  `TARGET = libmatmul.so`, `clean` target. Comment explaining why
  `-march=native`/`-ffast-math` are withheld (see invariant 5).
- Build: `make -C native/c` (WSL) → `libmatmul.so` lands in `native/c/`.
  Must be `-Wall -Wextra` clean.

### Step 2 — `ops/backend_c.py` + selector flip

- Load `libmatmul.so` by path computed from `__file__` (never hardcoded
  absolute paths); `ctypes.CDLL` at import.
- Register `argtypes`/`restype` for both `matmul_naive` and
  `matmul_blocked`: `POINTER(c_double)` x3 + `c_int` x3, `restype = None`.
- `_flatten(A) -> (ctypes array, rows, cols)`, `_unflatten(flat, n, m)`
  helpers; caller-allocated output buffer of `(c_double * (n*m))`.
- `matmul(A, B, variant="blocked")`, variant in `{'naive', 'blocked'}`.
  Default `"blocked"` is what `train.py`/`compare_backends.py` use;
  `benchmark_matmul.py` exercises both explicitly.
- Same public signature as `backend_python.matmul(A, B)`; re-export
  `add_bias`, `transpose`, `elementwise` from `backend_python`.
- `ops/__init__.py`: `get_backend("c")` imports and returns
  `ops.backend_c` (replace the `NotImplementedError`).

### Step 3 — `tests/test_ops_c.py` (replace the 2 skip stubs)

- Golden 2x2 from phase 1 (`[[19,22],[43,50]]`) for both `naive` and
  `blocked`.
- Seeded property tests vs `backend_python.matmul` across random
  non-square shapes, **including boundary sizes** `k=1`, 65, 127,
  non-multiple-of-`BLOCK_SIZE` dims — blocking bugs hide there. Assert all
  three (python, c-naive, c-blocked) agree within `1e-9` absolute
  tolerance (not exact — summation order differs).
- Full pipeline test: `train.py --backend c --epochs 500`, same seed as
  phase-1 golden; final loss matches `--backend python` within `1e-6`;
  golden points classify correctly.
- **Cross-environment behavior (decision 2026-08-03):** module-level
  load guard — if `libmatmul.so` can't be loaded (native Windows), all C
  tests `pytest.skip` with a clear "build with make -C native/c under
  WSL2" message, keeping the whole suite green in both environments.

### Step 4 — two minor fixes (decided 2026-08-03)

- `compare_backends.py` `_golden_ok` (line ~261): `Network(...)` call is
  missing the required `seed` arg → regenerate path silently reports
  `golden ok = NO`. Fix: pass `meta["seed"]`.
- `docs/02_c_phase.md`: "re-renders ... without retraining when called
  without arguments" is wrong — the code requires `--regenerate`. Fix the
  doc sentence, not the CLI default.

### Step 5 — two-tier verification (per `02_c_phase.md`)

- **Demo tier (correctness):** `train.py --backend c` (500 ep, seed 0)
  → loss parity `1e-6` vs python, all 4 golden points correct. C looking
  equal or slower at demo scale is expected, not a bug.
- **Showcase tier (efficiency):** `compare_backends.py` picks up `c`
  automatically → `benchmark_report.md` table gains c-naive/c-blocked
  rows (epoch ms, phase split, speedup vs python, final loss, golden ok)
  and `animations/backend_comparison.mp4` gains the new lines. Measured
  on this machine: c-blocked ≈ **4-4.5x per epoch (python ~78 ms, c ~17 ms
  median interleaved)** and **~14-30x per matmul** at showcase shapes —
  the epoch figure is capped by the frozen-python O(n) ops + per-call
  list-of-lists marshalling (see status note above; the ~10x epoch target
  in the earlier draft was not reachable under the frozen interface and
  `02_c_phase.md` now reflects the honest, evidence-backed bound).
- `benchmark_matmul.py --backend c --variant naive|blocked --sizes
  16,...,2048` appends to `benchmark_results.csv`; the estimate-then-ask
  safety logic handles the new ceiling. (CLI already supports these
  flags — no edits needed.)
- `animate.py --log-dir logs/showcase_c` works unchanged, no edits.
- Everything above runs under WSL's `python3` (pytest included).

### Step 6 — docs + commit

- Update this file's tracking as work lands.
- Small scoped commits, one logical change each (code first, then
  artifacts/docs). Commit source only — `*.so`/`*.o`, logs, CSVs, mp4s
  are gitignored (reproducible scripts, not outputs).

## Definition of done for phase 2 (from `02_c_phase.md`)

- [ ] `native/c/libmatmul.so` builds cleanly via `make -C native/c`, no
      warnings under `-Wall -Wextra`.
- [ ] `train.py --backend c` produces the same converged loss (within
      `1e-6`) as `train.py --backend python` on the same seed; same 4
      golden points classify correctly.
- [ ] `tests/test_ops_c.py` passes, including boundary-size blocking
      cases (skips gracefully on Windows when the .so can't load).
- [ ] `benchmark_results.csv`/`benchmark_report.md` show naive-C beating
      naive-Python by ~1-2 orders of magnitude, and blocked-C beating
      naive-C further at larger sizes (the gap must widen as size grows
      past cache size — if not, the blocking has a bug or `BLOCK_SIZE` is
      poorly tuned for this machine's cache).
- [ ] `animate.py --log-dir logs/<c-run>` works unchanged.
- [ ] Showcase-tier comparison: c-blocked at an order of magnitude (or
      better) at the **matmul level** (shaped ~14-30x, sweep ~20x at 512,
      per-epoch ~4-4.5x bounded by the frozen-python surface + per-call
      marshalling — documented in `02_c_phase.md`, checkpoint text
      corrected to measured numbers), demo-tier loss parity (`1e-6`) and
      golden points still hold (bit-identical verified).
- [ ] Working tree committed; plan.md tracking updated.
