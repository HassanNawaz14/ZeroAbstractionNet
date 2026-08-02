# Phase 3 Plan — x86-64 Assembly Matmul Backend (working plan, 2026-08-03)

## Session status (updated 2026-08-03)

- **Phase 2 is CLOSED and accepted as-is.** Last commit `0370804`; working
  tree clean. All phase-2 DoD items ticked (clean `libmatmul.so` build,
  demo-tier loss parity bit-identical, `test_ops_c.py` green in both
  environments — WSL `30 passed / 3 skipped`, Windows `14 passed / 4
  skipped`, sweep shows the blocked-gap widening, `animate.py` works
  unchanged, workspace committed).
- **The one corrected checkpoint is now a recorded decision:** the original
  phase-2 DoD wanted ≥10× *per epoch* at showcase; measurement gave
  **~4-4.5×/epoch** (python ~78 ms vs c ~17 ms median) while the **matmul
  level** hit the order of magnitude (~16× at `200x32x32`, ~20× at 512 in
  the sweep). The per-epoch bound is the frozen-interface ceiling — pure
  Python `elementwise`/`transpose`/`add_bias` (~13 ms/step at n=200) plus
  ~0.9 ms of list-of-lists marshalling per matmul call (~8 calls/step). We
  are NOT re-opening `network.py`/`train.py`/`animate.py` or the Python ops
  to chase that number.
- **Framing for phase 3: C was never the point; it is the fair baseline.**
  ASM is where the real win shows. Expectation, stated honestly in the
  phase-3 DoD below: order-of-magnitude (or better) at the **matmul level**
  (float32 + AVX2/FMA, ~30-60× over c-blocked at 512-2048), and only a
  modest bump per-epoch at showcase (float32 marshalling is cheaper, but
  the frozen-Python surface still caps the ratio at roughly **5-6×**). We
  report both numbers, never just the flattering one.
- **Environment verified for phase 3:** nasm 2.16.01, i5-6200U (Skylake) —
  AVX2 + FMA3 present, NO AVX-512 (reject `zmm`), gcc 13.3.0, make
  4.3, ffmpeg, matplotlib installed in WSL. `RESULTS.md` does not exist yet
  (this phase creates it). `profile_baseline.txt` (phase-1 motivating
  baseline) exists.

---

## Governing invariants (from AGENT.md + docs — do not violate)

1. **Two tiers forever:** demo tier (defaults) for correctness — golden
   points, loss parity tolerances; showcase tier (explicit flags
   `--layers 2,32,32,1 --n-per-quadrant 50 --epochs 250 --lr 2.5
   --log-every 5`) for all efficiency claims. Defaults never drift toward
   showcase scale, and never "optimize the demo up to showcase"
   (marshalling > compute at demo scale is expected, not a bug).
2. **Frozen surface:** `network.py`, `train.py`, `animate.py`, the dataset
   generator, and the log schema must NOT change in phase 3 except the
   `--backend` CLI option. If phase-3 work seems to require a change, stop
   and flag.
3. **Only `matmul` is native per backend.** `add_bias`, `transpose`,
   `elementwise` stay pure Python, re-exported from `backend_python` in
   every phase.
4. **Phase-3 precision decision (from `03_asm_phase.md`):** switch the asm
   kernel to **float32** (YMM holds 8 f32 vs 4 f64 — the whole point of
   hand-writing asm). Consequence: correctness tests vs phases 1/2 use
   **relative tolerance**, and `train.py --backend asm` runs the network in
   float32 — a legitimate expected precision downgrade for this backend
   only; never silently upcast back to double. 
5. **Baselines stay forever:** `matmul_naive` (C) and the asm stages
   (`matmul_asm_scalar`, `matmul_asm_vectorized`, `matmul_asm`) are all
   permanently exported so the benchmark shows the full progression.
   C builds also stay on `-O2` (no `-march`/`-ffast-math`/holdback on
   auto-vectorization) so asm's SIMD story is distinct.
6. **Determinism is load-bearing:** same seed + config → same result within
   each phase's documented tolerance (exact phase-1, `1e-9` abs phase-2,
   relative phase-3). No unseeded randomness, no uninitialized memory reads
   (asm: zero the output buffer including boundary tiles).
7. **Run everything native from WSL2** (Ubuntu 24.04): Windows Python
   cannot `.CDLL` an ELF `.so`. Repo shared via `/mnt/c/...` — edit on
   Windows, build+test in WSL. WSL commands must avoid bash loops; run
   long things via temp scripts with `PYTHONPATH=.`.
8. **Non-goals for phase 3 (from `03_asm_phase.md` — stop here):** no
   multithreading, no AVX-512, don't try to beat system BLAS (within
   3-10× of `numpy.dot` is a strong result), no speculative prefetch or
   micro-tuning unless stage C shows a measured bottleneck pointing at one.

---

## Phase-3 scope (source of truth: `docs/03_asm_phase.md`)

Add an `asm` backend following the exact `ops`-interface pattern of
`backend_c.py`, plus three native/B implementation stages that de-risk each
other:

```
native/asm/
├── matmul.asm        # 3 exported symbols, below
└── Makefile          # libmatmul_asm.so (nasm → .o → gcc -shared)
```

### ABI contract (SysV AMD64, Linux/WSL-default of gcc)
```
void matmul_asm(const float *A, const float *B, float *C, int n, int k, int m);
A=rdi B=rsi C=rdx n=ecx k=r8d m=r9d
```
Callee-saved regs (`rbx`, `rbp`, `r12`-`r15`) — push/pop if touched. YMM
registers are caller-saved; issue `vzeroupper` before `ret`. Vector width
= 256-bit YMM = 8×`float32`. Unaligned loads via `vmovups` (ctypes buffers
aren't 32-byte aligned).

### Milestone 1 — kernels, staged (build + test each before next)

- **Stage A — scalar** `matmul_asm_scalar`: pure scalar float32 triple
  loop, no SIMD. Sole goal: validate ABI plumbing with the simplest
  instruction sequence. **This must be correct before any SIMD is written.**
- **Stage B — vectorized** `matmul_asm_vectorized`: `vmovups` +
  `vfmadd231ps` 8-wide inner accumulation, horizontal reduce
  (`vextractf128`+`vaddps`+`vhaddps` or equiv) per 8-chunk, plus scalar
  cleanup for `k % 8 != 0` (the golden `k=2` case exposes a missing
  remainder handler immediately).
- **Stage C — blocked** `matmul_asm`: stage B's SIMD inner loop inside the
  phase-2 blocked tiling (start `BLOCK_SIZE 96`, re-tune for float32 — the
  tiles are half the bytes, so it may shift). This is the default variant.

`native/asm/Makefile`: `nasm -f elf64 matmul.asm`, `gcc -shared` the `.o`
directly (function already speaks SysV, no C shim). `clean` target.

### Milestone 2 — Python wrapper + selector flip

- `ops/backend_asm.py`: same shape as `backend_c.py` but `ctype.c_float`
  for the flat arrays (`POINTER(c_float)` x3, `c_int` x3). Reuse the
  bulk-marshalling lesson from phase 2: `array('f')`+`from_buffer`
  flatten, one `struct 'f'` unpack for unflatten (half the bytes of
  double, so cheaper per crossing). `variant ∈ {scalar, vectorized,
  blocked}` firing the three symbols, default `'blocked'`. Re-export
  `add_bias`/`transpose`/`elementwise` from `backend_python` (unchanged —
  precision loss happened at the C-boundary flatten/unflatten, not in the
  python elementwise code).
- `ops/__init__.py`: `get_backend("asm")` returns the module (replace the
  `NotImplementedError`).

### Step 3 — tests (`tests/test_ops_asm.py`, replaces the 3 stubs)

- Relative-tolerance golden 2×2 (`abs(got-expected)/max(1.0,|expected|)
  < 1e-4`) for all three variants (scalar should match tightly; SIMD
  reduction reordering is why vectorized/blocked get the looser bound).
- Property tests vs `backend_python` cast to float32, across random
  shapes **including `k % 8 != 0`** (stage B remainder) and
  **non-`BLOCK_SIZE` multiples** (stage C block bounds) — the off-by-one
  bug hotspots for hand-written asm.
- Pipeline: `train.py --backend asm --epochs 500`, same seed as 1/2;
  assert final loss within ~10% of the phase-1/2 result AND all 4 golden
  points classify correctly (that's the test that matters).
- Module-level load guard so native Windows auto-skips with the
  `make -C native/asm` hint (mirror `test_ops_c.py`); suite stays green in
  both environments.

### Step 4 — benchmarks & showcase (two-tier rule)

- `benchmark_matmul.py` already accepts `--backend asm --variant
  {scalar,vectorized,blocked}` — run the sweep (start 16..2048; asm is far
  faster, extend to 4096 if the estimate-first logic permits), regenerate
  `benchmark_results.csv`.
- `compare_backends.py` picks up `asm` automatically → 6-line comparison
  table (`python, c-naive, c-blocked, asm-scalar, asm-vectorized,
  asm-blocked`) with per-epoch ms, speedup, loss, golden-ok; plot gains the
  asm lines. Report matmul-level AND per-epoch honestly.
  Regenerate `benchmark_report.md` + `backend_comparison.mp4`.
- Showcase run `logs/showcase_asm` + `animate.py` (unchanged).

### Step 5 — `RESULTS.md` (project-close writeup)

At repo root: max matrix/network size trained in fixed budget per phase,
the final speedup table (all six variants), demo-tier correctness per
backend, and `profile_baseline.txt` (phase-1 profiling evidence) motivating
the whole exercise.

### Steps/commits

Milestone 1: three commits, one per stage (kernel + its test), source
first. Then wrapper+flip, tests, benchmarks/docs, `RESULTS.md`. Commit
source only — `.so`/`.o`, logs, CSVs, mp4s are gitignored (reproducible
scripts). Small scoped commits, don't one-commit the phase.

---

## Definition of done for phase 3 (from `03_asm_phase.md`)

- [ ] `native/asm/libmatmul_asm.so` builds cleanly via `make -C native/asm`.
- [ ] All three stages correct per `tests/test_ops_asm.py`, incl.
  non-multiple-of-8 `k` and non-`BLOCK_SIZE` `n`/`m` boundary cases.
- [ ] `train.py --backend asm` converges, all 4 golden points classify
  correctly (final loss ~10% of phases 1/2 — float32 tolerance).
- [ ] Matmul-level staged progression in `benchmark_report.md`/plot:
  `python < c-naive < c-blocked < asm-scalar < asm-vectorized <
  asm-blocked`, asm-blocked fastest (order-of-magnitude or better over
  c-blocked; not required to beat external BLAS).
- [ ] `compare_backends.py` shows the staged progression at showcase ratio
  (per-epoch speedup reported honestly; ~5-6× expected, bounded by the
  frozen surface).
- [ ] `animate.py --log-dir logs/<asm-run>` works unchanged.
- [ ] `RESULTS.md` written and accurate; working tree committed; plan.md
  tracking updated.