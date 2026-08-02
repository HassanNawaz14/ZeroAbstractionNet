# Phase 1 Closeout Plan — Two-Tier Benchmark/Log Infrastructure

Replaces the original phase-1 build plan (that work is done). Governs the
remaining phase-1 gaps and the `compare_backends` machinery that phases 2-3
will plug into. Details live in `docs/01_python_phase.md`,
`docs/02_c_phase.md`, `docs/03_asm_phase.md`, `README.md`, `AGENT.md` —
all already updated with the two-tier strategy below.

---

## Governing invariant: two tiers of runs

Every tool must work at both scales; defaults never change tier.

| Tier | Config | Purpose | Python epoch |
|---|---|---|---|
| **Demo (default)** | `[2,4,4,1]`, n=100 (`--n-per-quadrant 25`), 250 ep, lr 2.5 | pedagogy: boundary, golden points, 3-panel animation | ~5 ms |
| **Showcase (explicit)** | `--layers 2,32,32,1 --n-per-quadrant 50` (n=200), 250 ep, lr 2.5, `--log-every 5` | efficiency: C/asm speedups visible (compute ≫ ~1-5 µs ctypes marshalling) | ~160 ms |

Correctness checks run at demo tier; efficiency claims always cite
showcase-tier numbers.

## Status (done)

- Core MLP `[2,4,4,1]`, dataset, ops backends, tests (14 pass, 5 skipped =
  phase 2/3 stubs) — working
- `train.py` (log truncation fix), `animate.py` (fast persistent-artist
  renderer, dedup, frame budget, optional 4th phase-time panel),
  `benchmark_matmul.py` (`timed_best_of` helper), `profile_run.py`,
  `analyze_run.py`, `compare_backends.py` — working
- `profile_baseline.txt`, animations, `benchmark_results.csv`,
  `benchmark_shaped.csv`, `benchmark_report.md`,
  `animations/backend_comparison.mp4` — produced
- All 5 markdown files updated with the two-tier strategy; no stale
  "(planned)" markers remain
- `config.py` defaults aligned to the demo tier (`EPOCHS=250`, `LR=2.5`) —
  bare `python train.py` IS the demo tier, matching the two-tier docs
- Stub tests (`test_ops_c.py`, `test_ops_asm.py`) converted from failing
  `NotImplementedError` to `pytest.mark.skip` — phase-1 suite is green
- Demo-tier re-verified end-to-end this closeout: `python train.py` →
  loss 0.693→0.0118 in 250 ep, all 4 golden points correct,
  `animate.py` renders the 3-panel mp4 unchanged
- Showcase-tier artifacts verified present: `logs/showcase_python`
  (250 ep, lr 2.5, final loss 0.0167, golden ok), `benchmark_results.csv`
  (python sweep 16→512, 40.5s at 512 = ceiling), deduped
  `benchmark_shaped.csv` with `count` column, `benchmark_report.md`
  table, `animations/backend_comparison.mp4`
- Working tree committed (see git log: `553bb7f`, `41e055c`)

## Remaining work — ALL DONE (historical record; items 1-4 completed in
commits `553bb7f` and `41e055c` + this closeout)

## 1. Showcase-tier code support (small, additive)

- **`train.py`**: add `--layers 2,32,32,1` and `--probe-resolution`
  (defaults = `config.py`). `meta.json` schema unchanged.
- **`analyze_run.py`**: same two flags; dedupe shaped benchmark — identical
  shapes (e.g. two 32→32 layers) collapse to one row with a `count` column.
- **`profile_run.py`**: add `--layers` so it can profile the showcase net.
- **`animate.py`**: edge thinning — per layer, if connections > 300, draw
  only the top 300 by `|w|` and note it in the panel title. Must not
  trigger at demo scale (28 edges).
- **`benchmark_matmul.py`**: CSV schema → `backend,variant,size,seconds`
  (python → variant `naive`); file doesn't exist yet, schema change is free
  and needed for c/asm variant lines later.
- **`.gitignore`**: add `benchmark_shaped.csv` (generated artifact).

## 2. `compare_backends.py` (new — the benchmark_logs deliverable)

- Loop over backends `get_backend` can load (python now; c/asm join later
  with graceful skip until then).
- Per backend, run the showcase tier into `logs/showcase_<backend>`,
  collecting: epoch time, phase split, final loss, golden-point check,
  shaped-matmul times (reuse `timed_best_of`).
- Write **`benchmark_report.md`**: table
  `backend/variant | epoch ms | fwd/bwd/upd % | speedup vs python |
  final loss | golden ok` + shaped-matmul table.
- Render **`animations/backend_comparison.mp4`** — animated log-scale
  figure: panel A = sweep lines per backend/variant growing size-by-size
  (`benchmark_results.csv`), panel B = shaped-matmul times at showcase
  shapes, log-y (`benchmark_shaped.csv`).

## 3. Generate and verify artifacts

- `benchmark_matmul.py --backend python --sizes 16,32,64,128,256,512` →
  `benchmark_results.csv` (estimator skips >60s; 256 ≈ 5-10s = python ceiling).
- Showcase Python run → `logs/showcase_python`: verify ~160 ms/epoch, loss
  converges, golden points correct; `animate.py` render with visible
  phase-time bars (~52 records, ~1 min render).
- `analyze_run.py` showcase run → `benchmark_shaped.csv` (deduped,
  showcase shapes) + `logs/showcase_analysis/analysis.txt`.
- `compare_backends.py` → `benchmark_report.md` + comparison mp4.
- `pytest` all pass; demo-tier 3-panel path re-verified unchanged.
- Remove now-stale "(planned)" markers in the docs once the tooling lands.

## 4. Commit

Two commits (on approval): (a) docs + tier-support code, (b)
`compare_backends.py` + `benchmark_report.md`. Logs/animations/CSVs are
gitignored — reproducible scripts, not outputs, get committed.

---

# Phase 2 Kickoff — WSL2 environment & best practices (checked 2026-08-03)

## Verified WSL2 toolchain (Ubuntu 24.04, distro "Ubuntu", WSL 2.7.11)

| Tool | Status | Notes |
|---|---|---|
| `gcc` 13.3.0 | present | phase 2 build tool |
| `make` 4.3 | present | `make -C native/c` |
| `python3` 3.12.3 | present | runtime for `train.py --backend c` |
| `nasm` | **missing** | phase 3 only — not needed for phase 2 |
| `ffmpeg` | **missing** | only if rendering mp4 *inside* WSL |
| `pytest` / `matplotlib` (pip) | **missing** | needed to test/animate inside WSL |
| repo access | `/mnt/c/Users/HP/OneDrive/Desktop/ZeroAbstractionNet` | drvfs mount works |
| CPU | Intel i5-6200U, AVX2+FMA | relevant for phase 3 asm, not phase 2 |

## Recommended installs before phase 2 (all inside WSL2)

```bash
sudo apt-get update
sudo apt-get install -y nasm ffmpeg            # nasm for phase 3, ffmpeg for mp4
python3 -m pip install --user pytest matplotlib # test suite + animation in WSL
```

## Best practices for phase 2 (from 02_c_phase.md + AGENT.md)

1. **Run everything from WSL2.** Windows Python cannot `ctypes.CDLL` an
   ELF `.so`; the C backend only works under WSL's `python3`. The repo is
   shared via `/mnt/c/...`, so edit on Windows, build+test in WSL.
2. **Don't touch the frozen surface** — `network.py`, `train.py`,
   `animate.py`, dataset, log schema. Only add `native/c/*` +
   `ops/backend_c.py` and flip one line in `ops/__init__.py`.
3. **`double` in C**, argtypes registered via ctypes; flat row-major
   1D arrays cross the boundary; shapes passed as ints.
4. **Keep `matmul_naive` forever** as the baseline (rule 5); `-O2`
   only, no `-march=native`/`-mavx2`/`-ffast-math` (that's phase 3's
   story). Comment this in the Makefile.
5. **Correctness at demo tier** (loss parity `1e-6`, golden points);
   **efficiency at showcase tier** (`compare_backends.py` picks up the
   `c` backend automatically via `get_backend`). C looking "equal" at
   demo scale is expected, not a bug.
6. **Boundary-size tests are the bug-hunters**: non-multiple-of-64
   dims (`k=1`, 65, 127...) in `test_ops_c.py`.
7. `benchmark_matmul.py --backend c --variant naive|blocked --sizes
   16,...,2048` appends to `benchmark_results.csv`; the existing
   "estimate-then-ask" safety logic handles the new ceiling.
8. `.gitignore` already covers `*.so`/`*.o` — commit source
   (`.c/.h/Makefile`) only, never the built library.
9. Verify with `python3 -m pytest` in WSL before declaring done.

---

## Definition of Done (phase-1 closeout — both tiers)

- [x] `train.py --backend python` (demo tier) runs end-to-end, loss
      decreases, all 4 golden points classify correctly
- [x] Showcase-tier Python run trains/logs/animates at
      `[2,32,32,1]`/n=200 with visible per-epoch phase times
- [x] `benchmark_matmul.py --backend python` produces
      `benchmark_results.csv` (new `backend,variant,size,seconds` schema)
      with sizes up to the ~5-10s single-call ceiling
- [x] `analyze_run.py` produces deduped `benchmark_shaped.csv` with a
      `count` column at showcase shapes
- [x] `compare_backends.py` produces `benchmark_report.md` table and the
      animated log-scale `animations/backend_comparison.mp4`
      (python-only rows for now; c/asm lines appear as phases 2-3 land)
- [x] `profile_run.py` output saved as `profile_baseline.txt`; works with
      `--layers` at both tiers
- [x] `animate.py` produces working mp4/gif: 3-panel (demo) and 4-panel
      phase-time (showcase), with edge thinning active at showcase scale
- [x] All tests in `tests/` pass (14 passed, 5 skipped = phase 2/3 stubs)
- [x] No `import numpy` / `import torch` / `import tensorflow` in compute
      path
- [x] Docs (`docs/*.md`, `README.md`, `AGENT.md`) consistent with the
      implemented reality — no stale "(planned)" markers
- [x] Working tree committed (step 4)
