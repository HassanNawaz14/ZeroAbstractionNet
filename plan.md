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

- Core MLP `[2,4,4,1]`, dataset, ops backends, tests (14 pass) — working
- `train.py` (log truncation fix), `animate.py` (fast persistent-artist
  renderer, dedup, frame budget, optional 4th phase-time panel),
  `benchmark_matmul.py` (`timed_best_of` helper), `profile_run.py`,
  `analyze_run.py` — working
- `profile_baseline.txt`, animations for demo runs — produced
- All 5 markdown files updated with the two-tier strategy
- Working tree has uncommitted changes (docs + code above) — commit in step 4

## Remaining work, in order (each step testable before moving on)

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

## Definition of Done (phase-1 closeout — both tiers)

- [ ] `train.py --backend python` (demo tier) runs end-to-end, loss
      decreases, all 4 golden points classify correctly
- [ ] Showcase-tier Python run trains/logs/animates at
      `[2,32,32,1]`/n=200 with visible per-epoch phase times
- [ ] `benchmark_matmul.py --backend python` produces
      `benchmark_results.csv` (new `backend,variant,size,seconds` schema)
      with sizes up to the ~5-10s single-call ceiling
- [ ] `analyze_run.py` produces deduped `benchmark_shaped.csv` with a
      `count` column at showcase shapes
- [ ] `compare_backends.py` produces `benchmark_report.md` table and the
      animated log-scale `animations/backend_comparison.mp4`
      (python-only rows for now; c/asm lines appear as phases 2-3 land)
- [ ] `profile_run.py` output saved as `profile_baseline.txt`; works with
      `--layers` at both tiers
- [ ] `animate.py` produces working mp4/gif: 3-panel (demo) and 4-panel
      phase-time (showcase), with edge thinning active at showcase scale
- [ ] All tests in `tests/` pass
- [ ] No `import numpy` / `import torch` / `import tensorflow` in compute
      path
- [ ] Docs (`docs/*.md`, `README.md`, `AGENT.md`) consistent with the
      implemented reality — no stale "(planned)" markers
- [ ] Working tree committed (step 4)
