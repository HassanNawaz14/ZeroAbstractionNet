# Presentation & Online Deliverables — ZeroAbstractionNet (working plan, 2026-08-03)

## Context

Phase 1-3 engineering is DONE (`RESULTS.md` closed out; 94 tests green).
This plan covers the *online presentation* of the project — defeating the
two natural failure modes of a "language speed comparison" post:

- **The claim is unverifiable** → we embed honest, rerunnable measurement
  (machine card, best-of-3, reproducibility) and *visually* show the
  assembly (not a "trust me" AVX2 sticker).
- **The assets are the wrong shape for the venue** → GitHub README needs
  committed media; LinkedIn rewards <30s **square/portrait** mp4 with
  caption text burned into frames, not 16:9 landscape.

Everything reuses committed CSVs/logs (`benchmark_results.csv`,
`benchmark_shaped.csv`, `logs/showcase_*`) — **no new benchmark runs**
needed to produce the media.

## Venue-mapped deliverables

| # | Asset | New script | Inputs | Output | Where it lands |
|---|---|---|---|---|---|
| 1 | **"Same 5 seconds" budget reel** — progress bar trickles for python, reaches n=2048 for asm | `present/animate_budget.py` | `benchmark_results.csv`, `benchmark_shaped.csv` | 1080×1080 mp4, ≤20s, caption burned in | LinkedIn primary + README embed |
| 2 | **Stack drill-down** — same 8×8 matmul in Python → ctypes → `matmul.c` → `matmul.asm` (register-lane highlight) | `present/animate_stack.py` | static panels + code text; no new benchmarks | 16:9 mp4 ~25s | README primary (the "from scratch" proof), second LinkedIn clip |
| 3 | **README re-landing** — banner PNG + embeds (1)+(2), speedup tables up, plus a **"Measurement card"** (CPU/SKL-class, WSL2, single-core, best-of-3, AVX2/FMA flags) as a trust strip | `README.md` edit | existing tables + media | rendered README | GitHub |
| 4 | **Comparison recut (stretch)** — existing `backend_comparison.mp4` + endpoint "×237" annotations + 1s hold on final frame | `compare_backends.py` | existing CSVs | 16:9 mp4 | README below drill-down |
| 5 | **Gitignore carve-out** | DONE by user — `animations/*mp4/png/gif` no longer ignored; media commits with the repo | – | – | – |

**Priority: build 1 then 2** — 1 is the LinkedIn hook, 2 is the README
differentiator; 3-5 are mechanics bundled with whichever lands first.

## Numbers to bake into captions (already in `RESULTS.md` — keep exact)

- Same ~5s wall-clock budget: python fits **n=256** (2.64 s) → asm-blocked
  fits **n=2048** (4.96 s) = **8× the matrix, 64× the math, same budget**.
- Speedup vs python-naive at n=512: c-naive **20×**, c-blocked **46×**,
  asm-blocked **~237×**.
- asm-blocked vs c-blocked at n=1024: **4.5×** (the float32 same-again win).
- Honest costs stay visible: asm final loss 0.018767 vs 0.016710 (float32
  downgrade) — the drill-down and cards must not overclaim.

## Build 1 — `present/animate_budget.py`

- Read `benchmark_results.csv`; centre on the "fixed 5 s budget" story.
- One progress-bar lane per backend (`python-naive`, `c-blocked`,
  `asm-blocked`); x = wall-clock seconds; a marker lands when each backend
  completes its largest-n `n×n` matmul; on-screen size chips at
  n=256 / 1024 / 2048.
- Square figure (LinkedIn), caption band at the bottom third, sub-20s.
- Output: `present/budget_reel.mp4`; est. 60-90 s of matplotlib at ~120
  frames.

## Build 2 — `present/animate_stack.py`

- Four "levels" of the SAME 8×8 product: Python (3 highlighted
  for-loops) → C (pointer inner loop) → ctypes call under it → asm
  (blocked FMA kernel, 8 lanes). Annotate by *loading the real committed
  source* of `native/asm/matmul.asm` and highlighting the executing lines —
  the animation literally walks the repo (that is the "no libraries" proof).
- 16:9 primary (~30 s) + optional 1080×1080 crop.

## Verification rule (I can't see frames — so nothing ships unverified)

Any new mp4/png is validated by **pixel scan**: encode a frame to raw RGB
and assert the expected palette colours actually span the expected regions
(the same trick used to fix `backend_comparison.png`). An animation that
silently shows an empty figure is the worst possible README embed.

## Definition of done

- [ ] `present/animate_budget.py` + `present/animate_stack.py` committed,
      rendered, pixel-verified.
- [ ] `README.md` embeds banner PNG + drill-down + budget reel (mp4 HTML
      blocks), links `RESULTS.md`; speedup table kept above the fold.
- [ ] "Measurement card" (CPU/SKL-class, WSL2, single-core, best-of-3,
      `gcc -O2 -mavx2 -mfma`, NASM) present wherever speedups are quoted.
- [ ] `animations/*` media committed (gitignore carve-out already done);
      working tree clean; `plan.md` tracking updated.