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

Everything reuses committed data (`benchmark_results.csv`,
`benchmark_shaped.csv` — tracked with the repo so the media is rerunnable
from a fresh clone) plus local `logs/showcase_*` (gitignored; only
`compare_backends.py --regenerate` needs them) — **no new benchmark runs**
needed to produce the media.

Hard constraint for this phase: nothing in frozen phases 1-3 changes —
`present/` scripts are read-only consumers of the CSVs, logs, and committed
sources (AGENT.md rule 8). The only editable files are `README.md` (item 3)
and this tracker.

## Venue-mapped deliverables

| # | Asset | New script | Inputs | Output | Where it lands |
|---|---|---|---|---|---|
| 1 | **"Same 5 seconds" budget reel** — progress bar trickles for python, reaches n=2048 for asm | `present/animate_budget.py` | `benchmark_results.csv`, `benchmark_shaped.csv` | 1080×1080 mp4, ≤20s, caption burned in | LinkedIn primary + README embed |
| 2 | **Stack drill-down** — same 8×8 matmul in Python → ctypes → `matmul.c` → `matmul.asm` (register-lane highlight) | `present/animate_stack.py` | static panels + code text; no new benchmarks | 16:9 mp4 ~25s | README primary (the "from scratch" proof), second LinkedIn clip |
| 3 | **README re-landing** — banner PNG + embeds (1)+(2), speedup tables up, plus a **"Measurement card"** (CPU/SKL-class, WSL2, single-core, best-of-3, AVX2/FMA flags) as a trust strip | `README.md` edit | existing tables + media; banner → `present/banner.png` | rendered README | GitHub |
| 4 | **Comparison recut (stretch)** — existing `backend_comparison.mp4` + endpoint "×237" annotations + 1s hold on final frame | `present/recut_comparison.py` (new script; `compare_backends.py` stays untouched per AGENT.md rule 8) | existing `animations/backend_comparison.mp4` + CSVs | 16:9 mp4 | README below drill-down |
| 5 | **Gitignore carve-out** | DONE by user — `animations/*mp4/png/gif` no longer ignored; media commits with the repo | – | – | – |

**Priority: build 1 then 2** — 1 is the LinkedIn hook, 2 is the README
differentiator; 3-5 are mechanics bundled with whichever lands first.

## Numbers to bake into captions (already in `RESULTS.md` — keep exact)

- Same ~5s wall-clock budget: python fits **n=256** (2.64 s) → asm-blocked
  fits **n=2048** (4.96 s) = **8× the matrix size** (RESULTS.md/README
  wording — the honest math is 8× the dimension, hence 512× the flops:
  caption line "8× THE MATRIX SIZE · 512× THE MATH", not "64×").
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
- 16:9 primary (~30 s) → `present/stack_drilldown.mp4`; optional 1080×1080
  crop → `present/stack_drilldown_square.mp4`.

## Verification rule (I can't see frames — so nothing ships unverified)

Any new mp4/png is validated by **pixel scan**: encode a frame to raw RGB
(via `ffmpeg` rawvideo decode) and assert the expected palette colours
actually span the expected regions. The helper is written fresh as
`present/verify_frames.py`, shared by both new scripts — note there is NO
existing pixel-scan code in the repo to reuse (the historical
`backend_comparison.png` fix in commit 3f0a624 was a dedicated-figure
rewrite, not a scan). An animation that silently shows an empty figure is
the worst possible README embed.

## Definition of done

- [x] `present/animate_budget.py` → `present/budget_reel.mp4` and
      `present/animate_stack.py` → `present/stack_drilldown.mp4` committed,
      rendered, pixel-verified via `present/verify_frames.py`.
      (reel done 2026-08-04: 17 s, 1080×1080, 8 fps, 91 KB; verified by
      `python present/verify_frames.py` — canvas, caption band, 3 lanes
      fill+motion+ratio, budget line, headline text.)
      (drill-down done 2026-08-04: 30 s, 1920×1080, 24 fps, 1.75 MB;
      verified by `python present/verify_frames.py --spec stack` — canvas,
      intro title, per-phase code highlights with no cross-phase bleed,
      C-grid fills, breadcrumb chip, outro stack. Parallel frame renderer
      (`--frames/--fps/--workers`) keeps wall-clock bounded.)
- [ ] (Stretch) `present/recut_comparison.py` re-encodes the existing
      `backend_comparison.mp4` with endpoint "×237" annotation + 1 s hold;
      `compare_backends.py` untouched.
- [x] `README.md` embeds banner (`present/banner.png`) + drill-down +
      budget reel (mp4 HTML blocks), links `RESULTS.md`; speedup table kept
      above the fold.
      (done 2026-08-04, commit 4d90b75: full re-landing — banner, both
      films, stills, 3 showcase mp4s, backend_comparison mp4+png, 3 GIFs,
      dataset, two-tier strategy, full sweep/shaped/showcase tables,
      measurement card, usage + layout + testing sections, doc links.)
      (redesigned 2026-08-04 per review, commit 7e2fd1b: ASCII figlet
      header replaces the PNG banner; every film gets its own section with
      top-level `<video>` embeds (HTML-table video embeds were not
      rendering); data shown via varied media — tables, `<details>`,
      ASCII bars/diagrams (net, matrix, quadrants, budget lanes, profile);
      AI/DL motifs in the design.)
- [x] "Measurement card" (CPU/SKL-class, WSL2, single-core, best-of-3,
      `gcc -O2`, NASM + `gcc -shared`) present wherever speedups are quoted in
      hand-written docs (canonical copy in `README.md`; `benchmark_report.md`
      is generator output and deliberately keeps no card — see AGENT.md
      rule 8). (done 2026-08-04: canonical card sits directly under the
      numbers section in README.md; RESULTS.md narrative lives under it.)
- [x] `animations/*` + `present/*` media committed (gitignore carve-out
      already done); benchmark CSVs (`benchmark_results.csv`,
      `benchmark_shaped.csv`) committed with the repo; working tree clean;
      `plan.md` tracking updated.