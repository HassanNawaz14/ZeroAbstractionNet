# AGENT.md — ZEROABSTRACTIONNET

## What this project is
A 3-phase educational/portfolio project: a from-scratch feedforward neural
network trained on a tiny deterministic dataset, where the matrix
multiplication backend is progressively replaced — pure Python → C → hand-
written x86-64 AVX2/FMA assembly — to empirically measure and demonstrate
the speedup at each stage. A synchronized animation of training (network
diagram + decision boundary + loss curve) is a first-class deliverable, not
an afterthought.

The full technical spec lives in three files under `docs/` — **read the
relevant one in full before writing any code for that phase**:
- `docs/01_python_phase.md`
- `docs/02_c_phase.md`
- `docs/03_asm_phase.md`

These three files are the source of truth for interfaces, schemas, and
function signatures. If anything below conflicts with them, the phase docs
win — this file governs *process and conventions*, not the technical spec.

After phases 1-3 closed, `plan.md` is the active plan for the online
presentation deliverables (`present/` scripts, README re-landing, committed
media). The phase docs stay the technical source of truth; `plan.md` governs
what the presentation phase builds and how it is verified.

## Hard rules — do not violate these even if it seems more convenient
1. **Phases are sequential and gated.** Do not start phase 2 work until
   phase 1's "Definition of done" checklist (in `docs/01_python_phase.md`) is
   fully satisfied and confirmed with the user. Same for phase 2 → 3. If
   asked to "just get ahead" on a later phase, push back and confirm that's
   really intended before doing it.
2. **No NumPy/PyTorch/TensorFlow/JAX in the compute path, ever**, in any
   phase, for any reason — not even "just to double check my C output is
   right." Use the pure-Python phase-1 backend as the correctness oracle
   instead, per each phase doc's testing section.
3. **The `ops` backend interface is frozen once phase 1 is done.**
   `network.py`, `train.py`, `animate.py`, the dataset generator, and the
   log schema must not change during phase 2 or phase 3 except for the
   `--backend` CLI option itself. If a later phase seems to require
   changing one of these files, stop and flag it to the user — it likely
   means the phase-1 interface was under-specified, not that it's fine to
   patch around.
4. **Determinism is load-bearing.** Same seed + same config must always
   produce the same result within each phase's documented tolerance
   (exact for phase 1, `1e-9` abs for phase 2, relative tolerance for
   phase 3 — see each doc). Never introduce unseeded randomness (no bare
   `random.random()`, no relying on dict ordering across Python versions,
   no uninitialized memory reads in C/asm).
5. **Every phase's "naive"/baseline implementation stays in the codebase
   permanently** as a benchmark reference (e.g. `matmul_naive` in C,
   `matmul_asm_scalar` in asm) — never delete or "clean up" a baseline
   once a faster version exists. The whole point of the project is the
   before/after comparison.
6. **Don't add scope that isn't in the phase docs**: no multithreading, no
   AVX-512, no attempt to match/beat system BLAS, no CNN/RNN layers, no
   Jupyter notebooks, no mini-batch shuffling by default. If you think one
   of these would genuinely improve the project, say so and ask — don't
   just add it.
7. **Test before declaring a stage done.** Each phase doc has a "Definition
   of done" checklist — treat it literally as a checklist. Run the actual
   tests/benchmarks and show the output; don't assert something passes
   without having run it in this session.
8. **Presentation scripts are read-only consumers.** Everything under
   `present/` may read the committed benchmark CSVs
   (`benchmark_results.csv`, `benchmark_shaped.csv`), local `logs/`
   (gitignored), and the committed sources (`native/c/matmul.c`,
   `native/asm/matmul.asm`), but must never modify `network.py`,
   `train.py`, `animate.py`, `ops/*`, `config.py`, `compare_backends.py`,
   the dataset generator, or the native sources. matplotlib/ffmpeg are
   fine in `present/` (rendering/verification only); they never enter the
   training/inference compute path. If a presentation deliverable seems to
   require editing one of the frozen files, stop and ask — build it as a
   new script instead.

## Two-tier run strategy — ALWAYS consider both tiers before planning anything

The project deliberately runs at two scales. Before planning or executing
any training, benchmark, profile, or animation work, state which tier each
run targets; defaults never change tier.

1. **Demo tier (defaults, pedagogy):** `[2,4,4,1]`, 100 points,
   `--epochs 250 --lr 2.5`. The readable 3-panel animation, golden points,
   and loss curve. All correctness checks (golden-point classification,
   cross-backend loss parity) run here.
2. **Showcase tier (efficiency):** `--layers 2,32,32,1 --n-per-quadrant 50`
   (200 points), `--epochs 250 --lr 2.5 --log-every 5`. Where C/asm
   speedups are measured and shown. A pure-Python epoch is ~160 ms here
   vs ~5 ms at demo tier, making the ~1-5 µs ctypes marshalling overhead
   of a native backend call negligible and the 30-80× compute speedup
   real.

Why this exists: at demo tier a native backend looks equal or slower
(marshalling > compute), which would make the project's headline feature
— measured C/asm speedups — invisible. Never "optimize the defaults up to
showcase scale"; showcase runs are explicit CLI flags only. At showcase
scale `animate.py` thins the network diagram to the top ~300 weights per
layer. Efficiency claims must always cite showcase-tier numbers (via
`compare_backends.py`, `benchmark_report.md`, `benchmark_results.csv`,
`benchmark_shaped.csv` — the two CSVs are committed repo data since the
presentation phase, so the numbers are rerunnable from a fresh clone),
never demo-tier timings.

## Environment assumptions
- Linux or WSL2 (the phase-2/3 toolchain — `gcc`, `nasm`, `make`, ELF
  shared objects — assumes this; note clearly if you detect you're on
  native Windows without WSL2, don't silently try to adapt the asm/linker
  steps for MSVC/MASM).
- Toolchain needed: `python3` (3.10+), `gcc`, `nasm`, `make`, `matplotlib`,
  and `ffmpeg` (for mp4 animation export — `animate.py` should degrade to
  gif if ffmpeg isn't found, per the phase-1 doc, rather than crash).
  Presentation scripts (`present/`) additionally pipe frames through
  ffmpeg's rawvideo decoder for pixel verification; run them in the same
  WSL2 environment the benchmarks ran in.
- No cloud/notebook dependency for this project by design — everything
  should run from a local terminal. If something can't run locally, that's
  a problem to flag, not to route around by suggesting Colab.

## Workflow / how to work in this repo
- Before writing code for a phase, restate (briefly) the relevant
  interfaces/schemas you're about to implement, drawn from the phase doc,
  so a mismatch is caught before code exists rather than after.
- Prefer small, verifiable steps: implement → write/run the test for that
  piece → confirm it passes → move on. Don't write all of a phase's files
  and then test at the very end.
- When a phase doc leaves something ambiguous or up to you (it will say so
  explicitly, e.g. "agent's choice"), make a reasonable choice, state what
  you chose and why in a code comment or commit message, and move on
  rather than stopping to ask — but do ask when a choice would affect a
  *frozen interface* (rule 3 above).
- Keep commits small and scoped to one logical change (e.g. "phase1: dataset
  generator + golden point tests", not one giant commit per phase).
- After finishing a phase's Definition of Done checklist, summarize: what
  was built, what the benchmark/test numbers actually showed, and
  explicitly ask before starting the next phase.

## Code style
- Python: standard library only in the compute path (per rule 2); `black`-
  formatted; type hints on function signatures per the phase docs; no
  cleverness that obscures the nested-loop baseline nature of phase 1 code
  — it's supposed to look naive, that's the pedagogical point.
- C: `-Wall -Wextra` clean, no warnings suppressed without a comment
  explaining why.
- Asm: NASM syntax, comment every non-obvious instruction sequence
  (especially the horizontal-reduction and block-loop sections) — this
  code needs to be readable for a blog post, not just correct.

## Dataset characteristics (heuristics, not code)

- **Task:** 2D binary classification with an XOR-like decision boundary.
  Label = 1 in quadrants I & III (x*y > 0), label = 0 in II & IV (x*y < 0).
- **Input domain:** [-1, 1]², axes excluded.  Balanced per quadrant.
- **Layout:** Jittered grid (not pure random) for even visual coverage — grid
  spacing ± small seeded jitter.
- **Deterministic:** seeded `random.Random(seed)` — same seed, same data.
  The noise_std knob (default 0.0) adds Gaussian coordinate noise.
- **Golden test points:** (0.5,0.5)→1, (-0.5,-0.5)→1, (0.5,-0.5)→0, (-0.5,0.5)→0.
  These four must always classify correctly after training.
- **Size:** `n_per_quadrant * 4` points (default 25×4 = 100).  Training is
  full-batch (no minibatch shuffling) for reproducibility.
- **Probe grid:** Uniform `resolution × resolution` grid over [-1, 1]²
  (default 40×40 = 1600 points), forward-passed each log step for the
  animation heatmap.  Never used for training.

## What "done" looks like for the whole project
`RESULTS.md` (specified at the end of `docs/03_asm_phase.md`) exists and is
accurate, all three `Definition of done` checklists are checked off, and
`animate.py` produces working animations for at least one run per backend.

For the presentation phase (active now, governed by `plan.md`), "done" is
`plan.md`'s Definition of done: `present/animate_budget.py` +
`present/animate_stack.py` (and the stretch `present/recut_comparison.py`)
rendered and pixel-verified, `README.md` embedding them plus the banner,
benchmark CSVs committed, working tree clean.
