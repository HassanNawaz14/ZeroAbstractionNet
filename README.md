# ZEROABSTRACTIONNET

A from-scratch feedforward neural network trained on a tiny XOR-quadrant
dataset, with matrix-multiplication backend progressively replaced from
pure Python → C → hand-written x86-64 AVX2/FMA assembly to empirically
measure each stage's speedup.

## Dataset — XOR quadrants

The dataset is a synthetic 2D binary classification problem with 4 golden
test points that are trivial to verify by eye:

| Point | Label | Quadrant |
|-------|-------|----------|
| ( 0.5,  0.5) | 1 | I   |
| (-0.5, -0.5) | 1 | III |
| ( 0.5, -0.5) | 0 | II  |
| (-0.5,  0.5) | 0 | IV  |

**Rule:** label = 1 iff `x * y > 0` (quadrants I & III), else 0 (II & IV).
Points on the axes (`x == 0` or `y == 0`) are never generated.

Key properties:
- **Deterministic** — seeded RNG, same seed = same dataset every time.
- **Balanced** — exactly `n_per_quadrant` points per quadrant (default 25 → 100 total, 50 per class).
- **Jittered grid layout** — points are placed on a uniform grid within each quadrant with small random jitter, not pure scatter. This ensures even visual coverage of the input space for the animation heatmap.
- **Noise knob** — `noise_std` (default 0.0) adds Gaussian jitter that can flip labels near axes, useful for testing robustness. Core demo always uses 0.0.
- **Probe grid** — a separate uniform `resolution × resolution` grid over [-1, 1]² (default 40 → 1600 points) used only for rendering the decision-boundary heatmap during animation, never for training.

The generator lives in `data/generate_data.py` — this is the single source of
all dataset loading for every phase.

## Two tiers of runs — demo vs showcase

Every training/inference/benchmark/visualization tool in this repo must
keep working at **both** of these scales. The distinction is load-bearing
and applies to every phase (Python, C, assembly):

| | Demo tier (default) | Showcase tier (efficiency) |
|---|---|---|
| Architecture | `[2, 4, 4, 1]` (config default) | `--layers 2,32,32,1` |
| Dataset | 100 points (`--n-per-quadrant 25`) | 200 points (`--n-per-quadrant 50`) |
| Run | `--epochs 250 --lr 2.5` | same flags |
| Purpose | pedagogy — readable decision boundary, golden points, loss curve, 3-panel animation | making the C/asm speedups visible and measurable |
| Typical epoch (pure Python) | ~5 ms | ~160 ms |

**Why two tiers:** a C/asm backend call carries ~1-5 µs of ctypes
marshalling overhead. At demo scale an entire epoch is only ~2-10 ms, so
a native backend looks equal (or slower) there and the project's headline
feature — the measured speedup of C and assembly — would be invisible.
At showcase scale compute dwarfs marshalling and the 30-80× speedup
becomes real, measurable, and visible: per-epoch phase times in the
animation's 4th panel, `analyze_run.py`'s shaped-matmul table,
`compare_backends.py`'s comparison table, and the animated log-scale
benchmark plot.

**Rules:**
- Never change `config.py` defaults to showcase scale — the tiny demo is
  the pedagogy artifact. Showcase runs are always explicit flags.
- Correctness checks (golden points, cross-backend loss parity) run at
  demo tier; efficiency checks (speedups, benchmark tables) run at
  showcase tier.
- At showcase scale `animate.py` draws only the top ~300 weights per
  layer by magnitude, keeping the network diagram readable and the
  animation fast.

Showcase example:

    python train.py --layers 2,32,32,1 --n-per-quadrant 50 --epochs 250 --lr 2.5 --log-every 5 --log-dir logs/showcase_python
    python animate.py --log-dir logs/showcase_python --out animations/showcase_python.mp4

The cross-backend comparison machinery — `compare_backends.py`
(per-backend showcase runs, comparison table, animated log-scale plot
built from `benchmark_results.csv` + `benchmark_shaped.csv`) — and
`benchmark_report.md` are the efficiency deliverables of the whole project.

## Phase docs

- [Phase 1 — Pure Python](docs/01_python_phase.md)
- [Phase 2 — C Backend](docs/02_c_phase.md)
- [Phase 3 — Assembly Backend](docs/03_asm_phase.md)
