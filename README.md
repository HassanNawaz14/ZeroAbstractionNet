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

## Phase docs

- [Phase 1 — Pure Python](docs/01_python_phase.md)
- [Phase 2 — C Backend](docs/02_c_phase.md)
- [Phase 3 — Assembly Backend](docs/03_asm_phase.md)
