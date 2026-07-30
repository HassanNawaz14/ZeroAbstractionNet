# ZEROABSTRACTIONNET

A from-scratch feedforward neural network trained on a tiny XOR-quadrant
dataset, with matrix-multiplication backend progressively replaced from
pure Python → C → hand-written x86-64 AVX2/FMA assembly to empirically
measure each stage's speedup.

## Phase docs

- [Phase 1 — Pure Python](docs/01_python_phase.md)
- [Phase 2 — C Backend](docs/02_c_phase.md)
- [Phase 3 — Assembly Backend](docs/03_asm_phase.md)
