# Phase 1 — Pure Python (No NumPy) ANN Core

## Project name
`ZeroAbstractionNet`

## Big picture
This is phase 1 of a 3-phase project (Python → C → x86-64 assembly). We are
building a from-scratch feedforward neural network (MLP) with **no ML
frameworks and no NumPy in the compute path**. The goal of this phase is a
correct, deterministic, fully-instrumented pure-Python implementation that
later phases will accelerate by swapping out only the matrix-multiplication
backend — nothing else in the codebase should need to change between phases.

Read this whole file before writing code. Phases 2 and 3 depend on the
interfaces defined here being stable.

## Non-goals (do not do these, even if they seem like natural improvements)
- No NumPy, PyTorch, TensorFlow, JAX, or any array/autodiff library anywhere
  in the training/inference path. `pandas`/`numpy` may be used later purely
  for post-hoc analysis or plotting *inputs*, never for matmul/dot/gradient
  computation.
- No Jupyter notebooks. Everything is plain `.py` scripts runnable from a
  terminal.
- No convolutional/recurrent/attention layers. Dense (fully-connected)
  layers only.
- No mini-batch shuffling/stochasticity by default — training must be
  reproducible run-to-run given the same config (see "Determinism").
- No premature optimization of the pure-Python code (no clever vectorized
  tricks, no `array` module tricks to fake speed). It should look like
  straightforward nested-loop Python. The whole point of later phases is to
  contrast this baseline against C and asm.

## Directory structure (create this now; later phases add files, not rename)
```
cpu-native-ann/
├── config.py                  # all hyperparams, seeds, dataset params in one place
├── data/
│   └── generate_data.py       # deterministic XOR-quadrant dataset generator
├── ops/
│   ├── __init__.py            # backend selector (get_backend(name))
│   ├── backend_python.py      # pure-python matmul/dot — THIS PHASE
│   ├── backend_c.py           # phase 2 (create empty stub now, raise NotImplementedError)
│   ├── backend_asm.py         # phase 3 (create empty stub now, raise NotImplementedError)
│   └── activations.py         # sigmoid/tanh/relu + derivatives (pure python, all phases)
├── native/
│   ├── c/                     # phase 2 will populate
│   └── asm/                   # phase 3 will populate
├── network.py                 # Network class: forward/backward, uses ops backend
├── train.py                   # training loop CLI
├── benchmark_matmul.py         # standalone matmul-only scaling benchmark, all backends
├── profile_run.py              # cProfile harness for the training loop
├── animate.py                  # reads logs/, renders animations — backend-agnostic
├── logs/                       # jsonl run logs (create .gitkeep)
├── tests/
│   ├── test_ops_python.py
│   ├── test_ops_c.py           # phase 2 (stub now)
│   ├── test_ops_asm.py         # phase 3 (stub now)
│   └── test_forward_backward.py
└── README.md
```

## Determinism requirement
Every run with the same `config.py` must produce bit-identical loss curves
and weights (within phase 1/2's double-precision arithmetic). This matters
because phase 2 and 3 correctness tests will diff their outputs against this
phase's outputs. Achieve this by:
- Using Python's stdlib `random.Random(seed)` (a local instance, not the
  global `random` module state) for weight init and any data generation
  noise.
- Full-batch gradient descent only (no minibatch shuffling) in the default
  config.
- All floats are Python `float` (i.e. IEEE-754 double).

## Dataset spec — "XOR quadrants"
Deterministic, no external files, no randomness required to generate the
core set (a `noise_std` knob exists but defaults to `0.0`).

**Rule:** a point `(x, y)` with `x, y ∈ [-1, 1]` has label
`1` if `x * y > 0` (i.e. quadrants 1 and 3), else `0` (quadrants 2 and 4).
Points exactly on an axis (`x == 0` or `y == 0`) are excluded from
generation.

**Generator contract** — `data/generate_data.py`:
```python
def generate_dataset(n_per_quadrant: int, seed: int, noise_std: float = 0.0) -> tuple[list[list[float]], list[float]]:
    """
    Returns (X, y):
      X: list of [x, y] pairs, len == 4 * n_per_quadrant
      y: list of labels (0.0 or 1.0), same length, same order as X
    Points are drawn on a jittered grid within each quadrant (not pure
    random scatter) so the dataset is visually even and reproducible.
    If noise_std > 0, add Gaussian noise via the seeded RNG (may flip a
    point's true quadrant near the axes — acceptable, mirrors real label
    noise, but keep default 0.0 for the core demo).
    """
```

**Golden test points** (hardcode these in `tests/test_forward_backward.py`
and reuse in `data/generate_data.py` docstring for reference — these are
sanity-checkable by eye and must always classify correctly once trained):
```
(0.5, 0.5)   -> 1
(-0.5, -0.5) -> 1
(0.5, -0.5)  -> 0
(-0.5, 0.5)  -> 0
```

**Probe grid**: `generate_data.py` must also expose
`generate_probe_grid(resolution: int) -> list[list[float]]`, a uniform
`resolution x resolution` grid over `[-1, 1] x [-1, 1]`, used only for
rendering the decision-boundary heatmap during animation (not for
training). Default `resolution=40` (1600 points — cheap to run forward-pass
inference on every log step).

## Network spec
- Architecture: configurable list of layer sizes, default `[2, 4, 1]`
  (2 inputs, 1 hidden layer of 4 neurons, 1 output).
- Hidden activation: `tanh`. Output activation: `sigmoid`. Loss: binary
  cross-entropy.
- Weight init: for each layer, sample uniform in
  `[-1/sqrt(fan_in), 1/sqrt(fan_in)]` using the seeded `random.Random`
  instance from `config.py`. Biases init to `0.0`.
- Optimizer: plain full-batch gradient descent, fixed learning rate
  (config default `lr=0.5`, since the dataset/network are tiny this
  converges in a few hundred epochs — tune if needed but keep it simple,
  no momentum/Adam).

`network.py` contract:
```python
class Network:
    def __init__(self, layer_sizes: list[int], backend, seed: int): ...
    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        """Returns (predictions, cache) where cache holds every layer's
        pre-activation and post-activation values, needed for backward()
        and for animation logging."""
    def backward(self, X, y_true, cache) -> dict:
        """Returns gradients per layer (weights, biases)."""
    def update(self, grads, lr: float) -> None: ...
    def get_state(self) -> dict:
        """Returns JSON-serializable snapshot: weights, biases per layer.
        Used by the logger."""
```
`Network` must call **only** `ops` backend functions for matmul/dot — never
raw `for` loops over matrix elements inside `network.py` itself. This keeps
the backend swap (phase 2/3) confined entirely to the `ops/` package.

## `ops` backend interface (THE contract — must be identical across all 3 phases)
`ops/backend_python.py` must implement, using nothing but Python lists,
loops, and stdlib `math`:
```python
def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """A is n x k, B is k x m, returns n x m. Plain triple-nested loop."""

def add_bias(A: list[list[float]], b: list[float]) -> list[list[float]]:
    """Adds bias vector b to every row of A."""

def transpose(A: list[list[float]]) -> list[list[float]]: ...

def elementwise(A: list[list[float]], fn) -> list[list[float]]:
    """Applies fn to every element, returns new matrix."""
```
`ops/activations.py` (pure python, shared by all phases — this is NOT part
of the matmul-swap, it's cheap O(n) elementwise math and stays pure Python
forever):
```python
def sigmoid(x: float) -> float: ...
def sigmoid_prime_from_output(s: float) -> float: ...  # s = sigmoid(x) already computed
def tanh_prime_from_output(t: float) -> float: ...      # t = tanh(x) already computed
```
`ops/__init__.py`:
```python
def get_backend(name: str):
    """name in {'python', 'c', 'asm'}. Imports and returns the module.
    Phase 1: only 'python' is implemented; 'c' and 'asm' raise
    NotImplementedError with a clear message, so train.py --backend c
    fails loudly and obviously until phase 2 lands."""
```

## Training loop — `train.py`
CLI (use `argparse`):
```
python train.py --backend python --epochs 500 --lr 0.5 --n-per-quadrant 25 \
                 --seed 0 --log-every 5 --log-dir logs/run_001
```
Loop structure:
1. Load config, generate dataset + probe grid (seeded).
2. Build `Network`.
3. For each epoch: forward → loss → backward → update.
4. Every `--log-every` epochs (and always on epoch 0 and the final epoch),
   write a log record (see schema below).
5. Print epoch/loss to stdout every `--log-every` epochs too, for a quick
   terminal sanity check.

## Two-tier run strategy — demo vs showcase

All runs happen at one of two deliberate scales. Every tool built in this
phase (`train.py`, `benchmark_matmul.py`, `profile_run.py`, `animate.py`,
`analyze_run.py`) must keep working at both, and no default may drift
toward the bigger one — the distinction is a project-wide invariant, not a
phase-1 nicety.

**Demo tier (default, pedagogy).** Architecture `[2, 4, 4, 1]` (the
`config.py` default), 100 points (`--n-per-quadrant 25`),
`--epochs 250 --lr 2.5`. This is what the classic deliverables are about:
a decision boundary readable at a glance, the four golden points, the loss
curve, and the 3-panel animation. All correctness checks (golden-point
classification, cross-backend loss parity) are always done here.

**Showcase tier (explicit, efficiency).** `--layers 2,32,32,1
--n-per-quadrant 50` (200 points), `--epochs 250 --lr 2.5
--log-every 5`. This is the tier where backend speedups are measured and
displayed: a pure-Python epoch is ~160 ms here (vs ~5 ms at demo scale),
so the ~1-5 µs ctypes marshalling overhead of a native backend call is
negligible and the real compute speedup becomes visible in per-epoch
phase times, in `analyze_run.py`'s shaped-matmul table, and in
`compare_backends.py`'s comparison table and animated log-scale plot.

```bash
python train.py --layers 2,32,32,1 --n-per-quadrant 50 --epochs 250 \
                 --lr 2.5 --log-every 5 --log-dir logs/showcase_python
python animate.py --log-dir logs/showcase_python --out animations/showcase_python.mp4
```

Notes:
- Never change `config.py` defaults to showcase scale; showcase runs are
  explicit CLI flags only.
- Probe grid: the 1600-point default probe costs ~0.8 s per logged epoch
  in pure Python at showcase scale — keep `--log-every` at 5 (or higher)
  for long Python showcase runs, or reduce `--probe-resolution`.
- At showcase scale `animate.py` draws only the top ~300 weights per
  layer by magnitude so the network diagram stays readable and frames
  stay fast.

## Log schema (this is what `animate.py` consumes — keep it stable)
One file per run: `logs/run_XXX/epochs.jsonl` (JSON Lines, one JSON object
per logged epoch). Also write `logs/run_XXX/meta.json` once at start with
static info.

`meta.json`:
```json
{
  "backend": "python",
  "layer_sizes": [2, 4, 1],
  "lr": 0.5,
  "seed": 0,
  "n_per_quadrant": 25,
  "dataset_points": [[0.12, 0.87], ...],
  "dataset_labels": [1.0, 0.0, ...],
  "probe_grid": [[-1.0, -1.0], ...],
  "probe_grid_resolution": 40
}
```

Each line of `epochs.jsonl`:
```json
{
  "epoch": 15,
  "loss": 0.3821,
  "weights": [[[...]], [[...]]],
  "biases": [[...], [...]],
  "dataset_predictions": [0.91, 0.04, ...],
  "probe_predictions": [0.12, 0.88, ...],
  "wall_time_sec": 0.0041
}
```
Note: `probe_predictions` is the network's output for every point in the
fixed `probe_grid` from `meta.json` — this is what becomes the animated
decision-boundary heatmap. `weights`/`biases` are the full parameter
snapshot — this drives the animated network diagram.

## Benchmark script — `benchmark_matmul.py`
Separate from the tiny demo network. This answers "how big a matmul can my
CPU handle in reasonable time" independent of the XOR demo.
```
python benchmark_matmul.py --backend python --sizes 16,32,64,128,256,512 --repeats 3
```
For each size `n` in `--sizes`, generate two random `n x n` matrices
(seeded), time `ops.matmul(A, B)` (best-of-`--repeats`, using
`time.perf_counter`), and append a row to
`benchmark_results.csv`: `backend,size,seconds`. This CSV is the artifact
phases 2 and 3 will append to, producing the final comparison table.
Print a warning and skip sizes where a single call exceeds ~30s (avoid
hanging on huge pure-Python matmuls — 512x512 in pure Python may already be
close to that ceiling on this hardware; the script should time a small size
first and extrapolate before attempting large ones, printing an estimated
time and asking for `--force` to proceed if the estimate exceeds 60s).

## Profiling script — `profile_run.py`
```
python profile_run.py --epochs 50 --backend python
```
Runs the training loop under `cProfile`, prints the top 15 functions by
cumulative time (`pstats.Stats(...).sort_stats('cumulative').print_stats(15)`),
and separately prints wall-clock time split into forward / backward /
update using manual `time.perf_counter()` calls around each phase in the
loop (don't rely on cProfile alone for this breakdown — add explicit
timers). This confirms matmul dominates before we bother writing C/asm for
it — save this output, it's the evidence for the blog post.

## Animation — `animate.py`
This is explicitly one of the most important deliverables. Style reference:
Patrick Winston's MIT AI course chalkboard-style network diagrams — clear,
labeled, one idea per frame, not flashy.
```
python animate.py --log-dir logs/run_001 --out animations/run_001.mp4
```
Reads `meta.json` + `epochs.jsonl`, builds a `matplotlib.animation.FuncAnimation`
with **three synchronized panels** in one figure, one frame per logged
epoch:
1. **Left**: network diagram — circles for neurons positioned by layer,
   lines for weights colored by sign (e.g. blue positive / red negative)
   and line width or alpha scaled by `|weight|`. Bias shown as a small
   label or separate node per neuron.
2. **Center**: decision boundary — `probe_predictions` reshaped to
   `resolution x resolution` and rendered with `imshow`/`contourf` (a
   diverging colormap centered at 0.5), with the actual dataset points
   scattered on top colored by true label (so you can see misclassified
   points visually as training progresses).
3. **Right**: loss curve so far (line plot, x-axis = epoch, growing each
   frame).

Title of each frame shows `epoch` and `loss`. Save as mp4 (`ffmpeg` writer)
or gif fallback if `ffmpeg` isn't available — detect and warn, don't crash.
This script must be **backend-agnostic**: it never touches `ops/`, only
reads logs, so it works unchanged for phase 2 and 3 runs.

## Testing — `tests/test_ops_python.py`
Hand-verify a tiny fixed case in a comment, then assert against it:
```python
# A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
# A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
def test_matmul_2x2():
    assert matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]) == [[19, 22], [43, 50]]
```
Also test non-square shapes (e.g. 2x3 times 3x1) and a shape-mismatch
`ValueError`.

`tests/test_forward_backward.py`: build a `Network([2, 2, 1])` with
hardcoded (not random) weights, hand-compute the expected forward pass
output for one input on paper (put the arithmetic in a comment), assert the
code matches. Then run a few hundred epochs on the golden XOR points from
the dataset spec above and assert all four end up on the correct side of
0.5.

## Definition of done for phase 1
- [ ] `train.py --backend python` runs end-to-end, loss decreases, all 4
      golden points classify correctly after training.
- [ ] `benchmark_matmul.py --backend python` produces
      `benchmark_results.csv` with at least sizes up to where a single
      matmul takes ~5-10s (find this empirically — that's your "ceiling"
      data point for the pure-Python baseline).
- [ ] `profile_run.py` output clearly shows `matmul` (or the function that
      calls it) as the dominant cost — save this output as
      `profile_baseline.txt`, it's referenced in phase 2's motivation.
- [ ] `animate.py` produces a working mp4/gif with all three panels
      synced and readable.
- [ ] Both tiers work end-to-end: demo-tier defaults untouched, and a
      showcase-tier Python run (`--layers 2,32,32,1 --n-per-quadrant 50
      --epochs 250 --lr 2.5`) trains, logs, and animates correctly.
- [ ] All tests in `tests/` pass.
- [ ] No `import numpy` / `import torch` / `import tensorflow` anywhere
      except optionally inside a clearly separate, optional analysis
      script that is not imported by anything else.
