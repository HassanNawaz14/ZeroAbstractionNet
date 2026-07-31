# Phase 1 Implementation Plan

Based on `docs/01_python_phase.md` — implement in this order, each step testable before moving on.

---

## 1. `ops/backend_python.py`
- `matmul(A, B)` — triple-nested loop, list-of-lists in/out
- `add_bias(A, b)` — broadcast bias to every row
- `transpose(A)` — standard row/column swap
- `elementwise(A, fn)` — apply `fn` to every element

## 2. `ops/activations.py`
- `sigmoid(x)`, `sigmoid_prime_from_output(s)`, `tanh_prime_from_output(t)`
- Pure `math` functions, operates on individual floats

## 3. `network.py` — `Network` class
- `__init__(layer_sizes, backend, seed)` — seeded weight init in `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`, biases to 0.0
- `forward(X)` → `(predictions, cache)` — tanh hidden, sigmoid output
- `backward(X, y_true, cache)` → grads dict — binary cross-entropy, backprop
- `update(grads, lr)` — full-batch GD step
- `get_state()` → JSON-serializable dict of weights/biases

## 4. `tests/test_ops_python.py`
- `test_matmul_2x2` — hardcoded golden case with comment
- `test_matmul_non_square` — e.g. 2×3 × 3×1
- `test_matmul_shape_mismatch` → `ValueError`

## 5. `tests/test_forward_backward.py`
- `test_forward_known_weights` — hardcoded weights, hand-computed expected output
- `test_golden_points_converge` — train a few epochs, assert all 4 golden points classify correctly

## 6. `train.py`
- `argparse` CLI: `--backend`, `--epochs`, `--lr`, `--seed`, `--n-per-quadrant`, `--log-every`, `--log-dir`
- Loop: generate dataset → build Network → forward → loss → backward → update
- Logging: `logs/run_XXX/meta.json` + `epochs.jsonl` (JSON Lines)
- Print epoch/loss every `--log-every`

## 7. `benchmark_matmul.py`
- Size sweep over `--sizes`, best-of-`--repeats` timing via `time.perf_counter`
- Estimate-and-warn before attempting large sizes (>60s)
- Append to `benchmark_results.csv`

## 8. `profile_run.py`
- cProfile harness, print top 15 by cumulative time
- Manual `time.perf_counter()` split for forward / backward / update

## 9. `animate.py`
- Read `meta.json` + `epochs.jsonl`
- 3 panels: network diagram | decision-boundary heatmap | loss curve
- mp4 output (gif fallback if ffmpeg absent)

---

## Definition of Done (from phase 1 doc)

- [ ] `train.py --backend python` runs end-to-end, loss decreases, all 4 golden points classify correctly
- [ ] `benchmark_matmul.py --backend python` produces `benchmark_results.csv` with sizes up to ~5-10s ceiling
- [ ] `profile_run.py` output saved as `profile_baseline.txt`
- [ ] `animate.py` produces working mp4/gif with 3 synced panels
- [ ] All tests in `tests/` pass
- [ ] No `import numpy` / `import torch` / `import tensorflow` in compute path
