"""Tests for ops/backend_c.py (phase 2).

Requires native/c/libmatmul.so — build it with `make -C native/c` inside
WSL2/Linux. On environments where the shared library can't be loaded (e.g.
native Windows, or not yet built), the whole module is skipped with a build
hint instead of failing, so the suite stays green in both environments.

Per docs/02_c_phase.md, correctness is verified at the DEMO tier: C output
must match the pure-Python backend within 1e-9 absolute tolerance (not exact
equality — summation order differs), and train.py --backend c must reach the
same converged loss as --backend python within 1e-6 with all golden points
correct. Efficiency is NOT tested here — that's showcase tier,
compare_backends.py's job.
"""

import json
import os
import random
import subprocess
import sys

import pytest

from analyze_run import GOLDEN_POINTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from ops import backend_c
except OSError as exc:  # libmatmul.so missing or unloadable (e.g. Windows)
    pytest.skip(
        f"phase 2 C backend can't load libmatmul.so ({exc}); build it first "
        f"with `make -C native/c` inside WSL2/Linux",
        allow_module_level=True,
    )

from ops.backend_python import matmul as py_matmul


def _max_abs_diff(A, B):
    """Max |a - b| over two equal-shaped list-of-lists matrices."""
    return max(
        abs(a - b)
        for row_a, row_b in zip(A, B)
        for a, b in zip(row_a, row_b)
    )


# Golden 2x2 case, identical to the phase-1 test:
# A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
# A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
GOLDEN_A = [[1, 2], [3, 4]]
GOLDEN_B = [[5, 6], [7, 8]]
GOLDEN_C = [[19, 22], [43, 50]]


def test_matmul_2x2_naive():
    assert backend_c.matmul(GOLDEN_A, GOLDEN_B, variant="naive") == GOLDEN_C


def test_matmul_2x2_blocked():
    assert backend_c.matmul(GOLDEN_A, GOLDEN_B, variant="blocked") == GOLDEN_C


# Boundary sizes are exactly where blocking bugs hide: k=1 (no division into
# full blocks), dims straddling BLOCK_SIZE (64) and its multiples (65, 100,
# 127, 128, 129), and layer-count shapes the demo/showcase nets actually use.
BOUNDARY_SHAPES = [
    (1, 1, 1),        # single element
    (3, 1, 5),        # k=1
    (7, 65, 3),       # m < BLOCK_SIZE, k > BLOCK_SIZE
    (65, 63, 65),     # n, m just past the block edge
    (100, 127, 50),   # k=127 (two partial blocks)
    (127, 100, 65),   # n=127 partial last block row
    (129, 128, 129),  # everything past the block edge
    (200, 32, 32),    # showcase hidden layer
    (50, 32, 1),      # showcase output layer
    (100, 100, 100),  # backward-style k = n
]


@pytest.mark.parametrize("n,k,m", BOUNDARY_SHAPES)
def test_c_matches_python_random_shapes(n, k, m):
    rng = random.Random(12345)
    A = [[rng.uniform(-1.0, 1.0) for _ in range(k)] for _ in range(n)]
    B = [[rng.uniform(-1.0, 1.0) for _ in range(m)] for _ in range(k)]

    expected = py_matmul(A, B)
    for variant in ("naive", "blocked"):
        got = backend_c.matmul(A, B, variant=variant)
        diff = _max_abs_diff(got, expected)
        assert diff < 1e-9, (
            f"variant={variant} shape=({n},{k},{m}): max diff {diff:.2e} >= 1e-9"
        )


def test_naive_and_blocked_agree_on_network_shape():
    """Both C variants must agree with each other too (they share the C
    contract but differ in accumulation order; tolerance 1e-9)."""
    rng = random.Random(7)
    A = [[rng.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(200)]
    B = [[rng.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)]
    naive = backend_c.matmul(A, B, variant="naive")
    blocked = backend_c.matmul(A, B, variant="blocked")
    assert _max_abs_diff(naive, blocked) < 1e-9


def test_shape_mismatch_raises():
    # A is 2x2, B is 1x2 -> k=2 != len(B)=1.
    with pytest.raises(ValueError):
        backend_c.matmul([[1, 2], [3, 4]], [[5, 6]])


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        backend_c.matmul([[1.0]], [[1.0]], variant="bogus")


def _train_via_cli(backend, log_dir, epochs=500):
    """Run the real train.py CLI (demo tier, seed 0) and return the final
    logged loss. Exercises the full CLI->ops->backend plumbing."""
    subprocess.run(
        [sys.executable, "train.py", "--backend", backend,
         "--epochs", str(epochs), "--seed", "0", "--log-dir", log_dir],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    records = [json.loads(line) for line in open(os.path.join(log_dir, "epochs.jsonl"))]
    return records[-1]["loss"]


def _golden_points_ok(log_dir):
    """Classify the 4 golden points with the C run's final logged weights."""
    meta = json.load(open(os.path.join(log_dir, "meta.json")))
    records = [json.loads(line) for line in open(os.path.join(log_dir, "epochs.jsonl"))]
    from network import Network
    net = Network(meta["layer_sizes"], backend_c, meta["seed"])
    net.weights = records[-1]["weights"]
    net.biases = records[-1]["biases"]
    preds = [row[0] for row in net.forward([pt for pt, _ in GOLDEN_POINTS])[0]]
    return preds, all(
        (p > 0.5) == (label == 1.0)
        for p, (_, label) in zip(preds, GOLDEN_POINTS)
    )


def test_full_pipeline_loss_parity_and_golden(tmp_path):
    """train.py --backend c vs --backend python, 500 epochs, same seed:
    final loss within 1e-6 and the 4 golden points still classify correctly."""
    py_dir = str(tmp_path / "py")
    c_dir = str(tmp_path / "c")

    py_loss = _train_via_cli("python", py_dir)
    c_loss = _train_via_cli("c", c_dir)

    assert abs(py_loss - c_loss) < 1e-6, (
        f"final loss parity broken: python {py_loss:.10f} vs c {c_loss:.10f}"
    )

    preds, golden_ok = _golden_points_ok(c_dir)
    assert golden_ok, (
        "golden points misclassified by the c backend run: "
        f"preds={[round(p, 4) for p in preds]}"
    )