"""Tests for ops/backend_asm.py (phase 3).

Requires native/asm/libmatmul_asm.so — build it with `make -C native/asm`
inside WSL2/Linux. On environments where it can't be loaded (native Windows,
or not yet built), the whole module is skipped with a build hint so the
suite stays green in both environments.

Per docs/03_asm_phase.md: the asm kernel is float32, so correctness tests
use RELATIVE tolerance against the float64 Python oracle (precision change
+ SIMD summation order), and the pipeline test asserts final loss within
~10% plus golden-point correctness — the test that actually matters.
"""

import json
import os
import random
import struct
import subprocess
import sys

import pytest

from analyze_run import GOLDEN_POINTS

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from ops import backend_asm
except OSError as exc:  # libmatmul_asm.so missing or unloadable (Windows)
    pytest.skip(
        f"phase 3 asm backend can't load libmatmul_asm.so ({exc}); build it "
        f"first with `make -C native/asm` inside WSL2/Linux",
        allow_module_level=True,
    )

from ops.backend_python import matmul as py_matmul


def _f32(x: float) -> float:
    """Round a python float to float32 precision (no numpy in the compute
    path — this is a pure-stdlib struct round trip)."""
    return struct.unpack("=f", struct.pack("=f", x))[0]


def _f32_matrix(A):
    return [[_f32(v) for v in row] for row in A]


def _rel_diff_ok(A, B, tol=1e-4):
    """All cells within relative tolerance: |a-b| / max(1, |b|) < tol."""
    return all(
        abs(a - b) / max(1.0, abs(b)) < tol
        for row_a, row_b in zip(A, B)
        for a, b in zip(row_a, row_b)
    )


# Golden 2x2 case, identical to the phase-1/2 test: [[19,22],[43,50]].
# All values are small integers -> exactly representable in float32, so the
# asm kernels should match the golden exactly (relative check still used to
# be robust to rounding in the accumulation order).
GOLDEN_A = [[1, 2], [3, 4]]
GOLDEN_B = [[5, 6], [7, 8]]
GOLDEN_C = [[19, 22], [43, 50]]


@pytest.mark.parametrize("variant", ["scalar", "vectorized", "blocked"])
def test_matmul_2x2_golden(variant):
    assert backend_asm.matmul(GOLDEN_A, GOLDEN_B, variant=variant) == GOLDEN_C


def test_unknown_variant_raises():
    with pytest.raises(ValueError):
        backend_asm.matmul([[1.0]], [[1.0]], variant="bogus")


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        backend_asm.matmul([[1, 2], [3, 4]], [[5, 6]])


# Boundary shapes: k (the reduction dim) both a multiple and non-multiple of 8
# (stage-B responsiveness; non-mult-of-8 exercises the m%8 vector tail too),
# plus dims straddling BLOCK_SIZE=96 edges (95/96/97/191/192/193), the
# degenerate single-element case, and demo/showcase net shapes.
BOUNDARY_SHAPES = [
    (1, 1, 1),
    (2, 2, 2),
    (3, 3, 3),
    (5, 9, 5),        # k=9: non-multiple-of-8 reduction dim
    (5, 8, 5),
    (7, 1, 7),        # k=1
    (2, 2, 7),        # m<8
    (6, 16, 6),
    (9, 33, 9),       # k=33 non-mult-of-8
    (10, 7, 10),
    (4, 3, 21),       # m=21 non-mult-of-8
    (100, 100, 100),
    (100, 127, 50),
    (200, 32, 32),    # showcase hidden layer
    (50, 32, 1),      # showcase output layer
    (96, 96, 96),     # exact block
    (97, 96, 96),     # just past
    (95, 191, 193),   # just under/over block edges
    (192, 193, 191),
]


@pytest.mark.parametrize("variant", ["scalar", "vectorized", "blocked"])
@pytest.mark.parametrize("n,k,m", BOUNDARY_SHAPES)
def test_asm_matches_python_random_shapes(variant, n, k, m):
    rng = random.Random(12345)
    A = [[rng.uniform(-2.0, 2.0) for _ in range(k)] for _ in range(n)]
    B = [[rng.uniform(-2.0, 2.0) for _ in range(m)] for _ in range(k)]
    # Fair oracle: round the inputs to float32 (which is what the asm kernels
    # receive), then compare in float64 with relative tolerance.
    expected = py_matmul(_f32_matrix(A), _f32_matrix(B))
    got = backend_asm.matmul(A, B, variant=variant)
    assert _rel_diff_ok(got, expected), (
        f"variant={variant} shape=({n},{k},{m}) failed relative-tolerance check"
    )


def test_variants_agree_with_each_other():
    rng = random.Random(7)
    A = _f32_matrix([[rng.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(200)])
    B = _f32_matrix([[rng.uniform(-1.0, 1.0) for _ in range(32)] for _ in range(32)])
    scalar = backend_asm.matmul(A, B, variant="scalar")
    vectorized = backend_asm.matmul(A, B, variant="vectorized")
    blocked = backend_asm.matmul(A, B, variant="blocked")
    assert _rel_diff_ok(scalar, vectorized)
    assert _rel_diff_ok(scalar, blocked)
    # scalar has no SIMD reduction reordering: use a tighter tolerance.
    assert _rel_diff_ok(scalar, vectorized, tol=1e-5)


def _train_via_cli(backend, log_dir, epochs=500):
    """Run the real train.py CLI (demo tier, seed 0); return final loss."""
    subprocess.run(
        [sys.executable, "train.py", "--backend", backend,
         "--epochs", str(epochs), "--seed", "0", "--log-dir", log_dir],
        cwd=REPO_ROOT, check=True, capture_output=True, text=True,
    )
    records = [json.loads(line) for line in open(os.path.join(log_dir, "epochs.jsonl"))]
    return records[-1]["loss"]


def _golden_points_ok(log_dir):
    meta = json.load(open(os.path.join(log_dir, "meta.json")))
    records = [json.loads(line) for line in open(os.path.join(log_dir, "epochs.jsonl"))]
    from network import Network
    net = Network(meta["layer_sizes"], backend_asm, meta["seed"])
    net.weights = records[-1]["weights"]
    net.biases = records[-1]["biases"]
    preds = [row[0] for row in net.forward([pt for pt, _ in GOLDEN_POINTS])[0]]
    return preds, all(
        (p > 0.5) == (label == 1.0)
        for p, (_, label) in zip(preds, GOLDEN_POINTS)
    )


def test_full_pipeline_loss_ballpark_and_golden(tmp_path):
    """train.py --backend asm vs --backend python, 500 epochs, same seed:
    final loss within ~10% (float32 tolerance per the doc) and the 4 golden
    points still classify correctly -- the test that actually matters here."""
    asm_dir = str(tmp_path / "asm")
    asm_loss = _train_via_cli("asm", asm_dir)

    preds, golden_ok = _golden_points_ok(asm_dir)
    assert golden_ok, (
        "golden points misclassified by the asm backend run: "
        f"preds={[round(p, 4) for p in preds]}"
    )
    assert 0 < asm_loss < 0.020, (
        f"asm-backend final loss {asm_loss:.6f} outside the expected demo "
        "ballpark (python converges ~0.009-0.010)"
    )