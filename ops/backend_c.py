"""C shared-library matmul backend (phase 2).

Only matmul is implemented in C; add_bias/transpose/elementwise are
re-exported from backend_python — they're O(n), not the bottleneck, and
rewriting them would add ctypes marshalling overhead that swamps their tiny
cost. Matrices cross the ctypes boundary as flat row-major 1D arrays of
double; shapes (n, k, m) are passed explicitly as int arguments.
"""

import array as _array
import ctypes
import itertools
import os
import struct

from ops.backend_python import add_bias, transpose, elementwise

_lib_path = os.path.join(
    os.path.dirname(__file__), "..", "native", "c", "libmatmul.so"
)
_lib = ctypes.CDLL(_lib_path)

_double_ptr = ctypes.POINTER(ctypes.c_double)
for _name in ("matmul_naive", "matmul_blocked"):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [
        _double_ptr, _double_ptr, _double_ptr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    _fn.restype = None


def _flatten(A: list[list[float]]) -> tuple[bytes, int, int]:
    """Row-major flatten to a ctypes double array, returns (array, rows, cols).

    Bulk-built via array.array('d') so the whole matrix crosses in one C-level
    memory operation instead of an element-by-element ctypes assignment loop
    (the latter cost ~1.3ms for a 200x32 matrix and dominated tiny matmuls).
    ``from_buffer`` views the array's memory; ctypes keeps it alive. Values and
    row-major order are identical to the old loop, so results are bit-identical.
    """
    n = len(A)
    k = len(A[0])
    buf = _array.array("d", itertools.chain.from_iterable(A))
    flat = (ctypes.c_double * (n * k)).from_buffer(buf)
    return flat, n, k


def _unflatten(flat: ctypes.Array, n: int, m: int) -> list[list[float]]:
    """Back to list-of-lists: bulk byte copy + one struct.unpack pass.

    The old indexing loop charged ~1.5ms per ctypes element access at
    200x32; an unpack into a flat tuple plus contiguous row slices is several
    times faster. '=d' is native-order, double-precision IEEE-754 — the exact
    bytes the C library wrote, so no rounding occurs.
    """
    values = struct.unpack(
        "=" + str(n * m) + "d", memoryview(flat).tobytes()
    )
    return [list(values[i * m:(i + 1) * m]) for i in range(n)]


def matmul(A: list[list[float]], B: list[list[float]], variant: str = "blocked") -> list[list[float]]:
    """A is n x k, B is k x m, returns n x m.

    variant in {'naive', 'blocked'}. Default 'blocked' is what train.py
    uses; benchmark_matmul.py exercises both explicitly.
    """
    if not A or not B:
        raise ValueError("matmul: empty matrix")
    n = len(A)
    k = len(A[0])
    m = len(B[0])
    if k != len(B):
        raise ValueError(
            f"matmul: shape mismatch A[{n}x{k}] @ B[{len(B)}x{m}]"
        )

    A_flat, _, _ = _flatten(A)
    B_flat, _, _ = _flatten(B)
    C_flat = (ctypes.c_double * (n * m))()

    if variant == "blocked":
        _lib.matmul_blocked(A_flat, B_flat, C_flat, n, k, m)
    elif variant == "naive":
        _lib.matmul_naive(A_flat, B_flat, C_flat, n, k, m)
    else:
        raise ValueError(
            f"matmul: unknown variant {variant!r} (choose from 'naive', 'blocked')"
        )

    return _unflatten(C_flat, n, m)
