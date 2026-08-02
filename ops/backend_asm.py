"""Assembly (AVX2/FMA) shared-library matmul backend (phase 3).

Phase-3 precision decision: the asm kernels operate in float32 (YMM holds
8 floats vs. 4 doubles — the entire reason for hand-writing assembly). The
flatten/unflatten layer rounds every value to float32, so selecting this
backend runs the whole network in float32. That is the intended, expected
precision downgrade for this backend only — never upcast back to double
here; it would defeat the point of the phase.

add_bias/transpose/elementwise are re-exported from backend_python (same
pattern as phase 2). The python-side functions see float32-rounded floats,
so their double-precision math is fine — the precision loss already
happened at the boundary, not in the elementwise code.
"""

import array as _array
import ctypes
import itertools
import os
import struct

from ops.backend_python import add_bias, transpose, elementwise

_lib_path = os.path.join(
    os.path.dirname(__file__), "..", "native", "asm", "libmatmul_asm.so"
)
_lib = ctypes.CDLL(_lib_path)

_float_ptr = ctypes.POINTER(ctypes.c_float)
for _name in ("matmul_asm_scalar", "matmul_asm_vectorized", "matmul_asm"):
    _fn = getattr(_lib, _name)
    _fn.argtypes = [
        _float_ptr, _float_ptr, _float_ptr,
        ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    _fn.restype = None

_VARIANT_SYMBOL = {
    "scalar": "matmul_asm_scalar",
    "vectorized": "matmul_asm_vectorized",
    "blocked": "matmul_asm",
}


def _flatten(A: list[list[float]]) -> tuple[ctypes.Array, int, int]:
    """Row-major flatten to a ctypes FLOAT32 array, returns (array, rows, cols).

    array('f') rounds each value to float32 — that rounding is the backend's
    precision downgrade, applied once per crossing. Bulk transfer via
    from_buffer (the phase-2 marshalling lesson); results are bit-stable.
    """
    n = len(A)
    k = len(A[0])
    buf = _array.array("f", itertools.chain.from_iterable(A))
    flat = (ctypes.c_float * (n * k)).from_buffer(buf)
    return flat, n, k


def _unflatten(flat: ctypes.Array, n: int, m: int) -> list[list[float]]:
    """Back to list-of-lists: bulk byte copy + one struct.unpack pass.

    '=f' reads the exact bytes the assembly wrote (float32 echoed back as
    python floats), so nothing is re-rounded on the way out.
    """
    values = struct.unpack(
        "=" + str(n * m) + "f", memoryview(flat).tobytes()
    )
    return [list(values[i * m:(i + 1) * m]) for i in range(n)]


def matmul(A: list[list[float]], B: list[list[float]], variant: str = "blocked") -> list[list[float]]:
    """A is n x k, B is k x m, returns n x m (float32).

    variant in {'scalar', 'vectorized', 'blocked'}; default 'blocked'
    (the stage-C kernel) is what train.py/compare_backends.py use while
    benchmark_matmul.py exercises all three explicitly.
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
    if variant not in _VARIANT_SYMBOL:
        raise ValueError(
            f"matmul: unknown variant {variant!r} "
            "(choose from 'scalar', 'vectorized', 'blocked')"
        )

    A_flat, _, _ = _flatten(A)
    B_flat, _, _ = _flatten(B)
    C_flat = (ctypes.c_float * (n * m))()

    fn = getattr(_lib, _VARIANT_SYMBOL[variant])
    fn(A_flat, B_flat, C_flat, n, k, m)

    return _unflatten(C_flat, n, m)