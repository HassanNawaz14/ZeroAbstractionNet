"""Pure-Python list-of-lists matmul backend (phase 1)."""


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """A is n x k, B is k x m, returns n x m. Plain triple-nested loop."""
    if not A or not B:
        raise ValueError("matmul: empty matrix")
    n = len(A)
    k = len(A[0])
    m = len(B[0])
    if k != len(B):
        raise ValueError(
            f"matmul: shape mismatch A[{n}x{k}] @ B[{len(B)}x{m}]"
        )
    C = []
    for i in range(n):
        row = []
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            row.append(s)
        C.append(row)
    return C


def add_bias(A: list[list[float]], b: list[float]) -> list[list[float]]:
    """Adds bias vector b to every row of A."""
    if not A:
        return []
    m = len(A[0])
    if len(b) != m:
        raise ValueError(
            f"add_bias: bias length {len(b)} != row size {m}"
        )
    result = []
    for row in A:
        new_row = []
        for j in range(m):
            new_row.append(row[j] + b[j])
        result.append(new_row)
    return result


def transpose(A: list[list[float]]) -> list[list[float]]:
    """Returns the transpose of A."""
    if not A:
        return []
    n = len(A)
    m = len(A[0])
    result = []
    for j in range(m):
        new_row = []
        for i in range(n):
            new_row.append(A[i][j])
        result.append(new_row)
    return result


def elementwise(A: list[list[float]], fn) -> list[list[float]]:
    """Applies fn to every element, returns new matrix."""
    result = []
    for row in A:
        new_row = []
        for val in row:
            new_row.append(fn(val))
        result.append(new_row)
    return result
