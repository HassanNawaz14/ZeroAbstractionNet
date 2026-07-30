"""Assembly shared-library matmul backend (phase 3 — not yet implemented)."""


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    raise NotImplementedError(
        "Assembly backend not yet implemented. Use --backend python for phase 1."
    )


def add_bias(A: list[list[float]], b: list[float]) -> list[list[float]]:
    raise NotImplementedError(
        "Assembly backend not yet implemented. Use --backend python for phase 1."
    )


def transpose(A: list[list[float]]) -> list[list[float]]:
    raise NotImplementedError(
        "Assembly backend not yet implemented. Use --backend python for phase 1."
    )


def elementwise(A: list[list[float]], fn) -> list[list[float]]:
    raise NotImplementedError(
        "Assembly backend not yet implemented. Use --backend python for phase 1."
    )
