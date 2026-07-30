"""Pure-Python list-of-lists matmul backend (phase 1)."""


def matmul(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    raise NotImplementedError


def add_bias(A: list[list[float]], b: list[float]) -> list[list[float]]:
    raise NotImplementedError


def transpose(A: list[list[float]]) -> list[list[float]]:
    raise NotImplementedError


def elementwise(A: list[list[float]], fn) -> list[list[float]]:
    raise NotImplementedError
