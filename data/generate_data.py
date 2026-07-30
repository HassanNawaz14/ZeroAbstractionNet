"""Deterministic XOR-quadrant dataset generator."""


def generate_dataset(
    n_per_quadrant: int, seed: int, noise_std: float = 0.0
) -> tuple[list[list[float]], list[float]]:
    raise NotImplementedError


def generate_probe_grid(resolution: int) -> list[list[float]]:
    raise NotImplementedError
