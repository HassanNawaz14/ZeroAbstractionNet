"""Deterministic XOR-quadrant dataset generator.

Golden test points (always classify correctly once trained):
  ( 0.5,  0.5) -> 1
  (-0.5, -0.5) -> 1
  ( 0.5, -0.5) -> 0
  (-0.5,  0.5) -> 0
"""

import math
import random


def generate_dataset(
    n_per_quadrant: int, seed: int, noise_std: float = 0.0
) -> tuple[list[list[float]], list[float]]:
    """Returns (X, y) for the XOR-quadrant classification problem.

    X : list of [x, y] pairs, length == 4 * n_per_quadrant
    y : list of labels (0.0 or 1.0), same order as X

    Points are drawn on a jittered grid within each quadrant so the dataset
    is visually even, not a random scatter.  The grid is sized to
    accommodate exactly n_per_quadrant points; the remainder (when
    n_per_quadrant is not a perfect square) is discarded.

    Rule: label = 1  iff  x * y > 0   (quadrants I and III)
           label = 0  iff  x * y < 0   (quadrants II and IV)
    Points on the axes (x == 0 or y == 0) are never generated.
    When noise_std > 0, Gaussian noise is added to coordinates via the
    seeded RNG, which may push a point across an axis (= label noise).
    """
    rng = random.Random(seed)
    X: list[list[float]] = []
    labels: list[float] = []

    grid_size = math.ceil(math.sqrt(n_per_quadrant))

    quadrants = [
        (1.0, 1.0),    # Q1: x>0, y>0  -> label 1
        (-1.0, 1.0),   # Q2: x<0, y>0  -> label 0
        (-1.0, -1.0),  # Q3: x<0, y<0  -> label 1
        (1.0, -1.0),   # Q4: x>0, y<0  -> label 0
    ]

    for qx, qy in quadrants:
        label = 1.0 if qx * qy > 0 else 0.0
        points: list[list[float]] = []

        for gi in range(grid_size):
            for gj in range(grid_size):
                t = (gi + 0.5) / grid_size
                u = (gj + 0.5) / grid_size

                x = qx * (1.0 - t)
                y = qy * (1.0 - u)

                if noise_std > 0.0:
                    x += rng.gauss(0.0, noise_std)
                    y += rng.gauss(0.0, noise_std)

                points.append([x, y])

        rng.shuffle(points)
        selected = points[:n_per_quadrant]

        for pt in selected:
            X.append(pt)
            labels.append(label)

    return X, labels


def generate_probe_grid(resolution: int) -> list[list[float]]:
    """Returns a uniform `resolution x resolution` grid over [-1, 1]^2.

    Used only for rendering the decision-boundary heatmap during animation.
    """
    grid: list[list[float]] = []
    for i in range(resolution):
        for j in range(resolution):
            x = -1.0 + (2.0 * i + 1.0) / resolution
            y = -1.0 + (2.0 * j + 1.0) / resolution
            grid.append([x, y])
    return grid
