"""Feedforward neural network using pluggable ops backend."""


class Network:
    def __init__(self, layer_sizes: list[int], backend, seed: int):
        raise NotImplementedError

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        raise NotImplementedError

    def backward(self, X, y_true, cache) -> dict:
        raise NotImplementedError

    def update(self, grads, lr: float) -> None:
        raise NotImplementedError

    def get_state(self) -> dict:
        raise NotImplementedError
