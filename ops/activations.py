"""Activation functions and their derivatives (pure Python, shared by all phases)."""


def sigmoid(x: float) -> float:
    raise NotImplementedError


def sigmoid_prime_from_output(s: float) -> float:
    raise NotImplementedError


def tanh_prime_from_output(t: float) -> float:
    raise NotImplementedError
