"""Activation functions and their derivatives (pure Python, shared by all phases)."""

import math


def sigmoid(x: float) -> float:
    """Sigmoid activation: 1 / (1 + exp(-x))."""
    if x < -45:
        return 0.0
    if x > 45:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def sigmoid_prime_from_output(s: float) -> float:
    """Derivative of sigmoid given s = sigmoid(x) already computed: s * (1 - s)."""
    return s * (1.0 - s)


def tanh_prime_from_output(t: float) -> float:
    """Derivative of tanh given t = tanh(x) already computed: 1 - t^2."""
    return 1.0 - t * t
