"""End-to-end forward/backward and golden-point convergence tests.

Architecture [2, 2, 1] for the known-weights test:

  W1 = [[0.5, -0.5],           b1 = [0.1, -0.1]
         [0.3, -0.3]]

  W2 = [[1.0],                  b2 = [0.0]
        [-1.0]]

Input X = [[1.0, 0.5]]:
  z1  = X @ W1 + b1
      = [1*0.5+0.5*0.3 + 0.1,  1*(-0.5)+0.5*(-0.3) + (-0.1)]
      = [0.75, -0.75]
  a1  = tanh(z1)  = [0.63514895238, -0.63514895238]

  z2  = a1 @ W2 + b2
      = 0.63514895238*1.0 + (-0.63514895238)*(-1.0) + 0.0
      = 1.27029790476
  a2  = sigmoid(z2)  = 1/(1+exp(-1.27029790476))  ≈ 0.78080331246
"""

import math
from network import Network
from ops import get_backend

backend = get_backend("python")


def test_forward_known_weights():
    W1 = [[0.5, -0.5], [0.3, -0.3]]
    b1 = [0.1, -0.1]
    W2 = [[1.0], [-1.0]]
    b2 = [0.0]

    net = Network([2, 2, 1], backend, seed=0)
    net.weights = [W1, W2]
    net.biases = [b1, b2]

    X = [[1.0, 0.5]]
    pred, cache = net.forward(X)

    expected = 0.78080331246
    assert abs(pred[0][0] - expected) < 1e-5, f"{pred[0][0]} != {expected}"

    assert "a0" in cache
    assert "z1" in cache
    assert "a1" in cache
    assert "z2" in cache
    assert "a2" in cache


def test_backward_shapes():
    net = Network([2, 4, 1], backend, seed=0)
    X = [[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]
    y = [1.0, 1.0, 0.0, 0.0]

    _, cache = net.forward(X)
    grads = net.backward(X, y, cache)

    assert len(grads["W1"]) == 2 and len(grads["W1"][0]) == 4
    assert len(grads["b1"]) == 4
    assert len(grads["W2"]) == 4 and len(grads["W2"][0]) == 1
    assert len(grads["b2"]) == 1


def test_backward_updates_decrease_loss():
    net = Network([2, 4, 1], backend, seed=0)
    X = [[0.5, 0.5], [-0.5, -0.5], [0.5, -0.5], [-0.5, 0.5]]
    y = [1.0, 1.0, 0.0, 0.0]

    pred0, cache0 = net.forward(X)
    loss0 = Network.binary_cross_entropy(y, pred0)

    for _ in range(10):
        _, cache = net.forward(X)
        grads = net.backward(X, y, cache)
        net.update(grads, 0.5)

    pred1, _ = net.forward(X)
    loss1 = Network.binary_cross_entropy(y, pred1)

    assert loss1 < loss0, f"loss did not decrease: {loss0} -> {loss1}"


def test_golden_points_converge():
    golden = [
        ([0.5, 0.5], 1.0),
        ([-0.5, -0.5], 1.0),
        ([0.5, -0.5], 0.0),
        ([-0.5, 0.5], 0.0),
    ]
    X = [p for p, _ in golden]
    y = [l for _, l in golden]

    net = Network([2, 4, 1], backend, seed=0)

    for epoch in range(500):
        pred, cache = net.forward(X)
        grads = net.backward(X, y, cache)
        net.update(grads, 0.5)

    pred, _ = net.forward(X)
    for i, (pt, expected) in enumerate(golden):
        got = pred[i][0]
        if expected == 1.0:
            assert got > 0.5, f"{pt}: {got} <= 0.5 (expected {expected})"
        else:
            assert got < 0.5, f"{pt}: {got} >= 0.5 (expected {expected})"
