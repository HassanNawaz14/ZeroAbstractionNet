"""Feedforward neural network using pluggable ops backend."""

import math
import random

from ops.activations import sigmoid, sigmoid_prime_from_output, tanh_prime_from_output


class Network:
    def __init__(self, layer_sizes: list[int], backend, seed: int):
        self.layer_sizes = layer_sizes
        self.backend = backend
        rng = random.Random(seed)

        self.weights: list[list[list[float]]] = []
        self.biases: list[list[float]] = []

        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = 1.0 / math.sqrt(fan_in)
            W = []
            for _ in range(fan_in):
                row = []
                for _ in range(fan_out):
                    row.append(rng.uniform(-limit, limit))
                W.append(row)
            b = [0.0] * fan_out
            self.weights.append(W)
            self.biases.append(b)

    def forward(self, X: list[list[float]]) -> tuple[list[list[float]], list]:
        cache = {}
        a = X
        cache["a0"] = a

        for i in range(len(self.weights)):
            z = self.backend.matmul(a, self.weights[i])
            z = self.backend.add_bias(z, self.biases[i])
            cache[f"z{i + 1}"] = z

            if i == len(self.weights) - 1:
                a = self.backend.elementwise(z, sigmoid)
            else:
                a = self.backend.elementwise(z, math.tanh)
            cache[f"a{i + 1}"] = a

        return a, cache

    def backward(self, X, y_true, cache) -> dict:
        n = len(X)
        num_layers = len(self.weights)
        grads = {}

        if y_true and not isinstance(y_true[0], list):
            y_true_m = [[v] for v in y_true]
        else:
            y_true_m = y_true

        da = None
        for i in range(num_layers - 1, -1, -1):
            z = cache[f"z{i + 1}"]
            a_prev = cache[f"a{i}"]

            if i == num_layers - 1:
                a = cache[f"a{i + 1}"]
                dz = []
                for j in range(n):
                    row = []
                    for k in range(len(a[0])):
                        row.append(a[j][k] - y_true_m[j][k])
                    dz.append(row)
            else:
                a = cache[f"a{i + 1}"]
                dz = []
                for j in range(n):
                    row = []
                    for k in range(len(z[0])):
                        row.append(da[j][k] * tanh_prime_from_output(a[j][k]))
                    dz.append(row)

            dW = self.backend.matmul(self.backend.transpose(a_prev), dz)

            db = []
            for k in range(len(dz[0])):
                s = 0.0
                for j in range(n):
                    s += dz[j][k]
                db.append(s)

            if i == num_layers - 1:
                da = self.backend.matmul(dz, self.backend.transpose(self.weights[i]))
            else:
                da = self.backend.matmul(dz, self.backend.transpose(self.weights[i]))

            grads[f"W{i + 1}"] = dW
            grads[f"b{i + 1}"] = db

        return grads

    def update(self, grads, lr: float) -> None:
        for i in range(len(self.weights)):
            W = self.weights[i]
            dW = grads[f"W{i + 1}"]
            for r in range(len(W)):
                for c in range(len(W[0])):
                    W[r][c] -= lr * dW[r][c]

            b = self.biases[i]
            db = grads[f"b{i + 1}"]
            for j in range(len(b)):
                b[j] -= lr * db[j]

    def get_state(self) -> dict:
        return {
            "weights": self.weights,
            "biases": self.biases,
        }

    @staticmethod
    def binary_cross_entropy(y_true, y_pred: list[list[float]]) -> float:
        n = len(y_pred)
        if y_true and not isinstance(y_true[0], list):
            y_true_m = [[v] for v in y_true]
        else:
            y_true_m = y_true
        loss = 0.0
        eps = 1e-15
        for i in range(n):
            for j in range(len(y_pred[0])):
                p = max(min(y_pred[i][j], 1.0 - eps), eps)
                loss -= y_true_m[i][j] * math.log(p) + (1.0 - y_true_m[i][j]) * math.log(1.0 - p)
        return loss / n
