"""Tests for ops/backend_python.py.

# A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
# A @ B = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
"""

from ops.backend_python import matmul, add_bias, transpose, elementwise


def test_matmul_2x2():
    A = [[1, 2], [3, 4]]
    B = [[5, 6], [7, 8]]
    expected = [[19, 22], [43, 50]]
    assert matmul(A, B) == expected


def test_matmul_non_square():
    A = [[1, 2, 3], [4, 5, 6]]   # 2 x 3
    B = [[7], [8], [9]]           # 3 x 1
    # expected: [[1*7+2*8+3*9], [4*7+5*8+6*9]] = [[50], [122]]
    expected = [[50], [122]]
    assert matmul(A, B) == expected


def test_matmul_shape_mismatch():
    A = [[1, 2], [3, 4]]   # 2 x 2
    B = [[5, 6]]            # 1 x 2  (k=1 != A's k=2)
    try:
        matmul(A, B)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_add_bias():
    A = [[1.0, 2.0], [3.0, 4.0]]
    b = [0.5, -0.5]
    expected = [[1.5, 1.5], [3.5, 3.5]]
    assert add_bias(A, b) == expected


def test_add_bias_empty():
    assert add_bias([], []) == []


def test_add_bias_shape_mismatch():
    try:
        add_bias([[1, 2]], [1, 2, 3])
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_transpose():
    A = [[1, 2, 3], [4, 5, 6]]
    expected = [[1, 4], [2, 5], [3, 6]]
    assert transpose(A) == expected


def test_transpose_empty():
    assert transpose([]) == []


def test_elementwise():
    A = [[1.0, 2.0], [3.0, 4.0]]
    expected = [[2.0, 3.0], [4.0, 5.0]]
    assert elementwise(A, lambda x: x + 1.0) == expected


def test_elementwise_empty():
    assert elementwise([], lambda x: x) == []
