import numpy as np

from utils import A, b


def inv(matrix: np.matrix) -> np.matrix:
    n = len(matrix)
    k = n - 1

    if n == 1:
        return np.matrix(1 / matrix[0, 0])

    Ap = matrix[:k, :k]
    V, U = matrix[k, :k], matrix[:k, k]

    Ap_inv = inv(Ap)

    alpha = 1 / (matrix[k, k] - V * Ap_inv * U).item()
    Q = -V * Ap_inv * alpha
    P = Ap_inv - Ap_inv * U * Q
    R = - Ap_inv * U * alpha

    A_inv = np.matrix([[0.0] * n for _ in range(n)])
    A_inv[:k, :k] = P
    A_inv[k, :k] = Q[0]

    A_inv[:k, k] = R[:, 0]
    A_inv[k, k] = alpha

    return A_inv


def main():
    print(inv(A) - np.linalg.inv(A))
    print(inv(A) * b)


if __name__ == '__main__':
    main()
