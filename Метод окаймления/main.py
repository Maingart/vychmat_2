import numpy as np
from numpy import matrix


def inv(A: matrix) -> matrix:
    n = len(A)
    k = n - 1

    if n == 1:
        return np.matrix(1 / A[0, 0])

    Ap = A[:k, :k]
    V, U = A[k, :k], A[:k, k]

    Ap_inv = inv(Ap)

    alpha = 1 / (A[k, k] - V * Ap_inv * U).item()
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
    mat = np.matrix([
        [0.411, 0.421, -0.333, 0.313, -0.141, -0.381, 0.245],
        [0.241, 0.705, 0.139, -0.409, 0.321, 0.0625, 0.101],
        [0.123, -0.239, 0.502, 0.901, 0.243, 0.819, 0.321],
        [0.413, 0.309, 0.801, 0.865, 0.423, 0.118, 0.183],
        [0.241, -0.221, -0.243, 0.134, 1.274, 0.712, 0.423],
        [0.281, 0.525, 0.719, 0.118, -0.974, 0.808, 0.923],
        [0.246, -0.301, 0.231, 0.813, -0.702, 1.223, 1.105],
    ])

    b = np.matrix([0.096, 1.252, 1.024, 1.023, 1.155, 1.937, 1.673]).T

    print(inv(mat) - np.linalg.inv(mat))
    print(inv(mat) * b)


if __name__ == '__main__':
    main()
