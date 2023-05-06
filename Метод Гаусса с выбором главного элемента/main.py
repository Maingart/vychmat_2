import numpy as np

np.set_printoptions(linewidth=1000)


DTYPE = 'float64'


def solve_gauss(matrix: np.matrix) -> np.ndarray:
    n = matrix.shape[0]

    for k in range(n - 1):
        ind = k + np.argmax(np.abs(matrix[k:, k]))
        matrix[k, :], matrix[ind, :] = np.copy(matrix[ind, :]), np.copy(matrix[k, :])

        for i in range(k + 1, n):
            frac = matrix[i, k] / matrix[k, k]
            matrix[i, :] -= matrix[k, :] * frac

    x = np.transpose(np.matrix([0.0 for _ in range(n)]))
    for k in range(n - 1, -1, -1):
        x[k, 0] = (matrix[k, -1] - matrix[k, k:n] * x[k:n, 0]) / matrix[k, k]

    return x


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

    x = np.round(solve_gauss(np.c_[mat, b]), 3)

    print(*x)


if __name__ == '__main__':
    main()
