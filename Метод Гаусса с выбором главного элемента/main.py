import numpy as np

from utils import A, b

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
    x = np.round(solve_gauss(np.c_[A, b]), 3)

    print(*x)


if __name__ == '__main__':
    main()
