import numpy as np


DTYPE = 'float64'


def solve_gauss(m):
    n = m.shape[0]
    for k in range(n - 1):
        ind = k + np.argmax(np.abs(m[k:, k]))
        if ind != k:
            m[k, :], m[ind, :] = np.copy(m[ind, :]), np.copy(m[k, :])

        for i in range(k + 1, n):
            frac = m[i, k] / m[k, k]
            m[i, :] -= m[k, :] * frac

    print(np.round(m, 3), '\n')

    x = np.transpose(np.matrix([0.0 for _ in range(n)]))
    for k in range(n - 1, -1, -1):
        x[k, 0] = (m[k, -1] - m[k, k:n] * x[k:n, 0]) / m[k, k]

    return x


def main():
    A = np.array([
        [3, 4, -9, 5],
        [-15, -12, 50, -16],
        [-27, -36, 73, 8],
        [9, 12, -10, -16],
    ], dtype=DTYPE)
    b = np.array([-14, 44, 142, -76], dtype=DTYPE)

    x = np.round(solve_gauss(np.c_[A, b]), 3)

    print(*x)


if __name__ == '__main__':
    main()
