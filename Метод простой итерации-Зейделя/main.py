import numpy as np

from utils import A, b


def simple_iterations(mat: np.matrix, _: np.ndarray) -> [np.matrix, int]:
    return [mat, 0]


def seidel_method(mat: np.matrix, _: np.ndarray) -> [np.matrix, int]:
    return [mat, 0]


def main():
    x_simple_iterations, count_simple_iterations = simple_iterations(A, b)
    x_seidel_method, count_seidel_method = seidel_method(A, b)

    print(x_simple_iterations)
    print(x_seidel_method)
    print(x_simple_iterations - x_seidel_method)
    print(count_simple_iterations, count_seidel_method)


if __name__ == '__main__':
    main()
