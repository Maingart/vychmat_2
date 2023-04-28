# Метод окаймления

## Реализация
```python
def inv(A: matrix, depth=0) -> matrix:
    n = len(A)
    k = n - 1

    if n == 1:
        return np.matrix(1 / A[0, 0])

    Ap = A[:k, :k]
    V, U = A[k, :k], A[:k, k]

    Ap_inv = inv(Ap, depth + 1)

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
```
## Использование
```python
def main():
    mat = np.matrix([
        [1, 2, 4],
        [3, 3, 2],
        [4, 1, 3]
    ])

    print('Моя реализация: ', inv(mat), 'numpy: ', np.linalg.inv(mat), sep='\n')
```
```pycon
Моя реализация: 
[[-0.22580645  0.06451613  0.25806452]
 [ 0.03225806  0.41935484 -0.32258065]
 [ 0.29032258 -0.22580645  0.09677419]]
numpy: 
[[-0.22580645  0.06451613  0.25806452]
 [ 0.03225806  0.41935484 -0.32258065]
 [ 0.29032258 -0.22580645  0.09677419]]
```