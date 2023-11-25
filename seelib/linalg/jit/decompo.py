import numpy as np
from numba import njit, prange
from ...array.swap import swap2d_
        
@njit(parallel = True)
def cholesky(x: np.ndarray) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    assert x.shape[0] == x.shape[1], "x must be a square matrix"
    x = x.copy()
    for j in range(x.shape[0]):
        for k in range(j):
            x[j,j] -= x[j,k] ** 2
        x[j,j] = np.sqrt(x[j,j])
        for i in range(j+1, x.shape[0]):
            for k in range(j):
                x[i,j] -= x[i,k] * x[j,k]
            x[i,j] /= x[j,j]
    return np.tril(x)

@njit(parallel = True)
def _gem(x, allow_permute = True, eps = 1E-6):
    assert len(x.shape) == 2, "x must be a matrix"
    i = j = 0
    while i < x.shape[0] - 1 and j < x.shape[1]:
        if allow_permute:
            temp = i+1
            for k in range(i+1, x.shape[0]):
                if np.abs(x[k,j]) > np.abs(x[i,j]):
                    temp = k
            swap2d_(x, i, temp, axis = 0)
        if np.isclose(x[i,j], 0, atol = eps):
            j += 1
            continue
        else:
            for k in prange(i+1, x.shape[0]):
                x[k] -= x[i] * x[k,j] / x[i,j]
            i += 1
            j += 1
    return x