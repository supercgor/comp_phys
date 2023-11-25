import numpy as np
from numba import njit

@njit(parallel = True)
def Usolver(U: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Ux = b, where U is an upper triangular matrix."""
    assert U.shape[0] == U.shape[1], "U must be a square matrix."
    assert U.shape[0] == b.shape[0], "U and b must have the same number of rows."
    n = len(U)
    x = b.copy()
    for i in range(n-1, -1, -1):
        for j in range(i+1, n):
            x[i] -= U[i,j] * x[j]
        x[i] /= U[i, i]
    return x

@njit(parallel = True)
def Lsolver(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve Lx = b, where L is a lower triangular matrix."""
    assert L.shape[0] == L.shape[1], "L must be a square matrix."
    assert L.shape[0] == b.shape[0], "L and b must have the same number of rows."
    n = len(L)
    x = b.copy()
    for i in range(n):
        for j in range(i):
            x[i] -= L[i, j] * x[j]
        x[i] /= L[i,i]
    return x