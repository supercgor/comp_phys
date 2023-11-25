import numpy as np
from ...linalg import gem
from ...linalg.jit import Usolver

def square_spline(x: np.ndarray, y: np.ndarray, X: np.ndarray, mind: int = 0, mval: float = 0.0) -> np.ndarray:
    n = x.shape[0]
    dy,dx = np.diff(y), np.diff(x)
    h = 2 * (dy / dx)
    U = np.eye(n, n) + np.eye(n, n, 1)
    U[-1] = 0
    U[-1, mind] = 1
    h = np.append(h, mval)
    print(U)
    U, hp = gem(U, h[...,None])
    M = np.squeeze(Usolver(U, hp))
    Y = np.empty_like(X)
    a = np.empty_like(x)
    a[0] = y[0] - M[0] * dx[0] / 2
    a[1:] = M[1:] * dx / 2 + y[1:]
    i = j = 0
    while i < X.shape[0] and j < x.shape[0]-1:
        if x[j] > X[i]:
            i += 1
        elif X[i] > x[j+1]:
            j += 1
        else:
            Y[i] = - M[j]/2 / dx[j] * (X[i] - x[j+1]) ** 2 + M[j+1] / 2 / dx[j] * (X[i]- x[j]) ** 2 + a[j]
            i += 1
    return Y