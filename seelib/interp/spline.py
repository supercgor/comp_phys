import numpy as np
from numpy.typing import NDArray
from typing import Literal
from ..linalg import gem
from ..linalg.jit import Usolver

class CubicSpline():
    def __init__(self, x: NDArray, y: NDArray, bc_type: str | dict = 'natural'):
        if bc_type == "natural":
            self._bc_type = {0: (2, 0.0), -1: (2, 0.0)}
        elif bc_type == "periodic":
            assert False, "Not implemented"
            assert np.isclose(y[0], y[-1]), "y must be periodic"
            self._bc_type = {'p': None}
            x = np.append(x, x[1] - x[0] + x[-1])
            y = np.append(y, y[1])
        else:
            pass
        n = x.shape[0]
        eqs = np.zeros((n, n))
        d = np.zeros(n)
        mu = np.zeros(n)
        lam = np.zeros(n)
        dx, dy = np.diff(x), np.diff(y)

        mu[:-2] = dx[:-1] / (dx[:-1] + dx[1:]) # n-1
        lam[:-2] = 1 - mu[:-2] # n-1
        d[:-2] = 6 * (y[:-2] / dx[:-1] / (dx[:-1] + dx[1:]) + y[2:] / dx[1:] / (dx[:-1] + dx[1:]) - y[1:-1] / dx[:-1] / dx[1:])
    
        eqs += np.diag(mu) + np.roll(np.diag(lam), 2, axis=1)
        eqs[:-2,1:-1] += 2 * np.eye(n - 2)
        for i, (ind, elems) in enumerate(self._bc_type.items(), 1):
            if 'p' == ind:
                d = d[:-1]
                dx = dx[:-1]
                dy = dy[:-1]
                x = x[:-1]
                y = y[:-1]
                eqs[-3,0] = eqs[-3,-1]
                eqs = eqs[:-1, :-1]
                eqs[-1, 0] = 2
                eqs[-1, 1] = dx[0] / (dx[0] + dx[-1])
                eqs[-1, -1] = dx[-1] / (dx[0] + dx[-1])
                d = np.roll(d, 1)
                d[0] = 6 * (dy[0]/dx[0] - dy[-1]/dx[-1])/(dx[0]+ dx[-1])
                print(d)
                print(eqs)
                
            else:
                div, val = elems
                if div == 1:
                    if ind < 0 and ind + 1 >= 0:
                        eqs[-i, ind-1] = dx[ind]/6
                        eqs[-i, ind] = dx[ind]/3
                        d[-i] = val - dy[ind] / dx[ind]
                    else:
                        eqs[-i, ind] = - dx[ind] / 3
                        eqs[-i, ind+1] = - dx[ind] / 6
                        d[-i] = val - dy[ind] / dx[ind]
                elif div == 2:
                    eqs[-i, ind] = 1
                    d[-i] = val

        U, hp = gem(eqs, d[...,None]) # n
        self._M = np.squeeze(Usolver(U, hp))  # n
        self._a = dy/dx - dx * np.diff(self._M) / 6 # n-1
        self._b = y[:-1] - dx ** 2 * self._M[:-1] / 6 # n-1
        self._x = x
    
    def __call__(self, X: NDArray) -> NDArray:
        Y = np.empty_like(X)
        i = j = 0
        while i < X.shape[0] and j < self._x.shape[0]-1:
            Y[i] = -self._M[j] * (X[i] - self._x[j+1]) ** 3 / 6 / (self._x[j+1] - self._x[j]) + self._M[j+1] * (X[i] - self._x[j]) ** 3 / 6 / (self._x[j+1] - self._x[j]) + self._a[j] * (X[i] - self._x[j]) + self._b[j]
            if self._x[j] > X[i]:
                i += 1
            elif X[i] > self._x[j+1] and j + 1 < self._x.shape[0] - 1:
                j += 1
            else:
                i += 1
        Y = np.where(np.isclose(Y, 0), 0, Y)
        return Y
        

def cubic_spline(x, y, X, m1ind = 0, m2ind = -1, m1 = ..., m2 = ..., dm1 = ..., dm2 = ..., ddm1 = ..., ddm2 = ...):
    def get_m(ind, val):
        raise NotImplementedError
    def get_dm(ind, val):
        eq = np.zeros((x.shape[0]))
        if ind < 0 and ind + 1 >= 0:
            eq[ind-1] = dx[ind]/6
            eq[ind] = dx[ind]/3
            b = val - dy[ind] / dx[ind]
        else:
            eq[ind] = - dx[ind] / 3
            eq[ind+1] = - dx[ind] / 6
            b = val - dy[ind] / dx[ind]
        return eq, b
    def get_ddm(ind, val):
        eq = np.zeros((x.shape[0]))
        eq[ind] = 1
        return eq, val
    dx, dy = np.diff(x), np.diff(y)
    eqs = []
    bs = []
    if m1 is not ...:
        eq, b = get_m(m1ind, m1)
        eqs.append(eq)
        bs.append(b)
    if m2 is not ...:
        eq, b = get_m(m2ind, m2)
        eqs.append(eq)
        bs.append(b)
    if dm1 is not ...:
        eq, b = get_dm(m1ind, dm1)
        eqs.append(eq)
        bs.append(b)
    if dm2 is not ...:
        eq, b = get_dm(m2ind, dm2)
        eqs.append(eq)
        bs.append(b)
    if ddm1 is not ...:
        eq, b = get_ddm(m1ind, ddm1)
        eqs.append(eq)
        bs.append(b)
    if ddm2 is not   ...:
        eq, b = get_ddm(m2ind, ddm2)
        eqs.append(eq)
        bs.append(b)
    if len(eqs) != 2:
        raise ValueError("You must specify 2 of m1, m2, dm1, dm2, ddm1, ddm2")
    n = x.shape[0]
    eqs = np.stack(eqs, axis = 0)
    b = np.array(bs)
    mu = dx[:-1] / (dx[:-1] + dx[1:])
    lam = 1 - mu
    d = 6 * (y[:-2] / dx[:-1] / (dx[:-1] + dx[1:]) + y[2:] / dx[1:] / (dx[:-1] + dx[1:]) - y[1:-1] / dx[:-1] / dx[1:])
    std_eq = np.zeros((n,n))
    std_eq[:-2,:-2] += np.diag(mu)
    std_eq[:-2,1:-1] += 2 * np.eye(n-2)
    std_eq[:-2,2:] += np.diag(lam)
    std_eq[-2:] = eqs
    d = np.append(d, b)

    U, hp = gem(std_eq, d[...,None]) # n
    M = np.squeeze(Usolver(U, hp))  # n
    Y = np.empty_like(X)
    a = dy/dx - dx * np.diff(M) / 6 # n-1
    b = y[:-1] - dx ** 2 * M[:-1] / 6 # n-1
    i = j = 0
    while i < X.shape[0] and j < x.shape[0]-1:
        Y[i] = -M[j] * (X[i] - x[j+1]) ** 3 / 6 / dx[j] + M[j+1] * (X[i] - x[j]) ** 3 / 6 / dx[j] + a[j] * (X[i] - x[j]) + b[j]
        if x[j] > X[i]:
            i += 1
        elif X[i] > x[j+1] and j + 1 < x.shape[0] - 1:
            j += 1
        else:
            i += 1
    return Y
    
def square_spline(x: np.ndarray, y: np.ndarray, X: np.ndarray, mind: int = 0, mval: float = 0.0) -> np.ndarray:
    n = x.shape[0]
    dy,dx = np.diff(y), np.diff(x)
    h = 2 * (dy / dx)
    U = np.eye(n, n) + np.eye(n, n, 1)
    U[-1] = 0
    U[-1, mind] = 1
    h = np.append(h, mval)
    U, hp = gem(U, h[...,None])
    M = np.squeeze(Usolver(U, hp))
    Y = np.empty_like(X)
    a = np.empty_like(x)
    a[0] = y[0] - M[0] * dx[0] / 2
    a[1:] = M[1:] * dx / 2 + y[1:]
    i = j = 0
    while i < X.shape[0] and j < x.shape[0]-1:
        Y[i] = - M[j]/2 / dx[j] * (X[i] - x[j+1]) ** 2 + M[j+1] / 2 / dx[j] * (X[i]- x[j]) ** 2 + a[j]
        if x[j] > X[i]:
            i += 1
        elif X[i] > x[j+1]:
            j += 1
        else:
            i += 1
    return Y

def curve_spline(ys, N, method: Literal["cubic", "square"]= "cubic"):
    # ys: (n, m), N: 1000
    X = np.linspace(0, 1, N)
    x = np.linspace(0 ,1, ys.shape[0])
    out = []
    for m in range(ys.shape[1]):
        if method == "cubic":
            out.append(cubic_spline(x, ys[:,m], X, ddm1=0, ddm2=0))
        elif method == "square":
            out.append(square_spline(x, ys[:,m], X))
        else:
            raise ValueError("method must be 'cubic' or 'square'")
    return np.stack(out, axis = 1)
        
        
    