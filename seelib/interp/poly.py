import numpy as np
from ..linalg import Lsolver

def lagrange(x, y, X):
    xjm = x[None, :] - x[:, None] + np.eye(x.shape[0])
    L = np.divide((X[:,None,None] - x[None, None, :]),xjm[None, ...])
    D = np.diag_indices_from(L[0])
    L[:, D[0], D[1]] = 1
    L = L.prod(axis = -1)
    L = L * y[None, :]
    L = L.sum(axis = -1)
    return L if x.shape[0] % 2 else -L

def newton(x, y, X):
    Mij = np.tril(x[:, None] - x[None, :])
    Mij = np.cumprod(Mij, axis = -1)
    Mij[:, -1] = 1
    Mij = np.roll(Mij, 1, axis = -1)
    A = Lsolver(Mij, y)
    N = X[:, None] - x[None, :] # 5 3
    N = np.roll(N, 1, axis = -1)
    N[...,0] = 1
    N = N.cumprod(axis = -1)
    N = (N * A[None, :]).sum(axis = -1)
    return N

def neville(x, y, X):
    T = np.tile(y,(X.shape[0], 1)) # 100, j
    X = X[:, None]
    for k in range(1, x.shape[0]):
        T = (T[:, 1:] * (X - x[None, :-k]) - T[:, :-1] * (X - x[None, k:])) / (x[None, k:] - x[None, :-k])
    return T[...,0]

# if __name__ == "__main__":
    # x =np.linspace(0, 1, 10)
    # y = np.random.rand(10)
    # X = np.linspace(0, 1, 1000)
    # import matplotlib.pyplot as plt
    # import timeit
    # fig, ax = plt.subplots(1, 3)
    # print(timeit.timeit("lagrange(x, y, X)", globals = globals(), number = 10))
    # ax[0].plot(x, y, 'o')
    # ax[0].plot(X, lagrange(x, y, X))
    # ax[0].set_title("Lagrange")
    # print(timeit.timeit("newton(x, y, X)", globals = globals(), number = 10))
    # ax[1].plot(x, y, 'o')
    # ax[1].plot(X, newton(x, y, X))
    # ax[1].set_title("Newton")
    # print(timeit.timeit("neville(x, y, X)", globals = globals(), number = 10))
    # ax[2].plot(x, y, 'o')
    # ax[2].plot(X, neville(x, y, X))
    # ax[2].set_title("Neville")
    # plt.show()
    