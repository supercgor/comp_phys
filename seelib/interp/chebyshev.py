import numpy as np

def chebyshev(f, X, N, xmin = -5, xmax = 5):
    """Compute the Chebyshev interpolant of f at the points X."""
    x = (np.arange(2*N, step=2) + 1) * np.pi / (2 * N)
    x = np.flip(x)
    m = np.arange(N)
    mx = np.cos(m[:, None] * x[None,:]) * (xmax - xmin)/2 + (xmax + xmin)/2
    cm = (mx * f(mx[(1,),:])).sum(-1) * 2 / N
    cm[0] /= 2
    cm /= (xmax - xmin)/2
    X = (X - (xmax + xmin)/2) * 2 / (xmax - xmin)
    Tmx = np.cos(m[:, None] * np.arccos(X[None,:]))
    Tx = (Tmx * cm[:, None]).sum(axis = 0)
    return Tx
    

def __f(x):
    return 1 / (1 + x ** 2)

if __name__ == "__main__":
    X = np.linspace(-8, 7, 1000)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot(X, __f(X))
    ax.plot(X, chebyshev(__f, X, 11, -8, 7.9))
    plt.show()