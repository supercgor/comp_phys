import numpy as np
def newton_cortes(f, x = [0,1,100], deg = 2):
    """
    Approximate the integral of f(x) from a to b by the trapezoidal rule.
    """
    x = np.linspace(*x)
    conv = __newton_kernal(deg)
    left = (len(x)-1) % deg
    dx= x[1] - x[0]
    I = f(x)
    i = 0
    out = 0
    while i +deg < len(I):
        out += conv * I[i:i+deg+1] * dx * deg
        i += deg
    out = np.sum(out)
    if left != 0:
        conv_left = __newton_kernal(left)
        out += np.sum(conv_left * I[-left-1:] * dx * left)
    return out

def gauss_legendre(f, xmin = -1, xmax = 1, deg = 10):
    xp, wp = np.polynomial.legendre.leggauss(deg)
    xp = (xmax - xmin)/2 * xp + (xmax + xmin)/2
    print(xp.shape, wp.shape)
    return np.sum(wp * f(xp)) * (xmax - xmin)/2
    
    
def __newton_kernal(deg):
    if deg == 0:
        conv = np.array([1])
    elif deg == 1:
        conv = np.array([1, 1])/2
    elif deg == 2:
        conv = np.array([1, 4, 1])/6
    elif deg == 3:
        conv = np.array([1, 3, 3, 1])/8
    elif deg == 4:
        conv = np.array([7, 32, 12, 32, 7])/90
    elif deg == 5:
        conv = np.array([19, 75, 50, 50, 75, 19])/288
    elif deg == 6:
        conv = np.array([41, 216, 27, 272, 27, 216, 41])/840
    else:
        raise ValueError(f"deg must be [1, 6], {deg} is given")
    return conv

# Define the arrays
if __name__ == "__main__":
    X = np.linspace(0, 1, 100).astype(np.float64)
    f = lambda x: np.exp(x)
    print("Gauss-Legendre")
    print(gauss_legendre(f, 0, 1, 10))
    for i in range(1, 7):
        print(f"n = {i}")
        print(newton_cortes(f, [0, 1, 100], i))
        print()
