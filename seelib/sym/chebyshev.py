from sympy import symbols, simplify, Integer

def chebyshev(n, x):
    if isinstance(x, str):
        x = symbols(x)
    if n == 0:
        return Integer(1)
    elif n == 1:
        return x
    else:
        return 2 * x * chebyshev(n-1, x) - chebyshev(n-2, x)