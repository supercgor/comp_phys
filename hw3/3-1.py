import numpy as np
import matplotlib.pyplot as plt
from sympy import simplify, symbols, latex

def f(x):
    out = x ** 6 + \
          3 * x ** 5 + \
          4 * x ** 4 + \
          x ** 3 / 3 + \
          2 * x ** 2 + \
          x - 10
    return out
# parameters
t = 6

def T(x, n):
    if n == 0:
        return 1
    elif n == 1:
        return x
    else:
        return 2 * x * T(x, n-1) - T(x, n-2)

ks = np.arange(0, t)
print(len(ks))
p = (np.cos((ks + 0.5) * np.pi / t) * 2) + 1

c0 = 1/t * np.sum(f(p))
ps = []
cs = [c0]

for i in range(1, t):
    pm = np.cos(i * np.pi * (ks + 0.5) / t)
    cs.append(2/t * np.sum(pm * f(p)))
for c in cs:
    print(f"{c:.1f}")

x = np.linspace(-1, 3, 100)

plt.plot(x, f(x), label="f(x)")
y = (x - 1) / 2

S = 0
xsym = symbols("x")
Ssym = 0
for i in range(0, t):
    Ssym += cs[i] * T(xsym, i)
    
Ssym = simplify(Ssym).subs(xsym, (xsym - 1)/2)
for i in range(0, t):
    S += cs[i] * T(y, i)
plt.plot(x, S, label="S(x)", c="r")
plt.scatter(p, f(p), c="r")
plt.title(f"{t}th-order Chebyshev Interpolation of f(x)")
print(latex(simplify(Ssym)))
plt.grid()
plt.legend()
plt.show()