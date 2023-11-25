import sys, os
sys.path.append(os.path.curdir)

from seelib import interp
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, formatter={'float': '{: 0.2f}'.format})
def f(x):
    return 1 / (1 + 25 * x ** 2)

fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_axes([0.1,0.1, 0.8, 0.8])
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.grid()
ax1.set_xlim(-1.03, 1.03)
ax1.set_ylim(-0.1, 1.1)
fig.suptitle('Runge Phenomenon', fontsize = 20, y=0.95, fontweight='bold')

x = np.linspace(-1, 1, 21)

X0 = np.linspace(-1, 1, 41)
X = np.linspace(-1, 1, 501)

ax1.plot(X, f(X), label='$f(x)=\\frac{1}{1+25x^2}$', c='grey', linestyle='--', linewidth=4)

Y0 = interp.lagrange(x, f(x), X0)
Y = interp.lagrange(x, f(x), X)

ax1.scatter(X0, Y0, label='Lagrange', marker='o', c='g')
ax1.plot(X, Y, c="g", alpha=0.5)

Y1 = interp.chebyshev(f, X0, N=21, xmin= -1, xmax = 1)
Y = interp.chebyshev(f, X, N=21, xmin= -1, xmax = 1)

ax1.scatter(X0, Y1, label='Chebyshev', marker='o', c='r')
ax1.plot(X, Y, c="r", alpha=0.5)

Y2 = interp.cubic_spline(x, f(x), X0, ddm1=0, ddm2=0)
Y = interp.cubic_spline(x, f(x), X, ddm1=0, ddm2=0)

ax1.scatter(X0, Y2, label='Cubic Spline', marker='o', c='b')
ax1.plot(X, Y, c="b", alpha=0.5)

ax1.legend(loc='upper left', ncol=1)

ax2 = fig.add_axes([0.6,0.58, 0.27, 0.27])
ax2.set_title('Zoomed View', fontsize = 16)

Y0 = interp.lagrange(x, f(x), X0)
Y = interp.lagrange(x, f(x), X)

ax2.grid()
ax2.set_xlim(-1.03, 1.03)
# no ticks

ax2.plot(X, Y, c="g", linewidth=1)
ax2.plot(X, f(X), label='$f(x)=\\frac{1}{1+25x^2}$', c='grey', linestyle='--', linewidth=2)
ax2.scatter(X0, Y0, label='Lagrange', marker='o', c='g', s=10)

compare = np.stack([
    X0,
    f(X0),
    Y0,
    np.abs(Y0 - f(X0)),
    Y1,
    np.abs(Y1 - f(X0)),
    Y2,
    np.abs(Y2 - f(X0)),
], axis=0)

print(compare.T)

# print(Y0)
# print(f(X0))
# print(np.abs(Y0 - f(X0)))


plt.show()