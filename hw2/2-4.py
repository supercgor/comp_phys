import os, sys
sys.path.append(os.path.curdir)

import numpy as np
from seelib.interp import curve_spline, cubic_spline
import matplotlib.pyplot as plt

np.set_printoptions(precision=2, formatter={'float': '{: 0.2f}'.format})
ps = np.linspace(0, 2* np.pi, 400)
phi = np.linspace(0, 2 * np.pi, 9)

r = lambda phi: 1 - np.cos(phi)

x = r(phi) * np.cos(phi)
y = r(phi) * np.sin(phi)

print(np.stack([phi, x, y], axis=1))
xs = r(ps) * np.cos(ps)
ys = r(ps) * np.sin(ps)

tX = cubic_spline(phi, x, ps, ddm1=0, ddm2=0)
tY = cubic_spline(phi, y, ps, ddm1=0, ddm2=0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
ax1.plot(ps, xs, label='original', linestyle='--')
ax2.plot(ps, ys, label='original', linestyle='--')
ax1.plot(ps, tX, label='spline')
ax2.plot(ps, tY, label='spline')
ax1.set_xlabel('$\phi$', fontsize=15)
ax2.set_xlabel('$\phi$', fontsize=15)
ax1.set_xticks(np.linspace(0, 2 * np.pi, 5))
ax2.set_xticks(np.linspace(0, 2 * np.pi, 5))
ax1.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax2.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
ax1.set_ylabel('$x$', fontsize=15)
ax2.set_ylabel('$y$', fontsize=15)
ax1.scatter(phi, x, label='data')
ax2.scatter(phi, y, label='data')
ax1.legend()
ax2.legend()
ax1.grid()
ax2.grid()
ax1.set_title('$x = (1 - \cos(\phi))\cos(\phi)$')
ax2.set_title('$y = (1 - \cos(\phi))\sin(\phi)$')

plt.figure(figsize=(8, 8))
plt.title('$r = 1 - \cos(\phi)$', fontsize=15)
plt.scatter(x, y, label='data')
plt.plot(xs, ys, label='original', linestyle='--')
plt.plot(tX, tY, label='spline')
plt.xlabel('$\phi$', fontsize=15)
plt.ylabel('$r$', fontsize=15)
plt.legend()
plt.grid()
plt.show()
