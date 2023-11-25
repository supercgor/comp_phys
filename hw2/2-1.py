import sys
import os
sys.path.append(os.path.curdir)

import numpy as np
import seelib as sl

#gem
x = np.array([
    [0.05, 0.07, 0.06, 0.05],
    [0.07, 0.10, 0.08, 0.07],
    [0.06, 0.08, 0.10, 0.09],
    [0.05, 0.07, 0.09, 0.10]
])

b = np.array([0.23, 0.32, 0.33, 0.31])[...,None]

U, bp = sl.linalg.gem(x, b)
sol = sl.linalg.Usolver(U, bp)

print(sol[:,0])

#cholesky
L = sl.linalg.cholesky(x)
y = sl.linalg.Lsolver(L, b)
sol = sl.linalg.Usolver(L.T, y)

print(sol[:,0])
