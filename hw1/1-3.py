import os
import sys
sys.path.append(os.path.curdir)
import numpy as np
import seelib as sl

np.set_printoptions(linewidth=200)
for i in range(1, 11):
    Hn = sl.array.hilbert(i)
    b = np.ones(i)

    U, P = sl.linalg.gem(Hn, return_aux=True)
    

    solution = sl.linalg.Usolver(U, P @ b)
    print(solution)
    # print(Hn @ solution)

for i in range(1, 11):
    Hn = sl.array.hilbert(i)
    b = np.ones(i)
    
    L = sl.linalg.cholesky(Hn)
    solution = sl.linalg.Lsolver(L, b)
    solution = sl.linalg.Usolver(L.T, solution)
    print(solution)
    # print(Hn @ solution)
    
"""Out
[1.]
[-2.  6.]
[  3. -24.  30.]
[  -4.   60. -180.  140.]
[    5.  -120.   630. -1120.   630.]
[-6.000e+00  2.100e+02 -1.680e+03  5.040e+03 -6.300e+03  2.772e+03]
[ 7.00000004e+00 -3.36000002e+02  3.78000002e+03 -1.68000001e+04  3.46500001e+04 -3.32640001e+04  1.20120000e+04]
[-7.99999954e+00  5.03999972e+02 -7.55999961e+03  4.61999978e+04 -1.38599994e+05  2.16215991e+05 -1.68167993e+05  5.14799981e+04]
[ 8.99992146e+00 -7.19994491e+02  1.38599056e+04 -1.10879320e+05  4.50447486e+05 -1.00900283e+06  1.26125404e+06 -8.23676381e+05  2.18789103e+05]
[-9.99704969e+00  9.89744533e+02 -2.37545500e+04  2.40190393e+05 -1.26102316e+06  3.78312843e+06 -6.72565032e+06  7.00024576e+06 -3.93767681e+06  9.23660503e+05]
[1.]
[-2.  6.]
[  3. -24.  30.]
[  -4.   60. -180.  140.]
[    5.  -120.   630. -1120.   630.]
[-6.000e+00  2.100e+02 -1.680e+03  5.040e+03 -6.300e+03  2.772e+03]
[ 6.99999999e+00 -3.36000000e+02  3.78000000e+03 -1.68000000e+04  3.46500000e+04 -3.32640000e+04  1.20120000e+04]
[-7.99999802e+00  5.03999891e+02 -7.55999855e+03  4.61999920e+04 -1.38599978e+05  2.16215969e+05 -1.68167978e+05  5.14799937e+04]
[ 8.99989933e+00 -7.19992997e+02  1.38598808e+04 -1.10879145e+05  4.50446854e+05 -1.00900156e+06  1.26125258e+06 -8.23675512e+05  2.18788889e+05]
[-9.99706763e+00  9.89747614e+02 -2.37546413e+04  2.40191412e+05 -1.26102876e+06  3.78314551e+06 -6.72568069e+06  7.00027706e+06 -3.93769412e+06  9.23664479e+05]
"""

import numpy as np
def c(n):
    def factorial(n):
        return np.prod(np.arange(1, n+1, dtype=np.float64))
    out = 1
    for i in range(1,n):
        out *= factorial(i)
    return out
for n in range(1, 11):
    print(f"ln det(H_{n}) = {4* np.log(c(n)) - np.log(c(2 * n))}")