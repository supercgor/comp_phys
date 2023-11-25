# %%
# Calculate $e^{-x}$
import numpy as np
import math
cutoff = 20
dtype = np.float32
# %%
# (a)
xs = np.arange(0, 100, 10, dtype=dtype)
values = np.zeros_like(xs)
for i in range(cutoff):
    values = values + ((-1)**i) * (xs**i) / math.factorial(i)
print(f"Direct expansion\n{values}")

# %%
# (b)
si = dtype(1)
values = np.zeros_like(xs)
for i in range(cutoff):
    values += si
    si = -si * xs / (i + 1)
print(f"Recursive expansion\n{values}")
# %%
# (c)
values = np.zeros_like(xs)
for i in range(cutoff):
    values += (xs**i) / math.factorial(i)
values = 1 / values
print(f"Inverse expansion\n{values}")

values = np.zeros_like(xs)
si = dtype(1)
for i in range(cutoff):
    values += si
    si = si * xs / (i + 1)
values = 1 / values
print(f"Inverse recursive expansion\n{values}")

print(f"True values\n{np.exp(-xs)}")

# %%
from decimal import Decimal, getcontext
getcontext().prec = 2048

def factorial(n: Decimal):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)
    

de = []
for x in range(0, 100, 10):
    v = 0
    for i in range(500):
        in1 = (-1) ** i
        if i == 0:
            in2 = Decimal(1)
        else:
            in2 = Decimal(x) ** i
        in3 = factorial(Decimal(i))
        v += in1 * in2 / in3
    de.append(v)
for d in de:
    print(f"{d:.6e}", end=" ")
# %%
