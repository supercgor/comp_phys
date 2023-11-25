# %%
# (a)
import numpy as np
n = 7
A = np.triu(np.full((n,n), -1), 1) + np.diag(np.ones(n), 0)
print(f"det(A) = {np.diag(A).prod()}")
# %%
# (b)
Aprime = A.copy()
IA = np.eye(n)
for ind in range(n-1, 0, -1):
    change = IA[(-1,),:] * Aprime[:-1, ind][:, None]/ Aprime[ind, ind]
    IA[:ind] -= IA[(ind,),:] * Aprime[:ind, ind][:, None]/ Aprime[ind, ind]

print(f"Inverse of A = \n{IA}")
IA = np.linalg.inv(A)
print(f"True inverse of A = \n{IA}")
# %%
# (c)