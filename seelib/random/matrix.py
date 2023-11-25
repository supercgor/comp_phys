import numpy as np

def pos_def(n: int):
    out = np.random.rand(n,n)
    out[:, :] = out @ out.T
    out += out.T
    return out
