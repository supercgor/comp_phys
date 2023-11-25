import numpy as np
from numba import njit

@njit()
def argafter(x, X, equal = True):
    """
    Find the index of the first element in x that is greater than X.
    Args:
        x (float): values to compare with
        X (np.ndarray): sorted array

    Returns:
        int: index of the first element in x that is greater than X
    """
    i = 0
    if equal:
        while i < x.shape[0] and x[i] <= X:
            i += 1
        return i
    else:
        while i < x.shape[0] and x[i] < X:
            i += 1
        return i