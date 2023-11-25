from numba import njit

@njit()
def swap1d(x, i, j):
    return swap1d_(x.copy(), i, j)

@njit()
def swap2d(x, i, j, axis = 0):
    return swap2d_(x.copy(), i, j, axis)

@njit()
def swap3d(x, i, j, axis = 0):
    return swap3d_(x.copy(), i, j, axis)


@njit()
def swap1d_(x, i, j):
    temp = x[i].copy()
    x[i] = x[j]
    x[j] = temp
    return x

@njit()
def swap2d_(x, i, j, axis = 0):
    if axis == 0:
        temp = x[i].copy()
        x[i] = x[j]
        x[j] = temp
    elif axis == 1:
        temp = x[:,i].copy()
        x[:,i] = x[:,j]
        x[:,j] = temp
    else:
        raise ValueError("axis must be between 0 and 1")
    return x

@njit()
def swap3d_(x, i, j, axis = 0):
    if axis == 0:
        temp = x[i].copy()
        x[i] = x[j]
        x[j] = temp
    elif axis == 1:
        temp = x[:,i].copy()
        x[:,i] = x[:,j]
        x[:,j] = temp
    elif axis == 2:
        temp = x[:,:,i].copy()
        x[:,:,i] = x[:,:,j]
        x[:,:,j] = temp
    else:
        raise ValueError("axis must be between 0 and 2")
    return x