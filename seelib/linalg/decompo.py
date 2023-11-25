import numpy as np

def gem(x: np.ndarray, P: np.ndarray = ..., allow_permute: bool = True, return_aux: bool = True, eps: float = ...) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Use Gaussian elimination to turn x to U.
    Args:
        x (np.ndarray): square matrix
        P (np.ndarry, optional): marking matrix. Default to identity matrix, `P @ X = U`
        allow_permute (bool, optional): allow row permutation. Defaults to True.
        return_aux (bool, optional): return P. Defaults to True.
        eps (float, optional): tolerance. Defaults to np.finfo(x.dtype).eps * 2.

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]: _description_
    """
    assert len(x.shape) == 2, "x must be a matrix"
    if eps is ...:
        eps = np.finfo(x.dtype).eps * 2
    if P is ...:
        P = np.eye(x.shape[0])
    xp = np.concatenate([x, P], axis = 1)
    def gauss_elim(xpi):
        xpi[1:] -= xpi[0:1] * xpi[1:, 0:1] / xpi[0, 0:1]
    def row_permute(xpi: np.ndarray):
        ind = np.argsort(np.abs(xpi[:,0]))[::-1]
        xpi[:, :] = xpi[ind, :]
    i = j = 0
    while i < x.shape[0] - 1 and j < x.shape[1]:
        if allow_permute:
            row_permute(xp[i:,j:])
        if np.isclose(xp[i,j], 0, atol = eps):
            j += 1
            continue
        else:
            gauss_elim(xp[i:,j:])
            i += 1
            j += 1
    xp[np.isclose(xp, 0, atol = eps)] = 0
    return (xp[:,:x.shape[1]], xp[:,x.shape[1]:]) if return_aux else xp[:x.shape[0]]
        
    
def cholesky(x: np.ndarray) -> np.ndarray:
    assert x.shape[0] == x.shape[1], "x must be a square matrix"
    x = x.copy()
    for j in range(x.shape[0]):
        x[j,j] -= x[j,:j] @ x[j,:j]
        x[j,j] = np.sqrt(x[j,j])
        for i in range(j+1, x.shape[0]):
            x[i,j] -= x[i,:j] @ x[j,:j]
            x[i,j] /= x[j,j]
    return np.tril(x)