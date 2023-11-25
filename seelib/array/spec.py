import numpy as np

def hilbert(n: int) -> np.ndarray:
    """Hilbert matrix of order n."""
    return np.array([[1 / (i + j + 1) for j in range(n)] for i in range(n)])