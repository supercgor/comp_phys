import numpy as np
from math import sqrt
# L = np.array([[sqrt(2)/2, sqrt(2)/4, sqrt(2)/8], 
#               [0, sqrt(2)/2, sqrt(2)/4], 
#               [0, 0, sqrt(6)/3]]).T
L = np.array([[1, 0, 0], 
              [-1/2, 1, 0], 
              [0, -2/3, 1]])
D = np.diag([2, 3/2, 1/3])
H = np.array([[2, -1, 0],
             [-1, 2, -1],
             [0, -1, 2]])
print(L @ D @ L.T)

H = np.array([
    [2, -1, 0, 0],
    [-1, 2, -1, 0],
    [0, -1, 2, -1],
    [0, 0, -1, 2]])

print(np.linalg.eigvals(H))