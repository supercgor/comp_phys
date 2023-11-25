# %%

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Q = np.array([[1, 1, 1], [1, 0, -1], [1, -2, 1]])
Q = Q / np.linalg.norm(Q, axis = 1, keepdims = True)
simga = np.diag([0, 1, 3])

# %%
theta = np.linspace(0, 2*np.pi, 100)
xp = 0 * theta
yp = np.sqrt(2) * np.cos(theta)
zp = np.sqrt(2/3) * np.sin(theta)
x, y, z = Q.T @ np.stack([xp, yp, zp], axis = 0)

ax.plot(x, y, z, label = "$u^TBu$")
# %%


u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)

vectors = np.array([[0, 0, 0, 1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)], [0, 0, 0, 1/np.sqrt(2), 0, -1/np.sqrt(2)], [0, 0, 0, 1/np.sqrt(6), -2/np.sqrt(6), 1/np.sqrt(6)]])

ax.plot_surface(x, y, z, alpha = 0.2, color = "grey", label="unit sphere")

for vector in vectors:
    X, Y, Z, U, V, W = vector
    ax.quiver(X, Y, Z, U, V, W, length=1, linewidth=3)

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_zticks([-1, -0.5, 0, 0.5, 1])

ax.set_box_aspect([1,1,1])

plt.show()