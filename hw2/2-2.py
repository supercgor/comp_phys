import sys
import os
sys.path.append(os.path.curdir)

import numpy as np
import seelib
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


def f(x):
    return np.cos(x ** 2)

def fp(x):
    return -2 * x * np.sin(x ** 2)

x = np.array([0.0,0.6,0.9])
y = f(x)
X = np.linspace(0,0.9,100)

FIG = plt.figure(figsize=(12, 10),constrained_layout=True)
fig1, fig2 = FIG.subfigures(2, 1, height_ratios=[1, 1])
FIG.suptitle('Cubic Spline Interpolation\n', fontsize = 20)
fig1.suptitle('$d^2g|_{x=x_0,x_2}=0$', fontsize = 16)
fig2.suptitle('$dg|_{x=x_0,x_2}=df$', fontsize = 16)
ax1,ax2,ax3 = fig1.subplots(1,3, sharex=True)
ax4,ax5,ax6 = fig2.subplots(1,3, sharex=True)

for ax in [ax1, ax4]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.plot(X,f(X),label='$f(x)=cos(x^2)$')
    ax.plot(x,y,'o', label='data')
    ax.title.set_text('My CubicSpline')

for ax in [ax2, ax5]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.plot(X,f(X),label='$f(x)=cos(x^2)$')
    ax.plot(x,y,'o', label='data')
    ax.title.set_text('Scipy CubicSpline')
    
for ax in [ax3, ax6]:
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    ax.title.set_text('Difference')
    
Y1 = seelib.interp.cubic_spline(x,y,X,ddm1=0, ddm2=0)
ax1.plot(X, Y1, label='')

cs = CubicSpline(x, y, bc_type='natural')
Y2 = cs(X)
ax2.plot(X, Y2, label='$dg|_{x_0,x_2}=df$')

Y3 = seelib.interp.cubic_spline(x,y,X,dm1=fp(0), dm2=fp(0.9))
ax4.plot(X, Y3, label='')

cs = CubicSpline(x, y, bc_type=((1, fp(0)), (1, fp(0.9))))
Y4 = cs(X)
ax5.plot(X, Y4, label='$dg|_{x_0,x_2}=df$')

ax3.plot(X, Y1 - Y2)

ax6.plot(X, Y3 - Y4)

#plt.tight_layout()
plt.legend()
plt.show()