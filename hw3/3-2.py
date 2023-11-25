import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return (1 + x/2 + x**2/12)/(1 - x/2 + x**2/12)

X = np.linspace(-3, 3, 100)
plt.title("Pade[2,2] Approximation of $e^x$")
plt.plot(X, np.exp(X), label="f(x)")
plt.plot(X, f(X), label="R(x)")
plt.grid()
plt.legend()
plt.show()