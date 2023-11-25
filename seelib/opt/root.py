import numpy as np

def bisection(f, x_min, x_max, eps = 1E-15):
    while x_max - x_min > eps:
        x_mid = (x_max + x_min)/2
        #print(".", end="")
        #print(x_mid)
        if f(x_mid) * f(x_min) < 0:
            x_max = x_mid
        else:
            x_min = x_mid
    return (x_max + x_min)/2, f((x_max + x_min)/2)

def newton_ralphson(f, df, x_init, eps = 1E-15):
    dx = 1
    while True:
        #print(".", end="")
        fx, dfx = f(x_init), df(x_init)
        if abs(fx) < eps or abs(dx) < eps:
            return x_init, fx
        dx = -fx / dfx
        x_init += dx

def secant(f, x_init, eps = 1E-15):
    x0 = x_init - 0.1
    x1 = x_init + 0.1
    f0 = f(x0)
    f1 = f(x1)
    while abs(f1) > eps and abs(x1 - x0) > eps:
        #print(".", end="")
        x1, x0 = x1 - f1 * (x1 - x0) / (f1 - f0), x1
        f1, f0 = f(x1), f1
    return x1, f1
    
if __name__ == "__main__":
    f = lambda x: x - 2 * np.sin(x)
    df = lambda x: 1 - 2 * np.cos(x)
    print(bisection(f, -3, 3, eps= 1E-15))
    print(newton_ralphson(f, df, 3, eps= 1E-15))
    print(secant(f, 3, eps= 1E-15))
    
    f = lambda x: x ** 2 - 4 * x * np.sin(x) + (2 * np.sin(x)) ** 2
    df = lambda x: 2 * x - 4 * np.sin(x) - 4 * x * np.cos(x) + 8 * np.sin(x) * np.cos(x)
    print(bisection(f, -3, 3, eps= 1E-15))
    print(newton_ralphson(f, df, 3, eps= 1E-15))
    print(secant(f, 3, eps= 1E-15))