import numpy as np

def eta(x, theta):
    f1 = np.exp(-x*(1+theta))
    f2 = np.exp(-(x-7*theta)**2/(4*theta))

    return f1 + 0.5*f2

def zeta(x, theta=0.4):
    f1 = np.exp(-theta*x)
    f2 = np.sin((1+theta)*x)

    return f1*(1-(1-theta)*f2)