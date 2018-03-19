import numpy as np

def cm2inch(size):
    return size/2.54

def linear_reg(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    X = np.vander(x, 2)
    coeffs, res, rank, sing_vals = np.linalg.lstsq(X, y)
    mx = x.sum()/len(x)
    sx = float(((x - mx)**2).sum())
    if len(x) > 2:
        r2 = 1. - res/(y.size*np.var(y))
    else:
        r2 = 0
    return coeffs, r2

