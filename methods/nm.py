import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
import os
import warnings
    
def run(f, x0, max_iter=10, tol=1e-4, plot=False, func_name="function"):
    n = len(x0)
    x0 = np.array(x0)
    
    simplex = [x0]
    step = 0.05

    for i in range(n):
        x = np.copy(x0)
        x[i] += step
        simplex.append(x)
    
    res = [[x, f(x)] for x in simplex]

    for _ in range(max_iter):
        res, conv = nm(f, res, tol)
        if conv:
            break
    
    best = min(res, key=lambda x: x[1])
    print(f"best point: {best[0]}, value: {best[1]}")
    return best[1]

def nm(f, res, tol=1e-4):
    
    # order
    res.sort(key=lambda x: x[1])
    xs = [r[0] for r in res]
    fs = [r[1] for r in res]

    # check convergence
    if np.max(np.abs(np.array(fs) - fs[0])) < tol:
        return res, True

    # centroid
    n = len(res[0][0])
    c = np.mean([x for x, _ in res[:-1]], axis=0)

    # reflection
    worst = res[-1][0]
    alpha = 1.0
    xr = c + alpha*(c - worst)
    rval = f(xr)

    if res[0][1] <= rval < res[-2][1]:
        res[-1] = [xr, rval] 
        return res
    
    # expansion
    if rval < res[0][1]:
        gamma = 2.0
        xe = c + gamma*(xr - c)
        eval = f(xe)
        if eval < rval:
            res[-1] = [xe, eval]
            return res, False
        else:
            res[-1] = [xr, rval]
            return res, False

    # contraction
    rho = 0.5
    if rval >= res[-2][1]:
        if rval < res[-1][1]:
            xc = c + rho*(xr - c)
            cval = f(xc)
            if cval < rval:
                res[-1] = [xc, cval]
                return res, False

        else:
            xc = c + rho*(res[-1][0] - c)
            cval = f(xc)
            if cval < res[-1][1]:
                res[-1] = [xc, cval]
                return res, False
        # shrink
        sigma = 0.5
        best = res[0][0]
        new_res = [res[0]]
        for i in range(1, len(res)):
            x_shrunk = best + sigma*(res[i][0] - best)
            new_res.append([x_shrunk, f(x_shrunk)])
        return new_res, False
    return res, False