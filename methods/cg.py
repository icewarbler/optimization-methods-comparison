import numpy as np
import scipy.optimize as sopt
    
def run(f, gradf, x0, max_iter=100, tol=1e-6):
    xs = [x0]
    gs = [gradf(x0)]
    ss = [-gs[0]]
    
    for i in range(max_iter):
        f_ls = lambda alpha: f(xs[i] + alpha * ss[i])
        best_alpha = sopt.golden(f_ls)

        next_x = xs[i] + best_alpha * ss[i]
        xs.append(next_x)

        next_g = gradf(xs[-1])
        gs.append(next_g)
        if np.linalg.norm(next_g) < tol:
            break

        beta = np.dot(next_g, next_g) / np.dot(gs[i], gs[i])
        next_s = -gs[-1] + beta * ss[i]
        ss.append(next_s)
    return xs, gs, ss