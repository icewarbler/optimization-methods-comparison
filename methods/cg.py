import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
    
def run(f, gradf, x0, max_iter=100, tol=1e-6, plot=False):
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

    if plot:
        plot_iterations(f, xs)

    return xs, gs, ss

def plot_iterations(f, xs):
    xmesh, ymesh = np.mgrid[-10:10:50j,-10:10:50j]
    f_mesh = f(np.array([xmesh, ymesh]))
    plt.axis("equal")
    plt.contour(xmesh, ymesh, f_mesh, 50)
    xs_array = np.array(xs)
    plt.plot(xs_array.T[0], xs_array.T[1], "x-")
    plt.savefig("plots/cg_plot.png")