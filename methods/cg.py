import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
    
def run(f, gradf, x0, max_iter=100, tol=1e-6, plot=False, func_name="function"):
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
        plot_iterations(f, xs, func_name)

    return xs, gs, ss

def plot_iterations(f, xs, func_name="function"):
#    xmesh, ymesh = np.mgrid[-10:10:50j,-10:10:50j]
#    f_mesh = f(np.array([xmesh, ymesh]))
    xs_array = np.array(xs)
    x_min, x_max = xs_array[:, 0].min(), xs_array[:, 0].max()
    y_min, y_max = xs_array[:, 1].min(), xs_array[:, 1].max()

    x_pad = 0.1 * (x_max - x_min)
    y_pad = 0.1 * (y_max - y_min)

    x_low, x_high = x_min - x_pad, x_max + x_pad
    y_low, y_high = y_min - y_pad, y_max + y_pad

    xmesh, ymesh = np.mgrid[x_low:x_high:200j, y_low:y_high:200j]
    f_mesh = np.zeros_like(xmesh)
    for i in range(xmesh.shape[0]):
        for j in range(xmesh.shape[1]):
            point = np.array([xmesh[i, j], ymesh[i, j]])
            f_mesh[i, j] = f(point)

    plt.figure()
    plt.axis("equal")
    plt.contour(xmesh, ymesh, f_mesh, levels=50)
    plt.plot(xs_array[:, 0], xs_array[:, 1], "x-", color="red")
#    xs_array = np.array(xs)
 #   plt.plot(xs_array.T[0], xs_array.T[1], "x-")
    plt.savefig(f"plots/{func_name}_cg_plot.png")