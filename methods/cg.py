import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt
import matplotlib.pyplot as plt
import os
import warnings
    
def run(f, gradf, x0, max_iter=100, tol=1e-6, plot=False, func_name="function", steepness=False):
    xs = [x0]
    gs = [gradf(x0)]
    ss = [-gs[0]]
    
    for i in range(max_iter):
        best_alpha = sopt.line_search(f, gradf, xs[i], ss[i])[0]
        if best_alpha is None:
            best_alpha = 1e-4

        next_x = xs[i] + best_alpha * ss[i]
        xs.append(next_x)

        next_g = gradf(xs[-1])
        gs.append(next_g)

        if np.linalg.norm(next_g) < tol:
            break

        # use Polak-Ribiere instead of Fletcher-Reeves
        y = next_g - gs[i]
        beta = max(0, np.dot(next_g, y) / np.dot(gs[i], gs[i]))

        if i % len(x0) == 0 or np.dot(next_g, ss[i]) > 0:
            next_s = -next_g
        else:
            next_s = -next_g + beta * ss[i]

        ss.append(next_s)

    if plot:
        plot_iterations(f, xs, func_name)
    
    if steepness:
        plot_steepness_iterations(gs, func_name)

    return xs[-1], f(xs[-1]), len(xs)

def plot_iterations(f, xs, func_name="function"):
    os.makedirs("plots", exist_ok=True)
    xs_array = np.array(xs)
    x_min, x_max = xs_array[:, 0].min(), xs_array[:, 0].max()
    y_min, y_max = xs_array[:, 1].min(), xs_array[:, 1].max()

    x_pad = 0.1*(x_max - x_min)
    y_pad = 0.1*(y_max - y_min)

    x_low, x_high = x_min - x_pad, x_max + x_pad
    y_low, y_high = y_min - y_pad, y_max + y_pad

    xmesh, ymesh = np.mgrid[x_low:x_high:200j, y_low:y_high:200j]
    f_mesh = f(np.array([xmesh, ymesh]))

    plt.figure()
    plt.axis("equal")
    plt.contour(xmesh, ymesh, f_mesh, levels=50)
    plt.plot(xs_array[:, 0], xs_array[:, 1], "x-", color="red")
    plt.plot(xs_array[-1, 0], xs_array[-1, 1], "o", color="blue", markersize=10, label="final point")
    plt.savefig(f"plots/{func_name}_cg_plot.png")

def plot_steepness_iterations(gs, func_name="function"):
    os.makedirs("plots", exist_ok=True)

    grad_norms = [la.norm(g) for g in gs]
    iterations = list(range(len(gs)))

    plt.figure() 
    plt.plot(iterations, grad_norms)
    plt.xlabel("iteration")
    plt.ylabel("||grad||")
    plt.yscale("log")
    plt.title("Steepness")
    plt.grid(True)
    plt.savefig(f"plots/{func_name}_cg_steepness.png")