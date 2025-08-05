import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt
import matplotlib.pyplot as plt
import os
import warnings

def run(f, gradf, x0, max_iter=100, tol=1e-6, plot=False, func_name="function"):
    xs = [x0]
    bs = [np.eye(len(x0))]
    ps = []
    
    for i in range(max_iter):
        grad = gradf(xs[-1])


        print(f"iter {i}: ||grad|| = {la.norm(grad):.2e}, f(x) = {f(xs[-1]):.6f}")
        
        if la.norm(grad) < tol:
            break

        p_k = -la.solve(bs[-1], grad)
        ps.append(p_k)

        best_alpha = sopt.line_search(f, gradf, xs[-1], p_k)[0]
        if best_alpha is None:
            best_alpha = 1.0
        
        s_k = best_alpha * p_k
        x_next = xs[-1] + s_k

        y_k = gradf(x_next) - grad

        s_k = s_k[:, None]
        y_k = y_k[:, None]

        b_next = bs[-1] + ((y_k @ y_k.T) / (y_k.T @ s_k)) - ((bs[-1] @ s_k @ s_k.T @ bs[-1].T) / (s_k.T @  bs[-1] @ s_k))
        
        xs.append(x_next)
        bs.append(b_next)
    
    if plot:
        plot_iterations(f, xs, func_name)
    
    return xs, bs, ps

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
    # f_mesh = np.zeros_like(xmesh)
    # for i in range(xmesh.shape[0]):
    #     for j in range(xmesh.shape[1]):
    #         point = np.array([xmesh[i, j], ymesh[i, j]])
    #         f_mesh[i, j] = f(point)

    plt.figure()
    plt.axis("equal")
    plt.contour(xmesh, ymesh, f_mesh, levels=50)
    plt.plot(xs_array[:, 0], xs_array[:, 1], "x-", color="red")
    plt.plot(xs_array[-1, 0], xs_array[-1, 1], "o", color="blue", markersize=10, label="final point")
    plt.savefig(f"plots/{func_name}_bfgs_plot.png")