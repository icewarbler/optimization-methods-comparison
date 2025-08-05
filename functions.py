import numpy as np
import numpy.linalg as la
import scipy.optimize as sopt

def quadratic(xy):
    np.random.seed(1)
    n = 2
    Q, _ = la.qr(np.random.randn(n, n))
    A = Q @ (np.diag(np.random.rand(n)) @ Q.T)
    b = np.random.randn(n)
    x, y = xy
    return 0.5*(A[0,0]*x*x + 2*A[1,0]*x*y +  A[1,1]*y*y) - x*b[0] - y*b[1]

def grad_quadratic(xy):
    np.random.seed(1)
    n = 2
    Q, _ = la.qr(np.random.randn(n, n))
    A = Q @ (np.diag(np.random.rand(n)) @ Q.T)
    b = np.random.randn(n)
    x, y = xy
    return np.array([
            A[0,0]*x + A[0,1]*y - b[0],
            A[1,0]*x + A[1,1]*y - b[1]
        ])

def rosenbrock(xy):
    x, y = xy
    return (1 - x)**2 + 100*(y - x**2)**2

def grad_rosenbrock(xy):
    x, y = xy
    return np.array([
        -2*(1 - x)**2 - 400*x * (y - x**2),
        200*(y - x**2)
    ])

def rastrigin(xy, A=10):
    x, y = xy
    return (
        A * 2
        + (x**2 - A * np.cos(2 * np.pi * x))
        + (y**2 - A * np.cos(2 * np.pi * y))
    )

def grad_rastrigin(xy, A=10):
    x, y = xy
    dx = 2 * x + 2 * np.pi * A * np.sin(2 * np.pi * x)
    dy = 2 * y + 2 * np.pi * A * np.sin(2 * np.pi * y)
    return np.array([dx, dy])

def get_function(name):
    if name == "rosenbrock":
        return rosenbrock, grad_rosenbrock
    elif name == "rastrigin":
        return rastrigin, grad_rastrigin
    elif name == "quadratic":
        return quadratic, grad_quadratic
    else:
        raise ValueError(f"Unknown function: {name}")