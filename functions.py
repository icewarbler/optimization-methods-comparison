import numpy as np
import numpy.linalg as la

def quadratic(x):
    np.random.seed(1)
    n = 2
    Q, _ = la.qr(np.random.randn(n, n))
    A = Q @ (np.diag(np.random.rand(n)) @ Q.T)
    b = np.random.randn(n)
    return 0.5*(A[0,0]*x[0]*x[0] + 2*A[1,0]*x[0]*x[1] +  A[1,1]*x[1]*x[1]) - x[0]*b[0] - x[1]*b[1]

def grad_quadratic(x):
    np.random.seed(1)
    n = 2
    Q, _ = la.qr(np.random.randn(n, n))
    A = Q @ (np.diag(np.random.rand(n)) @ Q.T)
    b = np.random.randn(n)
    return np.array([
            A[0,0]*x[0] + A[0,1]*x[1] - b[0],
            A[1,0]*x[0] + A[1,1]*x[1] - b[1]
        ])

def rosenbrock(x):
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def grad_rosenbrock(x):
    return np.array([
        -2*(1 - x[0])**2 - 400*x[0] * (x[1] - x[0]**2),
        200*(x[1] - x[0]**2)
    ])

def rastrigin(x, A=10):
    x = np.asarray(x)
    n = len(x)
    return A*n + np.sum(x**2 - A * cos(2*np.pi*x))

def grad_rastrigin(x, A=10):
    x = np.asarray(x)
    return 2*x + 2*np.pi*A*np.sin(2*np.pi*x) 

def get_function(name):
    if name == "rosenbrock":
        return rosenbrock, grad_rosenbrock
    elif name == "rastrigin":
        return rastrigin, grad_rastrigin
    else:
        raise ValueError(f"Unknown function: {name}")